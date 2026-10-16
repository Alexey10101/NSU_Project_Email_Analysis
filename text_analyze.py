"""
Считает текстовые признаки и оценивает вероятность редактирования письма ИИ.
При прямом запуске обучает логистический классификатор и сохраняет его в text_classifier.json.
При анализе загружает готовую модель и объясняет вклад признаков.
"""

import json
import math
import random
import re
import sys
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from parser import EmailParser, clean_email_body


def calculate_sentence_variance(text: str) -> Dict[str, float]:
    raw_sentences = re.split(r"[.!?]+(?:\s+|$)", text)
    lengths = [len(s.strip().split()) for s in raw_sentences if len(s.strip().split()) > 0]

    if len(lengths) < 2:
        return {
            "sentence_count": len(lengths),
            "avg_sentence_len": float(lengths[0]) if lengths else 0.0,
            "sentence_len_variance": 0.0,
            "burstiness_ratio": 0.0
        }

    avg_len = statistics.mean(lengths)
    var = statistics.variance(lengths)
    std = statistics.stdev(lengths)
    burstiness = std / avg_len if avg_len > 0 else 0.0

    return {
        "sentence_count": len(lengths),
        "avg_sentence_len": round(avg_len, 2),
        "sentence_len_variance": round(var, 2),
        "burstiness_ratio": round(burstiness, 2)
    }


def calculate_vocabulary_richness(text: str) -> float:
    words = re.findall(r"\b[a-zA-Zа-яА-ЯёЁ0-9_-]+\b", text.lower())
    if not words:
        return 0.0
    return round(len(set(words)) / len(words), 2)


AI_BUZZWORDS = [
    "важно отметить", "следует подчеркнуть", "в заключение", "таким образом",
    "кроме того", "является ключевым", "необходимо учитывать", "подводя итог",
    "безусловно", "играет важную роль", "надеюсь, это письмо застанет вас",
    "furthermore", "moreover", "in conclusion", "it is important to note",
    "crucial", "delve", "testament", "tapestry", "in summary", "foster"
]


def count_ai_buzzwords(text: str) -> int:
    text_lower = text.lower()
    return sum(text_lower.count(phrase) for phrase in AI_BUZZWORDS)


def detect_emotional_markers(text: str) -> Dict[str, int]:
    return {
        "multi_punctuation_count": len(re.findall(r"(\!{2,}|\?{2,}|\?\!|\!\?)", text)),
        "bracket_smiles_count": len(re.findall(r"(\){2,}|\({2,})", text)),
        "caps_words_count": len(re.findall(r"\b[A-ZА-ЯЁ]{2,}\b", text))
    }


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "text_classifier.json"
FEATURE_NAMES = (
    "sentence_count", "avg_sentence_len", "sentence_len_variance", "burstiness_ratio",
    "vocab_richness_ttr", "ai_buzzwords_count", "multi_punctuation_count",
    "bracket_smiles_count", "caps_words_count",
)
FEATURE_LABELS = (
    "число предложений", "средняя длина предложения", "дисперсия длины предложений",
    "разброс длины предложений", "разнообразие словаря", "число заданных фраз",
    "повторяющаяся пунктуация", "смайлы-скобки", "слова заглавными буквами",
)


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, value))))


def feature_vector(features: dict) -> list:
    return [math.log1p(max(0.0, float(features[name]))) for name in FEATURE_NAMES]


def fit_logistic(vectors, labels, sample_weights, epochs=300, penalty=0.01):
    weights = [0.0] * len(vectors[0])
    bias = 0.0
    total_weight = sum(sample_weights)
    for _ in range(epochs):
        gradient = [0.0] * len(weights)
        bias_gradient = 0.0
        for vector, label, sample_weight in zip(vectors, labels, sample_weights):
            value = bias + sum(w * x for w, x in zip(weights, vector))
            error = (sigmoid(value) - label) * sample_weight
            bias_gradient += error
            for index, x in enumerate(vector):
                gradient[index] += error * x
        for index in range(len(weights)):
            weights[index] -= 0.1 * (gradient[index] / total_weight + penalty * weights[index])
        bias -= 0.1 * bias_gradient / total_weight
    return weights, bias


class TextClassifier:
    def __init__(self, model: dict):
        if model["feature_names"] != list(FEATURE_NAMES):
            raise ValueError("формат модели изменился, запусти text_analyze.py для обучения")
        self.model = model

    def transform(self, features: dict) -> list:
        return [(x - mean) / scale for x, mean, scale in
                zip(feature_vector(features), self.model["mean"], self.model["scale"])]

    def decision(self, features: dict) -> float:
        return self.model["bias"] + sum(w * x for w, x in
                                        zip(self.model["weights"], self.transform(features)))

    def probability(self, features: dict) -> float:
        return sigmoid(self.model["calibration_weight"] * self.decision(features)
                       + self.model["calibration_bias"])

    def explain(self, features: dict) -> list:
        contributions = [w * x * self.model["calibration_weight"] for w, x in
                         zip(self.model["weights"], self.transform(features))]
        strongest = sorted(range(len(contributions)), key=lambda i: abs(contributions[i]), reverse=True)[:2]
        reasons = []
        for index in strongest:
            if abs(contributions[index]) < 0.01:
                continue
            direction = "повышает" if contributions[index] > 0 else "снижает"
            reasons.append(f"{FEATURE_LABELS[index]}: {features[FEATURE_NAMES[index]]}; "
                           f"{direction} оценку")
        if features["sentence_count"] < 2:
            reasons.append("мало предложений: оценку короткого текста следует трактовать осторожно")
        return reasons or ["признаки близки к средним значениям обучающей выборки"]


def evaluate_text_ai_probability(features: dict) -> Tuple[Optional[float], List[str]]:
    if features["sentence_count"] == 0:
        return None, ["в письме нет текста для оценки"]
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("классификатор ещё не обучен: запусти text_analyze.py")
    classifier = TextClassifier(json.loads(MODEL_PATH.read_text(encoding="utf-8")))
    return round(classifier.probability(features) * 100.0, 1), classifier.explain(features)


def read_training_data(dataset_dir: Path):
    parser = EmailParser()
    groups = {}
    for folder, label in (("human", 0), ("edited", 1)):
        directory = dataset_dir / folder
        if not directory.is_dir():
            raise FileNotFoundError(f"не найдена папка: {directory}")
        for path in sorted(directory.glob("*.eml")):
            match = re.fullmatch(r"(sample_\d+)(?:_.*)?", path.stem)
            if not match:
                raise ValueError(f"не удалось определить пару по имени: {path.name}")
            parsed = parser.parse_from_file(str(path))
            text = clean_email_body(parsed.text_plain or parsed.extracted_text_from_html)
            if not text:
                raise ValueError(f"письмо пустое после очистки: {path}")
            features = extract_all_features(text, include_score=False)
            groups.setdefault(match[1], []).append((features, label))
    if len(groups) < 20 or any({label for _, label in rows} != {0, 1} for rows in groups.values()):
        raise ValueError("нужны хотя бы 20 полных групп с человеческим и отредактированным текстом")
    return groups


def collect_rows(groups, names):
    features, labels, weights = [], [], []
    for name in names:
        rows = groups[name]
        counts = {label: sum(y == label for _, y in rows) for label in (0, 1)}
        for values, label in rows:
            features.append(values)
            labels.append(label)
            # дополнительные версии не увеличивают вес одной исходной записи
            weights.append(1.0 / counts[label])
    return features, labels, weights


def train_classifier(dataset_dir: Path = BASE_DIR / "GPT3.5_dataset") -> dict:
    print("чтение писем и извлечение признаков...", flush=True)
    groups = read_training_data(dataset_dir)
    names = sorted(groups)
    random.Random(42).shuffle(names)
    train_end, calibration_end = int(len(names) * 0.7), int(len(names) * 0.85)
    train_names, calibration_names, test_names = names[:train_end], names[train_end:calibration_end], names[calibration_end:]
    train_features, train_labels, train_weights = collect_rows(groups, train_names)
    calibration_features, calibration_labels, calibration_weights = collect_rows(groups, calibration_names)
    test_features, test_labels, test_weights = collect_rows(groups, test_names)

    # масштаб признаков вычисляется только по обучающим данным
    vectors = [feature_vector(values) for values in train_features]
    total = sum(train_weights)
    mean = [sum(row[i] * w for row, w in zip(vectors, train_weights)) / total for i in range(len(FEATURE_NAMES))]
    scale = [max(math.sqrt(sum(w * (row[i] - mean[i]) ** 2 for row, w in
                              zip(vectors, train_weights)) / total), 1e-8) for i in range(len(FEATURE_NAMES))]
    scaled = [[(x - m) / s for x, m, s in zip(row, mean, scale)] for row in vectors]
    print(f"обучение на {len(train_names)} группах...", flush=True)
    weights, bias = fit_logistic(scaled, train_labels, train_weights)
    model = dict(version=1, feature_names=list(FEATURE_NAMES), mean=mean, scale=scale,
                 weights=weights, bias=bias, calibration_weight=1.0, calibration_bias=0.0)
    classifier = TextClassifier(model)

    # проценты настраиваются на отдельной части данных, тест остаётся нетронутым
    logits = [[classifier.decision(values)] for values in calibration_features]
    calibration_weights_fit, calibration_bias = fit_logistic(logits, calibration_labels,
                                                            calibration_weights, epochs=400, penalty=0.001)
    model.update(calibration_weight=calibration_weights_fit[0], calibration_bias=calibration_bias)
    predictions = [classifier.probability(values) for values in test_features]
    tp = sum(w for p, y, w in zip(predictions, test_labels, test_weights) if p >= 0.5 and y == 1)
    fp = sum(w for p, y, w in zip(predictions, test_labels, test_weights) if p >= 0.5 and y == 0)
    fn = sum(w for p, y, w in zip(predictions, test_labels, test_weights) if p < 0.5 and y == 1)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    metrics = dict(precision=precision, recall=recall,
                   f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    model.update(metrics=metrics, seed=42,
                 split_groups=dict(train=len(train_names), calibration=len(calibration_names), test=len(test_names)))
    MODEL_PATH.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"модель сохранена: {MODEL_PATH}")
    print(f"проверка на {len(test_names)} группах: precision={precision:.3f}, recall={recall:.3f}, "
          f"f1={metrics['f1']:.3f}")
    return metrics


def extract_all_features(cleaned_text: str, include_score: bool = True) -> dict:
    variance_metrics = calculate_sentence_variance(cleaned_text)
    vocab_richness = calculate_vocabulary_richness(cleaned_text)
    buzzwords = count_ai_buzzwords(cleaned_text)
    emotional_flags = detect_emotional_markers(cleaned_text)

    features = {
        **variance_metrics,
        "vocab_richness_ttr": vocab_richness,
        "ai_buzzwords_count": buzzwords,
        **emotional_flags
    }

    if include_score:
        prob, reasons = evaluate_text_ai_probability(features)
        features["text_ai_score"] = prob
        features["text_ai_reasons"] = reasons

    return features


def analyze_email_from_file(file_path: str) -> dict:
    parser = EmailParser()
    parsed = parser.parse_from_file(file_path)
    raw_body = parsed.text_plain if parsed.text_plain else parsed.extracted_text_from_html
    clean_body = clean_email_body(raw_body)
    features = extract_all_features(clean_body)

    return {
        "subject": parsed.subject,
        "cleaned_text": clean_body,
        "images_count": len(parsed.images),
        **features
    }


def analyze_email_from_bytes(raw_bytes: bytes) -> dict:
    parser = EmailParser()
    parsed = parser.parse_from_bytes(raw_bytes)
    raw_body = parsed.text_plain if parsed.text_plain else parsed.extracted_text_from_html
    clean_body = clean_email_body(raw_body)
    features = extract_all_features(clean_body)

    return {
        "subject": parsed.subject,
        "cleaned_text": clean_body,
        "images_count": len(parsed.images),
        **features
    }


if __name__ == "__main__":
    try:
        train_classifier()
    except (OSError, ValueError, KeyError) as error:
        print(f"ошибка обучения: {error}", file=sys.stderr)
        sys.exit(1)
