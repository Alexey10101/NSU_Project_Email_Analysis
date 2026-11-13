"""
извлекает общие признаки письма и применяет сохранённую модель catboost.
возвращает проценты, итоговый индекс, вердикт и вклад признаков в json.
отсутствующие нейросетевые оценки не подменяются случайными числами.
"""

import argparse
import hashlib
import json
import math
import sys
from email.errors import MessageError
from pathlib import Path

from parser import EmailParser, clean_email_body
from text_analyze import FEATURE_NAMES as TEXT_FEATURES, FEATURE_LABELS as TEXT_LABELS, extract_all_features, sigmoid
from html_analyze import FEATURE_NAMES as HTML_FEATURES, FEATURE_LABELS as HTML_LABELS, extract_all_features as extract_html_features
from model_utils import analyze_path, print_error, print_json


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "catboost_email_model.json"
CALIBRATION_PATH = BASE_DIR / "models" / "catboost_calibration.json"
METADATA_PATH = BASE_DIR / "models" / "catboost_metadata.json"
FEATURE_NAMES = tuple("text_" + name for name in TEXT_FEATURES) + tuple("html_" + name for name in HTML_FEATURES) + ("has_html",)
FEATURE_LABELS = dict(zip(FEATURE_NAMES, (*TEXT_LABELS, *HTML_LABELS, "наличие html")))


def email_features(raw_bytes):
    parsed = EmailParser().parse_from_bytes(raw_bytes)
    text = clean_email_body(parsed.text_plain or parsed.extracted_text_from_html)
    text_values = extract_all_features(text, include_score=False)
    html_values = extract_html_features(parsed.text_html, parsed.headers, parsed.text_plain)
    features = {"text_" + name: float(text_values[name]) for name in TEXT_FEATURES}
    features.update({"html_" + name: float(html_values[name]) for name in HTML_FEATURES})
    features["has_html"] = int(bool(parsed.text_html))
    return parsed, features


class BoostingClassifier:
    def __init__(self, model_path=MODEL_PATH, calibration_path=CALIBRATION_PATH, metadata_path=METADATA_PATH):
        from catboost import CatBoostClassifier

        self.metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        self.calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
        digest = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
        if digest != self.metadata["model_sha256"] or digest != self.calibration["model_sha256"]:
            raise ValueError("модель catboost, описание и калибровка не совпадают")
        if self.metadata.get("version") != 1 or not all(name in FEATURE_NAMES for name in self.metadata["feature_names"]):
            raise ValueError("формат признаков catboost изменился, нужно повторное обучение")
        if not 0 <= self.calibration["threshold"] <= 1:
            raise ValueError("некорректный порог catboost")
        if not all(math.isfinite(self.calibration[name]) for name in ("weight", "bias")) or self.calibration["weight"] < 0:
            raise ValueError("повреждены параметры калибровки catboost")
        self.model = CatBoostClassifier()
        self.model.load_model(str(model_path), format="json")
        if self.model.feature_names_ != self.metadata["feature_names"]:
            raise ValueError("порядок признаков не совпадает с моделью")

    def vector(self, features):
        values = [float(features[name]) for name in self.metadata["feature_names"]]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("в признаках найдены некорректные числа")
        return values

    def evaluate(self, features, threshold=None):
        from catboost import Pool

        vector = self.vector(features)
        pool = Pool([vector], feature_names=self.metadata["feature_names"])
        logit = float(self.model.predict(pool, prediction_type="RawFormulaVal")[0])
        probability = sigmoid(self.calibration["weight"] * logit + self.calibration["bias"])
        threshold = self.calibration["threshold"] if threshold is None else threshold
        if not 0 <= threshold <= 1:
            raise ValueError("порог должен быть от 0 до 1")
        contributions = self.model.get_feature_importance(pool, type="ShapValues", thread_count=2)[0][:-1]
        strongest = sorted(range(len(contributions)), key=lambda i: abs(contributions[i]), reverse=True)[:3]
        reasons = []
        for index in strongest:
            contribution = float(contributions[index]) * self.calibration["weight"]
            if abs(contribution) < 0.001:
                continue
            name = self.metadata["feature_names"][index]
            direction = "повышает" if contribution > 0 else "снижает"
            reasons.append(dict(feature=name, value=vector[index], contribution=round(contribution, 6),
                                reason=f"{FEATURE_LABELS[name]}: {vector[index]:g}; {direction} оценку"))
        return dict(ai_index=round(probability, 6), ai_score=round(probability * 100, 1),
                    verdict="AI-Edited" if probability >= threshold else "Human-Written", threshold=threshold,
                    reasons=reasons or [dict(reason="признаки близки к средним значениям обучающей выборки")])


def analyze_email_from_bytes(raw_bytes, classifier=None, threshold=None):
    parsed, features = email_features(raw_bytes)
    result = dict(subject=parsed.subject, features=features, parse_warnings=parsed.warnings,
                  ai_index=None, ai_score=None, verdict=None, status="not_trained",
                  missing_blocks=["нейросетевой анализ текста", "оценка изображений"], reasons=[])
    if features["text_sentence_count"] == 0 and not features["has_html"]:
        result.update(status="no_content", reasons=[dict(reason="нет текста или html для оценки")])
        return result
    if classifier is None:
        if not all(path.is_file() for path in (MODEL_PATH, CALIBRATION_PATH, METADATA_PATH)):
            result["reasons"] = [dict(reason="catboost ещё не обучен: запусти training/train_boosting.py")]
            return result
        classifier = BoostingClassifier()
    result.update(classifier.evaluate(features, threshold), status="ok", method=classifier.metadata["mode"],
                  used_blocks=classifier.metadata["used_blocks"])
    if "html_features" not in classifier.metadata["used_blocks"]:
        result["missing_blocks"].append("обученная оценка html")
    if result["missing_blocks"]:
        result["status"] = "partial"
    return result


def analyze_email_from_file(file_path, **options):
    return analyze_email_from_bytes(Path(file_path).read_bytes(), **options)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("file", type=Path, help="письмо или папка с eml")
    cli.add_argument("--output", type=Path)
    cli.add_argument("--threshold", type=float)
    args = cli.parse_args()
    try:
        classifier = BoostingClassifier() if MODEL_PATH.is_file() else None
        result = analyze_path(args.file, analyze_email_from_file, classifier=classifier, threshold=args.threshold)
        print_json(result, args.output)
        return 0
    except (OSError, ValueError, KeyError, ImportError, MessageError, RecursionError) as error:
        print_error(error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
