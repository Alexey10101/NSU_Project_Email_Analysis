"""
Считает длины предложений, разнообразие словаря, заданные фразы и эмоциональные маркеры.
Использует парсер для чтения письма и очистки текста.
Выдает локальную вероятность генерации текста ИИ (0-100%) и понятные объяснения (explainability).
"""

import re
import statistics
from typing import Dict, List, Tuple

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



import math

def evaluate_text_ai_probability(features: dict) -> Tuple[float, List[str]]:
    """
    Математически строгий расчет вероятности ИИ на основе логистической кривой (Log-Odds)
    с непрерывной зависимостью от дисперсии и степенной эскалацией повторяющихся признаков.
    """

    z = 0.0
    reasons = []

    sentence_count = features["sentence_count"]
    if sentence_count < 2:
        return 50.0, ["Письмо слишком короткое (меньше 2 предложений) — статистический анализ нейтрален"]


    b = features["burstiness_ratio"]
    b_neutral = 0.55

    z_burst = 2.0 * ((b_neutral - b) / (b_neutral + b + 0.1))
    z += z_burst

    if z_burst > 0.3:
        reasons.append(f"Монотонный ритм предложений (Burstiness={b}) плавно повышает риск ИИ")
    elif z_burst < -0.3:
        reasons.append(f"Живой, неравномерный темп текста (Burstiness={b}) указывает на человека")


    buzz_count = features["ai_buzzwords_count"]
    if buzz_count > 0:
        z_buzz = 0.55 * (buzz_count ** 1.35)
        z += z_buzz
        reasons.append(f"Обнаружены клише ИИ ({buzz_count} шт.): кумулятивный вклад +{round(z_buzz, 2)} к лог-шансам")

    human_signals = (
        features["multi_punctuation_count"] * 1.0 +
        features["bracket_smiles_count"] * 1.2 +
        features["caps_words_count"] * 0.7
    )

    if human_signals > 0:
        
        z_human = -0.75 * (human_signals ** 1.3)
        z += z_human
        reasons.append(f"Эмоциональные маркеры человека ({human_signals:.1f} балла хаоса) резко снижают вероятность ИИ")


    avg_len = features["avg_sentence_len"]
    
    if 10.0 <= avg_len <= 20.0:
        distance_from_ideal = abs(avg_len - 15.0)
        z_len = max(0.0, 0.4 - (distance_from_ideal * 0.08))
        if z_len > 0.1 and b < 0.5:
            z += z_len
            reasons.append(f"Средняя длина предложений ({avg_len} слов) типична для LLM-структур")

    z_clamped = max(-6.0, min(6.0, z))
    probability = 1.0 / (1.0 + math.exp(-z_clamped))
    percentage = round(probability * 100.0, 1)

    return percentage, reasons


def extract_all_features(cleaned_text: str) -> dict:
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
