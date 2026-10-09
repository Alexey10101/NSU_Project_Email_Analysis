"""
Считает длины предложений, разнообразие словаря, заданные фразы и эмоциональные маркеры.
Использует парсер для чтения письма и очистки текста, возвращает словарь признаков.
"""

import re
import statistics
from typing import Dict

from parser import EmailParser, clean_email_body


def calculate_sentence_variance(text: str) -> Dict[str, float]:
    # делим текст на предложения и считаем слова
    raw_sentences = re.split(r"[.!?]+(?:\s+|$)", text)

    lengths = [len(s.strip().split()) for s in raw_sentences if len(s.strip().split()) > 0]

    # для дисперсии нужны хотя бы два предложения
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
    # приводим слова к нижнему регистру, знаки препинания не учитываем
    words = re.findall(r"\b[a-zA-Zа-яА-ЯёЁ0-9_-]+\b", text.lower())

    total_words = len(words)
    if total_words == 0:
        return 0.0

    unique_words = len(set(words))

    ttr = unique_words / total_words
    return round(ttr, 2)


# фразы для отдельного текстового признака
AI_BUZZWORDS = [

    "важно отметить", "следует подчеркнуть", "в заключение", "таким образом",
    "кроме того", "является ключевым", "необходимо учитывать", "подводя итог",
    "безусловно", "играет важную роль", "надеюсь, это письмо застанет вас",

    "furthermore", "moreover", "in conclusion", "it is important to note",
    "crucial", "delve", "testament", "tapestry", "in summary", "foster"
]


def count_ai_buzzwords(text: str) -> int:
    text_lower = text.lower()
    matches_count = 0

    for phrase in AI_BUZZWORDS:
        matches_count += text_lower.count(phrase)

    return matches_count


def detect_emotional_markers(text: str) -> Dict[str, int]:
    # повторяющиеся знаки, смайлы-скобки и слова заглавными буквами
    multi_punct = len(re.findall(r"(\!{2,}|\?{2,}|\?\!|\!\?)", text))

    bracket_smiles = len(re.findall(r"(\){2,}|\({2,})", text))

    caps_words = len(re.findall(r"\b[A-ZА-ЯЁ]{2,}\b", text))

    return {
        "multi_punctuation_count": multi_punct,
        "bracket_smiles_count": bracket_smiles,
        "caps_words_count": caps_words
    }


def extract_all_features(cleaned_text: str) -> dict:
    # собираем все текстовые признаки в один словарь
    variance_metrics = calculate_sentence_variance(cleaned_text)
    vocab_richness = calculate_vocabulary_richness(cleaned_text)
    buzzwords = count_ai_buzzwords(cleaned_text)
    emotional_flags = detect_emotional_markers(cleaned_text)

    return {
        **variance_metrics,
        "vocab_richness_ttr": vocab_richness,
        "ai_buzzwords_count": buzzwords,
        **emotional_flags
    }


def analyze_email_from_bytes(raw_bytes: bytes) -> dict:
    parser = EmailParser()
    parsed = parser.parse_from_bytes(raw_bytes)

    # обычный текст в приоритете, иначе берём текст из html
    raw_body = parsed.text_plain if parsed.text_plain else parsed.extracted_text_from_html

    clean_body = clean_email_body(raw_body)

    features = extract_all_features(clean_body)

    return {
        "subject": parsed.subject,
        "cleaned_text": clean_body,
        "images_count": len(parsed.images),
        **features
    }


def analyze_email_from_file(file_path: str) -> dict:
    parser = EmailParser()
    parsed = parser.parse_from_file(file_path)
    # обычный текст в приоритете, иначе берём текст из html
    raw_body = parsed.text_plain if parsed.text_plain else parsed.extracted_text_from_html
    clean_body = clean_email_body(raw_body)
    features = extract_all_features(clean_body)

    return {
        "subject": parsed.subject,
        "cleaned_text": clean_body,
        "images_count": len(parsed.images),
        **features
    }
