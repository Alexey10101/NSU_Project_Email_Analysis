import re
import statistics
from typing import Dict, List

# Импортируем наш парсер и модуль очистки текста из соседнего файла parser.py
from parser import EmailParser, clean_email_body


# =====================================================================
# 1. ПРОВЕРКА: ДИСПЕРСИЯ И РИТМ ПРЕДЛОЖЕНИЙ (BURSTINESS)
# =====================================================================
def calculate_sentence_variance(text: str) -> Dict[str, float]:
    """
    Анализирует дисперсию длины предложений.
    
    """
    # Разбиваем текст на предложения по знакам завершения (. ! ?)
    raw_sentences = re.split(r"[.!?]+(?:\s+|$)", text)
    
    # Считаем количество слов в каждом непустом предложении
    lengths = [len(s.strip().split()) for s in raw_sentences if len(s.strip().split()) > 0]

    # Если в тексте меньше двух предложений, математически дисперсию посчитать нельзя
    if len(lengths) < 2:
        return {
            "sentence_count": len(lengths),
            "avg_sentence_len": float(lengths[0]) if lengths else 0.0,
            "sentence_len_variance": 0.0,
            "burstiness_ratio": 0.0
        }

    # Математические расчеты
    avg_len = statistics.mean(lengths)                 # Средняя длина предложения
    var = statistics.variance(lengths)                 # Дисперсия (разброс длин в квадрате)
    std = statistics.stdev(lengths)                    # Стандартное отклонение (разброс в словах)
    burstiness = std / avg_len if avg_len > 0 else 0.0 # Коэффициент вариации (главный нормализованный признак)

    return {
        "sentence_count": len(lengths),
        "avg_sentence_len": round(avg_len, 2),
        "sentence_len_variance": round(var, 2),
        "burstiness_ratio": round(burstiness, 2)
    }


# =====================================================================
# 2. ПРОВЕРКА: БОГАТСТВО СЛОВАРЯ (TYPE-TOKEN RATIO / TTR)
# =====================================================================
def calculate_vocabulary_richness(text: str) -> float:
    """
    Считает коэффициент лексического разнообразия (TTR):
    Отношение уникальных слов ко всем словам в тексте.
    
    """
    # Выделяем только слова (игнорируем знаки препинания и приводим всё к нижнему регистру)
    words = re.findall(r"\b[a-zA-Zа-яА-ЯёЁ0-9_-]+\b", text.lower())
    
    total_words = len(words)
    if total_words == 0:
        return 0.0

    # Множество set() автоматически оставляет только уникальные слова
    unique_words = len(set(words))
    
    # Формула: Уникальные слова / Всего слов
    ttr = unique_words / total_words
    return round(ttr, 2)


# =====================================================================
# 3. ПРОВЕРКА: СЛОВА-КЛИШЕ НЕЙРОСЕТЕЙ (AI BUZZWORDS)
# =====================================================================
# Список характерных оборотов, которыми злоупотребляют ChatGPT, Claude и другие LLM
AI_BUZZWORDS = [
    # Русскоязычные клише
    "важно отметить", "следует подчеркнуть", "в заключение", "таким образом",
    "кроме того", "является ключевым", "необходимо учитывать", "подводя итог",
    "безусловно", "играет важную роль", "надеюсь, это письмо застанет вас",
    
    # Англоязычные клише
    "furthermore", "moreover", "in conclusion", "it is important to note",
    "crucial", "delve", "testament", "tapestry", "in summary", "foster"
]

def count_ai_buzzwords(text: str) -> int:
    """
    Считает количество найденных шаблонных фраз, типичных для нейросетей.

    """
    text_lower = text.lower()
    matches_count = 0

    for phrase in AI_BUZZWORDS:
        # Считаем, сколько раз данная фраза встретилась в тексте
        matches_count += text_lower.count(phrase)

    return matches_count


# =====================================================================
# 4. ПРОВЕРКА: ЭМОЦИОНАЛЬНЫЕ ЗНАКИ И ЧЕЛОВЕЧЕСКИЙ ХАОС
# =====================================================================
def detect_emotional_markers(text: str) -> Dict[str, int]:
    """
    Ищет признаки неформального человеческого письма:
    повторяющиеся знаки, смайлы-скобочки и слова капсом.
    """
    # 1. Поиск множественных знаков: ??, !!!, ?!, !?
    multi_punct = len(re.findall(r"(\!{2,}|\?{2,}|\?\!|\!\?)", text))

    # 2. Поиск текстовых смайлов-скобочек: )))) или ((((
    bracket_smiles = len(re.findall(r"(\){2,}|\({2,})", text))

    # 3. Поиск слов КАПСОМ длиной от 2 букв (например: "СРОЧНО", "СПАСИБО", "ASAP")
    # Проверяем, чтобы это было отдельное слово из заглавных букв
    caps_words = len(re.findall(r"\b[A-ZА-ЯЁ]{2,}\b", text))

    return {
        "multi_punctuation_count": multi_punct,
        "bracket_smiles_count": bracket_smiles,
        "caps_words_count": caps_words
    }


# =====================================================================
# 5. СКВОЗНОЙ АНАЛИЗАТОР (ОБЪЕДИНЯЕТ ВСЕ ПРОВЕРКИ ДЛЯ ТАБЛИЦЫ)
# =====================================================================
def extract_all_features(cleaned_text: str) -> dict:
    """Запускает все проверки на очищенном тексте и собирает словарь признаков."""
    variance_metrics = calculate_sentence_variance(cleaned_text)
    vocab_richness = calculate_vocabulary_richness(cleaned_text)
    buzzwords = count_ai_buzzwords(cleaned_text)
    emotional_flags = detect_emotional_markers(cleaned_text)

    # Объединяем все вычисленные признаки в одну строчку
    return {
        **variance_metrics,
        "vocab_richness_ttr": vocab_richness,
        "ai_buzzwords_count": buzzwords,
        **emotional_flags
    }


def analyze_email_from_bytes(raw_bytes: bytes) -> dict:
    """Полный цикл: от сырых байт письма до строки с признаками."""
    # Шаг 1: Парсим структуру письма
    parser = EmailParser()
    parsed = parser.parse_from_bytes(raw_bytes)

    # Шаг 2: Берем текст (plain в приоритете, либо вычищенный из HTML)
    raw_body = parsed.text_plain if parsed.text_plain else parsed.extracted_text_from_html

    # Шаг 3: Отрезаем цитаты, историю переписки и 'Sent from my iPhone'
    clean_body = clean_email_body(raw_body)

    # Шаг 4: Считаем все наши метрики
    features = extract_all_features(clean_body)

    return {
        "subject": parsed.subject,
        "cleaned_text": clean_body,
        "images_count": len(parsed.images),
        **features
    }


def analyze_email_from_file(file_path: str) -> dict:
    """То же самое, но принимает путь к файлу .eml на компьютере."""
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
