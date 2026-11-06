"""
С помощью random выбирает человеческое письмо, ищет ему соответствующее редактированное ИИ,
парсит их и выводит все признаки анализа.
"""

import random
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "analysis"))
from parser import EmailParser
from text_analyze import analyze_email_from_file
from html_analyze import analyze_email_from_file as analyze_html
from boosting_analyze import analyze_email_from_file as analyze_boosting


def compare_pair(dataset_dir: Path, filename: str) -> None:
    paths = {
        "Написано человеком": dataset_dir / "human" / filename,
        "Отредактировано GPT-3.5": dataset_dir / "edited" / filename,
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Не найдено письмо пары: {path}")

    parser = EmailParser()
    for label, path in paths.items():
        parsed = parser.parse_from_file(str(path))
        features = analyze_email_from_file(str(path))
        print("=" * 60)
        print(f"{label}: {path.name}")
        print(f"Папка: {path.parent.name}")
        print(f"Полный путь: {path}")
        print(f"Тема: {parsed.subject}")
        print(f"Отправитель: {parsed.sender}")
        print(f"Получатели: {', '.join(parsed.to)}")
        print(f"Дата: {parsed.date}")
        raw_body = parsed.text_plain or parsed.extracted_text_from_html
        print("\nТекст письма:")
        print(raw_body)
        if features["cleaned_text"] != raw_body:
            print("\nТекст после очистки:")
            print(features["cleaned_text"])
        print("\nПризнаки анализа:")
        for key, value in features.items():
            if key not in {"subject", "cleaned_text"}:
                print(f"{key:26}: {value}")
        html = analyze_html(str(path))
        print("\nHTML-анализ:")
        if html["html_ai_score"] is None:
            print("; ".join(html["html_ai_reasons"]))
        else:
            print(f"уверенность в ИИ-правках: {html['html_ai_score']}%")
            print("; ".join(html["html_ai_reasons"]))
        boosting = analyze_boosting(str(path))
        print("\nCatBoost:")
        if boosting["ai_score"] is not None:
            print(f"уверенность в ИИ-правках: {boosting['ai_score']}%")
        print("; ".join(row["reason"] for row in boosting["reasons"]))
        if boosting.get("missing_blocks"):
            print("пока не участвуют: " + ", ".join(boosting["missing_blocks"]))


if __name__ == "__main__":
    base_dir = BASE_DIR
    dataset_dir = base_dir / "datasets" / "PASTED" / "data" / "test"
    human_emails = sorted(path for path in (dataset_dir / "human").glob("*.eml") if path.is_file())
    if not human_emails:
        raise FileNotFoundError(f"В {dataset_dir / 'human'} нет писем. Сначала запусти скрипт загрузки.")
    selected_human = random.choice(human_emails)
    print(f"Случайно выбрана пара: {selected_human.name}")
    compare_pair(dataset_dir, selected_human.name)
