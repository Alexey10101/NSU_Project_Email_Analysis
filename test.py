"""
С помощью random выбирает человеческое письмо, затем изет ему соответствующее редактированное ИИ,
парсит их и выводит все признаки анализа.
"""

import random
from pathlib import Path
from parser import EmailParser
from text_analyze import analyze_email_from_file


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


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    dataset_dir = base_dir / "GPT3.5_dataset"
    human_emails = sorted(path for path in (dataset_dir / "human").glob("*.eml") if path.is_file())
    if not human_emails:
        raise FileNotFoundError(f"В {dataset_dir / 'human'} нет писем. Сначала запусти скрипт загрузки.")
    selected_human = random.choice(human_emails)
    print(f"Случайно выбрана пара: {selected_human.name}")
    compare_pair(dataset_dir, selected_human.name)
