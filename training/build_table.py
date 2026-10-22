"""
собирает признаки писем из datasets/PASTED/data, сохраняет группы и разбиения,
извлекает локальные оценки и сохраняет компактную таблицу для бустинга.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "analysis"))
import pandas as pd
from text_analyze import analyze_email_from_file, dataset_files


def build_meta_dataset(limit_per_class: int = None):
    base_dir = BASE_DIR
    dataset_dir = base_dir / "datasets" / "PASTED" / "data"

    tasks = list(dataset_files(dataset_dir))

    rows = []

    print(">>> Начинаем обработку писем для таблицы...\n")

    counts = {}
    for file_path, label, group, split in tasks:
        category = (split, label)
        if limit_per_class and counts.get(category, 0) >= limit_per_class:
            continue
        features = analyze_email_from_file(str(file_path))
        rows.append({
            "filename": file_path.name,
            "group_id": group.rsplit(":", 1)[-1],
            "split": split,
            "text_ai_score": features["text_ai_score"],
            "burstiness_ratio": features["burstiness_ratio"],
            "ai_buzzwords_count": features["ai_buzzwords_count"],
            "vocab_richness_ttr": features["vocab_richness_ttr"],
            "sentence_count": features["sentence_count"],
            "target": label
        })
        counts[category] = counts.get(category, 0) + 1
        if len(rows) % 100 == 0:
            print(f"обработано {len(rows)} писем...")

    df = pd.DataFrame(rows)
    output_path = base_dir / "training" / "dataset_table.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")

    print("\n" + "=" * 50)
    print(f"Таблица успешно создана! Сохранено строк: {len(df)}")
    print(f"Файл сохранен: {output_path}")
    print("=" * 50)
    print("\nПервые 5 строк таблицы:")
    print(df.head())


if __name__ == "__main__":
    build_meta_dataset(limit_per_class=None)