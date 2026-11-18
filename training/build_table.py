"""
Собирает признаки писем из datasets/PASTED/data, сохраняет группы и разбиения.
Объединяет стилометрические признаки, локальные скоры и нейросетевой анализ (DetectGPT).
"""

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "analysis"))

import pandas as pd
from parser import EmailParser, clean_email_body
from text_analyze import extract_all_features, dataset_files
from neural_analyze import get_detector


def build_meta_dataset(limit_per_class: int = None, use_neural: bool = True):
    base_dir = BASE_DIR
    fast_dir = Path("/content/datasets/PASTED/data")
    dataset_dir = fast_dir if fast_dir.is_dir() else (base_dir / "datasets" / "PASTED" / "data")

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Датасет не найден в {dataset_dir}! Сначала запустите скрипт загрузки.")

    tasks = list(dataset_files(dataset_dir))
    parser = EmailParser()

    if use_neural:
        print(">>> Загрузка нейросетевого детектора (DistilBERT + GPT-2)...")
        neural_detector = get_detector()

    rows = []
    print(">>> Начинаем обработку писем для таблицы...\n")

    counts = {}
    for idx, (file_path, label, group, split) in enumerate(tasks, 1):
        category = (split, label)
        if limit_per_class and counts.get(category, 0) >= limit_per_class:
            continue

        try:
            parsed = parser.parse_from_file(str(file_path))
            raw_body = parsed.text_plain or parsed.extracted_text_from_html
            cleaned_body = clean_email_body(raw_body)

            # 1. Стилометрические признаки (Текстовый блок)
            text_features = extract_all_features(cleaned_body, include_score=True)

            # 2. Нейросетевые признаки (DetectGPT блок)
            if use_neural:
                neural_features = neural_detector.evaluate(cleaned_body, n_perturbations=4)
            else:
                neural_features = {
                    "neural_discrepancy": 0.0,
                    "neural_original_loss": 0.0,
                    "neural_ai_score": 50.0
                }

            row = {
                "filename": file_path.name,
                "group_id": group.rsplit(":", 1)[-1],
                "split": split,
                # Текстовые признаки:
                "text_ai_score": text_features.get("text_ai_score", 50.0),
                "burstiness_ratio": text_features["burstiness_ratio"],
                "ai_buzzwords_count": text_features["ai_buzzwords_count"],
                "vocab_richness_ttr": text_features["vocab_richness_ttr"],
                "sentence_count": text_features["sentence_count"],
                # Нейросетевые признаки:
                "neural_discrepancy": neural_features["neural_discrepancy"],
                "neural_original_loss": neural_features["neural_original_loss"],
                "neural_ai_score": neural_features["neural_ai_score"],
                # Таргет:
                "target": label
            }
            rows.append(row)
            counts[category] = counts.get(category, 0) + 1

            if len(rows) % 25 == 0:
                print(f"Обработано {len(rows)} писем...")

        except Exception as e:
            print(f"Ошибка при обработке {file_path.name}: {e}")

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
    cli = argparse.ArgumentParser()
    cli.add_argument("--limit", type=int, default=100, help="Ограничение писем на класс в каждом сплите (по умолчанию 100)")
    cli.add_argument("--no-neural", action="store_true", help="Отключить нейросетевой блок для ускорения")
    args = cli.parse_args()

    build_meta_dataset(limit_per_class=args.limit, use_neural=not args.no_neural)
