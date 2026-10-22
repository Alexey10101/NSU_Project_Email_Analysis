"""
Скрипт проходит по папкам 'human' и 'edited' датасета GPT3.5_dataset,
извлекает локальные оценки и сохраняет компактную таблицу для бустинга.
"""

from pathlib import Path
import pandas as pd
from text_analyze import analyze_email_from_file


def build_meta_dataset(limit_per_class: int = None):
    base_dir = Path(__file__).resolve().parent
    dataset_dir = base_dir / "GPT3.5_dataset"

    human_dir = dataset_dir / "human"
    edited_dir = dataset_dir / "edited"

    if not human_dir.exists() or not edited_dir.exists():
        raise FileNotFoundError(
            f"Не найден датасет в {dataset_dir}. Сначала запустите загрузку датасета!"
        )

    tasks = [
        (human_dir, 0, "Человек"),
        (edited_dir, 1, "ИИ (GPT-3.5)")
    ]

    rows = []

    print(">>> Начинаем обработку писем для таблицы...\n")

    for folder, label, label_name in tasks:
        files = sorted(p for p in folder.glob("*.eml") if p.is_file())
        
        if limit_per_class:
            files = files[:limit_per_class]

        print(f"Обработка категории '{label_name}' (всего файлов: {len(files)})...")

        for idx, file_path in enumerate(files, 1):
            try:
                features = analyze_email_from_file(str(file_path))


                rows.append({
                    "filename": file_path.name,
                    "text_ai_score": features["text_ai_score"],       # Наш главный локальный скор (0-100%)
                    "burstiness_ratio": features["burstiness_ratio"], # Ритм предложений
                    "ai_buzzwords_count": features["ai_buzzwords_count"], # Количество клише
                    "vocab_richness_ttr": features["vocab_richness_ttr"], # Богатство словаря
                    "sentence_count": features["sentence_count"],     # Длина текста (в предложениях)
                    "target": label                                   # Метка: 0 (человек) или 1 (ИИ)
                })

                if idx % 100 == 0:
                    print(f"  Обработано {idx}/{len(files)}...")

            except Exception as e:
                print(f"Ошибка при обработке {file_path.name}: {e}")

    df = pd.DataFrame(rows)
    output_path = base_dir / "dataset_table.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")

    print("\n" + "=" * 50)
    print(f"Таблица успешно создана! Сохранено строк: {len(df)}")
    print(f"Файл сохранен: {output_path}")
    print("=" * 50)
    print("\nПервые 5 строк таблицы:")
    print(df.head())


if __name__ == "__main__":
    build_meta_dataset(limit_per_class=None)