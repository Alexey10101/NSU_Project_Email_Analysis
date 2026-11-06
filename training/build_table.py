"""
собирает текстовые и html-признаки для обучения catboost.
сохраняет исходные группы, train/validation/test и описание таблицы.
готовые оценки моделей не используются, чтобы избежать утечки обучающих данных.
"""

import argparse
import csv
import hashlib
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "analysis"))

from boosting_analyze import FEATURE_NAMES, email_features
from model_utils import read_dataset, save_json, split_summary


TABLE_PATH = BASE_DIR / "training" / "data" / "dataset_table.csv"
DATASET_PATH = BASE_DIR / "datasets" / "PASTED" / "data"


def build_meta_dataset(limit_per_class=None, dataset_dir=DATASET_PATH, output_path=TABLE_PATH):
    datasets = dataset_dir if isinstance(dataset_dir, (list, tuple)) else [dataset_dir]
    records = []
    for directory in datasets:
        rows = read_dataset(directory)
        if limit_per_class is not None:
            if limit_per_class < 1:
                raise ValueError("ограничение должно быть положительным")
            selected = {split: sorted({row["group_id"] for row in rows if row["split"] == split})[:limit_per_class]
                        for split in ("train", "validation", "test")}
            rows = [row for row in rows if row["group_id"] in selected[row["split"]]]
        records.extend(rows)
    if len({row["group_id"] for row in records}) < 20:
        raise ValueError("для таблицы нужны хотя бы 20 групп")
    print("чтение писем и извлечение признаков...", flush=True)
    rows = []
    hashes = {}
    group_splits = {}
    for record in records:
        split, group = record["split"], record["group_id"]
        if group in group_splits and group_splits[group] != split:
            raise ValueError("группа пересекается между источниками и разбиениями")
        group_splits[group] = split
        raw = record["path"].read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest in hashes:
            if hashes[digest] != split:
                raise ValueError("одинаковое письмо попало в разные разбиения")
            raise ValueError("одинаковое письмо включено в таблицу несколько раз")
        hashes[digest] = split
        _, features = email_features(raw)
        if features["text_sentence_count"] == 0 and not features["has_html"]:
            raise ValueError(f"нет содержимого для обучения: {record['filename']}")
        rows.append(dict(filename=record["filename"], group_id=group, split=split, target=record["target"],
                         sample_weight=record["weight"], source_sha256=digest, **features))
        if len(rows) % 500 == 0:
            print(f"обработано {len(rows)}/{len(records)} писем...", flush=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["filename", "group_id", "split", "target", "sample_weight", "source_sha256", *FEATURE_NAMES]
    temporary = output_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output_path)
    metadata = dict(version=1, feature_names=list(FEATURE_NAMES), datasets=[str(Path(p).resolve()) for p in datasets],
                    table_sha256=hashlib.sha256(output_path.read_bytes()).hexdigest(), splits=split_summary(records),
                    mode="feature_baseline", scores_used=False,
                    note="это признаки доступных блоков; нейросетевые оценки и изображения ещё не объединяются")
    save_json(output_path.with_suffix(".json"), metadata)
    print(f"таблица: {output_path}; строк: {len(rows)}")
    return metadata


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dataset", nargs="+", type=Path, default=[DATASET_PATH])
    cli.add_argument("--output", type=Path, default=TABLE_PATH)
    cli.add_argument("--limit", type=int, help="групп в каждом разбиении; пара всегда остаётся целой")
    args = cli.parse_args()
    try:
        build_meta_dataset(args.limit, args.dataset, args.output)
    except (OSError, ValueError, KeyError) as error:
        print(f"ошибка подготовки таблицы: {error}", file=sys.stderr)
        sys.exit(1)
