"""
обучает catboost на таблице признаков, калибрует проценты по validation.
сохраняет модель и калибровку в models, метрики в reports, журналы в logs.
test используется только для итоговой проверки, группы писем не пересекаются.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "analysis"))

from boosting_analyze import FEATURE_NAMES
from model_utils import calculate_metrics, calibrate, save_json
from text_analyze import sigmoid


TABLE_PATH = BASE_DIR / "training" / "data" / "dataset_table.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_PATH = BASE_DIR / "training" / "reports" / "catboost_metrics.json"
LOG_DIR = BASE_DIR / "training" / "logs" / "catboost"


def read_table(table_path):
    table_path = Path(table_path)
    if not table_path.is_file():
        raise ValueError("сначала запусти training/build_table.py")
    description = json.loads(table_path.with_suffix(".json").read_text(encoding="utf-8"))
    if description.get("version") != 1 or description.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("формат таблицы изменился, нужно пересоздать её через build_table.py")
    digest = hashlib.sha256(table_path.read_bytes()).hexdigest()
    if description["table_sha256"] != digest:
        raise ValueError("таблица не совпадает с описанием, нужно пересоздать её")
    groups = defaultdict(list)
    source_splits = {}
    with table_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["target"] = int(row["target"])
        row["sample_weight"] = float(row["sample_weight"])
        if row["target"] not in (0, 1) or row["split"] not in ("train", "validation", "test"):
            raise ValueError("некорректная метка или разбиение в таблице")
        for name in FEATURE_NAMES:
            row[name] = float(row[name])
            if not math.isfinite(row[name]):
                raise ValueError(f"некорректное число в признаке {name}")
        if not math.isfinite(row["sample_weight"]) or row["sample_weight"] <= 0:
            raise ValueError("веса должны быть положительными")
        digest = row["source_sha256"]
        if digest in source_splits and source_splits[digest] != row["split"]:
            raise ValueError("одинаковое письмо попало в разные разбиения")
        source_splits[digest] = row["split"]
        groups[row["group_id"]].append(row)
    if len(groups) < 20:
        raise ValueError("нужны хотя бы 20 групп писем")
    for group, values in groups.items():
        if len({row["split"] for row in values}) != 1 or {row["target"] for row in values} != {0, 1}:
            raise ValueError(f"разбиение или состав группы нарушены: {group}")
        for label in (0, 1):
            selected = [row for row in values if row["target"] == label]
            if any(abs(row["sample_weight"] - 1 / len(selected)) > 1e-9 for row in selected):
                raise ValueError("вес оригинала и его вариантов не сбалансирован")
    parts = {split: [row for row in rows if row["split"] == split] for split in ("train", "validation", "test")}
    if any({row["target"] for row in values} != {0, 1} for values in parts.values()):
        raise ValueError("в каждом разбиении должны быть оба класса")
    return parts, description


def train_model(table_path=TABLE_PATH, model_dir=MODEL_DIR, report_path=REPORT_PATH, log_dir=LOG_DIR,
                iterations=300, max_fpr=0.2):
    from catboost import CatBoostClassifier, Pool
    import catboost

    if iterations < 1 or not 0 <= max_fpr <= 1:
        raise ValueError("iterations должен быть положительным, max_fpr — от 0 до 1")
    parts, description = read_table(table_path)
    train = parts["train"]
    html_labels = {row["target"] for row in train if row["has_html"]}
    html_groups = {row["group_id"] for row in train if row["has_html"]}
    has_html_training = html_labels == {0, 1} and len(html_groups) >= 20
    features = [name for name in FEATURE_NAMES if name.startswith("text_") or has_html_training]
    used_blocks = ["text_features"] + (["html_features"] if has_html_training else [])
    pools = {split: Pool([[row[name] for name in features] for row in rows],
                         label=[row["target"] for row in rows], weight=[row["sample_weight"] for row in rows],
                         feature_names=features) for split, rows in parts.items()}
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    parameters = dict(iterations=iterations, learning_rate=0.05, depth=4, loss_function="Logloss",
                      random_seed=42, thread_count=2, train_dir=str(log_dir), verbose=False)
    model = CatBoostClassifier(**parameters)
    print(f"обучение catboost на {len(train)} письмах...", flush=True)
    model.fit(pools["train"])
    validation = parts["validation"]
    logits = [float(x) for x in model.predict(pools["validation"], prediction_type="RawFormulaVal")]
    calibration = calibrate(logits, [row["target"] for row in validation],
                            [row["sample_weight"] for row in validation], max_fpr)
    test = parts["test"]
    test_logits = model.predict(pools["test"], prediction_type="RawFormulaVal")
    probabilities = [sigmoid(calibration["weight"] * float(x) + calibration["bias"]) for x in test_logits]
    metrics = calculate_metrics(probabilities, [row["target"] for row in test],
                                [row["sample_weight"] for row in test], calibration["threshold"])
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "catboost_email_model.json"
    temporary = model_path.with_suffix(".json.tmp")
    model.save_model(str(temporary), format="json")
    temporary.replace(model_path)
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    calibration["model_sha256"] = digest
    metadata = dict(version=1, model_sha256=digest, feature_names=features, mode="feature_baseline",
                    used_blocks=used_blocks, missing_blocks=["llm", "image"], catboost_version=catboost.__version__,
                    parameters=parameters, table_sha256=description["table_sha256"], datasets=description["datasets"],
                    splits={split: dict(emails=len(rows), groups=len({row["group_id"] for row in rows}))
                            for split, rows in parts.items()},
                    train_groups=sorted({row["group_id"] for row in train}),
                    note="обучена модель по доступным признакам; объединение четырёх готовых оценок требует общего размеченного набора")
    report = dict(model_sha256=digest, mode=metadata["mode"], used_blocks=used_blocks, splits=metadata["splits"],
                  threshold=calibration["threshold"], validation=calibration["validation_metrics"], test=metrics,
                  feature_importance={name: float(value) for name, value in zip(features, model.feature_importances_)},
                  meets_quality_target=metrics["f1"] >= 0.7 and metrics["roc_auc"] >= 0.7 and metrics["fpr"] <= 0.2,
                  note="калибровка и порог получены только на validation; test использован один раз для итоговой проверки")
    save_json(model_dir / "catboost_calibration.json", calibration)
    save_json(model_dir / "catboost_metadata.json", metadata)
    save_json(report_path, report)
    print(f"модель: {model_path}\nотчёт: {report_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--table", type=Path, default=TABLE_PATH)
    cli.add_argument("--models", type=Path, default=MODEL_DIR)
    cli.add_argument("--report", type=Path, default=REPORT_PATH)
    cli.add_argument("--logs", type=Path, default=LOG_DIR)
    cli.add_argument("--iterations", type=int, default=300)
    cli.add_argument("--max-fpr", type=float, default=0.2)
    args = cli.parse_args()
    try:
        train_model(args.table, args.models, args.report, args.logs, args.iterations, args.max_fpr)
    except (OSError, ValueError, KeyError, ImportError) as error:
        print(f"ошибка обучения catboost: {error}", file=sys.stderr)
        sys.exit(1)
