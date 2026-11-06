"""
общие проверки разметки, разбиения данных, калибровка оценок и расчёт метрик.
родственные письма остаются в одной части выборки, порог выбирается по validation.
"""

import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from email.errors import MessageError
from itertools import groupby
from pathlib import Path

from text_analyze import dataset_files, fit_logistic, sigmoid


def save_json(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(values, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def print_error(error, context=None):
    print(f"ошибка {type(error).__name__}: {error if context is None else context}", file=sys.stderr)


def print_json(values, output=None):
    if output:
        save_json(output, values)
    print(json.dumps(values, ensure_ascii=False, indent=2))


def analyze_path(path, analyze, status_key="status", **options):
    if not path.is_dir():
        return analyze(str(path), **options)

    result = []
    for email in sorted(path.rglob("*")):
        if not email.is_file() or email.suffix.lower() != ".eml":
            continue
        try:
            result.append(dict(analyze(str(email), **options), file=str(email)))
        except (OSError, ValueError, TypeError, MessageError, RecursionError) as error:
            print_error(error, email)
            result.append({"file": str(email), status_key: "parse_error", "error": str(error)})
    return result


def read_dataset(directory):
    directory = Path(directory)
    records = []
    groups = defaultdict(list)
    splits = {}
    annotations = directory / "annotations.jsonl"
    metadata = {}
    if annotations.is_file():
        for line in annotations.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                name = row["file"]
                if name in metadata:
                    raise ValueError(f"повторная разметка: {name}")
                metadata[name] = row
    for path, label, group, split in dataset_files(directory):
        relative = path.relative_to(directory).as_posix()
        annotation = metadata.get(relative, {})
        if metadata and relative not in metadata:
            raise ValueError(f"нет разметки письма: {relative}")
        if annotation and annotation.get("ai_label") != label:
            raise ValueError(f"нет достоверной AI-разметки или метка противоречит папке: {relative}")
        if annotation.get("split", split) != split:
            raise ValueError(f"разбиение не совпадает с папкой: {relative}")
        group = group.split(":", 1)[-1] if split else group
        if group in splits and splits[group] != split:
            raise ValueError(f"группа попала в разные разбиения: {group}")
        splits[group] = split
        record = dict(path=path, filename=relative, group_id=group, split=split,
                      target=label, annotation=annotation)
        records.append(record)
        groups[group].append(record)
    if len(groups) < 20 or any({row["target"] for row in rows} != {0, 1} for rows in groups.values()):
        raise ValueError("нужно хотя бы 20 полных групп: оригинал и его ИИ-версии")
    if not any(splits.values()):
        names = sorted(groups)
        random.Random(42).shuffle(names)
        first, second = int(len(names) * 0.7), int(len(names) * 0.85)
        splits = {name: "train" if i < first else "validation" if i < second else "test"
                  for i, name in enumerate(names)}
        for row in records:
            row["split"] = splits[row["group_id"]]
    # одинаковые файлы тоже не должны оказаться в разных разбиениях
    hashes = {}
    for row in records:
        digest = hashlib.sha256(row["path"].read_bytes()).hexdigest()
        if digest in hashes and hashes[digest] != row["split"]:
            raise ValueError(f"одинаковое письмо попало в разные разбиения: {row['filename']}")
        hashes[digest] = row["split"]
    for split in ("train", "validation", "test"):
        if {row["target"] for row in records if row["split"] == split} != {0, 1}:
            raise ValueError(f"в {split} нужны оба класса")
    counts = defaultdict(int)
    for row in records:
        counts[(row["group_id"], row["target"])] += 1
    for row in records:
        row["weight"] = 1.0 / counts[(row["group_id"], row["target"])]
    return records


def calculate_metrics(probabilities, labels, weights=None, threshold=0.5):
    weights = [1.0] * len(labels) if weights is None else weights
    if not (len(probabilities) == len(labels) == len(weights)) or not labels:
        raise ValueError("оценки, метки и веса должны быть одинаковой непустой длины")
    if any(y not in (0, 1) for y in labels) or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("нужны бинарные метки и оценки от 0 до 1")
    if any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError("веса должны быть положительными")
    tp = fp = fn = tn = 0.0
    for p, y, w in zip(probabilities, labels, weights):
        if p >= threshold:
            if y:
                tp += w
            else:
                fp += w
        elif y:
            fn += w
        else:
            tn += w
    negative = positive = wins = 0.0
    for _, tied in groupby(sorted(zip(probabilities, labels, weights)), key=lambda row: row[0]):
        rows = list(tied)
        pos = sum(w for _, y, w in rows if y == 1)
        neg = sum(w for _, y, w in rows if y == 0)
        wins += pos * (negative + 0.5 * neg)
        negative += neg
        positive += pos
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return dict(precision=precision, recall=recall,
                f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                roc_auc=wins / (positive * negative) if positive and negative else None,
                fpr=fp / (fp + tn) if fp + tn else 0.0,
                accuracy=(tp + tn) / sum(weights),
                brier=sum(w * (p - y) ** 2 for p, y, w in zip(probabilities, labels, weights)) / sum(weights),
                confusion=dict(tp=tp, fp=fp, fn=fn, tn=tn))


def calibrate(logits, labels, weights, max_fpr=0.2):
    if not 0 <= max_fpr <= 1 or set(labels) != {0, 1}:
        raise ValueError("для калибровки нужны оба класса и max_fpr от 0 до 1")
    if not (len(logits) == len(labels) == len(weights)) or not all(math.isfinite(x) for x in logits):
        raise ValueError("неверная длина или некорректные оценки для калибровки")
    if any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError("веса калибровки должны быть положительными")
    fitted, bias = fit_logistic([[x] for x in logits], labels, weights, epochs=400, penalty=0.001)
    # калибровка не должна переворачивать порядок оценок модели
    weight = max(0.0, fitted[0])
    probabilities = [sigmoid(weight * x + bias) for x in logits]
    candidates = sorted({0.0, 0.5, 1.0, *probabilities})
    valid = [(t, calculate_metrics(probabilities, labels, weights, t)) for t in candidates]
    threshold, metrics = max((row for row in valid if row[1]["fpr"] <= max_fpr),
                             key=lambda row: (row[1]["f1"], -row[1]["fpr"], row[0]))
    return dict(weight=weight, bias=bias, threshold=threshold, max_fpr=max_fpr,
                fitted_on="validation", validation_metrics=metrics)


def split_summary(records):
    return {split: dict(emails=sum(row["split"] == split for row in records),
                        groups=len({row["group_id"] for row in records if row["split"] == split}))
            for split in ("train", "validation", "test")}
