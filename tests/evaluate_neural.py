"""
проверка нейросетевого анализа на парных письмах pasted.
порог и калибровка подбираются на validation, итоговые метрики считаются на test.
"""

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "analysis"))

from model_utils import calculate_metrics, calibrate, print_error, print_json, read_dataset
from neural_analyze import get_detector
from parser import EmailParser, clean_email_body
from text_analyze import sigmoid


def evaluate_split(records, split, detector, limit, perturbations, seed):
    groups = defaultdict(list)
    for row in records:
        if row["split"] == split:
            groups[row["group_id"]].append(row)
    names = sorted(groups)
    random.Random(seed).shuffle(names)

    parser = EmailParser()
    scores, labels, weights = [], [], []
    total, scored = Counter(), Counter()
    for group in names[:limit]:
        for row in groups[group]:
            parsed = parser.parse_from_file(str(row["path"]))
            text = clean_email_body(parsed.text_plain or parsed.extracted_text_from_html)
            score = detector.evaluate(text, perturbations, seed)["neural_discrepancy"]
            target = row["target"]
            total[target] += 1
            if score is not None:
                scored[target] += 1
                scores.append(score)
                labels.append(target)
                weights.append(row["weight"])

    if set(labels) != {0, 1}:
        raise ValueError(f"в {split} нужны оценки для обоих классов")
    coverage = {name: dict(total=total[label], scored=scored[label])
                for label, name in ((0, "human"), (1, "edited"))}
    return scores, labels, weights, coverage


def evaluate_dataset(dataset, detector, limit=50, perturbations=5, seed=42, max_fpr=0.2):
    if limit < 1 or perturbations < 1 or not 0 <= max_fpr <= 1:
        raise ValueError("нужны положительные limit и perturbations, max-fpr должен быть от 0 до 1")
    records = read_dataset(dataset)
    scores, labels, weights, validation_coverage = evaluate_split(
        records, "validation", detector, limit, perturbations, seed)
    calibration = calibrate(scores, labels, weights, max_fpr)

    scores, labels, weights, test_coverage = evaluate_split(
        records, "test", detector, limit, perturbations, seed)
    probabilities = [sigmoid(calibration["weight"] * score + calibration["bias"]) for score in scores]
    test = calculate_metrics(probabilities, labels, weights, calibration["threshold"])
    raw_auc = calculate_metrics([sigmoid(score) for score in scores], labels, weights)["roc_auc"]
    return dict(mask_model=detector.mask_name, eval_model=detector.eval_name, limit_groups=limit,
                perturbations=perturbations, seed=seed, calibration=calibration, test=test,
                test_raw_roc_auc=raw_auc, coverage=dict(validation=validation_coverage, test=test_coverage))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dataset", type=Path, default=BASE_DIR / "datasets" / "PASTED" / "data")
    cli.add_argument("--output", type=Path)
    cli.add_argument("--limit", type=int, default=50, help="групп в каждом разбиении")
    cli.add_argument("--perturbations", type=int, default=5)
    cli.add_argument("--seed", type=int, default=42)
    cli.add_argument("--max-fpr", type=float, default=0.2)
    args = cli.parse_args()
    try:
        print_json(evaluate_dataset(args.dataset, get_detector(), args.limit,
                                    args.perturbations, args.seed, args.max_fpr), args.output)
    except (OSError, ValueError, ImportError, RuntimeError) as error:
        print_error(error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
