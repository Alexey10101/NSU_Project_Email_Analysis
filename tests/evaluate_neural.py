"""
проверяет нейросетевой анализ на парных письмах pasted.
калибровка и порог подбираются на validation, итоговые метрики считаются на test.
"""

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "analysis"))

from model_utils import calculate_metrics, calibrate, print_error, print_json, read_dataset
from neural_analyze import get_detector
from parser import EmailParser, clean_email_body
from text_analyze import sigmoid


def evaluate_dataset(dataset, detectors, limit=50, perturbations=4, seed=42, max_fpr=0.2):
    if limit < 1 or perturbations < 1 or not 0 <= max_fpr <= 1:
        raise ValueError("нужны положительные limit и perturbations, max-fpr должен быть от 0 до 1")
    records = read_dataset(dataset)
    reports = {name: dict(mask_model=model.mask_name, eval_model=model.eval_name,
                          perturbations=perturbations, seed=seed, coverage={})
               for name, model in detectors.items()}
    parser = EmailParser()
    chart = {}

    for split in ("validation", "test"):
        groups = defaultdict(list)
        for row in records:
            if row["split"] == split:
                groups[row["group_id"]].append(row)
        names = sorted(groups)
        random.Random(seed).shuffle(names)
        selected = names[:limit]
        observations = []
        seconds = defaultdict(float)
        for name in selected:
            group = []
            for row in groups[name]:
                parsed = parser.parse_from_file(row["path"])
                text = clean_email_body(parsed.text_plain or parsed.extracted_text_from_html)
                values = {}
                for key, model in detectors.items():
                    start = perf_counter()
                    values[key] = model.evaluate(text, perturbations, seed)["neural_discrepancy"]
                    seconds[key] += perf_counter() - start
                group.append((row["target"], row["weight"], values))
            if all(value is not None for _, _, values in group for value in values.values()):
                observations.extend(group)

        if {label for label, _, _ in observations} != {0, 1}:
            raise ValueError(f"в {split} нужны оцениваемые пары для обоих классов")
        labels = [label for label, _, _ in observations]
        weights = [weight for _, weight, _ in observations]
        for key, report in reports.items():
            scores = [values[key] for _, _, values in observations]
            report["coverage"][split] = dict(groups_selected=len(selected), emails_scored=len(scores),
                                             emails_selected=sum(len(groups[name]) for name in selected),
                                             inference_seconds=round(seconds[key], 3))
            if split == "validation":
                report["calibration"] = calibrate(scores, labels, weights, max_fpr)
                continue
            cal = report["calibration"]
            probabilities = [sigmoid(cal["weight"] * score + cal["bias"]) for score in scores]
            report["test"] = calculate_metrics(probabilities, labels, weights, cal["threshold"])
            report["test_raw_roc_auc"] = calculate_metrics([sigmoid(x) for x in scores], labels, weights)["roc_auc"]
            chart[key] = (labels, probabilities, cal["threshold"])
    return reports, chart


def main(roberta_default=False):
    cli = argparse.ArgumentParser(description=__doc__)
    colab = Path("/content/datasets/PASTED/data")
    cli.add_argument("--dataset", type=Path, default=colab if colab.is_dir() else BASE_DIR / "datasets/PASTED/data")
    cli.add_argument("--output", type=Path)
    cli.add_argument("--limit", type=int, default=50, help="групп в каждом разбиении")
    cli.add_argument("--perturbations", type=int, default=4)
    cli.add_argument("--seed", type=int, default=42)
    cli.add_argument("--max-fpr", type=float, default=0.2)
    mode = cli.add_mutually_exclusive_group()
    mode.add_argument("--roberta", action="store_true", default=roberta_default)
    mode.add_argument("--compare", action="store_true", help="проверить оба варианта на одинаковых парах")
    cli.add_argument("--plot", type=Path)
    args = cli.parse_args()
    try:
        detectors = {}
        if not args.roberta or args.compare:
            detectors["distilbert"] = get_detector()
        if args.roberta or args.compare:
            from neural_roberta import get_detector as get_roberta

            detectors["roberta"] = get_roberta()
        reports, chart = evaluate_dataset(args.dataset, detectors, args.limit, args.perturbations,
                                          args.seed, args.max_fpr)
        print_json(reports if args.compare else next(iter(reports.values())), args.output)
        if args.plot is not None:
            from evaluate_roberta import plot_report

            name = "roberta" if "roberta" in chart else "distilbert"
            plot_report(*chart[name], args.plot)
    except (OSError, ValueError, ImportError, RuntimeError) as error:
        print_error(error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
