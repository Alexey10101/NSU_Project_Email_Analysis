"""
собирает таблицу для эксперимента с roberta, сохраняя полные пары pasted.
"""

import argparse
import sys
from pathlib import Path

from build_table import build_meta_dataset
from model_utils import print_error

BASE_DIR = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    colab = Path("/content/datasets/PASTED/data")
    cli.add_argument("--dataset", type=Path, default=colab if colab.is_dir() else BASE_DIR / "datasets/PASTED/data")
    cli.add_argument("--output", type=Path, default=BASE_DIR / "training/data/neural_table.csv")
    cli.add_argument("--limit", type=int, default=100, help="групп в каждом разбиении")
    cli.add_argument("--perturbations", type=int, default=4)
    cli.add_argument("--seed", type=int, default=42)
    cli.add_argument("--no-neural", action="store_true")
    args = cli.parse_args()
    try:
        detector = None
        if not args.no_neural:
            from neural_roberta import get_detector

            detector = get_detector()
        build_meta_dataset(args.limit, args.dataset, args.output, detector, args.perturbations, args.seed)
    except (OSError, KeyError, ValueError, ImportError, RuntimeError) as error:
        print_error(error)
        sys.exit(1)
