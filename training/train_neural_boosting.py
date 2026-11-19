"""
обучает отдельный экспериментальный catboost в формате основного анализатора.
"""

import argparse
import sys

from train_boosting import BASE_DIR, train_model
from model_utils import print_error


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--iterations", type=int, default=300)
    cli.add_argument("--max-fpr", type=float, default=0.2)
    cli.add_argument("--without-neural", action="store_true")
    args = cli.parse_args()
    suffix = "comparison" if args.without_neural else "neural"
    try:
        train_model(BASE_DIR / "training/data/neural_table.csv", BASE_DIR / "models" / suffix,
                    BASE_DIR / "training/reports" / f"catboost_{suffix}_metrics.json",
                    BASE_DIR / "training/logs" / f"catboost_{suffix}", args.iterations, args.max_fpr,
                    not args.without_neural)
    except (OSError, KeyError, ValueError, ImportError, RuntimeError) as error:
        print_error(error)
        sys.exit(1)
