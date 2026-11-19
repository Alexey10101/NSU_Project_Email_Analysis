"""
запускает проверку roberta и строит график по откалиброванным оценкам.
"""

import sys
from pathlib import Path

from evaluate_neural import main


def plot_report(labels, probabilities, threshold, path):
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for label, title in ((0, "оригиналы"), (1, "ИИ-версии")):
        axes[0].hist([p for p, y in zip(probabilities, labels) if y == label],
                     bins=12, range=(0, 1), alpha=0.6, label=title)
    axes[0].axvline(threshold, color="black", linestyle="--", label="порог")
    axes[0].set(xlabel="калиброванная оценка", ylabel="писем")
    axes[0].legend()
    RocCurveDisplay.from_predictions(labels, probabilities, ax=axes[1])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    sys.exit(main(roberta_default=True))
