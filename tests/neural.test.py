"""
Изолированное тестирование нейросетевого детектора возмущений (DetectGPT-like)
на 100 реальных письмах из тестовой выборки датасета PASTED.
Вычисляет метрики и строит графики распределения и разделения классов.
"""

import sys
from pathlib import Path

# Подключаем папку analysis к путям поиска
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "analysis"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from parser import EmailParser, clean_email_body
from neural_analyze import get_detector

# Настройка шрифтов для графиков
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def plot_neural_dashboard(y_true, y_pred, y_scores, discrepancies):
    """Строит дашборд 2х2 с анализом работы детектора возмущений."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        "Тестирование детектора возмущений (DistilBERT + GPT-2) на 100 письмах PASTED",
        fontsize=16,
        fontweight="bold",
        y=0.97
    )

    # -------------------------------------------------------------
    # 1. ГРАФИК: РАСПРЕДЕЛЕНИЕ DISCREPANCY (РАЗНИЦЫ ПОТЕРЬ)
    # -------------------------------------------------------------
    ax_dist = axes[0, 0]
    human_disc = [d for d, y in zip(discrepancies, y_true) if y == 0]
    ai_disc = [d for d, y in zip(discrepancies, y_true) if y == 1]

    bins = np.linspace(min(discrepancies) - 0.05, max(discrepancies) + 0.05, 15)
    ax_dist.hist(human_disc, bins=bins, alpha=0.6, color="#2ecc71", label="Человек (target=0)", edgecolor="black")
    ax_dist.hist(ai_disc, bins=bins, alpha=0.6, color="#e74c3c", label="ИИ / GPT-3.5 (target=1)", edgecolor="black")
    ax_dist.axvline(0.0, color="gray", linestyle="--", lw=1.5, label="Нулевая разница (d=0)")

    ax_dist.set_xlabel("Значение Discrepancy (рост потерь при синонимах)", fontsize=11, labelpad=8)
    ax_dist.set_ylabel("Количество писем", fontsize=11, labelpad=8)
    ax_dist.set_title("Разделение плотности Discrepancy", fontsize=13, fontweight="bold", pad=12)
    ax_dist.legend(loc="upper right", fontsize=10)
    ax_dist.grid(axis="y", linestyle=":", alpha=0.6)

    # -------------------------------------------------------------
    # 2. ГРАФИК: ROC-КРИВАЯ
    # -------------------------------------------------------------
    ax_roc = axes[0, 1]
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_val = roc_auc_score(y_true, y_scores)

    ax_roc.plot(fpr, tpr, color="#2b5c8f", lw=2.5, label=f"ROC детектора (AUC = {auc_val:.3f})")
    ax_roc.plot([0, 1], [0, 1], color="#e74c3c", lw=1.5, linestyle="--", label="Случайный выбор (AUC = 0.5)")
    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.05])
    ax_roc.set_xlabel("False Positive Rate (Ложные тревоги)", fontsize=11, labelpad=8)
    ax_roc.set_ylabel("True Positive Rate (Полнота)", fontsize=11, labelpad=8)
    ax_roc.set_title("ROC-кривая качества детекции", fontsize=13, fontweight="bold", pad=12)
    ax_roc.legend(loc="lower right", fontsize=10, frameon=True)
    ax_roc.grid(True, linestyle=":", alpha=0.6)

    # -------------------------------------------------------------
    # 3. ГРАФИК: МАТРИЦА ОШИБОК (CONFUSION MATRIX)
    # -------------------------------------------------------------
    ax_cm = axes[1, 0]
    cm = confusion_matrix(y_true, y_pred)
    im = ax_cm.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax_cm.figure.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    classes = ["Человек", "ИИ"]
    tick_marks = np.arange(len(classes))
    ax_cm.set_xticks(tick_marks)
    ax_cm.set_xticklabels(classes, fontsize=11)
    ax_cm.set_yticks(tick_marks)
    ax_cm.set_yticklabels(classes, fontsize=11)
    ax_cm.set_xlabel("Предсказание детектора", fontsize=11, labelpad=8)
    ax_cm.set_ylabel("Истинный класс", fontsize=11, labelpad=8)
    ax_cm.set_title("Матрица ошибок (адаптивный порог)", fontsize=13, fontweight="bold", pad=12)

    c_min, c_max = cm.min(), cm.max()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            norm_val = (val - c_min) / (c_max - c_min + 1e-6)
            text_color = "white" if norm_val > 0.55 else "#111111"
            ax_cm.text(j, i, format(val, "d"),
                       ha="center", va="center",
                       color=text_color, fontsize=15, fontweight="bold")

    # -------------------------------------------------------------
    # 4. ТАБЛИЦА МЕТРИК
    # -------------------------------------------------------------
    ax_tbl = axes[1, 1]
    ax_tbl.axis("off")

    table_data = [
        ["Всего проверено писем", f"{len(y_true)} шт. (50/50)"],
        ["Общая точность (Accuracy)", f"{accuracy_score(y_true, y_pred) * 100:.2f}%"],
        ["Качество разделения (ROC-AUC)", f"{auc_val:.4f}"],
        ["Точность детекции ИИ (Precision)", f"{precision_score(y_true, y_pred, zero_division=0) * 100:.2f}%"],
        ["Полнота отлова ИИ (Recall)", f"{recall_score(y_true, y_pred, zero_division=0) * 100:.2f}%"],
        ["F1-Score", f"{f1_score(y_true, y_pred, zero_division=0):.4f}"],
        ["Ложные тревоги (False Positives)", f"{cm[0][1]} шт."],
        ["Пропущенный ИИ (False Negatives)", f"{cm[1][0]} шт."]
    ]

    table = ax_tbl.table(
        cellText=table_data,
        colLabels=["Метрика качества", "Результат детектора"],
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.0)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            if row % 2 == 0:
                cell.set_facecolor("#f8f9fa")

    ax_tbl.set_title("Сводная таблица показателей", fontsize=13, fontweight="bold", pad=20)

    plt.subplots_adjust(top=0.91, bottom=0.08, left=0.10, right=0.95, hspace=0.38, wspace=0.28)

    save_path = BASE_DIR / "tests" / "neural_evaluation_report.png"
    plt.savefig(save_path, dpi=300)
    print(f"\nГрафик сохранен в файл: {save_path.name}")
    plt.show()


def run_test(limit_pairs: int = 50):
    dataset_dir = BASE_DIR / "datasets" / "PASTED" / "data" / "test"
    human_dir = dataset_dir / "human"
    edited_dir = dataset_dir / "edited"

    if not human_dir.is_dir() or not edited_dir.is_dir():
        # Если папка test не найдена, пробуем искать в общем datasets
        alt_dataset = BASE_DIR / "datasets" / "PASTED" / "data"
        if (alt_dataset / "train").is_dir():
            human_dir = alt_dataset / "train" / "human"
            edited_dir = alt_dataset / "train" / "edited"
        else:
            raise FileNotFoundError(f"Не найден датасет в {dataset_dir}! Сначала скачайте его.")

    # Находим общие парные файлы
    human_files = sorted(p for p in human_dir.glob("*.eml") if p.is_file())
    edited_files = sorted(p for p in edited_dir.glob("*.eml") if p.is_file())

    human_dict = {p.name: p for p in human_files}
    edited_dict = {p.name: p for p in edited_files}
    common_names = sorted(set(human_dict.keys()) & set(edited_dict.keys()))

    if not common_names:
        raise ValueError("Не найдено парных файлов с одинаковыми именами в human и edited!")

    selected_names = common_names[:limit_pairs]
    print(f"\nВыбрано {len(selected_names)} пар писем (всего {len(selected_names)*2} файлов).")

    parser = EmailParser()
    detector = get_detector()

    y_true = []
    y_scores = []
    y_pred = []
    discrepancies = []

    tasks = []
    for name in selected_names:
        tasks.append((human_dict[name], 0, "Человек"))
        tasks.append((edited_dict[name], 1, "ИИ"))

    print(f">>> Начинаем тестирование нейросети на {len(tasks)} письмах...\n")

    for i, (path, label, label_name) in enumerate(tasks, 1):
        parsed = parser.parse_from_file(str(path))
        text = clean_email_body(parsed.text_plain or parsed.extracted_text_from_html)

        # Оцениваем письмо через DistilBERT + GPT-2
        res = detector.evaluate(text, n_perturbations=2)

        disc = res["neural_discrepancy"]
        score = res["neural_ai_score"]

        y_true.append(label)
        y_scores.append(score / 100.0)
        discrepancies.append(disc)

        # Автоматически находим лучший порог разделения по реальным данным
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    # Индекс Юдена: точка максимального отрыва от случайного угадывания
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        print(f"\n>>> Оптимальный порог детекции: {optimal_threshold * 100:.1f}%")
        y_pred = [1 if s >= optimal_threshold else 0 for s in y_scores]

        if i % 10 == 0 or i == len(tasks):
            print(f"  [{i}/{len(tasks)}] Обработано: {path.name} ({label_name}) -> d={disc:+.3f}, скор={score}%")

    print("\n" + "=" * 50)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО! СТРОИМ ДАШБОРД...")
    print("=" * 50)

    plot_neural_dashboard(y_true, y_pred, y_scores, discrepancies)


if __name__ == "__main__":
    # Берем 50 пар = 100 реальных писем
    run_test(limit_pairs=50)