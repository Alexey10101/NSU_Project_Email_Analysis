"""
Обучение модели CatBoost на мета-таблице признаков
с построением визуального дашборда (2х2):
1. Матрица ошибок (с исправленным контрастом текста)
2. ROC-кривая качества классификации
3. Рейтинг важности признаков (Feature Importance)
4. Сводная таблица ключевых метрик (plt.table)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from catboost import CatBoostClassifier

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def plot_results(y_test, y_pred, y_proba, feature_names, importances):
    """Отрисовывает аккуратный дашборд 2х2 с графиками и таблицей."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    fig.suptitle(
        "Отчет по оценке модели детекции ИИ (CatBoost)",
        fontsize=18,
        fontweight="bold",
        y=0.97
    )


    ax_cm = axes[0, 0]
    cm = confusion_matrix(y_test, y_pred)
    im = ax_cm.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax_cm.figure.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    classes = ["Человек", "ИИ"]
    tick_marks = np.arange(len(classes))
    ax_cm.set_xticks(tick_marks)
    ax_cm.set_xticklabels(classes, fontsize=12)
    ax_cm.set_yticks(tick_marks)
    ax_cm.set_yticklabels(classes, fontsize=12)
    ax_cm.set_xlabel("Предсказание модели", fontsize=12, labelpad=10)
    ax_cm.set_ylabel("Истинный класс", fontsize=12, labelpad=10)
    ax_cm.set_title("Матрица ошибок (Confusion Matrix)", fontsize=14, fontweight="bold", pad=12)


    c_min, c_max = cm.min(), cm.max()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            norm_val = (val - c_min) / (c_max - c_min + 1e-6)
            text_color = "white" if norm_val > 0.55 else "#111111"
            ax_cm.text(
                j, i, format(val, "d"),
                ha="center", va="center",
                color=text_color, fontsize=16, fontweight="bold"
            )


    ax_roc = axes[0, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_val = roc_auc_score(y_test, y_proba)

    ax_roc.plot(fpr, tpr, color="#2b5c8f", lw=2.5, label=f"ROC (AUC = {auc_val:.3f})")
    ax_roc.plot([0, 1], [0, 1], color="#e74c3c", lw=1.5, linestyle="--", label="Случайный выбор (AUC = 0.5)")
    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.05])
    ax_roc.set_xlabel("False Positive Rate (Ложные тревоги)", fontsize=11, labelpad=8)
    ax_roc.set_ylabel("True Positive Rate (Полнота)", fontsize=11, labelpad=8)
    ax_roc.set_title("ROC-кривая качества классификации", fontsize=14, fontweight="bold", pad=12)
    ax_roc.legend(loc="lower right", fontsize=11, frameon=True)
    ax_roc.grid(True, linestyle=":", alpha=0.6)


    ax_feat = axes[1, 0]
    sorted_idx = np.argsort(importances)
    y_pos = np.arange(len(sorted_idx))

    bars = ax_feat.barh(
        y_pos, [importances[i] for i in sorted_idx],
        color="#3498db", edgecolor="#2980b9", height=0.65
    )
    ax_feat.set_yticks(y_pos)
    ax_feat.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=11)
    ax_feat.set_xlabel("Относительный вклад (%)", fontsize=11, labelpad=8)
    ax_feat.set_title("Важность признаков (Feature Importance)", fontsize=14, fontweight="bold", pad=12)
    ax_feat.grid(axis="x", linestyle=":", alpha=0.6)

    max_w = max(importances) if len(importances) > 0 else 100
    ax_feat.set_xlim(0, max_w * 1.18)

    for bar in bars:
        width = bar.get_width()
        ax_feat.text(
            width + max_w * 0.02, bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}%", va="center", ha="left",
            fontsize=10, fontweight="bold", color="#2c3e50"
        )

    ax_tbl = axes[1, 1]
    ax_tbl.axis("off")

    table_data = [
        ["Общая точность (Accuracy)", f"{accuracy_score(y_test, y_pred) * 100:.2f}%"],
        ["Качество разделения (ROC-AUC)", f"{auc_val:.4f}"],
        ["Точность детекции ИИ (Precision)", f"{precision_score(y_test, y_pred) * 100:.2f}%"],
        ["Полнота отлова ИИ (Recall)", f"{recall_score(y_test, y_pred) * 100:.2f}%"],
        ["F1-Score (Баланс метрик)", f"{f1_score(y_test, y_pred):.4f}"],
        ["Ложные тревоги (False Positives)", f"{cm[0][1]} шт."],
        ["Пропущенный ИИ (False Negatives)", f"{cm[1][0]} шт."]
    ]

    table = ax_tbl.table(
        cellText=table_data,
        colLabels=["Показатель качества", "Значение модели"],
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

    ax_tbl.set_title("Сводная таблица ключевых метрик", fontsize=14, fontweight="bold", pad=20)

    plt.subplots_adjust(top=0.91, bottom=0.08, left=0.10, right=0.95, hspace=0.38, wspace=0.28)

    save_path = Path(__file__).resolve().parent / "model_evaluation_report.png"
    plt.savefig(save_path, dpi=300)
    print(f"\nГрафик сохранен в файл: {save_path.name}")

    plt.show()


def train_model():
    base_dir = Path(__file__).resolve().parent
    table_path = base_dir / "dataset_table.csv"

    if not table_path.exists():
        raise FileNotFoundError(f"Файл {table_path} не найден! Сначала запустите build_table.py")

    df = pd.read_csv(table_path)
    print(f"Загружено строк из таблицы: {len(df)}")

    features_to_drop = ["filename", "target"]
    feature_cols = [c for c in df.columns if c not in features_to_drop]
    X = df[feature_cols]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Обучение CatBoost
    model = CatBoostClassifier(
        iterations=200,
        learning_rate=0.05,
        depth=4,
        random_seed=42,
        verbose=False
    )
    print(">>> Идет обучение CatBoost...")
    model.fit(X_train, y_train)
    print(">>> Модель обучена!\n")

    # Предсказания
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Важность признаков
    importances = model.get_feature_importance()

    # Сохраняем веса модели на диск
    model_save_path = base_dir / "catboost_email_model.cbm"
    model.save_model(str(model_save_path))
    print(f"Модель сохранена в: {model_save_path.name}")

    # Отрисовка графиков
    plot_results(y_test, y_pred, y_proba, feature_cols, importances)


if __name__ == "__main__":
    train_model()