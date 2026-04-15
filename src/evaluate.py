import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.preprocessing import label_binarize
import joblib

from src.config import *
from src.preprocessing import get_classification_data, get_regression_data, load_splits
from src import logger as log


# ═══════════════════════════════════════════════════════════════════════════
#  CLASSIFICATION EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_classification():
    log.section("Classification Evaluation", "🎯")

    data = load_splits("clf") or get_classification_data()
    X_test, y_test = data["X_test"], data["y_test"]
    le = data["label_encoder"]
    class_names = list(le.classes_)
    n_classes = len(class_names)

    model_names = ["RandomForest", "XGBoost", "SVM", "VotingEnsemble", "StackingEnsemble"]
    results = {}

    for name in model_names:
        path = os.path.join(MODELS_DIR, f"clf_{name}.pkl")
        if not os.path.exists(path):
            log.warning(f"{name} — model not found, skipping")
            continue
        model = joblib.load(path)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "y_pred": y_pred,
            "y_proba": y_proba,
        }

    # ── Comparison table ───────────────────────────────────────────────────
    log.metrics_table(
        "Classification Evaluation",
        ["Model", "Accuracy", "Precision", "Recall", "F1-wt", "F1-mac"],
        [[n, m["accuracy"], m["precision"], m["recall"], m["f1_weighted"], m["f1_macro"]] for n, m in results.items()],
        highlight_best=1,
    )

    # ── Confusion matrices ─────────────────────────────────────────────────
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    for ax, (name, m) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, m["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=class_names, yticklabels=class_names)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "eval_clf_confusion_matrices.png"), dpi=150)
    plt.close()
    log.success("Confusion matrices saved")

    # ── ROC curves (One-vs-Rest) ───────────────────────────────────────────
    y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))
    colors = sns.color_palette("husl", n_classes)

    for name, m in results.items():
        if m["y_proba"] is None:
            continue
        fig, ax = plt.subplots(figsize=(10, 8))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], m["y_proba"][:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i],
                    label=f"{class_names[i]} (AUC={roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curves — {name}", fontsize=13, fontweight="bold")
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"eval_clf_roc_{name}.png"), dpi=150)
        plt.close()
    log.success("ROC curves saved")

    # ── Classification report for best model ───────────────────────────────
    best_name = max(results, key=lambda k: results[k]["f1_weighted"])
    log.info(f"Best model: {best_name}")
    log.info("Classification report:\n" + classification_report(
        y_test, results[best_name]["y_pred"], target_names=class_names))

    # ── Bar chart comparison ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_weighted"]
    x = np.arange(len(results))
    width = 0.2
    for i, metric in enumerate(metrics_to_plot):
        vals = [results[n][metric] for n in results]
        ax.bar(x + i * width, vals, width, label=metric.replace("_", " ").title())
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(results.keys(), rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Classification Model Comparison", fontsize=14, fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "eval_clf_comparison.png"), dpi=150)
    plt.close()
    log.success("Classification comparison bar chart saved")

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  REGRESSION EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_regression():
    log.section("Regression Evaluation", "📊")

    data = load_splits("reg") or get_regression_data()
    X_test, y_test = data["X_test"], data["y_test"]

    model_names = ["RandomForest", "XGBoost", "Ridge", "VotingEnsemble", "StackingEnsemble"]
    results = {}

    for name in model_names:
        path = os.path.join(MODELS_DIR, f"reg_{name}.pkl")
        if not os.path.exists(path):
            log.warning(f"{name} — model not found, skipping")
            continue
        model = joblib.load(path)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        results[name] = {
            "r2": r2_score(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "mse": mse,
            "rmse": np.sqrt(mse),
            "y_pred": y_pred,
        }

    # ── Comparison table ───────────────────────────────────────────────────
    log.metrics_table(
        "Regression Evaluation",
        ["Model", "R²", "MAE", "MSE", "RMSE"],
        [[n, m["r2"], m["mae"], m["mse"], m["rmse"]] for n, m in results.items()],
        highlight_best=1,
    )

    # ── Actual vs Predicted ────────────────────────────────────────────────
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    for ax, (name, m) in zip(axes, results.items()):
        ax.scatter(y_test, m["y_pred"], alpha=0.6, s=30, edgecolors="k", linewidths=0.3)
        mn, mx = min(y_test.min(), m["y_pred"].min()), max(y_test.max(), m["y_pred"].max())
        ax.plot([mn, mx], [mn, mx], "r--", linewidth=2, label="Perfect fit")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{name}\nR²={m['r2']:.3f}", fontsize=11, fontweight="bold")
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "eval_reg_actual_vs_predicted.png"), dpi=150)
    plt.close()
    log.success("Actual vs Predicted plots saved")

    # ── Residual plots ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    for ax, (name, m) in zip(axes, results.items()):
        residuals = y_test - m["y_pred"]
        ax.scatter(m["y_pred"], residuals, alpha=0.6, s=30, edgecolors="k", linewidths=0.3)
        ax.axhline(y=0, color="r", linestyle="--", linewidth=2)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual")
        ax.set_title(f"{name} Residuals", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "eval_reg_residuals.png"), dpi=150)
    plt.close()
    log.success("Residual plots saved")

    # ── Bar chart comparison ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    names_list = list(results.keys())

    axes[0].bar(names_list, [results[n]["r2"] for n in names_list], color="steelblue")
    axes[0].set_title("R² Score Comparison", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("R²")
    axes[0].tick_params(axis="x", rotation=30)

    x = np.arange(len(names_list))
    w = 0.35
    axes[1].bar(x - w / 2, [results[n]["mae"] for n in names_list], w, label="MAE", color="salmon")
    axes[1].bar(x + w / 2, [results[n]["rmse"] for n in names_list], w, label="RMSE", color="steelblue")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names_list, rotation=30, ha="right")
    axes[1].set_title("Error Metrics Comparison", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Value (Million $)")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "eval_reg_comparison.png"), dpi=150)
    plt.close()
    log.success("Comparison bar chart saved")

    return results


# ═══════════════════════════════════════════════════════════════════════

def run():
    clf_results = evaluate_classification()
    reg_results = evaluate_regression()
    log.success(f"All evaluation plots saved to {PLOTS_DIR}/")
    return clf_results, reg_results


if __name__ == "__main__":
    run()
