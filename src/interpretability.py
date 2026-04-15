import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import os
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shap
from lime.lime_tabular import LimeTabularExplainer
import joblib

from src.config import *
from src.preprocessing import get_classification_data, get_regression_data, load_splits
from src import logger as log


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _pick_best_tree_model(task="clf"):
    """Return the best individual tree-based model (RF or XGBoost) for SHAP.
    Falls back to RandomForest if XGBoost causes SHAP compatibility issues."""
    summary = joblib.load(os.path.join(MODELS_DIR, f"{task}_results_summary.pkl"))
    tree_names = [n for n in ["RandomForest", "XGBoost"] if n in summary]
    if not tree_names:
        return None, None
    metric_key = "f1" if task == "clf" else "r2"
    best = max(tree_names, key=lambda n: summary[n][metric_key])
    model = joblib.load(os.path.join(MODELS_DIR, f"{task}_{best}.pkl"))
    # Test SHAP compatibility — fall back to RandomForest if needed
    try:
        shap.TreeExplainer(model)
    except Exception:
        if best != "RandomForest" and "RandomForest" in tree_names:
            log.warning(f"SHAP incompatible with {best}, falling back to RandomForest")
            best = "RandomForest"
            model = joblib.load(os.path.join(MODELS_DIR, f"{task}_{best}.pkl"))
    return model, best


# ═══════════════════════════════════════════════════════════════════════════
#  SHAP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def shap_classification(data):
    log.section("SHAP: Classification", "🔍")
    model, model_name = _pick_best_tree_model("clf")
    if model is None:
        log.warning("No tree-based classification model found. Skipping SHAP.")
        return

    X_test = data["X_test"]
    feature_names = data["feature_names"]
    class_names = list(data["label_encoder"].classes_)

    # Use a sample for speed
    n_samples = min(100, X_test.shape[0])
    X_sample = X_test[:n_samples]

    log.info(f"Using {model_name} on {n_samples} samples ...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Summary plot (all classes)
    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      class_names=class_names, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "shap_clf_summary.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    log.success("SHAP summary plot saved")

    # Bar plot (mean |SHAP|)
    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      class_names=class_names, plot_type="bar", show=False,
                      max_display=15)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "shap_clf_bar.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    log.success("SHAP bar plot saved")

    # Waterfall plot for first sample (class with highest prediction)
    try:
        explainer_v2 = shap.TreeExplainer(model)
        shap_values_v2 = explainer_v2(X_sample[:1])
        if hasattr(shap_values_v2, "values") and shap_values_v2.values.ndim == 3:
            pred_class = int(model.predict(X_sample[:1])[0])
            sv_single = shap_values_v2[:, :, pred_class]
        else:
            sv_single = shap_values_v2
        sv_single.feature_names = feature_names
        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(sv_single[0], show=False, max_display=15)
        plt.title(f"SHAP Waterfall — Classification (predicted: {class_names[pred_class] if shap_values_v2.values.ndim == 3 else 'best'})",
                  fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "shap_clf_waterfall.png"), dpi=150,
                    bbox_inches="tight")
        plt.close()
        log.success("SHAP waterfall plot saved")
    except Exception as e:
        log.warning(f"Waterfall plot skipped: {e}")


def shap_regression(data):
    log.section("SHAP: Regression", "🔍")
    model, model_name = _pick_best_tree_model("reg")
    if model is None:
        log.warning("No tree-based regression model found. Skipping SHAP.")
        return

    X_test = data["X_test"]
    feature_names = data["feature_names"]
    n_samples = min(100, X_test.shape[0])
    X_sample = X_test[:n_samples]

    log.info(f"Using {model_name} on {n_samples} samples ...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "shap_reg_summary.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    log.success("SHAP summary plot saved")

    # Bar plot
    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      plot_type="bar", show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "shap_reg_bar.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    log.success("SHAP bar plot saved")

    # Dependence plots for top 3 features
    mean_abs = np.abs(shap_values).mean(axis=0)
    top3_idx = np.argsort(mean_abs)[-3:][::-1]
    for rank, idx in enumerate(top3_idx):
        plt.figure()
        shap.dependence_plot(idx, shap_values, X_sample,
                             feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(
            os.path.join(PLOTS_DIR, f"shap_reg_dependence_top{rank + 1}.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close()
    log.success("SHAP dependence plots saved (top 3)")

    # Waterfall plot for first regression sample
    try:
        explainer_v2 = shap.TreeExplainer(model)
        shap_values_v2 = explainer_v2(X_sample[:1])
        shap_values_v2.feature_names = feature_names
        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(shap_values_v2[0], show=False, max_display=15)
        plt.title("SHAP Waterfall — Regression (Sample 1)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "shap_reg_waterfall.png"), dpi=150,
                    bbox_inches="tight")
        plt.close()
        log.success("SHAP waterfall plot saved")
    except Exception as e:
        log.warning(f"Waterfall plot skipped: {e}")


# ═══════════════════════════════════════════════════════════════════════
#  LIME ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def lime_classification(data):
    log.section("LIME: Classification", "🍋")
    model_path = os.path.join(MODELS_DIR, "clf_best_model.pkl")
    if not os.path.exists(model_path):
        log.warning("Best classification model not found. Skipping LIME.")
        return

    model = joblib.load(model_path)
    X_train = data["X_train"]
    X_test = data["X_test"]
    feature_names = data["feature_names"]
    class_names = list(data["label_encoder"].classes_)

    explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        random_state=RANDOM_STATE,
    )

    # Explain 3 random test samples
    np.random.seed(RANDOM_STATE)
    indices = np.random.choice(X_test.shape[0], size=min(3, X_test.shape[0]), replace=False)

    for rank, idx in enumerate(indices):
        exp = explainer.explain_instance(
            X_test[idx], model.predict_proba, num_features=10
        )
        fig = exp.as_pyplot_figure()
        fig.set_size_inches(10, 6)
        fig.suptitle(f"LIME — Classification Sample {rank + 1}", fontsize=13,
                     fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, f"lime_clf_sample_{rank + 1}.png"), dpi=150)
        plt.close(fig)

    log.success(f"LIME classification explanations saved ({len(indices)} samples)")


def lime_regression(data):
    log.section("LIME: Regression", "🍋")
    model_path = os.path.join(MODELS_DIR, "reg_best_model.pkl")
    if not os.path.exists(model_path):
        log.warning("Best regression model not found. Skipping LIME.")
        return

    model = joblib.load(model_path)
    X_train = data["X_train"]
    X_test = data["X_test"]
    feature_names = data["feature_names"]

    explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        mode="regression",
        random_state=RANDOM_STATE,
    )

    np.random.seed(RANDOM_STATE)
    indices = np.random.choice(X_test.shape[0], size=min(3, X_test.shape[0]), replace=False)

    for rank, idx in enumerate(indices):
        exp = explainer.explain_instance(
            X_test[idx], model.predict, num_features=10
        )
        fig = exp.as_pyplot_figure()
        fig.set_size_inches(10, 6)
        fig.suptitle(f"LIME — Regression Sample {rank + 1}", fontsize=13,
                     fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, f"lime_reg_sample_{rank + 1}.png"), dpi=150)
        plt.close(fig)

    log.success(f"LIME regression explanations saved ({len(indices)} samples)")


# ═══════════════════════════════════════════════════════════════════════════

def run():
    log.section("Interpretability & Explainability", "🔬")

    clf_data = load_splits("clf") or get_classification_data()
    reg_data = load_splits("reg") or get_regression_data()

    shap_classification(clf_data)
    shap_regression(reg_data)
    lime_classification(clf_data)
    lime_regression(reg_data)

    log.success(f"All interpretability plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    run()
