import numpy as np
import sys
import os
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.CRITICAL)
logging.getLogger("mlflow.models.model").setLevel(logging.CRITICAL)
logging.getLogger("mlflow.sklearn").setLevel(logging.CRITICAL)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
import joblib

from src.config import *
from src.preprocessing import get_classification_data, save_splits
from src import logger as log


def _safe_log_params(params):
    """Log only serialisable parameters to MLflow."""
    for k, v in params.items():
        if isinstance(v, (int, float, str, bool)):
            try:
                mlflow.log_param(k, v)
            except Exception:
                pass
        elif v is None:
            try:
                mlflow.log_param(k, "None")
            except Exception:
                pass


def train_and_log(model, name, X_train, y_train, X_test, y_test, label_encoder, grid_cv=None):
    """Train a model, evaluate it, and log everything to MLflow.
    If grid_cv is provided (dict with cv_mean, cv_std), skip fit & cross_val_score."""
    with mlflow.start_run(run_name=name):
        _safe_log_params(model.get_params())

        if grid_cv is None:
            model.fit(X_train, y_train)
            cv = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
            cv_mean, cv_std = cv.mean(), cv.std()
        else:
            cv_mean, cv_std = grid_cv["cv_mean"], grid_cv["cv_std"]

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1_mac = f1_score(y_test, y_pred, average="macro")
        f1_w = f1_score(y_test, y_pred, average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1_mac)
        mlflow.log_metric("f1_weighted", f1_w)
        mlflow.log_metric("cv_mean_accuracy", cv_mean)
        mlflow.log_metric("cv_std_accuracy", cv_std)
        mlflow.log_metric("n_classes", len(label_encoder.classes_))

        # Log confusion matrix as artifact
        from sklearn.metrics import confusion_matrix
        import json
        cm = confusion_matrix(y_test, y_pred)
        cm_dict = {
            "labels": list(label_encoder.classes_),
            "matrix": cm.tolist(),
        }
        cm_path = os.path.join(MODELS_DIR, f"clf_{name}_confusion_matrix.json")
        with open(cm_path, "w") as f:
            json.dump(cm_dict, f, indent=2)
        mlflow.log_artifact(cm_path)

        mlflow.sklearn.log_model(model, "model")

        log.info(f"{name} trained")
        log.metric("Accuracy", acc)
        log.metric("F1 (macro)", f1_mac)
        log.metric("F1 (weighted)", f1_w)
        log.metric("CV Accuracy", cv_mean)
        log.metric("CV Std", cv_std)

    return model, acc, f1_w


def run():
    """Train all classification models with hyper-parameter tuning & ensembles."""
    log.section("Loading classification data", "📂")

    data = get_classification_data()
    save_splits(data, "clf")
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    le = data["label_encoder"]

    log.info(f"Classes ({len(le.classes_)}): {list(le.classes_)}")
    log.info(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]}  Features: {X_train.shape[1]}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_DIR)
    mlflow.set_experiment("Cybersecurity_Classification")

    results = {}

    with log.progress_bar(5, "Classification Models") as progress:
        task = progress.add_task("Training models...", total=5)

        # ── 1. Random Forest ───────────────────────────────────────────────────
        log.section("[1/5] Random Forest (GridSearchCV)", "🌲")
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "n_estimators": [100, 200],
                "max_depth": [10, None],
                "min_samples_split": [2, 5],
            },
            cv=5,
            scoring="f1_weighted",
            n_jobs=-1,
        )
        rf_grid.fit(X_train, y_train)
        rf_best_params = rf_grid.best_params_
        log.info(f"Best params: {rf_best_params}")

        rf_cv = {"cv_mean": rf_grid.best_score_, "cv_std": rf_grid.cv_results_['std_test_score'][rf_grid.best_index_]}
        rf_model, rf_acc, rf_f1 = train_and_log(
            rf_grid.best_estimator_, "RandomForest", X_train, y_train, X_test, y_test, le, grid_cv=rf_cv
        )
        results["RandomForest"] = {"model": rf_model, "accuracy": rf_acc, "f1": rf_f1}
        progress.update(task, advance=1, description="[1/5] Random Forest ✔")

        # ── 2. XGBoost ─────────────────────────────────────────────────────────
        log.section("[2/5] XGBoost (GridSearchCV)", "🚀")
        xgb_grid = GridSearchCV(
            XGBClassifier(random_state=RANDOM_STATE, eval_metric="mlogloss"),
            {
                "n_estimators": [100, 200],
                "max_depth": [3, 6],
                "learning_rate": [0.01, 0.1],
            },
            cv=5,
            scoring="f1_weighted",
            n_jobs=-1,
        )
        xgb_grid.fit(X_train, y_train)
        xgb_best_params = xgb_grid.best_params_
        log.info(f"Best params: {xgb_best_params}")

        xgb_cv = {"cv_mean": xgb_grid.best_score_, "cv_std": xgb_grid.cv_results_['std_test_score'][xgb_grid.best_index_]}
        xgb_model, xgb_acc, xgb_f1 = train_and_log(
            xgb_grid.best_estimator_, "XGBoost", X_train, y_train, X_test, y_test, le, grid_cv=xgb_cv
        )
        results["XGBoost"] = {"model": xgb_model, "accuracy": xgb_acc, "f1": xgb_f1}
        progress.update(task, advance=1, description="[2/5] XGBoost ✔")

        # ── 3. SVM ─────────────────────────────────────────────────────────────
        log.section("[3/5] SVM (GridSearchCV)", "🎯")
        svm_grid = GridSearchCV(
            SVC(random_state=RANDOM_STATE, probability=True),
            {"C": [0.1, 1, 10], "kernel": ["rbf"], "gamma": ["scale", "auto"]},
            cv=5,
            scoring="f1_weighted",
            n_jobs=-1,
        )
        svm_grid.fit(X_train, y_train)
        svm_best_params = svm_grid.best_params_
        log.info(f"Best params: {svm_best_params}")

        svm_cv = {"cv_mean": svm_grid.best_score_, "cv_std": svm_grid.cv_results_['std_test_score'][svm_grid.best_index_]}
        svm_model, svm_acc, svm_f1 = train_and_log(
            svm_grid.best_estimator_, "SVM", X_train, y_train, X_test, y_test, le, grid_cv=svm_cv
        )
        results["SVM"] = {"model": svm_model, "accuracy": svm_acc, "f1": svm_f1}
        progress.update(task, advance=1, description="[3/5] SVM ✔")

        # ── 4. Voting Ensemble ─────────────────────────────────────────────────
        log.section("[4/5] Voting Ensemble (soft)", "🗳️")
        voting_clf = VotingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(**rf_best_params, random_state=RANDOM_STATE)),
                ("xgb", XGBClassifier(**xgb_best_params, random_state=RANDOM_STATE,
                                      eval_metric="mlogloss")),
                ("svm", SVC(**svm_best_params, random_state=RANDOM_STATE, probability=True)),
            ],
            voting="soft",
        )
        voting_clf, v_acc, v_f1 = train_and_log(
            voting_clf, "VotingEnsemble", X_train, y_train, X_test, y_test, le
        )
        results["VotingEnsemble"] = {"model": voting_clf, "accuracy": v_acc, "f1": v_f1}
        progress.update(task, advance=1, description="[4/5] Voting ✔")

        # ── 5. Stacking Ensemble ───────────────────────────────────────────────
        log.section("[5/5] Stacking Ensemble", "📚")
        stacking_clf = StackingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(**rf_best_params, random_state=RANDOM_STATE)),
                ("xgb", XGBClassifier(**xgb_best_params, random_state=RANDOM_STATE,
                                      eval_metric="mlogloss")),
                ("svm", SVC(**svm_best_params, random_state=RANDOM_STATE, probability=True)),
            ],
            final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            cv=5,
        )
        stacking_clf, s_acc, s_f1 = train_and_log(
            stacking_clf, "StackingEnsemble", X_train, y_train, X_test, y_test, le
        )
        results["StackingEnsemble"] = {"model": stacking_clf, "accuracy": s_acc, "f1": s_f1}
        progress.update(task, advance=1, description="[5/5] Stacking ✔")

    # ── Results table ──────────────────────────────────────────────────────
    log.metrics_table(
        "Classification Results",
        ["Model", "Accuracy", "F1 (weighted)"],
        [[n, r["accuracy"], r["f1"]] for n, r in results.items()],
        highlight_best=2,
    )

    # ── Save best model ───────────────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["f1"])
    log.champion(best_name, {
        "Accuracy": results[best_name]["accuracy"],
        "F1 (weighted)": results[best_name]["f1"],
    })

    for name, r in results.items():
        joblib.dump(r["model"], os.path.join(MODELS_DIR, f"clf_{name}.pkl"))

    joblib.dump(results[best_name]["model"], os.path.join(MODELS_DIR, "clf_best_model.pkl"))
    joblib.dump(
        {n: {"accuracy": r["accuracy"], "f1": r["f1"]} for n, r in results.items()},
        os.path.join(MODELS_DIR, "clf_results_summary.pkl"),
    )

    log.success(f"All classification models saved to {MODELS_DIR}/")
    return results


if __name__ == "__main__":
    run()
