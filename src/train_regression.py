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
    RandomForestRegressor,
    VotingRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import mlflow
import mlflow.sklearn
import joblib

from src.config import *
from src.preprocessing import get_regression_data, save_splits
from src import logger as log


def _safe_log_params(params):
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


def train_and_log(model, name, X_train, y_train, X_test, y_test, grid_cv=None):
    """Train a regression model, evaluate, and log to MLflow.
    If grid_cv is provided (dict with cv_mean, cv_std), skip fit & cross_val_score."""
    with mlflow.start_run(run_name=name):
        _safe_log_params(model.get_params())

        if grid_cv is None:
            model.fit(X_train, y_train)
            cv = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
            cv_mean, cv_std = cv.mean(), cv.std()
        else:
            cv_mean, cv_std = grid_cv["cv_mean"], grid_cv["cv_std"]

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("cv_mean_r2", cv_mean)
        mlflow.log_metric("cv_std_r2", cv_std)

        # Log predictions summary as artifact
        import json
        summary = {
            "model": name,
            "r2": round(r2, 4),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "cv_mean_r2": round(cv_mean, 4),
            "cv_std_r2": round(cv_std, 4),
            "y_test_range": [round(float(y_test.min()), 2), round(float(y_test.max()), 2)],
        }
        summary_path = os.path.join(MODELS_DIR, f"reg_{name}_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact(summary_path)

        mlflow.sklearn.log_model(model, "model")

        log.info(f"{name} trained")
        log.metric("R²", r2)
        log.metric("MAE", mae)
        log.metric("RMSE", rmse)
        log.metric("CV R² (mean)", cv_mean)
        log.metric("CV Std", cv_std)

    return model, r2, mae


def run():
    """Train all regression models with hyper-parameter tuning & ensembles."""
    log.section("Loading regression data", "📂")

    data = get_regression_data()
    save_splits(data, "reg")
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    log.info(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]}  Features: {X_train.shape[1]}")
    log.info(f"Target range: [{y_train.min():.2f}, {y_train.max():.2f}]")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_DIR)
    mlflow.set_experiment("Cybersecurity_Regression")

    results = {}

    with log.progress_bar(5, "Regression Models") as progress:
        task = progress.add_task("Training models...", total=5)

        # ── 1. Random Forest Regressor ─────────────────────────────────────────
        log.section("[1/5] Random Forest Regressor (GridSearchCV)", "🌲")
        rf_grid = GridSearchCV(
            RandomForestRegressor(random_state=RANDOM_STATE),
            {
                "n_estimators": [100, 200],
                "max_depth": [10, None],
                "min_samples_split": [2, 5],
            },
            cv=5,
            scoring="r2",
            n_jobs=-1,
        )
        rf_grid.fit(X_train, y_train)
        rf_bp = rf_grid.best_params_
        log.info(f"Best params: {rf_bp}")

        rf_cv = {"cv_mean": rf_grid.best_score_, "cv_std": rf_grid.cv_results_['std_test_score'][rf_grid.best_index_]}
        rf_model, rf_r2, rf_mae = train_and_log(
            rf_grid.best_estimator_, "RandomForest", X_train, y_train, X_test, y_test, grid_cv=rf_cv
        )
        results["RandomForest"] = {"model": rf_model, "r2": rf_r2, "mae": rf_mae}
        progress.update(task, advance=1, description="[1/5] Random Forest ✔")

        # ── 2. XGBoost Regressor ───────────────────────────────────────────
        log.section("[2/5] XGBoost Regressor (GridSearchCV)", "🚀")
        xgb_grid = GridSearchCV(
            XGBRegressor(random_state=RANDOM_STATE, eval_metric="rmse"),
            {
                "n_estimators": [100, 200],
                "max_depth": [3, 6],
                "learning_rate": [0.01, 0.1],
            },
            cv=5,
            scoring="r2",
            n_jobs=-1,
        )
        xgb_grid.fit(X_train, y_train)
        xgb_bp = xgb_grid.best_params_
        log.info(f"Best params: {xgb_bp}")

        xgb_cv = {"cv_mean": xgb_grid.best_score_, "cv_std": xgb_grid.cv_results_['std_test_score'][xgb_grid.best_index_]}
        xgb_model, xgb_r2, xgb_mae = train_and_log(
            xgb_grid.best_estimator_, "XGBoost", X_train, y_train, X_test, y_test, grid_cv=xgb_cv
        )
        results["XGBoost"] = {"model": xgb_model, "r2": xgb_r2, "mae": xgb_mae}
        progress.update(task, advance=1, description="[2/5] XGBoost ✔")

        # ── 3. Ridge Regression ────────────────────────────────────────────
        log.section("[3/5] Ridge Regression (GridSearchCV)", "🎯")
        ridge_grid = GridSearchCV(
            Ridge(),
            {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
            cv=5,
            scoring="r2",
            n_jobs=-1,
        )
        ridge_grid.fit(X_train, y_train)
        ridge_bp = ridge_grid.best_params_
        log.info(f"Best params: {ridge_bp}")

        ridge_cv = {"cv_mean": ridge_grid.best_score_, "cv_std": ridge_grid.cv_results_['std_test_score'][ridge_grid.best_index_]}
        ridge_model, ridge_r2, ridge_mae = train_and_log(
            ridge_grid.best_estimator_, "Ridge", X_train, y_train, X_test, y_test, grid_cv=ridge_cv
        )
        results["Ridge"] = {"model": ridge_model, "r2": ridge_r2, "mae": ridge_mae}
        progress.update(task, advance=1, description="[3/5] Ridge ✔")

        # ── 4. Voting Regressor ────────────────────────────────────────────
        log.section("[4/5] Voting Regressor", "🗳️")
        voting_reg = VotingRegressor(
            estimators=[
                ("rf", RandomForestRegressor(**rf_bp, random_state=RANDOM_STATE)),
                ("xgb", XGBRegressor(**xgb_bp, random_state=RANDOM_STATE, eval_metric="rmse")),
                ("ridge", Ridge(**ridge_bp)),
            ]
        )
        voting_reg, v_r2, v_mae = train_and_log(
            voting_reg, "VotingEnsemble", X_train, y_train, X_test, y_test
        )
        results["VotingEnsemble"] = {"model": voting_reg, "r2": v_r2, "mae": v_mae}
        progress.update(task, advance=1, description="[4/5] Voting ✔")

        # ── 5. Stacking Regressor ───────────────────────────────────────────
        log.section("[5/5] Stacking Regressor", "📚")
        stacking_reg = StackingRegressor(
            estimators=[
                ("rf", RandomForestRegressor(**rf_bp, random_state=RANDOM_STATE)),
                ("xgb", XGBRegressor(**xgb_bp, random_state=RANDOM_STATE, eval_metric="rmse")),
                ("ridge", Ridge(**ridge_bp)),
            ],
            final_estimator=Ridge(alpha=1.0),
            cv=5,
        )
        stacking_reg, s_r2, s_mae = train_and_log(
            stacking_reg, "StackingEnsemble", X_train, y_train, X_test, y_test
        )
        results["StackingEnsemble"] = {"model": stacking_reg, "r2": s_r2, "mae": s_mae}
        progress.update(task, advance=1, description="[5/5] Stacking ✔")

    # ── Results table ──────────────────────────────────────────────────────
    log.metrics_table(
        "Regression Results",
        ["Model", "R²", "MAE"],
        [[n, r["r2"], r["mae"]] for n, r in results.items()],
        highlight_best=1,
    )

    # ── Save best model ───────────────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["r2"])
    log.champion(best_name, {
        "R²": results[best_name]["r2"],
        "MAE": results[best_name]["mae"],
    })

    for name, r in results.items():
        joblib.dump(r["model"], os.path.join(MODELS_DIR, f"reg_{name}.pkl"))

    joblib.dump(results[best_name]["model"], os.path.join(MODELS_DIR, "reg_best_model.pkl"))
    joblib.dump(
        {n: {"r2": r["r2"], "mae": r["mae"]} for n, r in results.items()},
        os.path.join(MODELS_DIR, "reg_results_summary.pkl"),
    )

    log.success(f"All regression models saved to {MODELS_DIR}/")
    return results


if __name__ == "__main__":
    run()
