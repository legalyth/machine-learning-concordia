import sys
import os
import time
import warnings
import logging

# Suppress noisy third-party logs from console
warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.CRITICAL)
logging.getLogger("mlflow.models.model").setLevel(logging.CRITICAL)
logging.getLogger("mlflow.sklearn").setLevel(logging.CRITICAL)
logging.getLogger("mlflow.utils.autologging_utils").setLevel(logging.CRITICAL)
logging.getLogger("xgboost").setLevel(logging.CRITICAL)

os.environ["MLFLOW_DISABLE_WARNINGS"] = "true"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.eda import run as run_eda
from src.train_classification import run as run_classification
from src.train_regression import run as run_regression
from src.evaluate import run as run_evaluation
from src.interpretability import run as run_interpretability
from src import logger as log


def main():
    pipeline_start = time.time()

    log.banner()

    # Step 1 — Exploratory Data Analysis
    t0 = time.time()
    log.step_header(1, 5, "EXPLORATORY DATA ANALYSIS", "📊")
    run_eda()
    log.success(f"EDA completed in {time.time() - t0:.1f}s")

    # Step 2 — Classification Model Training
    t0 = time.time()
    log.step_header(2, 5, "CLASSIFICATION MODEL TRAINING", "🤖")
    run_classification()
    log.success(f"Classification training completed in {time.time() - t0:.1f}s")

    # Step 3 — Regression Model Training
    t0 = time.time()
    log.step_header(3, 5, "REGRESSION MODEL TRAINING", "📈")
    run_regression()
    log.success(f"Regression training completed in {time.time() - t0:.1f}s")

    # Step 4 — Model Evaluation
    t0 = time.time()
    log.step_header(4, 5, "MODEL EVALUATION", "🔍")
    run_evaluation()
    log.success(f"Evaluation completed in {time.time() - t0:.1f}s")

    # Step 5 — Interpretability & Explainability
    t0 = time.time()
    log.step_header(5, 5, "INTERPRETABILITY & EXPLAINABILITY", "🧠")
    run_interpretability()
    log.success(f"Interpretability completed in {time.time() - t0:.1f}s")

    total = time.time() - pipeline_start
    log.info(f"Total pipeline time: {total:.1f}s ({total/60:.1f} min)")
    log.pipeline_complete()


if __name__ == "__main__":
    main()
