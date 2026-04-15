import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.config import (
    MODELS_DIR,
    NUMERIC_FEATURES_CLF,
    CATEGORICAL_FEATURES_CLF,
    NUMERIC_FEATURES_REG,
    CATEGORICAL_FEATURES_REG,
)
from src.preprocessing import feature_engineering
from api.schemas import (
    ClassificationInput,
    RegressionInput,
    ClassificationOutput,
    RegressionOutput,
    HealthResponse,
)
import logging
_logger = logging.getLogger("api")
# ── Global model store ─────────────────────────────────────────────────────
_models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at startup."""
    required = {
        "clf_model": "clf_best_model.pkl",
        "clf_preprocessor": "clf_preprocessor.pkl",
        "clf_label_encoder": "clf_label_encoder.pkl",
        "reg_model": "reg_best_model.pkl",
        "reg_preprocessor": "reg_preprocessor.pkl",
    }
    missing = [f for f in required.values() if not os.path.exists(os.path.join(MODELS_DIR, f))]
    if missing:
        _logger.error(f"Missing model files: {missing}. Run main.py first.")
        raise RuntimeError(f"Missing model files: {missing}. Train models first with: python main.py")
    for key, filename in required.items():
        _models[key] = joblib.load(os.path.join(MODELS_DIR, filename))
    _logger.info("All models loaded successfully")
    yield
    _models.clear()


app = FastAPI(
    title="Cybersecurity Threat Prediction API",
    description="Predict attack types and financial losses from cybersecurity incidents.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────────────
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    with open(os.path.join(_TEMPLATE_DIR, "index.html"), encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", models_loaded=bool(_models))


@app.post("/predict/attack-type", response_model=ClassificationOutput)
async def predict_attack_type(input_data: ClassificationInput):
    try:
        row = {
            "Country": input_data.Country,
            "Year": input_data.Year,
            "Target Industry": input_data.Target_Industry,
            "Financial Loss (in Million $)": input_data.Financial_Loss,
            "Number of Affected Users": input_data.Number_of_Affected_Users,
            "Attack Source": input_data.Attack_Source,
            "Security Vulnerability Type": input_data.Security_Vulnerability_Type,
            "Defense Mechanism Used": input_data.Defense_Mechanism_Used,
            "Incident Resolution Time (in Hours)": input_data.Incident_Resolution_Time,
        }
        df = pd.DataFrame([row])
        df = feature_engineering(df)

        X = df[NUMERIC_FEATURES_CLF + CATEGORICAL_FEATURES_CLF]
        X_processed = _models["clf_preprocessor"].transform(X)

        prediction = _models["clf_model"].predict(X_processed)
        probabilities = _models["clf_model"].predict_proba(X_processed)

        le = _models["clf_label_encoder"]
        attack_type = le.inverse_transform(prediction)[0]
        confidence = float(probabilities.max())

        class_probs = {
            le.classes_[i]: round(float(probabilities[0][i]), 4)
            for i in range(len(le.classes_))
        }

        return ClassificationOutput(
            prediction=attack_type,
            confidence=round(confidence, 4),
            class_probabilities=class_probs,
        )
    except Exception as e:
        _logger.exception("Classification prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed. Check input values.")


@app.post("/predict/financial-loss", response_model=RegressionOutput)
async def predict_financial_loss(input_data: RegressionInput):
    try:
        row = {
            "Country": input_data.Country,
            "Year": input_data.Year,
            "Attack Type": input_data.Attack_Type,
            "Target Industry": input_data.Target_Industry,
            "Number of Affected Users": input_data.Number_of_Affected_Users,
            "Attack Source": input_data.Attack_Source,
            "Security Vulnerability Type": input_data.Security_Vulnerability_Type,
            "Defense Mechanism Used": input_data.Defense_Mechanism_Used,
            "Incident Resolution Time (in Hours)": input_data.Incident_Resolution_Time,
        }
        df = pd.DataFrame([row])
        df = feature_engineering(df)

        X = df[NUMERIC_FEATURES_REG + CATEGORICAL_FEATURES_REG]
        X_processed = _models["reg_preprocessor"].transform(X)

        prediction = _models["reg_model"].predict(X_processed)
        predicted_value = max(0.0, float(prediction[0]))

        return RegressionOutput(
            predicted_financial_loss=round(predicted_value, 2),
            unit="Million $",
        )
    except Exception as e:
        _logger.exception("Regression prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed. Check input values.")


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
