"""
app.py — FastAPI serving layer for the Predictive Maintenance model.

Run locally:  uvicorn app:app --reload
Swagger docs: http://127.0.0.1:8000/docs
Metrics:      http://127.0.0.1:8000/metrics   (scraped by Prometheus)
"""
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)

ARTIFACTS_DIR = Path("artifacts")
# device models present after dropping S1F0 (drop_first=True drops the first alphabetically)
KNOWN_DEVICE_MODELS = ["S1F1", "W1F0", "W1F1", "Z1F0", "Z1F1"]

app = FastAPI(
    title="Predictive Maintenance API",
    description="Predicts device failure from IoT sensor telemetry.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---- Prometheus custom metric: count predictions by outcome ----
PREDICTIONS = Counter(
    "pdm_predictions_total", "Total predictions served", ["result"]
)

MODEL = None
SCALER = None
FEATURE_ORDER = None


# ---------- request / response schemas ----------
class DeviceMetricsRequest(BaseModel):
    date: str = Field(..., examples=["2015-03-17"])
    device: str = Field(..., examples=["S1F01085"])
    metric1: float = Field(..., examples=[215630672])
    metric2: float = Field(..., examples=[55])
    metric3: float = Field(..., examples=[0])
    metric4: float = Field(..., examples=[52])
    metric5: float = Field(..., examples=[6])
    metric6: float = Field(..., examples=[407438])
    metric7: float = Field(..., examples=[0])
    metric8: float = Field(..., examples=[0])
    metric9: float = Field(..., examples=[7])


class PredictionResponse(BaseModel):
    prediction: int
    failure_probability: float | None
    label: str


# ---------- startup: load artifacts once ----------
@app.on_event("startup")
def load_artifacts():
    global MODEL, SCALER, FEATURE_ORDER
    MODEL = joblib.load(ARTIFACTS_DIR / "model.pkl")
    FEATURE_ORDER = json.loads((ARTIFACTS_DIR / "feature_order.json").read_text())
    logging.info(f"Model loaded; expects {len(FEATURE_ORDER)} features.")

    scaler_path = ARTIFACTS_DIR / "scaler.pkl"
    if scaler_path.exists():
        SCALER = joblib.load(scaler_path)
        logging.info("Scaler loaded.")
    else:
        SCALER = None
        logging.warning("No scaler.pkl — predicting on raw inputs.")

    # expose Prometheus /metrics (import here so the app still runs if the lib is absent)
    try:
        from prometheus_fastapi_instrumentator import Instrumentator
        Instrumentator().instrument(app).expose(app)
        logging.info("Prometheus instrumentation active at /metrics")
    except Exception as e:
        logging.warning(f"Prometheus instrumentator not active: {e}")


# ---------- feature engineering (must MATCH training exactly) ----------
def build_feature_row(req: DeviceMetricsRequest) -> pd.DataFrame:
    d = pd.to_datetime(req.date)
    row = {
        "metric1": req.metric1, "metric2": req.metric2, "metric3": req.metric3,
        "metric4": req.metric4, "metric5": req.metric5, "metric6": req.metric6,
        "metric7": req.metric7, "metric8": req.metric8, "metric9": req.metric9,
        "day_of_week": d.dayofweek,
        "day_of_month": d.day,
        "is_weekend": 1 if d.dayofweek >= 5 else 0,
        "month": d.month,
        "week": int(d.isocalendar().week),
    }
    device_model = str(req.device)[:4]
    for dm in KNOWN_DEVICE_MODELS:
        row[f"device_model_{dm}"] = 1 if device_model == dm else 0
    df = pd.DataFrame([row]).reindex(columns=FEATURE_ORDER, fill_value=0)
    return df


# ---------- endpoints ----------
@app.get("/")
def root():
    return {"status": "ok", "message": "Predictive Maintenance API. See /docs"}


@app.get("/health")
def health():
    return {"model_loaded": MODEL is not None,
            "features_expected": len(FEATURE_ORDER or [])}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: DeviceMetricsRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        X = build_feature_row(request)
        X_input = SCALER.transform(X) if SCALER is not None else X.to_numpy()

        pred = int(MODEL.predict(X_input)[0])
        proba = None
        if hasattr(MODEL, "predict_proba"):
            proba = float(MODEL.predict_proba(X_input)[0][1])

        PREDICTIONS.labels(result="failure" if pred == 1 else "healthy").inc()
        return PredictionResponse(
            prediction=pred,
            failure_probability=proba,
            label="FAILURE PREDICTED" if pred == 1 else "Healthy",
        )
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
