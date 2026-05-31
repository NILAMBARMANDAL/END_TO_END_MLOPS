"""
export_model.py — exports the trained model + feature schema from the ZenML pipeline
into plain files (artifacts/) that FastAPI / Docker load directly.

Run:  python export_model.py
"""
import json
import joblib
import logging
from pathlib import Path
from zenml.client import Client

logging.basicConfig(level=logging.INFO)

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Verified column order for THIS dataset. The order follows how the notebook CREATES
# the columns (date parts first as day_of_week, day_of_month, is_weekend, then month, week),
# and get_dummies appends the device_model_* dummies last. It is NOT alphabetical.
FEATURE_ORDER = [
    "metric1", "metric2", "metric3", "metric4", "metric5",
    "metric6", "metric7", "metric8", "metric9",
    "day_of_week", "day_of_month", "is_weekend", "month", "week",
    "device_model_S1F1", "device_model_W1F0", "device_model_W1F1",
    "device_model_Z1F0", "device_model_Z1F1",
]


def export():
    client = Client()
    pipeline = client.get_pipeline("continuous_deployment_pipeline")
    last_run = pipeline.last_successful_run
    logging.info(f"Using run: {last_run.id}")

    trained_model = last_run.steps["train_model"].output.load()
    joblib.dump(trained_model, ARTIFACTS_DIR / "model.pkl")
    logging.info("Saved artifacts/model.pkl")

    # prefer the exact training feature order if the model preserved it
    try:
        order = list(trained_model.feature_names_in_)
        logging.info("Using feature order recovered from the trained model.")
    except AttributeError:
        order = FEATURE_ORDER
        logging.info("Model had no feature_names_in_; using hard-coded FEATURE_ORDER.")

    (ARTIFACTS_DIR / "feature_order.json").write_text(json.dumps(order, indent=2))
    logging.info("Saved artifacts/feature_order.json")

    scaler_src = Path("model/scaler.joblib")
    if scaler_src.exists():
        joblib.dump(joblib.load(scaler_src), ARTIFACTS_DIR / "scaler.pkl")
        logging.info("Saved artifacts/scaler.pkl")
    else:
        logging.warning("No scaler at model/scaler.joblib; API will run unscaled.")

    logging.info("Export complete -> ./artifacts/")


if __name__ == "__main__":
    export()
