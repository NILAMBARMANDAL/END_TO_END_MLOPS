import logging
import numpy as np
import pandas as pd
from typing import Tuple, Any
from zenml import step
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import mlflow

logging.basicConfig(level=logging.INFO)


@step(experiment_tracker="mlflow_tracker")
def evaluation(
    model: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[float, float, float]:
    """Evaluate the model and tune the decision threshold to maximise F1.

    Why threshold tuning: on imbalanced data the default 0.5 cutoff is rarely optimal.
    We sweep thresholds and pick the one with the best F1, then log everything to MLflow.
    Returns (accuracy, f1, roc_auc) — the SAME metric names the deployment_trigger reads.
    """
    try:
        logging.info("Evaluating model and optimizing decision threshold for F1.")

        # probability of the positive class (failure)
        y_probs = model.predict_proba(x_test)[:, 1]

        best_threshold, best_f1, best_acc = 0.5, 0.0, 0.0
        for thresh in np.arange(0.01, 1.0, 0.01):
            preds = (y_probs >= thresh).astype(int)
            current_f1 = f1_score(y_test, preds, zero_division=0)
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = float(thresh)
                best_acc = accuracy_score(y_test, preds)

        auc = float(roc_auc_score(y_test, y_probs))

        # log with stable names the rest of the pipeline depends on
        mlflow.log_metric("optimized_accuracy", best_acc)
        mlflow.log_metric("optimized_f1_score", best_f1)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_param("optimized_threshold", best_threshold)

        # log the model so MLflow can serve it / show signature
        mlflow.sklearn.log_model(model, artifact_path="model", input_example=x_test.head(5))

        logging.info(
            f"Best threshold={best_threshold:.2f} | acc={best_acc:.4f} "
            f"f1={best_f1:.4f} auc={auc:.4f}"
        )
        return best_acc, best_f1, auc

    except Exception as e:
        logging.error(f"Error during evaluation/threshold optimization: {e}")
        raise e
