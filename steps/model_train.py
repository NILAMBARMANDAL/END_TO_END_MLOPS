import logging
import pandas as pd
from typing import Any, Dict
from pydantic import BaseModel, Field
from zenml import step
import mlflow

from src.model_dev import (
    LogisticRegressionModel,
    RandomForestModel,
    XGBoostModel,
    LightGBMModel,
    LinearRegressionModel,
)

logging.basicConfig(level=logging.INFO)


class ModelNameConfig(BaseModel):
    """Configuration schema for model training, populated from config.yaml.

    model_name      -> which algorithm to train
    fine_tuning     -> reserved flag for a hyperparameter-search path (off by default)
    hyperparameters -> dict of model kwargs (n_estimators, max_depth, ...) so you tune
                       the model from YAML without editing Python.
    """
    model_name: str = "randomforest"
    fine_tuning: bool = False
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)


# registry: add a new model => add one line here. (Open/Closed principle)
MODEL_REGISTRY = {
    "logistic_regression": LogisticRegressionModel,
    "randomforest": RandomForestModel,
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
    "linear_regression": LinearRegressionModel,
}


@step(experiment_tracker="mlflow_tracker")
def train_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig,
) -> Any:
    """Trains the model named in config, using hyperparameters from config.yaml."""
    try:
        # autolog records params/metrics/model to MLflow automatically for these frameworks
        mlflow.sklearn.autolog(log_datasets=False, log_models=True)
        mlflow.xgboost.autolog(silent=True, log_datasets=False)
        mlflow.lightgbm.autolog(silent=True, log_datasets=False)

        logging.info(f"Training model='{config.model_name}' hyperparameters={config.hyperparameters}")

        model_class = MODEL_REGISTRY.get(config.model_name)
        if model_class is None:
            raise ValueError(
                f"Model '{config.model_name}' unsupported. "
                f"Choose one of: {list(MODEL_REGISTRY.keys())}"
            )

        model_instance = model_class()
        return model_instance.train(x_train, y_train, **config.hyperparameters)

    except Exception as e:
        logging.error(f"Pipeline tracking error during training invocation: {e}")
        raise e
