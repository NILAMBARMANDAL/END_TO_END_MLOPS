import os
import json
import numpy as np
import pandas as pd
import logging

from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.client import Client

# MLflow Integrations
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService

# Step modules
import steps.ingest_data as ingest_mod
import steps.clean_data as clean_mod
import steps.model_train as train_mod
import steps.evaluation as eval_mod
import steps.deployment_trigger as trigger_mod

from pipelines.utils import get_data_for_test

logging.basicConfig(level=logging.INFO)
docker_settings = DockerSettings(required_integrations=["mlflow"])


# ==========================================
#          INFERENCE PIPELINE STEPS
# ==========================================
@step(enable_cache=False)
def dynamic_importer() -> str:
    """Fetch a small JSON batch for batch-inference testing."""
    return get_data_for_test()


@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:
    """Locate the running MLflow model server started by the deployment step."""
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()
    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )
    if not existing_services:
        raise RuntimeError("No MLflow prediction service is deployed.")
    return existing_services[0]


@step
def predictor(service: MLFlowDeploymentService, data: str) -> np.ndarray:
    """Run batch predictions. Works via the MLflow daemon, or falls back to loading the
    trained model straight from ZenML storage (needed on Windows where the daemon is flaky)."""
    try:
        parsed = json.loads(data)
        df_inf = pd.DataFrame(parsed["data"])

        client = Client()
        last_run = client.get_pipeline("continuous_deployment_pipeline").last_successful_run
        trained_model = last_run.steps["train_model"].output.load()

        # align columns to what the model was trained on
        try:
            expected = list(trained_model.feature_names_in_)
            df_inf.columns = expected[: len(df_inf.columns)]
            df_inf = df_inf.reindex(columns=expected, fill_value=0)
            final_array = df_inf.to_numpy()
        except AttributeError:
            n = df_inf.shape[0]
            final_array = np.zeros((n, 19))
            cols = min(19, df_inf.shape[1])
            final_array[:, :cols] = df_inf.iloc[:, :cols].to_numpy()

        if service.is_running:
            logging.info("Predicting via active MLflow server.")
            return service.predict(final_array)
        logging.warning("MLflow daemon inactive -> inline fallback prediction.")
        preds = trained_model.predict(final_array)
        print(f"\n[SUCCESS] PREDICTIONS:\n{preds}\n")
        return preds

    except Exception as e:
        logging.error(f"Error in predictor step: {e}")
        raise e


# ==========================================
#                PIPELINES
# ==========================================
@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    data_path: str = "./data/raw_data/predictive_maintenance_dataset.csv",
    min_f1: float = 0.60,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    """Train -> evaluate -> (gate) -> deploy the model automatically if it is good enough."""
    df = ingest_mod.ingest_df(data_path=data_path)
    x_train, x_test, y_train, y_test = clean_mod.clean_df(df)

    # config (model_name + hyperparameters) is injected from config.yaml at runtime
    model = train_mod.train_model(
        x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test
    )

    accuracy, f1, auc = eval_mod.evaluation(model, x_test, y_test)

    # gate on F1 (correct for imbalanced data), not on the meaningless r2_score
    deploy_decision = trigger_mod.deployment_trigger(current_f1=f1, min_f1=min_f1)

    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deploy_decision,
        workers=workers,
        timeout=timeout,
    )


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    """Batch inference against the deployed model."""
    batch_data = dynamic_importer()
    service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
    )
    predictor(service=service, data=batch_data)
