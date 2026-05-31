import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import cast
import click
from rich import print
from dotenv import load_dotenv

from pipelines.deployment_pipeline import (
    continuous_deployment_pipeline,
    inference_pipeline,
)
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService

load_dotenv()

DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"


@click.command()
@click.option("--config", "-c",
              type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
              default=DEPLOY_AND_PREDICT)
@click.option("--min-f1", default=0.60,
              help="Minimum F1 required to deploy the model.")
def main(config: str, min_f1: float):
    deployer = MLFlowModelDeployer.get_active_model_deployer()
    do_deploy = config in (DEPLOY, DEPLOY_AND_PREDICT)
    do_predict = config in (PREDICT, DEPLOY_AND_PREDICT)

    data_path = os.getenv(
        "SEED_CSV_PATH",
        "./data/raw_data/predictive_maintenance_dataset.csv",
    )

    if do_deploy:
        continuous_deployment_pipeline.with_options(
            config_path="config.yaml",
            enable_cache=False,
        )(data_path=data_path, min_f1=min_f1, workers=3, timeout=300)

    if do_predict:
        inference_pipeline(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
        )

    print(
        f"\n[italic green]mlflow ui --backend-store-uri '{get_tracking_uri()}'[/italic green]"
        "\n...to inspect runs in the MLflow UI.\n"
    )

    services = deployer.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        model_name="model",
    )
    if services:
        svc = cast(MLFlowDeploymentService, services[0])
        if svc.is_running:
            print(f"MLflow prediction server is live at:\n    {svc.prediction_url}\n")
        elif svc.is_failed:
            print(f"MLflow server failed: {svc.status.last_error}")
    else:
        print("No MLflow server running. Run with --config deploy first.")


if __name__ == "__main__":
    main()
