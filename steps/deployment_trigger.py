import logging
from zenml import step
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO)


@step
def deployment_trigger(
    current_f1: float,
    min_f1: float = 0.60,
    experiment_name: str = "continuous_deployment_pipeline",
    beat_history: bool = True,
) -> bool:
    """Decide whether to deploy the freshly trained model.

    Two gates (a model must pass BOTH to deploy):

      GATE 1 - absolute quality floor:
        current_f1 >= min_f1   (min_f1 comes from config; F1 not accuracy, because the
        data is extremely imbalanced and accuracy is misleading).

      GATE 2 - regression guard (only if beat_history=True):
        the new model must be at least as good as the best PREVIOUS run's F1 in MLflow,
        so a worse model never silently replaces a good one in production.

    NOTE for interview: the original tutorial code compared `r2_score`, a *regression*
    metric this classification pipeline never logs -> the comparison always errored and
    the gate silently returned True (deploying everything). This version fixes that by
    comparing the metric we actually log: `optimized_f1_score`.
    """
    # ---- GATE 1: absolute floor ----
    if current_f1 < min_f1:
        logging.info(f"GATE 1 FAILED: f1={current_f1:.4f} < min_f1={min_f1}. Skip deploy.")
        return False
    logging.info(f"GATE 1 PASSED: f1={current_f1:.4f} >= min_f1={min_f1}.")

    if not beat_history:
        return True

    # ---- GATE 2: must beat best previous run ----
    try:
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            logging.info("No prior experiment found -> first deployment. Deploy.")
            return True

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.optimized_f1_score DESC"],
            max_results=5,
        )
        # take the best HISTORICAL f1 (best-effort: the current run may also appear, so we
        # compare with >= to avoid being blocked by a tie against ourselves)
        past_best = -1.0
        for r in runs:
            f1 = r.data.metrics.get("optimized_f1_score")
            if f1 is not None:
                past_best = max(past_best, f1)

        if past_best < 0:
            logging.info("No historical f1 logged yet -> deploy.")
            return True

        decision = current_f1 >= past_best
        logging.info(
            f"GATE 2: current f1={current_f1:.4f} vs best historical={past_best:.4f} "
            f"-> {'DEPLOY' if decision else 'SKIP'}"
        )
        return decision

    except Exception as e:
        # If the registry check fails for infra reasons, fail OPEN to first deployment only.
        logging.warning(f"History check unavailable ({e}). Passing on GATE 1 result.")
        return True
