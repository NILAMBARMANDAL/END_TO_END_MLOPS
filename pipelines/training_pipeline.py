from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import evaluation


@pipeline(enable_cache=False)
def maintenance_training_pipeline(data_path: str = ""):
    """End-to-end training: ingest -> clean -> train -> evaluate.
    config (model_name + hyperparameters) is injected from config.yaml at runtime."""
    df = ingest_df(data_path=data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(x_train=X_train, x_test=X_test, y_train=y_train, y_test=y_test)
    accuracy, f1, auc = evaluation(model, X_test, y_test)
