import os
import logging
import pandas as pd
from src.clean_data import DataCleaning

logging.basicConfig(level=logging.INFO)


def get_data_for_test() -> str:
    """Build a small JSON batch for inference testing.

    Reads a sample, runs the production cleaning, drops the target + any text columns,
    and returns a split-oriented JSON string. Path comes from env (no hard-coded laptop path).
    """
    try:
        csv_path = os.getenv(
            "SEED_CSV_PATH",
            "./data/raw_data/predictive_maintenance_dataset.csv",
        )
        df = pd.read_csv(csv_path).sample(n=100, random_state=42)

        try:
            X_train, _, _, _ = DataCleaning(df).execute_cleaning_and_splitting()
            df_cleaned = pd.DataFrame(X_train)
        except Exception:
            df_cleaned = df.copy()

        for col in ["Target", "failure", "review_score"]:
            if col in df_cleaned.columns:
                df_cleaned.drop(columns=[col], inplace=True)

        df_numeric = df_cleaned.select_dtypes(include=["number"])
        return df_numeric.to_json(orient="split")

    except Exception as e:
        logging.error(f"Error generating test batch: {e}")
        raise e
