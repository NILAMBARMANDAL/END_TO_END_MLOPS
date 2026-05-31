import logging
import pandas as pd
from typing import Tuple
from zenml import step
from src.clean_data import DataCleaning  # Import the worker class

@step
def clean_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """ZenML step orchestrator that invokes the underlying src data cleaning logic."""
    try:
        logging.info("ZenML Orchestrator: Invoking structural cleaning engine from src.")
        
        # Instantiate your src worker class and run it
        cleaner = DataCleaning(df)
        X_train, X_test, y_train, y_test = cleaner.execute_cleaning_and_splitting()
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logging.error(f"Failed to execute clean_df step: {e}")
        raise e