import logging
from abc import ABC, abstractmethod
from typing import Union, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """Abstract Class defining strategy for handling data."""
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, Tuple]:
        pass

class DataPreprocessStrategy(DataStrategy):
    """Strategy for cleaning predictive maintenance data."""
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            # Dropping operational identifiers not required for mathematical relationships
            columns_to_drop = ["date", "device"]
            data = data.drop(columns=columns_to_drop, errors="ignore")
            
            # Numeric filtering & missing value imputation
            data = data.select_dtypes(include=[np.number])
            data = data.fillna(data.median())
            return data
        except Exception as e:
            logging.error(f"Error in data preprocessing strategy: {e}")
            raise e

class DataDivideStrategy(DataStrategy):
    """Strategy for separating target labels and splitting dataset into train/test splits."""
    def handle_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        try:
            # 'failure' is the primary target binary metric
            X = data.drop(columns=["failure"], errors="ignore")
            y = data["failure"]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error in data splitting strategy: {e}")
            raise e

class DataCleaning:
    """Context class prioritizing specific strategy implementations."""
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy
        
    def handle_data(self) -> Union[pd.DataFrame, Tuple]:
        return self.strategy.handle_data(self.data)