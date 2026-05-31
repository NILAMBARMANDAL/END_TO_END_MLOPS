import logging
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
import os
import joblib
class DataCleaning:
    """Replicates the exact feature engineering and balancing sequence from the EDA notebook."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def execute_cleaning_and_splitting(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Runs the notebook transformations, under-sampling, and standard scaling."""
        try:
            logging.info("Executing notebook feature engineering sequence inside src pipeline layer.")
            
            # 1. Replicate notebook Cell 6 & 10: Extract device_model and drop 'Z1F2'
            if "device" in self.df.columns:
                self.df["device_model"] = self.df["device"].apply(lambda x: str(x)[:4])
                self.df = self.df[self.df["device_model"] != "Z1F2"].reset_index(drop=True)
                self.df.drop("device", axis=1, inplace=True)
            
            # 2. Replicate notebook Cell 12: Drop device_rest
            if "device_rest" in self.df.columns:
                self.df.drop("device_rest", axis=1, inplace=True)

            # 3. Replicate notebook Cell 17 & 19: Extract time intelligence features
            if "date" in self.df.columns:
                self.df['date'] = pd.to_datetime(self.df['date'])
                self.df['day_of_week'] = self.df['date'].dt.dayofweek
                self.df['day_of_month'] = self.df['date'].dt.day
                self.df['is_weekend'] = self.df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
                self.df['month'] = self.df['date'].dt.month
                self.df['week'] = self.df['date'].dt.isocalendar().week.astype(int)
                self.df.drop(['date'], axis=1, inplace=True)
            
            # 4. Replicate notebook Cell 20: Get dummies for categorical variables
            self.df = pd.get_dummies(self.df, drop_first=True)
            
            # Ensure boolean columns from get_dummies are explicitly integers for Scikit-Learn
            for col in self.df.select_dtypes(include=['bool']).columns:
                self.df[col] = self.df[col].astype(int)

            # Isolate features and target
            X = self.df.drop(columns=["failure"]) 
            y = self.df["failure"]
            
            # 5. Replicate notebook Cell 24: Apply RandomUnderSampler to full dataset
            logging.info(f"Target count before sampling: {np.bincount(y)}")
            rus = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = rus.fit_resample(X, y)
            
            # 6. Replicate notebook Cell 26: Split 80/20
            X_train, X_test, y_train, y_test = train_test_split(
                X_resampled, y_resampled, test_size=0.2, random_state=42
            )
            
         # 7. Replicate notebook Cell 26: Standardize features
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        
            os.makedirs("model", exist_ok=True)
            joblib.dump(scaler, "model/scaler.joblib")
            logging.info("Saved fitted scaler to model/scaler.joblib")

            logging.info(f"Notebook architecture alignment complete! Matrix shape: {X_train_scaled.shape}")
            
            return X_train_scaled, X_test_scaled, y_train, y_test
            
        except Exception as e:
            logging.error(f"Error executing notebook processing block: {e}")
            raise e