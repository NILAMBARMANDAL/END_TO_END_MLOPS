import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

logging.basicConfig(level=logging.INFO)


# 1. Base Abstract Class  (Factory / Strategy pattern)
class Model(ABC):
    """Abstract base for all model wrappers. Every model implements .train()."""
    @abstractmethod
    def train(self, x_train, y_train, **hyperparameters):
        """Train and return a fitted model. **hyperparameters lets config.yaml tune the model."""
        pass


# 2. Logistic Regression Wrapper (kept as an interpretable baseline)
class LogisticRegressionModel(Model):
    """Classifier baseline for binary failure prediction."""
    def train(self, x_train, y_train, **hyperparameters):
        try:
            logging.info(f"Training Logistic Regression with overrides: {hyperparameters}")
            params = {"max_iter": 1000, "class_weight": "balanced", "random_state": 42}
            params.update(hyperparameters)  # config.yaml values win
            clf = LogisticRegression(**params)
            clf.fit(x_train, y_train)
            return clf
        except Exception as e:
            logging.error(f"Error while training Logistic Regression: {e}")
            raise e


# 3. Random Forest Wrapper (YOUR chosen production model)
class RandomForestModel(Model):
    """Random Forest classifier. Hyperparameters are injected from config.yaml so you can
    tune the model WITHOUT touching code -> reproducible, config-driven experiments."""
    def train(self, x_train, y_train, **hyperparameters):
        try:
            logging.info(f"Training Random Forest with overrides: {hyperparameters}")
            # sensible defaults; any key present in config.yaml overrides these
            params = {
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": -1,
            }
            params.update(hyperparameters)
            clf = RandomForestClassifier(**params)
            clf.fit(x_train, y_train)
            return clf
        except Exception as e:
            logging.error(f"Error within Random Forest framework: {e}")
            raise e


# 4. XGBoost Wrapper
class XGBoostModel(Model):
    """XGBoost classifier for balanced data."""
    def train(self, x_train, y_train, **hyperparameters):
        try:
            logging.info(f"Training XGBoost with overrides: {hyperparameters}")
            params = {
                "n_estimators": 100,
                "scale_pos_weight": 1,
                "eval_metric": "logloss",
                "random_state": 42,
            }
            params.update(hyperparameters)
            clf = XGBClassifier(**params)
            clf.fit(x_train, y_train)
            return clf
        except Exception as e:
            logging.error(f"Error within XGBoost framework: {e}")
            raise e


# 5. LightGBM Wrapper
class LightGBMModel(Model):
    """LightGBM classifier for fast tree-based training."""
    def train(self, x_train, y_train, **hyperparameters):
        try:
            logging.info(f"Training LightGBM with overrides: {hyperparameters}")
            params = {"n_estimators": 100, "class_weight": "balanced", "random_state": 42}
            params.update(hyperparameters)
            clf = LGBMClassifier(**params)
            clf.fit(x_train, y_train)
            return clf
        except Exception as e:
            logging.error(f"Error within LightGBM framework: {e}")
            raise e


# 6. Linear Regression Wrapper (kept for completeness; NOT for this binary target)
class LinearRegressionModel(Model):
    """Linear Regression. failure is binary, so this is reference-only, never deployed."""
    def train(self, x_train, y_train, **hyperparameters):
        try:
            logging.info("Training Linear Regression (reference only).")
            reg = LinearRegression(**hyperparameters)
            reg.fit(x_train, y_train)
            return reg
        except Exception as e:
            logging.error(f"Error within Linear Regression framework: {e}")
            raise e
