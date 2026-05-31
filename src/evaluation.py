from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

class Evaluation(ABC):
    """Abstract strategy block for performance assessments."""
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> float:
        pass

class AccuracyScore(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> float:
        return float(accuracy_score(y_true, y_pred))

class F1Score(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> float:
        return float(f1_score(y_true, y_pred))

class RocAucScore(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> float:
        return float(roc_auc_score(y_true, y_proba))