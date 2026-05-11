"""
MedAI inference preprocessor — must live in a stable module so it can be unpickled.
"""
import numpy as np
import pandas as pd
from typing import List


class MedAIPreprocessor:
    """Stateful sklearn-based preprocessor for MedAI inference."""

    def __init__(self, imputer, scaler, feature_names: List[str], continuous_cols: List[str]):
        self.imputer = imputer
        self.scaler = scaler
        self.feature_names = feature_names
        self.continuous_cols = continuous_cols
        self.feature_order = feature_names
        self.is_fitted = True

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.feature_names].copy()
        X_imp = pd.DataFrame(
            self.imputer.transform(X),
            columns=self.feature_names,
        )
        X_imp[self.continuous_cols] = self.scaler.transform(X_imp[self.continuous_cols])
        return X_imp[self.feature_names].values
