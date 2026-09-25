"""Pairwise classification model."""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class MatcherConfig:
    C: float = 2.0
    class_weight: str | dict | None = "balanced"
    max_iter: int = 2000
    random_state: int = 42


class PairMatcher:
    def __init__(self, config: MatcherConfig | None = None):
        self.config = config or MatcherConfig()
        self.model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=self.config.C,
                        class_weight=self.config.class_weight,
                        max_iter=self.config.max_iter,
                        random_state=self.config.random_state,
                    ),
                ),
            ]
        )
        self.feature_names: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PairMatcher":
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.feature_names is None:
            raise RuntimeError("Model has not been fitted.")
        X = X.reindex(columns=self.feature_names, fill_value=0.0)
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "PairMatcher":
        obj = joblib.load(path)
        if not isinstance(obj, PairMatcher):
            raise TypeError(f"Unexpected model type in {path}")
        return obj
