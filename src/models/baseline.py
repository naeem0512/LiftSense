from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class BaselineModel:
    pipeline: Pipeline

    @classmethod
    def create(cls) -> "BaselineModel":
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=300, multi_class="auto")),
        ])
        return cls(pipeline=pipeline)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.pipeline.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def save(self, path: str) -> None:
        from joblib import dump

        dump(self.pipeline, path)

    @classmethod
    def load(cls, path: str) -> "BaselineModel":
        from joblib import load

        pipeline = load(path)
        return cls(pipeline=pipeline)

    def report(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        preds = self.predict(X)
        return classification_report(y, preds, output_dict=True)
