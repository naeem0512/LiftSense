from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report

from ..persistence import ensure_dir, save_json


@dataclass
class RandomForestModel:
    estimator: RandomForestClassifier

    @classmethod
    def create(cls, seed: int) -> "RandomForestModel":
        estimator = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=seed,
        )
        return cls(estimator=estimator)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.estimator.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(X)

    def save(self, path: Path) -> None:
        from joblib import dump

        ensure_dir(path.parent)
        dump(self.estimator, path)

    @classmethod
    def load(cls, path: Path) -> "RandomForestModel":
        from joblib import load

        estimator = load(path)
        return cls(estimator=estimator)

    def report(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        preds = self.predict(X)
        return classification_report(y, preds, output_dict=True)

    def feature_importance(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: Tuple[str, ...],
        output_dir: Path,
        seed: int,
    ) -> Dict[str, Any]:
        impurity = self.estimator.feature_importances_
        perm = permutation_importance(
            self.estimator,
            X_test,
            y_test,
            n_repeats=10,
            random_state=seed,
            n_jobs=-1,
        )

        data = {
            "feature_names": list(feature_names),
            "impurity_importance": impurity.tolist(),
            "permutation_mean": perm.importances_mean.tolist(),
            "permutation_std": perm.importances_std.tolist(),
        }

        json_path = output_dir / "feature_importance.json"
        plot_path = output_dir / "plots" / "feature_importance.png"
        ensure_dir(plot_path.parent)
        save_json(data, json_path)

        order = np.argsort(impurity)[::-1]
        plt.figure(figsize=(8, 4))
        plt.bar(np.array(feature_names)[order], impurity[order])
        plt.ylabel("Importance")
        plt.title("Random Forest Feature Importance")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        return data
