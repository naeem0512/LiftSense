from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.baseline import BaselineModel
from src.models.random_forest import RandomForestModel


def _make_dataset(n_samples: int = 30) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_samples, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def test_baseline_model_fit_predict():
    X, y = _make_dataset()
    model = BaselineModel.create()
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape


def test_random_forest_fit_predict():
    X, y = _make_dataset()
    model = RandomForestModel.create(seed=0)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape
