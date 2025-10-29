from __future__ import annotations

import numpy as np
import pandas as pd

from src.datasets import make_windows


def test_make_windows_stride():
    data = {
        "velocity": np.arange(30),
        "force": np.arange(30),
        "grip": np.arange(30),
        "heart_rate": np.arange(30),
        "hr_delta": np.arange(30),
        "power": np.arange(30),
        "efficiency": np.arange(30),
        "fatigue": np.tile([0, 1, 2], 10)[:30],
    }
    df = pd.DataFrame(data)
    dataset = make_windows(df, window_size=5, stride=2, feature_cols=list(df.columns[:-1]))
    expected_windows = (30 - 5) // 2 + 1
    assert dataset.features.shape[0] == expected_windows
    assert dataset.features.shape[1:] == (5, len(df.columns) - 1)
