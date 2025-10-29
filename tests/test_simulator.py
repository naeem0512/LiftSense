from __future__ import annotations

import numpy as np

from src.config import config
from src.data_simulator import simulate_session


def test_simulator_ranges():
    result = simulate_session(config, save_plot=False)
    df = result.dataframe

    assert not df.isna().any().any()
    assert df["velocity"].between(0.05, 1.5).all()
    assert df["force"].between(40, 600).all()
    assert df["grip"].between(5, 70).all()
    assert df["heart_rate"].between(50, 200).all()
    assert set(df["fatigue"].unique()).issubset({0, 1, 2})
