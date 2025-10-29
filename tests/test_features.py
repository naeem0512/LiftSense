from __future__ import annotations

import pandas as pd

from src.features import FEATURE_COLUMNS, build_features


def test_build_features_returns_expected_columns():
    data = {col: list(range(5)) for col in FEATURE_COLUMNS}
    df = pd.DataFrame(data)
    features = build_features(df)
    assert list(features.columns) == FEATURE_COLUMNS
