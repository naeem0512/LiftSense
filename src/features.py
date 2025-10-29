from __future__ import annotations

from typing import Iterable, List

import pandas as pd

FEATURE_COLUMNS: List[str] = [
    "velocity",
    "force",
    "grip",
    "heart_rate",
    "hr_delta",
    "power",
    "efficiency",
]


def build_features(df: pd.DataFrame, feature_cols: Iterable[str] | None = None) -> pd.DataFrame:
    columns = list(feature_cols) if feature_cols is not None else FEATURE_COLUMNS
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")
    return df[columns].copy()
