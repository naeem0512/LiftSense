from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def train_test_split_df(df: pd.DataFrame, test_size: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["fatigue"])
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


@dataclass
class WindowedDataset:
    features: np.ndarray
    labels: np.ndarray


def make_windows(
    df: pd.DataFrame,
    window_size: int,
    stride: int,
    feature_cols: Sequence[str],
    label_col: str = "fatigue",
) -> WindowedDataset:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    data = df[feature_cols].to_numpy(dtype=float)
    labels = df[label_col].to_numpy(dtype=int)
    num_samples = data.shape[0]
    windows: List[np.ndarray] = []
    window_labels: List[int] = []

    for start in range(0, num_samples - window_size + 1, stride):
        end = start + window_size
        window = data[start:end]
        label_window = labels[start:end]
        majority_label = np.bincount(label_window, minlength=3).argmax()
        windows.append(window)
        window_labels.append(int(majority_label))

    if not windows:
        raise ValueError("No windows were created; check window_size and stride")

    return WindowedDataset(features=np.stack(windows), labels=np.array(window_labels, dtype=int))
