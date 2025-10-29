from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List

import numpy as np

from .config import config
from .data_simulator import simulate_session
from .features import FEATURE_COLUMNS, build_features
from .models.baseline import BaselineModel
from .models.bilstm import BiLSTMModel
from .models.random_forest import RandomForestModel


@dataclass
class SlidingWindowPredictor:
    window_size: int
    stride: int
    model
    feature_list: List[str] = field(default_factory=lambda: FEATURE_COLUMNS)
    buffer: Deque[Dict[str, float]] = field(init=False)
    step_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.buffer = deque(maxlen=self.window_size)

    def push(self, rep: Dict[str, float]) -> Dict[str, float] | None:
        self.buffer.append(rep)
        self.step_count += 1
        if len(self.buffer) < self.window_size:
            return None
        if (self.step_count - self.window_size) % self.stride != 0:
            return None
        window = np.array([[entry[feat] for feat in self.feature_list] for entry in self.buffer], dtype=float)
        if hasattr(self.model, "model"):
            probs = self.model.predict_proba(window[None, :, :])[0]
        else:
            aggregated = window.mean(axis=0, keepdims=True)
            probs = self.model.predict_proba(aggregated)[0]
        prediction = int(np.argmax(probs))
        return {"prediction": prediction, "probabilities": probs.tolist()}


def load_model(model_type: str):
    if model_type == "bilstm":
        return BiLSTMModel.load(Path("results/model.h5"))
    if model_type == "baseline":
        return BaselineModel.load("results/model.joblib")
    if model_type == "random_forest":
        return RandomForestModel.load(Path("results/model.joblib"))
    raise ValueError(f"Unsupported model type: {model_type}")


def stream_predictions(model_type: str, window_size: int, stride: int) -> List[Dict[str, float]]:
    session = simulate_session(config, save_plot=False).dataframe
    features = build_features(session)
    records = features.to_dict(orient="records")

    model = load_model(model_type)
    predictor = SlidingWindowPredictor(window_size, stride, model)

    outputs: List[Dict[str, float]] = []
    for rep in records:
        result = predictor.push(rep)
        if result:
            outputs.append(result)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time windowed predictions")
    parser.add_argument("--model", choices=["baseline", "random_forest", "bilstm"], required=True)
    parser.add_argument("--window-size", type=int, default=config.windowing.window_size)
    parser.add_argument("--stride", type=int, default=config.windowing.stride)
    args = parser.parse_args()

    outputs = stream_predictions(args.model, args.window_size, args.stride)
    print(json.dumps(outputs[:5], indent=2))


if __name__ == "__main__":
    main()
