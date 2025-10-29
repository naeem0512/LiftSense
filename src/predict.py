from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

from .config import AppConfig, config
from .data_simulator import simulate_session
from .datasets import make_windows
from .features import FEATURE_COLUMNS, build_features
from .models.baseline import BaselineModel
from .models.random_forest import RandomForestModel
from .models.bilstm import BiLSTMModel
from .persistence import ensure_dir, save_json


MODEL_PATHS = {
    "baseline": config.artifacts.baseline,
    "random_forest": config.artifacts.random_forest,
    "bilstm": config.artifacts.bilstm,
}


def load_model(model_type: str, model_path: Optional[Path] = None):
    path = Path(model_path) if model_path else Path(MODEL_PATHS[model_type])
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path}. Train the model first.")
    if model_type == "baseline":
        return BaselineModel.load(str(path))
    if model_type == "random_forest":
        return RandomForestModel.load(path)
    if model_type == "bilstm":
        return BiLSTMModel.load(path)
    raise ValueError(f"Unknown model type: {model_type}")


def run_prediction(
    cfg: AppConfig,
    df: pd.DataFrame,
    model_type: str,
    window_size: int,
    stride: int,
    model_path: Optional[Path] = None,
) -> Dict[str, Any]:
    artifact_path = Path(model_path) if model_path else Path(MODEL_PATHS[model_type])
    model = load_model(model_type, artifact_path)
    feature_df = build_features(df)

    if model_type == "bilstm":
        dataset = make_windows(df, window_size, stride, FEATURE_COLUMNS)
        X = dataset.features
        preds = model.predict(X)
        probs = model.predict_proba(X)
    else:
        X = feature_df.to_numpy()
        preds = model.predict(X)
        probs = model.predict_proba(X)

    distribution = (
        pd.Series(preds).value_counts().sort_index().rename_axis("fatigue_class")
    )

    summary = {
        "model": model_type,
        "num_reps": int(df.shape[0]),
        "predictions": preds.tolist(),
        "probabilities_head": probs[:5].tolist(),
        "feature_means": feature_df.mean().to_dict(),
        "class_distribution": {
            str(int(k)): {"count": int(v), "ratio": float(v / len(preds))}
            for k, v in distribution.items()
        },
        "artifact_path": str(artifact_path),
        "window": {
            "size": window_size,
            "stride": stride,
            "applied": model_type == "bilstm",
        },
    }

    ensure_dir(cfg.output_dir)
    save_json(summary, cfg.output_dir / "latest_session.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict fatigue on a session")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--simulate", type=int, help="Simulate a session when set to 1")
    group.add_argument("--csv", type=Path, help="Path to CSV containing session data")
    parser.add_argument("--model", choices=["baseline", "random_forest", "bilstm"], required=True)
    parser.add_argument("--model-path", type=Path, help="Override the saved model path")
    parser.add_argument("--window-size", type=int, default=config.windowing.window_size)
    parser.add_argument("--stride", type=int, default=config.windowing.stride)
    args = parser.parse_args()

    cfg = config

    if args.simulate == 1:
        df = simulate_session(cfg, save_plot=False).dataframe
    elif args.csv:
        df = pd.read_csv(args.csv)
    else:
        raise ValueError("Either --simulate 1 or --csv must be provided")

    result = run_prediction(
        cfg,
        df,
        args.model,
        args.window_size,
        args.stride,
        model_path=args.model_path,
    )
    logger.info("Prediction summary saved: {}", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
