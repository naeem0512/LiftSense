from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from .config import AppConfig, config
from .data_simulator import save_dataframe, simulate_session
from .datasets import make_windows
from .features import FEATURE_COLUMNS, build_features
from .models.baseline import BaselineModel
from .models.random_forest import RandomForestModel
from .models.bilstm import BiLSTMModel
from .persistence import ensure_dir, save_json


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro")),
    }


def train_baseline(
    cfg: AppConfig,
    df: pd.DataFrame,
    output_dir: Path,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    X = build_features(df).to_numpy()
    y = df["fatigue"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.training.test_size,
        random_state=cfg.random_seed,
        stratify=y,
    )

    model = BaselineModel.create()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = evaluate_predictions(y_test, preds)

    model_path = cfg.artifacts.baseline
    ensure_dir(model_path.parent)
    model.save(str(model_path))

    return metrics, {"report": model.report(X_test, y_test)}


def train_random_forest(
    cfg: AppConfig,
    df: pd.DataFrame,
    output_dir: Path,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    X = build_features(df).to_numpy()
    y = df["fatigue"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.training.test_size,
        random_state=cfg.random_seed,
        stratify=y,
    )

    model = RandomForestModel.create(cfg.random_seed)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = evaluate_predictions(y_test, preds)

    model.feature_importance(
        X_test,
        y_test,
        feature_names=tuple(FEATURE_COLUMNS),
        output_dir=output_dir,
        seed=cfg.random_seed,
    )

    model_path = cfg.artifacts.random_forest
    ensure_dir(model_path.parent)
    model.save(model_path)

    return metrics, {"report": model.report(X_test, y_test)}


def train_bilstm(
    cfg: AppConfig,
    df: pd.DataFrame,
    window_size: int,
    stride: int,
    output_dir: Path,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    dataset = make_windows(df, window_size, stride, FEATURE_COLUMNS)
    X = dataset.features
    y = dataset.labels

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=cfg.training.test_size,
        random_state=cfg.random_seed,
        stratify=y,
    )

    model = BiLSTMModel.create((X.shape[1], X.shape[2]), cfg.training.lr)
    history_info = model.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=cfg.training.max_epochs,
        batch_size=cfg.training.batch_size,
        patience=cfg.training.patience,
        output_dir=output_dir,
    )

    preds = model.predict(X_val)
    metrics = evaluate_predictions(y_val, preds)

    model_path = cfg.artifacts.bilstm
    ensure_dir(model_path.parent)
    model.save(model_path)

    report = classification_report(y_val, preds, output_dict=True)
    extras = {
        "history": history_info["history"],
        "plot_path": history_info["plot_path"],
        "report": report,
    }

    return metrics, extras


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LiftSense models")
    parser.add_argument("--model", choices=["baseline", "random_forest", "bilstm"], required=True)
    parser.add_argument("--windows", type=int, default=0, help="Use sliding windows")
    parser.add_argument("--window-size", type=int, default=config.windowing.window_size)
    parser.add_argument("--stride", type=int, default=config.windowing.stride)
    parser.add_argument("--save-csv", type=Path, default=None)
    args = parser.parse_args()

    cfg = config
    ensure_dir(cfg.output_dir)

    start_time = time.time()
    simulation = simulate_session(cfg)
    df = simulation.dataframe
    if args.save_csv:
        save_dataframe(df, args.save_csv)

    metrics: Dict[str, float]
    extras: Dict[str, float]

    if args.model == "baseline":
        metrics, extras = train_baseline(cfg, df, cfg.output_dir)
    elif args.model == "random_forest":
        metrics, extras = train_random_forest(cfg, df, cfg.output_dir)
    elif args.model == "bilstm":
        metrics, extras = train_bilstm(cfg, df, args.window_size, args.stride, cfg.output_dir)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    elapsed = time.time() - start_time

    report = {
        "model": args.model,
        "metrics": metrics,
        "extras": extras,
        "timing": {"seconds": elapsed},
        "windowing": {
            "enabled": bool(args.windows),
            "window_size": args.window_size,
            "stride": args.stride,
        },
    }

    metrics_path = cfg.output_dir / "training_metrics.json"
    save_json(report, metrics_path)

    logger.info("Training complete: {}", metrics)


if __name__ == "__main__":
    main()
