from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import AppConfig, config


@dataclass
class SimulationResult:
    dataframe: pd.DataFrame
    plot_path: Optional[Path]


def _z_scores(values: np.ndarray) -> np.ndarray:
    mean = values.mean()
    std = values.std()
    if std == 0:
        return np.zeros_like(values)
    return (values - mean) / std


def simulate_session(cfg: AppConfig | None = None, save_plot: bool = True) -> SimulationResult:
    cfg = cfg or config
    sim_cfg = cfg.simulation
    rng = np.random.default_rng(cfg.random_seed)

    reps = sim_cfg.reps
    t = np.linspace(0.0, 1.0, reps)

    velocity_mean = rng.normal(sim_cfg.vel_mu_sigma[0], sim_cfg.vel_mu_sigma[1])
    force_mean = rng.normal(sim_cfg.force_mu_sigma[0], sim_cfg.force_mu_sigma[1])
    grip_mean = rng.normal(sim_cfg.grip_mu_sigma[0], sim_cfg.grip_mu_sigma[1])

    velocity_sigma = sim_cfg.vel_mu_sigma[1]
    force_sigma = sim_cfg.force_mu_sigma[1]
    grip_sigma = sim_cfg.grip_mu_sigma[1]

    k_v, k_f, k_g = 0.25, 0.15, 0.05

    velocity_trend = velocity_mean * (1 - k_v * t)
    force_trend = force_mean * (1 - k_f * t)
    grip_trend = grip_mean * (1 - k_g * t)

    velocity = rng.normal(velocity_trend, velocity_sigma)
    force = rng.normal(force_trend, force_sigma)
    grip = rng.normal(grip_trend, grip_sigma)

    velocity = np.clip(velocity, 0.05, 1.5)
    force = np.clip(force, 40, 600)
    grip = np.clip(grip, 5, 70)

    hr_start = rng.uniform(sim_cfg.hr_rest[0], sim_cfg.hr_rest[1])
    hr_target = rng.uniform(sim_cfg.hr_work[0], sim_cfg.hr_work[1])

    hr_lin = hr_start + t * (hr_target - hr_start)

    sigma_hr = 3.0
    ar = sim_cfg.ar_hr
    noise = rng.normal(0.0, sigma_hr, size=reps)
    ar_errors = np.zeros(reps)
    for i in range(1, reps):
        ar_errors[i] = ar * ar_errors[i - 1] + noise[i]
    heart_rate = hr_lin + ar_errors
    heart_rate = np.clip(heart_rate, 50, 200)

    velocity += rng.normal(0.0, sim_cfg.noise * 0.2, size=reps)
    force += rng.normal(0.0, sim_cfg.noise * 25, size=reps)
    grip += rng.normal(0.0, sim_cfg.noise * 2, size=reps)

    heart_rate = np.clip(heart_rate, 50, 200)

    df = pd.DataFrame(
        {
            "rep": np.arange(1, reps + 1),
            "velocity": velocity,
            "force": force,
            "grip": grip,
            "heart_rate": heart_rate,
        }
    )

    df["hr_delta"] = df["heart_rate"].diff().fillna(0.0)
    df["power"] = df["velocity"] * df["force"]
    df["efficiency"] = df["power"] / (df["heart_rate"] - hr_start + 1)

    fatigue_score = _z_scores(-df["velocity"].to_numpy()) + _z_scores(-df["force"].to_numpy()) + _z_scores(
        df["heart_rate"].to_numpy()
    )

    tertiles = np.quantile(fatigue_score, [1 / 3, 2 / 3])
    fatigue = np.digitize(fatigue_score, tertiles, right=False)

    flip_mask = rng.random(reps) < 0.05
    fatigue = np.where(flip_mask, rng.integers(0, 3, size=reps), fatigue)

    df["fatigue"] = fatigue.astype(int)

    plot_path = None
    if save_plot:
        plot_path = cfg.output_dir / "plots" / "simulated_biometrics.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(df["rep"], df["velocity"], label="Velocity (m/s)")
        plt.plot(df["rep"], df["force"], label="Force (N)")
        plt.plot(df["rep"], df["heart_rate"], label="Heart Rate (bpm)")
        plt.xlabel("Rep")
        plt.ylabel("Value")
        plt.title("Simulated Biometrics")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    return SimulationResult(dataframe=df, plot_path=plot_path)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main(save_csv: Optional[Path] = None) -> None:
    cfg = config
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    result = simulate_session(cfg)
    if save_csv is not None:
        save_dataframe(result.dataframe, save_csv)
    print(json.dumps(result.dataframe.head().to_dict(orient="records"), indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simulate LiftSense session")
    parser.add_argument("--save-csv", type=Path, default=None, help="Optional path to save the simulated session")
    args = parser.parse_args()
    main(args.save_csv)
