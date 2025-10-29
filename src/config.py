from pathlib import Path
from typing import Tuple

from pydantic import Field
from pydantic_settings import BaseSettings


class SimulationConfig(BaseSettings):
    reps: int = Field(200, description="Number of repetitions in a simulated session")
    hr_rest: Tuple[int, int] = Field((60, 72), description="Range for resting heart rate")
    hr_work: Tuple[int, int] = Field((100, 150), description="Range for working heart rate")
    vel_mu_sigma: Tuple[float, float] = Field((0.6, 0.15), description="Velocity mean/sigma")
    force_mu_sigma: Tuple[float, float] = Field((250.0, 60.0), description="Force mean/sigma")
    grip_mu_sigma: Tuple[float, float] = Field((25.0, 6.0), description="Grip mean/sigma")
    noise: float = Field(0.08, description="Noise magnitude scale")
    ar_hr: float = Field(0.9, description="Autoregressive heart rate coefficient")


class WindowConfig(BaseSettings):
    window_size: int = Field(20, description="Number of reps per window")
    stride: int = Field(5, description="Stride for sliding window")


class TrainingConfig(BaseSettings):
    model_type: str = Field("baseline", description="Model type identifier")
    test_size: float = Field(0.2, description="Test split ratio")
    max_epochs: int = Field(20, description="Maximum epochs for neural models")
    batch_size: int = Field(64, description="Batch size for neural models")
    lr: float = Field(1e-3, description="Learning rate for neural models")
    patience: int = Field(4, description="Early stopping patience")


class ArtifactConfig(BaseSettings):
    baseline: Path = Field(Path("results/baseline_model.joblib"), description="Baseline model artifact")
    random_forest: Path = Field(Path("results/random_forest_model.joblib"), description="Random Forest model artifact")
    bilstm: Path = Field(Path("results/bilstm_model.h5"), description="Bi-LSTM model artifact")


class AppConfig(BaseSettings):
    random_seed: int = Field(42, description="Global random seed")
    output_dir: Path = Field(Path("results"), description="Output directory")
    simulation: SimulationConfig = SimulationConfig()
    windowing: WindowConfig = WindowConfig()
    training: TrainingConfig = TrainingConfig()
    artifacts: ArtifactConfig = ArtifactConfig()


config = AppConfig()
