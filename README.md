# LiftSense

LiftSense is a reproducible research environment for simulating resistance-training sessions, training fatigue classifiers, and visualising results via a Dash dashboard. The system runs locally without external services and supports classical ML as well as deep-learning models.

## Features

- Vectorised simulation of rep-level biometrics (velocity, force, grip, heart rate) with fatigue drift, AR(1) noise, and derived fatigue labels.
- Feature extraction utilities and windowed dataset generation.
- Baseline logistic regression, Random Forest with feature importance, and Bi-LSTM sliding-window models.
- Training CLI that produces metrics, plots, and persisted artefacts under `results/`.
- Prediction CLI capable of simulating or loading CSV sessions.
- Real-time sliding-window predictor for streaming scenarios.
- Dash dashboard to inspect latest results and feature importance.

## Getting Started

```bash
make setup
```

### Train Models

```bash
make baseline  # logistic regression
make rf        # random forest
make bilstm    # bi-directional LSTM (installs ML extras)
```

### Predict on a Simulated Session

```bash
make predict
```

### Launch Dashboard

```bash
make dash
```

### Real-time Sliding Window Demo

```bash
make realtime
```

### Clean Generated Artefacts

```bash
make clean
```

## Tests

Run automated tests with:

```bash
pytest
```
