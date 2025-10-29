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

Each run stores its artefact independently: `results/baseline_model.joblib`,
`results/random_forest_model.joblib`, and `results/bilstm_model.h5`, alongside
`results/training_metrics.json`, updated classification reports, and, for the
Bi-LSTM, refreshed training curves.

### Predict on a Session (Simulated or CSV)

The prediction entry-point defaults to the Bi-LSTM and exposes knobs through
Make variables:

```bash
make predict                               # Bi-LSTM on a simulated session
MODEL=random_forest make predict           # switch to the Random Forest
MODEL=baseline make predict SIMULATE=0 CSV=data/session.csv
MODEL=bilstm make predict MODEL_PATH=/tmp/bilstm.h5  # custom artefact path
```

The console output (also saved to `results/latest_session.json`) now includes a
class distribution summary, the artefact path used, and explicit window
settings. When running the Bi-LSTM you should see the fatigue distribution shift
toward classes `1` and `2` in later reps, confirming the temporal model is in
play.

### Launch Dashboard

```bash
make dash
```

When the dashboard starts successfully you should see Dash announce that it is
serving on `http://0.0.0.0:8050/` along with the usual Flask development server
banner. Subsequent `GET` log lines for `/_dash-...` assets and the biometrics
plot indicate the frontend is loading data from the generated artefacts.

### Real-time Sliding Window Demo

```bash
make realtime
MODEL=baseline make realtime
MODEL=bilstm MODEL_PATH=/tmp/bilstm.h5 make realtime WINDOW_SIZE=30 STRIDE=10
```

The real-time preview reports which artefact was loaded, the window settings,
how many predictions were emitted, and shows the first few probability vectors.
Early reps should lean toward low-fatigue predictions before climbing as the
session progresses.

### Clean Generated Artefacts

```bash
make clean
```

## Tests

Run automated tests with:

```bash
pytest
```
