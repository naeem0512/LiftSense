from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import dash
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

RESULTS_DIR = Path("results")


def load_json(path: Path) -> Dict:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def layout_dashboard(app: Dash) -> html.Div:
    metrics = load_json(RESULTS_DIR / "training_metrics.json")
    latest = load_json(RESULTS_DIR / "latest_session.json")
    feature_importance = load_json(RESULTS_DIR / "feature_importance.json")

    kpi_cards = []
    for key in ["accuracy", "macro_f1", "micro_f1"]:
        value = metrics.get("metrics", {}).get(key, 0)
        kpi_cards.append(
            dbc.Card(
                [
                    dbc.CardHeader(key.replace("_", " ").title()),
                    dbc.CardBody(html.H3(f"{value:.3f}" if value else "N/A")),
                ],
                className="m-2",
            )
        )

    predictions = latest.get("predictions", [])
    if predictions:
        prediction_counts = pd.Series(predictions).value_counts().sort_index()
        class_dist = px.bar(prediction_counts, labels={"index": "Fatigue Class", "value": "Count"})
    else:
        class_dist = go.Figure()

    biometrics_path = RESULTS_DIR / "plots" / "simulated_biometrics.png"
    biometrics_img = html.Img(src=str(biometrics_path), style={"width": "100%", "maxWidth": "600px"})

    feature_fig = go.Figure()
    if feature_importance:
        names = feature_importance.get("feature_names", [])
        values = feature_importance.get("impurity_importance", [])
        feature_fig = px.bar(x=names, y=values, labels={"x": "Feature", "y": "Importance"})

    layout = dbc.Container(
        [
            html.H1("LiftSense Dashboard"),
            dbc.Row(kpi_cards),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(figure=class_dist), md=6),
                    dbc.Col(biometrics_img, md=6),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(figure=feature_fig), md=12),
                ]
            ),
        ],
        fluid=True,
    )

    return layout


def main() -> None:
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = layout_dashboard(app)
    app.run_server(debug=False)


if __name__ == "__main__":
    main()
