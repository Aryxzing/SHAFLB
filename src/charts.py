"""Plotly figures for the dashboard. Colours are fixed per entity so a model keeps its colour everywhere."""

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve, roc_curve

from src.models import METRIC_LABELS

# Validated categorical palette, assigned in slot order and bound to entities.
MODEL_COLORS = {
    "centralized_lr": "#2a78d6",
    "fedavg": "#eb6834",
    "fedprox": "#1baf7a",
    "centralized_rf": "#eda100",
    "dp_fedavg": "#e87ba4",
}
MODEL_NAMES = {
    "centralized_lr": "Centralised LR",
    "centralized_rf": "Centralised RF",
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "dp_fedavg": "DP-FedAvg",
}
CLASS_COLORS = {"negative": "#86b6ef", "positive": "#e34948"}
LOCAL_COLOR = "#a8a79f"
BLUE_RAMP = [[0, "#cde2fb"], [0.5, "#5598e7"], [1, "#104281"]]
GRID = "rgba(128,128,128,0.18)"


def _layout(fig: go.Figure, height: int = 360, **kwargs) -> go.Figure:
    settings = dict(
        height=height,
        margin=dict(l=8, r=8, t=36, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title=None),
        hoverlabel=dict(font_size=12),
    )
    fig.update_layout(**{**settings, **kwargs})
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def hospital_volumes(hospitals: list[dict]) -> go.Figure:
    names = [h["hospital_id"].replace("_", " ") for h in hospitals]
    fig = go.Figure()
    for key, label, cls in (("negative_samples", "Not readmitted", "negative"),
                            ("positive_samples", "Readmitted <30d", "positive")):
        fig.add_bar(
            x=names, y=[h[key] for h in hospitals], name=label, marker_color=CLASS_COLORS[cls],
            marker_line=dict(width=2, color="rgba(0,0,0,0)"),
            hovertemplate="%{x}<br>" + label + ": %{y:,}<extra></extra>",
        )
    fig.update_layout(barmode="stack", bargap=0.45)
    fig.update_yaxes(title="Patients")
    return _layout(fig)


def readmission_rates(hospitals: list[dict], overall: float) -> go.Figure:
    names = [h["hospital_id"].replace("_", " ") for h in hospitals]
    rates = [h["readmission_rate_pct"] for h in hospitals]
    fig = go.Figure(go.Bar(
        x=names, y=rates, marker_color=CLASS_COLORS["positive"], text=[f"{r:.1f}%" for r in rates],
        textposition="outside", cliponaxis=False, hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
        showlegend=False,
    ))
    fig.add_hline(y=overall, line_dash="dot", line_color="#8a8a85",
                  annotation_text=f"Overall {overall:.1f}%", annotation_position="top left")
    fig.update_layout(bargap=0.45)
    fig.update_yaxes(title="Readmitted within 30 days (%)", range=[0, max(rates) * 1.3])
    return _layout(fig)


def convergence(runs: dict, metric: str = "roc_auc", reference: dict | None = None) -> go.Figure:
    """One line per federated run; optional dashed horizontal line for the centralised reference."""
    fig = go.Figure()
    for name, run in runs.items():
        hist = run["history"]
        fig.add_scatter(
            x=[h["round"] for h in hist], y=[h["metrics"][metric] for h in hist],
            name=run.get("label", MODEL_NAMES.get(name, name)), mode="lines+markers",
            line=dict(width=2, color=MODEL_COLORS.get(name, "#2a78d6")), marker=dict(size=8),
            hovertemplate="Round %{x}: %{y:.4f}<extra>%{fullData.name}</extra>",
        )
    if reference:
        fig.add_hline(y=reference[metric], line_dash="dash", line_color=MODEL_COLORS["centralized_lr"],
                      annotation_text="Centralised LR", annotation_position="bottom right")
    fig.update_xaxes(title="Communication round", dtick=1)
    fig.update_yaxes(title=METRIC_LABELS.get(metric, metric))
    return _layout(fig, hovermode="x unified")


def metric_bars(models: dict, keys: list[str], metrics: list[str]) -> go.Figure:
    fig = go.Figure()
    for key in keys:
        fig.add_bar(
            x=[METRIC_LABELS[m] for m in metrics], y=[models[key][m] for m in metrics],
            name=MODEL_NAMES[key], marker_color=MODEL_COLORS[key],
            hovertemplate="%{x}: %{y:.4f}<extra>" + MODEL_NAMES[key] + "</extra>",
        )
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    fig.update_yaxes(range=[0, 1], title="Score")
    return _layout(fig)


def roc_pr_curves(y_true, probas: dict, kind: str = "roc") -> go.Figure:
    fig = go.Figure()
    for key, p in probas.items():
        if kind == "roc":
            x, y, _ = roc_curve(y_true, p)
        else:
            y, x, _ = precision_recall_curve(y_true, p)
        step = max(1, len(x) // 400)  # thin the curve for a lighter payload
        fig.add_scatter(x=x[::step], y=y[::step], name=MODEL_NAMES[key], mode="lines",
                        line=dict(width=2, color=MODEL_COLORS[key]),
                        hovertemplate="(%{x:.3f}, %{y:.3f})<extra>" + MODEL_NAMES[key] + "</extra>")
    if kind == "roc":
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                        line=dict(dash="dot", width=1, color="#8a8a85"), hoverinfo="skip")
        fig.update_xaxes(title="False positive rate", range=[0, 1])
        fig.update_yaxes(title="True positive rate", range=[0, 1.02])
    else:
        base = float(np.mean(y_true))
        fig.add_hline(y=base, line_dash="dot", line_color="#8a8a85",
                      annotation_text=f"Prevalence {base:.1%}", annotation_position="top right")
        fig.update_xaxes(title="Recall", range=[0, 1])
        fig.update_yaxes(title="Precision", range=[0, 1.02])
    return _layout(fig, height=380)


def confusion(cm: dict, title: str) -> go.Figure:
    z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    total = sum(cm.values())
    labels = ["Not readmitted", "Readmitted"]
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels, colorscale=BLUE_RAMP, showscale=False, xgap=2, ygap=2,
        text=[[f"{v:,}<br>{v / total:.1%}" for v in row] for row in z], texttemplate="%{text}",
        hovertemplate="Actual %{y}<br>Predicted %{x}<br>%{z:,}<extra></extra>",
    ))
    fig.update_xaxes(title="Predicted", side="bottom", showgrid=False)
    fig.update_yaxes(title="Actual", autorange="reversed", showgrid=False)
    return _layout(fig, height=300, title=dict(text=title, font=dict(size=14)))


def local_vs_federated(local_only: list[dict], federated: dict, metric: str = "roc_auc") -> go.Figure:
    names = [r["hospital_id"].replace("_", " ") for r in local_only]
    vals = [r["metrics"][metric] for r in local_only]
    fig = go.Figure(go.Bar(
        x=names, y=vals, name="Trained on own data only", marker_color=LOCAL_COLOR,
        text=[f"{v:.3f}" for v in vals], textposition="outside", cliponaxis=False,
        hovertemplate="%{x} local-only: %{y:.4f}<extra></extra>",
    ))
    fig.add_hline(y=federated[metric], line_color=MODEL_COLORS["fedavg"], line_width=2,
                  annotation_text=f"Federated model {federated[metric]:.3f}", annotation_position="top left")
    lo = min(vals + [federated[metric]])
    fig.update_yaxes(title=METRIC_LABELS[metric], range=[max(0, lo - 0.05), max(vals + [federated[metric]]) + 0.02])
    fig.update_layout(bargap=0.45, showlegend=False)
    return _layout(fig)


def coefficients(names: list[str], weights: np.ndarray, top: int = 12) -> go.Figure:
    order = np.argsort(np.abs(weights))[::-1][:top][::-1]
    w = weights[order]
    fig = go.Figure(go.Bar(
        x=w, y=[names[i] for i in order], orientation="h",
        marker_color=[CLASS_COLORS["positive"] if v > 0 else CLASS_COLORS["negative"] for v in w],
        hovertemplate="%{y}: %{x:+.3f}<extra></extra>", showlegend=False,
    ))
    fig.update_xaxes(title="Log-odds weight (red raises risk, blue lowers it)")
    left = 12 + 7 * max(len(names[i]) for i in order)  # room for the longest label
    return _layout(fig, height=max(320, 26 * top), margin=dict(l=left, r=8, t=12, b=8))


def update_norms(history: list[dict]) -> go.Figure:
    hospitals = list(history[0]["update_norms"])
    z = [[h["update_norms"][hid] for h in history] for hid in hospitals]
    fig = go.Figure(go.Heatmap(
        z=z, x=[h["round"] for h in history], y=[h.replace("_", " ") for h in hospitals],
        colorscale=BLUE_RAMP, xgap=2, ygap=2, colorbar=dict(title="L2 norm", thickness=10),
        hovertemplate="%{y}, round %{x}<br>Update norm %{z:.3f}<extra></extra>",
    ))
    fig.update_xaxes(title="Round", dtick=1, showgrid=False)
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return _layout(fig, height=280)


def risk_gauge(value: float, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value * 100, number=dict(suffix="%", valueformat=".1f"),
        gauge=dict(
            axis=dict(range=[0, 100], ticksuffix="%", tickvals=[0, 25, 50, 75, 100]),
            bar=dict(color=color, thickness=0.35),
            threshold=dict(line=dict(color="#8a8a85", width=2), thickness=0.8, value=50),
        ),
    ))
    return _layout(fig, height=210, margin=dict(l=44, r=44, t=24, b=0))
