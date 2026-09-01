from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

CHART_CONFIG = {
    "scrollZoom": False,
    "displayModeBar": False,
    "displaylogo": False,
    "responsive": True,
    "doubleClick": False,
    "showTips": False,
}


def full_split_chart(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    split_colors = {
        "train": "#16a34a",
        "validation": "#f59e0b",
        "test": "#dc2626",
        "out_of_scope": "#64748b",
    }

    if "dataset_split" in df.columns:
        for split_name, label in [
            ("train", "Train"),
            ("validation", "Validation"),
            ("test", "Test"),
            ("out_of_scope", "Outside paper split"),
        ]:
            part = df[df["dataset_split"] == split_name]
            if part.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=part["date"],
                    y=part["close"],
                    mode="lines",
                    name=label,
                    line=dict(color=split_colors[split_name], width=1.9),
                    hovertemplate=f"{label}<br>Date=%{{x}}<br>Close=%{{y:,.2f}}<extra></extra>",
                )
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["close"],
                mode="lines",
                name="Close",
                line=dict(color="#2563eb", width=1.9),
                hovertemplate="Date=%{x}<br>Close=%{y:,.2f}<extra></extra>",
            )
        )
    fig.update_layout(
        height=410,
        title=title,
        margin=dict(l=12, r=12, t=56, b=16),
        yaxis_title="Close price",
        xaxis_title="Date",
        dragmode=False,
        hovermode="x unified",
        xaxis=dict(rangeslider=dict(visible=False), fixedrange=True),
        yaxis=dict(fixedrange=True),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def forecast_prediction_chart(
    predictions: pd.DataFrame,
    split: str,
    title: str,
    baseline: pd.DataFrame | None = None,
) -> go.Figure:
    """Parity scatter: predicted vs actual close. Points on the diagonal = accurate.

    Author (official) and our retrained predictions are overlaid; both hugging the
    y=x line shows the retrain reproduced the paper model. If ``baseline`` is given,
    the naive-persistence baseline is added as a third cloud for comparison.
    """
    part = predictions[predictions["split"] == split]
    official = part[part["result_type"] == "official_checkpoint"]
    retrained = part[part["result_type"] == "retrained_checkpoint"]

    naive = pd.DataFrame()
    if baseline is not None and not baseline.empty:
        naive = baseline[
            (baseline["split"] == split) & (baseline["model"] == "naive_persistence")
        ]

    lo = float(part["target_close"].min())
    hi = float(part["target_close"].max())
    pad = (hi - lo) * 0.03

    fig = go.Figure()
    # Perfect-prediction reference line (y = x).
    fig.add_trace(
        go.Scatter(
            x=[lo - pad, hi + pad],
            y=[lo - pad, hi + pad],
            mode="lines",
            name="Perfect prediction (y = x)",
            line=dict(color="#94a3b8", width=1.5, dash="dash"),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=official["target_close"],
            y=official["predicted_close"],
            mode="markers",
            name="Author model (official)",
            marker=dict(color="#2563eb", size=6, opacity=0.45, line=dict(width=0)),
            hovertemplate="Author<br>actual=%{x:,.0f}<br>pred=%{y:,.0f}<extra></extra>",
        )
    )
    if not naive.empty:
        fig.add_trace(
            go.Scatter(
                x=naive["target_close"],
                y=naive["predicted_close"],
                mode="markers",
                name="Naive baseline (yesterday's close)",
                marker=dict(color="#f59e0b", size=5, opacity=0.35, line=dict(width=0)),
                hovertemplate="Naive<br>actual=%{x:,.0f}<br>pred=%{y:,.0f}<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=retrained["target_close"],
            y=retrained["predicted_close"],
            mode="markers",
            name="Our retrained model",
            marker=dict(color="#dc2626", size=6, opacity=0.45, line=dict(width=0)),
            hovertemplate="Retrained<br>actual=%{x:,.0f}<br>pred=%{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=460,
        title=title,
        margin=dict(l=12, r=12, t=56, b=16),
        xaxis_title="Actual BTC close (USD)",
        yaxis_title="Predicted BTC close (USD)",
        dragmode=False,
        xaxis=dict(range=[lo - pad, hi + pad], fixedrange=True, constrain="domain"),
        yaxis=dict(range=[lo - pad, hi + pad], fixedrange=True, scaleanchor="x", scaleratio=1),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def forecast_timeseries_chart(
    predictions: pd.DataFrame,
    result_type: str = "retrained_checkpoint",
    title: str = "Predicted vs actual BTC close — paper split (train / val / test)",
) -> go.Figure:
    """Paper-style time series: actual close (blue) overlaid with the model's predicted
    close colored by split (train / val / test), over the full paper timeline.

    Mirrors the CryptoMamba paper's prediction figure (scripts/evaluation.py pred plot):
    the predicted line hugging the actual line is the visual proof of reproduction.
    """
    part = predictions[predictions["result_type"] == result_type].copy()
    if part.empty:
        return go.Figure()
    part["_d"] = pd.to_datetime(part["prediction_date"], errors="coerce")
    part = part.dropna(subset=["_d"]).sort_values("_d")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=part["_d"], y=part["target_close"], mode="lines", name="Actual close",
            line=dict(color="#2563eb", width=1.9),
            hovertemplate="Actual<br>%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
        )
    )
    for split_name, label, color in (
        ("train", "Predicted · Train", "#dc2626"),
        ("val", "Predicted · Val", "#16a34a"),
        ("test", "Predicted · Test", "#db2777"),
    ):
        seg = part[part["split"] == split_name]
        if seg.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=seg["_d"], y=seg["predicted_close"], mode="lines", name=label,
                line=dict(color=color, width=1.4),
                hovertemplate=f"{label}<br>%{{x|%Y-%m-%d}}<br>$%{{y:,.0f}}<extra></extra>",
            )
        )
    fig.update_layout(
        height=460,
        title=title,
        margin=dict(l=12, r=12, t=56, b=16),
        yaxis_title="BTC close (USD)",
        xaxis_title="Date",
        dragmode=False,
        hovermode="x unified",
        xaxis=dict(rangeslider=dict(visible=False), fixedrange=True),
        yaxis=dict(fixedrange=True),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def forecast_error_chart(
    predictions: pd.DataFrame,
    baseline_predictions: pd.DataFrame | None = None,
    split: str = "test",
    title: str = "Daily forecast error — CryptoMamba-v vs naive (test period)",
) -> go.Figure:
    """Absolute percentage error per day: |predicted - actual| / actual * 100.

    Unlike the price overlay (where everything hugs the actual line at the $0-70K scale),
    this surfaces the real day-to-day miss and lets CryptoMamba-v be compared directly with
    the naive baseline. Spikes = big-miss days. The mean of each line is its MAPE.
    """

    def _ape(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["_d"] = pd.to_datetime(d["prediction_date"], errors="coerce")
        d = d.dropna(subset=["_d"]).sort_values("_d")
        d["_ape"] = (d["predicted_close"] - d["target_close"]).abs() / d["target_close"] * 100.0
        return d

    cm = predictions[
        (predictions["result_type"] == "retrained_checkpoint") & (predictions["split"] == split)
    ]
    if cm.empty:
        return go.Figure()
    cm = _ape(cm)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cm["_d"], y=cm["_ape"], mode="lines", name="CryptoMamba-v",
            line=dict(color="#dc2626", width=1.4),
            hovertemplate="CryptoMamba-v<br>%{x|%Y-%m-%d}<br>error %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_hline(y=float(cm["_ape"].mean()), line=dict(color="#dc2626", width=1, dash="dot"))

    if baseline_predictions is not None and not baseline_predictions.empty:
        nv = baseline_predictions[
            (baseline_predictions["model"] == "naive_persistence")
            & (baseline_predictions["split"] == split)
        ]
        if not nv.empty:
            nv = _ape(nv)
            fig.add_trace(
                go.Scatter(
                    x=nv["_d"], y=nv["_ape"], mode="lines", name="Naive (yesterday's close)",
                    line=dict(color="#f59e0b", width=1.4),
                    hovertemplate="Naive<br>%{x|%Y-%m-%d}<br>error %{y:.2f}%<extra></extra>",
                )
            )
            fig.add_hline(y=float(nv["_ape"].mean()), line=dict(color="#f59e0b", width=1, dash="dot"))

    fig.update_layout(
        height=420,
        title=title,
        margin=dict(l=12, r=12, t=56, b=16),
        yaxis_title="Absolute % error  (|pred − actual| / actual)",
        xaxis_title="Date",
        dragmode=False,
        hovermode="x unified",
        xaxis=dict(rangeslider=dict(visible=False), fixedrange=True),
        yaxis=dict(fixedrange=True, rangemode="tozero"),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def candle_chart(df: pd.DataFrame, title: str, prediction: dict | None = None, height: int = 430) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="BTC candle",
            increasing_line_color="#16a34a",
            decreasing_line_color="#dc2626",
        )
    )
    if prediction:
        fig.add_trace(
            go.Scatter(
                x=[df["date"].iloc[-1], prediction["prediction_date"]],
                y=[float(prediction["last_close"]), float(prediction["predicted_close"])],
                mode="lines+markers",
                name="Prediction",
                line=dict(color="#f97316", width=3, dash="dash"),
                marker=dict(size=9),
            )
        )
    fig.update_layout(
        height=height,
        title=title,
        margin=dict(l=12, r=12, t=48, b=12),
        dragmode=False,
        hovermode="x unified",
        xaxis=dict(rangeslider=dict(visible=False), fixedrange=True),
        yaxis=dict(fixedrange=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def roi_chart(sim_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    colors = ["#2563eb", "#0f766e"]
    fig.add_trace(
        go.Bar(
            x=sim_df["strategy"],
            y=sim_df["roi_pct"],
            marker_color=colors,
            text=sim_df["roi_pct"].map(lambda x: f"{x:+.2f}%"),
            textposition="outside",
        )
    )
    fig.update_layout(height=320, margin=dict(l=12, r=12, t=28, b=12), yaxis_title="ROI (%)", showlegend=False)
    return fig


# --- Model-comparison charts (demo-friendly benchmark views) ---

_HIGHLIGHT_COLOR = "#1e3a8a"   # CryptoMamba-v (the model under test)
_NEUTRAL_COLOR = "#cbd5e1"     # reference baselines


def leaderboard_bar_chart(
    labels: list[str],
    values: list[float],
    *,
    value_label: str,
    lower_is_better: bool,
    highlight_substr: str = "CryptoMamba",
    value_fmt: str = "{:,.1f}",
    height: int = 360,
) -> go.Figure:
    """Ranked bar chart (leaderboard). Best bar first; CryptoMamba-v highlighted."""
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=not lower_is_better)
    sorted_labels = [labels[i] for i in order]
    sorted_values = [values[i] for i in order]
    colors = [
        _HIGHLIGHT_COLOR if highlight_substr.lower() in label.lower() else _NEUTRAL_COLOR
        for label in sorted_labels
    ]
    fig = go.Figure(
        go.Bar(
            x=sorted_labels,
            y=sorted_values,
            marker_color=colors,
            text=[value_fmt.format(v) for v in sorted_values],
            textposition="outside",
            hovertemplate="%{x}<br>" + value_label + "=%{y}<extra></extra>",
        )
    )
    direction = "lower is better" if lower_is_better else "higher is better"
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=44, b=86),
        title=f"{value_label} — ranked ({direction})",
        yaxis_title=value_label,
        showlegend=False,
        xaxis=dict(fixedrange=True, tickangle=-22),
        yaxis=dict(fixedrange=True),
    )
    return fig


def tradeoff_scatter_chart(
    models: list[str],
    rmse: list[float],
    dir_acc: list[float],
    *,
    highlight_substr: str = "CryptoMamba",
    height: int = 430,
) -> go.Figure:
    """RMSE (y, lower better) vs directional accuracy (x, higher better). Best = bottom-right."""
    fig = go.Figure()
    for model, x, y in zip(models, dir_acc, rmse):
        highlighted = highlight_substr.lower() in model.lower()
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                text=[model],
                textposition="top center",
                marker=dict(
                    size=18 if highlighted else 12,
                    color=_HIGHLIGHT_COLOR if highlighted else "#94a3b8",
                    line=dict(width=1, color="#1e293b"),
                ),
                showlegend=False,
                hovertemplate=f"{model}<br>Directional acc=%{{x:.1f}}%<br>RMSE=%{{y:,.1f}}<extra></extra>",
            )
        )
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=48, b=48),
        title="Error vs direction — best models sit bottom-right",
        xaxis_title="Directional accuracy (%) — higher is better →",
        yaxis_title="RMSE — lower is better ↓",
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True),
    )
    return fig


def grouped_bar_chart(
    categories: list[str],
    series: dict[str, list[float]],
    *,
    title: str,
    y_title: str,
    value_fmt: str = "{:,.2f}",
    colors: list[str] | None = None,
    hline: float | None = None,
    hline_label: str = "",
    height: int = 360,
) -> go.Figure:
    """Generic grouped bar chart (categories x named series). Optional reference hline."""
    palette = colors or ["#94a3b8", "#2563eb", _HIGHLIGHT_COLOR, "#0f766e"]
    fig = go.Figure()
    for index, (name, vals) in enumerate(series.items()):
        fig.add_trace(
            go.Bar(
                name=name,
                x=categories,
                y=vals,
                marker_color=palette[index % len(palette)],
                text=[value_fmt.format(v) for v in vals],
                textposition="outside",
                hovertemplate=f"{name}<br>%{{x}}=%{{y}}<extra></extra>",
            )
        )
    if hline is not None:
        fig.add_hline(
            y=hline,
            line_dash="dot",
            line_color="#b45309",
            annotation_text=hline_label,
            annotation_position="top right",
        )
    fig.update_layout(
        barmode="group",
        height=height,
        margin=dict(l=12, r=12, t=46, b=40),
        title=title,
        yaxis_title=y_title,
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
