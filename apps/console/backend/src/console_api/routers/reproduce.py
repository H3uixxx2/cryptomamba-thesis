"""Reproduce screen API — reuses load_reproduce_artifacts + charts.py.

The artifact loader (reproduce_artifacts.py) does the real validation/consistency
work and is reused unchanged. The presentation-table formatters in the Streamlit
``reproduce_page.py`` import streamlit, so we do NOT import them here; instead we
assemble display tables directly from the loaded artifact frames whose gap/verdict
columns are already pre-computed at artifact-generation time. No model/eval logic
is reimplemented. Missing/invalid artifacts yield status=NOT_READY (never a 500).
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from fastapi import APIRouter

from ..core import config
from .. import logic
from ..loaders.final_evidence import (
    EvidenceIntegrityError,
    FinalEvidence,
    records as final_records,
)
from ..loaders.forecast_robustness import get_forecast_robustness, not_ready_robustness
from ..loaders.reproduction_evidence import build_reproduction_350d, not_ready_reproduction
from .data import fig_json

router = APIRouter(prefix="/api/reproduce", tags=["reproduce"])

# Mandatory baseline set for the FULL thesis gate (mirrors reproduce_page).
MANDATORY_BASELINES = {"naive_persistence", "lstm", "gru", "itransformer"}

# One stable colour per model so a model is recognisable across every chart.
MODEL_COLORS = {
    "CryptoMamba-v": "#3E8EF2",   # our model — signal blue
    "naive_persistence": "#9CA3AF",
    "lstm": "#F2B01E",
    "gru": "#EC4899",
    "itransformer": "#14B8A6",
}
_FALLBACK_COLOR = "#9CA3AF"


def _color_for(model: str) -> str:
    return MODEL_COLORS.get(model, _FALLBACK_COLOR)


def _bar_by_model(
    models: list[str],
    values: list[float],
    *,
    y_title: str,
    lower_is_better: bool,
    value_fmt: str = "{:.2f}",
) -> go.Figure:
    """Ranked, per-model coloured bar chart. CM-v gets a bright outline so it
    stands out; bars are sorted best→worst so the ranking reads left to right."""
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=not lower_is_better)
    m = [models[i] for i in order]
    v = [float(values[i]) for i in order]
    fig = go.Figure(
        go.Bar(
            x=m,
            y=v,
            marker_color=[_color_for(x) for x in m],
            marker_line_color=["#E5E7EB" if _is_cm(x) else "rgba(0,0,0,0)" for x in m],
            marker_line_width=[2 if _is_cm(x) else 0 for x in m],
            text=[value_fmt.format(val) for val in v],
            textposition="outside",
            textfont=dict(size=11),
            cliponaxis=False,
            hovertemplate="%{x}: %{y:.2f}<extra></extra>",
        )
    )
    span = (max(v) - min(v)) or 1.0
    top = max(v) + span * 0.18
    bottom = min(0.0, min(v))
    fig.update_layout(height=330, yaxis=dict(title=y_title, range=[bottom, top]))
    return fig


def _tradeoff_chart(models: list[str], rmse: list[float], diracc: list[float]) -> go.Figure:
    """RMSE vs directional-accuracy scatter with hand-placed, non-overlapping labels
    and padded ranges so no label is clipped at the frame edge."""
    label_pos = {
        "naive_persistence": "middle right",
        "CryptoMamba-v": "middle left",
        "itransformer": "middle right",
        "lstm": "middle left",
        "gru": "middle right",
    }
    fig = go.Figure(
        go.Scatter(
            x=diracc,
            y=rmse,
            mode="markers+text",
            text=models,
            textposition=[label_pos.get(m, "top center") for m in models],
            textfont=dict(size=11, color="#D4D4D8"),
            marker=dict(
                size=[20 if _is_cm(m) else 12 for m in models],
                color=[_color_for(m) for m in models],
                line=dict(
                    width=[2 if _is_cm(m) else 0 for m in models],
                    color="#E5E7EB",
                ),
            ),
            cliponaxis=False,
            hovertemplate="%{text}<br>dir-acc %{x:.2f}%<br>RMSE %{y:.1f}<extra></extra>",
        )
    )
    dmin, dmax = min(diracc), max(diracc)
    rmin, rmax = min(rmse), max(rmse)
    dpad = (dmax - dmin) * 0.20 or 6
    rpad = (rmax - rmin) * 0.16 or 80
    fig.update_layout(
        height=380,
        xaxis=dict(title="Directional accuracy (%) — cao hơn tốt hơn →", range=[dmin - dpad, dmax + dpad]),
        yaxis=dict(title="RMSE — thấp hơn tốt hơn ↓", range=[rmin - rpad, rmax + rpad]),
    )
    return fig


def _records(df: pd.DataFrame | None) -> list[dict]:
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient="records"))


def _retrained_row(forecast: pd.DataFrame) -> dict | None:
    if forecast is None or forecast.empty:
        return None
    mask = forecast["result_type"].astype(str).str.contains("retrain", case=False, na=False)
    rows = forecast[mask]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def _is_cm(name: str) -> bool:
    n = name.lower()
    return "cryptomamba" in n or "cmamba" in n


def _cm_dir_acc(a: "logic.ReproduceArtifacts") -> float | None:
    sig = a.significance
    if sig is not None and not sig.empty and "cm_directional_acc_pct" in sig.columns:
        vals = sig["cm_directional_acc_pct"].dropna()
        if not vals.empty:
            return float(vals.iloc[0])
    return None


def _cm_dir_coverage(a: "logic.ReproduceArtifacts") -> float | None:
    sig = a.significance
    if sig is not None and not sig.empty and "cm_directional_coverage_pct" in sig.columns:
        vals = sig["cm_directional_coverage_pct"].dropna()
        if not vals.empty:
            return float(vals.iloc[0])
    return None


def _model_metrics_frame(a: "logic.ReproduceArtifacts") -> pd.DataFrame:
    """Tidy evaluation metrics across baselines plus one CryptoMamba-v row.

    Presentation assembly only — values come straight from pre-computed artifact
    frames. We collapse the official/retrained CM-v rows into a single retrained
    CryptoMamba-v entry (official ≈ paper, shown in the forecast table) and attach
    its directional accuracy from the significance file.
    """
    rows: list[dict] = []
    comp = a.baseline_comparison
    if comp is not None and not comp.empty:
        for _, r in comp.iterrows():
            name = str(r.get("model", ""))
            if _is_cm(name):
                continue  # CM-v handled once below from the retrained forecast row
            rows.append(
                {
                    "model": name,
                    "RMSE": r.get("RMSE"),
                    "MAPE_pct": r.get("MAPE_pct"),
                    "dir_acc_pct": r.get("directional_accuracy_strict_pct"),
                    "dir_coverage_pct": r.get("directional_coverage_pct"),
                }
            )
    retr = _retrained_row(a.forecast_metrics)
    if retr is not None:
        rows.append(
            {
                "model": "CryptoMamba-v",
                "RMSE": retr.get("RMSE"),
                "MAPE_pct": retr.get("MAPE_pct"),
                "dir_acc_pct": _cm_dir_acc(a),
                "dir_coverage_pct": _cm_dir_coverage(a),
            }
        )
    return pd.DataFrame(rows)


def _retrained_test_predictions(a: "logic.ReproduceArtifacts") -> pd.DataFrame | None:
    """Per-day retrained CryptoMamba-v predictions on the paper test split."""
    fp = a.forecast_predictions
    if fp is None or fp.empty:
        return None
    needed = {"current_close", "target_close", "predicted_close"}
    if not needed.issubset(fp.columns):
        return None
    m = fp[
        fp["result_type"].astype(str).str.contains("retrain", case=False, na=False)
        & (fp["split"].astype(str) == "test")
    ]
    return m.sort_values("prediction_date") if not m.empty else None


def _forecast_timeseries_chart(d: pd.DataFrame) -> go.Figure:
    """Canonical forecasting figure: actual vs predicted close over the test period."""
    x = d["prediction_date"].astype(str).tolist()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=[float(v) for v in d["target_close"]], mode="lines",
            name="Actual close", line=dict(color="#D4D4D8", width=1.6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=[float(v) for v in d["predicted_close"]], mode="lines",
            name="CryptoMamba-v dự đoán", line=dict(color="#3E8EF2", width=1.6),
        )
    )
    fig.update_layout(
        height=360,
        yaxis=dict(title="BTC close ($)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _parity_chart(d: pd.DataFrame) -> go.Figure:
    """Predicted vs actual scatter with the y=x reference line (regression fit diagnostic)."""
    actual = [float(v) for v in d["target_close"]]
    pred = [float(v) for v in d["predicted_close"]]
    lo, hi = min(min(actual), min(pred)), max(max(actual), max(pred))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y = x (dự đoán hoàn hảo)",
                   line=dict(color="#71717A", width=1, dash="dash"))
    )
    fig.add_trace(
        go.Scatter(x=actual, y=pred, mode="markers", name="Ngày test",
                   marker=dict(size=5, color="#3E8EF2", opacity=0.5),
                   hovertemplate="actual $%{x:.0f}<br>pred $%{y:.0f}<extra></extra>")
    )
    fig.update_layout(
        height=360,
        xaxis=dict(title="Actual close ($)"),
        yaxis=dict(title="Predicted close ($)", scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _dir_confusion_chart(d: pd.DataFrame) -> go.Figure:
    """2x2 directional confusion matrix (predicted vs actual up/down) — visualises the
    directional-accuracy headline; the diagonal share equals CM-v's dir-acc."""
    cur = d["current_close"].to_numpy()
    tgt = d["target_close"].to_numpy()
    prd = d["predicted_close"].to_numpy()
    pu, au = prd > cur, tgt > cur
    A = int((au & pu).sum())    # actual up, predicted up   (correct)
    B = int((au & ~pu).sum())   # actual up, predicted down (miss)
    C = int((~au & pu).sum())   # actual down, predicted up (false alarm)
    D = int((~au & ~pu).sum())  # actual down, predicted down (correct)
    total = A + B + C + D or 1
    counts = [[A, B], [C, D]]
    correct = [[1, 0], [0, 1]]  # drive colour by correctness, not raw count
    text = [
        [f"{A}<br>{A / total * 100:.1f}%", f"{B}<br>{B / total * 100:.1f}%"],
        [f"{C}<br>{C / total * 100:.1f}%", f"{D}<br>{D / total * 100:.1f}%"],
    ]
    fig = go.Figure(
        go.Heatmap(
            z=correct,
            x=["Dự đoán: Lên", "Dự đoán: Xuống"],
            y=["Thực tế: Lên", "Thực tế: Xuống"],
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=16, color="#FFFFFF"),
            colorscale=[[0, "rgba(242,176,30,0.22)"], [1, "rgba(47,191,113,0.55)"]],
            showscale=False,
            xgap=5,
            ygap=5,
            customdata=counts,
            hovertemplate="%{y} / %{x}: %{customdata} ngày<extra></extra>",
        )
    )
    fig.update_layout(height=340, yaxis=dict(autorange="reversed"))
    return fig


def _trading_replay_chart(df: pd.DataFrame) -> go.Figure:
    """Paper vs Official vs Retrained final balance per strategy. Official ≈ Paper
    (pipeline verified, green); the retrained bar's shortfall is the disclosed gap."""
    order = ["vanilla", "smart", "smart_w_short"]
    label = {"vanilla": "Vanilla", "smart": "Smart", "smart_w_short": "Smart+Short"}
    modes = [m for m in order if m in set(df["trade_mode"].astype(str))]

    def val(result_type: str, mode: str, col: str) -> float | None:
        sub = df[(df["result_type"].astype(str) == result_type) & (df["trade_mode"].astype(str) == mode)]
        return float(sub[col].iloc[0]) if not sub.empty else None

    paper = [val("official_checkpoint", m, "paper_final_balance") or val("retrained_checkpoint", m, "paper_final_balance") for m in modes]
    official = [val("official_checkpoint", m, "final_balance") for m in modes]
    retrained = [val("retrained_checkpoint", m, "final_balance") for m in modes]
    x = [label.get(m, m) for m in modes]

    def _txt(vals):
        return [f"{v:.0f}" if v is not None else "" for v in vals]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Paper (mục tiêu)", x=x, y=paper, marker_color="#9CA3AF",
                         text=_txt(paper), textposition="outside", textfont=dict(size=10)))
    fig.add_trace(go.Bar(name="Official — xác minh pipeline", x=x, y=official, marker_color="#2FBF71",
                         text=_txt(official), textposition="outside", textfont=dict(size=10)))
    fig.add_trace(go.Bar(name="Retrained — model của tôi", x=x, y=retrained, marker_color="#F2B01E",
                         text=_txt(retrained), textposition="outside", textfont=dict(size=10)))
    vals = [v for v in (paper + official + retrained) if v is not None]
    top = (max(vals) * 1.18) if vals else 1
    fig.update_layout(
        height=360,
        barmode="group",
        yaxis=dict(title="Balance cuối", range=[0, top]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _charts(a: "logic.ReproduceArtifacts") -> dict:
    """Best-effort charts; any chart whose inputs are incomplete is omitted, never 500."""
    out: dict[str, Any] = {}

    # Trading replay: paper vs official vs retrained final balance per strategy.
    try:
        rp = a.trading_replay_metrics
        if rp is not None and not rp.empty:
            rp_test = (
                rp[rp["split"].astype(str).str.lower() == "test"]
                if "split" in rp.columns
                else rp
            )
            if rp_test is not None and not rp_test.empty:
                out["trading_replay"] = fig_json(_trading_replay_chart(rp_test))
    except Exception:  # noqa: BLE001
        pass

    # ML-standard forecast diagnostics on the retrained test split.
    rt = _retrained_test_predictions(a)
    if rt is not None:
        try:
            out["forecast_ts"] = fig_json(_forecast_timeseries_chart(rt))
        except Exception:  # noqa: BLE001
            pass
        try:
            out["parity"] = fig_json(_parity_chart(rt))
        except Exception:  # noqa: BLE001
            pass
        try:
            out["dir_confusion"] = fig_json(_dir_confusion_chart(rt))
        except Exception:  # noqa: BLE001
            pass

    # Forecast daily %-error (primary discriminator).
    try:
        if a.forecast_predictions is not None and not a.forecast_predictions.empty:
            fig = logic.charts.forecast_error_chart(
                a.forecast_predictions,
                baseline_predictions=a.baseline_predictions
                if (a.baseline_predictions is not None and not a.baseline_predictions.empty)
                else None,
                split="test",
                title="Sai số % theo ngày · test",
            )
            out["forecast_error"] = fig_json(fig)
    except Exception:  # noqa: BLE001 - chart is optional, tables carry the evidence
        pass

    # Gap-vs-paper bars with 5% tolerance line.
    try:
        retr = _retrained_row(a.forecast_metrics)
        if retr is not None:
            gaps = [retr.get("RMSE_gap_pct"), retr.get("MAE_gap_pct"), retr.get("MAPE_gap_pct")]
            if all(g is not None for g in gaps):
                fig = logic.charts.grouped_bar_chart(
                    ["RMSE", "MAE", "MAPE"],
                    {"Our retrain": [float(g) for g in gaps]},
                    title="Sai số model retrain so với paper",
                    y_title="Gap (%)",
                    hline=float(retr.get("tolerance_pct", 5.0)),
                    hline_label="ngưỡng 5%",
                )
                out["gap"] = fig_json(fig)
    except Exception:  # noqa: BLE001
        pass

    # Per-model comparison charts (RMSE, MAPE, directional accuracy, trade-off).
    try:
        mm = _model_metrics_frame(a)

        rmse_rows = mm.dropna(subset=["RMSE"])
        if not rmse_rows.empty:
            out["leaderboard_rmse"] = fig_json(
                _bar_by_model(
                    rmse_rows["model"].tolist(),
                    [float(v) for v in rmse_rows["RMSE"].tolist()],
                    y_title="RMSE (thấp hơn tốt hơn)",
                    lower_is_better=True,
                    value_fmt="{:.0f}",
                )
            )

        mape_rows = mm.dropna(subset=["MAPE_pct"])
        if not mape_rows.empty:
            out["mape_bar"] = fig_json(
                _bar_by_model(
                    mape_rows["model"].tolist(),
                    [float(v) for v in mape_rows["MAPE_pct"].tolist()],
                    y_title="MAPE % (thấp hơn tốt hơn)",
                    lower_is_better=True,
                    value_fmt="{:.2f}",
                )
            )

        dir_only = mm.dropna(subset=["dir_acc_pct"])
        if not dir_only.empty:
            out["diracc_bar"] = fig_json(
                _bar_by_model(
                    dir_only["model"].tolist(),
                    [float(v) for v in dir_only["dir_acc_pct"].tolist()],
                    y_title="Directional accuracy % (cao hơn tốt hơn)",
                    lower_is_better=False,
                    value_fmt="{:.1f}",
                )
            )

        dir_rows = mm.dropna(subset=["RMSE", "dir_acc_pct"])
        if not dir_rows.empty:
            out["tradeoff"] = fig_json(
                _tradeoff_chart(
                    dir_rows["model"].tolist(),
                    [float(v) for v in dir_rows["RMSE"].tolist()],
                    [float(v) for v in dir_rows["dir_acc_pct"].tolist()],
                )
            )
    except Exception:  # noqa: BLE001
        pass

    return out


def _evidence(a: "logic.ReproduceArtifacts") -> dict:
    ms = a.model_selection or {}
    fx = a.inference_fixture or {}
    av = a.artifact_validation or {}
    return {
        "checkpoint_sha256": ms.get("checkpoint_sha256"),
        "source_commit": ms.get("source_commit"),
        "selection_reason": ms.get("selection_reason"),
        "expected_tensor_shape": fx.get("expected_tensor_shape"),
        "expected_predicted_close": fx.get("expected_predicted_close"),
        "validation_status": av.get("status"),
        "forecast_rows": av.get("forecast_rows"),
        "prediction_rows": av.get("prediction_rows"),
        "replay_rows": av.get("replay_rows"),
        "baseline_rows": av.get("baseline_rows"),
    }


def _final_evaluation_fields() -> tuple[dict, list[str]]:
    try:
        evidence = FinalEvidence(config.FINAL_EVIDENCE_DIR)
        paper = evidence.read_csv("forecast/paper_reported_metrics.csv")
        controlled = evidence.read_csv("forecast/controlled_forecast_metrics.csv")
        paired = evidence.read_csv("forecast/paired_significance_tests.csv")
    except EvidenceIntegrityError as exc:
        error = str(exc)
        return (
            {
                "final_evidence_status": "NOT_READY",
                "final_evidence_error": error,
                "paper_reported": [],
                "controlled_local": [],
                "paired_tests": [],
                "reproduction_350d": not_ready_reproduction(error),
                "forecast_robustness": not_ready_robustness(error),
            },
            [error],
        )

    errors: list[str] = []
    try:
        reproduction = build_reproduction_350d(evidence, config.CORE_ROOT)
    except EvidenceIntegrityError as exc:
        error = f"350-day reproduction: {exc}"
        errors.append(error)
        reproduction = not_ready_reproduction(error)
    try:
        robustness = get_forecast_robustness(evidence)
    except EvidenceIntegrityError as exc:
        error = f"forecast robustness: {exc}"
        errors.append(error)
        robustness = not_ready_robustness(error)

    fields = {
        "final_evidence_status": "READY",
        "final_evidence_error": None,
        "final_evidence_sha256": evidence.package_sha256,
        "paper_reported": final_records(paper),
        "controlled_local": final_records(controlled),
        "paired_tests": final_records(paired),
        "paired_test_definitions": {
            "diebold_mariano": "Paired squared-error differential with HAC lag 1.",
            "wilcoxon": "Paired absolute-error differential.",
            "scope": "Only identical local target dates are paired; paper aggregates are not tested against local rows.",
        },
        "reproduction_350d": reproduction,
        "forecast_robustness": robustness,
    }
    return fields, errors


@router.get("")
def get_reproduce() -> dict:
    final_fields, final_errors = _final_evaluation_fields()
    a = logic.load_reproduce_artifacts(
        evaluation_dir=config.EVALUATION_DIR,
        provenance_dir=config.REPRODUCE_PROVENANCE_DIR,
        selected_checkpoint_path=config.SELECTED_CHECKPOINT_PATH,
    )

    present = sorted(a.baseline_models_present)
    missing = sorted(MANDATORY_BASELINES - set(a.baseline_models_present))

    base = {
        "status": a.status,
        "forecast_status": a.forecast_status,
        "replay_status": a.replay_status,
        "baseline_status": a.baseline_status,
        "baseline_models_present": present,
        "baseline_models_missing": missing,
        "errors": list(a.errors),
        **final_fields,
    }
    if final_errors:
        base["status"] = "NOT_READY"
        base["errors"].extend(final_errors)
    if base["status"] != "READY":
        return base  # honest NOT_READY; frontend renders the errors, no fabrication

    # Trading replay is reported per split; expose test split (the headline period).
    replay = a.trading_replay_metrics
    replay_test = (
        replay[replay["split"].astype(str).str.lower() == "test"]
        if replay is not None and not replay.empty and "split" in replay.columns
        else replay
    )

    base.update(
        {
            "forecast_metrics": _records(a.forecast_metrics),
            "trading_replay_test": _records(replay_test),
            "baseline_comparison": _records(a.baseline_comparison),
            "significance": _records(a.significance),
            "model_metrics": _records(_model_metrics_frame(a)),
            "evidence": _evidence(a),
            "charts": _charts(a),
        }
    )
    return base
