"""Trading screen business logic.

Two clearly separated capabilities (Phase 4 spec):
  * :func:`simulate_one_day` — the one-day decision demo (reuses trading_logic.simulate_trade).
  * :func:`build_backtest` / :func:`build_replay` — the thesis-grade historical
    chronological backtest, read from artifacts produced offline by
    ``scripts/run_backtest.py``. No model and no recompute here: pure artifact
    presentation. A missing artifact yields NOT_READY, never a fabricated number.
"""
from __future__ import annotations

import json
import logging

import pandas as pd
import plotly.graph_objects as go

from ..core import config
from ..core.errors import InvalidInputError
from .. import logic
from ..loaders.final_evidence import (
    EvidenceIntegrityError,
    FinalEvidence,
    records as final_records,
)
from .data_service import fig_json

log = logging.getLogger(__name__)

# Display order + colours for the four strategies.
_STRATEGY_ORDER = ["buy_hold", "vanilla", "smart", "smart_w_short"]
_STRATEGY_LABEL = {
    "buy_hold": "Buy & Hold",
    "vanilla": "Vanilla",
    "smart": "Smart",
    "smart_w_short": "Smart+Short",
}
_STRATEGY_COLOR = {
    "buy_hold": "#94a3b8",
    "vanilla": "#2563eb",
    "smart": "#22c55e",
    "smart_w_short": "#a855f7",
}


_ACTION_COLOR = {"BUY": "var(--signal-green)", "SELL": "var(--signal-red)", "HOLD": "var(--signal-amber)"}


def simulate_one_day(
    *,
    current: float,
    predicted: float,
    capital: float,
    btc: float,
    risk: float,
    realized_move: float,
) -> dict:
    """One-day decision demo for the given price/portfolio inputs."""
    realized_price = current * (1 + realized_move / 100.0)
    sim = logic.trading_logic.simulate_trade(
        current, predicted, capital, btc, risk, realized_price
    )
    rows = []
    for _, r in sim.iterrows():
        rows.append(
            {
                "strategy": r["strategy"],
                "action": r["action"],
                "action_color": _ACTION_COLOR.get(str(r["action"]), "var(--text-primary)"),
                "trade_size": r["trade_size"],
                "end_value": float(r["end_value"]),
                "pnl": float(r["pnl"]),
                "roi_pct": float(r["roi_pct"]),
            }
        )
    roi_fig = logic.charts.roi_chart(sim)
    return {
        "start_value": capital + btc * current,
        "assumed_close": realized_price,
        "move_pct": logic.trading_logic.pct(predicted, current),
        "rows": rows,
        "charts": {"roi": json.loads(roi_fig.to_json())},
    }


# --------------------------------------------------------------------------- #
# Historical chronological backtest (artifact presentation only)
# --------------------------------------------------------------------------- #

_METRICS_CSV = "trading_metrics.csv"
_EQUITY_CSV = "trading_equity_curve.csv"
_REGIME_CSV = "regime_metrics.csv"
_PREDICTIONS_CSV = "forecast_predictions.csv"
_METADATA_JSON = "trading_backtest_metadata.json"


def _read_csv(name: str) -> pd.DataFrame | None:
    path = config.EVALUATION_DIR / name
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, OSError, UnicodeDecodeError) as exc:
        # Present-but-unreadable artifact: log it (observable) and degrade to NOT_READY.
        log.warning("trading artifact %s exists but is unreadable: %s", path, exc)
        return None


def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records"))


def _final_trading_fields() -> dict:
    try:
        evidence = FinalEvidence(config.FINAL_EVIDENCE_DIR)
        paper = evidence.read_csv("trading/paper_replay_metrics.csv")
        corrected = evidence.read_csv("trading/corrected_trading_metrics.csv")
        metadata = evidence.read_json("trading/corrected_trading_metadata.json")
    except EvidenceIntegrityError as exc:
        return {
            "final_evidence_status": "NOT_READY",
            "final_evidence_error": str(exc),
            "paper_replay": [],
            "corrected": [],
            "corrected_metadata": {},
        }
    return {
        "final_evidence_status": "READY",
        "final_evidence_error": None,
        "final_evidence_sha256": evidence.package_sha256,
        "paper_replay": final_records(paper),
        "corrected": final_records(corrected),
        "corrected_metadata": metadata,
    }


def _order_strategies(values) -> list[str]:
    present = [s for s in _STRATEGY_ORDER if s in set(values)]
    # keep any unexpected strategy names rather than silently dropping them
    return present + [s for s in dict.fromkeys(values) if s not in _STRATEGY_ORDER]


def _equity_figure(equity: pd.DataFrame, y: str, fill_zero: bool = False) -> go.Figure:
    fig = go.Figure()
    for strategy in _order_strategies(equity["strategy"].unique()):
        sub = equity[equity["strategy"] == strategy].sort_values("date")
        fig.add_trace(
            go.Scatter(
                x=sub["date"].tolist(),
                y=[float(v) for v in sub[y].tolist()],
                mode="lines",
                name=_STRATEGY_LABEL.get(strategy, strategy),
                line=dict(color=_STRATEGY_COLOR.get(strategy), width=2),
                fill="tozeroy" if fill_zero else None,
            )
        )
    fig.update_layout(
        height=340,
        margin=dict(l=12, r=12, t=20, b=36),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True),
    )
    return fig


def _backtest_charts(metrics: pd.DataFrame, equity: pd.DataFrame, ref_cost: float) -> dict:
    out: dict = {}
    strategies = _order_strategies(metrics["strategy"].unique())

    # Strategy comparison — final balance at the reference cost (+ break-even line).
    try:
        ref = metrics[metrics["transaction_cost_pct"] == ref_cost]
        by_strat = {s: ref[ref["strategy"] == s] for s in strategies}
        finals = [float(by_strat[s]["final_balance"].iloc[0]) for s in strategies if not by_strat[s].empty]
        labels = [_STRATEGY_LABEL.get(s, s) for s in strategies if not by_strat[s].empty]
        initial = float(metrics["initial_balance"].iloc[0]) if not metrics.empty else 100.0
        if finals:
            fig = logic.charts.grouped_bar_chart(
                labels,
                {f"Final balance @ {ref_cost:g}% cost": finals},
                title="So sánh chiến lược",
                y_title="Final balance ($)",
                hline=initial,
                hline_label="starting capital",
            )
            out["comparison"] = fig_json(fig)
    except Exception:  # noqa: BLE001
        pass

    # Cost sensitivity — final balance per cost scenario, one bar group per strategy.
    try:
        costs = sorted(float(c) for c in metrics["transaction_cost_pct"].unique())
        series = {}
        for s in strategies:
            vals = []
            for c in costs:
                cell = metrics[(metrics["strategy"] == s) & (metrics["transaction_cost_pct"] == c)]
                vals.append(float(cell["final_balance"].iloc[0]) if not cell.empty else None)
            series[_STRATEGY_LABEL.get(s, s)] = vals
        if series:
            fig = logic.charts.grouped_bar_chart(
                [f"{c:g}%" for c in costs],
                series,
                title="Độ nhạy theo phí giao dịch",
                y_title="Final balance ($)",
                colors=[_STRATEGY_COLOR[s] for s in strategies],
            )
            out["cost_sensitivity"] = fig_json(fig)
    except Exception:  # noqa: BLE001
        pass

    # Equity curve + drawdown at the reference cost.
    try:
        eq = equity[equity["transaction_cost_pct"] == ref_cost]
        if not eq.empty:
            out["equity"] = fig_json(_equity_figure(eq, "portfolio_value"))
            out["drawdown"] = fig_json(_equity_figure(eq, "drawdown_pct", fill_zero=True))
    except Exception:  # noqa: BLE001
        pass

    return out


def build_backtest(
    *,
    result_type: str = "retrained_checkpoint",
    split: str = "test",
    ref_cost: float = 0.1,
) -> dict:
    """Historical chronological backtest for one scenario, from frozen artifacts."""
    final_fields = _final_trading_fields()
    if final_fields["final_evidence_status"] != "READY":
        return {
            "status": "NOT_READY",
            "error": final_fields["final_evidence_error"],
            **final_fields,
        }
    metrics = _read_csv(_METRICS_CSV)
    equity = _read_csv(_EQUITY_CSV)
    regime = _read_csv(_REGIME_CSV)
    if metrics is None or metrics.empty:
        exists = (config.EVALUATION_DIR / _METRICS_CSV).exists()
        reason = (
            f"{_METRICS_CSV} exists under {config.EVALUATION_DIR} but is empty or unreadable"
            if exists
            else f"{_METRICS_CSV} not found under {config.EVALUATION_DIR}"
        )
        return {
            "status": "NOT_READY",
            "error": (
                f"{reason}. Run CryptoMamba/scripts/run_backtest.py to (re)generate the backtest artifacts."
            ),
            **final_fields,
        }

    available = {
        "result_types": sorted(metrics["result_type"].unique().tolist()),
        "splits": sorted(metrics["split"].unique().tolist()),
        "costs": sorted(float(c) for c in metrics["transaction_cost_pct"].unique()),
        "strategies": _order_strategies(metrics["strategy"].unique()),
    }
    if result_type not in available["result_types"]:
        result_type = available["result_types"][0]
    if split not in available["splits"]:
        split = available["splits"][0]
    # Snap to the nearest available cost so float round-trip through the CSV (e.g. 0.1)
    # can never miss the exact-equality filters in _backtest_charts and silently blank a chart.
    ref_cost = min(available["costs"], key=lambda c: abs(c - ref_cost))

    msel = metrics[(metrics["result_type"] == result_type) & (metrics["split"] == split)].copy()
    eqsel = (
        equity[(equity["result_type"] == result_type) & (equity["split"] == split)].copy()
        if equity is not None and not equity.empty
        else pd.DataFrame()
    )
    rgsel = (
        regime[(regime["result_type"] == result_type) & (regime["split"] == split)].copy()
        if regime is not None and not regime.empty
        else pd.DataFrame()
    )

    metadata = {}
    meta_path = config.EVALUATION_DIR / _METADATA_JSON
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text())
        except Exception:  # noqa: BLE001
            metadata = {}

    return {
        "status": "READY",
        "result_type": result_type,
        "split": split,
        "ref_cost": ref_cost,
        "available": available,
        "metrics": _records(msel),
        "regime": _records(rgsel) if not rgsel.empty else [],
        "metadata": metadata,
        "charts": _backtest_charts(msel, eqsel, ref_cost) if not eqsel.empty or not msel.empty else {},
        **final_fields,
    }


def build_replay(
    *,
    result_type: str = "retrained_checkpoint",
    split: str = "test",
    strategy: str = "smart",
    ref_cost: float = 0.1,
) -> dict:
    """Return one artifact-backed daily decision timeline for interactive replay.

    This endpoint never runs the model or backtest engine. It only joins the
    persisted forecast and equity rows for the selected historical scenario.
    """
    metrics = _read_csv(_METRICS_CSV)
    equity = _read_csv(_EQUITY_CSV)
    predictions = _read_csv(_PREDICTIONS_CSV)
    missing = [
        name
        for name, frame in (
            (_METRICS_CSV, metrics),
            (_EQUITY_CSV, equity),
            (_PREDICTIONS_CSV, predictions),
        )
        if frame is None or frame.empty
    ]
    if missing:
        return {
            "status": "NOT_READY",
            "error": f"Missing or unreadable replay artifacts: {', '.join(missing)}",
        }

    available = {
        "result_types": sorted(metrics["result_type"].unique().tolist()),
        "splits": sorted(metrics["split"].unique().tolist()),
        "costs": sorted(float(c) for c in metrics["transaction_cost_pct"].unique()),
        "strategies": _order_strategies(metrics["strategy"].unique()),
    }
    matched_cost = next(
        (cost for cost in available["costs"] if abs(cost - ref_cost) <= 1e-9),
        None,
    )
    if (
        result_type not in available["result_types"]
        or split not in available["splits"]
        or strategy not in available["strategies"]
        or matched_cost is None
    ):
        raise InvalidInputError(
            f"Unsupported replay selection. Available values: {available}"
        )
    ref_cost = matched_cost

    scenario = (
        (metrics["result_type"] == result_type)
        & (metrics["split"] == split)
        & (metrics["strategy"] == strategy)
        & (metrics["transaction_cost_pct"] == ref_cost)
    )
    summary_rows = metrics[scenario]
    equity_rows = equity[
        (equity["result_type"] == result_type)
        & (equity["split"] == split)
        & (equity["strategy"] == strategy)
        & (equity["transaction_cost_pct"] == ref_cost)
    ].copy()
    forecast_rows = predictions[
        (predictions["result_type"] == result_type)
        & (predictions["split"] == split)
    ].copy()
    if summary_rows.empty or equity_rows.empty or forecast_rows.empty:
        return {
            "status": "NOT_READY",
            "error": "The selected replay scenario has no persisted artifact rows.",
        }

    equity_rows["date"] = equity_rows["date"].astype(str)
    forecast_rows["prediction_date"] = forecast_rows["prediction_date"].astype(str)
    candles = logic.DATASET_SERVICE.load_paper_sample().processed_df[
        ["date", "open", "high", "low", "close", "volume"]
    ].copy()
    candles["date"] = candles["date"].astype(str)
    forecast_columns = [
        "prediction_date",
        "window_start_date",
        "window_end_date",
        "current_close",
        "target_close",
        "predicted_close",
        "predicted_return_pct",
        "actual_return_pct",
    ]
    merged = equity_rows.merge(
        forecast_rows[forecast_columns],
        left_on="date",
        right_on="prediction_date",
        how="inner",
        validate="one_to_one",
    ).merge(
        candles[["date", "open", "high", "low", "close"]],
        on="date",
        how="inner",
        validate="one_to_one",
    ).sort_values("date")
    if len(merged) != len(equity_rows):
        return {
            "status": "NOT_READY",
            "error": (
                "Replay artifact dates do not reconcile: "
                f"{len(equity_rows)} equity rows vs {len(merged)} joined forecast rows."
            ),
        }

    merged["decision_date"] = merged["window_end_date"].astype(str)
    merged["outcome_date"] = merged["date"].astype(str)
    merged.insert(0, "step", range(len(merged)))
    row_columns = [
        "step",
        "date",
        "decision_date",
        "outcome_date",
        "open",
        "high",
        "low",
        "close",
        "window_start_date",
        "window_end_date",
        "current_close",
        "target_close",
        "predicted_close",
        "predicted_return_pct",
        "actual_return_pct",
        "action",
        "cash",
        "position_btc",
        "portfolio_value",
        "drawdown_pct",
        "trade_value",
        "transaction_cost",
    ]
    summary = _records(summary_rows.iloc[[0]])[0]
    curve_last_value = float(merged["portfolio_value"].iloc[-1])
    summary["curve_last_value"] = curve_last_value
    summary["terminal_valuation_delta"] = float(summary["final_balance"]) - curve_last_value
    forecast_source = forecast_rows.iloc[0]
    candle_start = str(forecast_rows["window_start_date"].min())
    candle_end = str(forecast_rows["prediction_date"].max())
    replay_candles = candles[
        (candles["date"] >= candle_start) & (candles["date"] <= candle_end)
    ].sort_values("date")
    return {
        "status": "READY",
        "result_type": result_type,
        "split": split,
        "strategy": strategy,
        "ref_cost": ref_cost,
        "available": available,
        "summary": summary,
        "candles": _records(replay_candles),
        "rows": _records(merged[row_columns]),
        "provenance": {
            "evidence_kind": "historical_backtest",
            "order_execution": False,
            "checkpoint": str(summary.get("checkpoint", "")),
            "model": str(summary.get("model", "")),
            "source_commit": str(forecast_source.get("source_commit", "")),
            "protocol": str(summary.get("protocol", "")),
            "artifacts": [_PREDICTIONS_CSV, _METRICS_CSV, _EQUITY_CSV],
            "valuation_basis": (
                "portfolio_value is marked at current_close after each decision; "
                "final_balance values the final open position at the last target_close"
            ),
        },
    }
