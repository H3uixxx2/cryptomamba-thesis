"""Corrected same-close self-financing trading evaluation.

Kept separate from the published replay in :mod:`utils.trade` /
``scripts/run_backtest.py`` because the accounting differs: here every order is
fee-aware, each signal uses only the current and predicted close, and a single
ledger carries cash, signed BTC position, costs and marked equity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Sequence

import numpy as np
import pandas as pd


STRATEGIES = ("buy_hold", "vanilla", "smart", "smart_w_short")
REQUIRED_COLUMNS = {
    "model_id",
    "split",
    "signal_date",
    "prediction_date",
    "current_close",
    "target_close",
    "predicted_close",
}
TRADING_DAYS_PER_YEAR = 365
EPSILON = 1e-10

EQUITY_COLUMNS = [
    "signal_date",
    "prediction_date",
    "execution_price",
    "valuation_price",
    "predicted_close",
    "forecast_return_pct",
    "target_weight",
    "cash_before",
    "position_before_btc",
    "cash",
    "position_btc",
    "short_collateral",
    "gross_exposure_at_execution",
    "gross_exposure_marked",
    "portfolio_value",
    "drawdown_pct",
    "action",
    "trade_value",
    "transaction_cost",
    "borrow_cost",
    "self_financing_error",
]


@dataclass(frozen=True)
class SelfFinancingConfig:
    initial_equity: float = 100.0
    transaction_cost_pct: float = 0.2
    borrow_cost_bps_per_day: float = 0.0
    risk_pct: float = 2.0
    vanilla_threshold: float = 0.01
    max_long_exposure: float = 1.0
    max_short_exposure: float = 1.0
    gross_exposure_cap: float = 1.0


@dataclass(frozen=True)
class SelfFinancingResult:
    final_equity: float
    roi_pct: float
    max_drawdown_pct: float
    sharpe: float
    turnover_notional: float
    number_of_trades: int
    total_fees: float
    total_borrow_cost: float
    reconciliation_error: float
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)


def _validate_config(config: SelfFinancingConfig) -> None:
    values = {
        "initial_equity": config.initial_equity,
        "transaction_cost_pct": config.transaction_cost_pct,
        "borrow_cost_bps_per_day": config.borrow_cost_bps_per_day,
        "risk_pct": config.risk_pct,
        "vanilla_threshold": config.vanilla_threshold,
        "max_long_exposure": config.max_long_exposure,
        "max_short_exposure": config.max_short_exposure,
        "gross_exposure_cap": config.gross_exposure_cap,
    }
    for name, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if config.initial_equity <= 0.0:
        raise ValueError("initial_equity must be positive")
    if not 0.0 <= config.transaction_cost_pct < 100.0:
        raise ValueError("transaction_cost_pct must be in [0, 100)")
    if config.borrow_cost_bps_per_day < 0.0:
        raise ValueError("borrow_cost_bps_per_day must be non-negative")
    if config.risk_pct <= 0.0:
        raise ValueError("risk_pct must be positive")
    if config.vanilla_threshold < 0.0:
        raise ValueError("vanilla_threshold must be non-negative")
    if not 0.0 <= config.max_long_exposure <= 1.0:
        raise ValueError("max_long_exposure must be in [0, 1]")
    if not 0.0 <= config.max_short_exposure <= 1.0:
        raise ValueError("max_short_exposure must be in [0, 1]")
    if not 0.0 < config.gross_exposure_cap <= 1.0:
        raise ValueError("gross_exposure_cap must be in (0, 1]")
    if config.max_long_exposure > config.gross_exposure_cap:
        raise ValueError("max_long_exposure exceeds gross_exposure_cap")
    if config.max_short_exposure > config.gross_exposure_cap:
        raise ValueError("max_short_exposure exceeds gross_exposure_cap")


def _validated_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"prediction columns missing: {sorted(missing)}")
    frame = predictions.copy().reset_index(drop=True)
    if frame.empty:
        return frame
    if frame["prediction_date"].duplicated().any():
        raise ValueError("prediction_date values must be unique; duplicate found")

    prediction_dates = pd.to_datetime(
        frame["prediction_date"], format="%Y-%m-%d", errors="coerce"
    )
    signal_dates = pd.to_datetime(
        frame["signal_date"], format="%Y-%m-%d", errors="coerce"
    )
    if prediction_dates.isna().any() or signal_dates.isna().any():
        raise ValueError("signal_date and prediction_date must use YYYY-MM-DD")
    if not prediction_dates.is_monotonic_increasing:
        raise ValueError("prediction_date must already be sorted chronologically")
    if not (signal_dates < prediction_dates).all():
        raise ValueError("signal_date must be strictly before prediction_date")

    prices = frame[["current_close", "target_close", "predicted_close"]].to_numpy(
        dtype=float
    )
    if not np.isfinite(prices).all():
        raise ValueError("prices must be finite")
    if (prices <= 0.0).any():
        raise ValueError("prices must be positive")
    return frame


def _policy_target_weight(
    strategy: str,
    *,
    forecast_return: float,
    current_weight: float,
    config: SelfFinancingConfig,
    first: bool,
) -> tuple[float, bool]:
    """Return desired signed exposure and whether the policy explicitly holds."""
    if strategy == "buy_hold":
        if first:
            return min(config.max_long_exposure, config.gross_exposure_cap), False
        return current_weight, True
    if strategy == "vanilla":
        if abs(forecast_return) < config.vanilla_threshold:
            return current_weight, True
        return (
            min(config.max_long_exposure, config.gross_exposure_cap)
            if forecast_return > 0.0
            else 0.0
        ), False
    scaled = forecast_return / (config.risk_pct / 100.0)
    if strategy == "smart":
        return min(config.max_long_exposure, max(0.0, scaled)), False
    if strategy == "smart_w_short":
        return min(
            config.max_long_exposure,
            max(-config.max_short_exposure, scaled),
        ), False
    raise ValueError(f"unknown strategy: {strategy!r}")


def _desired_notional(
    *,
    equity_before: float,
    current_notional: float,
    target_weight: float,
    fee_rate: float,
) -> float:
    """Solve target notional on post-fee equity.

    For desired notional ``n``, post-trade equity is
    ``E - fee * abs(n-current_notional)``.  The unique solution of
    ``n = target_weight * post_trade_equity`` is found by bisection.  This
    includes fees in both long and short sizing and remains valid when an order
    crosses through zero.
    """
    if abs(target_weight) <= EPSILON:
        return 0.0

    span = equity_before / max(1.0 - fee_rate, EPSILON)
    span += abs(current_notional) + equity_before
    lower, upper = -span, span

    def residual(notional: float) -> float:
        post_fee_equity = equity_before - fee_rate * abs(
            notional - current_notional
        )
        return notional - target_weight * post_fee_equity

    if residual(lower) > 0.0 or residual(upper) < 0.0:
        raise RuntimeError("could not bracket fee-aware target exposure")
    for _ in range(100):
        middle = (lower + upper) / 2.0
        if residual(middle) < 0.0:
            lower = middle
        else:
            upper = middle
    return (lower + upper) / 2.0


def _empty_result(initial_equity: float) -> SelfFinancingResult:
    return SelfFinancingResult(
        final_equity=initial_equity,
        roi_pct=0.0,
        max_drawdown_pct=0.0,
        sharpe=0.0,
        turnover_notional=0.0,
        number_of_trades=0,
        total_fees=0.0,
        total_borrow_cost=0.0,
        reconciliation_error=0.0,
        equity_curve=pd.DataFrame(columns=EQUITY_COLUMNS),
    )


def _result(
    initial_equity: float,
    rows: list[dict[str, Any]],
) -> SelfFinancingResult:
    if not rows:
        return _empty_result(initial_equity)
    equity_curve = pd.DataFrame(rows, columns=EQUITY_COLUMNS)
    values = np.concatenate(
        ([initial_equity], equity_curve["portfolio_value"].to_numpy(dtype=float))
    )
    peaks = np.maximum.accumulate(values)
    drawdowns = (values - peaks) / peaks * 100.0
    equity_curve["drawdown_pct"] = drawdowns[1:]
    returns = np.diff(values) / values[:-1]
    return_std = float(returns.std(ddof=0))
    sharpe = (
        float(returns.mean() / return_std * math.sqrt(TRADING_DAYS_PER_YEAR))
        if return_std > 0.0
        else 0.0
    )
    final_equity = float(equity_curve.iloc[-1]["portfolio_value"])
    return SelfFinancingResult(
        final_equity=final_equity,
        roi_pct=(final_equity / initial_equity - 1.0) * 100.0,
        max_drawdown_pct=float(-drawdowns.min()),
        sharpe=sharpe,
        turnover_notional=float(equity_curve["trade_value"].sum()),
        number_of_trades=int((equity_curve["action"] != "HOLD").sum()),
        total_fees=float(equity_curve["transaction_cost"].sum()),
        total_borrow_cost=float(equity_curve["borrow_cost"].sum()),
        reconciliation_error=abs(
            final_equity - float(equity_curve.iloc[-1]["portfolio_value"])
        ),
        equity_curve=equity_curve,
    )


def simulate_self_financing(
    predictions: pd.DataFrame,
    strategy: str,
    config: SelfFinancingConfig,
) -> SelfFinancingResult:
    """Run one strategy using a causal signal and one self-financing ledger."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy!r}")
    _validate_config(config)
    frame = _validated_predictions(predictions)
    if frame.empty:
        return _empty_result(config.initial_equity)

    fee_rate = config.transaction_cost_pct / 100.0
    borrow_rate = config.borrow_cost_bps_per_day / 10_000.0
    cash = float(config.initial_equity)
    position = 0.0
    rows: list[dict[str, Any]] = []

    for index, row in frame.iterrows():
        execution_price = float(row["current_close"])
        valuation_price = float(row["target_close"])
        predicted_close = float(row["predicted_close"])
        equity_before = cash + position * execution_price
        if not math.isfinite(equity_before) or equity_before <= 0.0:
            raise RuntimeError("portfolio equity must remain finite and positive")

        current_notional = position * execution_price
        current_weight = current_notional / equity_before
        forecast_return = predicted_close / execution_price - 1.0
        target_weight, explicit_hold = _policy_target_weight(
            strategy,
            forecast_return=forecast_return,
            current_weight=current_weight,
            config=config,
            first=index == 0,
        )
        target_weight = min(
            config.gross_exposure_cap,
            max(-config.gross_exposure_cap, target_weight),
        )

        if explicit_hold:
            desired_position = position
        else:
            desired_notional = _desired_notional(
                equity_before=equity_before,
                current_notional=current_notional,
                target_weight=target_weight,
                fee_rate=fee_rate,
            )
            desired_position = desired_notional / execution_price

        cash_before = cash
        position_before = position
        delta = desired_position - position
        if delta > EPSILON:
            affordable = cash / (execution_price * (1.0 + fee_rate))
            delta = min(delta, affordable)
        trade_value = abs(delta) * execution_price
        transaction_cost = trade_value * fee_rate
        if trade_value <= EPSILON * max(1.0, equity_before):
            delta = 0.0
            trade_value = 0.0
            transaction_cost = 0.0
            action = "HOLD"
        else:
            action = "BUY" if delta > 0.0 else "SELL"
            cash -= delta * execution_price + transaction_cost
            position += delta

        tolerance = EPSILON * max(1.0, equity_before)
        if abs(cash) <= tolerance:
            cash = 0.0
        if cash < -tolerance:
            raise RuntimeError("self-financing execution created negative cash")

        equity_after_execution = cash + position * execution_price
        if not math.isfinite(equity_after_execution) or equity_after_execution <= 0.0:
            raise RuntimeError("post-trade equity must remain finite and positive")
        gross_at_execution = abs(position) * execution_price / equity_after_execution
        if gross_at_execution > config.gross_exposure_cap + 1e-8:
            raise RuntimeError("execution exceeded gross exposure cap")

        short_collateral = max(0.0, -position) * valuation_price
        borrow_cost = short_collateral * borrow_rate
        cash -= borrow_cost
        portfolio_value = cash + position * valuation_price
        if not math.isfinite(portfolio_value) or portfolio_value <= 0.0:
            raise RuntimeError("marked portfolio equity must remain finite and positive")
        gross_marked = abs(position) * valuation_price / portfolio_value
        expected_cash = (
            cash_before
            - delta * execution_price
            - transaction_cost
            - borrow_cost
        )
        self_financing_error = abs(cash - expected_cash)
        if self_financing_error > tolerance:
            raise RuntimeError("cash ledger failed self-financing reconciliation")

        rows.append(
            {
                "signal_date": str(row["signal_date"]),
                "prediction_date": str(row["prediction_date"]),
                "execution_price": execution_price,
                "valuation_price": valuation_price,
                "predicted_close": predicted_close,
                "forecast_return_pct": forecast_return * 100.0,
                "target_weight": target_weight,
                "cash_before": cash_before,
                "position_before_btc": position_before,
                "cash": cash,
                "position_btc": position,
                "short_collateral": short_collateral,
                "gross_exposure_at_execution": gross_at_execution,
                "gross_exposure_marked": gross_marked,
                "portfolio_value": portfolio_value,
                "drawdown_pct": 0.0,
                "action": action,
                "trade_value": trade_value,
                "transaction_cost": transaction_cost,
                "borrow_cost": borrow_cost,
                "self_financing_error": self_financing_error,
            }
        )

    return _result(config.initial_equity, rows)


def run_corrected_matrix(
    predictions: pd.DataFrame,
    *,
    configs: Sequence[SelfFinancingConfig],
) -> dict[str, Any]:
    """Evaluate each model/split/strategy/cost without overwriting a row."""
    missing = {"model_id", "split"} - set(predictions.columns)
    if missing:
        raise ValueError(f"prediction columns missing: {sorted(missing)}")
    metrics_rows: list[dict[str, Any]] = []
    equity_frames: list[pd.DataFrame] = []
    for (model_id, split), frame in predictions.groupby(
        ["model_id", "split"], sort=True
    ):
        frame = frame.reset_index(drop=True)
        for strategy in STRATEGIES:
            for config in configs:
                result = simulate_self_financing(frame, strategy, config)
                metrics_rows.append(
                    {
                        "model_id": str(model_id),
                        "split": str(split),
                        "strategy": strategy,
                        "transaction_cost_pct": config.transaction_cost_pct,
                        "borrow_cost_bps_per_day": config.borrow_cost_bps_per_day,
                        "initial_equity": config.initial_equity,
                        "final_equity": result.final_equity,
                        "ROI_pct": result.roi_pct,
                        "max_drawdown_pct": result.max_drawdown_pct,
                        "Sharpe": result.sharpe,
                        "turnover_notional": result.turnover_notional,
                        "number_of_trades": result.number_of_trades,
                        "total_fees": result.total_fees,
                        "total_borrow_cost": result.total_borrow_cost,
                        "reconciliation_error": result.reconciliation_error,
                    }
                )
                equity = result.equity_curve.copy()
                equity.insert(0, "transaction_cost_pct", config.transaction_cost_pct)
                equity.insert(0, "strategy", strategy)
                equity.insert(0, "split", str(split))
                equity.insert(0, "model_id", str(model_id))
                equity_frames.append(equity)

    return {
        "metrics": pd.DataFrame(metrics_rows),
        "equity": (
            pd.concat(equity_frames, ignore_index=True)
            if equity_frames
            else pd.DataFrame()
        ),
        "metadata": {
            "engine": "self_financing_same_close_v1",
            "mark_basis": "target_close",
            "execution_basis": "signal_day_close",
            "annualization_days": TRADING_DAYS_PER_YEAR,
            "strategies": list(STRATEGIES),
            "gross_exposure_cap_default": 1.0,
        },
    }
