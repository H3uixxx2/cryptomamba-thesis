"""Chronological transaction-cost trading backtest.

Drives the paper trading logic (``utils.trade``) over the frozen per-date
predictions in ``output/evaluation/forecast_predictions.csv`` — no model, no GPU,
no live API. Adds explicit transaction costs, a buy & hold baseline, per-day
equity curves and Sharpe/drawdown/trade-count metrics on top.

Decision at day ``t`` uses only ``current_close`` (today's price) and
``predicted_close`` (the model's forecast) — never ``target_close`` (the
realised next-day price), which is used solely to value the resulting position.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import sys
from argparse import ArgumentParser
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

from utils.trade import buy_sell_smart, buy_sell_smart_w_short, buy_sell_vanilla

STRATEGIES = ("buy_hold", "vanilla", "smart", "smart_w_short")
REGIMES = ("bull", "bear", "sideways")
DEFAULT_COSTS = (0.0, 0.1, 0.2)
DEFAULT_RESULT_TYPES = ("official_checkpoint", "retrained_checkpoint")
DEFAULT_SPLITS = ("test", "val")
MODEL_NAME = "CryptoMamba-v"
TRADE_EPS = 1e-12
TRADING_DAYS_PER_YEAR = 365  # BTC trades every calendar day


REGIME_WINDOW = 30
REGIME_UP_PCT = 10.0
REGIME_DOWN_PCT = 10.0


def classify_regimes(closes, window: int = REGIME_WINDOW,
                     up_pct: float = REGIME_UP_PCT, down_pct: float = REGIME_DOWN_PCT):
    """Label each day bull/bear/sideways by trailing %-change of close over `window` days.

    Transparent rule (persisted to metadata): compare close[t] to close[max(0, t-window)].
    > +up_pct => bull, < -down_pct => bear, otherwise sideways. Early days (< window of
    history) use the shortest available lookback and therefore tend to read sideways.
    """
    arr = [float(x) for x in closes]
    labels = []
    for t, today in enumerate(arr):
        ref = arr[max(0, t - window)]
        pct = (today - ref) / ref * 100.0 if ref else 0.0
        if pct > up_pct:
            labels.append("bull")
        elif pct < -down_pct:
            labels.append("bear")
        else:
            labels.append("sideways")
    return labels


def _max_drawdown_pct(values) -> float:
    """Largest peak-to-trough decline (%), running peak. Mirrors utils.trade max_drawdown."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    peak = np.maximum.accumulate(arr)
    drawdown = (arr - peak) / peak
    return float(-drawdown.min() * 100.0)


def _sharpe(portfolio_values) -> float:
    """Annualised Sharpe of daily portfolio returns (rf=0, population std)."""
    arr = np.asarray(portfolio_values, dtype=float)
    if arr.size < 2:
        return 0.0
    returns = np.diff(arr) / arr[:-1]
    std = returns.std(ddof=0)
    if std == 0:
        return 0.0
    return float(returns.mean() / std * math.sqrt(TRADING_DAYS_PER_YEAR))


@dataclass
class BacktestConfig:
    strategy: str
    transaction_cost_pct: float
    risk_pct: float = 2.0
    initial_balance: float = 100.0
    vanilla_threshold: float = 0.01


@dataclass
class BacktestResult:
    final_balance: float = 0.0
    number_of_trades: int = 0
    roi_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe: float = 0.0
    win_rate_pct: float = 0.0
    initial_balance: float = 0.0
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)


def _decide(strategy: str, is_first: bool, today: float, pred: float,
            balance: float, shares: float, config: BacktestConfig):
    """Apply one strategy step. Returns post-decision (balance, shares), cost-free."""
    if strategy == "buy_hold":
        if is_first:  # buy everything once, then hold forever
            shares += balance / today
            balance = 0.0
        return balance, shares
    if strategy == "vanilla":
        return buy_sell_vanilla(today, pred, balance, shares, tr=config.vanilla_threshold)
    if strategy == "smart":
        return buy_sell_smart(today, pred, balance, shares, risk=config.risk_pct)
    if strategy == "smart_w_short":
        return buy_sell_smart_w_short(today, pred, balance, shares, risk=config.risk_pct)
    raise ValueError(f"unknown strategy: {strategy!r}")


def simulate(predictions: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
    if config.strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {config.strategy!r}")

    initial_balance = float(config.initial_balance)
    if predictions.empty:
        return BacktestResult(
            final_balance=initial_balance,
            initial_balance=initial_balance,
            equity_curve=pd.DataFrame(
                columns=[
                    "date", "cash", "position_btc", "portfolio_value",
                    "drawdown_pct", "action", "trade_value", "transaction_cost",
                ]
            ),
        )

    df = predictions.sort_values("prediction_date").reset_index(drop=True)
    balance = initial_balance
    shares = 0.0
    n_trades = 0
    n_wins = 0
    rows = []
    portfolio_values = []
    peak = balance

    for i, row in df.iterrows():
        today = float(row["current_close"])
        pred = float(row["predicted_close"])
        target = float(row["target_close"])
        shares_before = shares

        balance, shares = _decide(config.strategy, i == 0, today, pred, balance, shares, config)

        traded_shares = shares - shares_before
        trade_value = abs(traded_shares) * today
        cost = trade_value * config.transaction_cost_pct / 100.0
        balance -= cost

        action = "HOLD"
        if traded_shares > TRADE_EPS:
            action = "BUY"
        elif traded_shares < -TRADE_EPS:
            action = "SELL"
        if action != "HOLD":
            n_trades += 1
            # A trade "wins" when the realised next-day move agrees with its direction:
            # a buy expects a rise, a sell/reduce expects a fall.
            if (action == "BUY" and target > today) or (action == "SELL" and target < today):
                n_wins += 1

        portfolio = shares * today + balance
        portfolio_values.append(portfolio)
        peak = max(peak, portfolio)
        drawdown_pct = (portfolio - peak) / peak * 100.0 if peak > 0 else 0.0

        rows.append(
            {
                "date": row["prediction_date"],
                "cash": balance,
                "position_btc": shares,
                "portfolio_value": portfolio,
                "drawdown_pct": drawdown_pct,
                "action": action,
                "trade_value": trade_value,
                "transaction_cost": cost,
            }
        )

    final_balance = balance + shares * float(df["target_close"].iloc[-1])
    roi_pct = (final_balance - initial_balance) / initial_balance * 100.0 if initial_balance else 0.0
    max_drawdown_pct = _max_drawdown_pct([initial_balance] + portfolio_values)
    sharpe = _sharpe(portfolio_values)
    win_rate_pct = (n_wins / n_trades * 100.0) if n_trades else 0.0

    return BacktestResult(
        final_balance=final_balance,
        number_of_trades=n_trades,
        roi_pct=roi_pct,
        max_drawdown_pct=max_drawdown_pct,
        sharpe=sharpe,
        win_rate_pct=win_rate_pct,
        initial_balance=initial_balance,
        equity_curve=pd.DataFrame(rows),
    )


EQUITY_COLUMNS = [
    "result_type", "checkpoint", "split", "strategy", "transaction_cost_pct",
    "date", "cash", "position_btc", "portfolio_value", "drawdown_pct",
    "action", "trade_value", "transaction_cost",
]
METRICS_COLUMNS = [
    "result_type", "model", "checkpoint", "split", "strategy", "transaction_cost_pct",
    "initial_balance", "final_balance", "ROI_pct", "max_drawdown_pct", "Sharpe",
    "number_of_trades", "win_rate_pct", "risk_pct", "protocol", "status",
]
REGIME_COLUMNS = [
    "result_type", "checkpoint", "split", "regime", "strategy", "transaction_cost_pct",
    "samples", "RMSE", "MAE", "MAPE_pct", "ROI_pct", "max_drawdown_pct",
]


def _regime_trading(equity_curve: pd.DataFrame, regime_labels):
    """Regime-conditional ROI/MDD: compound only the daily returns on regime days.

    Returns {regime: (roi_pct, max_drawdown_pct)}. Transparent definition: build an
    equity index from 1.0 over the days labelled with that regime, using each day's own
    return vs the previous calendar day (sliced from the full chronological run, so no
    look-ahead is introduced)."""
    pv = equity_curve["portfolio_value"].to_numpy(dtype=float)
    out = {}
    for regime in REGIMES:
        eq = [1.0]
        for i in range(1, len(pv)):
            if regime_labels[i] == regime:
                r = (pv[i] / pv[i - 1] - 1.0) if pv[i - 1] else 0.0
                eq.append(eq[-1] * (1.0 + r))
        roi = (eq[-1] - 1.0) * 100.0 if len(eq) > 1 else 0.0
        out[regime] = (roi, _max_drawdown_pct(eq))
    return out


def _protocol(source_commit) -> str:
    base = (
        "forecast-driven utils.trade strategies; explicit per-transaction cost on traded "
        "notional; chronological; decision at t uses current_close+predicted_close only; "
        f"regime=trailing {REGIME_WINDOW}d +/-{REGIME_UP_PCT:g}%/{REGIME_DOWN_PCT:g}%"
    )
    return f"{base}; source_commit={source_commit}" if source_commit else base


def run_full_backtest(
    predictions: pd.DataFrame,
    *,
    result_types=DEFAULT_RESULT_TYPES,
    splits=DEFAULT_SPLITS,
    strategies=STRATEGIES,
    costs=DEFAULT_COSTS,
    risk_pct: float = 2.0,
    initial_balance: float = 100.0,
    source_commit=None,
    regime_window: int = REGIME_WINDOW,
) -> dict:
    """Run the full strategy x cost x checkpoint x split backtest matrix.

    Returns {"metrics": df, "equity": df, "regime": df}. Rows are kept distinct per
    (result_type, checkpoint, split, strategy, transaction_cost_pct) — never overwritten.
    """
    protocol = _protocol(source_commit)
    metrics_rows, equity_frames, regime_rows = [], [], []

    for result_type in result_types:
        for split in splits:
            sub = predictions[
                (predictions["result_type"] == result_type) & (predictions["split"] == split)
            ].copy()
            if sub.empty:
                continue
            sub = sub.sort_values("prediction_date").reset_index(drop=True)
            checkpoint = str(sub["checkpoint"].iloc[0])
            regimes = classify_regimes(sub["current_close"].tolist(), window=regime_window)

            # Forecast error per regime is strategy-independent — compute once.
            regime_forecast = {}
            for regime in REGIMES:
                mask = [lab == regime for lab in regimes]
                rsub = sub[pd.Series(mask, index=sub.index)]
                samples = int(len(rsub))
                if samples:
                    err = (rsub["predicted_close"] - rsub["target_close"]).to_numpy(dtype=float)
                    rmse = float(np.sqrt(np.mean(err ** 2)))
                    mae = float(np.mean(np.abs(err)))
                    mape = float(np.mean(np.abs(err) / rsub["target_close"].to_numpy(dtype=float)) * 100.0)
                else:
                    rmse = mae = mape = float("nan")
                regime_forecast[regime] = (samples, rmse, mae, mape)

            for strategy in strategies:
                for cost in costs:
                    res = simulate(
                        sub,
                        BacktestConfig(
                            strategy=strategy,
                            transaction_cost_pct=cost,
                            risk_pct=risk_pct,
                            initial_balance=initial_balance,
                        ),
                    )
                    metrics_rows.append(
                        {
                            "result_type": result_type, "model": MODEL_NAME, "checkpoint": checkpoint,
                            "split": split, "strategy": strategy, "transaction_cost_pct": cost,
                            "initial_balance": initial_balance, "final_balance": res.final_balance,
                            "ROI_pct": res.roi_pct, "max_drawdown_pct": res.max_drawdown_pct,
                            "Sharpe": res.sharpe, "number_of_trades": res.number_of_trades,
                            "win_rate_pct": res.win_rate_pct, "risk_pct": risk_pct,
                            "protocol": protocol, "status": "OK",
                        }
                    )

                    eq = res.equity_curve.copy()
                    eq["result_type"] = result_type
                    eq["checkpoint"] = checkpoint
                    eq["split"] = split
                    eq["strategy"] = strategy
                    eq["transaction_cost_pct"] = cost
                    equity_frames.append(eq[EQUITY_COLUMNS])

                    regime_trade = _regime_trading(res.equity_curve, regimes)
                    for regime in REGIMES:
                        samples, rmse, mae, mape = regime_forecast[regime]
                        roi, mdd = regime_trade[regime]
                        regime_rows.append(
                            {
                                "result_type": result_type, "checkpoint": checkpoint, "split": split,
                                "regime": regime, "strategy": strategy, "transaction_cost_pct": cost,
                                "samples": samples, "RMSE": rmse, "MAE": mae, "MAPE_pct": mape,
                                "ROI_pct": roi, "max_drawdown_pct": mdd,
                            }
                        )

    return {
        "metrics": pd.DataFrame(metrics_rows, columns=METRICS_COLUMNS),
        "equity": pd.concat(equity_frames, ignore_index=True) if equity_frames else pd.DataFrame(columns=EQUITY_COLUMNS),
        "regime": pd.DataFrame(regime_rows, columns=REGIME_COLUMNS),
    }


def write_artifacts(output_dir, results: dict, metadata: dict | None = None) -> dict:
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "trading_metrics.csv": "metrics",
        "trading_equity_curve.csv": "equity",
        "regime_metrics.csv": "regime",
    }
    paths = {}
    for filename, key in files.items():
        path = output_dir / filename
        results[key].to_csv(path, index=False)
        paths[filename] = path
    if metadata is not None:
        meta_path = output_dir / "trading_backtest_metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2, default=str))
        paths["trading_backtest_metadata.json"] = meta_path
    return paths


def _get_args():
    parser = ArgumentParser(description="Phase 4 chronological transaction-cost trading backtest.")
    parser.add_argument("--evaluation_dir", type=str, default=None,
                        help="Dir holding forecast_predictions.csv and where artifacts are written.")
    parser.add_argument("--risk", type=float, default=2.0, help="Smart-strategy risk band, in percent.")
    parser.add_argument("--balance", type=float, default=100.0, help="Initial balance.")
    return parser.parse_args()


def main() -> None:
    args = _get_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    eval_dir = pathlib.Path(args.evaluation_dir) if args.evaluation_dir else root / "output" / "evaluation"
    predictions = pd.read_csv(eval_dir / "forecast_predictions.csv")
    source_commit = (
        str(predictions["source_commit"].iloc[0]) if "source_commit" in predictions.columns else None
    )

    results = run_full_backtest(
        predictions,
        result_types=DEFAULT_RESULT_TYPES,
        splits=DEFAULT_SPLITS,
        strategies=STRATEGIES,
        costs=DEFAULT_COSTS,
        risk_pct=args.risk,
        initial_balance=args.balance,
        source_commit=source_commit,
    )
    metadata = {
        "regime_window_days": REGIME_WINDOW,
        "regime_up_pct": REGIME_UP_PCT,
        "regime_down_pct": REGIME_DOWN_PCT,
        "strategies": list(STRATEGIES),
        "transaction_cost_pct_scenarios": list(DEFAULT_COSTS),
        "risk_pct": args.risk,
        "initial_balance": args.balance,
        "vanilla_threshold": BacktestConfig("vanilla", 0.0).vanilla_threshold,
        "source_commit": source_commit,
        "protocol": _protocol(source_commit),
    }
    paths = write_artifacts(eval_dir, results, metadata=metadata)
    m = results["metrics"]
    print(f"Wrote {len(m)} trading_metrics rows to {paths['trading_metrics.csv']}")
    print(f"Wrote {len(results['equity'])} equity-curve rows; {len(results['regime'])} regime rows")
    cols = ["result_type", "split", "strategy", "transaction_cost_pct", "final_balance", "ROI_pct", "max_drawdown_pct", "Sharpe"]
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(m[cols].to_string(index=False))


if __name__ == "__main__":
    main()
