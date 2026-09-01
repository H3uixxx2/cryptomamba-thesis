from __future__ import annotations

from typing import Any

import pandas as pd


def pct(predicted: float, current: float) -> float:
    if current == 0:
        return 0.0
    return (predicted - current) / current * 100


def vanilla_signal(current: float, predicted: float, threshold: float = 0.01) -> str:
    move = (predicted - current) / current
    if move > threshold:
        return "buy"
    if move < -threshold:
        return "sell"
    return "hold"


def smart_signal(current: float, predicted: float, risk_pct: float) -> tuple[str, float]:
    band = predicted * risk_pct / 100
    if band <= 0:
        return "hold", 0.0
    if current > predicted + band:
        return "sell", 100.0
    if current > predicted:
        return "sell", round(max(0.0, min(100.0, (current - predicted) / band * 100)), 1)
    if current > predicted - band:
        return "buy", round(max(0.0, min(100.0, (predicted - current) / band * 100)), 1)
    return "buy", 100.0


def simulate_trade(current: float, predicted: float, capital: float, btc: float, risk: float, realized_price: float) -> pd.DataFrame:
    start_value = capital + btc * current
    rows: list[dict[str, Any]] = []

    action = vanilla_signal(current, predicted)
    cash_v, btc_v = capital, btc
    if action == "buy" and cash_v > 0:
        btc_v += cash_v / current
        cash_v = 0.0
    elif action == "sell" and btc_v > 0:
        cash_v += btc_v * current
        btc_v = 0.0
    end_value = cash_v + btc_v * realized_price
    rows.append({
        "strategy": "Vanilla",
        "action": action.upper(),
        "trade_size": "100%" if action != "hold" else "0%",
        "end_value": end_value,
        "pnl": end_value - start_value,
        "roi_pct": (end_value - start_value) / start_value * 100 if start_value else 0,
    })

    action, amount = smart_signal(current, predicted, risk)
    cash_s, btc_s = capital, btc
    fraction = amount / 100
    if action == "buy" and cash_s > 0:
        spend = cash_s * fraction
        btc_s += spend / current
        cash_s -= spend
    elif action == "sell" and btc_s > 0:
        sold = btc_s * fraction
        cash_s += sold * current
        btc_s -= sold
    end_value = cash_s + btc_s * realized_price
    rows.append({
        "strategy": "Smart",
        "action": action.upper(),
        "trade_size": f"{amount:.1f}%",
        "end_value": end_value,
        "pnl": end_value - start_value,
        "roi_pct": (end_value - start_value) / start_value * 100 if start_value else 0,
    })
    return pd.DataFrame(rows)
