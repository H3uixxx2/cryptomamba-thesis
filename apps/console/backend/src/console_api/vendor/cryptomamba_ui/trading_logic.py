from __future__ import annotations

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
