"""Request models for the Data screen."""
from __future__ import annotations

from pydantic import BaseModel, Field


class WindowCandle(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    split: str


class WindowRequest(BaseModel):
    candles: list[WindowCandle]
    prediction_date: str
    risk: float = Field(2.0, ge=0.5, le=10.0)
