"""Request models for the Trading screen."""
from __future__ import annotations

from pydantic import BaseModel, Field


class SimulateRequest(BaseModel):
    current: float = Field(..., gt=0)  # last close
    predicted: float = Field(..., gt=0)  # model forecast
    capital: float = Field(10000.0, ge=0)  # USD cash
    btc: float = Field(0.0, ge=0)  # BTC held
    risk: float = Field(2.0, ge=0.5, le=10.0)
    realized_move: float = Field(0.0, ge=-15.0, le=15.0)  # assumed next-day % move
