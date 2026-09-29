"""Trading screen HTTP endpoints: the backtest and the daily replay, both read from frozen artifacts.

All logic lives in ``services.trading_service``.
"""
from __future__ import annotations

from fastapi import APIRouter

from ..services import trading_service
from ._http import handle_domain_errors

router = APIRouter(prefix="/api/trading", tags=["trading"])


@router.get("/backtest")
@handle_domain_errors
def backtest(
    result_type: str = "retrained_checkpoint",
    split: str = "test",
    ref_cost: float = 0.1,
) -> dict:
    return trading_service.build_backtest(
        result_type=result_type, split=split, ref_cost=ref_cost
    )


@router.get("/replay")
@handle_domain_errors
def replay(
    result_type: str = "retrained_checkpoint",
    split: str = "test",
    strategy: str = "smart",
    ref_cost: float = 0.1,
) -> dict:
    """One artifact-backed daily decision timeline. Never runs the model."""
    return trading_service.build_replay(
        result_type=result_type, split=split, strategy=strategy, ref_cost=ref_cost
    )
