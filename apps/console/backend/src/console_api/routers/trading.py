"""Trading screen endpoints: backtest and daily replay."""
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
    """Daily decision timeline of one scenario."""
    return trading_service.build_replay(
        result_type=result_type, split=split, strategy=strategy, ref_cost=ref_cost
    )
