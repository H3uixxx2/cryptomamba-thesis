"""Trading screen HTTP endpoints.

Two clearly separated capabilities: the one-day decision demo (``POST /simulate``)
and the historical chronological backtest read from frozen artifacts
(``GET /backtest``, ``GET /replay``). All logic lives in ``services.trading_service``.
"""
from __future__ import annotations

from fastapi import APIRouter

from ..schemas.trading import SimulateRequest
from ..services import trading_service
from ._http import handle_domain_errors

router = APIRouter(prefix="/api/trading", tags=["trading"])


@router.post("/simulate")
@handle_domain_errors
def simulate(req: SimulateRequest) -> dict:
    return trading_service.simulate_one_day(
        current=req.current,
        predicted=req.predicted,
        capital=req.capital,
        btc=req.btc,
        risk=req.risk,
        realized_move=req.realized_move,
    )


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
