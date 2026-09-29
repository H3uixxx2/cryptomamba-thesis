"""Training-run evidence HTTP endpoint (seeds and ablation)."""
from __future__ import annotations

from fastapi import APIRouter

from ..services import runs_service
from ._http import handle_domain_errors

router = APIRouter(prefix="/api/runs", tags=["runs"])


@router.get("")
@handle_domain_errors
def runs() -> dict:
    return runs_service.build_runs()
