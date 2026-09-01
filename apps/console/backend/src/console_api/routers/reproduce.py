"""Reproduce / Evaluation screen HTTP endpoint."""
from __future__ import annotations

from fastapi import APIRouter

from ..services import reproduce_service
from ._http import handle_domain_errors

router = APIRouter(prefix="/api/reproduce", tags=["reproduce"])


@router.get("")
@handle_domain_errors
def get_reproduce() -> dict:
    return reproduce_service.build_reproduce()
