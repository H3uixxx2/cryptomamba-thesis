"""Model configuration HTTP endpoint."""
from __future__ import annotations

from fastapi import APIRouter

from ..services import config_service
from ._http import handle_domain_errors

router = APIRouter(prefix="/api/config", tags=["config"])


@router.get("")
@handle_domain_errors
def model_config() -> dict:
    return config_service.build_config()
