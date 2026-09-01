"""Architecture screen HTTP endpoint."""
from __future__ import annotations

from fastapi import APIRouter

from ..services import architecture_service
from ._http import handle_domain_errors

router = APIRouter(prefix="/api/architecture", tags=["architecture"])


@router.get("")
@handle_domain_errors
def architecture() -> dict:
    return architecture_service.build_architecture()
