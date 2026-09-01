"""Shared transport helper: domain error -> HTTP response.

Only :class:`console_api.core.errors.ConsoleError` is translated. Any other
exception propagates, so an unexpected defect surfaces as a 500 and is never
mislabelled as a client error.
"""
from __future__ import annotations

import functools
import inspect
from typing import Any, Callable

from fastapi import HTTPException

from ..core.errors import ConsoleError


def handle_domain_errors(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Translate ConsoleError into HTTPException, preserving the endpoint signature."""
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await fn(*args, **kwargs)
            except ConsoleError as exc:
                raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

        return async_wrapper

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except ConsoleError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    return wrapper
