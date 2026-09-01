"""Domain errors raised by the service layer.

Services must not import FastAPI. They raise these; the routers translate them
into HTTP responses. Anything *not* raised as a ConsoleError is an unexpected
defect and is allowed to surface as a 500 — it is never masked as a client error.
"""
from __future__ import annotations


class ConsoleError(Exception):
    """Base class for expected, user-facing service failures.

    ``status_code`` may be overridden per instance when a lower layer (e.g. the
    checkpoint worker adapter) already determined the right status.
    """

    status_code = 500

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code


class InvalidInputError(ConsoleError):
    """The caller supplied data the pipeline legitimately rejects (-> 400)."""

    status_code = 400


class PayloadTooLargeError(ConsoleError):
    """The upload exceeds the configured size cap (-> 413)."""

    status_code = 413


class UpstreamError(ConsoleError):
    """A remote dependency (live model API) failed or answered malformed (-> 502)."""

    status_code = 502


class EvidenceUnavailableError(ConsoleError):
    """Verified evidence needed to answer the request is missing (-> 503)."""

    status_code = 503
