"""Domain errors raised by services; routers map them to HTTP responses.
Any other exception surfaces as a 500.
"""
from __future__ import annotations


class ConsoleError(Exception):
    """Expected, user-facing service failure; ``status_code`` can be overridden per instance."""

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
