from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


class ApiClientError(RuntimeError):
    """Raised when the remote model API cannot satisfy a request."""


@dataclass(frozen=True)
class CryptoMambaApiClient:
    base_url: str
    timeout_seconds: int = 60

    def __post_init__(self) -> None:
        if not self.base_url or not self.base_url.strip():
            raise ValueError("API base URL is required")
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/predict", json=payload)

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        # ngrok free tier serves an HTML browser-warning interstitial unless this
        # header is present; without it the API response is unreachable HTML.
        headers = {"ngrok-skip-browser-warning": "true", **kwargs.pop("headers", {})}
        try:
            response = requests.request(
                method, url, timeout=self.timeout_seconds, headers=headers, **kwargs
            )
        except requests.RequestException as exc:
            raise ApiClientError(f"Cannot reach CryptoMamba API at {url}: {exc}") from exc

        content_type = response.headers.get("content-type", "")
        body: Any
        if "application/json" in content_type:
            body = response.json()
        else:
            body = response.text

        if response.status_code >= 400:
            raise ApiClientError(f"API returned HTTP {response.status_code}: {body}")
        if not isinstance(body, dict):
            raise ApiClientError(f"API returned non-JSON object response: {body}")
        if "error" in body:
            raise ApiClientError(str(body["error"]))
        return body
