from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from console_api.vendor.cryptomamba_ui.data import MODEL_WINDOW_SIZE


class OfflinePredictionError(ValueError):
    """Raised when the frozen offline_prediction.json is missing or malformed.

    The caller must render an explicit NOT READY state — never a fake value.
    """


# Keys the predict result renderer reads off the prediction dict.
_REQUIRED_RESPONSE_KEYS = (
    "last_close",
    "predicted_close",
    "prediction_date",
    "vanilla_action",
    "smart_action",
)


def load_offline_prediction(path: str | Path) -> dict[str, Any]:
    """Load the frozen defense-day backup produced by the serve notebook (§8).

    The artifact nests the captured response under ``response`` with its own
    ``inference_type == "live"`` (it genuinely WAS a live call at freeze time).
    Loading it for display is an OFFLINE action, so we FE-override the label:
      * extract ``response`` as the prediction dict,
      * force ``inference_type = "offline"`` (honest source label — hard rule #1),
      * return provenance + the frozen input window separately.

    Raises OfflinePredictionError on any problem so the caller renders NOT READY.
    Never returns a partial or faked prediction.
    """
    path = Path(path)
    if not path.is_file():
        raise OfflinePredictionError(f"file not found: {path}")
    try:
        artifact = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise OfflinePredictionError(f"unreadable JSON: {exc}") from exc

    if not isinstance(artifact, dict) or artifact.get("artifact_type") != "offline_prediction":
        raise OfflinePredictionError("not an offline_prediction artifact")

    response = artifact.get("response")
    if not isinstance(response, dict):
        raise OfflinePredictionError("missing 'response' block")

    missing = [key for key in _REQUIRED_RESPONSE_KEYS if key not in response]
    if missing:
        raise OfflinePredictionError(f"response missing keys: {', '.join(missing)}")

    payload = artifact.get("payload")
    candles = payload.get("candles") if isinstance(payload, dict) else None
    if not isinstance(candles, list) or len(candles) < MODEL_WINDOW_SIZE:
        raise OfflinePredictionError(
            f"payload must carry >= {MODEL_WINDOW_SIZE} candles for the input window"
        )

    prediction = dict(response)
    prediction["inference_type"] = "offline"  # FE-override: honest source label
    return {
        "prediction": prediction,
        "window_candles": candles,
        "provenance": {
            "checkpoint_sha256": artifact.get("checkpoint_sha256")
            or response.get("checkpoint_sha256", ""),
            "source_commit": artifact.get("source_commit") or response.get("source_commit", ""),
            "generated_at_utc": artifact.get("generated_at_utc", ""),
            "warning": artifact.get("warning", "Frozen offline prediction — not a live call."),
        },
    }
