"""Bounded subprocess adapter for the core checkpoint-inference worker."""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import re
import subprocess
from typing import Any

from ..core import config


log = logging.getLogger(__name__)

_REQUIRED_RESULT_FIELDS = {
    "model_id",
    "inference_type",
    "prediction_date",
    "predicted_close",
    "last_close",
    "expected_return_pct",
    "window_days",
    "window_start",
    "window_end",
    "checkpoint_sha256",
    "source_commit",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class CheckpointInferenceError(RuntimeError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail[: config.MAX_WORKER_ERROR_CHARS]


def _worker_error_detail(stderr: str, *, fallback: str) -> str:
    bounded = (stderr or "")[-config.MAX_WORKER_ERROR_CHARS :].strip()
    if not bounded:
        return fallback[: config.MAX_WORKER_ERROR_CHARS]
    # The core worker writes one JSON error object.  If a dependency emitted a
    # warning first, inspect the final line without exposing the full stderr.
    last_line = bounded.splitlines()[-1]
    try:
        payload = json.loads(last_line)
    except json.JSONDecodeError:
        return fallback[: config.MAX_WORKER_ERROR_CHARS]
    if isinstance(payload, dict) and isinstance(payload.get("message"), str):
        return payload["message"][: config.MAX_WORKER_ERROR_CHARS]
    return bounded[: config.MAX_WORKER_ERROR_CHARS]


def run_checkpoint_inference(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute one trusted worker process; never synthesize a prediction."""
    unavailable = [
        str(path)
        for path in (config.CORE_PYTHON, config.CHECKPOINT_WORKER)
        if not Path(path).is_file()
    ]
    if unavailable:
        raise CheckpointInferenceError(
            503, f"Checkpoint worker is unavailable: {', '.join(unavailable)}"
        )
    try:
        request_json = json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise CheckpointInferenceError(400, f"Invalid prediction payload: {exc}") from exc

    try:
        completed = subprocess.run(
            [str(config.CORE_PYTHON), str(config.CHECKPOINT_WORKER)],
            input=request_json,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=config.CHECKPOINT_TIMEOUT_SECONDS,
            check=False,
            cwd=config.CORE_ROOT,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise CheckpointInferenceError(
            504,
            f"Checkpoint inference exceeded {config.CHECKPOINT_TIMEOUT_SECONDS} seconds.",
        ) from exc
    except (FileNotFoundError, NotADirectoryError, PermissionError, OSError) as exc:
        raise CheckpointInferenceError(
            503, f"Checkpoint worker is unavailable: {exc}"
        ) from exc

    if completed.returncode != 0:
        status_code = {2: 400, 3: 503, 4: 500}.get(completed.returncode, 500)
        if completed.stderr:
            log.error(
                "Checkpoint worker exited with code %s: %s",
                completed.returncode,
                completed.stderr[-config.MAX_WORKER_ERROR_CHARS :],
            )
        detail = _worker_error_detail(
            completed.stderr,
            fallback=f"Checkpoint worker exited with code {completed.returncode}.",
        )
        raise CheckpointInferenceError(status_code, detail)

    if len(completed.stdout) > config.MAX_WORKER_STDOUT_CHARS:
        raise CheckpointInferenceError(500, "Checkpoint worker response is too large.")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise CheckpointInferenceError(
            500, "Checkpoint worker returned invalid JSON."
        ) from exc
    if not isinstance(result, dict):
        raise CheckpointInferenceError(
            500, "Checkpoint worker returned a non-object response."
        )
    missing = sorted(_REQUIRED_RESULT_FIELDS - result.keys())
    if missing:
        raise CheckpointInferenceError(
            500,
            f"Checkpoint worker response is missing fields: {', '.join(missing)}",
        )
    string_fields = (
        "model_id",
        "inference_type",
        "prediction_date",
        "window_start",
        "window_end",
        "checkpoint_sha256",
        "source_commit",
    )
    if any(not isinstance(result[field], str) for field in string_fields):
        raise CheckpointInferenceError(500, "Checkpoint worker returned invalid field types.")
    numeric_fields = (
        "predicted_close",
        "last_close",
        "expected_return_pct",
    )
    if any(
        isinstance(result[field], bool)
        or not isinstance(result[field], (int, float))
        or not math.isfinite(float(result[field]))
        for field in numeric_fields
    ):
        raise CheckpointInferenceError(500, "Checkpoint worker returned invalid numeric values.")
    if (
        float(result["predicted_close"]) <= 0.0
        or float(result["last_close"]) <= 0.0
        or isinstance(result["window_days"], bool)
        or not isinstance(result["window_days"], int)
        or result["window_days"] <= 0
        or result["model_id"] != payload.get("model_id")
        or result["prediction_date"] != payload.get("prediction_date")
        or result["inference_type"] != "local_frozen_checkpoint"
        or _SHA256.fullmatch(result["checkpoint_sha256"]) is None
        or _COMMIT.fullmatch(result["source_commit"]) is None
    ):
        raise CheckpointInferenceError(500, "Checkpoint worker response failed validation.")
    return result


__all__ = ["CheckpointInferenceError", "run_checkpoint_inference"]
