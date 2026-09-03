"""Strict CPU inference for the two frozen checkpoints (CM-v, S5-Full).

Bypasses Lightning and rebuilds the model directly from ``contracts.MODEL_SPECS``:
the checkpoint bytes are hashed against the pinned SHA-256, the state dict is
loaded with ``strict=True``, and the feature tensor is built to the per-model
window/normalisation contract. There is no fallback path — any mismatch raises.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timezone
import hashlib
import math
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, cast

import torch

from models.cmamba import CMamba
from models.cmamba_t import CMambaT
from thesis_pipeline.contracts import MODEL_SPECS


InferenceModelId = Literal["cmamba_v_reproduced", "s5_full"]
INFERENCE_MODEL_IDS: tuple[InferenceModelId, ...] = (
    "cmamba_v_reproduced",
    "s5_full",
)
TIMESTAMP_UTC_HOUR = 7


class PredictionValidationError(ValueError):
    """The request cannot be evaluated without violating the model contract."""


@dataclass(frozen=True)
class Candle:
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def parsed_date(self) -> date:
        return _parse_iso_date(self.date, field="candle date")

    @property
    def timestamp_utc_0700(self) -> int:
        instant = datetime.combine(
            self.parsed_date,
            time(hour=TIMESTAMP_UTC_HOUR),
            tzinfo=timezone.utc,
        )
        return int(instant.timestamp())


@dataclass(frozen=True)
class PredictionRequest:
    model_id: InferenceModelId
    candles: tuple[Candle, ...]
    prediction_date: str


@dataclass(frozen=True)
class PredictionResult:
    model_id: str
    inference_type: str
    prediction_date: str
    predicted_close: float
    last_close: float
    expected_return_pct: float
    window_days: int
    window_start: str
    window_end: str
    checkpoint_sha256: str
    source_commit: str


@dataclass(frozen=True)
class PreparedModelInput:
    model_id: InferenceModelId
    tensor: torch.Tensor
    window: tuple[Candle, ...]
    last_close: float

    def reconstruct_close(self, raw_output: float | torch.Tensor) -> float:
        output = torch.as_tensor(raw_output, dtype=torch.float32).reshape(-1)[0]
        if self.model_id == "cmamba_v_reproduced":
            return float(output.item())
        anchor = torch.tensor(self.last_close, dtype=torch.float32)
        return float((output * anchor + anchor).item())


def _parse_iso_date(value: object, *, field: str) -> date:
    if not isinstance(value, str):
        raise PredictionValidationError(f"{field} must be an ISO date string")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise PredictionValidationError(
            f"{field} must use valid YYYY-MM-DD format"
        ) from exc
    if parsed.isoformat() != value:
        raise PredictionValidationError(f"{field} must use YYYY-MM-DD format")
    return parsed


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PredictionValidationError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise PredictionValidationError(f"{field} must be finite")
    return number


def _parse_candle(payload: object, *, index: int) -> Candle:
    if not isinstance(payload, Mapping):
        raise PredictionValidationError(f"candles[{index}] must be an object")
    required = ("date", "open", "high", "low", "close", "volume")
    missing = [name for name in required if name not in payload]
    if missing:
        raise PredictionValidationError(
            f"candles[{index}] is missing fields: {', '.join(missing)}"
        )
    candle = Candle(
        date=cast(str, payload["date"]),
        open=_finite_number(payload["open"], field=f"candles[{index}].open"),
        high=_finite_number(payload["high"], field=f"candles[{index}].high"),
        low=_finite_number(payload["low"], field=f"candles[{index}].low"),
        close=_finite_number(payload["close"], field=f"candles[{index}].close"),
        volume=_finite_number(payload["volume"], field=f"candles[{index}].volume"),
    )
    _validate_candle(candle, index=index)
    return candle


def _validate_candle(candle: Candle, *, index: int) -> None:
    candle.parsed_date
    prices = (candle.open, candle.high, candle.low, candle.close)
    if not all(math.isfinite(value) for value in (*prices, candle.volume)):
        raise PredictionValidationError(f"candles[{index}] values must be finite")
    if any(value <= 0.0 for value in prices) or candle.volume < 0.0:
        raise PredictionValidationError(
            f"candles[{index}] has invalid non-positive price or negative volume"
        )
    if (
        candle.low > candle.high
        or candle.low > min(candle.open, candle.close)
        or candle.high < max(candle.open, candle.close)
    ):
        raise PredictionValidationError(f"candles[{index}] has invalid OHLC geometry")


def _validate_request(request: PredictionRequest) -> date:
    if request.model_id not in INFERENCE_MODEL_IDS:
        raise PredictionValidationError(
            f"model_id must be one of: {', '.join(INFERENCE_MODEL_IDS)}"
        )
    prediction_date = _parse_iso_date(
        request.prediction_date, field="prediction_date"
    )
    if not request.candles:
        raise PredictionValidationError("candles must not be empty")
    previous: date | None = None
    for index, candle in enumerate(request.candles):
        _validate_candle(candle, index=index)
        current = candle.parsed_date
        if previous is not None and current <= previous:
            raise PredictionValidationError(
                "candle dates must be unique and strictly increasing"
            )
        previous = current
    return prediction_date


def parse_request(payload: object) -> PredictionRequest:
    """Parse and fully validate a JSON-compatible prediction request."""
    if not isinstance(payload, Mapping):
        raise PredictionValidationError("request must be a JSON object")
    model_id = payload.get("model_id")
    if not isinstance(model_id, str) or model_id not in INFERENCE_MODEL_IDS:
        raise PredictionValidationError(
            f"model_id must be one of: {', '.join(INFERENCE_MODEL_IDS)}"
        )
    candles_payload = payload.get("candles")
    if not isinstance(candles_payload, Sequence) or isinstance(
        candles_payload, (str, bytes)
    ):
        raise PredictionValidationError("candles must be an array")
    request = PredictionRequest(
        model_id=cast(InferenceModelId, model_id),
        candles=tuple(
            _parse_candle(candle, index=index)
            for index, candle in enumerate(candles_payload)
        ),
        prediction_date=cast(str, payload.get("prediction_date")),
    )
    _validate_request(request)
    return request


def prepare_model_input(request: PredictionRequest) -> PreparedModelInput:
    """Build the exact causal feature tensor used by a frozen checkpoint."""
    prediction_date = _validate_request(request)
    spec = MODEL_SPECS[request.model_id]
    available = tuple(
        candle for candle in request.candles if candle.parsed_date < prediction_date
    )
    if len(available) < spec.window_days:
        raise PredictionValidationError(
            f"{request.model_id} requires at least {spec.window_days} history candles "
            f"strictly before prediction_date; received {len(available)}"
        )
    window = available[-spec.window_days :]
    last_close = float(torch.tensor(window[-1].close, dtype=torch.float32).item())

    if request.model_id == "cmamba_v_reproduced":
        channels = (
            [float(candle.timestamp_utc_0700) for candle in window],
            [candle.open for candle in window],
            [candle.high for candle in window],
            [candle.low for candle in window],
            [candle.close for candle in window],
            [candle.volume / 1e9 for candle in window],
        )
    else:
        channels = (
            [candle.open / last_close - 1.0 for candle in window],
            [candle.high / last_close - 1.0 for candle in window],
            [candle.low / last_close - 1.0 for candle in window],
            [candle.close / last_close - 1.0 for candle in window],
            [math.log1p(candle.volume / 1e9) for candle in window],
        )
    tensor = torch.tensor(channels, dtype=torch.float32).unsqueeze(0)
    return PreparedModelInput(
        model_id=request.model_id,
        tensor=tensor,
        window=window,
        last_close=last_close,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _construct_model(model_id: InferenceModelId) -> torch.nn.Module:
    if model_id == "cmamba_v_reproduced":
        return CMamba(
            num_features=6,
            hidden_dims=[14, 16, 32, 1],
            d_states=64,
            layer_density=4,
        )
    return CMambaT(
        num_features=5,
        window_size=60,
        d_model=32,
        n_blocks=4,
        d_state=16,
        mlp_ratio=2,
        drop=0.1,
    )


def _load_model(
    model_id: InferenceModelId,
    checkpoint_path: Path,
    expected_sha256: str,
) -> tuple[torch.nn.Module, str]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    actual_sha256 = _sha256(checkpoint_path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"checkpoint hash mismatch for {model_id}: expected {expected_sha256}, "
            f"received {actual_sha256}"
        )
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"checkpoint is unreadable: {checkpoint_path}") from exc
    if not isinstance(checkpoint, Mapping) or not isinstance(
        checkpoint.get("state_dict"), Mapping
    ):
        raise RuntimeError("checkpoint is incompatible: state_dict is missing")

    state_dict: dict[str, Any] = {}
    for key, value in checkpoint["state_dict"].items():
        if not isinstance(key, str):
            raise RuntimeError("checkpoint is incompatible: non-string state key")
        stripped = key[len("model.") :] if key.startswith("model.") else key
        state_dict[stripped] = value

    model = _construct_model(model_id)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"checkpoint is incompatible with {model_id}") from exc
    model.to(device="cpu")
    model.eval()
    return model, actual_sha256


class CheckpointPredictor:
    """Resolve, verify, and execute one of the approved frozen checkpoints."""

    def __init__(
        self,
        *,
        core_root: Path | None = None,
        checkpoint_paths: Mapping[str, Path] | None = None,
        checkpoint_hashes: Mapping[str, str] | None = None,
    ) -> None:
        self.core_root = (
            Path(core_root).resolve()
            if core_root is not None
            else Path(__file__).resolve().parents[1]
        )
        self.checkpoint_paths = dict(checkpoint_paths or {})
        self.checkpoint_hashes = dict(checkpoint_hashes or {})
        if self.checkpoint_paths.keys() != self.checkpoint_hashes.keys():
            raise ValueError(
                "checkpoint path overrides require matching checkpoint hash overrides"
            )
        unknown = set(self.checkpoint_paths) - set(INFERENCE_MODEL_IDS)
        if unknown:
            raise ValueError(f"unknown checkpoint override model(s): {sorted(unknown)}")

    def _checkpoint_binding(self, model_id: InferenceModelId) -> tuple[Path, str]:
        if model_id in self.checkpoint_paths:
            return (
                self.checkpoint_paths[model_id].resolve(),
                self.checkpoint_hashes[model_id],
            )
        spec = MODEL_SPECS[model_id]
        if spec.checkpoint_path is None or spec.checkpoint_sha256 is None:
            raise RuntimeError(f"no checkpoint binding exists for {model_id}")
        return self.core_root / spec.checkpoint_path, spec.checkpoint_sha256

    def predict(self, request: PredictionRequest) -> PredictionResult:
        prepared = prepare_model_input(request)
        checkpoint_path, expected_sha256 = self._checkpoint_binding(request.model_id)
        model, actual_sha256 = _load_model(
            request.model_id, checkpoint_path, expected_sha256
        )
        with torch.inference_mode():
            raw_output = model(prepared.tensor)
        predicted_close = prepared.reconstruct_close(raw_output)
        if not math.isfinite(predicted_close) or predicted_close <= 0.0:
            raise RuntimeError(
                f"checkpoint produced an invalid close prediction for {request.model_id}"
            )
        spec = MODEL_SPECS[request.model_id]
        if spec.source_commit is None:
            raise RuntimeError(f"source provenance is missing for {request.model_id}")
        return PredictionResult(
            model_id=request.model_id,
            inference_type="local_frozen_checkpoint",
            prediction_date=request.prediction_date,
            predicted_close=predicted_close,
            last_close=prepared.last_close,
            expected_return_pct=(predicted_close / prepared.last_close - 1.0) * 100.0,
            window_days=spec.window_days,
            window_start=prepared.window[0].date,
            window_end=prepared.window[-1].date,
            checkpoint_sha256=actual_sha256,
            source_commit=spec.source_commit,
        )


def predict_next_close(request: PredictionRequest) -> PredictionResult:
    return CheckpointPredictor().predict(request)


__all__ = [
    "Candle",
    "CheckpointPredictor",
    "PredictionRequest",
    "PredictionResult",
    "PredictionValidationError",
    "PreparedModelInput",
    "parse_request",
    "predict_next_close",
    "prepare_model_input",
]
