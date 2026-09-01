from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import subprocess

import pytest
import torch

from thesis_pipeline.inference import (
    CheckpointPredictor,
    PredictionValidationError,
    parse_request,
    predict_next_close,
    prepare_model_input,
)


CORE_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = CORE_ROOT / "tests/fixtures/thesis_golden_inference.json"
GOLDEN = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def _request(case_id: str):
    return parse_request(GOLDEN[case_id]["request"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_cmamba_preprocessing_matches_training_contract() -> None:
    request = _request("cmamba_v_reproduced")

    prepared = prepare_model_input(request)

    assert tuple(prepared.tensor.shape) == (1, 6, 14)
    candles = request.candles
    assert prepared.tensor[0, 0].tolist() == pytest.approx(
        [float(candle.timestamp_utc_0700) for candle in candles]
    )
    assert prepared.tensor[0, 1].tolist() == pytest.approx(
        [candle.open for candle in candles]
    )
    assert prepared.tensor[0, 2].tolist() == pytest.approx(
        [candle.high for candle in candles]
    )
    assert prepared.tensor[0, 3].tolist() == pytest.approx(
        [candle.low for candle in candles]
    )
    assert prepared.tensor[0, 4].tolist() == pytest.approx(
        [candle.close for candle in candles]
    )
    assert prepared.tensor[0, 5].tolist() == pytest.approx(
        [candle.volume / 1e9 for candle in candles]
    )


def test_s5_preprocessing_matches_training_contract() -> None:
    request = _request("s5_full")

    prepared = prepare_model_input(request)

    assert tuple(prepared.tensor.shape) == (1, 5, 60)
    last_close = prepared.last_close
    expected = [
        [candle.open / last_close - 1.0 for candle in request.candles],
        [candle.high / last_close - 1.0 for candle in request.candles],
        [candle.low / last_close - 1.0 for candle in request.candles],
        [candle.close / last_close - 1.0 for candle in request.candles],
        [math.log1p(candle.volume / 1e9) for candle in request.candles],
    ]
    torch.testing.assert_close(
        prepared.tensor[0], torch.tensor(expected, dtype=torch.float32)
    )
    assert prepared.reconstruct_close(0.01) == pytest.approx(last_close * 1.01)


@pytest.mark.parametrize("case_id", ["cmamba_v_reproduced", "s5_full"])
def test_cpu_inference_matches_frozen_prediction(case_id: str) -> None:
    case = GOLDEN[case_id]

    result = predict_next_close(_request(case_id))

    assert result.predicted_close == pytest.approx(
        case["expected_predicted_close"], abs=case["absolute_tolerance"]
    )
    assert result.last_close == pytest.approx(case["last_close"])
    assert result.checkpoint_sha256 == case["checkpoint_sha256"]
    assert result.source_commit == case["source_commit"]
    assert result.inference_type == "local_frozen_checkpoint"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload["candles"][0].update(date="2022/01/01"), "date"),
        (
            lambda payload: payload["candles"].__setitem__(
                1, dict(payload["candles"][0])
            ),
            "strictly increasing",
        ),
        (
            lambda payload: payload["candles"].__setitem__(
                slice(0, 2), reversed(payload["candles"][:2])
            ),
            "strictly increasing",
        ),
        (lambda payload: payload["candles"][0].update(close=float("nan")), "finite"),
        (
            lambda payload: payload["candles"][0].update(low=20.0, high=10.0),
            "geometry",
        ),
        (
            lambda payload: payload["candles"][0].update(close=1e12),
            "geometry",
        ),
    ],
)
def test_request_validation_rejects_invalid_candles(mutate, message: str) -> None:
    payload = json.loads(json.dumps(GOLDEN["cmamba_v_reproduced"]["request"]))
    mutate(payload)

    with pytest.raises(PredictionValidationError, match=message):
        parse_request(payload)


@pytest.mark.parametrize("case_id", ["cmamba_v_reproduced", "s5_full"])
def test_inference_rejects_insufficient_history(case_id: str) -> None:
    request = _request(case_id)
    short_request = replace(request, candles=request.candles[1:])

    with pytest.raises(PredictionValidationError, match="history"):
        prepare_model_input(short_request)


def test_parser_rejects_unknown_model() -> None:
    payload = json.loads(json.dumps(GOLDEN["cmamba_v_reproduced"]["request"]))
    payload["model_id"] = "arima"

    with pytest.raises(PredictionValidationError, match="model_id"):
        parse_request(payload)


def test_predictor_rejects_missing_checkpoint(tmp_path: Path) -> None:
    predictor = CheckpointPredictor(core_root=tmp_path)

    with pytest.raises(FileNotFoundError, match="checkpoint"):
        predictor.predict(_request("cmamba_v_reproduced"))


def test_predictor_rejects_incompatible_state_dict(tmp_path: Path) -> None:
    checkpoint = tmp_path / "incompatible.ckpt"
    torch.save({"state_dict": {"model.not_a_real_weight": torch.ones(1)}}, checkpoint)
    model_id = "cmamba_v_reproduced"
    predictor = CheckpointPredictor(
        core_root=CORE_ROOT,
        checkpoint_paths={model_id: checkpoint},
        checkpoint_hashes={model_id: _sha256(checkpoint)},
    )

    with pytest.raises(RuntimeError, match="incompatible"):
        predictor.predict(_request(model_id))


def test_prediction_uses_only_candles_before_prediction_date() -> None:
    request = _request("cmamba_v_reproduced")
    target = replace(
        request.candles[-1],
        date=request.prediction_date,
        open=50_000.0,
        high=50_100.0,
        low=49_900.0,
        close=50_000.0,
    )
    extended = replace(request, candles=(*request.candles, target))

    prepared = prepare_model_input(extended)

    assert prepared.window == request.candles
    assert prepared.last_close == pytest.approx(request.candles[-1].close)


def test_cli_emits_one_json_object_without_stdout_noise() -> None:
    payload = json.dumps(GOLDEN["cmamba_v_reproduced"]["request"])

    completed = subprocess.run(
        [str(CORE_ROOT / ".venv/bin/python"), "scripts/checkpoint_inference.py"],
        cwd=CORE_ROOT,
        input=payload,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    response = json.loads(completed.stdout)
    assert response["model_id"] == "cmamba_v_reproduced"
    assert response["predicted_close"] == pytest.approx(
        GOLDEN["cmamba_v_reproduced"]["expected_predicted_close"],
        abs=GOLDEN["cmamba_v_reproduced"]["absolute_tolerance"],
    )
    assert completed.stdout.count("{") == 1
