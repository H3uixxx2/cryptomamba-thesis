"""Contracts for the bounded core worker and checksum-verified evidence loader."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from console_api.core import config
from console_api.loaders.checkpoint_inference import (
    CheckpointInferenceError,
    run_checkpoint_inference,
)
from console_api.loaders.final_evidence import EvidenceIntegrityError, FinalEvidence
from console_api.main import app


client = TestClient(app)
CORE_FIXTURE = (
    config.CORE_ROOT / "tests/fixtures/thesis_golden_inference.json"
)
GOLDEN = json.loads(CORE_FIXTURE.read_text(encoding="utf-8"))


def _worker_result(model_id: str = "cmamba_v_reproduced") -> dict:
    case = GOLDEN[model_id]
    return {
        "model_id": model_id,
        "inference_type": "local_frozen_checkpoint",
        "prediction_date": case["request"]["prediction_date"],
        "predicted_close": case["expected_predicted_close"],
        "last_close": case["last_close"],
        "expected_return_pct": 0.8854217274422371,
        "window_days": len(case["request"]["candles"]),
        "window_start": case["request"]["candles"][0]["date"],
        "window_end": case["request"]["candles"][-1]["date"],
        "checkpoint_sha256": case["checkpoint_sha256"],
        "source_commit": case["source_commit"],
    }


class TestCheckpointAdapter(unittest.TestCase):
    def test_default_core_python_preserves_virtualenv_entrypoint(self) -> None:
        if os.getenv("CRYPTO_MAMBA_CORE_PYTHON"):
            self.skipTest("core Python is explicitly overridden")
        self.assertEqual(
            config.CORE_PYTHON,
            config.CORE_ROOT / ".venv/bin/python",
        )

    @patch("console_api.loaders.checkpoint_inference.subprocess.run")
    def test_success_uses_exact_bounded_worker_command(self, run) -> None:
        payload = GOLDEN["cmamba_v_reproduced"]["request"]
        expected = _worker_result()
        run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(expected), stderr=""
        )

        result = run_checkpoint_inference(payload)

        self.assertEqual(result, expected)
        run.assert_called_once_with(
            [str(config.CORE_PYTHON), str(config.CHECKPOINT_WORKER)],
            input=json.dumps(payload, allow_nan=False, separators=(",", ":")),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=config.CHECKPOINT_TIMEOUT_SECONDS,
            check=False,
            cwd=config.CORE_ROOT,
            shell=False,
        )

    @patch("console_api.loaders.checkpoint_inference.subprocess.run")
    def test_timeout_maps_to_504(self, run) -> None:
        run.side_effect = subprocess.TimeoutExpired(cmd="worker", timeout=120)

        with self.assertRaises(CheckpointInferenceError) as caught:
            run_checkpoint_inference({"model_id": "cmamba_v_reproduced"})

        self.assertEqual(caught.exception.status_code, 504)

    @patch("console_api.loaders.checkpoint_inference.subprocess.run")
    def test_missing_executable_maps_to_503(self, run) -> None:
        run.side_effect = FileNotFoundError("python missing")

        with self.assertRaises(CheckpointInferenceError) as caught:
            run_checkpoint_inference({"model_id": "cmamba_v_reproduced"})

        self.assertEqual(caught.exception.status_code, 503)

    @patch("console_api.loaders.checkpoint_inference.log.error")
    @patch("console_api.loaders.checkpoint_inference.subprocess.run")
    def test_worker_exit_codes_map_without_leaking_unbounded_stderr(self, run, _log) -> None:
        cases = ((2, 400), (3, 503), (4, 500), (9, 500))
        for returncode, expected_status in cases:
            with self.subTest(returncode=returncode):
                run.return_value = subprocess.CompletedProcess(
                    args=[],
                    returncode=returncode,
                    stdout="",
                    stderr=json.dumps(
                        {"error_type": "failure", "message": "x" * 20_000}
                    ),
                )
                with self.assertRaises(CheckpointInferenceError) as caught:
                    run_checkpoint_inference({"model_id": "cmamba_v_reproduced"})
                self.assertEqual(caught.exception.status_code, expected_status)
                self.assertLessEqual(
                    len(caught.exception.detail), config.MAX_WORKER_ERROR_CHARS
                )

    @patch("console_api.loaders.checkpoint_inference.subprocess.run")
    def test_invalid_or_non_object_stdout_is_internal_failure(self, run) -> None:
        for stdout in ("not-json", "[]"):
            with self.subTest(stdout=stdout):
                run.return_value = subprocess.CompletedProcess(
                    args=[], returncode=0, stdout=stdout, stderr=""
                )
                with self.assertRaises(CheckpointInferenceError) as caught:
                    run_checkpoint_inference({"model_id": "cmamba_v_reproduced"})
                self.assertEqual(caught.exception.status_code, 500)


class TestFinalEvidence(unittest.TestCase):
    def test_verified_package_reads_csv_and_json(self) -> None:
        evidence = FinalEvidence(config.FINAL_EVIDENCE_DIR)

        controlled = evidence.read_csv("forecast/controlled_forecast_metrics.csv")
        provenance = evidence.read_json("model/checkpoint_provenance.json")

        self.assertEqual(len(controlled), 6)
        self.assertEqual(
            set(controlled["model_id"]),
            {"cmamba_v_reproduced", "s5_full", "naive_persistence"},
        )
        self.assertEqual(
            set(provenance["models"]),
            {"cmamba_v_reproduced", "s5_full", "naive_persistence"},
        )

    def test_tampering_and_path_traversal_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            copied = Path(tmp) / "final"
            shutil.copytree(config.FINAL_EVIDENCE_DIR, copied)
            metrics = copied / "forecast/controlled_forecast_metrics.csv"
            metrics.write_text(metrics.read_text() + "tampered\n", encoding="utf-8")
            with self.assertRaises(EvidenceIntegrityError):
                FinalEvidence(copied)

        evidence = FinalEvidence(config.FINAL_EVIDENCE_DIR)
        with self.assertRaises(EvidenceIntegrityError):
            evidence.read_csv("../outside.csv")


class TestCheckpointApi(unittest.TestCase):
    def test_setup_makes_checkpoint_primary_and_lists_only_approved_models(self) -> None:
        response = client.get("/api/predict")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["default_mode"], "checkpoint")
        self.assertEqual(
            {model["id"] for model in body["checkpoint_models"]},
            {"cmamba_v_reproduced", "s5_full"},
        )
        self.assertNotIn("s5-150", json.dumps(body).lower())

    @patch("console_api.routers.predict.run_checkpoint_inference")
    def test_checkpoint_endpoint_returns_real_worker_provenance(self, run) -> None:
        run.return_value = _worker_result()
        request = GOLDEN["cmamba_v_reproduced"]["request"]

        response = client.post("/api/predict/checkpoint", json=request)

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["inference_type"], "local_checkpoint")
        self.assertEqual(body["window_days"], 14)
        self.assertEqual(body["checkpoint_sha256"], GOLDEN["cmamba_v_reproduced"]["checkpoint_sha256"])
        self.assertEqual(body["source_commit"], GOLDEN["cmamba_v_reproduced"]["source_commit"])
        self.assertEqual(body["predicted_close"], GOLDEN["cmamba_v_reproduced"]["expected_predicted_close"])
        self.assertNotIn("mock", json.dumps(body).lower())
        run.assert_called_once_with(request)

    @patch("console_api.routers.predict.run_checkpoint_inference")
    def test_checkpoint_rejects_target_not_day_after_selected_window(self, run) -> None:
        request = deepcopy(GOLDEN["cmamba_v_reproduced"]["request"])
        request["prediction_date"] = "2022-12-01"
        worker_result = _worker_result()
        worker_result["prediction_date"] = request["prediction_date"]
        run.return_value = worker_result

        response = client.post("/api/predict/checkpoint", json=request)

        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("one calendar day", response.json()["detail"])
        run.assert_not_called()

    @patch("console_api.routers.predict.run_checkpoint_inference")
    def test_checkpoint_rejects_gap_in_selected_daily_window(self, run) -> None:
        request = deepcopy(GOLDEN["cmamba_v_reproduced"]["request"])
        request["candles"][0]["date"] = "2022-11-01"
        worker_result = _worker_result()
        worker_result["window_start"] = request["candles"][0]["date"]
        run.return_value = worker_result

        response = client.post("/api/predict/checkpoint", json=request)

        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("contiguous daily", response.json()["detail"])
        run.assert_not_called()

    @patch("console_api.routers.predict.run_checkpoint_inference")
    def test_checkpoint_fails_closed_on_evidence_provenance_mismatch(self, run) -> None:
        request = GOLDEN["cmamba_v_reproduced"]["request"]
        mismatches = {
            "checkpoint_sha256": {"checkpoint_sha256": "0" * 64},
            "source_commit": {"source_commit": "0" * 40},
            "window_days": {
                "window_days": 13,
                "window_start": request["candles"][1]["date"],
            },
        }

        for field, changes in mismatches.items():
            with self.subTest(field=field):
                worker_result = _worker_result()
                worker_result.update(changes)
                run.return_value = worker_result

                response = client.post("/api/predict/checkpoint", json=request)

                self.assertEqual(response.status_code, 500, response.text)
                self.assertIn("provenance", response.json()["detail"].lower())
                run.reset_mock()

    @patch("console_api.routers.predict.run_checkpoint_inference")
    def test_checkpoint_worker_errors_keep_http_status(self, run) -> None:
        run.side_effect = CheckpointInferenceError(400, "requires 60 history candles")

        response = client.post(
            "/api/predict/checkpoint", json=GOLDEN["s5_full"]["request"]
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("60 history", response.json()["detail"])

    def test_architecture_exposes_only_current_s5_full_evidence(self) -> None:
        response = client.get("/api/architecture")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["status"], "READY")
        self.assertEqual(body["model"]["display_name"], "CMamba-T / S5-Full")
        self.assertEqual(
            {row["model_id"] for row in body["controlled_metrics"]}, {"s5_full"}
        )
        self.assertTrue(body["paired_tests"])
        self.assertNotIn("s5-150", json.dumps(body).lower())
        self.assertNotIn("ladder", body)
        self.assertNotIn("round2", body)


if __name__ == "__main__":
    unittest.main()
