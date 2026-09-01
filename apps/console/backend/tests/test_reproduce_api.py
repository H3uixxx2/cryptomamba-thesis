"""Contract tests for the Reproduce API. With real artifacts present it must be
READY and expose the evidence tables; the loader must degrade to NOT_READY (not a
500) when the artifact directory is missing."""
import importlib
from pathlib import Path
import shutil
import tempfile
import unittest

import pandas as pd
from fastapi.testclient import TestClient

from console_api.loaders.final_evidence import EvidenceIntegrityError
from console_api.main import app
from console_api.loaders.reproduction_evidence import _load_test_predictions

client = TestClient(app)


class TestReproduceApi(unittest.TestCase):
    def test_ready_shape_with_real_artifacts(self):
        r = client.get("/api/reproduce")
        self.assertEqual(r.status_code, 200)
        d = r.json()
        # Status flags always present regardless of READY/NOT_READY.
        for key in ("status", "forecast_status", "replay_status", "baseline_status",
                    "baseline_models_present", "errors", "final_evidence_status",
                    "paper_reported", "controlled_local", "paired_tests"):
            self.assertIn(key, d)
        self.assertEqual(d["final_evidence_status"], "READY")
        self.assertTrue(d["paper_reported"])
        self.assertEqual(len(d["controlled_local"]), 6)
        self.assertTrue(d["paired_tests"])
        self.assertEqual(
            {row["model_id"] for row in d["controlled_local"]},
            {"cmamba_v_reproduced", "s5_full", "naive_persistence"},
        )
        if d["status"] != "READY":
            self.skipTest(f"artifacts not READY in this env: {d['errors']}")
        for key in ("forecast_metrics", "trading_replay_test", "baseline_comparison",
                    "significance", "model_metrics", "evidence", "charts"):
            self.assertIn(key, d)
        # Forecast comparison has the gap/verdict columns straight from the CSV.
        self.assertTrue(d["forecast_metrics"])
        self.assertIn("RMSE", d["forecast_metrics"][0])
        self.assertIn("status", d["forecast_metrics"][0])
        # Exactly one CryptoMamba-v entry in the cross-model frame (no duplicates).
        cm = [m for m in d["model_metrics"] if "cryptomamba" in m["model"].lower()]
        self.assertEqual(len(cm), 1, d["model_metrics"])
        self.assertTrue(all("dir_coverage_pct" in m for m in d["model_metrics"]))
        naive = next(m for m in d["model_metrics"] if m["model"] == "naive_persistence")
        self.assertEqual(naive["dir_coverage_pct"], 0.0)
        self.assertEqual(cm[0]["dir_coverage_pct"], 100.0)
        # Evidence carries the frozen checkpoint provenance.
        self.assertTrue(d["evidence"]["checkpoint_sha256"])

    def test_350_day_reproduction_is_recomputed_from_pinned_predictions(self):
        response = client.get("/api/reproduce")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("reproduction_350d", response.json())

        result = response.json()["reproduction_350d"]

        self.assertEqual(result["status"], "READY")
        self.assertEqual(result["split"], "test")
        self.assertEqual(result["samples"], 350)
        self.assertEqual(result["date_from"], "2023-10-01")
        self.assertEqual(result["date_to"], "2024-09-14")
        self.assertEqual(result["criterion"]["threshold_pct"], 5.0)
        self.assertIn("project", result["criterion"]["label"].lower())
        self.assertIn("not published", result["criterion"]["label"].lower())
        self.assertEqual(
            result["evidence"]["source_artifact"],
            "output/evaluation/forecast_predictions.csv",
        )
        self.assertEqual(
            result["evidence"]["source_sha256"],
            "223e33da6c9eed7e2b39c67d646c97f24ba1372465ec7dc82e8f36932691ae72",
        )
        self.assertNotIn("/Users/", str(result["evidence"]))

        rows = {row["result_type"]: row for row in result["rows"]}
        self.assertEqual(set(rows), {"official_checkpoint", "retrained_checkpoint"})
        self.assertAlmostEqual(rows["official_checkpoint"]["RMSE"], 1598.092417, places=6)
        self.assertAlmostEqual(rows["official_checkpoint"]["MAE"], 1120.659548, places=6)
        self.assertAlmostEqual(rows["official_checkpoint"]["MAPE_pct"], 2.034328, places=6)
        self.assertAlmostEqual(rows["retrained_checkpoint"]["RMSE"], 1612.352672, places=6)
        self.assertAlmostEqual(rows["retrained_checkpoint"]["MAE"], 1132.540558, places=6)
        self.assertAlmostEqual(rows["retrained_checkpoint"]["MAPE_pct"], 2.048596, places=6)
        self.assertAlmostEqual(
            rows["retrained_checkpoint"]["MAPE_gap_pct"], 0.717598, places=6
        )
        self.assertTrue(
            all("MAPE_pct_gap_pct" not in row for row in rows.values())
        )
        self.assertTrue(all(row["status"] == "PASS" for row in rows.values()))
        self.assertEqual(result["paper_reference"]["model_id"], "cmamba_v")

    def test_forecast_robustness_matches_frozen_primary_and_sensitivity_results(self):
        response = client.get("/api/reproduce")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("forecast_robustness", response.json())

        robustness = response.json()["forecast_robustness"]

        self.assertEqual(robustness["status"], "READY")
        self.assertEqual(
            robustness["protocol"],
            {
                "method": "paired non-circular moving-block bootstrap",
                "resamples": 50_000,
                "primary_block_length": 7,
                "block_lengths": [5, 7, 14],
                "random_seed": 230_813,
                "confidence_level": 0.95,
                "retraining": False,
            },
        )
        self.assertEqual(len(robustness["rmse"]), 6)
        self.assertEqual(len(robustness["rmse_difference"]), 2)
        self.assertEqual(len(robustness["direction"]), 4)
        self.assertEqual(len(robustness["direction_difference"]), 2)
        self.assertEqual(len(robustness["periods"]), 4)
        self.assertEqual(len(robustness["sensitivity"]), 6)
        self.assertTrue(all(row["rmse_ci_includes_zero"] for row in robustness["sensitivity"]))
        self.assertTrue(all(row["direction_ci_includes_zero"] for row in robustness["sensitivity"]))
        self.assertTrue(all(row["samples"] == 152 for row in robustness["periods"]))
        expected_sensitivity = {
            ("val", 5): (-55.80, 5.95, -3.29, 10.86),
            ("test", 5): (-116.46, 23.53, -3.62, 12.83),
            ("val", 7): (-55.69, 6.54, -2.96, 10.86),
            ("test", 7): (-114.49, 23.44, -3.29, 12.50),
            ("val", 14): (-53.02, 4.23, -2.30, 10.20),
            ("test", 14): (-106.55, 16.23, -2.63, 11.51),
        }
        actual_sensitivity = {
            (row["split"], row["block_length"]): (
                round(row["rmse_ci_low"], 2),
                round(row["rmse_ci_high"], 2),
                round(row["direction_ci_low_pp"], 2),
                round(row["direction_ci_high_pp"], 2),
            )
            for row in robustness["sensitivity"]
        }
        self.assertEqual(actual_sensitivity, expected_sensitivity)
        actual_periods = {
            row["period"]: (
                row["date_from"],
                row["date_to"],
                round(row["cmamba_v_rmse"], 2),
                round(row["s5_full_rmse"], 2),
                round(row["persistence_rmse"], 2),
                round(row["cmamba_v_direction_pct"], 2),
                round(row["s5_full_direction_pct"], 2),
            )
            for row in robustness["periods"]
        }
        self.assertEqual(
            actual_periods,
            {
                "Val H1": ("2022-11-16", "2023-04-16", 599.05, 561.60, 559.57, 50.66, 48.68),
                "Val H2": ("2023-04-17", "2023-09-15", 565.49, 553.04, 550.27, 54.61, 49.34),
                "Test H1": ("2023-11-16", "2024-04-15", 1693.50, 1669.26, 1672.69, 59.21, 53.29),
                "Test H2": ("2024-04-16", "2024-09-14", 1713.47, 1652.35, 1642.26, 55.92, 51.32),
            },
        )
        self.assertEqual(
            robustness["evidence"]["source_artifact"],
            "forecast/controlled_predictions.csv",
        )
        self.assertEqual(
            robustness["evidence"]["source_sha256"],
            "c947d129db33d592ccb6d26e26104b5aab28022e88fea32029e983a2acb3dcd6",
        )
        self.assertNotIn("/Users/", str(robustness["evidence"]))

        test_rmse = next(
            row for row in robustness["rmse_difference"] if row["split"] == "test"
        )
        self.assertAlmostEqual(test_rmse["estimate"], -42.685593, places=6)
        self.assertAlmostEqual(test_rmse["ci_low"], -114.489121, places=6)
        self.assertAlmostEqual(test_rmse["ci_high"], 23.440531, places=6)
        test_direction = next(
            row for row in robustness["direction_difference"] if row["split"] == "test"
        )
        self.assertAlmostEqual(test_direction["mcnemar_exact_p"], 0.223613, places=6)

    def test_forecast_robustness_is_stable_across_repeated_requests(self):
        first_body = client.get("/api/reproduce").json()
        second_body = client.get("/api/reproduce").json()
        self.assertIn("forecast_robustness", first_body)
        self.assertIn("forecast_robustness", second_body)
        first = first_body["forecast_robustness"]
        second = second_body["forecast_robustness"]

        self.assertEqual(first, second)

    def test_not_ready_when_artifacts_missing(self):
        # Point the backend at a non-existent core root, reload, and assert graceful NOT_READY.
        import os
        from console_api.core import config

        original = os.environ.get("CRYPTO_MAMBA_CORE_ROOT")
        os.environ["CRYPTO_MAMBA_CORE_ROOT"] = "/tmp/__cm_does_not_exist__"
        try:
            importlib.reload(config)
            from console_api.routers import reproduce as repro
            importlib.reload(repro)
            local = TestClient(_app_with(repro))
            r = local.get("/api/reproduce")
            self.assertEqual(r.status_code, 200)
            d = r.json()
            self.assertEqual(d["status"], "NOT_READY")
            self.assertTrue(d["errors"])
            self.assertIn("reproduction_350d", d)
            self.assertEqual(d["reproduction_350d"]["status"], "NOT_READY")
            self.assertEqual(d["reproduction_350d"]["rows"], [])
            self.assertNotIn("paper_reference", d["reproduction_350d"])
        finally:
            if original is None:
                os.environ.pop("CRYPTO_MAMBA_CORE_ROOT", None)
            else:
                os.environ["CRYPTO_MAMBA_CORE_ROOT"] = original
            importlib.reload(config)
            from console_api.routers import reproduce as repro
            importlib.reload(repro)

    def test_negative_targets_are_rejected_before_mape(self):
        from console_api.core import config

        source = config.CORE_ROOT / "output" / "evaluation" / "forecast_predictions.csv"
        predictions = pd.read_csv(source)
        selected = predictions[
            (predictions["split"] == "test")
            & predictions["result_type"].isin(
                ("official_checkpoint", "retrained_checkpoint")
            )
        ].copy()
        selected["target_close"] = -selected["target_close"].abs()
        with tempfile.TemporaryDirectory() as tmp:
            invalid = Path(tmp) / "forecast_predictions.csv"
            selected.to_csv(invalid, index=False)

            with self.assertRaises(EvidenceIntegrityError):
                _load_test_predictions(invalid.read_bytes())

    def test_tampered_pinned_350_day_source_fails_closed(self):
        import os
        from console_api.core import config

        original_env = os.environ.get("CRYPTO_MAMBA_CORE_ROOT")
        original_source = (
            config.CORE_ROOT / "output" / "evaluation" / "forecast_predictions.csv"
        )
        original_evaluation_dir = config.EVALUATION_DIR
        original_provenance_dir = config.REPRODUCE_PROVENANCE_DIR
        original_checkpoint = config.SELECTED_CHECKPOINT_PATH
        with tempfile.TemporaryDirectory() as tmp:
            core_root = Path(tmp)
            copied_source = core_root / "output" / "evaluation" / "forecast_predictions.csv"
            copied_source.parent.mkdir(parents=True)
            shutil.copyfile(original_source, copied_source)
            copied_source.write_bytes(copied_source.read_bytes() + b"\n")
            os.environ["CRYPTO_MAMBA_CORE_ROOT"] = str(core_root)
            try:
                importlib.reload(config)
                # Keep the already-validated legacy artifacts available. The response
                # must still omit them rather than falling back when the pinned RQ1
                # source itself fails integrity.
                config.EVALUATION_DIR = original_evaluation_dir
                config.REPRODUCE_PROVENANCE_DIR = original_provenance_dir
                config.SELECTED_CHECKPOINT_PATH = original_checkpoint
                from console_api.routers import reproduce as repro
                importlib.reload(repro)
                local = TestClient(_app_with(repro))

                response = local.get("/api/reproduce")
                body = response.json()

                self.assertEqual(response.status_code, 200)
                self.assertEqual(body["status"], "NOT_READY")
                self.assertIn("reproduction_350d", body)
                self.assertEqual(body["reproduction_350d"]["status"], "NOT_READY")
                self.assertEqual(body["reproduction_350d"]["rows"], [])
                self.assertIn("checksum mismatch", body["reproduction_350d"]["error"])
                self.assertNotIn("forecast_metrics", body)
                self.assertNotIn("charts", body)
            finally:
                if original_env is None:
                    os.environ.pop("CRYPTO_MAMBA_CORE_ROOT", None)
                else:
                    os.environ["CRYPTO_MAMBA_CORE_ROOT"] = original_env
                importlib.reload(config)
                from console_api.routers import reproduce as repro
                importlib.reload(repro)

    def test_missing_final_evidence_fails_both_new_contracts_closed(self):
        import os
        from console_api.core import config

        original = os.environ.get("CRYPTO_MAMBA_FINAL_EVIDENCE")
        os.environ["CRYPTO_MAMBA_FINAL_EVIDENCE"] = "/tmp/__cm_final_evidence_missing__"
        try:
            importlib.reload(config)
            from console_api.routers import reproduce as repro
            importlib.reload(repro)
            local = TestClient(_app_with(repro))

            response = local.get("/api/reproduce")
            body = response.json()

            self.assertEqual(response.status_code, 200)
            self.assertEqual(body["status"], "NOT_READY")
            self.assertEqual(body["final_evidence_status"], "NOT_READY")
            self.assertEqual(body["reproduction_350d"]["status"], "NOT_READY")
            self.assertEqual(body["forecast_robustness"]["status"], "NOT_READY")
            self.assertEqual(body["reproduction_350d"]["rows"], [])
            self.assertEqual(body["forecast_robustness"]["rmse"], [])
        finally:
            if original is None:
                os.environ.pop("CRYPTO_MAMBA_FINAL_EVIDENCE", None)
            else:
                os.environ["CRYPTO_MAMBA_FINAL_EVIDENCE"] = original
            importlib.reload(config)
            from console_api.routers import reproduce as repro
            importlib.reload(repro)


def _app_with(repro_module):
    from fastapi import FastAPI

    a = FastAPI()
    a.include_router(repro_module.router)
    return a


if __name__ == "__main__":
    unittest.main()
