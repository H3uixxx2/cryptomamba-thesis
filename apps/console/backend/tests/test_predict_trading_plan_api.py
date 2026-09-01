"""Contract tests for Predict / Trading / Plan APIs.

Live prediction needs a remote Colab API, so we only assert request validation
there; the offline path is exercised against the real frozen artifact."""
import unittest

from fastapi.testclient import TestClient

from console_api.main import app

client = TestClient(app)


class TestPredictApi(unittest.TestCase):
    def test_setup_shape(self):
        d = client.get("/api/predict").json()
        for k in ("min_date", "max_date", "default_date", "last_close", "last_date",
                  "model_train_horizon", "offline_available", "offline_dates",
                  "default_mode", "checkpoint_models"):
            self.assertIn(k, d)

    def test_setup_exposes_offline_date_list(self):
        """The offline date picker must be a closed set of real test dates (a
        dropdown), never free text — so the UI can only offer selectable dates."""
        d = client.get("/api/predict").json()
        dates = d.get("offline_dates")
        self.assertIsInstance(dates, list)
        if not dates:
            self.skipTest("offline forecast artifact absent")
        # sorted, de-duplicated, and the advertised bounds are exactly its ends
        self.assertEqual(dates, sorted(dates))
        self.assertEqual(len(dates), len(set(dates)))
        self.assertEqual(d["offline_min_date"], dates[0])
        self.assertEqual(d["offline_max_date"], dates[-1])
        self.assertEqual(d["offline_default_date"], dates[-1])

    def test_offline_path(self):
        setup = client.get("/api/predict").json()
        d = client.get("/api/predict/offline").json()
        if not d.get("available"):
            self.skipTest(f"offline artifact absent: {d.get('error')}")
        # Hard rule: a frozen prediction is always labelled offline, never live.
        self.assertEqual(d["inference_type"], "offline")
        for k in ("predicted_close", "last_close", "move_pct", "vanilla_action",
                  "smart_action", "ood", "window", "charts"):
            self.assertIn(k, d)
        self.assertIn("candle", d["charts"])
        # Date-selectable path serves a historical test date, so the realised
        # outcome is known and returned; the single golden-fixture fallback
        # instead carries a `provenance` block. Exactly one contract applies.
        self.assertTrue(("actual_close" in d) or ("provenance" in d))
        if setup.get("offline_default_date"):
            self.assertEqual(d["prediction_date"], setup["offline_default_date"])

    def test_offline_exposes_artifact_backed_raw_and_affine_outputs(self):
        setup = client.get("/api/predict").json()
        target = setup.get("offline_default_date")
        if not target:
            self.skipTest("offline forecast artifact absent")

        d = client.get("/api/predict/offline", params={"date": target}).json()
        variants = {row["id"]: row for row in d.get("forecast_variants", [])}
        self.assertEqual(set(variants), {"raw", "affine"})
        self.assertEqual(variants["raw"]["source"], "official_checkpoint")
        self.assertEqual(
            variants["affine"]["source"],
            "official_checkpoint_affine_calibrated",
        )
        self.assertEqual(variants["raw"]["prediction_date"], target)
        self.assertEqual(variants["affine"]["prediction_date"], target)
        self.assertNotEqual(
            variants["raw"]["predicted_close"],
            variants["affine"]["predicted_close"],
        )
        self.assertEqual(d["predicted_close"], variants["raw"]["predicted_close"])
        self.assertEqual(d["window"]["size"], 14)
        self.assertEqual(len(d["window"]["rows"]), 14)

    def test_offline_date_param_is_honoured(self):
        """A specific in-range test date must return that date's frozen row."""
        setup = client.get("/api/predict").json()
        dates = setup.get("offline_dates") or []
        if not dates:
            self.skipTest("offline forecast artifact absent")
        target = dates[len(dates) // 2]  # a mid-range date, not just the default
        d = client.get("/api/predict/offline", params={"date": target}).json()
        self.assertTrue(d.get("available"))
        self.assertEqual(d["inference_type"], "offline")
        self.assertEqual(d["prediction_date"], target)

    def test_offline_out_of_range_date_snaps_not_errors(self):
        """Garbage/out-of-range dates must snap to a real test date, never 500."""
        setup = client.get("/api/predict").json()
        dates = setup.get("offline_dates") or []
        if not dates:
            self.skipTest("offline forecast artifact absent")
        r = client.get("/api/predict/offline", params={"date": "2050-12-31"})
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertTrue(d.get("available"))
        self.assertIn(d["prediction_date"], dates)  # snapped onto a valid date

    def test_live_requires_api_url(self):
        # Empty api_url must be rejected at validation, not attempted.
        r = client.post("/api/predict/live", json={"api_url": "", "risk": 2.0})
        self.assertEqual(r.status_code, 422)


class TestTradingApi(unittest.TestCase):
    def test_simulate_shape(self):
        r = client.post("/api/trading/simulate", json={
            "current": 27000.0, "predicted": 27500.0, "capital": 10000.0,
            "btc": 0.0, "risk": 2.0, "realized_move": 1.5,
        })
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertEqual(len(d["rows"]), 2)  # Vanilla + Smart
        self.assertEqual({row["strategy"] for row in d["rows"]}, {"Vanilla", "Smart"})
        self.assertIn("roi", d["charts"])
        self.assertAlmostEqual(d["assumed_close"], 27000.0 * 1.015, places=2)

    def test_simulate_rejects_nonpositive_price(self):
        r = client.post("/api/trading/simulate", json={"current": 0, "predicted": 100})
        self.assertEqual(r.status_code, 422)


class TestTradingBacktestApi(unittest.TestCase):
    def test_backtest_shape(self):
        d = client.get("/api/trading/backtest").json()
        if d.get("status") != "READY":
            self.skipTest(f"backtest artifacts absent: {d.get('error')}")
        for k in ("result_type", "split", "ref_cost", "available", "metrics", "regime",
                  "metadata", "charts", "final_evidence_status", "paper_replay",
                  "corrected", "corrected_metadata"):
            self.assertIn(k, d)
        self.assertEqual(d["final_evidence_status"], "READY")
        self.assertTrue(d["paper_replay"])
        self.assertEqual(len(d["corrected"]), 72)
        # selection defaults to the thesis-primary retrained checkpoint, test split
        self.assertEqual(d["result_type"], "retrained_checkpoint")
        self.assertEqual(d["split"], "test")
        # 4 strategies x 3 costs = 12 metric rows for one checkpoint+split
        self.assertEqual(len(d["metrics"]), 12)
        row = d["metrics"][0]
        for col in ("strategy", "transaction_cost_pct", "final_balance", "ROI_pct",
                    "max_drawdown_pct", "Sharpe", "number_of_trades", "win_rate_pct"):
            self.assertIn(col, row)
        # regime rows present (bull/bear/sideways x strategies x costs)
        self.assertTrue(d["regime"])
        self.assertEqual({r["regime"] for r in d["regime"]}, {"bull", "bear", "sideways"})
        # transparent regime rules surfaced from metadata
        self.assertIn("regime_window_days", d["metadata"])
        # charts assembled
        for c in ("comparison", "cost_sensitivity", "equity", "drawdown"):
            self.assertIn(c, d["charts"])

    def test_backtest_split_param_selects_val(self):
        d = client.get("/api/trading/backtest", params={"split": "val"}).json()
        if d.get("status") != "READY":
            self.skipTest("backtest artifacts absent")
        self.assertEqual(d["split"], "val")
        self.assertTrue(all(r["split"] == "val" for r in d["metrics"]))

    def test_replay_returns_chronological_artifact_rows(self):
        response = client.get(
            "/api/trading/replay",
            params={
                "result_type": "official_checkpoint",
                "split": "test",
                "strategy": "smart_w_short",
                "ref_cost": 0.1,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        d = response.json()
        if d.get("status") != "READY":
            self.skipTest(f"replay artifacts absent: {d.get('error')}")

        self.assertEqual(d["result_type"], "official_checkpoint")
        self.assertEqual(d["split"], "test")
        self.assertEqual(d["strategy"], "smart_w_short")
        self.assertEqual(d["ref_cost"], 0.1)
        self.assertEqual(len(d["rows"]), 350)
        self.assertGreaterEqual(len(d["candles"]), 364)
        self.assertEqual(
            [candle["date"] for candle in d["candles"]],
            sorted(candle["date"] for candle in d["candles"]),
        )

        dates = [row["date"] for row in d["rows"]]
        self.assertEqual(dates, sorted(dates))
        self.assertEqual(len(dates), len(set(dates)))
        for column in (
            "open",
            "high",
            "low",
            "close",
            "window_start_date",
            "window_end_date",
            "decision_date",
            "outcome_date",
            "current_close",
            "target_close",
            "predicted_close",
            "predicted_return_pct",
            "actual_return_pct",
            "action",
            "cash",
            "position_btc",
            "portfolio_value",
            "drawdown_pct",
            "trade_value",
            "transaction_cost",
        ):
            self.assertIn(column, d["rows"][0])
        self.assertEqual(d["rows"][0]["decision_date"], d["rows"][0]["window_end_date"])
        self.assertEqual(d["rows"][0]["outcome_date"], d["rows"][0]["date"])
        self.assertLess(d["rows"][0]["decision_date"], d["rows"][0]["outcome_date"])
        self.assertAlmostEqual(d["rows"][0]["close"], d["rows"][0]["target_close"], places=2)

        self.assertEqual(d["summary"]["number_of_trades"], 340)
        self.assertEqual(d["provenance"]["evidence_kind"], "historical_backtest")
        self.assertFalse(d["provenance"]["order_execution"])

    def test_replay_rejects_unknown_filters_instead_of_switching_experiment(self):
        response = client.get(
            "/api/trading/replay",
            params={
                "result_type": "missing",
                "split": "missing",
                "strategy": "missing",
                "ref_cost": 9.9,
            },
        )
        self.assertEqual(response.status_code, 400, response.text)
        d = response.json()
        self.assertIn("Unsupported replay selection", d["detail"])

    def test_replay_discloses_terminal_valuation_difference(self):
        response = client.get(
            "/api/trading/replay",
            params={
                "result_type": "official_checkpoint",
                "split": "test",
                "strategy": "smart_w_short",
                "ref_cost": 0.0,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        d = response.json()
        if d.get("status") != "READY":
            self.skipTest(f"replay artifacts absent: {d.get('error')}")

        summary = d["summary"]
        self.assertAlmostEqual(
            summary["terminal_valuation_delta"],
            summary["final_balance"] - summary["curve_last_value"],
            places=9,
        )
        self.assertIn("valuation_basis", d["provenance"])
        self.assertEqual(
            [row["step"] for row in d["rows"]],
            list(range(len(d["rows"]))),
        )


class TestPlanApi(unittest.TestCase):
    def test_plan_shape(self):
        d = client.get("/api/plan").json()
        self.assertEqual(len(d["page_status"]), 5)
        self.assertTrue(d["artifact_slots"])
        for key in ("validated", "pending", "demo_story"):
            self.assertTrue(d[key])
        # Each artifact slot reports a real present/missing badge.
        for slot in d["artifact_slots"]:
            self.assertIn(slot["badge_text"], ("PRESENT", "MISSING"))
        # Usability study is always disclosed as pending (deliberate deviation).
        self.assertIn("Usability study", {p["name"] for p in d["pending"]})

    def test_plan_state_driven_no_stale_pending(self):
        """A completed item must sit in validated, never in pending — and the
        Trading row must reflect the real backtest artifact state."""
        d = client.get("/api/plan").json()
        slots = {s["name"]: s["present"] for s in d["artifact_slots"]}
        validated = {v["name"] for v in d["validated"]}
        pending = {p["name"] for p in d["pending"]}
        trading = next(p for p in d["page_status"] if p["page"].startswith("04"))

        if slots.get("trading_metrics.csv"):
            # backtest built -> Trading READY, listed validated, not pending
            self.assertEqual(trading["status"], "READY")
            self.assertIn("Full trading backtest", validated)
            self.assertNotIn("Full trading backtest", pending)
        else:
            self.assertEqual(trading["status"], "DEMO")
            self.assertIn("Full trading backtest", pending)


if __name__ == "__main__":
    unittest.main()
