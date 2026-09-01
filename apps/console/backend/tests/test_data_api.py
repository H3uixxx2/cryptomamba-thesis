"""Contract tests for the Data API. Asserts the documented response shape and that
malformed uploads are client errors (400), not server crashes (500)."""
import io
import unittest
from datetime import date, timedelta
from unittest.mock import patch

from fastapi.testclient import TestClient

from console_api.main import app

client = TestClient(app)


class TestDataApi(unittest.TestCase):
    def test_paper_mode_shape(self):
        r = client.get("/api/data", params={"mode": "paper"})
        self.assertEqual(r.status_code, 200)
        d = r.json()
        for key in (
            "source",
            "metrics",
            "splits",
            "candles",
            "window",
            "prediction_payload",
            "tensor",
            "pipeline",
            "processing",
        ):
            self.assertIn(key, d)
        self.assertNotIn("charts", d)
        self.assertEqual(d["tensor"]["shape"], [1, 6, 14])
        self.assertEqual(d["window"]["size"], 14)
        self.assertEqual(len(d["window"]["rows"]), 14)
        self.assertEqual(len(d["prediction_payload"]["candles"]), 14)
        self.assertGreaterEqual(len(d["metrics"]), 1)
        # splits cover the chronological ordering
        self.assertTrue(any(s["split"] == "test" for s in d["splits"]))
        self.assertEqual(len(d["pipeline"]), 5)
        self.assertEqual(d["pipeline"][0]["artifact"], "2,191 raw rows")
        self.assertEqual(d["pipeline"][1]["artifact"], "2,191 daily candles")
        self.assertEqual(d["pipeline"][2]["artifact"], "2,191 processed rows · 3 splits")
        self.assertEqual(
            d["processing"],
            {
                "raw_rows": 2191,
                "daily_rows": 2191,
                "processed_rows": 2191,
                "split_strategy": "paper_date",
            },
        )

    def test_paper_mode_explorer_contract_is_chronological_and_consistent(self):
        r = client.get("/api/data", params={"mode": "paper"})
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()

        candle_dates = [row["date"] for row in d["candles"]]
        self.assertEqual(candle_dates, sorted(candle_dates))
        self.assertEqual(len(candle_dates), len(set(candle_dates)))
        self.assertTrue(
            all(
                set(("date", "open", "high", "low", "close", "volume", "split")) <= set(row)
                for row in d["candles"]
            )
        )

        window_rows = d["window"]["rows"]
        self.assertEqual(window_rows, d["candles"][-14:])
        self.assertEqual(d["window"]["start"], window_rows[0]["date"])
        self.assertEqual(d["window"]["end"], window_rows[-1]["date"])

        payload = d["prediction_payload"]
        payload_dates = [row["date"] for row in payload["candles"]]
        self.assertEqual(payload_dates, [row["date"] for row in window_rows])
        payload_keys = ("date", "open", "high", "low", "close", "volume")
        self.assertEqual(
            payload["candles"],
            [{key: row[key] for key in payload_keys} for row in window_rows],
        )
        expected_prediction_date = (
            date.fromisoformat(window_rows[-1]["date"]) + timedelta(days=1)
        ).isoformat()
        self.assertEqual(payload["prediction_date"], expected_prediction_date)
        self.assertEqual(payload["risk"], 2.0)

    def test_invalid_mode_rejected(self):
        r = client.get("/api/data", params={"mode": "bogus"})
        self.assertEqual(r.status_code, 422)

    def test_upload_happy_path(self):
        rows = ["date,open,high,low,close,volume"]
        # 30 strictly-dated daily candles (>= 14, enough for window + split)
        for i in range(1, 31):
            d = f"2020-01-{i:02d}"
            base = 100 + i
            rows.append(f"{d},{base},{base+2},{base-2},{base+1},{1000+i}")
        rows.append("2020-01-01,102,104,100,103,500")
        csv = "\n".join(rows).encode()
        r = client.post(
            "/api/data/upload",
            files={"file": ("custom.csv", io.BytesIO(csv), "text/csv")},
        )
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertEqual(d["source"]["strategy"], "chronological_ratio")
        self.assertEqual(d["tensor"]["shape"], [1, 6, 14])
        self.assertEqual(d["pipeline"][0]["artifact"], "31 raw rows")
        self.assertEqual(d["pipeline"][1]["artifact"], "30 daily candles")
        self.assertEqual(d["pipeline"][2]["artifact"], "30 processed rows · 3 splits")
        self.assertEqual(
            d["processing"],
            {
                "raw_rows": 31,
                "daily_rows": 30,
                "processed_rows": 30,
                "split_strategy": "chronological_ratio",
            },
        )

    def test_upload_bad_csv_is_400(self):
        bad = b"foo,bar\n1,2\n3,4\n"
        r = client.post(
            "/api/data/upload",
            files={"file": ("bad.csv", io.BytesIO(bad), "text/csv")},
        )
        self.assertEqual(r.status_code, 400)
        self.assertIn("detail", r.json())

    def test_upload_csv_with_nul_byte_is_400(self):
        r = client.post(
            "/api/data/upload",
            files={"file": ("nul.csv", io.BytesIO(b"foo\x00bar\n"), "text/csv")},
        )

        self.assertEqual(r.status_code, 400, r.text)
        self.assertIn("Could not read CSV", r.json()["detail"])

    def test_upload_does_not_mask_unexpected_response_defects_as_client_errors(self):
        rows = ["date,open,high,low,close,volume"]
        for day in range(1, 15):
            rows.append(f"2020-01-{day:02d},100,102,98,101,1000")
        no_raise_client = TestClient(app, raise_server_exceptions=False)

        with patch("console_api.routers.data._build_response", side_effect=RuntimeError("defect")):
            r = no_raise_client.post(
                "/api/data/upload",
                files={"file": ("valid.csv", io.BytesIO("\n".join(rows).encode()), "text/csv")},
            )

        self.assertEqual(r.status_code, 500, r.text)

    def test_upload_rejects_when_raw_rows_collapse_below_14_daily_candles(self):
        rows = ["date,open,high,low,close,volume"]
        # 20 valid intraday rows satisfy the raw-row minimum but collapse to only
        # 10 daily candles after aggregation.
        for day in range(1, 11):
            for hour in (0, 12):
                base = 100 + day
                rows.append(
                    f"2020-01-{day:02d}T{hour:02d}:00:00Z,"
                    f"{base},{base+2},{base-2},{base+1},{1000+day}"
                )

        r = client.post(
            "/api/data/upload",
            files={"file": ("intraday.csv", io.BytesIO("\n".join(rows).encode()), "text/csv")},
        )

        self.assertEqual(r.status_code, 400, r.text)
        self.assertIn("at least 14 daily candles", r.json()["detail"])

    def test_upload_rejects_impossible_ohlc_values(self):
        rows = ["date,open,high,low,close,volume"]
        for day in range(1, 15):
            base = 100 + day
            if day == 7:
                # Positive and numeric, but impossible as a candle: high is below
                # open/close and low is above them.
                rows.append(f"2020-01-{day:02d},{base},{base-2},{base+2},{base+1},{1000+day}")
            else:
                rows.append(f"2020-01-{day:02d},{base},{base+2},{base-2},{base+1},{1000+day}")

        r = client.post(
            "/api/data/upload",
            files={"file": ("invalid-ohlc.csv", io.BytesIO("\n".join(rows).encode()), "text/csv")},
        )

        self.assertEqual(r.status_code, 400, r.text)
        self.assertIn("Invalid OHLC candle invariants", r.json()["detail"])

    def test_upload_rejects_non_finite_numeric_values(self):
        rows = ["date,open,high,low,close,volume"]
        for day in range(1, 15):
            volume = "inf" if day == 7 else str(1000 + day)
            base = 100 + day
            rows.append(f"2020-01-{day:02d},{base},{base+2},{base-2},{base+1},{volume}")

        r = client.post(
            "/api/data/upload",
            files={"file": ("non-finite.csv", io.BytesIO("\n".join(rows).encode()), "text/csv")},
        )

        self.assertEqual(r.status_code, 400, r.text)
        self.assertIn("Invalid numeric OHLCV", r.json()["detail"])

    def test_window_endpoint_builds_selected_model_contract(self):
        candles = self._window_candles()

        r = client.post(
            "/api/data/window",
            json={"candles": candles, "prediction_date": "2020-01-15", "risk": 2.0},
        )

        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertEqual(set(d), {"window", "tensor", "prediction_payload"})
        self.assertEqual(d["window"]["size"], 14)
        self.assertEqual(d["window"]["rows"], candles)
        self.assertEqual(d["tensor"]["shape"], [1, 6, 14])
        self.assertEqual(d["prediction_payload"]["prediction_date"], "2020-01-15")
        self.assertEqual(len(d["prediction_payload"]["candles"]), 14)
        dates = [row["date"] for row in d["window"]["rows"]]
        self.assertEqual(dates, sorted(set(dates)))
        self.assertTrue(all(day < d["prediction_payload"]["prediction_date"] for day in dates))
        payload_keys = ("date", "open", "high", "low", "close", "volume")
        self.assertEqual(
            d["prediction_payload"]["candles"],
            [{key: row[key] for key in payload_keys} for row in d["window"]["rows"]],
        )

    def test_window_endpoint_rejects_non_exact_unsorted_or_duplicate_windows(self):
        valid = self._window_candles()
        cases = (
            valid[:-1],
            list(reversed(valid)),
            [*valid[:-1], valid[-2]],
        )

        for candles in cases:
            with self.subTest(candle_count=len(candles), dates=[row["date"] for row in candles]):
                r = client.post(
                    "/api/data/window",
                    json={"candles": candles, "prediction_date": "2020-01-15"},
                )
                self.assertEqual(r.status_code, 400, r.text)

    def test_window_endpoint_rejects_input_on_or_after_prediction_date(self):
        r = client.post(
            "/api/data/window",
            json={"candles": self._window_candles(), "prediction_date": "2020-01-14"},
        )

        self.assertEqual(r.status_code, 400, r.text)
        self.assertIn("strictly before prediction_date", r.json()["detail"])

    @staticmethod
    def _window_candles() -> list[dict]:
        return [
            {
                "date": f"2020-01-{day:02d}",
                "open": float(100 + day),
                "high": float(102 + day),
                "low": float(98 + day),
                "close": float(101 + day),
                "volume": float(1000 + day),
                "split": "train",
            }
            for day in range(1, 15)
        ]


if __name__ == "__main__":
    unittest.main()
