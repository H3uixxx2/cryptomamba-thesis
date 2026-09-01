from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.evaluate_baselines import (
    EvaluationConfig,
    build_naive_persistence_predictions,
    metric_rows,
    sample_count,
    validate_split,
    write_artifacts,
)


class EvaluateBaselinesTest(unittest.TestCase):
    def test_sample_count_matches_released_dataset_len(self) -> None:
        self.assertEqual(sample_count(row_count=365, window_size=14, mode="released"), 350)
        self.assertEqual(sample_count(row_count=365, window_size=14, mode="complete"), 351)

    def test_naive_predictions_use_last_close_as_next_prediction(self) -> None:
        df = self._daily_frame([100.0 + i for i in range(20)])
        config = EvaluationConfig(
            data_dir=Path("/tmp/data"),
            output_dir=Path("/tmp/out"),
            window_size=14,
            horizon_days=1,
            training_config=None,
            source_sample_count_mode="released",
        )

        predictions = build_naive_persistence_predictions(df, "test", config)

        self.assertEqual(len(predictions), 5)
        first = predictions.iloc[0]
        self.assertEqual(first["window_start_date"], "2024-01-01")
        self.assertEqual(first["window_end_date"], "2024-01-14")
        self.assertEqual(first["prediction_date"], "2024-01-15")
        self.assertEqual(first["predicted_close"], 113.0)
        self.assertEqual(first["target_close"], 114.0)

    def test_metrics_make_directional_coverage_explicit_for_hold_baseline(self) -> None:
        df = self._daily_frame([100.0 + i for i in range(20)])
        config = EvaluationConfig(
            data_dir=Path("/tmp/data"),
            output_dir=Path("/tmp/out"),
            window_size=14,
            horizon_days=1,
            training_config=None,
            source_sample_count_mode="released",
        )
        predictions = build_naive_persistence_predictions(df, "test", config)

        metrics = metric_rows(predictions, config)

        row = metrics.iloc[0]
        self.assertEqual(row["samples"], 5)
        self.assertEqual(row["RMSE"], 1.0)
        self.assertEqual(row["MAE"], 1.0)
        self.assertEqual(row["directional_coverage_pct"], 0.0)
        self.assertTrue(pd.isna(row["directional_accuracy_when_predicting_pct"]))

    def test_validate_split_and_write_artifacts(self) -> None:
        df = self._daily_frame([100.0 + i for i in range(20)])
        quality = validate_split(df, "test", window_size=14, mode="released", step_seconds=86400)
        self.assertEqual(quality.samples, 5)
        self.assertEqual(quality.step_violations, 0)

        with tempfile.TemporaryDirectory() as tmp:
            config = EvaluationConfig(
                data_dir=Path(tmp) / "data",
                output_dir=Path(tmp) / "out",
                window_size=14,
                horizon_days=1,
                training_config=None,
                source_sample_count_mode="released",
            )
            predictions = build_naive_persistence_predictions(df, "test", config)
            metrics = metric_rows(predictions, config)
            write_artifacts(predictions, metrics, [quality], config)

            self.assertTrue((config.output_dir / "baseline_metrics.csv").exists())
            self.assertTrue((config.output_dir / "baseline_predictions.csv").exists())
            self.assertTrue((config.output_dir / "data_quality.csv").exists())
            self.assertTrue((config.output_dir / "baseline_metadata.json").exists())

    @staticmethod
    def _daily_frame(closes: list[float]) -> pd.DataFrame:
        dates = pd.date_range("2024-01-01", periods=len(closes), freq="D")
        timestamps = (dates.view("int64") // 1_000_000_000).astype(int)
        return pd.DataFrame(
            {
                "Timestamp": timestamps,
                "Open": closes,
                "High": [value + 1 for value in closes],
                "Low": [value - 1 for value in closes],
                "Close": closes,
                "Volume": [1_000_000.0 for _ in closes],
                "date": dates.strftime("%Y-%m-%d"),
            }
        )


if __name__ == "__main__":
    unittest.main()
