from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thesis_pipeline.contracts import MODEL_SPECS
from thesis_pipeline.evaluation import (
    build_aligned_predictions,
    forecast_metrics,
    paired_tests,
)


CORE_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def aligned() -> pd.DataFrame:
    return build_aligned_predictions(CORE_ROOT)


def test_registry_contains_only_approved_local_models() -> None:
    assert set(MODEL_SPECS) == {
        "cmamba_v_reproduced",
        "s5_full",
        "naive_persistence",
    }
    assert MODEL_SPECS["cmamba_v_reproduced"].parameter_count == 136_952
    assert MODEL_SPECS["s5_full"].parameter_count == 57_249
    assert MODEL_SPECS["naive_persistence"].checkpoint_path is None


def test_alignment_has_three_models_on_identical_dates(aligned: pd.DataFrame) -> None:
    assert len(aligned) == 304 * 3 * 2
    for split in ("val", "test"):
        part = aligned[aligned["split"] == split]
        assert part.groupby("model_id").size().to_dict() == {
            "cmamba_v_reproduced": 304,
            "naive_persistence": 304,
            "s5_full": 304,
        }
        dates = part.groupby("model_id")["prediction_date"].apply(tuple)
        assert dates.nunique() == 1
        timestamps = part.groupby("model_id")["prediction_timestamp"].apply(tuple)
        assert timestamps.nunique() == 1


def test_alignment_preserves_actual_prices_and_builds_true_persistence(
    aligned: pd.DataFrame,
) -> None:
    for (_, _), group in aligned.groupby(["split", "prediction_date"]):
        assert group["current_close"].max() - group["current_close"].min() <= 1e-3
        assert group["target_close"].max() - group["target_close"].min() <= 1e-3
    naive = aligned[aligned["model_id"] == "naive_persistence"]
    assert np.array_equal(
        naive["predicted_close"].to_numpy(), naive["current_close"].to_numpy()
    )


def test_s5_frozen_rmse_replays_on_both_splits(aligned: pd.DataFrame) -> None:
    metrics = forecast_metrics(aligned).set_index(["model_id", "split"])

    assert metrics.loc[("s5_full", "val"), "RMSE"] == pytest.approx(
        557.3386170525702, abs=1e-9
    )
    assert metrics.loc[("naive_persistence", "val"), "RMSE"] == pytest.approx(
        554.9411029218376, abs=1e-9
    )
    assert metrics.loc[("s5_full", "test"), "RMSE"] == pytest.approx(
        1660.8279957505451, abs=1e-9
    )
    assert metrics.loc[("naive_persistence", "test"), "RMSE"] == pytest.approx(
        1657.5455648651607, abs=1e-9
    )


def test_persistence_direction_accuracy_is_null_without_coverage(
    aligned: pd.DataFrame,
) -> None:
    metrics = forecast_metrics(aligned)
    naive = metrics[metrics["model_id"] == "naive_persistence"]

    assert naive["directional_coverage_pct"].eq(0.0).all()
    assert naive["directional_accuracy_pct"].isna().all()
    assert naive["predicted_up_pct"].eq(0.0).all()


def test_metrics_report_split_dates_samples_parameters_and_hashes(
    aligned: pd.DataFrame,
) -> None:
    metrics = forecast_metrics(aligned)

    assert metrics["samples"].eq(304).all()
    assert metrics["date_from"].notna().all()
    assert metrics["date_to"].notna().all()
    assert metrics["parameter_count"].isin([0, 57_249, 136_952]).all()
    learned = metrics[metrics["model_id"] != "naive_persistence"]
    assert learned["checkpoint_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()


def test_paired_tests_use_squared_dm_and_absolute_wilcoxon(
    aligned: pd.DataFrame,
) -> None:
    results = paired_tests(aligned)

    assert len(results) == 2 * 3
    assert results["samples"].eq(304).all()
    assert results["dm_loss"].eq("squared_error").all()
    assert results["dm_hac_lag"].eq(1).all()
    assert results["wilcoxon_loss"].eq("absolute_error").all()
    assert not results[["split", "model_a", "model_b"]].duplicated().any()

    val = results[
        (results["split"] == "val")
        & (results["model_a"] == "s5_full")
        & (results["model_b"] == "naive_persistence")
    ].iloc[0]
    test = results[
        (results["split"] == "test")
        & (results["model_a"] == "s5_full")
        & (results["model_b"] == "naive_persistence")
    ].iloc[0]
    assert val["wilcoxon_p_value"] == pytest.approx(0.017301007139379713)
    assert test["wilcoxon_p_value"] == pytest.approx(0.9765954360189278)


def test_every_test_pair_has_the_same_ordered_dates(aligned: pd.DataFrame) -> None:
    results = paired_tests(aligned)
    for row in results.itertuples(index=False):
        left = aligned[
            (aligned["split"] == row.split)
            & (aligned["model_id"] == row.model_a)
        ]["prediction_date"].tolist()
        right = aligned[
            (aligned["split"] == row.split)
            & (aligned["model_id"] == row.model_b)
        ]["prediction_date"].tolist()
        assert left == right
