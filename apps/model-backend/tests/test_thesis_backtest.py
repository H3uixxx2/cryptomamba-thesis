from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from thesis_pipeline.backtest import (
    SelfFinancingConfig,
    run_corrected_matrix,
    simulate_self_financing,
)


def prediction_frame(
    *,
    currents: list[float] | None = None,
    targets: list[float] | None = None,
    predicted: list[float] | None = None,
    model_id: str = "cmamba_v_reproduced",
    split: str = "test",
) -> pd.DataFrame:
    current_values = currents or [100.0, 110.0, 105.0]
    target_values = targets or [110.0, 105.0, 120.0]
    predicted_values = predicted or [102.0, 105.0, 110.0]
    if not (len(current_values) == len(target_values) == len(predicted_values)):
        raise ValueError("fixture lengths must match")
    signal_dates = pd.date_range("2024-01-01", periods=len(current_values), freq="D")
    prediction_dates = signal_dates + pd.Timedelta(days=1)
    return pd.DataFrame(
        {
            "model_id": model_id,
            "split": split,
            "signal_date": signal_dates.strftime("%Y-%m-%d"),
            "prediction_date": prediction_dates.strftime("%Y-%m-%d"),
            "current_close": current_values,
            "target_close": target_values,
            "predicted_close": predicted_values,
        }
    )


def test_all_in_long_includes_fee_in_position_size() -> None:
    frame = prediction_frame(currents=[100.0], targets=[110.0], predicted=[102.0])

    result = simulate_self_financing(
        frame,
        "buy_hold",
        SelfFinancingConfig(transaction_cost_pct=0.2),
    )

    row = result.equity_curve.iloc[0]
    assert row["cash"] >= 0.0
    assert row["position_btc"] == pytest.approx(100.0 / (100.0 * 1.002))
    assert row["transaction_cost"] == pytest.approx(row["trade_value"] * 0.002)


def test_actions_do_not_change_when_only_realised_targets_change() -> None:
    base = prediction_frame(
        currents=[100.0, 100.0, 100.0],
        targets=[101.0, 99.0, 103.0],
        predicted=[101.0, 98.0, 103.0],
    )
    changed = base.assign(target_close=[99.0, 102.0, 97.0])

    left = simulate_self_financing(base, "smart_w_short", SelfFinancingConfig())
    right = simulate_self_financing(changed, "smart_w_short", SelfFinancingConfig())

    assert left.equity_curve["target_weight"].tolist() == right.equity_curve[
        "target_weight"
    ].tolist()


def test_terminal_equity_is_the_final_curve_value() -> None:
    result = simulate_self_financing(
        prediction_frame(), "smart", SelfFinancingConfig()
    )

    assert result.final_equity == pytest.approx(
        result.equity_curve.iloc[-1]["portfolio_value"]
    )
    assert result.reconciliation_error <= 1e-9


def test_zero_signal_path_has_no_trades_and_preserves_cash() -> None:
    frame = prediction_frame(
        currents=[100.0, 101.0, 99.0],
        targets=[101.0, 99.0, 102.0],
        predicted=[100.0, 101.0, 99.0],
    )

    result = simulate_self_financing(frame, "smart", SelfFinancingConfig())

    assert result.number_of_trades == 0
    assert result.turnover_notional == 0.0
    assert result.total_fees == 0.0
    assert result.final_equity == 100.0
    assert result.equity_curve["action"].eq("HOLD").all()


def test_short_is_capped_and_charged_borrow_cost() -> None:
    frame = prediction_frame(currents=[100.0], targets=[90.0], predicted=[80.0])
    no_borrow = simulate_self_financing(
        frame,
        "smart_w_short",
        SelfFinancingConfig(transaction_cost_pct=0.2, borrow_cost_bps_per_day=0.0),
    )
    with_borrow = simulate_self_financing(
        frame,
        "smart_w_short",
        SelfFinancingConfig(transaction_cost_pct=0.2, borrow_cost_bps_per_day=10.0),
    )

    row = with_borrow.equity_curve.iloc[0]
    assert row["position_btc"] < 0.0
    assert row["gross_exposure_at_execution"] <= 1.0 + 1e-9
    assert row["short_collateral"] == pytest.approx(
        abs(row["position_btc"]) * row["valuation_price"]
    )
    assert with_borrow.total_borrow_cost > 0.0
    assert with_borrow.final_equity < no_borrow.final_equity


def test_buy_hold_final_equity_is_monotone_in_cost() -> None:
    values = [
        simulate_self_financing(
            prediction_frame(),
            "buy_hold",
            SelfFinancingConfig(transaction_cost_pct=cost),
        ).final_equity
        for cost in (0.0, 0.1, 0.2)
    ]

    assert values[0] >= values[1] >= values[2]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.assign(current_close=np.nan), "finite"),
        (lambda frame: frame.assign(target_close=-1.0), "positive"),
        (lambda frame: pd.concat([frame, frame], ignore_index=True), "duplicate"),
        (lambda frame: frame.iloc[::-1].reset_index(drop=True), "sorted"),
        (lambda frame: frame.assign(signal_date=frame["prediction_date"]), "strictly before"),
    ],
)
def test_invalid_prediction_events_fail_closed(mutation, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        simulate_self_financing(
            mutation(prediction_frame()), "smart", SelfFinancingConfig()
        )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (SelfFinancingConfig(initial_equity=0.0), "initial_equity"),
        (SelfFinancingConfig(transaction_cost_pct=-0.1), "transaction_cost_pct"),
        (SelfFinancingConfig(borrow_cost_bps_per_day=-1.0), "borrow"),
        (SelfFinancingConfig(risk_pct=0.0), "risk_pct"),
        (SelfFinancingConfig(gross_exposure_cap=1.1), "gross_exposure_cap"),
        (SelfFinancingConfig(max_short_exposure=1.1), "max_short_exposure"),
    ],
)
def test_invalid_configs_fail_closed(
    config: SelfFinancingConfig, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        simulate_self_financing(prediction_frame(), "smart", config)


def test_unknown_strategy_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown strategy"):
        simulate_self_financing(
            prediction_frame(), "oracle", SelfFinancingConfig()
        )


def test_matrix_keeps_models_splits_strategies_and_costs_distinct() -> None:
    frames = [
        prediction_frame(model_id=model_id, split=split)
        for model_id in (
            "cmamba_v_reproduced",
            "s5_full",
            "naive_persistence",
        )
        for split in ("val", "test")
    ]
    predictions = pd.concat(frames, ignore_index=True)
    configs = [
        replace(SelfFinancingConfig(), transaction_cost_pct=cost)
        for cost in (0.0, 0.2)
    ]

    result = run_corrected_matrix(predictions, configs=configs)

    metrics = result["metrics"]
    assert len(metrics) == 3 * 2 * 4 * 2
    assert not metrics.duplicated(
        ["model_id", "split", "strategy", "transaction_cost_pct"]
    ).any()
    assert set(result) == {"metrics", "equity", "metadata"}
    assert result["metadata"]["engine"] == "self_financing_same_close_v1"
    assert metrics["reconciliation_error"].le(1e-9).all()
