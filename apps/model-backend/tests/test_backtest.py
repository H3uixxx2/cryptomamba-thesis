from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from scripts.run_backtest import (
    BacktestConfig,
    classify_regimes,
    run_full_backtest,
    simulate,
    write_artifacts,
)

EVAL_DIR = Path(__file__).resolve().parents[1] / "output" / "evaluation"
FORECAST_PREDICTIONS = EVAL_DIR / "forecast_predictions.csv"
TRADING_REPLAY = EVAL_DIR / "trading_replay_metrics.csv"


def _load_predictions(result_type: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(FORECAST_PREDICTIONS)
    sub = df[(df["result_type"] == result_type) & (df["split"] == split)].copy()
    return sub.sort_values("prediction_date").reset_index(drop=True)


def _paper_replay_row(result_type: str, split: str, trade_mode: str):
    df = pd.read_csv(TRADING_REPLAY)
    row = df[
        (df["result_type"] == result_type)
        & (df["split"] == split)
        & (df["trade_mode"] == trade_mode)
    ]
    return row.iloc[0]


def _frame(currents, targets, preds, start: str = "2024-01-01") -> pd.DataFrame:
    """Build a minimal chronological prediction frame for the backtest engine."""
    dates = pd.date_range(start, periods=len(currents), freq="D").strftime("%Y-%m-%d")
    return pd.DataFrame(
        {
            "prediction_date": dates,
            "current_close": [float(x) for x in currents],
            "target_close": [float(x) for x in targets],
            "predicted_close": [float(x) for x in preds],
        }
    )


class BuyHoldTest(unittest.TestCase):
    def test_buy_hold_is_deterministic_and_ignores_predictions(self) -> None:
        # Buy everything on the first day, hold, value the open position at the last target.
        # Predictions are garbage on purpose: buy & hold must not look at them.
        df = _frame(currents=[100, 110, 120], targets=[110, 120, 130], preds=[999, 1, 500])
        res = simulate(
            df,
            BacktestConfig(strategy="buy_hold", transaction_cost_pct=0.0, initial_balance=100.0),
        )
        # 1 BTC bought at 100 -> held -> valued at last target 130.
        self.assertAlmostEqual(res.final_balance, 130.0, places=6)
        self.assertEqual(res.number_of_trades, 1)


class MetricsTest(unittest.TestCase):
    def test_roi_pct_is_relative_gain_over_initial_balance(self) -> None:
        df = _frame(currents=[100, 110, 120], targets=[110, 120, 130], preds=[999, 1, 500])
        res = simulate(df, BacktestConfig("buy_hold", 0.0, initial_balance=100.0))
        # final 130 on 100 initial -> +30%
        self.assertAlmostEqual(res.roi_pct, 30.0, places=6)

    def test_max_drawdown_uses_running_peak_including_initial(self) -> None:
        # buy & hold 1 BTC at 100, price peaks at 120 then falls to 90.
        df = _frame(currents=[100, 120, 90], targets=[120, 90, 90], preds=[1, 1, 1])
        res = simulate(df, BacktestConfig("buy_hold", 0.0, initial_balance=100.0))
        # trough 90 vs peak 120 -> 25% drawdown
        self.assertAlmostEqual(res.max_drawdown_pct, 25.0, places=6)

    def test_sharpe_from_daily_portfolio_returns_population_std_annualised(self) -> None:
        df = _frame(currents=[100, 120, 90], targets=[120, 90, 90], preds=[1, 1, 1])
        res = simulate(df, BacktestConfig("buy_hold", 0.0, initial_balance=100.0))
        # portfolio rows [100,120,90] -> returns [0.2,-0.25]; mean -0.025, pop std 0.225
        # daily sharpe -0.111111 * sqrt(365) = -2.12277
        self.assertAlmostEqual(res.sharpe, -2.12277, delta=0.01)

    def test_win_rate_counts_directionally_correct_trades(self) -> None:
        # day0 BUY then price falls (loss); day1 SELL then price falls (win); day2 HOLD.
        df = _frame(currents=[100, 100, 200], targets=[80, 80, 200], preds=[200, 50, 200])
        res = simulate(df, BacktestConfig("vanilla", 0.0, initial_balance=100.0))
        self.assertEqual(res.number_of_trades, 2)
        self.assertAlmostEqual(res.win_rate_pct, 50.0, places=6)


class InvariantsTest(unittest.TestCase):
    def test_transaction_cost_charged_exactly_once_per_trade(self) -> None:
        # day0 buys 1 BTC at 100 (notional 100), day1 holds.
        df = _frame(currents=[100, 50], targets=[50, 50], preds=[200, 50])
        free = simulate(df, BacktestConfig("vanilla", 0.0))
        costed = simulate(df, BacktestConfig("vanilla", 0.2))
        self.assertEqual(costed.number_of_trades, 1)
        # 0.2% of notional 100 = 0.2, charged once (not 0.4).
        self.assertAlmostEqual(costed.equity_curve["transaction_cost"].sum(), 0.2, places=9)
        self.assertAlmostEqual(free.final_balance - costed.final_balance, 0.2, places=9)

    def test_no_look_ahead_decisions_independent_of_realised_target(self) -> None:
        currents = [100, 110, 90, 120]
        preds = [130, 90, 130, 110]
        a = _frame(currents, targets=[110, 90, 120, 100], preds=preds)
        b = _frame(currents, targets=[999, 1, 500, 2], preds=preds)  # different "future"
        ra = simulate(a, BacktestConfig("smart", 0.1))
        rb = simulate(b, BacktestConfig("smart", 0.1))
        # Decisions (actions + position path + trade count) must be identical:
        # only the realised target changed, and it must not influence any decision.
        self.assertEqual(list(ra.equity_curve["action"]), list(rb.equity_curve["action"]))
        self.assertEqual(
            list(ra.equity_curve["position_btc"]), list(rb.equity_curve["position_btc"])
        )
        self.assertEqual(ra.number_of_trades, rb.number_of_trades)

    def test_final_balance_reconciles_with_equity_curve_terminal_state(self) -> None:
        df = _frame(currents=[100, 120, 90], targets=[120, 90, 95], preds=[200, 50, 200])
        res = simulate(df, BacktestConfig("smart", 0.1))
        last = res.equity_curve.iloc[-1]
        expected = last["cash"] + last["position_btc"] * float(df["target_close"].iloc[-1])
        self.assertAlmostEqual(res.final_balance, expected, places=9)


class RegimeTest(unittest.TestCase):
    def test_trailing_window_thresholds_label_bull_bear_sideways(self) -> None:
        closes = [100, 100, 110, 130, 130, 90]
        # window 2, +/-10%: t-by-t trailing change vs close[max(0,t-2)]
        labels = classify_regimes(closes, window=2, up_pct=10.0, down_pct=10.0)
        self.assertEqual(
            labels, ["sideways", "sideways", "sideways", "bull", "bull", "bear"]
        )


class EdgeCaseTest(unittest.TestCase):
    def test_empty_predictions_returns_initial_balance_no_trades(self) -> None:
        df = _frame(currents=[], targets=[], preds=[])
        res = simulate(df, BacktestConfig("smart", 0.1, initial_balance=100.0))
        self.assertEqual(res.number_of_trades, 0)
        self.assertAlmostEqual(res.final_balance, 100.0, places=9)
        self.assertEqual(len(res.equity_curve), 0)

    def test_all_hold_series_makes_no_trades_and_keeps_balance(self) -> None:
        # vanilla holds when |pred-today|/today < threshold (here pred == today).
        df = _frame(currents=[100, 100, 100], targets=[100, 100, 100], preds=[100, 100, 100])
        res = simulate(df, BacktestConfig("vanilla", 0.2, initial_balance=100.0))
        self.assertEqual(res.number_of_trades, 0)
        self.assertAlmostEqual(res.final_balance, 100.0, places=9)
        self.assertAlmostEqual(res.equity_curve["transaction_cost"].sum(), 0.0, places=9)


class ZeroCostReconciliationTest(unittest.TestCase):
    """At 0% cost the engine must reproduce the paper utils.trade.trade() replay
    (output/evaluation/trading_replay_metrics.csv) it shares predictions with."""

    def _check(self, result_type: str, split: str, strategy: str, trade_mode: str) -> None:
        preds = _load_predictions(result_type, split)
        res = simulate(
            preds,
            BacktestConfig(
                strategy=strategy,
                transaction_cost_pct=0.0,
                risk_pct=2.0,
                initial_balance=100.0,
            ),
        )
        paper = _paper_replay_row(result_type, split, trade_mode)
        # Both final balance AND max drawdown must reproduce the paper replay at 0% cost.
        self.assertAlmostEqual(res.final_balance, float(paper["final_balance"]), delta=0.05)
        self.assertAlmostEqual(res.max_drawdown_pct, float(paper["max_drawdown_pct"]), delta=0.05)

    def test_official_test_vanilla_matches_paper_replay(self) -> None:
        self._check("official_checkpoint", "test", "vanilla", "vanilla")

    def test_official_test_smart_matches_paper_replay(self) -> None:
        self._check("official_checkpoint", "test", "smart", "smart")

    def test_official_test_smart_w_short_matches_paper_replay(self) -> None:
        self._check("official_checkpoint", "test", "smart_w_short", "smart_w_short")

    def test_retrained_val_smart_matches_paper_replay(self) -> None:
        self._check("retrained_checkpoint", "val", "smart", "smart")


class ArtifactMatrixTest(unittest.TestCase):
    def _small_matrix(self):
        preds = pd.read_csv(FORECAST_PREDICTIONS)
        return run_full_backtest(
            preds,
            result_types=["official_checkpoint"],
            splits=["test"],
            strategies=["vanilla", "smart"],
            costs=[0.0, 0.2],
            risk_pct=2.0,
            initial_balance=100.0,
        )

    def test_metrics_rows_unique_per_checkpoint_split_strategy_cost(self) -> None:
        m = self._small_matrix()["metrics"]
        self.assertEqual(len(m), 2 * 2)  # 2 strategies x 2 costs
        key = ["result_type", "checkpoint", "split", "strategy", "transaction_cost_pct"]
        self.assertEqual(int(m.duplicated(subset=key).sum()), 0)

    def test_cost_scenarios_do_not_overwrite_each_other(self) -> None:
        m = self._small_matrix()["metrics"]
        v0 = m[(m.strategy == "vanilla") & (m.transaction_cost_pct == 0.0)]["final_balance"].iloc[0]
        v2 = m[(m.strategy == "vanilla") & (m.transaction_cost_pct == 0.2)]["final_balance"].iloc[0]
        self.assertGreater(v0, v2)  # higher cost -> lower final balance

    def test_metrics_has_required_columns(self) -> None:
        m = self._small_matrix()["metrics"]
        required = {
            "result_type", "model", "checkpoint", "split", "strategy",
            "transaction_cost_pct", "initial_balance", "final_balance", "ROI_pct",
            "max_drawdown_pct", "Sharpe", "number_of_trades", "win_rate_pct",
            "risk_pct", "protocol", "status",
        }
        self.assertTrue(required.issubset(set(m.columns)), required - set(m.columns))

    def test_equity_and_regime_required_columns(self) -> None:
        out = self._small_matrix()
        eq_required = {
            "result_type", "checkpoint", "split", "strategy", "transaction_cost_pct",
            "date", "cash", "position_btc", "portfolio_value", "drawdown_pct",
            "action", "trade_value", "transaction_cost",
        }
        self.assertTrue(eq_required.issubset(set(out["equity"].columns)))
        rg_required = {
            "result_type", "checkpoint", "split", "regime", "strategy",
            "transaction_cost_pct", "samples", "RMSE", "MAE", "MAPE_pct",
            "ROI_pct", "max_drawdown_pct",
        }
        self.assertTrue(rg_required.issubset(set(out["regime"].columns)))

    def test_write_artifacts_emits_three_csvs(self) -> None:
        import tempfile

        out = self._small_matrix()
        with tempfile.TemporaryDirectory() as d:
            paths = write_artifacts(Path(d), out)
            for key in ("trading_metrics.csv", "trading_equity_curve.csv", "regime_metrics.csv"):
                self.assertTrue((Path(d) / key).exists(), key)
            self.assertEqual(len(pd.read_csv(paths["trading_metrics.csv"])), 4)


if __name__ == "__main__":
    unittest.main()
