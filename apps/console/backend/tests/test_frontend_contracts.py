"""Static regression guards for thesis-critical frontend labels and diagrams."""
from pathlib import Path
import unittest


WEBAPP_SRC = Path(__file__).resolve().parents[2] / "frontend" / "src"
WEBAPP_ROOT = WEBAPP_SRC.parent


class TestFrontendContracts(unittest.TestCase):
    def test_document_language_matches_the_english_interface(self) -> None:
        source = (WEBAPP_ROOT / "index.html").read_text(encoding="utf-8")

        self.assertIn('<html lang="en">', source)

    def test_forecast_caption_reports_the_actual_window_length(self) -> None:
        source = (WEBAPP_SRC / "components" / "ForecastChart.tsx").read_text(
            encoding="utf-8"
        )

        self.assertIn("${candles.length} daily Bitcoin candles", source)
        self.assertNotIn("Fourteen daily Bitcoin candles", source)

    def test_data_timeline_keeps_edge_labels_inside_the_visible_range(self) -> None:
        source = (WEBAPP_SRC / "components" / "DataExplorerChart.tsx").read_text(
            encoding="utf-8"
        )

        self.assertIn("const EDGE_PADDING_BARS", source)
        self.assertIn("from: -EDGE_PADDING_BARS", source)
        self.assertIn("to: candles.length - 1 + EDGE_PADDING_BARS", source)
        self.assertIn("minBarSpacing: 0.1", source)
        self.assertNotIn("getVisibleLogicalRange()", source)

    def test_data_timeline_exposes_pointable_ohlcv_values(self) -> None:
        source = (WEBAPP_SRC / "components" / "DataExplorerChart.tsx").read_text(
            encoding="utf-8"
        )

        self.assertIn('aria-label="Inspected OHLCV candle"', source)
        self.assertIn("chart.subscribeCrosshairMove(crosshairHandler)", source)
        self.assertIn("Open", source)
        self.assertIn("High", source)
        self.assertIn("Low", source)
        self.assertIn("Close", source)
        self.assertIn("Volume", source)
        self.assertNotIn("Hover a candle to inspect", source)

    def test_data_window_uses_two_visible_weeks_without_horizontal_scroll(self) -> None:
        source = (WEBAPP_SRC / "screens" / "DataScreen.tsx").read_text(
            encoding="utf-8"
        )
        window_lens = source[source.index("function SlidingWindowLens") : source.index("export function DataScreen")]

        self.assertIn('aria-label="Fourteen-day causal input window"', window_lens)
        self.assertIn("grid-cols-7", window_lens)
        self.assertNotIn("overflow-x-auto", window_lens)
        self.assertNotIn("min-w-[650px]", window_lens)

    def test_s5_diagram_draws_all_sixty_time_tokens(self) -> None:
        source = (WEBAPP_SRC / "components" / "ScanAxisDiagram.tsx").read_text(
            encoding="utf-8"
        )

        self.assertIn("Array.from({ length: 60 })", source)
        self.assertNotIn("Array.from({ length: 30 })", source)

    def test_checkpoint_ready_callout_requires_sufficient_history(self) -> None:
        source = (WEBAPP_SRC / "screens" / "PredictScreen.tsx").read_text(
            encoding="utf-8"
        )

        self.assertIn("const checkpointReady = Boolean(", source)
        self.assertIn(
            '!result && predict.mode === "checkpoint" && checkpointReady', source
        )
        self.assertNotIn(
            '!result && predict.mode === "checkpoint" && !predict.loading && !predict.error',
            source,
        )

    def test_checkpoint_ood_result_is_visibly_labeled_as_non_evidence(self) -> None:
        source = (WEBAPP_SRC / "screens" / "PredictScreen.tsx").read_text(
            encoding="utf-8"
        )

        self.assertIn('predict.mode === "checkpoint" && result.ood', source)
        self.assertIn(
            '<Callout tone="amber" title="Out-of-distribution checkpoint forecast">',
            source,
        )
        self.assertIn("qualitative and extrapolative only", source)
        self.assertIn("not thesis evidence or trading evidence", source)

    def test_checkpoint_flow_uses_source_neutral_dataset_wording(self) -> None:
        source = (WEBAPP_SRC / "screens" / "PredictScreen.tsx").read_text(
            encoding="utf-8"
        )

        self.assertIn("Active dataset → frozen checkpoint", source)
        self.assertNotIn("Uploaded data → frozen checkpoint", source)

    def test_checkpoint_date_input_is_bounded_by_active_dataset(self) -> None:
        source = (WEBAPP_SRC / "screens" / "PredictScreen.tsx").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "const { minimumPredictionDate, maximumPredictionDate }", source
        )
        self.assertIn("min={minimumPredictionDate}", source)
        self.assertIn("max={maximumPredictionDate}", source)
        self.assertIn("predict.predictDate >= minimumPredictionDate", source)
        self.assertIn("predict.predictDate <= maximumPredictionDate", source)

    def test_evaluation_uses_academic_direction_presentation(self) -> None:
        source = (WEBAPP_SRC / "screens" / "ReproduceScreen.tsx").read_text(
            encoding="utf-8"
        )

        self.assertIn("Persistence predicts the current Close", source)
        for stale in (
            "N/A · 0% coverage",
            "Evidence manifest",
            "Artifact traceability",
            "Manifest SHA-256",
        ):
            with self.subTest(stale=stale):
                self.assertNotIn(stale, source)

    def test_checkpoint_prediction_omits_delivery_metadata(self) -> None:
        source = (WEBAPP_SRC / "screens" / "PredictScreen.tsx").read_text(
            encoding="utf-8"
        )

        self.assertIn("{model.window_days}-day window", source)
        for stale in (
            "Artifact traceability",
            "Checkpoint SHA-256",
            "Source commit",
            "model.checkpoint_sha256",
            "result.checkpoint_sha256",
            "result.source_commit",
        ):
            with self.subTest(stale=stale):
                self.assertNotIn(stale, source)

    def test_trading_fee_sensitivity_uses_corrected_test_rows(self) -> None:
        source = (WEBAPP_SRC / "screens" / "TradingScreen.tsx").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "data.corrected_metadata.transaction_cost_pct_scenarios", source
        )
        self.assertIn("const declaredFeeScenarios = Array.from(new Set(", source)
        self.assertIn(
            "const feeScenarios = declaredFeeScenarios.filter(isCompleteFeeScenario)",
            source,
        )
        self.assertNotIn("const FEE_SCENARIOS", source)
        self.assertIn(
            'data.corrected.filter((row) => row.split === "test")', source
        )
        self.assertIn("Fee sensitivity · corrected self-financing · test", source)
        self.assertIn("hasCompleteFeeMatrix", source)
        self.assertIn("Number.isFinite(row.final_equity)", source)
        self.assertIn(
            "buyHoldBeatsEveryForecastStrategy = hasCompleteFeeMatrix &&", source
        )
        self.assertIn("buyHoldBeatsEveryForecastStrategy", source)
        self.assertIn(
            "Buy & Hold finishes above every forecast-driven model-strategy row",
            source,
        )

    def test_trading_preserves_the_three_evidence_surfaces(self) -> None:
        source = (WEBAPP_SRC / "screens" / "TradingScreen.tsx").read_text(
            encoding="utf-8"
        )

        for label in ("Corrected self-financing", "Paper replay", "One-Day What-If"):
            with self.subTest(label=label):
                self.assertIn(label, source)

    def test_trading_fee_sensitivity_defaults_to_accessible_line_charts(self) -> None:
        source = (WEBAPP_SRC / "screens" / "TradingScreen.tsx").read_text(
            encoding="utf-8"
        )

        self.assertIn('import { Plot } from "@/components/Plot"', source)
        self.assertIn('type: "scatter"', source)
        self.assertIn('mode: "lines+markers"', source)
        self.assertIn('modelId: "cmamba_v_reproduced"', source)
        self.assertIn('modelId: "s5_full"', source)
        self.assertIn('name: "Buy & Hold (shared)"', source)
        self.assertIn("ariaLabel=", source)
        self.assertIn("accessibleSummary=", source)

    def test_trading_exact_fee_matrix_is_collapsed_by_default(self) -> None:
        source = (WEBAPP_SRC / "screens" / "TradingScreen.tsx").read_text(
            encoding="utf-8"
        )

        details = source.index("<details")
        summary = source.index("<summary", details)
        label = source.index("View exact values", summary)
        table = source.index("<table", label)
        self.assertLess(details, summary)
        self.assertLess(summary, label)
        self.assertLess(label, table)
        self.assertNotIn("<details open", source)

    def test_trading_reference_cost_metrics_are_collapsed_by_default(self) -> None:
        source = (WEBAPP_SRC / "screens" / "TradingScreen.tsx").read_text(
            encoding="utf-8"
        )

        panel = source.index("Corrected self-financing · test · thesis reference cost")
        details = source.index("<details", panel)
        label = source.index("View reference-cost metrics", details)
        table = source.index("<table", label)
        self.assertLess(panel, details)
        self.assertLess(details, label)
        self.assertLess(label, table)

    def test_trading_hides_hashes_and_uses_human_replay_statuses(self) -> None:
        source = (WEBAPP_SRC / "screens" / "TradingScreen.tsx").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("Manifest SHA-256", source)
        self.assertNotIn("data.final_evidence_sha256", source)
        self.assertNotIn("Reading verified", source)
        self.assertNotIn("checksum-verified", source)
        self.assertIn('"Within local replay criterion"', source)
        self.assertNotIn('"Replay criterion met"', source)
        self.assertNotIn("Exact replay match", source)
        self.assertIn('"Does not match"', source)

    def test_buy_hold_is_presented_as_one_shared_market_benchmark(self) -> None:
        source = (WEBAPP_SRC / "screens" / "TradingScreen.tsx").read_text(
            encoding="utf-8"
        )

        self.assertIn("Shared market benchmark", source)
        self.assertIn("firstBuyHoldIndex", source)
        self.assertIn('row.strategy !== "buy_hold"', source)
        self.assertIn("fmt(buyHold?.initial_equity, 0)", source)
        self.assertNotIn("initial equity 100", source)

    def test_strategy_labels_use_the_exact_thesis_term(self) -> None:
        source = (WEBAPP_SRC / "lib" / "format.ts").read_text(encoding="utf-8")

        self.assertIn('smart_w_short: "Extended Smart"', source)
        self.assertNotIn('smart_w_short: "Smart + Short"', source)

    def test_app_shows_a_concise_research_disclaimer(self) -> None:
        source = (WEBAPP_SRC / "App.tsx").read_text(encoding="utf-8")

        self.assertIn("Local research prototype", source)
        self.assertIn("Not investment advice", source)


if __name__ == "__main__":
    unittest.main()
