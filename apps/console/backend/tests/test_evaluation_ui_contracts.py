"""Static guards for the submitted-thesis evidence shown on Evaluation."""
from pathlib import Path
import unittest


WEBAPP_SRC = Path(__file__).resolve().parents[2] / "frontend" / "src"


class TestEvaluationUiContracts(unittest.TestCase):
    def test_rq1_reproduction_is_explicitly_separate_from_controlled_comparison(self) -> None:
        source = (WEBAPP_SRC / "screens" / "ReproduceScreen.tsx").read_text(
            encoding="utf-8"
        )

        self.assertIn("result.reproduction_350d", source)
        self.assertIn("RQ1 reproduction · original", source)
        self.assertIn("data.samples", source)
        self.assertIn("controlledSamples", source)
        self.assertIn("project reproduction criterion", source)
        self.assertIn("gaps.length === 3 && gaps.every(Number.isFinite)", source)
        self.assertNotIn("registered sensitivity", source)
        self.assertNotIn("registered paired", source)

    def test_final_frozen_forecast_robustness_is_visible_and_bounded(self) -> None:
        source = (WEBAPP_SRC / "screens" / "ReproduceScreen.tsx").read_text(
            encoding="utf-8"
        )

        self.assertIn("result.forecast_robustness", source)
        self.assertIn("Frozen-forecast robustness", source)
        self.assertIn('protocol.resamples.toLocaleString("en-US")', source)
        self.assertIn("protocol.block_lengths.join", source)
        self.assertIn("Crossing zero does not prove equivalence", source)
        self.assertIn("not retraining, multi-seed, or future-regime evidence", source)
        self.assertIn("Chronological halves", source)
        self.assertIn("same checkpoints, not independent holdouts", source)
        self.assertNotIn("50,000 frozen-forecast resamples", source)

    def test_api_types_keep_the_350_and_304_date_scopes_distinct(self) -> None:
        source = (WEBAPP_SRC / "lib" / "api.ts").read_text(encoding="utf-8")

        self.assertIn("export interface Reproduction350d", source)
        self.assertIn("export interface ForecastRobustness", source)
        self.assertIn("reproduction_350d: Reproduction350d", source)
        self.assertIn("forecast_robustness: ForecastRobustness", source)

    def test_evaluation_success_language_is_local_and_neutral(self) -> None:
        source = (WEBAPP_SRC / "screens" / "ReproduceScreen.tsx").read_text(
            encoding="utf-8"
        )

        self.assertIn("Within local criterion", source)
        self.assertIn("All intervals cross zero", source)
        self.assertNotIn("Local evidence ready", source)
        self.assertNotIn("Evidence verified", source)
        self.assertNotIn("checksum-checked", source)
        self.assertNotIn('actions={<Pill tone={ready', source)
        self.assertNotIn(">Reproduced<", source)
        self.assertNotIn("Conclusion unchanged", source)
        self.assertIn('label="Reproduced CM-v · test RMSE"', source)
        self.assertNotIn('label="Local CM-v · test RMSE"', source)

    def test_evaluation_is_visual_first_with_zero_referenced_forests(self) -> None:
        source = (WEBAPP_SRC / "screens" / "ReproduceScreen.tsx").read_text(
            encoding="utf-8"
        )

        for figure in (
            "gapToPaperFigure",
            "primaryRmseForestFigure",
            "primaryDirectionForestFigure",
            "sensitivityRmseForestFigure",
            "sensitivityDirectionForestFigure",
            "chronologicalRmseFigure",
            "chronologicalDirectionFigure",
            "paperParameterRmseFigure",
        ):
            self.assertIn(figure, source)
        self.assertGreaterEqual(source.count("zeroReferenceLine()"), 4)
        self.assertIn('row.model_id === "cmamba_v"', source)
        self.assertIn("Parameter count vs paper-reported RMSE", source)
        self.assertIn("Negative ΔRMSE favors S5-Full", source)
        self.assertIn("positive ΔDirection favors CM-v", source)
        self.assertIn("parameter axis uses a logarithmic scale", source)
        self.assertIn("const upper = Math.max(data.criterion.threshold_pct, ...gaps)", source)
        self.assertIn("range: [0, upper]", source)

    def test_exact_tables_are_collapsed_instead_of_default_scroll_surfaces(self) -> None:
        source = (WEBAPP_SRC / "screens" / "ReproduceScreen.tsx").read_text(
            encoding="utf-8"
        )

        self.assertIn("function ExactValues", source)
        self.assertEqual(source.count("View exact values"), 1)
        self.assertGreaterEqual(source.count("<ExactValues>"), 7)
        self.assertNotIn('className="-mx-4 overflow-x-auto"', source)

    def test_rq1_exact_values_omit_the_redundant_paper_reference_row(self) -> None:
        source = (WEBAPP_SRC / "screens" / "ReproduceScreen.tsx").read_text(
            encoding="utf-8"
        )
        reproduction_panel = source[
            source.index("function ReproductionPanel") : source.index("function RobustnessPanel")
        ]

        self.assertNotIn("Paper reference · Table 3", reproduction_panel)
        self.assertIn("Paper-reported reference · CryptoMamba-v paper Table 3", source)

    def test_paired_test_rows_use_human_model_labels(self) -> None:
        source = (WEBAPP_SRC / "screens" / "ReproduceScreen.tsx").read_text(
            encoding="utf-8"
        )

        self.assertIn("prettyModel(row.model_a)", source)
        self.assertIn("prettyModel(row.model_b)", source)
        self.assertIn("prettyModel(row.lower_mse_model)", source)
        self.assertIn("prettyModel(row.lower_mae_model)", source)
        self.assertNotIn("{row.comparison}</td>", source)


if __name__ == "__main__":
    unittest.main()
