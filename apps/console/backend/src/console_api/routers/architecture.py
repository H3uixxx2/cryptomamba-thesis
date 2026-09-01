"""Evidence-backed architecture endpoint for the approved S5-Full experiment."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from ..core import config
from ..loaders.final_evidence import EvidenceIntegrityError, FinalEvidence, records


router = APIRouter(prefix="/api/architecture", tags=["architecture"])


def _training_metadata(summary: dict[str, Any]) -> dict[str, Any]:
    runs = summary.get("runs")
    run = runs[0] if isinstance(runs, list) and runs and isinstance(runs[0], dict) else {}
    selected = summary.get("selected")
    selected = selected if isinstance(selected, dict) else {}
    # Metric and p-value fields in the legacy summary are deliberately excluded;
    # the controlled CSV and paired-test CSV below are the current authorities.
    return {
        "config": summary.get("config"),
        "commit": summary.get("commit"),
        "branch": summary.get("branch"),
        "gpu": summary.get("gpu"),
        "torch": summary.get("torch"),
        "mamba_ssm": summary.get("mamba_ssm"),
        "seed": selected.get("seed"),
        "best_epoch": selected.get("best_epoch") or run.get("best_epoch"),
        "training_seconds": run.get("secs"),
    }


@router.get("")
def architecture() -> dict:
    try:
        evidence = FinalEvidence(config.FINAL_EVIDENCE_DIR)
        provenance = evidence.read_json("model/checkpoint_provenance.json")
        summary = evidence.read_json("model/s5_full_summary.json")
        metrics = evidence.read_csv("forecast/controlled_forecast_metrics.csv")
        paired = evidence.read_csv("forecast/paired_significance_tests.csv")
        trading = evidence.read_csv("trading/corrected_trading_metrics.csv")
        trading_metadata = evidence.read_json(
            "trading/corrected_trading_metadata.json"
        )
        models = provenance["models"]
        s5 = models["s5_full"]
        cmv = models["cmamba_v_reproduced"]
    except (EvidenceIntegrityError, KeyError, TypeError, ValueError) as exc:
        return {
            "status": "NOT_READY",
            "message": str(exc),
        }

    s5_metrics = metrics[metrics["model_id"].eq("s5_full")].copy()
    s5_paired = paired[
        paired["model_a"].eq("s5_full") | paired["model_b"].eq("s5_full")
    ].copy()
    reference_cost = float(trading_metadata["thesis_reference_cost_pct"])
    s5_trading = trading[
        trading["model_id"].eq("s5_full")
        & trading["transaction_cost_pct"].eq(reference_cost)
    ].copy()

    return {
        "status": "READY",
        "final_evidence_sha256": evidence.package_sha256,
        "model": s5,
        "parameter_comparison": {
            "reproduced_cm_v": int(cmv["parameter_count"]),
            "s5_full": int(s5["parameter_count"]),
            "s5_to_cm_v_pct": round(
                int(s5["parameter_count"]) / int(cmv["parameter_count"]) * 100.0,
                1,
            ),
        },
        "source_execution": {
            "input": "[B, 5, 60]",
            "feature_order": ["Open", "High", "Low", "Close", "Volume"],
            "normalization": "Price / last observed close - 1; log1p(Volume / 1e9)",
            "scan_axis": "60 chronological day tokens",
            "stages": [
                "Per-day linear embedding 5 -> 32",
                "4 causal CMBlock stages",
                "Last-day LayerNorm readout",
                "Relative-return head 32 -> 1",
            ],
        },
        "training": _training_metadata(summary),
        "controlled_metrics": records(s5_metrics),
        "paired_tests": records(s5_paired),
        "corrected_trading": records(s5_trading),
        "reference_cost_pct": reference_cost,
    }
