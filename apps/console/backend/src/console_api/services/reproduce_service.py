"""Evaluation screen: frozen thesis evidence for RQ1 and RQ2, from the sealed bundle plus the 350-day reproduction recomputed from the pinned predictions.
A missing or inconsistent artifact yields status=NOT_READY. Only CM-v, CryptoMamba-T (``s5_full``) and persistence have per-date predictions and enter paired tests;
the paper's other models are served as ``paper_reported`` aggregates.
"""
from __future__ import annotations

from ..core import config
from ..loaders.final_evidence import (
    EvidenceIntegrityError,
    FinalEvidence,
    records as final_records,
)
from ..loaders.forecast_robustness import get_forecast_robustness, not_ready_robustness
from ..loaders.reproduction_evidence import build_reproduction_350d, not_ready_reproduction


def _final_evaluation_fields() -> tuple[dict, list[str]]:
    try:
        evidence = FinalEvidence(config.FINAL_EVIDENCE_DIR)
        paper = evidence.read_csv("forecast/paper_reported_metrics.csv")
        controlled = evidence.read_csv("forecast/controlled_forecast_metrics.csv")
        paired = evidence.read_csv("forecast/paired_significance_tests.csv")
    except EvidenceIntegrityError as exc:
        error = str(exc)
        return (
            {
                "final_evidence_status": "NOT_READY",
                "final_evidence_error": error,
                "paper_reported": [],
                "controlled_local": [],
                "paired_tests": [],
                "reproduction_350d": not_ready_reproduction(error),
                "forecast_robustness": not_ready_robustness(error),
            },
            [error],
        )

    errors: list[str] = []
    try:
        reproduction = build_reproduction_350d(evidence, config.CORE_ROOT)
    except EvidenceIntegrityError as exc:
        error = f"350-day reproduction: {exc}"
        errors.append(error)
        reproduction = not_ready_reproduction(error)
    try:
        robustness = get_forecast_robustness(evidence)
    except EvidenceIntegrityError as exc:
        error = f"forecast robustness: {exc}"
        errors.append(error)
        robustness = not_ready_robustness(error)

    fields = {
        "final_evidence_status": "READY",
        "final_evidence_error": None,
        "final_evidence_sha256": evidence.package_sha256,
        "paper_reported": final_records(paper),
        "controlled_local": [
            {**row, "display_name": config.MODEL_DISPLAY_NAMES.get(row["model_id"], row["display_name"])}
            for row in final_records(controlled)
        ],
        "paired_tests": final_records(paired),
        "paired_test_definitions": {
            "diebold_mariano": "Paired squared-error differential with HAC lag 1.",
            "wilcoxon": "Paired absolute-error differential.",
            "scope": "Only identical local target dates are paired; paper aggregates are not tested against local rows.",
        },
        "reproduction_350d": reproduction,
        "forecast_robustness": robustness,
    }
    return fields, errors


def build_reproduce() -> dict:
    """Evaluation screen payload for RQ1 and RQ2."""
    fields, errors = _final_evaluation_fields()
    status = "READY" if fields["final_evidence_status"] == "READY" and not errors else "NOT_READY"
    return {"status": status, "errors": errors, **fields}
