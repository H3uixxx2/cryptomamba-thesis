"""Evaluation screen business logic.

Everything served here comes from the checksum-verified evidence bundle plus the
350-day reproduction recomputed from the pinned prediction artifact. Nothing is
recomputed from a model and nothing is estimated: a missing, tampered or
inconsistent artifact yields ``status=NOT_READY`` with the reason, never a 500
and never a substituted number.

Only CM-v, S5-Full and naive persistence have per-date local predictions, so only
those three are paired. LSTM / GRU / iTransformer / S-Mamba exist solely as
aggregate rows and are served verbatim under ``paper_reported`` — they carry no
per-date series and therefore cannot enter a paired test.
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
        "controlled_local": final_records(controlled),
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
    """Evaluation screen contract: the frozen thesis evidence for RQ1 and RQ2.

    Everything comes from the checksum-verified evidence bundle plus the 350-day
    reproduction recomputed from the pinned predictions. There is no local
    multi-baseline comparison: the paper's LSTM/GRU/iTransformer/S-Mamba rows are
    reported as published aggregates only (``paper_reported``).
    """
    fields, errors = _final_evaluation_fields()
    status = "READY" if fields["final_evidence_status"] == "READY" and not errors else "NOT_READY"
    return {"status": status, "errors": errors, **fields}
