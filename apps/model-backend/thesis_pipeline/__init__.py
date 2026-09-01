"""Artifact-backed pipeline used by the final CryptoMamba-v thesis."""

from thesis_pipeline.backtest import (
    SelfFinancingConfig,
    SelfFinancingResult,
    run_corrected_matrix,
    simulate_self_financing,
)
from thesis_pipeline.contracts import MODEL_SPECS, ModelSpec
from thesis_pipeline.evaluation import (
    build_aligned_predictions,
    forecast_metrics,
    paired_tests,
)
from thesis_pipeline.inference import (
    Candle,
    CheckpointPredictor,
    PredictionRequest,
    PredictionResult,
    PredictionValidationError,
    parse_request,
    predict_next_close,
    prepare_model_input,
)
from thesis_pipeline.package import PackageReceipt, package_evidence, verify_evidence

__all__ = [
    "MODEL_SPECS",
    "Candle",
    "CheckpointPredictor",
    "ModelSpec",
    "PackageReceipt",
    "PredictionRequest",
    "PredictionResult",
    "PredictionValidationError",
    "SelfFinancingConfig",
    "SelfFinancingResult",
    "build_aligned_predictions",
    "forecast_metrics",
    "paired_tests",
    "parse_request",
    "package_evidence",
    "predict_next_close",
    "prepare_model_input",
    "run_corrected_matrix",
    "simulate_self_financing",
    "verify_evidence",
]
