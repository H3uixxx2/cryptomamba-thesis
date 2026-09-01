"""Frozen identities and paths for the final thesis evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


ModelId = Literal["cmamba_v_reproduced", "s5_full", "naive_persistence"]
PredictionMode = Literal["price", "relative_return", "persistence"]


@dataclass(frozen=True)
class ModelSpec:
    model_id: ModelId
    display_name: str
    window_days: int
    parameter_count: int
    checkpoint_path: Path | None
    checkpoint_sha256: str | None
    prediction_mode: PredictionMode
    source_commit: str | None


MODEL_SPECS: dict[ModelId, ModelSpec] = {
    "cmamba_v_reproduced": ModelSpec(
        model_id="cmamba_v_reproduced",
        display_name="Reproduced CM-v",
        window_days=14,
        parameter_count=136_952,
        checkpoint_path=Path(
            "output/reproduce_colab_train/checkpoints/cmamba_v_best_colab_train.ckpt"
        ),
        checkpoint_sha256=(
            "ad5ec21bb2582e1f935837f620ed8d1ecb280e5568571783bd8c89ab717cc510"
        ),
        prediction_mode="price",
        source_commit="fef42727861a9c3a32241283ab981ef21a49e291",
    ),
    "s5_full": ModelSpec(
        model_id="s5_full",
        display_name="CMamba-T / S5-Full",
        window_days=60,
        parameter_count=57_249,
        checkpoint_path=Path(
            "output/improve_track_evidence/s5_full/checkpoints/"
            "s5_full__seed23__epoch321-val-rmse523.4394.ckpt"
        ),
        checkpoint_sha256=(
            "9b658f09019426fc723c17fa1eeb5cbd92284f8995935e0cf24d3f760a59ffa5"
        ),
        prediction_mode="relative_return",
        source_commit="672faa9fb7cb4499da17d6a3afbd65a9301f170f",
    ),
    "naive_persistence": ModelSpec(
        model_id="naive_persistence",
        display_name="Naive persistence",
        window_days=1,
        parameter_count=0,
        checkpoint_path=None,
        checkpoint_sha256=None,
        prediction_mode="persistence",
        source_commit=None,
    ),
}

