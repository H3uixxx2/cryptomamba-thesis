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
            "output/seed_runs/cmamba_v__seed23/epoch949-val-rmse572.6860.ckpt"
        ),
        checkpoint_sha256=(
            "afb28215d656ea54421efb00e3c60ec9d8a818fff186c194be9a98b55dcecdf0"
        ),
        prediction_mode="price",
        source_commit="672faa9fb7cb4499da17d6a3afbd65a9301f170f",
    ),
    "s5_full": ModelSpec(
        model_id="s5_full",
        display_name="CMamba-T / S5-Full",
        window_days=60,
        parameter_count=57_249,
        checkpoint_path=Path(
            "output/seed_runs/s5_full__seed23/epoch321-val-rmse527.3419.ckpt"
        ),
        checkpoint_sha256=(
            "a2f8b0f39411e199e73ee22a8ae3055d3fc4218fa9d670b31af93d2e190df291"
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

