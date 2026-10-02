"""Training, model and data-split configuration files, served verbatim through an allow-list.
A missing, oversized or non-UTF-8 file makes the payload NOT_READY.
"""
from __future__ import annotations

from ..core import config

MAX_CONFIG_BYTES = 16_384

# (id, group, title, path relative to the model backend, what it defines)
CONFIG_FILES: tuple[tuple[str, str, str, str, str], ...] = (
    ("data_split", "Data", "Chronological split", "configs/data_configs/mode_1.yaml",
     "Train, validation and test intervals; half-open, so each ends the day before the next begins."),
    ("cmv_training", "CryptoMamba-v", "Training", "configs/training/cmamba_v.yaml",
     "Optimiser, learning rate and schedule, weight decay, epochs, and which model it trains."),
    ("cmv_model", "CryptoMamba-v", "Model", "configs/models/CryptoMamba/v2.yaml",
     "Architecture: block widths (the first is the 14-day window), depth per stage, state size."),
    ("cmt_training", "CryptoMamba-T", "Training", "configs/training/s5_full.yaml",
     "Same schedule at learning rate 0.001, return output, and the three input-normalisation flags."),
    ("cmt_model", "CryptoMamba-T", "Model", "configs/models/CryptoMamba/t2.yaml",
     "Architecture: window is its own parameter, 4 blocks of width 32, state size 16."),
    ("registry", "Registry", "Model registry", "configs/models/archs.yaml",
     "Maps the model name a training file uses to the model file that builds it."),
)

# Display names for internal run ids.
LABELS: tuple[tuple[str, str], ...] = (
    ("CMamba_v2", "CryptoMamba-v (model configuration)"),
    ("cmamba_v", "CryptoMamba-v (training configuration)"),
    ("CMambaT_w60", "CryptoMamba-T (model configuration, 60-day window)"),
    ("s5_full", "CryptoMamba-T (training configuration and evidence id)"),
)

# Set on the command line of scripts/training.py, not in the files above.
OUTSIDE_FILES: tuple[str, ...] = (
    "Batch size 32 for every run.",
    "Seeds 23, 24 and 25.",
    "The checkpoint kept is the epoch with the lowest validation RMSE.",
)


def build_config() -> dict:
    files = []
    for file_id, group, title, relative, purpose in CONFIG_FILES:
        path = (config.CORE_ROOT / relative).resolve()
        try:
            path.relative_to(config.CORE_ROOT)
        except ValueError:
            return {"status": "NOT_READY", "message": f"path escapes the model backend: {relative}"}
        if path.is_symlink() or not path.is_file():
            return {"status": "NOT_READY", "message": f"configuration file is missing: {relative}"}
        if path.stat().st_size > MAX_CONFIG_BYTES:
            return {"status": "NOT_READY", "message": f"configuration file is too large: {relative}"}
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return {"status": "NOT_READY", "message": f"configuration file is unreadable: {relative}"}
        files.append({
            "id": file_id, "group": group, "title": title,
            "path": f"apps/model-backend/{relative}", "purpose": purpose, "text": text,
        })
    return {
        "status": "READY",
        "files": files,
        "labels": [{"internal": a, "name": b} for a, b in LABELS],
        "outside_files": list(OUTSIDE_FILES),
    }
