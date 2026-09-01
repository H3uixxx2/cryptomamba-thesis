"""Path resolution + settings for the CryptoMamba Console backend.

The console holds no model/data logic. It:
  - imports the vendored ``cryptomamba_ui`` package (``console_api.vendor.cryptomamba_ui``),
  - reads frozen artifacts from the model-backend app (``apps/model-backend/output``),
  - reads the checksum-verified evidence bundle from the ``evidence/`` submodule,
  - shells out to the model-backend venv for real frozen-checkpoint inference.

All external locations are overridable via environment variables.
"""
from __future__ import annotations

import os
from pathlib import Path

_HERE = Path(__file__).resolve()
PKG_ROOT = _HERE.parents[1]        # .../backend/src/console_api
BACKEND_ROOT = _HERE.parents[3]    # .../console/backend
CONSOLE_ROOT = _HERE.parents[4]    # .../apps/console
MONOREPO_ROOT = _HERE.parents[6]   # repo root

# --- model-backend app: produces the frozen artifacts the loaders read ---
CORE_ROOT = Path(
    os.getenv("CRYPTO_MAMBA_CORE_ROOT", MONOREPO_ROOT / "apps" / "model-backend")
).expanduser().resolve()

# Frozen-checkpoint inference runs in the model-backend venv (PyTorch/Mamba live there,
# not in this backend).
CORE_PYTHON = Path(
    os.getenv("CRYPTO_MAMBA_CORE_PYTHON", CORE_ROOT / ".venv/bin/python")
).expanduser().absolute()
CHECKPOINT_WORKER = CORE_ROOT / "scripts" / "checkpoint_inference.py"
CHECKPOINT_TIMEOUT_SECONDS = 120
MAX_WORKER_ERROR_CHARS = 2_048
MAX_WORKER_STDOUT_CHARS = 65_536

# --- evidence submodule: checksum-verified thesis-final bundle ---
FINAL_EVIDENCE_DIR = Path(
    os.getenv("CRYPTO_MAMBA_FINAL_EVIDENCE", MONOREPO_ROOT / "evidence" / "final")
).expanduser().resolve()

EVALUATION_DIR = CORE_ROOT / "output" / "evaluation"
REPRODUCE_DIR = CORE_ROOT / "output" / "reproduce_colab_train"
REPRODUCE_PROVENANCE_DIR = REPRODUCE_DIR / "provenance"
SELECTED_CHECKPOINT_PATH = REPRODUCE_DIR / "checkpoints" / "cmamba_v_best_colab_train.ckpt"
OFFLINE_PREDICTION_PATH = EVALUATION_DIR / "offline_prediction.json"
AFFINE_PREDICTIONS_PATH = EVALUATION_DIR / "affine_calibration" / "predictions.csv"

# Optional default live API URL (Colab/ngrok). Usually pasted in the UI at demo time.
API_URL_ENV = os.getenv("CRYPTO_MAMBA_API_URL", "").strip()

# Paper split fixture shipped with the console.
SAMPLE_DATA_PATH = CONSOLE_ROOT / "sample_data" / "btc_ohlcv_paper_splits.csv"

# Frontend: the React/Tailwind bundle built from frontend/ into web-dist/.
WEB_DIST_DIR = CONSOLE_ROOT / "web-dist"


def frontend_dir() -> Path:
    """Directory served at ``/`` — the built React app."""
    return WEB_DIST_DIR


def health() -> dict:
    return {
        "status": "ok",
        "core_root": str(CORE_ROOT),
        "core_root_exists": CORE_ROOT.is_dir(),
        "evaluation_dir": str(EVALUATION_DIR),
        "evaluation_dir_exists": EVALUATION_DIR.exists(),
        "core_python_exists": CORE_PYTHON.is_file(),
        "checkpoint_worker_exists": CHECKPOINT_WORKER.is_file(),
        "final_evidence_dir": str(FINAL_EVIDENCE_DIR),
        "final_evidence_exists": FINAL_EVIDENCE_DIR.is_dir(),
        "sample_data_exists": SAMPLE_DATA_PATH.exists(),
        "frontend_dir": str(frontend_dir()),
        "frontend_built": (WEB_DIST_DIR / "index.html").exists(),
    }
