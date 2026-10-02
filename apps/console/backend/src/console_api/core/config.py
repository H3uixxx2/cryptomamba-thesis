"""Paths and settings for the Console backend; every external location is overridable by environment variable."""
from __future__ import annotations

import os
from pathlib import Path

_HERE = Path(__file__).resolve()
CONSOLE_ROOT = _HERE.parents[4]    # .../apps/console
MONOREPO_ROOT = _HERE.parents[6]   # repo root

# model-backend: frozen artifacts and the inference worker
CORE_ROOT = Path(
    os.getenv("CRYPTO_MAMBA_CORE_ROOT", MONOREPO_ROOT / "apps" / "model-backend")
).expanduser().resolve()

# Interpreter that has PyTorch; it runs the checkpoint worker.
CORE_PYTHON = Path(
    os.getenv("CRYPTO_MAMBA_CORE_PYTHON", CORE_ROOT / ".venv/bin/python")
).expanduser().absolute()
CHECKPOINT_WORKER = CORE_ROOT / "scripts" / "checkpoint_inference.py"

# Display names only; the sealed evidence keeps the original display_name.
MODEL_DISPLAY_NAMES = {"s5_full": "CryptoMamba-T"}
CHECKPOINT_TIMEOUT_SECONDS = 120
MAX_WORKER_ERROR_CHARS = 2_048
MAX_WORKER_STDOUT_CHARS = 65_536

# Sealed evidence bundle (read-only)
FINAL_EVIDENCE_DIR = Path(
    os.getenv("CRYPTO_MAMBA_FINAL_EVIDENCE", MONOREPO_ROOT / "evidence")
).expanduser().resolve()

EVALUATION_DIR = CORE_ROOT / "output" / "evaluation"

# Default URL of the optional live API.
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
