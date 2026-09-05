#!/usr/bin/env bash
# Launch the CryptoMamba Console (FastAPI + built React frontend) on http://127.0.0.1:8600
set -euo pipefail
cd "$(dirname "$0")/.."
BACKEND=backend
PY="${BACKEND}/.venv/bin/python"
[ -x "$PY" ] || { echo "Create the backend venv first:"; echo "  python3 -m venv ${BACKEND}/.venv"; echo "  ${BACKEND}/.venv/bin/python -m pip install --upgrade pip"; echo "  ${BACKEND}/.venv/bin/pip install -e ${BACKEND}"; exit 1; }
exec env PYTHONPATH="${BACKEND}/src" "$PY" -m uvicorn console_api.main:app --host 127.0.0.1 --port 8600 "$@"
