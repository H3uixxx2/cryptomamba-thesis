#!/bin/sh
set -eu

(cd /app/evidence && sha256sum -c SHA256SUMS --quiet) || {
  echo "evidence bundle failed checksum verification; refusing to start" >&2
  exit 1
}

exec python -m uvicorn console_api.main:app --host 0.0.0.0 --port "${PORT}"
