#!/usr/bin/env bash
# Build the React console frontend into web-dist/ (served by the FastAPI app).
# Requires pnpm. Run after cloning and after any change under frontend/src.
set -euo pipefail
cd "$(dirname "$0")/../frontend"
pnpm install --frozen-lockfile
pnpm build
echo "Built -> $(cd .. && pwd)/web-dist"
