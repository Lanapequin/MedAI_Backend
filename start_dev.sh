#!/usr/bin/env bash
# start_dev.sh — launch MedAI backend in development mode (hot-reload)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── macOS: fix Homebrew Python linking against system libexpat / libomp ──
if [[ "$(uname)" == "Darwin" ]]; then
  EXPAT_LIB="$(brew --prefix expat 2>/dev/null)/lib" || EXPAT_LIB=""
  LIBOMP_LIB="$(brew --prefix libomp 2>/dev/null)/lib" || LIBOMP_LIB=""
  if [[ -n "$EXPAT_LIB" || -n "$LIBOMP_LIB" ]]; then
    export DYLD_LIBRARY_PATH="${EXPAT_LIB}:${LIBOMP_LIB}:${DYLD_LIBRARY_PATH:-}"
  fi
fi

# ── Validate prerequisites ───────────────────────────────────────────
if [[ ! -d venv ]]; then
  echo "ERROR: Python virtual environment not found."
  echo "Run the setup steps from README.md first:"
  echo "  python3.11 -m venv venv && venv/bin/pip install -r requirements.txt"
  exit 1
fi

if [[ ! -f models/triage_model.joblib ]]; then
  echo "WARNING: Model not found. Training now (this takes ~5 minutes)..."
  venv/bin/python train_model.py
fi

echo "Starting MedAI Backend on http://localhost:8000"
echo "API docs: http://localhost:8000/docs"
exec venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
