#!/usr/bin/env bash
# Rebuild the original Cosmos Policy CUDA-12.8 / LIBERO environment from uv.lock.
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it first: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
uv sync --locked --extra cu128 --group libero --python 3.10
echo "Environment created from uv.lock. Run autodl_env/verify_environment.py next."
