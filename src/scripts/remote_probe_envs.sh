#!/usr/bin/env bash
set -euo pipefail
MM=/home1/mabo1215/.corey-wsl-tools/bin/micromamba

echo "=== base env Python ==="
MAMBA_ROOT_PREFIX=/home1/mabo1215/.corey-wsl-tools $MM run -n base python --version 2>&1 || true

echo "=== quamba-py310 Python ==="
MAMBA_ROOT_PREFIX=/home1/mabo1215/.adama-micromamba $MM run -n quamba-py310 python --version 2>&1 || true

echo "=== quamba-py310 packages ==="
MAMBA_ROOT_PREFIX=/home1/mabo1215/.adama-micromamba $MM run -n quamba-py310 pip list 2>/dev/null | head -30 || true
