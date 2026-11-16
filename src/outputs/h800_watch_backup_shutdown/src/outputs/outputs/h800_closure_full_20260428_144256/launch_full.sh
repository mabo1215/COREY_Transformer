#!/usr/bin/env bash
set -euo pipefail
cd /root/Corey_Transformer
source ./use_system_cache.sh
export PATH=/root/miniconda3/bin:$PATH
export PYTHON_BIN=/root/miniconda3/bin/python
export OUTPUT_BASE="src/outputs/h800_closure_full_20260428_144256"
export MAX_SAMPLES=20
export RUN_MULTIBLOCK_PROBE=1
export RUN_DIVERSITY=1
export RUN_FA3_BASELINE=1
export RUN_MAMBA2_BASELINE=1
bash src/scripts/run_h800_closure_experiments.sh
