#!/usr/bin/env bash
# Light taco-r1 check for no-GPU mode (little host memory: no torch import, no compilation).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]:-}")/tpds_common.sh" 2>/dev/null || source src/experiments/remote/tpds_common.sh
rc=0
git log --oneline -1
bash -n src/experiments/remote/taco_lane_pro6000.sh || rc=1
"$PY" -m py_compile src/experiments/tpds/e27_varlen.py src/experiments/tpds/packed_steps.py \
    src/experiments/tpds/kernels.py src/experiments/tpds/serving_key_churn.py src/experiments/tpds/e2_oracle_sweep.py || rc=1
ls -la "$OUT_DIR/packed_steps_taco.json" || rc=1
free -g | head -2
check_space "$OUT_DIR" 5 || rc=1
echo "check rc=$rc"
exit $rc
