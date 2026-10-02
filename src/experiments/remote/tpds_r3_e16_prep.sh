#!/usr/bin/env bash
# tpds-r3 E16/E16b prep on the RTX PRO 6000 (CPU only; safe in no-GPU mode).
#   1. Python environment (torch, triton, numpy, scipy, pynvml) and the captured traces (input "real").
#   2. Nsight Compute: binary, version, and the E16b metrics listed for the GB202 chip.
#   3. Runtime-chunk reference scan for sm_120 (tpds_r2_prep.sh with TORCH_CUDA_ARCH_LIST=12.0),
#      so that E16 also times the reference kernel. Set TPDS_E16_REF=0 to skip.
#   4. bash -n on the lane, py_compile on the E16 scripts, CPU test of the counter parser.
#
#   wsl bash src/revloop/remote.sh exec pro6000 src/experiments/remote/tpds_r3_e16_prep.sh
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]:-}")/tpds_common.sh" 2>/dev/null || source src/experiments/remote/tpds_common.sh
rc=0
lanelog "tpds-r3 E16 prep on $(hostname) with $PY"
check_space . 10 || rc=1
nvidia-smi -L 2>/dev/null || echo "no GPU visible (expected in no-GPU mode)"

echo "== python environment"
"$PY" - <<'PYEOF' || rc=1
import importlib.metadata as md
import torch, triton, numpy, scipy
print("torch", torch.__version__, "cuda", torch.version.cuda, "triton", triton.__version__,
      "numpy", numpy.__version__, "scipy", scipy.__version__)
try:
    import pynvml  # noqa: F401
    print("pynvml ok")
except ImportError:
    print("pynvml missing (gpu_state falls back to nvidia-smi)")
PYEOF

echo "== traces"
if [ -d "$TRACE_DIR" ] && [ -n "$(find "$TRACE_DIR" -name '*mamba1*' -print -quit 2>/dev/null)" ]; then
  echo "traces: $(find "$TRACE_DIR" -type f | wc -l) files, $(du -sh "$TRACE_DIR" | cut -f1)"
else
  echo "Mamba-1 traces missing at $TRACE_DIR: input 'real' will be skipped (others unaffected)"
fi

echo "== Nsight Compute"
NCU=$("$PY" -c "from src.experiments.tpds.e16_counters import find_ncu; print(find_ncu() or '')")
if [ -z "$NCU" ]; then
  echo "ncu not found (PATH, CUDA_HOME/bin, /opt/nvidia/nsight-compute, /usr/local/cuda*/bin)"; rc=1
else
  "$NCU" --version | tail -1
  "$NCU" --list-chips 2>/dev/null | tr ',' '\n' | grep -i -E "gb20|gb10" | tr '\n' ' '; echo
  "$PY" - "$NCU" <<'PYEOF' || rc=1
import sys
from src.experiments.tpds.e16_counters import METRICS, available_metrics
for chip in ("gb202", None):
    keep, drop = available_metrics(sys.argv[1], METRICS, chip)
    print(f"chip {chip or 'device'}: {len(keep)}/{len(METRICS)} metrics listed; not listed: {drop}")
    if drop != list(METRICS):
        break
PYEOF
fi

if [ "${TPDS_E16_REF:-1}" = 1 ]; then
  echo "== runtime-chunk reference scan for sm_120"
  TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.0} bash src/experiments/remote/tpds_r2_prep.sh \
    || echo "runtime-chunk build failed: E16 runs the Triton kernel only"
fi

echo "== static checks"
bash -n src/experiments/remote/tpds_lane_r3_e16.sh || rc=1
"$PY" -m py_compile src/experiments/tpds/e16_counters.py src/experiments/tpds/e9_placement.py || rc=1
"$PY" - <<'PYEOF' || rc=1
import tempfile
from pathlib import Path
from src.experiments.tpds.e16_counters import METRICS, join_rows, parse_ncu_csv
hdr = ["ID", "Kernel Name"] + list(METRICS)
lines = ['==PROF== x', ",".join(f'"{h}"' for h in hdr), ",".join('""' for _ in hdr)]
lines += [",".join(f'"{v}"' for v in [i, "_mamba1_chunked_scan_kernel"] + ["1,000.5"] * len(METRICS)) for i in range(4)]
p = Path(tempfile.mkdtemp()) / "t.csv"
p.write_text("\n".join(lines) + "\n")
rows = parse_ncu_csv(p)
cells = [{"seq": s, "design": "A", "L": 512, "batch": 1, "chunk": 64, "input": "g", "mode": "act", "round": 0,
          "launches": 2} for s in range(2)]
out, err = join_rows(cells, rows)
assert len(rows) == 4 and rows[0][METRICS[0]] == 1000.5 and err is None and len(out) == 4, (rows, err)
print("counter parser ok")
PYEOF
lanelog "tpds-r3 E16 prep done (rc=$rc)"
exit $rc
