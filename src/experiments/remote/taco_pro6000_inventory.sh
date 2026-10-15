#!/usr/bin/env bash
# Inventory of the RTX PRO 6000 host for taco-r1 (no-GPU mode; read-only).
set -uo pipefail
source src/experiments/remote/tpds_common.sh 2>/dev/null
echo "== python: $PY"
"$PY" - <<'PYEOF'
import importlib.metadata as md, inspect
for p in ("torch", "triton", "mamba_ssm", "flash-linear-attention", "fla-core", "transformers", "scipy", "numpy"):
    try:
        print(p, md.version(p))
    except Exception as e:
        print(p, "missing")
try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    print("mamba_chunk_scan_combined", inspect.signature(mamba_chunk_scan_combined))
except Exception as e:
    print("ssd import failed:", e)
try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    print("chunk_gated_delta_rule", inspect.signature(chunk_gated_delta_rule))
except Exception as e:
    print("gdn import failed:", e)
PYEOF
echo "== .tpds_env"; cat .tpds_env 2>/dev/null | sed 's/\(TOKEN\|KEY\)=.*/\1=<redacted>/'
echo "== trace dirs"
for d in src/outputs/tpds_traces* /autodl-fs/data/* /root/autodl-tmp/*traces*; do
  [ -d "$d" ] && echo "$d: $(find "$d" -maxdepth 2 -type f | wc -l) files, $(du -sh "$d" 2>/dev/null | cut -f1)"
done
echo "== HF cache"
ls -d ${HF_HOME:-$HOME/.cache/huggingface}/hub/models--* 2>/dev/null | sed 's#.*/models--##'
ls -d /root/autodl-tmp/hf_models/* 2>/dev/null
echo "== disk"; df -h /root/autodl-tmp . | tail -n +1
echo "== git"; git log --oneline -1; git status --short | head -5
