#!/usr/bin/env bash
# E14 prep (CPU only): download the Mamba-1 1.4B and 2.8B checkpoints with aria2 from the
# Hugging Face mirror into $TPDS_E14_MODEL_ROOT (default /root/autodl-tmp/hf_models), resumably.
# Writes "ARIA_EXIT <code>" at the end of src/logs/tpds/e14_download.log, which the E14 step of
# tpds_lane_r3.sh waits for. Safe in no-GPU mode.
#
#   wsl bash src/revloop/remote.sh launch vgpu3090 tpds-r3-dl src/experiments/remote/tpds_r3_e14_download.sh
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/tpds_common.sh" 2>/dev/null || source src/experiments/remote/tpds_common.sh
ROOT_DIR=${TPDS_E14_MODEL_ROOT:-/root/autodl-tmp/hf_models}
DL_LOG=$LOG_DIR/e14_download.log
MODELS=${TPDS_E14_MODELS:-state-spaces/mamba-1.4b-hf state-spaces/mamba-2.8b-hf}
EP=${HF_ENDPOINT:-https://hf-mirror.com}
mkdir -p "$ROOT_DIR"
check_space "$ROOT_DIR" 40 || { echo "ARIA_EXIT 90" >> "$DL_LOG"; exit 1; }
lanelog "E14 download start: $MODELS -> $ROOT_DIR via $EP"
INPUT=$(mktemp)
for mid in $MODELS; do
  "$PY" - "$mid" "$ROOT_DIR/${mid##*/}" "$EP" >> "$INPUT" <<'PYEOF' || { echo "ARIA_EXIT 91" >> "$DL_LOG"; exit 1; }
import sys
from huggingface_hub import HfApi
mid, out, ep = sys.argv[1:4]
files = [f for f in HfApi(endpoint=ep).list_repo_files(mid)
         if f.endswith((".json", ".safetensors", ".txt", ".model")) and not f.startswith(".")]
for f in files:
    print(f"{ep}/{mid}/resolve/main/{f}\n  dir={out}\n  out={f}")
PYEOF
done
grep -c "resolve/main" "$INPUT" | xargs -I{} lanelog "E14 download: {} files"
aria2c -c -x 8 -s 8 -j 4 --auto-file-renaming=false --allow-overwrite=true --summary-interval=60 \
  --console-log-level=warn -i "$INPUT" >> "$DL_LOG" 2>&1
rc=$?
rm -f "$INPUT"
for mid in $MODELS; do
  [ -f "$ROOT_DIR/${mid##*/}/config.json" ] || { lanelog "E14 download: config.json missing for $mid"; [ $rc = 0 ] && rc=92; }
done
echo "ARIA_EXIT $rc" >> "$DL_LOG"
lanelog "E14 download done (aria2 exit $rc); $(du -sh "$ROOT_DIR" | cut -f1) in $ROOT_DIR"
exit $rc
