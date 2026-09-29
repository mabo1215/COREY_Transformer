#!/usr/bin/env bash
# Delete the E11 traces and checkpoints on the RTX 3090 host (author-approved, 2026-09-30).
# Removes only these paths, after checking that the E11 result files are committed:
#   src/outputs/tpds_traces_1p4b, src/outputs/tpds_traces_2p8b          (E11 traces)
#   /root/autodl-tmp/hf_models/mamba-1.4b-hf, mamba-2.8b-hf            (aria2 downloads)
#   HF cache entries models--state-spaces--mamba-1.4b-hf / -2.8b-hf     (partial downloads)
# The 370M traces (src/outputs/tpds_traces) are kept.
#
#   wsl bash src/revloop/remote.sh exec vgpu3090 src/experiments/remote/tpds_r2_cleanup_e11.sh
set -uo pipefail
ROOT=/root/autodl-tmp/COREY_Transformer_tpds
cd "$ROOT" || exit 1
( [ -f /etc/network_turbo ] && source /etc/network_turbo >/dev/null 2>&1; timeout 120 git pull -q --ff-only ) \
  || { echo "git pull failed; not deleting anything"; exit 1; }
for f in src/experiments/results/tpds/e2_mamba1_vgpu3090_1p4b.csv src/experiments/results/tpds/e2_mamba1_vgpu3090_2p8b.csv \
         src/experiments/results/tpds/e2_mamba1_vgpu3090_1p4b.json src/experiments/results/tpds/e2_mamba1_vgpu3090_2p8b.json; do
  git ls-files --error-unmatch "$f" >/dev/null 2>&1 || { echo "not in git: $f; not deleting anything"; exit 1; }
  git diff --quiet -- "$f" || { echo "differs from git: $f; not deleting anything"; exit 1; }
done
echo "E11 result files are committed; deleting:"
TARGETS=(src/outputs/tpds_traces_1p4b src/outputs/tpds_traces_2p8b
         /root/autodl-tmp/hf_models/mamba-1.4b-hf /root/autodl-tmp/hf_models/mamba-2.8b-hf
         /root/.cache/huggingface/hub/models--state-spaces--mamba-1.4b-hf
         /root/.cache/huggingface/hub/models--state-spaces--mamba-2.8b-hf)
for t in "${TARGETS[@]}"; do
  if [ -e "$t" ]; then echo "  $(du -sh "$t" 2>/dev/null | cut -f1)	$t"; rm -rf -- "$t"; else echo "  (absent)	$t"; fi
done
rmdir /root/autodl-tmp/hf_models 2>/dev/null || true
df -h /root/autodl-tmp | tail -1
