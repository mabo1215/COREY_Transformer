#!/usr/bin/env bash
# Remote multi-GPU policy_corey matrix runner - launched from Windows via WSL wrapper
# Prerequisites: .env file with remote login on line 1, password on line 2
# Usage: wsl -e bash -c "bash src/scripts/remote_run_policy_corey_from_windows.sh"

set -euo pipefail

# Read .env credentials
ENV_FILE=".env"
REMOTE=$(head -1 "$ENV_FILE" | tr -d '\r')
PASSWORD=$(sed -n '2p' "$ENV_FILE" | tr -d '\r')
HOST_KEY='ssh-ed25519 255 SHA256:Jj7AizwqBqF1buL3ZBUiE5P37N9XXvel+rxwrYIPty0'
PLINK='C:\\Program Files\\PuTTY\\plink.exe'
PSCP='C:\\Program Files\\PuTTY\\pscp.exe'

# Remote paths
REMOTE_ROOT='/home1/mabo1215/COREY_Transformer'
REMOTE_OUTPUT='src/outputs/revision_matrix_4task5_policy_corey_remote'

echo "[remote-corey] remote=$REMOTE"

# Sync code
echo "[remote-corey] syncing code to $REMOTE_ROOT ..."
for dir in src/algorithms src/experiments src/scripts; do
  echo "  syncing $dir ..."
  "$PSCP" -batch -hostkey "$HOST_KEY" -pw "$PASSWORD" -r "$PWD/$dir" "$REMOTE:$REMOTE_ROOT/src/" 2>/dev/null || true
done

# Launch remote execution
echo "[remote-corey] launching remote policy_corey matrix ..."
REMOTE_CMD="
set -euo pipefail
cd $REMOTE_ROOT
export REPO_ROOT=$REMOTE_ROOT
export MAMBA_ROOT_PREFIX=/home1/mabo1215/.adama-micromamba
export ENV_NAME=quamba-py310
export MM=/home1/mabo1215/.corey-wsl-tools/bin/micromamba
export SCHEDULER_POLICY=corey
export OUTPUT_DIR=$REMOTE_OUTPUT
for model in mamba-370m mamba-1.4b mamba-2.8b; do
  export MODELS=\$model
  if [[ \"\$model\" == \"mamba-2.8b\" ]]; then
    export EVAL_PERPLEXITY=0
  else
    export EVAL_PERPLEXITY=1
  fi
  echo \"[remote-corey-msg] executing model=\$model eval_ppl=\$EVAL_PERPLEXITY\"
  bash src/scripts/wsl_run_checkpoint_matrix.sh
done
bash src/scripts/wsl_run_checkpoint_matrix.sh  # Consolidate
"

"$PLINK" -ssh -batch -hostkey "$HOST_KEY" -pw "$PASSWORD" "$REMOTE" "$REMOTE_CMD"

echo "[remote-corey] done"
