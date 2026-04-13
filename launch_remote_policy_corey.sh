#!/bin/bash
# Remote policy_corey matrix launcher
cd /home1/mabo1215/COREY_Transformer
export REPO_ROOT=/home1/mabo1215/COREY_Transformer
export SCHEDULER_POLICY=corey
export MAX_SAMPLES=5
export OUTPUT_DIR=src/outputs/revision_matrix_4task5_policy_corey_remote

for model in mamba-370m mamba-1.4b mamba-2.8b; do
  export MODELS="$model"
  if [[ "$model" == "mamba-2.8b" ]]; then
    export EVAL_PERPLEXITY=0
  else
    export EVAL_PERPLEXITY=1
  fi
  echo "[remote-policy-corey] executing model=$model eval_ppl=$EVAL_PERPLEXITY"
  bash src/scripts/wsl_run_checkpoint_matrix.sh
  echo "[remote-policy-corey] completed model=$model"
done

# Final consolidation
export MODELS="mamba-370m mamba-1.4b mamba-2.8b"
echo "[remote-policy-corey] consolidating aggregate summary"
bash src/scripts/wsl_run_checkpoint_matrix.sh
echo "[remote-policy-corey] matrix run completed"
