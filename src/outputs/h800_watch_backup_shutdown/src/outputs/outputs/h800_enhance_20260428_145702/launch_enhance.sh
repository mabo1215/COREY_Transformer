#!/usr/bin/env bash
set -euo pipefail
cd /root/Corey_Transformer
source ./use_system_cache.sh
export PATH=/root/miniconda3/bin:$PATH
export PYTHON_BIN=/root/miniconda3/bin/python
export PYTHONPATH=/root/Corey_Transformer/src:/root/Corey_Transformer:${PYTHONPATH:-}
export BASE="src/outputs/h800_enhance_20260428_145702"
mkdir -p "$BASE/logs"
run_stage() {
  name="$1"; shift
  out="$BASE/$name"
  if [[ -f "$out/.stage_done" ]]; then
    echo "[skip] $name"
    return 0
  fi
  mkdir -p "$out"
  echo "[$(date '+%F %T')] RUN $name: $*"
  "$@" > "$BASE/logs/${name}.log" 2>&1
  touch "$out/.stage_done"
  echo "[$(date '+%F %T')] DONE $name"
}
run_stage integrated_repeats20   "$PYTHON_BIN" src/experiments/run_integrated_end_to_end.py     --model mamba-370m --new-tokens 32 --warmup 2 --repeats 20 --num-bins 256     --output-dir "$BASE/integrated_repeats20"
run_stage diversity_extra20   "$PYTHON_BIN" src/experiments/run_real_workload_diversity_h800.py     --model mamba-370m --dataset-root src/data/longbench_subset     --tasks narrativeqa qasper gov_report multifieldqa_en --samples-per-task 20     --extra-regime-file "$BASE/extra_regimes/extra_code.jsonl"     --extra-regime-file "$BASE/extra_regimes/extra_logs.jsonl"     --extra-regime-file "$BASE/extra_regimes/extra_tables.jsonl"     --extra-regime-file "$BASE/extra_regimes/extra_repetition.jsonl"     --max-extra-per-regime 20 --output-dir "$BASE/diversity_extra20"
run_stage pythia410m_fa3_50   "$PYTHON_BIN" src/experiments/run_hf_longbench_baseline.py     --model-name pythia-410m-fa3-50 --model-id EleutherAI/pythia-410m     --dataset-root src/data/longbench_subset --tasks narrativeqa qasper gov_report multifieldqa_en     --max-samples 50 --device cuda --dtype bfloat16 --attn-implementation flash_attention_3     --output-dir "$BASE/pythia410m_fa3_50"
run_stage mamba2_ssd_50   "$PYTHON_BIN" src/experiments/run_hf_longbench_baseline.py     --model-name mamba2-ssd-2.7b-50 --model-id benchang1110/mamba2-2.7b-hf     --dataset-root src/data/longbench_subset --tasks narrativeqa qasper gov_report multifieldqa_en     --max-samples 50 --device cuda --dtype bfloat16 --trust-remote-code     --output-dir "$BASE/mamba2_ssd_50"
echo "[$(date '+%F %T')] ALL_ENHANCE_DONE $BASE"
