#!/usr/bin/env bash
set -euo pipefail

REMOTE="mabo1215@10.147.20.176"
REMOTE_ROOT="/home1/mabo1215/COREY_Transformer/src/data/longbench_subset"
LOCAL_ROOT="/mnt/c/source/COREY_Transformer/src/data/longbench_subset"

ssh -o BatchMode=yes "$REMOTE" "mkdir -p $REMOTE_ROOT/narrativeqa $REMOTE_ROOT/qasper $REMOTE_ROOT/multifieldqa_en $REMOTE_ROOT/gov_report"

for t in narrativeqa qasper multifieldqa_en gov_report; do
  head -n 120 "$LOCAL_ROOT/$t/test.jsonl" | ssh -o BatchMode=yes "$REMOTE" "cat > $REMOTE_ROOT/$t/test.jsonl"
done

ssh -o BatchMode=yes "$REMOTE" "wc -l $REMOTE_ROOT/narrativeqa/test.jsonl $REMOTE_ROOT/qasper/test.jsonl $REMOTE_ROOT/multifieldqa_en/test.jsonl $REMOTE_ROOT/gov_report/test.jsonl"
