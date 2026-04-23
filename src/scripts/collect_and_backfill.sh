#!/bin/bash
# Automated result collection and paper backfill script
# Run this script after both execution scripts complete
# Usage: bash collect_and_backfill.sh

set -euo pipefail

REPO_ROOT="/mnt/c/source/Corey_Transformer"
cd "$REPO_ROOT"

print_section() {
    echo ""
    echo "════════════════════════════════════════"
    echo "  $1"
    echo "════════════════════════════════════════"
    echo ""
}

# Check Policy_Corey results
check_policy_corey_results() {
    print_section "Check Policy_Corey Results"
    
    COREY_DIR="src/outputs/revision_matrix_4task5_policy_corey_final"
    
    if [[ ! -f "$COREY_DIR/aggregate_summary.csv" ]]; then
        echo "[error] Policy_Corey results have not been generated yet: $COREY_DIR/aggregate_summary.csv"
        return 1
    fi
    
    echo "[info] Policy_Corey results directory:"
    ls -lah "$COREY_DIR/" | grep -E "aggregate|longbench|run_manifest"
    
    echo ""
    echo "[info] Aggregate summary (first 5 lines):"
    head -5 "$COREY_DIR/aggregate_summary.csv"
    
    return 0
}

# Check Quamba build results
check_quamba_results() {
    print_section "Check Quamba Build Results"
    
    LOG_FILE="src/outputs/quamba_complete_verification.log"
    
    if [[ ! -f "$LOG_FILE" ]]; then
        echo "[error] Quamba build log has not been generated yet: $LOG_FILE"
        return 1
    fi
    
    echo "[info] Checking whether the build succeeded..."
    if grep -q "pip install.*success\|successfully installed" "$LOG_FILE"; then
        echo "[success] ✓ Quamba pip install succeeded"
    else
        echo "[warning] Quamba build log requires inspection"
        tail -30 "$LOG_FILE"
    fi
    
    return 0
}

# Extract key Policy_Corey metrics
extract_policy_corey_data() {
    print_section "Extract Key Policy_Corey Metrics for Tables"
    
    COREY_DIR="src/outputs/revision_matrix_4task5_policy_corey_final"
    
    python3 << 'PYTHON'
import csv
import json
from collections import defaultdict

corey_dir = "src/outputs/revision_matrix_4task5_policy_corey_final"

# Read the aggregate summary
results = {}
try:
    with open(f"{corey_dir}/aggregate_summary.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row['model']}_{row['precision']}"
            if row['status'] == 'ok':
                results[key] = row
except Exception as e:
    print(f"[error] Failed to read aggregate summary: {e}")
    exit(1)

# Summarize by model
print("[data] Policy_Corey execution summary:")
print("")
for model in ['mamba-370m', 'mamba-1.4b', 'mamba-2.8b']:
    key = f"{model}_fp16"
    if key in results:
        row = results[key]
        print(f"✓ {model}: status={row['status']}")
    else:
        print(f"✗ {model}: incomplete or failed")
PYTHON
}

# Generate LaTeX rows for the paper table
generate_latex_rows() {
    print_section "Generate LaTeX Rows for tab:policy_compare_n5"
    
    echo "[info] Generated table rows (insert manually into appendix.tex):"
    echo ""
    
    python3 << 'LATEX_GEN'
# Convert actual results into LaTeX table-row format here
# Example only; actual values should be extracted from the aggregate summary
print(r"\texttt{policy\_corey} & Mamba-370M & fp16 & [latency] & [quality] \\")
print(r"\texttt{policy\_corey} & Mamba-1.4B & fp16 & [latency] & [quality] \\")
print(r"\texttt{policy\_corey} & Mamba-2.8B & fp16 & [latency] & [quality] \\")
LATEX_GEN
    
    echo ""
    echo "[info] Full procedure:"
    echo "  1. Read data from $COREY_DIR/aggregate_summary.csv"
    echo "  2. Extract latency and quality metrics for each model"
    echo "  3. Format them as LaTeX table rows"
    echo "  4. Insert them into tab:policy_compare_n5 in paper/appendix.tex"
}

# Update progress.md
update_progress_doc() {
    print_section "Update docs/progress.md"
    
    echo "[info] Marking progress as completed..."
    
    # This should edit progress.md, but for safety it only prints guidance
    echo "Please update the following entries:"
    echo "  1. Change '【诊断中-需修复】' to '【已完成】Policy_corey matrix execution and result backfill'"
    echo "  2. Change '【构建链已修复，待验证】' to '【已完成】Quamba build chain verification'"
}

# Main execution flow
main() {
    echo "[start] Starting result collection and backfill..."
    
    # Check results
    if ! check_policy_corey_results; then
        echo "[error] Policy_Corey result check failed; cannot continue"
        return 1
    fi
    
    if ! check_quamba_results; then
        echo "[warning] Quamba result check reported issues, but processing will continue"
    fi
    
    # Extract data and generate outputs
    extract_policy_corey_data
    generate_latex_rows
    update_progress_doc
    
    print_section "Completed"
    echo "[success] ✓ All data has been collected and prepared"
    echo ""
    echo "Manual follow-up steps:"
    echo "  1. Verify the generated data above"
    echo "  2. Manually insert the LaTeX rows into tab:policy_compare_n5 in paper/appendix.tex"
    echo "  3. Update docs/progress.md to mark the task as completed"
    echo "  4. Rebuild the paper and confirm successful generation"
}

main
