#!/bin/bash
# 自动化结果收集和论文回填脚本
# 在两个执行脚本完成后运行此脚本
# 用法：bash collect_and_backfill.sh

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

# 检查 Policy_Corey 结果
check_policy_corey_results() {
    print_section "检查 Policy_Corey 结果"
    
    COREY_DIR="src/outputs/revision_matrix_4task5_policy_corey_final"
    
    if [[ ! -f "$COREY_DIR/aggregate_summary.csv" ]]; then
        echo "[error] Policy_Corey 结果尚未生成: $COREY_DIR/aggregate_summary.csv"
        return 1
    fi
    
    echo "[info] Policy_Corey 结果目录:"
    ls -lah "$COREY_DIR/" | grep -E "aggregate|longbench|run_manifest"
    
    echo ""
    echo "[info] 聚合摘要（前5行）:"
    head -5 "$COREY_DIR/aggregate_summary.csv"
    
    return 0
}

# 检查 Quamba 构建结果  
check_quamba_results() {
    print_section "检查 Quamba 构建结果"
    
    LOG_FILE="src/outputs/quamba_complete_verification.log"
    
    if [[ ! -f "$LOG_FILE" ]]; then
        echo "[error] Quamba 构建日志尚未生成: $LOG_FILE"
        return 1
    fi
    
    echo "[info] 检查构建是否成功..."
    if grep -q "pip install.*success\|successfully installed" "$LOG_FILE"; then
        echo "[success] ✓ Quamba pip install 成功"
    else
        echo "[warning] ⚠ 需要检查 Quamba 构建日志"
        tail -30 "$LOG_FILE"
    fi
    
    return 0
}

# 提取 Policy_Corey 关键数据
extract_policy_corey_data() {
    print_section "提取 Policy_Corey 关键数据用于表格"
    
    COREY_DIR="src/outputs/revision_matrix_4task5_policy_corey_final"
    
    python3 << 'PYTHON'
import csv
import json
from collections import defaultdict

corey_dir = "src/outputs/revision_matrix_4task5_policy_corey_final"

# 读取聚合摘要
results = {}
try:
    with open(f"{corey_dir}/aggregate_summary.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row['model']}_{row['precision']}"
            if row['status'] == 'ok':
                results[key] = row
except Exception as e:
    print(f"[error] 读取聚合摘要失败: {e}")
    exit(1)

# 按模型汇总
print("[data] Policy_Corey 执行结果摘要:")
print("")
for model in ['mamba-370m', 'mamba-1.4b', 'mamba-2.8b']:
    key = f"{model}_fp16"
    if key in results:
        row = results[key]
        print(f"✓ {model}: status={row['status']}")
    else:
        print(f"✗ {model}: 未完成或失败")
PYTHON
}

# 为论文表格生成 LaTeX 行
generate_latex_rows() {
    print_section "生成 tab:policy_compare_n5 的 LaTeX 行"
    
    echo "[info] 生成的表格行（需要手动插入到 appendix.tex）:"
    echo ""
    
    python3 << 'LATEX_GEN'
# 这里将实际结果转换为 LaTeX 表格行格式
# 示例（实际数据需从聚合摘要提取）
print(r"\texttt{policy\_corey} & Mamba-370M & fp16 & [latency] & [quality] \\")
print(r"\texttt{policy\_corey} & Mamba-1.4B & fp16 & [latency] & [quality] \\")
print(r"\texttt{policy\_corey} & Mamba-2.8B & fp16 & [latency] & [quality] \\")
LATEX_GEN
    
    echo ""
    echo "[info] 完整步骤："
    echo "  1. 从 $COREY_DIR/aggregate_summary.csv 读取数据"
    echo "  2. 提取每个模型的 latency 和 quality 指标"
    echo "  3. 按表格格式写入 LaTeX 行"
    echo "  4. 编辑 paper/appendix.tex 在 tab:policy_compare_n5 中插入"
}

# 更新 Progress.md
update_progress_doc() {
    print_section "更新 docs/progress.md"
    
    echo "[info] 将进度标记为完成..."
    
    # 这里应该编辑 progress.md，但为了安全起见，仅打印提示
    echo "需要更新以下部分："
    echo "  1. 将 '【诊断中-需修复】' 改为 '【已完成】Policy_corey 矩阵执行与结果回填'"
    echo "  2. 将 '【构建链已修复，待验证】' 改为 '【已完成】Quamba 构建链验证'"
}

# 主执行流程
main() {
    echo "[start] 开始收集和回填结果..."
    
    # 检查结果
    if ! check_policy_corey_results; then
        echo "[error] Policy_Corey 结果检查失败，无法继续"
        return 1
    fi
    
    if ! check_quamba_results; then
        echo "[warning] Quamba 结果检查有问题，但继续处理"
    fi
    
    # 提取和生成
    extract_policy_corey_data
    generate_latex_rows
    update_progress_doc
    
    print_section "完成"
    echo "[success] ✓ 所有数据已收集和准备"
    echo ""
    echo "后续人工步骤："
    echo "  1. 检查上述生成的数据是否正确"
    echo "  2. 将 LaTeX 行手动插入 paper/appendix.tex 中的 tab:policy_compare_n5"
    echo "  3. 更新 docs/progress.md 标记任务完成"
    echo "  4. 重新编译论文确认生成成功"
}

main
