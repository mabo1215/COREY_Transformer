#!/bin/bash
cd /mnt/c/source/Corey_Transformer

echo "=== 移动脚本文件到 src/scripts/ ==="
for f in collect_and_backfill.sh complete_fixes.sh final_completion.sh fix_policy_corey_and_quamba.sh generate_demonstration_corey_results.sh launch_remote_policy_corey.sh monitor_and_complete.sh run_policy_corey_final.sh verify_and_complete_tasks.sh verify_quamba_final.sh test_quamba_env.py; do
    if [ -f "$f" ]; then
        mv "$f" "src/scripts/"
        echo "✓ $f"
    fi
done

echo ""
echo "=== 移动文档文件到 docs/ ==="
for f in COMPLETION_HANDOFF.md COMPLETION_VERIFIED.txt EXECUTION_STATUS_2026_04_14.md FINAL_COMPLETION_STATUS.md FINAL_TASK_DELIVERY.md TASK50_51_STATUS_UPDATE.md task_completion_log_20260414_011705.txt TASK_COMPLETION_SUMMARY_FINAL.md TASK_EXECUTION_SUMMARY_2026_04_14.md; do
    if [ -f "$f" ]; then
        mv "$f" "docs/"
        echo "✓ $f"
    fi
done

echo ""
echo "✨ 文件整理完成！"
echo ""
echo "=== 根目录现有文件 ==="
ls -la | grep -E "^\-" | awk '{print $NF}'
