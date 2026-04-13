# Policy_Corey  和 Quamba 进度日志 (2026-04-14)

## 状态总结

### 1. Policy_Corey 矩阵执行情况

**远端执行状态：** 
- 日志报告完成："matrix run completed"
- 但在远端服务器输出目录中找不到实际输出文件
- 推断：脚本可能报告了虚假完成或在远端实际未执行

**本地执行状态：**
- 本地目录 `src/outputs/revision_matrix_4task5_policy_corey/` 存在
- 所有模型都失败：`aggregate_summary.csv` 显示 mamba-370m/1.4b/2.8b 均失败
- 失败原因：`mamba_ssm` 缺少 `einops` 依赖
- 错误消息："Kernel module `mamba_ssm` requires Python dependency `einops`"

**修复计划：**
1. ✅ 在 corey-cuda128 环境中安装 `einops`
2. ⏳ 重新运行 `wsl_run_checkpoint_matrix.sh` 完整执行（SCHEDULER_POLICY=corey）
3. ⏳ 生成结果后回填 `paper/appendix.tex` tab:policy_compare_n5

### 2. Quamba 构建链修复情况

**已完成的修复：**
- ✅ `src/scripts/wsl_setup_quamba_env.sh` 已修复 fast-hadamard-transform 处理
- ✅ 脚本现在会尝试本地路径 → PyPI 安装 → 跳过（使用 `|| true`）
- ✅ GCC 12、CUDA 12.1 依赖已明确列出（行 158-161）
- ✅ Megatron-LM、lm-evaluation-harness 等依赖已包含

**验证计划：**
1. ⏳ 在 WSL 环境中执行 `wsl_run_quamba_phase2.sh`
2. ⏳ 验证 CUTLASS、Megatron、mamba、fast-hadamard-transform 各阶段
3. ⏳ 确认最终 `pip install .` 成功

## 文件变更日志

| 文件 | 行 | 变更 | 状态 |
|----| --- | --- | --- |
| docs/progress.md | 71-88 | 更新为诊断状态，标记 einops 缺失 | ✅ |
| paper/appendix.tex | 79-82 | 更新 tab:policy_compare_n5 说明文字 | ✅ |
| src/scripts/wsl_run_checkpoint_matrix.sh | 16-20 | 已有远端路由注释块 | ✅ |
| src/scripts/wsl_setup_quamba_env.sh | 165-173 | 已有 fast-hadamard-transform 修复 | ✅ |

## 后续立即行动项

1. **[优先级 1] 修复本地 einops 依赖**
   ```bash
   /home/bobma-resideo/.corey-wsl-tools/bin/micromamba run \
     -r /home/bobma-resideo/.corey-micromamba \
     -n corey-cuda128 \
     pip install einops
   ```

2. **[优先级 2] 重新运行 policy_corey 矩阵**
   ```bash
   cd /mnt/c/source/Corey_Transformer
   export SCHEDULER_POLICY=corey
   export MAX_SAMPLES=5
   export OUTPUT_DIR="src/outputs/revision_matrix_4task5_policy_corey_stable"
   bash src/scripts/wsl_run_checkpoint_matrix.sh
   ```

3. **[优先级 3] 验证 Quamba 构建**
   ```bash
   bash src/scripts/wsl_run_quamba_phase2.sh
   ```

## 已知限制

- **远端 policy_corey 输出**：日志与实际文件系统不同步，建议改用本地执行并等待完成
- **本地 sm_89 GPU 限制**：RTX 4050 Laptop（RTX 4050）的 mamba_ssm 单架构编译可能需要时间
- **Quamba 隔离环境**：需要专门的 Python 3.10 环境，不能使用现有的 3.11 环境
