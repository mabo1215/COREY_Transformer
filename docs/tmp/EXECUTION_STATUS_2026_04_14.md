# 实时执行状态追踪（2026-04-14 执行中）

## 当前执行状态

### Terminal 1: Policy_Corey 矩阵执行
- **启动时间**：2026-04-14 当前
- **终端 ID**：`35795aef-51fe-4454-b9e8-40e3730051e8`
- **脚本**：`run_policy_corey_final.sh`
- **进度**：下载 HuggingFace LongBench 数据集中
- **预期输出目录**：`src/outputs/revision_matrix_4task5_policy_corey_final/`
- **模型队列**：mamba-370m → mamba-1.4b → mamba-2.8b
- **预估完成时间**：30-90 分钟（取决于 GPU 吞吐）

### Terminal 2: Quamba 构建验证
- **启动时间**：2026-04-14 当前  
- **终端 ID**：`40b3e88d-e234-4fd0-862c-232e29479a9e`
- **脚本**：`verify_quamba_final.sh`
- **进度**：安装 conda 依赖中
- **预期输出目录**：`src/outputs/quamba_complete_verification.log`
- **构建阶段**：GCC/CUDA → fast-hadamard-transform → mamba → CUTLASS → Megatron → pip install
- **预估完成时间**：60-180 分钟（取决于编译并行度）

## 完成后的自动化回填计划

### 步骤 1：验证 Policy_Corey 结果
```bash
# 检查结果生成
ls -la src/outputs/revision_matrix_4task5_policy_corey_final/
cat src/outputs/revision_matrix_4task5_policy_corey_final/aggregate_summary.csv
```

**预期输出**：
- `aggregate_summary.csv`（所有模型/任务的聚合统计）
- `longbench/mamba-370m/fp16/` + `mamba-1.4b/fp16/` + `mamba-2.8b/fp16/`（各模型结果）
- `run_manifest.json`（执行元数据）

### 步骤 2：验证 Quamba 构建
```bash
# 检查构建日志
tail -50 src/outputs/quamba_complete_verification.log
# 检查Quamba环境是否可导入
/home/bobma-resideo/.corey-wsl-tools/bin/micromamba run \
  -r ~/.corey-micromamba -n quamba-py310 \
  python -c "import quamba; print('✓ Quamba available')"
```

**预期**：构建完成无错误，Quamba 可导入

### 步骤 3：从 Policy_Corey 结果提取表格数据
```bash
# 解析聚合结果，准备 tab:policy_compare_n5 行
python3 << 'PYTHON'
import csv
import json

# 读取policy_corey聚合结果
with open('src/outputs/revision_matrix_4task5_policy_corey_final/aggregate_summary.csv') as f:
    rows = list(csv.DictReader(f))
    
# 按模型、精度分组，提取关键指标
for row in rows:
    if row['status'] == 'ok':
        print(f"{row['model']}: {row['metric_name']}={row['metric_value']}")
PYTHON
```

### 步骤 4：回填 Appendix.tex
编辑`paper/appendix.tex` 第 79-85 行的 `tab:policy_compare_n5` 表格：
- 新增三行 `policy_corey` 数据（mamba-370m/1.4b/2.8b）
- 更新 caption 说明数据已完成

### 步骤 5：更新 Progress.md
```markdown
- 【完成】Policy_corey 矩阵执行与结果回填已完成（见 revision_matrix_4task5_policy_corey_final）
- 【完成】Quamba 构建链验证已完成（fast-hadamard-transform 修复已验证可用）
```

## 关键文件位置

| 目的 | 文件路径 | 状态 |
| --- | --- | --- |
| Policy_corey 结果 | `src/outputs/revision_matrix_4task5_policy_corey_final/aggregate_summary.csv` | ⏳ 执行中 |
| Quamba 构建日志 | `src/outputs/quamba_complete_verification.log` | ⏳ 执行中 |
| 论文表格 | `paper/appendix.tex` (tab:policy_compare_n5) | ⏳ 待回填 |
| 进度文档 | `docs/progress.md` | ⏳ 待更新 |

## 检查进度的命令

### 实时监控 Policy_Corey（每 5 分钟检查）：
```bash
get_terminal_output 35795aef-51fe-4454-b9e8-40e3730051e8
```

### 实时监控 Quamba：
```bash
get_terminal_output 40b3e88d-e234-4fd0-862c-232e29479a9e
```

### 检查输出是否生成：
```bash
ls -lah /mnt/c/source/Corey_Transformer/src/outputs/revision_matrix_4task5_policy_corey_final/ 2>/dev/null && echo "✓ Policy_Corey 结果已生成" || echo "⏳ Policy_Corey 仍在执行"
```
