#!/bin/bash
# src/scripts/run_all_experiments.sh
# 依次批量运行四个新实验脚本

#!/bin/bash
# src/scripts/run_all_experiments.sh
# 依次批量运行四个新实验脚本（已内联所有命令）


# 1. 外部 Baseline 跑分 (LongBench四子集)
for subset in narrativeqa qasper gov_report multifieldqa_en; do
	python3 experiments/run_external_baselines.py --models rwkv flashattention mamba2 \
		--data-file src/data/longbench_subset/$subset/test.jsonl --device cuda
done

# 2. Quamba 量化推理基准 (LongBench四子集)
for subset in narrativeqa qasper gov_report multifieldqa_en; do
	python3 experiments/run_quamba_quant_benchmark.py --model-id benchang1110/mamba2-2.7b-hf \
		--quant-backend awq --bits 4 --group-size 128 \
		--data-file src/data/longbench_subset/$subset/test.jsonl --device cuda
done

# 3. End-to-end Fused Kernel Benchmark（不依赖数据集）
python src/experiments/run_fused_kernel_benchmark.py --num-ops 8 --device cuda

# 4. policy_corey/mamba-2.7B Ablation Sweep (LongBench四子集)
for subset in narrativeqa qasper gov_report multifieldqa_en; do
	python3 experiments/run_policy_corey_ablation.py --model-id benchang1110/mamba2-2.7b-hf \
		--n 20 --policy corey \
		--data-file src/data/longbench_subset/$subset/test.jsonl --device cuda
done
