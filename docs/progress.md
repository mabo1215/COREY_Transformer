# 论文进度

最后更新：本轮 revision cycle 已继续推进并重新验证编译。

## 已全部修改

- 任务 1：在 `paper/main.tex` 中新增 `Entropy-Regularized Fusion Optimization`，将融合调度写成带硬件约束的优化问题，并补充动态规划求解器与自适应熵阈值。这样正文对“entropy-guided”不再只停留在启发式描述。
- 任务 2：在 `paper/main.tex` 中新增 `Theoretical Analysis`，补充熵增长、融合可行性、量化稳定性三组理论化表述与 proof sketch，使方法论叙述更符合 NeurIPS 论文对理论支撑的预期。
- 任务 3：将真实 Mamba 实验建议改写为诚实的 `Protocol for Pretrained SSM Evaluation` 与 `LongBench-Oriented Evaluation Protocol`，明确模型、数据集、指标、基线和硬件，但不伪造尚未运行的结果。
- 任务 4：在 `paper/main.tex` 中新增 `Triton Kernel Integration`，补充 kernel pipeline、fused kernel pseudocode 与 tile scheduling 设计，使系统路线更完整。
- 任务 5：在 `paper/appendix.tex` 中新增 `Detailed Proofs`，把熵增长、融合深度上界、量化稳定性的附录证明细化为正式 appendix 内容。
- 任务 6：从正文移除 `Rebuttal Points` 这类不适合论文主体的内容，并重新编译论文，确认主文稿和附录仍可成功生成 PDF。
- 任务 7：继续补全 LongBench inference harness design、真实 Mamba integration skeleton 与更严格的 entropy-majorization 定理表述；其中代码 skeleton 已加入 `src/`，正文与附录的熵证明也已改写为基于 doubly-stochastic mixing 的严格条件版。

## 未修改或部分修改

- 【已阻挡】尚未新增真实预训练模型推理代码与 benchmark 脚本。当前仓库只有原型实验管线，缺少可直接运行 Mamba-370M / 1.4B / 2.8B 的完整部署代码。
- 【已阻挡】尚未补充真实 LongBench、WikiText-103、PG19 的结果表、能耗统计与 checkpoint-level perplexity 结果。原因是对应实验尚未实际执行，不能在论文中伪造数值。
- 【进行中】下一阶段可继续把 protocol section 升级为真实结果 section，前提是先在 `src/` 中补齐真实模型 benchmark 实现并产出可复现结果。
