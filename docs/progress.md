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
- 任务 8：将 `run_longbench_inference.py` 扩展为同时支持本地 JSONL 与 Hugging Face `datasets` 读取，并在 `mamba_integration.py` 中补充 AWQ / GPTQ 量化 backend 的加载骨架；同时修复 LaTeX 中 figure/table/algorithm 的 `hyperref` anchor 命名，清除 duplicate destination 警告。
- 任务 9：继续把 LongBench runner 扩展为更稳健的多 schema 兼容版本，加入 batch inference、optional perplexity side-eval，以及可直接通过本机 Ollama API 运行的 backend；同时用本地 `llama3:latest` 完成了一次 smoke run，验证输出文件可正常生成。
- 任务 10：新增独立 `.venv311`，安装 `torch/transformers/datasets` 并将 runner 扩展为统一输出 LongBench 与 WikiText-103 / PG19 的评测协议；同时使用用户指定的 `mlx-community/mamba-1.4b-hf-f32` 作为请求模型 ID，在 Windows/HF 路径下自动解析到其基座 `state-spaces/mamba-1.4b-hf` 并完成真实 HF Mamba-1.4B smoke run。
- 任务 11：使用本机 `mathstral:latest` 审核论文中 entropy majorization 定理与证明的数学表述，并据此收紧正文 theorem wording 与 appendix 中的 proof phrasing。

## 未修改或部分修改

- 【已阻挡】尚未新增真实预训练模型推理代码与 benchmark 脚本。当前仓库只有原型实验管线，缺少可直接运行 Mamba-370M / 1.4B / 2.8B 的完整部署代码。
- 【已阻挡】尚未补充真实 LongBench、WikiText-103、PG19 的结果表、能耗统计与 checkpoint-level perplexity 结果。原因是对应实验尚未实际执行，不能在论文中伪造数值。
- 【已阻挡】当前机器的 `ollama` 仅安装了 `llama3:latest`，Ollama 官方模型库检索结果中也未发现现成的 Mamba-1.4B 条目；同时当前 `.venv` 为 Python 3.14，尚未具备 `torch/transformers/autoawq/auto-gptq` 运行栈，因此真实 Mamba-1.4B benchmark 仍需额外模型与环境准备。
- 【已阻挡】`mlx-community/mamba-1.4b-hf-f32` 是 MLX 模型，不能在当前 Windows `torch/transformers` 路径中直接作为原生权重运行，因此实际实验只能解析到其基座 `state-spaces/mamba-1.4b-hf`。
- 【已阻挡】GPTQ 路径在当前 `auto-gptq` 上已验证到模型检查阶段，但明确报错 `mamba isn't supported yet`；AWQ 路径在 Windows 上虽然成功安装，但其依赖链仍卡在 `transformers.models.phi3` / kernel extension 兼容问题，因此两条量化路径目前都无法对 Mamba-1.4B 完成真实量化推理。
- 【已阻挡】PG19 在当前 `datasets` 入口下返回 `Dataset scripts are no longer supported, but found pg19.py`，因此 unified protocol 已支持该数据集的记录与降级，但本机当前环境尚未得到可直接运行的 PG19 HF 入口。
- 【进行中】下一阶段可继续把 protocol section 升级为真实结果 section，前提是先在 `src/` 中补齐真实模型 benchmark 实现并产出可复现结果。
