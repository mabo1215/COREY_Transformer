# 论文进度

最后更新：本轮 revision cycle 已继续推进，并补完 WSL2 下的 1.4B checkpoint sanity-check。

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
- 任务 12：按本轮 `docs/revision_suggestions.tex` 完成正文与附录的论文级修订：统一熵记号并引入正式 theorem 环境、将主文表格补齐 Static Fusion 真值行并改用 `[t]` 浮动、明确 `quality drop` 仅为原型 proxy 指标、把 checkpoint smoke-test 数值降级到附录、补充 prototype hardware/software 环境说明，并扩展 Related Work / BibTeX 以纳入 FlashAttention、Triton、SmoothQuant 与 AWQ。
- 任务 13：补齐真实官方 checkpoint benchmark 闭环：扩展 `run_longbench_inference.py` 以支持 `max_length` 与关闭 entropy hook，新增 `run_official_mamba_benchmark.py` 做 warmup/repeat/内存记录，并在 `state-spaces/mamba-370m-hf` 上完成一次真实 HF benchmark，产出 `src/outputs/official_hf_benchmark/` 与 `src/outputs/official_hf_benchmark_fastpath/`。同时清理 LaTeX 日志中的 duplicate hyperref destination 与 appendix overfull hbox，使论文编译日志显著收敛。
- 任务 14：确认本机已切换到可用的 NVIDIA 环境后，将 `.venv` 中的 `torch` 重装为 `2.11.0+cu128`，验证 `torch.cuda.is_available()` 与 RTX 3070 可见，并在 GPU 上完成一次真实官方 HF Mamba benchmark，产出 `src/outputs/official_hf_benchmark_gpu/`。同时把 benchmark metadata 改为显式记录 `fast_path_status`，避免把“GPU 但无官方 fused kernel”的运行误标为 deployment-grade。
- 任务 15：新增 `docs/wsl2_cuda128_migration.md`、`scripts/wsl_setup_cuda128_env.sh` 与 `scripts/wsl_run_official_benchmark.sh`，把 WSL2 CUDA 12.8 对齐迁移清单和一键 benchmark 命令集落到仓库中；同时确认 micromamba root 必须放在 WSL Linux 文件系统而非 `/mnt/c/...`，并进一步定位到 `causal-conv1d` 的上游 `setup.py` 会硬编码多架构 `nvcc` 编译目标，因而其构建耗时不能仅靠 `TORCH_CUDA_ARCH_LIST` 收敛。
- 任务 16：按当前 `docs/revision_suggestions.tex` 继续完成“可写即改”的 revision 收口：将论文标题、摘要、引言与结论进一步改写为 prototype-study framing，补入主文的 hyperparameter / baseline 细节表，明确 Static Fusion 的固定分组定义与 `quality drop` 仅为 diagnostic proxy；同时在附录中补齐真实复现参数、修复 adaptive threshold 与 EMA 共用 `\lambda` 的符号冲突、把 Hadamard 量化稳定性改写为可验证的峰值坐标 / clipping bound，并补入仅限算法级的 entropy overhead 说明，避免把未测得的 GPU 开销写成既成事实。
- 任务 17：在当前 `.venv` 中重新确认 Hugging Face Mamba checkpoint 路径可直接用于真实 GPU sanity-check，并新增 `mamba-1.4b` 的官方 benchmark 输出到 `src/outputs/official_hf_benchmark_gpu_14b/`；同时复核 `.venv` 现为 Python 3.11 + `torch 2.11.0+cu128`，并确认 `install_python_packages` 给出的“已安装 `mamba-ssm` / `causal-conv1d`”并未真正落入当前工作区解释器，实际以 `.venv` 内 `pip install` 复查后仍被 `nvcc` 缺失和上游 `mamba-ssm` 构建错误阻断。
- 任务 18：在 WSL2 中转而复用可用的 `adama-cuda128` CUDA 12.8 环境，确认 `torch 2.11.0+cu128`、`transformers 5.5.0`、`datasets 4.8.4`、`triton 3.6.0` 与 RTX 3070 均可见，并完成 `mamba-1.4b` 官方 benchmark 的真实 WSL2 GPU sanity-check，产出 `src/outputs/official_hf_benchmark_wsl_14b/`。结果显示 NarrativeQA smoke 上 32 token 的平均延迟为 2194.588 ms、吞吐 14.641 tok/s、token-F1 为 0.173913、峰值 RSS 为 1626.7109 MB，WikiText-103 单样本 perplexity 为 525.633179；同时 metadata 继续给出 `fast_path_available=false`、`deployment_grade=false`，说明 WSL2 已能提供真实 Linux GPU fallback 证据，但尚不足以把 checkpoint 结果提升为主文对比表。
- 任务 19：继续修复 WSL2 `adama-cuda128` 的官方 fast-path 依赖链：为 `wsl_setup_cuda128_env.sh` 增补 `einops` 基础依赖检查，给 `mamba-ssm` 源码安装路径加入单架构 `sm_86` patch，并最终在 WSL2 上恢复 `mamba_ssm` / `causal_conv1d` 的官方 fast path。随后在 `state-spaces/mamba-1.4b-hf` 上完成一轮新的官方 benchmark，产出 `src/outputs/official_hf_benchmark_wsl_fastpath_14b/`；结果为 NarrativeQA smoke 上 32 token 平均延迟 1149.3544 ms、吞吐 27.9101 tok/s、token-F1 0.173913、峰值 RSS 1618.8164 MB，WikiText-103 perplexity 524.900574，metadata 明确记录 `fast_path_available=true` 与 `deployment_grade=true`。同时据此把论文中 checkpoint-level 证据段落改写为“已验证跑通，但任务覆盖和样本规模仍待扩展”的更精确表述。

## 未修改或部分修改

- 【部分缓解】仓库现已具备可直接运行的真实预训练模型推理与 benchmark 脚本，并已在 HF 官方路径上完成 `mamba-370m` 与 `mamba-1.4b` 的 GPU sanity-check；但 `mamba-2.8b`、deployment-grade fast path、以及面向论文主结果的成体系 checkpoint 对比仍未完成。
- 【部分缓解】现已补入部分真实 WikiText-103 perplexity 与本地 LongBench smoke 结果，但仍缺完整 LongBench、PG19、能耗统计与可进入主文表格的系统性 checkpoint-level 结果。
- 【已解除】当前 `.venv` 已是 Python 3.11 并具备 `torch/transformers/autoawq/auto-gptq` 运行栈，因此真实 Mamba-1.4B benchmark 的“环境未就绪”问题已解除；剩余阻塞转为 fast-path 扩展、量化兼容性与更完整数据集评测。
- 【已阻挡】`mlx-community/mamba-1.4b-hf-f32` 是 MLX 模型，不能在当前 Windows `torch/transformers` 路径中直接作为原生权重运行，因此实际实验只能解析到其基座 `state-spaces/mamba-1.4b-hf`。
- 【已阻挡】GPTQ 路径在当前 `auto-gptq` 上已验证到模型检查阶段，但明确报错 `mamba isn't supported yet`；AWQ 路径在 Windows 上虽然成功安装，但其依赖链仍卡在 `transformers.models.phi3` / kernel extension 兼容问题，因此两条量化路径目前都无法对 Mamba-1.4B 完成真实量化推理。
- 【已阻挡】PG19 在当前 `datasets` 入口下返回 `Dataset scripts are no longer supported, but found pg19.py`，因此 unified protocol 已支持该数据集的记录与降级，但本机当前环境尚未得到可直接运行的 PG19 HF 入口。
- 【已阻挡】本轮 review 中要求的真实 GPU Triton benchmark、公平外部 baseline 对比以及 Theorem 1 的经验条件验证，目前仍缺少可运行内核、硬件测量和对应实验输出，故只能先将主文论断收紧到“controlled prototype study”层面，不能伪造 deployment-grade 结果。
- 【已阻挡】虽然 `.venv` 已切换到 `torch 2.11.0+cu128` 且 GPU benchmark 已能在 RTX 3070 上真实运行，但 `mamba-ssm` / `causal-conv1d` 仍未安装成功，因此官方 HF 路径的 metadata 继续给出 `fast_path_available=false` 与 `deployment_grade=false`。
- 【已阻挡】当前机器安装的是 CUDA Toolkit 13.2，而可用的 PyTorch wheel 为 `cu128`；在 `--no-build-isolation` 下重装 `mamba-ssm` / `causal-conv1d` 时，构建过程已进入 extension 阶段，但被 `The detected CUDA version (13.2) mismatches the version that was used to compile PyTorch (12.8)` 阻断，同时对应的 Windows 预编译 wheel URL 也返回 404。
- 【已阻挡】Triton 在当前 Windows 环境中既无法直接 `import triton`，也无法通过 `pip install triton` 获得可用 wheel；因此本机当前无法执行 Triton kernel benchmark，除非切换到支持 Triton 的 Linux/WSL2 CUDA 环境。
- 【部分缓解】WSL2 侧的 `adama-cuda128` Linux CUDA 12.8 环境现已恢复 `mamba-ssm` / `causal-conv1d` 官方 fast path，并已产出 deployment-grade 的 `mamba-1.4b` 官方 benchmark 与 4 任务 LongBench 子集结果；当前剩余问题已从“kernel 路径不可用”转为“checkpoint-level 证据覆盖仍然过窄”，即仍缺更大任务集、更多样本、Mamba-2.8B、以及真实 static fusion / COREY 对比。
