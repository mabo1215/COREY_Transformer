下面我给你一个**不接受负结果叙事时的可执行转向方案**。核心判断是：**不能再重复现在的 scheduler matrix**。当前 H800 证据已经说明，在同一类 homogeneous real-checkpoint workload 上，adaptive/proxy 很难赢过 best static oracle：Static-512 是最佳固定块，adaptive/proxy 全部更慢；但 routed path 的质量保持基本成立，3/3 greedy exact-match，PPL 只在数值噪声范围变化。

## 总结结论

你要把问题从：

> “在同一个 workload 上，entropy scheduler 能否打败 best static chunk？”

改成：

> “在真实 mixed-regime serving workload 中，单一 static chunk 无法同时适配不同 regime；regime-aware adaptive scheduler 能否超过 global best static oracle？”

这是唯一比较合理的正结果路线。因为如果 workload 本身很单一，best static oracle 已经是固定 chunk 的最优点，adaptive 除非靠噪声，否则很难稳定超过它。当前论文也已经明确：现有真实 workload 基本坍缩到同一个 chunk bucket，尚未构造出不同 prompt/sequence/kernel-work 区间需要不同最优 static chunk 的真实 mixed regime。

---

# 1. 全新 Scheduler / Kernel / Workload 路线

## 路线 A：Regime-aware guarded adaptive scheduler

不要再用单一 entropy → chunk 规则。新 scheduler 应该改成：

```text
features = {
    sequence_length,
    batch_size,
    layer_id,
    prompt_type_proxy,
    post_conv_entropy,
    variance,
    kurtosis,
    sparsity_ratio,
    active_token_ratio,
    estimated_kernel_time
}

if predicted_gain(best_dynamic_chunk, static_512) > margin:
    use predicted dynamic chunk
else:
    fallback to static_512
```

这叫 **guarded adaptive scheduler**。它的好处是：在当前 LongBench 这种单一 regime 下，它自动退回 Static-512，不会比 best static 慢；只有在检测到明显不同 regime 时才切换 chunk。

审稿人会更容易接受，因为它不是强行说 entropy always helps，而是说：

> adaptive routing is useful only when workload regimes are heterogeneous; otherwise it degenerates safely to the static oracle.

## 路线 B：Learned scheduler baseline

可以做一个非常低成本、明确可行的 learned scheduler：

```text
Input features:
- prompt length
- layer id
- entropy
- variance
- kurtosis
- sparsity
- batch size
- first 1--2 layer probe statistics

Label:
- fastest chunk among {128, 256, 512, 1024, 2048}

Model:
- decision tree
- random forest
- small MLP
- or even a rule table
```

训练数据不需要很大。每个 regime 取 20–50 prompts，每个 prompt/layer 跑 static chunk sweep，标注最快 chunk。然后比较：

```text
Static-512 global oracle
vs.
Entropy rule
vs.
Moment proxy
vs.
Learned guarded scheduler
```

关键是 baseline 要设成：

```text
best single static chunk over the whole mixed workload
```

不要设成 per-regime oracle。否则 adaptive 永远很难赢。

## 路线 C：Kernel 路线必须降 Python overhead

当前 routed 负结果很大原因不是 recurrence-preserving routing 不成立，而是 Python-level entropy / monkey-patch / histogram overhead 太高。论文里也已经写到 full histogram 是负的，sampled histogram 只能在 passive baseline 上接近或小幅快，但 unified static-oracle ablation 仍为负。

下一步 kernel 路线应该是：

```text
1. 保留 patched selective_scan_cuda.fwd_with_chunk_size
2. 把 feature extraction 从 Python histogram 改为 CUDA-side reduction
3. 只计算 cheap features:
   - mean
   - variance
   - absmax
   - sparsity ratio
4. 用 C++/CUDA-side rule table 决定 chunk
5. 用 CUDA Graph / pre-captured chunk path 降 launch variance
```

也就是说，scheduler 不能再是 Python hook 版本。你要把它变成：

> device-side cheap proxy scheduler

而不是：

> Python histogram scheduler

优先尝试路线B 

---

# 2. 新 mixed-regime workload 大概需要多久？

分三档：

| 目标                                        |            时间 | 是否需要 H800 | 产出                                                    |
| ----------------------------------------- | ------------: | --------- | ----------------------------------------------------- |
| **低成本证据筛选**                               |       0.5–1 天 | 不需要       | 找哪些 prompt family 会产生不同 entropy/proxy/chunk bucket    |
| **3090 smoke + feature scan**             |         1–2 天 | 不需要 H800  | 判断是否值得开 H800                                          |
| **H800 full mixed-regime ablation**       | 0.5–1 天付费 GPU | 需要        | 表格：global static oracle vs adaptive/learned scheduler |
| **完整 learned scheduler + quality subset** |         3–5 天 | 需要少量 H800 | 可投稿级补充实验                                              |

我的建议是：

> 先用 1 天做 no-H800 mixed-regime discovery。只有发现至少 3 个 regime 的 static-optimal chunk 明显不同，再开 H800。

成功门槛建议写死：

```text
A regime is useful only if:
1. its best static chunk differs from Static-512 by at least one bucket;
2. the latency gap is > 5%;
3. the mixed-workload weighted adaptive latency can beat global best static by >= 10%;
4. routed quality remains unchanged at task level.
```

如果 1 天内筛不出这样的 regime，就不要继续烧 H800。

---

# 3. Learned / Compiler baseline 的明确可行方案

## Learned baseline：可做，建议做

最简单版本：

```text
Model: DecisionTreeClassifier(max_depth=3)
Input: [L, layer_id, entropy, variance, kurtosis, sparsity]
Output: chunk ∈ {128, 256, 512, 1024, 2048}
Fallback: if predicted margin < 3%, use static_512
```

这个 baseline 好处是实现快，也容易解释。论文里可以说：

> We compare COREY against a lightweight learned scheduler trained from static chunk-sweep labels.

这比只比较 random / no_entropy / hist / proxy 更强。

## Compiler baseline：只做低成本版本，不要承诺 XLA 完整 baseline

XLA/nvFuser 对 `mamba_ssm` 自定义 CUDA extension 不一定能直接优化 selective scan。更明确可行的 baseline 是：

```text
Compiler-assisted static specialization:
1. torch.compile / TorchInductor for surrounding PyTorch graph if supported
2. CUDA Graph replay for static chunk paths
3. pre-captured static_128/static_256/static_512/static_1024 graphs
4. compare:
   - Python scheduler
   - CUDA Graph static
   - CUDA Graph guarded adaptive
```

XLA 可以暂时写成 future work，因为它主要适合 XLA/HLO 图，不适合直接拿来优化当前 custom CUDA selective_scan path。当前项目记录里也把 learned scheduler、XLA/nvFuser/compiler fusion baseline 标为未完成，需要决定是否 future work 或指定低成本 baseline。

---

# 4. 如何探索新的真实 mixed-regime 证据？

现在不能只换几个 LongBench task。因为你已经试过 LongBench、84/164 prompt stress、code/log/table/repetition，仍然多数坍缩到少数 chunk bucket。

建议按下面顺序探索：

## Step 1：构造真实 serving trace，而不是普通 benchmark

候选 regime：

```text
R1: short chat prompts, 256–512 tokens
R2: long document QA, 4k–8k tokens
R3: code repository snippets, 2k–8k tokens
R4: JSON / CSV / log / stacktrace workloads
R5: repeated boilerplate / policy documents
R6: mixed-language Chinese-English prompts
R7: OCR-like forms / tabular records
R8: hash / UUID / base64-heavy logs
```

注意：base64 / UUID / logs 虽然看起来怪，但在真实系统日志、安全审计、RAG indexing 中是合理 workload，比纯 synthetic perturbation 更容易被 reviewer 接受。

## Step 2：先做 feature-only scan

不用先跑完整 generation。先只跑：

```text
prompt -> selected layers -> feature extraction
```

记录：

```text
entropy mean/std
variance
kurtosis
sparsity
predicted chunk
layer-wise chunk diversity
prompt-wise chunk diversity
```

如果所有 regime 还是 chunk=512 或 chunk=1024，就立即停止。

## Step 3：对候选 regime 做 static chunk sweep

对每个 regime 单独跑：

```text
static_128
static_256
static_512
static_1024
static_2048
```

你要找的不是 “adaptive 现在快不快”，而是：

```text
不同 regime 的 best static chunk 是否不同
```

理想结果类似：

```text
short chat       -> best chunk 128
long doc QA      -> best chunk 512
logs/base64      -> best chunk 1024
code             -> best chunk 256
```

如果能找到这个表，adaptive 才有机会打败 global static oracle。

## Step 4：构造 mixed workload 的核心表

最终你需要一张 reviewer 看得懂的表：

```text
Regime        Weight   Best static chunk   Static-512 latency   Regime-oracle latency
Short chat    25%      128                 ...
Long QA       25%      512                 ...
Logs/JSON     25%      1024                ...
Code          25%      256                 ...

Global best static: chunk = ?
Guarded adaptive: selected per regime
Speedup over global static: >=10%
```

这张表比继续堆 histogram ablation 更有价值。

---

# 5. Quality check 怎么“更好”？

这里要改表达：**routing 不应该提高 quality，只应该保持 quality**。所以目标不是 “quality 更好”，而是 “quality evidence 更强”。

当前只有 3 prompts sanity check，论文也承认它不是 task-level LongBench quality study。

建议补：

```text
LongBench routed quality subset:
- narrativeqa: 20 samples
- qasper: 20 samples
- multifieldqa_en: 20 samples
- gov_report: 20 samples
```

报告：

```text
1. greedy exact-match rate
2. token mismatch count
3. task metric delta
4. PPL ratio mean/std/max
5. failure case count
```

通过标准：

```text
metric delta <= 0.001
PPL ratio within [0.995, 1.005]
token exact-match >= 95% if deterministic path is not bitwise identical
```

如果 exact-match 仍是 100%，这会很强。

---

# 6. 推荐执行顺序

我建议按这个顺序推进：

```text
Day 1:
  Build mixed-regime prompt pool.
  Run feature-only scan.
  Find whether chunk diversity exists.

Day 2:
  Run static chunk sweep on only promising regimes.
  Compute theoretical mixed-workload upper bound:
    global best static vs per-regime oracle.

Day 3:
  Implement guarded adaptive scheduler.
  Add learned decision-tree scheduler.
  Run 3090 smoke if possible.

Day 4:
  Open H800 only if expected gain >=10%.
  Run:
    static global sweep
    guarded adaptive
    learned scheduler
    task-level quality subset

Day 5:
  If positive:
    rewrite paper as "regime-aware adaptive routing".
  If negative:
    accept negative-result framing and submit elsewhere / workshop / systems feasibility track.
```

---

# 7. 最关键的写作转向

如果你坚持正结果，论文主 claim 不能再是：

```text
COREY entropy scheduler beats static oracle on Mamba checkpoint inference.
```

要改成：

```text
COREY-style runtime routing is beneficial under heterogeneous serving regimes where a single static chunk is suboptimal across the workload mixture.
```

也就是说，**adaptive/proxy 赢的对象是 global static oracle over a mixed workload**，不是每个单独 regime 的 static oracle。

这个路线是目前最有希望把 Weak Reject 拉回来的方向。当前同类 H800 matrix 已经给出负结论，继续重复跑只会加强负结果；真正有价值的是证明 workload heterogeneity + guarded/learned scheduler 可以把 static oracle 的定义从 “单一 workload 最优” 转成 “混合 workload 全局固定 chunk 最优”。
