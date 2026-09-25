我先做一个“问题→证据”的对照核查：从 revision_suggestions.tex 的 Potential Issues 条目逐条映射到当前 main.tex、appendix.tex、ref.bib 和 progress.md 的已落地修改，然后给你一份可追溯的清单。

Read [](file:///c%3A/source/COREY_Transformer/docs/revision_suggestions.tex#40-40), lines 40 to 128

Searched for regex `round\(|operatorname\{round\}|H_\{\text\{ref\}\}|policy_off|policy_static|policy_corey|Table~\\ref\{tab:routed_ablation\}|Code and data` (`**/paper/main.tex`), 7 results

Searched for regex `legacy default|H_\{\\min\}|H_\{\\max\}|argmax|prev\[|fwd_with_chunk_size|Table~2|Table~\\ref\{tab:real-gpu-three-policy\}|round\(|operatorname\{round\}|T\(e\)` (`**/paper/appendix.tex`), 19 results

Searched for regex `yang2024parallelizing|Hardware-efficient linear attention kernels for state-space-like models` (`**/paper/ref.bib`), 1 result

Searched for text `PAT 审稿意见系统性修正` (`**/docs/progress.md`), 1 result

我继续补充“可核对证据行号”，重点核查你最关心的每条 Potential Issues 是否真的在正文/附录里已改，另外会标出仍未闭环或可能回退的项。

Read [](file:///c%3A/source/COREY_Transformer/paper/main.tex#118-118), lines 118 to 140

Read [](file:///c%3A/source/COREY_Transformer/paper/main.tex#48-48), lines 48 to 64

Read [](file:///c%3A/source/COREY_Transformer/paper/main.tex#332-332), lines 332 to 350

Read [](file:///c%3A/source/COREY_Transformer/paper/appendix.tex#760-760), lines 760 to 840

Read [](file:///c%3A/source/COREY_Transformer/paper/appendix.tex#862-862), lines 862 to 910

Read [](file:///c%3A/source/COREY_Transformer/paper/ref.bib#120-120), lines 120 to 134

Searched for regex `unless .*permutation|Theorem 2|doubly-stochastic|strict inequality` (`**/paper/main.tex`), 4 results

Searched for regex `H\(R\)|U\(R\)|region|input tensor|output tensor|policy_off|policy_static|policy_corey|without explicit warmup|first-call JIT|0\.413|3\.30/64|dispatch cost|Table 37|17\.0|17\.5` (`**/paper/appendix.tex`), 22 results

Read [](file:///c%3A/source/COREY_Transformer/paper/main.tex#140-140), lines 140 to 190

Read [](file:///c%3A/source/COREY_Transformer/paper/appendix.tex#700-700), lines 700 to 760

Read [](file:///c%3A/source/COREY_Transformer/paper/appendix.tex#940-940), lines 940 to 1010

Read [](file:///c%3A/source/COREY_Transformer/docs/progress.md#88-88), lines 88 to 180

Read [](file:///c%3A/source/COREY_Transformer/paper/appendix.tex#136-136), lines 136 to 190

Searched for text `Exact majorization test as an alternative diagnostic` (`**/paper/appendix.tex`), 1 result

Read [](file:///c%3A/source/COREY_Transformer/paper/appendix.tex#908-908), lines 908 to 932

Searched for text `Theorem 编号不一致` (`**/docs/progress.md`), 1 result

Searched for text `Algorithm 1：未存储 argmax predecessor` (`**/docs/progress.md`), 1 result

Searched for text `where the distinction between raw and normalized is clear from the column units` (`**/paper/main.tex`), 1 result

Searched for text `The dispatch cost per kernel launch is $\approx0.052$` (`**/paper/appendix.tex`), 1 result

已做完对照核查。结论是：`# Potential Issues And Suggestions` 里的核心问题大部分已经落地，少数是“部分处理”或“当前文本有回退风险”。

**逐条对应（含证据与修改依据）**
1. Eq.2 floor/round 矛盾：已解决  
证据：公式已改为 `round`，见 main.tex。  
依据：数学一致性，代入 $r=0.83$ 应得到 512 而不是 256。

2. Triton 映射公式 448 vs 512：已解决  
证据：附录已改为 2 的幂映射，并明确旧公式废弃，见 appendix.tex、appendix.tex。  
依据：与主文 Eq.2 保持一致、可复现一致性。

3. Theorem 2 的 and/or 逻辑错误：已解决  
证据：主文 informal theorem 已是 “or”，见 main.tex。  
依据：与概率论/majorization 条件逻辑一致，和 formal proof 对齐。

4. Region entropy \(H(R)\) 定义不清：已解决  
证据：明确写成 \(H(R)=\widehat H(z_{in}(R))\)，见 appendix.tex。  
依据：可复现性（DP 目标函数必须明确输入张量定义）。

5. 记号混用（\(H,\widehat H,\widetilde H\)）：已解决  
证据：主文新增 notation 约定段，见 main.tex。  
依据：数学可读性与跨章节一致性。

6. Adaptive threshold 中 \(H_{min}, H_{max}\) 未定义：已解决  
证据：已显式定义 \(H_{min}=0, H_{max}=\log K\)，见 appendix.tex。  
依据：公式完整性、边界条件明确。

7. Algorithm 1 未记录 argmax predecessor：已解决（代码已修）  
证据：已加入 `prev[i]` 回溯记录，见 appendix.tex。  
依据：动态规划路径恢复的正确性。  
备注：进度文档曾标为遗留，见 progress.md，但源码已修，进度记录略滞后。

8. Hadamard 熵下降机制解释不自洽：已解决  
证据：主文与附录均改为 “非零均值/DC 分量 + 动态分箱范围放大” 机制，见 main.tex、appendix.tex。  
依据：与观测到的 \(\Delta H<0\) 一致的机制解释。

9. “Exact majorization test” 建议：已吸收为补充诊断  
证据：新增 exact majorization paragraph，见 appendix.tex。  
依据：理论严谨性增强（从近似 Sinkhorn 扩展到可精确判别条件）。

10. Surrogate dispatch cost 0.413ms 计算错误：已解决  
证据：改为约 0.052ms，见 appendix.tex。  
依据：与 static-64 的 64 calls 一致的算术修正。

11. 17.5% 数值错误：已解决  
证据：现为 17.0%，见 appendix.tex。  
依据：按表中数值重算后的正确百分比。

12. Table 2/23 latency-speedup 解耦说明缺失：已解决  
证据：Table 2 caption 已明确 speedup 来源与 isolated kernel latency 区别，见 main.tex。  
依据：避免“同表不同度量”引发误读。

13. 错误表号引用（Table 1→Table 2 / Appendix 表）：已解决  
证据：已改为 “Table 2 of the main paper” 与附录正确表，见 appendix.tex、appendix.tex。  
依据：交叉引用准确性。

14. 混合测量行误导（mixed-regime 表含 H800 单条）：已解决  
证据：主文表已显式拆分并注明“不同硬件不可直接比较”，见 main.tex。  
依据：实验可比性与读者防误解。

15. `yang2024parallelizing` 引用条目混入标题：已解决  
证据：该条目当前无错误 note 混入，见 ref.bib。  
依据：参考文献准确性。

16. Code URL 缺失：当前未闭环/有回退  
证据：abstract 的 URL 目前仍被注释，见 main.tex。  
依据：NeurIPS checklist 与正文一致性要求。  
说明：这条在进度里写过“已取消注释”，见 progress.md，但当前文件状态显示已注释，建议视为回退项。

17. Table 15 计算口径不一致（warmup/JIT）：部分处理  
证据：表注已明确口径差异，见 appendix.tex。  
依据：透明披露已做，但“完全同口径重跑”本身尚未完成，严格来说属于部分修复。

18. Theorem 编号一致性：仍是低优先级遗留  
证据：进度文件仍记录为遗留，见 progress.md。  
依据：主要是排版编号层面，不影响当前结论正确性。

**你问的“修改依据”可归纳为 6 类**
1. 数学一致性：公式代入必须与表格/文字结论一致。  
2. 可复现性：定义、变量、算法回溯步骤必须可执行。  
3. 逻辑正确性：定理条件的逻辑连接词与 proof 一致。  
4. 度量一致性：速度、延迟、基线口径必须同一语义。  
5. 可比性与公平性：跨硬件/跨流程数据必须显式分栏或注释。  
6. 学术可追溯性：引用、表号、URL 必须与正文声明一致。

如果你要，我可以下一步直接给你一份“仅剩未闭环项”的最小修复清单（1页以内），按“必须修 / 可留作camera-ready”分组。