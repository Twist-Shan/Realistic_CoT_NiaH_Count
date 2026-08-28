# Internal counter 相关实验汇总

日期：2026-08-26<br>
模型：Qwen3-8B；部分实验同时覆盖 Gemma4-E4B<br>
目的：汇总我们为检验 thinking trace 中是否存在 internal counter 所做的主要实验、控制、结果与可支持的 claim。

## 1. 结论先行

目前的证据**不支持**以下强主张：

> 模型在 thinking trace 中维护一个紧凑的内部寄存器，并对它逐项执行 `counter <- counter + 1`。

目前最符合全部结果的机制是：

> 在 Qwen3-8B 的列表式 counting trace 中，每个有效事件会留下 marker-indexed、分布式的 K/V memory；模型在 item boundary 或最终 answer query 附近对这些记录做晚期 contextual aggregation / reconstruction，形成可解码、可因果搬运的 count-like progress state。该 state 不是已验证的逐步 `+1` 寄存器。

最关键的证据链是：

1. **有 count-like state。** Qwen 的自然 item boundary 可在 held-out seeds 上以 92%–94% exact 解码当前 ordinal；最终 answer-query 的晚层 residual 在 Qwen、Gemma 上都可 9/9 搬运 donor count。
2. **它在局部是因果的。** 同一 boundary 的 donor state 可在 23/30 trials 中被读成 donor；完整 natural commit state 会显著改变 donor-successor 的 retrieval / candidate score。
3. **但它不按 `+1` 递推。** 强行改变当前 state 后，经过同一个下一 item，状态几乎总是回到 receiver 的自然下一值。宽算子扫描中 next-state reset accuracy 为 97.08%，target `+1` accuracy 仅 0.625%。
4. **历史主要存在 K/V event memory 中。** 单个 boundary 不是 bottleneck；marker token 的 K/V splice 可恢复 85.5% 的 later-state contrast，K/V 的主要有效层带为 L20–23。
5. **晚期读取比在线寄存器更可信。** 三事件 factorial 与 early-stop behavior 均显示 marker entries 对 event/count readout 有显著、可累积的因果作用，但冻结的 exact-count/full-sufficiency gate 未通过。

模型范围需要区分：marker-ledger、overwrite 和递推算子实验目前主要是 **Qwen-only**；Gemma 最清楚的结果是晚期 answer-count state 和 full-commit successor-score sensitivity，尚未证明与 Qwen 使用同一 event-memory circuit。

## 2. 证据等级与读数约定

- **Confirmation**：settings、seeds 和 estimand 在查看对应 outcomes 前冻结；可以讨论确认性显著性。
- **Discovery / diagnostic**：用于定位机制或排除解释；即使 CI 不跨 0，也不应写成独立 confirmation。
- **Post-hoc**：读取 confirmation 后新增的分析；只能用于提出下一轮主假设。
- **Candidate-score effect**：teacher-forced 候选序列的 log-probability 改变，不等于 greedy generation 改变。
- **Exact / adoption**：模型实际生成或候选 argmax 是否变成目标；这是比 margin 更强的行为充分性指标。
- “统计上非零”不等于“符合 counter”。例如 retention 的 CI 可以高于 0，但若远低于理论要求的 1，仍然是对 `+1` recurrence 的负证据。

## 3. 实验总览

| 实验族 | 核心问题 | 最主要结果 | 结论等级 |
|---|---|---|---|
| 原始 READ→WRITE→COMMIT→NEXT READ 链 | retrieval、carrier、commit、next query 是否构成因果 loop？ | 两模型 retrieval bank 必要；carrier→commit rescue 为正；full commit 改变 next-query routing | recurrent retrieval-control loop supported；不是 `+1` 证明 |
| Item-span restoration | 单个 item hidden state 是否足以输出当前 count？ | 旧 log-margin 显著；改用 greedy integer 后各层仅 3%–5%，均不显著 | 单 item counter **不支持** |
| Native closing-suffix restoration | count 是否在晚期 closing/readout state 中形成？ | Qwen L32/L33 restored = Source = 92/100；wrong donor 跟随 donor 92/100 | 强 prospective positive，Qwen-only |
| Boundary bottleneck | 单 boundary 是否是可独立读取的 counter cell？ | boundary/item/boundary-bank source ceiling 均为 0；连续 list prefix 为 5/9 | 紧凑 boundary bottleneck **不支持** |
| Boundary probe + transplant | boundary 是否编码并局部使用 count/ordinal？ | probe 92%–94%；same-boundary donor 23/30 | held-out positive |
| Transition equivariance | 修改当前 count 后，下一 item 是否得到 donor+1？ | immediate edit 强；下一步 retention 约 0.004，0/40 exact | `+1` recurrence **明确失败** |
| Distributed residual/KV scans | 递推载体是否更分布式？ | K/V 有 0.09–0.36 analog retention，但 exact 仍近零且不对称 | 分布式 leakage positive；counter negative |
| Recurrence-operator scan | 哪类状态转移最符合数据？ | reset 97.08%，`+1` 0.625%；local arms reset 100% | reset + analog leakage |
| Natural anomaly geometry | geometry 跟 ordinal、unique count 还是 final total？ | 更接近 local ordinal；duplicate 下不稳定 | canonical progress manifold |
| Final answer-query transplant | 最终是否形成 compact count state？ | 两模型晚层 donor adoption 9/9，controls 全保持 receiver | 强 discovery positive |
| Full-commit successor control | natural commit state 是否控制下一 retrieval？ | score/routing 显著随 donor；实际 successor adoption 很弱 | causal score influence；行为充分性未证 |
| Event edit / overwrite | 下一 boundary 如何忽略被 patch 的旧 count？ | current donor 60/60；next reset 54/60；recurrent separation 0/30 | contextual reconstruction |
| Marker K/V circuit | event history 写在哪里、何时读？ | marker K/V 0.835；L20–23 主导；直接 marker→marker edge 失败 | marker cache carrier confirmed |
| Three-entry ledger factorial | 多个 marker entries 是否独立累积并控制 count？ | hidden/behavior specificity 显著；frozen exact/full recovery gate 失败 | event-memory substrate supported；exact ledger 未证 |

## 4. 各实验的设计与结果

### 4.0 原始报告中的 causal loop：支持 retrieval-control recurrence，不等于 arithmetic recurrence

最初的 native-thinking mechanism report 将局部机制拆成 `targeted retrieval -> grammar carrier -> item-end commit -> next targeted retrieval -> answer`。正式实验均以 seed 为独立单位，使用 20 discovery + 10 non-overlapping confirmation seeds；Qwen 冻结 Top-128 retrieval bank，Gemma 冻结 Top-6。

| 原始实验 | 干预与 endpoint | Qwen confirmation | Gemma confirmation | 如何解释 |
|---|---|---:|---:|---|
| Targeted retrieval necessity | 在注册 query 处关闭 selected bank；看下一 city 是否检索失败 | selected/random failure 93.1%/3.4%，差 +89.7 pp | 83.3%/11.5%，差 +71.9 pp | targeted banks 是 next-city retrieval 的必要路径 |
| READ -> carrier | 固定同一 teacher-forced trace，只 mask query heads；测 query 后 carrier deformation | +0.228 [0.138, 0.364]；selected-random +0.089 [0.018, 0.199] | +0.051 [0.041, 0.061]；selected-random +0.009 [-0.008, 0.023] | 两模型 damage 到达 carrier；Gemma 的 bank-identity specificity 较弱 |
| Carrier -> commit | 保持同一 selected-head damage，cumulative clamp clean carrier；测最终 commit 向 clean 恢复 | +0.391 [0.186, 0.656]，约恢复 damage 的 58.5% | +0.030 [0.010, 0.053]，约恢复 51.3% | carrier 对 later commit 有配对因果作用 |
| Commit -> next query（原 5.3） | 把 natural donor commit patch 到 receiver；测 targeted bank 是否转向 donor successor | full-self +4.749 [2.120, 7.960]；full-orthogonal +4.763 [2.132, 7.866] | +0.491 [0.304, 0.688]；+0.126 [-0.046, 0.271] | full commit 控制 routing；Gemma specificity 有限制 |
| Answer source necessity | 等长置零 prompt records 或完整 trace；看 greedy final exact | clean 97%，prompt-record blank 97%，full-trace blank 1% | 70%，70%，12% | 已生成 trace 是 final answer 的主要自然信息源 |
| Terminal grammar state -> answer margin | 在固定受损 trace 中恢复最后 item 的 frozen marker-core state | restoration +2.375 [0.700, 4.363]；vs ordinary +2.000 [0.612, 3.862] | +1.922 [0.922, 2.938]；+0.992 [0.289, 1.620] | controlled local bridge 成立；未证明 free-running sufficiency |

这些结果确实接上了一条 `READ -> WRITE/CARRIER -> COMMIT -> NEXT READ` 的因果控制 loop；但它只要求 commit state 改变下一次 retrieval routing，不要求数值状态满足 `S_{k+1}=S_k+1`。后续 boundary-equivariance、operator scan 和 greedy successor assay 正是针对这一逻辑缺口；它们显示 routing score 可被 donor 影响，但 count offset 不随下一 item 保留。因此原来的 “recurrent counting pathway” 最好改写为 **recurrent retrieval-control pathway carrying ordinal/progress information**。

### 4.1 Item hidden-state restoration：从“margin 显著”到“greedy exact 不显著”

**更早的 standardized no-running-index assay。** 在 outcome-blind teacher-forced `- City: score` grammar 中，将 prompt records 和全部 items 换成等长普通 tokens，再 cumulative restore 第 k 个完整 item span。Qwen exact 从 0.062 到 0.150，gain +0.087 [0.025, 0.163]，target-margin gain +1.906 [1.324, 2.867]；Gemma exact 从 0.013 到 0.050，gain +0.037 [0.000, 0.113]，margin +0.515 [0.262, 0.761]。两模型均未达到预设的 old-HTML magnitude gate。这是 graded information 的正结果，但不是 standalone counter sufficiency。

**做法。** 在 N=10 的 list-shaped traces 中，以等 token 长度 scrub prompt needles、pre-list reasoning、显式 index/running count；构造保留 item 1...k 的 Source 与全部 item 被替换的 Blank。将 Source 的 item-k span 在单个 decoder-block input layer transplant 到 Blank。每模型使用 20 discovery + 10 held-out confirmation seeds。

**早期 candidate-score 结果。** 旧主指标是答案数字加 termination 的候选序列 log-score，而不是实际 greedy 输出。

| 模型 | 三层平均 margin effect | 95% CI | sign-flip p | Source / Blank / Restored exact |
|---|---:|---:|---:|---:|
| Qwen | +1.036 | [0.463, 1.678] | 0.00293 | 47.0% / 11.0% / 16.7% |
| Gemma | +0.429 | [0.219, 0.674] | 0.00488 | 50.0% / 8.0% / 18.7% |

这证明 item span 可传递 graded count-correlated information，但旧 score 混入了 termination 和未恢复的 Blank KV context。

**校正后的 greedy-integer rerun。** 使用同一 frozen cohort，让模型在原生 `Total:` 后实际 greedy generate integer，并逐层报告。

- Qwen：L16/L12/L0 restored exact = 5/100、5/100、3/100；Holm p 均为 0.752。
- Gemma：L16/L20/L4 = 5/100、3/100、4/100；Holm p = 0.502、0.502、0.378。
- Qwen 成功只出现在 k=1–2；Gemma 只出现在 k=1–3。

**结论。** “单个 item span 是覆盖 1...10 的 standalone counter”未得到支持。诊断性 all-items restoration 比 single-item 更强（Qwen margin-gap closure 76.1% vs 44.9%；Gemma 87.8% vs 47.9%），更符合分布式证据。Gemma 的 residual-only identity control 还因 per-layer embedding/shared K/V 未被 patch 而失败，因此其 localization 百分比只能描述性使用。

### 4.2 Native closing-suffix restoration：晚期 count readout 的强正结果

**做法。** 不再 patch item 本身，而是捕获 item k 后最小原生 reasoning-close + `Total:` suffix 的 hidden states，并一次 transplant 到几何匹配的 Blank；行为指标是实际 greedy integer。Discovery 后冻结 L28/L32/L33，并新采 outcome-blind 10-seed Qwen confirmation。

| 条件 | L28 | L32 | L33 |
|---|---:|---:|---:|
| Source | 92/100 | 92/100 | 92/100 |
| Blank | 0/100 | 0/100 | 0/100 |
| Matched suffix restoration | 90/100 | **92/100** | **92/100** |
| Wrong-k donor：跟随 receiver target | 1/100 | 1/100 | 1/100 |
| Wrong-k donor：跟随 donor | 91/100 | **92/100** | **92/100** |

L32/L33 在每个 seed 上都精确复现 Source；artifact 没有单独报告 formal p，但这是 prospective confirmation，且 seed-level fidelity 一致。所有 confirmation traces 原本是 indexed lists，再做等长去 index，因此它证明的是 counterfactual suffix-state sufficiency，不是自然无编号列表中的持续 counter。Gemma 因 Source ceiling 不稳定，没有做对应 confirmation。

**结论。** Qwen 在晚期 closing/readout suffix 中形成了可搬运的 count state；强 code 位于 readout 后期，而非单个 bullet span。

### 4.3 Full-native boundary bottleneck：单个 boundary 不是独立 memory cell

**做法。** 保留完整 native trace 和原生 `Total:` 位置，用 4D attention graph cut 让 suffix 只能读取指定 memory span，未来 items/recap 虽仍在序列中但不可达。

- one boundary、last-4 tokens、complete item、all boundaries through k、separated item spans：Source exact 均为 0。
- 只有连续 list prefix through k 给出 partial source ceiling：5/9；Blank 0/6。
- 在无 boundary source ceiling 的情况下，boundary donor transplant 和 counted/no-op transition 均为 0。

这是 3-seed smoke，不能作强 null confirmation；但它说明 local boundary 并非可由 suffix 独立消费的 compact counter，连续列表结构更重要。

### 4.4 Boundary geometry 与 same-site causal transport

**做法。** Qwen-only；完整 native traces，等长 scrub 显式 count cues，不加 `Total:`。在每个 item 后的自然 boundary 记录 state，20 discovery seeds 拟合 probe，10 held-out seeds 固定读取 L15/L16/L24。

| 指标 | 结果 |
|---|---:|
| L15 / L16 / L24 frozen probe exact | 94% / 93% / 92% |
| MAE | 0.06 / 0.07 / 0.10 |
| donor 2/5/8 在同一 B5 的 persistence | 23/30 |
| 处理 item 6 后的 donor-conditioned `+1` | 7/30 |
| donor 2→3 / 5→6 / 8→9 | 0/10 / 7/10 / 0/10 |

7 个 next hits 全是 self-donor 5，即 receiver 本来就会自然到 6；没有 donor-dependent increment。

**结论。** boundary 上存在高度 ordered、可局部搬运的 count/ordinal representation；但它是否表示 semantic count 仍与 bullet ordinal/position 混淆，并且它不是已验证的递推 state。

### 4.5 Natural anomaly geometry：更像 local enumeration progress

**做法。** 用不含 anomaly seeds 的 clean traces 冻结 probe，再读取 14 个 held-out anomaly seeds、17 个自然异常 requests；无 patch、无 suffix。

| 层 | MAE to local ordinal | MAE to final answer | MAE to gold total | seed-level 更接近 local ordinal |
|---|---:|---:|---:|---:|
| L15 | 0.58 | 2.75 | 2.67 | 13/14 |
| L16 | 0.55 | 2.74 | 2.66 | 13/14 |
| L24 | 0.52 | 2.92 | 2.78 | 13/14 |

三条自然 omission traces 最终只列出 8/8/9 项、gold 为 9/9/10；L16 最终 coordinate 为 7.98/8.27/8.47。clean adjacent increment 约为 +1，但 13 个 duplicate transitions 高度不稳定：L24 median +0.367，6/13 接近 no-op，0/13 接近标准 +1，并有回退和大跳。

**结论。** geometry 稳定跟随 trace 的 local ordinal/progress，而不是 gold total；duplicate 下的失稳不支持一个稳健 semantic counter。

### 4.6 `+1` transition 与 carrier 扩展：系统性负结果

这些实验都保持原 task、token stream 和 native continuation 不变；先确保当前 state 被实际改变，再读取下一个或后续 boundaries。

| 干预 | 当前 first stage | 下一步结果 | 判断 |
|---|---:|---:|---|
| Adjacent count-subspace swap | 8/9 donor exact | 3/9 donor+1；全是 clean prediction | fail |
| Relative transition-equivariance，10-seed confirmation | 38/40 current predictions changed | retention 0.00399 [0.00016, 0.00767]；next 0/40 changed | strong fail |
| Last-four item suffix6 KV field，held-out | 有方向性 first stage | next retention 0.1187；-1=0.193，+1=0.030；无正确离散改变 | weak asymmetric leakage |
| Boundary + last-four suffix6，discovery | current 40/40 exact | retention 0.0559 [0.0403, 0.0732]；next 0/40 changed | fail |
| Full-history K+V，natural scale | current 220/240 exact | retention 0.0913 [0.0841, 0.0987]；next 0/240 exact | specific leakage |
| Full-history K+V，8x | current 237/240 | retention 0.2327 [0.2043, 0.2592]；next 10/240 exact | saturates far below 1 |
| Unified residual+K/V，discovery | current→next slope 0.362 [0.291, 0.445] | next 0/40 exact；successor semantic score 朝反方向 | feature persistence, not operator |

Full-history K/V 的 aligned-minus-orthogonal effect 在 scale 1 为 0.0956 [0.0855, 0.1056]，说明 analog leakage 是 count-specific、不是 generic noise；但它仍远低于预设 recurrence gate 0.8–1.2。Token-resolved movie 进一步显示：offset 在下一 item 的第一个 progress bin 已从 1 降到 0.084，最终 boundary 仅 0.091；没有隐藏的平滑 +1 ramp。

### 4.7 Recurrence-operator scan：reset 明显优于 `+1`

**做法。** Qwen 20 discovery seeds。Whole-state 扫 receivers 3/5/7 × donors 1...9，共 480 transitions；局部 residual、residual+K/V 及 norm-matched controls 共 960 transitions。比较 reset、identity、固定 shifts、`+1` 和多种连续 leaky-reset operators。

| Carrier | N | Current target | Next reset | Target `+1` | OLS retention | IV retention |
|---|---:|---:|---:|---:|---:|---:|
| Whole state | 480 | 100% | **97.08% [93.54%, 99.38%]** | **0.625% [0%, 1.25%]** | 0.071 | 0.069 |
| Residual subspace | 240 | 34.58% | **100%** | 0% | -0.002 | -0.005 |
| Residual + K/V | 240 | 35.42% | **100%** | 0% | 0.265 [0.218, 0.321] | 0.333 [0.282, 0.391] |

在 residual 和 residual+K/V 中，分别有 83、85 个 trials 成功把 current argmax 改为 target；这些成功 first-stage trials 的 next donor+1 仍为 0。Whole-state soft RMSE 也由 donor-lookup/leaky reset 类算子最好，`+1` 最差。

**结论。** 覆盖的算子族中，最佳模型是 position/content-conditioned reset + carrier-dependent analog leakage，而不是 `+1` recurrence。该扫描是 discovery，不是独立 held-out confirmation，但效应规模与前述确认性 transition 结果一致。

### 4.8 Final answer-query readout：两模型都形成 compact terminal count state

**做法。** 完整 clean native trace，不截断、不加 suffix、不改 mask；在最终数字前的原生 answer-query token 单次 transplant donor post-block residual。每模型 3 seeds × counts 3/6/8，属于 discovery pilot。

- Qwen：L24 3/9 donor adoption，L28 7/9，L32/L35 9/9。
- Gemma：L32 5/9，L36/L40/L41 9/9。
- 最终层 full greedy generation：两模型均 9/9 跟随 donor。
- Qwen 180 个、Gemma 216 个 self/same-count layer controls 全部保持 receiver。

**结论。** 完整 trace 后，两模型的晚层 answer-query residual 都含有充分、近似标量的 count readout state。因为这是 3-seed discovery，且没有完成 source attribution，它不能说明 state 来自递归 counter。

**Broad-retrieval follow-up。** 某些自然 heads（尤其 Gemma）会广泛 attention 到多个 trace endpoints；但 whole-head ablation 不 source-specific，Qwen 首次 exact failure 需 512/1152 heads。精确 endpoint-key mask 后 18/18 仍正确。Value replacement 只在 Gemma 的 terminal item tails 上给出稳定 margin effect（all tails -2.62；last tail -1.56；earlier tails -0.32），Qwen 不跨 seed 稳定。因此只能写“存在 late readout，部分 heads 描述性地 broad-attend”，不能写“已因果证明从全部早期 items 求和”。

### 4.9 Full natural commit state 对下一 retrieval / successor 的控制

#### A. Targeted retrieval-bank specificity

**做法。** 在 natural item boundary patch 完整 commit vector；与 self、三个完整 `delta h` 等范数正交控制、opposite delta、wrong-ordinal natural donor 比较。20 discovery + 10 confirmation seeds；两 donor offsets 在 seed 内平均。

| 模型 confirmation | Full vs self | Full vs norm controls | Donor-identity double difference | Formal gate |
|---|---:|---:|---:|---|
| Qwen | 4.747 [2.071, 7.936] | 4.511 [1.869, 7.685] | 4.518 [2.369, 7.098] | **PASS** |
| Gemma | 0.470 [0.314, 0.628] | 0.063 [-0.086, 0.191] | 0.511 [0.247, 0.775] | **FAIL norm-control gate** |

这些数值是各模型 frozen targeted attention bank 的 summed-mass units，不能跨模型比较绝对大小。Qwen 支持 natural full commit state 对 donor-specific next-query routing 的因果控制；Gemma 的 donor identity 成立，但未证明自然 donor 优于 generic 等范数方向。

#### B. Unindexed native next-bullet candidate scoring

**做法。** 使用无显式 index 的 native bullet traces，比较 receiver 的十个原生 bullet candidates；full donor、self、norm controls 和 wrong natural donor 均为 on-trace interventions。主要结果为候选完整序列 log-probability 差。

| 模型 | Full donor vs self | vs norm controls | Donor identity | 实际 donor-successor adoption |
|---|---:|---:|---:|---:|
| Qwen | +8.60 [4.69, 12.48], p=0.00195 | +8.55 [4.58, 12.53], p=0.00195 | +3.21 [1.10, 6.04], p=0.00977 | **0/20**；二选一也 0/20 |
| Gemma | +5.31 [2.83, 7.90], p=0.00391 | +4.50 [2.12, 7.13], p=0.00586 | +9.04 [7.87, 10.46], p=0.00195 | **3/20** |

Qwen 的 donor-vs-receiver log-odds 从 self 的 -31.80 提高到 -23.20；Gemma 从 -20.19 提高到 -14.88。效应很大且显著，但没有跨过决策边界。

**结论。** Full commit state 携带 context-bound、donor-specific successor-control information；不能据此写“成功生成/控制下一项”，更不能把它当作 `+1` arithmetic operator。它也可能是 ordinal pointer、item identity 或 retrieval plan 的联合 state。

### 4.10 Event edit 与 contextual reconstruction（“overwrite”）

**做法。** 在 B5 将 whole state clamp 为 donor 4 或 6，同时让后续 trace保持 original、插入一个 valid event、或删除一个早期 event；明确区分 donor recurrence 与 edit-specific prefix reconstruction。

| 汇总（10 seeds） | 结果 |
|---|---:|
| current donor exact | 60/60 |
| next equals edit-specific target | 52/60 |
| next reset to paired clean run | 54/60 |
| donor 4/6 给出相同 next label | 26/30 pairs |
| 应有的 recurrent separation=2 | **0/30** |

仅看最后 5 个 replication seeds：current donor 30/30；next reset 25/30；donor-invariant 12/15；recurrent separation 0/15。固定物理位置交换 item identity 时，L15/L16 仍 8/8 跟随 physical-position label，0/8 跟随 future-content identity。

**结论。** 这不是神秘的“主动纠错指令”，而是 intact distributed prefix 对局部矛盾 state 的支配：下一 boundary 从 marker/payload grammar 和历史 events 重新形成 ordinal-like state。

### 4.11 Marker-keyed event memory：cache splice、K/V 与 layer band

#### A. 单事件 cache splice confirmation

valid 与 markerless insertion 具有相同长度、payload、closing、位置、mask，只在一个 marker token 不同；从同一 pre-insertion cache 出发，在相同绝对位置交换 K/V。

| Spliced region | Frozen 7-seed confirmation progress |
|---|---:|
| Full inserted event | 1.000 |
| **Marker token only** | **0.855** |
| Payload only | 0.084 |
| Closing only | 0.055 |
| Previous boundary / pre-event control | 0.000 |

Marker effect 7/7 positive，p=0.0233；marker-minus-closing = 0.800、marker-minus-payload = 0.771，均 p=0.0233。说明 later state 的主要 branch difference 存在 marker K/V，而不是 closing commit cell。

#### B. Marker K-only / V-only 与 layer-band confirmation

8 个 untouched confirmation seeds；冻结三项检验并做 Holm correction。

| Carrier / test | Effect | Significance |
|---|---:|---:|
| Marker K-only, all layers | 0.276 | 8/8 positive，描述性 |
| Marker V-only, all layers | **0.500** | raw p=0.0117，Holm p=0.035，pass |
| Marker K+V, all layers | **0.835** | 8/8 positive |
| Marker K+V L20–23 | 0.540 | selected-minus-other-bands=0.498，Holm p=0.035，pass |
| 其他五个 pre-read bands 的 K+V 均值 | 约 0.043 | — |
| `target marker -> inserted marker` edge specificity | **-0.043** | 8/8 negative，Holm p=0.035；正向 gate **fail** |

因此可说 V 携带较多 branch-sensitive event content、K 提供 address/match，且晚期 L20–23 完成主要整合；不能说已经定位到一条直接 marker-to-marker edge。`target marker -> inserted closing` 的 0.501 是读取 confirmation 后发现的 post-hoc 强线索，不能升级为确认性结论。

### 4.12 三事件 factorial 与早停 behavior

**做法。** 在原 item 6 前复制三个 events；只有三个 inserted marker tokens 独立切换，形成严格等 geometry 的 `2^3=8` cells。Cache arm 从 `111` 向 `000` 移植任意 marker-entry subset，从而测试三条 entries 是否可独立累积。

#### Hidden-state confirmation（7 untouched seeds）

| Frozen metric | Result | Gate |
|---|---:|---|
| Marker-entry specificity | **0.116**, 7/7，p=0.0233 | pass |
| 前两条 entry specificity | **0.152** | pass |
| 三条 marker full-subset progress | **0.519** | pass |
| Textual endpoint-axis slope / marker | **0.312**（理想线性 1/3） | pass |
| 三个 singleton slot means | 0.292 / 0.079 / 0.070 | 全部 >0 |

这排除了“只有最后一个 marker 有效”，并支持 multi-entry event-history readout。但 frozen natural count probe 在该 OOD factorial 上只有 0.057 count/marker、strict accuracy 0.125，因此不能说 local hidden state 已精确编码 6/7/8/9。

#### Behavioral confirmation（另一组 7 seeds）

在 physical item 6 后立即 early-stop，移除未来 items/recap，附加 token-exact native close + `Total:`；endpoint 为 counts 1–10 的完整 answer+termination candidate score。

| Frozen metric | Result | Gate |
|---|---:|---|
| Cache marker-entry specificity | **0.122**, 7/7，p=0.0233 | pass |
| 前两条 entry specificity | **0.129** | pass |
| Textual expected-count slope / marker | **1.116** | pass |
| Textual `000->111` expected-count contrast | **3.397** | pass |
| Marker full-subset recovery | **0.247**，要求 >=0.35 | **fail** |
| Strict candidate exact accuracy | **0.250**，要求 >=0.50 | **fail** |

七个 seeds 的 Hamming-level expected-count curve 都单调，但 1–2 markers 时明显 undercount，呈现阈值/协同，而不是三个 exchangeable 的独立 `+1`。

#### Post-hoc region scan

| Region | Behavioral-axis recovery |
|---|---:|
| Marker K-only / V-only, all layers | 0.065 / 0.102 |
| **Same-marker K+V, all layers** | **0.810** |
| Closing K+V | 0.148 |
| **Marker + closing K+V** | **0.894** |
| Non-marker event K+V | 0.421 |
| Full event K+V | 0.975 |

K/V joint synergy 为 `0.810 - (0.065 + 0.102) = 0.643`，很符合 key-content binding；但这是 post-hoc，不能挽救冻结的 full-sufficiency/exactness claim。

### 4.13 Anthropic-style count-manifold 复分析（exploratory）

**动机。** Anthropic 的 [line-breaking 研究](https://transformer-circuits.pub/2025/linebreaks/index.html)明确指出，高精度标量 probe 不代表模型沿一条直线表示 count；同一数量可能由 sparse feature family、低维 subspace、弯曲 manifold 和多类 probe distribution 共同刻画。我们因此不再只看 frozen probe 的 argmax/soft count，而是在完整 10-way probe-score vector 中重分析自然 count trajectory，并检验 textual marker 与因果 K/V splice 的位移方向。

**自然 N=10 trajectory。** 使用与 ledger 实验相同的 frozen probe，对 held-out clean N=10 traces 做 softmax-invariant score centering 和 within-seed centering。

| Layer | PC1 between-count variance | rank for 90% | linear trajectory R² | fixed `+1` vector R² | relevant tangent LOSO cosine |
|---:|---:|---:|---:|---:|---:|
| L15 | 0.636 | 3 | 0.604 | 0.039 | 0.328 |
| L16 | 0.679 | 3 | 0.602 | 0.030 | 0.359 |
| L24 | 0.310 | 5 | 0.263 | 0.013 | 0.475 |

在 probe-visible subspace 中，count trajectory 明显不是一条直线，同一个固定 increment vector 只能解释约 1–4% 的 sample transition energy。但跨 seed 的 local tangent 也只有中等稳定，因此这不是一个已经确认的 universal manifold。

**Causal displacement。** 在 L24 target boundary，真实 textual marker toggle 与 held-out natural `c→c+1` tangent 的平均 cosine 为 0.333 [0.304, 0.358]，三 marker 累积 `000→111` cosine 为 0.512。K/V splice 与跨-seed natural tangent 接近零，因 tangent reliability 有限，不能单独据此判定“无 count information”。更公平的 exact-matched comparison 是：同一 seed、marker slot、Hamming edge、layer 和 landmark 下，将 cache 位移与真实 textual toggle 的完整 score-vector 位移比较：

| L24 target-boundary cache carrier | matched one-edge cosine | textual-direction scale | `000→111` matched cosine |
|---|---:|---:|---:|
| Marker V, all layers | **0.442 [0.405, 0.465]** | 0.263 | **0.748** |
| Marker K, all layers | 0.256 [0.188, 0.318] | 0.083 | 0.479 |
| Marker K+V, L20–23 | 0.340 [0.289, 0.392] | 0.112 | 0.598 |
| Closing K+V, all layers | 0.140 [0.045, 0.229] | 0.031 | 0.161 |

All-layer V 与仅 L20–23 的 K+V 不是同剂量比较，不能据表中绝对值排序 K/V synergy。方向上，V 最稳定地复现 textual branch effect，K 提供较小但正向的 matched component；三 marker endpoint 的 alignment 明显高于单 edge，符合分布式、协同/阈值式 aggregation，而不是三个完全 exchangeable 的固定 `+1` writes。

**限制。** 这是看到既有结果后的 retrospective analysis，只覆盖 frozen probe row span；它既不是 raw-residual manifold 分析，也不是 sparse-feature/transcoder attribution graph。K/V 只通过其 downstream residual effect 被观察，尚未完成 head-specific QK/OV decomposition。

## 5. 推荐 claim

### 中文

> 在 Qwen3-8B 的列表式 counting trace 中，事件历史以 marker-indexed、分布式 K/V records 的形式保存；晚期 boundary 和 answer-query computation 会对这些 records 做 contextual aggregation/reconstruction，形成有序、可解码且局部因果有效的 count-like progress state。对当前 state 的干预在下一 item 中主要被重置，而非按 `+1` 等变传播，因此现有证据不支持一个持续维护的标量 internal counter。

若需要跨模型表述，应收窄为：

> Qwen3-8B 与 Gemma4-E4B 在完整 trace 后都会形成可搬运的 late answer-count state；目前只有 Qwen 有较完整的 marker-indexed event-memory 机制证据。

### English

> In Qwen3-8B, list events are stored as marker-indexed, distributed K/V records and are contextually aggregated or reconstructed at late item boundaries and the final answer query, yielding an ordered and causally effective count-like progress state. Intervened count offsets are predominantly reset at the next item rather than propagated equivariantly by a `+1` update, arguing against a maintained scalar counter register.

### 不应使用的表述

- “The model implements `counter <- counter + 1` in hidden states.”
- “A single item boundary is the recurrent counter register.”
- “We have confirmatorily identified an exact marker-only ledger algorithm.”
- “Broad attention over all early trace items has been causally proved to compute the answer.”
- “Qwen and Gemma use the same internal counting circuit.”

## 6. 还缺什么才能升级 claim

最小的下一轮确认应在全新 outcomes 上预注册：

1. primary carrier：same-marker K+V across all layers；
2. controls：K-only、V-only、closing-only、non-marker event K+V；
3. secondary：marker+closing K+V；
4. endpoint：immediate early-stop greedy/exact count，而不仅是 candidate margin；
5. 同时检验 full recovery、K×V synergy 和相对 non-marker specificity；
6. 若继续研究 full commit control，主要指标必须是 greedy next-bullet donor adoption，而非只看 log-score 提升。

在这些行为充分性标准通过前，建议使用 `distributed event memory`、`count-like progress state` 或 `late count readout`，避免把主机制简称为 arithmetic internal counter。

## 7. 主要结果文件

- [原始 native-thinking mechanism report](../reports/NiaH_Native-Thinking_report.html)
- [Item restoration 原始 confirmation](../results/v5_marker_scrubbed_list_counterfactual_restore_20260824/SUMMARY.md)
- [Greedy exact 校正](../results/v5_marker_scrubbed_list_greedy_restore_20260824/corrected_same_cohort/SUMMARY.md)
- [Closing-suffix confirmation](../results/v5_marker_scrubbed_list_greedy_restore_20260824/final_confirmation/SUMMARY_FINAL.md)
- [Boundary bottleneck](../results/v5_boundary_bottleneck_20260824/SUMMARY.md)
- [Boundary probe、equivariance、KV carrier、final readout 与 anomaly geometry](../results/v5_boundary_counter_probe_20260824/SUMMARY.md)
- [Unified carrier transition](../results/v5_unified_carrier_transition_20260825/discovery20_receiver5_scale1_v2/REPORT.md)
- [Recurrence operator scan](../results/v5_recurrence_operator_scan_20260825/REPORT.md)
- [Overwrite/contextual reconstruction](../results/v5_overwrite_mechanism_20260825/REPORT.md)
- [Marker ledger intuitive mechanism](../results/v5_overwrite_mechanism_20260825/INTUITIVE_MECHANISM.md)
- [Marker K/V、layer band 与 edge confirmation](../results/v5_overwrite_mechanism_20260825/MARKER_CIRCUIT_RESULTS.md)
- [Three-entry factorial 与 behavior](../results/v5_event_ledger_20260825/REPORT.md)
- [Anthropic-style count-manifold 复分析](../results/v5_anthropic_count_manifold_20260826/REPORT.md)
- [Full-commit specificity](../work/v5_native_count_stream/full_commit_specificity_20260825/RESULTS.md)
- [Qwen unindexed successor estimands](../work/internal_counter_unindexed_cohort_20260825/unindexed_full_commit/Qwen3-8B/confirmation_successor/analysis/estimands.json)
- [Gemma unindexed successor estimands](../work/internal_counter_unindexed_cohort_20260825/unindexed_full_commit/Gemma4-E4B/confirmation_successor/analysis/estimands.json)
