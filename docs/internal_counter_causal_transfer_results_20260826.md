# Native-thinking internal counter：无显式 index 的因果实验结果

日期：2026-08-26<br>
模型：Qwen3-8B<br>
状态：自然 frozen-prompt discovery、冻结确认及 broad generality scan 均已完成

## 一句话结论

目前不能 claim 模型在无显式 index 的 Native-thinking 中实现了一个稳定、可迁移、逐步 `+1` 的 internal counter。最稳健的正结果是：**event-closing hidden state 对 donor count 相对 receiver-original count 的偏好具有 graded causal influence，且这一效应在早期 donor（尤其 occurrence 2）最普遍**。但强因果 adoption 很少，continued-counting 不具 hop-2 persistence，maximum-count 规则不成立，因此证据更符合“分布式 ordinal/count information”而不是“显式 recurrent counter register”。

## 数据与因果卫生

- 主数据来自原始 frozen Native-thinking prompt；没有为了生成无编号格式而改 prompt。
- 每个 seed 只按结构选择最大 eligible `gold_count`，不读取最终正确性或任何 intervention outcome。
- 自然轨迹是 inline evidence sequence / audit sentence / completion recap，不存在可供复制的数字 index。
- prompt record spans 在行为读出中被 same-length background scrub。
- 所有 count readout 都在选定 event 后立即追加最小 `</think> ... Total:` query；原始 post-event reasoning 保留 **0 token**，因此不会重用诸如 “there are eight cities” 的 verbal-total recap。
- intervention 是 decoder-block input 的 all-layer clamp，并对 donor replacement 做 receiver-norm rescaling。
- count 候选为动态 `1..18`，报告完整 candidate distribution、candidate argmax 和 greedy generation。

Discovery 使用 seed 1246、1252（均为 `N=8`）。冻结确认使用之前未查看的 seed 1256（`N=10`）与 1261（`N=8`）。Broad scan 再按纯结构资格纳入所有 `N >= donor` 的自然 seed；它用于描述 generality，不当作新的独立确认。

## 结果强度分级

本文始终区分四个终点：

1. **graded effect**：donor-vs-original probability contrast 或 log-odds 增加；
2. **new candidate adoption**：blank candidate argmax 不是 donor，patched 后变为 donor；
3. **new greedy adoption**：blank greedy output 不是 donor，patched 后变为 donor；
4. **recurrence**：单状态 `k=1` transfer 后，后续至少两个 hop 和最终 count 均按递推公式更新。

只有第 4 层足以支持自然执行的 recurrent counter；本轮实验没有达到。

## 1. CountScope：存在 graded 信息，但强 adoption 稀少

### Discovery

在两个 `N=8` discovery seed、donor `2/5/7` 上，closing-boundary CountScope 的 paper CI 为正 **6/6**，但 donor candidate-argmax adoption 为 `0/6`，greedy adoption 为 `0/6`。因此 discovery 支持 graded causal information，不支持 causally sufficient count state。

### 冻结确认

确认前冻结的主要终点是：每个确认 seed 内 donor `2/5/7` 的平均 closing CI 为正。

- seed 1256，`N=10`：三个 CI 均略为负；
- seed 1261，`N=8`：三个 CI 均为正；
- 两个 seed 均无 donor candidate/greedy adoption。

所以预先冻结的跨-seed确认假设 **失败**。不能宣称 closing effect 在自然 no-index trace 中普适成立。

### Broad generality scan

为避免 paper CI 仅由压低 receiver-original count 驱动，主要描述量改为：

\[
\Delta\log\mathrm{odds}
=\log\frac{p'(d)}{p'(1)}-\log\frac{p(d)}{p(1)}.
\]

| donor occurrence | eligible seeds | `Δ log-odds > 0` | mean | median | new candidate adoption | new greedy adoption |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 8 | 7/8 | 1.321 | 1.197 | 2/8 | 0/8 |
| 5 | 5 | 4/5 | 0.494 | 0.730 | 0/5 | 1/5 |
| 7 | 4 | 2/4 | 0.430 | 0.450 | 0/4 | 0/4 |

关键个案：

- seed 1235、1253（均 `N=3`）中，donor-2 closing patch 分别把 candidate argmax 从 `1 → 2`、`4 → 2`；greedy 均仍为 0。
- seed 1236（`N=5`）中，donor-5 closing patch 把 greedy 从 `10 → 5`，target probability 从 `0.237 → 0.478`；但 blank candidate argmax 本来已是 5，因此这是一个真实的 greedy flip，但不是 candidate-level new adoption。
- seed 1256（`N=10`）是 donor 2、5 的共同反例。

可支持的最强表述是：**early event-closing states frequently carry causally usable relative ordinal/count information**。不能表述为每个 event 都保存一个足以独立决定 count 的 register。

## 2. Continued counting：未发现可迁移 recurrence

共 36 个条件：两个 seed，source endpoint `3/5/7`，`k=1/2/3`，closing/full 两种 region。

- `k=1` 是单状态 counter 的关键检验。
- seed 1246 的 `source=3, k=1, full` 可把最终输出从原始 8 推到公式目标 10；seed 1252 不复现。
- 该条件的 event-boundary CountScope readout没有得到预期 hop 1 = 4、hop 2 = 5；native successor greedy adoption 也为 0。
- 所有 `k=1` 条件均没有跨 seed 的 final adoption 或 hop-2 persistence。
- `source=3, k=3` 的最终目标恰好仍为 8，与 clean target 相同，属于非诊断性成功，不能作为 recurrence 证据。
- native-successor greedy adoption 在全部条件中为 0。

因此，单个 full-event patch 有时能直接改变最终 count，但没有证据表明模型从被移植状态继续执行 `+1` recurrence。

## 3. Linear additivity：一个 seed 可 steer，另一个近乎为零

用 7 个与 evaluation 分离的 discovery seed 拟合 layer 20--26 的 occurrence centroids，并测试 `μ_j - μ_i`、opposite 和 equal-norm orthogonal controls。

- `+1` 的真实 position-difference 在 seed 1246 上产生明显 donor-aligned shift `+0.669`，greedy 从 2 变为目标 3；controls 不 adoption。
- 同一干预在 seed 1252 上为 `-0.005`，无 adoption。
- `-1` direction 在两 seed 的 aligned shift 均为正，但第二个 seed 仅 `+0.00067`，且不明显胜过 controls。
- 全部 linear 条件的 candidate argmax adoption 为 0。

结论：存在可操纵的 affine count geometry 个案，但没有跨 trace 的稳健性；更不能据此说自然 forward pass 在执行向量加法。

## 4. Closing-boundary collapse：一个 trace 强依赖，另一个不依赖

将 occurrence-1 state 复制到后续所有 events：

- seed 1246：closing collapse 使正确 count 概率下降约 `0.998`，greedy `8 → 7`；full collapse 也强烈破坏，greedy `8 → 1`；opening single-token control 几乎无影响。
- seed 1252：opening、closing、full 三种 collapse 均基本保留正确输出。

这说明某些 trace 的 event boundary/full event states 对最终计数是必要的，但 effect 高度依赖 trace，且 full collapse 同样强，因此不能 claim 一个普适、separator-specific counter shortcut。

## 5. Maximum-count interchange：不支持 max operator

共 24 个条件，覆盖 `source < target` 和 `source > target`、`k=1/2/3`、closing/full。

- max-hypothesis candidate argmax adoption：`0/24`；
- 没有跨 seed、跨方向的 greedy max adoption；
- 某些 full `k=2/3` 条件有较大正 CI，但输出通常停留在 clean target 或落到其他 count。

因此不能 claim模型实现了论文所述的 maximum latent count operator。

## 当前最合适的 claim

建议正文使用：

> In naturally occurring, unindexed reasoning traces, event-boundary states often exert a graded causal influence on relative count preference, particularly at early occurrences. However, state transfer rarely induces a new count adoption, does not support persistent continued counting, and fails the maximum-count interchange test. The evidence therefore supports distributed, causally relevant ordinal information rather than a stable recurrent counter register.

中文：

> 在自然产生、无显式 index 的推理轨迹中，事件边界状态经常对相对 count 偏好产生 graded causal influence，且该效应在早期 occurrence 最稳定；但 state transfer 很少导致新的 count adoption，也不支持持续递推或 maximum-count interchange。因此，当前证据更支持分布式、因果相关的 ordinal information，而不是稳定的 recurrent counter register。

不建议使用：

- “模型在每一步维护并执行 `counter ← counter + 1`”；
- “closing token 是唯一 counter register”；
- “continued-counting 证明了 recurrence”；
- “模型实现了 max-like count composition”。

## 下一步最有信息量的实验

1. 新生成、完全未查看的 natural no-index cohort，预注册 donor 2 closing CountScope；主要终点用 donor-vs-original log-odds，并要求 baseline→patched new adoption。
2. 对 donor-2 的两个 new-candidate-adoption seed 做 layer-band necessity/sufficiency scan，寻找是否存在稳定的 causal band，而不是继续 all-layer clamp。
3. 构造 matched receiver，使 blank baseline 对 count 不偏置；否则 `N=5/8/10` prior 会掩盖 new adoption。
4. 若要 claim recurrence，必须以 `k=1` 为主，并同时要求 immediate hop-1、hop-2 和 final count adoption；不能用 `k=3` 或 clean-target-equal 条件替代。

## 产物

- Discovery CountScope/continued：`work/counting_mechanism_transfer_natural_20260826/Qwen3-8B/gpu0_countscope_continued_v2/`
- Discovery linear/collapse/maximum：`work/counting_mechanism_transfer_natural_20260826/Qwen3-8B/gpu1_linear_separator_maximum_v2/`
- Frozen confirmation：`work/counting_mechanism_transfer_natural_20260826/Qwen3-8B/confirmation_closing_countscope_v1/`
- Broad scans（含 baseline greedy/new adoption）：`broad_countscope_donor{2,5,7}_v2/`
- 运行计划与实现说明：`docs/counting_mechanism_transfer_20260826.md`

所有正式结论应以 `trials.jsonl` 和 `manifest.json` 为准；早期 smoke 与失败 run 仅用于实现诊断，不进入统计。
