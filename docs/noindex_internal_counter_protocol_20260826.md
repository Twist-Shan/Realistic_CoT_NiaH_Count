# Natural no-index internal-counter protocol

日期：2026-08-26<br>
模型：Qwen3-8B（主分析）；Gemma4-E4B 扫描因命中率过低暂停<br>
状态：Qwen fixed-N=3/N=10 的 first-pass no-index cohorts 已冻结（2026-08-27）

## 1. 研究问题与两层 claim

本轮把结论明确拆成两个互不替代的层次。

### Level 1：paper-standard operational internal counting

问题是：在原生、无 running item index 的 reasoning trace 中，逐事件 state 是否：

1. 按事件 occurrence 呈现 CountScope 式对角结构；
2. 对 count preference 有可重复的因果影响；
3. 在 continued-counting transfer 中至少表现出跨 trace 的 counting-state 可迁移性。

若 discovery 冻结的检验在 untouched confirmation 上通过，可以 claim：

> The model implements an operational internal counting mechanism in naturally
> occurring reasoning traces without a running item index: event-aligned states
> encode count progress and causally influence count readout/continuation.

这里的 `mechanism` 是功能性、因果性的 operational 定义，不自动蕴含一个 Markov register，
也不自动蕴含 arithmetic `+1`。

### Level 2：严格的递推 `+1` mechanism

独立问题是：是否存在一个 state `s_k` 与稳定更新算子 `F`，使得干预后的状态满足

\[
s_{k+1}=F(s_k, e_{k+1}), \qquad c(s_{k+1})=c(s_k)+1,
\]

并在新 seed、不同 donor/receiver、至少两个后续 hop 上保留 donor-relative offset。

现有 transition-equivariance、retention、operator-scan 与 event-edit 结果均未找到这一机制；
覆盖的条件更符合 intact prefix/event memory 对下一状态的 contextual reconstruction/reset。
因此即使 Level 1 成立，Level 2 仍应写为：

> We did not identify an invariant `+1` state-transition mechanism. Intervened
> count-like states were usually overwritten or reconstructed from distributed
> event history at the next event boundary.

“没有发现”只限定于已测试的 carrier、layer、site、scale 和 operator family；不能写成
数学上排除了模型内部任何可能的加法实现。

## 2. Frozen-prompt no-index cohort

### Prompt 与生成

- 使用 byte-identical 原始 V5 user prompt；不增加 user/system instruction 或 assistant prefix。
- greedy decoding：`do_sample=False`，每条最多 4096 new tokens。
- **固定 N=3**，不在同一 seed 内事后挑选最容易产生 no-index 格式的 N。
- N=3 是能观察 `1→2→3` 两个 transition 的最小非平凡长度；选择它是因为 causal claim
  不要求 N=10。这里的稀缺性主要针对“整段推理从不编号”的 global-clean sensitivity，
  不是 first-occurrence prefix-clean Primary。
- Qwen 与 Gemma 使用相同的 seed/split 规则，但独立生成、审计，并可在不同 seed 停止。

先完成 fixed N=3 的 20+10 cohort；随后从 seed 1234 重新开始、按完全相同规则独立扫描
fixed N=10 的 20+10 cohort。两个 N 不共享 quota，也不混合凑足 30。主要 feasibility
读数为每个 model × N 达到 quota 时的 `last_scanned_seed`、attempted seeds 与命中率。

### Seed 与 split 规则

完全复用现有 V5 causal / one-to-one supplement 的顺序：

- seed 1234–1253：discovery；
- seed 1254–1263：confirmation；
- seed 1264：discovery supplement；seed 1265：confirmation supplement；
- 从 1266 起，每连续三个 seed 固定分配两个 discovery、一个 confirmation；
- 两个 split 分别取按 seed 升序的前 20 / 前 10 个 format-eligible rows。

已有 frozen deterministic generation 优先复用；缺失 seed 才重新生成。选择规则为：

1. 只读取 format audit，不读取 correctness、patch、probe 或 intervention outcome；
2. 固定 N=3，不允许 seed-specific N fallback；
3. eligible seeds 在各自预定 split 内按数值升序；
4. 截取 discovery 前 20 和 confirmation 前 10；
5. confirmation 的 split assignment 不根据 discovery 格式率或 mechanism effect 改变。

### First-pass no-index primary population

Primary 不再由后置 parser-selected bullet/recap 决定，而由模型的**首次、无回放的完整证据提取**决定：

1. 在原始 reasoning 中，对每个 gold city-score record 定位首次局部 score-supported mention；
2. 按生成顺序去重，并定义
   \[
   t^*=\text{第 }K\text{ 个唯一 gold record 首次出现的末端};
   \]
3. 要求从 reasoning 起点到 \(t^*\) 一共只出现 K 次 score-supported gold-record evidence，
   即每条 gold record 恰好一次；任一 gold city-score 在 \(t^*\) 前重复出现都视为已经开始
   replay/rethink，不能进入 first-pass cohort；
4. 要求从 reasoning 起点到 \(t^*\) 不出现 record/item/excerpt
   编号、ordinal record label、`Count=k`、英文数词 running subtotal、numbered evidence line，
   或紧邻 city 的 `(k)` index；
5. primary mechanism sites 固定为这些 first-occurrence spans，所有读取和干预均限制在
   \(t\leq t^*\)。

\(t^*\) 之后的 rethink、重复证据或编号回顾不影响 Primary 资格，因为 causal Transformer 中未来文本不能反向影响
此前 hidden states。它可以作为干预后的下游输出，但不能被用作 primary state site。例如，先无编号
找齐十条证据、之后才生成 `(1)…(10)` recap 的 trace 属于 `prefix_clean`；若 `That's one/two`
出现在后续目标记录之前，则不合格。

Primary 选择仍不读取 final-answer correctness、probe、patch 或 intervention outcome。Gold registry
只用于确定首次出现的 city-score evidence span；这是预注册的 oracle segmentation，而不是按效果挑选。

### Frozen \(t^*\) token-boundary contexts

机制实验不需要把后置 recap 放进模型上下文。对每个已冻结 Primary row，保留完整 generation 作为
不可变 provenance，并另行构造

\[
x_{\mathrm{analysis}}=x_{\mathrm{prompt}}\;\Vert\;
x_{\mathrm{output}}[0:\tau(t^*)],
\]

其中 \(\tau(t^*)\) 是覆盖 `t_star_char` 的最小完整 output-token 前缀终点。该规则对所有 seed
完全相同；不重新生成、不按 seed 手工选择 cutoff，也不读取 final answer 或 mechanism outcome。

Qwen N=3 与 N=10 均已冻结 20 discovery + 10 confirmation 个新 context。真实 tokenizer 审计显示：

- 60/60 context 都是原 prompt 与原 output token IDs 的严格前缀组合；
- 60/60 都删除了 \(t^*\) 后的输出，因此 future recap 不进入 causal context；
- token-boundary spill 最大为 3 个字符，全部只是引号或换行，没有字母、数字或 recap cue；
- N=3 删除的后续 output tokens 为 108–736，中位数 279.5；N=10 为 142–286，中位数 207。

冻结文件位于各 cohort 目录的 `tstar_first_pass_v2/tstar_first_pass_contexts_v2.jsonl`，并由
`tstar_first_pass_manifest_v2.json` 记录 source hashes、seed/split、cutoff 和 artifact hash。
这些截断 context 称为 `first-pass no-index truncated cohort`，不能改称模型自然生成的
global-clean trace，也不能声称 \(t^*\) 后没有 rethink；只能说 rethink 不在 causal context 中。

### 历史 forced-immediate-Total replay（不属于新 cohort）

以下结果来自 2026-08-26 的旧 `prefix_clean_v4` cohort。该 cohort 当时没有排除 \(t^*\) 前的
score-supported evidence replay，且 N=10 旧样本还包含后来修复的 shorthand numbering 污染。
因此这些 readout 数字仅作为开发历史保留，**不得外推到 2026-08-27 first-pass cohort**，也没有
参与新 cohort 的选样。新样本目前尚未运行 forced Total 或任何 mechanism outcome。

旧实验为检验 \(t^*\) 后跳过 recap 是否仍能直接读出总数，在 seed/cohort 已冻结后，对每个 context
统一追加 byte-identical suffix `\n</think>\n\nTotal: `，再用 Qwen3-8B、bfloat16、SDPA
做最多 16 tokens 的 greedy decoding。该条件是 standardized forced-stop readout，不是未修改的
natural continuation。

- N=3：30/30 只生成整数 `3` 并立即 EOS；20/20 discovery、10/10 confirmation 均正确。
  与原完整 trace 的 Total 一致为 28/30；seed 1465、1609 的原 trace 错答 `5`，forced-stop
  改为正确的 `3`。
- N=10：30/30 都只生成一个整数并立即 EOS，但只有 20/30 输出正确的 `10`；另外 10/30
  输出 `9`。Discovery 为 12/20 正确，confirmation 为 8/10 正确。输出 `9` 的 seeds 为
  `1267, 1290, 1293, 1384, 1506, 1539, 1893, 1978, 1307, 1688`，其中最后两个属于
  confirmation。

因此，\(t^*\) 是“第 K 个唯一 evidence 已经进入上下文”的可靠 endpoint，但在 N=10 上不是
“count=K 已完成 commit”的可靠 endpoint。后续若需要 post-count state，应在 discovery 上冻结一个
小范围 separator/post-evidence delay scan，再一次性应用于 confirmation；不能把所有 \(t^*\) state
直接视为已提交的 count state。

完整 rows、逐 seed shards、CSV、frozen plan 和 manifest 分别保存在：

- `work/natural_noindex_counter_n3_forced_immediate_total_20260826/Qwen3-8B/`
- `work/natural_noindex_counter_n10_forced_immediate_total_20260826/Qwen3-8B/`

### 历史 N=10 post-evidence token-delay scan（不属于新 cohort）

该 scan 同样使用旧 cohort，只记录为什么最终放弃 `t+k`，不能作为新 first-pass cohort 的结果。
在旧 frozen N=10 discovery 上，从同一个 (t^*) 起点继续保留模型自然生成的
`1, 2, 4, 8` 个 token，再追加相同 forced-stop suffix。四个 discovery arms 均为 20 seeds；
边界只按 discovery 选择，confirmation 在冻结前不可见。

| post-(t^*) delay | discovery correct | explicit-cue-free | repeated known city |
|---:|---:|---:|---:|
| 1 | 12/20 | 20/20 | 1/20 |
| 2 | 12/20 | 20/20 | 1/20 |
| 4 | 13/20 | 20/20 | 1/20 |
| 8 | 12/20 | 18/20 | 2/20 |

最初的 v1 eligibility 还排除了 post-(t^*) 对任何 gold city 的重复提及，因此四个 delay
均不合格。检查后发现，`d=1,2,4` 唯一的违规来自 seed 1978 立即复述已经出现过的第十个城市
`London`：没有显式 count/index cue，也没有新增唯一 evidence。这个条件把“重复已有内容”错误地
等同于“泄漏 count”。因此，在已经查看 discovery、但尚未运行任何 delayed confirmation 时，
留下审计记录并修订为 v2：只排除显式 count/index/`Total` cue；重复已知城市单独报告，不再作为
eligibility。该修订是透明的 post-discovery amendment，不能描述为原始预注册规则。

v2 下 `d=1,2,4` 合格；按“discovery accuracy 最大、并列取最小 delay”的冻结规则选择
`d=4`。`d=8` 因 seeds 1569、1918 出现 `Wait, that's 10` running-progress cue 而不合格。
冻结后只在 10 个 untouched confirmation seeds 上运行 `d=4`：

- 8/10 输出正确的 `10`，2/10 输出 `9`；错误仍为 seeds 1307、1688；
- 10/10 post-(t^*) delay 均无显式 count/index cue，也没有重复 gold city；
- 与 `d=0` forced-immediate readout 逐 seed 完全相同，没有任何 confirmation seed 改变输出；
- discovery 的单点提升只来自 seed 1893 从 `9` 变为 `10`。

因此，`d=4` 是本设计按 discovery 冻结的 boundary，后续若复用该设计应保持它不变；但
confirmation **没有证实 delay 提升 count readout**。不能把它写成找到了更可靠的 count-commit
endpoint，也不能通过 delay scan 消除 N=10 的 8/10 endpoint/readout limitation。

完整远端产物位于
`work/natural_noindex_counter_n10_tstar_delay_scan_20260826/Qwen3-8B/`；本地完整审计副本位于
`work/audit_n10_tstar_delay_scan_20260826/`。`tstar_delay_scan_manifest_v3.json` 是最终汇总；
它保留并哈希引用未覆盖的 v2 manifest，只修正 v1/v2 eligibility 的报告口径，不改变 trials、
冻结 boundary 或 confirmation outcomes。

### Global-clean sensitivity subset

另报告嵌套的 `global_clean` 子集：整段 reasoning 中都没有 per-record
labeled/ordinal/numbered-line/parenthetical index。除正则审计外，原 trace parser 若将记录结构判为
`indexed`、`ordinal` 或 `inline_count`，也直接从 global-clean 排除，避免漏掉 `1. City - score`
一类表面变体。第 K 条证据后的单次 terminal aggregate total 允许存在，因为它是答案汇总而不是
running item index。后置 `(1)…(K)` recap、`third record` 等会从 global-clean 排除，但不自动
排除 prefix-clean Primary。

若 global-clean 样本量不足，只作敏感性分析，不把它伪装成与 Primary 同等把握的确认性检验。

### 2026-08-27 已冻结 Qwen first-pass cohorts

- N=3：扫描 seed 1234–2292，共 1059 个；Primary 为 20 discovery + 10 confirmation；其中
  global-clean 为 7 discovery + 4 confirmation（11/30）。
- N=10：扫描 seed 1234–2322，共 1089 个；Primary 为 20 discovery + 10 confirmation；其中
  global-clean 为 2 discovery + 1 confirmation（3/30）。
- 新冻结产物统一使用 `*_first_pass_noindex_v5` 与 `tstar_first_pass_v2` 文件名；旧 cohort 文件
  不覆盖。v5 同时排除 \(t^*\) 前的 evidence replay，并识别 `1. City - score` 这类 shorthand
  numbering。旧 v4 seed 列表已被取代，不能与新 cohort 混用。
- 原始 selected rows 保留 source generation 的顶层 `split` 以维持 provenance；正式实验 split
  以 `noindex_n3_cohort.split` / `noindex_n10_cohort.split`、cohort manifest 和导出的 t-star
  context `split` 为准。N=10 seed 1359 的旧 source 顶层 `split=confirmation`，正式 cohort split
  为 `discovery`；最终 context 已规范化为后者。

本地完整副本：

- `work/audit_n3_first_pass_noindex_20260827/`
- `work/audit_n10_first_pass_noindex_20260827/`

跨 cohort 的简要索引见 `work/noindex_first_pass_internal_counter_cohorts_20260827.md`。

## 3. Discovery → confirmation 纪律

1. 完成两模型的 20-seed discovery cohort 后，只在 discovery 上确定 layer/band、event site、
   donor occurrences、CountScope estimand、continued-counting patch region 与统计规则。
2. 将完整配置、代码版本、discovery/confirmation row SHA256 写入 manifest。
3. 在查看 confirmation intervention outcomes 前冻结配置。
4. 每模型仅对其 10 个 untouched confirmation seeds 做一次主检验；失败后不得用 confirmation
   重新选择 layer/site 并仍称为 confirmation。
5. candidate probability/log-odds、candidate argmax、greedy adoption 分开报告；graded effect
   不等同于行为 adoption。

## 4. 解释边界

可能出现的最终组合及其含义：

| Level 1 | Level 2 | 可支持解释 |
|---|---|---|
| pass | fail | 有 operational/distributed internal counting；未找到稳定 `+1` register |
| pass | pass | 有 causal counting state，且找到可泛化的递推更新机制 |
| fail | fail | 只能 claim count-correlated/locally causal information，不能 claim general counter |
| fail | pass | 逻辑上异常，需先排查 endpoint、selection 与 readout 定义 |

当前先验与既有结果最一致的是第一种或第三种；本轮实验的目的就是用对称的 Qwen/Gemma
20 discovery + 10 confirmation 设计区分二者。
