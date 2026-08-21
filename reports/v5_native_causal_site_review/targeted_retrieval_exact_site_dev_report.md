# Native-thinking targeted retrieval: grammar-specific exact-site development screen

## 结论

主设计现在以 `(model, target grammar) -> one exact query token` 为选择单位。head ranking 读取该 token 对目标 prompt record 的 attention；ablation 也只施加在同一个 token。旧的 multi-site retrieval window 不进入主链。

Gemma 的 development 结果没有支持“为了 grammar-aware 而强制使用不同位置”。相反，三个可定位的主要 grammar 都由 `p0_item_end` 胜出，但每个 grammar 单独估计 head bank：

| Target grammar | 选定位点 | clean-stable 中断数 | 判定 |
|---|---:|---:|---|
| `adjacent_rank_after_city` | `p0_item_end` | 14/29 | development selected |
| `adjacent_rank_before_city` | `p0_item_end` | 4/5 | small-n development selected |
| `same_unit_rank_before_city` | `p0_item_end` | 9/29 | development selected |
| `same_unit_rank_after_city` | — | 0/4 at every screened site | unresolved |
| `structural_invariant_bullet` | — | support insufficient | unresolved secondary cohort |

这里的“中断”来自自由生成，不是 next-city ranking surrogate。但需要区分两个 scorer 版本：旧表使用 v1 `first generated gold city`，会忽略先输出的非-gold/幻觉城市；2026-08-18 的 integrity audit 后，新增实验统一使用 v2 `first semantic city record`。v2 要求模型首先表达的 city record 就是目标，后续自我纠正不能把失败改记为成功。旧表在完成批量 v2 rescore 前只保留为 development evidence。

Qwen 的相同设计目前给出负的定位结果。对主要的 `adjacent_rank_after_city`，P0 top-32 仅中断 2/30；`unit_pre_d1`、`city_pre_d1`、`record_clause_pre_d1` 的 top-32 screen 都是 0/10。按 representation 的 item-end layer 18 做 exact-layer 全 32 query-head 正控也是 0/10。最后，在完全相同的 P0、attention 指标和 10 个 clean-stable transition 上把剂量从 K=32 增至 K=64，结果仍只有同一个 seed1259 失败：K32 为 1/10，K64 为 1/10，十个配对的 outcome 完全一致（discordant 0/10），K64 新增失败为 0/10。因此不把 K64 扩到 30，也不为 Qwen 冻结该 grammar 的 causal site。

## Qwen `adjacent_rank_before_city`：三位点 window 与 all-head 正控

为检验 exact token 是否选得过窄，在同一 transition 中联合干预：

1. `pre_marker_d1`：数字 marker 之前；
2. `post_marker`：已读入数字、尚未读入句点；
3. `city_pre_d1`：目标 city 之前的最后一个 token。

head bank 仍由 `post_marker` 对目标 prompt record 的同位点 attention mass 排名；同一 bank 的 selected pre-O slices 在三个位置同时清零。干预只在 prefill 应用一次，decode 不动。10 个 matched clean transition 的 v2 结果为：

| 条件 | Head 范围 | 首个语义 city record 中断 | 注册 token path 偏离 | 解释 |
|---|---:|---:|---:|---|
| clean | 0 | 0/10 | 0/10 | matched baseline |
| post-marker top-K window | 32 | 0/10 | 0/10 | 三点联合仍无效 |
| post-marker top-K window | 64 | 1/10 | 3/10 | 两个额外偏离只是括号/措辞变化，city 仍正确 |
| all-head window | 1152 = 36×32 | 4/10 | 10/10 | 4 条先输出错误城市；6 条虽格式改变，首个 city 仍正确 |

all-head 是 positive control，不是 sparse circuit claim。它在每层把三个 query token 的整个 attention pre-O 向量清零，但保留 residual、MLP、其他 token 和全部 decode steps。K32→K64→all-head 的语义中断率为 `0% → 10% → 40%`，说明 current attention-mass top-K 很可能遗漏 causal heads；但 all-head 仍有 60% retrieval 成功，故排序并非唯一问题。更早的 clause/unit 起点、rank 与 city 之间的连续 token，以及 residual/MLP bypass 仍需检验。

scorer audit 的触发例是：模型先生成 `Sapporo received a score ...`，之后才自我纠正到目标 `Copenhagen`。v1 因只搜索 gold registry 会记为成功；v2 正确记为 `wrong_non_gold_city_record`。相反，`[Another entry for Fukuoka with 62]` 虽偏离严格 token path，但首个语义 city 正确，v2 仍记为 retrieval 成功。

## 为什么 P0 胜出具有机制意义

`p0_item_end` 是第 k 个 item 已提交后的最后一个 token，也是模型开始组织第 k+1 个 event 的最早 event-specific 边界。对 `adjacent_rank_before_city`，P0 ablation 打断 4/5；但 `pre_marker_d1`、`post_marker` 和 `city_pre_d1` 均为 0/5。这更符合“先 retrieve，再生成 marker/city”的时序，而不支持“输出 marker 后才去找 city”。

`adjacent_rank_after_city` 中，较晚的 `unit_pre_d1` 只有 3/28，`city_pre_d1` 为 6/30，而 P0 为 14/29。`same_unit_rank_before_city` 中，`pre_marker_d1` 为 0/29，P0 为 9/29。两者都说明把 query 放得太靠近 city 或 visible marker 会错过主要 causal stage。

## 设计边界

- 所有结果都来自已检查过的 development seeds，不是 held-out confirmation。
- 位点筛选只运行 clean 与 literal top-K selected bank；它不作显著性推断。
- 稀有 grammar 的 4/5 或 0/4 只用于定位或标记 unresolved，不估计稳定 effect size。
- 如果 literal top-K 占满同一层的全部可干预 value-head groups，exact layer-matched random K=8 control 在组合上不存在。此时 K=8 只用于定位；冻结位点后用可精确匹配的 K<=4 对照，并把 K=8 作为剂量补充结果。
- 每个 grammar 的 bank 独立排名；即使两个 grammar 都选择 P0，也不能合并 source rows 后共用一个 bank。
- Qwen K64 是超出原注册 K<=32 范围的 exploratory dose escalation。它只能检验更宽 head removal 是否增加破坏，不能证明新增第 33--64 个 head 各自具有 retrieval 特异性。
- Qwen all-1152 是 intervention-integrity positive control；它不能用于声称 1152 个 heads 都属于 retrieval circuit，也不能与 top-K effect size 作等价剂量解释。
- 多位点 window 是对 exact-site 负结果的 development follow-up，不替代“一次只冻结一个 exact site”的主估计量。window 必须包含选择 head bank 的原始 `post_marker` 位点，并记录选择位点与干预位点的区别。
- Qwen 当前的负结果排除了“只需在同一 exact token 多拆一些高 target-attention head”这一简单修补，但尚不能区分 residual-stream bypass、跨 token 分布式 retrieval、MLP/其他 attention stage 承载，或 ablation target 与实际 causal write 不一致。

机器可读 policy 位于 `configs/realistic_niah_v5_causal_exact_site_policy_dev.json`。Qwen 已完成的 exact-site screen 仍未定位到强效位点，不能从 Gemma 外推 P0 路由。
