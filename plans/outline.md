# Realistic NIAH Counting 实验方案

状态：实验协议
日期：2026-07-23
第一阶段模型：`Qwen/Qwen3-8B`

## 1. 研究目标

本项目研究长上下文计数中的两种可能算法：

- **Noisy broad aggregation**：模型从整个上下文中汇总分散、带噪声的证据，然后直接读出一个数量。
- **Targeted retrieval and aggregation**：模型逐项寻找 needle，把检索结果外部化或保存在推理轨迹中，再对这些项目计数。

我们要检验的不只是“思考是否提高准确率”，而是模型是否会在不同 prompt 条件下切换算法。核心比较包括：

1. Direct non-thinking 能否在不同长度和 needle 数量下准确计数？
2. Explicit indexed enumeration 能否通过“逐项检索 → 编号列出 → 计数”提高表现？
3. 模型的原生 thinking 是否会自发形成类似的逐项检索过程？
4. Query first 是否能更早建立检索目标，从而减轻 query last 的困难？
5. 这些现象如何随上下文长度、needle 数、needle 密度、模型大小和模型架构变化？

## 2. 实验设计总览

### 2.1 主网格

| 维度 | 设置 |
|---|---|
| 插针后 passage 长度 \(T\) | \(2K, 5K, 10K\) tokens |
| Needle 数 \(N\) | \(1,2,3,4,5,6,8,10,20,30\) |
| Paired seeds | `1234..1238`，共 5 个 |
| Query order | query first、query last |
| Prompt mode | Direct non-thinking、Explicit indexed enumeration、Native thinking（模型支持时） |
| 第一阶段 | Qwen3-8B 完整网格 |

这里的 \(T\) 是 **插入全部 needles 之后的 passage 长度**，使用 canonical tokenizer 计数；不包含 system message、task block、chat template 或模型输出。

### 2.2 每个 \((T,N)\) 组合有多少数据

主网格共有：

\[
3\ \text{lengths}\times10\ \text{needle counts}=30\ \text{cells}.
\]

每个 cell 使用 5 个 seeds，因此包含 **5 条不同的 stimuli**。每条 stimulus 是一个固定的 haystack、needle 集合和插入位置：

\[
5\ \text{stimuli/cell}\times30\ \text{cells}=150\ \text{unique stimuli}.
\]

同一批 5 条 stimuli 会在不同 prompt mode、query order 和模型之间复用。条件变化不会重新抽取 haystack 或 needle。

必须区分两种计数：

- **测试集合大小**：每个 \((T,N)\) cell 有 5 条不同数据。
- **模型生成次数**：同一条数据在多个实验条件下会产生多次输出。

| 模型类型 | 每条 stimulus 的条件数 | 每个 cell 的不同 stimuli | 每个 cell 的生成数 | 每模型总生成数 |
|---|---:|---:|---:|---:|
| Qwen3、Gemma 4 | 3 modes × 2 query orders = 6 | 5 | 30 | 900 |
| Llama、OLMo Instruct | 2 modes × 2 query orders = 4 | 5 | 20 | 600 |

因此：

- Qwen3-8B 第一阶段：30 cells × 5 stimuli × 6 conditions = **900 次生成**。
- 五个六条件模型（3 个 Qwen + 2 个 Gemma）：4,500 次生成。
- 三个四条件模型（2 个 Llama + 1 个 OLMo）：1,800 次生成。
- 八个模型的主实验合计：**6,300 次生成**。
- 对所有八个模型合并计算时，每个 \((T,N)\) cell 仍只有 5 条独立 stimuli，但会产生 **210 个模型输出**。

### 2.3 其他实验的样本量

| 实验 | Cells | 每 cell stimuli | Seeds | 每 cell 生成数 | 总生成数 |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B smoke test | 6 | 1 | `1234` | 6 | 36 |
| Qwen3-8B 完整主网格 | 30 | 5 | `1234..1238` | 30 | 900 |
| Qwen3-8B decoding control | 8 | 5 | `1234..1238` | 40 | 320 |
| OLMo Think-SFT 可选对照 | 30 | 5 | `1234..1238` | 10 | 300 |

Smoke test 的 6 个 cells 为：

\[
T\in\{2K,10K\},\qquad N\in\{5,6,30\}.
\]

Decoding control 的 8 个 cells 为：

\[
T\in\{2K,10K\},\qquad N\in\{5,6,20,30\}.
\]

后续机制分析从主实验已经生成的数据中抽样，不再创建新的行为测试集。每个机制 cell 最多选择 5 条 stimuli，并在数据允许时平衡 correct 和 incorrect。

## 3. 模型与实验条件

| 论文标签 | Hugging Face model ID | Direct / Enumeration | 同 checkpoint native thinking |
|---|---|---:|---:|
| Qwen3-1.7B | `Qwen/Qwen3-1.7B` | 是 | 是 |
| Qwen3-8B | `Qwen/Qwen3-8B` | 是 | 是 |
| Qwen3-32B | `Qwen/Qwen3-32B` | 是 | 是 |
| Gemma4-E4B | `google/gemma-4-E4B-it` | 是 | 是 |
| Gemma4-12B | `google/gemma-4-12B-it` | 是 | 是 |
| Llama3.1-8B | `meta-llama/Llama-3.1-8B-Instruct` | 是 | 否 |
| Llama3.2-3B | `meta-llama/Llama-3.2-3B-Instruct` | 是 | 否 |
| OLMo-Hybrid-7B | `allenai/Olmo-Hybrid-Instruct-DPO-7B` | 是 | 否 |

每次正式运行都要保存具体 model revision 和 tokenizer revision，不能只记录 `main`。

Llama 没有与 Qwen3 相同的原生 thinking 开关，因此主实验不使用 prompted CoT 冒充 native thinking。

`allenai/Olmo-Hybrid-Think-SFT-7B` 可以作为可选的 family-level thinking 对照。它和 Instruct-DPO 不是同一个 checkpoint，所以不能解释为同一模型内部由 prompt 引起的算法切换。

## 4. 预注册假设

### H1：Direct 存在容量拐点

Direct non-thinking 的 exact-count accuracy 会随 \(N\) 增加而下降。在较高 \(N\) 下，预测数量可能出现次线性增长、饱和或固定吸引子。

### H2：Enumeration 将 retrieval 与 counting 分开

Explicit indexed enumeration 应提高 gold city-score pair 的 recall。其错误可以分成：

1. 清单完整且 `Total` 正确；
2. 清单完整但 `Total` 错误；
3. 清单不完整但 `Total` 碰巧正确；
4. 清单不完整且 `Total` 错误。

第 2 类是 self-count/readout 失败，第 3、4 类主要反映 retrieval 失败。

### H3：Native thinking 的收益与 trace coverage 相关

如果 native thinking 采用逐项检索，它的正确率应与 reasoning trace 中出现的正确 gold pairs 数量相关。只有准确率提升、但轨迹中没有逐项证据，不足以支持 targeted retrieval 解释。

### H4：Query order 与算法存在交互

Query last 对 Direct 的影响预计大于对 Enumeration 和 Native thinking 的影响，因为后两种方式可以在解码阶段重新建立目标并逐项搜索。

### H5：模型规模主要改变临界容量

在 Qwen3 家族内部，更大的模型预计具有更高的 \(N_{50}\)，即 exact accuracy 下降到 0.5 时对应的 needle 数。跨厂商比较用于研究架构差异，不强行解释成纯参数 scaling。

## 5. 数据生成

### 5.1 数据来源

- Haystack：`data/haystacks/paul_graham/`
- Entity pool：`data/entities/cities.csv`
- Needle template：`data/templates/niah_fact_single_template.txt`
- Canonical tokenizer：`Qwen/Qwen3-8B`
- 每条 stimulus 内的 city 不重复。
- Score 从 50–100 中无放回抽样。

每条 needle、实际字符位置、token span、normalized depth 和 source window 都写入 metadata。

### 5.2 长度定义

\(T\) 表示插入全部 needles 之后、加入 query 和 chat template 之前的 passage token 数，取值为 2K、5K 和 10K。Canonical tokenizer 固定为 `Qwen/Qwen3-8B`，每条 stimulus 必须满足：

\[
\operatorname{len}\!\left(
\operatorname{tokenize}_{\text{Qwen3-8B}}(\text{final passage})
\right)=T.
\]

对主网格中的每个 \(N\)，先为 needles 预留 token 预算，再相应缩短 filler。因此，在固定 \(T\) 内增加 \(N\) 不会同时增加总 passage 长度；它增加的是 needle 数量和 needle density。

名义 density 定义为：

\[
\rho=\frac{N}{T/1000}.
\]

同一个 raw-text passage 在所有模型、prompt mode 和 query order 中复用。由于 tokenizer 不同，还要为每个模型记录：

- \(H\)：canonical tokenizer 下插针前的 clean filler tokens；
- \(L^{\text{passage}}_m\)：该模型 tokenizer 下的 final passage tokens；
- \(L^{\text{input}}_m\)：加入 task、system message 和 chat template 后的完整 rendered input tokens；
- 模型实际 density

\[
\rho_m=\frac{N}{L^{\text{passage}}_m/1000}.
\]

主图使用名义 \(T\) 和 \(\rho\)，跨模型回归同时使用 \(L^{\text{passage}}_m\) 和 \(\rho_m\)。\(L^{\text{input}}_m\) 用于 context-window、显存和延迟分析，不作为 needle density 的分母。该定义与旧 4K 报告“先按 needle 预算缩短 filler，再插针并固定总 passage 长度”的口径一致。

### 5.3 Needle 插入规则

Needle 的位置选择和文本插入继续调用仓库的 `generate_dynamic_niah_dataset_v2`。长度预算由外层 deterministic wrapper 控制：

```text
TARGET_PASSAGE_TOKENS = T
TARGET_HAYSTACK_TOKENS = searched_clean_filler_budget
RANDOMIZE_NEEDLE_INSERTION = True
RANDOMIZE_NEEDLE_SEED = paired_seed
SENTENCE_LEVEL_INSERTION = True
WORD_LEVEL_INSERTION = False
INSERTION_POSITIONS = [0] * N
```

每条 stimulus 按以下顺序生成：

1. 由 paired seed 生成固定的 needle 文本；
2. 用 \(T-\) needle token budget 得到初始 clean filler budget；
3. 生成 clean haystack window，并使用现有 sentence-level 算法插入 needles；
4. 用 canonical tokenizer 重新计算 final passage 长度；
5. 在初始预算附近确定性搜索 clean filler budget，直到 final passage 恰好为 \(T\) tokens；
6. 如果当前 source window 在搜索范围内无法达到精确长度，使用 paired seed 派生的 retry index 更换 window 后重试；超过预设上限则明确失败。

不能在插针后直接截断 final passage，因为这可能截掉 needle 或系统性改变尾部深度。不能为了达到长度而删除、改写或缩短 needle。

实际插入过程仍遵循仓库实现：

1. 从 `_sentence_end_offsets` 返回的保守句末候选中随机选择位置；
2. sentence-level 模式以 `INSERTION_POSITIONS` 是否为 `None` 判断该槽是否启用；
3. 位置随机数由 `randomize_needle_seed + ex_idx` 派生；
4. 句末候选少于 needle 数时允许有放回采样；
5. 没有合法句末时使用现有的 word-boundary fallback；
6. 保留现有的文本格式、插入顺序、token-span verification 和 metadata schema。

### 5.4 Paired seeds

主实验使用：

```text
1234, 1235, 1236, 1237, 1238
```

`stimulus_id` 格式为：

```text
T{T}_N{N}_seed{seed}
```

每个 ID 唯一决定 essay/window、city-score pairs 和插入位置。同一 stimulus 在所有模型和条件中使用相同的 raw context SHA256。

### 5.5 数据审计

冻结 150 条 stimuli 前必须检查：

- gold pair 数严格等于 \(N\)；
- needle 文本和 token span 能无损回读；
- city 和 score 在单个 stimulus 中均不重复；
- 插入 metadata 完整记录有放回采样和 fallback；
- filler 中可能形成的 city-score 干扰项；
- filler contamination audit 没有发现额外的、会被 parser 当成 gold needle 的 city-score pair；
- canonical tokenizer 下 final passage token 数严格等于 \(T\)；
- 所有 needles 均完整保留，且长度搜索没有截断 final passage；
- task block、system message 和 chat template 没有被计入 \(T\)；
- clean filler token 数 \(H\)、length-search attempts 和 retry index 已保存；
- 每个模型的 \(L^{\text{passage}}_m\) 与 \(L^{\text{input}}_m\) 均已保存；
- 每个模型的完整输入没有超出 context window；
- 同一 stimulus 在所有条件下的 context hash 相同。

审计规则必须在查看模型结果前确定。

## 6. Prompt 设计

### 6.1 Query first 与 query last

两种 query order 只改变 task block 的位置。System message、passage、分隔符和输出格式必须保持一致。

Query first：

```text
<TASK BLOCK>

<passage>
{context}
</passage>
```

Query last：

```text
<passage>
{context}
</passage>

<TASK BLOCK>
```

Query last 的 passage 前不能出现提示模型寻找 city-score records 的 task cue。

### 6.2 Direct 与 Native thinking

```text
The passage contains one or more city-score audit records. A city-score
audit record names one city and gives that city's numeric score.

How many city-score audit records are in the passage?
In the final answer, output exactly one line in this form:
Total: <integer>
```

- Direct：通过官方开关关闭 thinking。
- Native thinking：messages 与 Direct 完全相同，只开启官方 thinking。
- Native thinking 的 raw reasoning 和 final answer 分开保存。

### 6.3 Explicit indexed enumeration

```text
The passage contains one or more city-score audit records. A city-score
audit record names one city and gives that city's numeric score.

Find every city-score audit record in the passage. In passage order, output
one record per line as:
<k>. <city>: <score>
where k starts at 1 and increases by 1.
Then output one final line:
Total: <integer>
Do not include any other text.
```

Enumeration 关闭 native thinking。评估器同时解析编号清单和 `Total`。

### 6.4 模型模板

- Qwen3：使用官方 chat template 的 `enable_thinking=False/True`。
- Gemma 4：使用官方 processor/template，并分离 thought 与 final。
- Llama：使用 checkpoint 自带 template，不传入不存在的 thinking 参数。
- OLMo：使用 checkpoint 自带 template；不同 checkpoint 的结果分开分析。

正式运行前要人工检查每种 Qwen3-8B rendered prompt 至少 3 条。

## 7. 解码设置

### Direct 与 Enumeration

- `do_sample=False`
- `temperature=0`
- Direct：`max_new_tokens=64`
- Enumeration：`max_new_tokens=1536`

### Native thinking

- Qwen3：`temperature=0.6, top_p=0.95, top_k=20, min_p=0`
- Gemma 4：`temperature=1.0, top_p=0.95, top_k=64`
- `max_new_tokens=4096`
- Generation seed 由 stimulus seed 确定并保存。

所有首次输出都保留。`finish_reason=length` 单独报告，不能把截断样本静默删除。Enumeration 可以用更长预算重跑做 sensitivity analysis，但主结果仍使用 first-pass output。

## 8. 执行顺序

### 阶段 A：实现与数据冻结

1. 实现 strict query-order prompt builder。
2. 实现 Qwen、Gemma、Llama 和 OLMo adapters。
3. 实现 count/list/thought parser 及测试。
4. 实现可断点续跑的 batched runner。
5. 实现 post-insertion passage-length wrapper 及精确长度回归测试。
6. 生成并审计 150 条 master stimuli。

### 阶段 B：Qwen3-8B smoke test

运行 36 次生成，验证：

- 六种条件的 prompt 顺序和 thinking 开关；
- 每条 passage 在 canonical tokenizer 下都严格等于对应 \(T\)；
- `N=6` 的 indexed-list/count parser；
- `N=30` enumeration 输出预算；
- reasoning/final 分离；
- provenance、计时和硬件信息是否完整；
- 本地结果写入与云端同步。

### 阶段 C：Qwen3-8B 完整网格

运行 900 次生成。工程放行标准：

- dataset audit 100% 通过；
- parser failure <1%；
- 非预期 truncation <1%；
- 六条件配对完整；
- context hash 完全一致；
- 每种 mode 手工复核至少 20 条输出；
- 抽取 20 条验证 vLLM 与 Transformers 的 parsed-answer parity。

### 阶段 D：其他模型

Qwen3-8B 结果和吞吐报告冻结后，再运行其余七个模型。科学结果可以为 null；只要工程质控通过，就不根据结果方向修改网格。

### 阶段 E：可选控制

预算允许时运行：

- Qwen3-8B decoding control：320 次；
- OLMo Think-SFT：300 次；
- 边界 cells 先追加到 20 seeds；需要精确估计 \(N_{50}\) 时再追加到 50 seeds；
- Llama prompted-CoT，单独标记为非 native 条件。

## 9. 保存内容与评估指标

### 9.1 每次生成必须保存

- run ID、git commit、dirty flag、resolved config；
- model/tokenizer ID 和 revision；
- stimulus ID、seed、\(T,N\)、context SHA256；
- gold pairs 和 realized insertion spans；
- prompt mode、query order、rendered prompt、input IDs；
- clean filler tokens \(H\)、target \(T\)、length-search attempts 和 retry index；
- \(L^{\text{passage}}_m\)、\(L^{\text{input}}_m\)、\(\rho\) 和 \(\rho_m\)；
- decoding config 和 generation seed；
- raw output、thought、final、token IDs；
- parsed count、parsed list、parse status、finish reason；
- input/output token 数、wall time 和峰值显存；
- GPU、driver、CUDA、Transformers/vLLM 版本。

### 9.2 Primary endpoint

\[
\text{Exact-count accuracy}=\mathbf{1}[\hat N=N].
\]

### 9.3 Secondary endpoints

- Signed error、absolute error、normalized absolute error；
- Enumeration pair precision、recall 和 F1；
- 按插入深度分桶的 pair recall；
- Duplicate 和 hallucinated-pair rate；
- `Total` 是否等于模型自己列出的条目数；
- Native thought 中的 gold-pair coverage 和 passage-order consistency；
- Parse failure、refusal、truncation；
- 输出 tokens、延迟与准确率的 Pareto frontier。

## 10. Empirical law 分析

### 10.1 必画曲线

1. \(X=N\)，\(Y=\) exact-count accuracy；按模型、mode 和 query order 分面。
2. \(X=\rho\) 或 \(\rho_m\)，\(Y=\) accuracy log-odds；检查不同长度和 tokenizer 下是否 data collapse。
3. \(X=N\)，\(Y=\mathbb{E}[\hat N]\) 和 signed error；加入 identity line。
4. \(X=\) normalized passage depth，\(Y=\) pair recall。
5. \(X=\) output tokens 或 latency，\(Y=\) accuracy。

### 10.2 临界容量

为每个 model × mode × query order × \(T\) 拟合单调 logistic 或 isotonic curve：

\[
N_{50}: P(\hat N=N)=0.5.
\]

如果观测范围没有跨过 0.5，只报告上界或下界，不进行无依据外推。

### 10.3 模型大小与容量

Qwen3 家族探索性拟合：

\[
\log N_{50}
=
\alpha_0+\alpha_P\log P+\alpha_L\log L^{\text{passage}}_m
+\alpha_M M+\alpha_Q Q+\text{interactions}.
\]

这里只有三个 Qwen sizes，因此结果属于 family-level exploratory law。

### 10.4 Density data collapse

尝试构造：

\[
D=\frac{N^a(L^{\text{passage}}_m/1000)^b}{(P/10^9)^c}
\]

并拟合：

\[
\operatorname{logit}P(\text{exact})
=
\theta_{\text{model,mode,query}}-\lambda\log D.
\]

使用 leave-one-\(N\)-out 和 leave-one-model-out 验证。跨架构无法 collapse 时，报告 architecture-specific curves。

### 10.5 Direct 的饱和形式

比较：

\[
\hat N=aN+b,
\]

\[
\hat N=aN^\beta+b,\qquad \beta<1,
\]

\[
\hat N=K(1-e^{-N/K})+b.
\]

如果 Direct 更符合饱和模型，而 Enumeration/Native 接近 identity，将构成算法差异的定量行为证据。

## 11. 统计分析

- `stimulus_id` 是配对和聚类单位。
- Primary model 使用 hierarchical logistic regression。
- 固定效应包括 mode、query order、\(\log L^{\text{passage}}_m\)、\(N\) spline 及关键交互；\(L^{\text{input}}_m\) 只进入延迟/显存分析和长度 sensitivity analysis。
- 同 cell 的条件比较使用 paired bootstrap 或 McNemar test。
- Bootstrap 时整条 stimulus bundle 一起抽样。
- 预注册 contrasts：
  1. Enumeration vs Direct；
  2. Native vs Direct；
  3. Query first vs Query last；
  4. mode × \(N\)；
  5. mode × query order。

每个 cell 只有 5 条独立 stimuli，accuracy 的最小变化单位为 0.20。若真实准确率接近 0.5，单格比例的最坏标准误约为 0.224，正态近似 95% 误差范围约为 ±0.44。因此这套 5-seed 网格定位为 **pilot / exploratory experiment**：单格结果用于发现趋势，pooled paired model 用于初步比较，不能凭它给出稳定的 \(N_{50}\) 或精细 scaling law。Seed 只有 5 个水平，也不应依赖 seed random-effect variance；可将 seed 作为 paired block。第二阶段只对准确率位于 0.2–0.8 的边界 cells 先追加到 20 seeds，必要时再追加到 50 seeds。

## 12. GPU 预算

以下是规划值，不替代上机后的实测。假设 BF16、vLLM continuous batching、平均 native output 约 512 tokens。固定 post-insertion passage 后，平均输入量比“\(T\) 个 filler tokens 再加 needles”的口径约少 3%；该差异小于当前估时区间，因此下表保持不变并视为略保守。

### 12.1 Qwen3-8B 的 900 次生成

| Lambda 卡型 | 预计推理时间 | 预计费用 |
|---|---:|---:|
| 1×A6000 48 GB | 1.8–3.3 h | 约 $2.0–3.6 |
| 1×GH200 96 GB | 0.5–0.9 h | 约 $1.1–2.1 |
| 1×H100 PCIe 80 GB | 0.7–1.3 h | 约 $2.3–4.3 |
| 1×H100 SXM 80 GB | 0.6–1.1 h | 约 $2.6–4.7 |
| 1×B200 180 GB | 0.3–0.7 h | 约 $2.1–4.9 |

以上是纯推理时间。首次安装、模型下载和冷启动另计约 0.5–2 h，因此第一次租卡从登录到完成 Qwen3-8B smoke + 900-run，端到端大约需要：A6000 2.3–5.3 h、GH200 1.0–2.9 h、H100 1.1–3.3 h、B200 0.8–2.7 h。36-run smoke 后，使用实际 prefill/decode throughput 重算正式预算。

### 12.2 八个模型的 6,300 次生成

| 方案 | 预计 wall time | 预计费用 |
|---|---:|---:|
| A6000 混合配置 | 15–26 h | 约 $22–38 |
| 1×GH200 依次运行 | 4.5–8 h | 约 $10–19 |
| H100，Qwen32 使用 2×H100 | 4–7 h | 约 $22–38 |
| 1×B200 依次运行 | 2.5–4.5 h | 约 $17–32 |

上表是纯推理估计。八个模型的环境准备、下载和模型切换建议另留 2–6 h，因此端到端 wall time 约为：A6000 混合配置 17–32 h、GH200 6.5–14 h、H100 6–13 h、B200 4.5–10.5 h。Qwen3-32B 优先使用 2×H100、1×GH200 或 1×B200。Gemma 4 和 OLMo 在正式运行前分别做 10K、\(N=30\) 的 microbenchmark 和 OOM test。

## 13. 结果目录与运行安全

```text
runs/
  realistic_niah_v1/
    dataset/
      stimuli.jsonl
      manifest.json
      contamination_audit.json
    <model_slug>/
      <prompt_mode>/
        <query_order>/
          shard-*.jsonl
          metrics.json
          run_manifest.json
    aggregate/
      cell_metrics.parquet
      paired_metrics.parquet
      law_fits/
      figures/
      report.html
```

每个 shard 先写临时文件，完成后 atomic rename。恢复运行按 request ID 去重。结果先保存在服务器本地磁盘，再打包并同步到 Google Drive；Google Drive 不作为运行时唯一写入位置。

遇到以下情况应暂停运行并保留现场：

- dataset hash 或 paired mapping 不一致；
- 任一 final passage 在 canonical tokenizer 下不等于目标 \(T\)；
- 任一 needle 被长度控制过程截断、删除、缩短或改写；
- length-search 超过 retry 上限，无法精确命中 \(T\)；
- thinking 开关没有产生预期模板差异；
- parser failure ≥1%；
- 某个条件系统性 OOM 或截断；
- 预计费用超过预算上界的 1.5 倍；
- Google Drive 同步失败且服务器本地剩余空间不足。

## 14. 后续机制分析

行为主实验完成后，在 Qwen3-8B 中选择：

- \(T=5K\)；
- \(N\in\{1,5,10,20,30\}\)；
- 三种 modes × 两种 query orders；
- 每个机制 cell 最多 5 条 stimuli，并在数据允许时平衡 correct/incorrect。

分析目标包括：

- Direct 是否表现为 broad needle routing 和压缩后的 scalar count state；
- Enumeration/Native 的输出步骤是否逐项对齐 needle；
- progress、retrieved marker、successor/stop 和 final count state 是否可区分；
- query order 是否主要改变早期 target setup。

行为结果只能说明“与算法切换一致”；机制结论还需要 hidden-state、attention 和 causal intervention 证据。

## 15. 最终交付物

第一阶段 Qwen3-8B 应交付：

1. 150 条冻结 stimuli、manifest，以及证明每条 canonical final passage 精确等于 \(T\) 的审计报告；
2. 36-run smoke 报告；
3. 900 条完整生成及 resolved configs；
4. cell-level 和 paired metrics；
5. 主要曲线、错误分解和初步 empirical-law fits；
6. GPU 吞吐、耗时和成本报告；
7. 可从 request ID 断点续跑的 runner；
8. Google Drive 归档位置与本地 SHA256 manifest。

## 参考

- Synthetic NIAH v10 report：`../../Synthetic_NiaH_like_Count/colab_results/v10_main_seed1234_20260712_172332/syn_v10_report.html`
- Realistic NIAH 4K report：`../../NIAH-4K-report-standalone.html`
- 数据生成器：`src/dataset_generation/dynamic_niah_v2.py`
- Qwen3 model cards：<https://huggingface.co/collections/Qwen/qwen3>
- Gemma 4 E4B：<https://huggingface.co/google/gemma-4-E4B-it>
- Gemma 4 12B：<https://huggingface.co/google/gemma-4-12B-it>
- OLMo Hybrid collection：<https://huggingface.co/collections/allenai/olmo-hybrid>
- Lambda SSH 文档：<https://docs.lambda.ai/public-cloud/on-demand/connecting-instance/>
