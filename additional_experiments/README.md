# Additional experiments：任务迁移与计数机制

当前状态：2026-09-06 已切换到 `ubuntu@68.209.74.143`，SSH、Filestream 和 CUDA 张量运算检查通过。已启动 `smoke_outputs_v2_a` 双模型小规模验证；Qwen 已产生生成与表征输出，Gemma 随后运行。机制结论须待完整结果与干预对照检查后给出。旧主机 `.112` 的 CUDA 802 故障仅作为历史记录保留。详见 `STATUS.json`。

用户确认的主要目标是验证现有的表征与因果机制。这里保留一个范围明确的迁移 pilot：表征读出、旧 broad head 集合的终端消融、检索聚合层和晚期答案层的状态迁移。Native-thinking 的中间 item-to-item 因果链尚未在本 pilot 中实现，需要在自然 trace 可用性检查后另行设计。

## 科学问题和控制

| 任务 | 改变的量 | 固定的量 | 输出 |
|---|---|---|---|
| `count_all` | 原任务记录数 N=2,4,6,8 | 旧 V4.4 构造和提示词 | `Total:<integer>` |
| `kth_needle` | k=2,4,6,8 | 同一 seed 的 10 条原始记录、顺序及整个 passage | `Needle:<city>\|<integer>` |
| `topic_count` | 目标主题记录数 N=2,4,6,8 | 总记录数 10、城市/分数/记录次序和背景 | `Total:<integer>` |

主题通过每条记录内新增的一句项目描述表示。两类为 astronomy 和 botany，数据中没有 `Topic:` 字段，也不直接写出这两个类别名称。目标主题在 seed 间平衡，并在 passage 前明确告知。不同 count 的目标记录集合按 seed 嵌套，主题句子的变化会改变 tokenizer 长度，必须报告实际 token 数。此设置只覆盖两个语义差异较大的主题。

两种模式分别使用 checkpoint 的 `enable_thinking=False/True`。Non-thinking 保留旧实验的回答前缀 prefill；Native-thinking 保留自然推理，不要求列举、固定格式的推理步骤或人工 assistant 推理前缀。原 `count_all` 的 Non-thinking query 和 Native-thinking user prompt 有逐字一致性测试。

只读来源为本地旧 V4.4 frozen stimuli，SHA-256：
`da4dd86142eb8a07f9a7e53497efd3375184c8e68367d4db994370fcb331f090`。

## 样本与机制

默认先运行 smoke：discovery seed 1234，confirmation seed 1254，levels={2,8}，共 12 条输入 × 两模式 × 两模型 = 48 次基础推理。该规模只检查可执行性，不支持稳定机制结论。

备用 pilot：discovery seeds 1234–1237；confirmation seeds 1254–1257；每个 seed 四个 level、三个任务。共有 96 条输入、192 份 user prompt、384 次基础推理。它复用旧实验 seed，是任务迁移的探索性分析，不是独立的新确认队列。

1. **表征读出。** 捕获每条 prompt record 的末 token、自然推理中保守识别的 item-end，以及最终答案值之前的 query。固定层为 Qwen L8/L19/L23/L29、Gemma L9/L16/L29/L37，均为 zero-based。计数参考任务 discovery 数据拟合 full-state ridge 和 count-centroid rank-3 子空间，直接测试新任务的 held-out seeds；同时报告新任务自身拟合、随机子空间、位置与序号基线。主题任务分别保存已经读过的总记录数和目标记录数。Probe 不证明因果使用。
2. **冻结 head 消融。** 从现有报告 membership CSV 导入 Qwen top-32、Gemma top-6 broad bank，不在新任务上排序。每个终端 query 比较 clean、旧 bank 和三个同层同数量随机 bank；random 与旧 bank 允许重叠，沿用旧采样定义。干预仅在一次 prefill 中的 query token 生效，随后生成完整短答案。
3. **终端状态迁移。** 同一 seed、同一任务，相邻 level 的 donor；Qwen L23/L29，Gemma L29/L37。比较 self-patch、完整 donor state、等位移范数的正交随机对照。检查实际模型 dtype 下的范数和正交误差，以及 self-patch 的逐 token 无变化。主要迁移指标是跟随 donor 的原模型预测，另存 donor 正确答案。
4. **Native-thinking 旧表征。** 复制已冻结的 Qwen L19/Gemma L16 `item_end_discovery_basis.npz` 及完整原始 provenance。只用于自然 item-end 的子空间读出，并在参考任务 discovery 上重新校准 readout；不能表述为旧 probe 权重的零样本迁移。自然 line-end 与旧 parser 的 site/grammar 可能不完全一致，保存逐行文本和显式序号标记供审计。

每个 confirmation case 的因果部分最多 11 次短答案生成：clean 1、bank 4、两层各 3 次 patch。Smoke 最多 264 次、pilot 最多 2,112 次因果生成，另有状态采集和注意力 forward。先依据 smoke 的耗时和显存决定 pilot 预算；目前没有实际 GPU 耗时估计。

## 精确 prompt 与审计文件

`protocol.py` 是 prompt 和标签的唯一生成逻辑。使用修正后、尚未进行任何模型推理的 `task_transfer_20260905_v2/frozen` 和 `task_transfer_smoke_20260905_v2/frozen`。

- `cases.jsonl`：完整 passage、来源、gold、每条记录及字符区间。
- `user_prompts.jsonl`：每条样本、每种模式的完整 user prompt 和 SHA-256。
- `prompt_examples/`：六份完整可阅读 prompt，各 task × mode 一份。
- `frozen_banks.json`：旧 head 成员、来源文件和 hash。
- `legacy_native_bases/`：旧 Native-thinking 子空间与原始 provenance。
- 模型运行时的 `captures/<mode>/<case>/prompt.json`：实际 chat template、thinking 参数、assistant prefill、完整 rendered prompt、input IDs、attention mask 与 hashes。
- `generation.json`：原始生成文本、token IDs、解析结果和截断标志。
- `sites.json`、`states.npz`：精确 token 位置、native site 缺失原因和少量选定层状态；不存全层全位置张量。
- `causal/`：逐次干预结果、head 成员、hook 次数、donor ID、控制范数和实际输出。

早期 `v1` 冻结输入只用于本地准备检查，保留供审计，不用于运行。`v2` 将目标主题明确放在 passage 前，消除了模型读 passage 时尚不知道目标主题的问题。

## 运行和隔离

全部新增文件在 `additional_experiments/`；旧代码、报告、数据不修改。运行结果在 `runs/`，压缩包在 `bundles/`，两者均由本目录 `.gitignore` 排除。未创建 Git commit，未提交 checkpoint、原始 trace 或 tensors。

远程挂载路径必须先在用户指定主机上确认。历史记录中的 `/lambda/nfs/CoT-Non-thinking-v4` 只能用作查找线索。连接恢复后，在实际 Filestream 的 `additional_experiments/task_transfer_20260905_v2/` 下解压独立 `repo_snapshot/`；不得覆盖现有 Non-thinking/Native-thinking 目录。

本地重新准备独立版本：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s additional_experiments -p 'test_*.py' -v
.\.venv\Scripts\python.exe additional_experiments\freeze.py --output additional_experiments\runs\NEW_RUN\frozen --smoke
```

远程在压缩包解出的 `repo_snapshot` 内，用原实验 CUDA 环境；模型版本由旧注册表固定为 Qwen3-8B `b968826...` 和 Gemma4-E4B `ee0ef60...`。`requirements-native-count-stream.txt` 记录旧环境依赖，包括 Transformers 5.14.1。使用现有环境和 checkpoint cache，先检查实际包版本和 GPU；不自动安装或更新 CUDA PyTorch。

```bash
export EXPERIMENT_PYTHON=/absolute/path/to/existing/venv/bin/python
export MODEL_CACHE=/absolute/path/to/existing/model/cache
export GPU_INDEX=0  # replace with an inspected idle GPU
export FROZEN_INPUTS="$PWD/additional_experiments/runs/task_transfer_smoke_20260905_v2/frozen"
export EXPERIMENT_OUTPUT=/confirmed/filestream/additional_experiments/task_transfer_20260905_v2/smoke_outputs
bash additional_experiments/run_gpu.sh
```

Launcher 依次运行 Qwen、Gemma，写独立日志；检查指定 GPU 没有超过 1 GiB 的已有占用，保持 checkpoint 离线读取。断点续跑须逐模型调用 `run.py --resume`，输入和源码 hash 必须不变。已有不完整 case 目录会明确失败，避免把失败结果当成成功续用；修复后的新版本应使用新输出目录。

## 解释边界

统计独立单位为 seed；先在 seed 内聚合，再计算 bootstrap 区间。四个 confirmation seeds 的区间只能作为探索性摘要。所有行为结果保留，包括错误、格式错误和截断。不能因结果正确才纳入主行为分析。

Native-thinking endpoint 必须来自原始生成 token 的精确前缀，且出现显式 reasoning 结束边界；不从思考中的 `Total:` 提取最终答案位置，也不悄悄重新分词。无法定位的案例记录缺失原因。推理文本不满足保守 item-end 解析时，不构造人工 trace。

Native-thinking 的终端 patch/消融保留原先的完整推理。这些结果只检验给定自然推理后的终端计算；此前推理可能已写出答案。完整 residual 的 donor transport 只支持该位置的状态可改变输出。旧 bank 在新任务上的效应也不足以证明唯一或完整相同的计数机制。

待 GPU 可用后，首先人工检查各任务自然 trace 和 prompt/token 边界，再决定是否需要不同主题难度、更多 seeds、native 中间 item-to-item patch，以及原实验更完整的 span-restoration/mediation 套件。所有这些扩展均需单独冻结设计和 prompt。
