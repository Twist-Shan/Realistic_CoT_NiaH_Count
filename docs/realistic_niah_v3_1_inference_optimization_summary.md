# Realistic NIAH V3.1：Inference 优化与实验分工总结

## 1. 结论摘要

1. **Empirical-law inference 已经使用 vLLM**，并已完成第一轮工程优化：48 个逻辑
   model-mode shards 被组织成 14 个物理 model bundles，每个模型在一次 worker
   attempt 中只加载一次，然后连续运行其注册的 prompt modes。
2. **不同模型目前已有不同的并发配置**，但这些参数是按模型规模设置的保守初值，
   尚未经过目标 H100 环境的系统 benchmark。
3. **当前没有按 haystack 长度动态改变 engine 配置**。固定配置主要照顾 20K
   最坏情况，因此短上下文可能没有充分利用 GPU。是否加入 length-aware batching，
   应在正式推理前通过不查看 accuracy 的吞吐测试决定。
4. **正式推理精度当前统一为 BF16**。两类 Nemotron 的 Mamba state cache 按模型
   要求使用 FP32。FP8、INT8 或其他量化会改变数值路径，不能仅作为无条件工程
   优化加入 confirmatory run。
5. **Empirical law 与 mechanism 分析相互独立**。前者只依赖冻结 stimuli、模型
   responses、解析结果和聚合统计；不需要 hidden states、attention、activation
   patching 或 causal intervention。Mechanism 不构成 empirical-law inference 的前置
   阻塞项。

## 2. 当前实验规模与已冻结输入

- 14 个模型；
- 48 个注册的模型 × prompt-mode 逻辑组合；
- 14 个物理 model bundles；
- 8 个长度：1K、2K、3K、5K、8K、10K、15K、20K；
- 14 个计数：1–10、12、15、18、20；
- 30 个配对 seeds；
- 3,360 条唯一 stimuli；
- 161,280 次正式模型请求。

冻结数据已通过独立审计，并公开在：

<https://huggingface.co/datasets/stwistzz/realistic-niah-count-empirical-law>

`stimuli.jsonl` SHA256：

```text
afed18fe24d3c684b7f342a3c5cc119fe3bd4033487d25c97cbbe4fc21c0d159
```

## 3. 当前 inference 实现

### 3.1 软件栈

- vLLM `0.25.1`；
- Transformers `5.5.3`；
- Mistral 模型使用 `mistral-common >=1.8.6,<2`；
- 默认 `max_model_len = 32768`；
- 默认权重/计算 dtype 为 `bfloat16`；
- `enable_prefix_caching = true`；
- 每个 GPU 一个 worker，当前 `tensor_parallel_size = 1`；
- `gpu_memory_utilization` 为 0.90 或 0.92。

### 3.2 已完成的优化

#### 一次模型加载运行全部模式

旧的逻辑执行方式需要 48 次模型加载。当前以模型为物理调度单位，只加载 14 次，
避免 34 次冗余 checkpoint load。逻辑输出仍保留为 48 个独立 shard，因而不改变
预注册的请求 ID、审计结构或 model-mode 比较。

#### 原子断点与单次 canonical merge

每个 generation batch 独立写入原子 checkpoint part。任务完成后才生成一次完整
`requests.jsonl`，避免每跑完一个小 batch 都重写不断增长的大文件。中断后可从已完成
parts 继续，不需要重新生成整组 responses。

#### 减少重复 prompt 存储

每条 response 保留 reconstructible prompt hashes、输入/输出 token 数、输出文本和
output token IDs，但不在每个结果行重复保存完整 prompt payload。这降低了磁盘写入、
网络同步和最终 merge 成本。

#### 优先运行预计较慢的模型

bundle scheduler 根据模型规模和 reasoning mode 设置静态优先级，优先 claim 31B/32B
和 native-thinking 工作，减少多 GPU 尾部等待。不过这些权重目前是结构化启发式，
还不是 H100 实测运行时间。

## 4. 当前按模型设置的并发参数

下面是 worker 脚本实际使用的配置，而不是建议值：

| 模型组 | `request_batch_size` | `max_num_seqs` | GPU memory utilization |
|---|---:|---:|---:|
| Qwen3-32B, Gemma4-31B | 1 | 1 | 0.92 |
| Gemma4-26B-A4B, Qwen3-14B | 2 | 2 | 0.92 |
| Gemma4-12B, Nemotron-Nano-v2-9B, GLM-4/Z1-9B | 4 | 4 | 0.90 |
| Qwen3-8B, Gemma4-E4B, Ministral-3 8B | 6 | 6 | 0.90 |
| Qwen3-4B, Nemotron-3-Nano-4B | 8 | 8 | 0.90 |

这套设置能降低 20K 请求发生 OOM 的风险，但很可能对 1K–5K 请求过于保守，特别是
31B/32B 模型目前一次只处理一个 sequence。

## 5. 长度和 batch size 是否应不同

### 5.1 计算上为什么不同

长上下文主要增加 prefill FLOPs 和 KV-cache 占用。在同一模型上，能够并行驻留的
20K sequences 通常显著少于 1K sequences。因此从纯性能角度，短 (L) 使用更高并发
是合理的。

### 5.2 为什么不应直接手工为每个长度写死 batch size

vLLM 的 continuous batching 和 PagedAttention 已经会根据可用 KV blocks 调度请求。
更高的 `max_num_seqs` 是上限，不代表长请求一定同时驻留。理想做法是：

1. 给每个模型设置足够大的候选 request queue；
2. 由 vLLM 根据 token/KV 预算降低长请求的实际并发；
3. 必要时按长度 bucket，减少长短请求互相造成的尾延迟；
4. 不重新加载模型，从而保留 model-bundle 优化。

当前代码的 `request_batch_size` 和 `max_num_seqs` 相同且都较小，尚未充分利用这一点。

### 5.3 科研上的额外约束

native-thinking modes 使用非零 temperature。改变 batch composition 或 tensor
parallelism 可能改变随机数消费顺序或浮点归约，从而改变生成 token。正式采用不同
batch 策略前，必须验证同一 request ID 在固定 seed 下的可复现性。

如果 batch size 与 (L) 系统相关，并且它又改变 stochastic output，那么 batch policy
会成为长度效应的潜在工程混杂。因此：

- 优先使用**模型级固定 engine profile + vLLM 动态调度**；
- 只有在 token-level determinism 测试通过后，才使用显式 length-aware batch policy；
- 若无法做到 request-level 可复现，则正式实验中保持每模型固定 batch policy。

## 6. 精度策略

### 6.1 当前正式策略

- 所有模型权重/主计算：BF16；
- Nemotron-Nano-v2-9B 和 Nemotron-3-Nano-4B：Mamba SSM cache 为 FP32；
- 不使用权重量化；
- 不使用 FP8 KV cache；
- 不同模型 revision 固定，不自动替换 checkpoint variant。

### 6.2 不建议仅为节省时间直接改用 FP8/INT8

对于 empirical law，量化后的 checkpoint 严格来说是不同的数值干预，可能改变
accuracy、bias、CoT style 和 truncation。它不是纯粹的基础设施加速。除非研究对象
明确改为量化模型，否则 confirmatory run 应保留 BF16。

可以探索 FP8 KV cache 或其他低精度路径，但需要：

1. 在代表性 (N,L) cells 上做 BF16 对照；
2. 比较 request-level output token IDs、parse rate、accuracy、bias 和 truncation；
3. 在看到正式结果前预先写明一致性阈值；
4. 未通过阈值时回退 BF16。

当前最稳妥的选择仍是 BF16；优化重点应放在调度、并发、I/O 和模型加载，而不是改变
模型数值精度。

## 7. Tensor parallelism

当前每个 worker 只暴露一张 GPU，`tensor_parallel_size=1`。4B–14B 模型应优先保留
单卡运行，因为 data parallel 的多模型吞吐通常优于用两张卡服务一个小模型。

31B/32B 模型在 H100 80GB 上可能单卡可运行，但 20K context 会压缩 KV-cache 并发。
是否对 Qwen3-32B 和 Gemma4-31B 使用 TP=2，应通过实测决定：

- 如果 TP=1 能稳定运行且 decode/prefill 吞吐合理，保留 TP=1，以维持 8 个并行
  workers；
- 如果 TP=1 的 20K 并发只能长期维持 1，且 TP=2 的单 bundle 吞吐提升足以抵消少一半
  workers，再考虑 TP=2；
- TP 配置一旦确定，应在正式 run manifest 中冻结，不在运行中按 accuracy 调整。

当前 launcher 是一 GPU 一 worker；支持 TP=2 需要同步修改 GPU claim 和资源调度，不能
只把 CLI 参数从 1 改成 2。

## 8. 建议的 H100 inference pilot

### 8.1 目的

pilot 只用于选择吞吐和内存配置，不用于选择表现更好的模型、prompt 或实验条件。
在 pilot 完成前不查看 accuracy/bias 汇总。

### 8.2 覆盖范围

每个模型至少覆盖：

- (L\in\{1K,10K,20K\})；
- (N\in\{1,10,20\})；
- 2 个固定 seeds；
- 该模型注册的 direct/enumeration/native-thinking modes。

模型可按规模分四组先选代表 checkpoint：

1. 4B/E4B；
2. 8B/9B；
3. 12B/14B；
4. 26B/31B/32B。

代表模型确定候选区间后，其余模型仍需至少做一次 1K/20K OOM 与模板 smoke test。

### 8.3 需要记录的指标

- model load 时间；
- time to first token；
- prefill tokens/s；
- decode tokens/s；
- requests/s；
- GPU utilization；
- 峰值显存和可用 KV cache；
- preemption/OOM 次数；
- batch wall time；
- output tokens/request 分布；
- 同 request ID 重跑的 token-level 一致率。

### 8.4 参数搜索顺序

1. 保持 BF16、TP=1 和当前 decoding 不变；
2. 以当前并发为安全 baseline；
3. 逐步提高 `request_batch_size` 和 `max_num_seqs`；
4. 如 vLLM 出现频繁 preemption，再约束 batched-token/KV 预算；
5. 只对 31B/32B 比较 TP=1 与 TP=2；
6. 仅在固定 seed 输出可复现时评估显式 length bucketing。

候选配置应按**最大稳定吞吐、无 OOM、可恢复、输出可复现**选择，而不能按 accuracy
选择。

## 9. Empirical law 与 mechanism 的关系

### 9.1 可以独立推进的部分

Empirical-law pipeline 的输入和输出为：

```text
frozen stimuli
  -> vLLM responses
  -> parsing / format compliance / CoT style
  -> accuracy and 10% trimmed bias
  -> N, L, logN, logL and interaction-law fitting
  -> held-seed / held-N / held-L / leave-one-model-out validation
```

该流程不读取：

- hidden states；
- attention maps；
- neurons/heads；
- activation patching；
- steering/ablation 输出；
- mechanism labels。

因此 mechanism 可以后续单独设计、单独运行，或者与 empirical inference 并行进行。

### 9.2 两部分应共享但不能混淆的内容

两条研究线最好共享：

- immutable model revisions；
- tokenizer/chat-template revisions；
- stimulus IDs 和 passage hashes；
- prompt-mode 定义；
- seed 和 provenance 记录。

这样后续可以把机制测量与行为结果按 request/stimulus ID 对齐。但这种关联不能自动
升级为 causal claim；causal mechanism 仍需要独立 intervention design。

### 9.3 推荐组织方式

- Empirical law：完整 161,280 responses，output-only，作为当前主实验；
- Mechanism：从冻结 stimulus bank 中预注册子集，保存 activations/干预结果；
- 两者使用不同 run roots、manifests、分析脚本和报告；
- mechanism 的结果不反向改变 empirical law 的主 estimand 或 model selection。

## 10. 当前未完成事项与建议讨论议程

### 已完成

- V3.1 预注册与实现；
- 3,360 stimuli 本地生成和独立审计；
- 数据公开上传 Hugging Face；
- 14-model bundle scheduler；
- 原子断点与 canonical merge；
- SciPy confirmatory / Torch CUDA optional analysis backend；
- 48 logical shards 和 161,280 requests 的完整审计规则。

### inference 开始前需要完成

1. 在最终 H100 image 上安装并锁定 vLLM/Transformers/CUDA 环境；
2. 对 14 个模型做 revision、tokenizer、chat template、reasoning delimiter smoke test；
3. 运行上述 H100 throughput/determinism pilot；
4. 冻结每模型 engine profile；
5. 判断 31B/32B 是否需要 TP=2；
6. 根据实测 bundle 时间重新校准 8-GPU 调度优先级；
7. 在任何正式 accuracy/bias 查看之前记录 inference-only amendment；
8. 再启动 161,280 请求的正式 inference。

### 建议与合作者集中讨论的五个问题

1. 目标机器是否确定为 8 × H100 80GB，还是存在 H100 94GB/H200/A100 混用？
2. 31B/32B 是否接受 TP=2，还是优先最大化跨模型 data parallelism？
3. native-thinking 在不同 batch composition 下是否能做到固定 seed 的 token-level
   reproducibility？
4. pilot 的吞吐/显存/重跑一致性阈值如何定义？
5. mechanism 子实验准备复用哪些模型、(N,L) cells 和 stimulus IDs？

## 11. 推荐最终原则

> 科学配置与性能配置分离：model revision、prompt、decoding、seed、dtype 和输出预算
> 属于冻结的科学协议；batch queue、worker scheduling、checkpoint I/O 等只有在证明不改变
> request-level 生成语义后，才作为性能优化调整。

当前最合理的下一步不是直接全量开跑，也不是立即换低精度，而是先在目标 H100 环境
完成一个小规模、仅看性能和可复现性的 inference pilot，然后冻结 profile，再启动正式
empirical-law inference。Mechanism 研究可以独立推进，不需要等待或阻塞这一流程。
