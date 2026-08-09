# Realistic NIAH V3.1：Inference Optimization Summary

## 结论

- 数据已经生成并审核：3,360 条 stimuli，对应 161,280 次模型请求。
- 正式推理使用 vLLM、BF16 和固定模型版本，不使用 FP8/INT8 量化。
- 先在 H100 上做小规模性能测试，再冻结配置并启动全量推理。
- Empirical law 与 mechanism 实验相互独立，可以并行推进。

## 目前已经使用的优化

| 优化 | 当前做法 | 作用 |
|---|---|---|
| vLLM 推理 | 使用 continuous batching 和 PagedAttention | 提高吞吐并降低 KV-cache 浪费 |
| 模型只加载一次 | 同一模型的所有 prompt modes 连续运行 | 14 次加载完成 48 个 shards，少加载 34 次 |
| Prefix caching | `enable_prefix_caching=true` | 复用共同前缀，减少重复 prefill |
| 分模型并发 | 大模型 batch 小，小模型 batch 大 | 降低 20K 上下文 OOM 风险 |
| 单卡并行 | 每张 GPU 一个 worker，当前 TP=1 | 让多张卡同时运行不同模型 |
| 慢任务优先 | 31B/32B 和 native-thinking 优先 | 减少多 GPU 最后的等待时间 |
| 断点续跑 | 每个 batch 原子写入 checkpoint part | 中断后不必重新运行已完成请求 |
| 单次合并 | 完成后才生成 canonical `requests.jsonl` | 避免反复重写大文件 |
| 精简输出 | 保存 prompt hash，不重复保存完整 prompt | 减少磁盘写入和同步量 |

目前**尚未启用**：按长度动态调整 batch、TP=2、FP8/INT8 量化、FP8 KV cache。这些必须先经过 H100 pilot。

## 当前模型配置

| 模型组 | Batch size | `max_num_seqs` | GPU 显存比例 |
|---|---:|---:|---:|
| Qwen3-32B、Gemma4-31B | 1 | 1 | 0.92 |
| Gemma4-26B-A4B、Qwen3-14B | 2 | 2 | 0.92 |
| Gemma4-12B、Nemotron 9B、GLM/Z1 9B | 4 | 4 | 0.90 |
| Qwen3-8B、Gemma4-E4B、Ministral 8B | 6 | 6 | 0.90 |
| Qwen3-4B、Nemotron 4B | 8 | 8 | 0.90 |

这些是避免 20K 长上下文 OOM 的保守初始值，不是最终最优值。

## H100 pilot 要验证什么

- 每个模型测试 1K、10K、20K 三种长度。
- 逐步提高 batch size，记录吞吐、显存、OOM 和重跑一致性。
- 仅对 31B/32B 比较 TP=1 与 TP=2。
- 只有固定 seed 的输出可复现时，才考虑按长度动态调整 batch。
- 配置只按稳定性和速度选择，不能根据 accuracy/bias 选择。

## 建议运行顺序

1. 锁定 vLLM、Transformers、CUDA、模型和 tokenizer 版本。
2. 对 14 个模型做加载与 prompt-format smoke test。
3. 运行 H100 pilot。
4. 冻结每个模型的 batch size、TP 和显存配置。
5. 启动 161,280 次正式推理。
6. 推理完成后在 CPU 或 GPU 上进行 parsing、统计和 law fitting。

## 不要改变的实验条件

- 模型 revision、prompt、seed、decoding、BF16 精度和输出预算必须固定。
- 性能优化只能调整调度、并发、checkpoint 和 I/O。
- 如果一种优化改变同一 request 的生成结果，就不能直接用于正式实验。

公开数据：<https://huggingface.co/datasets/stwistzz/realistic-niah-count-empirical-law>
