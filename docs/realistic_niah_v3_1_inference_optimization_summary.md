# Realistic NIAH V3.1：Inference Optimization Summary

## 结论

- 数据已经生成并审核：3,360 条 stimuli，对应 161,280 次模型请求。
- 正式推理使用 vLLM、BF16 和固定模型版本，不使用 FP8/INT8 量化。
- 先在 H100 上做小规模性能测试，再冻结配置并启动全量推理。
- Empirical law 与 mechanism 实验相互独立，可以并行推进。

## 已完成的优化

1. **每个模型只加载一次**：14 次模型加载即可完成 48 个 model-mode shards，避免重复加载。
2. **断点续跑**：每个 batch 原子写入；中断后从已完成部分继续。
3. **减少磁盘写入**：结果中保存 prompt hash 和必要输出，不重复保存完整 prompt。
4. **优先运行慢模型**：31B/32B 和 native-thinking 任务优先调度，减少多 GPU 尾部等待。
5. **启用 prefix caching**：复用相同前缀，降低重复 prefill 成本。

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
