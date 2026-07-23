# Lambda GPU / Codex 执行指南

本文档用于交接给负责 Lambda GPU 实验的 Codex。实验的科学定义以 `plans/outline.md` 为准。

## 1. 给接手 Codex 的任务

你负责在 Lambda GPU 实例上准备、验证并运行 Realistic NIAH Counting 实验。第一阶段只做：

1. 检查仓库和运行环境；
2. 补齐主实验所需的 runner、prompt builder、model adapter、evaluator 和结果同步；
3. 冻结 150 条 master stimuli；
4. 完成 Qwen3-8B 的 36-run smoke test；
5. smoke 通过后，完成 Qwen3-8B 的 900-run 主网格；
6. 生成完整的结果、日志、manifest、耗时报告和 Google Drive 归档。

不要在 Qwen3-8B 工程质控通过前运行其余模型。

## 2. Codex 应运行在哪里

模型推理必须在 Lambda GPU 实例上运行，但 Codex 不一定安装在服务器上。两种方式都可以：

### 方式 A：本地 Codex 通过 SSH 控制 Lambda

这是推荐方式。用户继续在 Windows 上使用当前 Codex；Codex 每次收到运行任务时，通过 SSH 在 Lambda 上执行命令。优点是本地文件、对话和权限管理更集中。

### 方式 B：在 Lambda 上运行 Codex

也可以在服务器终端中使用 Codex，然后把本文档作为任务说明。它并不比方式 A 更能使用 GPU；GPU 能力来自命令运行所在的 Lambda 实例。

结论：**不需要只能使用服务器端 Codex。**

## 3. SSH 连接条件

用户已有一条 Ed25519 公钥。公钥需要添加到 Lambda workspace，并在创建实例时选择。公钥不能单独用于登录；本地还必须存在与它匹配的私钥。

实例创建后，需要向本地 Codex提供：

- 实例公网 IP；
- 用户名，Lambda on-demand 实例通常为 `ubuntu`；
- 对应私钥的本地路径或已经配置好的 SSH host alias；
- GPU 型号和数量；
- 实例磁盘或 persistent filesystem 信息。

Lambda 官方连接形式为：

```bash
ssh -i '<PRIVATE_KEY_PATH>' ubuntu@<INSTANCE_IP>
```

Windows 的 SSH config 可以配置为：

```sshconfig
Host lambda-niah
    HostName <INSTANCE_IP>
    User ubuntu
    IdentityFile <PRIVATE_KEY_PATH>
    IdentitiesOnly yes
    ServerAliveInterval 30
    ServerAliveCountMax 4
```

配置后，本地 Codex 可以使用：

```bash
ssh -o BatchMode=yes lambda-niah '<REMOTE_COMMAND>'
```

不要使用 `StrictHostKeyChecking=no`。首次连接时核对 host fingerprint。

官方说明：<https://docs.lambda.ai/public-cloud/on-demand/connecting-instance/>

## 4. 仓库与科学协议

仓库：

```text
https://github.com/Twist-Shan/Realistic_CoT_NiaH_Count.git
```

开始工作前必须完整阅读：

```text
AGENTS.md
plans/outline.md
README.md
```

记录：

```bash
git remote -v
git status --short --branch
git rev-parse HEAD
```

任何正式 run 都必须绑定一个确定的 git commit。不要用无法追溯的 dirty worktree 直接跑完整实验。

## 5. 当前仓库的已知状态

仓库已有：

- realistic haystack/needle 数据生成代码；
- sentence-level insertion 和 metadata；
- Transformers 串行推理入口；
- CoT、hidden-state 和 attention 分析代码；
- 基础 tests；
- 面向 Colab mounted Drive 的归档逻辑。

主实验仍需要确认或补齐：

- strict query-first/query-last prompt builder；
- Direct、indexed enumeration、native thinking 三模式；
- Qwen/Gemma/Llama/OLMo adapters；
- 官方 reasoning/final parser；
- vLLM 或等价 batched runner；
- request-ID 幂等和断点续跑；
- 150 条 stimuli 的一次性 freezer；
- 固定 post-insertion passage 长度的 deterministic budget-search wrapper；
- Lambda 环境中的 Google Drive 同步。

`scripts/gen_responses.py` 当前是逐样本 Transformers generation，不能直接视为 6,300-run 主实验 runner。现有 Google Drive 路径判断主要针对 Colab 的 `/content/drive`，Lambda 上不能假定已经挂载 Drive。

## 6. 不可改变的实验定义

未经用户明确同意，不得修改：

- \(T=\{2K,5K,10K\}\)；
- \(T\) 是 canonical tokenizer 下插入 needles 后、加入 query/chat template 前的 passage tokens；
- \(N=\{1,2,3,4,5,6,8,10,20,30\}\)；
- seeds `1234..1238`；
- 每个 \((T,N)\) cell 5 条 stimuli；
- query first/query last 的严格位置定义；
- Direct、Enumeration、Native thinking 的 prompt 文本；
- 原有 sentence-level needle insertion 行为；
- 每个模型的官方 chat template 和 thinking 开关；
- first-pass output 保留规则；
- 同一 stimulus 跨条件和模型复用。

如果实现与 outline 冲突，先暂停并报告，不要自行“优化”实验设计。

## 7. 服务器环境检查

登录后先运行只读检查：

```bash
nvidia-smi
df -h
free -h
uname -a
python3 --version
git --version
```

报告：

- GPU 型号、数量、显存；
- driver 和 CUDA；
- 可用磁盘；
- 系统架构是 x86_64 还是 ARM64；
- 预计模型缓存位置；
- 运行结果的本地目录。

GH200 常见为 ARM 主机。任何依赖 wheel 或 kernel 不兼容都要在 smoke 前暴露。

## 8. 建议的服务器目录

本次实例已经挂载 persistent filesystem，使用以下固定目录：

```text
/lambda/nfs/Twist-CoT-Count-Multi-Model/
  Realistic_CoT_NiaH_Count/
  hf-cache/
  pip-cache/
  runs/
  archives/
  logs/
```

不要把模型 cache、hidden states 或大量运行结果写进 Git 仓库。

当前 SSH 为 `ubuntu@209.20.158.200`，GPU 为 1×H100 PCIe 80 GB。Python
环境位于 `/home/ubuntu/venvs/realistic-niah-vllm`。不要把模型或结果默认
写入系统盘。

## 9. 环境安装

本次实例固定使用：

```bash
cd /lambda/nfs/Twist-CoT-Count-Multi-Model/Realistic_CoT_NiaH_Count
bash scripts/lambda_python.sh -m pip install -r requirements-inference.txt
bash scripts/lambda_python.sh -m pip freeze
```

`requirements-inference.txt` 固定 `vLLM==0.25.1`。必须通过
`scripts/lambda_python.sh` 启动 Python；该 wrapper 会设置 persistent cache，
并把 vLLM 辅助 CUDA 13 runtime 与 PyTorch CUDA 12.8 runtime 加入动态库搜索
路径。已验证的核心版本为：

- Python 3.10；
- PyTorch 2.11.0+cu128；
- Transformers 5.14.1；
- vLLM 0.25.1。

不要为了某个模型无条件升级整个环境：

- Qwen3、Gemma 4 和 OLMo 可能需要不同版本；
- OLMo Hybrid 的 model card 要求较新的 Transformers；
- Gemma 4 使用 `AutoProcessor`/multimodal model class，但本实验只走 text path；
- 必要时为不同模型维护独立、固定版本的环境；
- 每个环境保存 `pip freeze`。

Hugging Face token 通过环境变量、secret manager 或交互式登录提供。不要把 token 写入 config、shell history、日志或 Git。

当前 Qwen、Gemma 4 和 OLMo 仓库公开可读；两个官方 Meta Llama 仓库为
gated。进入 Llama 阶段前，必须确认服务器登录的 Hugging Face 账号已接受
相应许可。

## 10. 先测试，再实现主运行

安装后先执行：

```bash
bash scripts/lambda_python.sh -m compileall src scripts
PYTHONPATH=src bash scripts/lambda_python.sh -m pytest
```

若完整测试很慢，可先运行和数据、prompt、parser、runner 直接相关的 tests，但在 smoke 前仍需给出完整测试结果或明确说明未运行部分。

实现新 runner 时遵循仓库结构：

- reusable logic 放在 `src/`；
- CLI 放在 `scripts/`；
- resolved config 放在 `configs/`；
- tests 放在 `tests/`；
- outputs 放在独立的 `runs/`。

不要把主逻辑堆进 notebook。

## 11. Master dataset

只生成一次 150 条 stimuli：

```text
3 lengths × 10 needle counts × 5 seeds = 150
```

每个 cell 必须恰好有：

```text
5 rows
5 distinct stimulus_id
seeds = 1234..1238
```

长度口径：

```text
target_passage_tokens = T
canonical_tokenizer = Qwen/Qwen3-8B
passage = clean filler + inserted needles
excluded_from_T = system message + task block + chat template + output
```

当前 `target_haystack_tokens` 表示插针前的 clean filler 长度，不能直接令 `target_haystack_tokens=T`。必须实现一个外层 deterministic wrapper：

1. 先生成固定 needle 文本；
2. 从 `T - needle_token_budget` 得到 clean filler 初值；
3. 按现有 sentence-level 插入算法生成 final passage；
4. 用 canonical tokenizer 重新计数；
5. 确定性搜索 clean filler budget，直到 final passage 恰好为 \(T\) tokens；
6. 当前 window 无法精确命中时，用 seed 派生的 retry index 更换 window；
7. 超过最大 retries 后失败，不能接受错误长度。

禁止在插针后裁剪 passage，也不能删除、缩短或改写 needle 来凑长度。

插入配置：

```text
target_passage_tokens = T
target_haystack_tokens = searched_clean_filler_budget
randomize_needle_insertion = true
randomize_needle_seed = paired_seed
sentence_level_insertion = true
word_level_insertion = false
insertion_positions = [0] * N
```

完成后输出：

- `stimuli.jsonl`
- `manifest.json`
- `contamination_audit.json`
- `cell_counts.json`
- SHA256 manifest

在模型生成开始前运行审计脚本，并明确打印：

```text
rows = 150
cells = 30
rows_per_cell = 5
seed_min = 1234
seed_max = 1238
duplicate_stimulus_id = 0
canonical_passage_length_mismatch = 0
truncated_or_missing_needles = 0
task_or_template_tokens_in_T = 0
```

`stimuli.jsonl` 还必须保存 clean filler tokens、canonical passage tokens、每个模型 tokenizer 下的 passage tokens、完整 rendered input tokens、length-search attempts 和 retry index。

## 12. Prompt 和 runner 验证

对 Qwen3-8B 各抽至少 3 条，保存并人工检查：

- Direct × query first/last；
- Enumeration × query first/last；
- Native thinking × query first/last。

必须验证：

- query last 的 passage 前没有 task cue；
- Direct 和 Native 的 messages 完全相同；
- 两者只差官方 thinking switch；
- Enumeration 关闭 thinking；
- 每条 final passage 在 canonical tokenizer 下严格等于对应 \(T\)；
- query first/last 只改变 passage 外部的 task block 位置，不改变 passage；
- rendered prompt、input IDs 和 generation boundary 被保存；
- count/list/thought parser 对 `N=1`、`N=6`、`N=30` 和截断输出有测试。

## 13. 36-run smoke test

Smoke cells：

```text
T = 2K, 10K
N = 5, 6, 30
seed = 1234
```

每个 cell 运行 6 个条件，共：

```text
6 cells × 1 stimulus × 6 conditions = 36 generations
```

Smoke 完成后先提交状态报告，不要直接静默进入完整 run。报告至少包含：

- 36/36 request 是否完成；
- canonical tokenizer 下的 passage length audit，按 \(T,N\) 汇总；
- clean filler token 数的 min/max；
- length-search attempts、window retries 和失败数；
- 每个条件的 parsed count；
- parser failure 和 truncation；
- rendered prompt 样例；
- thinking/final 是否正确分离；
- peak VRAM；
- prefill/decode throughput；
- wall time；
- 本地输出路径；
- Google Drive 归档路径及同步验证；
- 根据实测吞吐更新后的 900-run 时间估计。

以下条件全部满足后才能运行 Qwen3-8B 完整网格：

- dataset audit 通过；
- `canonical_passage_length_mismatch = 0`；
- `truncated_or_missing_needles = 0`；
- 36 个 request 无缺失；
- parser failure <1%；
- 非预期 truncation <1%；
- context hash 在六条件中一致；
- Drive 归档成功且服务器本地副本仍存在。

## 14. Qwen3-8B 完整主网格

规模：

```text
30 cells × 5 stimuli × 6 conditions = 900 generations
```

规划阶段的纯推理时间约为：

- A6000：1.8–3.3 h；
- GH200：0.5–0.9 h；
- H100：0.6–1.3 h；
- B200：0.3–0.7 h。

首次环境安装、Qwen3-8B 下载和冷启动另留 0.5–2 h。Smoke 完成后必须用实测 throughput 更新这些区间。

使用 shard 运行，例如每个 shard 50–200 requests。每个 request 使用稳定 ID：

```text
<model>/<prompt_mode>/<query_order>/<stimulus_id>
```

要求：

- shard 先写 `.tmp`，完成后 atomic rename；
- 每个 shard 完成后更新 manifest；
- 重启时按 request ID 去重；
- 不重复计费已经完成的 request；
- 每个 shard 保存 wall time 和 token throughput；
- 保留 first-pass output；
- 不因结果不符合假设而改 prompt 或 seed。

长任务应在 `tmux` 或等价的可恢复会话中运行：

```bash
tmux new -s niah
```

不要只依赖一个会随 SSH 断开而终止的前台进程。

## 15. Google Drive 结果同步

Lambda 服务器没有 Colab 的 `/content/drive` 挂载。建议使用 `rclone`，或实现具有相同语义的同步接口。

安全要求：

- OAuth credentials 和 tokens 不进入 Git；
- 主运行先写服务器本地磁盘；
- 不要在生成过程中向 Drive 写成千上万个小文件；
- 每个 shard 本地完成后再归档或批量同步；
- Drive 同步失败不能删除本地结果。

推荐结构：

```text
gdrive:Realistic_CoT_NiaH_Count/
  realistic_niah_v1/
    <run_id>/
      dataset/
      shards/
      aggregate/
      manifests/
```

推荐流程：

1. 本地完成 shard 并 atomic rename；
2. 写入 SHA256 manifest；
3. 将 shard 或 run directory 打包；
4. 上传到 Drive；
5. 检查远端文件存在、大小合理；
6. 记录 remote path、upload time 和 archive hash；
7. 保留本地副本，直到整个 run 完成并验证。

如果 `rclone` 尚未配置，先停止在同步测试阶段，请用户完成一次 OAuth 授权。不要伪造“已同步成功”。

## 16. 运行结果目录

```text
runs/
  realistic_niah_v1/
    dataset/
      stimuli.jsonl
      manifest.json
      contamination_audit.json
      cell_counts.json
    Qwen_Qwen3-8B/
      direct/
        query_first/
        query_last/
      enumeration/
        query_first/
        query_last/
      native_thinking/
        query_first/
        query_last/
      run_manifest.json
    aggregate/
      cell_metrics.parquet
      paired_metrics.parquet
      figures/
      report.html
```

## 17. 暂停条件

出现以下任一情况时暂停新 shard，保留已有结果并报告：

- dataset hash 或 paired mapping 不一致；
- 任一 final passage 在 canonical tokenizer 下不等于目标 \(T\)；
- 任一 needle 被长度控制过程截断、删除、缩短或改写；
- length-search 超过 retry 上限，无法精确命中 \(T\)；
- thinking switch 没有改变官方模板；
- parser failure ≥1%；
- 某条件系统性 OOM、截断或缺失；
- 预计费用超过预算上界 1.5 倍；
- 磁盘空间可能不足；
- Google Drive 同步持续失败；
- model/tokenizer revision 发生变化；
- 需要改变 prompt、seed、插入算法或输出预算。

## 18. 每次状态报告格式

```text
Instance:
GPU / count / VRAM:
Git commit:
Environment hash or pip-freeze path:
Run ID:
Stage:
Completed requests / expected:
Failed or missing requests:
Passage length mismatches:
Missing or truncated needles:
Length-search retries / failures:
Parser failures:
Truncations:
Peak VRAM:
Prefill tok/s:
Decode tok/s:
Elapsed time:
Estimated remaining time:
Estimated remaining cost:
Local result path:
Google Drive path:
Last verified archive/hash:
Blockers:
Next action:
```

## 19. 禁止事项

- 不要未经允许改变实验网格。
- 不要为每个 prompt 条件重新生成 stimulus。
- 不要把 `target_haystack_tokens=T` 当成固定插针后 passage 长度。
- 不要在插针后裁剪 final passage 来凑 \(T\)。
- 不要把 query、system message 或 chat template 的 tokens 计入 \(T\)。
- 不要把 Llama prompted CoT 标成 native thinking。
- 不要把 OLMo 不同 checkpoint 的比较解释成同 checkpoint 因果效应。
- 不要静默删除 parse failure、OOM 或 truncation。
- 不要只保存 aggregate metrics；必须保留逐 request 输出。
- 不要提交模型、cache、hidden states 或大型 run outputs。
- 不要打印 Hugging Face、Google 或其他访问令牌。
- 不要在用户未要求时 commit、push 或删除远端数据。

## 20. 第一阶段完成定义

这套 5-seed 网格是 pilot / exploratory experiment。每个 cell 的 accuracy 只能以 0.20 为单位变化，不应从单格结果宣称稳定的 \(N_{50}\) 或精细 scaling law。

只有以下项目全部存在，Qwen3-8B 第一阶段才算完成：

- 150 条 master stimuli，以及证明每条 canonical final passage 精确等于 \(T\) 的审计；
- clean filler budget、length-search attempts、retry index 和 model-specific passage/input lengths；
- 36-run smoke report；
- 900 个完整、无重复的 request；
- 六条件配对检查；
- raw outputs、parsed outputs 和 manifests；
- cell-level 与 paired metrics；
- GPU 时间/成本报告；
- 本地 SHA256 manifest；
- 已验证的 Google Drive 归档；
- 清楚记录的代码 commit 和模型 revision。
