# NSF ACCESS / Anvil：V3.1 Empirical-Law 提交方法

本文供代为提交作业时直接执行。提交内容是预注册的 V3.1 behavior / empirical-law
GPU 推理与最终合并；统计拟合不占用本次 H100 作业，待推理审计通过后再单独运行。

## 1. 固定远端来源与实验选择

### 代码

- GitHub：<https://github.com/Twist-Shan/Realistic_CoT_NiaH_Count>
- 固定 commit：`cdb6e8dd2781b83b46883999d1baa566822fff47`
- 不使用可移动的 `main`/branch tip；正式运行必须 checkout 上述 commit。
- 不做 sparse checkout。该 commit 中实际入口为
  `infra/anvil/realistic_niah_v3_1/submit_anvil.sh`，它会继续调用固定的
  prepare、bundle worker、merge、`src/realistic_niah_v3_1/` 和
  `configs/realistic_niah_v3_1.json`。

### 冻结数据

- Hugging Face：
  <https://huggingface.co/datasets/twistshan/realistic-niah-count-empirical-law>
- 固定 dataset revision：`af28be936adf92d40971aed4fa341c92b6ecf799`
- 下载该 revision 的完整原始 snapshot，不使用 Dataset Viewer/parquet 转换。
- snapshot 文件为：`.gitattributes`、`README.md`、`SHA256SUMS`、
  `stimuli.jsonl`、`manifest.json`、`audit_report.json`、
  `cell_counts.json`、`contamination_audit.json`。
- 运行时必需的是 `stimuli.jsonl`、`manifest.json`、`audit_report.json`；
  其余文件保留用于来源和完整性审计。

冻结 stimulus 选择是完整 Cartesian grid，不取子集：

- passage tokens：`1000, 2000, 3000, 5000, 8000, 10000, 15000, 20000`；
- needle count：`1–10, 12, 15, 18, 20`；
- paired seeds：`1234–1263`；
- 共 `8 × 14 × 30 = 3,360` 个 stimuli；
- `stimuli.jsonl` 应为 184,690,729 bytes，SHA256 为
  `afed18fe24d3c684b7f342a3c5cc119fe3bd4033487d25c97cbbe4fc21c0d159`。

### 模型与 prompt modes

不在提交时手选 checkpoint；以下 14 个 Hugging Face repo/revision 全部由
`configs/realistic_niah_v3_1.json` 和代码强制：

| Label | Hugging Face model ID | Revision |
| --- | --- | --- |
| Qwen3-4B | `Qwen/Qwen3-4B` | `1cfa9a7208912126459214e8b04321603b3df60c` |
| Qwen3-8B | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` |
| Qwen3-14B | `Qwen/Qwen3-14B` | `40c069824f4251a91eefaf281ebe4c544efd3e18` |
| Qwen3-32B | `Qwen/Qwen3-32B` | `9216db5781bf21249d130ec9da846c4624c16137` |
| Gemma4-E4B | `google/gemma-4-E4B-it` | `ee0ef6023621cff504d758262d4e04895a5af4a2` |
| Gemma4-12B | `google/gemma-4-12B-it` | `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` |
| Gemma4-26B-A4B | `google/gemma-4-26B-A4B-it` | `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` |
| Gemma4-31B | `google/gemma-4-31B-it` | `842da3794eaa0b77d5f08bae87a17459d91ff475` |
| Nemotron-Nano-v2-9B | `nvidia/NVIDIA-Nemotron-Nano-9B-v2` | `6533e8de2c68e4536bf7c411d7a3ce5734111476` |
| Nemotron-3-Nano-4B | `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` | `dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f` |
| GLM-4-9B-0414 | `zai-org/GLM-4-9B-0414` | `645b8482494e31b6b752272bf7f7f273ef0f3caf` |
| GLM-Z1-9B-0414 | `zai-org/GLM-Z1-9B-0414` | `b221b06fefb23ca320922cf6e68ab5f2fb82de81` |
| Ministral-3-Instruct-8B | `mistralai/Ministral-3-8B-Instruct-2512` | `5b26027e7b19eeb4b7352e1fed3926375dd2cb4d` |
| Ministral-3-Reasoning-8B | `mistralai/Ministral-3-8B-Reasoning-2512` | `81eaece1948f3875421d9a45bc55487d10e2d894` |

前 10 个 switchable 模型各跑 `direct`、`enumeration_index`、
`enumeration_bullet`、`native_thinking`；GLM-4 和 Ministral-Instruct 各跑前三个
non-thinking modes；GLM-Z1 和 Ministral-Reasoning 各只跑 `native_thinking`。

因此是 14 个物理模型 bundle、48 个逻辑 model-mode shard，每个 shard 使用全部
3,360 stimuli，共 161,280 个请求。

## 2. 登录并检查 allocation

```bash
ssh x-yzhong6@anvil.rcac.purdue.edu

mybalance
myquota
showpartitions
sfeatures
```

应能看到 allocation `mth260088-ai`、partition `ai` 和 feature `H100`。不要在
login node 上直接运行模型推理。

## 3. 从 GitHub 获取固定代码

```bash
CODE_COMMIT="cdb6e8dd2781b83b46883999d1baa566822fff47"

git clone https://github.com/Twist-Shan/Realistic_CoT_NiaH_Count.git \
  "$PROJECT/niah"
cd "$PROJECT/niah"
git checkout --detach "$CODE_COMMIT"

test "$(git rev-parse HEAD)" = "$CODE_COMMIT"
test -z "$(git status --short)"
```

如果 `$PROJECT/niah` 已存在，不要覆盖：先确认其中没有未提交工作，再执行
`git fetch origin` 和 `git checkout --detach "$CODE_COMMIT"`。正式提交要求完整
`.git` 和 clean worktree。

## 4. 在单卡 Slurm session 中准备环境与 Hugging Face 缓存

依赖安装、数据下载和模型预热不应占用 login node。用同一 allocation 先申请一个
短的单卡 compute shell；若已有可用的 CPU allocation，也可改用其 `shared`
partition。

```bash
sinteractive -A mth260088-ai -p ai -C H100 \
  -N 1 -n 1 --gpus-per-task=1 -c 8 --mem=120G -t 12:00:00
```

首次使用时：

```bash
module load modtree/gpu
module load conda

ENV_DIR="$PROJECT/envs/$USER/niah-v31"
conda create --prefix "$ENV_DIR" python=3.11 pip -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"
python -m pip install -r "$PROJECT/niah/requirements-inference-v3.txt"

mkdir -p "$PROJECT/hf-cache"
hf auth whoami || hf auth login
```

数据集本身公开且不 gated；`hf auth` 主要用于随后下载需要授权的模型。不要把
Hugging Face token 写入脚本或日志。

下载完整固定 snapshot：

```bash
RUN_ROOT="$PROJECT/runs/realistic_niah_v3_1/20260819_formal"

REALISTIC_NIAH_REPO_ROOT="$PROJECT/niah" \
REALISTIC_NIAH_PYTHON="$ENV_DIR/bin/python" \
REALISTIC_NIAH_HF_BIN="$ENV_DIR/bin/hf" \
  bash scripts/download_realistic_niah_v3_1_dataset.sh "$RUN_ROOT"
```

校验下载内容：

```bash
cd "$RUN_ROOT/dataset"
sha256sum -c SHA256SUMS
test "$(wc -l < stimuli.jsonl)" -eq 3360
test "$(stat -c %s stimuli.jsonl)" -eq 184690729

"$ENV_DIR/bin/python" -c \
  'import json; a=json.load(open("audit_report.json")); assert a["passed"] is True; assert a["rows_checked"]==3360; print("DATASET AUDIT PASS")'
```

不要调用 freeze 脚本重新生成数据，也不要下载旧 namespace
`stwistzz/realistic-niah-count-empirical-law`；该名称已重命名。

### 预热固定模型 snapshot

正式作业会从 `$PROJECT/hf-cache` 读取权重。先接受所有 gated model 的许可，并在
当前 compute shell 中把 14 个固定 revision 下载并校验；不要等 8×H100 正式
作业开始后再下载。

```bash
ENV_DIR="$PROJECT/envs/$USER/niah-v31"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"
cd "$PROJECT/niah"

MODEL_LIST="$PROJECT/v31-model-snapshots.tsv"
PYTHONPATH=src "$ENV_DIR/bin/python" -c \
  'from realistic_niah_v3_1.spec import MODEL_LABELS, MODEL_REVISIONS, MODEL_SPECS; print("\n".join(f"{MODEL_SPECS[x].model_id}\t{MODEL_REVISIONS[x]}" for x in MODEL_LABELS))' \
  > "$MODEL_LIST"

test "$(wc -l < "$MODEL_LIST")" -eq 14
test "$(cut -f1 "$MODEL_LIST" | sort -u | wc -l)" -eq 14

while IFS=$'\t' read -r model_id revision; do
  hf download "$model_id" \
    --revision "$revision" \
    --cache-dir "$PROJECT/hf-cache" \
    --max-workers 4
  hf cache verify "$model_id" \
    --revision "$revision" \
    --cache-dir "$PROJECT/hf-cache" \
    --fail-on-missing-files
done < "$MODEL_LIST"

exit
```

`MODEL_LIST` 直接由冻结代码 registry 生成，因此下载对象应与上表完全一致。若
任一模型返回 401/403，先在其 Hugging Face 页面接受许可并确认 `hf auth whoami`，
不能临时替换成别的 checkpoint。

若 12 小时内未下载完，退出后重复本节的单卡 session 和下载循环；`hf download`
会复用已有缓存并续传。确认全部校验通过后再提交正式作业。

## 5. Dry-run 后提交 8×H100

```bash
cd "$PROJECT/niah"
CODE_COMMIT="cdb6e8dd2781b83b46883999d1baa566822fff47"
RUN_ROOT="$PROJECT/runs/realistic_niah_v3_1/20260819_formal"

bash infra/anvil/realistic_niah_v3_1/submit_anvil.sh \
  "$RUN_ROOT" --workers 8 --expected-commit "$CODE_COMMIT" --dry-run

# 确认输出为 nodes=2、ntasks=8、H100、account=mth260088-ai 后提交：
bash infra/anvil/realistic_niah_v3_1/submit_anvil.sh \
  "$RUN_ROOT" --workers 8 --expected-commit "$CODE_COMMIT"
```

默认资源是 2 个 H100 节点、8 个单卡 worker、每 worker 12 CPU、每节点
480 GB 内存、最长 48 小时。这里是 8 个独立 worker 动态消费 14 个 bundle，
不是单模型 TP=8。不得修改 model revisions、BF16、seed、prompt、decoding 或
冻结数据。

## 6. 监控、续跑与验收

```bash
squeue -u "$USER"
scontrol show job JOB_ID
wait_time -j JOB_ID
tail -f "$RUN_ROOT/orchestration/slurm/"*.out
```

若失败或达到 48 小时上限，先确认旧 job 已离开 `squeue`，再用相同
`RUN_ROOT` 和提交命令重投。不要同时对同一 `RUN_ROOT` 提交两个作业。

成功后检查：

```bash
"$PROJECT/envs/$USER/niah-v31/bin/python" -c \
  'import json,os; p=os.path.join(os.environ["PROJECT"],"runs/realistic_niah_v3_1/20260819_formal/orchestration/final_shard_audit.json"); a=json.load(open(p)); assert a["passed"] is True; assert a["requests"]==a["unique_request_ids"]==161280; print("INFERENCE AUDIT PASS", p)'

jobinfo JOB_ID
seff JOB_ID
jobsu JOB_ID
mybalance
```

验收标准是 `final_shard_audit.json` 中 `passed=true`，且总请求数和唯一请求数
均为 161,280。后续 parsing、统计分析和 empirical-law fitting 应使用独立 CPU
作业或本地环境，不能在 login node 运行。

Anvil 官方文档：
[Job Submission](https://docs.rcac.purdue.edu/userguides/anvil/jobs/)；
[File Management](https://docs.rcac.purdue.edu/userguides/anvil/file_management/)。
