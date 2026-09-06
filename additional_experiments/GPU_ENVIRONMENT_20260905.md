# GPU 部署与环境检查：2026-09-05

SSH 权限已恢复；`ubuntu@68.209.74.112` 登录成功。实验包已部署，但 CUDA 驱动初始化返回 802，尚未运行任何新模型推理。以下记录不构成任务行为或机制实验结果。

## 已完成

- 确认 H100 80GB，显存 81,559 MiB，检查前占用 0 MiB；驱动 580.105.08，GPU 为 Pass-Through 模式。
- 确认 `/lambda/nfs/CoT-Native-thinking-v5` 为 NFS4 挂载。
- 独立目录：`/lambda/nfs/CoT-Native-thinking-v5/additional_experiments/task_transfer_20260905_v2/`。
- 已上传并解压 `repo_snapshot/`；压缩包远程 SHA-256 为 `19db1895cb979a535a3974b9c0d802fa12cb0337deaa89caa76145b1415d7bda`，与本地一致。未覆盖旧实验代码和结果。
- 使用既有 `venv_v6_20260828/bin/python`：Torch 2.7.0、Transformers 5.14.1；未安装或更换依赖。
- Smoke 与 pilot 冻结输入、旧源码和旧表征基的 hash 检查通过。
- 在 `hf_cache` 中确认固定版本 Qwen3-8B 的 5 个权重分片，以及 Gemma4-E4B 的单个权重文件均存在且非空。没有重新计算全部权重文件的内容 hash。
- 两模型的真实 tokenizer 对 smoke/pilot 全部任务、模式完成 432 次 prompt 渲染，均通过。第一次预检查错误地要求 Gemma 必须有分片索引；修正为兼容单文件权重后通过。原始失败日志保留，实验源码和冻结输入未改变。

| 模型 | Pilot non-thinking tokens | Pilot native-thinking tokens |
|---|---:|---:|
| Qwen3-8B | 10,107–10,261 | 10,088–10,241 |
| Gemma4-E4B | 10,215–10,617 | 10,204–10,605 |

## 当前阻塞和已尝试操作

1. `torch.cuda.is_available()` 为 `False`；最小 CUDA 张量分配报 `Error 802: system not yet initialized`。
2. 绕过 PyTorch、直接调用 `libcuda.so.1` 的 `cuInit(0)`，同样返回 **802**，因此故障已在实验代码之外复现。
3. `nvidia-smi -q` 中 Fabric 的 `State: In Progress`，`Status: N/A`。
4. 已尝试启动预装的 `nvidia-fabricmanager`。服务因无法查询 NVSwitch 设备而失败，报 `NV_WARN_NOTHING_TO_DO`。客户机未暴露实际 NVSwitch 设备；不能据此认定客户机 FM 未运行就是根因。
5. 确认没有计算进程后，对本实例 GPU 0 执行一次重置，工具报告重置成功；随后 Fabric 和 CUDA 状态未恢复。未重启或终止实例。

这些证据与云平台侧的 Fabric/分区初始化问题相符，但客户机日志不足以确认根因。NVIDIA 文档说明 H100 的 CUDA 初始化依赖 GPU 完成 Fabric 注册，并区分客户机与 Service VM 对 NVSwitch 的管理职责：[Fabric Manager 官方文档](https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/)。需要平台检查该实例对应的 GPU 注册和分区状态，或在用户确认的 CUDA 可用实例上继续。

## 证据与恢复入口

本地证据在 `runs/connectivity/remote_deployment_20260905/`；远程在上述隔离目录的 `deployment/`：

- `preflight_20260905_v2.json`：最终状态 `CPU_CHECKS_PASS_GPU_BLOCKED`，耗时 51.80 秒。
- `cuda_driver_check.txt`：`cuInit return code: 802`。
- `nvidia-smi_after_reset.txt` 和 `fabric-manager.log`：GPU 与服务状态。
- `launch_smoke_68_209_74_112.sh`：CUDA 检查通过后才启动 Qwen/Gemma smoke；检查失败时退出。

恢复后可运行：

```bash
bash /lambda/nfs/CoT-Native-thinking-v5/additional_experiments/task_transfer_20260905_v2/deployment/launch_smoke_68_209_74_112.sh
```

当前没有运行中的模型实验、后台重试或自动启动监控。精确 user prompts 在 `repo_snapshot/additional_experiments/runs/task_transfer_20260905_v2/frozen/user_prompts.jsonl`，六份完整示例在同级 `prompt_examples/`；模型实际 chat template 和 token IDs 将随每次模型运行记录。

## 可交给 Lambda 支持的说明（尚未发送）

Instance `68.209.74.112` is reachable over SSH and exposes one idle H100 80GB HBM3 in Pass-Through mode. NVIDIA driver: 580.105.08. CUDA initialization fails with error 802, independently reproduced by calling `cuInit(0)` directly through `libcuda.so.1`, without PyTorch or model code. `nvidia-smi -q` reports Fabric State `In Progress`, Status `N/A`. A single reset of the idle GPU completed successfully, but CUDA and Fabric status did not recover. Starting the installed guest Fabric Manager failed with `NV_WARN_NOTHING_TO_DO` while querying NVSwitch devices; no actual NVSwitch device is exposed to the guest. Please inspect the backing GPU's Fabric registration and partition initialization. Diagnostic logs are available. No experiment workload is running.
