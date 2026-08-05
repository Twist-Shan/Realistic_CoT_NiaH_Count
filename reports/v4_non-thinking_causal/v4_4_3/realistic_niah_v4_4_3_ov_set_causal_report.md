# Realistic NIAH V4.4.3-Set：OV vertical geometry 因果检验

## 1. 猜想

prompt running-index 与 answer-count 表示同一计数变量，但使用近似正交的 residual 载体方向；一个小型 attention-head set 通过 QK 定位和 OV 写回共同完成重编码。

可证伪预测包括：held-out OV mapping 保持同号；Z transport 超过等范数输出 control；donor-α 超过位置打乱 α；set-output span 内的 signed injection 产生正 dose response；answer-direction removal 比等范数正交 removal 更损害计数。

**本节结论：** 单头失败不能否定该猜想；合适的下一检验单位是预先冻结、并和同规模 matched set 比较的小型 head set。

## 2. 实验设计与定义

- discovery seeds 1234--1253；fit counts 1/3/5/7/9 选择 nested sets（Qwen K=1/2/3/4/6/8；Gemma K=1/2/3/4）；held-out counts 2/4/6/8/10 只评估。K=1 复用旧单头 run。
- screen seeds 1254--1258 做 alpha/Z/O 分阶段 patch；confirmation seeds 1259--1263 做 removal 与 set-reachable injection。
- set mapping：`m_S = sum_h M_OV^h u_prompt`，`r_S = cos(m_S, u_answer)`。
- transport：`T = (E[N|patch] - E[N|base]) / (donor_count - receiver_count)`。
- injection direction：`u_answer,S = normalize(P_col([W_O^h]_{h in S}) u_answer)`；斜率 `b = sum beta*DeltaE[N] / sum beta^2`。
- removal：error contrast 的正值与 correct-margin contrast 的负值共同支持方向性损伤。
- 每个 candidate set 配一个同层、同 K、成员不重叠且 OV 输出范数匹配的 control set。
- 因果统计以 seed 为单位做单侧 exact sign-flip；扩大 K 后，在每个模型、每类 causal family 内跨全部 layer×K 做 Benjamini-Hochberg 校正。

**本节结论：** set 选择、held-out 几何、screen 与 confirmation 完全分离；注入已限制到 set 输出子空间；主结论采用 BH q<=0.05，而不是从多个 K 中挑 raw p<=0.05。

## 3. 具体结果

### Qwen3-8B

| set | heads | fit map | held-out map | raw families (Z/I/R) | BH q (Z/I/R) | raw 2/3 | FDR 2/3 |
|---|---|---:|---:|---|---|---|---|
| L28K2 | 16,19 | 0.09863 | 0.05315 | True/True/True | 0.09375/0.04261/0.11719 | True | False |
| L28K3 | 16,19,31 | 0.1036 | 0.06344 | True/True/True | 0.09375/0.04261/0.11719 | True | False |
| L28K4 | 16,18,19,31 | 0.1089 | 0.0498 | True/True/True | 0.09375/0.04261/0.11719 | True | False |
| L28K6 | 1,3,16,18,19,31 | 0.1074 | 0.04729 | True/True/True | 0.09375/0.04261/0.11719 | True | False |
| L28K8 | 1,3,12,14,16,18,19,31 | 0.1048 | 0.05841 | True/True/False | 0.09375/0.04261/0.18750 | True | False |
| L29K2 | 3,29 | 0.03649 | 0.007636 | False/False/False | 0.97098/0.40179/1.00000 | False | False |
| L29K3 | 3,17,29 | 0.04286 | 0.007237 | False/False/False | 0.97098/1.00000/1.00000 | False | False |
| L29K4 | 0,3,17,29 | 0.04739 | 0.007274 | False/True/False | 0.97098/0.04261/1.00000 | False | False |
| L29K6 | 0,1,3,7,17,29 | 0.0515 | 0.005283 | False/False/False | 1.00000/0.07812/1.00000 | False | False |
| L29K8 | 0,1,3,7,17,19,24,29 | 0.05361 | -0.006336 | False/True/False | 0.97098/0.04261/1.00000 | False | False |
| L30K2 | 3,7 | 0.06506 | 0.02323 | False/False/False | 0.97098/0.36058/1.00000 | False | False |
| L30K3 | 3,6,7 | 0.06746 | 0.04305 | False/True/False | 0.97098/0.04261/1.00000 | False | False |
| L30K4 | 3,6,7,14 | 0.06705 | 0.02719 | False/True/False | 0.13393/0.04261/1.00000 | False | False |
| L30K6 | 1,3,6,7,14,30 | 0.0653 | 0.02259 | False/True/False | 0.13393/0.04261/1.00000 | False | False |
| L30K8 | 1,3,6,7,10,14,19,30 | 0.06337 | 0.012 | False/True/False | 0.17578/0.04261/1.00000 | False | False |

**本节结论：** Qwen3-8B 的未校正 2-of-3 support=True；BH-FDR 2-of-3 support=False。

### Gemma4-E4B

| set | heads | fit map | held-out map | raw families (Z/I/R) | BH q (Z/I/R) | raw 2/3 | FDR 2/3 |
|---|---|---:|---:|---|---|---|---|
| L36K2 | 0,1 | 0.06001 | 0.03312 | False/False/False | 1.00000/1.00000/1.00000 | False | False |
| L36K3 | 0,1,3 | 0.06328 | 0.03058 | False/False/False | 1.00000/1.00000/1.00000 | False | False |
| L36K4 | 0,1,3,6 | 0.05867 | 0.01922 | False/False/False | 1.00000/1.00000/1.00000 | False | False |
| L37K2 | 1,2 | 0.05844 | 0.0511 | True/False/False | 0.28125/0.14062/1.00000 | False | False |
| L37K3 | 1,2,4 | 0.05692 | 0.03625 | False/True/False | 1.00000/0.05625/1.00000 | False | False |
| L37K4 | 1,2,4,5 | 0.05356 | 0.008936 | False/True/False | 1.00000/0.05625/1.00000 | False | False |
| L38K2 | 1,2 | 0.03982 | 0.03075 | False/True/False | 1.00000/0.05625/1.00000 | False | False |
| L38K3 | 1,2,5 | 0.04047 | 0.04226 | False/True/False | 1.00000/0.05625/1.00000 | False | False |
| L38K4 | 0,1,2,5 | 0.04164 | 0.03745 | False/True/False | 0.28125/0.05625/1.00000 | False | False |

**本节结论：** Gemma4-E4B 的未校正 2-of-3 support=False；BH-FDR 2-of-3 support=False。

## 4. 综合分析

至少一个 nested head set 通过未校正的 2-of-3 exact-p 筛选，但在模型内、证据族内跨全部 layer×K 做 BH 校正后没有集合保留 2-of-3 支持；因此目前只能把这些 set 视为后续确认候选，不能宣称已确认小型 circuit。

set sufficiency、set specificity 与 member irreducibility 是三个不同命题。本实验用 candidate-vs-matched 检验前两者，但没有做 leave-one-head-out，因此不检验成员不可替代性。另有四个边界：预选层来自既有 geometry；每个 causal split 只有 5 seeds；只搜索同层 sets；更大的 K 可能只增加可干预子空间维数，因此必须结合 matched specificity 与边际增益解释。

**本节结论：** 只有 held-out mapping 与至少两类 BH-FDR matched-set 因果证据汇合，才支持小型 set 的因果充分性；raw 显著或单一 family 只作为确认线索。
