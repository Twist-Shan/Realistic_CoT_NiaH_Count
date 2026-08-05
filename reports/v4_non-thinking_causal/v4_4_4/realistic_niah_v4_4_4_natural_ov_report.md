# Realistic NIAH V4.4.4：natural OV transporter 补充因果实验

## 结论

预冻结的 Qwen L28 H16/H19 set 在四类必要证据上全部通过：结果支持其作为自然 OV transporter。

全局 intersection-union p=0.0045414，阈值 α=0.05。这里的全局 p 是四个预注册证据族 p 值的最大值；只要任一必要证据失败，完整机制主张就不成立。**本段结论：完整 natural-OV 主张为 True。**

## 猜想与可证伪预测

QK heads 与 OV heads 不要求相同：较早的 QK 集合可先形成 relay/count state，L28 的 OV set 再从进入该层的表示中读取并写回 answer-relevant count direction。由于同层 heads 并行，若两组是串联关系，QK 阶段通常必须早于 L28，或者 relay 已在 L28 输入前形成。**本段结论：本实验只检验下游 OV transporter，不把 QK 定位失败等同于 OV 失败。**

设每个 query head 的自然 pre-O 一单位 count step 为 d_z,h=W_V[g(h)]s_P，set 输出方向为 m_S=Σ_h W_O^h d_z,h。count-neutral 中心 z0 是独立 center seeds 上逐 head OLS 在 count=0 的截距；自然 carrier 系数为 a_S(z)=<W_O^S(z_S-z0,S),m_hat_S>/||m_S||。Injection 在真实 pre-O z slice 加 βd_z；removal 在 z-space 删除自然输出沿 m_S 的分量，其控制也位于同一 W_O^S span、post-O 范数相等且与 m_S 正交。**本段结论：所有输出变化都经过 selected heads 自己的 W_O；sufficiency、necessity 与 mediation 被分开检验。**

## 实验设定

模型固定为 Qwen3-8B、层固定为 L28、主 set 固定为 H16/H19。方向估计使用 seeds 1234–1253；count-neutral z 中心与 matched controls 使用全新 seeds 1264–1273；因果确认使用不重叠的全新 seeds 1274–1293。每个 seed 含 count 1–10；injection/removal 用 count 2/5/8；mediation 用 1→6、3→8、5→10。**本段结论：方向、控制选择与因果确认三者没有 seed 泄漏。**

主 set 与四个同 GQA relative-position 的 K=2 controls 比较，controls 在未查看 causal outcome 时按 natural-step norm、自然轴对 answer 的 cosine、W_O-span reachability 与 baseline set-output norm 匹配。K=3/4/6/8 是旧 discovery 已冻结的 nested sets，只进行 Holm 校正后的二级稳健性分析，不改变 K=2 主结论。**本段结论：不存在按本轮显著性挑层、挑 heads 或挑 K。**

## 主结果

| 证据族 | IUT p | 通过 α=.05 | 组成检验 |
|---|---:|---|---|
| natural_signal | 9.53674e-07 | True | natural_carrier_count_slope p=9.53674e-07, mean=0.2174; natural_carrier_count_slope__candidate_minus_control_mean p=9.53674e-07, mean=0.2154 |
| pre_o_injection | 9.53674e-07 | True | injection_dose_slope p=9.53674e-07, mean=0.06401; injection_dose_slope__candidate_minus_control_mean p=9.53674e-07, mean=0.06088 |
| centered_removal | 0.00141335 | True | removal_error_axis_minus_control p=0.000108719, mean=0.07322; removal_error_axis_minus_control__candidate_minus_control_mean p=9.44138e-05, mean=0.07586; removal_margin_axis_minus_control p=0.00104523, mean=-0.2646; removal_margin_axis_minus_control__candidate_minus_control_mean p=0.00141335, mean=-0.2677 |
| path_mediation | 0.0045414 | True | donor_patch_transport p=2.86102e-06, mean=0.07452; donor_patch_transport__candidate_minus_control_mean p=4.76837e-06, mean=0.07394; mediation_control_minus_axis_block p=0.00447845, mean=0.01359; mediation_control_minus_axis_block__candidate_minus_control_mean p=0.0045414, mean=0.01358 |

每个证据族采用 conjunction：候选 set 自身效应与 candidate-minus-control-mean 特异性都必须沿预注册方向显著；removal 还要求 error 增加与 correct-margin 降低同时成立；mediation 要求 donor patch 能 transport 且 orthogonal-control 相对 natural-axis block 保留更多 transport。**本段结论：表中的 family p 是最弱组成检验，而不是挑最小 p。**

自然 carrier/count slope=0.2174（95% CI 0.1816–0.2519，p=9.53674e-07）。真实 pre-O injection dose slope=0.0640 expected-count/β（95% CI 0.0464–0.0836，p=9.53674e-07）。**本段结论：自然 forward 中存在 count carrier，且真实 V→z→W_O channel 具有带符号充分性。**

相对同 span、等 post-O 范数的正交控制，natural-axis removal 使 absolute error 多增加 0.0732（95% CI 0.0439–0.1018，p=0.000108719），使 correct-count margin 多下降 0.2646（95% CI -0.4000–-0.1167，p=0.00104523）。**本段结论：centered z-space removal 支持该自然 channel 对计数是必要的。**

Donor-z patch 的 normalized transport=0.0745（95% CI 0.0513–0.0987，p=2.86102e-06）；相对正交控制，自然轴阻断额外消除 0.0136，约占 donor transport 的 18.2%（p=0.00447845）。**本段结论：同一自然 OV 轴部分介导 donor effect，而不是只对任意 perturbation 敏感。**

## 基线与数据审计

无干预候选答案准确率=0.460，expected-count MAE=1.062，确认样本=200。审计 all_checks_pass=True；observed rows={'dataset_rows': 300, 'confirmation_seed_shards': 20, 'natural_rows': 2200, 'directed_rows': 4620, 'mediation_rows': 900}。**本段结论：因果结果是在模型原始计数行为与完整 seed grid 上计算，且产物计数通过审计。**

## Nested-K 二级结果

| K | heads | natural Holm p | injection Holm p | removal Holm p |
|---:|---|---:|---:|---:|
| 2 | 16,19 | 4.76837e-06 | 4.76837e-06 | 0.00522614 |
| 3 | 16,19,31 | 4.76837e-06 | 4.76837e-06 | 0.0972204 |
| 4 | 16,18,19,31 | 4.76837e-06 | 4.76837e-06 | 0.0167503 |
| 6 | 1,3,16,18,19,31 | 4.76837e-06 | 4.76837e-06 | 0.0972204 |
| 8 | 1,3,12,14,16,18,19,31 | 4.76837e-06 | 4.76837e-06 | 0.0972204 |

Nested-K 中 natural signal 与 injection 在 K=2/3/4/6/8 均经 Holm 校正通过；centered removal 仅 K=2 与 K=4 通过，K=3/6/8 未通过。更大 K 同时扩大可干预子空间，不能在没有同 K matched controls 的情况下单独证明更大的 circuit 更真实。**本段结论：没有‘增加 K 就更显著’的模式；最稳健的主结论仍来自 K=2 matched-set 检验。**

## H16/H19 成员结构（二级）

| endpoint | joint − H16 − H19 | two-sided p |
|---|---:|---:|
| removal_error_axis_minus_control | -0.0032561 | 0.778831 |
| removal_margin_axis_minus_control | -0.085416 | 0.222876 |
| natural_carrier_count_slope | -0.20188 | 1.90735e-06 |
| injection_dose_slope | 7.1843e-05 | 0.971947 |

Injection 的联合项近似严格可加；removal-error 与 removal-margin 的交互也未显著。H19 单头的 necessity 强于 H16，联合 set 的 margin 损伤更大，但额外超加性协同尚未确认。natural carrier 系数会随 set 自身 m_S 重新归一化，因此其显著负交互不能解释成两头相互抵消。**本段结论：当前数据支持 H16/H19 近加性贡献，不支持宣称超加性协同。**

## 边界与下一步

即使四类 OV 证据汇合，本实验仍没有定位上游 source-position QK heads；完整链路还需 donor/source patch 先经 S_QK 产生 shift，再阻断 S_OV 验证 shift 消失。反之，如果 injection 成立但 centered removal 或 mediation 不成立，只能解释为该 W_V/W_O channel 可 steering，而不是模型自然使用它。**本段结论：本报告最多确认 L28 OV transporter，不声称已经证明完整 QK→relay→OV circuit。**
