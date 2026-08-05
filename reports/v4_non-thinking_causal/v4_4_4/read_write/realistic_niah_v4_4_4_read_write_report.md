# Realistic NIAH V4.4.4 补充：state 读取分解与 OV 写入报告

## 总结论

读取模式判定为 **mixed**；下游写入传播支持为 **True**；联合 read→write 路径支持为 **True**。

本实验复用父 V4.4.4 的 evaluation seeds，属于冻结候选后的机制扩展，不是全新 seed 的独立复现。*本段结论：当前统计量可检验机制分解，但最终发表级确认仍需 1294–1313 新 seed 复现。*

## 1. 猜想与可证伪预测

V4.4.4 已支持 L28 H16/H19 是自然 OV transporter，但没有说明它们的 pre-O state 主要来自 V-content 改变还是 alpha-routing 改变。本实验预测：真实 donor-Z transport 应能被二者分解，且真正的读取分量既应推动 donor count，也应被 frozen natural OV-axis block 特异性削弱。

```text
Δz_value = 1/2[(z_RD-z_RR)+(z_DD-z_DR)]
Δz_route = 1/2[(z_DR-z_RR)+(z_DD-z_RD)]
Δz_full  = Δz_value + Δz_route
```

其中 RR/DD 使用模型实际 fused pre-O 端点，RD/DR 使用 crossed alpha-V 计算。*本节结论：这是对同一次真实 donor-Z movement 的因果分账，不把 QK head 与 OV head 预设为同一组。*

## 2. 实验设定

- 模型/候选：Qwen3-8B, L28, heads=[16, 19]。
- axis discovery seeds：1264–1273；evaluation seeds：1274–1293。
- counts=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]；directed donor pairs=[(1, 6), (6, 1), (3, 8), (8, 3), (5, 10), (10, 5)]；write counts=[2, 5, 8]。
- post-block trace layers=[28, 29, 30, 31, 32, 33, 34, 35]；pre-O intervention beta=1.0。
- 主分析使用全部 paired trials；baseline-correct/wrong 仅在 discovery 固定的 count axes 上分层，不重新拟合 PCA/count axis。

*本节结论：方向估计、机制评价和 outcome 分层之间没有按结果重新选轴。*

## 3. 读取分解结果

| metric | mean | 95% CI | p |
|---|---:|---:|---:|
| read_routing_ov_mediation_specificity | 0.00992046 | [0.00765738, 0.0121909] | 9.53674e-07 |
| read_value_ov_mediation_specificity | 0.00515879 | [0.00307946, 0.00735632] | 5.62668e-05 |
| read_value_minus_routing_transport | 0.000769446 | [-0.0105333, 0.0128811] | 0.451241 |
| read_full_behavior_transport | 0.113967 | [0.0922412, 0.136131] | 9.53674e-07 |
| read_routing_behavior_transport | 0.0516721 | [0.0385528, 0.0653322] | 9.53674e-07 |
| read_value_behavior_transport | 0.0524416 | [0.0430125, 0.0623721] | 9.53674e-07 |

value family p=5.62668e-05；routing family p=9.53674e-07；value-minus-routing p=0.451241。

*本节结论：按冻结判据，读取模式为 **mixed**。只有 component transport 与 natural-OV mediation 同时成立才计为自然读取证据。*

## 4. OV 写入与层间传播

在 L28 真实 pre-O 边界施加 ±β natural z-step，并与同一 H16/H19 W_O span 内、等 post-O 范数的正交方向比较。纵向 estimand 为

```text
coefficient_l = <[h_l(+β)-h_l(-β)]/(2β), s_l> / ||s_l||²
```

其中 s_l 是 discovery seeds 上拟合的该层自然 answer-query count step。

| layer | natural slope | orthogonal slope | specificity | Holm p |
|---:|---:|---:|---:|---:|
| L28 | 0.0425218 | 0.0220175 | 0.0205043 | 7.62939e-06 |
| L29 | 0.0569861 | 0.0181821 | 0.038804 | 7.62939e-06 |
| L30 | 0.0508687 | 0.016046 | 0.0348227 | 7.62939e-06 |
| L31 | 0.044937 | 0.0177964 | 0.0271406 | 7.62939e-06 |
| L32 | 0.0427494 | 0.0213299 | 0.0214194 | 7.62939e-06 |
| L33 | 0.0451805 | 0.0220734 | 0.0231071 | 7.62939e-06 |
| L34 | 0.0368722 | 0.0235077 | 0.0133645 | 7.62939e-06 |
| L35 | 0.0397706 | 0.0241712 | 0.0155994 | 2.28882e-05 |

答案分布上的 natural-minus-orthogonal specificity mean=0.0685172, p=9.53674e-07；最终 L35 residual specificity=0.0155994, Holm p=2.28882e-05。

*本节结论：下游写入传播支持为 **True**。只有答案分布与最终层固定自然 count axis 同时优于正交控制，才判定写入存活。*

## 5. 正确/错误基线与证据边界

正确/错误分层只用于敏感性分析；由于所有轴均在分层前冻结，答错样本不会通过重新拟合改变 geometry。若某一层样本过少，其区间应视为描述性而不是主检验。

审计通过：**True**，共 10 项。未持久化 full hidden state、full V tensor 或 raw attention map。
eager/cache candidate-logit 差异仅作为数值诊断记录；硬门槛是 all-key eager alpha-V 对真实 pre-O endpoint 的相对 L2 重建误差。

即使联合检验为正，也只定位 terminal state-component → H16/H19 pre-O Z → natural OV axis → downstream count state → count distribution。它不定位构造 V-state 的更早 heads/MLP；该问题需要按本轮 read-mode 结果选择 upstream residual/V 或 Q/K path patch。

*本节结论：本报告可以判断模型在 terminal attention 中读了哪一类state 并如何写入，但不能把上游 state builder 一并宣称为已识别。*
