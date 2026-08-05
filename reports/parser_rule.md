# 当前 native-thinking 城市列表 parser 规则

本文档描述当前 V4.4 native-thinking 截断实验实际使用的 parser。它解析的是模型的 native-thinking 输出，不是 passage；而且它是使用 gold city 的实验性、oracle-assisted parser，不是可部署的通用 parser。

## 1. 解析范围

parser 只检查模型输出中的 reasoning span：

- Qwen：`<think>` 与最后一个 `</think>` 之间。
- Gemma：`<|channel>thought\n` 与第一个 `<channel|>` 之间。
- 如果 opening tag 缺失，则从输出开头开始。
- 如果 closing tag 缺失，则 reasoning span 延伸到输出末尾。

最终 `Total: n` 答案区不作为城市列表的一部分。

## 2. 合法城市项

一行必须同时满足以下三个条件，才会被接受为城市项：

1. 行首是受支持的 marker；
2. 这一行恰好匹配该样本的一个 gold city；
3. 该行必须有结束换行符。

parser 使用该样本的 gold city 集合做匹配，只要求城市名匹配；不要求 thinking 中写出 score，也不要求写出的 score 与 gold score 一致。

合法例子：

```text
1. Chicago received a score of 72.
* Baku — 98
Third: Taipei
```

不合法例子：

```text
1. Chicago and Baku
```

这一行包含两个 gold city，因而不是“恰好一个”。

```text
- I found another record.
```

这一行没有 gold city。

## 3. 支持的 marker

### 3.1 数字编号

```text
1. Chicago
2) Baku
3. Taipei
```

要求：

- 必须从 `1` 开始；
- 必须严格递增；
- 支持 `n.` 和 `n)`；
- marker 后必须有空格和非空内容。

### 3.2 英文序数

```text
First, Chicago
Second: Baku
Third. Taipei
```

要求：

- 大小写不敏感；
- 支持 First 到 Twentieth；
- 也支持 Firstly、Secondly 等 `-ly` 形式；
- 允许外围 Markdown，例如 `**First:** Chicago`；
- 允许 `.`, `)`, `:`, `,`, `-` 等 marker 结尾；
- 必须从 First 开始并严格递增；
- 当 gold `N>1` 时，单独一个 First 不足以证明这是目标 ordinal list，至少需要两个接受项。

### 3.3 Bullet

```text
- Chicago
* Baku
• Taipei
```

要求与说明：

- `-`、`*`、`•` 都接受；
- bullet 不要求编号；
- 同一个列表中允许 bullet 表面符号变化，因为三者属于同一个 bullet marker family；
- 普通 Markdown 强调中的星号不会自动算 bullet，必须是行首 `*` 后有空格和内容，例如 `* Chicago`。

## 4. 跨段拼接和 bridge line

两个接受项之间允许出现：

- 空行；
- 普通解释 prose；
- 小标题；
- 不构成合法“marker + 单一 gold city”的桥接行。

例如：

```text
1. Chicago

I will keep scanning the passage.

Additional records:
2. Baku
3. Taipei
```

这会被拼成同一个 numbered candidate，并记录 bridge line 数。该规则用于保留模型在列表中间插入解释段落后继续列城市的情况。

## 5. 列表终止证据

仅看到若干合法城市项还不足以 parse 成功；必须有后续证据证明模型已经离开该列表。支持的终止证据包括：

- 出现模型官方 thinking-close tag；
- 最后一个城市项后出现非空 trailing prose；
- 数字编号或英文序数重新从 1/First 开始；
- 数字编号或英文序数不再按预期递增；
- marker family 改变；
- 后续出现一个不含“恰好一个 gold city”的同类 marker；
- bullet 列表后出现 non-gold bullet。

截断边界永远放在“最后一个已接受城市项的结束换行符之后”。用来证明终止的后续 prose、坏 marker 或 non-gold bullet 不会被保留。

例一：

```text
1. Chicago
2. Baku
Therefore there are two records.
```

截断位置是 `2. Baku\n` 之后；`Therefore...` 不保留。

例二：

```text
* Chicago
* Baku
* Total records: 2
```

前两个项被接受，第三个 non-gold bullet 证明列表终止；截断仍在 `* Baku\n` 之后。

两个保守限制：

- 当 gold `N>1` 时，一个孤立 bullet 后立即接 non-gold bullet，不足以证明这是目标列表；
- 当 gold `N>1` 时，一个孤立 First 项不算成功 ordinal parse。

如果城市项是输出的最后一行、没有结束换行、没有 closing delimiter、也没有 trailing prose，则不能构造可靠截断边界，记为 no-hit。

## 6. Parse 成功不要求完整覆盖

当前版本不要求列表覆盖全部 gold city，也不禁止重复。这是它与早期 full-coverage parser 的关键区别。

遗漏仍可 parse：

```text
1. Chicago
2. Taipei
</think>
```

假设 gold 是 Chicago、Baku、Taipei，这仍是 parse hit，但 trace 分类为“有遗漏、无重复”。

重复仍可 parse：

```text
1. Chicago
2. Baku
3. Baku
</think>
```

它会根据是否已经覆盖全部 gold city，分类为“完整覆盖但有重复”或“有遗漏且有重复”。

因此必须区分：

```text
parse hit ≠ trace 正确 ≠ 最终答案正确
```

parser 返回第一个具有可靠终止边界的合法 candidate，而不是事后挑选 coverage 最大或最接近 gold 的 candidate。

## 7. Trace 分类

对 parser 接受的城市序列做 multiset 比较，类别互斥。

### 7.1 `one_to_one`

接受序列与 gold city multiset 完全相同，每个 gold city 恰好一次，没有遗漏或重复。

`one_to_one` 再分为：

- `forward`：与 gold catalog 顺序相同；
- `reverse`：与 gold catalog 完全反序；
- `other_permutation`：其他排列。

正序、反序和其他排列都属于 trace 正确；trace 正确不要求采用 passage 中的出现顺序。

### 7.2 `full_coverage_with_duplicates`

所有 gold city 都至少出现一次，但至少有一个重复项。

### 7.3 `partial_unique`

遗漏了一个或多个 gold city，但接受项之间没有重复。

### 7.4 `partial_with_duplicates`

既有遗漏，又有重复。

### 7.5 `no_parser_hit`

没有找到带可靠终止边界的 candidate。

## 8. 哪些 parse hit 会执行 cutoff

当前行为实验中，任何 parse hit 都可以接受 cutoff，包括：

- `one_to_one`；
- 完整覆盖但有重复；
- 有遗漏、无重复；
- 有遗漏且有重复。

这样可以分别测量 partial/full/duplicate trace 截断后的行为，而不是预先丢弃非一一对应 trace。

cutoff 流程：

1. 保留原始输出直到最后一个接受城市项的结束换行符；
2. 将字符边界与原始 output token IDs 做 exact alignment；
3. 如果边界穿过一个 token，只从第一个跨界 token 开始重新 tokenize；更早的 token prefix 必须完全不变；
4. 追加该模型官方 thinking-close token；
5. 使用原注册 seed、原温度、原 top-p、原 top-k 继续生成；
6. `已保留 prefix + close + continuation` 总长度不得超过原来的 4096-token 输出预算。

对齐失败、没有剩余预算或没有 parse hit 的样本不执行干预；在 all-sample policy 指标中，这些样本回退到原始 baseline 输出。

## 9. 正式 geometry 的筛选条件

主 geometry 只使用：

```text
parser hit
AND trace_one_to_one
AND cutoff 实际完成
AND cutoff 最终 exact-count 正确
```

即先分析“trace 与 gold 一一对应，且截断后答案正确”的样本。

strict sensitivity cohort 额外要求：

```text
原始未截断 baseline 也正确
```

running index 只在 one-to-one cohort 中定义为列表的第 `k` 项，因此严格满足：

```text
1 ≤ k ≤ N ≤ 10
```

重复或遗漏 trace 即使出现 11、12、14 个 visit，也不会被称为 running index，更不会混入主 geometry。

## 10. 当前明确不接受的常见格式

Qwen 常见的单行 prose 枚举：

```text
First, I found Chicago. Then I found Baku. Finally I found Taipei.
```

整段在同一行，没有逐行合法 marker，因此 no-hit。

无 marker 的逐行 prose：

```text
Chicago was one record.
Baku was another.
Taipei was the third.
```

城市虽然都被提到，但没有支持的行首 marker，也是 no-hit。

preflight 中，Qwen 的 209 个 no-hit 里有 194 个属于无 marker 的 prose 或行内枚举；这些样本中的绝大多数最终计数正确。因此 parser hit rate 不能解释为模型计数准确率。

no-hit 格式诊断只用于解释 parser miss，不会反过来放宽正式 parser，也不会把这些样本加入 cutoff 或主 geometry。

