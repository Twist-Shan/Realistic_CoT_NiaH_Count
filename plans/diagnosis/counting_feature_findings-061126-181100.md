# Counting Feature Analysis Findings

I've run the experiments, and some of the figures are unexpected. I want you to check your code based on my suggestions and the summary from Codex IDE below. Try to understand the arguments. You may agree or disagree; not all analyses are correct.

**General instructions.**
- Double check your code and statistical analysis.
- Propose your fixes before you ask to code.
- Ask to correct obvious typos (with minimal changes) in this document before you code.

## My diagnosis

My take is that there are outlier tokens with large hidden state norms, which distorts the principal direction calculation severely. To fix this issue, you can remove the outlier tokens from the PCA calculation. 
- For each example, calculate the raw hidden state norms, identify the token positions whose hidden state norms are larger than 5 times of the median
- When calculating PCA, don't include those outlier tokens.


## Codex IDE summary

This note summarizes the main unexpected behaviors found in the counting-feature artifacts for:

`run_20260611_224007_Qwen3-8B_match_count_easier_1000_needles_100_null_400`

## 1. The 2D plots mix raw and standardized feature spaces

The probes were trained with `standardize=True`, so their learned coefficients live in standardized hidden-state coordinates. However, `plot_probe_2d_projection` projects raw hidden states directly onto the learned probe vector:

```python
coords = np.column_stack([x @ u, x @ v])
```

This means the plotted "probe direction" is not the same coordinate system used by the trained probe. As a result, the 2D scatterplots should not be interpreted as faithful visualizations of the learned counting direction.

Relevant code:

- `NIAH/src/counting/feature_analysis.py`, `plot_probe_2d_projection`

## 2. The classification decision boundaries are especially unreliable

For classification plots, the function creates a 2D grid in the plotted subspace:

```python
basis_points = xx.reshape(-1, 1) * u[None, :] + yy.reshape(-1, 1) * v[None, :]
pred = predict_classification(classifier, basis_points).reshape(xx.shape)
```

Those `basis_points` are artificial raw hidden-state vectors centered around zero in raw feature space. But `predict_classification` then applies the classifier's saved standardization statistics, which were learned from real hidden states. This sends the decision-boundary grid far away from the actual data distribution.

This likely explains the abnormal classification plots that look like a large diagonal black line rather than meaningful class boundaries.

## 3. The classification projection direction is arbitrary

The notebook calls:

```python
plot_probe_2d_projection(clf.coef[0], ...)
```

For a multi-class classifier, `clf.coef[0]` is just the class-0 logit vector. It is not a natural "counting direction" for separating counts `0`, `1`, and `2`.

Better alternatives would be:

- Use the ridge probe direction for the x-axis.
- Use `clf.coef[class_2] - clf.coef[class_0]` as an approximate low-to-high count direction.
- Use PCA/SVD on the classifier coefficient matrix to find the dominant classification direction.

## 4. The ridge 2D plot is visually distorted by extreme projected points

The ridge 2D test plot is dominated by a few extreme projected points, which set very large axis limits and visually collapse most points. This makes the plot look almost empty or like only one or two points are present.

This may come from the same raw-vs-standardized coordinate mismatch, but the plotting code also lacks robust axis handling. For diagnostic plots, it would help to clip axes to robust percentiles, for example the 1st and 99th percentiles, while optionally reporting outliers separately.

The ridge scalar metrics are still internally plausible, but the 2D scatterplot should not be trusted in its current form.

## 5. The experiment is easier than the run name suggests

Although the run name includes `needles_100_null_400`, the actual generated dataset has:

- 20 total examples.
- 19 successful examples used for feature analysis.
- 3 needle records per example.
- Only 2 relevant inserted needles per example: `N1` and `N3`.
- The middle needle `N2` is not inserted.
- Every example has gold count `2`.

The target count is therefore always a fixed step pattern:

```text
0 -> 1 -> 2
```

The needle positions are also highly consistent:

- First matching needle starts around token position 147-148.
- Second matching needle starts around token position 475-481.

This means a probe can achieve good performance by learning position/template structure, rather than necessarily discovering a robust internal counting feature.

## Bottom Line

The 2D plots, especially the classification plots, are likely affected by plotting and coordinate-system bugs. The scalar probe metrics may still be useful, but the current run is also heavily confounded by fixed final counts and nearly fixed needle positions.

Recommended next steps:

- Redo `plot_probe_2d_projection` in standardized feature space.
- Use a meaningful multi-class classification direction instead of `clf.coef[0]`.
- Add robust axis limits for 2D plots.
- Rerun with varied final counts and more varied insertion positions.

