# Realistic NiaH eight-model HTML report

Open `report.html` in a browser. The report is offline; all images use relative
paths under `assets/`.

## Canonical reviewed version

This directory is the canonical report location. It includes the audited prompt
formats, per-model signed-bias analysis, and a formula/figure readability review.
All formulas use `T` for canonical passage length and `N` for true needle count.
Primary exact accuracy uses all 6,300 registered requests; absolute error and
signed bias remain conditional on the 5,385 parsed numeric outputs.

The readability pass regenerated Figures 7, 8, and 14 from the preserved CSV
tables and rewrote every figure caption to define its sample, axes, uncertainty,
and interpretation. Reapply the report-only pass in place with:

```powershell
python scripts/refine_report_readability.py --report-root .
```

## Rebuild

Use a Python environment with NumPy, pandas, SciPy and Matplotlib:

```powershell
python scripts/build_report.py `
  --run-root <extracted>\runs\realistic_niah_v1\six_models_formal_20260723T194300Z `
  --output-dir <new-empty-output-directory>
```

The builder refuses to overwrite an existing output directory. It validates
6,300 unique request IDs, expected per-model row counts, the complete N/T/seed
grid, and stimuli SHA256 `374dc935bf4c1403f705bb8b95ce686e5063647c83c609501e6f668e2331a5f1` before generating any result.

Primary accuracy retains parse failure, wrong format and truncation as failures.
Absolute error and signed bias are explicitly conditional on parsed outputs.

## Independent per-model signed-bias laws

This report version adds a model-by-model signed-bias analysis. It compares a
fixed, interpretable log-length/log-needle law with a bounded coordinate search,
uses four grouped validation schemes plus nested leave-one-seed-out selection,
and reports 500-replicate stimulus-cluster bootstrap intervals.

Reproduce only this addendum with:

```powershell
python scripts/build_model_bias_addendum.py `
  --base-report <verified-base-report-directory> `
  --output-dir <new-empty-output-directory>
```

## Audited prompt formats

This report version adds every frozen model x prompt-mode x query-order format.
The HTML shows both the single-user-message layer and the tokenizer-rendered
layer. Passage bodies are replaced by `[PASSAGE OMITTED]`; full rendered-prompt
SHA256 values are retained. The complete 42-row matrix is available in
`tables/model_prompt_format_examples.csv` and
`prompt_formats/model_prompt_formats.json`.

Reproduce this addendum with:

```powershell
python scripts/build_prompt_format_addendum.py `
  --base-report <verified-base-report-directory> `
  --output-dir <new-empty-output-directory> `
  --repo-root <Realistic_CoT_NiaH_Count-repository>
```

<!-- MODEL_BIAS_NOISE_V1 -->
## Model-wise bias/noise analysis

The canonical report includes a model-wise signed-bias and tail-noise section,
plus a Qwen shared-versus-model-specific slope comparison. Rebuild it with:

```powershell
python scripts/add_model_bias_noise_section.py --report-root "<report directory>"
```

Bias tables are conditional on parsed numeric outputs; primary exact accuracy
continues to count parse failure, format failure, and truncation as failures.

<!-- CORE_RESULTS_REPORT_V3 -->
## Core-results narrative rewrite (v3)

The canonical `report.html` is organized in the scientific order used for
interpretation: frozen prompt formats and concrete experimental settings,
query-first/query-last exact accuracy, within-model prompt/reasoning modes,
rule-based anomaly detection and mechanism diagnosis, low-accuracy failure
mechanisms, and finally the retained empirical-law results. Every major
scientific section explicitly states its calculation method and the conclusion
currently supported by the data.

Rebuild this report-only layer with:

```powershell
python scripts/rewrite_core_results_report.py --report-root "<canonical report directory>"
```

The script reads the preserved 6,300-row request table plus the local export
archive (read-only, for enumeration score-summing diagnostics), validates the
frozen model/mode registry, writes auditable summary CSVs and figures, and
refreshes the root SHA256 manifest. It does not modify request-level data,
prompt snapshots, raw outputs, or fitted parameters.

Audit the generated report with:

```powershell
python scripts/audit_core_results_report.py --report-root "<canonical report directory>"
```
