# Qwen query-last mode-specific empirical-law report v2

Open `report.html` for the self-contained narrative.

## Scope

- Qwen3-1.7B, Qwen3-8B, and Qwen3-32B only
- query order fixed to `query_last`
- direct, enumeration, and native thinking fitted independently
- 1,350 requests total; 150 requests per model x mode stratum
- 48 frozen candidates per target and stratum
- nested leave-one-seed-out evaluation after candidate selection
- best-predictive and one-standard-error pipelines reported side by side
- no request deletion or post-hoc parser changes

Each model x mode x target is allowed to select a different bounded response
surface in T and N. Exact correctness includes parse/format/truncation failures
as failures. Signed bias and absolute error remain conditional on parsed numeric
outputs.

## Rebuild

Use the same environment as the eight-model report (NumPy, pandas, SciPy,
Matplotlib):

```powershell
python scripts/build_qwen_mode_specific_report.py `
  --source-report-root "C:\Users\HP\Desktop\Research\UWM Yiqiao Zhong\CoT for Counting\Realistic_CoT_NiaH_Count\reports\niah_eight_model_report_20260724_model_bias" `
  --output-dir <new-empty-output-directory>
```

The builder refuses to write into a non-empty directory.

## Audit

```powershell
python scripts/audit_qwen_mode_specific_report.py --report-root .
```

The audit recomputes the 1,350-row filter and headline accuracy values, verifies
the complete 3 x 10 x 5 grid in every stratum and dual nested-prediction
coverage, checks tables/images/links/MathML, and validates `SHA256SUMS.tsv`.
