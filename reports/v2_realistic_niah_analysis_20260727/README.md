# Realistic NiaH V2 analysis reports

This directory contains two complementary reports:

- `01_prompt_accuracy_report.html`: prompt/mode accuracy and observed failure mechanisms, with long tables and figures collapsed by default.
- `02_bias_law_report.html`: model-by-mode signed-bias laws, natural-log candidates, the fixed `bias/N ~ ln(N/Lk)` diagnostic, expanded goodness-of-fit metrics, and Direct/Enumerate/Native common functional forms.

The current-prompt composite contains exactly 29 cells × 500 requests. V2.1
enumeration replacements supersede the old enumeration rows. The strict
Gemma4-12B Direct appendix supplies the current Direct cell for numerical-bias
analysis; its original V2 Direct run remains a prompt-failure comparator only.

Rebuild:

```powershell
python scripts/build_v2_reports.py --input tables/request_level_compact.csv --failure-audit-dir tables/prompt --output .
```

All formulas are exploratory empirical response surfaces. The candidate tables
include held-out R²/RMSE/MAE/median-AE/NRMSE, leave-N/L-out metrics, information
criteria, bootstrap coefficient stability, and residual diagnostics. See the
reports for the exact success, bias, validation, and comparison definitions.
