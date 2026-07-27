# Realistic NiaH V2 analysis reports

This directory contains two complementary reports:

- `01_prompt_accuracy_report.html`: prompt/mode accuracy and observed failure mechanisms.
- `02_bias_law_report.html`: model-by-mode signed-bias laws and cross-model commonality.

The current-prompt composite contains exactly 29 cells × 500 requests. V2.1
enumeration replacements supersede the old enumeration rows. The strict
Gemma4-12B Direct appendix supplies the current Direct cell for numerical-bias
analysis; its original V2 Direct run remains a prompt-failure comparator only.

Rebuild:

```powershell
python scripts/build_v2_reports.py --input tables/request_level_compact.csv --output .
```

All formulas are exploratory empirical response surfaces. See the reports for
the exact success, bias, cross-validation, and extrapolation definitions.
