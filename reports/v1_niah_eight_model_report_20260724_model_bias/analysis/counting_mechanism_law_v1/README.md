# Counting-mechanism empirical law v1

This directory contains the bounded, grouped-validation search requested for
nonthinking/direct, enumeration, and native-thinking/CoT modes.

- Standalone report: `report.html`
- Frozen search plan: `analysis_plan.md`
- Reproduction script: `scripts/run_counting_mechanism_law.py`
- Complete candidate and fold results: `tables/`
- Figures: `figures/`

Source SHA256: `329694728ee69f330f447c48892f194060808e7033a2f49129dff747ac71f150`

Run:

```powershell
& "<python-with-numpy-pandas-scipy-matplotlib>" `
  "scripts/run_counting_mechanism_law.py" `
  --report-root "C:\Users\HP\Desktop\Research\UWM Yiqiao Zhong\CoT for Counting\Realistic_CoT_NiaH_Count\reports\niah_eight_model_report_20260724_model_bias" `
  --output "C:\Users\HP\Desktop\Research\UWM Yiqiao Zhong\CoT for Counting\Realistic_CoT_NiaH_Count\reports\niah_eight_model_report_20260724_model_bias\analysis\counting_mechanism_law_v1" `
  --inject-main-report
```

<!-- COUNTING_MATH_ENV_V1 -->
## Formula typesetting

Displayed equations use native, offline MathML with a responsive scroll
container.  This changes presentation only; no fitted value or analysis table
is altered.  After rebuilding either analysis stage, rerun:

```powershell
python scripts/format_math_environment.py --report-root "<canonical report directory>"
```

