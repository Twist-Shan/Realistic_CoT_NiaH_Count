from __future__ import annotations

import html
import os
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from realistic_niah.prompts import (
    COMMON_COUNTING_CUE,
    DIRECT_QUERY_BLOCK,
    ENUMERATION_BULLET_QUERY_BLOCK,
    ENUMERATION_INDEX_QUERY_BLOCK,
    NATIVE_THINKING_QUERY_BLOCK,
)

from .analysis import (
    FEATURE_LABELS,
    behavior_tables,
    formula_text,
    predict_selected_law,
)
from .spec import (
    EXPECTED_REQUESTS,
    EXPECTED_STIMULI,
    INSERTION_DEPTH_MAX_FRACTION,
    INSERTION_DEPTH_MIN_FRACTION,
    NEEDLE_COUNTS,
    PASSAGE_LENGTHS,
    SEEDS,
)

MODE_ORDER = (
    "direct",
    "enumeration_index",
    "enumeration_bullet",
    "native_thinking",
)
SLOT_ORDER = (
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3-14B",
    "Qwen3-32B",
    "Gemma4-E4B",
    "Gemma4-12B",
    "Gemma4-26B-A4B",
    "Gemma4-31B",
    "Nemotron-Nano-v2-9B",
    "Nemotron-3-Nano-4B",
    "GLM-4/Z1-9B",
    "Ministral-3-8B pair",
)
MODE_LABELS = {
    "direct": "Direct",
    "enumeration_index": "Index enumeration",
    "enumeration_bullet": "Bullet enumeration",
    "native_thinking": "Native thinking",
}
TARGET_LABELS = {
    "parseable_exact_accuracy": "Parseable exact-count accuracy",
    "signed_mean_deviation": "Mean signed deviation",
    "absolute_mean_deviation": "Mean absolute deviation",
    "signed_median_deviation": "Median signed deviation",
    "signed_trimmed_mean_deviation": "10% trimmed signed deviation",
    "signed_deviation_sample_variance": "Signed-deviation sample variance",
}


def _ordered_slots(values: pd.Series) -> list[str]:
    observed = set(values.astype(str))
    return [slot for slot in SLOT_ORDER if slot in observed] + sorted(
        observed - set(SLOT_ORDER)
    )


def _mode_order(values: pd.Series) -> list[str]:
    observed = set(values.astype(str))
    return [mode for mode in MODE_ORDER if mode in observed] + sorted(
        observed - set(MODE_ORDER)
    )


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _relative_asset(path: Path, report_path: Path) -> str:
    return Path(
        os.path.relpath(path.resolve(), start=report_path.parent.resolve())
    ).as_posix()


def plot_accuracy_heatmap(
    summary: pd.DataFrame,
    path: Path,
) -> Path:
    slots = _ordered_slots(summary["comparison_slot"])
    modes = _mode_order(summary["prompt_mode"])
    collapsed = (
        summary.groupby(["comparison_slot", "prompt_mode"])
        .apply(
            lambda group: np.average(
                group["parseable_exact_accuracy"],
                weights=group["requests"],
            ),
            include_groups=False,
        )
        .rename("accuracy")
        .reset_index()
    )
    pivot = (
        collapsed.pivot(
            index="comparison_slot",
            columns="prompt_mode",
            values="accuracy",
        )
        .reindex(index=slots, columns=modes)
    )
    fig, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(
        pivot.to_numpy(dtype=float),
        vmin=0,
        vmax=1,
        cmap="viridis",
        aspect="auto",
    )
    axis.set_xticks(range(len(modes)), [MODE_LABELS.get(x, x) for x in modes])
    axis.set_yticks(range(len(slots)), slots)
    axis.set_xlabel("Prompt mode")
    axis.set_ylabel("Behavior comparison slot")
    axis.set_title("Parseable exact-count accuracy")
    for row in range(len(slots)):
        for column in range(len(modes)):
            value = pivot.iloc[row, column]
            label = "NA" if pd.isna(value) else f"{value:.1%}"
            color = "white" if not pd.isna(value) and value < 0.55 else "black"
            axis.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )
    fig.colorbar(image, ax=axis, label="Accuracy")
    fig.tight_layout()
    return _save_figure(fig, path)


def plot_outcome_composition(
    outcomes: pd.DataFrame,
    path: Path,
) -> Path:
    slots = _ordered_slots(outcomes["comparison_slot"])
    classes = (
        "strict_success",
        "format_only_failure",
        "undercount",
        "overcount",
        "parse_failure",
        "truncation",
    )
    colors = {
        "strict_success": "#2E8B57",
        "format_only_failure": "#E6AB02",
        "undercount": "#377EB8",
        "overcount": "#E41A1C",
        "parse_failure": "#984EA3",
        "truncation": "#555555",
    }
    fig, axes = plt.subplots(2, 2, figsize=(16, 13), sharex=True)
    for axis, mode in zip(axes.flat, MODE_ORDER):
        subset = outcomes.loc[outcomes["prompt_mode"] == mode]
        pivot = (
            subset.pivot(
                index="comparison_slot",
                columns="outcome_class",
                values="proportion",
            )
            .reindex(index=slots, columns=classes)
            .fillna(0.0)
        )
        left = np.zeros(len(slots))
        y = np.arange(len(slots))
        for outcome in classes:
            values = pivot[outcome].to_numpy(dtype=float)
            axis.barh(
                y,
                values,
                left=left,
                color=colors[outcome],
                label=outcome.replace("_", " "),
                height=0.72,
            )
            left += values
        axis.set_yticks(y, slots, fontsize=8)
        axis.invert_yaxis()
        axis.set_xlim(0, 1)
        axis.set_title(MODE_LABELS[mode])
        axis.set_xlabel("Proportion of requests")
        axis.grid(axis="x", alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "Exclusive outcome composition by model slot and prompt mode",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    return _save_figure(fig, path)


def plot_condition_accuracy(
    by_condition: pd.DataFrame,
    *,
    prompt_mode: str,
    path: Path,
) -> Path:
    subset = by_condition.loc[
        by_condition["prompt_mode"] == prompt_mode
    ].copy()
    slots = _ordered_slots(subset["comparison_slot"])
    lengths = sorted(subset["L"].unique())
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(lengths)))
    fig, axes = plt.subplots(3, 4, figsize=(18, 13), sharex=True, sharey=True)
    for axis, slot in zip(axes.flat, slots):
        slot_rows = subset.loc[subset["comparison_slot"] == slot]
        for color, length in zip(colors, lengths):
            line = slot_rows.loc[slot_rows["L"] == length].sort_values("N")
            axis.plot(
                line["N"],
                line["parseable_exact_accuracy"],
                marker="o",
                markersize=3,
                linewidth=1.2,
                color=color,
                label=f"{length // 1000}k",
            )
        axis.set_title(slot, fontsize=10)
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.22)
    for axis in axes.flat[len(slots) :]:
        axis.set_visible(False)
    for axis in axes[-1, :]:
        axis.set_xlabel("N (true record count)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Exact-count accuracy")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Final passage length L",
        loc="lower center",
        ncol=len(lengths),
        frameon=False,
    )
    fig.suptitle(
        f"{MODE_LABELS[prompt_mode]}: accuracy across N and L",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    return _save_figure(fig, path)


def plot_selected_law(
    requests: pd.DataFrame,
    *,
    target: str,
    prompt_mode: str,
    candidate_name: str,
    path: Path,
) -> Path:
    prediction = predict_selected_law(
        requests,
        target=target,
        prompt_mode=prompt_mode,
        candidate_name=candidate_name,
    )
    slots = _ordered_slots(prediction["comparison_slot"])
    lengths = sorted(prediction["L"].unique())
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(lengths)))
    fig, axes = plt.subplots(
        3,
        4,
        figsize=(18, 13),
        sharex=True,
        sharey=True,
    )
    for axis, slot in zip(axes.flat, slots):
        slot_rows = prediction.loc[prediction["comparison_slot"] == slot]
        for color, length in zip(colors, lengths):
            line = slot_rows.loc[slot_rows["L"] == length].sort_values("N")
            axis.scatter(
                line["N"],
                line["observed"],
                s=17,
                alpha=0.65,
                color=color,
            )
            axis.plot(
                line["N"],
                line["predicted"],
                linewidth=1.4,
                color=color,
                label=f"{length // 1000}k",
            )
        axis.set_title(slot, fontsize=10)
        axis.axhline(0, color="#888888", linewidth=0.7, alpha=0.6)
        axis.grid(alpha=0.22)
        if target == "parseable_exact_accuracy":
            axis.set_ylim(-0.03, 1.03)
    for axis in axes.flat[len(slots) :]:
        axis.set_visible(False)
    for axis in axes[-1, :]:
        axis.set_xlabel("N (true record count)")
    ylabel = TARGET_LABELS[target]
    for axis in axes[:, 0]:
        axis.set_ylabel(ylabel)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Final passage length L",
        loc="lower center",
        ncol=len(lengths),
        frameon=False,
    )
    fig.suptitle(
        f"{MODE_LABELS[prompt_mode]} — {TARGET_LABELS[target]}\n"
        f"points: observed condition estimates; lines: {candidate_name}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    return _save_figure(fig, path)


def _table_html(frame: pd.DataFrame, *, digits: int = 4) -> str:
    display = frame.copy()
    numeric = display.select_dtypes(include=[np.number]).columns
    display[numeric] = display[numeric].round(digits)
    return display.to_html(index=False, border=0, classes="data-table")


def _details(summary: str, body: str, *, open_by_default: bool = False) -> str:
    opened = " open" if open_by_default else ""
    return (
        f"<details{opened}><summary>{html.escape(summary)}</summary>"
        f"<div class='details-body'>{body}</div></details>"
    )


def _base_html(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<script>
window.MathJax = {{tex: {{inlineMath: [['\\\\(','\\\\)']], displayMath: [['\\\\[','\\\\]']]}}}};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
:root {{ --ink:#16202a; --muted:#5c6975; --line:#d9e0e6; --accent:#2457a7; }}
body {{ max-width:1180px; margin:0 auto; padding:32px 28px 80px;
  color:var(--ink); font:16px/1.58 system-ui,-apple-system,Segoe UI,sans-serif; }}
h1,h2,h3 {{ line-height:1.22; margin-top:1.5em; }}
h1 {{ border-bottom:2px solid var(--ink); padding-bottom:.3em; }}
.lede {{ color:var(--muted); font-size:1.08rem; }}
.callout {{ border-left:4px solid var(--accent); background:#f3f7fc;
  padding:12px 16px; margin:18px 0; }}
code,pre {{ font-family:ui-monospace,SFMono-Regular,Consolas,monospace; }}
pre {{ white-space:pre-wrap; background:#f6f7f8; border:1px solid var(--line);
  padding:14px; border-radius:6px; }}
figure {{ margin:24px 0; }}
figure img {{ width:100%; height:auto; border:1px solid var(--line); }}
figcaption {{ color:var(--muted); margin-top:8px; }}
details {{ border:1px solid var(--line); border-radius:7px; margin:14px 0; }}
summary {{ cursor:pointer; padding:11px 14px; font-weight:650; }}
.details-body {{ padding:0 14px 14px; overflow-x:auto; }}
.data-table {{ border-collapse:collapse; width:100%; font-size:.86rem; }}
.data-table th,.data-table td {{ padding:6px 8px; border-bottom:1px solid var(--line);
  text-align:right; white-space:nowrap; }}
.data-table th:first-child,.data-table td:first-child {{ text-align:left; }}
.formula {{ overflow-x:auto; padding:12px; background:#fafafa;
  border:1px solid var(--line); }}
.small {{ color:var(--muted); font-size:.9rem; }}
</style>
</head>
<body>{body}</body></html>"""


def write_behavior_report(
    *,
    output_path: Path,
    summary: pd.DataFrame,
    by_condition: pd.DataFrame,
    outcomes: pd.DataFrame,
    paired_comparisons: pd.DataFrame,
    plot_paths: list[Path],
) -> Path:
    weighted = (
        summary.groupby(["comparison_slot", "prompt_mode"])
        .apply(
            lambda group: np.average(
                group["parseable_exact_accuracy"],
                weights=group["requests"],
            ),
            include_groups=False,
        )
        .rename("accuracy")
        .reset_index()
    )
    best = (
        weighted.sort_values(
            ["comparison_slot", "accuracy"],
            ascending=[True, False],
        )
        .groupby("comparison_slot")
        .first()
        .reset_index()
    )
    best["accuracy"] = best["accuracy"].map(lambda value: f"{value:.1%}")
    prompt_blocks = {
        "Direct": DIRECT_QUERY_BLOCK,
        "Index enumeration": ENUMERATION_INDEX_QUERY_BLOCK,
        "Bullet enumeration": ENUMERATION_BULLET_QUERY_BLOCK,
        "Native thinking": NATIVE_THINKING_QUERY_BLOCK,
    }
    prompt_html = f"<pre>{html.escape(COMMON_COUNTING_CUE)}\n\n&lt;passage&gt;\n{{PASSAGE}}\n&lt;/passage&gt;</pre>"
    for label, block in prompt_blocks.items():
        prompt_html += (
            f"<h3>{html.escape(label)}</h3><pre>{html.escape(block)}</pre>"
        )

    figures = []
    for path in plot_paths:
        relative = _relative_asset(path, output_path)
        if "accuracy_heatmap" in path.name:
            caption = (
                "Rows are behavior-comparison slots and columns are prompt "
                "modes; each cell is parseable exact-count accuracy."
            )
        elif "outcome_composition" in path.name:
            caption = (
                "Exclusive error categories sum to one within each model × "
                "mode. Undercount and overcount require a parsed numeric answer."
            )
        else:
            caption = (
                "Horizontal axis: true needle count N. Vertical axis: exact "
                "accuracy across ten seeds. Color denotes final passage length L."
            )
        figures.append(
            f"<figure><img src='{html.escape(relative)}' alt='behavior plot'>"
            f"<figcaption>{html.escape(caption)}</figcaption></figure>"
        )
    long_tables = (
        _details(
            "Full model × checkpoint × mode summary",
            _table_html(summary),
        )
        + _details(
            "All N × L condition estimates",
            _table_html(by_condition),
        )
        + _details(
            "Exclusive outcome-class counts",
            _table_html(outcomes),
        )
        + _details(
            "All paired prompt-mode comparisons",
            _table_html(paired_comparisons),
        )
    )
    body = rf"""
<h1>Realistic NIAH V3 — Behavior Comparison</h1>
<p class="lede">Registered analysis of the behavior arm only. No activation,
attention, intervention, or causal claim is made in this report.</p>
<div class="callout"><strong>Primary result definition.</strong> A request is
correct only when an integer is successfully parsed and equals the true count
\(N\). A parseable but wrong integer is incorrect. Strict registered success
additionally requires the mode-specific output format and no length truncation.</div>
<h2>1. Experimental design</h2>
<p>The shared dataset contains {EXPECTED_STIMULI:,} stimuli:
\(7\) final passage lengths \(L={list(PASSAGE_LENGTHS)}\), \(14\) true
record counts \(N={list(NEEDLE_COUNTS)}\), and seeds
{SEEDS[0]}–{SEEDS[-1]}. Needle starts are restricted to the inclusive
{INSERTION_DEPTH_MIN_FRACTION:.0%}–{INSERTION_DEPTH_MAX_FRACTION:.0%}
character-depth interval of the final passage. Across the registered raw
checkpoints and mode assignments, the complete run contains
{EXPECTED_REQUESTS:,} requests.</p>
<p><strong>Conclusion.</strong> Every prompt mode is evaluated on the same
stimuli. Differences among modes are paired at the stimulus level; differences
inside the GLM and Ministral slots also include their explicitly documented
checkpoint change.</p>
<h2>2. Frozen prompts</h2>
{prompt_html}
<p><strong>Conclusion.</strong> The common cue and passage are identical.
Only the post-passage response strategy changes.</p>
<h2>3. Metrics and error accounting</h2>
<ul>
<li><strong>Parseable exact accuracy:</strong>
\(\mathbf{{1}}[\hat N\text{{ exists and }}\hat N=N]\).</li>
<li><strong>Strict registered accuracy:</strong> exact count, compliant final
format, and no length truncation.</li>
<li><strong>Signed deviation:</strong> \(d=\hat N-N\), defined only after
successful parsing; \(d&lt;0\) is undercount and \(d&gt;0\) is overcount.</li>
<li><strong>Exclusive outcome class:</strong> truncation → parse failure →
under/overcount → format-only failure → strict success, in that priority order.</li>
</ul>
<p><strong>Conclusion.</strong> The report never turns formatting or
truncation failures into numeric count errors and never hides them by
conditioning the accuracy denominator on parsing.</p>
<h2>4. Results</h2>
{''.join(figures)}
<h3>Best observed mode within each behavior slot</h3>
{_table_html(best)}
<p class="small">“Best” is descriptive on the full sample, not a significance
claim. Paired uncertainty should be used for confirmatory mode comparisons.</p>
<h3>Paired mode differences</h3>
<p>For every pair of modes within a behavior slot, the table below reports
\(\widehat{{p}}_B-\widehat{{p}}_A\): positive values favor mode B. The 95%
interval resamples the ten seeds as clusters, keeping all N×L cells from one
seed together. The exact McNemar test uses discordant stimulus pairs; its
p-values are Holm-adjusted across the six mode contrasts within each slot.</p>
{_details(
    "Paired risk differences, clustered intervals, and McNemar tests",
    _table_html(paired_comparisons),
    open_by_default=True,
)}
<p><strong>Conclusion.</strong> Descriptive mode rankings and paired
differences are both shown. For GLM and Ministral, a native-thinking contrast
also changes checkpoint and is therefore not a same-weight prompting effect.</p>
{long_tables}
<h2>5. Scope of conclusions</h2>
<p>These results support claims about behavioral accuracy, directional count
error, format compliance, and truncation under this registered prompt and
sampling setup. They do not identify an internal counting mechanism and do not
establish that a reasoning trace causally produced the answer.</p>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _base_html("Realistic NIAH V3 Behavior Comparison", body),
        encoding="utf-8",
    )
    return output_path


def _latex_formula(target: str, candidate_name: str) -> str:
    feature_map = {
        "N": "N",
        "L_k": "L/1000",
        "ln_N": "\\ln N",
        "ln_L_k": "\\ln(L/1000)",
        "density_per_1k": "\\frac{N}{L/1000}",
        "N_x_L_k": "N(L/1000)",
        "ln_N_x_ln_L_k": "\\ln N\\,\\ln(L/1000)",
    }
    from .analysis import CANDIDATES

    candidate = next(
        item for item in CANDIDATES if item.name == candidate_name
    )
    left = (
        "\\operatorname{logit}\\Pr(\\hat N=N)"
        if target == "parseable_exact_accuracy"
        else "\\mu_{m}(N,L)"
    )
    terms = ["\\alpha_m"] + [
        f"\\beta_{index + 1}{feature_map[feature]}"
        for index, feature in enumerate(candidate.features)
    ]
    return f"{left}=" + "+".join(terms)


def write_empirical_law_report(
    *,
    output_path: Path,
    selected: pd.DataFrame,
    comparisons: pd.DataFrame,
    coefficients: pd.DataFrame,
    plot_paths: list[Path],
) -> Path:
    selected_sections: list[str] = []
    for row in selected.sort_values(["target", "prompt_mode"]).itertuples():
        target = str(row.target)
        mode = str(row.prompt_mode)
        candidate = str(row.candidate)
        if target == "parseable_exact_accuracy":
            metrics = (
                f"held-out log loss={row.cv_log_loss_mean:.4f}; "
                f"Brier={row.cv_brier_mean:.4f}; "
                f"deviance explained={row.cv_deviance_explained_mean:.3f}"
            )
        else:
            metrics = (
                f"held-out R²={row.cv_r2_mean:.3f}; "
                f"MAE={row.cv_mae_mean:.3f}; RMSE={row.cv_rmse_mean:.3f}"
            )
        term_table = coefficients.loc[
            (coefficients["target"] == target)
            & (coefficients["prompt_mode"] == mode)
            & (coefficients["candidate"] == candidate)
        ]
        selected_sections.append(
            f"<h3>{html.escape(TARGET_LABELS[target])} — "
            f"{html.escape(MODE_LABELS[mode])}</h3>"
            f"<div class='formula'>\\[{_latex_formula(target, candidate)}\\]</div>"
            f"<p><strong>Selected candidate:</strong> {html.escape(candidate)}; "
            f"{html.escape(metrics)}.</p>"
            + _details(
                "Fitted coefficients and 95% intervals",
                _table_html(term_table),
            )
        )

    figure_sections: list[str] = []
    for path in plot_paths:
        relative = _relative_asset(path, output_path)
        stem = path.stem.replace("selected_", "")
        figure_sections.append(
            _details(
                stem.replace("_", " "),
                (
                    f"<figure><img src='{html.escape(relative)}' "
                    "alt='observed and fitted empirical law'>"
                    "<figcaption>Each panel is one behavior-comparison slot. "
                    "Points are observed N×L condition estimates; lines are "
                    "predictions from the selected shared-slope law. The "
                    "horizontal axis is N and color is L.</figcaption></figure>"
                ),
            )
        )

    low_support: list[str] = []
    for row in selected.itertuples():
        if str(row.target) == "parseable_exact_accuracy":
            if float(row.cv_deviance_explained_mean) < 0.1:
                low_support.append(
                    f"{MODE_LABELS[str(row.prompt_mode)]} accuracy"
                )
        elif float(row.cv_r2_mean) < 0.3:
            low_support.append(
                f"{MODE_LABELS[str(row.prompt_mode)]} "
                f"{TARGET_LABELS[str(row.target)]}"
            )
    support_note = (
        "The following selected forms remain weak out of sample and must not "
        "be presented as stable laws: "
        + ", ".join(low_support)
        + "."
        if low_support
        else "All selected forms clear the report's descriptive support flag; "
        "this still does not make them causal or valid outside the tested grid."
    )
    body = rf"""
<h1>Realistic NIAH V3 — Empirical Laws</h1>
<p class="lede">Shared-form, model-stratified response surfaces for accuracy
and count deviation. Candidate selection uses held-out seeds, not training
fit.</p>
<div class="callout"><strong>Unified-law form.</strong> For each prompt mode,
all behavior slots share the same \(N,L\) coefficients. Each slot receives only
its own intercept \(\alpha_m\). This is a single functional form with
model-specific baseline, not a separate curve search per model.</div>
<h2>1. Estimands</h2>
<p>Let \(\hat N\) be the parsed prediction and \(d=\hat N-N\).</p>
<ul>
<li>Accuracy is \(\Pr(\hat N=N)\) over every request; parse failures are zero.</li>
<li>Mean, median, and 10% trimmed signed deviations are conditional on
successful parsing. Their companion parse rate is reported in the behavior
report.</li>
<li>Absolute deviation is \(|d|\). Sample variance is
\(s_d^2=(n-1)^{{-1}}\sum_i(d_i-\bar d)^2\) within an N×L condition.</li>
</ul>
<p><strong>Conclusion.</strong> Accuracy measures total task success; bias
measures the direction of numeric errors among responses that supplied a
number. They answer different questions and are not interchangeable.</p>
<h2>2. Candidate coordinates and validation</h2>
<p>The bounded grid compares additive subsets and two hierarchical,
first-order interactions drawn from
\(N, L/1000, \ln N, \ln(L/1000), N/(L/1000)\). Interaction models always
include both parent terms. Five-fold cross-validation holds out complete
seeds across all N, L, models, and modes. Accuracy is ranked by held-out log
loss/Brier score; continuous targets by held-out condition-level R², MAE, and
RMSE. A one-standard-error-style rule chooses the simpler near-best form.
An interaction is eligible only when its fitted interaction coefficient has
\(p&lt;0.05\).</p>
<p><strong>Conclusion.</strong> This search is finite and preregisterable.
Every attempted formula is retained below; the report does not show only the
best-looking curve.</p>
<h2>3. Selected laws</h2>
{''.join(selected_sections)}
<div class="callout"><strong>Support check.</strong> {html.escape(support_note)}</div>
<h2>4. Observed points and selected curves</h2>
{''.join(figure_sections)}
<h2>5. Complete comparison tables</h2>
{_details("All candidate metrics", _table_html(comparisons))}
{_details("All fitted coefficients", _table_html(coefficients))}
<h2>6. Mathematical interpretation and limits</h2>
<p>A linear term means that the tested response changes by an approximately
constant amount per additional record (or per 1k passage tokens) inside this
finite grid. A log term means diminishing marginal change: equal multiplicative
changes in N or L produce approximately equal response changes. Density
\(N/(L/1000)\) asks whether records per thousand tokens collapse the two axes.
These are descriptive response surfaces. They neither prove an internal
counting algorithm nor justify extrapolation beyond N=1–20 and L=2k–20k.</p>
<p><strong>Conclusion.</strong> A high held-out score supports a compact
within-domain empirical regularity. A low or negative held-out score is a valid
finding that no reliable unified law was found for that target and mode.</p>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _base_html("Realistic NIAH V3 Empirical Laws", body),
        encoding="utf-8",
    )
    return output_path


def build_all_plots(
    *,
    requests: pd.DataFrame,
    selected: pd.DataFrame,
    output_dir: Path,
) -> tuple[list[Path], list[Path]]:
    summary, by_condition, outcomes = behavior_tables(requests)
    behavior_dir = output_dir / "behavior"
    empirical_dir = output_dir / "empirical_law"
    behavior_paths = [
        plot_accuracy_heatmap(
            summary,
            behavior_dir / "accuracy_heatmap.png",
        ),
        plot_outcome_composition(
            outcomes,
            behavior_dir / "outcome_composition.png",
        ),
    ]
    for mode in MODE_ORDER:
        if mode in set(by_condition["prompt_mode"]):
            behavior_paths.append(
                plot_condition_accuracy(
                    by_condition,
                    prompt_mode=mode,
                    path=behavior_dir / f"accuracy_N_L_{mode}.png",
                )
            )

    empirical_paths: list[Path] = []
    for row in selected.itertuples():
        target = str(row.target)
        mode = str(row.prompt_mode)
        candidate = str(row.candidate)
        empirical_paths.append(
            plot_selected_law(
                requests,
                target=target,
                prompt_mode=mode,
                candidate_name=candidate,
                path=(
                    empirical_dir
                    / f"selected_{target}_{mode}.png"
                ),
            )
        )
    return behavior_paths, empirical_paths
