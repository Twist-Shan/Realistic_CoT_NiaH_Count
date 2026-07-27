"""Apply offline MathML typesetting and responsive layout fixes to the report.

This is an idempotent post-processing step.  It intentionally changes only
HTML presentation and checksum manifests; the experiment data, fitted
parameters, validation metrics, figures, and analysis manifests are untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import re
from pathlib import Path


STYLE_MARKER = "COUNTING_MATH_ENV_V1"
STYLE_START = "/* COUNTING_MATH_ENV_V1:"
STYLE_END = "/* END COUNTING_MATH_ENV_V1 */"
README_MARKER = "<!-- COUNTING_MATH_ENV_V1 -->"

MATH_STYLE = r"""
/* COUNTING_MATH_ENV_V1: paper-style offline MathML and responsive layout */
main {
  min-width: 0;
  counter-reset: report-equation;
}
#counting-mechanism-law-v1,
#counting-mechanism-stage2,
#model-bias-noise-v1 {
  min-width: 0;
}
/*
 * Display equations intentionally follow a conventional paper layout:
 * no colored card, no shadow, centered mathematics, and a right-aligned
 * sequential equation number.  The surrounding prose carries definitions.
 */
.formula.math-equation,
.math-equation {
  width: min(100%, 1040px);
  max-width: 1040px;
  margin: 28px auto 30px;
  padding: 0 8px;
  border: 0;
  border-radius: 0;
  background: transparent;
  box-shadow: none;
}
.math-equation .equation-title {
  max-width: 920px;
  margin: 0 auto 9px;
  color: var(--ink);
  font-family: Inter, "Segoe UI", Arial, sans-serif;
  font-size: .92rem;
  font-weight: 700;
  letter-spacing: .01em;
  line-height: 1.45;
  text-transform: none;
}
.math-scroll {
  counter-increment: report-equation;
  display: grid;
  grid-template-columns: minmax(max-content, 1fr) 3.35rem;
  align-items: center;
  width: 100%;
  max-width: 100%;
  min-height: 3.15rem;
  overflow-x: auto;
  overflow-y: hidden;
  padding: 7px 0 8px;
  border: 0;
  background: transparent;
  scrollbar-width: thin;
}
.math-scroll::after {
  content: "(" counter(report-equation) ")";
  grid-column: 2;
  justify-self: end;
  align-self: center;
  padding-left: .65rem;
  color: var(--muted);
  font-family: "Times New Roman", "Cambria Math", serif;
  font-size: .97rem;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.math-scroll + .math-scroll {
  margin-top: 6px;
}
.math-scroll math[display="block"] {
  grid-column: 1;
  justify-self: center;
  display: block;
  width: max-content;
  max-width: none;
  margin: 0 1.15rem;
  color: var(--ink);
  font-family: "Cambria Math", "STIX Two Math", "Times New Roman", serif;
  font-size: 1.34rem;
  line-height: 1.55;
}
.math-equation .equation-note {
  max-width: 920px;
  margin: 8px auto 0;
  padding: 0;
  border: 0;
  color: var(--muted);
  font-size: .90rem;
  line-height: 1.65;
}
.inline-math {
  font-family: "Cambria Math", "STIX Two Math", "Times New Roman", serif;
  font-size: 1.02em;
  white-space: nowrap;
}
#counting-mechanism-law-v1 figure,
#counting-mechanism-stage2 figure,
#counting-mechanism-law-v1 .figure,
#counting-mechanism-stage2 .figure {
  width: min(100%, 1080px);
  max-width: 1080px;
  margin: 28px auto 36px;
}
#counting-mechanism-law-v1 figure img,
#counting-mechanism-stage2 figure img,
#counting-mechanism-law-v1 .figure img,
#counting-mechanism-stage2 .figure img {
  display: block;
  width: 100%;
  max-width: 100%;
  height: auto;
  border: 1px solid var(--line);
  background: #fff;
}
#counting-mechanism-law-v1 .table-wrap,
#counting-mechanism-stage2 .table-wrap {
  display: block;
  width: 100%;
  max-width: 100%;
  overflow-x: auto;
}
#counting-mechanism-law-v1 .table-wrap table,
#counting-mechanism-stage2 .table-wrap table {
  width: max-content;
  min-width: 100%;
}
@media (max-width: 720px) {
  .formula.math-equation,
  .math-equation {
    margin: 22px auto 24px;
    padding: 0;
  }
  .math-equation .equation-title {
    padding: 0 2px;
    font-size: .86rem;
  }
  .math-scroll {
    grid-template-columns: minmax(max-content, 1fr) 2.5rem;
    min-height: 2.8rem;
    padding: 5px 0 6px;
  }
  .math-scroll::after {
    padding-left: .45rem;
    font-size: .84rem;
  }
  .math-scroll math[display="block"] {
    margin: 0 .7rem;
    font-size: 1.08rem;
  }
  .math-equation .equation-note {
    padding: 0 2px;
    font-size: .86rem;
  }
}
/* END COUNTING_MATH_ENV_V1 */
"""


def atom(tag: str, text: str, attrs: str = "") -> str:
    return f"<{tag}{attrs}>{html.escape(text)}</{tag}>"


def mi(text: str) -> str:
    return atom("mi", text)


def mn(text: str) -> str:
    return atom("mn", text)


def mo(text: str) -> str:
    return atom("mo", text)


def mt(text: str) -> str:
    return atom("mtext", text)


def row(*items: str) -> str:
    return "<mrow>" + "".join(items) + "</mrow>"


def sub(base: str, script: str) -> str:
    return f"<msub>{base}{script}</msub>"


def sup(base: str, script: str) -> str:
    return f"<msup>{base}{script}</msup>"


def frac(num: str, den: str) -> str:
    return f"<mfrac>{num}{den}</mfrac>"


def sqrt(expr: str) -> str:
    return f"<msqrt>{expr}</msqrt>"


def paren(expr: str) -> str:
    return row(mo("("), expr, mo(")"))


def bracket(expr: str) -> str:
    return row(mo("["), expr, mo("]"))


def brace(expr: str) -> str:
    return row(mo("{"), expr, mo("}"))


def call(name: str, expr: str) -> str:
    return row(atom("mi", name, ' mathvariant="normal"'), paren(expr))


def log_ratio(symbol: str, scale: str, base: str = "ln") -> str:
    return call(base, frac(mi(symbol), mn(scale)))


def indicator(text: str) -> str:
    return row(atom("mi", "1", ' mathvariant="double-struck"'), brace(mt(text)))


def math_block(tex: str, body: str) -> str:
    label = html.escape(tex, quote=True)
    annotation = html.escape(tex)
    return (
        '<div class="math-scroll">'
        f'<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" '
        f'aria-label="{label}"><semantics>{body}'
        f'<annotation encoding="application/x-tex">{annotation}</annotation>'
        "</semantics></math></div>"
    )


def card(title: str, equations: list[tuple[str, str]], note: str = "") -> str:
    parts = [
        '<div class="formula equation-card math-equation">',
        f'<div class="equation-title">{title}</div>',
    ]
    parts.extend(math_block(tex, body) for tex, body in equations)
    if note:
        parts.append(f'<div class="equation-note">{note}</div>')
    parts.append("</div>")
    return "\n".join(parts)


def logit(expr: str) -> str:
    return row(atom("mi", "logit", ' mathvariant="normal"'), paren(expr))


def expectation(expr: str) -> str:
    return row(atom("mi", "E", ' mathvariant="double-struck"'), bracket(expr))


def make_cards() -> dict[str, str]:
    n_hat_i = sub(row("<mover>", mi("N"), mo("^"), "</mover>"), mi("i"))
    # The direct string above is deliberately avoided below where clarity matters.
    n_hat_i = f"<msub><mover>{mi('N')}{mo('^')}</mover>{mi('i')}</msub>"
    yi = sub(mi("Y"), mi("i"))
    parsed_i = sub(atom("mi", "parsed", ' mathvariant="normal"'), mi("i"))
    truncated_i = sub(atom("mi", "truncated", ' mathvariant="normal"'), mi("i"))
    ni = sub(mi("N"), mi("i"))
    bi = sub(mi("b"), mi("i"))
    ei = sub(mi("e"), mi("i"))
    zi = sub(mi("z"), mi("i"))

    primary = row(
        yi,
        mo("="),
        indicator("parsedᵢ=1, truncatedᵢ=0, N̂ᵢ=Nᵢ"),
        mo(","),
        mt("Accuracy"),
        mo("="),
        frac(mn("1"), mi("n")),
        "<munder>" + mo("∑") + mi("i") + "</munder>",
        yi,
    )
    errors = row(
        bi,
        mo("="),
        n_hat_i,
        mo("−"),
        ni,
        mo(","),
        ei,
        mo("="),
        mo("|"),
        bi,
        mo("|"),
        mo(","),
        parsed_i,
        mo("="),
        mn("1"),
    )
    density = row(
        mi("d"),
        mo("="),
        frac(row(mn("1000"), mi("N")), mi("T")),
        mo(","),
        mt("unit: needles per 1k canonical passage tokens"),
    )

    pm_args = row(sub(mi("p"), mi("m")), paren(row(mi("T"), mo(","), mi("N"), mo(","), mi("q"), mo(","), mi("o"))))
    am = sub(mi("A"), mi("m"))
    rm = sub(mi("r"), mi("m"))
    sm = sub(mi("s"), mi("m"))
    dq = sub(mi("δ"), mi("q"))
    go = sub(mi("γ"), mi("o"))
    hill_inside = row(
        mn("1"),
        mo("+"),
        am,
        sup(paren(frac(mi("T"), mn("5000"))), rm),
        sup(paren(frac(mi("N"), mn("5"))), sm),
        call("exp", row(mo("−"), dq, mo("−"), go)),
    )
    hill = row(pm_args, mo("="), sup(bracket(hill_inside), row(mo("−"), mn("1"))))
    hill_logit = row(
        logit(pm_args),
        mo("="),
        mo("−"),
        call("ln", am),
        mo("−"),
        rm,
        log_ratio("T", "5000"),
        mo("−"),
        sm,
        log_ratio("N", "5"),
        mo("+"),
        dq,
        mo("+"),
        go,
    )

    abs_line = row(zi, mo("="), call("ln", row(mn("1"), mo("+"), mo("|"), bi, mo("|"))))
    bm = sub(mi("B"), mi("m"))
    um = sub(mi("u"), mi("m"))
    vm = sub(mi("v"), mi("m"))
    eta_q = sub(mi("η"), mi("q"))
    kappa_o = sub(mi("κ"), mi("o"))
    abs_expect = row(
        expectation(row(zi, mo("|"), mt("parsed, m, T, N, q, o"))),
        mo("≈"),
        call("ln", bm),
        mo("+"),
        um,
        log_ratio("T", "5000"),
        mo("+"),
        vm,
        log_ratio("N", "5"),
        mo("+"),
        eta_q,
        mo("+"),
        kappa_o,
    )

    asinh_line = row(
        bi,
        mo("="),
        n_hat_i,
        mo("−"),
        ni,
        mo(","),
        zi,
        mo("="),
        call("arsinh", bi),
        mo("="),
        call("ln", row(bi, mo("+"), sqrt(row(sup(bi, mn("2")), mo("+"), mn("1"))))),
    )
    cm = row(sub(mi("c"), mi("m")), paren(row(mi("T"), mo(","), mi("N"), mo(","), mi("q"), mo(","), mi("o"))))
    centered = row(
        cm,
        mo("="),
        call("sinh", expectation(row(zi, mo("|"), mt("parsed, m, T, N, q, o")))),
    )

    alpha = sub(mi("α"), row(mi("m"), mo(","), mi("q"), mo(","), mi("o")))
    beta_t = sub(mi("β"), row(mi("m"), mo(","), mi("T")))
    beta_n = sub(mi("β"), row(mi("m"), mo(","), mi("N")))
    fixed_bias = row(
        expectation(row(zi, mo("|"), mt("parsed, m, T, N, q, o"))),
        mo("="),
        alpha,
        mo("+"),
        beta_t,
        call(row(mt("log"), sub(mn(""), mn(""))) if False else "log₂", frac(mi("T"), mn("5000"))),
        mo("+"),
        beta_n,
        call("log₂", frac(mi("N"), mn("5"))),
    )

    # Stage-1 response surfaces.
    pm_lno = row(sub(mi("p"), mi("m")), paren(row(mi("L"), mo(","), mi("N"), mo(","), mi("o"))))
    am_l = sub(mi("a"), mi("m"))
    om = sub(mi("o"), mi("m"))
    query_i = indicator("query-last")
    linear = row(
        am_l,
        mo("+"),
        rm,
        log_ratio("L", "5000"),
        mo("+"),
        sm,
        log_ratio("N", "5"),
        mo("+"),
        om,
        query_i,
    )
    survival = row(pm_lno, mo("="), call("exp", row(mo("−"), call("exp", linear))))
    logistic = row(logit(pm_lno), mo("="), linear)

    qm_lno = row(sub(mi("q"), mi("m")), paren(row(mi("L"), mo(","), mi("N"), mo(","), mi("o"))))
    q_linear = row(logit(qm_lno), mo("="), linear)
    p_all = row(
        mt("P"),
        paren(mt("all N gold pairs found")),
        mo("≈"),
        sup(qm_lno, mi("N")),
    )

    n_eff = sub(mi("N"), row(mt("eff"), mo(","), mi("m")))
    kappa_m = sub(mi("κ"), mi("m"))
    corr1 = row(
        mt("P"),
        paren(row(mt("all pairs found"), mo("|"), mi("L"), mo(","), mi("N"), mo(","), mi("m"), mo(","), mi("o"))),
        mo("="),
        sup(qm_lno, row(n_eff, paren(mi("N")))),
    )
    corr2 = row(
        row(n_eff, paren(mi("N"))),
        mo("="),
        kappa_m,
        sup(mi("N"), mi("τ")),
    )

    n_hat = "<mover>" + mi("N") + mo("^") + "</mover>"
    rel_err = frac(row(n_hat, mo("−"), mi("N")), mi("N"))
    abs_rel = frac(row(mo("|"), n_hat, mo("−"), mi("N"), mo("|")), mi("N"))
    hurdle = row(
        expectation(rel_err),
        mo("="),
        mt("P"),
        paren(mt("over")),
        expectation(row(abs_rel, mo("|"), mt("over"))),
        mo("−"),
        mt("P"),
        paren(mt("under")),
        expectation(row(abs_rel, mo("|"), mt("under"))),
    )
    exact_decomp = row(
        mt("P"),
        paren(mt("exact")),
        mo("="),
        mt("P"),
        paren(mt("parse")),
        mo("×"),
        mt("P"),
        paren(row(mt("exact"), mo("|"), mt("parse"))),
    )
    ref_power = row(
        logit(sub(mi("p"), mi("m"))),
        mo("="),
        sub(mi("a"), mi("m")),
        mo("+"),
        sub(mi("r"), mi("m")),
        log_ratio("L", "5000"),
        mo("+"),
        sub(mi("s"), mi("m")),
        log_ratio("N", "5"),
        mo("+"),
        mt("query-order nuisance"),
    )

    return {
        "Primary outcome：全请求 exact accuracy": card(
            "Primary outcome：全请求 exact accuracy",
            [(r"Y_i=\mathbb{1}\{\mathrm{parsed}_i=1,\ \mathrm{truncated}_i=0,\ \widehat N_i=N_i\},\quad \mathrm{Accuracy}=\frac1n\sum_iY_i", primary)],
        ),
        "Conditional error outcomes：只在 parsed outputs 上定义": card(
            "Conditional error outcomes：只在 parsed outputs 上定义",
            [(r"b_i=\widehat N_i-N_i,\quad e_i=|b_i|,\quad \mathrm{parsed}_i=1", errors)],
        ),
        "Needle density": card(
            "Needle density",
            [(r"d=\frac{1000N}{T}\quad [\mathrm{needles\ per\ 1k\ canonical\ passage\ tokens}]", density)],
        ),
        "推荐的 exact-accuracy law": card(
            "推荐的 exact-accuracy law",
            [
                (r"p_m(T,N,q,o)=\left[1+A_m(T/5000)^{r_m}(N/5)^{s_m}e^{-\delta_q-\gamma_o}\right]^{-1}", hill),
                (r"\operatorname{logit}p_m=-\ln A_m-r_m\ln(T/5000)-s_m\ln(N/5)+\delta_q+\gamma_o", hill_logit),
            ],
        ),
        "Conditional absolute-error law": card(
            "Conditional absolute-error law",
            [
                (r"z_i=\ln(1+|b_i|)", abs_line),
                (r"\mathbb E[z_i\mid\mathrm{parsed},m,T,N,q,o]\approx\ln B_m+u_m\ln(T/5000)+v_m\ln(N/5)+\eta_q+\kappa_o", abs_expect),
            ],
        ),
        "Robust signed-bias target": card(
            "Robust signed-bias target",
            [
                (r"b_i=\widehat N_i-N_i,\quad z_i=\operatorname{arsinh}(b_i)=\ln(b_i+\sqrt{b_i^2+1})", asinh_line),
                (r"c_m(T,N,q,o)=\sinh\{\mathbb E[z_i\mid\mathrm{parsed},m,T,N,q,o]\}", centered),
            ],
        ),
        "用于跨模型比较的固定函数族": card(
            "用于跨模型比较的固定函数族",
            [(r"\mathbb E[z_i\mid\mathrm{parsed},m,T,N,q,o]=\alpha_{m,q,o}+\beta_{m,T}\log_2(T/5000)+\beta_{m,N}\log_2(N/5)", fixed_bias)],
        ),
        "Failure-hazard / survival law": card(
            "Failure-hazard / survival law",
            [(r"p_m(L,N,o)=\exp\{-\exp[a_m+r_m\ln(L/5000)+s_m\ln(N/5)+o_m\mathbb{1}_{\mathrm{query-last}}]\}", survival)],
        ),
        "Logistic / Hill law": card(
            "Logistic / Hill law",
            [(r"\operatorname{logit}p_m(L,N,o)=a_m+r_m\ln(L/5000)+s_m\ln(N/5)+o_m\mathbb{1}_{\mathrm{query-last}}", logistic)],
        ),
        "Enumeration retrieval compounding": card(
            "Enumeration retrieval compounding",
            [
                (r"\operatorname{logit}q_m(L,N,o)=a_m+r_m\ln(L/5000)+s_m\ln(N/5)+o_m\mathbb{1}_{\mathrm{query-last}}", q_linear),
                (r"P(\mathrm{all\ }N\mathrm{\ gold\ pairs\ found})\approx q_m(L,N,o)^N", p_all),
            ],
        ),
        "Correlated-retrieval law": card(
            "Correlated-retrieval law",
            [
                (r"P(\mathrm{all\ pairs\ found}\mid L,N,m,o)=q_m(L,N,o)^{N_{\mathrm{eff},m}(N)}", corr1),
                (r"N_{\mathrm{eff},m}(N)=\kappa_mN^\tau", corr2),
            ],
        ),
        "Two-part bias law": card(
            "Two-part bias law",
            [(r"\mathbb E[(\widehat N-N)/N]=P(\mathrm{over})\mathbb E[|\widehat N-N|/N\mid\mathrm{over}]-P(\mathrm{under})\mathbb E[|\widehat N-N|/N\mid\mathrm{under}]", hurdle)],
        ),
        "Exact accuracy decomposition": card(
            "Exact accuracy decomposition",
            [(r"P(\mathrm{exact})=P(\mathrm{parse})P(\mathrm{exact}\mid\mathrm{parse})", exact_decomp)],
        ),
        "Comparable L/N order reference": card(
            "Comparable L/N order reference",
            [(r"\operatorname{logit}p_m=a_m+r_m\ln(L/5000)+s_m\ln(N/5)+\mathrm{query\ order\ nuisance}", ref_power)],
        ),
    }


def inject_style(text: str) -> str:
    if STYLE_START in text:
        if STYLE_END in text:
            pattern = re.compile(
                re.escape(STYLE_START) + r".*?" + re.escape(STYLE_END),
                re.DOTALL,
            )
            return pattern.sub(lambda _: MATH_STYLE.strip(), text, count=1)
        pattern = re.compile(re.escape(STYLE_START) + r".*?(?=</style>)", re.DOTALL)
        return pattern.sub(lambda _: MATH_STYLE.strip() + "\n", text, count=1)
    if "</style>" in text:
        return text.replace("</style>", MATH_STYLE + "\n</style>", 1)
    return text


def replace_formula_cards(text: str, cards: dict[str, str]) -> str:
    legacy_card = re.compile(
        r'<div class="formula equation-card">\s*'
        r'<div class="equation-title">(?P<title>.*?)</div>'
        r'.*?<div class="equation-note">(?P<note>.*?)</div>\s*</div>',
        re.DOTALL,
    )

    def card_repl(match: re.Match[str]) -> str:
        title = re.sub(r"<.*?>", "", match.group("title")).strip()
        replacement = cards.get(title)
        if replacement is None:
            return match.group(0)
        note = match.group("note").strip()
        closing = replacement.rfind("</div>")
        return (
            replacement[:closing]
            + f'<div class="equation-note">{note}</div>\n'
            + replacement[closing:]
        )

    text = legacy_card.sub(card_repl, text)

    titled = re.compile(
        r'<div class="formula"><strong>(?P<title>.*?)</strong><br>.*?</div>',
        re.DOTALL,
    )

    def titled_repl(match: re.Match[str]) -> str:
        title = re.sub(r"<.*?>", "", match.group("title")).strip()
        return cards.get(title, match.group(0))

    text = titled.sub(titled_repl, text)
    text = re.sub(
        r'<div class="formula">\s*P\(exact\)\s*=\s*P\(parse\)\s*×\s*P\(exact\s*\|\s*parse\)\.\s*</div>',
        lambda _: cards["Exact accuracy decomposition"],
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r'<div class="formula">\s*logit p_m\s*=\s*a_m.*?query-order nuisance\.\s*</div>',
        lambda _: cards["Comparable L/N order reference"],
        text,
        flags=re.DOTALL,
    )

    # Upgrade the already-formatted Stage-2 card when the selected mechanism
    # law changes.  The following paragraph is a stable boundary emitted by
    # the Stage-2 generator.
    title_marker = '<div class="equation-title">Correlated-retrieval law</div>'
    next_marker = "<p>选择结果"
    title_at = text.find(title_marker)
    if title_at >= 0:
        outer_at = text.rfind(
            '<div class="formula equation-card math-equation">', 0, title_at
        )
        next_at = text.find(next_marker, title_at)
        if outer_at >= 0 and next_at > outer_at:
            text = (
                text[:outer_at]
                + cards["Correlated-retrieval law"]
                + "\n"
                + text[next_at:]
            )
    return text


def inject_summary_and_nav(text: str) -> str:
    if 'href="#counting-mechanism-stage2"' not in text:
        text = text.replace(
            '<a href="#laws">Empirical law</a>',
            '<a href="#laws">Empirical law</a>\n'
            '    <a href="#counting-mechanism-stage2">Counting mechanism</a>',
            1,
        )
    marker = "<!-- COUNTING_LAW_SUMMARY_V1 -->"
    if marker not in text:
        summary_start = text.find('<section id="summary">')
        list_start = text.find('<ul class="tight">', summary_start)
        if summary_start >= 0 and list_start >= 0:
            insertion_at = list_start + len('<ul class="tight">')
            summary = f"""
{marker}
    <li><strong>Counting mechanism：</strong>枚举模式的逐 needle 检索不符合独立复合
      <span class="inline-math"><i>q</i><sup><i>N</i></sup></span>。held-out 数据支持
      <span class="inline-math"><i>N</i><sub>eff,m</sub>(<i>N</i>)=κ<sub>m</sub><i>N</i><sup>τ</sup></span>，
      其中共享 <span class="inline-math">τ=0.747</span>（seed-cluster bootstrap 95% CI 0.641–0.850）；
      blocked-cell <span class="inline-math">R<sup>2</sup>=0.758</span>，而独立基线只有 0.326。</li>
    <li><strong>Bias：</strong>正负 bias 需要分成 over/under 概率与条件误差量级。
      该 hurdle 分解在 enumeration 与 CoT 上把 held-out cell
      <span class="inline-math">R<sup>2</sup></span> 提高到 0.311 与 0.288；
      direct 的 signed bias 不支持同一分解，因此不把它包装成统一定律。</li>"""
            text = text[:insertion_at] + summary + text[insertion_at:]
    return text


def update_html(
    path: Path,
    cards: dict[str, str],
    add_style: bool,
    main_report: bool = False,
) -> bool:
    if not path.exists():
        return False
    original = path.read_text(encoding="utf-8")
    revised = replace_formula_cards(original, cards)
    if add_style:
        revised = inject_style(revised)
    if main_report:
        revised = inject_summary_and_nav(revised)
    if revised == original:
        return False
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(revised, encoding="utf-8", newline="\n")
    tmp.replace(path)
    return True


def update_readme(path: Path) -> bool:
    if not path.exists():
        return False
    original = path.read_text(encoding="utf-8")
    if README_MARKER in original:
        return False
    addition = f"""

{README_MARKER}
## Formula typesetting

Displayed equations use native, offline MathML with a responsive scroll
container.  This changes presentation only; no fitted value or analysis table
is altered.  After rebuilding either analysis stage, rerun:

```powershell
python scripts/format_math_environment.py --report-root "<canonical report directory>"
```
"""
    path.write_text(original.rstrip() + addition + "\n", encoding="utf-8", newline="\n")
    return True


def refresh_checksums(base: Path, manifest: Path) -> None:
    records: list[str] = []
    for path in sorted(base.rglob("*"), key=lambda p: str(p.relative_to(base)).lower()):
        if not path.is_file() or path.resolve() == manifest.resolve():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rel = str(path.relative_to(base))
        records.append(f"{digest}\t{rel}")
    manifest.write_text("\n".join(records) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", type=Path, required=True)
    args = parser.parse_args()

    root = args.report_root.resolve()
    analysis = root / "analysis" / "counting_mechanism_law_v1"
    cards = make_cards()
    targets = [
        (root / "report.html", True, True),
        (analysis / "report.html", True, False),
        (analysis / "stage2_fragment.html", False, False),
    ]
    changed = [
        str(p)
        for p, with_style, is_main in targets
        if update_html(p, cards, with_style, is_main)
    ]

    update_readme(analysis / "README.md")
    refresh_checksums(analysis, analysis / "SHA256SUMS.tsv")
    refresh_checksums(root, root / "SHA256SUMS.tsv")

    print(f"Updated {len(changed)} HTML file(s).")
    for path in changed:
        print(path)


if __name__ == "__main__":
    main()
