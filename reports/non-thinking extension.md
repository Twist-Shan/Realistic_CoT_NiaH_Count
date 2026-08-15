# Main findings for counting mechanism under non-thinking and thinking modes

*Updated Aug 15, 2026*

This note audits the questions in the original outline against the experiments currently available in the [integrated non-thinking report](NiaH_Non-thinking_report.html). “Answered” means that a concrete experiment directly addresses the question; “falsified” means that the precisely stated strong claim is contradicted by a matched intervention; “partially answered” means that the data constrain the answer but do not identify the full mechanism; “open” means that a useful decisive experiment has not been run; “closed” means that the present paper deliberately does not make the stronger claim and therefore will not spend more compute on it. The report’s 25-row audit preserves the original question numbers and moves questions 21 and 24 to the end, while distinguishing falsification from scope closure.

## Status summary

- **Answered:** the two hidden-state geometries; answer-query exact-count classification; token corruption; all-token endpoint/interior/ordinary controls; dense layerwise endpoint/full-span/ordinary restoration; restoration-induced broad-attention response; prompt rank-3 removal; broad maps and frozen head sets; all top-$K$ and correct-only ablations; retrieval-subspace mediation; answer-state patch/removal; adjacent-layer maps; local transport; and the present OV/residual evidence boundary.
- **Answered:** the integrated distributed-evidence→retrieval→late-state chain is now supported by a same-forward, 11-arm serial-mediation experiment in both models. This establishes partial serial mediation, not a unique or exhaustive route.
- **Falsified:** the strong classical-induction claim and the registered outside-context-source specificity claim both fail their matched causal criteria. The opening two-sentence counting-definition cue is also not necessary for an ordered shallow running geometry. These negative results leave the weaker earlier-span routing and distributed outside-context-support claims intact.
- **Partially answered:** why prompt readout appears early while answer executability appears late, and the detailed source of residual manifold scatter. The completed identity/context/position factorial constrains the latter but its held-out full models do not generalize above the intercept baseline.
- **Closed:** cross-final-count prefix invariance, because the paper does not claim an abstract final-$N$-invariant counter. Exhaustive per-token census is not required, and the fixed prompt-rank-3→answer-rank-3 identity path has separately been falsified.

## Non-thinking mechanism

The most defensible three-stage account is:

1. **Noisy prompt-side record forms early.** At each needle end, the hidden state varies systematically with the occurrence index, but individual samples do not form clean count clusters. This is a representation result, not evidence for a single necessary integer register.
2. **Broad retrieval occurs at the answer query.** In middle-to-late layers, a bank of heads allocates attention mass across multiple active needle spans. Ablating the frozen top-ranked bank changes the output more than layer-matched random-head ablation.
3. **The answer-side count state becomes executable late.** Full answer-state patching transfers the donor answer, late count-subspace removal damages behavior, and a count-aligned perturbation is selectively relayed across the next block.

The key distinction is therefore:

> Prompt needle ends contain a noisy, decodable running-index geometry; the answer query later contains a consolidated state that is much more causally connected to the generated number.

### Suggested main visualization

- **Stage 1:** frozen-PCA 3D prompt needle-end running-index trajectory, with selectable layer.
- **Stage 2:** answer-query layer-by-head broad-score attention map, with the frozen Qwen top-32 and Gemma top-8 marked.
- **Stage 3:** frozen-PCA 3D answer-query count geometry together with the layerwise answer patch/removal curves.

The behavior heatmaps by gold needle count should precede this mechanism figure, with Qwen and Gemma shown separately.

## Step 1: Noisy running-index

### Characterizing the geometry

#### Do needle-end states lie in a low-dimensional space?

**Answered, with an important qualification.** The count centroids are close to a three-dimensional trajectory, but the full sample cloud is substantially noisier.

| Model and site | Ridge $R^2$ | Ridge MAD | rank-3 capture, all states | rank-3 capture, count centroids | stable rank | $\eta^2_{\rm count}$ | cosine silhouette |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-8B needle end, L8 | 0.945 | 0.561 | 0.690 | 0.988 | 1.678 | 0.603 | -0.075 |
| Gemma4-E4B needle end, L9 | 0.719 | 1.249 | 0.598 | 0.979 | 2.443 | 0.383 | -0.043 |

Here, “rank-3 capture” is the fraction of centered squared norm retained by the first three frozen PCA axes. The centroid values, $0.988$ and $0.979$, show that the *mean count trajectory* is nearly three-dimensional. The lower all-state values, $0.690$ and $0.598$, show that a large amount of within-count variation remains outside those axes. Stable rank measures effective dimensionality of the centered state matrix. $\eta^2_{\rm count}$ is the fraction of total variance attributable to the count label. Cosine silhouette compares within-count and between-count cosine distances; values near zero or negative mean that individual samples do not form cleanly separated clusters.

Example: rank-3 capture $0.69$ means that the first three PCs retain 69% of the centered state energy, not that 69% of examples can be classified correctly.

#### How well can the running index be predicted?

**Answered by regression, not by an exact-count classifier.** At the representative layers, seed-held-out ridge regression gives Qwen $R^2=0.945$, MAD $=0.561$ running-index units, and Gemma $R^2=0.719$, MAD $=1.249$. The current experiment package does not contain a valid needle-end exact classifier, and the report intentionally does not present one. The prompt question is whether an ordered running-index trajectory exists, whereas exact final-count classification is evaluated at the answer query.

For regression MAD,

$$
\operatorname{MAD}_{\rm ridge}=\frac{1}{n}\sum_i |\hat n_i-n_i|.
$$

Example: true running indices $[2,5]$ and predictions $[3,4.5]$ give MAD $(1+0.5)/2=0.75$.

#### Does the empirical formula $h_t=\gamma(N(s_{\leq t}))\mathbf 1(t\in\text{needle span})+\text{noise}$ hold?

**Answered for the strong all-token formula, with a restricted positive result at endpoints.** At selected *needle-end positions*, the centroids define a low-dimensional ordered curve $\gamma(n)$, while poor silhouettes and incomplete all-state rank-3 capture quantify residual variation. Frozen all-token controls compare endpoint-gated, span-gated, and ungated prefix curves against category-only baselines: endpoint gating adds $\Delta R^2=+0.551$ for Qwen and $+0.326$ for Gemma, whereas needle-interior span gating gives $-0.057$ and $-0.036$. The evidence therefore supports the restricted descriptive statement

$$
h_{s,n,\ell}^{\rm needle\ end}=\gamma_\ell(n)+\epsilon_{s,n,\ell},
$$

not the stronger indicator formula over every token in a needle span. Ordinary and hard-negative controls do not show the same useful trajectory. This is sufficient to reject “one common $\gamma(n)$ is present at every span token”; it is not an exhaustive census proving that every non-needle token contains zero count information.

#### What is the source of the noise?

**Partially answered; no further decomposition is required for the present paper.** At the representative layers, count/seed-context/interaction account for $0.599/0.161/0.241$ of Qwen variance and $0.385/0.228/0.386$ of Gemma variance. Running index and absolute endpoint position are highly collinear ($\rho\approx0.965$); after grouped cubic absolute-position control, the frozen three-PC held-out $R^2$ falls to $0.043$ for Qwen and $0.004$ for Gemma. Earlier-span attention prefers active needles over matched ordinary spans, but a needle-only mask that removes outside context worsens both models, so outside-span context is not pure removable noise. Absolute position explains most of the displayed three-dimensional ordering; seed/context and their interaction explain substantial scatter. A randomized identity/context ANOVA could further separate the residual sources, but it would not change the current position-confounded, span-level mechanism claim.

### Why does the noisy running-index form in shallow layers?

**Partially answered by architecture and layerwise timing, but not by a dedicated intervention.** A needle-end token is processed immediately after the local needle phrase and can summarize the prefix count early. It therefore only needs a local, lossy trace useful for later retrieval. By contrast, the answer query occurs after the full long prompt and must collect evidence from many distant spans, combine it, and align it with the output vocabulary; this requires several retrieval and write stages, so its consolidated and causally executable count state appears later.

This also explains why the within-prompt running-index direction can be nearly orthogonal to the answer-query count direction: the two states occur at different token positions, are fitted in different local bases, and serve different computations. They encode the same latent variable without being the same vector or register.

#### Is the opening counting-definition cue necessary?

**The strong necessity claim is falsified, with a narrow intervention boundary.** V4.4.2 deletes exactly the two opening definition sentences but retains the passage, slots, counting question, numeric-output instruction, and assistant formatting. It uses seeds 1234–1243. For prompt running geometry, each model contributes ten final-$N=10$ prompts and 100 paired endpoint states.

At the preselected shallow sites, the fixed-$\alpha=1$ ridge is evaluated leave-one-seed-out in a pooled cue-present/cue-absent shared six-PC basis. Qwen L8 has centroid-topology CKA $0.9995$, ridge $R^2$ $0.845\to0.840$, and full-state count $\eta^2$ $0.645\to0.633$; Gemma L9 has CKA $0.9999$, ridge $R^2$ $0.343\to0.355$, and count $\eta^2$ $0.440\to0.433$. Thus an ordered, readable running geometry survives removal of the opening definitions. However, the paired cue displacement remains count-dependent: full-state interaction $\eta^2$ is $0.484$ for Qwen and $0.332$ for Gemma. The cue therefore modulates the full state even though it is not necessary for the ordering.

The correct conclusion is:

> The two opening counting-definition sentences are not necessary for the shallow running-index geometry. This experiment does not show that all task instructions are unnecessary, because the counting query and required numeric output remain present.

The exact CKA definition, layerwise curve, representative-layer table, and figure-axis descriptions are in Appendix A of the integrated report.

#### Does an induction head explain the running-index geometry?

**Open.** Earlier-span attention is compatible with an induction-like mechanism but is not sufficient evidence. A direct answer requires identifying candidate heads, testing the expected previous-occurrence attention pattern, and performing head/path ablation against layer-matched controls. The current report does not establish this.

### Causal effects at the prompt

#### What happens if the active needle tokens are corrupted?

**Answered: the active evidence is strongly causal.** Every active-needle span is replaced with an equal-length ordinary haystack span. The matched control replaces the same number of nearby ordinary spans while preserving sequence length, token budget, and answer-query position.

Let $\hat y_0$, $\hat y_N$, and $\hat y_C$ be the clean, needle-corrupted, and ordinary-control outputs, with gold count $N$. The reported absolute-error specificity is

$$
S_{\rm token}=\bigl(|\hat y_N-N|-|\hat y_0-N|\bigr)
-\bigl(|\hat y_C-N|-|\hat y_0-N|\bigr)
=|\hat y_N-N|-|\hat y_C-N|.
$$

Example: $N=8$, needle corruption outputs 1 and control outputs 7. Then $S_{\rm token}=7-1=6$ counts.

The measured specificities are Qwen $+8.930$ counts and Gemma $+8.780$; correct-to-wrong damage specificity is $+0.450$ and $+0.360$. Ordinary-control error increases are approximately zero (Qwen $-0.010$, Gemma $+0.040$). Thus the raw needle evidence is indispensable.

#### What happens if the decoded prompt rank-3 component is removed?

**Answered: the tested linear endpoint subspace is not a strong localized causal register.** At every active needle end, the discovery-fitted centered rank-3 count projection is removed. The control removes an orthogonal within-count-residual rank-3 component at the same positions and is scaled to the same *realized* Frobenius norm in each prompt.

The absolute-error specificity is

$$
S_{\rm prompt\ rank3}=|\hat y_{\rm rank3-remove}-N|
-|\hat y_{\rm orth-remove}-N|.
$$

Across tested layers, Qwen ranges from $-0.022$ to $+0.056$ counts and Gemma from $-0.011$ to $+0.022$. The separate endpoint results are Qwen $+0.056$ and Gemma $-0.022$. These are tiny compared with token corruption.

This resolves the apparent contradiction. Removing the *input evidence* at layer 0 destroys the facts to be counted. Removing one fitted linear component from the endpoint states does not erase all usable evidence because the information may be distributed across tokens, nonlinear, redundant across positions, or retrieved again from the original span. Consistently, full prompt-span donor-to-receiver patching reaches the donor gold count on 81.5% of Qwen and 91.9% of Gemma eligible correct-only cases: distributed prompt states are sufficient even though one linear endpoint subspace is not locally necessary.

The appropriate claim is therefore not “the prompt counter has no causal effect.” It is:

> Prompt needle ends contain a noisy counter-like geometry, and the underlying needle evidence is strongly causal, but the currently identified endpoint rank-3 direction has little direction-specific localized necessity.

### Which prompt positions and depths remain causally reusable?

**Answered by the canonical dense span-restoration sweep.** For seeds 1234–1263 and counts 1–10, the experiment first stores clean hidden states, corrupts every active needle span with equal-length ordinary text, and then restores clean states only once at one post-block layer. It scans endpoint-only, full-span, and token-budget-matched ordinary restoration at every layer: Qwen L0–L35 and Gemma L0–L41. This produces 33,300 Qwen rows and 38,700 Gemma rows.

With $E[c]$ denoting softmax expected count, the main score is

$$
S_{\rm restore}(\ell)=
\left(|E_{N\text{-corrupt}}-N|-|E_{N\text{-restored},\ell}-N|\right)
-
\left(|E_{O\text{-corrupt}}-N|-|E_{O\text{-restored},\ell}-N|\right).
$$

Example: for gold $N=8$, needle corruption has $E[c]=3$ (error 5), full-span restoration at L8 has $E[c]=6$ (error 2), and matched ordinary restoration repairs only $0.2$ count. Then $S_{\rm restore}=3-0.2=2.8$ counts. Positive values mean that restoring true needle evidence is more useful than restoring an equally large ordinary hidden region.

Endpoint-minus-ordinary stays near zero. Full-span restoration is positively detectable through approximately Qwen L20 and Gemma L16; Qwen declines over roughly L15–L22, whereas Gemma has a sharp L16→L17 cliff. Thus the reusable source is the complete needle span rather than a single endpoint register.

The canonical routing diagnostic then re-measures the final frozen answer-query head bank after each one-time restoration. For head $h$, let $M_h$ be total attention mass on active needle spans and $B_h=M_hC_h$ the broad score. First subtract the appropriate corrupt baseline within each source type, then subtract the equal-token ordinary-restoration response:

$$
\Delta M(\ell)=\frac{1}{|\mathcal H|}\sum_h\mathbb E[
(M_h^{N\text{-restored}(\ell)}-M_h^{N\text{-corrupt}})
-(M_h^{O\text{-restored}(\ell)}-M_h^{O\text{-corrupt}})],
$$

with $\Delta B(\ell)$ defined identically after replacing $M$ by $B$. The plot uses all canonical seeds 1234–1263 and counts 1–10, averaging 300 prompts per layer/head. Qwen uses the final frozen top-32 registry over L0–L35, and Gemma the final top-8 over L0–L41; neither heads nor layers are reselected from the response curve. Qwen $\Delta B$ at L0/L16/L20/L21/L24/L26 is $0.310/0.302/0.196/0.061/0.023/0.010$ and becomes zero at L27; Gemma at L0/L16/L17/L20/L22/L23 is $0.208/0.327/0.134/0.102/0.102/0$. These are attention-derived units, not output counts.

> 在较早层恢复完整 needle-span evidence，会重新配置后续 answer-query 的 broad retrieval；这一影响在 Qwen 约持续到 L20、Gemma 约持续到 L16。这里的“持续”专指同时带来行为修复的主要窗口；attention-only 的较弱尾部分别延至 L26 与 L22。它定位的是“prompt evidence 在深度上何时仍可被后续 retrieval 使用”，而不是 retrieval head 直接读取了哪个历史层。

A transformer head reads token states at its own input depth. Layerwise restoration changes how repaired prompt states evolve through subsequent blocks and whether they can still affect that later head; it does not give the head a pointer to a past layer.

## Step 2: Broad retrieval

### How is a broad retrieval head defined?

**Answered.** For active needle span $S_i$ and answer query $q$,

$$
m_{i,h}=\sum_{j\in S_i}\alpha_h(q,j),\qquad
M_h=\sum_i m_{i,h},\qquad
p_{i,h}=\frac{m_{i,h}}{M_h},
$$

$$
C_h=\frac{\exp\!\left(-\sum_i p_{i,h}\log p_{i,h}\right)}{N},
\qquad B_h=M_hC_h.
$$

$M_h$ is total attention mass on active needles, $C_h$ measures how broadly that mass covers the $N$ needles, and $B_h$ is the broad-retrieval score. Example: with four needles, masses $[0.1,0.1,0.1,0.1]$ give $M=0.4$, $C=1$, $B=0.4$; mass $[0.4,0,0,0]$ gives the same $M$ but $C=1/4$, so $B=0.1$.

The report includes layer-by-head attention maps. The horizontal axis is zero-based head index, the vertical axis is zero-based layer, each cell is discovery-mean $B_h$, and the marked cells are the frozen causal-ablation sets—not a raw token-by-token attention matrix.

### Which heads are in the frozen sets?

**Answered.** The full memberships are:

- **Qwen top-32:** L27H18, L28H19, L23H29, L23H13, L23H28, L24H16, L23H12, L26H26, L22H5, L21H11, L24H13, L23H31, L21H25, L21H23, L21H16, L21H18, L28H16, L23H30, L26H20, L21H27, L28H17, L9H19, L21H19, L26H21, L27H16, L34H28, L24H14, L23H10, L24H31, L28H18, L21H31, L24H29.
- **Gemma top-8:** L29H4, L35H2, L35H7, L35H1, L35H3, L29H2, L41H1, L29H0.

### Does ablating these heads have a causal effect?

**Answered, but the effect is distributed and non-monotone in $K$.** Candidate ablation removes the frozen top-$K$ heads simultaneously at the answer query. The control removes $K$ random heads with the same layer-count distribution. If $\hat y_K$, $\hat y_R$, and $\hat y_0$ are ranked, random-control, and clean outputs, the excess absolute count shift is

$$
\Delta_{\rm shift}(K)=
|\hat y_K-\hat y_0|-|\hat y_R-\hat y_0|.
$$

This measures movement away from the clean output, not error relative to gold. Example: clean output 8, ranked ablation 6, matched random ablation 7 gives $\Delta_{\rm shift}=2-1=1$ count.

| $K$ | Qwen $\Delta_{\rm shift}$ | Gemma $\Delta_{\rm shift}$ | Qwen clean-correct $\Delta$ wrong | Gemma clean-correct $\Delta$ wrong |
|---:|---:|---:|---:|---:|
| 1 | +0.020 | +0.257 | +0.022 | +0.128 |
| 2 | +0.040 | +0.170 | +0.035 | +0.031 |
| 4 | +0.083 | +0.087 | +0.073 | +0.046 |
| 8 | +0.040 | +0.767 | +0.030 | +0.199 |
| 16 | +0.520 | +0.287 | +0.317 | +0.203 |
| 32 | +1.623 | +0.243 | +0.587 | +0.128 |

The main effect sizes are Qwen top-32 $+1.623$ counts and Gemma top-8 $+0.767$. A one-count shift is meaningful because the candidates are adjacent integers and the clean models already have substantial exact accuracy; nevertheless, a shift of one is not “complete mechanistic explanation.” Small effects such as Gemma $K=4$, $+0.087$, should be described as detectable but small. Non-monotonicity means these heads are not independent scalar counters: adding heads changes redundancy, compensation, and the matched random-control set.

### Does broad retrieval actually send donor-count information downstream?

**Answered.** Donor source/pre-O states are patched into receiver runs and the induced downstream component is then blocked against an equal-norm orthogonal control. Qwen early-source patching produces $+0.1057$ donor-vs-receiver log-odds gain, with $+0.1709$ downstream mediation at L28 H16–H19. Gemma source donor transport is $+0.0889$, with $+0.0864$ exact residual mediation at L37. This supports a source-to-mediator path rather than relying on attention correlation alone.

## Optional: OV writing

### What does OV add beyond the attention map?

**Answered conceptually and causally for Qwen.** Attention weights say *where the head reads*. The V projection forms the retrieved content $z_h$, and $W_O^hz_h$ is the vector actually written into the shared answer residual. OV experiments ask whether this written vector naturally carries count information, whether signed injection moves the answer, whether matched removal hurts, and whether it mediates an upstream patch effect.

For Qwen L28 core set $\{\mathrm{H16,H19}\}$:

- natural carrier slope is $0.2174$ hidden-coordinate units per gold count;
- one intervention dose $\beta$ changes softmax expected count by $0.0640$ on average;
- natural-axis removal increases expected-count absolute error by $0.0732$ relative to an equal-norm orthogonal removal and decreases the correct-answer log-prob margin by $0.2646$ more;
- the frozen natural axis mediates 18.2% of the tested donor-z transport;
- full/routing/value normalized transports are $0.1140/0.0517/0.0524$, so both “where to read” and “what value is retrieved” contribute;
- H19 leave-one-out decrement is $0.1538$, showing non-redundancy within the set, not a single-head counter.

For Gemma, **the localized OV claim is not established**. The preregistered L35H2 and $\{\mathrm{L29H4,L35H2}\}$ sets do not pass the complete natural-carrier/injection/removal package. What is established is a distributed residual path: source transport $0.0889$, L37 exact residual mediation $0.0864$, frozen count-axis mediation $0.0458$, and L41 terminal count-adoption coefficient $0.2256$. The last number is a normalized residual projection, not 22.56% exact accuracy.

### Does the residual stream preserve an identity map across layers?

**No; the data support selective local transport, not a strict identity map.** For adjacent answer-query layers, local rank-3 centroid maps are predictable and bootstrap-stable late in the network. At the causal test boundaries:

- Qwen L28$\to$29 has CV normalized RMSE $0.1175$, bootstrap map error $0.0164$, and full-operator cosine to the next boundary $0.7674$.
- Gemma L36$\to$37 has CV normalized RMSE $0.0271$, bootstrap map error $0.0074$, and cosine $0.7956$.

Cosine below 1 means that consecutive ambient low-rank operators are directionally similar but not identical. More directly, a count-aligned one-dose perturbation propagates across one block with target-chord coefficient $0.9486$ for Qwen and $0.9779$ for Gemma, whereas equal-realized-norm orthogonal controls give $0.0069$ and $0.0020$. Thus the next block selectively accepts a count-aligned change; it need not copy the entire residual through an identity map.

## Step 3: The answer count state consolidates

### Answer-query geometry

**Answered.** Unlike the prompt needle end, the answer query has valid seed-held-out exact-count classifiers because every state corresponds to one final gold count.

| Model and site | L2-logistic accuracy / MAD | nearest-centroid accuracy / MAD | ridge $R^2$ / MAD | rank-3 all states | rank-3 centroids | stable rank | $\eta^2_{\rm count}$ | silhouette |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen answer query, L29 | 56.0% / 0.635 | 54.5% / 0.640 | 0.891 / 0.662 | 0.729 | 0.947 | 2.375 | 0.686 | 0.056 |
| Gemma answer query, L37 | 53.0% / 0.720 | 55.0% / 0.615 | 0.885 / 0.720 | 0.799 | 0.981 | 2.135 | 0.744 | -0.038 |

Ten-class chance accuracy is 10%. Nearest centroid assigns a test state to the count whose discovery centroid has the smallest distance. Classifier MAD is the mean absolute difference between the predicted and gold class; example: predictions $[4,8]$ for gold $[5,6]$ give MAD $(1+2)/2=1.5$ counts.

The high classifier and ridge scores show that count is readable in the full hidden space. The still-low silhouettes show why a two- or three-dimensional plot may have overlapping points: a low-dimensional *centroid trajectory* is not the same as ten clean sample clusters.

### Does the answer representation become stronger with depth?

**Answered.** The layerwise curves show different timing at the two sites. Prompt needle-end ridge decoding is already strong in shallow layers and later declines, while answer-query exact classification rises sharply in middle-to-late layers. The selected peak L2-logistic results are Qwen L29, 56.0% with MAD 0.635, and Gemma L39, 56.5% with MAD 0.655. This timing matches the proposed computation: prompt states record local occurrences early; the answer query must retrieve and consolidate globally before becoming predictive and executable.

### Is the late answer state causally connected to the output?

**Answered.** Full donor answer-query state patching at the same layer gives strict donor-gold hits of 96.6% for Qwen and 96.0% for Gemma on correct-only eligible pairs. Dense layer sweeps are near zero early and rise in middle-to-late layers, establishing when the state becomes sufficient to drive the donor answer.

Answer-query rank-3 removal also establishes direction-specific necessity. With an equal-realized-norm orthogonal removal control,

$$
S_{\rm answer\ rank3}(\ell)=
|\hat y_{\rm count-remove,\ell}-N|
-|\hat y_{\rm orth-remove,\ell}-N|.
$$

Peak absolute-error specificity is Qwen L28 $+0.878$ counts and Gemma L32 $+1.222$; late values remain large (Qwen L24 $+0.733$, L32 $+0.789$; Gemma L36 $+1.189$, L41 $+1.133$). This contrasts sharply with the near-zero prompt endpoint removal.

### Is the full causal chain “prompt subspace $\to$ answer subspace $\to$ prediction” established?

**Established as a partial stage-wise distributed chain, not as transport of one fixed three-dimensional axis.** Dense full-span restoration shows that early distributed prompt evidence remains reusable; the restoration-induced attention response shows that repairing it reconfigures later broad retrieval; frozen retrieval-subspace interventions establish a local natural aggregation window; late answer-state patch/removal and Qwen OV / Gemma residual mediation connect the downstream state to output. Experiment 19 additionally combines source restoration, retrieval-coordinate blocking, and late-coordinate blocking within the same seed-count forward. The ordered criteria pass for both models, so the arrows form a supported partial serial-mediation chain. Bypass and redundant routes remain, and the experiment does not establish uniqueness.

The current causal graph should therefore be written as

$$
\text{distributed prompt evidence}
\longrightarrow \text{broad retrieval and OV/residual write}
\longrightarrow \text{late answer count state}
\longrightarrow \text{count output},
$$

not as a proven transport of one fixed three-dimensional axis from needle ends to the answer query.

The fixed endpoint-rank-3 $\to$ answer-rank-3 direct-transport claim is specifically disfavored: endpoint rank-3 removal is near zero, while whole-span restoration is large and each later stage can use a different fitted basis. That strong identity-path claim is not the missing link.

#### Completed experiment 19: same-forward nested serial mediation

Use the canonical confirmation panel (seeds 1254–1263 × counts 1–10, 100 paired units per model) and freeze the already selected stages: Qwen source/retrieval/late layers L8/L23/L29 and Gemma L9/L29/L37. In one corrupt forward compare:

1. ordinary-span restoration control $O$;
2. true full-needle-span restoration $S$;
3. $S+R_\perp$ versus $S+R_\parallel$, equal-realized-norm orthogonal versus count-aligned block at the retrieval stage;
4. $S+T_\perp$ versus $S+T_\parallel$ at the late answer stage;
5. the full joint $2\times2$ factorial $S+R_a+T_b$ for $a,b\in\{\perp,\parallel\}$: $R_\perp T_\perp$, $R_\parallel T_\perp$, $R_\perp T_\parallel$, and $R_\parallel T_\parallel$.

For expected-count error $e(X)=|E[c]_X-N|$, define source repair and the two direction-specific mediation scores as

$$
G_S=e(O)-e(S),\qquad
M_R=e(S+R_\parallel)-e(S+R_\perp),\qquad
M_T=e(S+T_\parallel)-e(S+T_\perp).
$$

Example: if $N=8$, $O$ has $E[c]=3$ (error 5), $S$ has $E[c]=6$ (error 2), then $G_S=3$ counts. If $S+R_\perp$ has error 2.2 and $S+R_\parallel$ error 3.2, then $M_R=1$ count. In addition to behavior, record broad attention/broad score, the frozen retrieval coordinate, and the late count coordinate. The ordered criterion is: $S$ changes retrieval; $R_\parallel$ selectively reduces the downstream late coordinate and repair; $T_\parallel$ selectively damages output without changing the earlier retrieval readout.

The joint interaction is

$$
I_{RT}=[e(S+R_\parallel+T_\parallel)-e(S+R_\perp+T_\parallel)]
-[e(S+R_\parallel+T_\perp)-e(S+R_\perp+T_\perp)].
$$

Negative $I_{RT}$ means that the late block occludes part of the retrieval-block damage, values near zero indicate approximate additivity, and positive values indicate synergy. Example: if the aligned retrieval block adds 1.0 count error under $T_\perp$ but only 0.3 under $T_\parallel$, then $I_{RT}=0.3-1.0=-0.7$. This quantifies overlap and does not prove a unique pathway. The full design has 11 arm conditions, or 1,100 condition-forwards per model and 2,200 across both models.

**Result.** All 1,100 registered arms per model and the 10,000 seed-level bootstrap analysis pass audit. Qwen source repair is $2.674$ counts (95% CI $[2.358,2.991]$), retrieval mediation $0.327$ $[0.243,0.417]$, late mediation $1.118$ $[0.863,1.383]$, joint interaction $-0.382$ $[-0.513,-0.257]$, and remaining repair $1.477$ $[1.026,1.904]$. Gemma gives $2.670$ $[2.312,3.038]$, $0.521$ $[0.415,0.626]$, $1.215$ $[0.993,1.435]$, $-0.380$ $[-0.490,-0.273]$, and $1.291$ $[0.910,1.650]$, respectively. The negative interactions show partial occlusion between retrieval- and late-stage damage, while the positive remaining repair shows that the tested coordinates do not exhaust the source-restoration effect. Thus the experiment supports **same-forward partial serial mediation**, not a unique channel.

#### Completed experiment 22: classical induction-head identification

**Optional and low priority.** Existing earlier-span preference does not test the defining relation of a classical induction head. Use two strictly separated stages. First, rank candidate heads only from the canonical NIAH discovery seeds 1234–1253 using the already defined endpoint-to-earlier-span preference; do not use confirmation outcomes for head selection. Second, test those same frozen heads on an independent synthetic induction assay built from a model-specific pool of stable single-token anchors and successors. Generate 30 fixed base sequences per model and four fully token/position-matched conditions per sequence (120 forwards/model): `repeated_consistent`, `unique_anchor`, `successor_reassignment`, and `same_position_ordinary_repeat`. In `successor_reassignment`, keep two earlier successor tokens at the same positions and swap only the equal-length anchor identities immediately before them, so the successor selected by the previous-match relation moves without moving either candidate successor position.

For the current-anchor query $q_t$ in the synthetic sequence, define

$$
I_h=\mathbb E\,\alpha_h(q_t,\operatorname{succ}(\operatorname{prevmatch}_t))
-\mathbb E\,\alpha_h(q_t,\text{matched non-successor}).
$$

Example: attention mass $0.20$ on the identity-defined successor and $0.05$ on its matched control gives $I_h=0.15$. Under successor reassignment, a relation-following head should move its mass from successor 1 to successor 2 even though both successor positions are fixed; a positional head should remain at the old position. Only after this synthetic test, return to canonical confirmation seeds 1254–1263 × counts 1–10 and run three arms—natural, candidate-path block, and matched-control block—for 300 condition rows/model. On discovery data, first freeze the repeated record-template anchor, its query offset, and the previous-occurrence successor key offset for each retained head. The candidate arm then removes the frozen natural $\alpha V$ contribution for those current-anchor-query→previous-successor edges inside each needle; it does not recompute a fully renormalized QK counterfactual and does not assume that the needle endpoint itself is the induction query. Compare with the same number of non-successor edges matched by layer, head, key-distance bin, and pre-intervention mass, and read out the downstream endpoint update, later retrieval, and final count.

**Result.** Both synthetic gates retain an induction-like candidate (Qwen L5H13; Gemma L5H0), but the canonical matched causal criterion fails. On relation-present counts 2–10, candidate-minus-control expected-error effects are Qwen $-0.0219$ (95% CI $[-0.0331,-0.0108]$) and Gemma $-0.0121$ $[-0.0250,0.0013]$: blocking the registered candidate edges does not damage counting more than the matched controls in the preregistered direction. The strong “classical induction head forms the running index” claim is therefore **not supported**. The appropriate retained label is *earlier-span routing*; the result does not say that induction-like pattern matching never occurs elsewhere in the model.

#### Completed experiment 23: identity/context/position decomposition and outside-context specificity

**Useful only if the paper needs to explain the source of manifold scatter.** The existing V4.1–V4.4 panels, whose frozen design is documented in [`docs/realistic_niah_v4.md`](../docs/realistic_niah_v4.md), already perform a sequential robustness ladder: V4.1 fixes position/order/content; V4.2 varies position; V4.3 additionally varies the order of a fixed fact set; V4.4 additionally varies city-score content. This is valuable evidence that the geometry is not tied to one frozen prompt, but it is not a fully crossed design—each later panel releases another factor—so it cannot identify independent identity/context/position contributions or their interactions. To obtain that decomposition, at the frozen Qwen L8 and Gemma L9 sites run an identity $\times$ context $\times$ position $2\times2\times2$ paired factorial. Identity randomizes city/score surface forms from a tokenizer-length-matched pool; context permutes surrounding ordinary blocks within matched length/depth bins; position exchanges each record with an exact-token-length ordinary carrier at frozen gap-jittered slots while preserving record order, prompt length, and answer-query position. Use 30 seeds, all eight cells, and final $N=10$: 240 prompt-forwards and 2,400 endpoint states per model, with the original 20/10 discovery/confirmation split.

After subtracting the discovery count centroid in the frozen three-PC basis, fit the full factorial with seed-held-out prediction. For factor $F$,

$$
\Delta R_F^2=R^2(\text{full})-R^2(\text{full without every term containing }F).
$$

Example: if held-out $R^2$ falls from $0.60$ to $0.20$ after deleting position and its interactions, then $\Delta R_P^2=0.40$. This identifies the effect of the controlled manipulation; it is not a literal claim that 40% of natural neural variance is “caused by position.”

Finally, localize the existing coarse outside-context mask result. Freeze outside-context edges on discovery data and compare, on confirmation seeds 1254–1263 × counts 1–10, four arms: no block, candidate outside-edge block, layer/head/key-distance/edge-count-matched random outside-edge block, and pre-intervention-attention-mass-matched ordinary-edge block. This is 400 condition rows per model. A candidate-specific effect on frozen geometry, later broad retrieval, and count behavior would show that particular outside context is used; the existing result that deleting *all* outside context hurts is not sufficient for that claim.

**Result.** The exact $2\times2\times2$ factorial, 2,400 endpoint states, and 400-row outside-context panel pass audit for each model. The discovery-fitted rank-3 centroid bases retain $0.988$ of Qwen and $0.979$ of Gemma centroid energy, but the held-out full factorial $R^2$ values are negative ($-0.022$ and $-0.089$). Corresponding incremental $\Delta R^2$ values are small: Qwen identity/context/position $=-0.0001/-0.0004/+0.0175$; Gemma $=+0.0031/+0.0002/-0.0007$. These are controlled deformations in a non-generalizing predictive model, not natural variance shares. The registered outside-context specificity gate also fails both controls in both models: Qwen candidate-minus-distance expected error is $+0.0123$ (95% CI $[-0.0054,0.0265]$) and candidate-minus-attention $+0.0031$ $[-0.0100,0.0163]$; Gemma gives $-0.0022$ $[-0.0099,0.0056]$ and $+0.0039$ $[-0.0058,0.0129]$. Therefore the experiment does not identify a privileged outside-context edge set. It remains compatible with distributed contextual synergy, other edge registries, or effects that require a fully renormalized QK intervention.

## Paper-level conclusion

The evidence supports a representational-to-causal transition rather than a single counter being carried unchanged through the network. Needle ends form an early, low-dimensional but noisy and position-confounded running-index trajectory. Raw needle evidence is essential and causally reusable as a complete span, whereas the fitted endpoint rank-3 direction is not a strong localized register. Repairing early spans changes later broad retrieval; local retrieval-subspace interventions identify a partial aggregation pathway; late answer-query states become sufficient and direction-specifically necessary for the generated count. Same-forward serial mediation now links those stages directly in both models, while leaving measurable bypass/redundancy. Qwen additionally supports a localized L28 OV writer, whereas Gemma supports a more distributed late residual mediator. The classical-induction micro-circuit and registered outside-context-source specificity claims fail their matched causal criteria, so neither is part of the main mechanism claim.

## Remaining optional follow-up

The preregistered purposes, matched-pair construction, interventions, controls, readouts, decision rules, execution order, and audited outcomes for experiments 19, 22, and 23 are recorded in [`plans/nonthinking-followup-experiment-log-20260813.md`](../plans/nonthinking-followup-experiment-log-20260813.md). The central frozen V4.4.5 follow-up queue is complete. Remaining work is optional scope extension rather than a missing link in the stated mechanism: a fully renormalized QK/V intervention could refine the edge-level account, and external model/task replication could test generality. Question 21's opening-definition necessity claim and the strong registered claims in questions 22 and 23 are falsified; question 24 is closed as an out-of-scope abstract-counter claim.
