# Main findings for counting mechanism under nonthinking & thinking modes
*Aug 6, 2026*

Thoughts and summaries are condensed from [Twist's report](realistic_niah_v4_4_non_thinking_integrated_mechanism_report.html) and [Tianyu's report](v26_v44_native_9000_parsing_geometry_TWO_PART_cn.html).

## Nonthinking mechanism

*This part should occupy 1.5--2 pages of a paper. Put details are in the appendix.*

Below is a high-level description of the mechanism.
1. (Step 1) Noisy running-index forms in lower layer: Models loosely maintain a count state in prompt needle spans
2. (Step 2) Broad retrieval collects needle signals in middle layers: retrieval heads attend to needle spans at answer token.
3. (Step 3) Count states consolidate in upper layers: causal effects of counting direction are progressively strengthened

### A visualization illustration

Here is a suggested main figure for this part.
- For step 1, use a PCA visualization to show the prompt-span running-index curve
- For step 2, use certain attention visualization to explain "broad retrieval" and the mechanism of retrieval head
- For step 3, use another PCA visualization to show the consolidated answer-token count-state

## Step 1: Noisy running-index

Some analysis to consider.

### Charaterizing the geometry of noisy running-index

- Rank analysis. Collect the hidden states at the needle spans. Do they lie in a relatively low dimensional space? Some sample questions:
  - How much variance does a $k$-dim subspace explain the variance of the hidden states?
  - Calculate the stable rank or numeric rank of the matrix formed by these hidden states?
- Degree of compression
  - Regression and goodness of fit: How much does the running-index curve explain the variance of the hidden states? Calculate $R^2$ of the linear probe or a nonlinear version of the $R^2$?
  - Classification: as a baseline for understanding, if we run a classifier (like nearest neighbors) using the hidden states, what is the accuracy of predicting the correct count?
  - Clustering analysis: what's the quality of clusters (treating each running index as a cluster).
- Noise analysis
  - What's the source of the "noise" in the hidden states? If we believe that the noise come from variability of needle tokens, then we may do some analysis of variance (ANOVA) with randomized needle token generation.
  - If we believe that the noise includes tokens outside the needle span, then we can check something like this: mannual removing the attention to outside-needle-span tokens should improve representations.
- Projections of all hidden states
  - Apply the same PCA projection to all hidden states including non-needle-span tokens. Do we get mostly noisy points?
  - An empirical formula for characterizing hidden states: denoting $N(s_{\le t})$ to be the needle count in the prefix $s_{\le t}$, and $\gamma$ be a curve in the hidden-state space, then does the following formula hold?
  $$
    h_t = \gamma(N(s_{\le t})) \cdot \mathbf{1}(t \in \text{needle span}) + \mathrm{noise}
  $$

### Why noisy running-index is formed

- Does the instruction cue in the beginning matter?
  - Compare instructions with or without cues before the passage. Does cue strengthen the running-index? If so, by how much?
- Does induction head explain?
  - One possibility is that needles contain repetitive strings, so induction heads are activated and attend to earlier needle spans, thus accumulating the count
  - For example, we can check the attention pattern from one needle span to earlier needle spans. I remember I have some run results that support this earlier-span attention claim

### Causal effects
- Needle token ablation / corruption
  - How does corrupting the needle-span tokens (replaced by haystack tokens) change later hidden states, and prediction counts? Measure the difference in accuracy, MAD, and other metrics.
- Hidden states ablation / corruption
  - I believe Twist's current results suggest that ablating hidden states has little causal effects to later representation and prediction.
  - I am a bit confused, because if we ablate the needle tokens directly (layer 0), the prediction count should decrease in general.
  - Perhaps a better claim is that the causal effect of running index is weak and only found in lower layers. This looks more reasonable because the needle signals are transferred to answer token in lower layers, so the middle/upper layers won't have an impact.
- Subspace ablation.
  - If we believe that the running index functions in a low-dim subspace, then how does removing the low-dim subspace component from the hidden states causally change later hidden states and prediction counts? This would test $\gamma(N(s_{\le t})) \subset \mathcal{V}$ for some subspace $\mathcal{V}$.
  - A good reference is Hao Yan's paper on task vector. Jiajun Song's work on induction head also has subspace ablation analysis.

## Step 2: Broad retrieval

- Analysis of attention pattern
  - The simplest is attention scores and attention head ablation.


### (Optional) delayed OV rewriting

This is the most complicated mechanism. I think the possible argument is that retrieval and OV rewriting are not in the same layer. I think we can mention with a few sentences in the main paper, and put details in the appendix?
- The causal effects of OV of certain heads in deciding the orthogonal answer-token count direction
- Evidence that residual streams perserve an identity map across multiple layers---this is why retrieved info is relayed to upper layers
- Some futher evidence (...Twist's report is somewhat too complicated to grasp...) in the appendix?

## Step 3: count state consolidates

### Charaterizing the geometry of noisy running-index
We can repeat every geometry analyis "Charaterizing the geometry of noisy running-index" in Step 1 for a sample of hidden states at the answer token

### Causal analysis

I think Twist already has some results that support a stronger causal effects. In addition,
- Of particular interest is the subspace causal links. If we believe that

  > running-index curve/subspace --> answer-token curve/subspace --> final count prediction.

  Then, removing the first would damage the latter two? Removing the second would damage the last? Basically, we can testing the map between needle-span curve to answer-token curve.

### Layer analysis
Of particular interest here is perhaps layer effects. Does the representation become more compressed as we progress to upper layers? Some metrics or visualization about layerwise change would be great.
