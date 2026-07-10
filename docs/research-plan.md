# Research plan

## Aims

The project should focus on one of the following aims.
1. Representation of **counting feature** (how do LMs count?)
   - Representation of **numerical feature**
2. How do LMs encode and **compress** context?
3. Architecture comparsion 
   - Transformer, RNN, LSTM, Mamba
   - Memory vs compute cost
   - ID vs OOD loss, and other generalization properties
   - Bayesian inference
   - Variance of hidden states, etc
  
### Counting features

- sequence s = (s_1, s_2, s_3, ... s_T)
- well-defined targets: parenthsis, matching needle, character, etc..
- count: y_t = Y(s_{\le t}) \in {0, 1, 2, 3, ...} for any t
- hidden state: given a layer, the hidden state h_t \in R^d
- Q: how does h_t encode y_t?
- h_t = \gamma_t + \nu_t + noise_t
  - where \gamma_t = \gamma_t(y_t) \in R^d is a curve (special case is a line)
  - \nu_t is the representation of the current token (or neighboing token)
- Connection to our paper (latent-variable setting)
  - h_t = \sum_k \beta_{k,t} \theta_k + \nu_t + noise_t
- Measurement:
  - Linearity of \gamma_t? 
  - The variance of the noise_t, how does it depend on t, task, ...
- Connection to compression
  - Maximum compression: noise_t small, or not depend on s_{<t}
  - Partial compression: at anchor/outlier token position t, noise_t is small
  - No compression: 
    - When t < T, \gamma_t = 0, and noise_t is large
    - When t = T, noise_t is small

### Q1: analysis of noise 

- How does noise depend on context length, number of needles, layer index, etc.
- How does CoT use explicit tokens to "denoise" the count features?
- How to denoise the hidden states? Steering?

### Q2: nonlinearity of counting feature

- When does \gamma_t become nonlinear? Especially for large counts or long context
- Implication on steering and LoRA finetuning: does existing model adaptation techniques work for long context, or do we have to account for nonlinearity?

### Q3: Compression

- Pure retrieval (last token attends to all relevant needles) vs using counting feature to (roughly) summarize the number of needles in the prefix? 
- Related: counting increment, Markov property

## Key hypotheses

1. Counting features
   - Linear vectors / subspace (canonical linear representation hypothesis)
   - Nonlinear curves / manifold 
   - Piecewise linear vectors (? probably wrong?)
2. Compression of context
   - Maximal compression: prefix is encoded by a hidden state, e.g., RNN, LSTM, or Transformer in Markov chains, or ICL (via in-context vector)
   - Partial compression: short-range info is summarized and encoded into multiple anchor tokens (e.g., outlier tokens)
   - No compression: model relies on retrieval at the query tokens

*It is interesting to understand when nonlinear representation appears. Also it's interesting to characterize different compression in terms of mechanism, generalization behavior, etc.*

### About linearity vs nonlinearity

- For Dyck language, elementary experiment supports linearity, but today's experiment shows non-linearity
- For biased coin, representation of latent (success prob) is clearly nonlinear

  
## Tasks

1. Synthetic
   - Dyck languages, and other structure?
   - How do transformers represent key counting features (height, stack element)? 
   - Can we find nonlinearly represented features?
   - The effects of noise level and sequence length
2. Needle-in-a-haystack
   - Counting task in a long context: does a model robustly represent counting features? Does a model summarize the context?
   - How does CoT improve the accuracy? (Reducing noise in long context? Prevent lost-in-the-middle issue?)
3. Dyck structures in code and html
4. Character counting 
   - Similar to Anthropic's blog post? Why nonlinear curve?


## Main techniques

1. Probing: ridge regression, multiclass logistic regression, etc.
   - Nonlinear version of probing? Mapping hidden states h_t to count y_t nonlinearly?
2. Variance of hidden states, clustering
   - The idea is to calculate Var(h_t | y_t) for count-related y_t
3. Token-level ablation
4. Representation-level ablation & activation patching
   - Apply to hidden states (probably works)
   - Apply to activations of counting feature (no clue)

