# Plan

### General instructions
Principles: Read the plan carefully. 
- You are encouraged to raise clarification questions about typos, ambiguities, uncertainties, etc. Ask me if you need more information for making decisions. 
- Ask to correct obvious typos (with minimal changes) in this document before you code.
- Remember to update the `README.md` after coding
- If I ask you to include the Q&A interaction, summarize our interaction and add your summary under a subsection titled "#### Clarification Q&A (YYYY-MM-DD-time)" right below the requested section

## Counting features

I want to add hidden states analysis on counting related feature vectors. I want to build on the existing dataset generation and evaluation pipeline, and extend the analysis. You will likely write / revise scripts in the folder `src/counting`, and write a new notebook `notebooks/counting_feature_analysis.ipynb`. Similar to the previous notebook such as `counting_analysis.ipynb`, the new notebook will be used to run experiments on Google Colab.


### Notebook expectation
- The workflow should be compatible with existing NIAH datasets and tasks, especially `match_count` and `literal_count` tasks.
- The new notebook should inherit most of the stuff in Block 1 -- 5 from `notebooks/counting_analysis.ipynb`. However, you can remove all global variables involving ablation experiments, including `SELECT_EXAMPLE_ID`, those under `Token-level ablation settings`, ` Representation-level ablation settings`, and `Representation-level restore settings`. I don't need `ANALYZE_REASONING_TOKENS`, `MAX_NEW_TOKENS_FOR_COT`, `RUN_HIDDEN_STATE_ANALYSIS` and the comments around them.
- I won't need the old hidden states analysis (Block 6) and all ablation analysis in `counting_analysis.ipynb`.
- The new notebook should also follow Block 8 and 9 of `counting_analysis.ipynb` for I/O control.


### Extracting counting feature

I define *counting feature* as a vector in the representation space that the model uses to count/record matching needles. Intuitively, in the ideal scenario, each needle activates this counting feature, so that the hidden states increment the count statistic as the model autoregressively processes the prompt.

1. Filter examples to keep successful ones. If you can't find existing scored dataset, then evaluate the NIAH dataset and score each example; if you can find scored dataset (likely in folder `data/niah-example`), then skip the evaluation part. For the following steps, only use the examples that the model accurately matches the gold answer.
   - A useful info is the tokenized sequence length for each example. I hope that the lengths are very close, since the examples use the same haystack length and number of needles. You can print out basic info about the statistics of lengths, such as min, max, mean, std, etc.
   - Train/test splitting. Once you are done with filtering, split the set of examples. Reserve the test examples to later use.
2. Run a forward pass of each example and save hidden states at layers specified by  `LAYERS`. To save memory for now, a possible strategy is to save the hidden states in separate .pt files indexed by layers, e.g., `hidden_layer_{idx}.pt`. The .pt file should save a size (n, T, d) tensor, where n is the number of examples, T is the maximum sequence length, and d is the embedding dim. You can use `NaN` values for shorter sequences that do not have the maximum length.
3. Construct target counts. The idea is to create a tensor `target_count_y_t.pt` for each example that records the matching needle in the prefix. To be specific, let y_t be the number at position t of the tensor. Then y_t is defined to be the number of needles in the prompt before position t. This y_t is well-defined at off-needle positions, but requires careful handling within needles. I provide three choices below. You need to introduce a global variable `TARGET_COUNT_TYPE` in the notebook to select one choice.
   - Left-jump count: y_t increments as soon as t enters a needle span.
   - Right-jump count: y_t increments as soon as t leaves a needle span.
   - Interpolation count: y_t linearly interpolates between the number of *complete* needles before t and that number plus 1, when t lies in the needle span.
You can probably use existing or generated dataset / metadata that contains the info about the positions of needle spans. The final `target_count_y_t.pt` will be a size (n, T) tensor (not necessarily integer if we use interpolation count).
4. Linear probe (regression). Now that we have hidden states and target count, we want to run a ridge regression: viewing all (namely, n-by-T, minus NaN values) hidden states as a N-by-d feature matrix H, and all target count as a length N vector y, we want to find a vector u \in R^d such that H u \approx y. 
   - You can frame this mathematically as a standard L_2 regularized loss minimization || H u - y ||_2^2 + lambda ||u||^2
   - I think you can try existing solver in `sklearn` or `numpy`; try to think about potential issue for very large N.
   - If calling existing solver doesn't work, you can try solving this minimization with minibatch SGD. I believe there are existing code other people wrote, so if you adopt this, you need to do some search. (I suggest not to try this unless other ideas don't work. Please think carefully and suggest your plan.)
   - You need to save results, report and print standard statistics for ridge regression, such as R^2, and other statistical measures
5. Linear probe (classification). Similar to the previous step, but now you want to find a set of classification vectors such that H can predict y (as labels). I think the best shot is still using existing linear classification solvers. Save results and report standard classification statistics.
6. Visualization. For each of the previous two linear probes, make two plots:
   - Fig 1 (line fitting): Plotting y_t against h_t^T u. The y-axis is target y_t, and the x-axis is h_t^T u. Overlay a fitting line on {(y_t, h_t^T u)}. 
   - Fig 2 (2D visualization): Calculate the top principal direction v of H. Consider projecting all hidden states onto the subspace spanned by u and v. Use scatter plot to visualize the hidden states. Use different colors to indicated y_t. For classification, plot decision boundaries in addition.
7. Report regression/classification results and statistics for both linear probes on the test examples. Repeat step 6 on test examples.

### Implementation details

- Name files systematically. Pay attention to the I/O path. Check the file size. If a single file is too large (larger than 1GB), print a warning message.
- For linear probes, think carefully about the size of input matrices and compute costs. (Ask questions about this part!) 
- Run smoke tests for the statistical analysis like regression. Your code should pass tests on small-scale synthetic data.

### Clarification Q&A summary

- **Expected run size:** Target 20--100 examples per Colab run.
- **Expected sequence length:** Target 1000--2000 haystack tokens, plus additional chat-template, instruction, query, and assistant-prefix tokens.
- **Target runtime:** Assume an A100 high-RAM Colab runtime for the main experiments.
- **Probe fitting strategy:** Approximate/scalable probes are acceptable when they keep the flattened token matrix size bounded. Prefer token subsampling over exact all-token fitting if the exact design matrix would be too large.
- **Token subsampling:** Use token subsampling for linear probes. A reasonable default is to cap probe training at about 200,000 token positions per layer, while keeping all needle-span tokens and sampling off-needle tokens in a balanced way across count labels when possible. Add notebook comments explaining that users can raise or lower this cap based on available memory and runtime.
- **Interpolation and classification:** Disable classification when `TARGET_COUNT_TYPE = "interpolation"`, because the labels are fractional. Regression should still run.
- **Classification labels:** Use absolute integer count labels, e.g., `0, 1, ..., num_needles`.
- **Token inclusion:** Include all tokens in the tokenized uncontrolled prompt, including chat-template, instruction, query, and assistant-generation prefix tokens.
- **Post-needle target values:** After the last matching needle, keep `y_t` at the final count through the query/instruction suffix and any other remaining prompt tokens.
- **Position convention:** Use `h_t` to predict the count before token position `t`. For left-jump targets, increment only when token position `t` is the start token of a matching needle. For right-jump targets, increment only when token position `t` is the first token outside the matching needle span.
- **Needles to count:** Use the needles that contribute to `gold_answer` for the task. In particular, `match_count` and `literal_count` should count matching/gold-answer needles rather than unrelated/control insertions.
- **Prompt variant:** Analyze the uncontrolled prompt only. Ignore controlled prompts for this feature analysis because they were mainly useful for the previous ablation analysis.
- **Hidden-state dtype:** Save hidden states in `float16` or `bfloat16` to reduce disk usage, and cast chunks to `float32` during fitting when needed.
- **Probe artifact format:** Save learned probe vectors in `.pt` format and save JSON sidecars for metrics/configuration. It is also fine to save `.npz` copies if useful for analysis outside PyTorch.
- **Notebook defaults:** Provide a smoke-test mode that defaults to a small layer/example configuration first, while allowing users to switch to the full `LAYERS` list for the main run.

### Revision


1. Saving generated dataset. I want to introduce a global variable in the notebook `SAVE_GENERATED_DATA = False`. If true, a copy of the generated data will be saved to `REPO_DIR/data/niah-example/{setting_name}` where `setting_name` is a simplified run_name such as `Qwen3-8B_literal_count_easier_1000_needles_200_null_500` (stripping away the date and time).
   - The idea is that I want to avoid re-generating a dataset with the same setting if a dataset already exists.
   - This dataset-saving step should happen in the same time as zipping and saving full results
   - You also need to change the dataset loading logic. If the notebook can find the dataset with an exact setting (model name, prompt style, haystack length, insertion position, etc) that matches the info in the `setting_name`, then you can simply load the dataset and skip Block 4 and 5 in the notebook. Be sure to check the metadata carefully if you choose to do so, and print out a message.

2. Randomize needle insertion positions. Currently, the needle insertion is decided by a few global variables in the notebook, including
- `NUM_NEEDLES = 3`
- `INSERTION_POSITIONS = [200, None, 500]`
The issue is that the needle will be inserted at fixed positions. I want to provide flexibility. Please introduce an additional variable `RANDOMIZE_NEEDLE_INSERTION` (default False), and `RANDOMIZE_NEEDLE_SEED = 42`. If False, the workflow is exactly the same as existing notebook. If True, then
- **for each example**, we randomly draw `NUM_NEEDLES` positive integers in the range(50, `TARGET_HAYSTACK_TOKENS`-50), with the condition that any two positive integers are separated by at least 50. For example, `[50, 120, 400]` is okay but `[50, 90, 400]` is not good. Think about how you want to generate the random integers. 
- Finally, use the pattern of `INSERTION_POSITIONS` to set a corresponding integer to None. If `INSERTION_POSITIONS = [200, None, 500]`, then you need to change `[50, 120, 400]` to `[50, None, 400]`
- The randomness of needle lists depends on the random seed `RANDOMIZE_NEEDLE_SEED`.
- I believe that you will need to revise the workflow moderately: existing logic assumes that `INSERTION_POSITIONS` is fixed across all examples. The randomize_needle setting has needle insertion at random positions across examples.
- The pipeline for generating NIAH dataset, evaluation, and hidden states analysis should depend on the needle insertion positions for each example if `RANDOMIZE_NEEDLE_INSERTION` is True. Carefully check whether you need to revise other scripts.
- If `RANDOMIZE_NEEDLE_INSERTION`, the run_name will also append a string `_rand_insrt`.
  
#### Clarification Q&A (2026-06-12)

- **Dataset cache identity:** `setting_name` should include `NUM_EXAMPLES`, generation seeds (`GLOBAL_RANDOM_SEED`, `HAYSTACK_SEED`, `NEEDLE_SEED`), and `USE_THINKING` / `thinking_mode`. It does not need a separate tokenizer field because the tokenizer is assumed to match the model name.
- **Cache validation:** The workflow should use the folder name as a readable lookup key, but must validate JSON metadata such as `config.used.json` rather than relying on the folder name alone.
- **Cached artifacts:** The reusable cache under `REPO_DIR/data/niah-example/{setting_name}` should include both generated dataset files and scored prediction files when available. If cached generated data exists but cached predictions do not, skip only dataset generation and still run response generation/evaluation.
- **Prediction validity:** Cached predictions should only be reused when the validated metadata matches the current model/task/prompt/generation/evaluation-relevant settings. Mismatched metadata should invalidate the cache for the current run.
- **Cache timing:** Save/cache small generated and scored artifacts before final cleanup/archive, and do not copy hidden-state tensors or other large analysis artifacts into `data/niah-example`.
- **Random insertion order:** When `RANDOMIZE_NEEDLE_INSERTION=True`, sample `NUM_NEEDLES` separated positions, sort them, and then apply the `None` pattern from `INSERTION_POSITIONS` slotwise.
- **Random insertion constraints:** Draw positions first with the separation constraint, then apply the `None` pattern. Either inclusive or Python-style exclusive upper bounds near `TARGET_HAYSTACK_TOKENS - 50` are acceptable. Impossible settings should raise a clear `ValueError`.
- **Randomized setting names:** Randomized settings should append the fixed-position pattern and seed, for example `_needles_200_null_500_rand_insrt_seed_42`, so the key preserves the requested `None` pattern and random seed.
- **Per-row downstream analysis:** For randomized insertion runs, hidden-state and target-count analysis should derive expected needle counts per row from row metadata rather than from a single config-level fixed insertion list.
- **Backward compatibility:** Old cached datasets without randomization metadata should be treated as non-randomized (`RANDOMIZE_NEEDLE_INSERTION=False`).
- **Overwrite policy:** If `SAVE_GENERATED_DATA=True` and the cache directory already exists, do not overwrite it by default. Validate the existing cache and report the result instead.


## Steering counting feature

I want to examine whether the counting feature vector that I extracted from the above approach can steer the model's output. You will likely write new functions / scripts under `src/counting`, and revise the notebook in `notebooks/counting_feature_analysis.ipynb`.

**Assumption**: A global variable in the notebook or config param `LAYERS` is given. 
**Goal**: Conduct steering for each layer in `LAYERS` separately at the last token position.

1. Select test examples. If a run name in `data/niah-example/{run_name}` matches the current run setting, load the examples from `dynamic_niah_v2.jsonl`; if no such folder exists, generate a dataset and use the model to score the examples (same as Block 4 and 5 of `counting_feature_analysis.ipynb`). Then print the number of successful examples and unsuccessful examples. 
   - Introduce a new global variable `MAX_NUM_STEERING_EXAMPLES = 10` in the notebook. Select both successful and unsuccessful examples up to this value. Note: you may end up selecting fewer examples if the dataset is small or very unbalanced. Then, print the total number of selected examples, and the id of examples in the dataset.
2. Load the counting feature calculated from previous sections. The file should be something like `ridge_probe_layer_4.pt`. I think it should be a vector of same length as the embedding dim (ask me if it is untrue). Let the normalized vector be `v_l` where `l` is the layer index.
3. At the last token, calculate the inner product between hidden state and feature vector, and then calculate the standard deviation std(h^T v_l) across examples (AKA scale factor). Denote it by `sigma_l`.
4. Replace the hidden state at the last position with the steered version: `h_l += beta * sigma_l * v_l` where `beta` is the steering strength. Introduce a global variable in the notebook `STEERING_COEFF = [-4, -2, -1, -0.5, -0.25, 0.25, 0.5, 1, 2, 4]`. 
5. To generate new tokens, replace the hidden state `h_l` in each forward pass with the steered hidden state. 
   - Note: in the forward pass, you should apply steering at the current last token at every decoding step (NOT fixed original prompt last position). 
   - Use the same precomputed `sigma_l` for the whole generation by default.
6. Scoring the steering result. Introduce a new global variable in the notebook, say, `MAX_NEW_TOKEN_STEERING = 20`. When generation stops or hits the max_new_token, compare the result with the gold answer.
7. Loop the above steps for every layer in `LAYERS` and every choice of steering strength in `STEERING_COEFF`.
8. Save results. You decide what more info to save, but you should include the following.
   - For each layer, each steering strength, and each example, keep the model's predicted count before steering and after steering. 
   - Generate a table that summarizes, for each layer & steering strength, the accuracy before steering and after steering across successful examples, and across unsuccessful examples respectively


### Revision

After running some pilot experiments, I don't see steering changing the model's accuracy. I want to adopt a different steering strategy: instead of the last-token position, change hidden states at needle-span positions. You will likely revise files in `src/counting` and `notebooks/counting_feature_analysis.ipynb` based on existing workflow to meet the following requests.

1. When applying steering `h_l += beta * sigma_l * v_l`, instead of doing this at the current last-token position, do it at one of the needle-span positions.
   - For a given needle, find the needle-span positions, and then apply steering at every position in the needle span.
   - Scoring the steering result as before. Still use `MAX_NEW_TOKEN_STEERING = 20`. 
   - Loop steering + scoring over the needle span, steering strength, and layer. So in total you have num_needles * number steering strength of choices * number of layers steering experiments.
   - Add a single progress bar for the nested loops.
   - Save a complete result for each example, each combination. Then, generate a summary table that takes averages across examples.
   - Note: use the actual needles; ignore those uninserted needles with `None` values in `INSERTION_POSITIONS`.


