# Plan

## Single-example analysis

I want to do in-depth analysis for a single example, instead of looping over 20 examples. For sanity check, I will start with some existing analysis with light code. You need to mostly write and revise code under a newer folder `src/single_example`, and a newer notebook `notebooks/single-example-v2.ipynb`. 

1. Create the folder `src/single_example` if it is not already in the repo. Check if `notebooks/single-example-v2.ipynb` exists.
2. Use existing NIAH dataset: Find the generated data `dynamic_niah_v2.jsonl` under `data/niah-example`; assume the example id is given by the user (use the variable in notebook EXAMPLE_ID = 0); extract both the uncontrolled prompt and controlled prompt under EXAMPLE_ID.
3. Use existing inference workflow to apply tokenizer and chat-template to both prompts, which gives two input tensors. Identify the needle positions in the uncontrolled input tensor. Print the needle positions and validate that the needle positions are 3 consecutive segments of integers. Then, save both input tensors, and 3 segments of needle positions into a json file.
4. Use existing LLM inference pipeline to calculate the hidden states. Similar to Block 7 of notebook `analysis_hidden_states_v4.ipynb`, save hidden states as `hidden_inputs_{id}.pt`, and plot `figures/inputs_{id}.png`. DO NOT plot the PCA visaulization.
5. Conduct Q/K-cache outlier analysis. Similar to Block 11 of `(block 7 of analysis_hidden_states_v4.ipynb`, calculate the attention metrics and massive activation metrics and save files under `tables`; also save the figure under `inputs_*_qk_outliers.png`. In addition, save the plot-related data under `tables/inputs_*_qk_plot_data.json`.
6. Delete `hidden_inputs_{id}.pt` and tensor data generated from Q/K-cache calculation.  
7. Save & zip results.

**Output and saving files**. At the start of the colab notebook, generate the name of the run folder, e.g., `run_20260604_020115_Qwen_Qwen3-8B_task-argmax_prompt-easier_len-1000_needles-3` (or perhaps something shorter), based on Block 2 of the notebook `analysis_hidden_states_v4.ipynb`. Using this run id, I want to save all generated files under the folder `/content/run_{id}`, and the subfolder structures remain the same; Namely, I expect to have `figures`, `generate_data`, etc. After running the colab notebook is completed, zip all files, and move to a user-specified path `RESULTS_PATH`. By default, in the notebook, set `RESULTS_PATH` to the folder `results/single-example/run_{id}`.

**Additional requests**. Please clean up & revise the initial notebook `notebooks/single-example-v2.ipynb` I provided to you. Remove unnecessary variables and functions. Note that we don't need to generate datasets so there are redundant variables.


## Ablation analysis: token level

The next step is to evaluate how impactful the "critical tokens" (including outlier tokens) are. I want to build on the previous setup and analyze a single example indicated by the `EXAMPLE_ID` variable in the notebook.

1. Create a configuration file `configs/ablation.json`. It includes `num_critical_tokens = 10` and some other useful config parameters.
2. Identify several sets/patterns of "critical tokens". Each set consists of `num_critical_tokens` token positions from a particular pattern. NOTE: The token positions must lie outside of needle segments, so the critical tokens should not be any needle tokens. The patterns include
   - Massive activation tokens. At a layer given by a config param `critical_token_calc_layer = 24`, you can perhaps reuse `massive_tokens_outside_needles_all.csv` (column `norm_ratio_to_median`) to find the K strongest massive activation tokens, where K equals `num_critical_tokens` (same below). Order the positions according to how strong the massive activations are.
   - Attention sink tokens. Similarly, At a layer given by a config param `critical_token_calc_layer`, you can perhaps reuse or do similar calculation as in the attention-metrics file (likely `attention_sinks_topk.csv`).
   - Needle-sensitive tokens. The tokens outside the needle segments with smallest K cosine similarity (namely, the y-axis of the second subplot of `inputs_*_qk_outliers`) at the given layer `critical_token_calc_layer`.
   - Needle-tail tokens. For each needle, find the adjacent K token positions right after each needle ends. For example, if a needle lies in position t_1, ... t_2, then I am looking for t_2 + 1, ... t_2 + K.
You can generate some intermediate files such as `tables/ablation/needle_sensitive_tokens_outside_needles_all.csv`. Then, prepare an aggregate file (perhaps json file) under `tables`. For each pattern, the file saves the ordered token positions, tokens, and scores (if any).
3. Generate model responses and score responses under ablation.
   - Loop over the patterns, and `k in range(1, K+1)`. The goal is to ablate the top-k critical tokens for a given pattern.
   - Replace the k critical tokens by k randomly chosen tokens. To do this, you can first form a large pool (say, 5,000) of irrelevant tokens by using the haystack text files under `data/haystacks/paul_graham`. Then, uniformly sample from these pool to replace the k critical tokens.
   - We view the input tensor with the replacement as the ablated/perturbed input. Run the model inference to generate the response, perhaps reusing or copying existing pipeline (check `out = model.generate(**inputs, **generation_kwargs)` in `scripts/gen_responses.py` for example).
   - Score the accuracy of model generation for each ablated input. Since we only have one example per ablation, I think the accuracy is either 0 or 1, for a given random seed (Please double check.) Also score the accuracy for the unablated input as a baseline.
   - Aggregate the results into a `csv` file that contains the accuracy for each pattern and each k from 1 to K, together with the baseline result (which you should put in the top).
  
**Revision**. 
- Previously, "critical tokens" are top-ranking tokens of a particular pattern lying outside the needle positions. I want to add additional constraint: The "critical tokens" should not be the first 5 tokens or last 5 tokens of the input sequence.
- I also want to make some changes to include a loop over all examples. (i) Introduce a global variable `ALL_EXAMPLES` in the notebook, with default value being False. (ii) If `ALL_EXAMPLES` is true, then all the analysis is run for all examples, but looping over all examples in `niah-example/dynamic_niah_v2.jsonl`. (iii) If `ALL_EXAMPLES` is true, then within the run folder, results are saved to a subfolder `example_id_*` for each example. If I have 20 examples in total, then I expect 20 subfolders `example_id_*`. You save existing generated results to each subfolder. Finally, a summary file `ablation_results_all.csv` is generated alongside the subfolders; the summary file contains columns "Pattern", "k", "accuracy", where "accuracy" is the average accuracy over all examples.

**Revision: alternative dataset**.
- Currently, the analysis starts from loading an existing dataset `data/niah-example/dynamic_niah_v2.jsonl`. I want to use a different dataset that stills follows the format as in `dynamic_niah_v2` but based on different data generation configs (for example, using a different niah task or different needle positions). I want to slightly modify the workflow of the notebook `single-example-v2.ipynb` to process a different dataset.
- You will load the dataset from the path `data/niah-example/{run_name}/dynamic_niah_v2.jsonl` where the run_name is something like `Qwen_Qwen3-8B_task-argmax_example-0_prompt-easier_len-1000_needles-3`. If there is no such dataset, give an error message and abort running.
- You need to read the dataset, metadata, and configs from this jsonl dataset. Then, you will run the same or similar pipeline as in the curren notebook `single-example-v2.ipynb`. Carefully check whether you need to adjust the ablation analysis calculation, if I modify the configs/metadata of the dataset.

## Ablation analysis: representation level

Previously, I did ablation analysis at the token level (replacing critical tokens by random tokens). Now I want to conduct representation-level ablation analysis. You will likely create a script `src/single_example/ablation_representation_analysis.py` and revise the notebook `single-example-v2.ipynb`.

1. Create a configuration file `configs/ablation-representation.json`. It includes `num_critical_tokens = 10`, `randomize_from_top_layer = True` and some other useful config parameters. As before, I will also use K to refer to `num_critical_tokens`. Import the config file and use the values by default, but users can override `num_critical_tokens` and `randomize_from_top_layer` via global variables in the notebook.
2. You will load the dataset from the path `data/niah-example/{run_name}/dynamic_niah_v2.jsonl` where the run_name is something like `Qwen_Qwen3-8B_task-argmax_prompt-easier_len-1000_needles-3`. As before, the run_name is specified in the notebook as a global variable. If there is no such dataset, give an error message and abort running.
3. Profile the empirical distribution of hidden states. In order to sample from an appropriate distribution for replacing hidden states, I want to first calculate basic statistics of hidden states. 
   - Use (latter) half of the examples in the niah dataset. For each example, do a forward pass and save the hidden states into a .pt file. Calculate the mean and standard deviation of every coordinate of every hidden state at every layer; in other words, you will have num_layer * num_seq_length * num_hidden_dim values of mean & std for one example. 
   - To avoid using too much storage, I suggest that you use the running mean and std to implement this part, so that you don't need to maintain a file for all examples. I also suggest that you use bf16/fp16 and avoid fp32.
   - The final mean & std tensors are saved as a .pt file.
4. Find several sets/patterns of critical tokens. In the current repo, we already calculated several sets (patterns) of critical tokens, including massive activation tokens, needle-sensitive tokens, etc. You can check the code for generating `tables/ablation/critical_tokens.csv`, or the section `Ablation analysis: token level` in this document.
   - I want to add more sets of critical tokens. For each needle, the token positions of needle span naturally lead to a set of critical tokens. I will call needle-span tokens to refer to this category.
   - I also want additional critical token patterns on top of the existing ones. Additional set of massive activation tokens. This time, I don't want to exclude any positions. Just use the score to rank all tokens in the input sequence and pick the top K tokens.
   - Additional set of attention sink tokens. Similarly, this time, I don't want to exclude any positions. Just use the score to rank all tokens in the input sequence and pick the top K tokens.
   - Note: Except for the needle-span tokens, all other sets of critical tokens are capped by K.
5. Assume that `EXAMPLE_ID` (default 0) is given by the notebook; and `num_critical_tokens`, `randomize_from_top_layer` are given by the config & notebook. The main ablation experiments loop over both the patterns and `layer_idx` that go from layer 0 to the last layer of the model.
   - For the example given `EXAMPLE_ID`, by a forward pass to calculate the unablated hidden states. Save all hidden states to a .pt file.
   - Here is the replacement policy. For a given pattern, if `randomize_from_top_layer` is true, randomize the hidden states at corresponding critical token positions from layer `layer_idx` to the last layer; if `randomize_from_top_layer` is false, randomize the hidden states at corresponding critical token positions from layer 0 to layer `layer_idx`. When you do randomization, randomly draw a normally distributed random variable with the mean & std given by the previously saved statistics for each coodinate of the hidden states.
   - To run model inference on a test example, we can't use model.generate(). Instead, to generate each next token, we run the forward pass layer by layer; After each layer, apply the replacement policy at critical tokens at the right layers. 
   - Score the accuracy of model generation for each ablated input. Since we only have one example per ablation, I think the accuracy is either 0 or 1, for a given random seed (Please double check.) Also score the accuracy for the unablated input as a baseline.
   - Aggregate the results into a `csv` file that contains the accuracy for each pattern and each `layer_idx`, together with the baseline result (which you should put in the top).
6. Remember to delete .pt files larger than 100 MB before zipping the result and save to the Google Drive when running the colab notebook.

## Ablation analysis: representation level (continued)

In the previous section, I did ablation analysis by corrupting targeted hidden states given clean/informative underlying tokens. Now I want to do the complement: restoring targeted hidden states given noisy/uninformative underlying tokens. You will likely create a script `src/single_example/ablation_representation_analysis_restore.py` and revise the notebook `single-example-v2.ipynb`. As before, assume that `EXAMPLE_ID` (default 0) is given by the notebook; and `num_critical_tokens`, `randomize_from_top_layer` are given by the config & notebook.

1. Create a configuration file `configs/ablation-representation-restore.json`. It includes `num_critical_tokens = 10`, `randomize_from_top_layer = True` and some other useful config parameters. As before, I will also use K to refer to `num_critical_tokens`. Import the config file and use the values by default, but users can override `num_critical_tokens` and `randomize_from_top_layer` via global variables in the notebook.
2. You will load the dataset from the path `data/niah-example/{run_name}/dynamic_niah_v2.jsonl` where the run_name is something like `Qwen_Qwen3-8B_task-argmax_prompt-easier_len-1000_needles-3`. As before, the run_name is specified in the notebook as a global variable. If there is no such dataset, give an error message and abort running.
3. Do a clean run and save the unablated hidden states. Assume that `EXAMPLE_ID` (default 0) is given by the notebook. We do a normal forward pass and save all the hidden states into a .pt file.
4. Find several sets/patterns of critical tokens. Similar to the previous section, I want to include existing patterns, and additional add more.
   - Massive activation tokens, attention sink tokens, needle-sensitive tokens---both excluding & including the needle positions and edge positions---are already included in the pipeline, which totals 6 patterns.
   - Needle-span tokens. These patterns are already included in the pipeline, which gives patterns that equal to the number of needles.
   - Now, I want to include "Needle-tail tokens" (which appeared in token-level ablation analysis but not previous section). For each needle, find the adjacent K token positions right after each needle ends. For example, if a needle lies in position t_1, ... t_2, then I am looking for t_2 + 1, ... t_2 + K.
5. Form a large pool (say, 5,000) of irrelevant tokens. Use the haystack text files under `data/haystacks/paul_graham`. In the later step, you will uniformly sample from these pool to replace the critical tokens.
6. Run model inference with corrupted tokens and restore targeted hidden states. The main ablation experiments loop over both the patterns and `layer_idx` that go from layer 0 to the last layer of the model. More specifically, 
   - If `randomize_from_top_layer` is true, define the "layer range" to start from `layer_idx` and ends with the last layer; If `randomize_from_top_layer` is false, then layer range starts from 0 and ends with `layer_idx`.
   - For a test example indicated by `EXAMPLE_ID`, replace tokens at all needle positions with randomly drawn tokens, as described in Step 5. Once we have the replaced input sequence (with no real needles), run a forward pass layer by layer.
   - For hidden states at each layer, if the layer is in the layer range, then *restore* the hidden states at the token positions given by the current pattern---namely, replace the hidden states at targeted pattern positions by the previously saved unablated hidden states.
   - Once the forward pass runs through all layers and generates a token, then repeat this procedure autoregressively to generate the model's response (as we normally do with `model.generate()` but with restored hidden states calculation).
   - Score the accuracy of model generation for each pattern and each `layer_idx`. Since we only have one example per setting, I think the accuracy is either 0 or 1, for a given random seed (Please double check.) Also score the accuracy for the unablated input as a baseline.
   - Aggregate the results into a `csv` file that contains the accuracy for each pattern and each `layer_idx`, together with the baseline result (which you should put in the top).
7. Remember to delete .pt files larger than 100 MB before zipping the result and save to the Google Drive when running the colab notebook.

