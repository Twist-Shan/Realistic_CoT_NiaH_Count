# Plan

## Dynamic-dataset: More controlled format of NIAH task

I want to make major changes to the NIAH tasks again. It is better to keep use old scrips / python files / notebooks and add updated files; for example, you can generate new files as "generate_dynamic_niah_v2.py" and save it to the same folder as the current "generate_dynamic_niah_v2.py". The high-level idea is the generate tokenizer-specific dynamical NIAH dataset with improved control over needle insertion positions.


1. Storing the raw haystack files (Paul Graham Essays). As before, you can go to the link https://github.com/gkamradt/LLMTest_NeedleInAHaystack/tree/main/needlehaystack/PaulGrahamEssays to copy all the Paul Graham Essays into a data folder. I believe that there are 200+ text files in total. Use `scripts/gather_paul_graham_essays_v2.py` to fetch/sync essays. (Legacy helper `scripts/prepare_paul_essays_v0.py` is archived.) After that, filter / remove files smaller than 5KB. 
2. Create a configuration file "niah_dynamic" that saves the default parameters. The user specifies a number of values:
   - the NIAH task (default is argmax), 
   - an LLM tokenizer (default is Qwen3-8B), 
   - the number of NIAH examples (default is 100),
   - the target sequence length of the haystack (default 1000), 
   - the number of neeldes (default is 3), 
   - the insertion positions (default is a list [100,200,400]), 
   - Prompt style of NIAH tasks (default is "easier"). "Vanilla" means context + query, and "easier" means query + context
   - Thinking mode (default is False)
   - output directory (default is generated/dynamic_niah)
   - and multiple random seeds: one seed for generating the underlying haystack, a separate seed for generating each needle. If the random seeds are to None, the haystacks and needles will be indepedently sampled, so I expect to see different haystack and needles across generated NIAH examples. If any of the random seed is not None, then I expect to see the haystack or needle being fixed across generated NIAH examples.
3. We sample a text file from the pool of raw haystack files. Tokenize the text file (don't use special tokens or padding), and randomly select a window (token sequence) matching the target sequence length (as specified by the config file). Throw a warning if the whole tokenized text file does not reach the target sequence length.
4. Generate tokenized needles (don't use special tokens or padding). The current approach you use (i.e., city names and ratings) for generating needles looks fine. All the needles, N1, N2, N3, ..., are tokenized (don't use special tokens or padding). 
5. Obtaining haystack with needles. You just insert each needle at the insertion position specified in the config file. The resulting instance is a tensor sequence representing the haystack with inserted needles.
6. Consider an appropriate schema for representing the NIAH examples. You already have a good scheme with most of the desired elements, but just to be clear, I think you should include at least the following---the tokenized needle sequence, length of each tokenized needle, decoded text from tokenized sequence, relevant records, query, gold answers, etc. 
7. This final data for NIAH is likely represented by a jsonl file that contains entries equaling the target number of NIAH examples. 
8. Similar to what you currenlty have, I want to call "PYTHONPATH=src python scripts/generate_dynamic_niah_v2.py" to generate the NIAH data (e.g., jsonl file). Optional arguments can modify the default configuration values.
9. Some clean-up. I also want to pack functions such as "build_messages_easier", "build_messages_harder", "parse_prediction" into a separate python file. They serve the purpose of wrapping the NIAH questions into prompts.
    - Additionally, please clean up the code block that contains # Optional: extract hidden states for all token positions at a specified layer in the notebook "quick_eval_openllm_niah_argmax_colab_v3.ipynb". A messy part is finding the token positions of the inserted needles in the prompt. The strategy here is to match token sequence that represents the needle. But the code is messy. Pack token sequence matching in a separate python file.
  
## Analysis of hidden states

I want to study the hidden states of LLMs when they solve NIAH tasks. The main goal is to understand the behavior of the hidden states in the long context as the model processes both the haystack and the needles. I wrote some simple analysis in the notebook "quick_eval_openllm_niah_argmax_colab_v3.ipynb" and plan to reogranize, revise, and expand the analysis. Of course, you can wrap some core computation into python scripts, and only use notebook lightly. It's perhaps better if you write easy-to-test functions in python scripts so that you can test, and other hard-to-test code in notebooks so that I can interactively experiment, test, and visualize.

Your task is to revise existing files, and create new python scripts & notebooks. Below I'll lay out the main steps.
1. Generate the NIAH dataset with the control prompt, once I set the control switch in the config file. Assume that I only set one value of control_switch to be true (i.e., control for a single needle). Please print logging after generating this dataset.
2. Build the prompt for each NIAH example: you can refer to the notebook "quick_eval_openllm_niah_argmax_colab_v3.ipynb" where I use "build_messages" and get "inputs". Do this for both the normal prompt and the control prompt, which leads to two input tensors "inputs" and "inputs_control". However, you need to be careful and check whether the two tensors have the same length (tokenization may introduce a small difference in the sequence length). If the two tensors have different lengths, print a logging, calculate an offset to align the two tensors so that inputs[0, t:] equals inputs_control[0: (t+offset):] for t after the needle insertion position.
3. Do a forward pass and calculate the hidden states for both "inputs" and "inputs_control". The user specifies either a layer index or a list of layer indices. We use 3D tensors H and H_control to represent the hidden states. Compare H and H_control at every layer and every position, in terms of relative norm difference, cosine similarity, etc. Remember to take the offeset into account, e.g. cossim(H[layer_idx, t, :],  H_control[layer_idx, t+offset, :]) for cosine similarity cossim. For each pair of inputs, return a dictionary with keys being measurements such as "relative_norm_diff" and values being the 2D tensor storing results for every layer, and every position after the insertion position. Save the dictionary to a user specified directory; the default is "analysis/hidden_comp" (create the folders if it does not exist).
4. Make a plot for all measurements using subplots. For each type of measurement / subplot, the x-axis is the position index, and y-axis is the measurement value. Each layer is represented by one curve. The legend should indicate the layer index. You should make a figure and save to the analysis directory for every pair of inputs.
5. I expect resutls like "analysis/hidden_comp/inputs_0.pt", "analysis/hidden_comp/inputs_1.pt", etc., "analysis/hidden_comp/inputs_0.png", "analysis/hidden_comp/inputs_1.png", etc.


### Restructuring the repo

Consider revise the code so that the results of the experiments are saved according to the following structure.

```text
└── results/
    ├── run_{date}_{time}_{model}_{params}
        ├── figures/
        ├── tables/
        ├── generate_data/
        ├── logs.txt
        ├── analyze_hidden_states_config.json
        ...
```

- Each run generates a folder of the format run_{date}_{time}_{model}_{params}, where date and time record the starting time of the experiment. Params contain important configuration parameters, such as prompt_style, niah_task, etc.
- logs.txt contains the print output

### Improving the workflow

- Generate and score the model's responses. Currently, analyze_hidden_states.py does not generate responses, and we don't know the accuracy of the model for the NIAH task. Refer to the notebook "quick_eval_openllm_niah_argmax_colab_v3.ipynb". There is a code block that calls "out = model.generate" to generate responses, uses "parse_prediction(gen)" to parse the model's responses, and then scores the results. Consider writing a similar script to print the accuracy and save the results first "python scripts/gen_responses.py" before I call "python scripts/analyze_hidden_states.py". 
- Additional config parameters. You need several additional parameters, "output_pred_jsonl", "output_metrics_json" are output files for saving / scoring model generation. You can save both files under the run folder. You also need "max_new_tokens" when calling model.generate. The default value is 64 for nonthinking mode and 1024 for thinking mode. The "temperature" parameter is 0 (greedy decoding) by default. Please revise the config file and cfg dataclass to incorporate the additional config parameters.
- Another NIAH task. Besides argmax, I want to slightly modify the task: instead of retrieving the highest rated city and score, I want to retrieve the number of rated cities and the average score. This will impact the query, the gold answer (e.g., calculation of "the winner"), prediction scoring, and perhaps a few other things. Everything else (haystack, needles, insertion, control sequence, random seed, etc) remain the same. 
- Please save the print output to the file logs.txt. Currently, the print output is not saved.

  
### Further analysis: PCA visualization

Please revise existing code & generate new code to implement the following. In particular, I want to add more analysis on top of the two measurements in item 3. Please implement the following in a concise way. Don't introduce too many wrappers.
PCA visualization.
- Loading the hidden states. Assume that hidden states "hidden" and "hidden_control" are saved (as .pt files for example) in earlier hidden states calculation; if not, implement this part. Load the tensors and check whether "hidden" and "hidden_control" both have the shape (1, layers, seq_length, hidden_dimension) or (layers, seq_length, hidden_dimension), where layers is the number of specified layer indices, and seq_length is the sequence length of the input tensor to the model. Note that we generate hidden states each example at a time (namely batch_size is one). Check if the tensors have expected shape.
- Calculating the projection matrix. We researve 5 examples untouched as test data, and only use the remaining examples to find the projection matrix. For a given layer (layer_idx) and a given input example (id), we calculate the norm of hidden state from "hidden" at every token position. We filter out the largest 10% of hidden states. Then, we collect all hidden states across token positions and input examples (excluding the 5 test examples), which gives a matrix of shape (N, hidden_dimension), where N is the total number of hidden states after filtering at a given layer. We apply SVD to this matrix to get an orthonormal matrix V of shape (hidden_dimension, 2). This matrix will be our projection matrix to 2D for the given layer.
-  Projection to 2D. For a given layer and a given test examples (among the 5 untouched), we use the projection matrix V to project both uncontrolled and controlled hidden states into 2D. This yields seq_length number of points in 2D for both un-controled and controled hidden states.
- Plotting. Use two different colors (e.g., red vs blue) to visualize projected points of uncontrolled and controlled. Use color gradient to represent the position index t. Add legend / colorbar. Only do scatter plots, not line plots. This visualization will show both original hidden states and control hidden states across token positions.
- Save the result as "PCA_layer_{layer_idx}_inputs_{id}.png". Iterate over all specified layers and the 5 input examples.
     
### Further analysis: outlier analysis: attention-sink and massive-activation tokens

I want to write instructions to tell Codex to continue to code based on the repo. Specifically, I want to focus on **outlier tokens**. Remember our discussions earlier: we discussed attention sink tokens and massive activation tokens. I broadly use outlier tokens to refer to both. Here is a initial plan to implement the analysis. Overall, the task is to expand the analysis in the notebook "analysis_hidden_states_v4.ipynb". For this task, use uncontrolled prompts only (namely, inserting needles at every requested positions.)

1. Calculate the QK cache at select layers (remember the LAYERS variable), preferrably using the newly added script "capture_qk_qwen3.py". Save the tensors to "tensors" folder under the run folder. Be default, use all heads; but the users can also specify the heads to analyze via a variable called HEADS. For this part, you likely need to revise code in existing blocks, probably calling the right function and save tensors.
2. Calculate attention-related statistics.
   - You don't need doing inference over the model. You can just load the saved tensors and do offline analysis.
   - The avarage attention score of each token, namely, the average attention each token receives from later tokens
   - Attention at critical tokens. The critical tokens include tokenizer-related specicial symbols, include the BOS token, think token, etc. The critical tokens also include position-related tokens, such as the first 10 tokens, last 10 tokens, the 10 tokens following the inserted token positions. For each of the critical tokens, we want to have a attention vector of length T, where T is the sequence length. Please return the results in a well-organized manner.
   - Attention mass of needles. Namely, the average attention the needle tokens collectively receives from later tokens. 
   - Other statistics that your code in "analyze_qk_qwen3.py" contains, such as top-k and entropy.
   - Organize and save the computed statistics to the same "tensors" folder.
   - In the notebook, add a block to summarize the main findings about the attention-related statistics.
 - Calculate massively activating tokens.   
   - You need to do some initial analysis of the hidden state norms at layers specified by LAYERS, like checking the distribution. 
   - I am looking for hidden states at given layers that show very large norms. My preference is to select hidden states that have norms larger than 10 times (more generally, use a threshold hyperparameter instead of hardcoded 5). Collect all the corresponding tokens across the examples. Save the massively activating tokens (position, token, hidden state norm, etc.) for all layers and examples into a file. Print the massively activating tokens into a text file in the folder "tables".
   - Do further analysis about massively activating tokens across examples and across layers. Check if there are commonly massively activating tokens. Check whether there are patterns.
   - In the notebook, add a block to summarize the main findings about massively activating tokens.  
 3. Improving the figures.
   - Currently, the figure "inputs_0.png", "inputs_1.png", ... under "figures" of the run folder has two subplots. I want to add a few subplots above the existing two plots. The first subplot shows avarage attention score across token positions. It has the same x-axis as the other two subplots, e.g., token positions across the entire input sequence. Moreover, I want to overlay this subplot with multiple vertical dashed lines at selected token positions---the positions where tokens are identified as massively activating. So for each layer, there will be a line plot and a collection of vertical dashed lines, all using one color. Different layers use different layers, but the colors should be consistent with the other two subplots. You don't need to add a legend for this plot.
   - The second added subplot shows entropy statistics, instead of avarage attention score. Other things remain the same. 

### Further analysis: needle-sensitive tokens

1. For a pair of uncontrolled and controlled prompts, earlier analysis calculates the hidden states across token positions at specified layers. For a given layer, let us denote the cosine similarity to be C[t] at position t. If a needle occupies an interval [t_1, t_2] in the sequence, I will call [t_1, t_2+5] the *expanded needle segment*. We can collect a dictionary D consisting of C[t] such that t is outside all expanded needle segments.
2. I will call a token to be *needle-sensitive* if (i) the token's position is outside any expanded needle segments, and (ii) the cosine similarity between the pair of hidden states at the token position is among the M smallest values in the dictionary D. Here, M is a user-specified value, with a default value 20.
3. I want to print needle-sensitive tokens for all layers in the ascending order of the cosine similarity. And then write the info Layer id: {token, t, C[t]}, into a json file. Note that you should print token not the token id.
