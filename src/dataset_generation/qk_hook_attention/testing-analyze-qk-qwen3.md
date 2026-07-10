Create a Colab notebook named notebooks/test_analyze_qk_qwen3_colab.ipynb.

Goal:
Smoke-test and interactively inspect scripts/analyze_qk_qwen3.py using Q/K cache files produced by scripts/capture_qk_qwen3.py for Qwen/Qwen3-8B.

Assumptions:
- The repo already contains:
  - scripts/capture_qk_qwen3.py
  - scripts/analyze_qk_qwen3.py
- A Q/K cache directory exists, e.g. /content/qk_cache_test, with:
  - metadata.json
  - layer_00_q_raw.pt
  - layer_00_k_raw.pt
  - optionally layer_18_q_raw.pt, layer_18_k_raw.pt, layer_35_q_raw.pt, layer_35_k_raw.pt
- The cache was generated from Qwen/Qwen3-8B.
- Use attn_implementation="sdpa" for simplicity if the notebook needs to load the model.

Notebook requirements:

1. Locate and inspect cache directory
   - Define CACHE_DIR = Path("qk_cache_test").
   - Assert metadata.json exists.
   - List files in CACHE_DIR.
   - Load metadata.json and print important fields:
     model_name
     seq_len
     target_layers
     num_hidden_layers
     num_attention_heads
     num_key_value_heads
     head_dim
     dtype
     input_text or prompt preview if available
   - Print expected Q/K shapes:
     q_raw should be [1, T, 4096]
     k_raw should be [1, T, 1024]
     for Qwen3-8B.

2. Inspect raw Q/K tensor files
   - Load one layer, default layer 0:
     layer_00_q_raw.pt
     layer_00_k_raw.pt
   - Print shape, dtype, device, memory size in MiB.
   - Print simple summaries:
     mean, std, min, max, norm over last dimension for a few positions.
   - Include an interactive variable LAYER_IDX that I can change.

3. Load analyzer utilities
   - Import scripts/analyze_qk_qwen3.py as a module if possible.
   - If the script is CLI-only, use subprocess calls from notebook cells.
   - Prefer importing reusable functions if they exist.
   - Do not duplicate the entire analyzer code unless absolutely necessary.

4. Run a minimal analysis
   - Use layer 0 and head 0 first.
   - Use a few query positions:
     last token
     first token
     middle token
     a manually specified list like [0, T//4, T//2, T-1]
   - Define simple named spans:
     bos: [0, 1)
     first_16: [0, 16)
     middle_16: [T//2, T//2 + 16)
     last_16: [max(0, T-16), T)
   - Run analyze_qk_qwen3.py to compute:
     top-k attended positions
     attention entropy
     span mass to named spans
     local-window mass if supported
   - Save analysis results under /content/qk_analysis_test.

5. Display results interactively
   - Load the analyzer output files, such as JSON/CSV/PT depending on what the script writes.
   - Display a pandas DataFrame for:
     top-k attended token positions
     span masses
     entropy by query position
   - If token strings are available in metadata, include decoded token text next to positions.
   - Add a simple matplotlib plot:
     query position vs entropy
     span mass bar plot for the last token
   - Do not use seaborn.

6. Critical-token inspection
   - Add a cell where I can manually set:
     SPECIAL_POSITIONS = [...]
     NEEDLE_SPANS = {"needle_1": [(start, end)], ...}
     QUERY_POSITIONS = [...]
   - Re-run analyzer using these positions/spans.
   - Print a compact table:
     layer, head, query_position, span_name, attention_mass.

7.  Sanity checks
   - Check that probabilities for a query row approximately sum to 1 if the script exposes row probabilities.
   - Check that future positions are masked for causal attention.
   - Check that span masses are between 0 and 1.
   - Check that entropy is nonnegative and at most log(number of valid keys), up to numerical tolerance.
   - Add warnings if something looks wrong.

8.  Optional validation cell
   - For a very short input only, optionally compare reconstructed attention against Hugging Face output_attentions=True with attn_implementation="eager".
   - Make this clearly optional and do not run it by default.
   - Explain that this is only for short-context validation because output_attentions=True materializes the full attention matrix.

Style requirements:
- Keep notebook cells small and readable.
- Add markdown explanations before each major section.
- The notebook should be safe to run top-to-bottom after a cache has been generated.
- Do not change scripts/analyze_qk_qwen3.py or scripts/capture_qk_qwen3.py.
- Use Qwen/Qwen3-8B throughout.
- Use sdpa, not flash_attention_2, for any notebook-side model loading.