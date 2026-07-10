I want to test the scripts "capture_qk_qwen3.py" in folder "dataset_generation/qk_hook_attention" using Colab. Could you set up a simple notebook (smoke-test) for me to test on Qwen/Qwen3-8B. You can refer to "analysis_hidden_states_v4.ipynb" and perhaps copy some starting code blocks. When implementing the following, use separate code blocks if necessary.

Requirements:
1. Install/check dependencies: torch, transformers, etc. Check whether they are preinstalled, if no, 
2. Print GPU name and memory using nvidia-smi.
3. Log in to Hugging Face if needed, but do not hard-code tokens.
4. Use a short prompt first, around 128 tokens.
5. Run capture_qk_qwen3.py with target layers [0, 20, 35], save_dtype bfloat16, output into a subdirectory qk_cache_test.
6. After running, list saved files.
7. Load metadata.json and print important fields.
8. Load layer_00_q_raw.pt, layer_00_k_raw.pt, layer_35_q_raw.pt, layer_35_k_raw.pt and print shape, dtype, device, min/max/norm summary.
9. Add a cell that estimates expected tensor shapes for Qwen3-8B:
   q_raw: [1, T, 4096]
   k_raw: [1, T, 1024]
11. Keep the notebook simple and Do not change capture_qk_qwen3.py.