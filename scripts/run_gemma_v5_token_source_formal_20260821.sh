#!/usr/bin/env bash
set -euo pipefail

task_code_root=/home/ubuntu/CoT-Native-thinking-v5/code/Realistic_CoT_NiaH_Count
task_venv=/home/ubuntu/CoT-Native-thinking-v5/venv
task_run_root=/home/ubuntu/CoT-Native-thinking-v5/runs/v5_token_level_ablation_20260821/Gemma4-E4B
task_cache_root=/home/ubuntu/CoT-Native-thinking-v5/hf_cache
task_generations=work/v5_trace_parser_v2/Gemma4-E4B_generations_reparsed.jsonl
task_mechanism_config=configs/realistic_niah_v5_native_count_stream_confirmation_v1.json
task_v5_config=configs/realistic_niah_v5.json
task_formal_root=/home/ubuntu/CoT-Native-thinking-v5/runs/v5_native_grammar_specific_p0_20260820/Gemma4-E4B
task_target_plan=${task_formal_root}/causal_plan_adjacent_rank_after_city_p0_local_seed_event_k8_fullpanel_v1/retrieval_anchor_bank_plan.csv
task_target_registry=${task_formal_root}/head_behavior_adjacent_rank_after_city_p0_k8_fullpanel_pergrammarregistry_v2_v1/selected_anchor_registry.jsonl

task_capture=${task_run_root}/answer_broad_discovery_capture_all20_v1
task_trace_plan=${task_run_root}/answer_trace_items_top32_plan_all20_v1
task_prompt_plan=${task_run_root}/answer_prompt_records_top32_plan_all20_v1
task_trace_answer=${task_run_root}/answer_tracebank_top32_confirmation_all20_v1
task_prompt_answer=${task_run_root}/answer_promptbank_top32_confirmation_all20_v1
task_targeting=${task_run_root}/targeting_adj_p0_k8_confirmation_v1
task_logs=${task_run_root}/logs

mkdir -p "${task_logs}"
test -x "${task_venv}/bin/python"
source "${task_venv}/bin/activate"
cd "${task_code_root}"
export PYTHONDONTWRITEBYTECODE=1

task_failed() {
  task_exit=$?
  python3 -c 'import json,pathlib,sys; pathlib.Path(sys.argv[1]).write_text(json.dumps({"status":"FAIL","exit_code":int(sys.argv[2])},indent=2)+"\n")' \
    "${task_run_root}/gemma_token_source_formal_complete.json" "${task_exit}"
  exit "${task_exit}"
}
trap task_failed ERR

test -s "${task_generations}"
test -s "${task_target_plan}"
test -s "${task_target_registry}"

echo "[stage] launch 20-seed answer capture on GPU0"
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_realistic_niah_v5_count_stream.py capture-broad \
  --mechanism-config "${task_mechanism_config}" \
  --v5-config "${task_v5_config}" \
  --model Gemma4-E4B \
  --cache-dir "${task_cache_root}" \
  --device-map cuda:0 \
  --torch-dtype bfloat16 \
  --attention-backend sdpa \
  --generations "${task_generations}" \
  --seed-role development \
  --cohort parser_hit \
  --row-panel all \
  --output "${task_capture}" \
  >"${task_logs}/answer_capture.log" 2>&1 &
task_capture_pid=$!

echo "[stage] launch frozen P0 Top-8 targeting attribution on GPU1"
CUDA_VISIBLE_DEVICES=1 python3 scripts/run_realistic_niah_v5_token_level_ablation.py \
  --mode targeting \
  --config "${task_v5_config}" \
  --generations "${task_generations}" \
  --output "${task_targeting}" \
  --model Gemma4-E4B \
  --cache-dir "${task_cache_root}" \
  --device-map cuda:0 \
  --torch-dtype bfloat16 \
  --attention-backend sdpa \
  --bank-plan "${task_target_plan}" \
  --bank-size 8 \
  --anchor-role p0_item_end \
  --target-grammar-class adjacent_rank_after_city \
  --anchor-registry "${task_target_registry}" \
  --split confirmation \
  --seeds 1254 1255 1256 1257 1258 1259 1260 1261 1262 1263 \
  --matched-control-repeats 3 \
  --run-greedy \
  --max-new-tokens 32 \
  >"${task_logs}/targeting.log" 2>&1 &
task_target_pid=$!

wait "${task_capture_pid}"
echo "[stage] freeze trace-items and prompt-records Top-32 banks"
python3 scripts/run_realistic_niah_v5_count_stream.py plan-broad \
  --mechanism-config "${task_mechanism_config}" \
  --captures "${task_capture}" \
  --model Gemma4-E4B \
  --source-group trace_items \
  --bank-sizes 32 \
  --use-all-development-seeds \
  --random-seed 20260821 \
  --output "${task_trace_plan}" \
  >"${task_logs}/trace_plan.log" 2>&1
python3 scripts/run_realistic_niah_v5_count_stream.py plan-broad \
  --mechanism-config "${task_mechanism_config}" \
  --captures "${task_capture}" \
  --model Gemma4-E4B \
  --source-group prompt_records \
  --bank-sizes 32 \
  --use-all-development-seeds \
  --random-seed 20260821 \
  --output "${task_prompt_plan}" \
  >"${task_logs}/prompt_plan.log" 2>&1

echo "[stage] launch trace-bank answer confirmation on GPU0"
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_realistic_niah_v5_token_level_ablation.py \
  --mode answer \
  --config "${task_v5_config}" \
  --generations "${task_generations}" \
  --output "${task_trace_answer}" \
  --model Gemma4-E4B \
  --cache-dir "${task_cache_root}" \
  --device-map cuda:0 \
  --torch-dtype bfloat16 \
  --attention-backend sdpa \
  --bank-plan "${task_trace_plan}/answer_broad_head_plan.csv" \
  --bank-size 32 \
  --split confirmation \
  --seeds 1254 1255 1256 1257 1258 1259 1260 1261 1262 1263 \
  --max-new-tokens 32 \
  >"${task_logs}/trace_answer.log" 2>&1 &
task_trace_pid=$!

wait "${task_target_pid}"
echo "[stage] launch prompt-bank answer confirmation on GPU1"
CUDA_VISIBLE_DEVICES=1 python3 scripts/run_realistic_niah_v5_token_level_ablation.py \
  --mode answer \
  --config "${task_v5_config}" \
  --generations "${task_generations}" \
  --output "${task_prompt_answer}" \
  --model Gemma4-E4B \
  --cache-dir "${task_cache_root}" \
  --device-map cuda:0 \
  --torch-dtype bfloat16 \
  --attention-backend sdpa \
  --bank-plan "${task_prompt_plan}/answer_broad_head_plan.csv" \
  --bank-size 32 \
  --split confirmation \
  --seeds 1254 1255 1256 1257 1258 1259 1260 1261 1262 1263 \
  --max-new-tokens 32 \
  >"${task_logs}/prompt_answer.log" 2>&1 &
task_prompt_pid=$!

wait "${task_trace_pid}"
wait "${task_prompt_pid}"

echo "[stage] registered analyses"
python3 scripts/analyze_realistic_niah_v5_token_level_ablation.py \
  --input "${task_targeting}" \
  --output "${task_targeting}/analysis_registered_v1" \
  >"${task_logs}/analyze_targeting.log" 2>&1
python3 scripts/analyze_realistic_niah_v5_token_level_ablation.py \
  --input "${task_trace_answer}" \
  --output "${task_trace_answer}/analysis_registered_v1" \
  >"${task_logs}/analyze_trace_answer.log" 2>&1
python3 scripts/analyze_realistic_niah_v5_token_level_ablation.py \
  --input "${task_prompt_answer}" \
  --output "${task_prompt_answer}/analysis_registered_v1" \
  >"${task_logs}/analyze_prompt_answer.log" 2>&1

python3 -c 'import json,pathlib,sys; root=pathlib.Path(sys.argv[1]); expected={"answer_broad_discovery_capture_all20_v1":200,"answer_tracebank_top32_confirmation_all20_v1":100,"answer_promptbank_top32_confirmation_all20_v1":100,"targeting_adj_p0_k8_confirmation_v1":30}; patterns={"answer_broad_discovery_capture_all20_v1":"*.csv","answer_tracebank_top32_confirmation_all20_v1":"*.jsonl","answer_promptbank_top32_confirmation_all20_v1":"*.jsonl","targeting_adj_p0_k8_confirmation_v1":"*.jsonl"}; actual={name:len(list((root/name/"shards").glob(patterns[name]))) for name in expected}; audits={name:json.loads((root/name/"analysis_registered_v1"/"analysis_audit.json").read_text())["status"] for name in expected if name!="answer_broad_discovery_capture_all20_v1"}; ok=actual==expected and set(audits.values())=={"PASS"}; payload={"status":"PASS" if ok else "FAIL","expected_shards":expected,"actual_shards":actual,"analysis_status":audits}; (root/"gemma_token_source_formal_complete.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); print(json.dumps(payload,indent=2)); sys.exit(0 if ok else 1)' "${task_run_root}"

task_archive=/tmp/gemma_token_source_formal_20260821_v1.tgz
task_raw_archive=/tmp/gemma_answer_source_discovery_raw_20260821_v1.tgz
tar -C "${task_run_root}" --exclude='answer_broad_discovery_capture_all20_v1/shards' -czf "${task_archive}" \
  answer_broad_discovery_capture_all20_v1 \
  answer_trace_items_top32_plan_all20_v1 \
  answer_prompt_records_top32_plan_all20_v1 \
  answer_tracebank_top32_confirmation_all20_v1 \
  answer_promptbank_top32_confirmation_all20_v1 \
  targeting_adj_p0_k8_confirmation_v1 \
  gemma_token_source_formal_complete.json \
  logs
tar -C "${task_run_root}" -czf "${task_raw_archive}" answer_broad_discovery_capture_all20_v1/shards
sha256sum "${task_archive}" "${task_raw_archive}"
echo "[complete] Gemma token-source formal experiment PASS"
