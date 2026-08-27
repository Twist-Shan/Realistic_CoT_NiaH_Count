from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_EVIDENCE_SENTINEL = (
    REPO_ROOT
    / "reports"
    / "v5_native_final_localizers"
    / "analysis"
    / "qwen_final_merged_dose_grid.json"
)


@pytest.mark.skipif(
    not EXTERNAL_EVIDENCE_SENTINEL.exists(),
    reason="external Native-thinking evidence bundle is not installed",
)
def test_restructured_report_is_audited_defined_and_nonthinking_ordered(tmp_path: Path) -> None:
    output = tmp_path / "native.html"
    manifest = tmp_path / "manifest.json"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_v5_native_thinking_report_restructured.py"),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    text = output.read_text(encoding="utf-8")
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    section_ids = (
        "definitions",
        "summary",
        "design",
        "task",
        "representation",
        "retrieval",
        "write",
        "answer",
        "walkthrough",
        "comparison",
        "appendix",
        "audit",
    )
    offsets = [text.index(f'<section id="{section_id}">') for section_id in section_ids]
    assert offsets == sorted(offsets)
    assert "Qwen3-8B · frozen Top-128" in text
    assert "Gemma4-E4B · frozen Top-6" in text
    assert "20 discovery / 10 confirmation" in text
    assert "descriptive null" in text
    assert "does not walk 1→10" in text
    assert "natural end-to-end sufficiency 仍未证明" in text
    assert "BalancedAccuracy" in text
    assert "PC1, PC2, PC3" in text
    assert 'id="native-running-canvas"' in text
    assert 'id="native-final-canvas"' in text
    assert 'id="native-geometry-model"' in text
    assert 'id="native-running-layer"' in text
    assert 'id="native-final-layer"' in text
    assert "class NativePointCloud3D" in text
    assert "const NATIVE_GEOMETRY=" in text
    assert "frozen default" in text
    assert '"default_layer":18,"layers":{"0":' in text
    assert '"default_layer":34,"layers":{"0":' in text
    assert "实验目的" in text
    assert "简单例子" in text
    assert "5.1–5.2 共用什么样本与实验底座" in text
    assert "5.1 关闭 targeted bank 后，检索结果有没有写入 grammar carrier" in text
    assert "5.2 在同一 query damage 下，恢复 clean carrier 能否救回 item-end commit" in text
    assert "30 条独立 traces × 7 arms = 210 condition rows" in text
    assert "Qwen · discovery" in text
    assert "N=3 × 1 · N=5 × 1 · N=6 × 1 · N=7 × 3 · N=8 × 2 · N=9 × 5 · N=10 × 7" in text
    assert "Gemma · confirmation" in text
    assert "N=2 × 1 · N=4 × 1 · N=5 × 1 · N=8 × 2 · N=9 × 3 · N=10 × 2" in text
    assert "Qwen 在 post-block L19–L35 读取 carrier，共 17 层" in text
    assert "Qwen 在 L19–L34、Gemma 在 L16–L40" in text
    assert "5.1b NCC：carrier 不只是“变了”，是否朝错误 count centroid 移动" in text
    assert "图 3a-2 · Targeted-head mask 对 NCC count geometry 的影响" in text
    assert "5.1c 直接 count-output margin：绕过 NCC centroid，检验最终答案分布" in text
    assert "图 3a-3 · Retrieval-query mask 对最终 count-output margin 的影响" in text
    assert "四支 clean candidate accuracy 都是 1.000" in text
    assert "Qwen：直接答案读出有效，但没有 selected-bank-specific damage" in text
    assert "Gemma：最终答案 margin 出现小的 selected-specific signal" in text
    assert "5.1a 的 carrier RMS deformation 才是“局部 hidden state 是否被改变”的主终点" in text
    assert "没有任何一行满足" in text
    assert "registered existing-split retrospective extension" in text
    assert "为什么重新分支运行" in text
    assert "pre_marker_state" in text
    assert "Qwen City→rank 19/10、Rank→city 19/9" in text
    assert "Gemma 18/9、19/10" in text
    assert "City→rank：更好的位置已经找到，但结果仍是 null" in text
    assert "Qwen 的 clean confirmation readout 无效" in text
    assert "历史 frozen pooled NCC 与事后 layer×timing 诊断" in text
    assert "Qwen/Gemma 对应性审计：数据可复现，但不是同剂量、同 cohort 的模型比较" in text
    assert "同一 frozen layer 内拆开 timing 后，两模型呈现相同的符号结构" in text
    assert "Qwen frozen L23 实际只有 24/128 selected heads 已能影响读数" in text
    assert "旧 NPZ 的 condition metadata 中" in text
    assert "原 pooled headline 的符号差异来自 timing/layer/scale mixture" in text
    assert "图 3a · 关掉 targeted heads 后 carrier hidden state 如何变化" in text
    assert "图 3b 只回答两件事" in text
    assert "图 3b · 同一 head damage 下，clean carrier 能否救回 later commit" in text
    assert "约恢复原 damage 的 58.5% 和 51.3%" in text
    assert "5.2 main rescue · clean carrier → commit" in text
    assert "5.2 harsh control · semantic vs ordinary state" not in text
    assert "为什么 Qwen 的 matched-control 数值 +9.42" not in text
    assert "Appendix D · 失败 control 审计" in text
    assert "ordinary-state arm 不提供有效 specificity 证据" in text
    assert "5.3 Commit state 是否决定下一次 targeted query 读向哪里" in text
    assert "Discovery 每模型 20×6=120 pairs" in text
    assert "Norm-matched orthogonal patch" in text
    assert "Gemma 看起来较小，但不是主效应失败" in text
    assert "Qwen bank 有 128 heads，Gemma 只有 6 heads" in text
    assert "targeted retrieval→carrier→commit→next targeted retrieval" in text
    assert "图 5a panel / 横轴" in text
    assert "左 · cumulative" in text
    assert "横轴为何不是一条简单的“擦得越来越多”剂量轴" in text
    assert "Qwen 10 confirmation seeds / 45 requests，Gemma 9 seeds / 30 requests" in text
    assert "每条 trace 有 clean + uninformative + 5 semantic restores + 5 matched-random restores = 12 arms" in text
    assert "对候选答案字符串 1,…,10 做 teacher-forced sequence scoring" in text
    assert "为什么总览只写 controlled only？" in text
    assert "它不是“没有效果”，而是“效果适用范围较窄”" in text
    assert "confirmed†" in text
    assert "local matched-control specificity" in text
    assert "Raw attention mass 与 bank-summed mass" in text
    assert "图 2c · 可切换的 Native targeted-retrieval attention maps" in text
    assert "图 2d · Qwen Top-128 layer×head atlas（全宽）" in text
    assert "图 2e · Gemma Top-6 layer×head atlas（L0–20 / L21–41 分栏放大）" in text
    assert 'class="attention-atlas-stack"' in text
    assert "frozen Top-6" in text
    assert "data-attention-selector" in text
    assert "Appendix E · 其他 grammar 的 attention maps" in text
    assert "8 张 SVG 已内嵌在本 HTML" in text
    assert "图 E8 · Gemma structural-invariant bullet · target ordinal×ranked head" in text
    assert "Gold N=10，我们想测试“只恢复第 4 个 item 是否让答案走向 4”" in text
    assert "7.2 无显式 running index 的 20/10-seed old-HTML restoration" in text
    assert "图 6b · 无 running-index trace 的 full-item state 是否推动 early-stop k" in text
    assert "两模型都未达到预注册 old-HTML magnitude gate" in text
    for figure_index in range(1, 9):
        assert f'id="figure-e{figure_index}"' in text
    appendix_e = text[text.index('id="appendix-e"') : text.index("Appendix E 结论")]
    assert len(re.findall(r"<svg(?:\s|>)", appendix_e)) == 8
    assert text.count('<details class="appendix-block">') == 5
    assert '<details class="appendix-block" open>' not in text
    assert "Gemma4-E4B · frozen Top-8" not in text
    assert "完整串行链获得confirmation" not in text
    assert len(re.findall(r"<svg(?:\s|>)", text)) == text.count("</svg>")
    assert len(text.encode("utf-8")) < 4_500_000
    assert "Schema: realistic_niah_v5_native_thinking_restructured_v7" in text

    assert payload["status"] == "PASS"
    assert payload["scientific_contract"] == {
        "discovery_seed_count": 20,
        "confirmation_seed_count": 10,
        "outcome_blind": True,
        "selection_rank_used": False,
        "qwen_targeted_bank": 128,
        "gemma_targeted_bank": 6,
    }
    assert payload["claim_scope"]["recurrent_pathway_supported"] is True
    assert payload["claim_scope"]["exclusive_circuit_claimed"] is False
    assert payload["claim_scope"]["natural_end_to_end_single_state_sufficiency"] is False
    assert payload["claim_scope"]["single_seed_walkthrough_inferential"] is False
    assert payload["claim_scope"]["gemma_commit_to_query_direct_effect_confirmed"] is True
    assert payload["claim_scope"]["gemma_commit_to_query_local_specificity_qualified"] is True
    assert payload["claim_scope"]["gemma_narrow_pre_o_query_mediation_confirmed"] is False
    assert payload["claim_scope"]["qwen_free_running_terminal_restoration_confirmed"] is False
    assert payload["claim_scope"]["ncc_frozen_results_independently_reproduced"] is True
    assert payload["claim_scope"]["ncc_timing_stratified_recapture_complete"] is True
    assert payload["claim_scope"]["ncc_city_to_rank_marker_tokens_excluded"] is True
    assert payload["claim_scope"]["ncc_bank_matched_timing_raw_direction_corresponds_across_models"] is True
    assert payload["claim_scope"]["ncc_cross_model_effect_size_comparison_allowed"] is False
    assert payload["claim_scope"]["ncc_model_by_mask_interaction_confirmed"] is False
    assert payload["claim_scope"]["qwen_historical_ncc_pooled_directionally_supported"] is True
    assert payload["claim_scope"]["qwen_stratified_rank_to_city_readout_validity_pass"] is False
    assert payload["claim_scope"]["gemma_ncc_pooled_directionally_supported"] is False
    assert payload["claim_scope"]["gemma_stratified_rank_to_city_directional_specific_only"] is True
    assert payload["claim_scope"]["ncc_marker_free_city_to_rank_damage_supported"] is False
    assert payload["claim_scope"]["ncc_readout_validity_gate_post_analysis"] is True
    assert payload["claim_scope"]["qwen_full_bank_to_late_ncc_confirmed"] is False
    assert payload["claim_scope"]["gemma_full_bank_to_late_ncc_confirmed"] is False
    assert payload["claim_scope"]["gemma_l17_rank_before_midlayer_damage_exploratory"] is True
    assert payload["claim_scope"]["direct_count_output_margin_complete"] is True
    assert payload["claim_scope"]["direct_margin_all_clean_readouts_valid"] is True
    assert payload["claim_scope"]["qwen_direct_margin_directional_specific_supported"] is False
    assert payload["claim_scope"]["gemma_direct_margin_both_timings_directional_specific"] is True
    assert payload["claim_scope"]["gemma_direct_margin_interval_confirmed"] is False
    assert payload["claim_scope"]["direct_margin_confirmation_pristine_prospective"] is False
    assert payload["claim_scope"]["direct_margin_model_by_mask_interaction_tested"] is False
    assert payload["claim_scope"]["no_running_index_count_signal_confirmed"] is True
    assert payload["claim_scope"]["no_running_index_single_span_strong_sufficiency"] is False
    assert payload["claim_scope"]["no_running_index_panel_natural_generation"] is False
    assert payload["claim_scope"]["all_layer_pca3_is_descriptive"] is True
    assert payload["schema_version"] == "realistic_niah_v5_native_thinking_restructured_v7"
    assert "geometry_3d" in payload["derived_display_data_sha256"]


def test_shipped_restructured_report_is_self_contained() -> None:
    report = REPO_ROOT / "reports" / "NiaH_Native-Thinking_report.html"
    text = report.read_text(encoding="utf-8")
    assert '<img ' not in text
    assert len(re.findall(r"<svg(?:\s|>)", text)) == text.count("</svg>")
    assert 'id="native-running-canvas"' in text
    assert 'id="native-final-canvas"' in text
    assert "5.1–5.2 共用什么样本与实验底座" in text
    assert "5.1 关闭 targeted bank 后，检索结果有没有写入 grammar carrier" in text
    assert "5.2 在同一 query damage 下，恢复 clean carrier 能否救回 item-end commit" in text
    assert "30 条独立 traces × 7 arms = 210 condition rows" in text
    assert "图 3a · 关掉 targeted heads 后 carrier hidden state 如何变化" in text
    assert "图 3a-2 · Targeted-head mask 对 NCC count geometry 的影响" in text
    assert "图 3a-3 · Retrieval-query mask 对最终 count-output margin 的影响" in text
    assert "四支 clean candidate accuracy 都是 1.000" in text
    assert "Gemma：最终答案 margin 出现小的 selected-specific signal" in text
    assert "这个结论不依赖最终 count 是否翻转" in text
    assert "没有任何分支通过两道 interval gate" in text
    assert "City→rank：更好的位置已经找到，但结果仍是 null" in text
    assert "Qwen 的 clean confirmation readout 无效" in text
    assert "Qwen/Gemma 对应性审计：数据可复现，但不是同剂量、同 cohort 的模型比较" in text
    assert "同一 frozen layer 内拆开 timing 后，两模型呈现相同的符号结构" in text
    assert "原 pooled headline 的符号差异来自 timing/layer/scale mixture" in text
    assert "图 3b · 同一 head damage 下，clean carrier 能否救回 later commit" in text
    assert "约恢复原 damage 的 58.5% 和 51.3%" in text
    assert "Discovery 每模型 20×6=120 pairs" in text
    assert "Gemma 看起来较小，但不是主效应失败" in text
    assert "图 5a panel / 横轴" in text
    assert "Qwen 10 confirmation seeds / 45 requests，Gemma 9 seeds / 30 requests" in text
    assert "每条 trace 有 clean + uninformative + 5 semantic restores + 5 matched-random restores = 12 arms" in text
    assert "为什么总览只写 controlled only？" in text
    assert "图 6b · 无 running-index trace 的 full-item state 是否推动 early-stop k" in text
    assert "8 张 SVG 已内嵌在本 HTML" in text
    assert "图 E8 · Gemma structural-invariant bullet · target ordinal×ranked head" in text
    for figure_index in range(1, 9):
        assert f'id="figure-e{figure_index}"' in text
    appendix_e = text[text.index('id="appendix-e"') : text.index("Appendix E 结论")]
    assert len(re.findall(r"<svg(?:\s|>)", appendix_e)) == 8
    assert text.count('<details class="appendix-block">') == 5
    assert '<details class="appendix-block" open>' not in text
    assert len(text.encode("utf-8")) < 4_500_000
    assert "Schema: realistic_niah_v5_native_thinking_restructured_v7" in text
