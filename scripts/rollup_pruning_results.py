"""Roll up OpenVLA pruning ablation results.

This script is read-only with respect to model inference: it only reads each
method directory's summary / episode / step metrics and writes aggregate tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional


ROLLUP_FIELDS = [
    "method",
    "task_suite",
    "num_tasks",
    "num_trials",
    "num_episodes",
    "num_successes",
    "num_failures",
    "success_rate",
    "mean_episode_steps",
    "mean_episode_steps_success_only",
    "mean_episode_steps_failure_only",
    "mean_step_time_ms",
    "mean_step_time_ms_episode_std",
    "mean_model_forward_time_ms",
    "mean_model_forward_time_ms_episode_std",
    "mean_effective_inference_ms",
    "speedup_vs_baseline",
    "success_delta_vs_baseline",
    "mean_hook_total_ms",
    "mean_hook_total_ms_excluding_warmup",
    "p50_hook_total_ms",
    "p95_hook_total_ms",
    "max_hook_total_ms",
    "mean_token_mapping_ms",
    "mean_score_compute_ms",
    "mean_selection_ms",
    "mean_gather_ms",
    "mean_keep_ratio",
    "requested_keep_ratio_mean",
    "keep_ratio_source_distribution",
    "effective_keep_count_mean",
    "original_token_count_mean",
    "mean_visual_tokens_kept",
    "mean_num_visual_tokens_original",
    "mean_num_visual_tokens_kept",
    "dynamic_keep_ratio_mean",
    "dynamic_keep_ratio_std",
    "geo_risk_level_distribution",
    "geo_risk_low_ratio",
    "geo_risk_medium_ratio",
    "geo_risk_high_ratio",
    "geo_risk_interaction_lock_ratio",
    "interaction_lock_ratio",
    "valid_token_ratio",
    "valid_3d_token_ratio",
    "fallback_rate",
    "fallback_ratio",
    "fallback_reason_counts",
    # --- P6 selection diagnostics ---
    "selector_name_counts",
    "selector_function_name_counts",
    "selection_strategy_name_counts",
    "selection_stage_name_counts",
    "selector_success_ratio",
    "keep_indices_sorted_ratio",
    "keep_indices_unique_ratio",
    "keep_indices_out_of_bounds_ratio",
    "selection_error_counts",
    "selection_warning_counts",
    "keep_ratio_requested_mean",
    "keep_ratio_actual_mean",
    "retention_actual_mean",
    "num_visual_tokens_dropped_mean",
    "num_visual_tokens_original_total_mean",
    "num_visual_tokens_kept_total_mean",
    "selected_by_phase1_mean",
    "selected_by_phase2_mean",
    "selected_by_phase3_mean",
    "selected_by_fill_mean",
    "selected_by_fallback_mean",
    "selected_unattributed_mean",
    "phase_accounting_valid_ratio",
    "phase_accounting_error_counts",
    "reserved_edge_dropped_ratio_mean",
    "non_reserved_topk_count_mean",
    "non_reserved_kept_count_mean",
    "non_reserved_dropped_count_mean",
    "non_reserved_dropped_ratio_mean",
    "total_keep_budget_mean",
    "depth_edge_budget_mean",
    "robot_geo_budget_mean",
    "fill_budget_mean",
    "safety_budget_mean",
    "score_min_mean",
    "score_max_mean",
    "score_mean_mean",
    "score_std_mean",
    "phase_ratio_far",
    "phase_ratio_mid",
    "phase_ratio_near",
    "phase_ratio_fallback_safe",
    "mean_K_edge_actual",
    "mean_K_geo_actual",
    "mean_K_diverse_actual",
    "mean_geo_contact_score_topk",
    "mean_near_contact_score_topk",
    "mean_corridor_contact_score_topk",
    "mean_edge_score_ms",
    "mean_robot_mapping_ms",
    "mean_near_score_ms",
    "mean_corridor_score_ms",
    "mean_contact_score_ms",
    "mean_edge_selection_ms",
    "mean_geo_selection_ms",
    "mean_diverse_selection_ms",
    "mean_final_merge_ms",
    "max_delta_action",
    "max_jerk_action",
    # --- diagnostic fields v2 ---
    "action_has_nan_ratio",
    "action_has_inf_ratio",
    "action_min",
    "action_max",
    "action_mean",
    "action_std",
    "depth_valid_ratio",
    "depth_min",
    "depth_max",
    "depth_mean",
    "points_robot_min_x",
    "points_robot_min_y",
    "points_robot_min_z",
    "points_robot_max_x",
    "points_robot_max_y",
    "points_robot_max_z",
    "points_robot_mean_x",
    "points_robot_mean_y",
    "points_robot_mean_z",
    "points_robot_std_x",
    "points_robot_std_y",
    "points_robot_std_z",
    "points_cam_min_x",
    "points_cam_min_y",
    "points_cam_min_z",
    "points_cam_max_x",
    "points_cam_max_y",
    "points_cam_max_z",
    "points_cam_available",
    "points_cam_unavailable_reason",
    "num_valid_3d_tokens",
    "distance_to_gripper_min",
    "distance_to_gripper_mean",
    "distance_to_gripper_max",
    "robot_state_valid_ratio",
    "motion_direction_valid_ratio",
    "ee_position_x_mean",
    "ee_position_y_mean",
    "ee_position_z_mean",
    "contact_risk_lock_ratio",
    "gripper_proximity_lock_ratio",
    "region_stability_lock_ratio",
    # --- score component distribution stats ---
    "depth_edge_score_mean",
    "depth_edge_score_std",
    "depth_edge_score_min",
    "depth_edge_score_p50",
    "depth_edge_score_p90",
    "depth_edge_score_max",
    "depth_edge_score_positive_ratio",
    "distance_score_mean",
    "distance_score_std",
    "distance_score_min",
    "distance_score_p50",
    "distance_score_p90",
    "distance_score_max",
    "motion_cone_score_mean",
    "motion_cone_score_std",
    "motion_cone_score_min",
    "motion_cone_score_p50",
    "motion_cone_score_p90",
    "motion_cone_score_max",
    "motion_cone_score_positive_ratio",
    "motion_cone_score_zero_ratio",
    "motion_dir_norm_mean",
    "motion_dir_norm_min",
    "motion_dir_norm_max",
    "workspace_score_mean",
    "workspace_score_std",
    "workspace_score_max",
    "contact_risk_score_mean",
    "contact_risk_score_std",
    "contact_risk_score_min",
    "contact_risk_score_p50",
    "contact_risk_score_p90",
    "contact_risk_score_max",
    "final_geometry_score_mean",
    "final_geometry_score_std",
    "final_geometry_score_min",
    "final_geometry_score_p50",
    "final_geometry_score_p90",
    "final_geometry_score_max",
    "selected_depth_edge_score_mean",
    "selected_distance_score_mean",
    "selected_motion_cone_score_mean",
    "selected_workspace_score_mean",
    "selected_contact_risk_score_mean",
    "selected_final_score_mean",
    # Hybrid quota union stats
    "selected_by_depth_edge_count",
    "selected_by_robot_geo_count",
    "selected_by_uniform_count",
    "selected_by_fill_count",
    "depth_edge_quota_count",
    "robot_geo_quota_count",
    "uniform_quota_count",
    "fill_count",
    "overlap_depth_robot_geo",
    # --- hybrid fix: new score distribution diagnostics ---
    "contact_risk_top1_mean",
    "contact_risk_top5_mean",
    "contact_risk_top10_mean",
    "contact_risk_p95",
    "contact_risk_p99",
    "motion_cone_top5_mean",
    "motion_cone_score_max",
    "motion_cone_score_zero_ratio",
    "distance_score_top5_mean",
    "final_score_top5_mean",
    "final_score_concentration",
    "selected_contact_risk_score_mean",
    "selected_contact_risk_score_p50",
    "selected_contact_risk_score_p90",
    "selected_contact_risk_score_max",
    "selected_motion_cone_score_mean",
    "selected_distance_score_mean",
    "selected_final_score_mean",
    # --- Hybrid v1 new fields ---
    "w_edge",
    "w_near",
    "w_contact",
    "w_corr",
    "w_diverse",
    "edge_score_mean",
    "edge_score_max",
    "edge_score_std",
    "near_score_mean",
    "near_score_max",
    "near_score_std",
    "contact_score_mean",
    "contact_score_max",
    "contact_score_std",
    "corridor_score_mean",
    "corridor_score_max",
    "corridor_score_std",
    "diversity_score_mean",
    "diversity_score_max",
    "diversity_score_std",
    "final_hybrid_score_mean",
    "final_hybrid_score_max",
    "final_hybrid_score_std",
    "selected_grid_coverage_ratio",
    "grid_coverage_ratio",
    "selected_token_grid_entropy",
    # --- Temporal v1 adaptive threshold ---
    "adaptive_threshold_mean",
    "adaptive_threshold_max",
    "ema_used_for_selection",
    "score_ema_available_ratio",
    "lock_condition_failed_reason_distribution",
    # --- P0-4 depth metric stats ---
    "depth_metric_min",
    "depth_metric_max",
    "depth_metric_mean",
    "depth_metric_std",
    "depth_source_key",
    "depth_conversion",
    "depth_is_metric",
    "depth_unit",
    "depth_sim_available",
    # --- P0-4 transform metadata ---
    "transform_convention",
    "transform_inverse_used",
    "transform_source",
    "transform_convention_verified",
    "transform_convention_evidence",
    # --- P0-4 depth quality diagnostics ---
    "depth_suspicious_ratio",
    "depth_suspicious_steps",
    "missing_depth_metadata_steps",
    # --- P0-4 transform quality diagnostics ---
    "T_ambiguous_ratio",
    "T_ambiguous_steps",
    "missing_transform_metadata_steps",
    # --- P0-4 motion diagnostics ---
    "motion_norm_mean",
    "motion_norm_median",
    "motion_invalid_steps",
    "motion_valid_steps",
    "motion_invalid_but_cone_nonzero_steps",
    "motion_invalid_but_cone_nonzero_ratio",
    "missing_motion_direction_valid_steps",
    # --- P0-4 workspace diagnostics ---
    "workspace_valid_steps",
    "workspace_all_one_ratio",
    "workspace_all_one_steps",
    # --- P0-4 token selection quality ---
    "selected_robot_token_ratio",
    "retention_ratio",
    # ---- P1-1: Token selection attribution / top-k competition diagnostics ----
    "selected_token_count_mean",
    "selected_token_count_std",
    "dropped_token_count_mean",
    "dropped_token_count_std",
    "retention_ratio_mean",
    "retention_ratio_std",
    "selected_token_ratio_mean",
    "selected_token_ratio_std",
    "dropped_token_ratio_mean",
    "dropped_token_ratio_std",
    "num_visual_tokens_original_mean",
    "num_visual_tokens_kept_mean",
    "depth_edge_topk_count_mean",
    "robot_geo_topk_count_mean",
    "final_selected_count_mean",
    "depth_edge_topk_kept_in_final_count_mean",
    "robot_geo_topk_kept_in_final_count_mean",
    "depth_edge_topk_dropped_count_mean",
    "robot_geo_topk_dropped_count_mean",
    "depth_edge_topk_dropped_ratio_mean",
    "robot_geo_topk_dropped_ratio_mean",
    "overlap_depth_edge_robot_geo_count_mean",
    "overlap_depth_edge_robot_geo_ratio_mean",
    "selected_overlap_robot_geo_depth_edge_ratio_mean",
    "selected_overlap_robot_geo_motion_ratio_mean",
    "selected_overlap_robot_geo_near_ratio_mean",
    "depth_edge_topk_overlap_with_robot_geo_topk_mean",
    "robot_geo_topk_overlap_with_depth_edge_topk_mean",
    "selected_high_depth_edge_but_low_robot_geo_count_mean",
    "dropped_high_depth_edge_tokens_count_mean",
    "dropped_high_robot_geo_tokens_count_mean",
    "selected_token_u_mean_mean",
    "selected_token_u_std_mean",
    "selected_token_v_mean_mean",
    "selected_token_v_std_mean",
    "selected_token_bbox_u_min_mean",
    "selected_token_bbox_u_max_mean",
    "selected_token_bbox_v_min_mean",
    "selected_token_bbox_v_max_mean",
    "selected_token_near_gripper_pixel_dist_mean_mean",
    "selected_token_near_gripper_pixel_dist_median_mean",
    "workspace_score_min_mean",
    "workspace_score_unique_count_mean",
    "workspace_all_one_ratio",
    "workspace_all_one_steps",
    "workspace_valid_steps",
    "workspace_fallback_used_ratio",
    "workspace_valid_token_ratio_mean",
    "near_score_unique_count_mean",
    "depth_edge_score_unique_count_mean",
    "motion_cone_score_unique_count_mean",
    "final_score_unique_count_mean",
    # P5-fix: new edge_reserve metrics
    "edge_reserve_invalid_ratio",
    "edge_reserve_invalid_steps",
    "edge_scores_available_ratio",
    "edge_reserved_actual_mean",
    "edge_reserved_actual_std",
    "edge_reserved_survival_ratio_mean",
    "reserved_edge_topk_dropped_ratio_mean",
    "non_reserved_edge_topk_dropped_ratio_mean",
    "overall_depth_edge_topk_dropped_ratio_mean",
    "duplicate_after_exclusion_count_mean",
    "duplicate_with_original_hybrid_count_mean",
    # P5-fix: new accounting and invariant metrics
    "edge_reserve_accounting_valid_ratio",
    "edge_reserve_split_metrics_valid_ratio",
    "selected_by_edge_reserved_count_mean",
    "selected_by_phase1_hybrid_count_mean",
    "selected_by_phase2_diversity_count_mean",
    "selected_by_phase3_fallback_count_mean",
    "selected_by_fill_count_mean",
    "selected_by_unattributed_count_mean",
    "no_duplicate_final_indices_ratio",
    "final_indices_sorted_ratio",
    "selected_token_count_equals_kept_ratio",
    "retention_ratio_valid_ratio",
    # P11: branch_budget_v0 fields
    "branch_budget_v0_rate",
    "branch_accounting_valid_ratio",
    "branch_sum_equals_kept_ratio",
    "selected_by_depth_branch_count_mean",
    "selected_by_hybrid_branch_count_mean",
    "selected_by_fill_branch_count_mean",
    "selected_by_fallback_mean",
    "overlap_depth_edge_hybrid_count_mean",
    "overlap_depth_edge_hybrid_ratio_mean",
    "depth_edge_topk_count_mean",
    "depth_edge_topk_kept_in_final_count_mean",
    "depth_edge_topk_dropped_count_mean",
    "depth_edge_topk_dropped_ratio_mean",
    "hybrid_final_score_topk_count_mean",
    "hybrid_final_score_topk_kept_count_mean",
    "hybrid_final_score_topk_dropped_count_mean",
    "hybrid_final_score_topk_dropped_ratio_mean",
    "non_reserved_depth_edge_dropped_ratio_mean",
    "depth_edge_score_mean_mean",
    "hybrid_score_mean_mean",
    "final_hybrid_score_mean_mean",
    # P11.3: DE top-k attribution by branch
    "depth_edge_topk_kept_by_depth_branch_count_mean",
    "depth_edge_topk_kept_by_hybrid_branch_count_mean",
    "depth_edge_topk_kept_by_fill_branch_count_mean",
    "depth_edge_topk_kept_by_fallback_count_mean",
    "depth_edge_topk_survival_ratio_mean",
    # P11.3: Hybrid top-k attribution by branch
    "hybrid_final_score_topk_kept_by_depth_branch_count_mean",
    "hybrid_final_score_topk_kept_by_hybrid_branch_count_mean",
    "hybrid_final_score_topk_kept_by_fill_branch_count_mean",
    "hybrid_final_score_topk_kept_by_fallback_count_mean",
    "hybrid_final_score_topk_survival_ratio_mean",
    # P11.3: Legacy hybrid top-k aliases (P8 compat)
    "hybrid_topk_kept_by_depth_branch_count_mean",
    "hybrid_topk_kept_by_hybrid_branch_count_mean",
    "hybrid_topk_kept_by_fill_branch_count_mean",
    "hybrid_topk_kept_by_fallback_count_mean",
    "hybrid_topk_survival_ratio_mean",
]


def build_rollup(root: Path, methods: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
    root = Path(root)
    method_filter = set(methods or [])
    rows: List[Dict[str, Any]] = []
    if not root.exists():
        raise FileNotFoundError(f"Result root does not exist: {root}")

    for method_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        method = method_dir.name
        if method_filter and method not in method_filter:
            continue
        summary_path = method_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = _read_json(summary_path)
        step_records = _read_jsonl(method_dir / "per_step_metrics.jsonl")
        if not step_records:
            step_records = _read_step_csv(method_dir / "step_metrics.csv")
        row = summarize_method(method, summary, step_records)
        rows.append(row)
    _add_baseline_comparisons(rows)
    return rows


def summarize_method(method: str, summary: Dict[str, Any], step_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    model_ms = _first_number(summary.get("mean_model_forward_time_ms"), summary.get("mean_model_forward_ms"))
    hook_ms = _num(summary.get("mean_hook_total_ms"))
    if hook_ms is None:
        computed_effective_ms = model_ms
    elif model_ms is None:
        computed_effective_ms = None
    else:
        computed_effective_ms = model_ms + hook_ms
    effective_ms = _first_number(summary.get("mean_effective_inference_ms"), computed_effective_ms)

    fallback_rate = _fallback_rate(step_records)
    interaction_lock_ratio = _bool_rate(step_records, "interaction_lock")
    dynamic_stats = _mean_std_nullable(_numbers_from_records(step_records, "dynamic_keep_ratio"))
    risk_distribution = _categorical_distribution(step_records, "geo_risk_level")
    risk_ratio = risk_distribution or {}
    valid_3d_ratio = _first_number(
        summary.get("valid_3d_token_ratio"),
        summary.get("valid_token_ratio_mean"),
        summary.get("valid_depth_ratio"),
        _mean_nullable(_numbers_from_records(step_records, "valid_token_ratio")),
        _mean_nullable(_numbers_from_records(step_records, "valid_depth_ratio")),
    )
    mean_keep_ratio = _first_number(
        summary.get("mean_keep_ratio"),
        summary.get("actual_keep_ratio_mean"),
        summary.get("token_retention_ratio"),
    )
    mean_tokens_kept = _first_number(
        summary.get("num_visual_tokens_kept_mean"),
        summary.get("num_visual_tokens_kept"),
        summary.get("mean_visual_tokens_kept"),
    )

    row = {
        "method": method,
        "task_suite": summary.get("task_suite"),
        "num_tasks": summary.get("num_tasks_evaluated"),
        "num_trials": summary.get("num_trials_per_task"),
        "num_episodes": summary.get("num_episodes"),
        "num_successes": summary.get("num_successes"),
        "num_failures": summary.get("num_failures"),
        "success_rate": _first_number(summary.get("overall_success_rate"), summary.get("success_rate")),
        "mean_episode_steps": _first_number(summary.get("mean_episode_steps_all"), summary.get("mean_episode_steps"), summary.get("mean_steps")),
        "mean_episode_steps_success_only": summary.get("mean_episode_steps_success_only"),
        "mean_episode_steps_failure_only": summary.get("mean_episode_steps_failure_only"),
        "mean_step_time_ms": _first_number(summary.get("mean_step_wall_time_ms"), summary.get("mean_step_time_ms")),
        "mean_step_time_ms_episode_std": summary.get("std_step_wall_time_ms"),
        "mean_model_forward_time_ms": model_ms,
        "mean_model_forward_time_ms_episode_std": summary.get("std_model_forward_time_ms"),
        "mean_effective_inference_ms": effective_ms,
        "speedup_vs_baseline": None,
        "success_delta_vs_baseline": None,
        "mean_hook_total_ms": hook_ms,
        "mean_hook_total_ms_excluding_warmup": summary.get("mean_hook_total_ms_excluding_warmup"),
        "p50_hook_total_ms": summary.get("p50_hook_total_ms"),
        "p95_hook_total_ms": summary.get("p95_hook_total_ms"),
        "max_hook_total_ms": summary.get("max_hook_total_ms"),
        "mean_token_mapping_ms": summary.get("mean_token_mapping_time_ms"),
        "mean_score_compute_ms": summary.get("mean_score_compute_ms") or summary.get("mean_geometry_score_time_ms"),
        "mean_selection_ms": summary.get("mean_selection_ms"),
        "mean_gather_ms": summary.get("mean_gather_ms"),
        "mean_keep_ratio": mean_keep_ratio,
        "requested_keep_ratio_mean": summary.get("requested_keep_ratio_mean"),
        "keep_ratio_source_distribution": summary.get("keep_ratio_source_distribution"),
        "effective_keep_count_mean": summary.get("effective_keep_count_mean"),
        "original_token_count_mean": summary.get("original_token_count_mean"),
        "mean_visual_tokens_kept": mean_tokens_kept,
        "mean_num_visual_tokens_original": _first_number(summary.get("num_visual_tokens_original_mean"), summary.get("num_visual_tokens_original")),
        "mean_num_visual_tokens_kept": mean_tokens_kept,
        "dynamic_keep_ratio_mean": _first_number(summary.get("dynamic_keep_ratio_mean"), summary.get("mean_keep_ratio"), dynamic_stats.get("mean")),
        "dynamic_keep_ratio_std": _first_number(summary.get("dynamic_keep_ratio_std"), dynamic_stats.get("std")),
        "geo_risk_level_distribution": risk_distribution,
        "geo_risk_low_ratio": risk_ratio.get("low"),
        "geo_risk_medium_ratio": risk_ratio.get("medium"),
        "geo_risk_high_ratio": risk_ratio.get("high"),
        "geo_risk_interaction_lock_ratio": risk_ratio.get("interaction_lock"),
        "interaction_lock_ratio": _first_number(summary.get("interaction_lock_ratio"), interaction_lock_ratio),
        "valid_token_ratio": _first_number(summary.get("valid_token_ratio_mean"), summary.get("valid_depth_ratio")),
        "valid_3d_token_ratio": valid_3d_ratio,
        "fallback_rate": fallback_rate,
        "fallback_ratio": fallback_rate,
        "fallback_reason_counts": summary.get("fallback_reason_counts"),
        "selector_name_counts": summary.get("selector_name_counts") or _categorical_counts(step_records, "selector_name"),
        "selector_function_name_counts": summary.get("selector_function_name_counts") or _categorical_counts(step_records, "selector_function_name"),
        "selection_strategy_name_counts": summary.get("selection_strategy_name_counts") or _categorical_counts(step_records, "selection_strategy_name"),
        "selection_stage_name_counts": summary.get("selection_stage_name_counts") or _categorical_counts(step_records, "selection_stage_name"),
        "selector_success_ratio": _first_number(summary.get("selector_success_ratio"), _bool_rate(step_records, "selector_success")),
        "keep_indices_sorted_ratio": _first_number(summary.get("keep_indices_sorted_ratio"), _bool_rate(step_records, "keep_indices_sorted")),
        "keep_indices_unique_ratio": _first_number(summary.get("keep_indices_unique_ratio"), _bool_rate(step_records, "keep_indices_unique")),
        "keep_indices_out_of_bounds_ratio": _first_number(summary.get("keep_indices_out_of_bounds_ratio"), _bool_rate(step_records, "keep_indices_out_of_bounds")),
        "selection_error_counts": summary.get("selection_error_counts") or _categorical_counts(step_records, "selection_error"),
        "selection_warning_counts": summary.get("selection_warning_counts") or _categorical_counts(step_records, "selection_warning"),
        "keep_ratio_requested_mean": _first_number(summary.get("keep_ratio_requested_mean"), _mean_nullable(_numbers_from_records(step_records, "keep_ratio_requested"))),
        "keep_ratio_actual_mean": _first_number(summary.get("keep_ratio_actual_mean"), _mean_nullable(_numbers_from_records(step_records, "keep_ratio_actual"))),
        "retention_actual_mean": _first_number(summary.get("retention_actual_mean"), _mean_nullable(_numbers_from_records(step_records, "retention_actual"))),
        "num_visual_tokens_dropped_mean": _first_number(summary.get("num_visual_tokens_dropped_mean"), _mean_nullable(_numbers_from_records(step_records, "num_visual_tokens_dropped"))),
        "num_visual_tokens_original_total_mean": _first_number(summary.get("num_visual_tokens_original_total_mean"), _mean_nullable(_numbers_from_records(step_records, "num_visual_tokens_original_total"))),
        "num_visual_tokens_kept_total_mean": _first_number(summary.get("num_visual_tokens_kept_total_mean"), _mean_nullable(_numbers_from_records(step_records, "num_visual_tokens_kept_total"))),
        "selected_by_phase1_mean": _first_number(summary.get("selected_by_phase1_mean"), _mean_nullable(_numbers_from_records(step_records, "selected_by_phase1"))),
        "selected_by_phase2_mean": _first_number(summary.get("selected_by_phase2_mean"), _mean_nullable(_numbers_from_records(step_records, "selected_by_phase2"))),
        "selected_by_phase3_mean": _first_number(summary.get("selected_by_phase3_mean"), _mean_nullable(_numbers_from_records(step_records, "selected_by_phase3"))),
        "selected_by_fill_mean": _first_number(summary.get("selected_by_fill_mean"), _mean_nullable(_numbers_from_records(step_records, "selected_by_fill"))),
        "selected_by_fallback_mean": _first_number(summary.get("selected_by_fallback_mean"), _mean_nullable(_numbers_from_records(step_records, "selected_by_fallback"))),
        "selected_unattributed_mean": _first_number(summary.get("selected_unattributed_mean"), _mean_nullable(_numbers_from_records(step_records, "selected_unattributed"))),
        "phase_accounting_valid_ratio": _first_number(summary.get("phase_accounting_valid_ratio"), _bool_rate(step_records, "phase_accounting_valid")),
        "phase_accounting_error_counts": summary.get("phase_accounting_error_counts") or _categorical_counts(step_records, "phase_accounting_error"),
        "reserved_edge_dropped_ratio_mean": _first_number(summary.get("reserved_edge_dropped_ratio_mean"), _mean_nullable(_numbers_from_records(step_records, "reserved_edge_dropped_ratio"))),
        "non_reserved_topk_count_mean": _first_number(summary.get("non_reserved_topk_count_mean"), _mean_nullable(_numbers_from_records(step_records, "non_reserved_topk_count"))),
        "non_reserved_kept_count_mean": _first_number(summary.get("non_reserved_kept_count_mean"), _mean_nullable(_numbers_from_records(step_records, "non_reserved_kept_count"))),
        "non_reserved_dropped_count_mean": _first_number(summary.get("non_reserved_dropped_count_mean"), _mean_nullable(_numbers_from_records(step_records, "non_reserved_dropped_count"))),
        "non_reserved_dropped_ratio_mean": _first_number(summary.get("non_reserved_dropped_ratio_mean"), _mean_nullable(_numbers_from_records(step_records, "non_reserved_dropped_ratio"))),
        "total_keep_budget_mean": _first_number(summary.get("total_keep_budget_mean"), _mean_nullable(_numbers_from_records(step_records, "total_keep_budget"))),
        "depth_edge_budget_mean": _first_number(summary.get("depth_edge_budget_mean"), _mean_nullable(_numbers_from_records(step_records, "depth_edge_budget"))),
        "robot_geo_budget_mean": _first_number(summary.get("robot_geo_budget_mean"), _mean_nullable(_numbers_from_records(step_records, "robot_geo_budget"))),
        "fill_budget_mean": _first_number(summary.get("fill_budget_mean"), _mean_nullable(_numbers_from_records(step_records, "fill_budget"))),
        "safety_budget_mean": _first_number(summary.get("safety_budget_mean"), _mean_nullable(_numbers_from_records(step_records, "safety_budget"))),
        "score_min_mean": _first_number(summary.get("score_min_mean"), _mean_nullable(_numbers_from_records(step_records, "score_min"))),
        "score_max_mean": _first_number(summary.get("score_max_mean"), _mean_nullable(_numbers_from_records(step_records, "score_max"))),
        "score_mean_mean": _first_number(summary.get("score_mean_mean"), _mean_nullable(_numbers_from_records(step_records, "score_mean"))),
        "score_std_mean": _first_number(summary.get("score_std_mean"), _mean_nullable(_numbers_from_records(step_records, "score_std"))),
        "phase_ratio_far": summary.get("phase_ratio_far"),
        "phase_ratio_mid": summary.get("phase_ratio_mid"),
        "phase_ratio_near": summary.get("phase_ratio_near"),
        "phase_ratio_fallback_safe": summary.get("phase_ratio_fallback_safe"),
        "mean_K_edge_actual": summary.get("mean_K_edge_actual"),
        "mean_K_geo_actual": summary.get("mean_K_geo_actual"),
        "mean_K_diverse_actual": summary.get("mean_K_diverse_actual"),
        "mean_geo_contact_score_topk": summary.get("mean_geo_contact_score_topk"),
        "mean_near_contact_score_topk": summary.get("mean_near_contact_score_topk"),
        "mean_corridor_contact_score_topk": summary.get("mean_corridor_contact_score_topk"),
        "mean_edge_score_ms": summary.get("mean_edge_score_ms"),
        "mean_robot_mapping_ms": summary.get("mean_robot_mapping_ms"),
        "mean_near_score_ms": summary.get("mean_near_score_ms"),
        "mean_corridor_score_ms": summary.get("mean_corridor_score_ms"),
        "mean_contact_score_ms": summary.get("mean_contact_score_ms"),
        "mean_edge_selection_ms": summary.get("mean_edge_selection_ms"),
        "mean_geo_selection_ms": summary.get("mean_geo_selection_ms"),
        "mean_diverse_selection_ms": summary.get("mean_diverse_selection_ms"),
        "mean_final_merge_ms": summary.get("mean_final_merge_ms"),
        "max_delta_action": summary.get("max_delta_action"),
        "max_jerk_action": summary.get("max_jerk_action"),
        # --- patch v2: diagnostic fields ---
        "action_has_nan_ratio": _bool_rate(step_records, "action_has_nan"),
        "action_has_inf_ratio": _bool_rate(step_records, "action_has_inf"),
        "action_min": _mean_nullable(_numbers_from_records(step_records, "action_min")),
        "action_max": _mean_nullable(_numbers_from_records(step_records, "action_max")),
        "action_mean": _mean_nullable(_numbers_from_records(step_records, "action_mean")),
        "action_std": _stds_from_records(step_records, "action_std", "action_mean"),
        "depth_valid_ratio": _mean_nullable(_numbers_from_records(step_records, "depth_valid_ratio")),
        "depth_min": _mean_nullable(_numbers_from_records(step_records, "depth_min")),
        "depth_max": _mean_nullable(_numbers_from_records(step_records, "depth_max")),
        "depth_mean": _mean_nullable(_numbers_from_records(step_records, "depth_mean")),
        "points_robot_min_x": _mean_nullable(_numbers_from_records(step_records, "points_robot_min_xyz_x")),
        "points_robot_min_y": _mean_nullable(_numbers_from_records(step_records, "points_robot_min_xyz_y")),
        "points_robot_min_z": _mean_nullable(_numbers_from_records(step_records, "points_robot_min_xyz_z")),
        "points_robot_max_x": _mean_nullable(_numbers_from_records(step_records, "points_robot_max_xyz_x")),
        "points_robot_max_y": _mean_nullable(_numbers_from_records(step_records, "points_robot_max_xyz_y")),
        "points_robot_max_z": _mean_nullable(_numbers_from_records(step_records, "points_robot_max_xyz_z")),
        "points_cam_min_x": _mean_nullable(_numbers_from_records(step_records, "points_cam_min_xyz_x")),
        "points_cam_min_y": _mean_nullable(_numbers_from_records(step_records, "points_cam_min_xyz_y")),
        "points_cam_min_z": _mean_nullable(_numbers_from_records(step_records, "points_cam_min_xyz_z")),
        "points_cam_max_x": _mean_nullable(_numbers_from_records(step_records, "points_cam_max_xyz_x")),
        "points_cam_max_y": _mean_nullable(_numbers_from_records(step_records, "points_cam_max_xyz_y")),
        "points_cam_max_z": _mean_nullable(_numbers_from_records(step_records, "points_cam_max_xyz_z")),
        "num_valid_3d_tokens": _mean_nullable(_numbers_from_records(step_records, "num_valid_3d_tokens")),
        "distance_to_gripper_min": _mean_nullable(_numbers_from_records(step_records, "distance_to_gripper_min")),
        "distance_to_gripper_mean": _mean_nullable(_numbers_from_records(step_records, "distance_to_gripper_mean")),
        "distance_to_gripper_max": _mean_nullable(_numbers_from_records(step_records, "distance_to_gripper_max")),
        "robot_state_valid_ratio": _bool_rate(step_records, "robot_state_valid"),
        "motion_direction_valid_ratio": _bool_rate(step_records, "motion_direction_valid"),
        "ee_position_x_mean": _mean_nullable(_x_from_records(step_records, "ee_position")),
        "ee_position_y_mean": _mean_nullable(_y_from_records(step_records, "ee_position")),
        "ee_position_z_mean": _mean_nullable(_z_from_records(step_records, "ee_position")),
        "contact_risk_lock_ratio": _bool_rate(step_records, "interaction_lock_reason", "contact_risk"),
        "gripper_proximity_lock_ratio": _bool_rate(step_records, "interaction_lock_reason", "gripper_proximity"),
        "region_stability_lock_ratio": _bool_rate(step_records, "interaction_lock_reason", "region_stability"),
        "score_ema_available_ratio": summary.get("score_ema_available_ratio"),
        "lock_condition_failed_reason_distribution": summary.get("lock_condition_failed_reason_distribution"),
        # --- P0-4 depth metric stats (from step-level records) ---
        "depth_metric_min": _mean_nullable(_numbers_from_records(step_records, "depth_metric_min")),
        "depth_metric_max": _mean_nullable(_numbers_from_records(step_records, "depth_metric_max")),
        "depth_metric_mean": _mean_nullable(_numbers_from_records(step_records, "depth_metric_mean")),
        "depth_metric_std": _mean_nullable(_numbers_from_records(step_records, "depth_metric_std")),
        "depth_source_key": _mode_any_str(_strings_from_records(step_records, "depth_source_key")),
        "depth_conversion": _mode_any_str(_strings_from_records(step_records, "depth_conversion")),
        "depth_is_metric": _mode_any_bool(_bools_from_records(step_records, "depth_is_metric")),
        "depth_unit": _mode_any_str(_strings_from_records(step_records, "depth_unit")),
        "depth_sim_available": _mode_any_bool(_bools_from_records(step_records, "depth_sim_available")),
        # --- P0-4 transform metadata ---
        "transform_convention": _mode_any_str(_strings_from_records(step_records, "transform_convention")),
        "transform_inverse_used": _mode_any_bool(_bools_from_records(step_records, "transform_inverse_used")),
        "transform_source": _mode_any_str(_strings_from_records(step_records, "transform_source")),
        "transform_convention_verified": _mode_any_bool(_bools_from_records(step_records, "transform_convention_verified")),
        "transform_convention_evidence": _first_non_null(_strings_from_records(step_records, "transform_convention_evidence")),
        # --- P0-4 depth quality diagnostics ---
        "depth_suspicious_ratio": _depth_suspicious_ratio(step_records),
        "depth_suspicious_steps": _count_depth_suspicious(step_records),
        "missing_depth_metadata_steps": _count_missing_depth(step_records),
        # --- P0-4 transform quality diagnostics ---
        "T_ambiguous_ratio": _T_ambiguous_ratio(step_records),
        "T_ambiguous_steps": _count_T_ambiguous(step_records),
        "missing_transform_metadata_steps": _count_missing_transform(step_records),
        # --- P0-4 motion diagnostics ---
        "motion_norm_mean": _mean_nullable(_numbers_from_records(step_records, "motion_norm")),
        "motion_norm_median": _median_nullable(_numbers_from_records(step_records, "motion_norm")),
        "motion_invalid_steps": _count_motion_invalid(step_records),
        "motion_valid_steps": _count_motion_valid(step_records),
        "motion_invalid_but_cone_nonzero_steps": _count_motion_invalid_but_cone_nonzero(step_records),
        "motion_invalid_but_cone_nonzero_ratio": _motion_invalid_but_cone_nonzero_ratio(step_records),
        "missing_motion_direction_valid_steps": _count_missing_motion_direction(step_records),
        # --- P0-4 workspace diagnostics ---
        "workspace_valid_steps": _count_workspace_valid(step_records),
        "workspace_all_one_ratio": _workspace_all_one_ratio(step_records),
        "workspace_all_one_steps": _count_workspace_all_one(step_records),
        # --- P0-4 token selection quality ---
        "selected_robot_token_ratio": _selected_robot_token_ratio(step_records),
        "retention_ratio": _compute_retention_ratio(step_records),
        # score component distribution stats
        "depth_edge_score_mean": _mean_nullable(_numbers_from_records(step_records, "depth_edge_score_mean")),
        "depth_edge_score_std": _mean_nullable(_numbers_from_records(step_records, "depth_edge_score_std")),
        "depth_edge_score_min": _mean_nullable(_numbers_from_records(step_records, "depth_edge_score_min")),
        "depth_edge_score_p50": _mean_nullable(_numbers_from_records(step_records, "depth_edge_score_p50")),
        "depth_edge_score_p90": _mean_nullable(_numbers_from_records(step_records, "depth_edge_score_p90")),
        "depth_edge_score_max": _mean_nullable(_numbers_from_records(step_records, "depth_edge_score_max")),
        "depth_edge_score_positive_ratio": _mean_nullable(_numbers_from_records(step_records, "depth_edge_score_positive_ratio")),
        "distance_score_mean": _mean_nullable(_numbers_from_records(step_records, "distance_score_mean")),
        "distance_score_std": _mean_nullable(_numbers_from_records(step_records, "distance_score_std")),
        "distance_score_min": _mean_nullable(_numbers_from_records(step_records, "distance_score_min")),
        "distance_score_p50": _mean_nullable(_numbers_from_records(step_records, "distance_score_p50")),
        "distance_score_p90": _mean_nullable(_numbers_from_records(step_records, "distance_score_p90")),
        "distance_score_max": _mean_nullable(_numbers_from_records(step_records, "distance_score_max")),
        "motion_cone_score_mean": _mean_nullable(_numbers_from_records(step_records, "motion_cone_score_mean")),
        "motion_cone_score_std": _mean_nullable(_numbers_from_records(step_records, "motion_cone_score_std")),
        "motion_cone_score_min": _mean_nullable(_numbers_from_records(step_records, "motion_cone_score_min")),
        "motion_cone_score_p50": _mean_nullable(_numbers_from_records(step_records, "motion_cone_score_p50")),
        "motion_cone_score_p90": _mean_nullable(_numbers_from_records(step_records, "motion_cone_score_p90")),
        "motion_cone_score_max": _mean_nullable(_numbers_from_records(step_records, "motion_cone_score_max")),
        "motion_cone_score_positive_ratio": _mean_nullable(_numbers_from_records(step_records, "motion_cone_score_positive_ratio")),
        "motion_cone_score_zero_ratio": _mean_nullable(_numbers_from_records(step_records, "motion_cone_score_zero_ratio")),
        "motion_dir_norm_mean": _mean_nullable(_numbers_from_records(step_records, "motion_dir_norm_mean")),
        "motion_dir_norm_min": _mean_nullable(_numbers_from_records(step_records, "motion_dir_norm_min")),
        "motion_dir_norm_max": _mean_nullable(_numbers_from_records(step_records, "motion_dir_norm_max")),
        "workspace_score_mean": _mean_nullable(_numbers_from_records(step_records, "workspace_score_mean")),
        "workspace_score_std": _mean_nullable(_numbers_from_records(step_records, "workspace_score_std")),
        "workspace_score_max": _mean_nullable(_numbers_from_records(step_records, "workspace_score_max")),
        "contact_risk_score_mean": _mean_nullable(_numbers_from_records(step_records, "contact_risk_score_mean")),
        "contact_risk_score_std": _mean_nullable(_numbers_from_records(step_records, "contact_risk_score_std")),
        "contact_risk_score_min": _mean_nullable(_numbers_from_records(step_records, "contact_risk_score_min")),
        "contact_risk_score_p50": _mean_nullable(_numbers_from_records(step_records, "contact_risk_score_p50")),
        "contact_risk_score_p90": _mean_nullable(_numbers_from_records(step_records, "contact_risk_score_p90")),
        "contact_risk_score_max": _mean_nullable(_numbers_from_records(step_records, "contact_risk_score_max")),
        "final_geometry_score_mean": _mean_nullable(_numbers_from_records(step_records, "final_geometry_score_mean")),
        "final_geometry_score_std": _mean_nullable(_numbers_from_records(step_records, "final_geometry_score_std")),
        "final_geometry_score_min": _mean_nullable(_numbers_from_records(step_records, "final_geometry_score_min")),
        "final_geometry_score_p50": _mean_nullable(_numbers_from_records(step_records, "final_geometry_score_p50")),
        "final_geometry_score_p90": _mean_nullable(_numbers_from_records(step_records, "final_geometry_score_p90")),
        "final_geometry_score_max": _mean_nullable(_numbers_from_records(step_records, "final_geometry_score_max")),
        "selected_depth_edge_score_mean": _mean_nullable(_numbers_from_records(step_records, "selected_depth_edge_score_mean")),
        "selected_distance_score_mean": _mean_nullable(_numbers_from_records(step_records, "selected_distance_score_mean")),
        "selected_motion_cone_score_mean": _mean_nullable(_numbers_from_records(step_records, "selected_motion_cone_score_mean")),
        "selected_workspace_score_mean": _mean_nullable(_numbers_from_records(step_records, "selected_workspace_score_mean")),
        "selected_contact_risk_score_mean": _mean_nullable(_numbers_from_records(step_records, "selected_contact_risk_score_mean")),
        "selected_final_score_mean": _mean_nullable(_numbers_from_records(step_records, "selected_final_score_mean")),
        # Hybrid quota union stats
        "selected_by_depth_edge_count": _mean_nullable(_numbers_from_records(step_records, "selected_by_depth_edge_count")),
        "selected_by_robot_geo_count": _mean_nullable(_numbers_from_records(step_records, "selected_by_robot_geo_count")),
        "selected_by_uniform_count": _mean_nullable(_numbers_from_records(step_records, "selected_by_uniform_count")),
        "selected_by_fill_count": _mean_nullable(_numbers_from_records(step_records, "selected_by_fill_count")),
        "depth_edge_quota_count": _mean_nullable(_numbers_from_records(step_records, "depth_edge_quota_count")),
        "robot_geo_quota_count": _mean_nullable(_numbers_from_records(step_records, "robot_geo_quota_count")),
        "uniform_quota_count": _mean_nullable(_numbers_from_records(step_records, "uniform_quota_count")),
        "fill_count": _mean_nullable(_numbers_from_records(step_records, "fill_count")),
        "overlap_depth_robot_geo": _mean_nullable(_numbers_from_records(step_records, "overlap_depth_robot_geo")),
        # points_robot mean/std
        "points_robot_mean_x": _mean_nullable(_numbers_from_records(step_records, "points_robot_mean_xyz_x")),
        "points_robot_mean_y": _mean_nullable(_numbers_from_records(step_records, "points_robot_mean_xyz_y")),
        "points_robot_mean_z": _mean_nullable(_numbers_from_records(step_records, "points_robot_mean_xyz_z")),
        "points_robot_std_x": _mean_nullable(_numbers_from_records(step_records, "points_robot_std_xyz_x")),
        "points_robot_std_y": _mean_nullable(_numbers_from_records(step_records, "points_robot_std_xyz_y")),
        "points_robot_std_z": _mean_nullable(_numbers_from_records(step_records, "points_robot_std_xyz_z")),
        # camera frame availability
        "points_cam_available": _bool_rate(step_records, "points_cam_available") == 1.0 if _bool_rate(step_records, "points_cam_available") is not None else None,
        # --- hybrid fix: new score distribution diagnostics ---
        "contact_risk_top1_mean": _percentile(_numbers_from_records(step_records, "contact_risk_score_max"), 100),
        "contact_risk_top5_mean": _mean_nullable(_numbers_from_records(step_records, "contact_risk_score_p90")),
        "contact_risk_top10_mean": None,
        "contact_risk_p95": _percentile(_numbers_from_records(step_records, "contact_risk_score_max"), 95),
        "contact_risk_p99": _percentile(_numbers_from_records(step_records, "contact_risk_score_max"), 99),
        "motion_cone_top5_mean": _mean_nullable(_numbers_from_records(step_records, "motion_cone_score_p90")),
        "motion_cone_score_max": _mean_nullable(_numbers_from_records(step_records, "motion_cone_score_max")),
        "motion_cone_score_zero_ratio": _mean_nullable(_numbers_from_records(step_records, "motion_cone_score_zero_ratio")),
        "distance_score_top5_mean": _mean_nullable(_numbers_from_records(step_records, "distance_score_p90")),
        "final_score_top5_mean": _mean_nullable(_numbers_from_records(step_records, "final_geometry_score_p90")),
        "final_score_concentration": _score_concentration(step_records),
        "selected_contact_risk_score_mean": _mean_nullable(_numbers_from_records(step_records, "selected_contact_risk_score_mean")),
        "selected_contact_risk_score_p50": _mean_nullable(_numbers_from_records(step_records, "selected_contact_risk_score_p50")),
        "selected_contact_risk_score_p90": _mean_nullable(_numbers_from_records(step_records, "selected_contact_risk_score_p90")),
        "selected_contact_risk_score_max": _mean_nullable(_numbers_from_records(step_records, "selected_contact_risk_score_max")),
        "selected_motion_cone_score_mean": _mean_nullable(_numbers_from_records(step_records, "selected_motion_cone_score_mean")),
        "selected_distance_score_mean": _mean_nullable(_numbers_from_records(step_records, "selected_distance_score_mean")),
        "selected_final_score_mean": _mean_nullable(_numbers_from_records(step_records, "selected_final_score_mean")),
        # --- Hybrid v1 weighted score stats ---
        "w_edge": _first_number(
            summary.get("w_edge"),
            _mean_nullable(_numbers_from_records(step_records, "w_edge")),
        ),
        "w_near": _first_number(
            summary.get("w_near"),
            _mean_nullable(_numbers_from_records(step_records, "w_near")),
        ),
        "w_contact": _first_number(
            summary.get("w_contact"),
            _mean_nullable(_numbers_from_records(step_records, "w_contact")),
        ),
        "w_corr": _first_number(
            summary.get("w_corr"),
            _mean_nullable(_numbers_from_records(step_records, "w_corr")),
        ),
        "w_diverse": _first_number(
            summary.get("w_diverse"),
            _mean_nullable(_numbers_from_records(step_records, "w_diverse")),
        ),
        "edge_score_mean": _mean_nullable(_numbers_from_records(step_records, "edge_score_mean")),
        "edge_score_max": _mean_nullable(_numbers_from_records(step_records, "edge_score_max")),
        "edge_score_std": _mean_nullable(_numbers_from_records(step_records, "edge_score_std")),
        "near_score_mean": _mean_nullable(_numbers_from_records(step_records, "near_score_mean")),
        "near_score_max": _mean_nullable(_numbers_from_records(step_records, "near_score_max")),
        "near_score_std": _mean_nullable(_numbers_from_records(step_records, "near_score_std")),
        "contact_score_mean": _mean_nullable(_numbers_from_records(step_records, "contact_score_mean")),
        "contact_score_max": _mean_nullable(_numbers_from_records(step_records, "contact_score_max")),
        "contact_score_std": _mean_nullable(_numbers_from_records(step_records, "contact_score_std")),
        "corridor_score_mean": _mean_nullable(_numbers_from_records(step_records, "corridor_score_mean")),
        "corridor_score_max": _mean_nullable(_numbers_from_records(step_records, "corridor_score_max")),
        "corridor_score_std": _mean_nullable(_numbers_from_records(step_records, "corridor_score_std")),
        "diversity_score_mean": _mean_nullable(_numbers_from_records(step_records, "diversity_score_mean")),
        "diversity_score_max": _mean_nullable(_numbers_from_records(step_records, "diversity_score_max")),
        "diversity_score_std": _mean_nullable(_numbers_from_records(step_records, "diversity_score_std")),
        "final_hybrid_score_mean": _mean_nullable(_numbers_from_records(step_records, "final_hybrid_score_mean")),
        "final_hybrid_score_max": _mean_nullable(_numbers_from_records(step_records, "final_hybrid_score_max")),
        "final_hybrid_score_std": _mean_nullable(_numbers_from_records(step_records, "final_hybrid_score_std")),
        "selected_grid_coverage_ratio": _mean_nullable(_numbers_from_records(step_records, "selected_grid_coverage_ratio")),
        "grid_coverage_ratio": _first_number(
            summary.get("grid_coverage_ratio"),
            _mean_nullable(_numbers_from_records(step_records, "grid_coverage_ratio")),
            _mean_nullable(_numbers_from_records(step_records, "selected_grid_coverage_ratio")),
            _mean_nullable(_numbers_from_records(step_records, "selected_token_grid_entropy")),
        ),
        # --- Temporal v1 adaptive threshold ---
        "adaptive_threshold_mean": _mean_nullable(_numbers_from_records(step_records, "adaptive_threshold_mean")),
        "adaptive_threshold_max": _mean_nullable(_numbers_from_records(step_records, "adaptive_threshold_max")),
        "ema_used_for_selection": _bool_rate(step_records, "ema_used_for_selection"),
        # ---- P1-1: Token selection attribution / top-k competition diagnostics ----
        "selected_token_count_mean": _mean_nullable(_numbers_from_records(step_records, "selected_token_count")),
        "selected_token_count_std": _stds_from_records(step_records, "selected_token_count", "selected_token_count"),
        "dropped_token_count_mean": _mean_nullable(_numbers_from_records(step_records, "dropped_token_count")),
        "dropped_token_count_std": _stds_from_records(step_records, "dropped_token_count", "dropped_token_count"),
        "retention_ratio_mean": _mean_nullable(_numbers_from_records(step_records, "retention_ratio")),
        "retention_ratio_std": _stds_from_records(step_records, "retention_ratio", "retention_ratio"),
        "selected_token_ratio_mean": _mean_nullable(_numbers_from_records(step_records, "selected_token_ratio")),
        "selected_token_ratio_std": _stds_from_records(step_records, "selected_token_ratio", "selected_token_ratio"),
        "dropped_token_ratio_mean": _mean_nullable(_numbers_from_records(step_records, "dropped_token_ratio")),
        "dropped_token_ratio_std": _stds_from_records(step_records, "dropped_token_ratio", "dropped_token_ratio"),
        "num_visual_tokens_original_mean": _mean_nullable(_numbers_from_records(step_records, "num_visual_tokens_original")),
        "num_visual_tokens_kept_mean": _mean_nullable(_numbers_from_records(step_records, "num_visual_tokens_kept")),
        "depth_edge_topk_count_mean": _mean_nullable(_numbers_from_records(step_records, "depth_edge_topk_count")),
        "robot_geo_topk_count_mean": _mean_nullable(_numbers_from_records(step_records, "robot_geo_topk_count")),
        "final_selected_count_mean": _mean_nullable(_numbers_from_records(step_records, "final_selected_count")),
        "depth_edge_topk_kept_in_final_count_mean": _mean_nullable(_numbers_from_records(step_records, "depth_edge_topk_kept_in_final_count")),
        "robot_geo_topk_kept_in_final_count_mean": _mean_nullable(_numbers_from_records(step_records, "robot_geo_topk_kept_in_final_count")),
        "depth_edge_topk_dropped_count_mean": _mean_nullable(_numbers_from_records(step_records, "depth_edge_topk_dropped_count")),
        "robot_geo_topk_dropped_count_mean": _mean_nullable(_numbers_from_records(step_records, "robot_geo_topk_dropped_count")),
        "depth_edge_topk_dropped_ratio_mean": _mean_nullable(_numbers_from_records(step_records, "depth_edge_topk_dropped_ratio")),
        "robot_geo_topk_dropped_ratio_mean": _mean_nullable(_numbers_from_records(step_records, "robot_geo_topk_dropped_ratio")),
        "overlap_depth_edge_robot_geo_count_mean": _mean_nullable(_numbers_from_records(step_records, "overlap_depth_edge_robot_geo_count")),
        "overlap_depth_edge_robot_geo_ratio_mean": _mean_nullable(_numbers_from_records(step_records, "overlap_depth_edge_robot_geo_ratio")),
        "selected_overlap_robot_geo_depth_edge_ratio_mean": _mean_nullable(_numbers_from_records(step_records, "selected_overlap_robot_geo_depth_edge_ratio")),
        "selected_overlap_robot_geo_motion_ratio_mean": _mean_nullable(_numbers_from_records(step_records, "selected_overlap_robot_geo_motion_ratio")),
        "selected_overlap_robot_geo_near_ratio_mean": _mean_nullable(_numbers_from_records(step_records, "selected_overlap_robot_geo_near_ratio")),
        "depth_edge_topk_overlap_with_robot_geo_topk_mean": _mean_nullable(_numbers_from_records(step_records, "depth_edge_topk_overlap_with_robot_geo_topk")),
        "robot_geo_topk_overlap_with_depth_edge_topk_mean": _mean_nullable(_numbers_from_records(step_records, "robot_geo_topk_overlap_with_depth_edge_topk")),
        "selected_high_depth_edge_but_low_robot_geo_count_mean": _mean_nullable(_numbers_from_records(step_records, "selected_high_depth_edge_but_low_robot_geo_count")),
        "dropped_high_depth_edge_tokens_count_mean": _mean_nullable(_numbers_from_records(step_records, "dropped_high_depth_edge_tokens_count")),
        "dropped_high_robot_geo_tokens_count_mean": _mean_nullable(_numbers_from_records(step_records, "dropped_high_robot_geo_tokens_count")),
        # P5-fix: edge_reserve invariant checks
        "edge_reserve_invalid_ratio": _mean_boolean(step_records, "edge_reserve_invalid"),
        "edge_reserve_invalid_steps": _sum_from_records(step_records, "edge_reserve_invalid"),
        "edge_scores_available_ratio": _mean_boolean(step_records, "edge_scores_available"),
        "edge_reserved_actual_mean": _mean_nullable(_numbers_from_records(step_records, "edge_reserved_actual_count")),
        "edge_reserved_actual_std": _stds_nullable(step_records, "edge_reserved_actual_count"),
        "edge_reserved_survival_ratio_mean": _mean_nullable(_floats_from_records(step_records, "edge_reserved_survival_ratio")),
        "reserved_edge_topk_dropped_ratio_mean": _mean_nullable(_floats_from_records(step_records, "reserved_edge_topk_dropped_ratio")),
        "non_reserved_edge_topk_dropped_ratio_mean": _mean_nullable(_floats_from_records(step_records, "non_reserved_edge_topk_dropped_ratio")),
        "overall_depth_edge_topk_dropped_ratio_mean": _mean_nullable(_floats_from_records(step_records, "overall_depth_edge_topk_dropped_ratio")),
        "duplicate_after_exclusion_count_mean": _mean_nullable(_numbers_from_records(step_records, "duplicate_after_exclusion_count")),
        "duplicate_with_original_hybrid_count_mean": _mean_nullable(_numbers_from_records(step_records, "duplicate_with_original_hybrid_count")),
        # P5-fix: new accounting and invariant metrics
        "edge_reserve_accounting_valid_ratio": _mean_boolean(step_records, "edge_reserve_accounting_valid"),
        "edge_reserve_split_metrics_valid_ratio": _mean_boolean(step_records, "edge_reserve_split_metrics_valid"),
        "selected_by_edge_reserved_count_mean": _mean_nullable(_numbers_from_records(step_records, "selected_by_edge_reserved_count")),
        "selected_by_phase1_hybrid_count_mean": _mean_nullable(_numbers_from_records(step_records, "selected_by_phase1_hybrid_count")),
        "selected_by_phase2_diversity_count_mean": _mean_nullable(_numbers_from_records(step_records, "selected_by_phase2_diversity_count")),
        "selected_by_phase3_fallback_count_mean": _mean_nullable(_numbers_from_records(step_records, "selected_by_phase3_fallback_count")),
        "selected_by_fill_count_mean": _mean_nullable(_numbers_from_records(step_records, "selected_by_fill_count")),
        "selected_by_unattributed_count_mean": _mean_nullable(_numbers_from_records(step_records, "selected_by_unattributed_count")),
        "no_duplicate_final_indices_ratio": _mean_boolean(step_records, "no_duplicate_final_indices"),
        "final_indices_sorted_ratio": _mean_boolean(step_records, "final_indices_sorted"),
        "selected_token_count_equals_kept_ratio": _mean_boolean(step_records, "selected_token_count_equals_kept"),
        "retention_ratio_valid_ratio": _mean_boolean(step_records, "retention_ratio_valid"),
        "selected_token_u_mean_mean": _mean_nullable(_numbers_from_records(step_records, "selected_token_u_mean")),
        "selected_token_u_std_mean": _mean_nullable(_numbers_from_records(step_records, "selected_token_u_std")),
        "selected_token_v_mean_mean": _mean_nullable(_numbers_from_records(step_records, "selected_token_v_mean")),
        "selected_token_v_std_mean": _mean_nullable(_numbers_from_records(step_records, "selected_token_v_std")),
        "selected_token_bbox_u_min_mean": _mean_nullable(_numbers_from_records(step_records, "selected_token_bbox_u_min")),
        "selected_token_bbox_u_max_mean": _mean_nullable(_numbers_from_records(step_records, "selected_token_bbox_u_max")),
        "selected_token_bbox_v_min_mean": _mean_nullable(_numbers_from_records(step_records, "selected_token_bbox_v_min")),
        "selected_token_bbox_v_max_mean": _mean_nullable(_numbers_from_records(step_records, "selected_token_bbox_v_max")),
        "selected_token_near_gripper_pixel_dist_mean_mean": _mean_nullable(_numbers_from_records(step_records, "selected_token_near_gripper_pixel_dist_mean")),
        "selected_token_near_gripper_pixel_dist_median_mean": _mean_nullable(_numbers_from_records(step_records, "selected_token_near_gripper_pixel_dist_median")),
        "workspace_score_min_mean": _mean_nullable(_numbers_from_records(step_records, "workspace_score_min")),
        "workspace_score_unique_count_mean": _mean_nullable(_numbers_from_records(step_records, "workspace_score_unique_count")),
        "workspace_valid_token_ratio_mean": _mean_nullable(_numbers_from_records(step_records, "workspace_valid_token_ratio")),
        "near_score_unique_count_mean": _mean_nullable(_numbers_from_records(step_records, "near_score_unique_count")),
        "depth_edge_score_unique_count_mean": _mean_nullable(_numbers_from_records(step_records, "depth_edge_score_unique_count")),
        "motion_cone_score_unique_count_mean": _mean_nullable(_numbers_from_records(step_records, "motion_cone_score_unique_count")),
        "final_score_unique_count_mean": _mean_nullable(_numbers_from_records(step_records, "final_score_unique_count")),
        # P11: branch_budget_v0
        "branch_budget_v0_rate": _mean_boolean(step_records, "branch_budget_v0"),
        "branch_accounting_valid_ratio": _mean_boolean(step_records, "branch_accounting_valid"),
        "branch_sum_equals_kept_ratio": _mean_boolean(step_records, "branch_sum_equals_kept"),
        "selected_by_depth_branch_count_mean": _mean_nullable(_numbers_from_records(step_records, "selected_by_depth_branch_count")),
        "selected_by_hybrid_branch_count_mean": _mean_nullable(_numbers_from_records(step_records, "selected_by_hybrid_branch_count")),
        "selected_by_fill_branch_count_mean": _mean_nullable(_numbers_from_records(step_records, "selected_by_fill_branch_count")),
        "selected_by_fallback_mean": _mean_nullable(_numbers_from_records(step_records, "selected_by_fallback")),
        "overlap_depth_edge_hybrid_count_mean": _mean_nullable(_numbers_from_records(step_records, "overlap_depth_edge_hybrid_count")),
        "overlap_depth_edge_hybrid_ratio_mean": _mean_nullable(_floats_from_records(step_records, "overlap_depth_edge_hybrid_ratio")),
        "depth_edge_topk_count_mean": _mean_nullable(_numbers_from_records(step_records, "depth_edge_topk_count")),
        "depth_edge_topk_kept_in_final_count_mean": _mean_nullable(_numbers_from_records(step_records, "depth_edge_topk_kept_in_final_count")),
        "depth_edge_topk_dropped_count_mean": _mean_nullable(_numbers_from_records(step_records, "depth_edge_topk_dropped_count")),
        "depth_edge_topk_dropped_ratio_mean": _mean_nullable(_floats_from_records(step_records, "depth_edge_topk_dropped_ratio")),
        "hybrid_final_score_topk_count_mean": _mean_nullable(_numbers_from_records(step_records, "hybrid_final_score_topk_count")),
        "hybrid_final_score_topk_kept_count_mean": _mean_nullable(_numbers_from_records(step_records, "hybrid_final_score_topk_kept_count")),
        "hybrid_final_score_topk_dropped_count_mean": _mean_nullable(_numbers_from_records(step_records, "hybrid_final_score_topk_dropped_count")),
        "hybrid_final_score_topk_dropped_ratio_mean": _mean_nullable(_floats_from_records(step_records, "hybrid_final_score_topk_dropped_ratio")),
        "non_reserved_depth_edge_dropped_ratio_mean": _mean_nullable(_floats_from_records(step_records, "non_reserved_depth_edge_dropped_ratio")),
        "depth_edge_score_mean_mean": _mean_nullable(_floats_from_records(step_records, "depth_edge_score_mean")),
        "hybrid_score_mean_mean": _mean_nullable(_floats_from_records(step_records, "hybrid_score_mean")),
        "final_hybrid_score_mean_mean": _mean_nullable(_floats_from_records(step_records, "final_hybrid_score_mean")),
        # P11.3: DE top-k attribution by branch
        "depth_edge_topk_kept_by_depth_branch_count_mean": _mean_nullable(_numbers_from_records(step_records, "depth_edge_topk_kept_by_depth_branch_count")),
        "depth_edge_topk_kept_by_hybrid_branch_count_mean": _mean_nullable(_numbers_from_records(step_records, "depth_edge_topk_kept_by_hybrid_branch_count")),
        "depth_edge_topk_kept_by_fill_branch_count_mean": _mean_nullable(_numbers_from_records(step_records, "depth_edge_topk_kept_by_fill_branch_count")),
        "depth_edge_topk_kept_by_fallback_count_mean": _mean_nullable(_numbers_from_records(step_records, "depth_edge_topk_kept_by_fallback_count")),
        "depth_edge_topk_survival_ratio_mean": _mean_nullable(_floats_from_records(step_records, "depth_edge_topk_survival_ratio")),
        # P11.3: Hybrid top-k attribution by branch
        "hybrid_final_score_topk_kept_by_depth_branch_count_mean": _mean_nullable(_numbers_from_records(step_records, "hybrid_final_score_topk_kept_by_depth_branch_count")),
        "hybrid_final_score_topk_kept_by_hybrid_branch_count_mean": _mean_nullable(_numbers_from_records(step_records, "hybrid_final_score_topk_kept_by_hybrid_branch_count")),
        "hybrid_final_score_topk_kept_by_fill_branch_count_mean": _mean_nullable(_numbers_from_records(step_records, "hybrid_final_score_topk_kept_by_fill_branch_count")),
        "hybrid_final_score_topk_kept_by_fallback_count_mean": _mean_nullable(_numbers_from_records(step_records, "hybrid_final_score_topk_kept_by_fallback_count")),
        "hybrid_final_score_topk_survival_ratio_mean": _mean_nullable(_floats_from_records(step_records, "hybrid_final_score_topk_survival_ratio")),
        # P11.3: Legacy hybrid top-k aliases (P8 compat)
        "hybrid_topk_kept_by_depth_branch_count_mean": _mean_nullable(_numbers_from_records(step_records, "hybrid_topk_kept_by_depth_branch_count")),
        "hybrid_topk_kept_by_hybrid_branch_count_mean": _mean_nullable(_numbers_from_records(step_records, "hybrid_topk_kept_by_hybrid_branch_count")),
        "hybrid_topk_kept_by_fill_branch_count_mean": _mean_nullable(_numbers_from_records(step_records, "hybrid_topk_kept_by_fill_branch_count")),
        "hybrid_topk_kept_by_fallback_count_mean": _mean_nullable(_numbers_from_records(step_records, "hybrid_topk_kept_by_fallback_count")),
        "hybrid_topk_survival_ratio_mean": _mean_nullable(_floats_from_records(step_records, "hybrid_topk_survival_ratio")),
    }
    return {k: _json_safe(row.get(k)) for k in ROLLUP_FIELDS}


def _add_baseline_comparisons(rows: List[Dict[str, Any]]) -> None:
    baseline = None
    for row in rows:
        if row.get("method") == "baseline_none_keep100":
            baseline = row
            break
    if baseline is None:
        return
    baseline_effective_ms = _num(baseline.get("mean_effective_inference_ms"))
    baseline_success = _num(baseline.get("success_rate"))
    for row in rows:
        effective_ms = _num(row.get("mean_effective_inference_ms"))
        success = _num(row.get("success_rate"))
        if baseline_effective_ms is not None and effective_ms is not None and effective_ms > 0:
            row["speedup_vs_baseline"] = _json_safe(baseline_effective_ms / effective_ms)
        if baseline_success is not None and success is not None:
            row["success_delta_vs_baseline"] = _json_safe(success - baseline_success)


def write_rollup(root: Path, rows: List[Dict[str, Any]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "rollup_summary.json"
    csv_path = root / "rollup_summary.csv"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROLLUP_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _read_step_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _fallback_rate(records: List[Dict[str, Any]]) -> Optional[float]:
    vals = []
    for record in records:
        value = record.get("fallback_used")
        parsed = _parse_bool(value)
        if parsed is not None:
            vals.append(parsed)
    return sum(vals) / len(vals) if vals else None


def _bool_rate(records: List[Dict[str, Any]], key: str, contains: Optional[str] = None) -> Optional[float]:
    vals = []
    for record in records:
        value = record.get(key)
        if value in (None, ""):
            continue
        if isinstance(value, str):
            if contains is not None:
                vals.append(contains in value.strip().lower())
            else:
                parsed = _parse_bool(value)
                if parsed is not None:
                    vals.append(parsed)
        else:
            parsed = _parse_bool(value)
            if parsed is not None:
                vals.append(parsed)
    return sum(vals) / len(vals) if vals else None


def _numbers_from_records(records: List[Dict[str, Any]], key: str) -> List[float]:
    out: List[float] = []
    for record in records:
        value = _num(record.get(key))
        if value is not None:
            out.append(value)
    return out


def _mean_nullable(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _floats_from_records(records: List[Dict[str, Any]], key: str) -> List[float]:
    out: List[float] = []
    for record in records:
        value = record.get(key)
        if value is not None:
            try:
                out.append(float(value))
            except (ValueError, TypeError):
                pass
    return out


def _mean_boolean(records: List[Dict[str, Any]], key: str) -> Optional[float]:
    vals = []
    for record in records:
        value = record.get(key)
        parsed = _parse_bool(value)
        if parsed is not None:
            vals.append(1.0 if parsed else 0.0)
    return sum(vals) / len(vals) if vals else None


def _sum_from_records(records: List[Dict[str, Any]], key: str) -> Optional[int]:
    total = 0
    found = False
    for record in records:
        value = record.get(key)
        parsed_bool = _parse_bool(value)
        if parsed_bool is not None:
            found = True
            total += 1 if parsed_bool else 0
            continue
        if value in (None, ""):
            continue
        found = True
        try:
            total += int(value)
        except (ValueError, TypeError):
            pass
    return total if found else None


def _stds_nullable(records: List[Dict[str, Any]], key: str) -> Optional[float]:
    vals = _numbers_from_records(records, key)
    if len(vals) < 2:
        return None
    import statistics
    return float(statistics.stdev(vals))


def _mean_std_nullable(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "std": None}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values) if len(values) > 1 else 0.0
    return {"mean": mean, "std": math.sqrt(variance)}


def _categorical_distribution(records: List[Dict[str, Any]], key: str) -> Optional[Dict[str, float]]:
    values = []
    for record in records:
        value = record.get(key)
        if value in (None, ""):
            continue
        values.append(str(value))
    if not values:
        return None
    counts = Counter(values)
    total = float(len(values))
    return {name: count / total for name, count in sorted(counts.items())}


def _categorical_counts(records: List[Dict[str, Any]], key: str) -> Optional[Dict[str, int]]:
    values = []
    for record in records:
        value = record.get(key)
        if value in (None, "", "none", "None"):
            continue
        values.append(str(value))
    if not values:
        return None
    return dict(sorted(Counter(values).items()))


def _parse_bool(value: Any) -> Optional[bool]:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return bool(value)
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1", "yes", "y"):
            return True
        if normalized in ("false", "0", "no", "n"):
            return False
    return None


def _first_number(*values: Any) -> Optional[float]:
    for value in values:
        value = _num(value)
        if value is not None:
            return value
    return None


def _num(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    return value


def _percentile(values: List[float], p: float) -> Optional[float]:
    """Compute a percentile of a list of floats."""
    if not values:
        return None
    sorted_vals = sorted(values)
    idx = (p / 100.0) * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _score_concentration(records: List[Dict[str, Any]]) -> Optional[float]:
    """Compute score concentration = mean(top1/mean) across steps."""
    pairs = []
    for r in records:
        top1 = _num(r.get("geometry_score_max"))
        mean = _num(r.get("geometry_score_mean"))
        if top1 is not None and mean is not None and mean > 0:
            pairs.append((top1, mean))
    if not pairs:
        return None
    ratios = [t / m for t, m in pairs if m > 0]
    return sum(ratios) / len(ratios) if ratios else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roll up OpenVLA pruning ablation results")
    parser.add_argument("--root", type=Path, required=True, help="Ablation output root")
    parser.add_argument("--methods", type=str, default=None, help="Optional comma-separated method directory names")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()] if args.methods else None
    rows = build_rollup(args.root, methods=methods)
    write_rollup(args.root, rows)
    print(f"Wrote {args.root / 'rollup_summary.json'}")
    print(f"Wrote {args.root / 'rollup_summary.csv'}")


def _z_from_records(records: List[Dict[str, Any]], key: str) -> List[float]:
    out: List[float] = []
    for record in records:
        val = record.get(key)
        if val is None or val == "":
            continue
        try:
            parts = str(val).split(",")
            if len(parts) >= 3:
                v = float(parts[2].strip())
                if math.isfinite(v):
                    out.append(v)
        except (TypeError, ValueError, IndexError):
            continue
    return out


# --- P0-4 helper functions: string / bool list extractors ---

def _strings_from_records(records: List[Dict[str, Any]], key: str) -> List[str]:
    """Extract non-null/non-empty string values for a field."""
    out: List[str] = []
    for record in records:
        val = record.get(key)
        if val is not None and val != "":
            out.append(str(val))
    return out


def _bools_from_records(records: List[Dict[str, Any]], key: str) -> List[bool]:
    """Extract non-null boolean values for a field."""
    out: List[bool] = []
    for record in records:
        val = record.get(key)
        if val is not None:
            out.append(bool(val))
    return out


def _first_non_null(values: List[str]) -> Optional[str]:
    """Return first non-null/non-empty value, or None."""
    for v in values:
        if v is not None and v != "":
            return str(v)
    return None


def _mode_any_str(values: List[str]) -> Optional[str]:
    """Mode (most common value) for string fields."""
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def _mode_any_bool(values: List[bool]) -> Optional[bool]:
    """Mode for boolean fields."""
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def _median_nullable(values: List[float]) -> Optional[float]:
    """Median, skipping None."""
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return None
    sorted_vals = sorted(finite)
    n = len(sorted_vals)
    if n % 2 == 0:
        return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0
    return sorted_vals[n // 2]


# --- P0-4 diagnostic functions ---

_EPS = 1e-6


def _count_motion_invalid(records: List[Dict[str, Any]]) -> int:
    """Count steps where motion_direction_valid == False."""
    return sum(1 for r in records if r.get("motion_direction_valid") is False)


def _count_motion_valid(records: List[Dict[str, Any]]) -> int:
    """Count steps where motion_direction_valid == True."""
    return sum(1 for r in records if r.get("motion_direction_valid") is True)


def _count_missing_motion_direction(records: List[Dict[str, Any]]) -> int:
    """Count steps where motion_direction_valid is None/missing."""
    return sum(1 for r in records if r.get("motion_direction_valid") is None)


def _count_motion_invalid_but_cone_nonzero(records: List[Dict[str, Any]]) -> int:
    """Count steps where motion_direction_valid=False BUT motion_cone_nonzero_ratio > eps."""
    count = 0
    for r in records:
        mdv = r.get("motion_direction_valid")
        mcnr = r.get("motion_cone_nonzero_ratio")
        mcsm = r.get("motion_cone_score_mean")
        if mdv is False:
            # Cone nonzero: either nonzero_ratio > eps OR score_mean > eps
            if (mcnr is not None and mcnr > _EPS) or (mcsm is not None and mcsm > _EPS):
                count += 1
    return count


def _motion_invalid_but_cone_nonzero_ratio(records: List[Dict[str, Any]]) -> Optional[float]:
    """Ratio of motion_invalid_but_cone_nonzero steps among steps with motion_direction_valid==False."""
    total_invalid = sum(1 for r in records if r.get("motion_direction_valid") is False)
    if total_invalid == 0:
        return None
    bad = _count_motion_invalid_but_cone_nonzero(records)
    return bad / total_invalid if total_invalid > 0 else None


def _count_workspace_all_one(records: List[Dict[str, Any]]) -> int:
    """Count steps where workspace_score_mean == 1 (all tokens score 1)."""
    count = 0
    for r in records:
        ws = _num(r.get("workspace_score_mean"))
        if ws is not None and abs(ws - 1.0) < _EPS:
            count += 1
    return count


def _count_workspace_valid(records: List[Dict[str, Any]]) -> int:
    """Count steps where workspace_score_mean is not None."""
    return sum(1 for r in records if _num(r.get("workspace_score_mean")) is not None)


def _workspace_all_one_ratio(records: List[Dict[str, Any]]) -> Optional[float]:
    """Ratio of workspace_all_one steps among steps with workspace_score_mean not None."""
    total = _count_workspace_valid(records)
    if total == 0:
        return None
    return _count_workspace_all_one(records) / total


def _depth_suspicious(records: List[Dict[str, Any]]) -> List[int]:
    """Indices of records where depth is suspicious."""
    suspicious = []
    for i, r in enumerate(records):
        da = r.get("depth_available") or r.get("geometry_available")
        dim = r.get("depth_is_metric")
        dc = r.get("depth_conversion")
        dm_mean = r.get("depth_metric_mean")
        # Suspicious: has depth but not metric, or raw_no_sim_fallback, or suspicious range
        if da and dim is False:
            suspicious.append(i)
        elif da and dc == "raw_no_sim_fallback":
            suspicious.append(i)
        elif da and dm_mean is not None:
            # Suspicious if metric_mean ≈ 0.95-1.0 (raw z-buffer range) and not robosuite depth
            if 0.95 <= float(dm_mean) <= 1.05 and dc != "robosuite_get_real_depth_map":
                suspicious.append(i)
    return suspicious


def _count_depth_suspicious(records: List[Dict[str, Any]]) -> Optional[int]:
    """Number of suspicious depth steps."""
    s = _depth_suspicious(records)
    return len(s) if s else 0


def _depth_suspicious_ratio(records: List[Dict[str, Any]]) -> Optional[float]:
    """Ratio of suspicious depth steps."""
    total = sum(1 for r in records if r.get("depth_is_metric") is not None or r.get("depth_conversion") is not None)
    if total == 0:
        return None
    return len(_depth_suspicious(records)) / total


def _count_missing_depth(records: List[Dict[str, Any]]) -> int:
    """Count steps missing depth metadata."""
    return sum(1 for r in records if r.get("depth_is_metric") is None and r.get("depth_conversion") is None)


def _T_ambiguous(records: List[Dict[str, Any]]) -> List[int]:
    """Indices of records where transform is ambiguous."""
    ambiguous = []
    for i, r in enumerate(records):
        tc = r.get("transform_convention")
        tcv = r.get("transform_convention_verified")
        tps = r.get("T_projection_status")
        # Ambiguous: verified=False OR transform missing OR projection_status=ambiguous/both
        if tc is None:
            ambiguous.append(i)
        elif tcv is False:
            ambiguous.append(i)
        elif tps in ("ambiguous", "both"):
            ambiguous.append(i)
    return ambiguous


def _count_T_ambiguous(records: List[Dict[str, Any]]) -> Optional[int]:
    """Number of ambiguous transform steps."""
    a = _T_ambiguous(records)
    return len(a) if a else 0


def _T_ambiguous_ratio(records: List[Dict[str, Any]]) -> Optional[float]:
    """Ratio of ambiguous transform steps."""
    total = sum(1 for r in records if r.get("transform_convention") is not None or r.get("transform_convention_verified") is not None)
    if total == 0:
        return None
    return len(_T_ambiguous(records)) / total


def _count_missing_transform(records: List[Dict[str, Any]]) -> int:
    """Count steps missing transform metadata."""
    return sum(1 for r in records if r.get("transform_convention") is None)


def _selected_robot_token_ratio(records: List[Dict[str, Any]]) -> Optional[float]:
    """Ratio of robot tokens selected (selected_by_robot_geo_count / num_visual_tokens_kept)."""
    ratios = []
    for r in records:
        by_geo = _num(r.get("selected_by_robot_geo_count"))
        kept = _num(r.get("num_visual_tokens_kept"))
        if by_geo is not None and kept is not None and int(kept) > 0:
            ratios.append(float(by_geo) / float(kept))
    return _mean_nullable(ratios) if ratios else None


def _compute_retention_ratio(records: List[Dict[str, Any]]) -> Optional[float]:
    """retention_ratio = num_kept / num_original (mean across steps)."""
    ratios = []
    for r in records:
        kept = _num(r.get("num_visual_tokens_kept"))
        orig = _num(r.get("num_visual_tokens_original"))
        if kept is not None and orig is not None and int(orig) > 0:
            ratios.append(float(kept) / float(orig))
    return _mean_nullable(ratios) if ratios else None


def _x_from_records(records: List[Dict[str, Any]], key: str) -> List[float]:
    out: List[float] = []
    for record in records:
        val = record.get(key)
        if val is None or val == "":
            continue
        try:
            parts = str(val).split(",")
            if len(parts) >= 1:
                v = float(parts[0].strip())
                if math.isfinite(v):
                    out.append(v)
        except (TypeError, ValueError, IndexError):
            continue
    return out


def _y_from_records(records: List[Dict[str, Any]], key: str) -> List[float]:
    out: List[float] = []
    for record in records:
        val = record.get(key)
        if val is None or val == "":
            continue
        try:
            parts = str(val).split(",")
            if len(parts) >= 2:
                v = float(parts[1].strip())
                if math.isfinite(v):
                    out.append(v)
        except (TypeError, ValueError, IndexError):
            continue
    return out


def _stds_from_records(records: List[Dict[str, Any]], std_key: str, mean_key: str) -> Optional[float]:
    out: List[float] = []
    for record in records:
        val = record.get(std_key)
        if val is None or val == "":
            continue
        try:
            v = float(val)
            if math.isfinite(v):
                out.append(v)
        except (TypeError, ValueError):
            continue
    if len(out) < 2:
        return None
    mean = sum(out) / len(out)
    variance = sum((value - mean) ** 2 for value in out) / len(out)
    return math.sqrt(variance)


if __name__ == "__main__":
    main()
