"""Diagnostics and visualization methods for VisualTokenPruningHook."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..core.metrics import HookMetrics
from ..core.visualization import save_geo_debug_visualization, save_pruning_visualization
from ..core.utils import (
    _compute_token_selection_attribution,
    _first_present,
    _selected_score_stats,
)


class HookDiagnosticsMixin:
    def _apply_selection_metrics(
        self,
        metrics: HookMetrics,
        selection_meta: Dict[str, Any],
        keep_indices_np: np.ndarray,
        aux_metrics: Dict[str, Any],
    ) -> None:
        # Preserve a fallback_used=True that an earlier missing-input branch
        # (e.g. missing_robot_state for legacy robot_geo_* strategies) already
        # set on metrics. The bulk copy loop below pulls fallback_used straight
        # from selection_meta, and the generic select_keep_indices() path returns
        # fallback_used=False, which would otherwise clobber the real signal.
        _prior_fallback_used = bool(getattr(metrics, "fallback_used", False))
        for key in (
            "K_total",
            "K_edge_target",
            "K_geo_target",
            "K_diverse_target",
            "K_edge_actual",
            "K_geo_actual",
            "K_diverse_actual",
            "selected_by_edge_count",
            "selected_by_geo_count",
            "selected_by_diverse_count",
            "overlap_edge_geo_before_dedup",
            # spatial diversity
            "selected_token_grid_entropy",
            # Hybrid v1 simple-name score stats (from select_hybrid_v1 metadata)
            "edge_score_mean", "edge_score_max", "edge_score_std",
            "near_score_mean", "near_score_max", "near_score_std",
            "contact_score_mean", "contact_score_max", "contact_score_std",
            "corridor_score_mean", "corridor_score_max", "corridor_score_std",
            "diversity_score_mean", "diversity_score_max", "diversity_score_std",
            "final_hybrid_score_mean", "final_hybrid_score_max", "final_hybrid_score_std",
            "w_edge", "w_near", "w_contact", "w_corr", "w_diverse",
            "selected_grid_coverage_ratio", "grid_coverage_ratio",
            # Hybrid dynamic (robot_geo_hybrid_dynamic_v0) selection counts
            "selected_by_depth_edge_count", "selected_by_contact_count",
            "selected_by_distance_contact_count", "selected_by_motion_count",
            "selected_by_uniform_count", "selected_by_fill_count",
            "motion_gate_effective", "overlap_depth_contact",
            # temporal / hybrid_temporal fields that come from selection_meta
            "ema_used_for_selection",
            "score_ema_available",
            "lock_condition_failed_reason",
            "topk_contact_lock",
            "elevated_current_lock",
            "gripper_lock",
            "region_lock",
            "requested_keep_ratio",
            "keep_ratio_source",
            "effective_keep_count",
            "original_token_count",
            "actual_keep_ratio",
            "attribution_missing_reason",
            "adaptive_threshold_mean",
            "adaptive_threshold_max",
            "fallback_used",
            # P1-1: token selection attribution
            "workspace_score_min",
            "workspace_score_unique_count",
            "workspace_all_one",
            "workspace_all_one_reason",
            "workspace_fallback_used",
            "workspace_bounds",
            "workspace_valid_token_ratio",
            "near_score_unique_count",
            "depth_edge_score_unique_count",
            "motion_cone_score_unique_count",
            "final_score_unique_count",
            "depth_edge_topk_count",
            "robot_geo_topk_count",
            "hybrid_final_score_topk_count",
            "depth_edge_topk_kept_in_final_count",
            "robot_geo_topk_kept_in_final_count",
            "hybrid_final_score_topk_kept_count",
            "depth_edge_topk_dropped_count",
            "robot_geo_topk_dropped_count",
            "hybrid_final_score_topk_dropped_count",
            "depth_edge_topk_dropped_ratio",
            "robot_geo_topk_dropped_ratio",
            "hybrid_final_score_topk_dropped_ratio",
            "overlap_depth_edge_robot_geo_count",
            "overlap_depth_edge_robot_geo_ratio",
            "selected_near_token_ratio",
            "selected_motion_token_ratio",
            "selected_workspace_token_ratio",
            "selected_robot_token_ratio",
            "selected_scene_token_ratio",
            "selected_background_token_ratio",
            "selected_high_depth_edge_but_low_robot_geo_count",
            "dropped_high_depth_edge_tokens_count",
            "dropped_high_robot_geo_tokens_count",
            "depth_edge_quota_count",
            "robot_geo_quota_count",
            "selected_overlap_robot_geo_depth_edge_ratio",
            "selected_overlap_robot_geo_motion_ratio",
            "selected_overlap_robot_geo_near_ratio",
            "depth_edge_topk_overlap_with_robot_geo_topk",
            "robot_geo_topk_overlap_with_depth_edge_topk",
            "selected_token_u_mean",
            "selected_token_u_std",
            "selected_token_v_mean",
            "selected_token_v_std",
            "selected_token_bbox_u_min",
            "selected_token_bbox_u_max",
            "selected_token_bbox_v_min",
            "selected_token_bbox_v_max",
            "selected_token_grid_quadrant_histogram",
            "selected_token_near_gripper_pixel_dist_mean",
            "selected_token_near_gripper_pixel_dist_median",
            # P5: edge_reserve ablation metrics
            "edge_reserve_enabled",
            "edge_reserve_ratio",
            "edge_reserved_target_count",
            "edge_reserved_actual_count",
            "edge_reserved_survival_ratio",
            "final_selected_count",
            "selected_by_edge_reserved_count",
            "selected_by_original_hybrid_count",
            "selected_by_fill_count",
            "duplicate_edge_hybrid_count",
            "K_edge_reserve_target",
            "K_edge_reserve_actual",
            # P5-fix: new renamed duplicate metrics
            "duplicate_after_exclusion_count",
            "duplicate_with_original_hybrid_count",
            # P5-fix: reserved / non-reserved split diagnostics
            "reserved_edge_topk_count",
            "reserved_edge_kept_count",
            "reserved_edge_dropped_count",
            "reserved_edge_topk_dropped_ratio",
            "non_reserved_edge_topk_count",
            "non_reserved_edge_kept_count",
            "non_reserved_edge_dropped_count",
            "non_reserved_edge_topk_dropped_ratio",
            "overall_depth_edge_topk_count",
            "overall_depth_edge_topk_kept_count",
            "overall_depth_edge_topk_dropped_count",
            "overall_depth_edge_topk_dropped_ratio",
            # P5-fix: new accounting fields
            "selected_by_phase1_hybrid_count",
            "selected_by_phase2_diversity_count",
            "selected_by_phase3_fallback_count",
            "selected_by_unattributed_count",
            "diagnostic_k_small",
            "diagnostic_k_large",
            # P5-fix: edge_reserve invalid flag and diagnostics
            "edge_reserve_invalid",
            "edge_reserve_invalid_reason",
            "edge_scores_available",
            "edge_scores_shape",
            "edge_scores_finite_ratio",
            # P6 normalized selection diagnostics
            "selector_name",
            "selector_function_name",
            "selection_strategy_name",
            "selection_stage_name",
            "keep_indices_unique",
            "keep_indices_out_of_bounds",
            "keep_ratio_requested",
            "keep_ratio_actual",
            "retention_actual",
            "selection_error",
            "selection_warning",
            "selected_by_phase1",
            "selected_by_phase2",
            "selected_by_phase3",
            "selected_by_fill",
            "selected_by_fallback",
            "selected_unattributed",
            "phase_accounting_sum",
            "phase_accounting_valid",
            "phase_accounting_error",
            "reserved_edge_dropped_ratio",
            "non_reserved_topk_count",
            "non_reserved_kept_count",
            "non_reserved_dropped_count",
            "non_reserved_dropped_ratio",
            "total_keep_budget",
            "depth_edge_budget",
            "robot_geo_budget",
            "fill_budget",
            "safety_budget",
            "score_min",
            "score_max",
            "score_mean",
            "score_std",
            "num_visual_tokens_original_total",
            "num_visual_tokens_kept_total",
            "num_visual_tokens_dropped",
            # P15: ACGTP-v1 metrics
            "acgtp_v1",
            "acgtp_w_scene_layout",
            "acgtp_w_depth_structure",
            "acgtp_w_contact_ring",
            "acgtp_w_motion_corridor",
            "acgtp_self_core_radius_px",
            "acgtp_contact_ring_inner_px",
            "acgtp_contact_ring_outer_px",
            "acgtp_self_core_token_count",
            "acgtp_self_core_token_ratio",
            "acgtp_contact_ring_token_count",
            "acgtp_contact_ring_token_ratio",
            "acgtp_contact_ring_gated_token_count",
            "acgtp_contact_ring_valid",
            "acgtp_scene_layout_score_mean",
            "acgtp_scene_layout_score_max",
            "acgtp_support_plane_token_count",
            "acgtp_object_component_token_count",
            "acgtp_boundary_token_count",
            "acgtp_scene_fill_candidate_count",
            "acgtp_scene_fill_candidate_ratio",
            # P6 new fields
            "acgtp_support_plane_candidate_count",
            "acgtp_scene_support_plane_cap_ratio",
            "acgtp_scene_support_plane_cap_used",
            "acgtp_scene_support_plane_fallback_used",
            "acgtp_scene_support_plane_fallback_reason",
            "acgtp_scene_object_component_fallback_used",
            "acgtp_scene_object_component_fallback_reason",
            "acgtp_scene_object_component_num_components",
            "acgtp_scene_boundary_fallback_used",
            "acgtp_scene_boundary_fallback_reason",
            "acgtp_scene_boundary_from_object_count",
            "acgtp_scene_boundary_from_depth_count",
            "acgtp_scene_layout_ms",
            "acgtp_contact_ring_ms",
            "acgtp_motion_corridor_ms",
            "acgtp_scene_selected_support_plane_count",
            "acgtp_scene_selected_object_component_count",
            "acgtp_scene_selected_boundary_count",
            "acgtp_scene_selected_relation_count",
            "acgtp_scene_support_plane_selected_ratio",
            "acgtp_motion_corridor_valid",
            "acgtp_motion_corridor_score_mean",
            "acgtp_motion_corridor_score_max",
            "acgtp_motion_corridor_length_m",
            "acgtp_motion_norm_m",
            "acgtp_motion_ema_alpha",
            "acgtp_depth_structure_score_mean",
            "acgtp_depth_structure_score_max",
            "acgtp_action_constraint_score_mean",
            "acgtp_action_constraint_score_max",
            "acgtp_action_constraint_source",
            "acgtp_future_action_constraint_enabled",
            "acgtp_future_action_constraint_valid",
            "acgtp_future_action_constraint_disabled_reason",
            "acgtp_future_action_constraint_score_mean",
            "acgtp_future_action_constraint_score_max",
            "acgtp_object_side_contact_score_mean",
            "acgtp_object_side_contact_score_max",
            "acgtp_swept_motion_risk_score_mean",
            "acgtp_swept_motion_risk_score_max",
            "acgtp_collision_contact_risk_score_mean",
            "acgtp_collision_contact_risk_score_max",
            "acgtp_contact_object_overlap_count",
            "acgtp_robot_self_penalty_count",
            "acgtp_action_constraint_ms",
            "acgtp_dynamic_enabled",
            "acgtp_dynamic_phase",
            "acgtp_dynamic_candidate_phase",
            "acgtp_dynamic_previous_phase",
            "acgtp_dynamic_hysteresis_state",
            "acgtp_dynamic_risk",
            "acgtp_dynamic_confidence",
            "acgtp_dynamic_keep_ratio",
            "acgtp_dynamic_keep_k",
            "acgtp_dynamic_base_keep_ratio",
            "acgtp_dynamic_raw_keep_ratio",
            "acgtp_dynamic_phase_min_keep_ratio",
            "acgtp_dynamic_phase_max_keep_ratio",
            "acgtp_dynamic_lock_strength",
            "acgtp_dynamic_uncertainty_boost",
            "acgtp_dynamic_risk_boost",
            "acgtp_dynamic_prune_gain",
            "acgtp_dynamic_keep_reason",
            "acgtp_dynamic_layout_motion_alignment",
            "acgtp_dynamic_binary_alignment",
            "acgtp_dynamic_contact_phase_gate",
            "acgtp_dynamic_contact_peak",
            "acgtp_dynamic_contact_mean",
            "acgtp_dynamic_contact_ratio",
            "acgtp_dynamic_motion_peak",
            "acgtp_dynamic_motion_mean",
            "acgtp_dynamic_motion_ratio",
            "acgtp_dynamic_physical_ratio",
            "acgtp_dynamic_high_contact",
            "acgtp_dynamic_high_contact_coverage",
            "acgtp_dynamic_high_contact_legacy",
            "acgtp_dynamic_shadow_contact_guard",
            "acgtp_dynamic_high_motion",
            "acgtp_dynamic_strong_layout",
            "acgtp_dynamic_action_peak",
            "acgtp_dynamic_action_mean",
            "acgtp_dynamic_depth_valid_ratio",
            "acgtp_dynamic_fill_candidate_count",
            "acgtp_dynamic_fill_candidate_ratio",
            "acgtp_dynamic_candidate_gap_count",
            "acgtp_dynamic_candidate_gap_ratio",
            "acgtp_dynamic_candidate_clamped",
            "acgtp_dynamic_scene_weight",
            "acgtp_dynamic_depth_weight",
            "acgtp_dynamic_contact_weight",
            "acgtp_dynamic_motion_weight",
            "acgtp_dynamic_hard_protect_ratio",
            "acgtp_dynamic_budget_vector",
            "acgtp_dynamic_disabled_reason",
            "acgtp_selector_version",
            "acgtp_quota_policy",
            "acgtp_fill_policy",
            "acgtp_hard_protect_count",
            "acgtp_hard_protect_ratio",
            "acgtp_hard_protect_valid",
            "acgtp_scene_quota",
            "acgtp_depth_quota",
            "acgtp_contact_quota",
            "acgtp_motion_quota",
            "acgtp_scene_quota_weight",
            "acgtp_depth_quota_weight",
            "acgtp_contact_quota_weight",
            "acgtp_motion_quota_weight",
            "acgtp_scene_allocated",
            "acgtp_depth_allocated",
            "acgtp_contact_allocated",
            "acgtp_motion_allocated",
            "acgtp_coverage_fill_candidate_count",
            "acgtp_coverage_fill_candidate_ratio",
            "selected_by_scene_layout_count",
            "selected_by_depth_structure_count",
            "selected_by_contact_ring_count",
            "selected_by_motion_corridor_count",
            "selected_by_constrained_fill_count",
            "selected_by_acgtp_fallback_count",
            "overlap_scene_depth_count",
            "overlap_scene_contact_count",
            "overlap_contact_motion_count",
            "overlap_scene_motion_count",
            "overlap_depth_contact_count",
            "overlap_depth_motion_count",
            "acgtp_branch_accounting_valid",
            "acgtp_branch_sum",
            "acgtp_branch_sum_error",
            "branch_accounting_valid",
            "branch_sum_equals_kept",
            "acgtp_hard_protect_ratio_config",
            "acgtp_fallback_used",
            "acgtp_fallback_reason",
            "acgtp_motion_disabled_reason",
            "acgtp_scene_layout_scores",
            "acgtp_contact_ring_scores",
            "acgtp_motion_corridor_scores",
            "acgtp_action_constraint_scores",
            "acgtp_constrained_fill_mask",
            "acgtp_robot_self_core_mask",
            "acgtp_final_kept",
            "acgtp_expected_kept",
            "acgtp_actual_keep_ratio",
            # P16: ACGTP-v2 semantic/attention/fallback fields. These are
            # copied here as well as in the native v2 branch so input-fallback
            # paths still produce a complete CSV schema.
            "acgtp_v2",
            "acgtp_v2_semantic_enabled",
            "acgtp_v2_semantic_backend",
            "acgtp_v2_semantic_confidence",
            "acgtp_v2_semantic_unavailable",
            "acgtp_v2_semantic_fallback_reason",
            "acgtp_v2_release_quota",
            "strict_fallback_dispatch_used",
            "delegated_selector_name",
            "fallback_dispatch_to_v1",
            "semantic_backend",
            "semantic_unavailable",
            "semantic_confidence",
            "semantic_available",
            "semantic_quota_released",
            "selected_by_semantic_count",
            "attention_backend",
            "attention_source",
            "attention_available",
            "attention_confidence",
            "attention_quota_released",
            "selected_by_attention_count",
            "attention_only_token_count",
            "attention_selected_by_final_count",
            "attention_top_count",
            "safe_drop_candidate_count",
            "high_attention_low_geometry_count",
            "high_geometry_low_attention_count",
        ):
            if key in selection_meta:
                setattr(metrics, key, selection_meta.get(key))
        # OR back the pre-existing fallback signal so the bulk copy never
        # downgrades a real missing-input fallback to False.
        if _prior_fallback_used:
            metrics.fallback_used = True
        if selection_meta.get("fallback_reason") and metrics.fallback_reason is None:
            metrics.fallback_reason = selection_meta.get("fallback_reason")
            metrics.fallback_used = True
            metrics.keep_ratio_source = "fallback"

        # P5-fix: edge_reserve accounting invariant checks
        _selected_by_edge_reserved = int(selection_meta.get("selected_by_edge_reserved_count") or 0)
        _selected_by_phase1 = int(selection_meta.get("selected_by_phase1_hybrid_count") or 0)
        _selected_by_phase2 = int(selection_meta.get("selected_by_phase2_diversity_count") or 0)
        _selected_by_phase3 = int(selection_meta.get("selected_by_phase3_fallback_count") or 0)
        _selected_by_fill = int(selection_meta.get("selected_by_fill_count") or 0)
        _selected_by_unattributed = int(selection_meta.get("selected_by_unattributed_count") or 0)
        _final_selected = int(selection_meta.get("final_selected_count") or 0)
        _k_diag_small = int(selection_meta.get("diagnostic_k_small") or 0)
        _k_diag_large = int(selection_meta.get("diagnostic_k_large") or 0)
        _reserved_drop_ratio = selection_meta.get("reserved_edge_topk_dropped_ratio")
        _edge_reserve_enabled = selection_meta.get("edge_reserve_enabled") is True
        _edge_reserve_seen = any(
            key in selection_meta
            for key in (
                "edge_reserve_enabled",
                "selected_by_edge_reserved_count",
                "reserved_edge_topk_count",
                "non_reserved_edge_topk_count",
                "overall_depth_edge_topk_count",
            )
        )

        # Invariant 1: all selected_by_* sum to final_selected_count
        _attributed_sum = _selected_by_edge_reserved + _selected_by_phase1 + _selected_by_phase2 + _selected_by_phase3
        _accounting_valid = bool(_attributed_sum == _final_selected and _selected_by_unattributed == 0)
        _accounting_error = None
        if _attributed_sum != _final_selected:
            _accounting_error = f"attributed_sum={_attributed_sum} != final_selected={_final_selected}"
        elif _selected_by_unattributed > 0:
            _accounting_error = f"unattributed={_selected_by_unattributed} > 0"

        # Invariant 2: if edge_reserve_enabled and survival=1.0 then reserved_drop_ratio=0
        _split_valid = True
        if _edge_reserve_enabled and _reserved_drop_ratio is not None and _reserved_drop_ratio > 0:
            _split_valid = False

        # Invariant 3: selected_token_count equals num_visual_tokens_kept
        _retention_valid = True
        if metrics.selected_token_count is not None and metrics.num_visual_tokens_kept is not None:
            _retention_valid = bool(metrics.selected_token_count == metrics.num_visual_tokens_kept)

        # Invariant 4: keep_indices have no duplicates and are sorted
        idx_raw = np.asarray(keep_indices_np, dtype=np.int64) if keep_indices_np is not None else np.array([], dtype=np.int64)
        _no_dup = bool(len(idx_raw) == len(set(idx_raw))) if idx_raw.size > 0 else True
        _sorted = bool(np.all(idx_raw[:-1] <= idx_raw[1:])) if idx_raw.size > 1 else True

        # P5-fix: record invariant check results
        metrics.edge_reserve_accounting_valid = bool(_accounting_valid) if _edge_reserve_seen else None
        metrics.edge_reserve_accounting_error = str(_accounting_error) if (_edge_reserve_seen and _accounting_error) else None
        metrics.edge_reserve_split_metrics_valid = bool(_split_valid) if _edge_reserve_seen else None
        metrics.diagnostic_k_small = _k_diag_small if _k_diag_small > 0 else None
        metrics.diagnostic_k_large = _k_diag_large if _k_diag_large > 0 else None
        metrics.no_duplicate_final_indices = _no_dup
        metrics.final_indices_sorted = _sorted
        metrics.selected_token_count_equals_kept = _retention_valid
        metrics.retention_ratio_valid = _retention_valid

        idx = np.asarray(keep_indices_np, dtype=np.int64)
        _n_from_aux = aux_metrics.get("num_tokens")
        if _n_from_aux is not None and int(_n_from_aux) > 0:
            num_tokens = int(_n_from_aux)
        else:
            num_tokens = int(idx.size) if idx.size else 256
        if idx.size == 0:
            return

        # Selected token statistics for each score component
        selected_stats = _selected_score_stats(aux_metrics, idx, num_tokens)
        for key, value in selected_stats.items():
            if value is not None and hasattr(metrics, key):
                setattr(metrics, key, value)

        # P1-1: Token selection attribution and top-k competition diagnostics
        token_grid_shape = aux_metrics.get("token_grid_shape", (16, 16))
        try:
            attribution = _compute_token_selection_attribution(
                aux_metrics, idx, num_tokens, token_grid_shape
            )
        except Exception as exc:
            attribution = {}
            metrics.attribution_missing_reason = (
                f"_compute_token_selection_attribution_error:{type(exc).__name__}:{exc}"
            )
        for key, value in attribution.items():
            if value is not None and hasattr(metrics, key):
                setattr(metrics, key, value)

    def _record_selection_path_diagnostics(
        self,
        metrics: "HookMetrics",
        selector_success: bool,
        exc: Optional[Exception],
        requested_strategy: str,
        effective_strategy: str,
        selector_name: str,
        fallback_selector: Optional[str],
        keep_indices_np: Optional[np.ndarray],
        num_tokens: int,
        selection_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record Stage X selection-path integrity diagnostics on every pruning step."""
        meta = selection_meta or {}
        metrics.requested_pruning_strategy = requested_strategy
        metrics.effective_pruning_strategy = effective_strategy
        metrics.selector_name = meta.get("selector_name", metrics.selector_name)
        metrics.selector_function_name = meta.get("selector_function_name", selector_name)
        metrics.selection_strategy_name = meta.get("selection_strategy_name", effective_strategy)
        metrics.selection_stage_name = meta.get("selection_stage_name")
        metrics.selector_success = selector_success
        # Do NOT override metrics.fallback_used here.
        # fallback_used is already used in this codebase for "fallback due to missing inputs"
        # (e.g., geometry/robot metrics unavailable). Stage X only annotates selector-* fallback.
        if exc is not None:
            metrics.selector_exception_type = type(exc).__name__
            msg = str(exc)
            metrics.selector_exception_msg = msg[:200] if msg else None
            if metrics.selection_error is None:
                metrics.selection_error = f"{type(exc).__name__}:{metrics.selector_exception_msg}"
        else:
            metrics.selector_exception_type = None
            metrics.selector_exception_msg = None
        metrics.fallback_selector_name = meta.get("fallback_selector_name", fallback_selector)
        if keep_indices_np is not None:
            metrics.keep_indices_count = int(len(keep_indices_np))
            idx = np.asarray(keep_indices_np, dtype=np.int64)
            metrics.keep_indices_sorted = bool(
                np.all(idx[:-1] <= idx[1:]) if idx.shape[0] > 1 else True
            )
            metrics.keep_indices_unique = bool(np.unique(idx).size == idx.size)
            metrics.keep_indices_out_of_bounds = bool(np.any((idx < 0) | (idx >= int(num_tokens))))
            metrics.actual_retention_ratio = float(len(keep_indices_np)) / float(num_tokens) if num_tokens else None
            metrics.keep_ratio_actual = metrics.actual_retention_ratio
            metrics.retention_actual = metrics.actual_retention_ratio
            # Infer keep_indices_source
            if selector_success:
                src = metrics.selector_function_name
            elif metrics.fallback_selector_name is not None:
                src = f"fallback_{metrics.fallback_selector_name}"
            else:
                src = None
            metrics.keep_indices_source = src
        else:
            metrics.keep_indices_count = None
            metrics.keep_indices_sorted = None
            metrics.keep_indices_unique = None
            metrics.keep_indices_out_of_bounds = None
            metrics.actual_retention_ratio = None
            metrics.keep_ratio_actual = None
            metrics.retention_actual = None
            metrics.keep_indices_source = None

        for key in (
            "keep_ratio_requested",
            "selection_error",
            "selection_warning",
            "selected_by_phase1",
            "selected_by_phase2",
            "selected_by_phase3",
            "selected_by_fill",
            "selected_by_fallback",
            "selected_unattributed",
            "phase_accounting_sum",
            "phase_accounting_valid",
            "phase_accounting_error",
            "reserved_edge_dropped_ratio",
            "non_reserved_topk_count",
            "non_reserved_kept_count",
            "non_reserved_dropped_count",
            "non_reserved_dropped_ratio",
            "total_keep_budget",
            "depth_edge_budget",
            "robot_geo_budget",
            "fill_budget",
            "safety_budget",
            "score_min",
            "score_max",
            "score_mean",
            "score_std",
            # P7: hybrid_budget_v2 fields (copied from selection_meta)
            "hybrid_budget_v2",
            "depth_edge_budget_ratio",
            "robot_contact_budget_ratio",
            "safety_budget_ratio",
            "K_depth_actual",
            "K_robot_actual",
            "K_fill_actual",
            "overlap_depth_robot_count",
            "overlap_depth_robot_diagnostic",
            "depth_edge_candidates_count",
            "robot_geo_candidates_count",
            "depth_edge_reserved_kept_count",
            "robot_geo_reserved_kept_count",
            "fill_from_depth_count",
            "fill_from_robot_count",
            "fill_from_other_count",
            "num_visual_tokens_original_total",
            "num_visual_tokens_kept_total",
            "num_visual_tokens_dropped",
            # P11.3: DE top-k attribution by branch (selector.py populates these in metadata)
            "depth_edge_topk_kept_by_depth_branch_count",
            "depth_edge_topk_kept_by_hybrid_branch_count",
            "depth_edge_topk_kept_by_fill_branch_count",
            "depth_edge_topk_kept_by_fallback_count",
            "depth_edge_topk_survival_ratio",
            # P11.3: Hybrid/final-score top-k attribution by branch
            "hybrid_final_score_topk_kept_by_depth_branch_count",
            "hybrid_final_score_topk_kept_by_hybrid_branch_count",
            "hybrid_final_score_topk_kept_by_fill_branch_count",
            "hybrid_final_score_topk_kept_by_fallback_count",
            "hybrid_final_score_topk_survival_ratio",
            # P11.3: Legacy hybrid top-k aliases
            "hybrid_topk_kept_by_depth_branch_count",
            "hybrid_topk_kept_by_hybrid_branch_count",
            "hybrid_topk_kept_by_fill_branch_count",
            "hybrid_topk_kept_by_fallback_count",
            "hybrid_topk_survival_ratio",
            # P11: branch_budget_v0 fields
            "branch_budget_v0",
            "hybrid_action_budget",
            "diversity_fill_budget",
            "temporal_budget",
            "branch_accounting_valid",
            "branch_sum_equals_kept",
            "selected_by_depth_branch",
            "selected_by_hybrid_branch",
            "selected_by_fill",
            "selected_by_fallback",
            "overlap_depth_edge_hybrid_count",
            "overlap_depth_edge_hybrid_ratio",
            "non_reserved_depth_edge_dropped_ratio",
            "non_reserved_depth_edge_count",
            "non_reserved_depth_edge_kept",
            "non_reserved_depth_edge_dropped",
            "branch_budget_depth_ratio_override",
            "branch_budget_hybrid_ratio_override",
            "depth_edge_budget_actual",
            "hybrid_action_budget_actual",
            "diversity_fill_budget_actual",
            "depth_branch_indices",
            "hybrid_branch_indices",
            "fill_branch_indices",
        ):
            if key in meta:
                setattr(metrics, key, meta.get(key))
        if metrics.keep_ratio_requested is None:
            metrics.keep_ratio_requested = metrics.requested_keep_ratio

        # If pruning is enabled, always record the requested/effective strategy.
        # For strategies that don't use selector fallback (normal case), also fill success=True.
        if metrics.requested_pruning_strategy is None:
            metrics.requested_pruning_strategy = requested_strategy
        if metrics.effective_pruning_strategy is None:
            metrics.effective_pruning_strategy = effective_strategy
        if metrics.selector_function_name is None:
            metrics.selector_function_name = selector_name
        if metrics.selector_success is None:
            metrics.selector_success = bool(selector_success)

    def _maybe_save_visualization(
        self,
        *,
        keep_indices_np: np.ndarray,
        scores: Optional[np.ndarray],
        aux_metrics: Dict[str, Any],
        selection_meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        latest = self.geometry_recorder.get_latest() if self.geometry_recorder is not None else None
        if latest is None:
            return None
        episode_id = int(getattr(latest, "episode_id", 0))
        step_id = int(getattr(latest, "step_id", 0))
        if self.cfg.get("save_pruning_debug", False):
            debug_tasks = self.cfg.get("debug_tasks")
            if debug_tasks:
                task_name = str(getattr(latest, "task_name", ""))
                task_idx = task_name.split("_")[-1] if task_name.startswith("task_") else None
                allowed = {str(x).strip() for x in str(debug_tasks).split(",") if str(x).strip()}
                if task_name not in allowed and task_idx not in allowed:
                    return None
        target_episode = self.cfg.get("pruning_vis_episode")
        target_step = self.cfg.get("pruning_vis_step")
        if target_episode is not None and int(target_episode) != episode_id:
            return None
        if target_step is not None and int(target_step) != step_id:
            return None
        try:
            base_output = str(self.cfg.get("save_dir") or "outputs")
            cache = aux_metrics.get("cache", {})
            score_maps = {
                "depth_edge_score": aux_metrics.get("edge_scores"),
                "gripper_distance_score": aux_metrics.get("near_scores"),
                "near_score": aux_metrics.get("near_scores"),
                "motion_corridor_score": aux_metrics.get("corridor_scores"),
                "corridor_score": aux_metrics.get("corridor_scores"),
                "near_contact_score": aux_metrics.get("near_contact_scores"),
                "corridor_contact_score": aux_metrics.get("corridor_contact_scores"),
                "geo_contact_score": aux_metrics.get("geo_contact_scores"),
                "final_geometry_score": scores,
            }
            selection_masks = None
            if selection_meta:
                selection_masks = {
                    "edge": selection_meta.get("selected_edge_indices", []),
                    "geo": selection_meta.get("selected_geo_indices", []),
                    "diverse": selection_meta.get("selected_diverse_indices", []),
                }
            return save_pruning_visualization(
                output_dir=base_output,
                method=self.config.strategy,
                episode_id=episode_id,
                step_id=step_id,
                rgb=getattr(latest, "rgb", None),
                depth=getattr(latest, "depth", None),
                token_u=cache.get("u") if isinstance(cache, dict) else None,
                token_v=cache.get("v") if isinstance(cache, dict) else None,
                keep_indices=keep_indices_np,
                score_maps=score_maps,
                selection_masks=selection_masks,
                token_grid_shape=self._latest_token_grid_shape or self.config.token_grid_shape,
            )
        except Exception as exc:
            if self.config.debug:
                print(f"[PRUNING VIS] warning: failed to save pruning visualization: {exc}")
            return None

    def _maybe_save_geo_debug_visualization(
        self,
        *,
        keep_indices_np: np.ndarray,
        scores: Optional[np.ndarray],
        aux_metrics: Dict[str, Any],
    ) -> Optional[str]:
        if not self.config.enable_geo_debug:
            return None
        if self.config.max_debug_frames <= 0:
            return None
        if self._geo_debug_frames_saved >= self.config.max_debug_frames:
            return None
        latest = self.geometry_recorder.get_latest() if self.geometry_recorder is not None else None
        if latest is None:
            return None
        step_id = int(getattr(latest, "step_id", 0))
        if step_id % self.config.geo_debug_interval != 0:
            return None
        try:
            cache = aux_metrics.get("cache", {})
            dynamic_decision = aux_metrics.get("dynamic_decision") or {}
            component_summary = dynamic_decision.get("component_summary", {}) if isinstance(dynamic_decision, dict) else {}
            dynamic_info = {
                "dynamic_keep_ratio": dynamic_decision.get("keep_ratio") if isinstance(dynamic_decision, dict) else None,
                "risk_level": dynamic_decision.get("risk_level") if isinstance(dynamic_decision, dict) else None,
                "risk_score": dynamic_decision.get("risk_score") if isinstance(dynamic_decision, dict) else None,
                "reason": dynamic_decision.get("reason") if isinstance(dynamic_decision, dict) else None,
                "num_high_contact_tokens": component_summary.get("num_high_contact_tokens") if isinstance(component_summary, dict) else None,
                "num_valid_3d_tokens": component_summary.get("num_valid_3d_tokens") if isinstance(component_summary, dict) else None,
            }
            score_maps = {
                "distance_to_gripper_score": _first_present(aux_metrics, "rule_v0_distance_scores", "near_scores"),
                "motion_cone_score": _first_present(aux_metrics, "rule_v0_motion_cone_scores", "corridor_scores"),
                "contact_risk_score": _first_present(aux_metrics, "rule_v0_contact_risk_scores", "geo_contact_scores"),
                "depth_edge_score": aux_metrics.get("edge_scores"),
                "final_geometry_score": scores,
            }
            path = save_geo_debug_visualization(
                enabled=True,
                output_dir=str(self.cfg.get("save_dir") or "outputs"),
                method=self.config.strategy,
                episode_id=int(getattr(latest, "episode_id", 0)),
                step_id=step_id,
                keep_indices=keep_indices_np,
                score_maps=score_maps,
                dynamic_info=dynamic_info,
                rgb=getattr(latest, "rgb", None),
                token_u=cache.get("u") if isinstance(cache, dict) else None,
                token_v=cache.get("v") if isinstance(cache, dict) else None,
                token_grid_shape=self._latest_token_grid_shape or self.config.token_grid_shape,
            )
            if path is not None:
                self._geo_debug_frames_saved += 1
            return path
        except Exception as exc:
            if self.config.debug:
                print(f"[GEO DEBUG VIS] warning: failed to save geometry debug visualization: {exc}")
            return None

    def _save_token_selection_debug_visualization(
        self,
        *,
        keep_indices_np: np.ndarray,
        scores: Optional[np.ndarray],
        aux_metrics: Dict[str, Any],
        selection_meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """P1: Save token selection debug overlay (first 3 steps of hybrid_temporal_v1 runs)."""
        try:
            from ..core.visualization import save_token_selection_debug_visualization
        except ImportError:
            return None
        latest = self.geometry_recorder.get_latest() if self.geometry_recorder is not None else None
        if latest is None:
            return None
        episode_id = int(getattr(latest, "episode_id", 0))
        step_id = int(getattr(latest, "step_id", 0))
        grid_shape = aux_metrics.get("token_grid_shape", (16, 16))
        return save_token_selection_debug_visualization(
            output_dir=str(self.cfg.get("save_dir") or "outputs"),
            method=self.config.strategy,
            episode_id=episode_id,
            step_id=step_id,
            rgb=getattr(latest, "rgb", None),
            token_u=aux_metrics.get("token_u"),
            token_v=aux_metrics.get("token_v"),
            keep_indices=keep_indices_np,
            depth_edge_scores=aux_metrics.get("edge_scores"),
            robot_geo_scores=(
                aux_metrics.get("hybrid_final_scores")
                if aux_metrics.get("hybrid_final_scores") is not None
                else aux_metrics.get("final_scores")
            ),
            gripper_pixel=aux_metrics.get("gripper_pixel"),
            token_grid_shape=grid_shape,
        )

    def _reconstruct_dropped_token_sets(
        self,
        *,
        keep_indices_np: np.ndarray,
        aux_metrics: Dict[str, Any],
        selection_meta: Optional[Dict[str, Any]],
        num_tokens: int,
        grid_shape: Tuple[int, int],
    ) -> Dict[str, set]:
        """P8.3: Reconstruct token sets for dropped-token visualization.

        This reconstructs the sets from scores already stored in aux_metrics and
        counts stored in selection_meta. It does NOT change selection logic or scores.

        Priority order for sourcing counts (highest to lowest):
          1. selection_meta (authoritative — from the real selector accounting)
          2. aux_metrics edge_scores (for reconstruction fallback)
          3. int(round(k_final * ratio)) (last resort, with reconstruction_exact=false)

        Returns a dict with:
          - depth_edge_topk_indices
          - robot_geo_topk_indices
          - depth_edge_topk_dropped_indices
          - robot_geo_topk_dropped_indices
          - reserved_edge_indices
          - non_reserved_edge_dropped_indices
          - reconstruction_metadata: dict with count sources and exactness flags
        """
        import numpy as np

        result: Dict[str, set] = {
            "depth_edge_topk_indices": set(),
            "robot_geo_topk_indices": set(),
            "depth_edge_topk_dropped_indices": set(),
            "robot_geo_topk_dropped_indices": set(),
            "reserved_edge_indices": set(),
            "non_reserved_edge_dropped_indices": set(),
            # P11.3: Branch attribution sets
            "depth_branch_indices": set(),
            "hybrid_branch_indices": set(),
            "fill_branch_indices": set(),
            "fallback_branch_indices": set(),
        }
        reconstruction_meta: Dict[str, Any] = {
            "depth_edge_topk_count_source": None,
            "depth_edge_topk_count_exact": False,
            "reserved_edge_count_source": None,
            "reserved_edge_count_exact": False,
            "non_reserved_edge_dropped_count_source": None,
            "non_reserved_edge_dropped_count_exact": False,
        }

        n = int(num_tokens)
        if n <= 0:
            result["reconstruction_metadata"] = reconstruction_meta
            return result

        idx = np.asarray(list(keep_indices_np), dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < n)]
        keep_set = set(int(i) for i in idx.tolist())
        all_set = set(range(n))
        dropped_set = all_set - keep_set

        k_final = len(keep_set)

        # ── Depth-edge top-k ────────────────────────────────────────────────
        # Priority: 1) selection_meta (authoritative), 2) aux_metrics edge_scores
        # (score-based reconstruction is exact because scores are deterministic).
        de_topk: set = set()
        edge_arr = aux_metrics.get("edge_scores")
        if edge_arr is not None:
            edge_flat = np.asarray(edge_arr, dtype=np.float32).reshape(-1)
            if edge_flat.size >= n:
                # Try authoritative count from selection_meta first
                _sm_de_count = selection_meta.get("depth_edge_topk_count") if selection_meta else None
                if _sm_de_count is not None:
                    _de_k = int(_sm_de_count)
                    reconstruction_meta["depth_edge_topk_count_source"] = "selection_meta"
                    reconstruction_meta["depth_edge_topk_count_exact"] = True
                else:
                    # Fall back to score-based: use the same ratio the selector used
                    _de_k = max(1, int(round(k_final * 0.80)))
                    reconstruction_meta["depth_edge_topk_count_source"] = "reconstructed_from_edge_scores_and_ratio"
                    reconstruction_meta["depth_edge_topk_count_exact"] = False
                    reconstruction_meta["depth_edge_topk_ratio"] = 0.80
                edge_adj = np.where(np.isfinite(edge_flat), edge_flat, -np.inf)
                edge_order = np.argsort(-edge_adj)
                for i in range(min(_de_k, n)):
                    de_topk.add(int(edge_order[i]))
                result["depth_edge_topk_indices"] = de_topk
                result["depth_edge_topk_dropped_indices"] = de_topk - keep_set

        # ── Hybrid/final-score top-k ─────────────────────────────────────────
        # For depth_edge_fast: hybrid_final_scores == edge_scores, so this would
        # duplicate de_topk. We mark it null for depth_edge_fast in metadata later.
        # For hybrid/edge_reserve methods: this is a distinct signal (weighted combo).
        hybrid_arr = aux_metrics.get("hybrid_final_scores")
        final_arr = hybrid_arr if hybrid_arr is not None else aux_metrics.get("final_scores")
        if final_arr is not None:
            final_flat = np.asarray(final_arr, dtype=np.float32).reshape(-1)
            if final_flat.size >= n:
                # Use the same k as depth_edge top-k for consistency in cross-method comparison
                _rg_k = len(de_topk) if de_topk else max(1, int(round(k_final * 0.80)))
                final_adj = np.where(np.isfinite(final_flat), final_flat, -np.inf)
                final_order = np.argsort(-final_adj)
                rg_topk: set = set()
                for i in range(min(_rg_k, n)):
                    rg_topk.add(int(final_order[i]))
                result["robot_geo_topk_indices"] = rg_topk
                result["robot_geo_topk_dropped_indices"] = rg_topk - keep_set

        # ── Reserved edge tokens (edge_reserve methods only) ─────────────────
        # Priority: 1) selection_meta reserved_edge_topk_count (authoritative),
        # 2) aux_metrics edge_scores + k from selection_meta, 3) score fallback.
        if edge_arr is not None and "edge_reserve" in self.config.strategy:
            edge_flat2 = np.asarray(edge_arr, dtype=np.float32).reshape(-1)
            if edge_flat2.size >= n:
                edge_adj2 = np.where(np.isfinite(edge_flat2), edge_flat2, -np.inf)
                edge_order2 = np.argsort(-edge_adj2)

                # Try authoritative count from selection_meta
                _sm_res_count = selection_meta.get("reserved_edge_topk_count") if selection_meta else None
                if _sm_res_count is not None:
                    _res_k = int(_sm_res_count)
                    reconstruction_meta["reserved_edge_count_source"] = "selection_meta"
                    reconstruction_meta["reserved_edge_count_exact"] = True
                else:
                    # Try to reconstruct from edge_scores using edge_reserve_k from aux_metrics
                    _erk = aux_metrics.get("edge_reserve_k")
                    if _erk is not None:
                        _res_k = int(_erk)
                        reconstruction_meta["reserved_edge_count_source"] = "reconstructed_from_edge_scores_and_aux_metrics_edge_reserve_k"
                        reconstruction_meta["reserved_edge_count_exact"] = True
                    else:
                        # Last resort: approximate (not exact)
                        _res_k = max(0, int(round(0.40 * k_final)))
                        reconstruction_meta["reserved_edge_count_source"] = "fallback_int_0_40_k_final"
                        reconstruction_meta["reserved_edge_count_exact"] = False
                        reconstruction_meta["reserved_edge_fallback_ratio"] = 0.40

                res_set: set = set()
                for i in range(min(_res_k, n)):
                    res_set.add(int(edge_order2[i]))
                result["reserved_edge_indices"] = res_set

                # Non-reserved edge dropped = dropped tokens that were in DE top-k but NOT reserved
                non_res_in_de_topk = de_topk - res_set if de_topk else set()
                result["non_reserved_edge_dropped_indices"] = non_res_in_de_topk & dropped_set

                # Track non_reserved source
                _sm_nres_drop = selection_meta.get("non_reserved_edge_dropped_count") if selection_meta else None
                if _sm_nres_drop is not None:
                    reconstruction_meta["non_reserved_edge_dropped_count_source"] = "selection_meta"
                    reconstruction_meta["non_reserved_edge_dropped_count_exact"] = True
                else:
                    reconstruction_meta["non_reserved_edge_dropped_count_source"] = "reconstructed"
                    reconstruction_meta["non_reserved_edge_dropped_count_exact"] = True  # exact from scores

        # ── P11.3: Branch attribution sets from selection_meta ─────────────────
        if selection_meta is not None:
            _depth_branch = selection_meta.get("depth_branch_indices")
            _hybrid_branch = selection_meta.get("hybrid_branch_indices")
            _fill_branch = selection_meta.get("fill_branch_indices")
            if _depth_branch is not None and isinstance(_depth_branch, (list, set, tuple)):
                result["depth_branch_indices"] = set(int(i) for i in _depth_branch if 0 <= int(i) < n)
            if _hybrid_branch is not None and isinstance(_hybrid_branch, (list, set, tuple)):
                result["hybrid_branch_indices"] = set(int(i) for i in _hybrid_branch if 0 <= int(i) < n)
            if _fill_branch is not None and isinstance(_fill_branch, (list, set, tuple)):
                result["fill_branch_indices"] = set(int(i) for i in _fill_branch if 0 <= int(i) < n)

        result["reconstruction_metadata"] = reconstruction_meta
        return result

    def _save_dropped_token_debug_visualization(
        self,
        *,
        keep_indices_np: np.ndarray,
        aux_metrics: Dict[str, Any],
        selection_meta: Optional[Dict[str, Any]],
        num_tokens: int,
    ) -> Optional[str]:
        """P8: Save enhanced overlay with dropped token categories.

        This is visualization-only. It only calls save_token_selection_debug_with_dropped
        which renders token sets; it does NOT change any selection logic or scores.
        """
        try:
            from ..core.visualization import save_token_selection_debug_with_dropped
        except ImportError:
            return None

        latest = self.geometry_recorder.get_latest() if self.geometry_recorder is not None else None
        if latest is None:
            return None
        if getattr(latest, "rgb", None) is None:
            return None

        episode_id = int(getattr(latest, "episode_id", 0))
        step_id = int(getattr(latest, "step_id", 0))
        grid_shape = aux_metrics.get("token_grid_shape", (16, 16))

        token_sets = self._reconstruct_dropped_token_sets(
            keep_indices_np=keep_indices_np,
            aux_metrics=aux_metrics,
            selection_meta=selection_meta,
            num_tokens=num_tokens,
            grid_shape=grid_shape,
        )

        return save_token_selection_debug_with_dropped(
            output_dir=str(self.cfg.get("save_dir") or "outputs"),
            method=self.config.strategy,
            episode_id=episode_id,
            step_id=step_id,
            rgb=getattr(latest, "rgb", None),
            token_u=aux_metrics.get("token_u"),
            token_v=aux_metrics.get("token_v"),
            keep_indices=keep_indices_np,
            depth_edge_topk_indices=list(token_sets.get("depth_edge_topk_indices", set())),
            depth_edge_dropped_indices=list(token_sets.get("depth_edge_topk_dropped_indices", set())),
            robot_geo_topk_indices=list(token_sets.get("robot_geo_topk_indices", set())),
            robot_geo_dropped_indices=list(token_sets.get("robot_geo_topk_dropped_indices", set())),
            reserved_edge_indices=list(token_sets.get("reserved_edge_indices", set())),
            non_reserved_edge_dropped_indices=list(token_sets.get("non_reserved_edge_dropped_indices", set())),
            # P11.3: Branch attribution sets from selection_meta
            depth_branch_indices=list(token_sets.get("depth_branch_indices", set())),
            hybrid_branch_indices=list(token_sets.get("hybrid_branch_indices", set())),
            fill_branch_indices=list(token_sets.get("fill_branch_indices", set())),
            gripper_pixel=aux_metrics.get("gripper_pixel"),
            token_grid_shape=grid_shape,
            selection_meta=selection_meta,
            reconstruction_metadata=token_sets.get("reconstruction_metadata", {}),
        )
