"""Legacy/debug projector runtime path."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..signals.temporal import decide_acgtp_dynamic_budget, compute_dynamic_keep_ratio
from ..core.metrics import HookMetrics, HookTiming
from .acgtp_v1 import select_acgtp_v1
from ..methods.acgtp_v2 import select_acgtp_v2_fast, select_acgtp_v2
from .branch_budget import select_branch_budget_v0
from .hybrid import (
    select_hybrid_budget_v2,
    select_hybrid_quota_v2,
    select_hybrid_v1,
    select_hybrid_v1_edge_reserve,
)
from ..methods.baselines import select_keep_indices, select_tokens_contact_budget
from ..methods.utils import finalize_selection_debug_info, validate_keep_indices
from ..signals.semantic import compute_task_semantic_anchors, parse_instruction_terms
from ..strategy_registry import (
    ACGTP_STRATEGIES,
    EARLY_GEOMETRY_FALLBACK_STRATEGIES,
    EDGE_RESERVE_LEGACY_STRATEGIES,
    ROBOT_STATE_REQUIRED_LEGACY_STRATEGIES,
    SELF_HANDLED_SELECTOR_STRATEGIES,
    ROBOT_GEO_SCORE_STRATEGIES,
    TOKEN_SELECTION_DEBUG_STRATEGIES,
)


class HookLegacyRuntimeMixin:
    def _run(self, visual_tokens: torch.Tensor) -> Tuple[torch.Tensor, HookMetrics]:
        num_tokens = int(visual_tokens.shape[1])
        keep_count = self.config.keep_count(num_tokens)
        metrics = HookMetrics(
            num_visual_tokens_original=num_tokens,
            num_visual_tokens_kept=num_tokens,
            num_visual_tokens_pruned=0,
            keep_ratio=1.0,
            requested_keep_ratio=float(self.config.keep_ratio),
            keep_ratio_source=self._default_keep_ratio_source(),
            effective_keep_count=keep_count,
            original_token_count=num_tokens,
            pruning_strategy=self.config.strategy,
            pruning_method=self.config.strategy,
            protected_token_ratio=1.0,
            pruned_token_ratio=0.0,
            timing=HookTiming(),
        )

        _skip_postprocess_diag = False
        if not self.config.enabled:
            metrics.keep_indices_sorted = True
            metrics.keep_indices_unique = True
            metrics.keep_indices_out_of_bounds = False
            metrics.duplicate_indices_count = 0
            metrics.effective_keep_count = num_tokens
            metrics.original_token_count = num_tokens
            metrics.requested_keep_ratio = float(self.config.keep_ratio)
            metrics.actual_keep_ratio = 1.0
            metrics.keep_ratio_requested = float(self.config.keep_ratio)
            metrics.keep_ratio_actual = 1.0
            metrics.retention_actual = 1.0
            metrics.actual_retention_ratio = 1.0
            metrics.num_visual_tokens_dropped = 0
            metrics.num_visual_tokens_original_total = num_tokens
            metrics.num_visual_tokens_kept_total = num_tokens
            metrics.keep_ratio_source = "cli_keep_ratio"
            metrics.selector_name = "none"
            metrics.selection_strategy_name = "none"
            metrics.selection_stage_name = "disabled_no_pruning"
            metrics.selector_success = None
            metrics.fallback_used = False
            metrics.fallback_reason = None
            self._prepare_position_preserve_info(
                keep_indices_np=np.arange(num_tokens, dtype=np.int64),
                num_tokens=int(num_tokens),
                metrics=metrics,
            )
            return visual_tokens, metrics
        latest_for_warmup = self.geometry_recorder.get_latest() if self.geometry_recorder is not None else None
        metrics.is_warmup_step = self._mark_warmup_step(latest_for_warmup)

        # ── 1. Uniform init of ALL intermediate state ───────────────────────
        scores: Optional[np.ndarray] = None
        valid_mask: Optional[np.ndarray] = None
        fallback_reason: Optional[str] = None
        keep_indices_np: Optional[np.ndarray] = None
        selection_meta: Dict[str, Any] = {}
        robot_metrics: Dict[str, Any] = {}

        if self._acgtp_fast_runtime_enabled():
            return self._run_acgtp_fast_runtime(
                visual_tokens,
                metrics,
                keep_count=keep_count,
                num_tokens=num_tokens,
            )

        # P8: Determine which aux metrics are active for this strategy.
        # Used by visualization and geo_debug — avoids the broken
        #   robot_metrics if "robot_metrics" in locals() else depth_metrics ...
        # pattern where robot_metrics={} is always in locals() even for depth_edge_fast.
        if self.config.strategy in ("depth_edge_fast", "depth_edge_fast_diverse"):
            scores, valid_mask, depth_metrics, fallback_reason = self._compute_depth_edge_scores(num_tokens)
            metrics.timing.token_mapping_ms = depth_metrics.get("token_mapping_ms")
            metrics.timing.depth_sampling_ms = depth_metrics.get("depth_sampling_ms")
            metrics.timing.depth_sample_ms = metrics.timing.depth_sampling_ms
            metrics.timing.score_compute_ms = depth_metrics.get("score_compute_ms")
            metrics.timing.depth_edge_score_ms = metrics.timing.score_compute_ms
            metrics.valid_token_ratio = depth_metrics.get("valid_token_ratio")
            metrics.geometry_score_mean = depth_metrics.get("score_mean")
            metrics.geometry_score_max = depth_metrics.get("score_max")
            metrics.geometry_score_std = depth_metrics.get("score_std")
            metrics.depth_edge_score_mean = depth_metrics.get("score_mean")
            depth_metrics.setdefault("edge_scores", scores)
            depth_metrics.setdefault("final_scores", scores)
            depth_metrics.setdefault("hybrid_final_scores", scores)
            # P1-1: Add token_u, token_v, gripper_pixel for attribution if available from cache
            if "u" in depth_metrics.get("cache", {}):
                depth_metrics.setdefault("token_u", depth_metrics["cache"]["u"])
            if "v" in depth_metrics.get("cache", {}):
                depth_metrics.setdefault("token_v", depth_metrics["cache"]["v"])
            depth_metrics.setdefault("token_grid_shape", self.config.token_grid_shape)
            # P0: depth conversion metadata
            metrics.depth_source_key = depth_metrics.get("depth_source_key")
            metrics.depth_conversion = depth_metrics.get("depth_conversion")
            metrics.depth_is_metric = depth_metrics.get("depth_is_metric")
            metrics.depth_unit = depth_metrics.get("depth_unit")
            metrics.depth_sim_available = depth_metrics.get("depth_sim_available")
            metrics.depth_raw_min = depth_metrics.get("depth_raw_min")
            metrics.depth_raw_max = depth_metrics.get("depth_raw_max")
            metrics.depth_raw_mean = depth_metrics.get("depth_raw_mean")
            metrics.depth_raw_std = depth_metrics.get("depth_raw_std")
            metrics.depth_metric_min = depth_metrics.get("depth_metric_min")
            metrics.depth_metric_max = depth_metrics.get("depth_metric_max")
            metrics.depth_metric_mean = depth_metrics.get("depth_metric_mean")
            metrics.depth_metric_std = depth_metrics.get("depth_metric_std")
        elif self.config.strategy in ROBOT_GEO_SCORE_STRATEGIES:
            scores, valid_mask, robot_metrics, fallback_reason = self._compute_robot_geo_near_scores(num_tokens)
            metrics.timing.token_mapping_ms = robot_metrics.get("token_mapping_ms")
            metrics.timing.depth_sampling_ms = robot_metrics.get("depth_sampling_ms")
            metrics.timing.depth_sample_ms = metrics.timing.depth_sampling_ms
            metrics.timing.score_compute_ms = robot_metrics.get("score_compute_ms")
            metrics.timing.depth_edge_score_ms = robot_metrics.get("depth_edge_score_ms")
            metrics.timing.token_xyz_projection_ms = robot_metrics.get("token_xyz_projection_ms")
            metrics.timing.score_fusion_ms = robot_metrics.get("score_fusion_ms")
            metrics.timing.edge_score_ms = robot_metrics.get("edge_score_ms")
            metrics.timing.robot_mapping_ms = robot_metrics.get("robot_mapping_ms")
            metrics.timing.near_score_ms = robot_metrics.get("near_score_ms")
            metrics.timing.corridor_score_ms = robot_metrics.get("corridor_score_ms")
            metrics.timing.contact_score_ms = robot_metrics.get("contact_score_ms")
            metrics.geometry_available = robot_metrics.get("geometry_available")
            metrics.robot_state_available = robot_metrics.get("robot_state_available")
            metrics.camera_available = robot_metrics.get("camera_available")
            metrics.valid_token_ratio = robot_metrics.get("valid_token_ratio")
            metrics.geometry_score_mean = robot_metrics.get("geometry_score_mean")
            metrics.geometry_score_max = robot_metrics.get("geometry_score_max")
            metrics.geometry_score_std = robot_metrics.get("geometry_score_std")
            metrics.depth_edge_score_mean = robot_metrics.get("depth_edge_score_mean")
            metrics.distance_score_mean = robot_metrics.get("mean_near_score")
            metrics.direction_score_mean = robot_metrics.get("motion_cone_score_mean") or robot_metrics.get("corridor_strength_mean")
            metrics.d_min = robot_metrics.get("d_min")
            # P0: depth conversion metadata
            metrics.depth_source_key = robot_metrics.get("depth_source_key")
            metrics.depth_conversion = robot_metrics.get("depth_conversion")
            metrics.depth_is_metric = robot_metrics.get("depth_is_metric")
            metrics.depth_unit = robot_metrics.get("depth_unit")
            metrics.depth_sim_available = robot_metrics.get("depth_sim_available")
            metrics.depth_raw_min = robot_metrics.get("depth_raw_min")
            metrics.depth_raw_max = robot_metrics.get("depth_raw_max")
            metrics.depth_raw_mean = robot_metrics.get("depth_raw_mean")
            metrics.depth_raw_std = robot_metrics.get("depth_raw_std")
            metrics.depth_metric_min = robot_metrics.get("depth_metric_min")
            metrics.depth_metric_max = robot_metrics.get("depth_metric_max")
            metrics.depth_metric_mean = robot_metrics.get("depth_metric_mean")
            metrics.depth_metric_std = robot_metrics.get("depth_metric_std")
            # P0-4: transform convention (T_robot_cam_forward)
            metrics.transform_convention = robot_metrics.get("transform_convention")
            metrics.transform_inverse_used = robot_metrics.get("transform_inverse_used")
            metrics.transform_source = robot_metrics.get("transform_source")
            metrics.transform_convention_verified = robot_metrics.get("transform_convention_verified")
            metrics.transform_convention_evidence = robot_metrics.get("transform_convention_evidence")
            metrics.mean_near_score = robot_metrics.get("mean_near_score")
            metrics.max_near_score = robot_metrics.get("max_near_score")
            metrics.motion_norm = robot_metrics.get("motion_norm")
            metrics.corridor_strength_mean = robot_metrics.get("corridor_strength_mean")
            metrics.corridor_active = robot_metrics.get("corridor_active")
            metrics.d_corridor_min = robot_metrics.get("d_corridor_min")
            metrics.edge_score_mean = robot_metrics.get("edge_score_mean")
            metrics.motion_cone_score_mean = robot_metrics.get("motion_cone_score_mean")
            metrics.motion_cone_score_max = robot_metrics.get("motion_cone_score_max")
            metrics.workspace_score_mean = robot_metrics.get("workspace_score_mean")
            metrics.workspace_score_max = robot_metrics.get("workspace_score_max")
            metrics.contact_risk_score_mean = robot_metrics.get("contact_risk_score_mean")
            metrics.contact_risk_score_max = robot_metrics.get("contact_risk_score_max")
            metrics.geo_risk_level = robot_metrics.get("geo_risk_level")
            metrics.geo_risk_score = robot_metrics.get("geo_risk_score")
            metrics.dynamic_keep_reason = robot_metrics.get("dynamic_keep_reason")
            metrics.num_high_contact_tokens = robot_metrics.get("num_high_contact_tokens")
            metrics.num_valid_3d_tokens = robot_metrics.get("num_valid_3d_tokens")
            metrics.interaction_lock = robot_metrics.get("interaction_lock")
            metrics.temporal_stability = robot_metrics.get("temporal_stability")
            metrics.history_length = robot_metrics.get("history_length")
            metrics.score_ema_enabled = robot_metrics.get("score_ema_enabled")
            metrics.near_contact_score_mean = robot_metrics.get("near_contact_score_mean")
            metrics.corridor_contact_score_mean = robot_metrics.get("corridor_contact_score_mean")
            metrics.geo_contact_score_mean = robot_metrics.get("geo_contact_score_mean")
            # New diagnostic fields: depth
            metrics.depth_min = robot_metrics.get("depth_min")
            metrics.depth_max = robot_metrics.get("depth_max")
            metrics.depth_mean = robot_metrics.get("depth_mean")
            metrics.depth_valid_ratio = robot_metrics.get("valid_token_ratio")
            # New diagnostic fields: 3D token geometry
            metrics.points_robot_min_xyz = robot_metrics.get("points_robot_min_xyz")
            metrics.points_robot_max_xyz = robot_metrics.get("points_robot_max_xyz")
            metrics.points_cam_min_xyz = robot_metrics.get("points_cam_min_xyz")
            metrics.points_cam_max_xyz = robot_metrics.get("points_cam_max_xyz")
            metrics.extrinsics_available = robot_metrics.get("extrinsics_available")
            metrics.intrinsics_available = robot_metrics.get("intrinsics_available")
            metrics.camera_frame_name = robot_metrics.get("camera_frame_name")
            metrics.geometry_frame_name = robot_metrics.get("geometry_frame_name")
            # New diagnostic fields: robot state / gripper
            metrics.ee_position = robot_metrics.get("ee_position")
            metrics.robot_state_valid = robot_metrics.get("robot_state_valid")
            metrics.motion_direction_valid = robot_metrics.get("motion_direction_valid")
            metrics.motion_direction_xyz = robot_metrics.get("motion_direction_xyz")
            metrics.distance_to_gripper_min = robot_metrics.get("distance_to_gripper_min")
            metrics.distance_to_gripper_mean = robot_metrics.get("distance_to_gripper_mean")
            metrics.distance_to_gripper_max = robot_metrics.get("distance_to_gripper_max")
            # P1-1: Score component is_none/nonzero_ratio (from hybrid_v1 branch)
            metrics.contact_risk_scores_is_none = robot_metrics.get("contact_risk_scores_is_none")
            metrics.motion_cone_scores_is_none = robot_metrics.get("motion_cone_scores_is_none")
            metrics.workspace_scores_is_none = robot_metrics.get("workspace_scores_is_none")
            metrics.contact_nonzero_ratio = robot_metrics.get("contact_nonzero_ratio")
            metrics.motion_cone_nonzero_ratio = robot_metrics.get("motion_cone_nonzero_ratio")
            metrics.workspace_nonzero_ratio = robot_metrics.get("workspace_nonzero_ratio")
            # P1-1: temporal_v1 semantics fix
            metrics.temporal_enabled = robot_metrics.get("temporal_enabled")
            metrics.interaction_lock_triggered = robot_metrics.get("interaction_lock_triggered")
            metrics.interaction_lock_reason = robot_metrics.get("interaction_lock_reason")

        # P8: Assign active_aux_metrics based on which branch was taken above.
        # This replaces all broken "robot_metrics if 'robot_metrics' in locals()" checks
        # where robot_metrics={} is always in locals() even for depth_edge_fast.
        if self.config.strategy in ("depth_edge_fast", "depth_edge_fast_diverse"):
            active_aux_metrics = depth_metrics
        else:
            active_aux_metrics = robot_metrics

        strategy = self.config.strategy
        reserve_tokens = self.config.diverse_reserve_tokens
        if self.config.strategy == "robot_geo_dynamic" and fallback_reason is None:
            try:
                score_dict = {
                    "scores": scores,
                    "edge_scores": robot_metrics.get("edge_scores"),
                    "corridor_scores": robot_metrics.get("corridor_scores"),
                    "motion_norm": robot_metrics.get("motion_norm"),
                    "valid_depth_ratio": robot_metrics.get("valid_token_ratio"),
                }
                _, dynamic_stats = compute_dynamic_keep_ratio(
                    score_dict=score_dict,
                    p_robot=robot_metrics.get("token_points_robot"),
                    gripper_pos=robot_metrics.get("gripper_pos"),
                    valid_mask=valid_mask,
                    cfg=self.config,
                )
                keep_count = int(dynamic_stats.get("dynamic_keep_k", keep_count))
                reserve_tokens = int(dynamic_stats.get("dynamic_reserve_k", reserve_tokens))
                metrics.dynamic_enabled = True
                metrics.keep_ratio_source = "dynamic_decision"
                metrics.dynamic_phase = dynamic_stats.get("dynamic_phase")
                metrics.dynamic_keep_ratio = dynamic_stats.get("dynamic_keep_ratio")
                metrics.dynamic_keep_k = keep_count
                metrics.d_min = dynamic_stats.get("d_min", metrics.d_min)
                metrics.d_topk_mean = dynamic_stats.get("d_topk_mean")
                metrics.corridor_strength = dynamic_stats.get("corridor_strength")
                metrics.edge_concentration = dynamic_stats.get("edge_concentration")
                metrics.valid_depth_ratio = dynamic_stats.get("valid_depth_ratio")
                metrics.motion_norm = dynamic_stats.get("motion_norm", metrics.motion_norm)
                metrics.phase_far_count = 1 if metrics.dynamic_phase == "far" else 0
                metrics.phase_mid_count = 1 if metrics.dynamic_phase == "mid" else 0
                metrics.phase_near_count = 1 if metrics.dynamic_phase == "near" else 0
                metrics.phase_fallback_safe_count = 1 if metrics.dynamic_phase == "fallback_safe" else 0
            except Exception as exc:
                metrics.fallback_used = True
                metrics.keep_ratio_source = "fallback"
                metrics.fallback_reason = f"dynamic_scheduler_error:{type(exc).__name__}"
                strategy = "depth_edge_fast_diverse"
                keep_count = int(round(num_tokens * self.config.keep_ratio_mid))
                reserve_tokens = self.config.diverse_reserve_tokens
                fallback_reason = None
        if self.config.strategy == "robot_geo_dynamic_v0" and fallback_reason is None:
            dynamic_decision = robot_metrics.get("dynamic_decision") or {}
            dynamic_ratio = dynamic_decision.get("keep_ratio")
            if dynamic_ratio is not None:
                keep_count = int(round(num_tokens * float(dynamic_ratio)))
                keep_count = max(1, min(num_tokens, keep_count))
                metrics.dynamic_enabled = True
                metrics.keep_ratio_source = "dynamic_decision"
                metrics.dynamic_phase = dynamic_decision.get("risk_level")
                metrics.dynamic_keep_ratio = keep_count / num_tokens if num_tokens else None
                metrics.dynamic_keep_k = keep_count
                metrics.geo_risk_level = dynamic_decision.get("risk_level")
                metrics.geo_risk_score = dynamic_decision.get("risk_score")
                metrics.dynamic_keep_reason = dynamic_decision.get("reason")
                component_summary = dynamic_decision.get("component_summary") or {}
                metrics.num_high_contact_tokens = component_summary.get("num_high_contact_tokens")
                metrics.num_valid_3d_tokens = component_summary.get("num_valid_3d_tokens")
                metrics.phase_far_count = 1 if metrics.dynamic_phase == "low" else 0
                metrics.phase_mid_count = 1 if metrics.dynamic_phase == "medium" else 0
                metrics.phase_near_count = 1 if metrics.dynamic_phase == "high" else 0
                metrics.phase_fallback_safe_count = 0
            else:
                mid_ratio = float(self.config.dynamic_keep_ratio_config.get("mid_keep_ratio", self.config.keep_ratio_mid))
                keep_count = int(round(num_tokens * mid_ratio))
                metrics.dynamic_enabled = True
                metrics.keep_ratio_source = "dynamic_mid_keep_ratio"
                metrics.dynamic_phase = "medium"
                metrics.dynamic_keep_ratio = keep_count / num_tokens if num_tokens else None
                metrics.dynamic_keep_k = keep_count
                metrics.dynamic_keep_reason = "missing_dynamic_decision_fallback"
        if self.config.strategy == "robot_geo_temporal_v0" and fallback_reason is None:
            dynamic_decision = robot_metrics.get("dynamic_decision") or {}
            dynamic_ratio = dynamic_decision.get("keep_ratio")
            if dynamic_ratio is None:
                dynamic_ratio = float(self.config.dynamic_keep_ratio_config.get("mid_keep_ratio", self.config.keep_ratio_mid))
                metrics.keep_ratio_source = "dynamic_mid_keep_ratio"
            else:
                metrics.keep_ratio_source = "dynamic_decision"
            temporal = self._temporal_history.detect_interaction_lock(
                contact_risk_score=robot_metrics.get("rule_v0_contact_risk_scores"),
                final_scores=scores,
                motion_direction=robot_metrics.get("motion_direction"),
                valid_3d_ratio=robot_metrics.get("valid_token_ratio"),
                dynamic_keep_ratio=dynamic_ratio,
                config=self.config,
                gripper_pos=robot_metrics.get("gripper_pos"),
                token_points_robot=robot_metrics.get("token_points_robot"),
            )
            lock_ratio = float(self.config.dynamic_keep_ratio_config.get("max_keep_ratio", self.config.keep_ratio_near))
            chosen_ratio = max(float(dynamic_ratio), lock_ratio) if temporal.get("interaction_lock") else float(dynamic_ratio)
            keep_count = max(1, min(num_tokens, int(round(num_tokens * chosen_ratio))))
            metrics.dynamic_enabled = True
            metrics.dynamic_phase = "interaction_lock" if temporal.get("interaction_lock") else dynamic_decision.get("risk_level", "medium")
            metrics.dynamic_keep_ratio = keep_count / num_tokens if num_tokens else None
            metrics.dynamic_keep_k = keep_count
            metrics.geo_risk_level = metrics.dynamic_phase
            metrics.geo_risk_score = dynamic_decision.get("risk_score")
            metrics.dynamic_keep_reason = "temporal_interaction_lock" if temporal.get("interaction_lock") else dynamic_decision.get("reason", "single_frame_rule")
            metrics.interaction_lock = bool(temporal.get("interaction_lock"))
            metrics.interaction_lock_reason = str(temporal.get("lock_reason", "none"))
            metrics.temporal_stability = temporal.get("temporal_stability")
            metrics.history_length = temporal.get("history_length")
            metrics.score_ema_enabled = bool(temporal.get("score_ema_enabled"))
            component_summary = dynamic_decision.get("component_summary") or {}
            metrics.num_high_contact_tokens = component_summary.get("num_high_contact_tokens")
            metrics.num_valid_3d_tokens = component_summary.get("num_valid_3d_tokens")
            metrics.phase_far_count = 1 if metrics.dynamic_phase == "low" else 0
            metrics.phase_mid_count = 1 if metrics.dynamic_phase == "medium" else 0
            metrics.phase_near_count = 1 if metrics.dynamic_phase in ("high", "interaction_lock") else 0
            metrics.phase_fallback_safe_count = 0
        if self.config.strategy in EARLY_GEOMETRY_FALLBACK_STRATEGIES and fallback_reason is None:
            strategy = "depth_edge_fast_diverse"
        if self.config.strategy == "robot_geo_contact_budget" and fallback_reason is not None:
            metrics.fallback_used = True
            metrics.keep_ratio_source = "fallback"
            metrics.fallback_reason = fallback_reason
            strategy = "depth_edge_fast_diverse"
            fallback_reason = None
        if self.config.strategy in (ROBOT_STATE_REQUIRED_LEGACY_STRATEGIES - ACGTP_STRATEGIES) and fallback_reason in (
            "missing_robot_state",
            "missing_camera",
            "missing_camera_intrinsics",
            "missing_camera_extrinsics",
        ):
            metrics.fallback_used = True
            metrics.keep_ratio_source = "fallback"
            metrics.fallback_reason = fallback_reason
            strategy = "depth_edge_fast_diverse"
            if self.config.strategy in ("robot_geo_dynamic", "robot_geo_dynamic_v0", "robot_geo_temporal_v0"):
                mid_ratio = float(self.config.dynamic_keep_ratio_config.get("mid_keep_ratio", self.config.keep_ratio_mid))
                keep_count = int(round(num_tokens * mid_ratio))
                metrics.dynamic_enabled = True
                metrics.dynamic_phase = "medium"
                metrics.dynamic_keep_ratio = keep_count / num_tokens if num_tokens else None
                metrics.dynamic_keep_k = keep_count
                metrics.geo_risk_level = "medium"
                metrics.dynamic_keep_reason = fallback_reason
            fallback_reason = None
        if fallback_reason is not None:
            metrics.fallback_used = True
            metrics.keep_ratio_source = "fallback"
            metrics.fallback_reason = fallback_reason
            if self.config.strategy == "robot_geo_dynamic":
                strategy = "uniform_grid"
                keep_count = int(round(num_tokens * self.config.keep_ratio_near))
                reserve_tokens = 0
                metrics.dynamic_enabled = True
                metrics.dynamic_phase = "fallback_safe"
                metrics.dynamic_keep_k = keep_count
                metrics.dynamic_keep_ratio = keep_count / num_tokens if num_tokens else None
                metrics.valid_depth_ratio = metrics.valid_token_ratio
                metrics.phase_far_count = 0
                metrics.phase_mid_count = 0
                metrics.phase_near_count = 0
                metrics.phase_fallback_safe_count = 1
            elif self.config.strategy == "robot_geo_dynamic_v0":
                strategy = "depth_edge_fast_diverse" if scores is not None else "uniform_grid"
                mid_ratio = float(self.config.dynamic_keep_ratio_config.get("mid_keep_ratio", self.config.keep_ratio_mid))
                keep_count = int(round(num_tokens * mid_ratio))
                metrics.dynamic_enabled = True
                metrics.dynamic_phase = "medium"
                metrics.dynamic_keep_ratio = keep_count / num_tokens if num_tokens else None
                metrics.dynamic_keep_k = keep_count
                metrics.geo_risk_level = "medium"
                dynamic_decision = robot_metrics.get("dynamic_decision") or {}
                metrics.dynamic_keep_reason = fallback_reason
                metrics.phase_far_count = 0
                metrics.phase_mid_count = 1
                metrics.phase_near_count = 0
                metrics.phase_fallback_safe_count = 0
            elif self.config.strategy == "robot_geo_temporal_v0":
                strategy = "depth_edge_fast_diverse" if scores is not None else "uniform_grid"
                mid_ratio = float(self.config.dynamic_keep_ratio_config.get("mid_keep_ratio", self.config.keep_ratio_mid))
                keep_count = int(round(num_tokens * mid_ratio))
                metrics.dynamic_enabled = True
                metrics.dynamic_phase = "medium"
                metrics.dynamic_keep_ratio = keep_count / num_tokens if num_tokens else None
                metrics.dynamic_keep_k = keep_count
                metrics.geo_risk_level = "medium"
                metrics.dynamic_keep_reason = fallback_reason
                metrics.interaction_lock = False
                metrics.interaction_lock_reason = "none"
                metrics.history_length = self._temporal_history.history_length
                metrics.phase_far_count = 0
                metrics.phase_mid_count = 1
                metrics.phase_near_count = 0
                metrics.phase_fallback_safe_count = 0
            elif self.config.strategy in ("robot_geo_hybrid_temporal_v1", "robot_geo_hybrid_temporal_edge_reserve_v1"):
                strategy = "depth_edge_fast_diverse" if scores is not None else "uniform_grid"
                mid_ratio = float(self.config.dynamic_keep_ratio_config.get("mid_keep_ratio", self.config.keep_ratio_mid))
                keep_count = int(round(num_tokens * mid_ratio))
                metrics.dynamic_enabled = True
                metrics.dynamic_phase = "medium"
                metrics.dynamic_keep_ratio = keep_count / num_tokens if num_tokens else None
                metrics.dynamic_keep_k = keep_count
                metrics.geo_risk_level = "medium"
                metrics.dynamic_keep_reason = fallback_reason
                metrics.interaction_lock = False
                metrics.interaction_lock_reason = "none"
                metrics.history_length = self._temporal_history.history_length
                metrics.score_ema_enabled = True
                metrics.score_ema_available = False
                metrics.ema_used_for_selection = False
                metrics.lock_condition_failed_reason = fallback_reason
                metrics.adaptive_threshold_mean = None
                metrics.adaptive_threshold_max = None
                metrics.interaction_lock_ratio = 0.0
                metrics.phase_far_count = 0
                metrics.phase_mid_count = 1
                metrics.phase_near_count = 0
                metrics.phase_fallback_safe_count = 0
            else:
                strategy = "none" if self.config.fallback_strategy == "no_pruning" else "uniform_grid"
                keep_count = num_tokens if strategy == "none" else keep_count

        select_start = time.perf_counter()
        if self.config.strategy in ACGTP_STRATEGIES and fallback_reason is not None:
            # ACGTP is a geometry expert. If required geometry inputs are missing,
            # do not run the branch selector on zero scores/all-token fill masks:
            # mark an explicit input fallback so diagnostics cannot be mistaken
            # for successful action-constrained token protection.
            _acgtp_input_fallback_reason = str(fallback_reason)
            _requested_keep_count = int(round(num_tokens * float(self.config.keep_ratio)))
            if scores is not None and valid_mask is not None and _acgtp_input_fallback_reason not in ("missing_depth", "missing_geometry"):
                _fallback_strategy = "depth_edge_fast_diverse"
                _fallback_keep_count = _requested_keep_count
            else:
                _fallback_strategy = "none" if self.config.fallback_strategy == "no_pruning" else "uniform_grid"
                _fallback_keep_count = num_tokens if _fallback_strategy == "none" else _requested_keep_count
            keep_indices_np, selection_meta = select_keep_indices(
                strategy=_fallback_strategy,
                num_tokens=num_tokens,
                keep_count=_fallback_keep_count,
                scores=scores,
                valid_mask=valid_mask,
                seed=self.config.seed,
                grid_size=self.config.token_grid_shape[0],
                cell_grid=self.config.cell_grid,
                reserve_tokens=reserve_tokens,
            )
            _fallback_kept = int(len(keep_indices_np))
            _is_v2 = self.config.strategy == "robot_geo_acgtp_v2"
            metrics.fallback_used = True
            metrics.keep_ratio_source = "fallback"
            metrics.fallback_reason = _acgtp_input_fallback_reason
            selection_meta.update({
                "strategy": _fallback_strategy,
                "requested_acgtp_strategy": self.config.strategy,
                "selector_name": "acgtp_input_fallback",
                "selector_function_name": "select_keep_indices",
                "selection_strategy_name": _fallback_strategy,
                "selection_stage_name": "acgtp_input_fallback",
                "fallback_used": True,
                "fallback_reason": _acgtp_input_fallback_reason,
                "input_fallback_used": True,
                "input_fallback_reason": _acgtp_input_fallback_reason,
                "acgtp_v1": not _is_v2,
                "acgtp_v2": _is_v2,
                "acgtp_fallback_used": True,
                "acgtp_fallback_reason": _acgtp_input_fallback_reason,
                "acgtp_hard_protect_count": 0,
                "acgtp_hard_protect_valid": False,
                "selected_by_scene_layout_count": 0,
                "selected_by_depth_structure_count": 0,
                "selected_by_contact_ring_count": 0,
                "selected_by_motion_corridor_count": 0,
                "selected_by_constrained_fill_count": 0,
                "selected_by_acgtp_fallback_count": _fallback_kept,
                "selected_by_semantic_count": 0,
                "selected_by_attention_count": 0,
                "selected_by_fallback": _fallback_kept,
                "selected_by_fill": 0,
                "selected_unattributed": 0,
                "acgtp_branch_accounting_valid": True,
                "acgtp_branch_sum": _fallback_kept,
                "acgtp_branch_sum_error": 0,
                "branch_accounting_valid": True,
                "branch_sum_equals_kept": True,
                "final_kept": _fallback_kept,
                "expected_kept": _fallback_keep_count,
                "K_total": _fallback_keep_count,
                "acgtp_final_kept": _fallback_kept,
                "acgtp_expected_kept": _fallback_keep_count,
                "acgtp_actual_keep_ratio": (_fallback_kept / num_tokens) if num_tokens else 0.0,
            })
            if _is_v2:
                selection_meta.update({
                    "strict_fallback_dispatch_used": False,
                    "delegated_selector_name": None,
                    "fallback_dispatch_to_v1": False,
                    "acgtp_v2_semantic_enabled": bool(self.config.acgtp_v2_semantic_enabled),
                    "acgtp_v2_semantic_backend": str(self.config.acgtp_v2_semantic_backend),
                    "acgtp_v2_semantic_confidence": 0.0,
                    "acgtp_v2_semantic_unavailable": True,
                    "acgtp_v2_semantic_fallback_reason": _acgtp_input_fallback_reason,
                    "acgtp_v2_release_quota": True,
                    "semantic_backend": str(self.config.acgtp_v2_semantic_backend),
                    "semantic_available": False,
                    "semantic_unavailable": True,
                    "semantic_confidence": 0.0,
                    "semantic_quota_released": True,
                    "attention_backend": "none",
                    "attention_source": "none",
                    "attention_available": False,
                    "attention_confidence": 0.0,
                    "attention_quota_released": True,
                    "attention_only_token_count": 0,
                    "attention_selected_by_final_count": 0,
                    "attention_top_count": 0,
                    "high_attention_low_geometry_count": 0,
                    "high_geometry_low_attention_count": 0,
                })
            fallback_reason = None
        if self.config.strategy in ("depth_edge_fast", "depth_edge_fast_diverse") and fallback_reason is None:
            keep_indices_np, selection_meta = select_keep_indices(
                strategy=self.config.strategy,
                num_tokens=num_tokens,
                keep_count=keep_count,
                scores=scores,
                valid_mask=valid_mask,
                seed=self.config.seed,
                grid_size=self.config.token_grid_shape[0],
                cell_grid=self.config.cell_grid,
                reserve_tokens=reserve_tokens,
            )
            selection_meta["fallback_used"] = False
            selection_meta["fallback_reason"] = None
            selection_meta["requested_keep_ratio"] = float(self.config.keep_ratio)
            selection_meta["keep_ratio_source"] = "cli_keep_ratio"
            selection_meta["effective_keep_count"] = int(len(keep_indices_np))
            selection_meta["original_token_count"] = int(num_tokens)
            metrics.keep_ratio_source = "cli_keep_ratio"
            self._record_selection_path_diagnostics(
                metrics=metrics,
                selector_success=True,
                exc=None,
                requested_strategy=self.config.strategy,
                effective_strategy=self.config.strategy,
                selector_name="select_keep_indices",
                fallback_selector=None,
                keep_indices_np=keep_indices_np,
                num_tokens=num_tokens,
            )
        elif self.config.strategy == "robot_geo_contact_budget" and strategy == "robot_geo_contact_budget" and fallback_reason is None:
            k_edge, k_geo, k_diverse = self._contact_budget_counts(keep_count)
            keep_indices_np, selection_meta = select_tokens_contact_budget(
                edge_score=robot_metrics.get("edge_scores"),
                geo_contact_score=robot_metrics.get("geo_contact_scores", scores),
                valid_mask=valid_mask,
                keep_k=keep_count,
                k_edge=k_edge,
                k_geo=k_geo,
                k_diverse=k_diverse,
                grid_h=self.config.token_grid_shape[0],
                grid_w=self.config.token_grid_shape[1],
                cells_h=self.config.cell_grid,
                cells_w=self.config.cell_grid,
                return_indices=bool(self.cfg.get("save_pruning_vis", False) or self.cfg.get("save_pruning_debug", False)),
                detailed_timing=self.config.detailed_pruning_timing,
            )
            self._record_selection_path_diagnostics(
                metrics=metrics,
                selector_success=True,
                exc=None,
                requested_strategy=self.config.strategy,
                effective_strategy=self.config.strategy,
                selector_name="select_tokens_contact_budget",
                fallback_selector=None,
                keep_indices_np=keep_indices_np,
                num_tokens=num_tokens,
            )
        elif self.config.strategy == "robot_geo_hybrid_v0":
            if fallback_reason is not None:
                metrics.fallback_used = True
                metrics.keep_ratio_source = "fallback"
                metrics.fallback_reason = fallback_reason
                strategy = "depth_edge_fast_diverse"
                keep_count = int(round(num_tokens * 0.75))
            else:
                try:
                    keep_indices_np, selection_meta = select_hybrid_quota_v2(
                        depth_edge_scores=robot_metrics.get("edge_scores"),
                        contact_risk_scores=robot_metrics.get("rule_v0_contact_risk_scores"),
                        distance_scores=robot_metrics.get("near_scores"),
                        motion_cone_scores=robot_metrics.get("rule_v0_motion_cone_scores"),
                        valid_mask=valid_mask,
                        keep_k=keep_count,
                        grid_h=self.config.token_grid_shape[0],
                        grid_w=self.config.token_grid_shape[1],
                        cell_grid=self.config.cell_grid,
                        seed=self.config.seed,
                    )
                    metrics.dynamic_enabled = True
                    metrics.dynamic_phase = "hybrid_v2"
                    metrics.dynamic_keep_ratio = keep_count / num_tokens if num_tokens else None
                    metrics.dynamic_keep_k = keep_count
                    metrics.dynamic_keep_reason = "hybrid_quota_v2"
                    metrics.phase_far_count = 0
                    metrics.phase_mid_count = 0
                    metrics.phase_near_count = 1
                    metrics.phase_fallback_safe_count = 0
                    self._record_selection_path_diagnostics(
                        metrics=metrics,
                        selector_success=True,
                        exc=None,
                        requested_strategy=self.config.strategy,
                        effective_strategy=self.config.strategy,
                        selector_name="select_hybrid_quota_v2",
                        fallback_selector=None,
                        keep_indices_np=keep_indices_np,
                        num_tokens=num_tokens,
                    )
                    for key in (
                        "selected_by_depth_edge_count", "selected_by_contact_count",
                        "selected_by_distance_contact_count", "selected_by_motion_count",
                        "selected_by_uniform_count", "selected_by_fill_count",
                        "K_depth_edge_quota", "K_contact_quota", "K_distance_contact_quota",
                        "K_motion_quota", "K_uniform_quota", "K_fill_quota",
                        "K_depth_edge_actual", "K_contact_actual", "K_distance_contact_actual",
                        "K_motion_actual", "K_uniform_actual", "K_fill_actual",
                        "motion_gate_tokens_total", "motion_gate_tokens_selected",
                        "motion_gate_effective", "overlap_depth_contact",
                        "overlap_depth_dist_contact", "overlap_depth_motion",
                    ):
                        if key in selection_meta:
                            setattr(metrics, key, selection_meta.get(key))
                except Exception as exc:
                    metrics.fallback_used = True
                    metrics.keep_ratio_source = "fallback"
                    metrics.fallback_reason = f"hybrid_v0_error:{type(exc).__name__}"
                    strategy = "depth_edge_fast_diverse"
                    keep_count = int(round(num_tokens * 0.75))
                    self._record_selection_path_diagnostics(
                        metrics=metrics,
                        selector_success=False,
                        exc=exc,
                        requested_strategy=self.config.strategy,
                        effective_strategy="depth_edge_fast_diverse",
                        selector_name="select_hybrid_quota_v2",
                        fallback_selector="depth_edge_fast_diverse",
                        keep_indices_np=None,
                        num_tokens=num_tokens,
                    )

        elif self.config.strategy == "robot_geo_hybrid_dynamic_v0":
            decision = robot_metrics.get("dynamic_decision") or {}
            dynamic_ratio = decision.get("keep_ratio")
            if dynamic_ratio is not None:
                keep_count = max(1, min(num_tokens, int(round(num_tokens * float(dynamic_ratio)))))
            if fallback_reason is not None:
                metrics.fallback_used = True
                metrics.keep_ratio_source = "fallback"
                metrics.fallback_reason = fallback_reason
                strategy = "depth_edge_fast_diverse"
                keep_count = int(round(num_tokens * 0.75))
                metrics.dynamic_enabled = True
                metrics.dynamic_phase = "medium"
                metrics.dynamic_keep_ratio = keep_count / num_tokens if num_tokens else None
                metrics.dynamic_keep_k = keep_count
                metrics.dynamic_keep_reason = fallback_reason
            else:
                try:
                    keep_indices_np, selection_meta = select_hybrid_quota_v2(
                        depth_edge_scores=robot_metrics.get("edge_scores"),
                        contact_risk_scores=robot_metrics.get("rule_v0_contact_risk_scores"),
                        distance_scores=robot_metrics.get("near_scores"),
                        motion_cone_scores=robot_metrics.get("rule_v0_motion_cone_scores"),
                        valid_mask=valid_mask,
                        keep_k=keep_count,
                        grid_h=self.config.token_grid_shape[0],
                        grid_w=self.config.token_grid_shape[1],
                        cell_grid=self.config.cell_grid,
                        seed=self.config.seed,
                    )
                    metrics.dynamic_enabled = True
                    metrics.dynamic_phase = f"hybrid_dynamic_{decision.get('risk_level', 'medium')}"
                    metrics.dynamic_keep_ratio = keep_count / num_tokens if num_tokens else None
                    metrics.dynamic_keep_k = keep_count
                    metrics.geo_risk_level = decision.get("risk_level")
                    metrics.geo_risk_score = decision.get("risk_score")
                    metrics.dynamic_keep_reason = f"hybrid_dynamic_{decision.get('reason', 'dynamic')}"
                    component_summary = decision.get("component_summary", {}) or {}
                    metrics.num_high_contact_tokens = component_summary.get("num_high_contact_tokens")
                    metrics.num_valid_3d_tokens = component_summary.get("num_valid_3d_tokens")
                    self._record_selection_path_diagnostics(
                        metrics=metrics,
                        selector_success=True,
                        exc=None,
                        requested_strategy=self.config.strategy,
                        effective_strategy=self.config.strategy,
                        selector_name="select_hybrid_quota_v2",
                        fallback_selector=None,
                        keep_indices_np=keep_indices_np,
                        num_tokens=num_tokens,
                    )
                    # All hybrid_dynamic fields propagated by _apply_selection_metrics() below
                except Exception as exc:
                    metrics.fallback_used = True
                    metrics.keep_ratio_source = "fallback"
                    metrics.fallback_reason = f"hybrid_dynamic_error:{type(exc).__name__}"
                    strategy = "depth_edge_fast_diverse"
                    keep_count = int(round(num_tokens * 0.75))
                    metrics.dynamic_enabled = True
                    metrics.dynamic_phase = "medium"
                    metrics.dynamic_keep_ratio = keep_count / num_tokens if num_tokens else None
                    metrics.dynamic_keep_k = keep_count
                    metrics.dynamic_keep_reason = f"hybrid_dynamic_fallback:{type(exc).__name__}"
                    self._record_selection_path_diagnostics(
                        metrics=metrics,
                        selector_success=False,
                        exc=exc,
                        requested_strategy=self.config.strategy,
                        effective_strategy="depth_edge_fast_diverse",
                        selector_name="select_hybrid_quota_v2",
                        fallback_selector="depth_edge_fast_diverse",
                        keep_indices_np=None,
                        num_tokens=num_tokens,
                    )

        if self.config.strategy == "robot_geo_hybrid_v1":
            if fallback_reason is not None:
                metrics.fallback_used = True
                metrics.keep_ratio_source = "fallback"
                metrics.fallback_reason = fallback_reason
                strategy = "depth_edge_fast_diverse"
                keep_count = int(round(num_tokens * 0.75))
            else:
                try:
                    hybrid_weights = self.config.hybrid_v1_weights
                    keep_indices_np, selection_meta = select_hybrid_v1(
                        depth_edge_scores=robot_metrics.get("edge_scores"),
                        near_scores=robot_metrics.get("near_scores"),
                        contact_risk_scores=robot_metrics.get("rule_v0_contact_risk_scores"),
                        corridor_scores=robot_metrics.get("rule_v0_motion_cone_scores"),
                        valid_mask=valid_mask,
                        keep_k=keep_count,
                        grid_h=self.config.token_grid_shape[0],
                        grid_w=self.config.token_grid_shape[1],
                        cell_grid=self.config.cell_grid,
                        seed=self.config.seed,
                        w_edge=float(hybrid_weights.get("w_edge", 0.45)),
                        w_near=float(hybrid_weights.get("w_near", 0.20)),
                        w_contact=float(hybrid_weights.get("w_contact", 0.20)),
                        w_corr=float(hybrid_weights.get("w_corr", 0.10)),
                        w_diverse=float(hybrid_weights.get("w_diverse", 0.05)),
                    )
                    metrics.dynamic_enabled = True
                    metrics.keep_ratio_source = "cli_keep_ratio"
                    metrics.dynamic_phase = "hybrid_v1"
                    metrics.dynamic_keep_ratio = keep_count / num_tokens if num_tokens else None
                    metrics.dynamic_keep_k = keep_count
                    metrics.dynamic_keep_reason = "hybrid_v1_weighted_score"
                    metrics.phase_far_count = 0
                    metrics.phase_mid_count = 0
                    metrics.phase_near_count = 1
                    metrics.phase_fallback_safe_count = 0
                    self._record_selection_path_diagnostics(
                        metrics=metrics,
                        selector_success=True,
                        exc=None,
                        requested_strategy=self.config.strategy,
                        effective_strategy=self.config.strategy,
                        selector_name="select_hybrid_v1",
                        fallback_selector=None,
                        keep_indices_np=keep_indices_np,
                        num_tokens=num_tokens,
                    )
                    # All hybrid v1 fields (score stats, weights, grid coverage, entropy)
                    # are propagated by _apply_selection_metrics() below
                except Exception as exc:
                    metrics.fallback_used = True
                    metrics.keep_ratio_source = "fallback"
                    metrics.fallback_reason = f"hybrid_v1_error:{type(exc).__name__}"
                    strategy = "depth_edge_fast_diverse"
                    keep_count = int(round(num_tokens * 0.75))
                    # robot_geo_hybrid_v1 is a SELF_HANDLED selector strategy, so the
                    # generic select_keep_indices() rescue further down is skipped.
                    # Generate the depth_edge_fast_diverse fallback selection here
                    # (mirroring the hybrid_temporal_v1 branch) so keep_indices_np is
                    # never left None on the error path.
                    keep_indices_np, selection_meta = select_keep_indices(
                        strategy=strategy,
                        num_tokens=num_tokens,
                        keep_count=keep_count,
                        scores=scores,
                        valid_mask=valid_mask,
                        seed=self.config.seed,
                        grid_size=self.config.token_grid_shape[0],
                        cell_grid=self.config.cell_grid,
                        reserve_tokens=reserve_tokens,
                    )
                    selection_meta["fallback_used"] = True
                    selection_meta["fallback_reason"] = metrics.fallback_reason
                    selection_meta["actual_keep_ratio"] = len(keep_indices_np) / num_tokens if num_tokens else 0.0
                    self._record_selection_path_diagnostics(
                        metrics=metrics,
                        selector_success=False,
                        exc=exc,
                        requested_strategy=self.config.strategy,
                        effective_strategy="depth_edge_fast_diverse",
                        selector_name="select_hybrid_v1",
                        fallback_selector="depth_edge_fast_diverse",
                        keep_indices_np=keep_indices_np,
                        num_tokens=num_tokens,
                    )

        if self.config.strategy == "robot_geo_hybrid_temporal_v1":
            if fallback_reason is not None:
                metrics.fallback_used = True
                metrics.keep_ratio_source = "fallback"
                metrics.fallback_reason = fallback_reason
                strategy = "depth_edge_fast_diverse"
                keep_count = int(round(num_tokens * 0.75))
                # ── Actually generate a valid selection ────────────────────────
                keep_indices_np, selection_meta = select_keep_indices(
                    strategy=strategy,
                    num_tokens=num_tokens,
                    keep_count=keep_count,
                    scores=scores,
                    valid_mask=valid_mask,
                    seed=self.config.seed,
                    grid_size=self.config.token_grid_shape[0],
                    cell_grid=self.config.cell_grid,
                    reserve_tokens=reserve_tokens,
                )
                selection_meta["fallback_used"] = True
                selection_meta["fallback_reason"] = fallback_reason
                selection_meta["ema_used_for_selection"] = False
                selection_meta["interaction_lock"] = False
                selection_meta["lock_reason"] = fallback_reason
                selection_meta["actual_keep_ratio"] = len(keep_indices_np) / num_tokens if num_tokens else 0.0
                # Fallback: dynamic_keep_ratio derived from actual selection, NOT from config
                dynamic_keep_ratio_fb = len(keep_indices_np) / num_tokens if num_tokens else 0.75
                metrics.dynamic_enabled = True
                metrics.dynamic_phase = "medium"
                metrics.dynamic_keep_ratio = dynamic_keep_ratio_fb
                metrics.dynamic_keep_k = len(keep_indices_np)
                metrics.geo_risk_level = "medium"
                metrics.dynamic_keep_reason = fallback_reason
                metrics.interaction_lock = False
                metrics.interaction_lock_reason = "none"
                metrics.history_length = self._temporal_history.history_length
                metrics.score_ema_enabled = True
                metrics.score_ema_available = False
                metrics.ema_used_for_selection = False
                metrics.lock_condition_failed_reason = fallback_reason
                metrics.adaptive_threshold_mean = None
                metrics.adaptive_threshold_max = None
                metrics.interaction_lock_ratio = 0.0
                metrics.phase_far_count = 0
                metrics.phase_mid_count = 1
                metrics.phase_near_count = 0
                metrics.phase_fallback_safe_count = 0
                self._record_selection_path_diagnostics(
                    metrics=metrics,
                    selector_success=True,
                    exc=None,
                    requested_strategy=self.config.strategy,
                    effective_strategy="depth_edge_fast_diverse",
                    selector_name="select_keep_indices",
                    fallback_selector="depth_edge_fast_diverse",
                    keep_indices_np=keep_indices_np,
                    num_tokens=num_tokens,
                )
            else:
                contact_risk_arr = robot_metrics.get("rule_v0_contact_risk_scores")
                contact_vals = np.asarray(contact_risk_arr, dtype=np.float32).reshape(-1) if contact_risk_arr is not None else None
                adaptive_min = float(self.config.temporal_adaptive_threshold_min)
                adaptive_pct = float(self.config.temporal_adaptive_threshold_percentile)
                adaptive_threshold = adaptive_min
                if contact_vals is not None and np.any(np.isfinite(contact_vals)):
                    pct_val = float(np.percentile(contact_vals[np.isfinite(contact_vals)], adaptive_pct))
                    adaptive_threshold = max(adaptive_min, pct_val)
                dynamic_ratio = (robot_metrics.get("dynamic_decision") or {}).get("keep_ratio")
                if dynamic_ratio is None:
                    dynamic_ratio = float(self.config.keep_ratio)
                    metrics.keep_ratio_source = "cli_keep_ratio"
                else:
                    metrics.keep_ratio_source = "dynamic_decision"
                temporal = self._temporal_history.detect_interaction_lock(
                    contact_risk_score=contact_risk_arr,
                    final_scores=scores,
                    motion_direction=robot_metrics.get("motion_direction"),
                    valid_3d_ratio=robot_metrics.get("valid_token_ratio"),
                    dynamic_keep_ratio=dynamic_ratio,
                    config=self.config,
                    gripper_pos=robot_metrics.get("gripper_pos"),
                    token_points_robot=robot_metrics.get("token_points_robot"),
                    adaptive_threshold=adaptive_threshold,
                )
                interaction_locked = bool(temporal.get("interaction_lock"))
                ema_available = bool(temporal.get("score_ema_enabled"))
                if interaction_locked:
                    lock_ratio = float(self.config.temporal_interaction_lock_conservative_ratio)
                    chosen_ratio = max(float(dynamic_ratio), lock_ratio)
                    metrics.keep_ratio_source = "dynamic_decision"
                else:
                    chosen_ratio = float(dynamic_ratio)
                keep_count = max(1, min(num_tokens, int(round(num_tokens * chosen_ratio))))
                _select_ok = False
                try:
                    hybrid_weights = self.config.hybrid_v1_weights
                    keep_indices_np, selection_meta = select_hybrid_v1(
                        depth_edge_scores=robot_metrics.get("edge_scores"),
                        near_scores=robot_metrics.get("near_scores"),
                        contact_risk_scores=contact_risk_arr,
                        corridor_scores=robot_metrics.get("rule_v0_motion_cone_scores"),
                        valid_mask=valid_mask,
                        keep_k=keep_count,
                        grid_h=self.config.token_grid_shape[0],
                        grid_w=self.config.token_grid_shape[1],
                        cell_grid=self.config.cell_grid,
                        seed=self.config.seed,
                        w_edge=float(hybrid_weights.get("w_edge", 0.45)),
                        w_near=float(hybrid_weights.get("w_near", 0.20)),
                        w_contact=float(hybrid_weights.get("w_contact", 0.20)),
                        w_corr=float(hybrid_weights.get("w_corr", 0.10)),
                        w_diverse=float(hybrid_weights.get("w_diverse", 0.05)),
                    )
                    _select_ok = True
                    self._record_selection_path_diagnostics(
                        metrics=metrics,
                        selector_success=True,
                        exc=None,
                        requested_strategy=self.config.strategy,
                        effective_strategy=self.config.strategy,
                        selector_name="select_hybrid_v1",
                        fallback_selector=None,
                        keep_indices_np=keep_indices_np,
                        num_tokens=num_tokens,
                    )
                except Exception as exc:
                    print(f"[PRUNING] select_hybrid_v1 exception (step={self._hook_step_counter}): {type(exc).__name__}: {exc}")
                    metrics.fallback_used = True
                    metrics.keep_ratio_source = "fallback"
                    metrics.fallback_reason = f"hybrid_temporal_v1_select_error:{type(exc).__name__}"
                    strategy = "depth_edge_fast_diverse"
                    keep_count = int(round(num_tokens * 0.75))
                    keep_indices_np = None
                    selection_meta = {}
                    self._record_selection_path_diagnostics(
                        metrics=metrics,
                        selector_success=False,
                        exc=exc,
                        requested_strategy=self.config.strategy,
                        effective_strategy="depth_edge_fast_diverse",
                        selector_name="select_hybrid_v1",
                        fallback_selector="depth_edge_fast_diverse",
                        keep_indices_np=None,
                        num_tokens=num_tokens,
                    )
                    metrics.dynamic_enabled = True
                    metrics.dynamic_phase = "medium"
                    metrics.dynamic_keep_ratio = keep_count / num_tokens if num_tokens else None
                    metrics.dynamic_keep_k = keep_count
                    metrics.interaction_lock = False
                    metrics.interaction_lock_reason = "select_error"
                    metrics.score_ema_enabled = True
                    metrics.score_ema_available = False
                    metrics.ema_used_for_selection = False
                    metrics.lock_condition_failed_reason = "select_error"
                    metrics.adaptive_threshold_mean = None
                    metrics.adaptive_threshold_max = None
                    metrics.interaction_lock_ratio = 0.0
                if _select_ok:
                    metrics.dynamic_enabled = True
                    metrics.dynamic_phase = "interaction_lock" if interaction_locked else "hybrid_temporal_v1"
                    metrics.dynamic_keep_ratio = keep_count / num_tokens if num_tokens else None
                    metrics.dynamic_keep_k = keep_count
                    metrics.geo_risk_level = metrics.dynamic_phase
                    metrics.dynamic_keep_reason = "temporal_interaction_lock" if interaction_locked else "hybrid_temporal_v1"
                    metrics.interaction_lock = interaction_locked
                    metrics.interaction_lock_reason = str(temporal.get("lock_reason", "none"))
                    metrics.temporal_stability = temporal.get("temporal_stability")
                    metrics.history_length = temporal.get("history_length")
                    metrics.score_ema_enabled = ema_available
                    metrics.score_ema_available = ema_available
                    # EMA is computed for diagnostics only in temporal_v1; selection still
                    # uses the current-frame hybrid score, so do not log it as used.
                    metrics.ema_used_for_selection = False
                    metrics.lock_condition_failed_reason = str(temporal.get("reason", "none"))
                    metrics.topk_contact_lock = bool(temporal.get("topk_contact_lock"))
                    metrics.elevated_current_lock = bool(temporal.get("elevated_current_lock"))
                    metrics.gripper_lock = bool(temporal.get("gripper_lock"))
                    metrics.region_lock = bool(temporal.get("region_lock"))
                    metrics.adaptive_threshold_mean = adaptive_threshold
                    metrics.adaptive_threshold_max = adaptive_threshold
                    metrics.interaction_lock_ratio = 1.0 if interaction_locked else 0.0
                    component_summary = (robot_metrics.get("dynamic_decision") or {}).get("component_summary", {}) or {}
                    metrics.num_high_contact_tokens = component_summary.get("num_high_contact_tokens")
                    metrics.num_valid_3d_tokens = component_summary.get("num_valid_3d_tokens")
                    metrics.phase_far_count = 0
                    metrics.phase_mid_count = 0
                    metrics.phase_near_count = 1 if interaction_locked else 0
                    metrics.phase_fallback_safe_count = 0
                    # All hybrid v1 fields propagated by _apply_selection_metrics() below

        # P5/P6: robot_geo_hybrid_temporal_edge_reserve variants — targeted edge-reserve ablation
        if self.config.strategy in EDGE_RESERVE_LEGACY_STRATEGIES:
            # P6: compute edge_reserve_k from edge_reserve_ratio if set
            if self.config.edge_reserve_ratio > 0:
                _computed_edge_reserve_k = int(round(self.config.edge_reserve_ratio * keep_count))
            else:
                _computed_edge_reserve_k = int(self.config.edge_reserve_k)
            edge_reserve_k = max(0, min(_computed_edge_reserve_k, keep_count))
            if fallback_reason is not None:
                metrics.fallback_used = True
                metrics.keep_ratio_source = "fallback"
                metrics.fallback_reason = fallback_reason
                strategy = "depth_edge_fast_diverse"
                keep_count = int(round(num_tokens * 0.75))
                keep_indices_np, selection_meta = select_keep_indices(
                    strategy=strategy,
                    num_tokens=num_tokens,
                    keep_count=keep_count,
                    scores=scores,
                    valid_mask=valid_mask,
                    seed=self.config.seed,
                    grid_size=self.config.token_grid_shape[0],
                    cell_grid=self.config.cell_grid,
                    reserve_tokens=reserve_tokens,
                )
                selection_meta["fallback_used"] = True
                selection_meta["fallback_reason"] = fallback_reason
                selection_meta["edge_reserve_enabled"] = False
                selection_meta["edge_reserve_ratio"] = 0.0
                selection_meta["edge_reserved_target_count"] = 0
                selection_meta["edge_reserved_actual_count"] = 0
                selection_meta["edge_reserved_survival_ratio"] = None
                selection_meta["final_selected_count"] = int(len(keep_indices_np))
                selection_meta["selected_by_edge_reserved_count"] = 0
                selection_meta["selected_by_original_hybrid_count"] = 0
                selection_meta["selected_by_fill_count"] = int(len(keep_indices_np))
                selection_meta["duplicate_edge_hybrid_count"] = 0
                selection_meta["duplicate_after_exclusion_count"] = 0
                selection_meta["duplicate_with_original_hybrid_count"] = 0
                selection_meta["edge_scores_available"] = False
                selection_meta["edge_scores_shape"] = None
                selection_meta["edge_scores_finite_ratio"] = None
                selection_meta["edge_reserve_invalid"] = True
                selection_meta["edge_reserve_invalid_reason"] = "fallback_triggered"
                selection_meta["reserved_edge_topk_count"] = None
                selection_meta["reserved_edge_kept_count"] = None
                selection_meta["reserved_edge_dropped_count"] = None
                selection_meta["reserved_edge_topk_dropped_ratio"] = None
                selection_meta["non_reserved_edge_topk_count"] = None
                selection_meta["non_reserved_edge_kept_count"] = None
                selection_meta["non_reserved_edge_dropped_count"] = None
                selection_meta["non_reserved_edge_topk_dropped_ratio"] = None
                selection_meta["overall_depth_edge_topk_count"] = None
                selection_meta["overall_depth_edge_topk_kept_count"] = None
                selection_meta["overall_depth_edge_topk_dropped_count"] = None
                selection_meta["overall_depth_edge_topk_dropped_ratio"] = None
                metrics.dynamic_enabled = True
                metrics.dynamic_phase = "medium"
                metrics.dynamic_keep_ratio = len(keep_indices_np) / num_tokens if num_tokens else 0.75
                metrics.dynamic_keep_k = len(keep_indices_np)
                metrics.geo_risk_level = "medium"
                metrics.dynamic_keep_reason = fallback_reason
                metrics.interaction_lock = False
                metrics.score_ema_enabled = True
                metrics.score_ema_available = False
                metrics.ema_used_for_selection = False
                self._record_selection_path_diagnostics(
                    metrics=metrics,
                    selector_success=True,
                    exc=None,
                    requested_strategy=self.config.strategy,
                    effective_strategy="depth_edge_fast_diverse",
                    selector_name="select_keep_indices",
                    fallback_selector="depth_edge_fast_diverse",
                    keep_indices_np=keep_indices_np,
                    num_tokens=num_tokens,
                )
            else:
                contact_risk_arr = robot_metrics.get("rule_v0_contact_risk_scores")
                contact_vals = np.asarray(contact_risk_arr, dtype=np.float32).reshape(-1) if contact_risk_arr is not None else None
                adaptive_min = float(self.config.temporal_adaptive_threshold_min)
                adaptive_pct = float(self.config.temporal_adaptive_threshold_percentile)
                adaptive_threshold = adaptive_min
                if contact_vals is not None and np.any(np.isfinite(contact_vals)):
                    pct_val = float(np.percentile(contact_vals[np.isfinite(contact_vals)], adaptive_pct))
                    adaptive_threshold = max(adaptive_min, pct_val)
                dynamic_ratio = (robot_metrics.get("dynamic_decision") or {}).get("keep_ratio")
                if dynamic_ratio is None:
                    dynamic_ratio = float(self.config.keep_ratio)
                    metrics.keep_ratio_source = "cli_keep_ratio"
                else:
                    metrics.keep_ratio_source = "dynamic_decision"
                temporal = self._temporal_history.detect_interaction_lock(
                    contact_risk_score=contact_risk_arr,
                    final_scores=scores,
                    motion_direction=robot_metrics.get("motion_direction"),
                    valid_3d_ratio=robot_metrics.get("valid_token_ratio"),
                    dynamic_keep_ratio=dynamic_ratio,
                    config=self.config,
                    gripper_pos=robot_metrics.get("gripper_pos"),
                    token_points_robot=robot_metrics.get("token_points_robot"),
                    adaptive_threshold=adaptive_threshold,
                )
                interaction_locked = bool(temporal.get("interaction_lock"))
                ema_available = bool(temporal.get("score_ema_enabled"))
                if interaction_locked:
                    lock_ratio = float(self.config.temporal_interaction_lock_conservative_ratio)
                    chosen_ratio = max(float(dynamic_ratio), lock_ratio)
                    metrics.keep_ratio_source = "dynamic_decision"
                else:
                    chosen_ratio = float(dynamic_ratio)
                keep_count = max(1, min(num_tokens, int(round(num_tokens * chosen_ratio))))
                _select_ok = False
                # P5-fix BUG A: Diagnostic + invariant check before select_hybrid_v1_edge_reserve
                _edge_scores_raw = robot_metrics.get("edge_scores")
                _edge_arr = np.asarray(_edge_scores_raw, dtype=np.float32).reshape(-1) if _edge_scores_raw is not None else None
                _edge_scores_available = _edge_arr is not None and _edge_arr.size > 0
                _edge_scores_shape = int(_edge_arr.size) if _edge_arr is not None else None
                _edge_finite_ratio = float(np.mean(np.isfinite(_edge_arr))) if _edge_arr is not None and _edge_arr.size > 0 else None
                if self.config.debug or self.config.edge_reserve_k > 0:
                    print(f"[EDGE_RESERVE] step={self._hook_step_counter}: "
                          f"edge_scores_available={_edge_scores_available}, "
                          f"edge_scores_shape={_edge_scores_shape}, "
                          f"edge_scores_finite_ratio={_edge_finite_ratio}, "
                          f"edge_reserve_k={edge_reserve_k}")
                try:
                    hybrid_weights = self.config.hybrid_v1_weights
                    # P14-B: Apply robot-self mask to near_scores before passing to selector.
                    _near_scores_for_select = robot_metrics.get("near_scores")
                    _self_mask_diag_er: Dict[str, Any] = {}
                    _self_mask_enabled_er = self.config.robot_self_mask_enabled
                    metrics.robot_self_mask_enabled = _self_mask_enabled_er
                    _gripper_pixel_er = robot_metrics.get("gripper_pixel")
                    _token_u_er = None
                    _token_v_er = None
                    if "cache" in robot_metrics and robot_metrics["cache"]:
                        _token_u_er = robot_metrics["cache"].get("u")
                        _token_v_er = robot_metrics["cache"].get("v")
                    _self_mask_arr_er = np.zeros(num_tokens, dtype=bool)
                    if _self_mask_enabled_er and _near_scores_for_select is not None and self.config.robot_self_mask_apply_to_near_score:
                        _near_arr_er = np.asarray(_near_scores_for_select, dtype=np.float32).reshape(-1)
                        _near_masked_er, _self_mask_arr_er, _self_mask_diag_er = self._apply_self_mask(
                            scores=_near_arr_er,
                            gripper_pixel=_gripper_pixel_er,
                            token_u=_token_u_er,
                            token_v=_token_v_er,
                            core_radius_px=self.config.robot_self_mask_core_radius_px,
                        )
                        _near_scores_for_select = _near_masked_er
                    keep_indices_np, selection_meta = select_hybrid_v1_edge_reserve(
                        depth_edge_scores=robot_metrics.get("edge_scores"),
                        near_scores=_near_scores_for_select,
                        contact_risk_scores=contact_risk_arr,
                        corridor_scores=robot_metrics.get("rule_v0_motion_cone_scores"),
                        valid_mask=valid_mask,
                        keep_k=keep_count,
                        edge_reserve_k=edge_reserve_k,
                        grid_h=self.config.token_grid_shape[0],
                        grid_w=self.config.token_grid_shape[1],
                        cell_grid=self.config.cell_grid,
                        seed=self.config.seed,
                        w_edge=float(hybrid_weights.get("w_edge", 0.45)),
                        w_near=float(hybrid_weights.get("w_near", 0.20)),
                        w_contact=float(hybrid_weights.get("w_contact", 0.20)),
                        w_corr=float(hybrid_weights.get("w_corr", 0.10)),
                        w_diverse=float(hybrid_weights.get("w_diverse", 0.05)),
                    )
                    _select_ok = True
                    # P14-B: Record self-mask diagnostics for edge_reserve
                    metrics.self_mask_available = _self_mask_diag_er.get("self_mask_available", False)
                    metrics.self_mask_core_radius_px = _self_mask_diag_er.get("self_mask_core_radius_px")
                    metrics.self_mask_token_count = _self_mask_diag_er.get("self_mask_token_count", 0)
                    metrics.self_mask_token_ratio = _self_mask_diag_er.get("self_mask_token_ratio", 0.0)
                    metrics.gripper_pixel_u = _self_mask_diag_er.get("gripper_pixel_u")
                    metrics.gripper_pixel_v = _self_mask_diag_er.get("gripper_pixel_v")
                    metrics.gripper_pixel_in_bounds = _self_mask_diag_er.get("gripper_pixel_in_bounds", False)
                    metrics.gripper_projection_valid = _self_mask_diag_er.get("gripper_projection_valid", False)
                    metrics.near_score_mean_before_self_mask = _self_mask_diag_er.get("near_score_mean_before_self_mask")
                    metrics.near_score_mean_after_self_mask = _self_mask_diag_er.get("near_score_mean_after_self_mask")
                    metrics.near_score_self_region_mean = _self_mask_diag_er.get("near_score_self_region_mean")
                    metrics.near_score_nonself_region_mean = _self_mask_diag_er.get("near_score_nonself_region_mean")
                    if keep_indices_np is not None and _self_mask_arr_er is not None and len(_self_mask_arr_er) > 0:
                        _sel_arr_er = np.asarray(keep_indices_np, dtype=np.int64)
                        _sel_self_er = int(np.sum(_self_mask_arr_er[_sel_arr_er])) if _sel_arr_er.size > 0 else 0
                        metrics.selected_self_mask_token_count = _sel_self_er
                        metrics.selected_self_mask_token_ratio = float(_sel_self_er / len(_sel_arr_er)) if len(_sel_arr_er) > 0 else 0.0
                    self._record_selection_path_diagnostics(
                        metrics=metrics,
                        selector_success=True,
                        exc=None,
                        requested_strategy=self.config.strategy,
                        effective_strategy=self.config.strategy,
                        selector_name="select_hybrid_v1_edge_reserve",
                        fallback_selector=None,
                        keep_indices_np=keep_indices_np,
                        num_tokens=num_tokens,
                    )
                    # P5-fix BUG A invariant: detect if edge_reserve is actually working
                    _er_actual = selection_meta.get("edge_reserved_actual_count", 0)
                    _er_target = selection_meta.get("edge_reserved_target_count", 0)
                    _er_invalid = False
                    _er_invalid_reason = None
                    if selection_meta.get("edge_reserve_enabled") is True:
                        if _er_actual is None or _er_actual == 0:
                            if not _edge_scores_available:
                                _er_invalid = True
                                _er_invalid_reason = "edge_scores_missing"
                            elif _edge_scores_available and _er_target > 0:
                                _er_invalid = True
                                _er_invalid_reason = "edge_reserve_mechanism_failed"
                        if _er_actual is not None and _er_target is not None and _er_target > 0:
                            if _er_actual < _er_target:
                                _er_invalid = True
                                _er_invalid_reason = "edge_reserve_partial_failure"
                    selection_meta["edge_reserve_invalid"] = _er_invalid
                    selection_meta["edge_reserve_invalid_reason"] = _er_invalid_reason
                    # P5-fix: always carry forward the edge_scores diagnostics
                    selection_meta["edge_scores_available"] = _edge_scores_available
                    selection_meta["edge_scores_shape"] = _edge_scores_shape
                    selection_meta["edge_scores_finite_ratio"] = _edge_finite_ratio
                except Exception as exc:
                    print(f"[PRUNING] select_hybrid_v1_edge_reserve exception (step={self._hook_step_counter}): {type(exc).__name__}: {exc}")
                    metrics.fallback_used = True
                    metrics.keep_ratio_source = "fallback"
                    metrics.fallback_reason = f"edge_reserve_select_error:{type(exc).__name__}"
                    strategy = "depth_edge_fast_diverse"
                    keep_count = int(round(num_tokens * 0.75))
                    keep_indices_np = None
                    selection_meta = {
                        "fallback_used": True,
                        "fallback_reason": f"edge_reserve_select_error:{type(exc).__name__}",
                        "edge_reserve_enabled": False,
                        "edge_reserve_ratio": 0.0,
                        "edge_reserved_target_count": 0,
                        "edge_reserved_actual_count": 0,
                        "edge_reserved_survival_ratio": None,
                        "final_selected_count": 0,
                        "selected_by_edge_reserved_count": 0,
                        "selected_by_original_hybrid_count": 0,
                        "selected_by_fill_count": 0,
                        "duplicate_edge_hybrid_count": 0,
                        "duplicate_after_exclusion_count": 0,
                        "duplicate_with_original_hybrid_count": 0,
                        "edge_scores_available": False,
                        "edge_scores_shape": None,
                        "edge_scores_finite_ratio": None,
                        "edge_reserve_invalid": True,
                        "edge_reserve_invalid_reason": "exception",
                        "reserved_edge_topk_count": None,
                        "reserved_edge_kept_count": None,
                        "reserved_edge_dropped_count": None,
                        "reserved_edge_topk_dropped_ratio": None,
                        "non_reserved_edge_topk_count": None,
                        "non_reserved_edge_kept_count": None,
                        "non_reserved_edge_dropped_count": None,
                        "non_reserved_edge_topk_dropped_ratio": None,
                        "overall_depth_edge_topk_count": None,
                        "overall_depth_edge_topk_kept_count": None,
                        "overall_depth_edge_topk_dropped_count": None,
                        "overall_depth_edge_topk_dropped_ratio": None,
                    }
                    metrics.dynamic_enabled = True
                    metrics.dynamic_phase = "medium"
                    metrics.dynamic_keep_ratio = keep_count / num_tokens if num_tokens else None
                    metrics.dynamic_keep_k = keep_count
                    metrics.interaction_lock = False
                    metrics.score_ema_enabled = True
                    metrics.score_ema_available = False
                    metrics.ema_used_for_selection = False
                    self._record_selection_path_diagnostics(
                        metrics=metrics,
                        selector_success=False,
                        exc=exc,
                        requested_strategy=self.config.strategy,
                        effective_strategy="depth_edge_fast_diverse",
                        selector_name="select_hybrid_v1_edge_reserve",
                        fallback_selector="depth_edge_fast_diverse",
                        keep_indices_np=None,
                        num_tokens=num_tokens,
                    )
                if _select_ok:
                    metrics.dynamic_enabled = True
                    metrics.dynamic_phase = "interaction_lock" if interaction_locked else "hybrid_temporal_edge_reserve_v1"
                    metrics.dynamic_keep_ratio = keep_count / num_tokens if num_tokens else None
                    metrics.dynamic_keep_k = keep_count
                    metrics.geo_risk_level = metrics.dynamic_phase
                    metrics.dynamic_keep_reason = "temporal_interaction_lock" if interaction_locked else "hybrid_temporal_edge_reserve_v1"
                    metrics.interaction_lock = interaction_locked
                    metrics.interaction_lock_reason = str(temporal.get("lock_reason", "none"))
                    metrics.temporal_stability = temporal.get("temporal_stability")
                    metrics.history_length = temporal.get("history_length")
                    metrics.score_ema_enabled = ema_available
                    metrics.score_ema_available = ema_available
                    metrics.ema_used_for_selection = False
                    metrics.lock_condition_failed_reason = str(temporal.get("reason", "none"))
                    metrics.topk_contact_lock = bool(temporal.get("topk_contact_lock"))
                    metrics.elevated_current_lock = bool(temporal.get("elevated_current_lock"))
                    metrics.gripper_lock = bool(temporal.get("gripper_lock"))
                    metrics.region_lock = bool(temporal.get("region_lock"))
                    metrics.adaptive_threshold_mean = adaptive_threshold
                    metrics.adaptive_threshold_max = adaptive_threshold
                    metrics.interaction_lock_ratio = 1.0 if interaction_locked else 0.0
                    component_summary = (robot_metrics.get("dynamic_decision") or {}).get("component_summary", {}) or {}
                    metrics.num_high_contact_tokens = component_summary.get("num_high_contact_tokens")
                    metrics.num_valid_3d_tokens = component_summary.get("num_valid_3d_tokens")
                    metrics.phase_far_count = 0
                    metrics.phase_mid_count = 0
                    metrics.phase_near_count = 1 if interaction_locked else 0
                    metrics.phase_fallback_safe_count = 0

        # P7: hybrid_budget_v2 — budget-based depth/robot protection
        if self.config.strategy == "hybrid_budget_v2":
            if fallback_reason is not None:
                metrics.fallback_used = True
                metrics.keep_ratio_source = "fallback"
                metrics.fallback_reason = fallback_reason
                strategy = "depth_edge_fast_diverse"
                keep_count = int(round(num_tokens * 0.75))
                keep_indices_np, selection_meta = select_keep_indices(
                    strategy=strategy,
                    num_tokens=num_tokens,
                    keep_count=keep_count,
                    scores=scores,
                    valid_mask=valid_mask,
                    seed=self.config.seed,
                    grid_size=self.config.token_grid_shape[0],
                    cell_grid=self.config.cell_grid,
                    reserve_tokens=reserve_tokens,
                )
                selection_meta["fallback_used"] = True
                selection_meta["fallback_reason"] = fallback_reason
                selection_meta["hybrid_budget_v2"] = False
                selection_meta["depth_edge_budget_ratio"] = float(self.config.hybrid_budget_v2_depth_edge_ratio)
                selection_meta["robot_contact_budget_ratio"] = float(self.config.hybrid_budget_v2_robot_contact_ratio)
                selection_meta["safety_budget_ratio"] = float(self.config.hybrid_budget_v2_safety_ratio)
                selection_meta["total_keep_budget"] = keep_count
                selection_meta["depth_edge_budget"] = None
                selection_meta["robot_geo_budget"] = None
                selection_meta["fill_budget"] = keep_count
                selection_meta["safety_budget"] = None
                selection_meta["K_depth_actual"] = None
                selection_meta["K_robot_actual"] = None
                selection_meta["K_fill_actual"] = None
                selection_meta["overlap_depth_robot_count"] = None
                selection_meta["overlap_depth_robot_diagnostic"] = None
                selection_meta["depth_edge_candidates_count"] = int(np.sum(valid_mask)) if valid_mask is not None else None
                selection_meta["robot_geo_candidates_count"] = int(np.sum(valid_mask)) if valid_mask is not None else None
                selection_meta["depth_edge_reserved_kept_count"] = None
                selection_meta["robot_geo_reserved_kept_count"] = None
                selection_meta["fill_from_depth_count"] = None
                selection_meta["fill_from_robot_count"] = None
                selection_meta["fill_from_other_count"] = None
                self._record_selection_path_diagnostics(
                    metrics=metrics,
                    selector_success=False,
                    exc=None,
                    requested_strategy=self.config.strategy,
                    effective_strategy=self.config.strategy,
                    selector_name="select_hybrid_budget_v2",
                    fallback_selector="depth_edge_fast_diverse",
                    keep_indices_np=keep_indices_np,
                    num_tokens=num_tokens,
                )
            else:
                _depth_scores = robot_metrics.get("edge_scores")
                _near_scores = robot_metrics.get("near_scores")
                _contact_scores = robot_metrics.get("rule_v0_contact_risk_scores", robot_metrics.get("contact_risk_scores"))
                # hybrid_budget_v2 uses depth_edge + robot_geo (use near_scores as proxy)
                _robot_geo = _near_scores
                try:
                    keep_indices_np, selection_meta = select_hybrid_budget_v2(
                        depth_edge_scores=_depth_scores,
                        robot_geo_scores=_robot_geo,
                        valid_mask=valid_mask,
                        keep_k=keep_count,
                        depth_edge_budget_ratio=float(self.config.hybrid_budget_v2_depth_edge_ratio),
                        robot_contact_budget_ratio=float(self.config.hybrid_budget_v2_robot_contact_ratio),
                        safety_budget_ratio=float(self.config.hybrid_budget_v2_safety_ratio),
                        grid_h=self.config.token_grid_shape[0],
                        grid_w=self.config.token_grid_shape[1],
                    )
                    _select_ok = True
                    # P7: Inject P7 fields into selection_meta BEFORE _record_selection_path_diagnostics
                    # so that finalize_selection_debug_info() inside it sees correct phase accounting.
                    # Include fallback_actual and selected_by_fallback_count for phase accounting.
                    _p7_fields = {
                        "hybrid_budget_v2": True,
                        "depth_edge_budget_ratio": float(self.config.hybrid_budget_v2_depth_edge_ratio),
                        "robot_contact_budget_ratio": float(self.config.hybrid_budget_v2_robot_contact_ratio),
                        "safety_budget_ratio": float(self.config.hybrid_budget_v2_safety_ratio),
                        "K_depth_actual": selection_meta.get("K_depth_actual"),
                        "K_robot_actual": selection_meta.get("K_robot_actual"),
                        "K_fill_actual": selection_meta.get("K_fill_actual"),
                        "overlap_depth_robot_count": selection_meta.get("overlap_depth_robot_count"),
                        "overlap_depth_robot_diagnostic": selection_meta.get("overlap_depth_robot_diagnostic"),
                        "depth_edge_candidates_count": selection_meta.get("depth_edge_candidates_count"),
                        "robot_geo_candidates_count": selection_meta.get("robot_geo_candidates_count"),
                        "depth_edge_reserved_kept_count": selection_meta.get("depth_edge_reserved_kept_count"),
                        "robot_geo_reserved_kept_count": selection_meta.get("robot_geo_reserved_kept_count"),
                        "fill_from_depth_count": selection_meta.get("fill_from_depth_count"),
                        "fill_from_robot_count": selection_meta.get("fill_from_robot_count"),
                        "fill_from_other_count": selection_meta.get("fill_from_other_count"),
                        # Phase accounting: include fallback count even if 0
                        "fallback_actual": selection_meta.get("fallback_actual"),
                        "selected_by_fallback_count": selection_meta.get("selected_by_fallback_count"),
                    }
                    for _k, _v in _p7_fields.items():
                        if _v is not None:
                            selection_meta[_k] = _v
                    # Also inject selected_by_fallback_count into selection_meta for finalize_debug_info
                    if "selected_by_fallback_count" not in selection_meta:
                        selection_meta["selected_by_fallback_count"] = selection_meta.get("fallback_actual")
                    self._record_selection_path_diagnostics(
                        metrics=metrics,
                        selector_success=True,
                        exc=None,
                        requested_strategy=self.config.strategy,
                        effective_strategy=self.config.strategy,
                        selector_name="select_hybrid_budget_v2",
                        fallback_selector=None,
                        keep_indices_np=keep_indices_np,
                        num_tokens=num_tokens,
                        selection_meta=selection_meta,
                    )
                    # P7: Directly compute and set correct phase accounting in metrics.
                    # finalize_selection_debug_info runs twice (inside _record_path and in post-processing),
                    # which corrupts hybrid_budget_v2 phase accounting. Override here.
                    _sm = selection_meta
                    _kept = metrics.num_visual_tokens_kept or 0
                    _p1 = int(_sm.get("selected_by_depth_edge_count") or 0)
                    _p2 = int(_sm.get("selected_by_robot_geo_count") or 0)
                    _pfill = int(_sm.get("selected_by_fill_count") or 0)
                    _pfb = int(_sm.get("fallback_actual") or _sm.get("selected_by_fallback_count") or 0)
                    _punattr = int(_sm.get("selected_unattributed") or 0)
                    _phase_sum = _p1 + _p2 + _pfill + _pfb
                    metrics.selected_by_phase1 = _p1
                    metrics.selected_by_phase2 = _p2
                    metrics.selected_by_phase3 = None
                    metrics.selected_by_fill = _pfill
                    metrics.selected_by_fallback = _pfb
                    metrics.selected_unattributed = _punattr
                    metrics.phase_accounting_sum = _phase_sum
                    if abs(_phase_sum - _kept) <= 2:
                        metrics.phase_accounting_valid = True
                        metrics.phase_accounting_error = None
                    else:
                        metrics.phase_accounting_valid = False
                        metrics.phase_accounting_error = f"phase_sum={_phase_sum} != kept={_kept}"
                    # Copy remaining P7 fields into metrics
                    metrics.hybrid_budget_v2 = True
                    metrics.depth_edge_budget_ratio = float(self.config.hybrid_budget_v2_depth_edge_ratio)
                    metrics.robot_contact_budget_ratio = float(self.config.hybrid_budget_v2_robot_contact_ratio)
                    metrics.safety_budget_ratio = float(self.config.hybrid_budget_v2_safety_ratio)
                    metrics.K_depth_actual = _sm.get("K_depth_actual")
                    metrics.K_robot_actual = _sm.get("K_robot_actual")
                    metrics.K_fill_actual = _sm.get("K_fill_actual")
                    metrics.overlap_depth_robot_count = _sm.get("overlap_depth_robot_count")
                    metrics.overlap_depth_robot_diagnostic = _sm.get("overlap_depth_robot_diagnostic")
                    metrics.depth_edge_candidates_count = _sm.get("depth_edge_candidates_count")
                    metrics.robot_geo_candidates_count = _sm.get("robot_geo_candidates_count")
                    metrics.depth_edge_reserved_kept_count = _sm.get("depth_edge_reserved_kept_count")
                    metrics.robot_geo_reserved_kept_count = _sm.get("robot_geo_reserved_kept_count")
                    metrics.fill_from_depth_count = _sm.get("fill_from_depth_count")
                    metrics.fill_from_robot_count = _sm.get("fill_from_robot_count")
                    metrics.fill_from_other_count = _sm.get("fill_from_other_count")
                    # hybrid_budget_v2 phase accounting is correct — skip post-processing
                    # _record_selection_path_diagnostics call to avoid re-corrupting it
                    _skip_postprocess_diag = True
                except Exception as exc:
                    print(f"[PRUNING] select_hybrid_budget_v2 exception (step={self._hook_step_counter}): {type(exc).__name__}: {exc}")
                    metrics.fallback_used = True
                    metrics.keep_ratio_source = "fallback"
                    metrics.fallback_reason = f"hybrid_budget_v2_error:{type(exc).__name__}"
                    keep_indices_np, selection_meta = select_keep_indices(
                        strategy="depth_edge_fast_diverse",
                        num_tokens=num_tokens,
                        keep_count=keep_count,
                        scores=_depth_scores,
                        valid_mask=valid_mask,
                        seed=self.config.seed,
                        grid_size=self.config.token_grid_shape[0],
                        cell_grid=self.config.cell_grid,
                        reserve_tokens=reserve_tokens,
                    )
                    self._record_selection_path_diagnostics(
                        metrics=metrics,
                        selector_success=False,
                        exc=exc,
                        requested_strategy=self.config.strategy,
                        effective_strategy="depth_edge_fast_diverse",
                        selector_name="select_hybrid_budget_v2",
                        fallback_selector="depth_edge_fast_diverse",
                        keep_indices_np=None,
                        num_tokens=num_tokens,
                    )
                    # hybrid_budget_v2 exception-fallback phase accounting is correct
                    _skip_postprocess_diag = True
                    selection_meta["hybrid_budget_v2"] = False
                    selection_meta["depth_edge_budget_ratio"] = float(self.config.hybrid_budget_v2_depth_edge_ratio)
                    selection_meta["robot_contact_budget_ratio"] = float(self.config.hybrid_budget_v2_robot_contact_ratio)
                    selection_meta["safety_budget_ratio"] = float(self.config.hybrid_budget_v2_safety_ratio)
                    selection_meta["total_keep_budget"] = keep_count
                    selection_meta["depth_edge_budget"] = None
                    selection_meta["robot_geo_budget"] = None
                    selection_meta["fill_budget"] = keep_count
                    selection_meta["safety_budget"] = None
                    selection_meta["K_depth_actual"] = None
                    selection_meta["K_robot_actual"] = None
                    selection_meta["K_fill_actual"] = None
                    selection_meta["overlap_depth_robot_count"] = None
                    selection_meta["overlap_depth_robot_diagnostic"] = None
                    selection_meta["depth_edge_candidates_count"] = int(np.sum(valid_mask)) if valid_mask is not None else None
                    selection_meta["robot_geo_candidates_count"] = int(np.sum(valid_mask)) if valid_mask is not None else None
                    selection_meta["depth_edge_reserved_kept_count"] = None
                    selection_meta["robot_geo_reserved_kept_count"] = None
                    selection_meta["fill_from_depth_count"] = None
                    selection_meta["fill_from_robot_count"] = None
                    selection_meta["fill_from_other_count"] = None
                    # hybrid_budget_v2 exception-fallback — skip post-processing diag
                    _skip_postprocess_diag = True

        _bbv0_strategy_check = str(self.config.strategy)
        if _bbv0_strategy_check == "robot_geo_branch_budget_v0":
            _depth_scores = robot_metrics.get("edge_scores")
            _hybrid_raw = robot_metrics.get("hybrid_final_scores")
            if _hybrid_raw is None:
                _hybrid_raw = robot_metrics.get("final_scores")
            _hybrid_scores = _hybrid_raw

            # P14-B: Apply robot-self mask to hybrid_final_scores if enabled.
            # self_mask is applied to near_score component of hybrid_final_scores only.
            _self_mask_diag: Dict[str, Any] = {}
            _self_mask_enabled = self.config.robot_self_mask_enabled
            metrics.robot_self_mask_enabled = _self_mask_enabled
            _gripper_pixel = robot_metrics.get("gripper_pixel")
            _token_u = None
            _token_v = None
            if "cache" in robot_metrics and robot_metrics["cache"]:
                _token_u = robot_metrics["cache"].get("u")
                _token_v = robot_metrics["cache"].get("v")
            if _self_mask_enabled:
                pass  # diagnostics collected below
            if _self_mask_enabled and _hybrid_scores is not None:
                _hybrid_scores, _self_mask_arr, _self_mask_diag = self._apply_self_mask(
                    scores=np.asarray(_hybrid_scores, dtype=np.float32),
                    gripper_pixel=_gripper_pixel,
                    token_u=_token_u,
                    token_v=_token_v,
                    core_radius_px=self.config.robot_self_mask_core_radius_px,
                )
                # Also apply to near_scores if available
                if robot_metrics.get("near_scores") is not None and self.config.robot_self_mask_apply_to_near_score:
                    _near_arr = np.asarray(robot_metrics["near_scores"], dtype=np.float32).reshape(-1)
                    _near_masked, _, _ = self._apply_self_mask(
                        scores=_near_arr,
                        gripper_pixel=_gripper_pixel,
                        token_u=_token_u,
                        token_v=_token_v,
                        core_radius_px=self.config.robot_self_mask_core_radius_px,
                    )
                    robot_metrics["near_scores"] = _near_masked
            else:
                _self_mask_arr = np.zeros(num_tokens, dtype=bool)

            # Track which scores were unavailable (for diagnostics)
            _de_unavailable = _depth_scores is None
            _hyb_unavailable = _hybrid_scores is None
            _fallback_reasons: List[str] = []
            if _de_unavailable:
                _fallback_reasons.append("depth_edge_scores_unavailable")
            if _hyb_unavailable:
                _fallback_reasons.append("hybrid_final_scores_unavailable")
            _fallback_reason = "|".join(_fallback_reasons) if _fallback_reasons else None

            _token_u = None
            _token_v = None
            if "cache" in robot_metrics and robot_metrics["cache"]:
                _token_u = robot_metrics["cache"].get("u")
                _token_v = robot_metrics["cache"].get("v")
            try:
                keep_indices_np, selection_meta = select_branch_budget_v0(
                    depth_edge_scores=_depth_scores,
                    hybrid_final_scores=_hybrid_scores,
                    valid_mask=valid_mask,
                    keep_k=keep_count,
                    depth_edge_budget=int(self.config.branch_budget_depth_tokens),
                    hybrid_action_budget=int(self.config.branch_budget_hybrid_tokens),
                    depth_edge_ratio_override=float(self.config.branch_budget_depth_ratio),
                    hybrid_action_ratio_override=float(self.config.branch_budget_hybrid_ratio),
                    grid_h=self.config.token_grid_shape[0],
                    grid_w=self.config.token_grid_shape[1],
                    token_u=_token_u,
                    token_v=_token_v,
                )
                _bb_fields = {
                    "branch_budget_v0": True,
                    "total_keep_budget": int(keep_count),
                    "depth_edge_budget": int(self.config.branch_budget_depth_tokens),
                    "hybrid_action_budget": int(self.config.branch_budget_hybrid_tokens),
                    "diversity_fill_budget": keep_count - int(self.config.branch_budget_depth_tokens) - int(self.config.branch_budget_hybrid_tokens),
                    "temporal_budget": None,
                    "branch_budget_depth_ratio_override": float(self.config.branch_budget_depth_ratio),
                    "branch_budget_hybrid_ratio_override": float(self.config.branch_budget_hybrid_ratio),
                    "score_path_fallback_reason": _fallback_reason,
                }
                for _k, _v in _bb_fields.items():
                    selection_meta[_k] = _v
                self._record_selection_path_diagnostics(
                    metrics=metrics,
                    selector_success=True,
                    exc=None,
                    requested_strategy=self.config.strategy,
                    effective_strategy=self.config.strategy,
                    selector_name="select_branch_budget_v0",
                    fallback_selector=None,
                    keep_indices_np=keep_indices_np,
                    num_tokens=num_tokens,
                    selection_meta=selection_meta,
                )
                # P11: Directly set phase accounting in metrics to prevent post-processing corruption
                _sm = selection_meta
                _kept = int(len(keep_indices_np)) if keep_indices_np is not None else 0
                metrics.selected_by_depth_branch_count = int(_sm.get("selected_by_depth_branch") or 0)
                metrics.selected_by_hybrid_branch_count = int(_sm.get("selected_by_hybrid_branch") or 0)
                metrics.selected_by_fill_branch_count = int(_sm.get("selected_by_fill") or 0)
                metrics.branch_budget_v0 = True
                metrics.branch_accounting_valid = bool(_sm.get("branch_accounting_valid", False))
                metrics.branch_sum_equals_kept = bool(_sm.get("branch_sum_equals_kept", False))
                metrics.overlap_depth_edge_hybrid_count = _sm.get("overlap_depth_edge_hybrid_count")
                metrics.overlap_depth_edge_hybrid_ratio = _sm.get("overlap_depth_edge_hybrid_ratio")
                metrics.non_reserved_depth_edge_dropped_ratio = _sm.get("non_reserved_depth_edge_dropped_ratio")
                # P14-B: Record self-mask diagnostics
                metrics.self_mask_available = _self_mask_diag.get("self_mask_available", False)
                metrics.self_mask_core_radius_px = _self_mask_diag.get("self_mask_core_radius_px")
                metrics.self_mask_token_count = _self_mask_diag.get("self_mask_token_count", 0)
                metrics.self_mask_token_ratio = _self_mask_diag.get("self_mask_token_ratio", 0.0)
                metrics.gripper_pixel_u = _self_mask_diag.get("gripper_pixel_u")
                metrics.gripper_pixel_v = _self_mask_diag.get("gripper_pixel_v")
                metrics.gripper_pixel_in_bounds = _self_mask_diag.get("gripper_pixel_in_bounds", False)
                metrics.gripper_projection_valid = _self_mask_diag.get("gripper_projection_valid", False)
                metrics.near_score_mean_before_self_mask = _self_mask_diag.get("near_score_mean_before_self_mask")
                metrics.near_score_mean_after_self_mask = _self_mask_diag.get("near_score_mean_after_self_mask")
                metrics.near_score_self_region_mean = _self_mask_diag.get("near_score_self_region_mean")
                metrics.near_score_nonself_region_mean = _self_mask_diag.get("near_score_nonself_region_mean")
                # Selected self-mask token counts (compute from keep_indices and self_mask_arr)
                if keep_indices_np is not None and _self_mask_arr is not None and len(_self_mask_arr) > 0:
                    _sel_arr = np.asarray(keep_indices_np, dtype=np.int64)
                    _sel_self = int(np.sum(_self_mask_arr[_sel_arr])) if _sel_arr.size > 0 else 0
                    metrics.selected_self_mask_token_count = _sel_self
                    metrics.selected_self_mask_token_ratio = float(_sel_self / len(_sel_arr)) if len(_sel_arr) > 0 else 0.0
                # Hybrid score before/after self-mask
                metrics.hybrid_score_mean_before_self_mask = _self_mask_diag.get("hybrid_score_mean_before_self_mask")
                metrics.hybrid_score_mean_after_self_mask = _self_mask_diag.get("hybrid_score_mean_after_self_mask")
                # Hybrid score stats (computed from _hybrid_scores if available)
                if _hybrid_scores is not None:
                    _hyb_a = np.asarray(_hybrid_scores, dtype=np.float32).reshape(-1)
                    _hyb_v = _hyb_a[valid_mask]
                    if _hyb_v.size > 0:
                        metrics.final_hybrid_score_mean = float(np.nanmean(_hyb_v))
                        metrics.final_hybrid_score_max = float(np.nanmax(_hyb_v))
                        metrics.final_hybrid_score_std = float(np.nanstd(_hyb_v))
                if _depth_scores is not None:
                    _de_a = np.asarray(_depth_scores, dtype=np.float32).reshape(-1)
                    _de_v = _de_a[valid_mask]
                    if _de_v.size > 0:
                        metrics.edge_score_mean = float(np.nanmean(_de_v))
                        metrics.edge_score_max = float(np.nanmax(_de_v))
                        metrics.edge_score_std = float(np.nanstd(_de_v))
                _skip_postprocess_diag = True
            except Exception as exc:
                print(f"[PRUNING] select_branch_budget_v0 exception (step={self._hook_step_counter}): {type(exc).__name__}: {exc}")
                metrics.fallback_used = True
                metrics.keep_ratio_source = "fallback"
                metrics.fallback_reason = f"branch_budget_v0_error:{type(exc).__name__}"
                keep_indices_np, selection_meta = select_keep_indices(
                    strategy="depth_edge_fast_diverse",
                    num_tokens=num_tokens,
                    keep_count=keep_count,
                    scores=_depth_scores,
                    valid_mask=valid_mask,
                    seed=self.config.seed,
                    grid_size=self.config.token_grid_shape[0],
                    cell_grid=self.config.cell_grid,
                    reserve_tokens=reserve_tokens,
                )
                self._record_selection_path_diagnostics(
                    metrics=metrics,
                    selector_success=False,
                    exc=exc,
                    requested_strategy=self.config.strategy,
                    effective_strategy="depth_edge_fast_diverse",
                    selector_name="select_branch_budget_v0",
                    fallback_selector="depth_edge_fast_diverse",
                    keep_indices_np=None,
                    num_tokens=num_tokens,
                )
                _skip_postprocess_diag = True
                selection_meta["branch_budget_v0"] = False
                selection_meta["total_keep_budget"] = keep_count
                selection_meta["depth_edge_budget"] = None
                selection_meta["hybrid_action_budget"] = None
                selection_meta["diversity_fill_budget"] = keep_count
                selection_meta["branch_accounting_valid"] = None
                selection_meta["branch_sum_equals_kept"] = None
                selection_meta["overlap_depth_edge_hybrid_count"] = None
                selection_meta["overlap_depth_edge_hybrid_ratio"] = None
                selection_meta["non_reserved_depth_edge_dropped_ratio"] = None
                selection_meta["fallback_used"] = True
                selection_meta["fallback_reason"] = f"branch_budget_v0_error:{type(exc).__name__}"
                metrics.branch_budget_v0 = False
                metrics.branch_accounting_valid = None
                metrics.branch_sum_equals_kept = None

        # ─────────────────────────────────────────────────────────────────────────
        # P15: robot_geo_acgtp_v1 — Action-Constrained Geometric Token Protection
        # ─────────────────────────────────────────────────────────────────────────
        _acgtp_strategy_check = str(self.config.strategy)
        if _acgtp_strategy_check == "robot_geo_acgtp_v1" and keep_indices_np is None:
            score_stats: Dict[str, Any] = {}  # Initialize so exception handler can reference it
            # ── Build all ACGTP-v1 branch scores ───────────────────────────────
            _scene_scores = robot_metrics.get("acgtp_scene_layout_scores")
            _de_scores = robot_metrics.get("edge_scores")
            _contact_scores = robot_metrics.get("acgtp_contact_ring_scores")
            _motion_scores = robot_metrics.get("acgtp_motion_corridor_scores")
            _action_constraint_scores = robot_metrics.get("acgtp_action_constraint_scores")
            _motion_valid = bool(robot_metrics.get("acgtp_motion_corridor_valid", False))
            _fill_mask = robot_metrics.get("acgtp_constrained_fill_mask")
            _self_core = robot_metrics.get("acgtp_self_core_mask")
            _token_u = None
            _token_v = None
            if "cache" in robot_metrics and robot_metrics["cache"]:
                _token_u = robot_metrics["cache"].get("u")
                _token_v = robot_metrics["cache"].get("v")

            # Fallback: if scene_layout not computed, derive from depth distribution
            if _scene_scores is None:
                _depth_vals = robot_metrics.get("depth_metric")
                if _depth_vals is not None:
                    _depth_a = np.asarray(_depth_vals, dtype=np.float32).reshape(-1)
                    _scene_scores = np.clip((_depth_a - 0.3) / 1.7, 0.0, 1.0)
                else:
                    _scene_scores = np.zeros(num_tokens, dtype=np.float32)

            # Fallback: if contact_ring not computed, use near_scores with self-mask
            if _contact_scores is None:
                _near_s = robot_metrics.get("near_scores")
                if _near_s is not None:
                    _near_a = np.asarray(_near_s, dtype=np.float32).reshape(-1)
                    # Simple self-filter: penalize near gripper tokens
                    _contact_scores = _near_a.copy()
                else:
                    _contact_scores = np.zeros(num_tokens, dtype=np.float32)

            # Fallback: if motion_corridor not computed, use motion_cone_scores
            if _motion_scores is None:
                _mc = robot_metrics.get("motion_cone_scores")
                if _mc is not None:
                    _motion_scores = np.asarray(_mc, dtype=np.float32).reshape(-1)
                else:
                    _motion_scores = np.zeros(num_tokens, dtype=np.float32)

            # Fallback: if depth_edge not available, use edge_scores from robot_metrics
            if _de_scores is None:
                _de_scores = robot_metrics.get("edge_scores")
            if _de_scores is None:
                _de_scores = np.zeros(num_tokens, dtype=np.float32)

            # Build constrained fill mask: tokens with valid depth + scene relevance
            if _fill_mask is None:
                _depth_metric = robot_metrics.get("depth_metric")
                if _depth_metric is not None:
                    _dm = np.asarray(_depth_metric, dtype=np.float32).reshape(-1)
                    _scene_s = np.asarray(_scene_scores, dtype=np.float32).reshape(-1)
                    _fill_mask = (np.isfinite(_dm) & (_dm > 0.1) & (_dm < 5.0) & (_scene_s > 0.0)).astype(np.float32)
                else:
                    _fill_mask = np.ones(num_tokens, dtype=np.float32)

            # Self-core mask for contact ring filtering
            if _self_core is None:
                _gripper_pixel = robot_metrics.get("gripper_pixel")
                if _gripper_pixel is not None and _token_u is not None:
                    _gx, _gy = float(_gripper_pixel[0]), float(_gripper_pixel[1])
                    _u_a = np.asarray(_token_u, dtype=np.float32).reshape(-1)
                    _v_a = np.asarray(_token_v, dtype=np.float32).reshape(-1)
                    _core_r = float(self.config.acgtp_self_core_radius_px)
                    _dist2 = (_u_a - _gx) ** 2 + (_v_a - _gy) ** 2
                    _self_core = (np.sqrt(_dist2) <= _core_r).astype(np.float32)
                else:
                    _self_core = np.zeros(num_tokens, dtype=np.float32)

            # Gripper pixel projection for contact_ring diagnostics
            _gripper_pixel_metrics = robot_metrics.get("gripper_pixel")
            # Full module results for diagnostics
            _contact_result_metrics = robot_metrics.get("acgtp_contact_ring_result")
            _motion_result_metrics = robot_metrics.get("acgtp_motion_corridor_result")
            _scene_result_metrics = robot_metrics.get("acgtp_scene_layout_result")
            _action_constraint_result_metrics = robot_metrics.get("acgtp_action_constraint_result")

            try:
                keep_indices_np, selection_meta = select_acgtp_v1(
                    scene_layout_scores=_scene_scores,
                    depth_edge_scores=_de_scores,
                    contact_ring_scores=_contact_scores,
                    motion_corridor_scores=_motion_scores,
                    valid_mask=valid_mask,
                    keep_k=keep_count,
                    constrained_fill_mask=_fill_mask,
                    token_u=_token_u,
                    token_v=_token_v,
                    grid_h=self.config.token_grid_shape[0],
                    grid_w=self.config.token_grid_shape[1],
                    w_scene_layout=float(self.config.acgtp_w_scene_layout),
                    w_depth_structure=float(self.config.acgtp_w_depth_structure),
                    w_contact_ring=float(self.config.acgtp_w_contact_ring),
                    w_motion_corridor=float(self.config.acgtp_w_motion_corridor),
                    hard_protect_ratio=float(self.config.acgtp_hard_protect_ratio),
                    motion_corridor_valid=_motion_valid,
                    self_core_mask=_self_core,
                    contact_ring_inner_px=float(self.config.acgtp_contact_ring_inner_px),
                    contact_ring_outer_px=float(self.config.acgtp_contact_ring_outer_px),
                    contact_requires_edge_or_object=bool(self.config.acgtp_contact_requires_edge_or_object),
                    depth_edge_score_for_gate=_de_scores,
                    # Pass real motion corridor config and result values for diagnostics
                    _motion_result_for_diag=_motion_result_metrics,
                    _scene_result_for_diag=_scene_result_metrics,
                    action_constraint_scores=_action_constraint_scores,
                    _action_constraint_result_for_diag=_action_constraint_result_metrics,
                )
                # ── Record ACGTP-v1 metrics ─────────────────────────────────
                _sm = selection_meta
                metrics.acgtp_v1 = True
                metrics.acgtp_w_scene_layout = float(self.config.acgtp_w_scene_layout)
                metrics.acgtp_w_depth_structure = float(self.config.acgtp_w_depth_structure)
                metrics.acgtp_w_contact_ring = float(self.config.acgtp_w_contact_ring)
                metrics.acgtp_w_motion_corridor = float(self.config.acgtp_w_motion_corridor)
                metrics.acgtp_self_core_radius_px = float(self.config.acgtp_self_core_radius_px)
                metrics.acgtp_contact_ring_inner_px = float(self.config.acgtp_contact_ring_inner_px)
                metrics.acgtp_contact_ring_outer_px = float(self.config.acgtp_contact_ring_outer_px)
                metrics.acgtp_self_core_token_count = _sm.get("acgtp_self_core_token_count")
                metrics.acgtp_self_core_token_ratio = _sm.get("acgtp_self_core_token_ratio")
                # contact_ring_token_count from selector = top-k gated candidates selected
                # The total ring/gated token counts come from the module result
                metrics.acgtp_contact_ring_token_count = _sm.get("acgtp_contact_ring_token_count")
                metrics.acgtp_contact_ring_token_ratio = _sm.get("acgtp_contact_ring_token_ratio")
                metrics.acgtp_contact_ring_gated_token_count = _sm.get("acgtp_contact_ring_gated_token_count")
                metrics.acgtp_contact_ring_valid = _sm.get("acgtp_contact_ring_valid")
                # Gripper pixel from module result
                if _contact_result_metrics is not None:
                    metrics.gripper_pixel_u = _contact_result_metrics.get("gripper_pixel_u")
                    metrics.gripper_pixel_v = _contact_result_metrics.get("gripper_pixel_v")
                    metrics.gripper_in_bounds = _contact_result_metrics.get("gripper_in_bounds")
                    # Full ring token counts from the module (not just top-k)
                    _ring_total = _contact_result_metrics.get("contact_ring_token_count")
                    if _ring_total is not None:
                        metrics.acgtp_contact_ring_token_count = _ring_total
                    _gated_total = _contact_result_metrics.get("contact_ring_gated_token_count")
                    if _gated_total is not None:
                        metrics.acgtp_contact_ring_gated_token_count = _gated_total
                # Scene layout diagnostics from module result
                if _scene_result_metrics is not None:
                    metrics.acgtp_support_plane_token_count = _scene_result_metrics.get("support_plane_token_count")
                    metrics.acgtp_object_component_token_count = _scene_result_metrics.get("object_component_token_count")
                    metrics.acgtp_boundary_token_count = _scene_result_metrics.get("boundary_token_count")
                metrics.acgtp_scene_layout_score_mean = _sm.get("acgtp_scene_layout_score_mean")
                metrics.acgtp_scene_layout_score_max = _sm.get("acgtp_scene_layout_score_max")
                if _scene_result_metrics is not None:
                    _sp_cnt = _scene_result_metrics.get("support_plane_token_count")
                    if _sp_cnt is not None:
                        metrics.acgtp_support_plane_token_count = _sp_cnt
                    _oc_cnt = _scene_result_metrics.get("object_component_token_count")
                    if _oc_cnt is not None:
                        metrics.acgtp_object_component_token_count = _oc_cnt
                    _bn_cnt = _scene_result_metrics.get("boundary_token_count")
                    if _bn_cnt is not None:
                        metrics.acgtp_boundary_token_count = _bn_cnt
                # P6: support_plane cap diagnostics
                metrics.acgtp_support_plane_candidate_count = _sm.get("acgtp_support_plane_candidate_count")
                metrics.acgtp_scene_support_plane_cap_ratio = _sm.get("acgtp_scene_support_plane_cap_ratio")
                metrics.acgtp_scene_support_plane_cap_used = _sm.get("acgtp_scene_support_plane_cap_used")
                metrics.acgtp_scene_support_plane_fallback_used = _sm.get("acgtp_scene_support_plane_fallback_used")
                metrics.acgtp_scene_support_plane_fallback_reason = _sm.get("acgtp_scene_support_plane_fallback_reason")
                # P6: object_component fallback
                metrics.acgtp_scene_object_component_fallback_used = _sm.get("acgtp_scene_object_component_fallback_used")
                metrics.acgtp_scene_object_component_fallback_reason = _sm.get("acgtp_scene_object_component_fallback_reason")
                metrics.acgtp_scene_object_component_num_components = _sm.get("acgtp_scene_object_component_num_components")
                # P6: boundary fallback + source
                metrics.acgtp_scene_boundary_fallback_used = _sm.get("acgtp_scene_boundary_fallback_used")
                metrics.acgtp_scene_boundary_fallback_reason = _sm.get("acgtp_scene_boundary_fallback_reason")
                metrics.acgtp_scene_boundary_from_object_count = _sm.get("acgtp_scene_boundary_from_object_count")
                metrics.acgtp_scene_boundary_from_depth_count = _sm.get("acgtp_scene_boundary_from_depth_count")
                # P6: per-branch timing
                metrics.acgtp_scene_layout_ms = score_stats.get("acgtp_scene_layout_ms")
                metrics.acgtp_contact_ring_ms = score_stats.get("acgtp_contact_ring_ms")
                metrics.acgtp_motion_corridor_ms = score_stats.get("acgtp_motion_corridor_ms")
                metrics.acgtp_scene_fill_candidate_count = _sm.get("acgtp_scene_fill_candidate_count")
                metrics.acgtp_scene_fill_candidate_ratio = _sm.get("acgtp_scene_fill_candidate_ratio")
                # Scene layout per-component selected attribution (post-selection, from selector metadata)
                metrics.acgtp_scene_selected_support_plane_count = _sm.get("acgtp_scene_selected_support_plane_count")
                metrics.acgtp_scene_selected_object_component_count = _sm.get("acgtp_scene_selected_object_component_count")
                metrics.acgtp_scene_selected_boundary_count = _sm.get("acgtp_scene_selected_boundary_count")
                metrics.acgtp_scene_selected_relation_count = _sm.get("acgtp_scene_selected_relation_count")
                metrics.acgtp_scene_support_plane_selected_ratio = _sm.get("acgtp_scene_support_plane_selected_ratio")
                metrics.acgtp_motion_corridor_valid = _motion_valid
                metrics.acgtp_motion_corridor_score_mean = _sm.get("acgtp_motion_corridor_score_mean")
                metrics.acgtp_motion_corridor_score_max = _sm.get("acgtp_motion_corridor_score_max")
                # Motion corridor diagnostics from module result (not fake zeros)
                if _motion_result_metrics is not None:
                    metrics.acgtp_motion_corridor_length_m = _sm.get("acgtp_motion_corridor_length_m")
                    metrics.acgtp_motion_norm_m = _motion_result_metrics.get("motion_norm_m")
                    metrics.acgtp_motion_ema_alpha = float(self.config.acgtp_motion_ema_alpha)
                else:
                    metrics.acgtp_motion_corridor_length_m = _sm.get("acgtp_motion_corridor_length_m")
                    metrics.acgtp_motion_norm_m = _sm.get("acgtp_motion_norm_m")
                    metrics.acgtp_motion_ema_alpha = _sm.get("acgtp_motion_ema_alpha")
                metrics.acgtp_depth_structure_score_mean = _sm.get("acgtp_depth_structure_score_mean")
                metrics.acgtp_depth_structure_score_max = _sm.get("acgtp_depth_structure_score_max")
                metrics.acgtp_action_constraint_score_mean = _sm.get("acgtp_action_constraint_score_mean")
                metrics.acgtp_action_constraint_score_max = _sm.get("acgtp_action_constraint_score_max")
                metrics.acgtp_action_constraint_source = _sm.get("acgtp_action_constraint_source")
                metrics.acgtp_future_action_constraint_enabled = _sm.get("acgtp_future_action_constraint_enabled")
                metrics.acgtp_future_action_constraint_valid = _sm.get("acgtp_future_action_constraint_valid")
                metrics.acgtp_future_action_constraint_disabled_reason = _sm.get("acgtp_future_action_constraint_disabled_reason")
                metrics.acgtp_future_action_constraint_score_mean = _sm.get("acgtp_future_action_constraint_score_mean")
                metrics.acgtp_future_action_constraint_score_max = _sm.get("acgtp_future_action_constraint_score_max")
                metrics.acgtp_object_side_contact_score_mean = _sm.get("acgtp_object_side_contact_score_mean")
                metrics.acgtp_object_side_contact_score_max = _sm.get("acgtp_object_side_contact_score_max")
                metrics.acgtp_swept_motion_risk_score_mean = _sm.get("acgtp_swept_motion_risk_score_mean")
                metrics.acgtp_swept_motion_risk_score_max = _sm.get("acgtp_swept_motion_risk_score_max")
                metrics.acgtp_collision_contact_risk_score_mean = _sm.get("acgtp_collision_contact_risk_score_mean")
                metrics.acgtp_collision_contact_risk_score_max = _sm.get("acgtp_collision_contact_risk_score_max")
                metrics.acgtp_contact_object_overlap_count = _sm.get("acgtp_contact_object_overlap_count")
                metrics.acgtp_robot_self_penalty_count = _sm.get("acgtp_robot_self_penalty_count")
                metrics.acgtp_action_constraint_ms = score_stats.get("acgtp_action_constraint_ms")
                metrics.acgtp_hard_protect_count = _sm.get("acgtp_hard_protect_count")
                metrics.acgtp_hard_protect_ratio = _sm.get("acgtp_hard_protect_ratio")
                metrics.acgtp_hard_protect_valid = _sm.get("acgtp_hard_protect_valid")
                metrics.selected_by_scene_layout_count = _sm.get("selected_by_scene_layout_count")
                metrics.selected_by_depth_structure_count = _sm.get("selected_by_depth_structure_count")
                metrics.selected_by_contact_ring_count = _sm.get("selected_by_contact_ring_count")
                metrics.selected_by_motion_corridor_count = _sm.get("selected_by_motion_corridor_count")
                metrics.selected_by_constrained_fill_count = _sm.get("selected_by_constrained_fill_count")
                metrics.selected_by_acgtp_fallback_count = _sm.get("selected_by_acgtp_fallback_count")
                metrics.overlap_scene_depth_count = _sm.get("overlap_scene_depth_count")
                metrics.overlap_scene_contact_count = _sm.get("overlap_scene_contact_count")
                metrics.overlap_contact_motion_count = _sm.get("overlap_contact_motion_count")
                metrics.overlap_scene_motion_count = _sm.get("overlap_scene_motion_count")
                metrics.overlap_depth_contact_count = _sm.get("overlap_depth_contact_count")
                metrics.overlap_depth_motion_count = _sm.get("overlap_depth_motion_count")
                metrics.acgtp_branch_accounting_valid = _sm.get("acgtp_branch_accounting_valid")
                metrics.acgtp_branch_sum = _sm.get("acgtp_branch_sum")
                metrics.acgtp_branch_sum_error = _sm.get("acgtp_branch_sum_error")
                metrics.branch_accounting_valid = _sm.get("branch_accounting_valid")
                metrics.branch_sum_equals_kept = _sm.get("branch_sum_equals_kept")
                metrics.acgtp_hard_protect_ratio_config = _sm.get("acgtp_hard_protect_ratio_config")
                metrics.acgtp_fallback_used = _sm.get("acgtp_fallback_used")
                metrics.acgtp_fallback_reason = _sm.get("acgtp_fallback_reason")
                metrics.acgtp_motion_disabled_reason = _sm.get("acgtp_motion_disabled_reason")
                metrics.acgtp_constrained_fill_mask = _sm.get("acgtp_constrained_fill_mask")
                metrics.acgtp_final_kept = _sm.get("final_kept")
                metrics.acgtp_expected_kept = _sm.get("expected_kept")
                metrics.acgtp_actual_keep_ratio = _sm.get("keep_ratio_actual")

                self._record_selection_path_diagnostics(
                    metrics=metrics,
                    selector_success=True,
                    exc=None,
                    requested_strategy=self.config.strategy,
                    effective_strategy=self.config.strategy,
                    selector_name="select_acgtp_v1",
                    fallback_selector=None,
                    keep_indices_np=keep_indices_np,
                    num_tokens=num_tokens,
                    selection_meta=selection_meta,
                )
                _skip_postprocess_diag = True

            except Exception as exc:
                print(f"[PRUNING] select_acgtp_v1 exception (step={self._hook_step_counter}): {type(exc).__name__}: {exc}")
                metrics.fallback_used = True
                metrics.keep_ratio_source = "fallback"
                metrics.fallback_reason = f"acgtp_v1_error:{type(exc).__name__}"
                keep_indices_np, selection_meta = select_keep_indices(
                    strategy="depth_edge_fast_diverse",
                    num_tokens=num_tokens,
                    keep_count=keep_count,
                    scores=_de_scores if _de_scores is not None else scores,
                    valid_mask=valid_mask,
                    seed=self.config.seed,
                    grid_size=self.config.token_grid_shape[0],
                    cell_grid=self.config.cell_grid,
                    reserve_tokens=reserve_tokens,
                )
                self._record_selection_path_diagnostics(
                    metrics=metrics,
                    selector_success=False,
                    exc=exc,
                    requested_strategy=self.config.strategy,
                    effective_strategy="depth_edge_fast_diverse",
                    selector_name="select_acgtp_v1",
                    fallback_selector="depth_edge_fast_diverse",
                    keep_indices_np=None,
                    num_tokens=num_tokens,
                )
                metrics.acgtp_v1 = False
                metrics.acgtp_fallback_used = True
                metrics.acgtp_fallback_reason = f"acgtp_v1_error:{type(exc).__name__}"
                _skip_postprocess_diag = True

        # ── P16: robot_geo_acgtp_v2 — Task-Semantic Anchor Branch ──────────────────────
        elif _acgtp_strategy_check == "robot_geo_acgtp_v2" and keep_indices_np is None:
            # Build geometry branch scores (same as v1)
            _scene_scores = robot_metrics.get("acgtp_scene_layout_scores")
            _de_scores = robot_metrics.get("edge_scores")
            _contact_scores = robot_metrics.get("acgtp_contact_ring_scores")
            _motion_scores = robot_metrics.get("acgtp_motion_corridor_scores")
            _action_constraint_scores = robot_metrics.get("acgtp_action_constraint_scores")
            _motion_valid = bool(robot_metrics.get("acgtp_motion_corridor_valid", False))
            _fill_mask = robot_metrics.get("acgtp_constrained_fill_mask")
            _self_core = robot_metrics.get("acgtp_self_core_mask")
            _token_u = None
            _token_v = None
            if "cache" in robot_metrics and robot_metrics["cache"]:
                _token_u = robot_metrics["cache"].get("u")
                _token_v = robot_metrics["cache"].get("v")

            # Fallback: if scene_layout not computed, derive from depth distribution
            if _scene_scores is None:
                _depth_vals = robot_metrics.get("depth_metric")
                if _depth_vals is not None:
                    _scene_scores = np.clip((np.asarray(_depth_vals, dtype=np.float32).reshape(-1) - 0.3) / 1.7, 0.0, 1.0)
                else:
                    _scene_scores = np.zeros(num_tokens, dtype=np.float32)
            # Fallback: if contact_ring not available
            if _contact_scores is None:
                _near_s = robot_metrics.get("near_scores")
                if _near_s is not None:
                    _contact_scores = np.asarray(_near_s, dtype=np.float32).reshape(-1)
                else:
                    _contact_scores = np.zeros(num_tokens, dtype=np.float32)
            # Fallback: if motion_corridor not available
            if _motion_scores is None:
                _mc = robot_metrics.get("motion_cone_scores")
                if _mc is not None:
                    _motion_scores = np.asarray(_mc, dtype=np.float32).reshape(-1)
                else:
                    _motion_scores = np.zeros(num_tokens, dtype=np.float32)
            if _de_scores is None:
                _de_scores = robot_metrics.get("edge_scores")
            if _de_scores is None:
                _de_scores = np.zeros(num_tokens, dtype=np.float32)
            # Build constrained fill mask
            if _fill_mask is None:
                _depth_metric = robot_metrics.get("depth_metric")
                if _depth_metric is not None:
                    _dm = np.asarray(_depth_metric, dtype=np.float32).reshape(-1)
                    _scene_s = np.asarray(_scene_scores, dtype=np.float32).reshape(-1)
                    _fill_mask = (np.isfinite(_dm) & (_dm > 0.1) & (_dm < 5.0) & (_scene_s > 0.0)).astype(np.float32)
                else:
                    _fill_mask = np.ones(num_tokens, dtype=np.float32)
            # Self-core mask
            if _self_core is None:
                _gripper_pixel = robot_metrics.get("gripper_pixel")
                if _gripper_pixel is not None and _token_u is not None:
                    _gx, _gy = float(_gripper_pixel[0]), float(_gripper_pixel[1])
                    _u_a = np.asarray(_token_u, dtype=np.float32).reshape(-1)
                    _v_a = np.asarray(_token_v, dtype=np.float32).reshape(-1)
                    _dist2 = (_u_a - _gx) ** 2 + (_v_a - _gy) ** 2
                    _self_core = (np.sqrt(_dist2) <= float(self.config.acgtp_self_core_radius_px)).astype(np.float32)
                else:
                    _self_core = np.zeros(num_tokens, dtype=np.float32)

            # Full module results for diagnostics
            _contact_result_metrics = robot_metrics.get("acgtp_contact_ring_result")
            _motion_result_metrics = robot_metrics.get("acgtp_motion_corridor_result")
            _scene_result_metrics = robot_metrics.get("acgtp_scene_layout_result")
            _action_constraint_result_metrics = robot_metrics.get("acgtp_action_constraint_result")

            # ── Task-Semantic Anchor Branch ────────────────────────────────────
            # Get instruction string from the robot metrics or episode state
            _instruction = None
            # Try to get instruction from robot_metrics (populated by the eval pipeline)
            if robot_metrics is not None:
                _instruction = robot_metrics.get("task_instruction") or robot_metrics.get("task_name")
            # Fall back to the latest task_name from geometry_recorder if available
            if not _instruction:
                _latest = self.geometry_recorder.get_latest() if self.geometry_recorder else None
                if _latest is not None:
                    _instruction = getattr(_latest, "task_name", None)
            if not _instruction:
                _instruction = ""

            _semantic_active = bool(self.config.acgtp_v2_semantic_enabled) and str(self.config.acgtp_v2_semantic_backend) != "none"
            if _semantic_active:
                # Parse instruction terms for diagnostics only when a semantic backend is active.
                _parsed = parse_instruction_terms(_instruction)

                _token_depth = robot_metrics.get("depth_metric")
                _token_depth_arr = np.asarray(_token_depth, dtype=np.float32).reshape(-1) if _token_depth is not None else None

                _sem_result = compute_task_semantic_anchors(
                    instruction=_instruction,
                    rgb=None,  # RGB not yet wired in this pipeline
                    token_u=_token_u,
                    token_v=_token_v,
                    token_depth=_token_depth_arr,
                    scene_result=_scene_result_metrics,
                    config=self.config,
                    semantic_enabled=True,
                    semantic_backend=str(self.config.acgtp_v2_semantic_backend),
                    w_semantic_target=float(self.config.acgtp_v2_w_semantic_target),
                    w_semantic_reference=float(self.config.acgtp_v2_w_semantic_reference),
                    w_semantic_relation=float(self.config.acgtp_v2_w_semantic_relation),
                    w_semantic_goal=float(self.config.acgtp_v2_w_semantic_goal),
                    target_cap_ratio=float(self.config.acgtp_v2_target_cap_ratio),
                    reference_cap_ratio=float(self.config.acgtp_v2_reference_cap_ratio),
                    relation_cap_ratio=float(self.config.acgtp_v2_relation_cap_ratio),
                    hard_ratio=float(self.config.acgtp_v2_semantic_hard_ratio),
                    release_quota_when_unavailable=bool(self.config.acgtp_v2_release_semantic_quota_when_unavailable),
                    grid_h=self.config.token_grid_shape[0],
                    grid_w=self.config.token_grid_shape[1],
                )
            else:
                _parsed = {
                    "parsed_target_terms": [],
                    "parsed_reference_terms": [],
                    "parsed_relation_terms": [],
                    "instruction_is_meaningful": False,
                }
                _sem_result = {
                    "semantic_anchor_scores": None,
                    "semantic_target_scores": None,
                    "semantic_reference_scores": None,
                    "semantic_relation_scores": None,
                    "semantic_goal_scores": None,
                    "semantic_confidence": 0.0,
                    "semantic_unavailable": True,
                    "semantic_fallback_reason": "semantic_disabled_fast_path",
                    "scene_layout_branch_active": False,
                    "scene_layout_available": False,
                    "scene_layout_confidence": 0.0,
                    "target_mask_count": 0,
                    "reference_mask_count": 0,
                    "relation_mask_count": 0,
                    "layout_anchor_mask_count": 0,
                    "scene_layout_indices": [],
                }

            _hist_decision: Dict[str, Any] = {"acgtp_history_enabled": False}
            if bool(getattr(self.config, "acgtp_history_enabled", False)):
                try:
                    _hist_decision = self._acgtp_history.prepare_step(
                        scene_scores=_scene_scores,
                        depth_scores=_de_scores,
                        contact_scores=_contact_scores,
                        motion_scores=_motion_scores,
                        action_scores=_action_constraint_scores,
                        valid_mask=valid_mask,
                        num_tokens=num_tokens,
                        gripper_pos=robot_metrics.get("gripper_pos"),
                        depth_valid_ratio=robot_metrics.get("depth_valid_ratio"),
                    )
                    _scene_scores = _hist_decision.get("scene_scores", _scene_scores)
                    _de_scores = _hist_decision.get("depth_scores", _de_scores)
                    _contact_scores = _hist_decision.get("contact_scores", _contact_scores)
                    _motion_scores = _hist_decision.get("motion_scores", _motion_scores)
                    _action_constraint_scores = _hist_decision.get("action_scores", _action_constraint_scores)
                except Exception as _hist_exc:
                    _hist_decision = {
                        "acgtp_history_enabled": False,
                        "acgtp_history_disabled_reason": f"history_error:{type(_hist_exc).__name__}:{str(_hist_exc)[:120]}",
                    }

            _dyn_decision: Dict[str, Any] = {"acgtp_dynamic_enabled": False}
            _dyn_w_scene = float(self.config.acgtp_w_scene_layout)
            _dyn_w_depth = float(self.config.acgtp_w_depth_structure)
            _dyn_w_contact = float(self.config.acgtp_w_contact_ring)
            _dyn_w_motion = float(self.config.acgtp_w_motion_corridor)
            _dyn_hard_ratio = float(self.config.acgtp_hard_protect_ratio)
            if bool(getattr(self.config, "acgtp_dynamic_enabled", False)):
                try:
                    _motion_norm_for_dyn = None
                    if isinstance(_motion_result_metrics, dict):
                        _motion_norm_for_dyn = _motion_result_metrics.get("motion_norm_m")
                    _dyn_decision = decide_acgtp_dynamic_budget(
                        scene_layout_scores=_scene_scores,
                        depth_structure_scores=_de_scores,
                        contact_ring_scores=_contact_scores,
                        motion_corridor_scores=_motion_scores,
                        action_constraint_scores=_action_constraint_scores,
                        valid_mask=valid_mask,
                        constrained_fill_mask=_fill_mask,
                        num_tokens=num_tokens,
                        base_keep_ratio=float(self.config.keep_ratio),
                        previous_state=self._acgtp_dynamic_state,
                        motion_corridor_valid=bool(_motion_valid),
                        motion_norm_m=_motion_norm_for_dyn,
                        depth_valid_ratio=robot_metrics.get("depth_valid_ratio"),
                        min_keep_ratio=max(
                            float(self.config.acgtp_dynamic_min_keep_ratio),
                            (
                                float(self.config.keep_ratio)
                                if not bool(getattr(self.config, "acgtp_dynamic_allow_below_base_keep_ratio", False))
                                else float(self.config.acgtp_dynamic_min_keep_ratio)
                            ) + (
                                float(self.config.acgtp_history_conservative_keep_boost)
                                if bool(_hist_decision.get("acgtp_history_conservative_mode")) else 0.0
                            ),
                        ),
                        max_keep_ratio=float(self.config.acgtp_dynamic_max_keep_ratio),
                        risk_boost_scale=float(self.config.acgtp_dynamic_risk_boost_scale),
                        confidence_prune_scale=float(self.config.acgtp_dynamic_confidence_prune_scale),
                        contact_phase_gate=str(getattr(self.config, "acgtp_dynamic_contact_phase_gate", "legacy_peak")),
                        respect_phase_min_on_candidate_gap=bool(getattr(self.config, "acgtp_dynamic_respect_phase_min_on_candidate_gap", False)),
                        shadow_contact_guard_enabled=bool(getattr(self.config, "acgtp_dynamic_shadow_contact_guard_enabled", False)),
                        shadow_contact_depth_weight_floor=float(getattr(self.config, "acgtp_dynamic_shadow_contact_depth_weight_floor", 0.30)),
                        shadow_contact_contact_weight_floor=float(getattr(self.config, "acgtp_dynamic_shadow_contact_contact_weight_floor", 0.24)),
                        shadow_contact_hard_ratio_floor=float(getattr(self.config, "acgtp_dynamic_shadow_contact_hard_ratio_floor", 0.70)),
                    )
                    _state = _dyn_decision.pop("_state", None)
                    if isinstance(_state, dict):
                        self._acgtp_dynamic_state = _state
                    keep_count = max(1, min(num_tokens, int(_dyn_decision.get("acgtp_dynamic_keep_k", keep_count))))
                    _dyn_w_scene = float(_dyn_decision.get("acgtp_dynamic_scene_weight", _dyn_w_scene))
                    _dyn_w_depth = float(_dyn_decision.get("acgtp_dynamic_depth_weight", _dyn_w_depth))
                    _dyn_w_contact = float(_dyn_decision.get("acgtp_dynamic_contact_weight", _dyn_w_contact))
                    _dyn_w_motion = float(_dyn_decision.get("acgtp_dynamic_motion_weight", _dyn_w_motion))
                    _dyn_hard_ratio = float(_dyn_decision.get("acgtp_dynamic_hard_protect_ratio", _dyn_hard_ratio))
                    if bool(_hist_decision.get("acgtp_history_conservative_mode")):
                        _dyn_hard_ratio = min(0.82, _dyn_hard_ratio + float(self.config.acgtp_history_conservative_hard_boost))
                        _dyn_w_scene += 0.03
                        _dyn_w_depth += 0.03
                        _ws = max(1e-8, _dyn_w_scene + _dyn_w_depth + _dyn_w_contact + _dyn_w_motion)
                        _dyn_w_scene, _dyn_w_depth, _dyn_w_contact, _dyn_w_motion = [
                            float(x / _ws) for x in (_dyn_w_scene, _dyn_w_depth, _dyn_w_contact, _dyn_w_motion)
                        ]
                        _dyn_decision["acgtp_history_keep_boost_applied"] = True
                    else:
                        _dyn_decision["acgtp_history_keep_boost_applied"] = False
                    metrics.dynamic_enabled = True
                    metrics.dynamic_phase = _dyn_decision.get("acgtp_dynamic_phase")
                    metrics.dynamic_keep_ratio = _dyn_decision.get("acgtp_dynamic_keep_ratio")
                    metrics.dynamic_keep_k = keep_count
                    metrics.dynamic_keep_reason = _dyn_decision.get("acgtp_dynamic_keep_reason")
                    metrics.geo_risk_score = _dyn_decision.get("acgtp_dynamic_risk")
                    metrics.geo_risk_level = _dyn_decision.get("acgtp_dynamic_phase")
                    metrics.keep_ratio_source = "acgtp_dynamic_controller"
                    metrics.effective_keep_count = keep_count
                except Exception as _dyn_exc:
                    _dyn_decision = {
                        "acgtp_dynamic_enabled": False,
                        "acgtp_dynamic_disabled_reason": f"controller_error:{type(_dyn_exc).__name__}:{str(_dyn_exc)[:120]}",
                    }
                    metrics.dynamic_enabled = False
                    metrics.dynamic_keep_reason = _dyn_decision["acgtp_dynamic_disabled_reason"]

            _use_acgtp_fast_selector = (
                bool(getattr(self.config, "acgtp_fast_selector_enabled", True))
                and not bool(getattr(self.config, "acgtp_full_diagnostics_enabled", False))
                and not _semantic_active
            )
            _acgtp_selector_fn = select_acgtp_v2_fast if _use_acgtp_fast_selector else select_acgtp_v2

            try:
                keep_indices_np, selection_meta = _acgtp_selector_fn(
                    scene_layout_scores=_scene_scores,
                    depth_edge_scores=_de_scores,
                    contact_ring_scores=_contact_scores,
                    motion_corridor_scores=_motion_scores,
                    semantic_anchor_scores=_sem_result.get("semantic_anchor_scores"),
                    semantic_target_scores=_sem_result.get("semantic_target_scores"),
                    semantic_reference_scores=_sem_result.get("semantic_reference_scores"),
                    semantic_relation_scores=_sem_result.get("semantic_relation_scores"),
                    semantic_goal_scores=_sem_result.get("semantic_goal_scores"),
                    valid_mask=valid_mask,
                    keep_k=keep_count,
                    constrained_fill_mask=_fill_mask,
                    token_u=_token_u,
                    token_v=_token_v,
                    grid_h=self.config.token_grid_shape[0],
                    grid_w=self.config.token_grid_shape[1],
                    w_scene_layout=_dyn_w_scene,
                    w_depth_structure=_dyn_w_depth,
                    w_contact_ring=_dyn_w_contact,
                    w_motion_corridor=_dyn_w_motion,
                    w_semantic=0.20,
                    hard_protect_ratio=_dyn_hard_ratio,
                    motion_corridor_valid=_motion_valid,
                    self_core_mask=_self_core,
                    contact_ring_inner_px=float(self.config.acgtp_contact_ring_inner_px),
                    contact_ring_outer_px=float(self.config.acgtp_contact_ring_outer_px),
                    contact_requires_edge_or_object=bool(self.config.acgtp_contact_requires_edge_or_object),
                    depth_edge_score_for_gate=_de_scores,
                    _motion_result_for_diag=_motion_result_metrics,
                    _scene_result_for_diag=_scene_result_metrics,
                    action_constraint_scores=_action_constraint_scores,
                    _action_constraint_result_for_diag=_action_constraint_result_metrics,
                    support_plane_cap_ratio=float(self.config.acgtp_scene_support_plane_cap_ratio),
                    # ── P16 semantic params ──
                    semantic_enabled=bool(self.config.acgtp_v2_semantic_enabled) and _semantic_active,
                    semantic_backend=str(self.config.acgtp_v2_semantic_backend),
                    semantic_confidence=float(_sem_result.get("semantic_confidence", 0.0)),
                    semantic_unavailable=bool(_sem_result.get("semantic_unavailable", True)),
                    semantic_fallback_reason=_sem_result.get("semantic_fallback_reason"),
                    release_semantic_quota_when_unavailable=bool(self.config.acgtp_v2_release_semantic_quota_when_unavailable),
                    w_semantic_target=float(self.config.acgtp_v2_w_semantic_target),
                    w_semantic_reference=float(self.config.acgtp_v2_w_semantic_reference),
                    w_semantic_relation=float(self.config.acgtp_v2_w_semantic_relation),
                    w_semantic_goal=float(self.config.acgtp_v2_w_semantic_goal),
                    target_cap_ratio=float(self.config.acgtp_v2_target_cap_ratio),
                    reference_cap_ratio=float(self.config.acgtp_v2_reference_cap_ratio),
                    relation_cap_ratio=float(self.config.acgtp_v2_relation_cap_ratio),
                    hard_semantic_ratio=float(self.config.acgtp_v2_semantic_hard_ratio),
                    parsed_target_terms=_parsed.get("parsed_target_terms"),
                    parsed_reference_terms=_parsed.get("parsed_reference_terms"),
                    parsed_relation_terms=_parsed.get("parsed_relation_terms"),
                    instruction_is_meaningful=bool(_parsed.get("instruction_is_meaningful", False)),
                    # ── P16 scene-layout branch params (from semantic backend) ──
                    scene_layout_branch_active=bool(_sem_result.get("scene_layout_branch_active", False)),
                    scene_layout_available=bool(_sem_result.get("scene_layout_available", False)),
                    scene_layout_confidence=float(_sem_result.get("scene_layout_confidence", 0.0)),
                    target_mask_count=int(_sem_result.get("target_mask_count", 0)),
                    reference_mask_count=int(_sem_result.get("reference_mask_count", 0)),
                    relation_mask_count=int(_sem_result.get("relation_mask_count", 0)),
                    layout_anchor_mask_count=int(_sem_result.get("layout_anchor_mask_count", 0)),
                    scene_layout_indices=_sem_result.get("scene_layout_indices", []),
                    # ── P16 attention branch params (disabled in strict fallback; enabled when configured) ──
                    acgtp_attention_enabled=False,  # disabled in this pipeline; set True to enable
                    acgtp_attention_backend="none",
                    acgtp_attention_min_confidence=0.0,
                    acgtp_attention_requires_geometry_alignment=True,
                    acgtp_attention_budget_ratio=0.10,
                    acgtp_attention_task_relevance_score=None,
                    acgtp_attention_task_relevance_mask=None,
                    acgtp_attention_source="none",
                    acgtp_attention_available=False,
                    acgtp_attention_confidence=0.0,
                )

                # Record ACGTP-v2 metrics
                if isinstance(_hist_decision, dict):
                    selection_meta.update({k: v for k, v in _hist_decision.items() if not k.endswith("_scores") and k not in ("scene_scores", "depth_scores", "contact_scores", "motion_scores", "action_scores")})
                if isinstance(_dyn_decision, dict):
                    selection_meta.update(_dyn_decision)
                _sm = selection_meta
                for _dyn_key in (
                    "acgtp_history_enabled",
            "acgtp_history_length",
            "acgtp_history_capacity",
            "acgtp_history_length_after_update",
            "acgtp_history_warmup",
            "acgtp_history_ema_available",
            "acgtp_history_smoothing_applied",
            "acgtp_history_conservative_mode",
            "acgtp_history_conservative_reason",
            "acgtp_history_disabled_reason",
            "acgtp_history_depth_change",
            "acgtp_history_keep_mask_iou",
            "acgtp_history_motion_stability",
            "acgtp_history_depth_valid_ratio",
            "acgtp_history_scene_ema_alpha",
            "acgtp_history_depth_ema_alpha",
            "acgtp_history_contact_ema_alpha",
            "acgtp_history_motion_ema_alpha",
            "acgtp_history_action_ema_alpha",
            "acgtp_history_phase_switch",
            "acgtp_history_force_conservative_next",
            "acgtp_history_force_conservative_reason",
            "acgtp_history_keep_boost_applied",
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
                ):
                    if hasattr(metrics, _dyn_key):
                        setattr(metrics, _dyn_key, _sm.get(_dyn_key))
                metrics.acgtp_v2 = True
                metrics.acgtp_v1 = False
                metrics.acgtp_selector_version = _sm.get("acgtp_selector_version")
                metrics.acgtp_quota_policy = _sm.get("acgtp_quota_policy")
                metrics.acgtp_fill_policy = _sm.get("acgtp_fill_policy")
                metrics.acgtp_scene_quota = _sm.get("acgtp_scene_quota")
                metrics.acgtp_depth_quota = _sm.get("acgtp_depth_quota")
                metrics.acgtp_contact_quota = _sm.get("acgtp_contact_quota")
                metrics.acgtp_motion_quota = _sm.get("acgtp_motion_quota")
                metrics.acgtp_scene_quota_weight = _sm.get("acgtp_scene_quota_weight")
                metrics.acgtp_depth_quota_weight = _sm.get("acgtp_depth_quota_weight")
                metrics.acgtp_contact_quota_weight = _sm.get("acgtp_contact_quota_weight")
                metrics.acgtp_motion_quota_weight = _sm.get("acgtp_motion_quota_weight")
                metrics.acgtp_scene_allocated = _sm.get("acgtp_scene_allocated")
                metrics.acgtp_depth_allocated = _sm.get("acgtp_depth_allocated")
                metrics.acgtp_contact_allocated = _sm.get("acgtp_contact_allocated")
                metrics.acgtp_motion_allocated = _sm.get("acgtp_motion_allocated")
                metrics.acgtp_coverage_fill_candidate_count = _sm.get("acgtp_coverage_fill_candidate_count")
                metrics.acgtp_coverage_fill_candidate_ratio = _sm.get("acgtp_coverage_fill_candidate_ratio")
                metrics.acgtp_v2_semantic_enabled = _sm.get("acgtp_v2_semantic_enabled")
                metrics.acgtp_v2_semantic_backend = _sm.get("acgtp_v2_semantic_backend")
                metrics.acgtp_v2_semantic_confidence = _sm.get("acgtp_v2_semantic_confidence")
                metrics.acgtp_v2_semantic_unavailable = _sm.get("acgtp_v2_semantic_unavailable")
                metrics.acgtp_v2_semantic_fallback_reason = _sm.get("acgtp_v2_semantic_fallback_reason")
                metrics.acgtp_v2_release_quota = _sm.get("acgtp_v2_release_quota")
                metrics.acgtp_v2_parsed_instruction_meaningful = _sm.get("acgtp_v2_parsed_instruction_meaningful")
                metrics.acgtp_v2_parsed_target_terms = str(_parsed.get("parsed_target_terms", []))
                metrics.acgtp_v2_parsed_reference_terms = str(_parsed.get("parsed_reference_terms", []))
                metrics.acgtp_v2_parsed_relation_terms = str(_parsed.get("parsed_relation_terms", []))
                metrics.acgtp_v2_w_semantic_target = _sm.get("acgtp_v2_w_semantic_target")
                metrics.acgtp_v2_w_semantic_reference = _sm.get("acgtp_v2_w_semantic_reference")
                metrics.acgtp_v2_w_semantic_relation = _sm.get("acgtp_v2_w_semantic_relation")
                metrics.acgtp_v2_w_semantic_goal = _sm.get("acgtp_v2_w_semantic_goal")
                metrics.acgtp_v2_semantic_target_token_count = _sm.get("acgtp_v2_semantic_target_token_count")
                metrics.acgtp_v2_semantic_target_kept_count = _sm.get("acgtp_v2_semantic_target_token_count")
                metrics.acgtp_v2_semantic_reference_token_count = _sm.get("acgtp_v2_semantic_reference_token_count")
                metrics.acgtp_v2_semantic_relation_token_count = _sm.get("acgtp_v2_semantic_relation_token_count")
                metrics.acgtp_v2_semantic_goal_token_count = _sm.get("acgtp_v2_semantic_goal_token_count")
                metrics.acgtp_v2_semantic_anchor_token_count = _sm.get("acgtp_v2_semantic_anchor_token_count")
                metrics.selected_by_semantic_target_count = _sm.get("selected_by_semantic_target_count")
                metrics.selected_by_semantic_reference_count = _sm.get("selected_by_semantic_reference_count")
                metrics.selected_by_semantic_relation_count = _sm.get("selected_by_semantic_relation_count")
                metrics.selected_by_semantic_goal_count = _sm.get("selected_by_semantic_goal_count")
                metrics.selected_by_scene_residual_fill_count = _sm.get("selected_by_scene_residual_fill_count")
                metrics.semantic_overlap_with_scene_count = _sm.get("semantic_overlap_with_scene_count")
                metrics.semantic_overlap_with_depth_count = _sm.get("semantic_overlap_with_depth_count")
                metrics.semantic_overlap_with_contact_count = _sm.get("semantic_overlap_with_contact_count")
                metrics.semantic_overlap_with_motion_count = _sm.get("semantic_overlap_with_motion_count")
                metrics.acgtp_v2_hard_semantic_quota = _sm.get("acgtp_v2_hard_semantic_quota")
                metrics.acgtp_v2_target_cap_k = _sm.get("acgtp_v2_target_cap_k")
                metrics.acgtp_v2_reference_cap_k = _sm.get("acgtp_v2_reference_cap_k")
                metrics.acgtp_v2_relation_cap_k = _sm.get("acgtp_v2_relation_cap_k")
                metrics.acgtp_scene_selected_residual_fill_count = _sm.get("acgtp_scene_selected_residual_fill_count")
                metrics.acgtp_scene_residual_fill_token_count = _sm.get("acgtp_scene_residual_fill_token_count")
                metrics.acgtp_scene_residual_fill_token_count_computed = _sm.get("acgtp_scene_residual_fill_token_count_computed")
                metrics.selected_by_semantic_count = _sm.get("selected_by_semantic_count")

                # P16: Strict fallback dispatch fields
                metrics.strict_fallback_dispatch_used = _sm.get("strict_fallback_dispatch_used")
                metrics.delegated_selector_name = _sm.get("delegated_selector_name")
                metrics.fallback_dispatch_to_v1 = _sm.get("fallback_dispatch_to_v1")
                # P16: Semantic state aliases
                metrics.semantic_backend = _sm.get("acgtp_v2_semantic_backend")
                metrics.semantic_unavailable = _sm.get("semantic_unavailable")
                metrics.semantic_confidence = _sm.get("semantic_confidence")
                metrics.semantic_available = _sm.get("acgtp_v2_semantic_available")
                metrics.semantic_quota_released = _sm.get("acgtp_v2_release_quota")
                # P16: Attention state
                metrics.attention_backend = _sm.get("acgtp_attention_backend")
                metrics.attention_source = _sm.get("acgtp_attention_source")
                metrics.attention_available = _sm.get("acgtp_attention_available")
                metrics.attention_confidence = _sm.get("acgtp_attention_confidence")
                metrics.attention_quota_released = _sm.get("acgtp_attention_quota_released")
                metrics.selected_by_attention_count = _sm.get("attention_selected_by_final_count")
                metrics.attention_only_token_count = _sm.get("acgtp_attention_only_token_count")
                metrics.attention_selected_by_final_count = _sm.get("attention_selected_by_final_count")
                metrics.attention_top_count = _sm.get("acgtp_attention_top_count")

                # Re-use v1 geometry branch metrics from selection_meta
                metrics.acgtp_w_scene_layout = _sm.get("acgtp_w_scene_layout")
                metrics.acgtp_w_depth_structure = _sm.get("acgtp_w_depth_structure")
                metrics.acgtp_w_contact_ring = _sm.get("acgtp_w_contact_ring")
                metrics.acgtp_w_motion_corridor = _sm.get("acgtp_w_motion_corridor")
                metrics.acgtp_self_core_radius_px = _sm.get("acgtp_self_core_radius_px")
                metrics.acgtp_contact_ring_inner_px = _sm.get("acgtp_contact_ring_inner_px")
                metrics.acgtp_contact_ring_outer_px = _sm.get("acgtp_contact_ring_outer_px")
                metrics.acgtp_self_core_token_count = _sm.get("acgtp_self_core_token_count")
                metrics.acgtp_self_core_token_ratio = _sm.get("acgtp_self_core_token_ratio")
                metrics.acgtp_contact_ring_token_count = _sm.get("acgtp_contact_ring_token_count")
                metrics.acgtp_contact_ring_token_ratio = _sm.get("acgtp_contact_ring_token_ratio")
                metrics.acgtp_contact_ring_gated_token_count = _sm.get("acgtp_contact_ring_gated_token_count")
                metrics.acgtp_contact_ring_valid = _sm.get("acgtp_contact_ring_valid")
                metrics.acgtp_support_plane_token_count = _sm.get("acgtp_support_plane_token_count")
                metrics.acgtp_support_plane_candidate_count = _sm.get("acgtp_support_plane_candidate_count")
                metrics.acgtp_object_component_token_count = _sm.get("acgtp_object_component_token_count")
                metrics.acgtp_boundary_token_count = _sm.get("acgtp_boundary_token_count")
                metrics.acgtp_scene_fill_candidate_count = _sm.get("acgtp_scene_fill_candidate_count")
                metrics.acgtp_scene_fill_candidate_ratio = _sm.get("acgtp_scene_fill_candidate_ratio")
                metrics.acgtp_scene_support_plane_cap_ratio = _sm.get("acgtp_scene_support_plane_cap_ratio")
                metrics.acgtp_scene_support_plane_cap_used = _sm.get("acgtp_scene_support_plane_cap_used")
                metrics.acgtp_scene_support_plane_fallback_used = _sm.get("acgtp_scene_support_plane_fallback_used")
                metrics.acgtp_scene_support_plane_fallback_reason = _sm.get("acgtp_scene_support_plane_fallback_reason")
                metrics.acgtp_scene_object_component_fallback_used = _sm.get("acgtp_scene_object_component_fallback_used")
                metrics.acgtp_scene_object_component_fallback_reason = _sm.get("acgtp_scene_object_component_fallback_reason")
                metrics.acgtp_scene_object_component_num_components = _sm.get("acgtp_scene_object_component_num_components")
                metrics.acgtp_scene_boundary_fallback_used = _sm.get("acgtp_scene_boundary_fallback_used")
                metrics.acgtp_scene_boundary_fallback_reason = _sm.get("acgtp_scene_boundary_fallback_reason")
                metrics.acgtp_scene_boundary_from_object_count = _sm.get("acgtp_scene_boundary_from_object_count")
                metrics.acgtp_scene_boundary_from_depth_count = _sm.get("acgtp_scene_boundary_from_depth_count")
                metrics.acgtp_scene_selected_support_plane_count = _sm.get("acgtp_scene_selected_support_plane_count")
                metrics.acgtp_scene_selected_object_component_count = _sm.get("acgtp_scene_selected_object_component_count")
                metrics.acgtp_scene_selected_boundary_count = _sm.get("acgtp_scene_selected_boundary_count")
                metrics.acgtp_scene_selected_relation_count = _sm.get("acgtp_scene_selected_relation_count")
                metrics.acgtp_scene_support_plane_selected_ratio = _sm.get("acgtp_scene_support_plane_selected_ratio")
                metrics.acgtp_motion_corridor_valid = _motion_valid
                metrics.acgtp_motion_corridor_score_mean = _sm.get("acgtp_motion_corridor_score_mean")
                metrics.acgtp_motion_corridor_score_max = _sm.get("acgtp_motion_corridor_score_max")
                metrics.acgtp_motion_norm_m = _sm.get("acgtp_motion_norm_m")
                metrics.acgtp_motion_ema_alpha = _sm.get("acgtp_motion_ema_alpha")
                metrics.acgtp_depth_structure_score_mean = _sm.get("acgtp_depth_structure_score_mean")
                metrics.acgtp_depth_structure_score_max = _sm.get("acgtp_depth_structure_score_max")
                metrics.acgtp_action_constraint_score_mean = _sm.get("acgtp_action_constraint_score_mean")
                metrics.acgtp_action_constraint_score_max = _sm.get("acgtp_action_constraint_score_max")
                metrics.acgtp_action_constraint_source = _sm.get("acgtp_action_constraint_source")
                metrics.acgtp_future_action_constraint_enabled = _sm.get("acgtp_future_action_constraint_enabled")
                metrics.acgtp_future_action_constraint_valid = _sm.get("acgtp_future_action_constraint_valid")
                metrics.acgtp_future_action_constraint_disabled_reason = _sm.get("acgtp_future_action_constraint_disabled_reason")
                metrics.acgtp_future_action_constraint_score_mean = _sm.get("acgtp_future_action_constraint_score_mean")
                metrics.acgtp_future_action_constraint_score_max = _sm.get("acgtp_future_action_constraint_score_max")
                metrics.acgtp_object_side_contact_score_mean = _sm.get("acgtp_object_side_contact_score_mean")
                metrics.acgtp_object_side_contact_score_max = _sm.get("acgtp_object_side_contact_score_max")
                metrics.acgtp_swept_motion_risk_score_mean = _sm.get("acgtp_swept_motion_risk_score_mean")
                metrics.acgtp_swept_motion_risk_score_max = _sm.get("acgtp_swept_motion_risk_score_max")
                metrics.acgtp_collision_contact_risk_score_mean = _sm.get("acgtp_collision_contact_risk_score_mean")
                metrics.acgtp_collision_contact_risk_score_max = _sm.get("acgtp_collision_contact_risk_score_max")
                metrics.acgtp_contact_object_overlap_count = _sm.get("acgtp_contact_object_overlap_count")
                metrics.acgtp_robot_self_penalty_count = _sm.get("acgtp_robot_self_penalty_count")
                metrics.acgtp_action_constraint_ms = robot_metrics.get("acgtp_action_constraint_ms")
                metrics.acgtp_hard_protect_count = _sm.get("acgtp_hard_protect_count")
                metrics.acgtp_hard_protect_ratio = _sm.get("acgtp_hard_protect_ratio")
                metrics.acgtp_hard_protect_valid = _sm.get("acgtp_hard_protect_valid")
                metrics.selected_by_scene_layout_count = _sm.get("selected_by_scene_layout_count")
                metrics.selected_by_depth_structure_count = _sm.get("selected_by_depth_structure_count")
                metrics.selected_by_contact_ring_count = _sm.get("selected_by_contact_ring_count")
                metrics.selected_by_motion_corridor_count = _sm.get("selected_by_motion_corridor_count")
                metrics.selected_by_constrained_fill_count = _sm.get("selected_by_constrained_fill_count")
                metrics.selected_by_acgtp_fallback_count = _sm.get("selected_by_acgtp_fallback_count")
                metrics.overlap_scene_depth_count = _sm.get("overlap_scene_depth_count")
                metrics.overlap_scene_contact_count = _sm.get("overlap_scene_contact_count")
                metrics.overlap_contact_motion_count = _sm.get("overlap_contact_motion_count")
                metrics.overlap_scene_motion_count = _sm.get("overlap_scene_motion_count")
                metrics.overlap_depth_contact_count = _sm.get("overlap_depth_contact_count")
                metrics.overlap_depth_motion_count = _sm.get("overlap_depth_motion_count")
                metrics.acgtp_branch_accounting_valid = _sm.get("acgtp_branch_accounting_valid")
                metrics.acgtp_branch_sum = _sm.get("acgtp_branch_sum")
                metrics.acgtp_branch_sum_error = _sm.get("acgtp_branch_sum_error")
                metrics.branch_accounting_valid = _sm.get("branch_accounting_valid")
                metrics.branch_sum_equals_kept = _sm.get("branch_sum_equals_kept")
                metrics.acgtp_hard_protect_ratio_config = _sm.get("acgtp_hard_protect_ratio")
                metrics.acgtp_fallback_used = _sm.get("acgtp_fallback_used")
                metrics.acgtp_fallback_reason = _sm.get("acgtp_fallback_reason")
                metrics.acgtp_motion_disabled_reason = _sm.get("acgtp_motion_disabled_reason")
                metrics.acgtp_constrained_fill_mask = _sm.get("acgtp_constrained_fill_mask")
                metrics.acgtp_final_kept = _sm.get("final_kept")
                metrics.acgtp_expected_kept = _sm.get("expected_kept")
                metrics.acgtp_actual_keep_ratio = _sm.get("keep_ratio_actual")

                # Gripper pixel from contact ring result
                if _contact_result_metrics is not None:
                    metrics.gripper_pixel_u = _contact_result_metrics.get("gripper_pixel_u")
                    metrics.gripper_pixel_v = _contact_result_metrics.get("gripper_pixel_v")
                    metrics.gripper_in_bounds = _contact_result_metrics.get("gripper_in_bounds")

                self._record_selection_path_diagnostics(
                    metrics=metrics,
                    selector_success=True,
                    exc=None,
                    requested_strategy=self.config.strategy,
                    effective_strategy=self.config.strategy,
                    selector_name=_sm.get("selector_function_name", "select_acgtp_v2"),
                    fallback_selector=None,
                    keep_indices_np=keep_indices_np,
                    num_tokens=num_tokens,
                    selection_meta=selection_meta,
                )
                _skip_postprocess_diag = True

            except Exception as exc:
                print(f"[PRUNING] {_acgtp_selector_fn.__name__} exception (step={self._hook_step_counter}): {type(exc).__name__}: {exc}")
                metrics.fallback_used = True
                metrics.keep_ratio_source = "fallback"
                metrics.fallback_reason = f"acgtp_v2_error:{type(exc).__name__}"
                keep_indices_np, selection_meta = select_keep_indices(
                    strategy="depth_edge_fast_diverse",
                    num_tokens=num_tokens,
                    keep_count=keep_count,
                    scores=_de_scores if _de_scores is not None else scores,
                    valid_mask=valid_mask,
                    seed=self.config.seed,
                    grid_size=self.config.token_grid_shape[0],
                    cell_grid=self.config.cell_grid,
                    reserve_tokens=reserve_tokens,
                )
                self._record_selection_path_diagnostics(
                    metrics=metrics,
                    selector_success=False,
                    exc=exc,
                    requested_strategy=self.config.strategy,
                    effective_strategy="depth_edge_fast_diverse",
                    selector_name=_acgtp_selector_fn.__name__,
                    fallback_selector="depth_edge_fast_diverse",
                    keep_indices_np=None,
                    num_tokens=num_tokens,
                )
                metrics.acgtp_v2 = False
                metrics.acgtp_fallback_used = True
        # for real exceptional paths, not for normal random/uniform/depth-edge use.
        # IMPORTANT: hybrid_budget_v2 and robot_geo_acgtp_v1 handle their own return path above (try/except block).
        if keep_indices_np is None and self.config.strategy not in SELF_HANDLED_SELECTOR_STRATEGIES:
            keep_indices_np, selection_meta = select_keep_indices(
                strategy=strategy,
                num_tokens=num_tokens,
                keep_count=keep_count,
                scores=scores,
                valid_mask=valid_mask,
                seed=self.config.seed,
                grid_size=self.config.token_grid_shape[0],
                cell_grid=self.config.cell_grid,
                reserve_tokens=reserve_tokens,
            )
            selection_meta.setdefault("fallback_used", bool(metrics.fallback_used))
            selection_meta.setdefault("fallback_reason", metrics.fallback_reason)
        # ~L871: Final validation block start
        # Check the local fallback_reason AND metrics.fallback_reason (some branches
        # set metrics.fallback_reason but may not have set the local variable).
        # NOTE: this None safety net must run BEFORE the effective_keep_count
        # setdefault below, otherwise int(len(keep_indices_np)) raises on the
        # legacy SELF_HANDLED paths whose selector threw and left it None.
        _existing_reason = fallback_reason or metrics.fallback_reason
        if keep_indices_np is None:
            _fallback_msg = _existing_reason or f"safe_fallback_uninitialized_strategy:{self.config.strategy}"
            if self.config.debug:
                print(f"[PRUNING] WARNING: keep_indices_np was None for strategy={self.config.strategy}, using safe fallback")
            if not metrics.fallback_used:
                metrics.fallback_used = True
            metrics.keep_ratio_source = "fallback"
            metrics.fallback_reason = _fallback_msg
            strategy = "depth_edge_fast" if scores is None else "depth_edge_fast_diverse"
            keep_count_safe = int(round(num_tokens * 0.75))
            keep_indices_np, selection_meta = select_keep_indices(
                strategy=strategy,
                num_tokens=num_tokens,
                keep_count=keep_count_safe,
                scores=scores,
                valid_mask=valid_mask,
                seed=self.config.seed,
                grid_size=self.config.token_grid_shape[0],
                cell_grid=self.config.cell_grid,
                reserve_tokens=reserve_tokens,
            )
            selection_meta["fallback_used"] = True
            selection_meta["fallback_reason"] = _fallback_msg
            selection_meta["actual_keep_ratio"] = len(keep_indices_np) / num_tokens if num_tokens else 0.0
            selection_meta["ema_used_for_selection"] = False
            selection_meta["interaction_lock"] = False
            selection_meta["lock_reason"] = _fallback_msg
        try:
            selection_meta.setdefault("requested_keep_ratio", float(self.config.keep_ratio))
            selection_meta.setdefault("keep_ratio_source", metrics.keep_ratio_source or self._default_keep_ratio_source())
            selection_meta.setdefault("effective_keep_count", int(len(keep_indices_np)))
            selection_meta.setdefault("original_token_count", int(num_tokens))
        except Exception as _setdefault_exc:
            print(f"[PRUNING] step={self._hook_step_counter} setdefault EXCEPTION: {_setdefault_exc}")
            raise

        # Ensure we never use stale indices from a previous call
        if len(keep_indices_np) == 0:
            _fallback_empty = _existing_reason or "safe_fallback_empty_indices"
            if not metrics.fallback_used:
                metrics.fallback_used = True
            metrics.keep_ratio_source = "fallback"
            metrics.fallback_reason = _fallback_empty
            strategy = "depth_edge_fast"
            keep_count_safe = int(round(num_tokens * 0.75))
            keep_indices_np, selection_meta = select_keep_indices(
                strategy=strategy,
                num_tokens=num_tokens,
                keep_count=keep_count_safe,
                scores=scores,
                valid_mask=valid_mask,
                seed=self.config.seed,
                grid_size=self.config.token_grid_shape[0],
                cell_grid=self.config.cell_grid,
                reserve_tokens=reserve_tokens,
            )
            selection_meta["fallback_used"] = True
            selection_meta["fallback_reason"] = _fallback_empty
            selection_meta["actual_keep_ratio"] = len(keep_indices_np) / num_tokens if num_tokens else 0.0

        # Always ensure sorted, unique indices
        keep_indices_np = np.unique(keep_indices_np).astype(np.int64)
        if keep_indices_np.ndim == 0:
            keep_indices_np = np.array([int(keep_indices_np)], dtype=np.int64)
        # Clamp to valid range
        keep_indices_np = keep_indices_np[(keep_indices_np >= 0) & (keep_indices_np < num_tokens)]
        # Ensure we have at least 1 token
        if len(keep_indices_np) == 0:
            keep_indices_np = np.array([0], dtype=np.int64)
        # Sort and deduplicate
        keep_indices_np = np.sort(np.unique(keep_indices_np))
        # ── actual_keep_ratio must come from final len / num_tokens, NOT config ─
        _final_actual_keep_ratio = len(keep_indices_np) / num_tokens if num_tokens else 1.0
        selection_meta["actual_keep_ratio"] = _final_actual_keep_ratio
        selection_meta.setdefault("requested_keep_ratio", float(self.config.keep_ratio))
        selection_meta.setdefault("keep_ratio_source", metrics.keep_ratio_source or self._default_keep_ratio_source())
        selection_meta.setdefault("effective_keep_count", int(len(keep_indices_np)))
        selection_meta.setdefault("original_token_count", int(num_tokens))
        metrics.actual_keep_ratio = _final_actual_keep_ratio
        metrics.effective_keep_count = int(len(keep_indices_np))
        metrics.original_token_count = int(num_tokens)

        # Stage X/P6: normalize selector diagnostics once on the common path.
        # Inline selector calls may have recorded older fields already; this
        # common pass fills missing fields without using one coarse guard.
        _sel_success = keep_indices_np is not None
        # ── hybrid_budget_v2: phase accounting already set in its branch; skip
        # post-processing _record_selection_path_diagnostics to avoid overwriting it
        if _skip_postprocess_diag:
            _skip_postprocess_diag = False
            # Still need to finalize scores and return
            pass
        else:
            _sel_exc = getattr(self, "_last_selector_exception", None)
            _sel_req = self.config.strategy
            _sel_eff = "depth_edge_fast_diverse" if (fallback_reason or getattr(metrics, "fallback_reason", None) or "").startswith("safe_fallback") else _sel_req
            _sel_name = (
                selection_meta.get("selector_function_name")
                or metrics.selector_function_name
                or getattr(self, "_last_selector_name", None)
                or "select_keep_indices"
            )
            _sel_fb = "depth_edge_fast_diverse" if _sel_req != _sel_eff else None
            if metrics.fallback_used and _sel_fb is None and _sel_eff != "depth_edge_fast_diverse":
                _sel_fb = selection_meta.get("fallback_selector_name")
            selection_meta = finalize_selection_debug_info(
                selection_meta,
                selector_function_name=str(_sel_name),
                strategy=str(selection_meta.get("strategy") or _sel_eff),
                keep_indices=keep_indices_np,
                num_tokens=num_tokens,
                keep_count=keep_count,
                scores=scores,
                requested_keep_ratio=float(self.config.keep_ratio),
                fallback_used=bool(metrics.fallback_used or selection_meta.get("fallback_used")),
                fallback_reason=metrics.fallback_reason or selection_meta.get("fallback_reason"),
                selection_error=selection_meta.get("selection_error"),
                selection_warning=selection_meta.get("selection_warning"),
            )
            self._record_selection_path_diagnostics(
                metrics=metrics,
                selector_success=_sel_success,
                exc=_sel_exc,
                requested_strategy=_sel_req,
                effective_strategy=str(selection_meta.get("strategy") or _sel_eff),
                selector_name=str(_sel_name),
                fallback_selector=_sel_fb or selection_meta.get("fallback_selector_name"),
                keep_indices_np=keep_indices_np,
                num_tokens=num_tokens,
                selection_meta=selection_meta,
            )
        # Reset per-step state
        if self.config.strategy == "robot_geo_acgtp_v2" and "robot_metrics" in locals() and not metrics.fallback_used:
            _hist_update = self._acgtp_history.update_after_selection(
                keep_indices=keep_indices_np,
                num_tokens=num_tokens,
                dynamic_decision=selection_meta,
                gripper_pos=robot_metrics.get("gripper_pos"),
            )
            selection_meta.update(_hist_update)
            for _hist_key in (
                "acgtp_history_enabled",
                "acgtp_history_length",
                "acgtp_history_capacity",
                "acgtp_history_length_after_update",
                "acgtp_history_warmup",
                "acgtp_history_ema_available",
                "acgtp_history_smoothing_applied",
                "acgtp_history_conservative_mode",
                "acgtp_history_conservative_reason",
                "acgtp_history_disabled_reason",
                "acgtp_history_depth_change",
                "acgtp_history_keep_mask_iou",
                "acgtp_history_motion_stability",
                "acgtp_history_depth_valid_ratio",
                "acgtp_history_scene_ema_alpha",
                "acgtp_history_depth_ema_alpha",
                "acgtp_history_contact_ema_alpha",
                "acgtp_history_motion_ema_alpha",
                "acgtp_history_action_ema_alpha",
                "acgtp_history_phase_switch",
                "acgtp_history_force_conservative_next",
                "acgtp_history_force_conservative_reason",
                "acgtp_history_keep_boost_applied",
            ):
                if hasattr(metrics, _hist_key):
                    setattr(metrics, _hist_key, selection_meta.get(_hist_key))
        self._last_selector_exception = None
        self._last_selector_name = None

        # selection_meta already set inside hybrid branches above; no redundant re-init needed
        if keep_indices_np is None and self.config.strategy not in (SELF_HANDLED_SELECTOR_STRATEGIES - {"robot_geo_branch_budget_v0", "robot_geo_acgtp_v2"}):
            keep_indices_np, selection_meta = select_keep_indices(
                strategy=strategy,
                num_tokens=num_tokens,
                keep_count=keep_count,
                scores=scores,
                valid_mask=valid_mask,
                seed=self.config.seed,
                grid_size=self.config.token_grid_shape[0],
                cell_grid=self.config.cell_grid,
                reserve_tokens=reserve_tokens,
            )
        metrics.timing.selection_ms = (time.perf_counter() - select_start) * 1000.0
        metrics.timing.pruning_time_ms = metrics.timing.selection_ms
        metrics.timing.topk_pruning_ms = metrics.timing.selection_ms
        if self.config.detailed_pruning_timing:
            metrics.timing.edge_selection_ms = selection_meta.get("edge_selection_ms")
            metrics.timing.geo_selection_ms = selection_meta.get("geo_selection_ms")
            metrics.timing.diverse_selection_ms = selection_meta.get("diverse_selection_ms")
            metrics.timing.final_merge_ms = selection_meta.get("final_merge_ms")

        validation = validate_keep_indices(keep_indices_np, len(keep_indices_np))
        metrics.keep_indices_sorted = validation["keep_indices_sorted"]
        metrics.duplicate_indices_count = validation["duplicate_indices_count"]
        _apply_aux = dict(active_aux_metrics)
        _apply_aux["num_tokens"] = num_tokens
        # P1-1: Record attribution_missing_reason for fallback path (robot_metrics not populated)
        if not _apply_aux:
            _apply_aux["attribution_missing_reason"] = (
                "hybrid_temporal_v1_fallback:robot_metrics_not_populated;"
                "edge_scores_unavailable_for_attribution"
            )
        self._apply_selection_metrics(metrics, selection_meta, keep_indices_np, _apply_aux)
        if self.config.strategy == "robot_geo_temporal_v0" and "robot_metrics" in locals() and not metrics.fallback_used:
            self._update_temporal_history_after_selection(
                keep_indices_np=keep_indices_np,
                num_tokens=num_tokens,
                scores=scores,
                robot_metrics=robot_metrics,
                dynamic_keep_ratio=metrics.dynamic_keep_ratio,
            )
        if self.config.strategy == "robot_geo_hybrid_temporal_v1" and "robot_metrics" in locals() and not metrics.fallback_used:
            self._update_temporal_history_after_selection(
                keep_indices_np=keep_indices_np,
                num_tokens=num_tokens,
                scores=scores,
                robot_metrics=robot_metrics,
                dynamic_keep_ratio=metrics.dynamic_keep_ratio,
            )
        if self.config.strategy == "robot_geo_hybrid_temporal_edge_reserve_v1" and "robot_metrics" in locals() and not metrics.fallback_used:
            self._update_temporal_history_after_selection(
                keep_indices_np=keep_indices_np,
                num_tokens=num_tokens,
                scores=scores,
                robot_metrics=robot_metrics,
                dynamic_keep_ratio=metrics.dynamic_keep_ratio,
            )

        internal_mode = self._internal_pruning_requested()
        if internal_mode:
            self._prepare_internal_pruning_plan(
                keep_indices_np=keep_indices_np,
                num_tokens=int(num_tokens),
                metrics=metrics,
                selection_meta=selection_meta,
                target_keep_ratio=float(selection_meta.get("acgtp_dynamic_keep_ratio") or selection_meta.get("requested_keep_ratio") or (len(keep_indices_np) / max(1, int(num_tokens)))),
            )
            metrics.timing.gather_ms = 0.0
            pruned = visual_tokens
            kept = int(np.asarray(keep_indices_np, dtype=np.int64).reshape(-1).size)
        else:
            gather_start = time.perf_counter()
            keep_indices = torch.as_tensor(keep_indices_np, dtype=torch.long, device=visual_tokens.device)
            pruned = visual_tokens.index_select(dim=1, index=keep_indices)
            metrics.timing.gather_ms = (time.perf_counter() - gather_start) * 1000.0
            kept = int(pruned.shape[1])
            metrics.compression_backend = "projector"
            metrics.projector_pruning_applied = bool(kept < int(num_tokens))
            metrics.internal_pruning_requested = False
            metrics.internal_pruning_plan_ready = False
            metrics.internal_pruning_applied = False

        metrics.num_visual_tokens_kept = kept
        metrics.num_visual_tokens_pruned = num_tokens - kept
        # P1: basic token count attribution fields
        metrics.selected_token_count = kept
        metrics.dropped_token_count = num_tokens - kept
        metrics.selected_token_ratio = kept / num_tokens if num_tokens else None
        metrics.dropped_token_ratio = (num_tokens - kept) / num_tokens if num_tokens else None
        metrics.retention_ratio = kept / num_tokens if num_tokens else None
        metrics.keep_ratio = kept / num_tokens if num_tokens else None
        metrics.actual_keep_ratio = metrics.keep_ratio
        metrics.keep_ratio_actual = metrics.keep_ratio
        metrics.retention_actual = metrics.keep_ratio
        metrics.effective_keep_count = kept
        metrics.original_token_count = num_tokens
        metrics.num_visual_tokens_dropped = num_tokens - kept
        metrics.num_visual_tokens_original_total = num_tokens
        metrics.num_visual_tokens_kept_total = kept
        if not internal_mode:
            self._prepare_position_preserve_info(
                keep_indices_np=keep_indices_np,
                num_tokens=int(num_tokens),
                metrics=metrics,
            )
        if metrics.requested_keep_ratio is None:
            metrics.requested_keep_ratio = float(self.config.keep_ratio)
        if metrics.keep_ratio_requested is None:
            metrics.keep_ratio_requested = metrics.requested_keep_ratio
        if metrics.keep_ratio_source is None:
            metrics.keep_ratio_source = self._default_keep_ratio_source()
        metrics.protected_token_ratio = metrics.keep_ratio
        metrics.pruned_token_ratio = 1.0 - metrics.keep_ratio if metrics.keep_ratio is not None else None
        metrics.selected_token_count_equals_kept = bool(metrics.selected_token_count == metrics.num_visual_tokens_kept)
        metrics.retention_ratio_valid = bool(metrics.retention_ratio == metrics.keep_ratio)
        metrics.pruning_result = {
            "num_tokens_before": num_tokens,
            "num_tokens_after": kept,
            "actual_keep_ratio": metrics.keep_ratio,
            "method": self.config.strategy,
            "effective_strategy": strategy,
            "selection_metadata": selection_meta,
            "compression_backend": metrics.compression_backend,
            "projector_pruning_applied": metrics.projector_pruning_applied,
            "internal_pruning_requested": metrics.internal_pruning_requested,
            "internal_pruning_plan_ready": metrics.internal_pruning_plan_ready,
            "fallback_used": metrics.fallback_used,
            "dynamic_phase": metrics.dynamic_phase,
            "dynamic_keep_k": metrics.dynamic_keep_k,
            "geo_risk_level": metrics.geo_risk_level,
            "geo_risk_score": metrics.geo_risk_score,
            "dynamic_keep_reason": metrics.dynamic_keep_reason,
            "num_high_contact_tokens": metrics.num_high_contact_tokens,
            "num_valid_3d_tokens": metrics.num_valid_3d_tokens,
            "interaction_lock": metrics.interaction_lock,
            "temporal_stability": metrics.temporal_stability,
            "history_length": metrics.history_length,
            "score_ema_enabled": metrics.score_ema_enabled,
            "score_ema_available": getattr(metrics, "score_ema_available", None),
            "ema_used_for_selection": getattr(metrics, "ema_used_for_selection", None),
            "lock_condition_failed_reason": getattr(metrics, "lock_condition_failed_reason", None),
            "keep_ratio_source": metrics.keep_ratio_source,
            "requested_keep_ratio": metrics.requested_keep_ratio,
            "effective_keep_count": metrics.effective_keep_count,
            "original_token_count": metrics.original_token_count,
            "adaptive_threshold_mean": getattr(metrics, "adaptive_threshold_mean", None),
            "adaptive_threshold_max": getattr(metrics, "adaptive_threshold_max", None),
            "interaction_lock_ratio": getattr(metrics, "interaction_lock_ratio", None),
            "interaction_lock_reason": getattr(metrics, "interaction_lock_reason", None),
            "selected_grid_coverage_ratio": getattr(metrics, "selected_grid_coverage_ratio", None),
            "grid_coverage_ratio": getattr(metrics, "grid_coverage_ratio", None),
            "final_hybrid_score_mean": getattr(metrics, "final_hybrid_score_mean", None),
            "final_hybrid_score_max": getattr(metrics, "final_hybrid_score_max", None),
        }
        if self.cfg.get("save_pruning_vis", False) or self.cfg.get("save_pruning_debug", False):
            vis_metrics = active_aux_metrics
            vis_path = self._maybe_save_visualization(
                keep_indices_np=keep_indices_np,
                scores=scores,
                aux_metrics=vis_metrics,
                selection_meta=selection_meta,
            )
            if vis_path is not None:
                metrics.pruning_result["visualization_dir"] = vis_path

        # P1: Token selection debug visualization (always for first 3 steps per method)
        _do_debug_vis = (
            self.cfg.get("save_token_selection_debug", False)
            and self._geo_debug_frames_saved < 3
            and self.config.strategy in ("robot_geo_hybrid_temporal_v1", "robot_geo_hybrid_temporal_edge_reserve_v1", "robot_geo_hybrid_v1", "depth_edge_fast")
        )
        if _do_debug_vis:
            debug_metrics = active_aux_metrics
            debug_path = self._save_token_selection_debug_visualization(
                keep_indices_np=keep_indices_np,
                scores=scores,
                aux_metrics=debug_metrics,
                selection_meta=selection_meta,
            )
            if debug_path is not None:
                self._geo_debug_frames_saved += 1
                metrics.pruning_result["token_selection_debug_dir"] = debug_path

        # P8: Enhanced dropped-token overlay (visualization-only, does not change selection)
        # Uses a SEPARATE counter so original + dropped overlays each get steps 0/1/2.
        _do_dropped_vis = (
            self.cfg.get("save_token_selection_debug", False)
            and self._dropped_overlay_frames_saved < 3
            and self.config.strategy in TOKEN_SELECTION_DEBUG_STRATEGIES
        )
        if _do_dropped_vis:
            debug_metrics2 = active_aux_metrics
            dropped_path = self._save_dropped_token_debug_visualization(
                keep_indices_np=keep_indices_np,
                aux_metrics=debug_metrics2,
                selection_meta=selection_meta,
                num_tokens=num_tokens,
            )
            if dropped_path is not None:
                self._dropped_overlay_frames_saved += 1
                metrics.pruning_result["token_selection_debug_dropped_dir"] = dropped_path

        if self.config.enable_geo_debug:
            debug_metrics = active_aux_metrics
            debug_path = self._maybe_save_geo_debug_visualization(
                keep_indices_np=keep_indices_np,
                scores=scores,
                aux_metrics=debug_metrics,
            )
            if debug_path is not None:
                metrics.pruning_result["geo_debug_dir"] = debug_path
        if metrics.d_min is not None and self.config.strategy in ("robot_geo_near", "robot_geo_corridor", "robot_geo_contact_budget", "robot_geo_rule_v0", "robot_geo_dynamic_v0", "robot_geo_temporal_v0", "robot_geo_dynamic"):
            distances = active_aux_metrics.get("distances")
            if distances is not None:
                d = np.asarray(distances, dtype=np.float32)
                valid_d = d[keep_indices_np[np.isfinite(d[keep_indices_np])]]
                metrics.d_mean_topk = float(np.mean(valid_d)) if valid_d.size else None
            corridor_distances = active_aux_metrics.get("corridor_distances")
            if corridor_distances is not None:
                cd = np.asarray(corridor_distances, dtype=np.float32)
                valid_cd = cd[keep_indices_np[np.isfinite(cd[keep_indices_np])]]
                metrics.d_corridor_mean_topk = float(np.mean(valid_cd)) if valid_cd.size else None
            corridor_scores = active_aux_metrics.get("corridor_scores")
            if corridor_scores is not None:
                cs = np.asarray(corridor_scores, dtype=np.float32)
                metrics.corridor_strength_topk_mean = float(np.mean(cs[keep_indices_np])) if keep_indices_np.size else None
            self._latest_stats = metrics.to_eval_stats()
        # --- P1-1 debug: print attribution_missing_reason if set ---
        if getattr(metrics, "attribution_missing_reason", None):
            print(f"[PRUNING] attribution_missing_reason: {metrics.attribution_missing_reason}")
        try:
            return pruned, metrics
        except Exception as exc:
            raise RuntimeError(
                f"_PruningHook.__call__ return failed: {type(exc).__name__}: {exc}"
            ) from exc
