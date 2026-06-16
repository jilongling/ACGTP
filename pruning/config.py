"""Configuration for external OpenVLA visual-token pruning."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from .strategy_registry import (
    ALWAYS_ENABLED_STRATEGIES,
    DYNAMIC_MID_KEEP_STRATEGIES,
    LEGACY_STRATEGIES,
    strategy_family,
    validate_strategy_runtime,
)

    # P7: hybrid budget v2 — budget-based depth/robot protection
    # P11: branch budget v0 — explicit depth/hybrid/fill branch budgets, no global competition
    # P15: ACGTP-v1 — Action-Constrained Geometric Token Protection
    # P16: ACGTP-v2 — Task-Conditioned Action-Constrained Geometry Token Protection

SUPPORTED_FALLBACKS = {"no_pruning", "uniform_grid"}
SUPPORTED_ROBOT_GEO_MODES = {"off", "rule_v0", "dynamic_rule_v0", "temporal_rule_v0", "hybrid_v0", "hybrid_v1"}

DEFAULT_GEO_SCORE_WEIGHTS = {
    "distance_to_gripper": 0.0,
    "motion_direction": 0.0,
    "depth_edge": 1.0,
    "workspace": 0.0,
    "contact_risk": 0.0,
    "temporal_stability": 0.0,
}

# Hybrid v1 weighted score formula:
#   final_score = w_edge * norm(edge) + w_near * norm(near)
#               + w_contact * norm(contact) + w_corr * norm(corridor)
#               + w_diverse * spatial_diversity_prior
DEFAULT_HYBRID_V1_WEIGHTS = {
    "w_edge": 0.45,
    "w_near": 0.20,
    "w_contact": 0.20,
    "w_corr": 0.10,
    "w_diverse": 0.05,
}

# Adaptive threshold for temporal v1: max(absolute_min, percentile(contact_risk, adaptive_percentile))
DEFAULT_TEMPORAL_ADAPTIVE_CONFIG = {
    "adaptive_threshold_min": 0.08,
    "adaptive_threshold_percentile": 85.0,
    "interaction_lock_conservative_ratio": 0.90,
}

DEFAULT_DYNAMIC_KEEP_RATIO_CONFIG = {
    "min_keep_ratio": 0.75,
    "mid_keep_ratio": 0.85,
    "max_keep_ratio": 0.95,
    "contact_risk_threshold": 0.5,
    "uncertainty_threshold": 0.5,
}


@dataclass
class PruningHookConfig:
    strategy: str = "none"
    allow_legacy_strategy: bool = False
    keep_ratio: float = 1.0
    fallback_strategy: str = "no_pruning"
    min_valid_token_ratio: float = 0.1
    seed: int = 7
    latency_mode: bool = False
    token_grid_shape: Tuple[int, int] = (16, 16)
    cell_grid: int = 4
    diverse_reserve_tokens: int = 32
    w_edge: float = 0.8
    w_near: float = 0.2
    w_corridor: float = 0.2
    sigma_near: float = 0.12
    sigma_corridor: float = 0.08
    corridor_length: float = 0.12
    min_motion_norm: float = 1e-4
    keep_ratio_far: float = 0.50
    keep_ratio_mid: float = 0.75
    keep_ratio_near: float = 0.875
    dynamic_reserve_ratio: float = 1.0 / 6.0
    min_valid_depth_ratio: float = 0.1
    near_threshold: float = 0.08
    mid_threshold: float = 0.18
    high_corridor_threshold: float = 0.5
    min_depth: float = 1e-6
    max_depth: float = 10.0
    require_robot_state: bool = False
    contact_budget_edge_ratio: float = 0.75
    contact_budget_geo_ratio: float = 0.125
    contact_budget_diverse_ratio: float = 0.125
    w_near_contact: float = 0.5
    w_corridor_contact: float = 0.8
    edge_gate_eps: float = 1e-6
    detailed_pruning_timing: bool = False
    enable_robot_geo_expert: bool = False
    robot_geo_mode: str = "off"
    enable_depth_token_mapping: bool = False
    enable_robot_state_adapter: bool = False
    enable_dynamic_keep_ratio: bool = False
    enable_geo_debug: bool = False
    max_debug_frames: int = 16
    geo_debug_interval: int = 1
    temporal_history_length: int = 5
    temporal_lock_min_frames: int = 2
    temporal_contact_risk_threshold: float = 0.3
    temporal_stability_threshold: float = 0.5
    temporal_motion_cos_threshold: float = 0.7
    temporal_ema_alpha: float = 0.6
    temporal_gripper_lock_dist: float = 0.15
    temporal_adaptive_threshold_min: float = 0.08
    temporal_adaptive_threshold_percentile: float = 85.0
    temporal_interaction_lock_conservative_ratio: float = 0.90
    hybrid_v1_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_HYBRID_V1_WEIGHTS))
    geo_score_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_GEO_SCORE_WEIGHTS))
    dynamic_keep_ratio_config: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_DYNAMIC_KEEP_RATIO_CONFIG))
    # P5/P6: edge_reserve ablation - force-preserve top-K depth_edge tokens before hybrid competition
    edge_reserve_k: int = 0
    edge_reserve_ratio: float = 0.0  # P6: ratio of keep_k to reserve (overrides edge_reserve_k if set)
    # P7: hybrid_budget_v2 config - budget ratios for depth/robot protection
    hybrid_budget_v2_depth_edge_ratio: float = 0.45
    hybrid_budget_v2_robot_contact_ratio: float = 0.25
    hybrid_budget_v2_safety_ratio: float = 0.00
    # P11: branch_budget_v0 config - explicit branch budgets for depth/hybrid/fill
    branch_budget_depth_tokens: int = 65
    branch_budget_hybrid_tokens: int = 80
    branch_budget_depth_ratio: float = 0.0  # overrides depth_tokens if > 0 (fraction of keep_k)
    branch_budget_hybrid_ratio: float = 0.0  # overrides hybrid_tokens if > 0 (fraction of keep_k)
    # P14-B: Minimal 2D robot-self mask — filters gripper-core tokens from near_score/hybrid
    robot_self_mask_enabled: bool = False
    robot_self_mask_core_radius_px: float = 16.0  # pixels around gripper projection
    robot_self_mask_penalty: float = 0.0  # multiplicative penalty (0.0 = no effect; 0.0 is a no-op, use >0)
    robot_self_mask_apply_to_near_score: bool = True
    robot_self_mask_apply_to_depth_edge: bool = False
    robot_self_mask_apply_to_fill: bool = False
    robot_self_mask_apply_to_final_hybrid: bool = True
    # P15: ACGTP-v1 — Action-Constrained Geometric Token Protection
    # Contact ring geometry (2D pixel distances from gripper projection)
    acgtp_self_core_radius_px: float = 16.0
    acgtp_contact_ring_inner_px: float = 24.0
    acgtp_contact_ring_outer_px: float = 48.0
    acgtp_contact_requires_edge_or_object: bool = True
    # Branch weights for ACGTP-v1 mixture scores
    acgtp_w_scene_layout: float = 0.30
    acgtp_w_depth_structure: float = 0.25
    acgtp_w_contact_ring: float = 0.25
    acgtp_w_motion_corridor: float = 0.20
    # Motion corridor parameters
    acgtp_motion_corridor_length_m: float = 0.15
    acgtp_motion_sigma_m: float = 0.06
    acgtp_motion_ema_alpha: float = 0.6
    # Scene layout: depth-based support plane estimation
    acgtp_scene_support_depth_min: float = 0.3
    acgtp_scene_support_depth_max: float = 2.0
    # P6: support plane cap ratio (max fraction of tokens that can be SP candidates)
    acgtp_scene_support_plane_cap_ratio: float = 0.30
    # P6: object proposal params
    acgtp_scene_object_min_area_tokens: int = 5
    acgtp_scene_object_height_residual_threshold: float = 0.04
    # ACGTP-v1 hard-protect ratio (fraction of keep_k from hard_protect before constrained fill)
    acgtp_hard_protect_ratio: float = 0.60
    # Step 8: fast rollout selector. Full diagnostics can still force the
    # legacy coverage-aware selector for audits.
    acgtp_fast_selector_enabled: bool = True
    acgtp_full_diagnostics_enabled: bool = False
    acgtp_runtime_mode: str = "fast"  # "fast" | "debug" | "audit"
    # Step 5: ACGTP dynamic phase/risk controller
    acgtp_dynamic_enabled: bool = True
    acgtp_dynamic_min_keep_ratio: float = 0.60
    acgtp_dynamic_max_keep_ratio: float = 0.95
    acgtp_dynamic_risk_boost_scale: float = 0.20
    acgtp_dynamic_confidence_prune_scale: float = 0.16
    acgtp_dynamic_allow_below_base_keep_ratio: bool = False
    acgtp_dynamic_contact_phase_gate: str = "legacy_peak"  # "legacy_peak" | "coverage" | "hybrid"
    acgtp_dynamic_phase_schedule: str = "legacy"  # "legacy" | "aggressive"
    acgtp_dynamic_branch_floor_enabled: bool = False
    acgtp_constrained_fill_max_ratio: float = 1.0
    acgtp_dynamic_respect_phase_min_on_candidate_gap: bool = False
    acgtp_dynamic_shadow_contact_guard_enabled: bool = False
    acgtp_dynamic_shadow_contact_depth_weight_floor: float = 0.30
    acgtp_dynamic_shadow_contact_contact_weight_floor: float = 0.24
    acgtp_dynamic_shadow_contact_hard_ratio_floor: float = 0.70
    # Lightweight depth/scene score reuse for the fast runtime path. This is
    # deliberately limited to depth-derived structure/layout scores; contact and
    # motion are still recomputed every step.
    acgtp_static_scene_cache_enabled: bool = True
    acgtp_static_scene_cache_depth_delta_threshold: float = 0.015
    acgtp_static_scene_cache_valid_iou_threshold: float = 0.95
    acgtp_static_scene_cache_max_age: int = 3
    # Latency-profile runtime cache. This reuses the previous ACGTP pruning
    # plan for a few visually stable steps; diagnostics keep it disabled.
    acgtp_latency_plan_cache_enabled: bool = False
    acgtp_latency_plan_cache_depth_delta_threshold: float = 0.120
    acgtp_latency_plan_cache_gripper_delta_threshold: float = 0.150
    acgtp_latency_plan_cache_max_age: int = 20
    # Step 6: ACGTP history stabilizer (no visual token cache)
    acgtp_history_enabled: bool = False
    acgtp_history_length: int = 5
    acgtp_history_scene_ema_alpha: float = 0.75
    acgtp_history_depth_ema_alpha: float = 0.75
    acgtp_history_contact_ema_alpha: float = 0.45
    acgtp_history_motion_ema_alpha: float = 0.45
    acgtp_history_action_ema_alpha: float = 0.50
    acgtp_history_depth_change_threshold: float = 0.18
    acgtp_history_keep_iou_threshold: float = 0.55
    acgtp_history_motion_stability_threshold: float = 0.25
    acgtp_history_conservative_keep_boost: float = 0.08
    acgtp_history_conservative_hard_boost: float = 0.06
    # VLA-Pruner-inspired lightweight attention guidance. The first backend is
    # a proxy score available in the projector hook; a future LLM-attention
    # backend can feed the same selector inputs.
    acgtp_attention_guidance_enabled: bool = False
    acgtp_attention_guidance_source: str = "action_proxy"  # "action_proxy" | "geometry_proxy"
    acgtp_attention_history_length: int = 3
    acgtp_attention_history_decay: float = 0.8
    acgtp_attention_budget_ratio: float = 0.12
    acgtp_attention_redundancy_filter_enabled: bool = True
    acgtp_attention_redundancy_weight: float = 0.35
    acgtp_attention_requires_geometry_alignment: bool = True
    # VLA-Pruner-style post-pruning metadata handling. Projector-level pruning
    # keeps a short sequence, but preserves original multimodal RoPE positions.
    acgtp_position_preserve_enabled: bool = True
    acgtp_compression_backend: str = "projector"  # "projector" | "internal"
    acgtp_internal_pruning_enabled: bool = False
    acgtp_internal_prune_layer: int = 2
    acgtp_internal_fail_on_backend_error: bool = True
    acgtp_internal_allow_projector_fallback: bool = False
    acgtp_internal_selection_mode: str = "geo_guarded"  # geometry_only | attention_diagnostic | geo_guarded | dynamic
    acgtp_internal_attention_enabled: bool = True
    acgtp_internal_attention_budget_ratio: float = 0.20
    acgtp_internal_history_budget_ratio: float = 0.15
    acgtp_internal_risk_adaptive_enabled: bool = False
    acgtp_internal_high_risk_keep_ratio: float = 0.85
    acgtp_internal_medium_risk_keep_ratio: float = 0.55
    acgtp_internal_low_risk_keep_ratio: float = 0.40
    # Coverage-based risk (final-design P3). Physical/action evidence only;
    # low geometry-attention IoU is diagnostic and gated. Tunable for sweeps.
    acgtp_internal_risk_coverage_weight: float = 3.0
    acgtp_internal_risk_mean_weight: float = 1.5
    acgtp_internal_risk_peak_weight: float = 0.15
    acgtp_internal_risk_physical_weight: float = 0.85
    acgtp_internal_risk_depth_weight: float = 0.15
    acgtp_internal_risk_disagreement_gate: float = 0.45
    acgtp_internal_risk_disagreement_max_bonus: float = 0.10
    acgtp_internal_risk_high_threshold: float = 0.65
    acgtp_internal_risk_medium_threshold: float = 0.35
    acgtp_internal_capture_decode_attention: bool = False
    acgtp_internal_trace_enabled: bool = True
    # Explicit geometry hard-protection: tokens whose normalized contact/action/
    # depth-boundary/motion score exceeds this quantile (over valid tokens) are
    # marked geo-protected and must survive internal pruning. The selector adds
    # them first and raises the keep budget instead of dropping them.
    acgtp_internal_geo_protect_quantile: float = 0.80
    acgtp_internal_geo_protect_max_ratio: float = 0.50
    # Execution-function-aware structured allocation. The hard geo-protect set
    # is added before these quotas; ratios apply to the remaining visual budget.
    acgtp_internal_functional_quota_enabled: bool = True
    acgtp_internal_latency_fast_path: bool = False
    acgtp_internal_layout_quota_ratio: float = 0.30
    acgtp_internal_contact_quota_ratio: float = 0.20
    acgtp_internal_motion_quota_ratio: float = 0.15
    acgtp_internal_semantic_quota_ratio: float = 0.12
    acgtp_internal_action_quota_ratio: float = 0.08
    acgtp_internal_fill_quota_ratio: float = 0.15
    # Step 7: explicit branch-ablation probe for causal/sensitivity rollout.
    # Comma-separated: scene,depth,contact,motion,fill. Empty keeps normal path.
    acgtp_ablate_branches: str = ""
    # P16: ACGTP-v2 — Task-Semantic Anchor Branch
    acgtp_v2_semantic_enabled: bool = False
    acgtp_v2_semantic_backend: str = "none"  # "none" | "grounding_dino" | "owl_vit" | "lseg"
    acgtp_v2_w_semantic_target: float = 1.0
    acgtp_v2_w_semantic_reference: float = 0.7
    acgtp_v2_w_semantic_relation: float = 0.5
    acgtp_v2_w_semantic_goal: float = 0.9
    acgtp_v2_semantic_hard_ratio: float = 0.20
    acgtp_v2_target_cap_ratio: float = 0.25
    acgtp_v2_reference_cap_ratio: float = 0.20
    acgtp_v2_relation_cap_ratio: float = 0.15
    acgtp_v2_release_semantic_quota_when_unavailable: bool = True
    debug: bool = False

    @classmethod
    def from_eval_cfg(cls, cfg: Dict[str, Any]) -> "PruningHookConfig":
        strategy = (
            cfg.get("pruning_strategy")
            or cfg.get("pruning_mode")
            or cfg.get("pruning_method")
            or "none"
        )
        runtime_mode = str(cfg.get("acgtp_runtime_mode", "audit" if _as_bool(cfg.get("acgtp_full_diagnostics_enabled", False)) else "fast") or "fast").strip().lower()
        if runtime_mode not in {"fast", "debug", "audit"}:
            runtime_mode = "fast"
        allow_legacy_strategy = _as_bool(cfg.get("allow_legacy_strategy", False))
        strategy = validate_strategy_runtime(
            str(strategy),
            allow_legacy_strategy=allow_legacy_strategy,
            runtime_mode=runtime_mode,
            full_diagnostics_enabled=_as_bool(cfg.get("acgtp_full_diagnostics_enabled", False)),
        )
        cfg["pruning_strategy_family"] = strategy_family(strategy)
        cfg["pruning_strategy_is_legacy"] = strategy in LEGACY_STRATEGIES
        cfg["pruning_strategy_is_core"] = strategy in {"none", "depth_edge_fast", "robot_geo_acgtp_v2"}
        fallback = str(cfg.get("fallback_strategy") or cfg.get("pruning_fallback_strategy", "no_pruning")).strip()
        if fallback not in SUPPORTED_FALLBACKS:
            fallback = "no_pruning"

        keep_ratio = float(cfg.get("keep_ratio", 1.0))
        keep_ratio = max(0.0, min(1.0, keep_ratio))
        robot_geo_mode = str(cfg.get("robot_geo_mode", "off")).strip().lower()
        return cls(
            strategy=strategy,
            allow_legacy_strategy=allow_legacy_strategy,
            keep_ratio=keep_ratio,
            fallback_strategy=fallback,
            min_valid_token_ratio=float(cfg.get("min_valid_token_ratio", 0.1)),
            seed=int(cfg.get("seed", 7)),
            latency_mode=_as_bool(cfg.get("latency_mode", False))
            or str(cfg.get("timing_profile", "") or "").strip().lower() == "latency",
            token_grid_shape=tuple(cfg.get("token_grid_shape", (16, 16))),
            cell_grid=int(cfg.get("cell_grid", 4)),
            diverse_reserve_tokens=int(cfg.get("diverse_reserve_tokens", 32)),
            w_edge=float(cfg.get("w_edge", 0.8)),
            w_near=float(cfg.get("w_near", 0.2)),
            w_corridor=float(cfg.get("w_corridor", 0.2)),
            sigma_near=float(cfg.get("sigma_near", 0.12)),
            sigma_corridor=float(cfg.get("sigma_corridor", 0.08)),
            corridor_length=float(cfg.get("corridor_length", 0.12)),
            min_motion_norm=float(cfg.get("min_motion_norm", 1e-4)),
            keep_ratio_far=float(cfg.get("keep_ratio_far", 0.50)),
            keep_ratio_mid=float(cfg.get("keep_ratio_mid", 0.75)),
            keep_ratio_near=float(cfg.get("keep_ratio_near", 0.875)),
            dynamic_reserve_ratio=float(cfg.get("dynamic_reserve_ratio", 1.0 / 6.0)),
            min_valid_depth_ratio=float(cfg.get("min_valid_depth_ratio", cfg.get("min_valid_token_ratio", 0.1))),
            near_threshold=float(cfg.get("near_threshold", 0.08)),
            mid_threshold=float(cfg.get("mid_threshold", 0.18)),
            high_corridor_threshold=float(cfg.get("high_corridor_threshold", 0.5)),
            min_depth=float(cfg.get("min_depth", 1e-6)),
            max_depth=float(cfg.get("max_depth", 10.0)),
            require_robot_state=bool(cfg.get("require_robot_state", False)),
            contact_budget_edge_ratio=float(cfg.get("contact_budget_edge_ratio", 0.75)),
            contact_budget_geo_ratio=float(cfg.get("contact_budget_geo_ratio", 0.125)),
            contact_budget_diverse_ratio=float(cfg.get("contact_budget_diverse_ratio", 0.125)),
            w_near_contact=float(cfg.get("w_near_contact", 0.5)),
            w_corridor_contact=float(cfg.get("w_corridor_contact", 0.8)),
            edge_gate_eps=float(cfg.get("edge_gate_eps", 1e-6)),
            detailed_pruning_timing=bool(cfg.get("detailed_pruning_timing", False)),
            enable_robot_geo_expert=_as_bool(cfg.get("enable_robot_geo_expert", False)),
            robot_geo_mode=robot_geo_mode,
            enable_depth_token_mapping=_as_bool(cfg.get("enable_depth_token_mapping", False)),
            enable_robot_state_adapter=_as_bool(cfg.get("enable_robot_state_adapter", False)),
            enable_dynamic_keep_ratio=_as_bool(cfg.get("enable_dynamic_keep_ratio", False)),
            enable_geo_debug=_as_bool(cfg.get("enable_geo_debug", False)),
            max_debug_frames=max(0, int(cfg.get("max_debug_frames", 16))),
            geo_debug_interval=max(1, int(cfg.get("geo_debug_interval", 1))),
            temporal_history_length=max(1, int(cfg.get("temporal_history_length", 5))),
            temporal_lock_min_frames=max(1, int(cfg.get("temporal_lock_min_frames", 2))),
            temporal_contact_risk_threshold=float(cfg.get("temporal_contact_risk_threshold", cfg.get("contact_risk_threshold", 0.3))),
            temporal_stability_threshold=float(cfg.get("temporal_stability_threshold", 0.5)),
            temporal_motion_cos_threshold=float(cfg.get("temporal_motion_cos_threshold", 0.7)),
            temporal_ema_alpha=max(0.0, min(1.0, float(cfg.get("temporal_ema_alpha", 0.6)))),
            temporal_gripper_lock_dist=float(cfg.get("temporal_gripper_lock_dist", 0.15)),
            temporal_adaptive_threshold_min=float(cfg.get("temporal_adaptive_threshold_min", 0.08)),
            temporal_adaptive_threshold_percentile=float(cfg.get("temporal_adaptive_threshold_percentile", 85.0)),
            temporal_interaction_lock_conservative_ratio=float(cfg.get("temporal_interaction_lock_conservative_ratio", 0.90)),
            geo_score_weights=_merge_float_dict(
                DEFAULT_GEO_SCORE_WEIGHTS,
                cfg.get("geo_score_weights"),
            ),
            hybrid_v1_weights=_merge_float_dict(
                DEFAULT_HYBRID_V1_WEIGHTS,
                cfg.get("hybrid_v1_weights"),
            ),
            dynamic_keep_ratio_config=_merge_float_dict(
                DEFAULT_DYNAMIC_KEEP_RATIO_CONFIG,
                cfg.get("dynamic_keep_ratio_config"),
            ),
            edge_reserve_k=max(0, int(cfg.get("edge_reserve_k", 0))),
            edge_reserve_ratio=max(0.0, float(cfg.get("edge_reserve_ratio", 0.0))),
            hybrid_budget_v2_depth_edge_ratio=float(cfg.get("hybrid_budget_v2_depth_edge_ratio", 0.45)),
            hybrid_budget_v2_robot_contact_ratio=float(cfg.get("hybrid_budget_v2_robot_contact_ratio", 0.25)),
            hybrid_budget_v2_safety_ratio=float(cfg.get("hybrid_budget_v2_safety_ratio", 0.00)),
            # P11: branch_budget_v0
            branch_budget_depth_tokens=max(1, int(cfg.get("branch_budget_depth_tokens", 65))),
            branch_budget_hybrid_tokens=max(1, int(cfg.get("branch_budget_hybrid_tokens", 80))),
            branch_budget_depth_ratio=max(0.0, min(1.0, float(cfg.get("branch_budget_depth_ratio", 0.0)))),
            branch_budget_hybrid_ratio=max(0.0, min(1.0, float(cfg.get("branch_budget_hybrid_ratio", 0.0)))),
            # P14-B: robot_self_mask
            robot_self_mask_enabled=_as_bool(cfg.get("robot_self_mask_enabled", False)),
            robot_self_mask_core_radius_px=max(1.0, float(cfg.get("robot_self_mask_core_radius_px", 16.0))),
            robot_self_mask_penalty=max(0.0, min(1.0, float(cfg.get("robot_self_mask_penalty", 0.0)))),
            robot_self_mask_apply_to_near_score=_as_bool(cfg.get("robot_self_mask_apply_to_near_score", True)),
            robot_self_mask_apply_to_depth_edge=_as_bool(cfg.get("robot_self_mask_apply_to_depth_edge", False)),
            robot_self_mask_apply_to_fill=_as_bool(cfg.get("robot_self_mask_apply_to_fill", False)),
            robot_self_mask_apply_to_final_hybrid=_as_bool(cfg.get("robot_self_mask_apply_to_final_hybrid", True)),
            debug=bool(cfg.get("geometry_debug", False)),
            # P15: ACGTP-v1
            acgtp_self_core_radius_px=max(1.0, float(cfg.get("acgtp_self_core_radius_px", 16.0))),
            acgtp_contact_ring_inner_px=max(1.0, float(cfg.get("acgtp_contact_ring_inner_px", 24.0))),
            acgtp_contact_ring_outer_px=max(1.0, float(cfg.get("acgtp_contact_ring_outer_px", 48.0))),
            acgtp_contact_requires_edge_or_object=_as_bool(cfg.get("acgtp_contact_requires_edge_or_object", True)),
            acgtp_w_scene_layout=max(0.0, float(cfg.get("acgtp_w_scene_layout", 0.30))),
            acgtp_w_depth_structure=max(0.0, float(cfg.get("acgtp_w_depth_structure", 0.25))),
            acgtp_w_contact_ring=max(0.0, float(cfg.get("acgtp_w_contact_ring", 0.25))),
            acgtp_w_motion_corridor=max(0.0, float(cfg.get("acgtp_w_motion_corridor", 0.20))),
            acgtp_motion_corridor_length_m=max(0.01, float(cfg.get("acgtp_motion_corridor_length_m", 0.15))),
            acgtp_motion_sigma_m=max(0.001, float(cfg.get("acgtp_motion_sigma_m", 0.06))),
            acgtp_motion_ema_alpha=max(0.0, min(1.0, float(cfg.get("acgtp_motion_ema_alpha", 0.6)))),
            acgtp_scene_support_depth_min=max(0.01, float(cfg.get("acgtp_scene_support_depth_min", 0.3))),
            acgtp_scene_support_depth_max=max(0.01, float(cfg.get("acgtp_scene_support_depth_max", 2.0))),
            acgtp_scene_support_plane_cap_ratio=max(0.05, min(0.9, float(cfg.get("acgtp_scene_support_plane_cap_ratio", 0.30)))),
            acgtp_scene_object_min_area_tokens=max(1, int(cfg.get("acgtp_scene_object_min_area_tokens", 5))),
            acgtp_scene_object_height_residual_threshold=max(0.005, float(cfg.get("acgtp_scene_object_height_residual_threshold", 0.04))),
            acgtp_hard_protect_ratio=max(0.0, min(1.0, float(cfg.get("acgtp_hard_protect_ratio", 0.60)))),
            acgtp_fast_selector_enabled=_as_bool(cfg.get("acgtp_fast_selector_enabled", strategy == "robot_geo_acgtp_v2")),
            acgtp_full_diagnostics_enabled=_as_bool(cfg.get("acgtp_full_diagnostics_enabled", False)) or runtime_mode == "audit",
            acgtp_runtime_mode=runtime_mode,
            # Step 5: ACGTP dynamic phase/risk controller
            acgtp_dynamic_enabled=_as_bool(cfg.get("acgtp_dynamic_enabled", strategy == "robot_geo_acgtp_v2")),
            acgtp_dynamic_min_keep_ratio=max(0.05, min(1.0, float(cfg.get("acgtp_dynamic_min_keep_ratio", 0.60)))),
            acgtp_dynamic_max_keep_ratio=max(0.05, min(1.0, float(cfg.get("acgtp_dynamic_max_keep_ratio", 0.95)))),
            acgtp_dynamic_risk_boost_scale=max(0.0, float(cfg.get("acgtp_dynamic_risk_boost_scale", 0.20))),
            acgtp_dynamic_confidence_prune_scale=max(0.0, float(cfg.get("acgtp_dynamic_confidence_prune_scale", 0.16))),
            acgtp_dynamic_allow_below_base_keep_ratio=_as_bool(
                cfg.get("acgtp_dynamic_allow_below_base_keep_ratio", False)
            ),
            acgtp_dynamic_contact_phase_gate=str(cfg.get("acgtp_dynamic_contact_phase_gate", "legacy_peak")),
            acgtp_dynamic_phase_schedule=str(cfg.get("acgtp_dynamic_phase_schedule", "legacy")),
            acgtp_dynamic_branch_floor_enabled=_as_bool(cfg.get("acgtp_dynamic_branch_floor_enabled", False)),
            acgtp_constrained_fill_max_ratio=max(
                0.0, min(1.0, float(cfg.get("acgtp_constrained_fill_max_ratio", 1.0)))
            ),
            acgtp_dynamic_respect_phase_min_on_candidate_gap=_as_bool(
                cfg.get("acgtp_dynamic_respect_phase_min_on_candidate_gap", False)
            ),
            acgtp_dynamic_shadow_contact_guard_enabled=_as_bool(
                cfg.get("acgtp_dynamic_shadow_contact_guard_enabled", False)
            ),
            acgtp_dynamic_shadow_contact_depth_weight_floor=max(
                0.0, min(1.0, float(cfg.get("acgtp_dynamic_shadow_contact_depth_weight_floor", 0.30)))
            ),
            acgtp_dynamic_shadow_contact_contact_weight_floor=max(
                0.0, min(1.0, float(cfg.get("acgtp_dynamic_shadow_contact_contact_weight_floor", 0.24)))
            ),
            acgtp_dynamic_shadow_contact_hard_ratio_floor=max(
                0.0, min(1.0, float(cfg.get("acgtp_dynamic_shadow_contact_hard_ratio_floor", 0.70)))
            ),
            acgtp_static_scene_cache_enabled=_as_bool(cfg.get("acgtp_static_scene_cache_enabled", True)),
            acgtp_static_scene_cache_depth_delta_threshold=max(
                0.0, float(cfg.get("acgtp_static_scene_cache_depth_delta_threshold", 0.015))
            ),
            acgtp_static_scene_cache_valid_iou_threshold=max(
                0.0, min(1.0, float(cfg.get("acgtp_static_scene_cache_valid_iou_threshold", 0.95)))
            ),
            acgtp_static_scene_cache_max_age=max(0, int(cfg.get("acgtp_static_scene_cache_max_age", 3))),
            acgtp_latency_plan_cache_enabled=_as_bool(cfg.get("acgtp_latency_plan_cache_enabled", False)),
            acgtp_latency_plan_cache_depth_delta_threshold=max(
                0.0, float(cfg.get("acgtp_latency_plan_cache_depth_delta_threshold", 0.120))
            ),
            acgtp_latency_plan_cache_gripper_delta_threshold=max(
                0.0, float(cfg.get("acgtp_latency_plan_cache_gripper_delta_threshold", 0.150))
            ),
            acgtp_latency_plan_cache_max_age=max(0, int(cfg.get("acgtp_latency_plan_cache_max_age", 20))),
            # Step 6: ACGTP history stabilizer
            acgtp_history_enabled=_as_bool(cfg.get("acgtp_history_enabled", False)),
            acgtp_history_length=max(1, int(cfg.get("acgtp_history_length", 5))),
            acgtp_history_scene_ema_alpha=max(0.0, min(1.0, float(cfg.get("acgtp_history_scene_ema_alpha", 0.75)))),
            acgtp_history_depth_ema_alpha=max(0.0, min(1.0, float(cfg.get("acgtp_history_depth_ema_alpha", 0.75)))),
            acgtp_history_contact_ema_alpha=max(0.0, min(1.0, float(cfg.get("acgtp_history_contact_ema_alpha", 0.45)))),
            acgtp_history_motion_ema_alpha=max(0.0, min(1.0, float(cfg.get("acgtp_history_motion_ema_alpha", 0.45)))),
            acgtp_history_action_ema_alpha=max(0.0, min(1.0, float(cfg.get("acgtp_history_action_ema_alpha", 0.50)))),
            acgtp_history_depth_change_threshold=max(0.0, float(cfg.get("acgtp_history_depth_change_threshold", 0.18))),
            acgtp_history_keep_iou_threshold=max(0.0, min(1.0, float(cfg.get("acgtp_history_keep_iou_threshold", 0.55)))),
            acgtp_history_motion_stability_threshold=max(-1.0, min(1.0, float(cfg.get("acgtp_history_motion_stability_threshold", 0.25)))),
            acgtp_history_conservative_keep_boost=max(0.0, float(cfg.get("acgtp_history_conservative_keep_boost", 0.08))),
            acgtp_history_conservative_hard_boost=max(0.0, float(cfg.get("acgtp_history_conservative_hard_boost", 0.06))),
            acgtp_attention_guidance_enabled=_as_bool(cfg.get("acgtp_attention_guidance_enabled", False)),
            acgtp_attention_guidance_source=str(cfg.get("acgtp_attention_guidance_source", "action_proxy")),
            acgtp_attention_history_length=max(1, int(cfg.get("acgtp_attention_history_length", 3))),
            acgtp_attention_history_decay=max(0.0, min(0.99, float(cfg.get("acgtp_attention_history_decay", 0.8)))),
            acgtp_attention_budget_ratio=max(0.0, min(0.5, float(cfg.get("acgtp_attention_budget_ratio", 0.12)))),
            acgtp_attention_redundancy_filter_enabled=_as_bool(cfg.get("acgtp_attention_redundancy_filter_enabled", True)),
            acgtp_attention_redundancy_weight=max(0.0, min(1.0, float(cfg.get("acgtp_attention_redundancy_weight", 0.35)))),
            acgtp_attention_requires_geometry_alignment=_as_bool(cfg.get("acgtp_attention_requires_geometry_alignment", True)),
            acgtp_position_preserve_enabled=_as_bool(cfg.get("acgtp_position_preserve_enabled", True)),
            acgtp_compression_backend=(
                "internal"
                if _as_bool(cfg.get("acgtp_internal_pruning_enabled", False))
                else str(cfg.get("acgtp_compression_backend", "projector") or "projector").strip().lower()
            ),
            acgtp_internal_pruning_enabled=_as_bool(cfg.get("acgtp_internal_pruning_enabled", False))
            or str(cfg.get("acgtp_compression_backend", "projector") or "projector").strip().lower() == "internal",
            acgtp_internal_prune_layer=max(0, int(cfg.get("acgtp_internal_prune_layer", 2))),
            acgtp_internal_fail_on_backend_error=_as_bool(cfg.get("acgtp_internal_fail_on_backend_error", True)),
            acgtp_internal_allow_projector_fallback=_as_bool(cfg.get("acgtp_internal_allow_projector_fallback", False)),
            acgtp_internal_selection_mode=str(cfg.get("acgtp_internal_selection_mode", "geo_guarded")).strip().lower(),
            acgtp_internal_attention_enabled=_as_bool(cfg.get("acgtp_internal_attention_enabled", True)),
            acgtp_internal_attention_budget_ratio=max(
                0.0, min(0.7, float(cfg.get("acgtp_internal_attention_budget_ratio", 0.20)))
            ),
            acgtp_internal_history_budget_ratio=max(
                0.0, min(0.7, float(cfg.get("acgtp_internal_history_budget_ratio", 0.15)))
            ),
            acgtp_internal_risk_adaptive_enabled=_as_bool(cfg.get("acgtp_internal_risk_adaptive_enabled", False)),
            acgtp_internal_high_risk_keep_ratio=max(
                0.05, min(1.0, float(cfg.get("acgtp_internal_high_risk_keep_ratio", 0.85)))
            ),
            acgtp_internal_medium_risk_keep_ratio=max(
                0.05, min(1.0, float(cfg.get("acgtp_internal_medium_risk_keep_ratio", 0.55)))
            ),
            acgtp_internal_low_risk_keep_ratio=max(
                0.05, min(1.0, float(cfg.get("acgtp_internal_low_risk_keep_ratio", 0.40)))
            ),
            acgtp_internal_risk_coverage_weight=max(0.0, float(cfg.get("acgtp_internal_risk_coverage_weight", 3.0))),
            acgtp_internal_risk_mean_weight=max(0.0, float(cfg.get("acgtp_internal_risk_mean_weight", 1.5))),
            acgtp_internal_risk_peak_weight=max(0.0, float(cfg.get("acgtp_internal_risk_peak_weight", 0.15))),
            acgtp_internal_risk_physical_weight=max(0.0, float(cfg.get("acgtp_internal_risk_physical_weight", 0.85))),
            acgtp_internal_risk_depth_weight=max(0.0, float(cfg.get("acgtp_internal_risk_depth_weight", 0.15))),
            acgtp_internal_risk_disagreement_gate=max(0.0, min(1.0, float(cfg.get("acgtp_internal_risk_disagreement_gate", 0.45)))),
            acgtp_internal_risk_disagreement_max_bonus=max(0.0, min(1.0, float(cfg.get("acgtp_internal_risk_disagreement_max_bonus", 0.10)))),
            acgtp_internal_risk_high_threshold=max(0.0, min(1.0, float(cfg.get("acgtp_internal_risk_high_threshold", 0.65)))),
            acgtp_internal_risk_medium_threshold=max(0.0, min(1.0, float(cfg.get("acgtp_internal_risk_medium_threshold", 0.35)))),
            acgtp_internal_capture_decode_attention=_as_bool(cfg.get("acgtp_internal_capture_decode_attention", False)),
            acgtp_internal_trace_enabled=_as_bool(cfg.get("acgtp_internal_trace_enabled", True)),
            acgtp_internal_geo_protect_quantile=max(
                0.0, min(1.0, float(cfg.get("acgtp_internal_geo_protect_quantile", 0.80)))
            ),
            acgtp_internal_geo_protect_max_ratio=max(
                0.0, min(1.0, float(cfg.get("acgtp_internal_geo_protect_max_ratio", 0.50)))
            ),
            acgtp_internal_functional_quota_enabled=_as_bool(
                cfg.get("acgtp_internal_functional_quota_enabled", True)
            ),
            acgtp_internal_latency_fast_path=_as_bool(cfg.get("acgtp_internal_latency_fast_path", False)),
            acgtp_internal_layout_quota_ratio=max(
                0.0, min(1.0, float(cfg.get("acgtp_internal_layout_quota_ratio", 0.30)))
            ),
            acgtp_internal_contact_quota_ratio=max(
                0.0, min(1.0, float(cfg.get("acgtp_internal_contact_quota_ratio", 0.20)))
            ),
            acgtp_internal_motion_quota_ratio=max(
                0.0, min(1.0, float(cfg.get("acgtp_internal_motion_quota_ratio", 0.15)))
            ),
            acgtp_internal_semantic_quota_ratio=max(
                0.0, min(1.0, float(cfg.get("acgtp_internal_semantic_quota_ratio", 0.12)))
            ),
            acgtp_internal_action_quota_ratio=max(
                0.0, min(1.0, float(cfg.get("acgtp_internal_action_quota_ratio", 0.08)))
            ),
            acgtp_internal_fill_quota_ratio=max(
                0.0, min(1.0, float(cfg.get("acgtp_internal_fill_quota_ratio", 0.15)))
            ),
            acgtp_ablate_branches=str(cfg.get("acgtp_ablate_branches", "") or ""),
            # P16: ACGTP-v2 — Task-Semantic Anchor Branch
            acgtp_v2_semantic_enabled=_as_bool(cfg.get("acgtp_v2_semantic_enabled", False)),
            acgtp_v2_semantic_backend=str(cfg.get("acgtp_v2_semantic_backend", "none")).strip().lower(),
            acgtp_v2_w_semantic_target=max(0.0, float(cfg.get("acgtp_v2_w_semantic_target", 1.0))),
            acgtp_v2_w_semantic_reference=max(0.0, float(cfg.get("acgtp_v2_w_semantic_reference", 0.7))),
            acgtp_v2_w_semantic_relation=max(0.0, float(cfg.get("acgtp_v2_w_semantic_relation", 0.5))),
            acgtp_v2_w_semantic_goal=max(0.0, float(cfg.get("acgtp_v2_w_semantic_goal", 0.9))),
            acgtp_v2_semantic_hard_ratio=max(0.0, min(1.0, float(cfg.get("acgtp_v2_semantic_hard_ratio", 0.20)))),
            acgtp_v2_target_cap_ratio=max(0.0, min(0.9, float(cfg.get("acgtp_v2_target_cap_ratio", 0.25)))),
            acgtp_v2_reference_cap_ratio=max(0.0, min(0.9, float(cfg.get("acgtp_v2_reference_cap_ratio", 0.20)))),
            acgtp_v2_relation_cap_ratio=max(0.0, min(0.9, float(cfg.get("acgtp_v2_relation_cap_ratio", 0.15)))),
            acgtp_v2_release_semantic_quota_when_unavailable=_as_bool(
                cfg.get("acgtp_v2_release_semantic_quota_when_unavailable", True)
            ),
        )

    @property
    def enabled(self) -> bool:
        if self.strategy in ALWAYS_ENABLED_STRATEGIES:
            return True
        return self.strategy != "none" and self.keep_ratio < 1.0

    def keep_count(self, num_tokens: int) -> int:
        # Only explicitly dynamic strategies may use dynamic_keep_ratio_config as
        # the default keep-count source. Fixed hybrid strategies must respect
        # CLI/config keep_ratio so names like keep075 and keep085 are meaningful.
        if self.strategy in DYNAMIC_MID_KEEP_STRATEGIES:
            mid_ratio = float(self.dynamic_keep_ratio_config.get("mid_keep_ratio", self.keep_ratio_mid))
            return int(round(num_tokens * mid_ratio))
        if self.strategy == "none" or self.keep_ratio >= 1.0:
            return int(num_tokens)
        return int(round(num_tokens * self.keep_ratio))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _merge_float_dict(defaults: Dict[str, float], override: Any) -> Dict[str, float]:
    out = dict(defaults)
    if override in (None, ""):
        return out
    if isinstance(override, str):
        try:
            override = json.loads(override)
        except json.JSONDecodeError:
            return out
    if not isinstance(override, dict):
        return out
    for key, value in override.items():
        if key in out:
            try:
                out[key] = float(value)
            except (TypeError, ValueError):
                pass
    return out
