"""Current ACGTP-v2 selector implementations."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Source: pruning/selectors/acgtp_v2_fast_selector.py
# ---------------------------------------------------------------------------
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .utils import finalize_selection_debug_info

def select_acgtp_v2_fast(
    scene_layout_scores: Optional[np.ndarray] = None,
    depth_edge_scores: Optional[np.ndarray] = None,
    contact_ring_scores: Optional[np.ndarray] = None,
    motion_corridor_scores: Optional[np.ndarray] = None,
    semantic_anchor_scores: Optional[np.ndarray] = None,
    semantic_target_scores: Optional[np.ndarray] = None,
    semantic_reference_scores: Optional[np.ndarray] = None,
    semantic_relation_scores: Optional[np.ndarray] = None,
    semantic_goal_scores: Optional[np.ndarray] = None,
    valid_mask: Optional[np.ndarray] = None,
    keep_k: int = 0,
    constrained_fill_mask: Optional[np.ndarray] = None,
    token_u: Optional[np.ndarray] = None,
    token_v: Optional[np.ndarray] = None,
    grid_h: int = 16,
    grid_w: int = 16,
    w_scene_layout: float = 0.25,
    w_depth_structure: float = 0.20,
    w_contact_ring: float = 0.20,
    w_motion_corridor: float = 0.15,
    w_semantic: float = 0.20,
    hard_protect_ratio: float = 0.60,
    motion_corridor_valid: bool = False,
    self_core_mask: Optional[np.ndarray] = None,
    contact_ring_inner_px: float = 24.0,
    contact_ring_outer_px: float = 48.0,
    contact_requires_edge_or_object: bool = True,
    depth_edge_score_for_gate: Optional[np.ndarray] = None,
    _motion_result_for_diag: Optional[Dict[str, Any]] = None,
    _scene_result_for_diag: Optional[Dict[str, Any]] = None,
    action_constraint_scores: Optional[np.ndarray] = None,
    _action_constraint_result_for_diag: Optional[Dict[str, Any]] = None,
    support_plane_cap_ratio: float = 0.30,
    semantic_enabled: bool = False,
    semantic_backend: str = "none",
    semantic_confidence: float = 0.0,
    semantic_unavailable: bool = True,
    semantic_fallback_reason: Optional[str] = None,
    release_semantic_quota_when_unavailable: bool = True,
    w_semantic_target: float = 1.0,
    w_semantic_reference: float = 0.7,
    w_semantic_relation: float = 0.5,
    w_semantic_goal: float = 0.9,
    min_scene_tokens: int = 0,
    min_depth_tokens: int = 0,
    min_contact_tokens: int = 0,
    min_motion_tokens: int = 0,
    constrained_fill_max_tokens: Optional[int] = None,
    acgtp_attention_enabled: bool = False,
    acgtp_attention_backend: str = "none",
    acgtp_attention_task_relevance_score: Optional[np.ndarray] = None,
    acgtp_attention_task_relevance_mask: Optional[np.ndarray] = None,
    acgtp_attention_source: str = "none",
    acgtp_attention_available: bool = False,
    acgtp_attention_confidence: float = 0.0,
    acgtp_attention_budget_ratio: float = 0.10,
    acgtp_attention_requires_geometry_alignment: bool = True,
    minimal_metadata: bool = False,
    **_: Any,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Fast ACGTP-v2 rollout selector.

    It keeps the action-constrained branch semantics, but replaces the legacy
    Python set/list coverage loop with vectorized top-k masks and lightweight
    diagnostics. The original select_acgtp_v2 remains the full-audit path.
    """

    select_start = time.perf_counter()

    def _flat_size(arr: Optional[np.ndarray]) -> int:
        if arr is None:
            return 0
        try:
            return int(np.asarray(arr).size)
        except Exception:
            return 0

    n = max(1, int(keep_k))
    for arr in (
        scene_layout_scores,
        depth_edge_scores,
        contact_ring_scores,
        motion_corridor_scores,
        semantic_anchor_scores,
        action_constraint_scores,
        acgtp_attention_task_relevance_score,
        acgtp_attention_task_relevance_mask,
        valid_mask,
    ):
        n = max(n, _flat_size(arr))

    keep_k = int(max(0, min(int(keep_k), n)))
    if keep_k <= 0:
        keep_indices_empty = np.asarray([], dtype=np.int64)
        metadata = {
            "strategy": "robot_geo_acgtp_v2",
            "selector_name": "select_acgtp_v2_fast",
            "selector_function_name": "select_acgtp_v2_fast",
            "selection_strategy_name": "robot_geo_acgtp_v2",
            "selection_stage_name": "acgtp_v2_fast_empty",
            "acgtp_v2": True,
            "acgtp_v1": False,
            "acgtp_selector_version": "acgtp_v2_2_fast_path",
            "acgtp_fast_selector_used": True,
        }
        return keep_indices_empty, finalize_selection_debug_info(
            metadata,
            selector_function_name="select_acgtp_v2_fast",
            strategy="robot_geo_acgtp_v2",
            keep_indices=keep_indices_empty,
            num_tokens=n,
            keep_count=0,
            requested_keep_ratio=0.0,
            fallback_used=False,
            fallback_reason=None,
        )

    def _arr(arr: Optional[np.ndarray], *, fill: float = 0.0) -> np.ndarray:
        out = np.full(n, fill, dtype=np.float32)
        if arr is None:
            return out
        try:
            flat = np.asarray(arr, dtype=np.float32).reshape(-1)
        except Exception:
            return out
        m = min(n, int(flat.size))
        if m > 0:
            out[:m] = flat[:m]
        return np.nan_to_num(out, nan=fill, posinf=fill, neginf=fill).astype(np.float32, copy=False)

    if valid_mask is not None and _flat_size(valid_mask) > 0:
        valid = _arr(valid_mask, fill=0.0) > 0.5
    else:
        valid = np.ones(n, dtype=bool)
    if not np.any(valid):
        valid[:] = True

    def _norm(scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        out = np.zeros(n, dtype=np.float32)
        finite_mask = valid & np.isfinite(scores)
        vals = scores[finite_mask]
        if vals.size == 0:
            return out
        lo = float(np.min(vals))
        hi = float(np.max(vals))
        if hi - lo > 1e-8:
            out[finite_mask] = (scores[finite_mask] - lo) / (hi - lo)
        out[~valid] = 0.0
        return out

    raw_scene = _arr(scene_layout_scores)
    raw_de = _arr(depth_edge_scores)
    raw_contact = _arr(contact_ring_scores)
    raw_motion = _arr(motion_corridor_scores)
    raw_action = _arr(action_constraint_scores)
    raw_self_core = _arr(self_core_mask) > 0.5 if self_core_mask is not None else np.zeros(n, dtype=bool)
    raw_fill = _arr(constrained_fill_mask) if constrained_fill_mask is not None else None

    norm_scene = _norm(raw_scene)
    norm_de = _norm(raw_de)
    norm_contact = _norm(raw_contact)
    norm_motion = _norm(raw_motion)
    norm_action = _norm(raw_action)

    if contact_requires_edge_or_object and depth_edge_score_for_gate is not None:
        gate = _norm(_arr(depth_edge_score_for_gate))
        gate_vals = gate[valid]
        gate_thr = float(np.percentile(gate_vals, 60)) if gate_vals.size else 0.0
        norm_contact[gate < gate_thr] = 0.0
    norm_contact[raw_self_core] = 0.0
    if not motion_corridor_valid:
        norm_motion[:] = 0.0

    attention_active = bool(acgtp_attention_enabled and acgtp_attention_available)
    if bool(semantic_enabled) and not bool(semantic_unavailable):
        raw_sem = (
            float(w_semantic_target) * _norm(_arr(semantic_target_scores))
            + float(w_semantic_reference) * _norm(_arr(semantic_reference_scores))
            + float(w_semantic_relation) * _norm(_arr(semantic_relation_scores))
            + float(w_semantic_goal) * _norm(_arr(semantic_goal_scores))
            + _norm(_arr(semantic_anchor_scores))
        )
        norm_sem = _norm(raw_sem)
    else:
        norm_sem = np.zeros(n, dtype=np.float32)
    if attention_active:
        raw_attn = _arr(acgtp_attention_task_relevance_score)
        raw_attn_mask = _arr(acgtp_attention_task_relevance_mask)
        norm_attn = _norm(raw_attn)
    else:
        raw_attn_mask = np.zeros(n, dtype=np.float32)
        norm_attn = np.zeros(n, dtype=np.float32)
    geom_alignment_mask_for_attn = valid & (
        (norm_scene > 0.0)
        | (norm_de > 0.0)
        | (norm_contact > 0.0)
        | (norm_motion > 0.0)
    )
    attention_top_mask = valid & (raw_attn_mask > 0.5)
    attention_only_mask = attention_top_mask & ~geom_alignment_mask_for_attn
    attention_candidate_mask = attention_top_mask
    if bool(acgtp_attention_requires_geometry_alignment):
        attention_candidate_mask = attention_candidate_mask & geom_alignment_mask_for_attn

    action_constraint_available = bool(np.any(raw_action[valid] > 0.0))
    if action_constraint_available:
        final_score = norm_action.copy()
        action_source = "future_action_constraint"
    else:
        final_score = (
            float(w_scene_layout) * norm_scene
            + float(w_depth_structure) * norm_de
            + float(w_contact_ring) * norm_contact
            + float(w_motion_corridor) * norm_motion
        )
        action_source = "branch_weighted_mixture"
    if np.any(norm_sem > 0.0):
        final_score = final_score + float(w_semantic) * norm_sem
    final_score[~valid] = -np.inf

    def _topk_indices(score: np.ndarray, k: int, eligible: np.ndarray, *, positive: bool = True) -> np.ndarray:
        if k <= 0:
            return np.asarray([], dtype=np.int64)
        score = np.asarray(score, dtype=np.float32).reshape(-1)
        mask = eligible & valid & np.isfinite(score)
        if positive:
            mask &= score > 0.0
        cand = np.flatnonzero(mask)
        if cand.size == 0:
            return np.asarray([], dtype=np.int64)
        if cand.size > k:
            part = cand[np.argpartition(-score[cand], k - 1)[:k]]
        else:
            part = cand
        order = np.lexsort((part, -score[part]))
        return part[order].astype(np.int64, copy=False)

    def _top_fraction_mask(score: np.ndarray, fraction: float) -> np.ndarray:
        mask = np.zeros(n, dtype=bool)
        if fraction <= 0.0:
            return mask
        k = max(1, min(n, int(np.ceil(float(n) * float(fraction)))))
        idx = _topk_indices(score, k, valid, positive=True)
        mask[idx] = True
        return mask

    if attention_active and not np.any(attention_top_mask):
        attn_k = max(1, int(round(float(keep_k) * max(0.0, float(acgtp_attention_budget_ratio)))))
        attention_top_mask[_topk_indices(norm_attn, attn_k, valid, positive=True)] = True
        attention_only_mask = attention_top_mask & ~geom_alignment_mask_for_attn
        attention_candidate_mask = attention_top_mask
        if bool(acgtp_attention_requires_geometry_alignment):
            attention_candidate_mask = attention_candidate_mask & geom_alignment_mask_for_attn

    sem_valid = bool(np.any(norm_sem[valid] > 0.0))
    scene_valid = bool(np.any(norm_scene[valid] > 0.0))
    depth_valid = bool(np.any(norm_de[valid] > 0.0))
    contact_valid = bool(np.any(norm_contact[valid] > 0.0))
    motion_valid = bool(motion_corridor_valid and np.any(norm_motion[valid] > 0.0))

    hard_k_total = min(keep_k, max(1, int(round(float(keep_k) * float(hard_protect_ratio)))))
    branch_specs = [
        ("semantic", sem_valid, float(max(0.0, w_semantic)), norm_sem, 1),
        ("scene", scene_valid, float(max(0.0, w_scene_layout)), norm_scene, 2),
        ("depth", depth_valid, float(max(0.0, w_depth_structure)), norm_de, 3),
        ("contact", contact_valid, float(max(0.0, w_contact_ring)), norm_contact, 4),
        ("motion", motion_valid, float(max(0.0, w_motion_corridor)), norm_motion, 5),
    ]
    active = [(name, weight, score, owner_code) for name, is_active, weight, score, owner_code in branch_specs if is_active]
    if not active:
        active = [("depth", 1.0, norm_de, 3)]

    weight_sum = sum(weight for _, weight, _, _ in active)
    if weight_sum <= 1e-8:
        active = [(name, 1.0, score, owner_code) for name, _, score, owner_code in active]
        weight_sum = float(len(active))

    quotas = {name: 0 for name, _, _, _, _ in branch_specs}
    quota_weights = {name: 0.0 for name, _, _, _, _ in branch_specs}
    raw_quota = {name: float(hard_k_total) * weight / weight_sum for name, weight, _, _ in active}
    for name, weight, _, _ in active:
        quota_weights[name] = float(weight / weight_sum)
        quotas[name] = int(np.floor(raw_quota[name]))
    while sum(quotas.values()) < hard_k_total:
        winner = max((name for name, _, _, _ in active), key=lambda name: (raw_quota[name] - quotas[name], quota_weights[name], name))
        quotas[winner] += 1
    while sum(quotas.values()) > hard_k_total:
        loser = min((name for name, _, _, _ in active if quotas[name] > 0), key=lambda name: (raw_quota[name] - quotas[name], quota_weights[name], name))
        quotas[loser] -= 1

    candidate_counts = {
        "scene": int(np.sum(valid & (norm_scene > 0.0))),
        "depth": int(np.sum(valid & (norm_de > 0.0))),
        "contact": int(np.sum(valid & (norm_contact > 0.0))),
        "motion": int(np.sum(valid & (norm_motion > 0.0))) if motion_valid else 0,
    }
    requested_floors = {
        "scene": max(0, int(min_scene_tokens)),
        "depth": max(0, int(min_depth_tokens)),
        "contact": max(0, int(min_contact_tokens)),
        "motion": max(0, int(min_motion_tokens)),
    }
    floor_targets = {}
    active_names = {name for name, _, _, _ in active}
    for name, requested in requested_floors.items():
        floor_targets[name] = min(requested, candidate_counts.get(name, 0)) if name in active_names else 0
    total_floor = int(sum(floor_targets.values()))
    if total_floor > hard_k_total and total_floor > 0:
        scale = float(hard_k_total) / float(total_floor)
        raw_floor = {name: floor_targets[name] * scale for name in floor_targets}
        scaled = {name: int(np.floor(raw_floor[name])) for name in floor_targets}
        remaining_floor = hard_k_total - int(sum(scaled.values()))
        priority = ("depth", "scene", "contact", "motion")
        for name in sorted(floor_targets, key=lambda key: (raw_floor[key] - scaled[key], key in priority), reverse=True):
            if remaining_floor <= 0:
                break
            if scaled[name] < floor_targets[name]:
                scaled[name] += 1
                remaining_floor -= 1
        floor_targets = scaled

    for name, floor in floor_targets.items():
        if floor > 0:
            quotas[name] = max(int(quotas.get(name, 0)), int(floor))
    while sum(quotas.values()) > hard_k_total:
        reducible = [name for name in quotas if quotas[name] > floor_targets.get(name, 0)]
        if not reducible:
            break
        loser = min(reducible, key=lambda name: (quota_weights.get(name, 0.0), quotas[name] - floor_targets.get(name, 0), name))
        quotas[loser] -= 1
    while sum(quotas.values()) < hard_k_total:
        winners = [name for name, _, _, _ in active if quotas[name] < candidate_counts.get(name, n)]
        if not winners:
            break
        winner = max(winners, key=lambda name: (raw_quota.get(name, 0.0) - quotas.get(name, 0), quota_weights.get(name, 0.0), name))
        quotas[winner] += 1

    selected = np.zeros(n, dtype=bool)
    owner = np.zeros(n, dtype=np.int8)
    allocated = {"semantic": 0, "scene": 0, "depth": 0, "contact": 0, "motion": 0}

    for name, _, score, owner_code in active:
        idx = _topk_indices(score, int(quotas.get(name, 0)), ~selected, positive=True)
        if idx.size > 0:
            selected[idx] = True
            owner[idx] = owner_code
            allocated[name] = int(idx.size)

    scene_fill_candidate_mask = valid & ((raw_fill > 0.5) if raw_fill is not None else (norm_scene > 0.0))
    depth_fill_candidate_mask = _top_fraction_mask(norm_de, 0.75)
    contact_fill_candidate_mask = _top_fraction_mask(norm_contact, 0.50)
    motion_fill_candidate_mask = _top_fraction_mask(norm_motion, 0.50) if motion_valid else np.zeros(n, dtype=bool)
    semantic_fill_candidate_mask = norm_sem > 0.0
    action_fill_candidate_mask = valid & np.isfinite(final_score) & (final_score > 0.0)
    coverage_fill_candidate_mask = valid & (
        semantic_fill_candidate_mask
        | attention_candidate_mask
        | (scene_fill_candidate_mask & ((norm_scene > 0.0) | (norm_de > 0.05)))
        | depth_fill_candidate_mask
        | contact_fill_candidate_mask
        | motion_fill_candidate_mask
        | action_fill_candidate_mask
    )

    fill_k = keep_k - int(np.sum(selected))
    fill_cap_applied = False
    fill_cap_tokens = None
    if constrained_fill_max_tokens is not None:
        fill_cap_tokens = max(0, int(constrained_fill_max_tokens))
        if fill_k > fill_cap_tokens:
            fill_k = fill_cap_tokens
            fill_cap_applied = True
    if fill_k > 0:
        fill_score = final_score + 0.04 * norm_attn + 0.08 * norm_scene + 0.06 * norm_de + 0.05 * norm_contact + 0.04 * norm_motion
        fill_idx = _topk_indices(fill_score, fill_k, coverage_fill_candidate_mask & ~selected, positive=False)
        if fill_idx.size > 0:
            selected[fill_idx] = True
            owner[fill_idx] = 6

    # If generic constrained-fill is capped, recycle the remaining budget back
    # into explicit geometry branches before using unconstrained fallback. This
    # keeps aggressive pruning from replacing scene/depth/contact/motion tokens
    # with arbitrary background tokens just to satisfy keep_k.
    remaining = keep_k - int(np.sum(selected))
    if remaining > 0:
        branch_reserve_mask = valid & ~selected & (
            (norm_sem > 0.0)
            | (norm_scene > 0.0)
            | (norm_de > 0.0)
            | (norm_contact > 0.0)
            | (norm_motion > 0.0)
            | attention_candidate_mask
        )
        branch_reserve_score = (
            np.where(np.isfinite(final_score), final_score, 0.0)
            + 0.04 * norm_attn
            + 0.16 * norm_de
            + 0.14 * norm_scene
            + 0.12 * norm_motion
            + 0.10 * norm_contact
            + 0.08 * norm_sem
        )
        reserve_idx = _topk_indices(branch_reserve_score, remaining, branch_reserve_mask, positive=True)
        if reserve_idx.size > 0:
            branch_scores = np.stack([norm_sem, norm_scene, norm_de, norm_contact, norm_motion], axis=0)
            branch_codes = np.asarray([1, 2, 3, 4, 5], dtype=np.int8)
            best_branch = np.argmax(branch_scores[:, reserve_idx], axis=0)
            selected[reserve_idx] = True
            owner[reserve_idx] = branch_codes[best_branch]

    fallback_reason = None
    remaining = keep_k - int(np.sum(selected))
    if remaining > 0:
        fb_idx = _topk_indices(final_score, remaining, valid & ~selected, positive=False)
        if fb_idx.size > 0:
            selected[fb_idx] = True
            owner[fb_idx] = 7
        fallback_reason = "constrained_fill_insufficient"

    keep_indices = np.flatnonzero(selected).astype(np.int64)
    if keep_indices.size > keep_k:
        trim_score = final_score[keep_indices]
        order = np.lexsort((keep_indices, -trim_score))
        keep_indices = np.sort(keep_indices[order[:keep_k]]).astype(np.int64)
        trim_mask = np.zeros(n, dtype=bool)
        trim_mask[keep_indices] = True
        owner[~trim_mask] = 0
        selected = trim_mask
    final_kept = int(keep_indices.size)

    semantic_count = int(np.sum(owner == 1))
    scene_count = int(np.sum(owner == 2))
    depth_count = int(np.sum(owner == 3))
    contact_count = int(np.sum(owner == 4))
    motion_count = int(np.sum(owner == 5))
    fill_count = int(np.sum(owner == 6))
    fb_count = int(np.sum(owner == 7))
    attention_count = int(np.sum(owner == 8))
    attention_selected_by_final_count = int(np.sum(attention_top_mask & selected))
    attention_only_count = int(np.sum(attention_only_mask))
    branch_sum = semantic_count + scene_count + depth_count + contact_count + motion_count + attention_count + fill_count + fb_count
    geometry_high_mask = valid & (
        (norm_sem > 0.20)
        | (norm_scene > 0.20)
        | (norm_de > 0.20)
        | (norm_contact > 0.20)
        | (norm_motion > 0.20)
    )
    low_geometry_mask = valid & ~geometry_high_mask
    safe_drop_mask = (~selected) & low_geometry_mask & ~attention_top_mask
    high_attention_low_geometry_count = int(np.sum(attention_top_mask & low_geometry_mask))
    high_geometry_low_attention_count = int(np.sum(geometry_high_mask & ~attention_top_mask))
    safe_drop_candidate_count = int(np.sum(safe_drop_mask))

    if bool(minimal_metadata):
        metadata = {
            "strategy": "robot_geo_acgtp_v2",
            "selector_name": "select_acgtp_v2_fast",
            "selector_function_name": "select_acgtp_v2_fast",
            "selection_strategy_name": "robot_geo_acgtp_v2",
            "selection_stage_name": "acgtp_v2_fast_dynamic_constrained_topk",
            "acgtp_v2": True,
            "acgtp_v1": False,
            "acgtp_fast_selector_used": True,
            "acgtp_selector_version": "acgtp_v2_2_fast_path_minimal",
            "acgtp_quota_policy": "fast_weighted_branch_topk",
            "acgtp_fill_policy": "fast_capped_fill_then_branch_reserve",
            "acgtp_motion_corridor_valid": bool(motion_corridor_valid),
            "selected_by_scene_layout_count": scene_count,
            "selected_by_depth_structure_count": depth_count,
            "selected_by_contact_ring_count": contact_count,
            "selected_by_motion_corridor_count": motion_count,
            "selected_by_attention_count": attention_selected_by_final_count,
            "selected_by_constrained_fill_count": fill_count,
            "selected_by_acgtp_fallback_count": fb_count,
            "selected_by_fallback": fb_count,
            "selected_unattributed": 0,
            "acgtp_fallback_used": fb_count > 0,
            "acgtp_fallback_reason": fallback_reason,
            "fallback_used": fb_count > 0,
            "fallback_reason": fallback_reason,
            "final_kept": final_kept,
            "expected_kept": keep_k,
            "K_total": keep_k,
            "acgtp_final_kept": final_kept,
            "acgtp_expected_kept": keep_k,
            "acgtp_actual_keep_ratio": float(final_kept) / float(n),
            "requested_keep_ratio": float(keep_k) / float(n) if n else None,
            "actual_keep_ratio": float(final_kept) / float(n) if n else None,
            "keep_ratio_actual": float(final_kept) / float(n) if n else None,
            "retention_actual": float(final_kept) / float(n) if n else None,
            "effective_keep_count": final_kept,
            "original_token_count": n,
            "num_visual_tokens_original_total": n,
            "num_visual_tokens_kept_total": final_kept,
            "num_visual_tokens_dropped": n - final_kept,
            "selection_time_ms": (time.perf_counter() - select_start) * 1000.0,
        }
        return keep_indices, metadata

    def _mean_max(score: np.ndarray) -> Dict[str, float]:
        vals = np.asarray(score, dtype=np.float32)[valid]
        if vals.size == 0:
            return {"mean": 0.0, "max": 0.0}
        return {"mean": float(np.mean(vals)), "max": float(np.max(vals))}

    scene_stats = _mean_max(norm_scene)
    de_stats = _mean_max(norm_de)
    contact_stats = _mean_max(norm_contact)
    motion_stats = _mean_max(norm_motion)
    acgtp_stats = _mean_max(np.where(np.isfinite(final_score), final_score, 0.0))
    semantic_stats = _mean_max(norm_sem)
    _acr_diag = _action_constraint_result_for_diag or {}
    _scene_diag = _scene_result_for_diag or {}
    _motion_diag = _motion_result_for_diag or {}

    metadata = {
        "strategy": "robot_geo_acgtp_v2",
        "selector_name": "select_acgtp_v2_fast",
        "selector_function_name": "select_acgtp_v2_fast",
        "selection_strategy_name": "robot_geo_acgtp_v2",
        "selection_stage_name": "acgtp_v2_fast_dynamic_constrained_topk",
        "acgtp_v2": True,
        "acgtp_v1": False,
        "acgtp_fast_selector_used": True,
        "acgtp_selector_version": "acgtp_v2_2_fast_path",
        "acgtp_quota_policy": "fast_weighted_branch_topk",
        "acgtp_fill_policy": "fast_capped_fill_then_branch_reserve",
        "strict_fallback_dispatch_used": False,
        "delegated_selector_name": None,
        "fallback_dispatch_to_v1": False,
        "acgtp_w_semantic": float(w_semantic),
        "acgtp_w_scene_layout": float(w_scene_layout),
        "acgtp_w_depth_structure": float(w_depth_structure),
        "acgtp_w_contact_ring": float(w_contact_ring),
        "acgtp_w_motion_corridor": float(w_motion_corridor),
        "acgtp_v2_w_semantic_target": float(w_semantic_target),
        "acgtp_v2_w_semantic_reference": float(w_semantic_reference),
        "acgtp_v2_w_semantic_relation": float(w_semantic_relation),
        "acgtp_v2_w_semantic_goal": float(w_semantic_goal),
        "acgtp_hard_protect_count": int(semantic_count + scene_count + depth_count + contact_count + motion_count),
        "acgtp_hard_protect_ratio": float(hard_protect_ratio),
        "acgtp_hard_protect_ratio_config": float(hard_protect_ratio),
        "acgtp_hard_protect_valid": int(semantic_count + scene_count + depth_count + contact_count + motion_count) <= keep_k,
        "acgtp_scene_quota": int(quotas.get("scene", 0)),
        "acgtp_depth_quota": int(quotas.get("depth", 0)),
        "acgtp_contact_quota": int(quotas.get("contact", 0)),
        "acgtp_motion_quota": int(quotas.get("motion", 0)),
        "acgtp_min_scene_tokens": int(floor_targets.get("scene", 0)),
        "acgtp_min_depth_tokens": int(floor_targets.get("depth", 0)),
        "acgtp_min_contact_tokens": int(floor_targets.get("contact", 0)),
        "acgtp_min_motion_tokens": int(floor_targets.get("motion", 0)),
        "acgtp_constrained_fill_cap_tokens": fill_cap_tokens,
        "acgtp_constrained_fill_cap_applied": bool(fill_cap_applied),
        "acgtp_v2_hard_semantic_quota": int(quotas.get("semantic", 0)),
        "acgtp_scene_quota_weight": float(quota_weights.get("scene", 0.0)),
        "acgtp_depth_quota_weight": float(quota_weights.get("depth", 0.0)),
        "acgtp_contact_quota_weight": float(quota_weights.get("contact", 0.0)),
        "acgtp_motion_quota_weight": float(quota_weights.get("motion", 0.0)),
        "acgtp_scene_allocated": int(allocated.get("scene", 0)),
        "acgtp_depth_allocated": int(allocated.get("depth", 0)),
        "acgtp_contact_allocated": int(allocated.get("contact", 0)),
        "acgtp_motion_allocated": int(allocated.get("motion", 0)),
        "acgtp_motion_corridor_valid": bool(motion_corridor_valid),
        "acgtp_motion_disabled_reason": _motion_diag.get("motion_disabled_reason") if not motion_corridor_valid else None,
        "acgtp_self_core_radius_px": max(0.0, float(contact_ring_inner_px) - 16.0),
        "acgtp_contact_ring_inner_px": float(contact_ring_inner_px),
        "acgtp_contact_ring_outer_px": float(contact_ring_outer_px),
        "acgtp_self_core_token_count": int(np.sum(raw_self_core)),
        "acgtp_self_core_token_ratio": float(np.sum(raw_self_core)) / float(n),
        "acgtp_contact_ring_token_count": int(np.sum(norm_contact > 0.0)),
        "acgtp_contact_ring_token_ratio": float(np.sum(norm_contact > 0.0)) / float(n),
        "acgtp_contact_ring_gated_token_count": int(np.sum(norm_contact > 0.0)),
        "acgtp_contact_ring_valid": bool(contact_valid),
        "acgtp_scene_layout_score_mean": scene_stats["mean"],
        "acgtp_scene_layout_score_max": scene_stats["max"],
        "acgtp_support_plane_token_count": int(_scene_diag.get("support_plane_token_count", int(np.sum(norm_scene > 0.0)))),
        "acgtp_support_plane_candidate_count": int(_scene_diag.get("support_plane_candidate_count", int(np.sum(scene_fill_candidate_mask)))),
        "acgtp_object_component_token_count": int(_scene_diag.get("object_component_token_count", 0)),
        "acgtp_boundary_token_count": int(_scene_diag.get("boundary_token_count", int(np.sum(norm_de > 0.0)))),
        "acgtp_scene_fill_candidate_count": int(np.sum(scene_fill_candidate_mask)),
        "acgtp_scene_fill_candidate_ratio": float(np.sum(scene_fill_candidate_mask)) / float(n),
        "acgtp_coverage_fill_candidate_count": int(np.sum(coverage_fill_candidate_mask)),
        "acgtp_coverage_fill_candidate_ratio": float(np.sum(coverage_fill_candidate_mask)) / float(n),
        "acgtp_scene_support_plane_cap_ratio": float(support_plane_cap_ratio),
        "acgtp_scene_support_plane_cap_used": bool(_scene_diag.get("support_plane_fallback_used", False)),
        "acgtp_scene_support_plane_fallback_used": bool(_scene_diag.get("support_plane_fallback_used", False)),
        "acgtp_scene_support_plane_fallback_reason": _scene_diag.get("support_plane_fallback_reason"),
        "acgtp_scene_object_component_fallback_used": bool(_scene_diag.get("object_component_fallback_used", False)),
        "acgtp_scene_object_component_fallback_reason": _scene_diag.get("object_component_fallback_reason"),
        "acgtp_scene_object_component_num_components": int(_scene_diag.get("object_component_num_components", 0)),
        "acgtp_scene_boundary_fallback_used": bool(_scene_diag.get("boundary_fallback_used", False)),
        "acgtp_scene_boundary_fallback_reason": _scene_diag.get("boundary_fallback_reason"),
        "acgtp_scene_boundary_from_object_count": int(_scene_diag.get("boundary_from_object_count", 0)),
        "acgtp_scene_boundary_from_depth_count": int(_scene_diag.get("boundary_from_depth_count", 0)),
        "acgtp_scene_selected_support_plane_count": None,
        "acgtp_scene_selected_object_component_count": None,
        "acgtp_scene_selected_boundary_count": None,
        "acgtp_scene_selected_relation_count": None,
        "acgtp_scene_support_plane_selected_ratio": None,
        "acgtp_motion_corridor_score_mean": motion_stats["mean"],
        "acgtp_motion_corridor_score_max": motion_stats["max"],
        "acgtp_motion_corridor_length_m": _motion_diag.get("corridor_length_m"),
        "acgtp_motion_norm_m": _motion_diag.get("motion_norm_m", 0.0),
        "acgtp_motion_ema_alpha": _motion_diag.get("ema_alpha", 0.6),
        "acgtp_depth_structure_score_mean": de_stats["mean"],
        "acgtp_depth_structure_score_max": de_stats["max"],
        "acgtp_action_constraint_score_mean": acgtp_stats["mean"],
        "acgtp_action_constraint_score_max": acgtp_stats["max"],
        "acgtp_action_constraint_source": action_source,
        "acgtp_future_action_constraint_enabled": action_constraint_available,
        "acgtp_future_action_constraint_valid": bool(_acr_diag.get("action_constraint_valid", action_constraint_available)),
        "acgtp_future_action_constraint_disabled_reason": _acr_diag.get("action_constraint_disabled_reason"),
        "acgtp_future_action_constraint_score_mean": _acr_diag.get("action_constraint_score_mean", acgtp_stats["mean"]),
        "acgtp_future_action_constraint_score_max": _acr_diag.get("action_constraint_score_max", acgtp_stats["max"]),
        "acgtp_object_side_contact_score_mean": _acr_diag.get("object_side_contact_score_mean"),
        "acgtp_object_side_contact_score_max": _acr_diag.get("object_side_contact_score_max"),
        "acgtp_swept_motion_risk_score_mean": _acr_diag.get("swept_motion_risk_score_mean"),
        "acgtp_swept_motion_risk_score_max": _acr_diag.get("swept_motion_risk_score_max"),
        "acgtp_collision_contact_risk_score_mean": _acr_diag.get("collision_contact_risk_score_mean"),
        "acgtp_collision_contact_risk_score_max": _acr_diag.get("collision_contact_risk_score_max"),
        "acgtp_contact_object_overlap_count": _acr_diag.get("contact_object_overlap_count"),
        "acgtp_robot_self_penalty_count": _acr_diag.get("robot_self_penalty_count"),
        "acgtp_v2_semantic_enabled": bool(semantic_enabled),
        "acgtp_v2_semantic_backend": str(semantic_backend),
        "acgtp_v2_semantic_confidence": float(semantic_confidence),
        "acgtp_v2_semantic_unavailable": bool(semantic_unavailable),
        "acgtp_v2_semantic_fallback_reason": semantic_fallback_reason,
        "acgtp_v2_release_quota": bool(release_semantic_quota_when_unavailable),
        "semantic_available": bool(sem_valid),
        "semantic_confidence": float(semantic_confidence),
        "semantic_unavailable": bool(semantic_unavailable),
        "selected_by_semantic_count": semantic_count,
        "selected_by_semantic_target_count": 0,
        "selected_by_semantic_reference_count": 0,
        "selected_by_semantic_relation_count": 0,
        "selected_by_semantic_goal_count": 0,
        "acgtp_v2_semantic_available": bool(sem_valid),
        "acgtp_v2_semantic_score_mean": semantic_stats["mean"],
        "acgtp_v2_semantic_score_max": semantic_stats["max"],
        "acgtp_v2_semantic_target_token_count": int(np.sum(_norm(_arr(semantic_target_scores)) > 0.0)) if semantic_target_scores is not None else 0,
        "acgtp_v2_semantic_reference_token_count": int(np.sum(_norm(_arr(semantic_reference_scores)) > 0.0)) if semantic_reference_scores is not None else 0,
        "acgtp_v2_semantic_relation_token_count": int(np.sum(_norm(_arr(semantic_relation_scores)) > 0.0)) if semantic_relation_scores is not None else 0,
        "acgtp_v2_semantic_goal_token_count": int(np.sum(_norm(_arr(semantic_goal_scores)) > 0.0)) if semantic_goal_scores is not None else 0,
        "acgtp_v2_semantic_anchor_token_count": int(np.sum(norm_sem > 0.0)),
        "semantic_overlap_with_scene_count": int(np.sum((norm_sem > 0.0) & (norm_scene > 0.0))),
        "semantic_overlap_with_depth_count": int(np.sum((norm_sem > 0.0) & (norm_de > 0.0))),
        "semantic_overlap_with_contact_count": int(np.sum((norm_sem > 0.0) & (norm_contact > 0.0))),
        "semantic_overlap_with_motion_count": int(np.sum((norm_sem > 0.0) & (norm_motion > 0.0))),
        "attention_backend": acgtp_attention_backend if attention_active else "none",
        "attention_source": acgtp_attention_source if attention_active else "none",
        "attention_available": bool(attention_active),
        "attention_confidence": float(acgtp_attention_confidence) if attention_active else 0.0,
        "attention_quota_released": not bool(attention_active),
        "attention_only_token_count": attention_only_count,
        "attention_selected_by_final_count": attention_selected_by_final_count,
        "attention_top_count": int(np.sum(attention_top_mask)),
        "safe_drop_candidate_count": safe_drop_candidate_count,
        "high_attention_low_geometry_count": high_attention_low_geometry_count,
        "high_geometry_low_attention_count": high_geometry_low_attention_count,
        "acgtp_attention_enabled": bool(acgtp_attention_enabled),
        "acgtp_attention_backend": acgtp_attention_backend if attention_active else "none",
        "acgtp_attention_source": acgtp_attention_source if attention_active else "none",
        "acgtp_attention_available": bool(attention_active),
        "acgtp_attention_confidence": float(acgtp_attention_confidence) if attention_active else 0.0,
        "acgtp_attention_quota_released": not bool(attention_active),
        "acgtp_attention_top_count": int(np.sum(attention_top_mask)),
        "acgtp_attention_candidate_count": int(np.sum(attention_candidate_mask)),
        "acgtp_attention_only_token_count": attention_only_count,
        "selected_by_scene_layout_count": scene_count,
        "selected_by_depth_structure_count": depth_count,
        "selected_by_contact_ring_count": contact_count,
        "selected_by_motion_corridor_count": motion_count,
        "selected_by_attention_count": attention_selected_by_final_count,
        "selected_by_constrained_fill_count": fill_count,
        "selected_by_acgtp_fallback_count": fb_count,
        "selected_by_phase1": scene_count + depth_count,
        "selected_by_phase2": contact_count + motion_count,
        "selected_by_fill": fill_count,
        "selected_by_fallback": fb_count,
        "selected_unattributed": 0,
        "overlap_scene_depth_count": int(np.sum((norm_scene > 0.0) & (norm_de > 0.0))),
        "overlap_scene_contact_count": int(np.sum((norm_scene > 0.0) & (norm_contact > 0.0))),
        "overlap_scene_motion_count": int(np.sum((norm_scene > 0.0) & (norm_motion > 0.0))),
        "overlap_contact_motion_count": int(np.sum((norm_contact > 0.0) & (norm_motion > 0.0))),
        "overlap_depth_contact_count": int(np.sum((norm_de > 0.0) & (norm_contact > 0.0))),
        "overlap_depth_motion_count": int(np.sum((norm_de > 0.0) & (norm_motion > 0.0))),
        "acgtp_branch_accounting_valid": branch_sum == final_kept,
        "acgtp_branch_sum": branch_sum,
        "acgtp_branch_sum_error": abs(branch_sum - final_kept),
        "branch_accounting_valid": branch_sum == final_kept,
        "branch_sum_equals_kept": branch_sum == final_kept,
        "acgtp_fallback_used": fb_count > 0,
        "acgtp_fallback_reason": fallback_reason,
        "fallback_used": fb_count > 0,
        "fallback_reason": fallback_reason,
        "final_kept": final_kept,
        "expected_kept": keep_k,
        "K_total": keep_k,
        "acgtp_final_kept": final_kept,
        "acgtp_expected_kept": keep_k,
        "acgtp_actual_keep_ratio": float(final_kept) / float(n),
        "acgtp_constrained_fill_mask": None,
        "selection_time_ms": (time.perf_counter() - select_start) * 1000.0,
    }

    return keep_indices, finalize_selection_debug_info(
        metadata,
        selector_function_name="select_acgtp_v2_fast",
        strategy="robot_geo_acgtp_v2",
        keep_indices=keep_indices,
        num_tokens=n,
        keep_count=keep_k,
        scores=np.where(np.isfinite(final_score), final_score, 0.0),
        requested_keep_ratio=float(keep_k) / float(n) if n else None,
        fallback_used=fb_count > 0,
        fallback_reason=fallback_reason,
    )

# ---------------------------------------------------------------------------
# Source: pruning/selectors/acgtp_v2_full_selector.py
# ---------------------------------------------------------------------------
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..legacy.acgtp_v1 import select_acgtp_v1
from .utils import finalize_selection_debug_info

def select_acgtp_v2(
    scene_layout_scores: Optional[np.ndarray],
    depth_edge_scores: Optional[np.ndarray],
    contact_ring_scores: Optional[np.ndarray],
    motion_corridor_scores: Optional[np.ndarray],
    semantic_anchor_scores: Optional[np.ndarray],
    semantic_target_scores: Optional[np.ndarray],
    semantic_reference_scores: Optional[np.ndarray],
    semantic_relation_scores: Optional[np.ndarray],
    semantic_goal_scores: Optional[np.ndarray],
    valid_mask: Optional[np.ndarray],
    keep_k: int,
    constrained_fill_mask: Optional[np.ndarray] = None,
    token_u: Optional[np.ndarray] = None,
    token_v: Optional[np.ndarray] = None,
    grid_h: int = 16,
    grid_w: int = 16,
    w_scene_layout: float = 0.25,
    w_depth_structure: float = 0.20,
    w_contact_ring: float = 0.20,
    w_motion_corridor: float = 0.15,
    w_semantic: float = 0.20,
    hard_protect_ratio: float = 0.60,
    motion_corridor_valid: bool = False,
    self_core_mask: Optional[np.ndarray] = None,
    contact_ring_inner_px: float = 24.0,
    contact_ring_outer_px: float = 48.0,
    contact_requires_edge_or_object: bool = True,
    depth_edge_score_for_gate: Optional[np.ndarray] = None,
    _motion_result_for_diag: Optional[Dict[str, Any]] = None,
    _scene_result_for_diag: Optional[Dict[str, Any]] = None,
    action_constraint_scores: Optional[np.ndarray] = None,
    _action_constraint_result_for_diag: Optional[Dict[str, Any]] = None,
    support_plane_cap_ratio: float = 0.30,
    # ── P16 semantic branch params ──────────────────────────────────────────
    semantic_enabled: bool = False,
    semantic_backend: str = "none",
    semantic_confidence: float = 0.0,
    semantic_unavailable: bool = True,
    semantic_fallback_reason: Optional[str] = None,
    release_semantic_quota_when_unavailable: bool = True,
    w_semantic_target: float = 1.0,
    w_semantic_reference: float = 0.7,
    w_semantic_relation: float = 0.5,
    w_semantic_goal: float = 0.9,
    target_cap_ratio: float = 0.25,
    reference_cap_ratio: float = 0.20,
    relation_cap_ratio: float = 0.15,
    hard_semantic_ratio: float = 0.20,
    # ── parsed instruction terms (for metrics) ──────────────────────────────
    parsed_target_terms: Optional[list] = None,
    parsed_reference_terms: Optional[list] = None,
    parsed_relation_terms: Optional[list] = None,
    instruction_is_meaningful: bool = False,
    # ── Task 4-5: scene-layout branch from semantic backend ─────────────────
    # These are passed through from compute_task_semantic_anchors / SemanticLayoutResult
    scene_layout_branch_active: bool = False,
    scene_layout_available: bool = False,
    scene_layout_confidence: float = 0.0,
    target_mask_count: int = 0,
    reference_mask_count: int = 0,
    relation_mask_count: int = 0,
    layout_anchor_mask_count: int = 0,
    scene_layout_indices: Optional[List[int]] = None,
    # ── P16-Extension: Attention task-relevance branch ─────────────────────
    # This is a GATED CANDIDATE signal only — inspired by VLA-Cache / VLA-IAP / VLA-Pruner.
    # It can NEVER replace action-constrained geometry and can NEVER cause a global
    # attention top-k. It enters the constrained union only, and only for tokens that
    # also satisfy geometry alignment (>= 1 geometry branch high).
    acgtp_attention_enabled: bool = False,
    acgtp_attention_backend: str = "none",
    acgtp_attention_min_confidence: float = 0.0,
    acgtp_attention_requires_geometry_alignment: bool = True,
    acgtp_attention_budget_ratio: float = 0.10,
    acgtp_attention_task_relevance_score: Optional[np.ndarray] = None,
    acgtp_attention_task_relevance_mask: Optional[np.ndarray] = None,
    acgtp_attention_source: str = "none",
    acgtp_attention_available: bool = False,
    acgtp_attention_confidence: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """ACGTP-v2: Task-Conditioned Action-Constrained Geometry Token Protection.

    Extends ACGTP-v1 with a Task-Semantic Anchor Branch. The five-branch priority
    order (highest to lowest) is:

        semantic > scene_layout > depth_structure > contact_ring > motion_corridor

    When ``semantic_unavailable=True`` (the default, no visual detector available):
      - semantic_confidence = 0.0
      - semantic branch produces zero scores
      - If release_semantic_quota_when_unavailable=True: semantic quota is NOT
        pre-allocated and is available for geometry branches.
      - Geometry branches (scene_layout, depth, contact, motion) handle all selection.
    """

    def _to_1d(arr) -> Optional[np.ndarray]:
        if arr is None:
            return None
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim == 0:
            return a.reshape(-1)
        if a.ndim == 2:
            return a.reshape(-1)
        return a.reshape(-1)

    raw_scene = _to_1d(scene_layout_scores)
    raw_de = _to_1d(depth_edge_scores)
    raw_contact = _to_1d(contact_ring_scores)
    raw_motion = _to_1d(motion_corridor_scores)
    raw_sem_anchor = _to_1d(semantic_anchor_scores)
    raw_sem_target = _to_1d(semantic_target_scores)
    raw_sem_ref = _to_1d(semantic_reference_scores)
    raw_sem_rel = _to_1d(semantic_relation_scores)
    raw_sem_goal = _to_1d(semantic_goal_scores)
    raw_valid = _to_1d(valid_mask) if valid_mask is not None else None
    raw_fill = _to_1d(constrained_fill_mask) if constrained_fill_mask is not None else None
    raw_self_core = _to_1d(self_core_mask) if self_core_mask is not None else None
    raw_gate = _to_1d(depth_edge_score_for_gate)

    # ── STRICT FALLBACK: literal dispatch to v1 ─────────────────────────────────
    # When semantic AND attention branches are both unavailable, v2 is a pure
    # wrapper around v1. We call v1 directly so keep_indices are bit-for-bit
    # identical — not merely "close" (Jaccard >= 0.90) but exact equal.
    #
    # strict_fallback is True when ALL of the following hold:
    #   - semantic_enabled == False  OR  semantic_backend == "none"
    #   - semantic_unavailable == True
    #   - acgtp_attention_enabled == False  OR  acgtp_attention_available == False
    #
    # In this path v2 does NOT run any of its normalization, scoring, hard-
    # protect, constrained-fill, or fallback logic. It returns v1's result
    # with v2-only diagnostic fields appended.
    _strict_fallback = (
        (not semantic_enabled or semantic_backend == "none")
        and semantic_unavailable
        and (not acgtp_attention_enabled or not acgtp_attention_available)
    )

    if _strict_fallback:
        # Map v2 geometry-only args to v1's exact signature.
        # Weights must match v1 defaults so the acgtp_scores mixture is identical.
        _v1_indices, _v1_meta = select_acgtp_v1(
            scene_layout_scores=scene_layout_scores,
            depth_edge_scores=depth_edge_scores,
            contact_ring_scores=contact_ring_scores,
            motion_corridor_scores=motion_corridor_scores,
            valid_mask=valid_mask,
            keep_k=keep_k,
            constrained_fill_mask=constrained_fill_mask,
            token_u=token_u,
            token_v=token_v,
            grid_h=grid_h,
            grid_w=grid_w,
            w_scene_layout=w_scene_layout,
            w_depth_structure=w_depth_structure,
            w_contact_ring=w_contact_ring,
            w_motion_corridor=w_motion_corridor,
            hard_protect_ratio=hard_protect_ratio,
            motion_corridor_valid=motion_corridor_valid,
            self_core_mask=self_core_mask,
            contact_ring_inner_px=contact_ring_inner_px,
            contact_ring_outer_px=contact_ring_outer_px,
            contact_requires_edge_or_object=contact_requires_edge_or_object,
            depth_edge_score_for_gate=depth_edge_score_for_gate,
            _motion_result_for_diag=_motion_result_for_diag,
            _scene_result_for_diag=_scene_result_for_diag,
            action_constraint_scores=action_constraint_scores,
            _action_constraint_result_for_diag=_action_constraint_result_for_diag,
            support_plane_cap_ratio=support_plane_cap_ratio,
        )

        # Build v2-style metadata by copying v1 fields and layering v2 diagnostics.
        _final_kept = len(_v1_indices)
        _branch_sum = (
            _v1_meta.get("selected_by_scene_layout_count", 0)
            + _v1_meta.get("selected_by_depth_structure_count", 0)
            + _v1_meta.get("selected_by_contact_ring_count", 0)
            + _v1_meta.get("selected_by_motion_corridor_count", 0)
            + _v1_meta.get("selected_by_constrained_fill_count", 0)
            + _v1_meta.get("selected_by_acgtp_fallback_count", 0)
        )
        _v2_meta: Dict[str, Any] = dict(_v1_meta)
        _v2_meta.update({
            # ── Version / strategy flags ─────────────────────────────────────
            "strategy": "robot_geo_acgtp_v2",
            "selector_function_name": "select_acgtp_v2",
            "selection_strategy_name": "robot_geo_acgtp_v2",
            "selection_stage_name": "acgtp_v2_strict_fallback",
            "acgtp_v2": True,
            "acgtp_v1": False,

            # ── Strict fallback dispatch record ────────────────────────────────
            "strict_fallback_dispatch_used": True,
            "delegated_selector_name": "select_acgtp_v1",
            "fallback_dispatch_to_v1": True,
            # Override selector_name so audit reports show "select_acgtp_v2" as the top-level
            # selector (delegation to select_acgtp_v1 is recorded in delegated_selector_name).
            "selector_name": "select_acgtp_v2",

            # ── P16 semantic branch (inactive / unavailable) ──────────────────
            "acgtp_v2_semantic_enabled": semantic_enabled,
            "acgtp_v2_semantic_backend": semantic_backend,
            "acgtp_v2_semantic_confidence": semantic_confidence,
            "acgtp_v2_semantic_unavailable": semantic_unavailable,
            "acgtp_v2_semantic_fallback_reason": semantic_fallback_reason,
            "acgtp_v2_release_quota": True,
            "semantic_available": False,
            "semantic_confidence": 0.0,
            "semantic_unavailable": semantic_unavailable,
            "selected_by_semantic_count": 0,
            "selected_by_semantic_target_count": 0,
            "selected_by_semantic_reference_count": 0,
            "selected_by_semantic_relation_count": 0,
            "selected_by_semantic_goal_count": 0,
            "acgtp_v2_semantic_available": False,
            "acgtp_v2_semantic_score_mean": 0.0,
            "acgtp_v2_semantic_score_max": 0.0,
            "acgtp_v2_semantic_target_token_count": 0,
            "acgtp_v2_semantic_reference_token_count": 0,
            "acgtp_v2_semantic_relation_token_count": 0,
            "acgtp_v2_semantic_goal_token_count": 0,
            "acgtp_v2_semantic_anchor_token_count": 0,
            "semantic_overlap_with_scene_count": 0,
            "semantic_overlap_with_depth_count": 0,
            "semantic_overlap_with_contact_count": 0,
            "semantic_overlap_with_motion_count": 0,

            # ── P16 branch weights (echoed from v1, may differ from v2 defaults) ──
            "acgtp_w_semantic": float(w_semantic),
            "acgtp_v2_w_semantic_target": float(w_semantic_target),
            "acgtp_v2_w_semantic_reference": float(w_semantic_reference),
            "acgtp_v2_w_semantic_relation": float(w_semantic_relation),
            "acgtp_v2_w_semantic_goal": float(w_semantic_goal),

            # ── Scene-layout branch (inactive in strict fallback) ─────────────
            "scene_layout_branch_active": False,
            "scene_layout_available": False,
            "scene_layout_confidence": 0.0,
            "scene_layout_branch_quota": 0,
            "selected_by_scene_layout_count": _v1_meta.get("selected_by_scene_layout_count", 0),
            "target_mask_count": 0,
            "reference_mask_count": 0,
            "relation_mask_count": 0,
            "layout_anchor_mask_count": 0,
            "scene_layout_indices": _v1_meta.get("scene_layout_indices", []),
            "overlap_scene_geometry_count": 0,

            # ── Attention branch (inactive in strict fallback) ───────────────
            "acgtp_attention_enabled": acgtp_attention_enabled,
            "acgtp_attention_backend": acgtp_attention_backend,
            "acgtp_attention_source": acgtp_attention_source,
            "acgtp_attention_available": False,
            "acgtp_attention_confidence": 0.0,
            "acgtp_attention_quota_released": True,
            "acgtp_attention_requires_geometry_alignment": acgtp_attention_requires_geometry_alignment,
            "acgtp_attention_budget_ratio": acgtp_attention_budget_ratio,
            "acgtp_attention_top_count": 0,
            "acgtp_attention_candidate_count": 0,
            "acgtp_attention_only_token_count": 0,
            "attention_only_token_count": 0,
            "attention_selected_by_final_count": 0,
            "attn_alignment_verified": False,

            # ── Accounting (mirrors v1) ────────────────────────────────────────
            "acgtp_branch_accounting_valid": (_branch_sum == _final_kept),
            "acgtp_branch_sum": _branch_sum,
            "acgtp_branch_sum_error": abs(_branch_sum - _final_kept),
            "branch_accounting_valid": (_branch_sum == _final_kept),
            "branch_sum_equals_kept": (_branch_sum == _final_kept),
            "final_kept": _final_kept,
            "expected_kept": keep_k,
            "K_total": keep_k,
            "acgtp_fallback_used": _v1_meta.get("acgtp_fallback_used", False),
            "acgtp_fallback_reason": _v1_meta.get("acgtp_fallback_reason"),
        })

        _diag_num_tokens = int(
            _v1_meta.get("num_visual_tokens_original")
            or _v1_meta.get("num_visual_tokens_original_total")
            or (raw_valid.size if raw_valid is not None else 0)
            or (raw_scene.size if raw_scene is not None else 0)
            or (raw_de.size if raw_de is not None else 0)
            or keep_k
        )

        # Finalize with ACGTP-v2 identity
        _v2_meta = finalize_selection_debug_info(
            _v2_meta,
            selector_function_name="select_acgtp_v2",
            strategy="robot_geo_acgtp_v2",
            keep_indices=_v1_indices,
            num_tokens=_diag_num_tokens,
            keep_count=keep_k,
            scores=None,
            requested_keep_ratio=float(keep_k) / float(_diag_num_tokens) if _diag_num_tokens else None,
            fallback_used=bool(_v1_meta.get("acgtp_fallback_used", False)),
            fallback_reason=_v1_meta.get("acgtp_fallback_reason"),
        )
        return _v1_indices, _v2_meta

    # ── V2 native path (semantic or attention branch available) ─────────────
    n = keep_k
    for arr in (raw_scene, raw_de, raw_contact, raw_motion, raw_sem_anchor):
        if arr is not None:
            n = max(n, int(arr.shape[0]))
    n = max(n, keep_k)

    if raw_valid is not None and raw_valid.shape[0] == n:
        valid = raw_valid.astype(bool)
    else:
        valid = np.ones(n, dtype=bool)

    keep_k = int(max(0, min(keep_k, n)))

    # ── Normalization helpers ─────────────────────────────────────────────
    def _norm(scores: np.ndarray) -> np.ndarray:
        a = np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
        finite = a[valid][np.isfinite(a[valid])]
        lo = float(np.min(finite)) if finite.size > 0 else 0.0
        hi = float(np.max(finite)) if finite.size > 0 else 0.0
        out = np.zeros(n, dtype=np.float32)
        if hi - lo > 1e-8:
            out[valid] = (a[valid] - lo) / (hi - lo)
        else:
            out[valid] = 0.0
        out[~valid] = -np.inf
        return out

    def _norm_safe(scores: np.ndarray) -> np.ndarray:
        if scores is None:
            return np.zeros(n, dtype=np.float32)
        a = np.nan_to_num(np.asarray(scores, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
        out = np.zeros(n, dtype=np.float32)
        v = a[valid]
        if v.size == 0:
            return out
        lo, hi = float(np.min(v)), float(np.max(v))
        if hi - lo > 1e-8:
            out[valid] = (a[valid] - lo) / (hi - lo)
        return out

    def _topk_order(scores: np.ndarray, k: int) -> np.ndarray:
        a = np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
        adj = np.where(valid, a, -np.inf)
        adj_neg = -adj
        order = np.lexsort((np.arange(n), adj_neg))
        result = []
        for idx in order:
            idx_i = int(idx)
            if valid[idx_i] and adj[idx_i] > -np.inf * 0.5:
                result.append(idx_i)
                if len(result) >= k:
                    break
        return np.asarray(result, dtype=np.int64)

    # ── Normalize all branch scores ──────────────────────────────────────
    norm_scene = _norm(raw_scene) if raw_scene is not None else np.zeros(n, dtype=np.float32)
    norm_de = _norm(raw_de) if raw_de is not None else np.zeros(n, dtype=np.float32)
    norm_contact = _norm(raw_contact) if raw_contact is not None else np.zeros(n, dtype=np.float32)
    norm_motion = _norm(raw_motion) if raw_motion is not None else np.zeros(n, dtype=np.float32)
    norm_sem_anchor = _norm(raw_sem_anchor) if raw_sem_anchor is not None else np.zeros(n, dtype=np.float32)

    norm_scene_s = _norm_safe(raw_scene)
    norm_de_s = _norm_safe(raw_de)
    norm_contact_s = _norm_safe(raw_contact)
    norm_motion_s = _norm_safe(raw_motion)
    norm_sem_anchor_s = _norm_safe(raw_sem_anchor)

    # ── Action-constraint score (mixture) ────────────────────────────────
    # When semantic is unavailable, zero out its weight to prevent zero-valued
    # semantic scores from causing subtle tie-breaking differences vs v1.
    effective_w_semantic = float(w_semantic)
    if semantic_unavailable or not semantic_enabled:
        effective_w_semantic = 0.0

    acgtp_scores = (
        effective_w_semantic * norm_sem_anchor_s
        + w_scene_layout * norm_scene_s
        + w_depth_structure * norm_de_s
        + w_contact_ring * norm_contact_s
        + w_motion_corridor * norm_motion_s
    )
    raw_action_constraint = _to_1d(action_constraint_scores)
    action_constraint_available = raw_action_constraint is not None and raw_action_constraint.shape[0] == n and np.any(raw_action_constraint > 0)
    if action_constraint_available:
        norm_action_constraint_s = _norm_safe(raw_action_constraint)
        acgtp_scores = norm_action_constraint_s
    else:
        norm_action_constraint_s = np.zeros(n, dtype=np.float32)
    acgtp_scores[~valid] = -np.inf

    # ── Contact ring gate ─────────────────────────────────────────────────
    norm_contact_gated = norm_contact.copy()
    if contact_requires_edge_or_object and raw_gate is not None:
        gate_scores = _norm_safe(raw_gate)
        gate_threshold = float(np.percentile(gate_scores[valid], 60)) if np.any(valid) else 0.0
        no_gate = gate_scores < gate_threshold
        norm_contact_gated[no_gate] = 0.0

    # ── Motion corridor gate ─────────────────────────────────────────────
    norm_motion_gated = norm_motion.copy()
    norm_motion_s_gated = norm_motion_s.copy()
    if not motion_corridor_valid:
        norm_motion_gated = np.full(n, -np.inf, dtype=np.float32)
        norm_motion_s_gated = np.zeros(n, dtype=np.float32)

    # ── Semantic branch validity ──────────────────────────────────────────
    semantic_is_active = (
        semantic_enabled
        and not semantic_unavailable
        and np.any(norm_sem_anchor > 0)
    )
    semantic_quota_released = semantic_unavailable and release_semantic_quota_when_unavailable

    # ── Geometry branch validity ──────────────────────────────────────────
    scene_valid = np.any((norm_scene > 0) & valid)
    de_valid = np.any((norm_de > 0) & valid)
    contact_valid = np.any((norm_contact_gated > 0) & valid)
    motion_valid_branch = motion_corridor_valid and np.any((norm_motion_gated > 0) & valid)

    # ── Hard-protect budget allocation ────────────────────────────────────
    hard_k_total = max(1, int(round(keep_k * hard_protect_ratio)))
    hard_k_total = min(hard_k_total, keep_k)

    if semantic_is_active and not semantic_quota_released:
        sem_hard_k = max(1, int(round(hard_k_total * hard_semantic_ratio)))
    else:
        sem_hard_k = 0

    geom_hard_k = hard_k_total - sem_hard_k

    def _allocate_weighted_branch_quotas(total_budget: int) -> Tuple[Dict[str, int], Dict[str, float]]:
        branch_specs = [
            ("scene", bool(scene_valid), float(max(0.0, w_scene_layout))),
            ("depth", bool(de_valid), float(max(0.0, w_depth_structure))),
            ("contact", bool(contact_valid), float(max(0.0, w_contact_ring))),
            ("motion", bool(motion_valid_branch), float(max(0.0, w_motion_corridor))),
        ]
        quotas = {name: 0 for name, _, _ in branch_specs}
        norm_weights = {name: 0.0 for name, _, _ in branch_specs}
        active = [(name, weight) for name, is_active, weight in branch_specs if is_active]
        if total_budget <= 0 or not active:
            return quotas, norm_weights

        weight_sum = sum(weight for _, weight in active)
        if weight_sum <= 1e-8:
            active = [(name, 1.0) for name, _ in active]
            weight_sum = float(len(active))

        raw_quota = {name: (weight / weight_sum) * float(total_budget) for name, weight in active}
        for name, weight in active:
            norm_weights[name] = float(weight / weight_sum)

        min_quota = 1 if total_budget >= len(active) else 0
        for name, _ in active:
            quotas[name] = max(min_quota, int(np.floor(raw_quota[name])))

        while sum(quotas.values()) > total_budget:
            candidates = [name for name, _ in active if quotas[name] > min_quota]
            if not candidates:
                break
            loser = min(candidates, key=lambda name: (raw_quota[name] - quotas[name], norm_weights[name], name))
            quotas[loser] -= 1

        while sum(quotas.values()) < total_budget:
            candidates = [name for name, _ in active]
            winner = max(candidates, key=lambda name: (raw_quota[name] - quotas[name], norm_weights[name], name))
            quotas[winner] += 1

        return quotas, norm_weights

    quota_map, quota_weight_map = _allocate_weighted_branch_quotas(geom_hard_k)
    scene_quota = quota_map["scene"]
    de_quota = quota_map["depth"]
    contact_quota = quota_map["contact"]
    motion_quota = quota_map["motion"]

    # Per-category caps for semantic branch
    target_cap_k = max(1, int(round(n * target_cap_ratio)))
    reference_cap_k = max(1, int(round(n * reference_cap_ratio)))
    relation_cap_k = max(1, int(round(n * relation_cap_ratio)))

    # ── Get candidate lists per branch ────────────────────────────────────
    sem_candidates = [i for i in _topk_order(norm_sem_anchor, sem_hard_k).tolist() if norm_sem_anchor[i] > 0]
    scene_candidates = [i for i in _topk_order(norm_scene, geom_hard_k).tolist() if norm_scene[i] > 0]
    de_candidates = [i for i in _topk_order(norm_de, geom_hard_k).tolist() if norm_de[i] > 0]
    contact_candidates = [i for i in _topk_order(norm_contact_gated, geom_hard_k).tolist() if norm_contact_gated[i] > 0]
    motion_candidates = [i for i in _topk_order(norm_motion_gated, geom_hard_k).tolist() if norm_motion_gated[i] > 0]

    # ── Overlap-aware constrained union ───────────────────────────────────
    # Priority: semantic > scene > depth > contact > motion
    selected: set[int] = set()
    selected_owner: Dict[int, str] = {}
    allocated: Dict[str, int] = {
        "semantic": 0, "scene": 0, "depth": 0, "contact": 0, "motion": 0
    }

    def _fill_branch(branch_name: str, candidates: list, quota: int) -> None:
        nonlocal selected, allocated
        for idx_i in candidates:
            if allocated[branch_name] >= quota:
                break
            if idx_i not in selected:
                selected.add(idx_i)
                selected_owner[idx_i] = branch_name
                allocated[branch_name] += 1

    def _fill_semantic_with_cap(
        candidates: list, total_quota: int,
        target_scores, ref_scores, rel_scores, goal_scores,
    ) -> None:
        nonlocal selected, allocated
        for cat_name, cat_scores, cap_k in [
            ("target", raw_sem_target, target_cap_k),
            ("reference", raw_sem_ref, reference_cap_k),
            ("relation", raw_sem_rel, relation_cap_k),
            ("goal", raw_sem_goal, relation_cap_k),
        ]:
            if cat_scores is None:
                continue
            cat_scores_1d = _norm_safe(cat_scores)
            cat_order = _topk_order(cat_scores_1d, cap_k)
            for idx_i in cat_order:
                if allocated["semantic"] >= total_quota:
                    break
                if idx_i not in selected:
                    selected.add(idx_i)
                    selected_owner[int(idx_i)] = "semantic"
                    allocated["semantic"] += 1

    # Priority 1: semantic
    if semantic_is_active:
        _fill_semantic_with_cap(
            sem_candidates, sem_hard_k,
            raw_sem_target, raw_sem_ref, raw_sem_rel, raw_sem_goal,
        )
    # Priority 2-5: geometry branches
    _fill_branch("scene", scene_candidates, scene_quota)
    _fill_branch("depth", de_candidates, de_quota)
    _fill_branch("contact", contact_candidates, contact_quota)
    _fill_branch("motion", motion_candidates, motion_quota)

    # ── Attention task-relevance gated-candidate integration ─────────────────
    # VLA-Cache / VLA-IAP / VLA-Pruner inspired:
    #   • Attention is a DIAGNOSTIC / OPTIONAL PROTECTION signal only.
    #   • It CANNOT replace action-constrained geometry.
    #   • It CANNOT cause selector to degrade into global attention top-k.
    #   • Attention-aligned tokens enter the CONSTRAINED FILL pool (union phase),
    #     NOT the hard-selected set. This preserves accounting validity.
    #   • Attention-only tokens (high attention, geometry weak) are diagnostic only.
    #   • Attention budget can NEVER reduce depth/contact/motion minimum quotas.
    #
    # VLA-Cache:  safe_drop = low_scene AND low_depth AND low_contact
    #             AND low_motion AND low_attention
    # VLA-IAP:    attention may focus background; needs geometry alignment gate
    # VLA-Pruner: semantic prefill attention ≠ action relevance; union only, no fusion
    #
    attn_candidates: List[int] = []
    attn_only_candidates: List[int] = []
    attn_geometry_aligned_candidates: List[int] = []
    attn_quota_released: bool = True
    attn_selected_by_final_count: int = 0
    attn_top_count: int = 0
    attn_alignment_verified: bool = False
    attn_candidate_source: str = acgtp_attention_source

    if acgtp_attention_enabled:
        raw_attn = _to_1d(acgtp_attention_task_relevance_score)
        raw_attn_mask = _to_1d(acgtp_attention_task_relevance_mask)

        attn_available = (
            acgtp_attention_available
            and acgtp_attention_confidence >= acgtp_attention_min_confidence
            and raw_attn is not None
            and np.any(raw_attn > 0)
        )

        if attn_available:
            attn_quota_released = False

            # Build per-geometry-branch high masks for alignment check (VLA-IAP gate).
            # Use normalized >= 0.50 threshold — equivalent to the fixed raw-score 0.50
            # threshold used in compute_safe_drop_diagnostic. This ensures the selector
            # and the diagnostic use consistent "high geometry" criteria when scores
            # follow a bimodal [LOW, HIGH] distribution (e.g. LOW: [0,0.15], HIGH: [0.70,1.0]).
            GEOM_HIGH_NORM = 0.50
            scene_high = (norm_scene >= GEOM_HIGH_NORM) if norm_scene is not None else np.zeros(n, dtype=bool)
            depth_high = (norm_de >= GEOM_HIGH_NORM) if norm_de is not None else np.zeros(n, dtype=bool)
            contact_high = (norm_contact_gated >= GEOM_HIGH_NORM) if norm_contact_gated is not None else np.zeros(n, dtype=bool)
            motion_high = (norm_motion_gated >= GEOM_HIGH_NORM) if norm_motion_gated is not None else np.zeros(n, dtype=bool)
            geom_alignment_mask = scene_high | depth_high | contact_high | motion_high

            # Normalise attention scores
            attn_norm = _norm_safe(raw_attn) if raw_attn is not None else np.zeros(n, dtype=np.float32)

            # Top-k attention mask from backend
            if raw_attn_mask is not None:
                attn_top_mask = np.asarray(raw_attn_mask, dtype=bool).reshape(-1)
            else:
                attn_top_mask = (attn_norm > 0)

            attn_top_count = int(np.sum(attn_top_mask))

            # Classify each attention-high token
            for idx_i in range(n):
                if not valid[idx_i]:
                    continue
                if idx_i >= len(attn_top_mask) or not attn_top_mask[idx_i]:
                    continue
                if acgtp_attention_requires_geometry_alignment and not geom_alignment_mask[idx_i]:
                    # VLA-IAP: attention high but geometry weak — diagnostic record only
                    attn_only_candidates.append(idx_i)
                else:
                    # VLA-Cache: attention high AND geometry-aligned → constrained union candidate
                    attn_geometry_aligned_candidates.append(idx_i)

            # Sort by attention score descending (stable by index)
            attn_geometry_aligned_candidates.sort(
                key=lambda i: (-float(attn_norm[i]) if i < len(attn_norm) else -np.inf, i)
            )
            attn_only_candidates.sort(
                key=lambda i: (-float(attn_norm[i]) if i < len(attn_norm) else -np.inf, i)
            )
            if len(attn_geometry_aligned_candidates) <= len(attn_only_candidates):
                # VLA-IAP-style low-alignment case: attention is more likely
                # background-biased than action-constrained, so keep it as a
                # diagnostic signal only and release its fill candidates.
                attn_geometry_aligned_candidates = []
                attn_quota_released = True
            attn_candidates = attn_geometry_aligned_candidates
            attn_alignment_verified = True
        else:
            attn_quota_released = True
    else:
        attn_quota_released = True

    # Hard-selected set (geometry only — attention is handled in fill phase)
    hard_selected = set(selected)
    hard_selected_list = sorted(hard_selected, key=lambda i: (-acgtp_scores[i] if valid[i] else -np.inf, i))

    # ── Constrained fill (extended with attention union candidates) ────────────
    # VLA-Pruner: attention branch enters constrained union only — no global top-k.
    # Order: (1) attention-aligned candidates, (2) geometry fill candidates, (3) safe fallback.
    remaining_k = keep_k - len(hard_selected)
    fallback_used = False
    fallback_reason: Optional[str] = None
    fallback_count = 0
    fill_selected: set[int] = set()

    def _top_fraction_mask(scores: np.ndarray, cap_ratio: float) -> np.ndarray:
        mask = np.zeros(n, dtype=bool)
        if scores is None or cap_ratio <= 0.0:
            return mask
        cap_k = max(1, int(np.ceil(float(n) * float(cap_ratio))))
        for idx_i in _topk_order(scores, cap_k).tolist():
            if valid[idx_i] and scores[idx_i] > 0.0:
                mask[idx_i] = True
        return mask

    if raw_fill is not None and raw_fill.shape[0] == n:
        scene_fill_candidate_mask = valid & (raw_fill > 0.5)
    else:
        scene_fill_candidate_mask = valid & (norm_scene_s > 0.0)
    depth_fill_candidate_mask = _top_fraction_mask(norm_de_s, 0.75)
    contact_fill_candidate_mask = _top_fraction_mask(norm_contact_gated, 0.50)
    motion_fill_candidate_mask = _top_fraction_mask(norm_motion_s_gated, 0.50) if motion_valid_branch else np.zeros(n, dtype=bool)
    coverage_fill_candidate_mask = valid & (
        scene_fill_candidate_mask
        | (norm_sem_anchor_s > 0.0)
        | depth_fill_candidate_mask
        | contact_fill_candidate_mask
        | motion_fill_candidate_mask
    )

    if remaining_k > 0:
        def _raw_fill_positive(idx_i: int) -> bool:
            return raw_fill is not None and idx_i < raw_fill.shape[0] and raw_fill[idx_i] > 0.5

        def _fill_membership(idx_i: int) -> Dict[str, bool]:
            return {
                "semantic": bool(norm_sem_anchor_s[idx_i] > 0.0),
                "scene": bool(_raw_fill_positive(idx_i) or norm_scene_s[idx_i] > 0.0),
                "depth": bool(depth_fill_candidate_mask[idx_i]),
                "contact": bool(contact_fill_candidate_mask[idx_i]),
                "motion": bool(motion_fill_candidate_mask[idx_i]),
            }

        selected_cells: set[Tuple[int, int]] = set()
        if token_u is not None and token_v is not None:
            try:
                tu = np.asarray(token_u).reshape(-1)
                tv = np.asarray(token_v).reshape(-1)
                for idx_i in hard_selected:
                    if idx_i < tu.shape[0] and idx_i < tv.shape[0]:
                        selected_cells.add((int(tu[idx_i]), int(tv[idx_i])))
            except Exception:
                tu = None
                tv = None
        else:
            tu = None
            tv = None

        fill_deficit = {
            "semantic": max(0, sem_hard_k - allocated["semantic"]),
            "scene": max(0, scene_quota - allocated["scene"]),
            "depth": max(0, de_quota - allocated["depth"]),
            "contact": max(0, contact_quota - allocated["contact"]),
            "motion": max(0, motion_quota - allocated["motion"]),
        }

        def _coverage_fill_score(idx_i: int) -> Tuple[float, int]:
            membership = _fill_membership(idx_i)
            branch_bonus = 0.0
            if membership["semantic"] and semantic_is_active:
                deficit_bonus = float(fill_deficit["semantic"]) / float(max(1, keep_k))
                branch_bonus += float(hard_semantic_ratio) * (1.0 + deficit_bonus)
            for branch_name in ("scene", "depth", "contact", "motion"):
                if not membership[branch_name]:
                    continue
                deficit_bonus = float(fill_deficit.get(branch_name, 0)) / float(max(1, keep_k))
                branch_bonus += float(quota_weight_map.get(branch_name, 0.0)) * (1.0 + deficit_bonus)

            spatial_bonus = 0.0
            if tu is not None and tv is not None and idx_i < tu.shape[0] and idx_i < tv.shape[0]:
                cell = (int(tu[idx_i]), int(tv[idx_i]))
                spatial_bonus = 1.0 if cell not in selected_cells else 0.0

            base = float(acgtp_scores[idx_i]) if np.isfinite(acgtp_scores[idx_i]) else 0.0
            score = base + 0.25 * branch_bonus + 0.05 * spatial_bonus
            return score, -idx_i

        # (1) Attention-aligned candidates enter fill pool (VLA-Cache gated union)
        # These are deduplicated against hard_selected.
        attn_fill_added = 0
        for idx_i in attn_geometry_aligned_candidates:
            if len(fill_selected) >= remaining_k:
                break
            if idx_i not in hard_selected and idx_i not in fill_selected:
                fill_selected.add(idx_i)
                selected_owner[int(idx_i)] = "fill"
                attn_fill_added += 1

        # (2) Geometry fill candidates
        fill_candidates = {
            i for i in range(n)
            if valid[i] and i not in hard_selected and i not in fill_selected and any(_fill_membership(i).values())
        }
        while fill_candidates and len(fill_selected) < remaining_k:
            best_idx = max(fill_candidates, key=_coverage_fill_score)
            fill_candidates.remove(best_idx)
            fill_selected.add(best_idx)
            selected_owner[int(best_idx)] = "fill"
            if tu is not None and tv is not None and best_idx < tu.shape[0] and best_idx < tv.shape[0]:
                selected_cells.add((int(tu[best_idx]), int(tv[best_idx])))

    remaining_k -= len(fill_selected)

    # ── Safe fallback ─────────────────────────────────────────────────────
    fallback_fill: set[int] = set()
    if remaining_k > 0:
        fallback_used = True
        fb_order = sorted(
            [i for i in range(n) if valid[i] and i not in hard_selected and i not in fill_selected],
            key=lambda i: (-acgtp_scores[i], i)
        )
        for idx_i in fb_order:
            if len(fallback_fill) >= remaining_k:
                break
            fallback_fill.add(idx_i)
            selected_owner[int(idx_i)] = "fallback"
            fallback_count += 1

    # ── Final union ──────────────────────────────────────────────────────
    all_selected = hard_selected | fill_selected | fallback_fill
    final_order = (
        hard_selected_list
        + sorted(fill_selected, key=lambda i: (-acgtp_scores[i], i))
        + sorted(fallback_fill, key=lambda i: (-acgtp_scores[i], i))
    )
    final_order = [i for i in final_order if i in all_selected]

    if len(final_order) > keep_k:
        final_order = final_order[:keep_k]

    keep_indices = np.sort(np.asarray(final_order, dtype=np.int64))
    keep_indices = keep_indices[(keep_indices >= 0) & (keep_indices < n)]
    keep_indices = np.unique(keep_indices)
    if len(keep_indices) > keep_k:
        keep_indices = keep_indices[:keep_k]

    final_kept = len(keep_indices)
    selected_set = set(int(i) for i in keep_indices)

    # ── Non-overlapping attribution ───────────────────────────────────────
    sem_only: set[int] = set()
    scene_only: set[int] = set()
    de_only: set[int] = set()
    contact_only: set[int] = set()
    motion_only: set[int] = set()
    fill_only: set[int] = set()
    fb_only: set[int] = set()

    for idx_i in sorted(selected_set):
        owner = selected_owner.get(int(idx_i))
        if owner == "semantic":
            sem_only.add(idx_i)
        elif owner == "scene":
            scene_only.add(idx_i)
        elif owner == "depth":
            de_only.add(idx_i)
        elif owner == "contact":
            contact_only.add(idx_i)
        elif owner == "motion":
            motion_only.add(idx_i)
        elif owner == "fill" or idx_i in fill_selected:
            fill_only.add(idx_i)
        elif owner == "fallback" or idx_i in fallback_fill:
            fb_only.add(idx_i)
        else:
            # Defensive accounting: a selected token should always have an
            # owner, but keep accounting conservative if future code inserts
            # tokens without tagging them.
            fill_only.add(idx_i)

    # ── Attention branch attribution ──────────────────────────────────────────
    # VLA-Pruner: attention does NOT get its own attribution category.
    # Count how many final tokens were filled by attention-aligned candidates.
    # attn_only_candidates (high attn, geometry weak) are diagnostic only.
    attn_only_count = sum(1 for i in selected_set if i in set(attn_only_candidates))
    attn_aligned_in_final = sum(1 for i in selected_set if i in set(attn_geometry_aligned_candidates))
    attn_selected_by_final_count = attn_aligned_in_final

    sem_count = len(sem_only)
    scene_count = len(scene_only)
    de_count = len(de_only)
    contact_count = len(contact_only)
    motion_count = len(motion_only)
    fill_count = len(fill_only)
    fb_count = len(fb_only)

    # Attention is NOT a separate branch in the accounting sum — it feeds into fill_only.
    # This preserves branch_sum == final_kept for accounting validity.
    branch_sum = sem_count + scene_count + de_count + contact_count + motion_count + fill_count + fb_count
    accounting_valid = (branch_sum == final_kept)

    # ── Semantic category sub-attribution ─────────────────────────────────
    sem_target_only: set[int] = set()
    sem_ref_only: set[int] = set()
    sem_rel_only: set[int] = set()
    sem_goal_only: set[int] = set()

    if raw_sem_target is not None:
        ts = _norm_safe(raw_sem_target)
        for idx_i in sem_only:
            if idx_i < n and ts[idx_i] > 0:
                sem_target_only.add(idx_i)
    if raw_sem_ref is not None:
        rs = _norm_safe(raw_sem_ref)
        for idx_i in sem_only:
            if idx_i < n and rs[idx_i] > 0:
                sem_ref_only.add(idx_i)
    if raw_sem_rel is not None:
        rls = _norm_safe(raw_sem_rel)
        for idx_i in sem_only:
            if idx_i < n and rls[idx_i] > 0:
                sem_rel_only.add(idx_i)
    if raw_sem_goal is not None:
        gs = _norm_safe(raw_sem_goal)
        for idx_i in sem_only:
            if idx_i < n and gs[idx_i] > 0:
                sem_goal_only.add(idx_i)

    # Semantic overlap with geometry branches
    sem_overlap_scene = len(sem_only & set(scene_candidates))
    sem_overlap_depth = len(sem_only & set(de_candidates))
    sem_overlap_contact = len(sem_only & set(contact_candidates))
    sem_overlap_motion = len(sem_only & set(motion_candidates))

    # ── Scene layout per-component attribution ───────────────────────────
    scene_selected_diag: Dict[str, Any] = {
        "acgtp_scene_selected_support_plane_count": 0,
        "acgtp_scene_selected_object_component_count": 0,
        "acgtp_scene_selected_boundary_count": 0,
        "acgtp_scene_selected_relation_count": None,
        "acgtp_scene_selected_residual_fill_count": None,
        "acgtp_scene_residual_fill_token_count": None,
        "acgtp_scene_residual_fill_token_count_computed": False,
        "acgtp_scene_support_plane_selected_ratio": 0.0,
        "acgtp_scene_relation_token_count": None,
        "acgtp_scene_relation_token_count_computed": False,
    }
    if _scene_result_for_diag is not None and all_selected:
        try:
            sp_cand_scores = _scene_result_for_diag.get("support_plane_candidate_scores")
            object_scores = _scene_result_for_diag.get("object_component_scores")
            boundary_scores = _scene_result_for_diag.get("boundary_scores")
            scene_layout_scores = _scene_result_for_diag.get("scene_layout_scores")

            n_local = n
            for arr in (sp_cand_scores, object_scores, boundary_scores, scene_layout_scores):
                if arr is not None:
                    try:
                        arr_n = int(np.asarray(arr, dtype=object).size)
                        n_local = max(n_local, arr_n)
                    except (TypeError, ValueError):
                        pass

            is_support = np.zeros(n_local, dtype=bool)
            is_object = np.zeros(n_local, dtype=bool)
            is_boundary = np.zeros(n_local, dtype=bool)
            is_residual = np.zeros(n_local, dtype=bool)

            if sp_cand_scores is not None:
                sp_arr = np.asarray(sp_cand_scores, dtype=np.float32).reshape(-1)
                if sp_arr.shape[0] == n_local:
                    is_support = sp_arr > 0.0
            if object_scores is not None:
                o = np.asarray(object_scores, dtype=np.float32).reshape(-1)
                if o.shape[0] == n_local:
                    is_object = o > 0.0
            if boundary_scores is not None:
                b = np.asarray(boundary_scores, dtype=np.float32).reshape(-1)
                if b.shape[0] == n_local:
                    is_boundary = b > 0.0
            if scene_layout_scores is not None:
                sl = np.asarray(scene_layout_scores, dtype=np.float32).reshape(-1)
                if sl.shape[0] == n_local:
                    scene_relevant = sl > 0.0
                    is_residual = scene_relevant & ~is_support & ~is_object & ~is_boundary

            sel_sp = sum(1 for i in all_selected if i < n_local and is_support[i])
            sel_oc = sum(1 for i in all_selected if i < n_local and is_object[i])
            sel_bn = sum(1 for i in all_selected if i < n_local and is_boundary[i])
            sel_residual = sum(1 for i in all_selected if i < n_local and is_residual[i])

            scene_selected_diag["acgtp_scene_selected_support_plane_count"] = sel_sp
            scene_selected_diag["acgtp_scene_selected_object_component_count"] = sel_oc
            scene_selected_diag["acgtp_scene_selected_boundary_count"] = sel_bn
            total_sp = int(np.sum(is_support))
            if np.any(is_residual):
                scene_selected_diag["acgtp_scene_selected_relation_count"] = sel_residual
                scene_selected_diag["acgtp_scene_residual_fill_token_count"] = int(np.sum(is_residual))
                scene_selected_diag["acgtp_scene_residual_fill_token_count_computed"] = True
                scene_selected_diag["acgtp_scene_relation_token_count"] = int(np.sum(is_residual))
                scene_selected_diag["acgtp_scene_relation_token_count_computed"] = True
            if total_sp > 0:
                scene_selected_diag["acgtp_scene_support_plane_selected_ratio"] = float(sel_sp) / float(total_sp)
        except Exception:
            pass

    # ── Overlap diagnostics ───────────────────────────────────────────────
    overlap_scene_de = len(set(scene_candidates) & set(de_candidates))
    overlap_scene_contact = len(set(scene_candidates) & set(contact_candidates))
    overlap_scene_motion = len(set(scene_candidates) & set(motion_candidates))
    overlap_contact_motion = len(set(contact_candidates) & set(motion_candidates))
    overlap_de_contact = len(set(de_candidates) & set(contact_candidates))
    overlap_de_motion = len(set(de_candidates) & set(motion_candidates))

    # ── Per-branch score statistics ───────────────────────────────────────
    def _score_stats(scores: np.ndarray) -> Dict[str, float]:
        if scores is None:
            return {"mean": 0.0, "max": 0.0}
        a = np.nan_to_num(np.asarray(scores, dtype=np.float32).reshape(-1), nan=0.0)
        v = a[valid]
        return {
            "mean": float(np.mean(v)) if v.size else 0.0,
            "max": float(np.max(v)) if v.size else 0.0,
        }

    sem_stats = _score_stats(norm_sem_anchor_s)
    scene_stats = _score_stats(norm_scene_s)
    de_stats = _score_stats(norm_de_s)
    contact_stats = _score_stats(norm_contact_gated)
    motion_stats = _score_stats(norm_motion_gated)
    acgtp_stats = _score_stats(acgtp_scores)

    # ── Scene layout module diagnostics ───────────────────────────────────
    _sp_cand_cnt = 0
    _sp_total_cnt = 0
    _oc_cnt = 0
    _bn_cnt = 0
    _sp_fallback_used = False
    _sp_fallback_reason: Optional[str] = None
    _oc_fallback_used = False
    _oc_fallback_reason: Optional[str] = None
    _bn_fallback_used = False
    _bn_fallback_reason: Optional[str] = None
    _oc_num_components = 0
    _bn_from_object = 0
    _bn_from_depth = 0

    if _scene_result_for_diag is not None:
        _sp_cand_cnt = _scene_result_for_diag.get("support_plane_candidate_count", 0)
        _sp_total_cnt = _scene_result_for_diag.get("support_plane_token_count", 0)
        _oc_cnt = _scene_result_for_diag.get("object_component_token_count", 0)
        _bn_cnt = _scene_result_for_diag.get("boundary_token_count", 0)
        _sp_fallback_used = bool(_scene_result_for_diag.get("support_plane_fallback_used", False))
        _sp_fallback_reason = _scene_result_for_diag.get("support_plane_fallback_reason")
        _oc_fallback_used = bool(_scene_result_for_diag.get("object_component_fallback_used", False))
        _oc_fallback_reason = _scene_result_for_diag.get("object_component_fallback_reason")
        _bn_fallback_used = bool(_scene_result_for_diag.get("boundary_fallback_used", False))
        _bn_fallback_reason = _scene_result_for_diag.get("boundary_fallback_reason")
        _oc_num_components = _scene_result_for_diag.get("object_component_num_components", 0)
        _bn_from_object = _scene_result_for_diag.get("boundary_from_object_count", 0)
        _bn_from_depth = _scene_result_for_diag.get("boundary_from_depth_count", 0)
    else:
        _sp_total_cnt = int(np.sum(norm_scene > 0)) if norm_scene is not None else 0
        _sp_cand_cnt = _sp_total_cnt
        _oc_cnt = int(np.sum(norm_scene > 0.5)) if norm_scene is not None else 0
        _bn_cnt = int(np.sum(norm_de > 0)) if norm_de is not None else 0

    fill_candidate_count = int(np.sum(scene_fill_candidate_mask))
    coverage_fill_candidate_count = int(np.sum(coverage_fill_candidate_mask))
    fill_candidate_ratio = float(fill_candidate_count) / float(n) if n > 0 else 0.0
    coverage_fill_candidate_ratio = float(coverage_fill_candidate_count) / float(n) if n > 0 else 0.0

    # Self-core diagnostics
    self_core_count = int(np.sum(raw_self_core)) if raw_self_core is not None else 0
    self_core_ratio = float(self_core_count) / float(n) if n > 0 else 0.0
    contact_ring_total = int(np.sum(norm_contact_gated > 0)) if norm_contact_gated is not None else 0
    contact_ring_ratio = float(contact_ring_total) / float(n) if n > 0 else 0.0
    contact_ring_gated_total = contact_ring_total

    # Motion corridor diagnostics
    _motion_norm = 0.0
    _motion_disabled_reason: Optional[str] = None
    _ema_alpha_config = 0.6
    if _motion_result_for_diag is not None:
        _motion_norm = _motion_result_for_diag.get("motion_norm_m", 0.0)
        _motion_disabled_reason = _motion_result_for_diag.get("motion_disabled_reason")
        _ema_alpha_config = _motion_result_for_diag.get("ema_alpha", 0.6)
    if not motion_corridor_valid and _motion_disabled_reason is None:
        _motion_disabled_reason = "motion_corridor_signal_unreliable"

    _acr_diag = _action_constraint_result_for_diag or {}

    # Semantic token counts
    sem_target_count = int(np.sum(norm_sem_anchor_s > 0)) if raw_sem_target is not None else 0
    sem_ref_count = int(np.sum(_norm_safe(raw_sem_ref) > 0)) if raw_sem_ref is not None else 0
    sem_rel_count = int(np.sum(_norm_safe(raw_sem_rel) > 0)) if raw_sem_rel is not None else 0
    sem_goal_count = int(np.sum(_norm_safe(raw_sem_goal) > 0)) if raw_sem_goal is not None else 0
    sem_anchor_count = int(np.sum(norm_sem_anchor_s > 0))

    # Constrained fill mask for debug recording
    constrained_fill_str = None
    if raw_fill is not None:
        try:
            import json
            constrained_fill_str = json.dumps([int(x) for x in raw_fill[:min(n, 512)]])
        except Exception:
            constrained_fill_str = str(list(raw_fill[:min(n, 512)]))

    metadata: Dict[str, Any] = {
        "strategy": "robot_geo_acgtp_v2",
        "selector_function_name": "select_acgtp_v2",
        "selection_strategy_name": "robot_geo_acgtp_v2",
        "selection_stage_name": "acgtp_v2_semantic_augmented_constrained_union",

        # Strategy flags
        "acgtp_v2": True,
        "acgtp_v1": False,
        "acgtp_selector_version": "acgtp_v2_1_weighted_coverage",
        "acgtp_quota_policy": "weight_proportional_release_invalid",
        "acgtp_fill_policy": "coverage_aware_constrained",

        # P16 semantic branch
        "acgtp_v2_semantic_enabled": semantic_enabled,
        "acgtp_v2_semantic_backend": semantic_backend,
        "acgtp_v2_semantic_confidence": semantic_confidence,
        "acgtp_v2_semantic_unavailable": semantic_unavailable,
        "acgtp_v2_semantic_fallback_reason": semantic_fallback_reason,
        "acgtp_v2_release_quota": semantic_quota_released,
        "acgtp_v2_parsed_instruction_meaningful": instruction_is_meaningful,
        "acgtp_v2_parsed_target_terms": parsed_target_terms or [],
        "acgtp_v2_parsed_reference_terms": parsed_reference_terms or [],
        "acgtp_v2_parsed_relation_terms": parsed_relation_terms or [],

        # Branch weights
        "acgtp_w_scene_layout": float(w_scene_layout),
        "acgtp_w_depth_structure": float(w_depth_structure),
        "acgtp_w_contact_ring": float(w_contact_ring),
        "acgtp_w_motion_corridor": float(w_motion_corridor),
        "acgtp_w_semantic": float(w_semantic),
        "acgtp_v2_w_semantic_target": float(w_semantic_target),
        "acgtp_v2_w_semantic_reference": float(w_semantic_reference),
        "acgtp_v2_w_semantic_relation": float(w_semantic_relation),
        "acgtp_v2_w_semantic_goal": float(w_semantic_goal),

        # Hard protect
        "acgtp_hard_protect_count": len(hard_selected),
        "acgtp_hard_protect_ratio": float(hard_protect_ratio),
        "acgtp_hard_protect_valid": len(hard_selected) <= keep_k,
        "acgtp_v2_hard_semantic_quota": sem_hard_k,
        "acgtp_v2_target_cap_k": target_cap_k,
        "acgtp_v2_reference_cap_k": reference_cap_k,
        "acgtp_v2_relation_cap_k": relation_cap_k,

        # Branch quotas
        "acgtp_scene_quota": scene_quota,
        "acgtp_depth_quota": de_quota,
        "acgtp_contact_quota": contact_quota,
        "acgtp_motion_quota": motion_quota,
        "acgtp_scene_quota_weight": quota_weight_map["scene"],
        "acgtp_depth_quota_weight": quota_weight_map["depth"],
        "acgtp_contact_quota_weight": quota_weight_map["contact"],
        "acgtp_motion_quota_weight": quota_weight_map["motion"],
        "acgtp_scene_allocated": allocated["scene"],
        "acgtp_depth_allocated": allocated["depth"],
        "acgtp_contact_allocated": allocated["contact"],
        "acgtp_motion_allocated": allocated["motion"],
        "acgtp_semantic_allocated": allocated["semantic"],

        # Motion corridor gate status
        "acgtp_motion_corridor_valid": bool(motion_corridor_valid),
        "acgtp_motion_disabled_reason": _motion_disabled_reason,

        # Self-core / contact ring
        "acgtp_self_core_radius_px": float(contact_ring_inner_px) - 16.0,
        "acgtp_contact_ring_inner_px": float(contact_ring_inner_px),
        "acgtp_contact_ring_outer_px": float(contact_ring_outer_px),
        "acgtp_self_core_token_count": self_core_count,
        "acgtp_self_core_token_ratio": self_core_ratio,
        "acgtp_contact_ring_token_count": contact_ring_total,
        "acgtp_contact_ring_token_ratio": contact_ring_ratio,
        "acgtp_contact_ring_gated_token_count": contact_ring_gated_total,
        "acgtp_contact_ring_valid": True,

        # Scene layout diagnostics
        "acgtp_scene_layout_score_mean": scene_stats["mean"],
        "acgtp_scene_layout_score_max": scene_stats["max"],
        "acgtp_support_plane_token_count": _sp_total_cnt,
        "acgtp_support_plane_candidate_count": _sp_cand_cnt,
        "acgtp_object_component_token_count": _oc_cnt,
        "acgtp_boundary_token_count": _bn_cnt,
        "acgtp_scene_fill_candidate_count": fill_candidate_count,
        "acgtp_scene_fill_candidate_ratio": fill_candidate_ratio,
        "acgtp_coverage_fill_candidate_count": coverage_fill_candidate_count,
        "acgtp_coverage_fill_candidate_ratio": coverage_fill_candidate_ratio,
        "acgtp_scene_support_plane_cap_ratio": float(support_plane_cap_ratio),
        "acgtp_scene_support_plane_cap_used": _sp_fallback_used,
        "acgtp_scene_support_plane_fallback_used": _sp_fallback_used,
        "acgtp_scene_support_plane_fallback_reason": _sp_fallback_reason,
        "acgtp_scene_object_component_fallback_used": _oc_fallback_used,
        "acgtp_scene_object_component_fallback_reason": _oc_fallback_reason,
        "acgtp_scene_object_component_num_components": _oc_num_components,
        "acgtp_scene_boundary_fallback_used": _bn_fallback_used,
        "acgtp_scene_boundary_fallback_reason": _bn_fallback_reason,
        "acgtp_scene_boundary_from_object_count": _bn_from_object,
        "acgtp_scene_boundary_from_depth_count": _bn_from_depth,

        # Scene layout per-component selected attribution (with corrected naming)
        "acgtp_scene_selected_support_plane_count": scene_selected_diag["acgtp_scene_selected_support_plane_count"],
        "acgtp_scene_selected_object_component_count": scene_selected_diag["acgtp_scene_selected_object_component_count"],
        "acgtp_scene_selected_boundary_count": scene_selected_diag["acgtp_scene_selected_boundary_count"],
        "acgtp_scene_selected_relation_count": scene_selected_diag["acgtp_scene_selected_relation_count"],
        # P16 corrected: geometry-only leftover is NOT semantic relation
        "acgtp_scene_selected_residual_fill_count": scene_selected_diag.get("acgtp_scene_selected_residual_fill_count"),
        "acgtp_scene_residual_fill_token_count": scene_selected_diag.get("acgtp_scene_residual_fill_token_count"),
        "acgtp_scene_residual_fill_token_count_computed": scene_selected_diag.get("acgtp_scene_residual_fill_token_count_computed", False),
        "acgtp_scene_support_plane_selected_ratio": scene_selected_diag["acgtp_scene_support_plane_selected_ratio"],
        "acgtp_scene_relation_token_count": scene_selected_diag.get("acgtp_scene_relation_token_count"),
        "acgtp_scene_relation_token_count_computed": scene_selected_diag.get("acgtp_scene_relation_token_count_computed", False),

        # Semantic branch diagnostics
        "acgtp_v2_semantic_score_mean": sem_stats["mean"],
        "acgtp_v2_semantic_score_max": sem_stats["max"],
        "acgtp_v2_semantic_target_token_count": sem_target_count,
        "acgtp_v2_semantic_reference_token_count": sem_ref_count,
        "acgtp_v2_semantic_relation_token_count": sem_rel_count,
        "acgtp_v2_semantic_goal_token_count": sem_goal_count,
        "acgtp_v2_semantic_anchor_token_count": sem_anchor_count,
        "selected_by_semantic_target_count": len(sem_target_only),
        "selected_by_semantic_reference_count": len(sem_ref_only),
        "selected_by_semantic_relation_count": len(sem_rel_only),
        "selected_by_semantic_goal_count": len(sem_goal_only),
        "selected_by_semantic_count": sem_count,
        "semantic_overlap_with_scene_count": sem_overlap_scene,
        "semantic_overlap_with_depth_count": sem_overlap_depth,
        "semantic_overlap_with_contact_count": sem_overlap_contact,
        "semantic_overlap_with_motion_count": sem_overlap_motion,
        "selected_by_scene_residual_fill_count": fill_count,

        # ── Task 3-5: Scene-layout branch (Task 4) ───────────────────────
        # semantic_available mirrors semantic_unavailable inverted
        "acgtp_v2_semantic_available": (
            semantic_enabled
            and not semantic_unavailable
            and np.any(norm_sem_anchor > 0)
        ),
        # Scene-layout branch state — use passed-in values from semantic backend
        "scene_layout_branch_active": scene_layout_branch_active,
        "scene_layout_available": scene_layout_available,
        "scene_layout_confidence": float(scene_layout_confidence),
        "scene_layout_branch_quota": sem_hard_k,
        # Mask counts from semantic/grounding backend (Task 5)
        "target_mask_count": target_mask_count,
        "reference_mask_count": reference_mask_count,
        "relation_mask_count": relation_mask_count,
        "layout_anchor_mask_count": layout_anchor_mask_count,
        # Indices selected by scene-layout branch (Task 4)
        "scene_layout_indices": list(scene_layout_indices) if scene_layout_indices else [],
        # Overlap with geometry (Task 5)
        "overlap_scene_geometry_count": overlap_scene_de + overlap_scene_contact + overlap_scene_motion,

        # Motion corridor diagnostics
        "acgtp_motion_corridor_score_mean": motion_stats["mean"],
        "acgtp_motion_corridor_score_max": motion_stats["max"],
        "acgtp_motion_norm_m": _motion_norm,
        "acgtp_motion_ema_alpha": _ema_alpha_config,

        # Depth structure diagnostics
        "acgtp_depth_structure_score_mean": de_stats["mean"],
        "acgtp_depth_structure_score_max": de_stats["max"],

        # Action constraint score
        "acgtp_action_constraint_score_mean": acgtp_stats["mean"],
        "acgtp_action_constraint_score_max": acgtp_stats["max"],
        "acgtp_action_constraint_source": "future_action_constraint" if action_constraint_available else "branch_weighted_mixture",
        "acgtp_future_action_constraint_enabled": bool(action_constraint_available),
        "acgtp_future_action_constraint_valid": bool(_acr_diag.get("action_constraint_valid", action_constraint_available)),
        "acgtp_future_action_constraint_disabled_reason": _acr_diag.get("action_constraint_disabled_reason"),
        "acgtp_future_action_constraint_score_mean": _acr_diag.get("action_constraint_score_mean", acgtp_stats["mean"]),
        "acgtp_future_action_constraint_score_max": _acr_diag.get("action_constraint_score_max", acgtp_stats["max"]),
        "acgtp_object_side_contact_score_mean": _acr_diag.get("object_side_contact_score_mean"),
        "acgtp_object_side_contact_score_max": _acr_diag.get("object_side_contact_score_max"),
        "acgtp_swept_motion_risk_score_mean": _acr_diag.get("swept_motion_risk_score_mean"),
        "acgtp_swept_motion_risk_score_max": _acr_diag.get("swept_motion_risk_score_max"),
        "acgtp_collision_contact_risk_score_mean": _acr_diag.get("collision_contact_risk_score_mean"),
        "acgtp_collision_contact_risk_score_max": _acr_diag.get("collision_contact_risk_score_max"),
        "acgtp_contact_object_overlap_count": _acr_diag.get("contact_object_overlap_count"),
        "acgtp_robot_self_penalty_count": _acr_diag.get("robot_self_penalty_count"),

        # Branch attribution (non-overlapping)
        "selected_by_scene_layout_count": scene_count,
        "selected_by_depth_structure_count": de_count,
        "selected_by_contact_ring_count": contact_count,
        "selected_by_motion_corridor_count": motion_count,
        "selected_by_constrained_fill_count": fill_count,
        "selected_by_acgtp_fallback_count": fb_count,

        # Aliases
        "selected_by_phase1": sem_count + scene_count + de_count,
        "selected_by_phase2": contact_count + motion_count,
        "selected_by_fill": fill_count,
        "selected_by_fallback": fb_count,
        "selected_unattributed": 0,

        # Overlap diagnostics
        "overlap_scene_depth_count": overlap_scene_de,
        "overlap_scene_contact_count": overlap_scene_contact,
        "overlap_scene_motion_count": overlap_scene_motion,
        "overlap_contact_motion_count": overlap_contact_motion,
        "overlap_depth_contact_count": overlap_de_contact,
        "overlap_depth_motion_count": overlap_de_motion,

        # ── P16-Extension: Attention task-relevance branch metadata ───────────
        # Inspired by VLA-Cache / VLA-IAP / VLA-Pruner.
        # Attention is a GATED CANDIDATE only — feeds into fill phase, not hard protect.
        # attention_selected_by_final_count counts tokens added from attention-aligned candidates.
        # attention_only_token_count (diagnostic) counts high-attn / geometry-weak tokens.
        "acgtp_attention_enabled": acgtp_attention_enabled,
        "acgtp_attention_backend": acgtp_attention_backend,
        "acgtp_attention_source": attn_candidate_source,
        "acgtp_attention_available": acgtp_attention_available,
        "acgtp_attention_confidence": acgtp_attention_confidence,
        "acgtp_attention_quota_released": attn_quota_released,
        "acgtp_attention_requires_geometry_alignment": acgtp_attention_requires_geometry_alignment,
        "acgtp_attention_budget_ratio": acgtp_attention_budget_ratio,
        "acgtp_attention_top_count": attn_top_count,
        "acgtp_attention_candidate_count": len(attn_geometry_aligned_candidates),
        "acgtp_attention_only_token_count": len(attn_only_candidates),
        "attention_only_token_count": len(attn_only_candidates),
        "geometry_only_token_count": 0,  # computed post-hoc by audit
        "attention_selected_by_final_count": attn_selected_by_final_count,
        "attn_alignment_verified": attn_alignment_verified,
        # Strict fallback dispatch fields — present in ALL v2 paths for audit clarity.
        # strict_fallback_dispatch_used=False for the native v2 path (semantic/attention available).
        # delegated_selector_name=None for the native v2 path (no delegation).
        "strict_fallback_dispatch_used": False,
        "delegated_selector_name": None,
        # selector_name appears in all v2 paths for audit clarity.
        "selector_name": "select_acgtp_v2",
        # Native v2 did not delegate to v1. Keep this false so audit logs do
        # not confuse "semantic disabled" with a literal v1 dispatch.
        "fallback_dispatch_to_v1": False,
        "semantic_unavailable": semantic_unavailable,
        "semantic_confidence": semantic_confidence,
        "selected_by_semantic_count": sem_count,
        "selected_by_scene_layout_count": scene_count,

        # ── Task 4: scene_layout_indices (tokens selected by scene-layout branch) ───
        # When semantic is active (mock/real detector), scene_layout branch is the
        # semantic branch. In fallback mode (semantic unavailable), the geometry
        # scene_layout branch handles scene relevance.
        "scene_layout_indices": sorted(list(scene_only)),

        # Accounting
        "acgtp_branch_accounting_valid": accounting_valid,
        "acgtp_branch_sum": branch_sum,
        "acgtp_branch_sum_error": abs(branch_sum - final_kept),
        "branch_accounting_valid": accounting_valid,
        "branch_sum_equals_kept": accounting_valid,

        # Fallback
        "acgtp_fallback_used": fb_count > 0,
        "acgtp_fallback_reason": fallback_reason,

        # Debug maps
        "acgtp_constrained_fill_mask": constrained_fill_str,
        "acgtp_scene_layout_scores": None,
        "acgtp_contact_ring_scores": None,
        "acgtp_motion_corridor_scores": None,
        "acgtp_action_constraint_scores": None,
        "acgtp_robot_self_core_mask": None,

        # Final stats
        "final_kept": final_kept,
        "expected_kept": keep_k,
        "K_total": keep_k,
        "grid_shape": [grid_h, grid_w],
    }

    metadata = finalize_selection_debug_info(
        metadata,
        selector_function_name="select_acgtp_v2",
        strategy="robot_geo_acgtp_v2",
        keep_indices=keep_indices,
        num_tokens=n,
        keep_count=keep_k,
        scores=acgtp_scores,
        requested_keep_ratio=float(keep_k) / float(n) if n else None,
        fallback_used=fb_count > 0,
        fallback_reason=fallback_reason,
    )
    return keep_indices, metadata
