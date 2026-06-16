"""Legacy ACGTP-v1 selector implementation."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..methods.utils import finalize_selection_debug_info

# ─────────────────────────────────────────────────────────────────────────────
# P15: ACGTP-v1 — Action-Constrained Geometric Token Protection
#
# Strategy philosophy:
#   1. Hard-protect tokens from 4 independent action-relevant branches:
#      scene_layout, depth_structure, contact_ring, motion_corridor
#   2. Use overlap-aware union; if total > keep_k, truncate by priority with
#      action_constraint_score as tiebreaker
#   3. Constrained scene fill: only scene-relevant tokens for remaining slots
#   4. Safe fallback: any valid token, sorted by action_constraint_score
#
# Each token has a clear attribution (why it was kept) and exhaustive metadata.
# ─────────────────────────────────────────────────────────────────────────────

def select_acgtp_v1(
    scene_layout_scores: Optional[np.ndarray],
    depth_edge_scores: Optional[np.ndarray],
    contact_ring_scores: Optional[np.ndarray],
    motion_corridor_scores: Optional[np.ndarray],
    valid_mask: Optional[np.ndarray],
    keep_k: int,
    constrained_fill_mask: Optional[np.ndarray] = None,
    token_u: Optional[np.ndarray] = None,
    token_v: Optional[np.ndarray] = None,
    grid_h: int = 16,
    grid_w: int = 16,
    w_scene_layout: float = 0.30,
    w_depth_structure: float = 0.25,
    w_contact_ring: float = 0.25,
    w_motion_corridor: float = 0.20,
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
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """ACGTP-v1: Action-Constrained Geometric Token Protection.

    Four independent action-relevant branches supply hard-protect candidates:
      - scene_layout:  tabletop/support plane + object components + boundaries
      - depth_structure: depth edge/boundary gradient (same signal as depth_edge_fast)
      - contact_ring:   self-filtered gripper outer ring (excludes robot-self core)
      - motion_corridor: swept path / contact risk (swept-corridor mixture)

    Branch protection is OVERLAP-AWARE:
      - Each active branch gets an allocated quota from the hard-protect budget
      - When branch candidates overlap, the overlapping tokens are attributed to the
        higher-priority branch, and the lower-priority branch is replenished from
        its next-best candidates until its quota is satisfied
      - If a branch has no valid candidates or motion is invalid, its quota is
        released and redistributed to remaining branches
      - Constrained fill (only from scene-relevant tokens) fills remaining slots
      - Safe fallback fills whatever remains

    Args:
        scene_layout_scores:  [N] scene-layout constraint scores.
        depth_edge_scores:   [N] depth edge/structure scores.
        contact_ring_scores: [N] self-filtered contact ring scores.
        motion_corridor_scores: [N] smoothed motion-corridor/contact-risk scores.
        valid_mask:         [N] boolean mask of valid tokens.
        keep_k:             Target number of tokens to keep.
        constrained_fill_mask: [N] boolean mask of scene-relevant fill candidates.
        token_u, token_v:  [N] integer pixel grid coordinates.
        grid_h, grid_w:    Grid dimensions.
        w_scene_layout:     Weight for scene_layout in action_constraint_score.
        w_depth_structure:  Weight for depth_structure.
        w_contact_ring:      Weight for contact_ring.
        w_motion_corridor:   Weight for motion_corridor.
        hard_protect_ratio: Fraction of keep_k from hard_protect before fill (default 0.60).
        motion_corridor_valid: Whether motion corridor signal is reliable.
        self_core_mask:     [N] boolean mask of robot-self-core tokens (to exclude from contact).
        contact_ring_inner_px: Inner radius for contact ring gate (pixel units).
        contact_ring_outer_px: Outer radius for contact ring gate.
        contact_requires_edge_or_object: Gate contact_ring with depth edge when True.
        depth_edge_score_for_gate: [N] depth_edge scores for contact_ring gate.
        _motion_result_for_diag: Full motion corridor result dict for real diagnostics.
        _scene_result_for_diag: Full scene layout result dict for real diagnostics.

    Returns:
        (sorted_keep_indices, selection_metadata)
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
    raw_valid = _to_1d(valid_mask) if valid_mask is not None else None
    raw_fill = _to_1d(constrained_fill_mask) if constrained_fill_mask is not None else None
    raw_self_core = _to_1d(self_core_mask) if self_core_mask is not None else None
    raw_gate = _to_1d(depth_edge_score_for_gate)

    n = keep_k
    for arr in (raw_scene, raw_de, raw_contact, raw_motion):
        if arr is not None:
            n = max(n, int(arr.shape[0]))
    n = max(n, keep_k)

    if raw_valid is not None and raw_valid.shape[0] == n:
        valid = raw_valid.astype(bool)
    else:
        valid = np.ones(n, dtype=bool)

    keep_k = int(max(0, min(keep_k, n)))

    # ── Normalization helper ───────────────────────────────────────────────
    def _norm(scores: np.ndarray) -> np.ndarray:
        a = np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
        adj = np.where(valid, a, -np.inf)
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

    def _branch_score(scores: Optional[np.ndarray]) -> np.ndarray:
        if scores is None:
            return np.zeros(n, dtype=np.float32)
        return _norm(scores)

    # ── Normalize branch scores ─────────────────────────────────────────────
    norm_scene = _branch_score(raw_scene)
    norm_de = _branch_score(raw_de)
    norm_contact = _branch_score(raw_contact)
    norm_motion = _branch_score(raw_motion)

    norm_scene_s = _norm_safe(raw_scene)
    norm_de_s = _norm_safe(raw_de)
    norm_contact_s = _norm_safe(raw_contact)
    norm_motion_s = _norm_safe(raw_motion)

    # ── Action-constraint score (mixture) ──────────────────────────────────
    acgtp_scores = (
        w_scene_layout * norm_scene_s
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

    # ── Contact ring gate: require depth_edge OR object when enabled ────────
    norm_contact_gated = norm_contact.copy()
    if contact_requires_edge_or_object and raw_gate is not None:
        gate_scores = _norm_safe(raw_gate)
        gate_threshold = float(np.percentile(gate_scores[valid], 60)) if np.any(valid) else 0.0
        no_gate = gate_scores < gate_threshold
        norm_contact_gated[no_gate] = 0.0

    # ── Motion corridor gate: zero-out when signal is invalid ─────────────
    norm_motion_gated = norm_motion.copy()
    norm_motion_s_gated = norm_motion_s.copy()
    if not motion_corridor_valid:
        norm_motion_gated = np.full(n, -np.inf, dtype=np.float32)
        norm_motion_s_gated = np.zeros(n, dtype=np.float32)

    # ── Branch validity assessment ─────────────────────────────────────────
    scene_valid = np.any((norm_scene > 0) & valid)
    de_valid = np.any((norm_de > 0) & valid)
    contact_valid = np.any((norm_contact_gated > 0) & valid)
    motion_valid_branch = motion_corridor_valid and np.any((norm_motion_gated > 0) & valid)

    # ── Hard-protect budget ─────────────────────────────────────────────────
    hard_k_total = max(1, int(round(keep_k * hard_protect_ratio)))
    hard_k_total = min(hard_k_total, keep_k)

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

    quota_map, quota_weight_map = _allocate_weighted_branch_quotas(hard_k_total)
    scene_quota = quota_map["scene"]
    de_quota = quota_map["depth"]
    contact_quota = quota_map["contact"]
    motion_quota = quota_map["motion"]

    # ── Get candidate lists per branch (sorted by descending score) ────────
    scene_candidates = [i for i in _topk_order(norm_scene, hard_k_total).tolist() if norm_scene[i] > 0]
    de_candidates = [i for i in _topk_order(norm_de, hard_k_total).tolist() if norm_de[i] > 0]
    contact_candidates = [i for i in _topk_order(norm_contact_gated, hard_k_total).tolist() if norm_contact_gated[i] > 0]
    motion_candidates = [i for i in _topk_order(norm_motion_gated, hard_k_total).tolist() if norm_motion_gated[i] > 0]

    # ── Overlap-aware constrained union with replenishment ───────────────────
    #
    # Priority order (highest to lowest): scene > depth > contact > motion
    # Each branch fills its quota. Overlapping tokens are attributed to the
    # higher-priority branch, and the lower-priority branch replenishes from
    # its remaining candidates until its quota is satisfied.
    #
    # This ensures NO branch is arbitrarily squeezed by overlap — it always
    # gets its quota filled unless it genuinely runs out of candidates.

    selected: set[int] = set()
    selected_owner: Dict[int, str] = {}
    allocated: Dict[str, int] = {"scene": 0, "depth": 0, "contact": 0, "motion": 0}

    def _fill_branch(branch_name: str, candidates: list, quota: int) -> None:
        nonlocal selected, allocated
        replenished = 0
        for idx_i in candidates:
            if allocated[branch_name] >= quota:
                break
            if idx_i not in selected:
                selected.add(idx_i)
                selected_owner[int(idx_i)] = branch_name
                allocated[branch_name] += 1
                replenished += 1

    def _overlap_with_higher_priority(idx_i: int) -> bool:
        return idx_i in selected

    # Scene (priority 1): fill from scene_candidates
    _fill_branch("scene", scene_candidates, scene_quota)

    # Depth (priority 2): fill from de_candidates, skipping already-selected
    _fill_branch("depth", de_candidates, de_quota)

    # Contact (priority 3): fill from contact_candidates, skipping already-selected
    _fill_branch("contact", contact_candidates, contact_quota)

    # Motion (priority 4): fill from motion_candidates, skipping already-selected
    _fill_branch("motion", motion_candidates, motion_quota)

    # Hard-selected set
    hard_selected = set(selected)
    hard_selected_list = sorted(hard_selected, key=lambda i: (-acgtp_scores[i] if valid[i] else -np.inf, i))

    # Remaining slots after hard_protect
    remaining_k = keep_k - len(hard_selected)
    fallback_used = False
    fallback_reason = None
    fallback_count = 0

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
        | depth_fill_candidate_mask
        | contact_fill_candidate_mask
        | motion_fill_candidate_mask
    )

    # ── Constrained scene fill ────────────────────────────────────────────
    fill_selected: set[int] = set()
    if remaining_k > 0:
        def _raw_fill_positive(idx_i: int) -> bool:
            return raw_fill is not None and idx_i < raw_fill.shape[0] and raw_fill[idx_i] > 0.5

        def _fill_membership(idx_i: int) -> Dict[str, bool]:
            return {
                "scene": bool(_raw_fill_positive(idx_i) or norm_scene_s[idx_i] > 0.0),
                "depth": bool(depth_fill_candidate_mask[idx_i]),
                "contact": bool(contact_fill_candidate_mask[idx_i]),
                "motion": bool(motion_fill_candidate_mask[idx_i]),
            }

        fill_candidates = {
            i for i in range(n)
            if valid[i] and i not in hard_selected and any(_fill_membership(i).values())
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
            "scene": max(0, scene_quota - allocated["scene"]),
            "depth": max(0, de_quota - allocated["depth"]),
            "contact": max(0, contact_quota - allocated["contact"]),
            "motion": max(0, motion_quota - allocated["motion"]),
        }

        def _coverage_fill_score(idx_i: int) -> Tuple[float, int]:
            membership = _fill_membership(idx_i)
            branch_bonus = 0.0
            for branch_name, is_member in membership.items():
                if not is_member:
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

        while fill_candidates and len(fill_selected) < remaining_k:
            best_idx = max(fill_candidates, key=_coverage_fill_score)
            fill_candidates.remove(best_idx)
            fill_selected.add(best_idx)
            selected_owner[int(best_idx)] = "fill"
            if tu is not None and tv is not None and best_idx < tu.shape[0] and best_idx < tv.shape[0]:
                selected_cells.add((int(tu[best_idx]), int(tv[best_idx])))

    remaining_k -= len(fill_selected)

    # ── Safe fallback fill ────────────────────────────────────────────────
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

    if len(fill_selected) == 0 and remaining_k > 0:
        fallback_reason = "constrained_fill_insufficient"
    elif fallback_count == 0:
        fallback_reason = None

    # ── Final union and priority ordering ──────────────────────────────────
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

    # ── Non-overlapping branch attribution ────────────────────────────────
    scene_only: set[int] = set()
    de_only: set[int] = set()
    contact_only: set[int] = set()
    motion_only: set[int] = set()
    fill_only: set[int] = set()
    fallback_only: set[int] = set()

    for idx_i in sorted(selected_set):
        owner = selected_owner.get(int(idx_i))
        if owner == "scene":
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
            fallback_only.add(idx_i)
        else:
            fill_only.add(idx_i)

    scene_count = len(scene_only)
    de_count = len(de_only)
    contact_count = len(contact_only)
    motion_count = len(motion_only)
    fill_count = len(fill_only)
    fb_count = len(fallback_only)

    branch_sum = scene_count + de_count + contact_count + motion_count + fill_count + fb_count
    accounting_valid = (branch_sum == final_kept)

    # ── Scene layout per-component selected attribution ──────────────────────
    # Computed from the actual selected token set (all_selected) using the
    # scene layout component masks from the module result.  This attributes each
    # kept token to its scene category: support_plane, object_component,
    # boundary, or relation (scene-relevant but not in any specific category).
    scene_selected_diag: Dict[str, Any] = {
        "acgtp_scene_selected_support_plane_count": 0,
        "acgtp_scene_selected_object_component_count": 0,
        "acgtp_scene_selected_boundary_count": 0,
        "acgtp_scene_selected_relation_count": None,
        "acgtp_scene_support_plane_selected_ratio": 0.0,
        "acgtp_scene_relation_token_count": None,
        "acgtp_scene_relation_token_count_computed": False,
    }
    if _scene_result_for_diag is not None and all_selected:
        try:
            # Build per-token category booleans from scene_result
            support_scores = _scene_result_for_diag.get("support_plane_scores")
            object_scores = _scene_result_for_diag.get("object_component_scores")
            boundary_scores = _scene_result_for_diag.get("boundary_scores")
            scene_layout_scores = _scene_result_for_diag.get("scene_layout_scores")

            n_local = n
            for arr in (support_scores, object_scores, boundary_scores, scene_layout_scores):
                if arr is not None:
                    try:
                        arr_n = int(np.asarray(arr, dtype=object).size)
                        n_local = max(n_local, arr_n)
                    except (TypeError, ValueError):
                        pass

            is_support = np.zeros(n_local, dtype=bool)
            is_object = np.zeros(n_local, dtype=bool)
            is_boundary = np.zeros(n_local, dtype=bool)
            is_relation = np.zeros(n_local, dtype=bool)

            # P6: prefer support_plane_candidate_scores (capped) over blanket support_plane_scores
            sp_cand_scores = _scene_result_for_diag.get("support_plane_candidate_scores")
            if sp_cand_scores is not None:
                sp_arr = np.asarray(sp_cand_scores, dtype=np.float32).reshape(-1)
                if sp_arr.shape[0] == n_local:
                    is_support = sp_arr > 0.0
            elif support_scores is not None:
                s = np.asarray(support_scores, dtype=np.float32).reshape(-1)
                if s.shape[0] == n_local:
                    is_support = s > 0.0

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
                    is_relation = scene_relevant & ~is_support & ~is_object & ~is_boundary

            # Count selected tokens in each category
            selected_sp = sum(1 for i in all_selected if is_support[i] if i < n_local)
            selected_oc = sum(1 for i in all_selected if is_object[i] if i < n_local)
            selected_bn = sum(1 for i in all_selected if is_boundary[i] if i < n_local)

            total_sp = int(np.sum(is_support))

            scene_selected_diag["acgtp_scene_selected_support_plane_count"] = selected_sp
            scene_selected_diag["acgtp_scene_selected_object_component_count"] = selected_oc
            scene_selected_diag["acgtp_scene_selected_boundary_count"] = selected_bn

            if np.any(is_relation):
                scene_selected_diag["acgtp_scene_selected_relation_count"] = sum(
                    1 for i in all_selected if is_relation[i] if i < n_local
                )
                scene_selected_diag["acgtp_scene_relation_token_count"] = int(np.sum(is_relation))
                scene_selected_diag["acgtp_scene_relation_token_count_computed"] = True

            if total_sp > 0:
                scene_selected_diag["acgtp_scene_support_plane_selected_ratio"] = float(selected_sp) / float(total_sp)
        except Exception:
            pass  # Never fail selection due to diagnostics

    # ── Overlap diagnostics ───────────────────────────────────────────────
    overlap_scene_de = len(set(scene_candidates) & set(de_candidates))
    overlap_scene_contact = len(set(scene_candidates) & set(contact_candidates))
    overlap_scene_motion = len(set(scene_candidates) & set(motion_candidates))
    overlap_contact_motion = len(set(contact_candidates) & set(motion_candidates))
    overlap_de_contact = len(set(de_candidates) & set(contact_candidates))
    overlap_de_motion = len(set(de_candidates) & set(motion_candidates))

    # ── Per-branch score statistics ─────────────────────────────────────
    def _score_stats(scores: np.ndarray) -> Dict[str, float]:
        if scores is None:
            return {"mean": 0.0, "max": 0.0}
        a = np.nan_to_num(np.asarray(scores, dtype=np.float32).reshape(-1), nan=0.0)
        v = a[valid]
        return {
            "mean": float(np.mean(v)) if v.size else 0.0,
            "max": float(np.max(v)) if v.size else 0.0,
        }

    scene_stats = _score_stats(norm_scene_s)
    de_stats = _score_stats(norm_de_s)
    contact_stats = _score_stats(norm_contact_gated)
    motion_stats = _score_stats(norm_motion_gated)
    acgtp_stats = _score_stats(acgtp_scores)

    # Scene layout component diagnostics — use module result if available
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

    # Self-core / contact ring diagnostics
    self_core_count = int(np.sum(raw_self_core)) if raw_self_core is not None else 0
    self_core_ratio = float(self_core_count) / float(n) if n > 0 else 0.0
    # Total tokens in the contact ring (24-48px), NOT gated top-k
    contact_ring_total = int(np.sum(norm_contact_gated > 0)) if norm_contact_gated is not None else 0
    contact_ring_ratio = float(contact_ring_total) / float(n) if n > 0 else 0.0
    # Gated count = tokens in ring that pass the depth-edge gate
    contact_ring_gated_total = contact_ring_total  # gate already applied in norm_contact_gated

    # Motion corridor diagnostics — use module result if available
    _motion_norm = 0.0
    _motion_disabled_reason = None
    _ema_alpha_config = 0.6
    if _motion_result_for_diag is not None:
        _motion_norm = _motion_result_for_diag.get("motion_norm_m", 0.0)
        _motion_disabled_reason = _motion_result_for_diag.get("motion_disabled_reason")
        _ema_alpha_config = _motion_result_for_diag.get("ema_alpha", 0.6)
    if not motion_corridor_valid and _motion_disabled_reason is None:
        _motion_disabled_reason = "motion_corridor_signal_unreliable"

    _acr_diag = _action_constraint_result_for_diag or {}

    # Constrained fill mask for debug recording
    constrained_fill_str = None
    if raw_fill is not None:
        try:
            import json
            constrained_fill_str = json.dumps([int(x) for x in raw_fill[:min(n, 512)]])
        except Exception:
            constrained_fill_str = str(list(raw_fill[:min(n, 512)]))

    metadata = {
        "strategy": "robot_geo_acgtp_v1",
        "selector_function_name": "select_acgtp_v1",
        "selection_strategy_name": "robot_geo_acgtp_v1",
        "selection_stage_name": "acgtp_v1_overlap_aware_constrained_union",

        # Strategy flag
        "acgtp_v1": True,
        "acgtp_selector_version": "acgtp_v2_1_weighted_coverage",
        "acgtp_quota_policy": "weight_proportional_release_invalid",
        "acgtp_fill_policy": "coverage_aware_constrained",

        # Branch weights
        "acgtp_w_scene_layout": float(w_scene_layout),
        "acgtp_w_depth_structure": float(w_depth_structure),
        "acgtp_w_contact_ring": float(w_contact_ring),
        "acgtp_w_motion_corridor": float(w_motion_corridor),

        # Hard protect
        "acgtp_hard_protect_count": len(hard_selected),
        "acgtp_hard_protect_ratio": float(hard_protect_ratio),
        "acgtp_hard_protect_valid": len(hard_selected) <= keep_k,
        "acgtp_hard_protect_ratio_config": float(hard_protect_ratio),
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

        # Motion corridor gate status
        "acgtp_motion_corridor_valid": bool(motion_corridor_valid),
        "acgtp_motion_disabled_reason": _motion_disabled_reason,

        # Self-core / contact ring (real values from config and module)
        "acgtp_self_core_radius_px": float(contact_ring_inner_px) - 16.0,  # self-core = 16px; ring inner = 24px
        "acgtp_contact_ring_inner_px": float(contact_ring_inner_px),
        "acgtp_contact_ring_outer_px": float(contact_ring_outer_px),
        "acgtp_self_core_token_count": self_core_count,
        "acgtp_self_core_token_ratio": self_core_ratio,
        "acgtp_contact_ring_token_count": contact_ring_total,
        "acgtp_contact_ring_token_ratio": contact_ring_ratio,
        "acgtp_contact_ring_gated_token_count": contact_ring_gated_total,
        "acgtp_contact_ring_valid": True,

        # Scene layout diagnostics (from module result or computed)
        "acgtp_scene_layout_score_mean": scene_stats["mean"],
        "acgtp_scene_layout_score_max": scene_stats["max"],
        # P6: use capped candidate count for support_plane
        "acgtp_support_plane_token_count": _sp_total_cnt,
        "acgtp_support_plane_candidate_count": _sp_cand_cnt,
        "acgtp_object_component_token_count": _oc_cnt,
        "acgtp_boundary_token_count": _bn_cnt,
        "acgtp_scene_fill_candidate_count": fill_candidate_count,
        "acgtp_scene_fill_candidate_ratio": fill_candidate_ratio,
        "acgtp_coverage_fill_candidate_count": coverage_fill_candidate_count,
        "acgtp_coverage_fill_candidate_ratio": coverage_fill_candidate_ratio,
        # P6: support_plane cap diagnostics
        "acgtp_scene_support_plane_cap_ratio": float(support_plane_cap_ratio),
        "acgtp_scene_support_plane_cap_used": _sp_fallback_used,
        "acgtp_scene_support_plane_fallback_used": _sp_fallback_used,
        "acgtp_scene_support_plane_fallback_reason": _sp_fallback_reason,
        # P6: object_component fallback
        "acgtp_scene_object_component_fallback_used": _oc_fallback_used,
        "acgtp_scene_object_component_fallback_reason": _oc_fallback_reason,
        "acgtp_scene_object_component_num_components": _oc_num_components,
        # P6: boundary fallback + source
        "acgtp_scene_boundary_fallback_used": _bn_fallback_used,
        "acgtp_scene_boundary_fallback_reason": _bn_fallback_reason,
        "acgtp_scene_boundary_from_object_count": _bn_from_object,
        "acgtp_scene_boundary_from_depth_count": _bn_from_depth,

        # Scene layout per-component selected attribution (post-selection diagnostics)
        # Only populated when _keep_indices and _scene_result_for_diag are both provided.
        "acgtp_scene_selected_support_plane_count": scene_selected_diag["acgtp_scene_selected_support_plane_count"],
        "acgtp_scene_selected_object_component_count": scene_selected_diag["acgtp_scene_selected_object_component_count"],
        "acgtp_scene_selected_boundary_count": scene_selected_diag["acgtp_scene_selected_boundary_count"],
        "acgtp_scene_selected_relation_count": scene_selected_diag["acgtp_scene_selected_relation_count"],
        "acgtp_scene_support_plane_selected_ratio": scene_selected_diag["acgtp_scene_support_plane_selected_ratio"],
        "acgtp_scene_relation_token_count": scene_selected_diag["acgtp_scene_relation_token_count"],
        "acgtp_scene_relation_token_count_computed": scene_selected_diag["acgtp_scene_relation_token_count_computed"],

        # Motion corridor diagnostics (real values from module result)
        "acgtp_motion_corridor_score_mean": motion_stats["mean"],
        "acgtp_motion_corridor_score_max": motion_stats["max"],
        "acgtp_motion_corridor_length_m": float(keep_k) / float(n) * 1.0 if n > 0 else 0.0,  # proportional placeholder; real value from module
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
        "selected_by_phase1": scene_count + de_count,
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
        selector_function_name="select_acgtp_v1",
        strategy="robot_geo_acgtp_v1",
        keep_indices=keep_indices,
        num_tokens=n,
        keep_count=keep_k,
        scores=acgtp_scores,
        requested_keep_ratio=float(keep_k) / float(n) if n else None,
        fallback_used=fb_count > 0,
        fallback_reason=fallback_reason,
    )
    return keep_indices, metadata


# ─────────────────────────────────────────────────────────────────────────────
# P16: ACGTP-v2 — Task-Conditioned Action-Constrained Geometry Token Protection
#
# Pipeline:
#   1. Compute semantic anchors (instruction parser + optional open-vocab detector).
#      When semantic_unavailable=True: semantic_confidence=0.0, quota released.
#   2. Five-branch hard protect (semantic > scene_layout > depth > contact > motion):
#      - semantic branch gets hard_semantic_quota tokens (top semantic_anchor_scores).
#      - Each branch has its own cap (target_cap_k, reference_cap_k, relation_cap_k).
#      - Branches with zero valid candidates release their quota.
#      - Overlap is attributed to higher-priority branch.
#   3. Constrained fill: only from (semantic anchors ∪ scene_layout ∪ depth).
#   4. Safe fallback: any valid token.
#
# Key invariants:
#   - keep_indices is sorted ascending, unique, len <= keep_k.
#   - semantic_confidence is LOW (0.0) in fallback mode; HIGH (>0.8) only with real detector.
#   - When semantic_unavailable=True and release_quota=True: semantic quota goes to other branches.
# ─────────────────────────────────────────────────────────────────────────────
