"""Legacy hybrid selector implementations."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..methods.utils import finalize_selection_debug_info, _normalize_for_selection

def select_hybrid_quota_union(
    depth_edge_scores: np.ndarray,
    robot_geo_scores: np.ndarray,
    valid_mask: Optional[np.ndarray],
    keep_k: int,
    depth_edge_quota_ratio: float = 0.50,
    robot_geo_quota_ratio: float = 0.20,
    uniform_quota_ratio: float = 0.05,
    grid_h: int = 16,
    grid_w: int = 16,
    cell_grid: int = 4,
    seed: int = 7,
    hybrid_final_scores: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Hybrid quota-union selection: 50% depth_edge + 20% robot_geo + 5% uniform, fill with hybrid scores.

    Args:
        depth_edge_scores: Raw depth-edge scores for all tokens.
        robot_geo_scores: Robot geometry final scores (may be less reliable).
        valid_mask: Boolean mask of valid tokens.
        keep_k: Target number of tokens to keep.
        depth_edge_quota_ratio: Fraction of keep_k from depth_edge top (default 0.50).
        robot_geo_quota_ratio: Fraction of keep_k from robot_geo top (default 0.20).
        uniform_quota_ratio: Fraction of keep_k from uniform spatial coverage (default 0.05).
        hybrid_final_scores: Optional hybrid final scores for tie-breaking/fill (default: weighted average).

    Returns:
        (keep_indices, selection_metadata)
    """
    n = int(depth_edge_scores.shape[0]) if depth_edge_scores is not None else int(robot_geo_scores.shape[0])
    valid = np.ones(n, dtype=np.bool_) if valid_mask is None else np.asarray(valid_mask, dtype=np.bool_).reshape(-1)

    depth_edge_quota = max(0, int(round(keep_k * depth_edge_quota_ratio)))
    robot_geo_quota = max(0, int(round(keep_k * robot_geo_quota_ratio)))
    uniform_quota = max(0, int(round(keep_k * uniform_quota_ratio)))
    fill_quota = keep_k - depth_edge_quota - robot_geo_quota - uniform_quota

    rng = np.random.RandomState(seed)

    # 1. Depth edge top-k
    depth_selected: set[int] = set()
    if depth_edge_scores is not None and depth_edge_quota > 0:
        depth_flat = np.nan_to_num(np.asarray(depth_edge_scores, dtype=np.float32).reshape(-1), nan=-np.inf)
        depth_adj = np.where(valid, depth_flat, -np.inf)
        depth_order = np.lexsort((np.arange(n), -depth_adj))
        count = 0
        for idx in depth_order:
            idx_i = int(idx)
            if not valid[idx_i]:
                continue
            depth_selected.add(idx_i)
            count += 1
            if count >= depth_edge_quota:
                break

    # 2. Robot geo top-k (from its own scoring)
    robot_selected: set[int] = set()
    if robot_geo_scores is not None and robot_geo_quota > 0:
        robot_flat = np.nan_to_num(np.asarray(robot_geo_scores, dtype=np.float32).reshape(-1), nan=-np.inf)
        robot_adj = np.where(valid, robot_flat, -np.inf)
        robot_order = np.lexsort((np.arange(n), -robot_adj))
        count = 0
        for idx in robot_order:
            idx_i = int(idx)
            if not valid[idx_i]:
                continue
            robot_selected.add(idx_i)
            count += 1
            if count >= robot_geo_quota:
                break

    # 3. Uniform spatial coverage tokens
    uniform_selected: set[int] = set()
    if uniform_quota > 0:
        if grid_h * grid_w == n:
            cell_h = max(1, grid_h // cell_grid)
            cell_w = max(1, grid_w // cell_grid)
            covered = 0
            for cr in range(cell_grid):
                for cc in range(cell_grid):
                    r0 = cr * cell_h
                    c0 = cc * cell_w
                    r1 = grid_h if cr == cell_grid - 1 else min(grid_h, r0 + cell_h)
                    c1 = grid_w if cc == cell_grid - 1 else min(grid_w, c0 + cell_w)
                    candidates = []
                    for r in range(r0, r1):
                        for c in range(c0, c1):
                            tok_idx = r * grid_w + c
                            if valid[tok_idx] and tok_idx not in (depth_selected | robot_selected | uniform_selected):
                                candidates.append(tok_idx)
                    if candidates:
                        chosen = rng.choice(candidates)
                        uniform_selected.add(chosen)
                        covered += 1
                    if covered >= uniform_quota:
                        break
        if len(uniform_selected) < uniform_quota:
            excluded = depth_selected | robot_selected | uniform_selected
            candidates = [i for i in range(n) if valid[i] and i not in excluded]
            if candidates:
                extra_n = min(uniform_quota - len(uniform_selected), len(candidates))
                extra = list(rng.choice(candidates, size=extra_n, replace=False))
                uniform_selected.update(extra)

    # Union of all selected
    union_selected = depth_selected | robot_selected | uniform_selected

    # Track overlap stats
    overlap_depth_robot = len(depth_selected & robot_selected)

    # 4. Fill remaining slots with hybrid scores
    fill_selected: set[int] = set()
    if len(union_selected) < keep_k and fill_quota > 0:
        # Use hybrid final scores if available, else weighted average
        if hybrid_final_scores is not None:
            fill_scores = np.nan_to_num(np.asarray(hybrid_final_scores, dtype=np.float32).reshape(-1), nan=-np.inf)
        else:
            d_norm = _normalize_for_selection(
                np.nan_to_num(np.asarray(depth_edge_scores if depth_edge_scores is not None else np.zeros(n), dtype=np.float32)),
                valid
            ) if depth_edge_scores is not None else np.zeros(n, dtype=np.float32)
            r_norm = _normalize_for_selection(
                np.nan_to_num(np.asarray(robot_geo_scores if robot_geo_scores is not None else np.zeros(n), dtype=np.float32)),
                valid
            ) if robot_geo_scores is not None else np.zeros(n, dtype=np.float32)
            fill_scores = 0.7 * d_norm + 0.3 * r_norm

        fill_adj = np.where(valid, fill_scores, -np.inf)
        fill_order = np.lexsort((np.arange(n), -fill_adj))
        count = 0
        for idx in fill_order:
            idx_i = int(idx)
            if idx_i in union_selected:
                continue
            if not valid[idx_i]:
                continue
            fill_selected.add(idx_i)
            count += 1
            if count >= fill_quota:
                break

    all_selected = union_selected | fill_selected

    # Build final ordered list (prioritize depth_edge, then robot_geo, then uniform, then fill)
    priority_order = list(depth_selected) + list(robot_selected) + list(uniform_selected) + list(fill_selected)
    priority_order = [x for x in priority_order if x in all_selected]

    # Sort by hybrid final score as tiebreaker
    if hybrid_final_scores is not None:
        final_scores_arr = np.nan_to_num(np.asarray(hybrid_final_scores, dtype=np.float32).reshape(-1), nan=-np.inf)
    elif depth_edge_scores is not None:
        d_norm = _normalize_for_selection(np.nan_to_num(np.asarray(depth_edge_scores, dtype=np.float32)), valid) if depth_edge_scores is not None else np.zeros(n)
        r_norm = _normalize_for_selection(np.nan_to_num(np.asarray(robot_geo_scores or np.zeros(n), dtype=np.float32)), valid) if robot_geo_scores is not None else np.zeros(n)
        final_scores_arr = 0.7 * d_norm + 0.3 * r_norm
    else:
        final_scores_arr = np.zeros(n)

    priority_order.sort(key=lambda i: (-final_scores_arr[i] if valid[i] else -np.inf, i))
    keep_indices = np.sort(np.asarray(priority_order[:keep_k], dtype=np.int64))

    metadata = {
        "strategy": "robot_geo_hybrid_v0",
        "K_total": int(keep_k),
        "K_depth_edge_quota": int(depth_edge_quota),
        "K_robot_geo_quota": int(robot_geo_quota),
        "K_uniform_quota": int(uniform_quota),
        "K_fill_quota": int(fill_quota),
        "K_depth_edge_actual": int(len(depth_selected)),
        "K_robot_geo_actual": int(len(robot_selected)),
        "K_uniform_actual": int(len(uniform_selected)),
        "K_fill_actual": int(len(fill_selected)),
        "selected_by_depth_edge_count": int(len(depth_selected)),
        "selected_by_robot_geo_count": int(len(robot_selected)),
        "selected_by_uniform_count": int(len(uniform_selected)),
        "selected_by_fill_count": int(len(fill_selected)),
        "depth_edge_quota_count": int(depth_edge_quota),
        "robot_geo_quota_count": int(robot_geo_quota),
        "uniform_quota_count": int(uniform_quota),
        "fill_count": int(len(fill_selected)),
        "overlap_depth_robot_geo": int(overlap_depth_robot),
        "final_kept": int(len(keep_indices)),
        "grid_shape": [grid_h, grid_w],
    }
    metadata = finalize_selection_debug_info(
        metadata,
        selector_function_name="select_hybrid_quota_union",
        strategy="robot_geo_hybrid_v0",
        keep_indices=keep_indices,
        num_tokens=n,
        keep_count=keep_k,
        scores=hybrid_final_scores,
        requested_keep_ratio=float(keep_k) / float(n) if n else None,
    )
    return keep_indices, metadata


def select_hybrid_quota_v2(
    depth_edge_scores: np.ndarray,
    contact_risk_scores: np.ndarray,
    distance_scores: np.ndarray,
    motion_cone_scores: np.ndarray,
    valid_mask: Optional[np.ndarray],
    keep_k: int,
    grid_h: int = 16,
    grid_w: int = 16,
    cell_grid: int = 4,
    seed: int = 7,
    depth_edge_quota_ratio: float = 0.50,
    contact_quota_ratio: float = 0.20,
    distance_contact_quota_ratio: float = 0.15,
    motion_quota_ratio: float = 0.05,
    uniform_quota_ratio: float = 0.05,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Hybrid quota-union selection v2: component-specific quotas with motion gating.

    Token quota allocation:
      - 50% from depth_edge_score top-k (stable structural signal)
      - 20% from contact_risk_score top-k (potential contact zones)
      - 15% from distance_contact hybrid (joint: normalized distance * contact_risk)
      - 5% from motion_cone_score top-k (gated by distance/workspace/contact)
      - 5% uniform diversity fill (spatial coverage)
      - Remaining: filled with depth_edge + final fused score

    Motion gating: motion_cone_score tokens are only selected if they also
    satisfy at least one of: (a) distance_score in top 40%, (b) contact_risk
    in top 40%, (c) depth_edge_score in top 40%. Prevents motion tokens from
    being selected in background regions.

    Args:
        depth_edge_scores: Raw depth/edge gradient scores [N].
        contact_risk_scores: Contact risk scores [N].
        distance_scores: Distance-to-gripper scores [N].
        motion_cone_scores: Motion corridor/cone scores [N].
        valid_mask: Boolean mask of valid tokens [N].
        keep_k: Target number of tokens to keep.
        grid_h, grid_w, cell_grid: Grid dimensions for spatial diversity.
        seed: Random seed for uniform selection.
        depth_edge_quota_ratio: Fraction of keep_k from depth_edge (default 0.50).
        contact_quota_ratio: Fraction of keep_k from contact_risk (default 0.20).
        distance_contact_quota_ratio: Fraction of keep_k from distance_contact (default 0.15).
        motion_quota_ratio: Fraction of keep_k from motion_cone (default 0.05).
        uniform_quota_ratio: Fraction of keep_k from uniform fill (default 0.05).

    Returns:
        (sorted_keep_indices, selection_metadata)
    """
    n = int(depth_edge_scores.shape[0]) if depth_edge_scores is not None else int(keep_k)
    if depth_edge_scores is None:
        raise ValueError("depth_edge_scores is required for hybrid_quota_v2")
    valid = np.ones(n, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool).reshape(-1)
    keep_k = int(max(0, min(keep_k, n)))

    # Compute quotas
    k_depth = max(0, int(round(keep_k * depth_edge_quota_ratio)))
    k_contact = max(0, int(round(keep_k * contact_quota_ratio)))
    k_dist_contact = max(0, int(round(keep_k * distance_contact_quota_ratio)))
    k_motion = max(0, int(round(keep_k * motion_quota_ratio)))
    k_uniform = max(0, int(round(keep_k * uniform_quota_ratio)))
    k_fill = keep_k - k_depth - k_contact - k_dist_contact - k_motion - k_uniform

    rng = np.random.RandomState(seed)

    # ----- Helper: normalize scores for comparisons -----
    def _norm(arr: np.ndarray) -> np.ndarray:
        a = np.nan_to_num(np.asarray(arr, dtype=np.float32).reshape(-1), nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
        adj = np.where(valid, a, -np.inf)
        v = a[valid]
        if v.size == 0 or v.max() - v.min() < 1e-8:
            return np.zeros(n, dtype=np.float32)
        out = np.zeros(n, dtype=np.float32)
        out[valid] = (v - v.min()) / max(v.max() - v.min(), 1e-8)
        return out

    def _topk_order(scores: np.ndarray) -> np.ndarray:
        a = np.nan_to_num(np.asarray(scores, dtype=np.float32).reshape(-1), nan=-np.inf)
        adj = np.where(valid, a, -np.inf)
        return np.lexsort((np.arange(n), -adj))

    def _gate_mask(scores: np.ndarray, k: int) -> np.ndarray:
        """Return boolean mask of top-k tokens from scores."""
        if k <= 0 or scores is None:
            return np.zeros(n, dtype=bool)
        order = _topk_order(scores)
        mask = np.zeros(n, dtype=bool)
        count = 0
        for idx in order:
            idx_i = int(idx)
            if valid[idx_i]:
                mask[idx_i] = True
                count += 1
                if count >= k:
                    break
        return mask

    depth_norm = _norm(depth_edge_scores)
    contact_norm = _norm(contact_risk_scores) if contact_risk_scores is not None else np.zeros(n, dtype=np.float32)
    dist_norm = _norm(distance_scores) if distance_scores is not None else np.zeros(n, dtype=np.float32)
    motion_norm = _norm(motion_cone_scores) if motion_cone_scores is not None else np.zeros(n, dtype=np.float32)

    # Gate for motion: top 40% of at least one of depth/distance/contact
    k_gate = max(1, int(round(n * 0.40)))
    depth_top40 = _gate_mask(depth_edge_scores, k_gate)
    dist_top40 = _gate_mask(distance_scores, k_gate)
    contact_top40 = _gate_mask(contact_risk_scores, k_gate) if contact_risk_scores is not None else np.zeros(n, dtype=bool)
    motion_gate = depth_top40 | dist_top40 | contact_top40

    # Masked motion: only tokens in top-40% of some other score AND in motion top-k
    motion_k_masked = _gate_mask(motion_cone_scores, k_motion)
    motion_selected_mask = motion_k_masked & motion_gate
    motion_selected = set(int(i) for i in np.where(motion_selected_mask)[0])

    # 1. Depth edge top-k
    depth_selected: set[int] = set()
    depth_order = _topk_order(depth_edge_scores)
    count = 0
    for idx in depth_order:
        idx_i = int(idx)
        if valid[idx_i]:
            depth_selected.add(idx_i)
            count += 1
            if count >= k_depth:
                break

    # 2. Contact risk top-k
    contact_selected: set[int] = set()
    if contact_risk_scores is not None and k_contact > 0:
        contact_order = _topk_order(contact_risk_scores)
        count = 0
        for idx in contact_order:
            idx_i = int(idx)
            if valid[idx_i] and idx_i not in depth_selected:
                contact_selected.add(idx_i)
                count += 1
                if count >= k_contact:
                    break

    # 3. Distance-contact hybrid top-k: norm(dist) * norm(contact)
    dist_contact_scores = dist_norm * contact_norm
    dist_contact_selected: set[int] = set()
    if k_dist_contact > 0:
        dc_order = _topk_order(dist_contact_scores)
        excluded = depth_selected | contact_selected | motion_selected
        count = 0
        for idx in dc_order:
            idx_i = int(idx)
            if valid[idx_i] and idx_i not in excluded:
                dist_contact_selected.add(idx_i)
                count += 1
                if count >= k_dist_contact:
                    break

    # 4. Motion gated top-k (already computed above)
    # motion_selected already contains tokens

    # 5. Uniform diversity fill
    excluded = depth_selected | contact_selected | dist_contact_selected | motion_selected
    uniform_selected: set[int] = set()
    if k_uniform > 0 and grid_h * grid_w == n:
        cell_h = max(1, grid_h // cell_grid)
        cell_w = max(1, grid_w // cell_grid)
        filled = 0
        for cr in range(cell_grid):
            for cc in range(cell_grid):
                if len(uniform_selected) >= k_uniform:
                    break
                r0 = cr * cell_h
                c0 = cc * cell_w
                r1 = grid_h if cr == cell_grid - 1 else min(grid_h, r0 + cell_h)
                c1 = grid_w if cc == cell_grid - 1 else min(grid_w, c0 + cell_w)
                candidates = []
                for r in range(r0, r1):
                    for c in range(c0, c1):
                        t = r * grid_w + c
                        if valid[t] and t not in excluded and t not in uniform_selected:
                            candidates.append(t)
                if candidates:
                    chosen = rng.choice(candidates)
                    uniform_selected.add(int(chosen))
                    filled += 1
            if len(uniform_selected) >= k_uniform:
                break
        if len(uniform_selected) < k_uniform:
            extra_cands = [i for i in range(n) if valid[i] and i not in excluded and i not in uniform_selected]
            if extra_cands:
                extra_n = min(k_uniform - len(uniform_selected), len(extra_cands))
                uniform_selected.update(int(x) for x in rng.choice(extra_cands, size=extra_n, replace=False))

    # Union of all selected
    union = depth_selected | contact_selected | dist_contact_selected | motion_selected | uniform_selected

    # 6. Fill remaining slots
    fill_selected: set[int] = set()
    if len(union) < keep_k and k_fill > 0:
        # Fusion: 0.6 depth_edge + 0.4 robot_geo (use contact_risk as proxy)
        fused = 0.6 * depth_norm + 0.4 * contact_norm
        fused[~valid] = -np.inf
        fill_order = _topk_order(fused)
        count = 0
        for idx in fill_order:
            idx_i = int(idx)
            if idx_i not in union and valid[idx_i]:
                fill_selected.add(idx_i)
                count += 1
                if count >= k_fill:
                    break
        # Safety fallback
        if len(union) + len(fill_selected) < keep_k:
            for idx in range(n):
                if len(union) + len(fill_selected) >= keep_k:
                    break
                idx_i = int(idx)
                if valid[idx_i] and idx_i not in union and idx_i not in fill_selected:
                    fill_selected.add(idx_i)

    all_selected = union | fill_selected

    # Final ordered list: depth_edge priority, then by fused score
    fused = 0.6 * depth_norm + 0.4 * contact_norm
    fused[~valid] = -np.inf
    sorted_list = sorted(all_selected, key=lambda i: (-fused[i] if valid[i] else -np.inf, i))
    keep_indices = np.sort(np.asarray(sorted_list[:keep_k], dtype=np.int64))

    metadata = {
        "strategy": "robot_geo_hybrid_v2",
        "K_total": int(keep_k),
        "K_depth_edge_quota": k_depth,
        "K_contact_quota": k_contact,
        "K_distance_contact_quota": k_dist_contact,
        "K_motion_quota": k_motion,
        "K_uniform_quota": k_uniform,
        "K_fill_quota": k_fill,
        "K_depth_edge_actual": int(len(depth_selected)),
        "K_contact_actual": int(len(contact_selected)),
        "K_distance_contact_actual": int(len(dist_contact_selected)),
        "K_motion_actual": int(len(motion_selected)),
        "K_uniform_actual": int(len(uniform_selected)),
        "K_fill_actual": int(len(fill_selected)),
        "selected_by_depth_edge_count": int(len(depth_selected)),
        "selected_by_contact_count": int(len(contact_selected)),
        "selected_by_distance_contact_count": int(len(dist_contact_selected)),
        "selected_by_motion_count": int(len(motion_selected)),
        "selected_by_uniform_count": int(len(uniform_selected)),
        "selected_by_fill_count": int(len(fill_selected)),
        "motion_gate_tokens_total": int(np.sum(motion_gate)),
        "motion_gate_tokens_selected": int(len(motion_selected)),
        "motion_gate_effective": bool(len(motion_selected) > 0),
        "overlap_depth_contact": int(len(depth_selected & contact_selected)),
        "overlap_depth_dist_contact": int(len(depth_selected & dist_contact_selected)),
        "overlap_depth_motion": int(len(depth_selected & motion_selected)),
        "final_kept": int(len(keep_indices)),
        "expected_kept": int(keep_k),
        "grid_shape": [grid_h, grid_w],
    }
    metadata = finalize_selection_debug_info(
        metadata,
        selector_function_name="select_hybrid_quota_v2",
        strategy="robot_geo_hybrid_v2",
        keep_indices=keep_indices,
        num_tokens=n,
        keep_count=keep_k,
        scores=fused,
        requested_keep_ratio=float(keep_k) / float(n) if n else None,
    )
    return keep_indices, metadata


def select_hybrid_v1(
    depth_edge_scores: np.ndarray,
    near_scores: np.ndarray,
    contact_risk_scores: np.ndarray,
    corridor_scores: np.ndarray,
    valid_mask: Optional[np.ndarray],
    keep_k: int,
    grid_h: int = 16,
    grid_w: int = 16,
    cell_grid: int = 4,
    seed: int = 7,
    w_edge: float = 0.45,
    w_near: float = 0.20,
    w_contact: float = 0.20,
    w_corr: float = 0.10,
    w_diverse: float = 0.05,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Hybrid v1: weighted multi-signal score with spatial diversity prior.

    Final score formula:
        final_score = w_edge * norm(edge) + w_near * norm(near)
                    + w_contact * norm(contact) + w_corr * norm(corridor)
                    + w_diverse * spatial_diversity_prior

    Selection: adaptive top-k over final scores, with grid-coverage fill.

    Args:
        depth_edge_scores: Raw depth-edge gradient scores [N].
        near_scores: Normalized near/gripper-distance scores [N].
        contact_risk_scores: Contact risk scores [N].
        corridor_scores: Corridor/motion-cone scores [N].
        valid_mask: Boolean mask of valid tokens [N].
        keep_k: Target number of tokens to keep.
        grid_h, grid_w, cell_grid: Grid dimensions for spatial diversity.
        seed: Random seed for diversity tie-breaking.
        w_edge, w_near, w_contact, w_corr, w_diverse: Score component weights.

    Returns:
        (sorted_keep_indices, selection_metadata)
    """
    n = int(depth_edge_scores.shape[0]) if depth_edge_scores is not None else int(keep_k)
    valid = np.ones(n, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool).reshape(-1)
    keep_k = int(max(0, min(keep_k, n)))

    def _norm(arr: np.ndarray, name: str = "score") -> np.ndarray:
        """Robust min-max normalization. Returns zeros when max==min (no NaN)."""
        a = np.nan_to_num(np.asarray(arr, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
        out = np.zeros(n, dtype=np.float32)
        lo = float(np.min(a[valid])) if np.any(valid) else 0.0
        hi = float(np.max(a[valid])) if np.any(valid) else 0.0
        if hi - lo > 1e-8:
            out[valid] = (a[valid] - lo) / (hi - lo)
        else:
            out[valid] = 0.0
        out[~valid] = 0.0
        return out

    def _stats(arr: np.ndarray) -> Dict[str, float]:
        """Compute mean/max/std over valid tokens."""
        a = np.nan_to_num(np.asarray(arr, dtype=np.float32).reshape(-1), nan=0.0)
        v = a[valid]
        return {
            f"mean": float(np.mean(v)) if v.size else 0.0,
            f"max": float(np.max(v)) if v.size else 0.0,
            f"std": float(np.std(v)) if v.size else 0.0,
        }

    # Normalize each component
    norm_edge = _norm(depth_edge_scores, "edge")
    norm_near = _norm(near_scores, "near")
    norm_contact = _norm(contact_risk_scores, "contact") if contact_risk_scores is not None else np.zeros(n, dtype=np.float32)
    norm_corr = _norm(corridor_scores, "corr") if corridor_scores is not None else np.zeros(n, dtype=np.float32)

    # Per-component stats for logging
    edge_stats = _stats(depth_edge_scores)
    near_stats = _stats(near_scores)
    contact_stats = _stats(contact_risk_scores) if contact_risk_scores is not None else {"mean": 0.0, "max": 0.0, "std": 0.0}
    corr_stats = _stats(corridor_scores) if corridor_scores is not None else {"mean": 0.0, "max": 0.0, "std": 0.0}

    # --- Spatial diversity prior ---
    # Lightweight: penalize tokens in already-covered grid cells
    diversity_prior = np.zeros(n, dtype=np.float32)
    if grid_h * grid_w == n and keep_k > 0:
        cell_h = max(1, grid_h // max(1, cell_grid))
        cell_w = max(1, grid_w // max(1, cell_grid))
        rng = np.random.RandomState(seed)
        for cr in range(cell_grid):
            for cc in range(cell_grid):
                r0 = cr * cell_h
                c0 = cc * cell_w
                r1 = grid_h if cr == cell_grid - 1 else min(grid_h, r0 + cell_h)
                c1 = grid_w if cc == cell_grid - 1 else min(grid_w, c0 + cell_w)
                cell_indices = [
                    r * grid_w + c
                    for r in range(r0, r1)
                    for c in range(c0, c1)
                    if r * grid_w + c < n
                ]
                # Diversity score: higher for uncovered cells, lower for already-high-score cells
                for idx in cell_indices:
                    if valid[idx]:
                        diversity_prior[idx] = 0.5  # baseline
    diversity_stats = _stats(diversity_prior)

    # --- Compute final hybrid score ---
    hybrid_scores = (
        w_edge * norm_edge
        + w_near * norm_near
        + w_contact * norm_contact
        + w_corr * norm_corr
        + w_diverse * diversity_prior
    )
    hybrid_stats = _stats(hybrid_scores)

    # --- Adaptive top-k selection with grid coverage ---
    selected: set[int] = set()

    # Phase 1: Select top-k by final score (guaranteed high-signal tokens)
    hybrid_adj = np.where(valid, hybrid_scores, -np.inf)
    order = np.lexsort((np.arange(n), -hybrid_adj))
    topk_count = max(1, int(round(keep_k * 0.80)))  # 80% from score ranking
    for idx in order:
        idx_i = int(idx)
        if valid[idx_i] and len(selected) < topk_count:
            selected.add(idx_i)
        if len(selected) >= topk_count:
            break

    # Phase 2: Grid coverage fill (remaining tokens for spatial diversity)
    remaining = keep_k - len(selected)
    if remaining > 0 and grid_h * grid_w == n:
        cell_h = max(1, grid_h // max(1, cell_grid))
        cell_w = max(1, grid_w // max(1, cell_grid))
        cells: list = []
        for cr in range(cell_grid):
            for cc in range(cell_grid):
                r0 = cr * cell_h
                c0 = cc * cell_w
                r1 = grid_h if cr == cell_grid - 1 else min(grid_h, r0 + cell_h)
                c1 = grid_w if cc == cell_grid - 1 else min(grid_w, c0 + cell_w)
                cell_indices = [
                    r * grid_w + c
                    for r in range(r0, r1)
                    for c in range(c0, c1)
                    if r * grid_w + c < n
                ]
                covered = sum(1 for idx in cell_indices if idx in selected)
                cells.append((covered, cr, cc, cell_indices))

        # Sort: prioritize cells with fewer covered tokens
        for _, cr, cc, cell_indices in sorted(cells, key=lambda x: (x[0], x[1], x[2])):
            if len(selected) >= keep_k:
                break
            candidates = [idx for idx in cell_indices if idx not in selected and valid[idx]]
            if not candidates:
                continue
            # Pick token with highest hybrid score within this cell
            best = min(candidates, key=lambda idx: (-hybrid_scores[idx], idx))
            selected.add(int(best))

    # Phase 3: Fallback fill (if still not enough)
    if len(selected) < keep_k:
        for idx in order:
            idx_i = int(idx)
            if valid[idx_i] and idx_i not in selected:
                selected.add(idx_i)
            if len(selected) >= keep_k:
                break

    # Build keep_indices before entropy / coverage computations that reference it
    keep_indices = np.sort(np.asarray(list(selected), dtype=np.int64))[:keep_k]

    # Compute grid coverage ratio
    if grid_h * grid_w == n and selected:
        cell_h = max(1, grid_h // max(1, cell_grid))
        cell_w = max(1, grid_w // max(1, cell_grid))
        total_cells = cell_grid * cell_grid
        covered_cells = 0
        for cr in range(cell_grid):
            for cc in range(cell_grid):
                r0 = cr * cell_h
                c0 = cc * cell_w
                r1 = grid_h if cr == cell_grid - 1 else min(grid_h, r0 + cell_h)
                c1 = grid_w if cc == cell_grid - 1 else min(grid_w, c0 + cell_w)
                # Count cell once if ANY token in it is selected
                cell_selected = any(
                    r * grid_w + c in selected
                    for r in range(r0, r1)
                    for c in range(c0, c1)
                )
                if cell_selected:
                    covered_cells += 1
        grid_coverage_ratio = float(covered_cells) / float(total_cells)
    else:
        grid_coverage_ratio = None

    # Token grid entropy
    entropy_val = None
    if grid_h * grid_w == n and len(keep_indices) > 0:
        cell_h = max(1, int(grid_h) // max(1, cell_grid))
        cell_w = max(1, int(grid_w) // max(1, cell_grid))
        cell_cov = np.zeros((cell_grid, cell_grid), dtype=np.float32)
        for idx in keep_indices:
            if int(idx) < n:
                r, c = int(idx) // int(grid_w), int(idx) % int(grid_w)
                cr, cc = min(cell_grid - 1, r // cell_h), min(cell_grid - 1, c // cell_w)
                cell_cov[cr, cc] = 1.0
        covered = cell_cov.sum()
        total = float(cell_grid * cell_grid)
        p = covered / total if total > 0 else 0.0
        if 0 < p < 1:
            entropy_val = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        else:
            entropy_val = 0.0

    metadata = {
        "strategy": "robot_geo_hybrid_v1",
        "K_total": int(keep_k),
        "K_topk": int(topk_count),
        "K_grid_fill": int(max(0, keep_k - topk_count)),
        "final_kept": int(len(keep_indices)),
        "expected_kept": int(keep_k),
        "grid_shape": [grid_h, grid_w],
        # Score component stats
        "edge_score_mean": edge_stats["mean"],
        "edge_score_max": edge_stats["max"],
        "edge_score_std": edge_stats["std"],
        "near_score_mean": near_stats["mean"],
        "near_score_max": near_stats["max"],
        "near_score_std": near_stats["std"],
        "contact_score_mean": contact_stats["mean"],
        "contact_score_max": contact_stats["max"],
        "contact_score_std": contact_stats["std"],
        "corridor_score_mean": corr_stats["mean"],
        "corridor_score_max": corr_stats["max"],
        "corridor_score_std": corr_stats["std"],
        "diversity_score_mean": diversity_stats["mean"],
        "diversity_score_max": diversity_stats["max"],
        "diversity_score_std": diversity_stats["std"],
        "final_hybrid_score_mean": hybrid_stats["mean"],
        "final_hybrid_score_max": hybrid_stats["max"],
        "final_hybrid_score_std": hybrid_stats["std"],
        # Weights
        "w_edge": float(w_edge),
        "w_near": float(w_near),
        "w_contact": float(w_contact),
        "w_corr": float(w_corr),
        "w_diverse": float(w_diverse),
        # Grid coverage
        "selected_grid_coverage_ratio": grid_coverage_ratio,
        "grid_coverage_ratio": grid_coverage_ratio,
    }

    # Add token grid entropy (Shannon entropy over grid coverage)
    if grid_h * grid_w == n and keep_indices is not None and len(keep_indices) > 0:
        grid_h_int = int(grid_h)
        grid_w_int = int(grid_w)
        cell_h = max(1, grid_h_int // max(1, cell_grid))
        cell_w = max(1, grid_w_int // max(1, cell_grid))
        cell_coverage = np.zeros((cell_grid, cell_grid), dtype=np.float32)
        for idx in keep_indices:
            if idx < n:
                r = int(idx) // grid_w_int
                c = int(idx) % grid_w_int
                cr = min(cell_grid - 1, r // cell_h)
                cc = min(cell_grid - 1, c // cell_w)
                cell_coverage[cr, cc] = 1.0
        covered = cell_coverage.sum()
        total = float(cell_grid * cell_grid)
        p = covered / total if total > 0 else 0.0
        if p > 0 and p < 1:
            entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        else:
            entropy = 0.0
        metadata["selected_token_grid_entropy"] = float(entropy)
    else:
        metadata["selected_token_grid_entropy"] = None

    metadata = finalize_selection_debug_info(
        metadata,
        selector_function_name="select_hybrid_v1",
        strategy="robot_geo_hybrid_v1",
        keep_indices=keep_indices,
        num_tokens=n,
        keep_count=keep_k,
        scores=hybrid_scores,
        requested_keep_ratio=float(keep_k) / float(n) if n else None,
    )
    return keep_indices, metadata


def select_hybrid_v1_edge_reserve(
    depth_edge_scores: np.ndarray,
    near_scores: np.ndarray,
    contact_risk_scores: np.ndarray,
    corridor_scores: np.ndarray,
    valid_mask: Optional[np.ndarray],
    keep_k: int,
    edge_reserve_k: int,
    grid_h: int = 16,
    grid_w: int = 16,
    cell_grid: int = 4,
    seed: int = 7,
    w_edge: float = 0.45,
    w_near: float = 0.20,
    w_contact: float = 0.20,
    w_corr: float = 0.10,
    w_diverse: float = 0.05,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Diagnostic variant: pre-reserve top-K depth_edge tokens before hybrid competition.

    P5 targeted edge-reserve ablation to test squeeze hypothesis:
    If robot_geo scores push depth_edge top-K tokens out of final selection,
    forcing an explicit edge reserve should recover success rate.

    Phase 0 (edge_reserve): Pre-select top-K depth_edge tokens into the selected set.
    Phase 1-3: Identical to select_hybrid_v1(), but edge-reserve tokens are pre-committed.

    Args:
        depth_edge_scores: Raw depth-edge gradient scores [N].
        near_scores: Normalized near/gripper-distance scores [N].
        contact_risk_scores: Contact risk scores [N].
        corridor_scores: Corridor/motion-cone scores [N].
        valid_mask: Boolean mask of valid tokens [N].
        keep_k: Target number of tokens to keep.
        edge_reserve_k: Number of top depth_edge tokens to force-preserve.
        grid_h, grid_w, cell_grid: Grid dimensions for spatial diversity.
        seed: Random seed for diversity tie-breaking.
        w_edge, w_near, w_contact, w_corr, w_diverse: Score component weights.

    Returns:
        (sorted_keep_indices, selection_metadata)
    """
    # Reshape all inputs to 1D; infer n from depth_edge_scores (the canonical length reference)
    def _to_1d(arr) -> np.ndarray:
        """Convert any array to 1D, defending against 2D [1,N] shapes and 0-d scalars."""
        if arr is None:
            return None
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim == 0:
            # 0-d scalar — reshape to 1D so shape[0] is valid
            return a.reshape(-1)
        if a.ndim == 2:
            # [1, N] from batch wrapper — take row 0
            if a.shape[0] == 1:
                return a.reshape(-1)
            # Unexpected [B, N] — flatten
            return a.reshape(-1)
        return a.reshape(-1)

    raw_depth_edge = _to_1d(depth_edge_scores)
    n = int(raw_depth_edge.shape[0]) if raw_depth_edge is not None else int(keep_k)

    # valid_mask: must be 1D [n]; defend against (1, n) from batch wrappers
    _raw_valid = _to_1d(valid_mask) if valid_mask is not None else None
    if _raw_valid is not None and _raw_valid.shape[0] == n:
        valid = _raw_valid.astype(bool)
    elif _raw_valid is not None and _raw_valid.shape[0] != n:
        # Shape mismatch — fall back to all-valid
        valid = np.ones(n, dtype=bool)
    else:
        valid = np.ones(n, dtype=bool)

    keep_k = int(max(0, min(keep_k, n)))
    edge_reserve_k = int(max(0, min(edge_reserve_k, keep_k)))

    def _norm(arr) -> np.ndarray:
        """Normalize arr to [0,1] over valid tokens. Input may be 1D or 2D [1,N]."""
        raw = _to_1d(arr)
        if raw is None:
            return np.zeros(n, dtype=np.float32)
        a = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.zeros(n, dtype=np.float32)
        lo = float(np.min(a[valid])) if np.any(valid) else 0.0
        hi = float(np.max(a[valid])) if np.any(valid) else 0.0
        if hi - lo > 1e-8:
            out[valid] = (a[valid] - lo) / (hi - lo)
        else:
            out[valid] = 0.0
        out[~valid] = 0.0
        return out

    def _stats(arr) -> Dict[str, float]:
        """Compute mean/max/std over valid tokens. Input may be 1D or 2D [1,N]."""
        raw = _to_1d(arr)
        if raw is None:
            return {"mean": 0.0, "max": 0.0, "std": 0.0}
        a = np.nan_to_num(raw, nan=0.0)
        v = a[valid]
        return {
            "mean": float(np.mean(v)) if v.size else 0.0,
            "max": float(np.max(v)) if v.size else 0.0,
            "std": float(np.std(v)) if v.size else 0.0,
        }

    edge_stats = _stats(depth_edge_scores)
    near_stats = _stats(near_scores)
    contact_stats = _stats(contact_risk_scores) if contact_risk_scores is not None else {"mean": 0.0, "max": 0.0, "std": 0.0}
    corr_stats = _stats(corridor_scores) if corridor_scores is not None else {"mean": 0.0, "max": 0.0, "std": 0.0}

    # --- Phase 0: Edge reserve (pre-commit top-K depth_edge tokens) ---
    edge_reserved_pre: set[int] = set()
    _edge_raw = _to_1d(depth_edge_scores)
    if edge_reserve_k > 0 and _edge_raw is not None:
        edge_flat = np.nan_to_num(_edge_raw, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
        edge_adj = np.where(valid, edge_flat, -np.inf)
        edge_order = np.argsort(-edge_adj)
        for idx in edge_order:
            idx_i = int(idx)
            if valid[idx_i] and len(edge_reserved_pre) < edge_reserve_k:
                edge_reserved_pre.add(idx_i)
            if len(edge_reserved_pre) >= edge_reserve_k:
                break

    selected: set[int] = set(edge_reserved_pre)
    remaining_slots = keep_k - len(edge_reserved_pre)

    # Adjusted weights: reduce edge contribution since we pre-reserved edge tokens
    w_edge_adj = max(0.0, w_edge - 0.10)
    w_near_adj = w_near + 0.05
    w_contact_adj = w_contact + 0.05

    # Normalize components
    norm_edge = _norm(depth_edge_scores)
    norm_near = _norm(near_scores)
    norm_contact = _norm(contact_risk_scores) if contact_risk_scores is not None else np.zeros(n, dtype=np.float32)
    norm_corr = _norm(corridor_scores) if corridor_scores is not None else np.zeros(n, dtype=np.float32)

    # Spatial diversity prior
    diversity_prior = np.zeros(n, dtype=np.float32)
    if grid_h * grid_w == n and keep_k > 0:
        cell_h = max(1, grid_h // max(1, cell_grid))
        cell_w = max(1, grid_w // max(1, cell_grid))
        for cr in range(cell_grid):
            for cc in range(cell_grid):
                r0 = cr * cell_h
                c0 = cc * cell_w
                r1 = grid_h if cr == cell_grid - 1 else min(grid_h, r0 + cell_h)
                c1 = grid_w if cc == cell_grid - 1 else min(grid_w, c0 + cell_w)
                cell_indices = [
                    r * grid_w + c for r in range(r0, r1) for c in range(c0, c1) if r * grid_w + c < n
                ]
                for idx in cell_indices:
                    if valid[idx]:
                        diversity_prior[idx] = 0.5
    diversity_stats = _stats(diversity_prior)

    # Hybrid score with adjusted weights (for selection)
    hybrid_scores = (
        w_edge_adj * norm_edge
        + w_near_adj * norm_near
        + w_contact_adj * norm_contact
        + w_corr * norm_corr
        + w_diverse * diversity_prior
    )
    hybrid_stats = _stats(hybrid_scores)

    # Also compute original-weight hybrid scores for P1 attribution diagnostics
    # (this lets _compute_token_selection_attribution compare depth_edge top-k vs hybrid top-k)
    hybrid_scores_original = (
        w_edge * norm_edge
        + w_near * norm_near
        + w_contact * norm_contact
        + w_corr * norm_corr
        + w_diverse * diversity_prior
    )

    # --- Phase 1: Top-k by hybrid score (80% of remaining slots) ---
    hybrid_adj = np.where(valid, hybrid_scores, -np.inf)
    order = np.lexsort((np.arange(n), -hybrid_adj))
    topk_count = max(1, int(round(remaining_slots * 0.80))) if remaining_slots > 0 else 0
    phase1_selected: set[int] = set()
    for idx in order:
        idx_i = int(idx)
        if valid[idx_i] and idx_i not in edge_reserved_pre and len(phase1_selected) < topk_count:
            phase1_selected.add(idx_i)
        if len(phase1_selected) >= topk_count:
            break
    selected.update(phase1_selected)

    # --- Phase 2: Grid coverage fill (adds to selected in-place; track additions) ---
    remaining_after_topk = keep_k - len(selected)
    _phase2_added: set[int] = set()
    if remaining_after_topk > 0 and grid_h * grid_w == n:
        cell_h = max(1, grid_h // max(1, cell_grid))
        cell_w = max(1, grid_w // max(1, cell_grid))
        cells: list = []
        for cr in range(cell_grid):
            for cc in range(cell_grid):
                r0 = cr * cell_h
                c0 = cc * cell_w
                r1 = grid_h if cr == cell_grid - 1 else min(grid_h, r0 + cell_h)
                c1 = grid_w if cc == cell_grid - 1 else min(grid_w, c0 + cell_w)
                cell_indices = [
                    r * grid_w + c for r in range(r0, r1) for c in range(c0, c1) if r * grid_w + c < n
                ]
                covered = sum(1 for idx in cell_indices if idx in selected)
                cells.append((covered, cr, cc, cell_indices))
        for _, cr, cc, cell_indices in sorted(cells, key=lambda x: (x[0], x[1], x[2])):
            if len(selected) >= keep_k:
                break
            candidates = [idx for idx in cell_indices if idx not in selected and valid[idx]]
            if not candidates:
                continue
            best = min(candidates, key=lambda idx: (-hybrid_scores[idx], idx))
            selected.add(int(best))
            _phase2_added.add(int(best))

    # --- Phase 3: Fallback fill (track additions) ---
    _phase3_added: set[int] = set()
    if len(selected) < keep_k:
        for idx in order:
            idx_i = int(idx)
            if valid[idx_i] and idx_i not in selected:
                selected.add(idx_i)
                _phase3_added.add(idx_i)
            if len(selected) >= keep_k:
                break

    keep_indices = np.sort(np.asarray(list(selected)[:keep_k], dtype=np.int64))

    # --- P5 metrics (before accounting) ---
    edge_reserved_actual = len(edge_reserved_pre & set(int(i) for i in keep_indices))
    edge_survival = float(edge_reserved_actual) / float(edge_reserve_k) if edge_reserve_k > 0 else None
    duplicate_count = len(edge_reserved_pre & phase1_selected)

    # Compute original-weight hybrid scores for P1 attribution diagnostics
    # (computed once; reused for _phase1_orig_set and overlap attribution)
    _phase1_orig_count = max(1, int(round(remaining_slots * 0.80)))
    _phase1_orig_set: set[int] = set()
    _orig_hybrid_order = np.argsort(-np.where(valid, hybrid_scores_original, -np.inf))
    for _idx in _orig_hybrid_order:
        _idx_i = int(_idx)
        if valid[_idx_i] and _idx_i not in edge_reserved_pre and len(_phase1_orig_set) < _phase1_orig_count:
            _phase1_orig_set.add(_idx_i)
        if len(_phase1_orig_set) >= _phase1_orig_count:
            break

    # Compute grid coverage and token grid entropy
    if grid_h * grid_w == n and selected:
        cell_h = max(1, grid_h // max(1, cell_grid))
        cell_w = max(1, grid_w // max(1, cell_grid))
        total_cells = cell_grid * cell_grid
        covered_cells = 0
        for cr in range(cell_grid):
            for cc in range(cell_grid):
                r0, c0 = cr * cell_h, cc * cell_w
                r1 = grid_h if cr == cell_grid - 1 else min(grid_h, r0 + cell_h)
                c1 = grid_w if cc == cell_grid - 1 else min(grid_w, c0 + cell_w)
                if any(r * grid_w + c in selected for r in range(r0, r1) for c in range(c0, c1)):
                    covered_cells += 1
        grid_coverage_ratio = float(covered_cells) / float(total_cells)
    else:
        grid_coverage_ratio = None

    entropy_val = None
    if grid_h * grid_w == n and len(keep_indices) > 0:
        cell_h = max(1, int(grid_h) // max(1, cell_grid))
        cell_w = max(1, int(grid_w) // max(1, cell_grid))
        cell_cov = np.zeros((cell_grid, cell_grid), dtype=np.float32)
        for idx in keep_indices:
            if int(idx) < n:
                r, c = int(idx) // int(grid_w), int(idx) % int(grid_w)
                cr, cc = min(cell_grid - 1, r // cell_h), min(cell_grid - 1, c // cell_w)
                cell_cov[cr, cc] = 1.0
        covered = cell_cov.sum()
        total = float(cell_grid * cell_grid)
        p = covered / total if total > 0 else 0.0
        if 0 < p < 1:
            entropy_val = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        else:
            entropy_val = 0.0

    # --- Attribution accounting (uses tracked phase additions) ---
    selected_set = set(int(i) for i in keep_indices)
    final_selected_count = len(selected_set)

    # Phase 1: tokens selected by hybrid score (adjusted weights) — pre-reserved are EXCLUDED
    selected_by_phase1_hybrid_count = len(phase1_selected & selected_set)

    # Phase 2: tokens from grid-diversity fill (tracked directly)
    selected_by_phase2_diversity_count = len(_phase2_added & selected_set)

    # Phase 3: tokens from fallback fill (tracked directly)
    selected_by_phase3_fallback_count = len(_phase3_added & selected_set)

    # selected_by_fill: ALL tokens not from edge_reserved or phase1 — same as phase2+phase3
    selected_by_edge_reserved_count = len(edge_reserved_pre & selected_set)
    selected_by_fill_count = selected_by_phase2_diversity_count + selected_by_phase3_fallback_count

    # Safety: any tokens that cannot be attributed to a known phase
    _attributed = (
        selected_by_edge_reserved_count
        + selected_by_phase1_hybrid_count
        + selected_by_phase2_diversity_count
        + selected_by_phase3_fallback_count
    )
    selected_by_unattributed_count = max(0, final_selected_count - _attributed)

    # --- Diagnostic K for cross-method comparison ---
    _k_diag_small = max(1, int(round(n * 0.20)))
    _k_diag_large = max(1, int(round(n * 0.80)))

    # --- Reserved / non-reserved split diagnostics ---
    # Reserved set: top edge_reserve_k tokens by depth_edge score (pre-reserved)
    # Non-reserved set: next k_diag_small tokens (for diagnostic drop comparison)
    # Overall: top k_diag_large tokens (k_final * 0.80 = 154, consistent denominator across methods)
    edge_scores_1d = np.nan_to_num(_to_1d(depth_edge_scores), nan=0.0)
    _all_edge_order = np.argsort(-np.where(valid, edge_scores_1d, -np.inf))

    # Reserved: first edge_reserve_k tokens in depth_edge order
    _reserved_set = set(int(i) for i in _all_edge_order[:edge_reserve_k] if valid[i])
    _reserved_kept = len(_reserved_set & selected_set)
    _reserved_dropped = len(_reserved_set) - _reserved_kept
    _reserved_dropped_ratio = float(_reserved_dropped) / len(_reserved_set) if _reserved_set else None

    # Non-reserved: next k_diag_small tokens (immediately after reserved block)
    _non_reserved_start = edge_reserve_k
    _non_reserved_end = edge_reserve_k + _k_diag_small
    _non_reserved_set = set(int(i) for i in _all_edge_order[_non_reserved_start:_non_reserved_end] if valid[i])
    _non_reserved_kept = len(_non_reserved_set & selected_set)
    _non_reserved_dropped = len(_non_reserved_set) - _non_reserved_kept
    _non_reserved_dropped_ratio = float(_non_reserved_dropped) / len(_non_reserved_set) if _non_reserved_set else None

    # Overall: top k_diag_large tokens (for cross-method comparison)
    _overall_set = set(int(i) for i in _all_edge_order[:_k_diag_large] if valid[i])
    _overall_kept = len(_overall_set & selected_set)
    _overall_dropped = len(_overall_set) - _overall_kept
    _overall_dropped_ratio = float(_overall_dropped) / len(_overall_set) if _overall_set else None

    # --- P5 overlap attribution (same logic as _compute_token_selection_attribution) ---
    # Use original-weight hybrid scores for the top-k comparison
    # so attribution can compare depth_edge top-k vs hybrid top-k drop
    _raw_hybrid_orig = hybrid_scores_original
    if _raw_hybrid_orig is None:
        _raw_hybrid_orig = np.zeros(n, dtype=np.float32)

    # depth_edge top-k (k_diag_small)
    K_topk = _k_diag_small
    edge_topk_set = set()
    edge_order_for_attr = np.argsort(-np.where(valid, edge_scores_1d, -np.inf))
    for idx in edge_order_for_attr:
        if valid[idx] and len(edge_topk_set) < K_topk:
            edge_topk_set.add(int(idx))
        if len(edge_topk_set) >= K_topk:
            break

    # hybrid top-k (original weights)
    hybrid_topk_set = set()
    hybrid_order = np.argsort(-np.where(valid, _raw_hybrid_orig, -np.inf))
    for idx in hybrid_order:
        if valid[idx] and len(hybrid_topk_set) < K_topk:
            hybrid_topk_set.add(int(idx))
        if len(hybrid_topk_set) >= K_topk:
            break

    overlap = len(edge_topk_set & hybrid_topk_set)
    depth_edge_dropped = len(edge_topk_set - selected_set)
    depth_edge_dropped_ratio = float(depth_edge_dropped) / float(K_topk) if K_topk > 0 else None
    overlap_depth_edge_robot_geo_ratio = float(overlap) / float(K_topk) if K_topk > 0 else None

    metadata = {
        "strategy": "robot_geo_hybrid_temporal_edge_reserve_v1",
        "K_total": int(keep_k),
        "K_edge_reserve_target": int(edge_reserve_k),
        "K_edge_reserve_actual": int(edge_reserved_actual),
        "edge_reserved_survival_ratio": edge_survival,
        "K_topk": int(topk_count),
        "K_grid_fill": int(max(0, keep_k - len(edge_reserved_pre) - topk_count)),
        "final_kept": int(len(keep_indices)),
        "expected_kept": int(keep_k),
        "grid_shape": [grid_h, grid_w],
        # P5 edge_reserve flags
        "edge_reserve_enabled": True,
        "edge_reserve_ratio": float(edge_reserve_k) / float(keep_k) if keep_k > 0 else 0.0,
        "edge_reserved_target_count": int(edge_reserve_k),
        "edge_reserved_actual_count": int(edge_reserved_actual),
        "edge_reserved_survival_ratio": edge_survival,
        "final_selected_count": int(final_selected_count),
        # P5-fix: corrected accounting — all selected_by_* sum to final_selected_count
        "selected_by_edge_reserved_count": int(selected_by_edge_reserved_count),
        "selected_by_original_hybrid_count": int(selected_by_phase1_hybrid_count),  # Phase 1 uses adjusted weights
        # P5-fix: renamed / new fields
        "selected_by_phase1_hybrid_count": int(selected_by_phase1_hybrid_count),  # Phase 1 (adjusted weights)
        "selected_by_phase2_diversity_count": int(selected_by_phase2_diversity_count),
        "selected_by_phase3_fallback_count": int(selected_by_phase3_fallback_count),
        "selected_by_fill_count": int(selected_by_fill_count),  # = phase2 + phase3
        "selected_by_unattributed_count": int(selected_by_unattributed_count),  # must be 0
        "duplicate_edge_hybrid_count": int(duplicate_count),
        # P5-fix: renamed duplicate metrics (original names kept as deprecated aliases)
        "duplicate_after_exclusion_count": int(duplicate_count),
        "duplicate_with_original_hybrid_count": int(len(edge_reserved_pre & _phase1_orig_set)),
        # P5-fix: diagnostic K
        "diagnostic_k_small": int(_k_diag_small),
        "diagnostic_k_large": int(_k_diag_large),
        # P5-fix: reserved / non-reserved split diagnostics
        "reserved_edge_topk_count": len(_reserved_set),
        "reserved_edge_kept_count": _reserved_kept,
        "reserved_edge_dropped_count": _reserved_dropped,
        "reserved_edge_topk_dropped_ratio": _reserved_dropped_ratio,
        "non_reserved_edge_topk_count": len(_non_reserved_set),
        "non_reserved_edge_kept_count": _non_reserved_kept,
        "non_reserved_edge_dropped_count": _non_reserved_dropped,
        "non_reserved_edge_topk_dropped_ratio": _non_reserved_dropped_ratio,
        "overall_depth_edge_topk_count": len(_overall_set),
        "overall_depth_edge_topk_kept_count": _overall_kept,
        "overall_depth_edge_topk_dropped_count": _overall_dropped,
        "overall_depth_edge_topk_dropped_ratio": _overall_dropped_ratio,
        # Score component stats
        "edge_score_mean": edge_stats["mean"],
        "edge_score_max": edge_stats["max"],
        "edge_score_std": edge_stats["std"],
        "near_score_mean": near_stats["mean"],
        "near_score_max": near_stats["max"],
        "near_score_std": near_stats["std"],
        "contact_score_mean": contact_stats["mean"],
        "contact_score_max": contact_stats["max"],
        "contact_score_std": contact_stats["std"],
        "corridor_score_mean": corr_stats["mean"],
        "corridor_score_max": corr_stats["max"],
        "corridor_score_std": corr_stats["std"],
        "diversity_score_mean": diversity_stats["mean"],
        "diversity_score_max": diversity_stats["max"],
        "diversity_score_std": diversity_stats["std"],
        "final_hybrid_score_mean": hybrid_stats["mean"],
        "final_hybrid_score_max": hybrid_stats["max"],
        "final_hybrid_score_std": hybrid_stats["std"],
        # Weights (original)
        "w_edge": float(w_edge),
        "w_near": float(w_near),
        "w_contact": float(w_contact),
        "w_corr": float(w_corr),
        "w_diverse": float(w_diverse),
        # Grid coverage
        "selected_grid_coverage_ratio": grid_coverage_ratio,
        "grid_coverage_ratio": grid_coverage_ratio,
        "selected_token_grid_entropy": entropy_val,
        # P1-style overlap attribution using original-weight hybrid scores
        "depth_edge_topk_dropped_ratio": depth_edge_dropped_ratio,
        "robot_geo_topk_dropped_ratio": None,  # not applicable for this variant
        "overlap_depth_edge_robot_geo_ratio": overlap_depth_edge_robot_geo_ratio,
    }

    metadata = finalize_selection_debug_info(
        metadata,
        selector_function_name="select_hybrid_v1_edge_reserve",
        strategy=str(metadata.get("strategy", "robot_geo_hybrid_temporal_edge_reserve_v1")),
        keep_indices=keep_indices,
        num_tokens=n,
        keep_count=keep_k,
        scores=hybrid_scores,
        requested_keep_ratio=float(keep_k) / float(n) if n else None,
    )
    return keep_indices, metadata


# =============================================================================
# P7: hybrid_budget_v2 — Budget-based hybrid selector
# Replaces global top-k competition with explicit budget allocation.
# Depth edge and robot geo each get a guaranteed budget; remaining slots filled
# with hybrid score. No global blending before selection.
# =============================================================================


def select_hybrid_budget_v2(
    depth_edge_scores: np.ndarray,
    robot_geo_scores: np.ndarray,
    valid_mask: Optional[np.ndarray],
    keep_k: int,
    depth_edge_budget_ratio: float = 0.45,
    robot_contact_budget_ratio: float = 0.25,
    safety_budget_ratio: float = 0.00,
    grid_h: int = 16,
    grid_w: int = 16,
    hybrid_scores: Optional[np.ndarray] = None,
    sort_keep_indices: bool = True,
    fallback_to_global_score: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Budget-based hybrid selector: no global top-k competition between depth edge and robot geo.

    Budget allocation (K = keep_k):
        K_depth   = round(K * depth_edge_budget_ratio)   # Stage 1: depth edge reserve
        K_robot   = round(K * robot_contact_budget_ratio)  # Stage 2: robot geo reserve
        K_safety  = round(K * safety_budget_ratio)         # Stage 3: spatial safety (not used v2)
        K_fill    = K - K_depth - K_robot - K_safety      # Stage 4: hybrid score fill

    Selection:
        Stage 1: top K_depth by depth_edge_scores  (pre-committed; robot geo cannot crowd out)
        Stage 2: top K_robot by robot_geo_scores  (excludes Stage-1 tokens; overlap tracked)
        Stage 3: spatial safety (null in v2)
        Stage 4: fill remaining slots with hybrid_scores (or 0.6*norm(depth)+0.4*norm(robot))
        Stage 5: fallback fill if Stages 1-4 do not reach K

    Args:
        depth_edge_scores: [N] depth/edge gradient scores.
        robot_geo_scores: [N] robot geometry/contact scores.
        valid_mask: [N] boolean valid token mask.
        keep_k: target token count.
        depth_edge_budget_ratio: fraction of K for depth edge (default 0.45).
        robot_contact_budget_ratio: fraction of K for robot geo (default 0.25).
        safety_budget_ratio: fraction of K for spatial safety (default 0.00; unused in v2).
        grid_h, grid_w: grid dimensions for diagnostics.
        hybrid_scores: optional [N] scores for Stage 4 fill.
        sort_keep_indices: sort final indices (default True).
        fallback_to_global_score: use combined score for fallback fill (default True).

    Returns:
        (keep_indices, metadata)
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

    raw_depth = _to_1d(depth_edge_scores)
    n = int(raw_depth.shape[0]) if raw_depth is not None else int(keep_k)

    raw_valid = _to_1d(valid_mask) if valid_mask is not None else None
    if raw_valid is not None and raw_valid.shape[0] == n:
        valid = raw_valid.astype(bool)
    elif raw_valid is not None:
        valid = np.ones(n, dtype=bool)
    else:
        valid = np.ones(n, dtype=bool)

    keep_k = int(max(0, min(keep_k, n)))

    k_depth = max(0, int(round(keep_k * depth_edge_budget_ratio)))
    k_robot = max(0, int(round(keep_k * robot_contact_budget_ratio)))
    k_safety = max(0, int(round(keep_k * safety_budget_ratio)))
    k_fill = max(0, keep_k - k_depth - k_robot - k_safety)

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
        return out

    def _topk_order(scores: np.ndarray) -> np.ndarray:
        a = np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
        adj = np.where(valid, a, -np.inf)
        return np.lexsort((np.arange(n), -adj))

    # ---- Stage 1: Depth edge reserve ----
    depth_selected: set[int] = set()
    if raw_depth is not None and k_depth > 0:
        depth_order = _topk_order(raw_depth)
        for idx in depth_order:
            idx_i = int(idx)
            if valid[idx_i] and len(depth_selected) < k_depth:
                depth_selected.add(idx_i)
            if len(depth_selected) >= k_depth:
                break

    # ---- Stage 2: Robot geo reserve (excludes Stage-1 tokens) ----
    robot_selected: set[int] = set()
    overlap_depth_robot = 0
    raw_robot = _to_1d(robot_geo_scores)
    if raw_robot is not None and k_robot > 0:
        robot_order = _topk_order(raw_robot)
        for idx in robot_order:
            idx_i = int(idx)
            if not valid[idx_i]:
                continue
            if idx_i in depth_selected:
                overlap_depth_robot += 1
                continue
            if len(robot_selected) < k_robot:
                robot_selected.add(idx_i)
            if len(robot_selected) >= k_robot:
                break

    reserved_set = depth_selected | robot_selected

    # ---- Stage 4: Fill from hybrid scores ----
    fill_selected: set[int] = set()
    fill_from_depth = 0
    fill_from_robot = 0
    fill_from_other = 0

    if len(reserved_set) < keep_k and k_fill > 0:
        if hybrid_scores is not None:
            fill_arr = np.nan_to_num(np.asarray(hybrid_scores, dtype=np.float32).reshape(-1), nan=-np.inf)
        else:
            d_norm = _norm(raw_depth) if raw_depth is not None else np.zeros(n, dtype=np.float32)
            r_norm = _norm(raw_robot) if raw_robot is not None else np.zeros(n, dtype=np.float32)
            fill_arr = 0.6 * d_norm + 0.4 * r_norm

        fill_order = _topk_order(fill_arr)
        for idx in fill_order:
            idx_i = int(idx)
            if idx_i in reserved_set or idx_i in fill_selected:
                continue
            if not valid[idx_i]:
                continue
            fill_selected.add(idx_i)
            if raw_depth is not None and raw_robot is not None:
                d_val = raw_depth[idx_i] if idx_i < raw_depth.size else -np.inf
                r_val = raw_robot[idx_i] if idx_i < raw_robot.size else -np.inf
                if d_val >= r_val and np.isfinite(d_val):
                    fill_from_depth += 1
                elif np.isfinite(r_val):
                    fill_from_robot += 1
                else:
                    fill_from_other += 1
            else:
                fill_from_other += 1
            if len(fill_selected) >= k_fill or len(reserved_set) + len(fill_selected) >= keep_k:
                break

    # ---- Stage 5: Fallback fill ----
    fallback_reason = None
    fallback_selected: set[int] = set()
    if len(reserved_set) + len(fill_selected) < keep_k:
        fallback_reason = "fill_insufficient_tokens"
        remaining_k = keep_k - len(reserved_set) - len(fill_selected)
        if raw_depth is not None:
            fallback_order = _topk_order(raw_depth)
        elif raw_robot is not None:
            fallback_order = _topk_order(raw_robot)
        else:
            fallback_order = np.arange(n, dtype=np.int64)
        for idx in fallback_order:
            idx_i = int(idx)
            if idx_i in reserved_set or idx_i in fill_selected or idx_i in fallback_selected:
                continue
            if not valid[idx_i]:
                continue
            fallback_selected.add(idx_i)
            remaining_k -= 1
            if remaining_k <= 0:
                break
        if len(reserved_set) + len(fill_selected) + len(fallback_selected) < keep_k:
            fallback_reason = "valid_tokens_less_than_keep_k"
            for idx_i in range(n):
                if idx_i in reserved_set or idx_i in fill_selected or idx_i in fallback_selected:
                    continue
                if not valid[idx_i]:
                    continue
                fallback_selected.add(idx_i)
                if len(reserved_set) + len(fill_selected) + len(fallback_selected) >= keep_k:
                    break

    all_selected = reserved_set | fill_selected | fallback_selected
    priority_order = list(depth_selected) + list(robot_selected) + list(fill_selected) + list(fallback_selected)
    priority_order = [x for x in priority_order if x in all_selected]

    if hybrid_scores is not None:
        final_scores_arr = np.nan_to_num(np.asarray(hybrid_scores, dtype=np.float32).reshape(-1), nan=-np.inf)
    elif raw_depth is not None:
        d_norm = _norm(raw_depth)
        r_norm = _norm(raw_robot) if raw_robot is not None else np.zeros(n, dtype=np.float32)
        final_scores_arr = 0.6 * d_norm + 0.4 * r_norm
    else:
        final_scores_arr = np.zeros(n, dtype=np.float32)

    priority_order.sort(key=lambda i: (-final_scores_arr[i] if valid[i] else -np.inf, i))

    if sort_keep_indices:
        keep_indices = np.sort(np.asarray(priority_order[:keep_k], dtype=np.int64))
    else:
        keep_indices = np.asarray(priority_order[:keep_k], dtype=np.int64)

    selected_set = set(int(i) for i in keep_indices)
    final_kept = len(selected_set)

    # ---- Phase accounting ----
    depth_actual = len(depth_selected & selected_set)
    robot_actual = len(robot_selected & selected_set)
    fill_actual = len(fill_selected & selected_set)
    fallback_actual = len(fallback_selected & selected_set)

    # ---- Diagnostic overlap: depth edge top-K vs robot geo top-K ----
    robot_topk_all: set[int] = set()
    if raw_robot is not None:
        robot_full_order = _topk_order(raw_robot)
        for idx in robot_full_order:
            idx_i = int(idx)
            if valid[idx_i]:
                robot_topk_all.add(idx_i)
            if len(robot_topk_all) >= k_robot:
                break
    overlap_depth_robot_diagnostic = len(depth_selected & robot_topk_all)

    # ---- Reserved / non-reserved split diagnostics ----
    depth_scores_1d = _to_1d(depth_edge_scores)
    _depth_order = np.argsort(-np.where(valid, depth_scores_1d, -np.inf)) if depth_scores_1d is not None else np.arange(n, dtype=np.int64)
    _reserved_set = set(int(i) for i in _depth_order[:k_depth] if valid[i])
    _reserved_kept = len(_reserved_set & selected_set)
    _reserved_dropped = len(_reserved_set) - _reserved_kept
    _reserved_dropped_ratio = float(_reserved_dropped) / float(len(_reserved_set)) if _reserved_set else None

    _robot_order = np.argsort(-np.where(valid, raw_robot, -np.inf)) if raw_robot is not None else np.arange(n, dtype=np.int64)
    _non_reserved_set: set[int] = set()
    for idx in _robot_order:
        idx_i = int(idx)
        if idx_i in _reserved_set:
            continue
        if valid[idx_i]:
            _non_reserved_set.add(idx_i)
        if len(_non_reserved_set) >= k_robot:
            break
    _non_reserved_kept = len(_non_reserved_set & selected_set)
    _non_reserved_dropped = len(_non_reserved_set) - _non_reserved_kept
    _non_reserved_dropped_ratio = float(_non_reserved_dropped) / float(len(_non_reserved_set)) if _non_reserved_set else None

    k_overall = max(1, int(round(n * 0.80)))
    _overall_set = set(int(i) for i in _depth_order[:k_overall] if valid[i])
    _overall_kept = len(_overall_set & selected_set)
    _overall_dropped = len(_overall_set) - _overall_kept
    _overall_dropped_ratio = float(_overall_dropped) / float(len(_overall_set)) if _overall_set else None

    metadata = {
        "strategy": "hybrid_budget_v2",
        "total_keep_budget": int(keep_k),
        "depth_edge_budget": int(k_depth),
        "robot_contact_budget": int(k_robot),
        "fill_budget": int(k_fill),
        "safety_budget": None,
        "depth_edge_budget_ratio": float(depth_edge_budget_ratio),
        "robot_contact_budget_ratio": float(robot_contact_budget_ratio),
        "safety_budget_ratio": float(safety_budget_ratio),
        "K_depth_actual": depth_actual,
        "K_robot_actual": robot_actual,
        "K_fill_actual": fill_actual,
        "fallback_actual": fallback_actual,
        "selected_by_depth_edge_count": depth_actual,
        "selected_by_robot_geo_count": robot_actual,
        "selected_by_fill_count": fill_actual,
        "selected_by_fallback_count": fallback_actual,
        "overlap_depth_robot_count": int(overlap_depth_robot),
        "overlap_depth_robot_diagnostic": int(overlap_depth_robot_diagnostic),
        "depth_edge_candidates_count": int(np.sum(valid)),
        "robot_geo_candidates_count": int(np.sum(valid)),
        "depth_edge_reserved_kept_count": depth_actual,
        "robot_geo_reserved_kept_count": robot_actual,
        "fill_from_depth_count": int(fill_from_depth),
        "fill_from_robot_count": int(fill_from_robot),
        "fill_from_other_count": int(fill_from_other),
        "reserved_edge_topk_count": len(_reserved_set),
        "reserved_edge_kept_count": _reserved_kept,
        "reserved_edge_dropped_count": _reserved_dropped,
        "reserved_edge_topk_dropped_ratio": _reserved_dropped_ratio,
        "non_reserved_edge_topk_count": len(_non_reserved_set),
        "non_reserved_edge_kept_count": _non_reserved_kept,
        "non_reserved_edge_dropped_count": _non_reserved_dropped,
        "non_reserved_edge_topk_dropped_ratio": _non_reserved_dropped_ratio,
        "overall_depth_edge_topk_count": len(_overall_set),
        "overall_depth_edge_topk_kept_count": _overall_kept,
        "overall_depth_edge_topk_dropped_count": _overall_dropped,
        "overall_depth_edge_topk_dropped_ratio": _overall_dropped_ratio,
        "K_total": int(keep_k),
        "final_kept": final_kept,
        "expected_kept": int(keep_k),
        "grid_shape": [grid_h, grid_w],
        "fallback_reason": fallback_reason,
        "hybrid_budget_v2": True,
    }

    metadata = finalize_selection_debug_info(
        metadata,
        selector_function_name="select_hybrid_budget_v2",
        strategy="hybrid_budget_v2",
        keep_indices=keep_indices,
        num_tokens=n,
        keep_count=keep_k,
        scores=final_scores_arr,
        requested_keep_ratio=float(keep_k) / float(n) if n else None,
        fallback_used=fallback_reason is not None,
        fallback_reason=fallback_reason,
    )
    return keep_indices, metadata
