"""Legacy branch-budget selector implementation."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..methods.utils import finalize_selection_debug_info

# ─────────────────────────────────────────────────────────────────────────────
# P11: Branch Budget v0
# Explicit branch budgets for depth/scene (Branch A), hybrid/action (Branch B),
# and diversity/fill (Branch C). No global top-k competition between branches.
# ─────────────────────────────────────────────────────────────────────────────

def select_branch_budget_v0(
    depth_edge_scores: np.ndarray,
    hybrid_final_scores: np.ndarray,
    valid_mask: Optional[np.ndarray],
    keep_k: int,
    depth_edge_budget: int = 65,
    hybrid_action_budget: int = 80,
    depth_edge_ratio_override: float = 0.0,
    hybrid_action_ratio_override: float = 0.0,
    grid_h: int = 16,
    grid_w: int = 16,
    token_u: Optional[np.ndarray] = None,
    token_v: Optional[np.ndarray] = None,
    sort_keep_indices: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Branch Budget v0: explicit depth/hybrid/fill branch budgets, no global competition.

    Budget allocation (K = keep_k):
        K_depth   = depth_edge_ratio_override * K  (if ratio > 0)  else depth_edge_budget
        K_hybrid  = hybrid_action_ratio_override * K (if ratio > 0) else hybrid_action_budget
        K_fill    = K - K_depth - K_hybrid  (at least 0)

    Selection:
        Branch A (depth/scene): top K_depth by depth_edge_scores (independent budget)
        Branch B (hybrid/action): top K_hybrid by hybrid_final_scores (independent budget)
        Branch C (diversity/fill): spatial grid coverage fill for remaining slots

    Each branch selects independently; overlap between branches is tracked but NOT
    excluded from selection (each branch gets its full budget from its score source).

    Args:
        depth_edge_scores: [N] depth/edge gradient scores.
        hybrid_final_scores: [N] hybrid/action final scores.
        valid_mask: [N] boolean valid token mask.
        keep_k: target token count.
        depth_edge_budget: fixed budget for Branch A (default 65).
        hybrid_action_budget: fixed budget for Branch B (default 80).
        depth_edge_ratio_override: if > 0, overrides depth_edge_budget as fraction of keep_k.
        hybrid_action_ratio_override: if > 0, overrides hybrid_action_budget as fraction of keep_k.
        grid_h, grid_w: grid dimensions for diversity fill.
        token_u, token_v: optional [N] UV grid coordinates for diversity fill.
        sort_keep_indices: sort final indices (default True).

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

    raw_de = _to_1d(depth_edge_scores)
    raw_hyb = _to_1d(hybrid_final_scores)
    n = max(int(raw_de.shape[0]) if raw_de is not None else keep_k,
            int(raw_hyb.shape[0]) if raw_hyb is not None else keep_k)

    raw_valid = _to_1d(valid_mask) if valid_mask is not None else None
    if raw_valid is not None and raw_valid.shape[0] == n:
        valid = raw_valid.astype(bool)
    else:
        valid = np.ones(n, dtype=bool)

    keep_k = int(max(0, min(keep_k, n)))

    # Resolve budgets
    if depth_edge_ratio_override > 0:
        k_depth = max(0, int(round(keep_k * depth_edge_ratio_override)))
    else:
        k_depth = max(0, min(depth_edge_budget, keep_k))

    if hybrid_action_ratio_override > 0:
        k_hyb = max(0, int(round(keep_k * hybrid_action_ratio_override)))
    else:
        k_hyb = max(0, min(hybrid_action_budget, keep_k))

    k_fill = max(0, keep_k - k_depth - k_hyb)

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

    def _spatial_fill_order(
        depth_n: np.ndarray,
        hyb_n: np.ndarray,
        u_arr: Optional[np.ndarray],
        v_arr: Optional[np.ndarray],
        grid_h: int,
        grid_w: int,
    ) -> np.ndarray:
        """Farthest spatial fill: pick tokens to maximize spatial coverage.

        IMPORTANT: This function is called BEFORE reserved_set is built,
        so it cannot exclude reserved tokens. The caller handles that via
        `if idx_i in reserved_set or idx_i in fill_selected: continue`.

        When UV is available: iterate grid cells in a round-robin pattern
        (every row, every column) and pick the highest-scoring token in each cell.
        When UV is unavailable: fall back to tokens sorted by combined score,
        which is intentionally different from pure depth/hybrid ordering
        (we add a tiny index-based tiebreaker to ensure diversity).
        """
        if u_arr is None or v_arr is None:
            fallback = np.zeros(n, dtype=np.float32)
            if depth_n is not None:
                fallback += depth_n
            if hyb_n is not None:
                fallback += hyb_n
            # Add a tiny diverse signal: prefer tokens NOT near each other in index space
            # (simple: add tiny random-ish value based on index to break ties)
            fallback += np.arange(n, dtype=np.float32) * 1e-6
            return _topk_order(fallback)

        u_a = np.asarray(u_arr, dtype=np.int32).reshape(-1)
        v_a = np.asarray(v_arr, dtype=np.int32).reshape(-1)
        u_a = np.clip(u_a, 0, grid_w - 1)
        v_a = np.clip(v_a, 0, grid_h - 1)

        cell_id = v_a * grid_w + u_a

        # Build per-cell lists of (score, index) pairs
        cell_tokens: Dict[int, list] = {}
        for i in range(n):
            cid = int(cell_id[i])
            if cid not in cell_tokens:
                cell_tokens[cid] = []
            # score = max(depth, hybrid) — the most useful token in this cell
            sc = max(
                float(depth_n[i]) if depth_n is not None else -np.inf,
                float(hyb_n[i]) if hyb_n is not None else -np.inf,
            )
            cell_tokens[cid].append((sc, i))

        # Sort tokens within each cell by score descending
        for cid in cell_tokens:
            cell_tokens[cid].sort(key=lambda x: -x[0])

        # Round-robin through cells: row by row, col by col
        order: list[int] = []
        for r in range(grid_h):
            for c in range(grid_w):
                cid = r * grid_w + c
                if cid in cell_tokens and cell_tokens[cid]:
                    order.append(cell_tokens[cid][0][1])

        # Any tokens not yet in order
        order_set = set(order)
        remaining = [i for i in range(n) if i not in order_set and valid[i]]
        order.extend(remaining)
        return np.array(order, dtype=np.int64)

    # ---- Branch A: Depth/Scene (independent budget) ----
    depth_selected: set[int] = set()
    depth_fallback_reason: Optional[str] = None
    if raw_de is not None and k_depth > 0:
        depth_order = _topk_order(raw_de)
        for idx in depth_order:
            idx_i = int(idx)
            if valid[idx_i] and len(depth_selected) < k_depth:
                depth_selected.add(idx_i)
            if len(depth_selected) >= k_depth:
                break
    elif k_depth > 0:
        depth_fallback_reason = "depth_edge_scores_unavailable"

    # ---- Branch B: Hybrid/Action (independent budget) ----
    hybrid_selected: set[int] = set()
    hybrid_fallback_reason: Optional[str] = None
    if raw_hyb is not None and k_hyb > 0:
        hyb_order = _topk_order(raw_hyb)
        for idx in hyb_order:
            idx_i = int(idx)
            if valid[idx_i] and len(hybrid_selected) < k_hyb:
                hybrid_selected.add(idx_i)
            if len(hybrid_selected) >= k_hyb:
                break
    elif k_hyb > 0:
        hybrid_fallback_reason = "hybrid_final_scores_unavailable"

    # Combined reserved set
    reserved_set = depth_selected | hybrid_selected

    # ---- Branch C: Diversity/Fill (fill remaining slots) ----
    # IMPORTANT: fill must exclude depth and hybrid tokens to maintain accounting integrity.
    # Tokens from reserved_set are NOT eligible for fill selection.
    fill_selected: set[int] = set()
    fill_fallback_reason: Optional[str] = None
    if len(reserved_set) < keep_k and k_fill > 0:
        de_norm = _norm(raw_de) if raw_de is not None else np.zeros(n, dtype=np.float32)
        hyb_norm = _norm(raw_hyb) if raw_hyb is not None else np.zeros(n, dtype=np.float32)

        fill_order = _spatial_fill_order(de_norm, hyb_norm, token_u, token_v, grid_h, grid_w)
        for idx in fill_order:
            idx_i = int(idx)
            # EXCLUDE all reserved tokens (depth + hybrid union) to maintain accounting
            if idx_i in reserved_set or idx_i in fill_selected:
                continue
            if not valid[idx_i]:
                continue
            fill_selected.add(idx_i)
            if len(reserved_set) + len(fill_selected) >= keep_k:
                break
        if len(fill_selected) == 0 and k_fill > 0:
            fill_fallback_reason = "no_valid_tokens_for_fill"
    elif k_fill <= 0:
        fill_fallback_reason = "no_fill_budget_remaining"

    # ---- Fallback Stage: fill any remaining slots ----
    fallback_selected: set[int] = set()
    fallback_reason: Optional[str] = None
    if len(reserved_set) + len(fill_selected) < keep_k:
        fallback_reason = "fill_insufficient_tokens"
        remaining_k = keep_k - len(reserved_set) - len(fill_selected)
        all_scores = np.zeros(n, dtype=np.float32)
        if raw_de is not None:
            all_scores += _norm(raw_de)
        if raw_hyb is not None:
            all_scores += _norm(raw_hyb)
        fb_order = _topk_order(all_scores)
        for idx in fb_order:
            idx_i = int(idx)
            if idx_i in reserved_set or idx_i in fill_selected or idx_i in fallback_selected:
                continue
            if not valid[idx_i]:
                continue
            fallback_selected.add(idx_i)
            remaining_k -= 1
            if remaining_k <= 0:
                break

    # ---- Final index set ----
    all_selected = reserved_set | fill_selected | fallback_selected

    # Build keep_indices with strict branch priority and no duplicates.
    # Priority order: depth > hybrid > fill > fallback.
    # Within each branch: sort by final_score DESC, index ASC.
    final_scores_arr = np.zeros(n, dtype=np.float32)
    if raw_hyb is not None:
        final_scores_arr = _norm(raw_hyb)
    elif raw_de is not None:
        final_scores_arr = _norm(raw_de)

    def _branch_order(branch_set: set) -> list:
        return sorted(branch_set, key=lambda i: (-final_scores_arr[i] if valid[i] else -np.inf, i))

    selected: list[int] = []
    selected_set: set[int] = set()

    for branch_order in [depth_selected, hybrid_selected, fill_selected, fallback_selected]:
        for idx_i in _branch_order(branch_order):
            if len(selected) >= keep_k:
                break
            if idx_i not in selected_set:
                selected_set.add(idx_i)
                selected.append(idx_i)
        if len(selected) >= keep_k:
            break

    if sort_keep_indices:
        keep_indices = np.sort(np.asarray(selected, dtype=np.int64))
    else:
        keep_indices = np.asarray(selected, dtype=np.int64)

    keep_indices = keep_indices[(keep_indices >= 0) & (keep_indices < n)]
    keep_indices = np.unique(keep_indices)
    if len(keep_indices) > keep_k:
        keep_indices = keep_indices[:keep_k]

    selected_set = set(int(i) for i in keep_indices)
    final_kept = len(selected_set)

    # ---- Phase accounting (non-overlapping, priority: depth > hybrid > fill > fallback) ----
    # Build non-overlapping sets: each token assigned to the highest-priority branch that selected it
    _depth_only: set[int] = set()
    _hybrid_only: set[int] = set()
    _fill_only: set[int] = set()
    _fallback_only: set[int] = set()

    for idx_i in sorted(selected_set):
        if idx_i in depth_selected:
            _depth_only.add(idx_i)
        elif idx_i in hybrid_selected:
            _hybrid_only.add(idx_i)
        elif idx_i in fill_selected:
            _fill_only.add(idx_i)
        elif idx_i in fallback_selected:
            _fallback_only.add(idx_i)

    depth_actual = len(_depth_only)
    hybrid_actual = len(_hybrid_only)
    fill_actual = len(_fill_only)
    fallback_actual = len(_fallback_only)

    # Accounting validation: non-overlapping sum must equal final_kept
    branch_sum = depth_actual + hybrid_actual + fill_actual + fallback_actual
    branch_accounting_valid = (branch_sum == final_kept)

    # ---- Overlap diagnostics ----
    overlap_de_hyb = len(depth_selected & hybrid_selected)
    overlap_de_hyb_ratio = float(overlap_de_hyb) / float(max(k_depth, 1))

    # DE top-k vs final selection.
    # P11.3 diagnostic semantics: this is the cross-method diagnostic top-k
    # (same convention as depth_edge_fast / edge_reserve), not merely the
    # 65-token depth branch budget. This answers which final branch retained
    # the broader DE top-k set. It does not change keep_indices.
    diagnostic_topk_k = max(1, min(n, int(round(keep_k * 0.80))))
    de_topk_all: set[int] = set()
    if raw_de is not None:
        de_full_order = _topk_order(raw_de)
        for idx in de_full_order:
            idx_i = int(idx)
            if valid[idx_i]:
                de_topk_all.add(idx_i)
            if len(de_topk_all) >= diagnostic_topk_k:
                break
    dep_topk_count = len(de_topk_all)
    dep_topk_in_final = len(de_topk_all & selected_set)
    dep_topk_dropped = dep_topk_count - dep_topk_in_final
    dep_topk_dropped_ratio = float(dep_topk_dropped) / float(max(dep_topk_count, 1))

    # Hybrid top-k vs final selection, using the same diagnostic top-k size.
    hyb_topk_all: set[int] = set()
    if raw_hyb is not None:
        hyb_full_order = _topk_order(raw_hyb)
        for idx in hyb_full_order:
            idx_i = int(idx)
            if valid[idx_i]:
                hyb_topk_all.add(idx_i)
            if len(hyb_topk_all) >= diagnostic_topk_k:
                break
    hyb_topk_count = len(hyb_topk_all)
    hyb_topk_in_final = len(hyb_topk_all & selected_set)
    hyb_topk_dropped = hyb_topk_count - hyb_topk_in_final
    hyb_topk_dropped_ratio = float(hyb_topk_dropped) / float(max(hyb_topk_count, 1))

    # Non-reserved DE drop (diagnostic DE top-k not in raw hybrid branch candidates)
    non_reserved_de = de_topk_all - hybrid_selected
    non_res_de_dropped = len(non_reserved_de - selected_set)
    non_res_de_dropped_ratio = float(non_res_de_dropped) / float(max(len(non_reserved_de), 1)) if non_reserved_de else None

    # Integrity checks
    keep_indices_unique = (np.unique(keep_indices).size == keep_indices.size)
    keep_indices_sorted = bool(np.all(keep_indices[:-1] <= keep_indices[1:])) if keep_indices.size > 1 else True
    keep_indices_in_range = bool(np.all((keep_indices >= 0) & (keep_indices < n)))
    selected_count_equals_budget = (final_kept == keep_k)

    metadata = {
        "strategy": "robot_geo_branch_budget_v0",
        "selector_function_name": "select_branch_budget_v0",
        "selection_strategy_name": "robot_geo_branch_budget_v0",
        "selection_stage_name": "branch_budget_de_hyb_fill_v0",

        # Budget targets
        "total_keep_budget": int(keep_k),
        "depth_edge_budget": int(k_depth),
        "hybrid_action_budget": int(k_hyb),
        "diversity_fill_budget": int(k_fill),
        "temporal_budget": None,
        "depth_edge_budget_ratio_override": float(depth_edge_ratio_override),
        "hybrid_action_budget_ratio_override": float(hybrid_action_ratio_override),

        # Actual counts selected
        "depth_edge_budget_actual": depth_actual,
        "hybrid_action_budget_actual": hybrid_actual,
        "diversity_fill_budget_actual": fill_actual,
        "temporal_budget_actual": None,

        # Selection accounting (mutually exclusive)
        "selected_by_depth_branch": depth_actual,
        "selected_by_hybrid_branch": hybrid_actual,
        "selected_by_fill": fill_actual,
        "selected_by_fallback": fallback_actual,
        "selected_by_temporal": 0,
        "selected_unattributed": 0,
        # Aliases expected by finalize_selection_debug_info (P7/P11 compat)
        "selected_by_fill_count": fill_actual,
        "selected_by_depth_edge_count": depth_actual,
        "selected_by_robot_geo_count": hybrid_actual,
        "selected_by_fallback_count": fallback_actual,
        "branch_accounting_valid": branch_accounting_valid,
        "branch_sum": branch_sum,
        "branch_sum_error": abs(branch_sum - final_kept),

        # Fallback tracking
        "depth_fallback_reason": depth_fallback_reason,
        "hybrid_fallback_reason": hybrid_fallback_reason,
        "fill_fallback_reason": fill_fallback_reason,
        "fallback_reason": fallback_reason,
        "fallback_actual": fallback_actual,

        # Competition diagnostics
        "depth_edge_topk_count": dep_topk_count,
        "depth_edge_topk_kept_in_final_count": dep_topk_in_final,
        "depth_edge_topk_dropped_count": dep_topk_dropped,
        "depth_edge_topk_dropped_ratio": dep_topk_dropped_ratio,
        "hybrid_final_score_topk_count": hyb_topk_count,
        "hybrid_final_score_topk_kept_count": hyb_topk_in_final,
        "hybrid_final_score_topk_dropped_count": hyb_topk_dropped,
        "hybrid_final_score_topk_dropped_ratio": hyb_topk_dropped_ratio,
        "overlap_depth_edge_hybrid_count": overlap_de_hyb,
        "overlap_depth_edge_hybrid_ratio": overlap_de_hyb_ratio,
        "non_reserved_depth_edge_dropped_ratio": non_res_de_dropped_ratio,
        "non_reserved_depth_edge_count": len(non_reserved_de),
        "non_reserved_depth_edge_kept": len(non_reserved_de & selected_set),
        "non_reserved_depth_edge_dropped": non_res_de_dropped,

        # ---- P11.3: DE top-k attribution by branch ----
        # Which branch ultimately kept each DE top-k token?
        # final_kept is partitioned into _depth_only, _hybrid_only, _fill_only, _fallback_only.
        # _depth_only ∩ de_topk_all = kept by depth branch
        # _hybrid_only ∩ de_topk_all = kept by hybrid branch
        # _fill_only ∩ de_topk_all = kept by fill branch
        # de_topk_all - final_kept = dropped
        "depth_edge_topk_kept_by_depth_branch_count": len(de_topk_all & _depth_only),
        "depth_edge_topk_kept_by_hybrid_branch_count": len(de_topk_all & _hybrid_only),
        "depth_edge_topk_kept_by_fill_branch_count": len(de_topk_all & _fill_only),
        "depth_edge_topk_kept_by_fallback_count": len(de_topk_all & _fallback_only),
        "depth_edge_topk_survival_ratio": len(de_topk_all & selected_set) / float(max(len(de_topk_all), 1)),

        # ---- P11.3: Hybrid/final-score top-k attribution by branch ----
        # Which branch ultimately kept each HYB top-k token?
        "hybrid_final_score_topk_kept_by_depth_branch_count": len(hyb_topk_all & _depth_only),
        "hybrid_final_score_topk_kept_by_hybrid_branch_count": len(hyb_topk_all & _hybrid_only),
        "hybrid_final_score_topk_kept_by_fill_branch_count": len(hyb_topk_all & _fill_only),
        "hybrid_final_score_topk_kept_by_fallback_count": len(hyb_topk_all & _fallback_only),
        "hybrid_final_score_topk_survival_ratio": len(hyb_topk_all & selected_set) / float(max(len(hyb_topk_all), 1)),

        # Legacy alias (for backward compatibility with P8 overlay integrity checker)
        "hybrid_topk_kept_by_depth_branch_count": len(hyb_topk_all & _depth_only),
        "hybrid_topk_kept_by_hybrid_branch_count": len(hyb_topk_all & _hybrid_only),
        "hybrid_topk_kept_by_fill_branch_count": len(hyb_topk_all & _fill_only),
        "hybrid_topk_kept_by_fallback_count": len(hyb_topk_all & _fallback_only),
        "hybrid_topk_survival_ratio": len(hyb_topk_all & selected_set) / float(max(len(hyb_topk_all), 1)),

        # Mutually-exclusive final branch index sets (for overlay visualization).
        # Raw branch candidates may overlap; these sets reflect the final
        # priority attribution depth > hybrid > fill > fallback.
        "depth_branch_indices": sorted(_depth_only),
        "hybrid_branch_indices": sorted(_hybrid_only),
        "fill_branch_indices": sorted(_fill_only),
        "fallback_branch_indices": sorted(_fallback_only),
        "raw_depth_branch_candidate_count": len(depth_selected),
        "raw_hybrid_branch_candidate_count": len(hybrid_selected),
        "final_keep_indices": sorted(int(i) for i in keep_indices),

        # Integrity
        "keep_indices_unique": keep_indices_unique,
        "keep_indices_sorted": keep_indices_sorted,
        "keep_indices_in_range": keep_indices_in_range,
        "selected_count_equals_budget": selected_count_equals_budget,
        "branch_sum_equals_kept": branch_accounting_valid,
        "final_kept": final_kept,
        "expected_kept": int(keep_k),
        "grid_shape": [grid_h, grid_w],
        "K_total": int(keep_k),
        "branch_budget_v0": True,
    }

    metadata = finalize_selection_debug_info(
        metadata,
        selector_function_name="select_branch_budget_v0",
        strategy="robot_geo_branch_budget_v0",
        keep_indices=keep_indices,
        num_tokens=n,
        keep_count=keep_k,
        scores=final_scores_arr,
        requested_keep_ratio=float(keep_k) / float(n) if n else None,
        fallback_used=fallback_reason is not None,
        fallback_reason=fallback_reason,
    )
    return keep_indices, metadata
