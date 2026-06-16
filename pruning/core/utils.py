"""Shared hook utility helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


def _selected_score_stats(aux_metrics: Dict[str, Any], idx: np.ndarray, num_tokens: int) -> Dict[str, Optional[float]]:
    """Compute score statistics over selected (kept) tokens only."""
    result: Dict[str, Optional[float]] = {}
    component_names = [
        ("edge_scores", "depth_edge_score"),
        ("near_scores", "distance_score"),
        ("rule_v0_motion_cone_scores", "motion_cone_score"),
        ("rule_v0_workspace_scores", "workspace_score"),
        ("rule_v0_contact_risk_scores", "contact_risk_score"),
        ("final_scores", "final_geometry_score"),
    ]
    for arr_key, prefix in component_names:
        arr = aux_metrics.get(arr_key)
        if arr is None:
            continue
        a = np.asarray(arr, dtype=np.float32).reshape(-1)
        if a.size == 0:
            continue
        valid_idx = idx[(idx >= 0) & (idx < num_tokens)]
        if valid_idx.size == 0:
            continue
        selected = a[valid_idx]
        selected = selected[np.isfinite(selected)]
        if selected.size == 0:
            continue
        result[f"selected_{prefix}_mean"] = float(np.mean(selected))
        sorted_sel = np.sort(selected)
        result[f"selected_{prefix}_p50"] = float(np.median(selected))
        p90_i = max(0, int(round(0.90 * selected.size)) - 1)
        result[f"selected_{prefix}_p90"] = float(sorted_sel[min(p90_i, selected.size - 1)])
        result[f"selected_{prefix}_max"] = float(np.max(selected))
    return result


def _append_score_stats(stats: Dict[str, Any], key: str, arr: np.ndarray, valid_mask: np.ndarray) -> None:
    """Compute and append distribution stats for a score array into the stats dict."""
    if arr is None:
        return
    a = np.asarray(arr, dtype=np.float32).reshape(-1)
    v = a[np.isfinite(a) & valid_mask]
    if v.size == 0:
        return
    prefix = key.replace("_scores", "_score")
    stats[f"{prefix}_mean"] = float(np.mean(v))
    stats[f"{prefix}_std"] = float(np.std(v))
    stats[f"{prefix}_min"] = float(np.min(v))
    stats[f"{prefix}_max"] = float(np.max(v))
    sorted_v = np.sort(v)
    stats[f"{prefix}_p50"] = float(np.median(v))
    p90_idx = max(0, int(round(0.90 * v.size)) - 1)
    stats[f"{prefix}_p90"] = float(sorted_v[min(p90_idx, v.size - 1)])
    if "motion_cone" in key:
        pos_ratio = float(np.mean(v > 1e-6))
        stats[f"{prefix}_positive_ratio"] = pos_ratio
        stats[f"{prefix}_zero_ratio"] = 1.0 - pos_ratio
        # motion direction norm (if stored)
        if "motion_direction" in stats and stats["motion_direction"] is not None:
            md = np.asarray(stats["motion_direction"], dtype=np.float32).reshape(-1)
            md_valid = np.isfinite(md)
            if np.any(md_valid):
                stats["motion_dir_norm_mean"] = float(np.mean(np.abs(md[md_valid])))
                stats["motion_dir_norm_min"] = float(np.min(np.abs(md[md_valid])))
                stats["motion_dir_norm_max"] = float(np.max(np.abs(md[md_valid])))


def _compute_token_selection_attribution(
    aux_metrics: Dict[str, Any],
    keep_indices: np.ndarray,
    num_tokens: int,
    token_grid_shape: Tuple[int, int],
) -> Dict[str, Any]:
    """Compute P1-1 token selection attribution and top-k competition diagnostics.

    Returns a flat dict of diagnostic fields.
    """
    result: Dict[str, Any] = {}
    n = int(num_tokens)
    if n <= 0:
        return result

    grid_h, grid_w = int(token_grid_shape[0]), int(token_grid_shape[1])
    idx = np.asarray(keep_indices, dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < n)]
    k_final = len(idx)
    keep_set = set(idx.tolist()) if k_final > 0 else set()

    # ---- Workspace score diagnostics ----
    ws_arr = aux_metrics.get("workspace_scores")
    if ws_arr is not None:
        ws = np.asarray(ws_arr, dtype=np.float32).reshape(-1)
        valid = np.isfinite(ws)
        ws_v = ws[valid] if np.any(valid) else np.array([], dtype=np.float32)
        if ws_v.size > 0:
            result["workspace_score_min"] = float(np.min(ws_v))
            result["workspace_score_unique_count"] = len(set(float(v) for v in ws_v))
            all_one = bool(np.allclose(ws_v, 1.0, atol=1e-6))
            result["workspace_all_one"] = all_one
            if all_one:
                bounds = aux_metrics.get("workspace_bounds")
                if bounds is not None:
                    result["workspace_all_one_reason"] = (
                        f"all_tokens_inside_workspace_bounds:{bounds}"
                    )
                else:
                    result["workspace_all_one_reason"] = (
                        "all_tokens_inside_default_workspace_bounds:"
                        "((-2.0,2.0),(-2.0,2.0),(-0.5,2.0))"
                    )
            valid_ws = np.sum(valid)
            result["workspace_valid_token_ratio"] = float(valid_ws) / float(n) if n > 0 else None
        else:
            result["workspace_score_min"] = None
            result["workspace_score_unique_count"] = 0
            result["workspace_all_one"] = None
            result["workspace_all_one_reason"] = "no_valid_workspace_scores"
            result["workspace_valid_token_ratio"] = 0.0

    ws_fallback = aux_metrics.get("workspace_fallback_used")
    if ws_fallback is not None:
        result["workspace_fallback_used"] = bool(ws_fallback)

    ws_bounds = aux_metrics.get("workspace_bounds")
    if ws_bounds is not None:
        result["workspace_bounds"] = str(ws_bounds)

    # ---- Score component unique counts ----
    for arr_key, field_name in [
        ("near_scores", "near_score_unique_count"),
        ("edge_scores", "depth_edge_score_unique_count"),
        ("motion_cone_scores", "motion_cone_score_unique_count"),
        ("final_scores", "final_score_unique_count"),
    ]:
        arr = aux_metrics.get(arr_key)
        if arr is not None:
            a = np.asarray(arr, dtype=np.float32).reshape(-1)
            valid = np.isfinite(a)
            v = a[valid] if np.any(valid) else np.array([], dtype=np.float32)
            result[field_name] = len(set(float(x) for x in v)) if v.size > 0 else 0

    # ---- Top-k overlap competition (depth_edge vs robot_geo vs final) ----
    # Get depth_edge scores for top-k
    edge_arr = aux_metrics.get("edge_scores")
    edge_scores_np = np.asarray(edge_arr, dtype=np.float32).reshape(-1) if edge_arr is not None else None

    # Get robot_geo final scores.
    # Priority: hybrid_final_scores (P1, precomputed) > final_scores (rule_v0 fallback)
    hybrid_arr = aux_metrics.get("hybrid_final_scores")
    final_arr = hybrid_arr if hybrid_arr is not None else aux_metrics.get("final_scores")
    final_scores_np = np.asarray(final_arr, dtype=np.float32).reshape(-1) if final_arr is not None else None

    if edge_scores_np is not None and edge_scores_np.size >= n:
        # depth_edge top-k: 80% of keep_k (matching select_hybrid_v1 phase 1)
        edge_topk_k = max(1, int(round(k_final * 0.80))) if k_final > 0 else 0
        edge_adj = np.where(np.isfinite(edge_scores_np), edge_scores_np, -np.inf)
        edge_order = np.lexsort((np.arange(n), -edge_adj))
        edge_topk = set(int(edge_order[i]) for i in range(min(edge_topk_k, n)))

        result["depth_edge_topk_count"] = len(edge_topk)
        result["depth_edge_topk_kept_in_final_count"] = len(edge_topk & keep_set)
        result["depth_edge_topk_dropped_count"] = len(edge_topk - keep_set)
        result["depth_edge_topk_dropped_ratio"] = (
            len(edge_topk - keep_set) / len(edge_topk) if edge_topk else None
        )
    else:
        result["depth_edge_topk_count"] = None
        result["depth_edge_topk_kept_in_final_count"] = None
        result["depth_edge_topk_dropped_count"] = None
        result["depth_edge_topk_dropped_ratio"] = None

    if final_scores_np is not None and final_scores_np.size >= n:
        # robot_geo top-k: same as depth_edge top-k count
        geo_topk_k = max(1, int(round(k_final * 0.80))) if k_final > 0 else 0
        geo_adj = np.where(np.isfinite(final_scores_np), final_scores_np, -np.inf)
        geo_order = np.lexsort((np.arange(n), -geo_adj))
        geo_topk = set(int(geo_order[i]) for i in range(min(geo_topk_k, n)))

        result["robot_geo_topk_count"] = len(geo_topk)
        result["robot_geo_topk_kept_in_final_count"] = len(geo_topk & keep_set)
        result["robot_geo_topk_dropped_count"] = len(geo_topk - keep_set)
        result["robot_geo_topk_dropped_ratio"] = (
            len(geo_topk - keep_set) / len(geo_topk) if geo_topk else None
        )

        if edge_scores_np is not None and edge_scores_np.size >= n:
            overlap = edge_topk & geo_topk
            result["overlap_depth_edge_robot_geo_count"] = len(overlap)
            total = len(edge_topk | geo_topk)
            result["overlap_depth_edge_robot_geo_ratio"] = len(overlap) / total if total > 0 else None
    else:
        result["robot_geo_topk_count"] = None
        result["robot_geo_topk_kept_in_final_count"] = None
        result["robot_geo_topk_dropped_count"] = None
        result["robot_geo_topk_dropped_ratio"] = None
        result["overlap_depth_edge_robot_geo_count"] = None
        result["overlap_depth_edge_robot_geo_ratio"] = None

    result["final_selected_count"] = k_final

    # ---- Selected token attribution ratios ----
    if k_final > 0 and n > 0:
        # Selected high depth_edge but low robot_geo count
        if edge_scores_np is not None and final_scores_np is not None:
            edge_topk_thresh = float(np.percentile(edge_scores_np[edge_scores_np > -np.inf], 80))
            geo_topk_thresh = float(np.percentile(final_scores_np[final_scores_np > -np.inf], 80))
            high_edge_low_geo = 0
            for i in idx:
                if i < n and edge_scores_np[i] >= edge_topk_thresh:
                    if i < n and final_scores_np[i] < geo_topk_thresh:
                        high_edge_low_geo += 1
            result["selected_high_depth_edge_but_low_robot_geo_count"] = high_edge_low_geo
        else:
            result["selected_high_depth_edge_but_low_robot_geo_count"] = None

        # Dropped high depth_edge count
        if edge_scores_np is not None:
            edge_topk_thresh = float(np.percentile(edge_scores_np[edge_scores_np > -np.inf], 80))
            dropped_high_edge = 0
            for i in range(n):
                if i not in keep_set and edge_scores_np[i] >= edge_topk_thresh:
                    dropped_high_edge += 1
            result["dropped_high_depth_edge_tokens_count"] = dropped_high_edge
        else:
            result["dropped_high_depth_edge_tokens_count"] = None

        # Dropped high robot_geo count
        if final_scores_np is not None:
            geo_topk_thresh = float(np.percentile(final_scores_np[final_scores_np > -np.inf], 80))
            dropped_high_geo = 0
            for i in range(n):
                if i not in keep_set and final_scores_np[i] >= geo_topk_thresh:
                    dropped_high_geo += 1
            result["dropped_high_robot_geo_tokens_count"] = dropped_high_geo
        else:
            result["dropped_high_robot_geo_tokens_count"] = None

    # ---- Selected token spatial distribution (UV grid coordinates) ----
    token_u = aux_metrics.get("token_u")
    token_v = aux_metrics.get("token_v")
    gripper_pixel = aux_metrics.get("gripper_pixel")

    if k_final > 0 and token_u is not None and token_v is not None:
        u_arr = np.asarray(token_u, dtype=np.float32).reshape(-1)
        v_arr = np.asarray(token_v, dtype=np.float32).reshape(-1)
        if u_arr.size >= n and v_arr.size >= n:
            u_sel = u_arr[idx[idx < u_arr.size]]
            v_sel = v_arr[idx[idx < v_arr.size]]
            if u_sel.size > 0:
                result["selected_token_u_mean"] = float(np.mean(u_sel))
                result["selected_token_u_std"] = float(np.std(u_sel)) if u_sel.size > 1 else 0.0
                result["selected_token_v_mean"] = float(np.mean(v_sel))
                result["selected_token_v_std"] = float(np.std(v_sel)) if v_sel.size > 1 else 0.0
                result["selected_token_bbox_u_min"] = int(np.min(u_sel))
                result["selected_token_bbox_u_max"] = int(np.max(u_sel))
                result["selected_token_bbox_v_min"] = int(np.min(v_sel))
                result["selected_token_bbox_v_max"] = int(np.max(v_sel))

            # Grid quadrant histogram: [TL, TR, BL, BR]
            if grid_h > 0 and grid_w > 0 and u_sel.size > 0:
                mid_u = float(np.median(u_arr[:n]))
                mid_v = float(np.median(v_arr[:n]))
                q_tl = int(np.sum((u_sel < mid_u) & (v_sel < mid_v)))
                q_tr = int(np.sum((u_sel >= mid_u) & (v_sel < mid_v)))
                q_bl = int(np.sum((u_sel < mid_u) & (v_sel >= mid_v)))
                q_br = int(np.sum((u_sel >= mid_u) & (v_sel >= mid_v)))
                result["selected_token_grid_quadrant_histogram"] = f"[{q_tl},{q_tr},{q_bl},{q_br}]"

            # Gripper pixel distance
            if gripper_pixel is not None:
                gx, gy = float(gripper_pixel[0]), float(gripper_pixel[1])
                if u_sel.size > 0 and v_sel.size > 0:
                    dists = np.sqrt((u_sel - gx) ** 2 + (v_sel - gy) ** 2)
                    result["selected_token_near_gripper_pixel_dist_mean"] = float(np.mean(dists))
                    result["selected_token_near_gripper_pixel_dist_median"] = float(np.median(dists))

    return result


def _first_present(values: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if values.get(key) is not None:
            return values.get(key)
    return None


def _arr_stats(arr: Any, stat: str = "min") -> Optional[str]:
    """Convert an array to a 'x,y,z' string summary for min/max/mean/std."""
    if arr is None:
        return None
    try:
        a = np.asarray(arr, dtype=np.float32).reshape(-1, 3)
        valid = np.all(np.isfinite(a), axis=1)
        if not np.any(valid):
            return None
        va = a[valid]
        if stat == "min":
            vals = np.nanmin(va, axis=0)
        elif stat == "max":
            vals = np.nanmax(va, axis=0)
        elif stat == "mean":
            vals = np.nanmean(va, axis=0)
        elif stat == "std":
            vals = np.nanstd(va, axis=0)
        else:
            return None
        return f"{float(vals[0]):.4f},{float(vals[1]):.4f},{float(vals[2]):.4f}"
    except Exception:
        return None


def _arr_to_str(arr: Any) -> Optional[str]:
    """Convert a 3-element array to 'x,y,z' string."""
    if arr is None:
        return None
    try:
        a = np.asarray(arr, dtype=np.float32).reshape(-1)
        vals = a[:3]
        if not np.all(np.isfinite(vals)):
            return None
        return f"{float(vals[0]):.4f},{float(vals[1]):.4f},{float(vals[2]):.4f}"
    except Exception:
        return None


# =============================================================================
# P1: Hybrid score helpers for selection attribution
# =============================================================================


def _norm_hybrid_component(arr: Optional[np.ndarray], valid_mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Min-max normalize a score array over valid tokens.

    Returns zeros for invalid entries. Matches the normalization logic in
    ``select_hybrid_v1._norm()``.
    """
    if arr is None:
        return None
    a = np.nan_to_num(np.asarray(arr, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    else:
        valid = np.isfinite(a) & (np.abs(a) < 1e10)
    out = np.zeros_like(a)
    if not np.any(valid):
        return out
    lo, hi = float(np.min(a[valid])), float(np.max(a[valid]))
    if hi - lo > 1e-8:
        out[valid] = (a[valid] - lo) / (hi - lo)
    return out


def _build_hybrid_final_scores(
    edge: Optional[np.ndarray],
    near: Optional[np.ndarray],
    contact: Optional[np.ndarray],
    corridor: Optional[np.ndarray],
    valid_mask: np.ndarray,
    weights: Dict[str, float],
) -> Optional[np.ndarray]:
    """Build hybrid v1 final scores using the same formula as ``select_hybrid_v1``.

    If any component is None, that weight is redistributed proportionally.
    """
    w_edge = float(weights.get("w_edge", 0.45))
    w_near = float(weights.get("w_near", 0.20))
    w_contact = float(weights.get("w_contact", 0.20))
    w_corr = float(weights.get("w_corr", 0.10))
    w_diverse = float(weights.get("w_diverse", 0.05))

    n = 0
    if edge is not None:
        n = int(np.asarray(edge).reshape(-1).shape[0])
    elif near is not None:
        n = int(np.asarray(near).reshape(-1).shape[0])

    if n == 0:
        return None

    if edge is not None:
        norm_edge = _norm_hybrid_component(edge)
    else:
        norm_edge = np.zeros(n, dtype=np.float32)

    if near is not None:
        norm_near = _norm_hybrid_component(near)
    else:
        norm_near = np.zeros(n, dtype=np.float32)

    if contact is not None:
        norm_contact = _norm_hybrid_component(contact)
    else:
        norm_contact = np.zeros(n, dtype=np.float32)

    if corridor is not None:
        norm_corr = _norm_hybrid_component(corridor)
    else:
        norm_corr = np.zeros(n, dtype=np.float32)

    valid = np.asarray(valid_mask, dtype=np.bool_).reshape(-1) if valid_mask is not None else np.ones(n, dtype=np.bool_)

    scores = w_edge * norm_edge + w_near * norm_near + w_contact * norm_contact + w_corr * norm_corr
    # No w_diverse since spatial diversity is selection-based, not score-based
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    scores[~valid] = 0.0
    return scores
