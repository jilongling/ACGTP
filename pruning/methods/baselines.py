"""Baseline and utility selector implementations."""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .registry import SIMPLE_SCORE_TOPK_STRATEGIES
from .utils import (
    finalize_selection_debug_info,
    select_score_topk,
    _normalize_for_selection,
)


def select_tokens_with_spatial_diversity(
    scores: np.ndarray,
    keep_count: int,
    reserve_k: int = 32,
    invalid_mask: Optional[np.ndarray] = None,
    grid_size: int = 16,
    cell_grid: int = 4,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Batch wrapper around the spatial-diversity selector.

    Args:
        scores: [B, N] or [N] score array.
        invalid_mask: optional boolean mask where True means invalid.
    """
    scores_np = np.asarray(scores, dtype=np.float32)
    was_1d = scores_np.ndim == 1
    if was_1d:
        scores_np = scores_np[None, :]
    if scores_np.ndim != 2:
        raise ValueError(f"scores must have shape [B, N] or [N], got {scores_np.shape}")

    if invalid_mask is None:
        valid_batch = np.ones(scores_np.shape, dtype=np.bool_)
    else:
        invalid_np = np.asarray(invalid_mask, dtype=np.bool_)
        if invalid_np.ndim == 1:
            invalid_np = invalid_np[None, :]
        valid_batch = ~invalid_np

    keep_rows = []
    metas = []
    depth_quota = max(0, int(keep_count) - min(int(reserve_k), int(keep_count)))
    for b in range(scores_np.shape[0]):
        idx, meta = select_depth_edge_diverse_indices(
            scores=scores_np[b],
            valid_mask=valid_batch[b],
            keep_total=int(keep_count),
            depth_quota=depth_quota,
            grid_size=grid_size,
            cell_grid=cell_grid,
        )
        keep_rows.append(idx)
        metas.append(meta)
    keep = np.stack(keep_rows, axis=0).astype(np.int64)
    return (keep[0] if was_1d else keep), {"batch_size": int(scores_np.shape[0]), "per_batch": metas}


def select_keep_indices(
    strategy: str,
    num_tokens: int,
    keep_count: int,
    scores: Optional[np.ndarray] = None,
    valid_mask: Optional[np.ndarray] = None,
    seed: int = 7,
    grid_size: int = 16,
    cell_grid: int = 4,
    reserve_tokens: int = 32,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if strategy == "none" or keep_count >= num_tokens:
        keep = np.arange(num_tokens, dtype=np.int64)
        meta = finalize_selection_debug_info(
            {"strategy": "none"},
            selector_function_name="select_keep_indices",
            strategy="none",
            keep_indices=keep,
            num_tokens=num_tokens,
            keep_count=num_tokens,
            scores=None,
            requested_keep_ratio=1.0,
        )
        return keep, meta
    if strategy == "random":
        rng = np.random.default_rng(seed)
        idx = np.arange(num_tokens)
        rng.shuffle(idx)
        keep = np.sort(idx[:keep_count]).astype(np.int64)
        meta = finalize_selection_debug_info(
            {"strategy": "random"},
            selector_function_name="select_keep_indices",
            strategy="random",
            keep_indices=keep,
            num_tokens=num_tokens,
            keep_count=keep_count,
            scores=None,
            requested_keep_ratio=float(keep_count) / float(num_tokens) if num_tokens else None,
        )
        return keep, meta
    if strategy == "uniform_grid":
        keep = select_uniform_grid_indices(num_tokens, keep_count)
        meta = finalize_selection_debug_info(
            {"strategy": "uniform_grid"},
            selector_function_name="select_keep_indices",
            strategy="uniform_grid",
            keep_indices=keep,
            num_tokens=num_tokens,
            keep_count=keep_count,
            scores=None,
            requested_keep_ratio=float(keep_count) / float(num_tokens) if num_tokens else None,
        )
        return keep, meta
    if scores is None:
        keep = select_uniform_grid_indices(num_tokens, keep_count)
        meta = finalize_selection_debug_info(
            {"strategy": strategy, "fallback": "uniform_grid"},
            selector_function_name="select_keep_indices",
            strategy=strategy,
            keep_indices=keep,
            num_tokens=num_tokens,
            keep_count=keep_count,
            scores=None,
            requested_keep_ratio=float(keep_count) / float(num_tokens) if num_tokens else None,
            fallback_used=True,
            fallback_reason="missing_scores_uniform_grid",
        )
        return keep, meta
    if strategy in SIMPLE_SCORE_TOPK_STRATEGIES:
        keep = select_score_topk(scores, keep_count, valid_mask)
        meta = finalize_selection_debug_info(
            {"strategy": strategy},
            selector_function_name="select_keep_indices",
            strategy=strategy,
            keep_indices=keep,
            num_tokens=num_tokens,
            keep_count=keep_count,
            scores=scores,
            requested_keep_ratio=float(keep_count) / float(num_tokens) if num_tokens else None,
        )
        return keep, meta
    if strategy == "depth_edge_fast_diverse":
        depth_quota = max(0, keep_count - min(reserve_tokens, keep_count))
        keep, meta = select_depth_edge_diverse_indices(
            scores=scores,
            valid_mask=valid_mask,
            keep_total=keep_count,
            depth_quota=depth_quota,
            grid_size=grid_size,
            cell_grid=cell_grid,
        )
        meta = finalize_selection_debug_info(
            meta,
            selector_function_name="select_keep_indices",
            strategy=strategy,
            keep_indices=keep,
            num_tokens=num_tokens,
            keep_count=keep_count,
            scores=scores,
            requested_keep_ratio=float(keep_count) / float(num_tokens) if num_tokens else None,
        )
        return keep, meta
    raise ValueError(f"Unknown pruning strategy: {strategy}")


def select_tokens_contact_budget(
    edge_score: np.ndarray,
    geo_contact_score: np.ndarray,
    valid_mask: Optional[np.ndarray],
    keep_k: int,
    k_edge: int,
    k_geo: int,
    k_diverse: int,
    grid_h: int = 16,
    grid_w: int = 16,
    cells_h: int = 4,
    cells_w: int = 4,
    return_indices: bool = False,
    detailed_timing: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Select tokens with an edge-first, contact-supplement budget.

    The returned indices are sorted so projector-token order is preserved.
    """
    edge = np.nan_to_num(np.asarray(edge_score, dtype=np.float32).reshape(-1), nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    geo = np.nan_to_num(np.asarray(geo_contact_score, dtype=np.float32).reshape(-1), nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    n = int(edge.shape[0])
    if geo.shape[0] != n:
        raise ValueError(f"edge_score and geo_contact_score length mismatch: {n} vs {geo.shape[0]}")

    keep_k = int(max(0, min(keep_k, n)))
    k_edge = int(max(0, min(k_edge, keep_k)))
    k_geo = int(max(0, min(k_geo, keep_k - k_edge)))
    k_diverse = int(max(0, keep_k - k_edge - k_geo))

    valid = np.ones(n, dtype=np.bool_) if valid_mask is None else np.asarray(valid_mask, dtype=np.bool_).reshape(-1)
    if valid.shape[0] != n:
        raise ValueError(f"valid_mask length mismatch: {valid.shape[0]} vs {n}")

    edge_norm = _normalize_for_selection(edge, valid)
    geo_norm = _normalize_for_selection(geo, valid)
    reserve_score = 0.7 * edge_norm + 0.3 * geo_norm
    reserve_score = np.nan_to_num(reserve_score, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    reserve_score[~valid] = -np.inf
    edge_order = _ordered_valid_indices(edge, valid)
    geo_order = _ordered_valid_indices(geo, valid)
    reserve_order = _ordered_valid_indices(reserve_score, valid)

    t0 = time.perf_counter()
    selected_edge = _take_from_order(edge_order, k_edge, excluded=set())
    edge_selection_ms = (time.perf_counter() - t0) * 1000.0 if detailed_timing else None
    edge_set = set(int(i) for i in selected_edge)
    t0 = time.perf_counter()
    raw_geo_top = _take_from_order(geo_order, k_geo, excluded=set())
    overlap_edge_geo_before_dedup = int(sum(1 for i in raw_geo_top if int(i) in edge_set))
    selected_geo = _take_from_order(geo_order, k_geo, excluded=edge_set)
    geo_selection_ms = (time.perf_counter() - t0) * 1000.0 if detailed_timing else None
    selected = set(int(i) for i in selected_edge)
    selected.update(int(i) for i in selected_geo)

    t0 = time.perf_counter()
    selected_diverse = _select_diverse_reserve(
        reserve_score=reserve_score,
        valid=valid,
        selected=selected,
        reserve_order=reserve_order,
        quota=k_diverse,
        grid_h=grid_h,
        grid_w=grid_w,
        cells_h=cells_h,
        cells_w=cells_w,
    )
    diverse_selection_ms = (time.perf_counter() - t0) * 1000.0 if detailed_timing else None
    selected.update(int(i) for i in selected_diverse)

    t0 = time.perf_counter()
    fallback_reason = None
    if len(selected) < keep_k:
        for idx in edge_order:
            idx = int(idx)
            if idx not in selected:
                selected.add(idx)
            if len(selected) >= keep_k:
                break
    if len(selected) < keep_k:
        fallback_reason = "valid_tokens_less_than_keep_k"
        for idx in range(n):
            if idx not in selected:
                selected.add(idx)
            if len(selected) >= keep_k:
                break

    final_priority = (
        [int(i) for i in selected_edge]
        + [int(i) for i in selected_geo]
        + [int(i) for i in selected_diverse]
        + [int(i) for i in reserve_order]
        + list(range(n))
    )
    final_ordered = []
    used = set()
    for idx in final_priority:
        if idx in selected and idx not in used:
            used.add(idx)
            final_ordered.append(idx)
        if len(final_ordered) >= keep_k:
            break
    keep_indices = np.sort(np.asarray(final_ordered[:keep_k], dtype=np.int64))
    final_merge_ms = (time.perf_counter() - t0) * 1000.0 if detailed_timing else None

    edge_actual = len(set(map(int, selected_edge)).intersection(set(map(int, keep_indices))))
    geo_actual = len(set(map(int, selected_geo)).intersection(set(map(int, keep_indices))))
    diverse_actual = len(set(map(int, selected_diverse)).intersection(set(map(int, keep_indices))))
    if edge_actual + geo_actual + diverse_actual < keep_indices.shape[0]:
        diverse_actual += int(keep_indices.shape[0] - edge_actual - geo_actual - diverse_actual)

    metadata = {
        "strategy": "robot_geo_contact_budget",
        "K_total": int(keep_k),
        "K_edge_target": int(k_edge),
        "K_geo_target": int(k_geo),
        "K_diverse_target": int(k_diverse),
        "K_edge_actual": int(edge_actual),
        "K_geo_actual": int(geo_actual),
        "K_diverse_actual": int(diverse_actual),
        "selected_by_edge_count": int(edge_actual),
        "selected_by_geo_count": int(geo_actual),
        "selected_by_diverse_count": int(diverse_actual),
        "overlap_edge_geo_before_dedup": int(overlap_edge_geo_before_dedup),
        "fallback_reason": fallback_reason,
        "final_kept": int(keep_indices.shape[0]),
    }
    if detailed_timing:
        metadata.update({
            "edge_selection_ms": edge_selection_ms,
            "geo_selection_ms": geo_selection_ms,
            "diverse_selection_ms": diverse_selection_ms,
            "final_merge_ms": final_merge_ms,
        })
    if return_indices:
        metadata.update({
            "selected_edge_indices": [int(i) for i in selected_edge.tolist()],
            "selected_geo_indices": [int(i) for i in selected_geo.tolist()],
            "selected_diverse_indices": [int(i) for i in selected_diverse.tolist()],
        })
    metadata = finalize_selection_debug_info(
        metadata,
        selector_function_name="select_tokens_contact_budget",
        strategy="robot_geo_contact_budget",
        keep_indices=keep_indices,
        num_tokens=n,
        keep_count=keep_k,
        scores=edge_score,
        requested_keep_ratio=float(keep_k) / float(n) if n else None,
        fallback_used=fallback_reason is not None,
        fallback_reason=fallback_reason,
    )
    return keep_indices, metadata


def _ordered_valid_indices(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    adjusted = np.where(valid, np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf, neginf=-np.inf), -np.inf)
    order = np.lexsort((np.arange(adjusted.shape[0]), -adjusted))
    return np.asarray([idx for idx in order if valid[int(idx)]], dtype=np.int64)


def _take_from_order(order: np.ndarray, k: int, excluded: set) -> np.ndarray:
    if k <= 0:
        return np.asarray([], dtype=np.int64)
    if not excluded:
        return np.asarray(order[:k], dtype=np.int64)
    selected = []
    for idx in order:
        idx_int = int(idx)
        if idx_int in excluded:
            continue
        selected.append(idx_int)
        if len(selected) >= k:
            break
    return np.asarray(selected, dtype=np.int64)


def _select_diverse_reserve(
    reserve_score: np.ndarray,
    valid: np.ndarray,
    selected: set,
    reserve_order: np.ndarray,
    quota: int,
    grid_h: int,
    grid_w: int,
    cells_h: int,
    cells_w: int,
) -> np.ndarray:
    if quota <= 0:
        return np.asarray([], dtype=np.int64)
    n = int(reserve_score.shape[0])
    reserve = []
    if grid_h * grid_w == n:
        cells = []
        for cr, cc, cell in _cached_cells(grid_h, grid_w, cells_h, cells_w):
            covered = sum(1 for idx in cell if idx in selected)
            cells.append((covered, cr, cc, cell))
        for _, cr, cc, cell in sorted(cells, key=lambda item: (item[0], item[1], item[2])):
            if len(reserve) >= quota:
                break
            candidates = [idx for idx in cell if idx not in selected and valid[idx]]
            if not candidates:
                continue
            best = min(candidates, key=lambda idx: (-reserve_score[idx], idx))
            selected.add(int(best))
            reserve.append(int(best))
    if len(reserve) < quota:
        for idx in reserve_order:
            idx = int(idx)
            if idx in selected:
                continue
            selected.add(idx)
            reserve.append(idx)
            if len(reserve) >= quota:
                break
    return np.asarray(reserve, dtype=np.int64)


@lru_cache(maxsize=16)
def _cached_cells(grid_h: int, grid_w: int, cells_h: int, cells_w: int) -> Tuple[Tuple[int, int, Tuple[int, ...]], ...]:
    cell_h = max(1, grid_h // max(1, cells_h))
    cell_w = max(1, grid_w // max(1, cells_w))
    cells = []
    for cr in range(cells_h):
        for cc in range(cells_w):
            r0, c0 = cr * cell_h, cc * cell_w
            r1 = grid_h if cr == cells_h - 1 else min(grid_h, r0 + cell_h)
            c1 = grid_w if cc == cells_w - 1 else min(grid_w, c0 + cell_w)
            cell = tuple(r * grid_w + c for r in range(r0, r1) for c in range(c0, c1))
            cells.append((cr, cc, cell))
    return tuple(cells)


def select_uniform_grid_indices(num_tokens: int, keep_count: int) -> np.ndarray:
    if keep_count >= num_tokens:
        return np.arange(num_tokens, dtype=np.int64)
    if keep_count <= 0:
        return np.array([], dtype=np.int64)
    raw = np.linspace(0, num_tokens - 1, keep_count)
    selected = []
    used = set()
    for value in raw:
        idx = int(round(float(value)))
        if idx not in used:
            used.add(idx)
            selected.append(idx)
    for idx in range(num_tokens):
        if len(selected) >= keep_count:
            break
        if idx not in used:
            used.add(idx)
            selected.append(idx)
    return np.sort(np.asarray(selected[:keep_count], dtype=np.int64))


def select_depth_edge_diverse_indices(
    scores: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    keep_total: int = 192,
    depth_quota: int = 160,
    grid_size: int = 16,
    cell_grid: int = 4,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    scores_np = np.nan_to_num(np.asarray(scores, dtype=np.float32), nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    n = scores_np.shape[0]
    valid = np.ones(n, dtype=np.bool_) if valid_mask is None else np.asarray(valid_mask, dtype=np.bool_)
    keep_total = int(max(0, min(keep_total, n)))
    depth_quota = int(max(0, min(depth_quota, keep_total)))
    reserve_quota = keep_total - depth_quota

    adjusted = np.where(valid, scores_np, -np.inf)
    order = np.lexsort((np.arange(n), -adjusted))
    depth_selected = order[:depth_quota].astype(np.int64)
    selected = set(int(i) for i in depth_selected)
    reserve_selected = []
    fallback_selected = 0

    if grid_size * grid_size == n and reserve_quota > 0:
        cell_size = max(1, grid_size // max(1, cell_grid))
        cells = []
        for cr in range(cell_grid):
            for cc in range(cell_grid):
                r0, c0 = cr * cell_size, cc * cell_size
                r1 = grid_size if cr == cell_grid - 1 else min(grid_size, r0 + cell_size)
                c1 = grid_size if cc == cell_grid - 1 else min(grid_size, c0 + cell_size)
                cell = [r * grid_size + c for r in range(r0, r1) for c in range(c0, c1)]
                covered = sum(1 for idx in cell if idx in selected)
                cells.append((covered, cr, cc, cell))
        for _, cr, cc, cell in sorted(cells, key=lambda x: (x[0], x[1], x[2])):
            if len(reserve_selected) >= reserve_quota:
                break
            candidates = [idx for idx in cell if idx not in selected and valid[idx]]
            if not candidates:
                continue
            best = min(candidates, key=lambda idx: (-adjusted[idx], idx))
            selected.add(int(best))
            reserve_selected.append(int(best))

    if len(reserve_selected) < reserve_quota:
        for idx in order:
            idx = int(idx)
            if idx in selected:
                continue
            selected.add(idx)
            reserve_selected.append(idx)
            fallback_selected += 1
            if len(reserve_selected) >= reserve_quota:
                break

    if len(selected) < keep_total:
        for idx in range(n):
            if idx not in selected:
                selected.add(idx)
            if len(selected) >= keep_total:
                break

    keep_indices = np.sort(np.asarray(list(selected), dtype=np.int64))[:keep_total]
    metadata = {
        "strategy": "depth_edge_fast_diverse",
        "depth_quota": int(depth_quota),
        "reserve_quota": int(reserve_quota),
        "reserve_selected": int(min(len(reserve_selected), reserve_quota)),
        "fallback_selected": int(fallback_selected),
        "grid_size": int(grid_size),
        "cell_grid": int(cell_grid),
        "final_kept": int(len(keep_indices)),
    }
    metadata = finalize_selection_debug_info(
        metadata,
        selector_function_name="select_depth_edge_diverse_indices",
        strategy="depth_edge_fast_diverse",
        keep_indices=keep_indices,
        num_tokens=n,
        keep_count=keep_total,
        scores=scores,
        requested_keep_ratio=float(keep_total) / float(n) if n else None,
    )
    return keep_indices, metadata
