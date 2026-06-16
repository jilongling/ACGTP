"""Shared selector helpers and diagnostics metadata normalization."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .registry import selection_stage_name


def _score_stats(scores: Optional[np.ndarray]) -> Dict[str, Optional[float]]:
    if scores is None:
        return {"score_min": None, "score_max": None, "score_mean": None, "score_std": None}
    arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"score_min": None, "score_max": None, "score_mean": None, "score_std": None}
    return {
        "score_min": float(np.min(finite)),
        "score_max": float(np.max(finite)),
        "score_mean": float(np.mean(finite)),
        "score_std": float(np.std(finite)),
    }


def _infer_stage_name(strategy: str, metadata: Dict[str, Any]) -> str:
    if metadata.get("selection_stage_name"):
        return str(metadata["selection_stage_name"])
    return selection_stage_name(strategy, fallback=bool(metadata.get("fallback")))


def finalize_selection_debug_info(
    metadata: Optional[Dict[str, Any]],
    *,
    selector_function_name: str,
    strategy: str,
    keep_indices: Optional[np.ndarray],
    num_tokens: int,
    keep_count: int,
    scores: Optional[np.ndarray] = None,
    requested_keep_ratio: Optional[float] = None,
    fallback_used: Optional[bool] = None,
    fallback_reason: Optional[str] = None,
    selection_error: Optional[str] = None,
    selection_warning: Optional[str] = None,
) -> Dict[str, Any]:
    """Attach the common diagnostics schema to selector metadata.

    This helper is logging-only. It must not alter keep_indices or selector
    ordering; it only describes the selection path that already ran.
    """
    info: Dict[str, Any] = dict(metadata or {})
    actual_strategy = str(info.get("strategy") or strategy)
    idx = None if keep_indices is None else np.asarray(keep_indices, dtype=np.int64).reshape(-1)
    kept = int(idx.size) if idx is not None else None
    unique = bool(np.unique(idx).size == idx.size) if idx is not None else None
    sorted_flag = bool(np.all(idx[:-1] <= idx[1:])) if idx is not None and idx.size > 1 else (True if idx is not None else None)
    out_of_bounds = bool(np.any((idx < 0) | (idx >= int(num_tokens)))) if idx is not None else False

    info.setdefault("selector_name", actual_strategy)
    info.setdefault("selector_function_name", selector_function_name)
    info.setdefault("selection_strategy_name", actual_strategy)
    info.setdefault("selection_stage_name", _infer_stage_name(actual_strategy, info))
    info.setdefault("num_visual_tokens_original", int(num_tokens))
    info.setdefault("num_visual_tokens_original_total", int(num_tokens))
    info.setdefault("num_visual_tokens_kept", kept)
    info.setdefault("num_visual_tokens_kept_total", kept)
    info.setdefault("num_visual_tokens_dropped", (int(num_tokens) - kept) if kept is not None else None)
    info.setdefault("keep_indices_count", kept)
    info.setdefault("keep_indices_unique", unique)
    info.setdefault("keep_indices_sorted", sorted_flag)
    info.setdefault("keep_indices_out_of_bounds", out_of_bounds)
    info.setdefault("keep_ratio_requested", requested_keep_ratio)
    info.setdefault("keep_ratio_actual", (float(kept) / float(num_tokens)) if kept is not None and num_tokens else None)
    info.setdefault("retention_actual", info.get("keep_ratio_actual"))
    info.setdefault("actual_retention_ratio", info.get("keep_ratio_actual"))
    info.setdefault("total_keep_budget", int(keep_count) if keep_count is not None else None)
    info.setdefault("fallback_used", bool(fallback_used) if fallback_used is not None else bool(info.get("fallback")))
    info.setdefault("fallback_reason", fallback_reason if fallback_reason is not None else info.get("fallback_reason"))
    # P7: selected_by_fallback_count is set by select_hybrid_budget_v2 in metadata.
    # Assign DIRECTLY (not setdefault) so that the correct value is preserved.
    # setdefault would overwrite the correct value with None.
    if "selected_by_fallback_count" not in info:
        info["selected_by_fallback_count"] = (metadata.get("selected_by_fallback_count") if metadata else None)
    info.setdefault("selection_error", selection_error)

    warnings = []
    if selection_warning:
        warnings.append(str(selection_warning))
    if out_of_bounds:
        warnings.append("keep_indices_out_of_bounds")
    if idx is not None and kept != int(keep_count) and not bool(info.get("fallback_used")):
        warnings.append(f"kept_count_mismatch:kept={kept},target={int(keep_count)}")
    info.setdefault("selection_warning", ";".join(warnings) if warnings else None)

    stats = _score_stats(scores)
    for key, value in stats.items():
        info.setdefault(key, value)

    # Normalized phase accounting. Detailed selector-specific fields remain
    # available; these generic fields are only for cross-selector checks.
    if "selected_by_phase1" not in info:
        if actual_strategy.startswith("robot_geo_hybrid_temporal_edge_reserve"):
            info["selected_by_phase1"] = int(info.get("selected_by_edge_reserved_count") or 0) + int(info.get("selected_by_phase1_hybrid_count") or 0)
            info["selected_by_phase2"] = info.get("selected_by_phase2_diversity_count")
            info["selected_by_phase3"] = info.get("selected_by_phase3_fallback_count")
            info["selected_by_fill"] = None
            info["selected_by_fallback"] = 0 if not info.get("fallback_used") else kept
            info["selected_unattributed"] = info.get("selected_by_unattributed_count")
        elif actual_strategy in ("robot_geo_hybrid_v2", "robot_geo_contact_budget") or any(k in info for k in ("selected_by_depth_edge_count", "selected_by_contact_count", "selected_by_geo_count")):
            fill_count = info.get("selected_by_fill_count")
            known = 0
            for key in (
                "selected_by_depth_edge_count",
                "selected_by_contact_count",
                "selected_by_distance_contact_count",
                "selected_by_motion_count",
                "selected_by_edge_count",
                "selected_by_geo_count",
                "selected_by_diverse_count",
                "selected_by_uniform_count",
            ):
                known += int(info.get(key) or 0)
            info["selected_by_phase1"] = known if known > 0 else None
            info["selected_by_phase2"] = None
            info["selected_by_phase3"] = None
            info["selected_by_fill"] = fill_count
            info["selected_by_fallback"] = 0 if not info.get("fallback_used") else kept
            accounted = known + int(fill_count or 0)
            info["selected_unattributed"] = max(0, int(kept or 0) - accounted) if kept is not None else None
        elif actual_strategy in ("hybrid_budget_v2", "robot_geo_branch_budget_v0"):
            # P7/P11: Budget-based hybrid. Phase 1=depth_edge, Phase 2=hybrid, Phase 3=null
            info["selected_by_phase1"] = info.get("selected_by_depth_edge_count")
            info["selected_by_phase2"] = info.get("selected_by_robot_geo_count")
            info["selected_by_phase3"] = None
            info["selected_by_fill"] = info.get("selected_by_fill_count")
            # Use actual fallback count; avoid double-counting
            fb_count = info.get("selected_by_fallback_count")
            info["selected_by_fallback"] = fb_count if fb_count is not None else (0 if not info.get("fallback_used") else kept)
            phase_sum = (
                int(info.get("selected_by_phase1") or 0)
                + int(info.get("selected_by_phase2") or 0)
                + int(info.get("selected_by_fill") or 0)
                + int(info.get("selected_by_fallback") or 0)
            )
            info["selected_unattributed"] = max(0, int(kept or 0) - phase_sum) if kept is not None else None
        elif bool(info.get("fallback_used")):
            info["selected_by_phase1"] = None
            info["selected_by_phase2"] = None
            info["selected_by_phase3"] = None
            info["selected_by_fill"] = None
            info["selected_by_fallback"] = kept
            info["selected_unattributed"] = 0 if kept is not None else None
        else:
            info["selected_by_phase1"] = kept
            info["selected_by_phase2"] = None
            info["selected_by_phase3"] = None
            info["selected_by_fill"] = None
            info["selected_by_fallback"] = 0 if kept is not None else None
            info["selected_unattributed"] = 0 if kept is not None else None

    phase_parts = [
        info.get("selected_by_phase1"),
        info.get("selected_by_phase2"),
        info.get("selected_by_phase3"),
        info.get("selected_by_fill"),
        info.get("selected_by_fallback"),
        info.get("selected_unattributed"),
    ]
    if kept is not None and any(v is not None for v in phase_parts):
        phase_sum = int(sum(int(v) for v in phase_parts if v is not None))
        info["phase_accounting_sum"] = phase_sum
        info["phase_accounting_valid"] = bool(phase_sum == kept)
        info["phase_accounting_error"] = None if phase_sum == kept else f"phase_sum={phase_sum} != kept={kept}"
    else:
        info.setdefault("phase_accounting_sum", None)
        info.setdefault("phase_accounting_valid", None)
        info.setdefault("phase_accounting_error", None)

    info.setdefault("depth_edge_budget", info.get("K_edge_target", info.get("depth_edge_quota_count", info.get("depth_quota"))))
    info.setdefault("robot_geo_budget", info.get("K_geo_target", info.get("robot_geo_quota_count")))
    info.setdefault("fill_budget", info.get("K_grid_fill", info.get("K_fill_quota", info.get("fill_count"))))
    info.setdefault("safety_budget", info.get("K_diverse_target", info.get("K_uniform_quota", info.get("uniform_quota_count"))))

    if "reserved_edge_topk_dropped_ratio" in info:
        info.setdefault("reserved_edge_dropped_ratio", info.get("reserved_edge_topk_dropped_ratio"))
    if "non_reserved_edge_topk_count" in info:
        info.setdefault("non_reserved_topk_count", info.get("non_reserved_edge_topk_count"))
        info.setdefault("non_reserved_kept_count", info.get("non_reserved_edge_kept_count"))
        info.setdefault("non_reserved_dropped_count", info.get("non_reserved_edge_dropped_count"))
        info.setdefault("non_reserved_dropped_ratio", info.get("non_reserved_edge_topk_dropped_ratio"))
    return info


def select_score_topk(
    scores: np.ndarray,
    keep_count: int,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    scores_np = np.nan_to_num(np.asarray(scores, dtype=np.float32), nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    if valid_mask is None:
        valid = np.ones(scores_np.shape[0], dtype=np.bool_)
    else:
        valid = np.asarray(valid_mask, dtype=np.bool_)
    adjusted = np.where(valid, scores_np, -np.inf)
    order = np.lexsort((np.arange(scores_np.shape[0]), -adjusted))
    selected = list(order[:keep_count])
    if len(selected) < keep_count:
        selected.extend([i for i in range(scores_np.shape[0]) if i not in selected][: keep_count - len(selected)])
    return np.sort(np.asarray(selected[:keep_count], dtype=np.int64))


def _normalize_for_selection(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(scores, dtype=np.float32)
    if not np.any(valid):
        return out
    vals = np.nan_to_num(scores[valid], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if hi - lo > 1e-8:
        out[valid] = (vals - lo) / (hi - lo)
    else:
        out[valid] = 0.0
    return out


def validate_keep_indices(keep_indices: np.ndarray, expected_count: int) -> Dict[str, Any]:
    idx = np.asarray(keep_indices, dtype=np.int64)
    unique_count = int(np.unique(idx).shape[0])
    return {
        "keep_indices_sorted": bool(np.all(idx[:-1] <= idx[1:])) if idx.shape[0] > 1 else True,
        "duplicate_indices_count": int(idx.shape[0] - unique_count),
        "final_kept": int(idx.shape[0]),
        "expected_kept": int(expected_count),
    }


_LEGACY_SELECTOR_EXPORTS = {
    "select_hybrid_quota_union",
    "select_hybrid_quota_v2",
    "select_hybrid_v1",
    "select_hybrid_v1_edge_reserve",
    "select_hybrid_budget_v2",
    "select_branch_budget_v0",
    "select_acgtp_v1",
}


def __getattr__(name: str) -> Any:
    """Lazy compatibility access for selector functions formerly imported here."""

    if name in _LEGACY_SELECTOR_EXPORTS:
        from .. import legacy as legacy_selectors

        return getattr(legacy_selectors, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
