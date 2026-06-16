"""Unified depth conversion utility for robosuite-based environments.

This module provides a single, authoritative function for converting raw
robosuite z-buffer depth to metric depth (meters). It also handles
already-metric depth and provides metadata about the conversion.

Conversion rules:
  1. If depth values are in [0,1] AND look like a z-buffer (values near 1.0
     with tiny variance), treat as robosuite raw z-buffer.
  2. If sim is available, call CU.get_real_depth_map() for correct conversion.
  3. If sim is unavailable, return raw depth with explicit "raw_no_sim_fallback"
     flag and depth_is_metric=False. Never silently pretend it's meters.
  4. If depth is clearly already metric (max > 1.5, reasonable variance),
     return as-is with "none_already_metric" flag.
  5. Guard against double conversion: if already converted (e.g. shape name
     contains "metric"), return as-is.

Usage:
    result = convert_depth_to_metric(depth_raw, sim=env_sim, source_key="agentview_depth")
    depth_metric = result["depth"]
    metadata     = result["metadata"]
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Minimum number of unique depth values to consider the depth non-constant.
_MIN_UNIQUE_FOR_VARIANCE_CHECK = 10


@dataclass
class DepthConversionResult:
    depth: Optional[np.ndarray]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, (np.bool_,)):
                d[k] = bool(v)
            elif isinstance(v, (np.integer,)):
                d[k] = int(v)
            elif isinstance(v, (np.floating,)):
                d[k] = float(v)
            elif isinstance(v, np.ndarray):
                d[k] = v.tolist()
        return d


def _array_stats(arr: np.ndarray) -> Dict[str, float]:
    flat = np.asarray(arr, dtype=np.float64).flatten()
    valid = flat[np.isfinite(flat)]
    if valid.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "median": 0.0}
    return {
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "median": float(np.median(valid)),
    }


def _looks_like_robosuite_zbuffer(arr: np.ndarray) -> bool:
    """Return True if depth array looks like a raw robosuite z-buffer.

    Robosuite raw z-buffer: values in [0,1], concentrated near 1.0,
    tiny std (typically < 0.05 for overhead agentview camera).
    """
    flat = np.asarray(arr, dtype=np.float64).flatten()
    valid = flat[np.isfinite(flat)]
    if valid.size == 0:
        return False
    mn, mx, mean, std = float(np.min(valid)), float(np.max(valid)), float(np.mean(valid)), float(np.std(valid))
    # Must be in [0,1] range
    if mn < -0.01 or mx > 1.05:
        return False
    # Must be concentrated near 1.0 (z-buffer: far=1.0, near=0.0)
    if mean < 0.90:
        return False
    # Must have tiny variance
    if std > 0.08:
        return False
    return True


def _looks_like_metric_depth(arr: np.ndarray) -> bool:
    """Return True if depth array looks like metric depth in meters.

    Heuristic: max > 1.5 AND reasonable variance.
    Also: max > 10.0 (millimeters) could be metric.
    """
    flat = np.asarray(arr, dtype=np.float64).flatten()
    valid = flat[np.isfinite(flat)]
    if valid.size < _MIN_UNIQUE_FOR_VARIANCE_CHECK:
        return False
    mn, mx, std = float(np.min(valid)), float(np.max(valid)), float(np.std(valid))
    # Clearly metric: max > 1.5m (typical tabletop scene: 0.5m to 3m)
    if mx >= 1.5:
        return True
    # Millimeters: 0.15m to 10m in mm would be 150 to 10000
    if mx >= 150.0 and mx <= 10000.0:
        return True
    return False


def _looks_like_already_converted(arr: np.ndarray) -> bool:
    """Return True if depth array looks like it has already been converted.

    After get_real_depth_map conversion, values should be:
    - In [0.5, 5.0] for typical tabletop scenes
    - Not concentrated near 1.0
    - Have reasonable variance (std > 0.1)
    """
    flat = np.asarray(arr, dtype=np.float64).flatten()
    valid = flat[np.isfinite(flat)]
    if valid.size < _MIN_UNIQUE_FOR_VARIANCE_CHECK:
        return False
    mn, mx, mean, std = float(np.min(valid)), float(np.max(valid)), float(np.mean(valid)), float(np.std(valid))
    if mn < -0.01 or mx > 10.0:
        return False  # out of reasonable range
    if mean >= 0.95 and std < 0.05:
        return False  # still looks like raw z-buffer
    if std > 0.05:
        return True
    return False


def convert_depth_to_metric(
    depth_raw: Any,
    sim: Any = None,
    source_key: Optional[str] = None,
    image_transform: Optional[str] = None,
    *,
    _debug: bool = False,
) -> DepthConversionResult:
    """Convert raw depth to metric depth (meters).

    Args:
        depth_raw: Raw depth array [H, W] or [H, W, 1].
        sim: Robosuite MujocoSimulator instance. If None, falls back to raw.
        source_key: Optional key name (e.g. "agentview_depth") used for logging.
        image_transform: Optional transform string (e.g. "rot180") applied before conversion.
        _debug: Enable debug logging.

    Returns:
        DepthConversionResult with:
          - depth: np.ndarray [H, W] float32 in meters, or None
          - metadata: dict with conversion details
    """
    key_name = source_key or "unknown"

    # Step 0: Normalize to numpy array
    if depth_raw is None:
        return DepthConversionResult(
            depth=None,
            metadata={
                "source_key": key_name,
                "conversion": "none_no_data",
                "depth_is_metric": False,
                "depth_unit": "unknown",
                "depth_raw_stats": None,
                "depth_metric_stats": None,
                "sim_available": sim is not None,
                "warning": "depth_raw is None",
            },
        )

    try:
        arr = np.asarray(depth_raw, dtype=np.float32)
    except Exception as exc:
        logger.warning("[DepthConversion] Failed to convert depth to array: %s", exc)
        return DepthConversionResult(
            depth=None,
            metadata={
                "source_key": key_name,
                "conversion": "error_parse_failed",
                "depth_is_metric": False,
                "depth_unit": "unknown",
                "depth_raw_stats": None,
                "depth_metric_stats": None,
                "sim_available": sim is not None,
                "warning": f"parse error: {exc}",
            },
        )

    # Squeeze [H, W, 1] -> [H, W]
    original_shape = arr.shape
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    arr = np.asarray(arr, dtype=np.float32)

    # Raw stats (on original raw values)
    raw_stats = _array_stats(arr)

    # Step 1: Apply image transform if specified
    if image_transform == "rot180":
        arr = arr[::-1, ::-1]
        if _debug:
            logger.debug("[DepthConversion] Applied rot180 transform")

    # Step 2: Determine conversion type
    looks_raw = _looks_like_robosuite_zbuffer(arr)
    looks_metric = _looks_like_metric_depth(arr)
    looks_converted = _looks_like_already_converted(arr)

    # Step 3: Check for already-metric key patterns
    is_metric_key = (
        (source_key is not None)
        and ("metric" in source_key.lower() or "real" in source_key.lower())
    )
    if is_metric_key:
        looks_metric = True
        looks_raw = False

    # Step 4: Double-conversion guard
    if looks_converted and not looks_raw:
        if _debug:
            logger.debug(
                "[DepthConversion] depth already looks converted "
                "(mean=%.4f, std=%.4f). Skipping conversion.",
                raw_stats["mean"], raw_stats["std"],
            )
        return DepthConversionResult(
            depth=arr,
            metadata={
                "source_key": key_name,
                "original_shape": list(original_shape),
                "conversion": "none_already_converted",
                "depth_is_metric": True,
                "depth_unit": "meters",
                "depth_raw_stats": raw_stats,
                "depth_metric_stats": raw_stats,
                "sim_available": sim is not None,
                "looks_like_robosuite_zbuffer": looks_raw,
                "looks_like_metric_depth": looks_metric,
                "looks_like_already_converted": looks_converted,
            },
        )

    # Step 5: Decide conversion path
    conversion = "none"
    depth_out: Optional[np.ndarray] = arr.copy()
    is_metric = False
    unit = "unknown"

    if looks_raw and not looks_metric:
        # ── RAW ROBOSUITE Z-BUFFER ──────────────────────────────────────────
        if sim is not None:
            try:
                from robosuite.utils import camera_utils as CU
                depth_out = CU.get_real_depth_map(sim, arr.copy()).astype(np.float32)
                conversion = "robosuite_get_real_depth_map"
                is_metric = True
                unit = "meters"
                if _debug:
                    logger.debug(
                        "[DepthConversion] Converted %s via get_real_depth_map: "
                        "raw [%.4f, %.4f] -> metric [%.4f, %.4f]",
                        key_name,
                        raw_stats["min"], raw_stats["max"],
                        float(np.min(depth_out)), float(np.max(depth_out)),
                    )
            except Exception as exc:
                logger.warning(
                    "[DepthConversion] get_real_depth_map failed for %s: %s. "
                    "Using raw depth as-is.",
                    key_name, exc,
                )
                conversion = "robosuite_get_real_depth_map_error"
                is_metric = False
                unit = "unknown"
        else:
            logger.warning(
                "[DepthConversion] %s looks like robosuite raw z-buffer "
                "(mean=%.4f, std=%.4f) but sim is not available. "
                "Cannot convert to metric. Returning raw depth with "
                "depth_is_metric=False.",
                key_name, raw_stats["mean"], raw_stats["std"],
            )
            conversion = "raw_no_sim_fallback"
            is_metric = False
            unit = "unknown"

    elif looks_metric and not looks_raw:
        # ── ALREADY METRIC DEPTH ────────────────────────────────────────────
        conversion = "none_already_metric"
        is_metric = True
        unit = "meters"
        if _debug:
            logger.debug(
                "[DepthConversion] %s already looks like metric depth "
                "(max=%.4f). Skipping conversion.",
                key_name, raw_stats["max"],
            )

    elif looks_raw and looks_metric:
        # ── AMBIGUOUS: treat as metric if sim is unavailable, raw if available ─
        if sim is not None:
            try:
                from robosuite.utils import camera_utils as CU
                depth_out = CU.get_real_depth_map(sim, arr.copy()).astype(np.float32)
                conversion = "robosuite_get_real_depth_map"
                is_metric = True
                unit = "meters"
            except Exception as exc:
                logger.warning(
                    "[DepthConversion] get_real_depth_map failed for ambiguous %s: %s",
                    key_name, exc,
                )
                conversion = "ambiguous_get_real_depth_map_error"
                is_metric = False
                unit = "unknown"
        else:
            conversion = "ambiguous_no_sim"
            is_metric = False
            unit = "unknown"

    else:
        # ── UNKNOWN TYPE ────────────────────────────────────────────────────
        conversion = "none_unknown_type"
        is_metric = False
        unit = "unknown"
        if _debug:
            logger.debug(
                "[DepthConversion] %s has unknown depth type "
                "(min=%.4f, max=%.4f, std=%.4f). Returning as-is.",
                key_name, raw_stats["min"], raw_stats["max"], raw_stats["std"],
            )

    # Compute metric stats
    metric_stats = _array_stats(depth_out) if depth_out is not None else None

    metadata = {
        "source_key": key_name,
        "original_shape": list(original_shape),
        "conversion": conversion,
        "depth_is_metric": is_metric,
        "depth_unit": unit,
        "depth_raw_stats": raw_stats,
        "depth_metric_stats": metric_stats,
        "sim_available": sim is not None,
        "looks_like_robosuite_zbuffer": looks_raw,
        "looks_like_metric_depth": looks_metric,
        "looks_like_already_converted": looks_converted,
        "is_metric_key_pattern": is_metric_key,
    }

    return DepthConversionResult(depth=depth_out, metadata=metadata)
