"""Runtime config normalization for pruning entrypoints."""

from __future__ import annotations

from typing import Any, Dict

from .strategy_registry import (
    ALWAYS_ENABLED_STRATEGIES,
    GEOMETRY_STRATEGIES,
    is_legacy_strategy,
    strategy_family,
    validate_strategy_runtime,
)


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def normalize_pruning_runtime_config(cfg: Dict[str, Any]) -> str:
    """Normalize pruning aliases and attach strategy metadata to ``cfg``."""

    strategy = (
        cfg.get("pruning_strategy")
        or cfg.get("pruning_mode")
        or cfg.get("pruning_method")
        or "none"
    )
    strategy = validate_strategy_runtime(
        strategy,
        allow_legacy_strategy=as_bool(cfg.get("allow_legacy_strategy", False)),
        runtime_mode=str(cfg.get("acgtp_runtime_mode", "fast") or "fast"),
        full_diagnostics_enabled=as_bool(cfg.get("acgtp_full_diagnostics_enabled", False)),
    )
    cfg["pruning_strategy"] = strategy
    cfg["pruning_mode"] = strategy
    cfg["pruning_method"] = strategy
    cfg["pruning_strategy_family"] = strategy_family(strategy)
    cfg["pruning_strategy_is_legacy"] = is_legacy_strategy(strategy)
    cfg["pruning_strategy_is_core"] = strategy in {"none", "depth_edge_fast", "robot_geo_acgtp_v2"}
    return strategy


def geometry_enabled_for(cfg: Dict[str, Any], strategy: str) -> bool:
    return bool(as_bool(cfg.get("geometry_enabled", False)) or strategy in GEOMETRY_STRATEGIES)


def pruning_enabled_for(cfg: Dict[str, Any], strategy: str) -> bool:
    keep_ratio = float(cfg.get("keep_ratio", 1.0) or 1.0)
    return bool(
        as_bool(cfg.get("pruning_enabled", False))
        or (strategy != "none" and (keep_ratio < 1.0 or strategy in ALWAYS_ENABLED_STRATEGIES))
    )
