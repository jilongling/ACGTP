"""Central registry for visual-token pruning strategies.

The project keeps many historical strategies for ablation and reproducibility.
Only a small subset should be treated as the current experiment surface.
"""

from __future__ import annotations

from typing import Set

from .legacy.strategies import (
    EDGE_RESERVE_LEGACY_STRATEGIES,
    LEGACY_ALWAYS_ENABLED_STRATEGIES,
    LEGACY_BRANCH_MIXTURE_SCORE_STRATEGIES,
    LEGACY_ROBOT_GEO_STRATEGIES,
    LEGACY_ROBOT_STATE_REQUIRED_STRATEGIES,
    LEGACY_STRATEGIES,
)
from .methods.registry import SELF_HANDLED_SELECTOR_STRATEGIES


BASELINE_STRATEGIES: Set[str] = {
    "none",
    "random",
    "uniform_grid",
    "depth_edge_fast",
    "depth_edge_fast_diverse",
}

CURRENT_STRATEGIES: Set[str] = {
    "robot_geo_acgtp_v2",
}

ACGTP_STRATEGIES: Set[str] = {
    "robot_geo_acgtp_v1",
    "robot_geo_acgtp_v2",
}

SUPPORTED_STRATEGIES: Set[str] = BASELINE_STRATEGIES | CURRENT_STRATEGIES | LEGACY_STRATEGIES

CORE_EXPERIMENT_STRATEGIES: Set[str] = {
    "none",
    "depth_edge_fast",
    "robot_geo_acgtp_v2",
}

ACTIVE_EXPERIMENT_STRATEGIES: Set[str] = CORE_EXPERIMENT_STRATEGIES | {
    "depth_edge_fast_diverse",
}

# Public comparison labels for the converged ACGTP line. These are report-level
# surfaces, not new pruning_strategy names; legacy/proxy/hybrid variants remain
# available only when explicitly selected by audit/probe scripts.
MAIN_EXPERIMENT_METHOD_LABELS: Set[str] = {
    "baseline_none",
    "baseline_none_keep100",
    "none",
    "projector_acgtp_legacy",
    "projector_acgtp_legacy_050",
    "internal_geometry_only",
    "internal_acgtp_geometry_only",
    "internal_acgtp_geometry_only_050",
    "internal_geo_guarded",
    "internal_acgtp_geo_guarded",
    "internal_acgtp_geo_guarded_050",
    "internal_dynamic",
    "internal_acgtp_dynamic",
    "internal_acgtp_dynamic_050",
}

AUDIT_ONLY_STRATEGIES: Set[str] = LEGACY_STRATEGIES | {"robot_geo_acgtp_v1"}

GEOMETRY_STRATEGIES: Set[str] = (
    {"depth_edge_fast", "depth_edge_fast_diverse"}
    | CURRENT_STRATEGIES
    | LEGACY_ROBOT_GEO_STRATEGIES
)

ROBOT_GEO_SCORE_STRATEGIES: Set[str] = CURRENT_STRATEGIES | LEGACY_ROBOT_GEO_STRATEGIES

ALWAYS_ENABLED_STRATEGIES: Set[str] = CURRENT_STRATEGIES | LEGACY_ALWAYS_ENABLED_STRATEGIES

DYNAMIC_MID_KEEP_STRATEGIES: Set[str] = {
    "robot_geo_dynamic",
    "robot_geo_dynamic_v0",
    "robot_geo_temporal_v0",
    "robot_geo_hybrid_dynamic_v0",
}

TEMPORAL_DIAGNOSTIC_STRATEGIES: Set[str] = {
    "robot_geo_hybrid_temporal_v1",
    "robot_geo_hybrid_temporal_edge_reserve_v1",
}

HYBRID_POST_DIAGNOSTIC_STRATEGIES: Set[str] = {
    "robot_geo_hybrid_temporal_v1",
    "robot_geo_hybrid_temporal_edge_reserve_v1",
    "robot_geo_hybrid_v1",
    "depth_edge_fast",
}

EARLY_GEOMETRY_FALLBACK_STRATEGIES: Set[str] = {
    "robot_geo_near",
    "robot_geo_corridor",
    "robot_geo_dynamic",
}

ROBOT_STATE_REQUIRED_LEGACY_STRATEGIES: Set[str] = (
    CURRENT_STRATEGIES | LEGACY_ROBOT_STATE_REQUIRED_STRATEGIES
)

BRANCH_MIXTURE_SCORE_STRATEGIES: Set[str] = CURRENT_STRATEGIES | LEGACY_BRANCH_MIXTURE_SCORE_STRATEGIES

TOKEN_SELECTION_DEBUG_STRATEGIES: Set[str] = {
    "robot_geo_hybrid_temporal_v1",
    "robot_geo_hybrid_temporal_edge_reserve_v1",
    "robot_geo_hybrid_v1",
    "depth_edge_fast",
    "robot_geo_branch_budget_v0",
    "robot_geo_acgtp_v1",
}


def normalize_strategy(strategy: str) -> str:
    s = str(strategy or "none").strip().lower()
    if s == "depth_edge":
        s = "depth_edge_fast"
    if s not in SUPPORTED_STRATEGIES:
        raise ValueError(f"Unknown pruning_strategy={strategy}. Supported: {sorted(SUPPORTED_STRATEGIES)}")
    return s


def is_geometry_strategy(strategy: str) -> bool:
    return normalize_strategy(strategy) in GEOMETRY_STRATEGIES


def is_robot_geo_score_strategy(strategy: str) -> bool:
    return normalize_strategy(strategy) in ROBOT_GEO_SCORE_STRATEGIES


def is_legacy_strategy(strategy: str) -> bool:
    return normalize_strategy(strategy) in LEGACY_STRATEGIES


def is_audit_only_strategy(strategy: str) -> bool:
    s = normalize_strategy(strategy)
    return s in AUDIT_ONLY_STRATEGIES and s not in CURRENT_STRATEGIES and s not in BASELINE_STRATEGIES


def legacy_strategy_allowed(
    strategy: str,
    *,
    allow_legacy_strategy: bool = False,
    runtime_mode: str = "fast",
    full_diagnostics_enabled: bool = False,
) -> bool:
    """Return whether a legacy strategy can run under the current runtime intent."""

    s = normalize_strategy(strategy)
    if s not in LEGACY_STRATEGIES:
        return True
    mode = str(runtime_mode or "fast").strip().lower()
    return bool(allow_legacy_strategy or full_diagnostics_enabled or mode == "audit")


def validate_strategy_runtime(
    strategy: str,
    *,
    allow_legacy_strategy: bool = False,
    runtime_mode: str = "fast",
    full_diagnostics_enabled: bool = False,
) -> str:
    """Normalize a strategy and reject legacy runs that were not explicitly opted in."""

    s = normalize_strategy(strategy)
    if not legacy_strategy_allowed(
        s,
        allow_legacy_strategy=allow_legacy_strategy,
        runtime_mode=runtime_mode,
        full_diagnostics_enabled=full_diagnostics_enabled,
    ):
        raise ValueError(
            f"Legacy pruning_strategy={s!r} requires --allow_legacy_strategy true "
            "or acgtp_runtime_mode=audit. Use robot_geo_acgtp_v2 for the current path."
        )
    return s


def is_main_experiment_method_label(label: str) -> bool:
    return str(label or "").strip() in MAIN_EXPERIMENT_METHOD_LABELS


def strategy_family(strategy: str) -> str:
    s = normalize_strategy(strategy)
    if s in CURRENT_STRATEGIES:
        return "current"
    if s in BASELINE_STRATEGIES:
        return "baseline"
    if s in LEGACY_STRATEGIES:
        return "legacy"
    return "unknown"
