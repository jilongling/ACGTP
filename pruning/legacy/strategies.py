"""Legacy strategy sets for audit and ablation paths."""

from __future__ import annotations

from typing import Set


EDGE_RESERVE_LEGACY_STRATEGIES: Set[str] = {
    "robot_geo_hybrid_temporal_edge_reserve_v1",
    "robot_geo_hybrid_temporal_edge_reserve025_v1",
    "robot_geo_hybrid_temporal_edge_reserve040_v1",
    "robot_geo_hybrid_temporal_edge_reserve055_v1",
    "robot_geo_hybrid_temporal_edge_reserve070_v1",
}

LEGACY_ROBOT_GEO_STRATEGIES: Set[str] = {
    "robot_geo_near",
    "robot_geo_corridor",
    "robot_geo_contact_budget",
    "robot_geo_rule_v0",
    "robot_geo_dynamic_v0",
    "robot_geo_temporal_v0",
    "robot_geo_hybrid_v0",
    "robot_geo_hybrid_dynamic_v0",
    "robot_geo_hybrid_v1",
    "robot_geo_hybrid_temporal_v1",
    "robot_geo_dynamic",
    "hybrid_budget_v2",
    "robot_geo_branch_budget_v0",
    "robot_geo_acgtp_v1",
} | EDGE_RESERVE_LEGACY_STRATEGIES

LEGACY_STRATEGIES: Set[str] = set(LEGACY_ROBOT_GEO_STRATEGIES)

LEGACY_ALWAYS_ENABLED_STRATEGIES: Set[str] = {
    "robot_geo_dynamic",
    "robot_geo_dynamic_v0",
    "robot_geo_temporal_v0",
    "robot_geo_hybrid_v0",
    "robot_geo_hybrid_dynamic_v0",
    "robot_geo_hybrid_v1",
    "robot_geo_hybrid_temporal_v1",
    "hybrid_budget_v2",
    "robot_geo_branch_budget_v0",
    "robot_geo_acgtp_v1",
} | EDGE_RESERVE_LEGACY_STRATEGIES

LEGACY_SELF_HANDLED_SELECTOR_STRATEGIES: Set[str] = {
    "robot_geo_hybrid_v0",
    "robot_geo_hybrid_dynamic_v0",
    "robot_geo_hybrid_v1",
    "robot_geo_hybrid_temporal_v1",
    "hybrid_budget_v2",
    "robot_geo_branch_budget_v0",
    "robot_geo_acgtp_v1",
} | EDGE_RESERVE_LEGACY_STRATEGIES

LEGACY_ROBOT_STATE_REQUIRED_STRATEGIES: Set[str] = {
    "robot_geo_near",
    "robot_geo_corridor",
    "robot_geo_rule_v0",
    "robot_geo_dynamic_v0",
    "robot_geo_temporal_v0",
    "robot_geo_dynamic",
    "robot_geo_hybrid_v1",
    "robot_geo_hybrid_temporal_v1",
    "robot_geo_acgtp_v1",
} | EDGE_RESERVE_LEGACY_STRATEGIES

LEGACY_BRANCH_MIXTURE_SCORE_STRATEGIES: Set[str] = {
    "robot_geo_hybrid_v1",
    "robot_geo_hybrid_temporal_v1",
    "robot_geo_branch_budget_v0",
    "robot_geo_acgtp_v1",
} | EDGE_RESERVE_LEGACY_STRATEGIES
