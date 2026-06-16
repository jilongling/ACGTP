"""Selector registry constants and stage names."""

from __future__ import annotations

from typing import Dict, Set

from ..legacy.strategies import LEGACY_SELF_HANDLED_SELECTOR_STRATEGIES


NO_SCORE_STRATEGIES: Set[str] = {"none", "random", "uniform_grid"}

SIMPLE_SCORE_TOPK_STRATEGIES: Set[str] = {
    "depth_edge_fast",
    "robot_geo_rule_v0",
    "robot_geo_dynamic_v0",
    "robot_geo_temporal_v0",
}

CURRENT_SELF_HANDLED_SELECTOR_STRATEGIES: Set[str] = {
    "robot_geo_acgtp_v2",
}

SELF_HANDLED_SELECTOR_STRATEGIES: Set[str] = (
    CURRENT_SELF_HANDLED_SELECTOR_STRATEGIES | LEGACY_SELF_HANDLED_SELECTOR_STRATEGIES
)

SELECTION_STAGE_NAMES: Dict[str, str] = {
    "none": "no_pruning",
    "random": "random_sample",
    "uniform_grid": "uniform_grid",
    "depth_edge_fast": "global_depth_edge_topk",
    "depth_edge_fast_diverse": "depth_edge_quota_spatial_diversity",
    "robot_geo_contact_budget": "contact_budget",
    "robot_geo_hybrid_v2": "hybrid_quota_v2",
    "robot_geo_hybrid_v1": "hybrid_score_grid_fill",
    "hybrid_budget_v2": "hybrid_budget_v2",
}


def selection_stage_name(strategy: str, *, fallback: bool = False) -> str:
    if fallback:
        return "fallback"
    s = str(strategy or "none")
    if s.startswith("robot_geo_hybrid_temporal_edge_reserve"):
        return "edge_reserve_hybrid_score_grid_fill"
    if s.startswith("robot_geo"):
        return "global_robot_geo_topk"
    return SELECTION_STAGE_NAMES.get(s, "selection")
