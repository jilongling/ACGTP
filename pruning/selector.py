"""Public selector compatibility surface for visual-token pruning.

Concrete implementations live in smaller modules:
- selector_core.py: simple/top-k/grid/contact selectors.
- acgtp_v2_selectors.py: ACGTP-v2 selectors.
- legacy_selectors.py: legacy ablation/audit selectors.
"""

from __future__ import annotations

from .methods.baselines import (
    select_tokens_with_spatial_diversity,
    select_keep_indices,
    select_tokens_contact_budget,
    select_uniform_grid_indices,
    select_depth_edge_diverse_indices,
)
from .methods.acgtp_v2 import select_acgtp_v2_fast, select_acgtp_v2
from .legacy import (
    select_hybrid_quota_union,
    select_hybrid_quota_v2,
    select_hybrid_v1,
    select_hybrid_v1_edge_reserve,
    select_hybrid_budget_v2,
    select_branch_budget_v0,
    select_acgtp_v1,
)
from .methods.utils import finalize_selection_debug_info, validate_keep_indices

__all__ = [
    "finalize_selection_debug_info",
    "validate_keep_indices",
    "select_tokens_with_spatial_diversity",
    "select_keep_indices",
    "select_tokens_contact_budget",
    "select_uniform_grid_indices",
    "select_depth_edge_diverse_indices",
    "select_acgtp_v2_fast",
    "select_acgtp_v2",
    "select_hybrid_quota_union",
    "select_hybrid_quota_v2",
    "select_hybrid_v1",
    "select_hybrid_v1_edge_reserve",
    "select_hybrid_budget_v2",
    "select_branch_budget_v0",
    "select_acgtp_v1",
]
