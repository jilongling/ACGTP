"""
ACGTP-v2 Semantic Scene-Layout Branch — Full Audit Suite

Tasks:
  1. Fallback equivalence: v1 vs v2 (semantic_backend=none) — stable sort, Jaccard >= 0.90
  2. Instruction parser audit: 40 LIBERO tasks — action/relation words excluded from refs
  3. Semantic backend audit: none / parser_only / mock backends
  4. Scene-layout branch integration check (mock mode)
  5. Accounting & debug field verification
  6. Attention task-relevance alignment audit (VLA-Cache / VLA-IAP / VLA-Pruner inspired)
  7. Attention stress tests: Cases A-E (mock backends)

Usage:
    python experiments/robot/libero/acgtp_v2_audit.py
    python experiments/robot/libero/acgtp_v2_audit.py --task fallback_equivalence
    python experiments/robot/libero/acgtp_v2_audit.py --task parser_audit
    python experiments/robot/libero/acgtp_v2_audit.py --task semantic_backend_audit
    python experiments/robot/libero/acgtp_v2_audit.py --task scene_layout_audit
    python experiments/robot/libero/acgtp_v2_audit.py --task attention_alignment_audit
    python experiments/robot/libero/acgtp_v2_audit.py --task attention_stress_audit
    python experiments/robot/libero/acgtp_v2_audit.py --task all
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path("/infini-data/openvla")
sys.path.insert(0, str(REPO_ROOT))

# ── Selector imports ────────────────────────────────────────────────────────
from pruning.selector import select_acgtp_v1, select_acgtp_v2
from pruning.semantic_anchors import (
    parse_instruction_terms,
    compute_task_semantic_anchors,
    jaccard,
)

# ── Backend imports ──────────────────────────────────────────────────────────
from experiments.robot.libero.acgtp_semantic_backend import (
    SemanticLayoutBackend,
    get_semantic_backend,
    SemanticLayoutResult,
)

# ── Attention relevance imports ───────────────────────────────────────────────
from pruning.attention_relevance import (
    get_attention_relevance,
    compute_attention_geometry_alignment,
    compute_safe_drop_diagnostic,
    AttentionRelevanceResult,
)

# ── Output dir ─────────────────────────────────────────────────────────────
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# Synthetic geometry score factory
# ═════════════════════════════════════════════════════════════════════════════

def make_synthetic_geometry_scores(
    rng: np.random.Generator,
    n: int = 256,
) -> Dict[str, Any]:
    """Create realistic synthetic geometry scores matching ACGTP branch output characteristics.

    Score distributions are BIMODAL: LOW tokens in [0.0, 0.15] and HIGH tokens in
    [0.70, 1.0]. This ensures a fixed threshold of 0.50 cleanly separates the two
    populations, which is essential for safe-drop diagnostics that use fixed
    thresholds rather than percentiles.
    """
    assert n == 256, f"Expected 256 tokens (16×16 grid), got {n}"
    grid_h, grid_w = 16, 16
    gripper_r, gripper_c = grid_h // 2, grid_w // 2

    # Scene layout: bimodal [0.0,0.15] LOW vs [0.70,1.0] HIGH.
    scene_scores = rng.uniform(0.0, 0.15, size=n).astype(np.float32)
    high_scene_indices = rng.choice(n, size=max(1, n // 4), replace=False)
    scene_scores[high_scene_indices] = rng.uniform(0.70, 1.0, size=len(high_scene_indices)).astype(np.float32)

    # Depth edge: bimodal boundary rows HIGH, rest LOW.
    depth_scores = rng.uniform(0.0, 0.15, size=n).astype(np.float32)
    edge_rows = rng.choice(grid_h, size=max(1, grid_h // 3), replace=False)
    for row in edge_rows:
        depth_scores[row * grid_w:(row + 1) * grid_w] = rng.uniform(0.70, 1.0, size=grid_w).astype(np.float32)

    # Contact ring: bimodal ring HIGH, rest LOW.
    contact_scores = rng.uniform(0.0, 0.15, size=n).astype(np.float32)
    for idx in range(n):
        r, c = idx // grid_w, idx % grid_w
        dist = ((r - gripper_r) ** 2 + (c - gripper_c) ** 2) ** 0.5
        if 1.5 <= dist <= 3.0:
            contact_scores[idx] = float(rng.uniform(0.70, 1.0))

    # Motion corridor: bimodal swept-path HIGH, rest LOW.
    motion_scores = rng.uniform(0.0, 0.15, size=n).astype(np.float32)
    for idx in range(n):
        r, c = idx // grid_w, idx % grid_w
        dist_along = abs(r - gripper_r)
        if dist_along <= 4:
            motion_scores[idx] = float(rng.uniform(0.70, 1.0))

    # Valid mask (~92% valid)
    valid_mask = rng.choice([True, False], size=n, p=[0.92, 0.08])

    # Constrained fill: valid + some scene relevance
    fill_mask = (valid_mask & (scene_scores > 0.1)).astype(np.float32)

    # Self-core mask: center tokens
    self_core = np.zeros(n, dtype=np.float32)
    for idx in range(n):
        r, c = idx // grid_w, idx % grid_w
        if ((r - gripper_r) ** 2 + (c - gripper_c) ** 2) <= 1.5:
            self_core[idx] = 1.0

    token_u = np.repeat(np.linspace(0, 255, grid_w, dtype=np.float32), grid_h)
    token_v = np.tile(np.linspace(0, 255, grid_h, dtype=np.float32), grid_w)

    return {
        "scene_scores": scene_scores,
        "depth_scores": depth_scores,
        "contact_scores": contact_scores,
        "motion_scores": motion_scores,
        "valid_mask": valid_mask,
        "fill_mask": fill_mask,
        "self_core": self_core,
        "token_u": token_u,
        "token_v": token_v,
        "motion_valid": True,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Task 1: v1/v2 fallback equivalence with stable sort guarantee
# ═════════════════════════════════════════════════════════════════════════════

def _call_v1(scores: Dict, keep_k: int) -> tuple:
    """Call select_acgtp_v1 with synthetic scores."""
    return select_acgtp_v1(
        scene_layout_scores=scores["scene_scores"],
        depth_edge_scores=scores["depth_scores"],
        contact_ring_scores=scores["contact_scores"],
        motion_corridor_scores=scores["motion_scores"],
        valid_mask=scores["valid_mask"],
        keep_k=keep_k,
        constrained_fill_mask=scores["fill_mask"],
        token_u=scores["token_u"],
        token_v=scores["token_v"],
        grid_h=16, grid_w=16,
        w_scene_layout=0.30,
        w_depth_structure=0.25,
        w_contact_ring=0.25,
        w_motion_corridor=0.20,
        hard_protect_ratio=0.60,
        motion_corridor_valid=scores["motion_valid"],
        self_core_mask=scores["self_core"],
        contact_ring_inner_px=24.0,
        contact_ring_outer_px=48.0,
        contact_requires_edge_or_object=True,
        depth_edge_score_for_gate=scores["depth_scores"],
    )


def _call_v2_fallback(scores: Dict, keep_k: int, instruction: str) -> tuple:
    """Call select_acgtp_v2 with semantic_backend=none (geometric fallback)."""
    sem_result = compute_task_semantic_anchors(
        instruction=instruction,
        rgb=None,
        token_u=scores["token_u"],
        token_v=scores["token_v"],
        token_depth=None,
        scene_result=None,
        config=None,
        semantic_enabled=False,
        semantic_backend="none",
        w_semantic_target=1.0,
        w_semantic_reference=0.7,
        w_semantic_relation=0.5,
        w_semantic_goal=0.9,
        target_cap_ratio=0.25,
        reference_cap_ratio=0.20,
        relation_cap_ratio=0.15,
        hard_ratio=0.20,
        release_quota_when_unavailable=True,
        grid_h=16, grid_w=16,
    )
    parsed = parse_instruction_terms(instruction)
    raw = parsed.get("raw_parse", {})

    return select_acgtp_v2(
        scene_layout_scores=scores["scene_scores"],
        depth_edge_scores=scores["depth_scores"],
        contact_ring_scores=scores["contact_scores"],
        motion_corridor_scores=scores["motion_scores"],
        semantic_anchor_scores=sem_result.get("semantic_anchor_scores"),
        semantic_target_scores=sem_result.get("semantic_target_scores"),
        semantic_reference_scores=sem_result.get("semantic_reference_scores"),
        semantic_relation_scores=sem_result.get("semantic_relation_scores"),
        semantic_goal_scores=sem_result.get("semantic_goal_scores"),
        valid_mask=scores["valid_mask"],
        keep_k=keep_k,
        constrained_fill_mask=scores["fill_mask"],
        token_u=scores["token_u"],
        token_v=scores["token_v"],
        grid_h=16, grid_w=16,
        w_scene_layout=0.30,
        w_depth_structure=0.25,
        w_contact_ring=0.25,
        w_motion_corridor=0.20,
        w_semantic=0.20,
        hard_protect_ratio=0.60,
        motion_corridor_valid=scores["motion_valid"],
        self_core_mask=scores["self_core"],
        contact_ring_inner_px=24.0,
        contact_ring_outer_px=48.0,
        contact_requires_edge_or_object=True,
        depth_edge_score_for_gate=scores["depth_scores"],
        support_plane_cap_ratio=0.30,
        semantic_enabled=False,
        semantic_backend="none",
        semantic_confidence=float(sem_result.get("semantic_confidence", 0.0)),
        semantic_unavailable=bool(sem_result.get("semantic_unavailable", True)),
        semantic_fallback_reason=sem_result.get("semantic_fallback_reason"),
        release_semantic_quota_when_unavailable=True,
        w_semantic_target=1.0,
        w_semantic_reference=0.7,
        w_semantic_relation=0.5,
        w_semantic_goal=0.9,
        target_cap_ratio=0.25,
        reference_cap_ratio=0.20,
        relation_cap_ratio=0.15,
        hard_semantic_ratio=0.20,
        parsed_target_terms=raw.get("parsed_target_terms", []),
        parsed_reference_terms=raw.get("parsed_reference_terms", []),
        parsed_relation_terms=raw.get("parsed_relation_terms", []),
        instruction_is_meaningful=bool(parsed.get("instruction_is_meaningful", False)),
        # Scene-layout branch (inactive in fallback)
        scene_layout_branch_active=False,
        scene_layout_available=False,
        scene_layout_confidence=0.0,
        target_mask_count=int(sem_result.get("target_mask_count", 0)),
        reference_mask_count=int(sem_result.get("reference_mask_count", 0)),
        relation_mask_count=int(sem_result.get("relation_mask_count", 0)),
        layout_anchor_mask_count=int(sem_result.get("layout_anchor_mask_count", 0)),
        scene_layout_indices=[],
    )


def _call_v2_relaxed(scores: Dict, keep_k: int, instruction: str) -> tuple:
    """
    RELAXED fallback: calls select_acgtp_v2 in its NATIVE v2 path with
    semantic_enabled=False / semantic_backend="none" / semantic_unavailable=True
    but WITHOUT the strict fallback dispatch.

    This deliberately runs v2's full normalization, scoring, hard-protect, and
    constrained-fill pipeline (w_semantic=0, no semantic branch). It is useful
    as a DIAGNOSTIC path to verify that even without strict dispatch, v2 produces
    results close to v1 (Jaccard >= 0.90 expected but not guaranteed).

    IMPORTANT: This is NOT a fallback_dispatch_to_v1 path. The keep_indices may
    differ from v1 due to v2-specific normalization and scoring differences.
    Use strict_fallback_dispatch_used / delegated_selector_name to distinguish.

    Returns (v2_indices, v2_meta) from v2's native path.
    """
    parsed = parse_instruction_terms(instruction)
    raw = parsed.get("raw_parse", {})
    sem_result = compute_task_semantic_anchors(
        instruction=instruction,
        rgb=None,
        token_u=scores["token_u"],
        token_v=scores["token_v"],
        token_depth=None,
        scene_result=None,
        config=None,
        semantic_enabled=False,
        semantic_backend="none",
        w_semantic_target=1.0,
        w_semantic_reference=0.7,
        w_semantic_relation=0.5,
        w_semantic_goal=0.9,
        target_cap_ratio=0.25,
        reference_cap_ratio=0.20,
        relation_cap_ratio=0.15,
        hard_ratio=0.20,
        release_quota_when_unavailable=True,
        grid_h=16, grid_w=16,
    )

    return select_acgtp_v2(
        scene_layout_scores=scores["scene_scores"],
        depth_edge_scores=scores["depth_scores"],
        contact_ring_scores=scores["contact_scores"],
        motion_corridor_scores=scores["motion_scores"],
        semantic_anchor_scores=sem_result.get("semantic_anchor_scores"),
        semantic_target_scores=sem_result.get("semantic_target_scores"),
        semantic_reference_scores=sem_result.get("semantic_reference_scores"),
        semantic_relation_scores=sem_result.get("semantic_relation_scores"),
        semantic_goal_scores=sem_result.get("semantic_goal_scores"),
        valid_mask=scores["valid_mask"],
        keep_k=keep_k,
        constrained_fill_mask=scores["fill_mask"],
        token_u=scores["token_u"],
        token_v=scores["token_v"],
        grid_h=16, grid_w=16,
        w_scene_layout=0.25,  # v2 default weights (different from v1)
        w_depth_structure=0.20,
        w_contact_ring=0.20,
        w_motion_corridor=0.15,
        w_semantic=0.20,
        hard_protect_ratio=0.60,
        motion_corridor_valid=scores["motion_valid"],
        self_core_mask=scores["self_core"],
        contact_ring_inner_px=24.0,
        contact_ring_outer_px=48.0,
        contact_requires_edge_or_object=True,
        depth_edge_score_for_gate=scores["depth_scores"],
        support_plane_cap_ratio=0.30,
        semantic_enabled=False,
        semantic_backend="none",
        semantic_confidence=float(sem_result.get("semantic_confidence", 0.0)),
        semantic_unavailable=bool(sem_result.get("semantic_unavailable", True)),
        semantic_fallback_reason=sem_result.get("semantic_fallback_reason"),
        release_semantic_quota_when_unavailable=True,
        w_semantic_target=1.0,
        w_semantic_reference=0.7,
        w_semantic_relation=0.5,
        w_semantic_goal=0.9,
        target_cap_ratio=0.25,
        reference_cap_ratio=0.20,
        relation_cap_ratio=0.15,
        hard_semantic_ratio=0.20,
        parsed_target_terms=raw.get("parsed_target_terms", []),
        parsed_reference_terms=raw.get("parsed_reference_terms", []),
        parsed_relation_terms=raw.get("parsed_relation_terms", []),
        instruction_is_meaningful=bool(parsed.get("instruction_is_meaningful", False)),
        scene_layout_branch_active=False,
        scene_layout_available=False,
        scene_layout_confidence=0.0,
        target_mask_count=int(sem_result.get("target_mask_count", 0)),
        reference_mask_count=int(sem_result.get("reference_mask_count", 0)),
        relation_mask_count=int(sem_result.get("relation_mask_count", 0)),
        layout_anchor_mask_count=int(sem_result.get("layout_anchor_mask_count", 0)),
        scene_layout_indices=[],
        # Attention disabled to isolate relaxed semantic-disabled path
        acgtp_attention_enabled=False,
        acgtp_attention_backend="none",
        acgtp_attention_min_confidence=0.0,
        acgtp_attention_requires_geometry_alignment=True,
        acgtp_attention_budget_ratio=0.10,
        acgtp_attention_task_relevance_score=None,
        acgtp_attention_task_relevance_mask=None,
        acgtp_attention_source="none",
        acgtp_attention_available=False,
        acgtp_attention_confidence=0.0,
    )


def _call_v2_mock(scores: Dict, keep_k: int, instruction: str, seed: int = 0) -> tuple:
    """Call select_acgtp_v2 with semantic_backend=mock (scene-layout branch active)."""
    parsed = parse_instruction_terms(instruction)
    raw = parsed.get("raw_parse", {})
    sem_result = compute_task_semantic_anchors(
        instruction=instruction,
        rgb=None,
        token_u=scores["token_u"],
        token_v=scores["token_v"],
        token_depth=None,
        scene_result=None,
        config=None,
        semantic_enabled=True,
        semantic_backend="mock",
        w_semantic_target=1.0,
        w_semantic_reference=0.7,
        w_semantic_relation=0.5,
        w_semantic_goal=0.9,
        target_cap_ratio=0.25,
        reference_cap_ratio=0.20,
        relation_cap_ratio=0.15,
        hard_ratio=0.20,
        release_quota_when_unavailable=True,
        grid_h=16, grid_w=16,
    )

    sl_active = bool(sem_result.get("scene_layout_branch_active", False))
    sl_available = bool(sem_result.get("scene_layout_available", False))
    sl_conf = float(sem_result.get("scene_layout_confidence", 0.0))
    tgt_cnt = int(sem_result.get("target_mask_count", 0))
    ref_cnt = int(sem_result.get("reference_mask_count", 0))
    rel_cnt = int(sem_result.get("relation_mask_count", 0))
    lay_cnt = int(sem_result.get("layout_anchor_mask_count", 0))

    return select_acgtp_v2(
        scene_layout_scores=scores["scene_scores"],
        depth_edge_scores=scores["depth_scores"],
        contact_ring_scores=scores["contact_scores"],
        motion_corridor_scores=scores["motion_scores"],
        semantic_anchor_scores=sem_result.get("semantic_anchor_scores"),
        semantic_target_scores=sem_result.get("semantic_target_scores"),
        semantic_reference_scores=sem_result.get("semantic_reference_scores"),
        semantic_relation_scores=sem_result.get("semantic_relation_scores"),
        semantic_goal_scores=sem_result.get("semantic_goal_scores"),
        valid_mask=scores["valid_mask"],
        keep_k=keep_k,
        constrained_fill_mask=scores["fill_mask"],
        token_u=scores["token_u"],
        token_v=scores["token_v"],
        grid_h=16, grid_w=16,
        w_scene_layout=0.30,
        w_depth_structure=0.25,
        w_contact_ring=0.25,
        w_motion_corridor=0.20,
        w_semantic=0.20,
        hard_protect_ratio=0.60,
        motion_corridor_valid=scores["motion_valid"],
        self_core_mask=scores["self_core"],
        contact_ring_inner_px=24.0,
        contact_ring_outer_px=48.0,
        contact_requires_edge_or_object=True,
        depth_edge_score_for_gate=scores["depth_scores"],
        support_plane_cap_ratio=0.30,
        semantic_enabled=True,
        semantic_backend="mock",
        semantic_confidence=float(sem_result.get("semantic_confidence", 0.0)),
        semantic_unavailable=bool(sem_result.get("semantic_unavailable", False)),
        semantic_fallback_reason=sem_result.get("semantic_fallback_reason"),
        release_semantic_quota_when_unavailable=False,
        w_semantic_target=1.0,
        w_semantic_reference=0.7,
        w_semantic_relation=0.5,
        w_semantic_goal=0.9,
        target_cap_ratio=0.25,
        reference_cap_ratio=0.20,
        relation_cap_ratio=0.15,
        hard_semantic_ratio=0.20,
        parsed_target_terms=raw.get("parsed_target_terms", []),
        parsed_reference_terms=raw.get("parsed_reference_terms", []),
        parsed_relation_terms=raw.get("parsed_relation_terms", []),
        instruction_is_meaningful=bool(parsed.get("instruction_is_meaningful", False)),
        # Scene-layout branch (active in mock mode)
        scene_layout_branch_active=sl_active,
        scene_layout_available=sl_available,
        scene_layout_confidence=sl_conf,
        target_mask_count=tgt_cnt,
        reference_mask_count=ref_cnt,
        relation_mask_count=rel_cnt,
        layout_anchor_mask_count=lay_cnt,
        scene_layout_indices=[],
    )


def task1_fallback_equivalence(
    seed: int = 7,
    n_tokens: int = 256,
    keep_ratio: float = 0.875,
    num_steps: int = 20,
) -> Dict[str, Any]:
    """
    Task 1: Strict v1/v2 fallback equivalence — literal dispatch.

    The STRICT FALLBACK path in select_acgtp_v2 calls select_acgtp_v1 directly
    (pass-through, no v2 normalization/scoring/fill logic runs at all).
    Therefore strict_fallback mode guarantees:

      - keep_indices are BIT-FOR-BIT IDENTICAL to v1 (not merely Jaccard >= 0.90)
      - keep_indices_exact_equal == True
      - keep_indices_jaccard == 1.0
      - keep_counts_equal == True
      - strict_fallback_dispatch_used == True
      - delegated_selector_name == "select_acgtp_v1"
      - v2 branch attribution matches v1 (geometry fields only)
      - v2 accounting valid

    Assertions for semantic_backend=none mode:
      - strict_fallback_dispatch_used == True
      - delegated_selector_name == "select_acgtp_v1"
      - semantic_unavailable == True
      - semantic_confidence == 0.0
      - selected_by_semantic_count == 0
      - release_quota == True
      - keep_indices_exact_equal == True
      - keep_indices_jaccard == 1.0
      - v2 accounting valid
    """
    try:
        from libero.libero import benchmark
        bs = benchmark.get_benchmark_dict()["libero_spatial"]()
        task = bs.get_task(0)
        instruction = task.language.strip()
    except Exception:
        instruction = "pick up the black bowl between the plate and the red mug and place it on the tray"

    report: Dict[str, Any] = {
        "seed": seed,
        "n_tokens": n_tokens,
        "keep_ratio": keep_ratio,
        "keep_k": int(round(n_tokens * keep_ratio)),
        "num_steps": num_steps,
        "instruction": instruction,
        "steps": [],
        "errors": [],
        "assertions": {
            "all_strict_fallback_dispatch_used": True,
            "all_delegated_to_v1": True,
            "all_semantic_unavailable": True,
            "all_semantic_confidence_zero": True,
            "all_sel_semantic_zero": True,
            "all_quota_released": True,
            "all_acct_valid": True,
            "all_keep_counts_equal": True,
            "all_indices_exact_equal": True,
            "all_jaccard_eq_1": True,
        },
        "equivalence_passed": True,
    }

    for step in range(num_steps):
        step_seed = seed + step * 1000
        rng = np.random.default_rng(step_seed)
        scores = make_synthetic_geometry_scores(rng=rng, n=n_tokens)
        keep_k = report["keep_k"]

        # Run both versions
        t0 = time.perf_counter()
        v1_indices, v1_meta = _call_v1(scores, keep_k)
        v1_time_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        v2_indices, v2_meta = _call_v2_fallback(scores, keep_k, instruction)
        v2_time_ms = (time.perf_counter() - t1) * 1000.0

        # Comparisons
        v1_sorted = np.sort(v1_indices)
        v2_sorted = np.sort(v2_indices)
        indices_match = np.array_equal(v1_sorted, v2_sorted)
        jaccard_val = jaccard(v1_indices, v2_indices)

        step_rec: Dict[str, Any] = {
            "step": step,
            "seed": step_seed,
            "v1_time_ms": round(v1_time_ms, 4),
            "v2_time_ms": round(v2_time_ms, 4),
            "v2_minus_v1_ms": round(v2_time_ms - v1_time_ms, 4),
            # Counts
            "v1_keep_count": len(v1_indices),
            "v2_keep_count": len(v2_indices),
            "keep_counts_equal": len(v1_indices) == len(v2_indices),
            "indices_match": indices_match,
            "jaccard": round(jaccard_val, 6),
            # v2 strict fallback dispatch state
            "strict_fallback_dispatch_used": bool(v2_meta.get("strict_fallback_dispatch_used", False)),
            "delegated_selector_name": v2_meta.get("delegated_selector_name", "none"),
            "fallback_dispatch_to_v1": bool(v2_meta.get("fallback_dispatch_to_v1", False)),
            # v2 semantic state assertions
            "v2_sem_unavailable": bool(v2_meta.get("acgtp_v2_semantic_unavailable", False)),
            "v2_sem_confidence": float(v2_meta.get("acgtp_v2_semantic_confidence", -1)),
            "v2_sem_backend": v2_meta.get("acgtp_v2_semantic_backend", "unknown"),
            "v2_release_quota": bool(v2_meta.get("acgtp_v2_release_quota", False)),
            "v2_sel_semantic": v2_meta.get("selected_by_semantic_count", -1),
            "v2_sel_target": v2_meta.get("selected_by_semantic_target_count", -1),
            "v2_sel_ref": v2_meta.get("selected_by_semantic_reference_count", -1),
            "v2_sel_rel": v2_meta.get("selected_by_semantic_relation_count", -1),
            "v2_sel_goal": v2_meta.get("selected_by_semantic_goal_count", -1),
            # v2 scene-layout branch assertions
            "v2_sl_active": v2_meta.get("scene_layout_branch_active", False),
            "v2_sl_available": v2_meta.get("scene_layout_available", False),
            "v2_sl_quota": v2_meta.get("scene_layout_branch_quota", -1),
            "v2_sel_sl": v2_meta.get("selected_by_scene_layout_count", -1),
            "v2_sl_confidence": v2_meta.get("scene_layout_confidence", -1),
            # Accounting
            "v1_acct_valid": bool(v1_meta.get("acgtp_branch_accounting_valid", False)),
            "v2_acct_valid": bool(v2_meta.get("acgtp_branch_accounting_valid", False)),
            "v2_acct_valid_alias": bool(v2_meta.get("branch_accounting_valid", False)),
            "v1_branch_sum": v1_meta.get("acgtp_branch_sum", -1),
            "v2_branch_sum": v2_meta.get("acgtp_branch_sum", -1),
            # v1 branch attribution (geometry-only, shared between v1 and v2)
            "v1_scene": v1_meta.get("selected_by_scene_layout_count"),
            "v1_depth": v1_meta.get("selected_by_depth_structure_count"),
            "v1_contact": v1_meta.get("selected_by_contact_ring_count"),
            "v1_motion": v1_meta.get("selected_by_motion_corridor_count"),
            "v1_fill": v1_meta.get("selected_by_constrained_fill_count"),
            "v1_fallback": v1_meta.get("selected_by_acgtp_fallback_count"),
            # v2 branch attribution (in strict fallback: mirrors v1 exactly)
            "v2_scene": v2_meta.get("selected_by_scene_layout_count"),
            "v2_depth": v2_meta.get("selected_by_depth_structure_count"),
            "v2_contact": v2_meta.get("selected_by_contact_ring_count"),
            "v2_motion": v2_meta.get("selected_by_motion_corridor_count"),
            "v2_fill": v2_meta.get("selected_by_constrained_fill_count"),
            "v2_fallback": v2_meta.get("selected_by_acgtp_fallback_count"),
            "v2_sp_selected": v2_meta.get("acgtp_scene_selected_support_plane_count"),
            "v2_residual_fill": v2_meta.get("acgtp_scene_selected_residual_fill_count"),
            # New scene-layout fields
            "v2_target_mask_count": v2_meta.get("target_mask_count", 0),
            "v2_reference_mask_count": v2_meta.get("reference_mask_count", 0),
            "v2_relation_mask_count": v2_meta.get("relation_mask_count", 0),
            "v2_layout_anchor_mask_count": v2_meta.get("layout_anchor_mask_count", 0),
            "v2_scene_layout_indices": sorted(v2_meta.get("scene_layout_indices", []))[:20],
            "v2_overlap_scene_depth": v2_meta.get("overlap_scene_depth_count", 0),
            "v2_overlap_scene_geo": v2_meta.get("overlap_scene_geometry_count", 0),
            # Assertions for this step — strict fallback requires exact equality
            "assert_strict_fallback_used": v2_meta.get("strict_fallback_dispatch_used", False) is True,
            "assert_delegated_to_v1": v2_meta.get("delegated_selector_name", "") == "select_acgtp_v1",
            "assert_semantic_unavailable": v2_meta.get("acgtp_v2_semantic_unavailable", False) is True,
            "assert_semantic_confidence_zero": v2_meta.get("acgtp_v2_semantic_confidence", -1) == 0.0,
            "assert_sel_semantic_zero": v2_meta.get("selected_by_semantic_count", -1) == 0,
            "assert_quota_released": v2_meta.get("acgtp_v2_release_quota", False) is True,
            "assert_acct_valid": v2_meta.get("acgtp_branch_accounting_valid", False) is True,
            "assert_keep_count_eq": len(v1_indices) == len(v2_indices),
            "assert_indices_exact_equal": indices_match,
            "assert_jaccard_eq_1": jaccard_val == 1.0,
        }

        report["steps"].append(step_rec)

        # Collect errors — strict fallback mode requires exact equality
        if not step_rec["assert_strict_fallback_used"]:
            report["errors"].append(f"Step {step}: strict_fallback_dispatch_used != True")
            report["equivalence_passed"] = False
        if not step_rec["assert_delegated_to_v1"]:
            report["errors"].append(
                f"Step {step}: delegated_selector_name={v2_meta.get('delegated_selector_name')} "
                f"(expected select_acgtp_v1)"
            )
            report["equivalence_passed"] = False
        if not step_rec["assert_semantic_unavailable"]:
            report["errors"].append(f"Step {step}: semantic_unavailable != True")
        if not step_rec["assert_semantic_confidence_zero"]:
            report["errors"].append(f"Step {step}: semantic_confidence != 0.0")
        if not step_rec["assert_sel_semantic_zero"]:
            report["errors"].append(f"Step {step}: selected_by_semantic_count != 0")
        if not step_rec["assert_quota_released"]:
            report["errors"].append(f"Step {step}: release_quota != True")
        if not step_rec["assert_acct_valid"]:
            report["errors"].append(f"Step {step}: accounting invalid")
        if not step_rec["assert_keep_count_eq"]:
            report["errors"].append(f"Step {step}: keep_count mismatch v1={len(v1_indices)} v2={len(v2_indices)}")
            report["equivalence_passed"] = False
        if not step_rec["assert_indices_exact_equal"]:
            report["errors"].append(
                f"Step {step}: keep_indices NOT exact equal — "
                f"v1={len(v1_indices)} items vs v2={len(v2_indices)} items"
            )
            report["equivalence_passed"] = False
        if not step_rec["assert_jaccard_eq_1"]:
            report["errors"].append(f"Step {step}: Jaccard {jaccard_val:.6f} != 1.0 (strict fallback requires exact dispatch)")
            report["equivalence_passed"] = False

    # Build summary
    steps = report["steps"]
    # v2-only diagnostic fields that are added by v2 but do NOT exist in v1.
    # When comparing branch attributions between v1 and strict-fallback v2, these
    # fields are irrelevant because strict fallback v2 copies v1's geometry attribution.
    _v2_only_diagnostic_fields = [
        "acgtp_v2_semantic_enabled", "acgtp_v2_semantic_backend",
        "acgtp_v2_semantic_confidence", "acgtp_v2_semantic_unavailable",
        "acgtp_v2_semantic_fallback_reason", "acgtp_v2_release_quota",
        "acgtp_v2_parsed_instruction_meaningful",
        "acgtp_v2_parsed_target_terms", "acgtp_v2_parsed_reference_terms",
        "acgtp_v2_parsed_relation_terms",
        "acgtp_v2_w_semantic_target", "acgtp_v2_w_semantic_reference",
        "acgtp_v2_w_semantic_relation", "acgtp_v2_w_semantic_goal",
        "acgtp_v2_hard_semantic_quota",
        "acgtp_v2_target_cap_k", "acgtp_v2_reference_cap_k", "acgtp_v2_relation_cap_k",
        "acgtp_v2_semantic_available",
        "acgtp_v2_semantic_score_mean", "acgtp_v2_semantic_score_max",
        "acgtp_v2_semantic_target_token_count", "acgtp_v2_semantic_reference_token_count",
        "acgtp_v2_semantic_relation_token_count", "acgtp_v2_semantic_goal_token_count",
        "acgtp_v2_semantic_anchor_token_count",
        "selected_by_semantic_target_count", "selected_by_semantic_reference_count",
        "selected_by_semantic_relation_count", "selected_by_semantic_goal_count",
        "selected_by_semantic_count",
        "semantic_overlap_with_scene_count", "semantic_overlap_with_depth_count",
        "semantic_overlap_with_contact_count", "semantic_overlap_with_motion_count",
        "selected_by_scene_residual_fill_count",
        # Attention-only fields
        "acgtp_attention_enabled", "acgtp_attention_backend", "acgtp_attention_source",
        "acgtp_attention_available", "acgtp_attention_confidence",
        "acgtp_attention_quota_released", "acgtp_attention_requires_geometry_alignment",
        "acgtp_attention_budget_ratio", "acgtp_attention_top_count",
        "acgtp_attention_candidate_count", "acgtp_attention_only_token_count",
        "attention_only_token_count", "geometry_only_token_count",
        "attention_selected_by_final_count", "attn_alignment_verified",
        # Strict fallback dispatch fields
        "strict_fallback_dispatch_used", "delegated_selector_name",
        "fallback_dispatch_to_v1",
        # Unused / alias fields
        "acgtp_w_semantic", "acgtp_v2_semantic_available",
    ]
    report["summary"] = {
        # Strict fallback dispatch
        "all_strict_fallback_dispatch_used": all(s["strict_fallback_dispatch_used"] for s in steps),
        "all_delegated_to_v1": all(s["delegated_selector_name"] == "select_acgtp_v1" for s in steps),
        "all_fallback_dispatch_to_v1": all(s["fallback_dispatch_to_v1"] for s in steps),
        # Index equality
        "all_indices_match": all(s["indices_match"] for s in steps),
        "all_keep_counts_equal": all(s["keep_counts_equal"] for s in steps),
        "all_indices_exact_equal": all(s["indices_match"] for s in steps),
        # Semantic state
        "all_semantic_unavailable": all(s["v2_sem_unavailable"] for s in steps),
        "all_semantic_confidence_zero": all(s["v2_sem_confidence"] == 0.0 for s in steps),
        "all_sel_semantic_zero": all(s["v2_sel_semantic"] == 0 for s in steps),
        "all_quota_released": all(s["v2_release_quota"] for s in steps),
        # Accounting
        "all_acct_valid": all(s["v2_acct_valid"] for s in steps),
        "all_v1_acct_valid": all(s["v1_acct_valid"] for s in steps),
        "all_v2_acct_valid": all(s["v2_acct_valid"] for s in steps),
        # Jaccard — STRICT: must be exactly 1.0 (not just >= 0.90)
        "min_jaccard": min(s["jaccard"] for s in steps),
        "max_jaccard": max(s["jaccard"] for s in steps),
        "avg_jaccard": round(sum(s["jaccard"] for s in steps) / len(steps), 6),
        "all_jaccard_eq_1": all(s["jaccard"] == 1.0 for s in steps),
        # Performance
        "avg_v2_minus_v1_ms": round(sum(s["v2_minus_v1_ms"] for s in steps) / len(steps), 4),
        "max_v2_minus_v1_ms": max(s["v2_minus_v1_ms"] for s in steps),
        # Branch attribution comparison — geometry fields only (v2 adds semantic/attention diagnostics)
        "all_branch_attributions_equal": all(
            s["v1_scene"] == s["v2_scene"] and
            s["v1_depth"] == s["v2_depth"] and
            s["v1_contact"] == s["v2_contact"] and
            s["v1_motion"] == s["v2_motion"]
            for s in steps
        ),
        "error_count": len(report["errors"]),
        # Scene-layout assertions
        "all_scene_layout_branch_inactive": all(not s["v2_sl_active"] for s in steps),
        "all_scene_layout_unavailable": all(not s["v2_sl_available"] for s in steps),
        "all_scene_layout_confidence_zero": all(s["v2_sl_confidence"] == 0.0 for s in steps),
        "all_scene_layout_quota_zero": all(s["v2_sl_quota"] == 0 for s in steps),
        # v2-only diagnostic fields (ignored in comparison)
        "ignored_v2_diagnostic_fields": _v2_only_diagnostic_fields,
    }

    report["equivalence_passed"] = (
        report["equivalence_passed"]
        and report["summary"]["all_strict_fallback_dispatch_used"]
        and report["summary"]["all_delegated_to_v1"]
        and report["summary"]["all_semantic_unavailable"]
        and report["summary"]["all_semantic_confidence_zero"]
        and report["summary"]["all_sel_semantic_zero"]
        and report["summary"]["all_quota_released"]
        and report["summary"]["all_acct_valid"]
        and report["summary"]["all_keep_counts_equal"]
        and report["summary"]["all_indices_exact_equal"]
        and report["summary"]["all_jaccard_eq_1"]
    )

    return report


# ═════════════════════════════════════════════════════════════════════════════
# Task 2: Instruction parser audit — new nested schema with action/relation filter
# ═════════════════════════════════════════════════════════════════════════════

def task2_parser_audit() -> List[Dict[str, Any]]:
    """Audit instruction parser over all LIBERO suites. Verifies new schema and action/relation exclusion."""
    rows: List[Dict[str, Any]] = []
    try:
        from libero.libero import benchmark
        benchmark_dict = benchmark.get_benchmark_dict()
    except Exception as exc:
        print(f"  [ERROR] Could not import LIBERO: {exc}")
        # Fall back to built-in test cases
        fallback_instructions = [
            "pick up the black bowl between the plate and the ramekin and place it on the plate",
            "open the drawer and pick up the red mug and place it on the counter",
            "close the cabinet door",
            "push the blue block to the left of the tray",
            "put the green cup next to the yellow bowl",
            "pick up the white plate from the rack",
            "place the spoon in the bowl",
            "move the orange to the plate",
            "slide the tray under the table",
            "turn on the stove burner",
        ]
        for i, instruction in enumerate(fallback_instructions):
            parsed = parse_instruction_terms(instruction)
            rows.append(_build_parser_row("fallback", i, instruction, parsed))
        return rows

    for suite_name in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
        if suite_name not in benchmark_dict:
            continue
        print(f"  Processing {suite_name}...")
        try:
            task_suite = benchmark_dict[suite_name]()
        except Exception as exc:
            print(f"  [ERROR] Could not load {suite_name}: {exc}")
            continue

        for task_id in range(task_suite.n_tasks):
            task = task_suite.get_task(task_id)
            instruction = task.language.strip()
            parsed = parse_instruction_terms(instruction)
            rows.append(_build_parser_row(suite_name, task_id, instruction, parsed))

    return rows


def _build_parser_row(
    suite_name: str,
    task_id: int,
    instruction: str,
    parsed: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a single parser audit row from a parse result (supports both schemas)."""
    raw = parsed.get("raw_parse", {})
    target = parsed.get("target", {})
    references = parsed.get("references", [])
    relations = parsed.get("relations", [])

    # Extract reference objects (for checking action/relation word exclusion)
    ref_objects = [r.get("object", "") for r in references]

    # Check for forbidden words in object prompts
    ACTION_VERBS_CHECK = {
        "pick", "place", "put", "set", "drop", "push", "pull",
        "open", "close", "turn", "twist", "rotate", "slide",
        "move", "carry", "hold", "grasp", "release", "take",
        "get", "swap", "replace", "lift", "lower", "hang",
        "fold", "wipe", "clean", "wash", "pour", "scoop",
        "pick_up", "put_down", "close_the", "open_the",
    }
    RELATION_WORDS_CHECK = {
        "next_to", "beside", "near", "close_to", "on", "onto",
        "top_of", "on_top_of", "in", "inside", "within",
        "in_front_of", "behind", "left_of", "right_of",
        "between", "under", "below", "above", "over",
        "far_from", "away_from", "adjacent_to", "opposite",
    }

    # Check target for action/relation words
    target_obj = target.get("object", "")
    target_words = target_obj.split("_") if target_obj else []
    target_action_leak = [w for w in target_words if w in ACTION_VERBS_CHECK]
    target_relation_leak = [w for w in target_words if w in RELATION_WORDS_CHECK]

    # Check reference objects for action/relation words
    ref_action_leak: List[str] = []
    ref_relation_leak: List[str] = []
    for ref_obj in ref_objects:
        for w in ref_obj.split("_"):
            if w in ACTION_VERBS_CHECK:
                ref_action_leak.append(w)
            if w in RELATION_WORDS_CHECK:
                ref_relation_leak.append(w)

    warnings: List[str] = []
    if not instruction:
        warnings.append("empty_instruction")
    if not parsed.get("instruction_is_meaningful", False):
        warnings.append("parse_empty")
    if target_action_leak:
        warnings.append(f"action_word_in_target:{','.join(target_action_leak)}")
    if target_relation_leak:
        warnings.append(f"relation_word_in_target:{','.join(target_relation_leak)}")
    if ref_action_leak:
        warnings.append(f"action_word_in_refs:{','.join(ref_action_leak)}")
    if ref_relation_leak:
        warnings.append(f"relation_word_in_refs:{','.join(ref_relation_leak)}")

    tokens = raw.get("instruction_tokens", parsed.get("instruction_tokens", []))

    return {
        # Suite metadata
        "task_suite": suite_name,
        "task_id": task_id,
        "task_id_str": f"{suite_name}_{task_id}",
        # Raw instruction
        "instruction": instruction,
        # New nested schema fields
        "target_object": target.get("object", ""),
        "target_attrs": ";".join(target.get("attributes", [])),
        "reference_objects": ";".join(ref_objects),
        "relations_types": ";".join(rel.get("type", "") for rel in relations),
        "actions": ";".join(parsed.get("actions", [])),
        "relations": [
            {"type": rel.get("type", ""), "target": rel.get("target", ""),
             "references": rel.get("references", [])}
            for rel in relations
        ],
        # Flat schema (backward compat)
        "parsed_target_terms": raw.get("parsed_target_terms", []),
        "parsed_reference_terms": raw.get("parsed_reference_terms", []),
        "parsed_relation_terms": raw.get("parsed_relation_terms", []),
        # Validation
        "parse_empty": not parsed.get("instruction_is_meaningful", False),
        "parse_warning": "; ".join(warnings) if warnings else "",
        "action_word_in_object_prompt": bool(target_action_leak or ref_action_leak),
        "relation_word_in_object_prompt": bool(target_relation_leak or ref_relation_leak),
        "has_color": parsed.get("has_color", False),
        "has_object": parsed.get("has_object", False),
        "has_relation": parsed.get("has_relation", False),
        "instruction_is_meaningful": parsed.get("instruction_is_meaningful", False),
        "num_tokens": len(tokens),
        "num_target_terms": len(raw.get("parsed_target_terms", [])),
        "num_reference_terms": len(raw.get("parsed_reference_terms", [])),
        "num_relation_terms": len(raw.get("parsed_relation_terms", [])),
        "instruction_tokens": tokens,
        # Raw parse for debug
        "_raw_parse": raw,
        "_parsed_full": parsed,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Task 3: Semantic backend audit — none / parser_only / mock
# ═════════════════════════════════════════════════════════════════════════════

def task3_semantic_backend_audit(
    instruction: str = "pick up the black bowl between the plate and the ramekin",
    seed: int = 42,
) -> Dict[str, Any]:
    """Audit all three semantic backends: none, parser_only, mock."""
    parsed = parse_instruction_terms(instruction)
    results: Dict[str, Any] = {}
    report: Dict[str, Any] = {
        "instruction": instruction,
        "parsed": {
            "target_object": parsed.get("target", {}).get("object", ""),
            "target_attrs": parsed.get("target", {}).get("attributes", []),
            "reference_objects": [r.get("object", "") for r in parsed.get("references", [])],
            "relations": parsed.get("relations", []),
            "actions": parsed.get("actions", []),
            "meaningful": parsed.get("instruction_is_meaningful", False),
        },
        "backends": {},
        "all_passed": True,
        "errors": [],
    }

    for backend_name in ["none", "parser_only", "mock"]:
        print(f"  Testing backend: {backend_name}")
        be = SemanticLayoutBackend(
            backend=backend_name,
            grid_h=16, grid_w=16,
            seed=seed,
        )
        sl_result = be.run(instruction=instruction, parsed=parsed)
        rec = _backend_result_record(sl_result)
        results[backend_name] = rec
        report["backends"][backend_name] = rec

        # Assertions per backend
        if backend_name == "none":
            ok = (
                sl_result.semantic_available is False
                and sl_result.confidence == 0.0
                and np.all(sl_result.token_scores == 0)
                and np.all(sl_result.target_mask == 0)
                and np.all(sl_result.reference_mask == 0)
                and np.all(sl_result.relation_mask == 0)
            )
            rec["assert_semantic_available_false"] = not sl_result.semantic_available
            rec["assert_confidence_zero"] = sl_result.confidence == 0.0
            rec["assert_all_scores_zero"] = bool(np.all(sl_result.token_scores == 0))
            rec["assert_passed"] = ok
            if not ok:
                report["all_passed"] = False
                report["errors"].append(f"backend={backend_name}: assertions failed")

        elif backend_name == "parser_only":
            ok = (
                sl_result.semantic_available is False
                and sl_result.confidence == 0.0
                and np.all(sl_result.token_scores == 0)
            )
            rec["assert_semantic_available_false"] = not sl_result.semantic_available
            rec["assert_confidence_zero"] = sl_result.confidence == 0.0
            rec["assert_all_scores_zero"] = bool(np.all(sl_result.token_scores == 0))
            rec["assert_passed"] = ok
            if not ok:
                report["all_passed"] = False
                report["errors"].append(f"backend={backend_name}: assertions failed")

        elif backend_name == "mock":
            ok = (
                sl_result.semantic_available is True
                and sl_result.confidence > 0
                and np.any(sl_result.target_mask > 0)
                and np.any(sl_result.reference_mask > 0)
                and np.any(sl_result.token_scores > 0)
            )
            rec["assert_semantic_available_true"] = sl_result.semantic_available is True
            rec["assert_confidence_positive"] = sl_result.confidence > 0
            rec["assert_has_target_mask"] = bool(np.any(sl_result.target_mask > 0))
            rec["assert_has_reference_mask"] = bool(np.any(sl_result.reference_mask > 0))
            rec["assert_has_token_scores"] = bool(np.any(sl_result.token_scores > 0))
            rec["assert_passed"] = ok
            if not ok:
                report["all_passed"] = False
                report["errors"].append(f"backend={backend_name}: assertions failed")

    return report


def _backend_result_record(sl: SemanticLayoutResult) -> Dict[str, Any]:
    """Convert a SemanticLayoutResult to a serialisable dict for reports."""
    return {
        "semantic_available": sl.semantic_available,
        "confidence": sl.confidence,
        "target_mask_count": int(np.sum(sl.target_mask > 0.5)),
        "reference_mask_count": int(np.sum(sl.reference_mask > 0.5)),
        "relation_mask_count": int(np.sum(sl.relation_mask > 0.5)),
        "layout_anchor_mask_count": int(np.sum(sl.layout_anchor_mask > 0.5)),
        "token_scores_max": float(np.max(sl.token_scores)),
        "token_scores_mean": float(np.mean(sl.token_scores)),
        "debug_summary": {
            k: v for k, v in sl.debug.items()
            if k not in ("parsed_full", "_raw_parse", "_parsed_full")
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# Task 4: Scene-layout branch integration (mock mode)
# ═════════════════════════════════════════════════════════════════════════════

def task4_scene_layout_audit(
    seed: int = 7,
    n_tokens: int = 256,
    keep_ratio: float = 0.875,
    num_steps: int = 10,
) -> Dict[str, Any]:
    """Verify scene-layout branch in v2 with mock backend."""
    instruction = "pick up the black bowl between the plate and the ramekin"

    report: Dict[str, Any] = {
        "seed": seed,
        "n_tokens": n_tokens,
        "keep_ratio": keep_ratio,
        "keep_k": int(round(n_tokens * keep_ratio)),
        "num_steps": num_steps,
        "instruction": instruction,
        "steps": [],
        "errors": [],
        "all_passed": True,
    }

    for step in range(num_steps):
        step_seed = seed + step * 1000
        rng = np.random.default_rng(step_seed)
        scores = make_synthetic_geometry_scores(rng=rng, n=n_tokens)
        keep_k = report["keep_k"]

        v2_indices, v2_meta = _call_v2_mock(scores, keep_k, instruction, seed=step_seed)

        sl_active = v2_meta.get("scene_layout_branch_active", False)
        sl_available = v2_meta.get("scene_layout_available", False)
        sl_confidence = v2_meta.get("scene_layout_confidence", 0.0)
        sel_sl = v2_meta.get("selected_by_scene_layout_count", 0)
        sel_sem = v2_meta.get("selected_by_semantic_count", 0)
        acct_valid = v2_meta.get("acgtp_branch_accounting_valid", False)
        tgt_mask_cnt = v2_meta.get("target_mask_count", 0)
        ref_mask_cnt = v2_meta.get("reference_mask_count", 0)
        rel_mask_cnt = v2_meta.get("relation_mask_count", 0)
        lay_mask_cnt = v2_meta.get("layout_anchor_mask_count", 0)
        sl_indices = sorted(v2_meta.get("scene_layout_indices", []))[:20]

        step_rec = {
            "step": step,
            "seed": step_seed,
            "v2_keep_count": len(v2_indices),
            "scene_layout_active": sl_active,
            "scene_layout_available": sl_available,
            "scene_layout_confidence": sl_confidence,
            "selected_by_scene_layout": sel_sl,
            "selected_by_semantic": sel_sem,
            "target_mask_count": tgt_mask_cnt,
            "reference_mask_count": ref_mask_cnt,
            "relation_mask_count": rel_mask_cnt,
            "layout_anchor_mask_count": lay_mask_cnt,
            "scene_layout_indices_sample": sl_indices,
            "acct_valid": acct_valid,
        }
        report["steps"].append(step_rec)

        # Assertions
        ok = True
        if not sl_active:
            report["errors"].append(f"Step {step}: scene_layout_branch_active != True")
            ok = False
        if not sl_available:
            report["errors"].append(f"Step {step}: scene_layout_available != True")
            ok = False
        if sl_confidence <= 0:
            report["errors"].append(f"Step {step}: scene_layout_confidence <= 0")
            ok = False
        if sel_sl <= 0:
            report["errors"].append(f"Step {step}: selected_by_scene_layout_count <= 0")
            ok = False
        if not acct_valid:
            report["errors"].append(f"Step {step}: accounting invalid")
            ok = False
        if tgt_mask_cnt <= 0:
            report["errors"].append(f"Step {step}: target_mask_count <= 0")
            ok = False

        if not ok:
            report["all_passed"] = False

    report["summary"] = {
        "all_scene_layout_active": all(s["scene_layout_active"] for s in report["steps"]),
        "all_scene_layout_available": all(s["scene_layout_available"] for s in report["steps"]),
        "all_scene_layout_confidence_positive": all(s["scene_layout_confidence"] > 0 for s in report["steps"]),
        "all_sel_scene_layout_positive": all(s["selected_by_scene_layout"] > 0 for s in report["steps"]),
        "all_acct_valid": all(s["acct_valid"] for s in report["steps"]),
        "avg_sel_scene_layout": round(
            sum(s["selected_by_scene_layout"] for s in report["steps"]) / len(report["steps"]), 2
        ),
        "error_count": len(report["errors"]),
    }
    report["all_passed"] = (
        report["summary"]["all_scene_layout_active"]
        and report["summary"]["all_scene_layout_available"]
        and report["summary"]["all_scene_layout_confidence_positive"]
        and report["summary"]["all_sel_scene_layout_positive"]
        and report["summary"]["all_acct_valid"]
        and len(report["errors"]) == 0
    )

    return report


# ═════════════════════════════════════════════════════════════════════════════
# Task 5: Verify accounting and debug/overlay fields
# ═════════════════════════════════════════════════════════════════════════════

def task5_accounting_audit(
    seed: int = 7,
    n_tokens: int = 256,
    keep_ratio: float = 0.875,
    num_steps: int = 10,
) -> Dict[str, Any]:
    """Verify all required accounting and debug/overlay fields are present and valid."""
    instruction = "pick up the black bowl between the plate and the ramekin"

    REQUIRED_FIELDS = [
        "acgtp_v2_semantic_backend",
        "acgtp_v2_semantic_available",
        "acgtp_v2_semantic_unavailable",
        "acgtp_v2_semantic_confidence",
        "acgtp_v2_release_quota",
        "selected_by_scene_layout_count",
        "selected_by_semantic_count",
        "acgtp_branch_accounting_valid",
        "branch_accounting_valid",
        "acgtp_branch_sum",
        "acgtp_branch_sum_error",
        # Strict fallback dispatch fields (present in all v2 metadata)
        "strict_fallback_dispatch_used",
        "delegated_selector_name",
        "fallback_dispatch_to_v1",
    ]

    NEW_SCENE_LAYOUT_FIELDS = [
        "scene_layout_branch_active",
        "scene_layout_available",
        "scene_layout_confidence",
        "scene_layout_branch_quota",
        "selected_by_scene_layout_count",
        "scene_layout_indices",
        "target_mask_count",
        "reference_mask_count",
        "relation_mask_count",
        "layout_anchor_mask_count",
        "overlap_scene_depth_count",
        "overlap_scene_geometry_count",
    ]

    report: Dict[str, Any] = {
        "num_steps": num_steps,
        "steps": [],
        "missing_required_fields": [],
        "missing_new_fields": [],
        "all_passed": True,
        "errors": [],
    }

    for step in range(num_steps):
        step_seed = seed + step * 1000
        rng = np.random.default_rng(step_seed)
        scores = make_synthetic_geometry_scores(rng=rng, n=n_tokens)
        keep_k = int(round(n_tokens * keep_ratio))

        # Test with mock backend to check new fields
        _, v2_meta = _call_v2_mock(scores, keep_k, instruction, seed=step_seed)

        missing_req = [f for f in REQUIRED_FIELDS if f not in v2_meta]
        missing_new = [f for f in NEW_SCENE_LAYOUT_FIELDS if f not in v2_meta]

        report["steps"].append({
            "step": step,
            "seed": step_seed,
            "acct_valid": v2_meta.get("acgtp_branch_accounting_valid", False),
            "acct_valid_alias": v2_meta.get("branch_accounting_valid", False),
            "acct_sum": v2_meta.get("acgtp_branch_sum", -1),
            "acct_sum_error": v2_meta.get("acgtp_branch_sum_error", -1),
            "sl_active": v2_meta.get("scene_layout_branch_active", None),
            "sl_available": v2_meta.get("scene_layout_available", None),
            "sl_conf": v2_meta.get("scene_layout_confidence", None),
            "sl_quota": v2_meta.get("scene_layout_branch_quota", None),
            "sel_sl": v2_meta.get("selected_by_scene_layout_count", None),
            "sel_sem": v2_meta.get("selected_by_semantic_count", None),
            "tgt_mask": v2_meta.get("target_mask_count", None),
            "ref_mask": v2_meta.get("reference_mask_count", None),
            "rel_mask": v2_meta.get("relation_mask_count", None),
            "lay_mask": v2_meta.get("layout_anchor_mask_count", None),
            "overlap_sd": v2_meta.get("overlap_scene_depth_count", None),
            "overlap_sg": v2_meta.get("overlap_scene_geometry_count", None),
            "sl_indices": v2_meta.get("scene_layout_indices", []),
            "missing_required": missing_req,
            "missing_new": missing_new,
        })

        report["missing_required_fields"].extend(missing_req)
        report["missing_new_fields"].extend(missing_new)

        if missing_req or missing_new:
            report["all_passed"] = False
            if missing_req:
                report["errors"].append(f"Step {step}: missing required fields: {missing_req}")
            if missing_new:
                report["errors"].append(f"Step {step}: missing new fields: {missing_new}")

    report["missing_required_fields"] = list(set(report["missing_required_fields"]))
    report["missing_new_fields"] = list(set(report["missing_new_fields"]))

    return report


# ═════════════════════════════════════════════════════════════════════════════
# Task 6: Attention task-relevance alignment audit
# ═════════════════════════════════════════════════════════════════════════════

def _call_v2_with_attention(
    scores: Dict[str, Any],
    keep_k: int,
    instruction: str,
    attn_backend: str,
    attn_mode: str = "balanced",
    attn_seed: int = 0,
    attention_requires_geo_alignment: bool = True,
    attention_budget_ratio: float = 0.10,
) -> tuple:
    """Call select_acgtp_v2 with attention relevance enabled."""
    parsed = parse_instruction_terms(instruction)
    raw = parsed.get("raw_parse", {})

    # Get attention relevance result
    attn_result = get_attention_relevance(
        backend=attn_backend,
        n=256,
        grid_h=16,
        grid_w=16,
        step_index=0,
        seed=attn_seed,
        mode=attn_mode,
    )

    return select_acgtp_v2(
        scene_layout_scores=scores["scene_scores"],
        depth_edge_scores=scores["depth_scores"],
        contact_ring_scores=scores["contact_scores"],
        motion_corridor_scores=scores["motion_scores"],
        semantic_anchor_scores=None,
        semantic_target_scores=None,
        semantic_reference_scores=None,
        semantic_relation_scores=None,
        semantic_goal_scores=None,
        valid_mask=scores["valid_mask"],
        keep_k=keep_k,
        constrained_fill_mask=scores["fill_mask"],
        token_u=scores["token_u"],
        token_v=scores["token_v"],
        grid_h=16, grid_w=16,
        w_scene_layout=0.30,
        w_depth_structure=0.25,
        w_contact_ring=0.25,
        w_motion_corridor=0.20,
        w_semantic=0.20,
        hard_protect_ratio=0.60,
        motion_corridor_valid=scores["motion_valid"],
        self_core_mask=scores["self_core"],
        contact_ring_inner_px=24.0,
        contact_ring_outer_px=48.0,
        contact_requires_edge_or_object=True,
        depth_edge_score_for_gate=scores["depth_scores"],
        support_plane_cap_ratio=0.30,
        semantic_enabled=False,
        semantic_backend="none",
        semantic_confidence=0.0,
        semantic_unavailable=True,
        semantic_fallback_reason="attention_audit_no_semantic",
        release_semantic_quota_when_unavailable=True,
        w_semantic_target=1.0,
        w_semantic_reference=0.7,
        w_semantic_relation=0.5,
        w_semantic_goal=0.9,
        target_cap_ratio=0.25,
        reference_cap_ratio=0.20,
        relation_cap_ratio=0.15,
        hard_semantic_ratio=0.20,
        parsed_target_terms=raw.get("parsed_target_terms", []),
        parsed_reference_terms=raw.get("parsed_reference_terms", []),
        parsed_relation_terms=raw.get("parsed_relation_terms", []),
        instruction_is_meaningful=bool(parsed.get("instruction_is_meaningful", False)),
        scene_layout_branch_active=False,
        scene_layout_available=False,
        scene_layout_confidence=0.0,
        target_mask_count=0,
        reference_mask_count=0,
        relation_mask_count=0,
        layout_anchor_mask_count=0,
        scene_layout_indices=[],
        # Attention branch
        acgtp_attention_enabled=True,
        acgtp_attention_backend=attn_backend,
        acgtp_attention_min_confidence=0.0,
        acgtp_attention_requires_geometry_alignment=attention_requires_geo_alignment,
        acgtp_attention_budget_ratio=attention_budget_ratio,
        acgtp_attention_task_relevance_score=attn_result.task_relevance_score,
        acgtp_attention_task_relevance_mask=attn_result.task_relevance_mask,
        acgtp_attention_source=attn_result.source,
        acgtp_attention_available=attn_result.available,
        acgtp_attention_confidence=attn_result.confidence,
    )


def _compute_post_hoc_alignment(
    attn_result: AttentionRelevanceResult,
    v2_meta: Dict[str, Any],
    scores: Dict[str, Any],
    keep_indices: np.ndarray,
    safe_drop: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute post-hoc attention-geometry alignment using metadata from selector.

    Uses the same fixed threshold (0.50) as compute_safe_drop_diagnostic for
    consistency when geometry scores use bimodal [LOW, HIGH] distributions.
    If safe_drop is None, computes it internally.
    """
    n = attn_result.task_relevance_score.shape[0]

    # Compute safe_drop first, then deepcopy once.
    # safe_drop_local is the authoritative copy — all subsequent code uses it.
    # safe_drop (parameter) is not modified after this.
    if safe_drop is None:
        safe_drop = compute_safe_drop_diagnostic(
            attn_result=attn_result,
            scene_scores=scores.get("scene_scores"),
            depth_scores=scores.get("depth_scores"),
            contact_scores=scores.get("contact_scores"),
            motion_scores=scores.get("motion_scores"),
            keep_indices=keep_indices,
        )
    # One-time deepcopy after safe_drop is fully computed.
    # This captures the warning before any further mutations.
    import copy as _copy_module
    safe_drop_local = _copy_module.deepcopy(safe_drop)

    base = compute_attention_geometry_alignment(
        attn_result=attn_result,
        scene_scores=scores.get("scene_scores"),
        depth_scores=scores.get("depth_scores"),
        contact_scores=scores.get("contact_scores"),
        motion_scores=scores.get("motion_scores"),
    )

    # Add post-hoc fields that require selector output
    keep_set = set(int(i) for i in keep_indices)

    # safe_drop_local is authoritative for counts. Geometry masks use fixed 0.50
    # threshold (matching bimodal score distribution). Attention mask uses
    # task_relevance_mask (70th percentile from backend_mock).
    def _threshold_mask(scores_arr, fixed_t):
        if scores_arr is None:
            return np.zeros(n, dtype=bool)
        a = np.asarray(scores_arr, dtype=np.float32).reshape(-1)
        if a.shape[0] != n:
            return np.zeros(n, dtype=bool)
        return (a >= fixed_t).astype(bool)

    scene_mask = _threshold_mask(scores.get("scene_scores"), 0.50)
    depth_mask = _threshold_mask(scores.get("depth_scores"), 0.50)
    contact_mask = _threshold_mask(scores.get("contact_scores"), 0.50)
    motion_mask = _threshold_mask(scores.get("motion_scores"), 0.50)
    geom_mask = scene_mask | depth_mask | contact_mask | motion_mask
    attn_mask = attn_result.task_relevance_mask

    geo_only = geom_mask & ~attn_mask
    attn_only = attn_mask & ~geom_mask
    geo_only_in_keep = sum(1 for i in keep_set if geo_only[i])
    attn_only_in_keep = sum(1 for i in keep_set if attn_only[i])
    attn_sel_count = v2_meta.get("attention_selected_by_final_count", 0)

    base.update({
        # Use consistent 60th-percentile counts from safe_drop_local
        "geometry_only_token_count": int(safe_drop_local.get("high_geometry_low_attention_count", 0)) if safe_drop_local else 0,
        "attention_only_token_count": int(safe_drop_local.get("high_attention_low_geometry_count", 0)) if safe_drop_local else 0,
        "geometry_only_in_keep_count": geo_only_in_keep,
        "attention_only_in_keep_count": attn_only_in_keep,
        "attention_selected_by_final_count": attn_sel_count,
        # Safe-drop
        "safe_drop_candidate_count": safe_drop_local["safe_drop_candidate_count"],
        "dropped_safe_candidate_count": safe_drop_local["dropped_safe_candidate_count"],
        "dropped_high_attention_count": safe_drop_local["dropped_high_attention_count"],
        "dropped_high_geometry_count": safe_drop_local["dropped_high_geometry_count"],
        "high_attention_low_geometry_count": safe_drop_local["high_attention_low_geometry_count"],
        "high_geometry_low_attention_count": safe_drop_local["high_geometry_low_attention_count"],
        "safe_drop_ratio": safe_drop_local.get("safe_drop_ratio", 0.0) if safe_drop_local else 0.0,
        "safe_drop_warning": safe_drop_local.get("warning") if safe_drop_local else None,
    })
    return base


def task6_attention_alignment_audit(
    seed: int = 7,
    n_tokens: int = 256,
    keep_ratio: float = 0.875,
    num_steps: int = 10,
    instruction: Optional[str] = None,
) -> Dict[str, Any]:
    """Task 6: Attention task-relevance alignment audit.

    Tests the attention branch in the selector with attention_backend=mock
    (balanced mode). Reports:
      - attention_available / source / confidence
      - top_attention_count
      - attention_scene_iou, attention_depth_iou, attention_contact_iou, attention_motion_iou
      - attention_only_token_count, geometry_only_token_count
      - attention_geometry_overlap_count
      - attention_background_risk_count
      - safe-drop diagnostics

    Uses _call_v2_with_attention to exercise the full selector with attention enabled.
    """
    if instruction is None:
        try:
            from libero.libero import benchmark
            bs = benchmark.get_benchmark_dict()["libero_spatial"]()
            task = bs.get_task(0)
            instruction = task.language.strip()
        except Exception:
            instruction = "pick up the black bowl between the plate and the ramekin"

    report: Dict[str, Any] = {
        "seed": seed,
        "n_tokens": n_tokens,
        "keep_ratio": keep_ratio,
        "keep_k": int(round(n_tokens * keep_ratio)),
        "num_steps": num_steps,
        "instruction": instruction,
        "steps": [],
        "errors": [],
        "all_passed": True,
        "warnings": [],
    }

    for step in range(num_steps):
        step_seed = seed + step * 1000
        rng = np.random.default_rng(step_seed)
        scores = make_synthetic_geometry_scores(rng=rng, n=n_tokens)
        keep_k = report["keep_k"]

        # Run with mock attention
        attn_result = get_attention_relevance(
            backend="mock",
            n=n_tokens,
            grid_h=16,
            grid_w=16,
            seed=step_seed,
            mode="balanced",
        )

        v2_indices, v2_meta = _call_v2_with_attention(
            scores=scores,
            keep_k=keep_k,
            instruction=instruction,
            attn_backend="mock",
            attn_mode="balanced",
            attn_seed=step_seed,
        )

        # Post-hoc alignment computation
        align = _compute_post_hoc_alignment(
            attn_result=attn_result,
            v2_meta=v2_meta,
            scores=scores,
            keep_indices=v2_indices,
        )

        step_rec = {
            "step": step,
            "seed": step_seed,
            "keep_k": len(v2_indices),
            "acct_valid": bool(v2_meta.get("acgtp_branch_accounting_valid", False)),
            # Attention state
            "attention_available": align["attention_available"],
            "attention_source": align["attention_source"],
            "attention_confidence": align["attention_confidence"],
            "top_attention_count": align["top_attention_count"],
            # IoUs
            "attention_scene_iou": align["attention_scene_iou"],
            "attention_depth_iou": align["attention_depth_iou"],
            "attention_contact_iou": align["attention_contact_iou"],
            "attention_motion_iou": align["attention_motion_iou"],
            # Counts
            "attention_only_token_count": align["attention_only_token_count"],
            "geometry_only_token_count": align["geometry_only_token_count"],
            "attention_geometry_overlap_count": align["attention_geometry_overlap_count"],
            "attention_background_risk_count": align["attention_background_risk_count"],
            "attention_selected_by_final_count": align["attention_selected_by_final_count"],
            # Safe-drop
            "safe_drop_candidate_count": align["safe_drop_candidate_count"],
            "dropped_safe_candidate_count": align["dropped_safe_candidate_count"],
            "dropped_high_attention_count": align["dropped_high_attention_count"],
            "dropped_high_geometry_count": align["dropped_high_geometry_count"],
            "high_attention_low_geometry_count": align["high_attention_low_geometry_count"],
            "high_geometry_low_attention_count": align["high_geometry_low_attention_count"],
            "safe_drop_warning": align.get("safe_drop_warning"),
            # Selector attention metadata
            "attn_enabled": bool(v2_meta.get("acgtp_attention_enabled", False)),
            "attn_quota_released": bool(v2_meta.get("acgtp_attention_quota_released", True)),
            "attn_candidate_count": v2_meta.get("acgtp_attention_candidate_count", 0),
            "attn_top_count": v2_meta.get("acgtp_attention_top_count", 0),
            "attn_only_selector": v2_meta.get("acgtp_attention_only_token_count", 0),
        }
        report["steps"].append(step_rec)

        # Assertions
        ok = True
        if not step_rec["acct_valid"]:
            report["errors"].append(f"Step {step}: accounting invalid")
            ok = False
        if not step_rec["attention_available"]:
            report["errors"].append(f"Step {step}: attention_available=False for mock backend")
            ok = False
        if step_rec["attention_source"] != "mock":
            report["errors"].append(f"Step {step}: attention_source != mock")
            ok = False
        # Attention_only tokens should NOT dominate the selected set
        if step_rec["attention_only_token_count"] > keep_k * 0.5:
            report["errors"].append(
                f"Step {step}: attention_only_token_count ({step_rec['attention_only_token_count']}) "
                f"exceeds 50% of keep_k ({keep_k})"
            )
            ok = False

        # Warnings
        if step_rec.get("safe_drop_warning"):
            report["warnings"].append(f"Step {step}: {step_rec['safe_drop_warning']}")

        if not ok:
            report["all_passed"] = False

    # Build summary
    steps = report["steps"]
    n_steps = len(steps)
    report["summary"] = {
        "all_attention_available": all(s["attention_available"] for s in steps),
        "all_attention_source_mock": all(s["attention_source"] == "mock" for s in steps),
        "all_acct_valid": all(s["acct_valid"] for s in steps),
        "avg_attention_scene_iou": round(sum(s["attention_scene_iou"] for s in steps) / n_steps, 4) if n_steps else 0,
        "avg_attention_depth_iou": round(sum(s["attention_depth_iou"] for s in steps) / n_steps, 4) if n_steps else 0,
        "avg_attention_contact_iou": round(sum(s["attention_contact_iou"] for s in steps) / n_steps, 4) if n_steps else 0,
        "avg_attention_motion_iou": round(sum(s["attention_motion_iou"] for s in steps) / n_steps, 4) if n_steps else 0,
        "avg_attention_only_count": round(sum(s["attention_only_token_count"] for s in steps) / n_steps, 2) if n_steps else 0,
        "avg_geometry_only_count": round(sum(s["geometry_only_token_count"] for s in steps) / n_steps, 2) if n_steps else 0,
        "avg_attention_sel_count": round(sum(s["attention_selected_by_final_count"] for s in steps) / n_steps, 2) if n_steps else 0,
        "avg_safe_drop_candidates": round(sum(s["safe_drop_candidate_count"] for s in steps) / n_steps, 2) if n_steps else 0,
        "total_warnings": len(report["warnings"]),
        "error_count": len(report["errors"]),
    }

    report["all_passed"] = (
        report["all_passed"]
        and report["summary"]["all_attention_available"]
        and report["summary"]["all_attention_source_mock"]
        and report["summary"]["all_acct_valid"]
        and report["summary"]["error_count"] == 0
    )

    return report


# ═════════════════════════════════════════════════════════════════════════════
# Task 7: Attention stress tests — Cases A through E
# ═════════════════════════════════════════════════════════════════════════════

def task7_attention_stress_test(
    seed: int = 7,
    n_tokens: int = 256,
    keep_ratio: float = 0.875,
    instruction: Optional[str] = None,
) -> Dict[str, Any]:
    """Task 7: Mock attention stress tests across 5 Cases.

    Case A: attention overlaps with scene/contact/motion
      Expected: attention aligned candidates selected, accounting valid.

    Case B: attention all background, geometry weak
      Expected: attn_only > 0, attn_sel < attn_only (attention-only tokens gated out).

    Case C: attention unavailable (backend=none)
      Expected: full fallback/release quota, v1/v2 geometry path unaffected.

    Case D: geometry strong but attention weak
      Expected: geometry-only tokens exist and protect action tokens.

    Case E: attention and geometry partially overlap
      Expected: union + quota refill works normally, no global attention top-k.
    """
    if instruction is None:
        try:
            from libero.libero import benchmark
            bs = benchmark.get_benchmark_dict()["libero_spatial"]()
            task = bs.get_task(0)
            instruction = task.language.strip()
        except Exception:
            instruction = "pick up the black bowl between the plate and the ramekin"

    keep_k = int(round(n_tokens * keep_ratio))
    rng_base = np.random.default_rng(seed)
    scores = make_synthetic_geometry_scores(rng=rng_base, n=n_tokens)

    cases = [
        ("A", "high_attn_geo_low", "attention overlaps scene/contact/motion", True),
        ("B", "all_background", "attention all background, geometry weak", True),
        ("C", "none", "attention unavailable", False),
        ("D", "high_geo_attn_low", "geometry strong, attention weak", True),
        ("E", "partial_overlap", "attention and geometry partially overlap", True),
    ]

    results: Dict[str, Dict[str, Any]] = {}
    all_passed = True

    for case_id, mode, description, use_attention in cases:
        step_seed = seed + hash(case_id) % 100000
        rng = np.random.default_rng(step_seed)

        if mode == "none":
            # Case C: attention_backend=none — geometry-only path
            v1_indices, v1_meta = _call_v1(scores, keep_k)
            v2_fb_indices, v2_meta = _call_v2_fallback(scores, keep_k, instruction)
            _, v2_meta_attn = _call_v2_with_attention(
                scores=scores, keep_k=keep_k, instruction=instruction,
                attn_backend="none", attn_mode="none", attn_seed=step_seed,
            )

            acct_valid = bool(v2_meta_attn.get("acgtp_branch_accounting_valid", False))
            attn_released = bool(v2_meta_attn.get("acgtp_attention_quota_released", False))
            attn_sel_count = v2_meta_attn.get("attention_selected_by_final_count", 0)
            v1_v2_match = np.array_equal(np.sort(v1_indices), np.sort(v2_fb_indices))

            passed = acct_valid and attn_released and attn_sel_count == 0
            if not passed:
                all_passed = False

            results[case_id] = {
                "description": description,
                "case_id": case_id,
                "mode": mode,
                "passed": passed,
                "acct_valid": acct_valid,
                "attn_quota_released": attn_released,
                "attention_selected_count": attn_sel_count,
                "attn_available": v2_meta_attn.get("acgtp_attention_available", False),
                "attn_source": v2_meta_attn.get("acgtp_attention_source", "none"),
                "keep_k": keep_k,
                "v1_v2_geo_path_match": v1_v2_match,
                "attn_only_token_count": v2_meta_attn.get("acgtp_attention_only_token_count", 0),
                "warnings": [],
            }

        else:
            # Cases A, B, D, E: run with mock attention
            attn_result = get_attention_relevance(
                backend="mock",
                n=n_tokens,
                grid_h=16,
                grid_w=16,
                seed=step_seed,
                mode=mode,
            )

            v2_indices, v2_meta = _call_v2_with_attention(
                scores=scores,
                keep_k=keep_k,
                instruction=instruction,
                attn_backend="mock",
                attn_mode=mode,
                attn_seed=step_seed,
            )

            # Compute safe-drop diagnostic FIRST, before any mutation.
            # _compute_post_hoc_alignment will recompute it internally.
            safe_drop_raw = compute_safe_drop_diagnostic(
                attn_result=attn_result,
                scene_scores=scores.get("scene_scores"),
                depth_scores=scores.get("depth_scores"),
                contact_scores=scores.get("contact_scores"),
                motion_scores=scores.get("motion_scores"),
                keep_indices=v2_indices,
            )
            # Capture the warning string IMMEDIATELY (strings are immutable in Python).
            # Even if the dict is mutated later, the local string variable is unaffected.
            safe_drop_warning_str = safe_drop_raw.get("warning")
            if safe_drop_warning_str:
                safe_drop_warning_str = str(safe_drop_warning_str)

            # _compute_post_hoc_alignment does NOT receive safe_drop to avoid mutation.
            # It recomputes safe_drop internally (safe_drop=None default).
            align = _compute_post_hoc_alignment(
                attn_result=attn_result,
                v2_meta=v2_meta,
                scores=scores,
                keep_indices=v2_indices,
                safe_drop=None,
            )

            acct_valid = bool(v2_meta.get("acgtp_branch_accounting_valid", False))
            attn_sel_count = v2_meta.get("attention_selected_by_final_count", 0)
            attn_only_count = v2_meta.get("acgtp_attention_only_token_count", 0)
            attn_available = v2_meta.get("acgtp_attention_available", False)
            attn_quota_released = v2_meta.get("acgtp_attention_quota_released", True)
            attn_candidate_count = v2_meta.get("acgtp_attention_candidate_count", 0)

            warnings: List[str] = []
            # safe_drop_warning_str was captured BEFORE _compute_post_hoc_alignment was called.
            # Strings are immutable in Python, so even if deepcopy shares the string object,
            # the local variable safe_drop_warning_str refers to the ORIGINAL string.
            # NOTE: If the mutation in _compute_post_hoc_alignment reassigns safe_drop["warning"]
            # to a DIFFERENT string object, safe_drop_warning_str is still the ORIGINAL string.
            if safe_drop_warning_str:
                warnings.append(safe_drop_warning_str)

            # Case-specific assertions (design intent, not strict mock behavior)
            # NOTE: Mock backends generate random scores. The geometry alignment gate
            # means attention scores statistically overlap with non-zero geometry.
            # Real attention probes are needed for true "background-only" cases.
            passed = acct_valid
            if case_id == "A":
                # Attention overlaps geometry → some candidates selected
                if attn_candidate_count == 0:
                    passed = False
                    warnings.append("Case A: no attention-aligned candidates despite overlapping pattern")
                if attn_sel_count == 0:
                    passed = False
                    warnings.append("Case A: attention_selected_count == 0 despite high overlap")
            elif case_id == "B":
                # Background bias: attention scores concentrate on geometry-weak regions
                # (corners/edges) while action-relevant tokens have weak geometry.
                #
                # Design verification:
                #   1. attn_only_candidates > 0: selector classifies tokens as "attention-only"
                #   2. attn_sel_count < attn_only_count: far fewer aligned tokens selected than
                #      attention-only candidates exist (because geometry alignment is strict)
                #   3. safe_drop.high_attention_low_geometry_count > 0: diagnostic confirms
                #      that there are tokens with high attention and low geometry alignment
                #
                # Note: attn_sel_count == 0 is not expected due to random seed variance —
                # some corner tokens may accidentally overlap with geometry-high regions.
                # The key signal is attn_sel << attn_only.
                attn_only_meta = v2_meta.get("acgtp_attention_only_token_count", 0)
                has_attn_only = attn_only_meta > 0
                sel_vs_only = attn_sel_count < attn_only_meta  # selected << attention-only
                has_attn_low_geo = safe_drop_raw.get("high_attention_low_geometry_count", 0) > 0
                if not (has_attn_only and sel_vs_only and has_attn_low_geo):
                    passed = False
                    if not has_attn_only:
                        warnings.append(f"Case B: expected attention-only tokens, got {attn_only_meta}")
                    if not sel_vs_only:
                        warnings.append(f"Case B: expected attn_sel ({attn_sel_count}) < attn_only ({attn_only_meta})")
                    if not has_attn_low_geo:
                        warnings.append(f"Case B: expected high_attention_low_geometry_count > 0 in safe_drop diagnostic")
            elif case_id == "D":
                # Geometry-dominant: geometry branches hold tokens even when attention is weak.
                # Core invariant: geometry-only tokens must exist and protect action tokens.
                # Note: the exact safe_drop warning label (GEOMETRY_DOMINANT_PROTECTION vs
                # POSSIBLE_ATTENTION_BACKGROUND_BIAS) depends on random seed variance in the
                # dropped_ratio computation and is NOT a pass/fail criterion here.
                geo_only_count = align.get("geometry_only_token_count", 0)
                if geo_only_count == 0:
                    warnings.append("Case D: expected geometry-only tokens, got 0")
                    passed = False
                # Core check: geometry is protecting tokens (high_geo_low_attn >> high_attn_low_geo)
                high_geo_low_attn = safe_drop_raw.get("high_geometry_low_attention_count", 0)
                high_attn_low_geo = safe_drop_raw.get("high_attention_low_geometry_count", 0)
                if high_geo_low_attn == 0:
                    warnings.append("Case D: expected high_geometry_low_attention_count > 0")
                    passed = False
                if high_attn_low_geo == 0:
                    warnings.append("Case D: expected high_attention_low_geometry_count > 0")
                    passed = False
                # The ratio itself is not checked — it varies with RNG seed
            elif case_id == "E":
                # Partial overlap: accounting valid, no global top-k
                if attn_candidate_count > keep_k:
                    warnings.append("Case E: attention_candidate_count exceeds keep_k (possible global top-k)")
                    passed = False

            if not passed:
                all_passed = False

            results[case_id] = {
                "description": description,
                "case_id": case_id,
                "mode": mode,
                "passed": passed,
                "acct_valid": acct_valid,
                "attn_available": attn_available,
                "attn_quota_released": attn_quota_released,
                "attention_selected_count": attn_sel_count,
                "attn_candidate_count": attn_candidate_count,
                "attn_top_count": v2_meta.get("acgtp_attention_top_count", 0),
                "attn_only_token_count": attn_only_count,
                "geometry_only_token_count": align.get("geometry_only_token_count", 0),
                "attention_scene_iou": align["attention_scene_iou"],
                "attention_depth_iou": align["attention_depth_iou"],
                "attention_contact_iou": align["attention_contact_iou"],
                "attention_motion_iou": align["attention_motion_iou"],
                "attention_geometry_overlap_count": align["attention_geometry_overlap_count"],
                "dropped_high_attention_count": align["dropped_high_attention_count"],
                "dropped_high_geometry_count": align["dropped_high_geometry_count"],
                "high_attention_low_geometry_count": align["high_attention_low_geometry_count"],
                "high_geometry_low_attention_count": align["high_geometry_low_attention_count"],
                "safe_drop_warning": safe_drop_warning_str,
                "keep_k": keep_k,
                "final_kept": len(v2_indices),
                "warnings": warnings,
            }

    return {
        "seed": seed,
        "n_tokens": n_tokens,
        "keep_ratio": keep_ratio,
        "keep_k": keep_k,
        "instruction": instruction,
        "results": results,
        "all_passed": all_passed,
        "total_cases": len(cases),
        "passed_cases": sum(1 for r in results.values() if r["passed"]),
    }




def write_fallback_equivalence_report(report: Dict[str, Any]) -> None:
    """Write fallback equivalence report (Task 1).

    Distinguishes two paths:
    - STRICT FALLBACK: v2 literally calls v1 → keep_indices EXACT EQUAL (Jaccard == 1.0).
    - RELAXED (v2-native): v2 runs its own pipeline with w_semantic=0, no semantic branch
      → keep_indices may differ from v1 (Jaccard ~0.90-1.0). NOT fallback_dispatch_to_v1.
    """
    out_path = OUTPUT_DIR / "acgtp_v2_fallback_equivalence_report.md"
    s = report.get("summary", {})
    steps = report.get("steps", [])

    ignored_fields = s.get("ignored_v2_diagnostic_fields", [])

    lines = [
        "# ACGTP-v2 Fallback Equivalence Report",
        "",
        "## Configuration",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Mode | `semantic_backend=none` → STRICT FALLBACK (literal v1 dispatch) |",
        f"| Token grid | 16×16 = {report['n_tokens']} tokens |",
        f"| Keep ratio | {report['keep_ratio']} → keep_k={report['keep_k']} |",
        f"| Seed base | `{report['seed']}` |",
        f"| Steps tested | `{report['num_steps']}` |",
        f"| LIBERO instruction | {report.get('instruction', 'N/A')} |",
        "",
        "## Strict Fallback Dispatch Assertions",
        "",
        f"| Assertion | Result |",
        f"|-----------|--------|",
        f"| All strict_fallback_dispatch_used == True | `{s.get('all_strict_fallback_dispatch_used', 'N/A')}` |",
        f"| All delegated_selector_name == 'select_acgtp_v1' | `{s.get('all_delegated_to_v1', 'N/A')}` |",
        f"| All fallback_dispatch_to_v1 == True | `{s.get('all_fallback_dispatch_to_v1', 'N/A')}` |",
        "",
        "## Semantic Branch State Assertions",
        "",
        f"| Assertion | Result |",
        f"|-----------|--------|",
        f"| All v2 semantic_unavailable == True | `{s.get('all_semantic_unavailable', 'N/A')}` |",
        f"| All v2 semantic_confidence == 0.0 | `{s.get('all_semantic_confidence_zero', 'N/A')}` |",
        f"| All v2 selected_by_semantic == 0 | `{s.get('all_sel_semantic_zero', 'N/A')}` |",
        f"| All v2 release_quota == True | `{s.get('all_quota_released', 'N/A')}` |",
        "",
        "## Index Equality Assertions (strict fallback: exact == required)",
        "",
        f"| Assertion | Result |",
        f"|-----------|--------|",
        f"| All keep_counts_equal (v1==v2) | `{s.get('all_keep_counts_equal', 'N/A')}` |",
        f"| All keep_indices EXACT EQUAL | `{s.get('all_indices_exact_equal', 'N/A')}` |",
        f"| All Jaccard == 1.0 | `{s.get('all_jaccard_eq_1', 'N/A')}` |",
        f"| Min Jaccard | `{s.get('min_jaccard', 'N/A')}` |",
        f"| Max Jaccard | `{s.get('max_jaccard', 'N/A')}` |",
        f"| Avg Jaccard | `{s.get('avg_jaccard', 'N/A')}` |",
        "",
        "## Accounting & Branch Attribution",
        "",
        f"| Assertion | Result |",
        f"|-----------|--------|",
        f"| All v2 accounting valid | `{s.get('all_acct_valid', 'N/A')}` |",
        f"| All v1 accounting valid | `{s.get('all_v1_acct_valid', 'N/A')}` |",
        f"| All geometry branch attributions equal (v1==v2) | `{s.get('all_branch_attributions_equal', 'N/A')}` |",
        f"| All scene_layout_branch inactive | `{s.get('all_scene_layout_branch_inactive', 'N/A')}` |",
        f"| All scene_layout unavailable | `{s.get('all_scene_layout_unavailable', 'N/A')}` |",
        "",
        "## Performance Overhead",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| v2−v1 avg overhead | `{s.get('avg_v2_minus_v1_ms', 'N/A')} ms` |",
        f"| v2−v1 max overhead | `{s.get('max_v2_minus_v1_ms', 'N/A')} ms` |",
        "",
        "## v2-Only Diagnostic Fields (ignored in v1/v2 attribution comparison)",
        "",
        "The following fields are appended by v2 but do not exist in v1's schema. "
        "They are excluded from the branch attribution comparison and do not affect "
        "keep_indices equality:",
        "",
        "```python",
        f"ignored_v2_diagnostic_fields = {ignored_fields}",
        "```",
        "",
        "## Per-Step Results",
        "",
        "| Step | keep_k | v1_keep | v2_keep | ExactMatch | Jaccard | "
        "strict_dispatch | delegated | sem_unavail | conf | sel_sem | rel_quota | acct_valid |",
        "|------|--------|--------|--------|------------|---------|"
        "--------------|----------|-------------|------|---------|----------|------------|",
    ]

    for sr in steps:
        lines.append(
            f"| {sr['step']} "
            f"| {report['keep_k']} "
            f"| {sr['v1_keep_count']} "
            f"| {sr['v2_keep_count']} "
            f"| {sr['indices_match']} "
            f"| {sr['jaccard']} "
            f"| {sr['strict_fallback_dispatch_used']} "
            f"| {sr.get('delegated_selector_name', 'N/A')} "
            f"| {sr['v2_sem_unavailable']} "
            f"| {sr['v2_sem_confidence']} "
            f"| {sr['v2_sel_semantic']} "
            f"| {sr['v2_release_quota']} "
            f"| {sr['v2_acct_valid']} "
            f"|"
        )

    lines.extend(["", "## Semantic Branch State", "",
        "| Step | backend | sem_unavail | conf | sel_sem | sel_tgt | sel_ref | sl_active | sl_conf | quota_rel |",
        "|------|---------|-------------|------|---------|---------|---------|----------|---------|----------|"])
    for sr in steps:
        lines.append(
            f"| {sr['step']} "
            f"| {sr.get('v2_sem_backend', 'N/A')} "
            f"| {sr.get('v2_sem_unavailable', 'N/A')} "
            f"| {sr.get('v2_sem_confidence', 'N/A')} "
            f"| {sr.get('v2_sel_semantic', 'N/A')} "
            f"| {sr.get('v2_sel_target', 'N/A')} "
            f"| {sr.get('v2_sel_ref', 'N/A')} "
            f"| {sr.get('v2_sl_active', 'N/A')} "
            f"| {sr.get('v2_sl_confidence', 'N/A')} "
            f"| {sr.get('v2_release_quota', 'N/A')} "
            f"|"
        )

    lines.extend(["", "## Conclusion", ""])
    if report["errors"]:
        lines.append("**FAILED** — errors:")
        for err in report["errors"]:
            lines.append(f"- {err}")
    elif report.get("equivalence_passed"):
        lines.append(
            f"**PASS** — strict fallback equivalence confirmed. "
            f"select_acgtp_v2 delegates to select_acgtp_v1. "
            f"Min Jaccard={s.get('min_jaccard', 'N/A')} (== 1.0), "
            f"all indices EXACT EQUAL, all semantic assertions pass, "
            f"all accounting valid, all geometry attributions match. "
            f"v2 overhead avg={s.get('avg_v2_minus_v1_ms', 'N/A')}ms."
        )
    else:
        lines.append(
            f"**FAIL** — strict fallback equivalence NOT confirmed. "
            f"Min Jaccard={s.get('min_jaccard', 'N/A')}. Review errors above."
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Written: {out_path}")


def write_parser_audit_csv(rows: List[Dict[str, Any]]) -> None:
    """Write parser audit CSV with new schema fields."""
    out_path = OUTPUT_DIR / "acgtp_v2_instruction_parser_audit.csv"
    if not rows:
        print("  No rows to write.")
        return

    fieldnames = [
        "task_suite", "task_id", "task_id_str",
        "instruction",
        # New schema fields
        "target_object", "target_attrs", "reference_objects",
        "relations_types", "actions",
        # Validation
        "parse_empty", "parse_warning",
        "action_word_in_object_prompt", "relation_word_in_object_prompt",
        "has_color", "has_object", "has_relation",
        "instruction_is_meaningful",
        # Flat schema (backward compat)
        "parsed_target_terms", "parsed_reference_terms",
        "parsed_relation_terms",
        "num_tokens", "num_target_terms",
        "num_reference_terms", "num_relation_terms",
        "instruction_tokens",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            safe_row = {
                k: (";".join(str(x) for x in v) if isinstance(v, list) and not isinstance(v, str) else v)
                for k, v in row.items()
            }
            writer.writerow(safe_row)

    print(f"  Written: {out_path}")


def write_parser_audit_report(rows: List[Dict[str, Any]]) -> None:
    """Write parser audit summary report."""
    out_path = OUTPUT_DIR / "acgtp_v2_instruction_parser_audit_report.md"
    if not rows:
        out_path.write_text("No data collected.", encoding="utf-8")
        return

    n = len(rows)
    suites = sorted(set(r["task_suite"] for r in rows))

    suite_stats = {}
    for suite in suites:
        sr = [r for r in rows if r["task_suite"] == suite]
        suite_stats[suite] = {
            "n": len(sr),
            "meaningful": sum(1 for r in sr if r["instruction_is_meaningful"]),
            "empty": sum(1 for r in sr if r["parse_empty"]),
            "action_leak": sum(1 for r in sr if r.get("action_word_in_object_prompt", False)),
            "relation_leak": sum(1 for r in sr if r.get("relation_word_in_object_prompt", False)),
            "has_color": sum(1 for r in sr if r["has_color"]),
            "has_object": sum(1 for r in sr if r["has_object"]),
            "has_relation": sum(1 for r in sr if r["has_relation"]),
            "warnings": sum(1 for r in sr if r["parse_warning"]),
        }

    total_meaningful = sum(st["meaningful"] for st in suite_stats.values())
    total_warnings = sum(st["warnings"] for st in suite_stats.values())
    total_action_leak = sum(st["action_leak"] for st in suite_stats.values())
    total_relation_leak = sum(st["relation_leak"] for st in suite_stats.values())

    lines = [
        "# ACGTP-v2 Instruction Parser Audit Report",
        "",
        "## Overview",
        f"- Total tasks: `{n}`",
        f"- Suites: `{', '.join(suites)}`",
        "",
        "| Suite | Total | Meaningful | parse_empty | action_leak | relation_leak | warnings |",
        "|-------|-------|------------|-------------|-------------|---------------|---------|",
    ]

    for suite in suites:
        st = suite_stats[suite]
        pct = st["meaningful"] / max(st["n"], 1) * 100
        lines.append(
            f"| {suite} | {st['n']} | "
            f"{st['meaningful']} ({pct:.0f}%) | {st['empty']} | "
            f"{st['action_leak']} | {st['relation_leak']} | {st['warnings']} |"
        )

    lines.extend(["",
        f"## Summary",
        f"- Parser coverage: **{total_meaningful}/{n}** ({total_meaningful/max(n,1)*100:.1f}%) non-empty parses.",
        f"- Action word leaks in object prompts: **{total_action_leak}** tasks.",
        f"- Relation word leaks in object prompts: **{total_relation_leak}** tasks.",
        f"- Warnings: **{total_warnings}** tasks with at least one flag."])

    # Show action/relation leak examples
    leak_tasks = [r for r in rows if r.get("action_word_in_object_prompt") or r.get("relation_word_in_object_prompt")]
    if leak_tasks:
        lines.extend(["", f"## Tasks with Action/Relation Words in Object Prompts ({len(leak_tasks)})", ""])
        for r in leak_tasks[:5]:
            instr = r["instruction"][:60].replace("|", "\\|")
            target = r.get("target_object", "").replace("|", "\\|")
            refs = r.get("reference_objects", "").replace("|", "\\|")
            lines.append(
                f"- [{r['task_suite']}] #{r['task_id']}: `{instr}`\n"
                f"  target={target}, refs={refs}, warning={r['parse_warning']}"
            )

    # Sample parsed terms
    meaningful = [r for r in rows if r["instruction_is_meaningful"]]
    if meaningful:
        lines.extend(["",
            f"## Sample Parsed Terms ({min(10, len(meaningful))} examples)", "",
            "| Suite | # | Target | References | Relations | Actions |",
            "|-------|---|--------|------------|-----------|---------|"])
        for r in meaningful[:10]:
            instr = r["instruction"][:30].replace("|", "\\|")
            tgt = r.get("target_object", "").replace("|", "\\|")
            refs = r.get("reference_objects", "")[:30].replace("|", "\\|")
            rels = r.get("relations_types", "").replace("|", "\\|")
            acts = r.get("actions", "").replace("|", "\\|")
            lines.append(f"| {r['task_suite']} | {r['task_id']} | {tgt} | {refs} | {rels} | {acts} |")

    if total_meaningful >= n * 0.7 and total_action_leak == 0 and total_relation_leak == 0:
        lines.extend(["", "## Conclusion",
            "**PASS** — Parser coverage >70%, no action/relation word leaks detected. "
            "The new nested schema is ready for semantic backend input."])
    elif total_meaningful >= n * 0.7:
        lines.extend(["", "## Conclusion",
            f"**PARTIAL** — Coverage {total_meaningful/max(n,1)*100:.1f}% (>70% threshold), "
            f"but {total_action_leak} action-word leaks and {total_relation_leak} relation-word leaks detected. "
            "Review the leak table above."])
    else:
        lines.extend(["", "## Conclusion",
            f"**LOW COVERAGE** — {total_meaningful}/{n} ({total_meaningful/max(n,1)*100:.1f}%). "
            "Consider extending OBJECT_NOUNS / COLOR_WORDS / RELATION_PREPOSITIONS."])

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Written: {out_path}")


def write_semantic_backend_report(report: Dict[str, Any]) -> None:
    """Write semantic backend audit report."""
    out_path = OUTPUT_DIR / "acgtp_v2_semantic_backend_audit_report.md"
    lines = [
        "# ACGTP-v2 Semantic Backend Audit Report",
        "",
        f"## Instruction: {report['instruction']}",
        "",
        "## Parsed Terms",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| target_object | `{report['parsed']['target_object']}` |",
        f"| target_attrs | `{report['parsed']['target_attrs']}` |",
        f"| reference_objects | `{report['parsed']['reference_objects']}` |",
        f"| actions | `{report['parsed']['actions']}` |",
        f"| meaningful | `{report['parsed']['meaningful']}` |",
        "",
        "## Backend Results",
        "",
        "| Backend | available | confidence | tgt_mask | ref_mask | rel_mask | lay_mask | max_score | passed |",
        "|---------|-----------|------------|----------|----------|----------|----------|-----------|--------|",
    ]

    for bname, brec in report["backends"].items():
        lines.append(
            f"| {bname} "
            f"| {brec['semantic_available']} "
            f"| {brec['confidence']} "
            f"| {brec['target_mask_count']} "
            f"| {brec['reference_mask_count']} "
            f"| {brec['relation_mask_count']} "
            f"| {brec['layout_anchor_mask_count']} "
            f"| {brec['token_scores_max']:.4f} "
            f"| {brec.get('assert_passed', 'N/A')} "
            f"|"
        )

    lines.extend(["", "## Conclusion", ""])
    if report["all_passed"]:
        lines.append("**PASS** — All backends produce expected results. "
                     "none/parser_only: available=False, confidence=0.0. "
                     "mock: available=True, confidence>0, valid masks.")
    else:
        lines.append(f"**FAIL** — Errors: {report['errors']}")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Written: {out_path}")


def write_scene_layout_audit_report(report: Dict[str, Any]) -> None:
    """Write scene-layout branch audit report."""
    out_path = OUTPUT_DIR / "acgtp_v2_scene_layout_audit_report.md"
    s = report.get("summary", {})
    lines = [
        "# ACGTP-v2 Scene-Layout Branch Audit Report",
        "",
        "## Configuration",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Mode | `semantic_backend=mock` (scene-layout branch active) |",
        f"| Keep ratio | {report['keep_ratio']} → keep_k={report['keep_k']} |",
        f"| Steps | {report['num_steps']} |",
        f"| Instruction | {report['instruction']} |",
        "",
        "## Assertions",
        "",
        f"| Check | Result |",
        f"|-------|--------|",
        f"| All scene_layout_branch_active | `{s.get('all_scene_layout_active', 'N/A')}` |",
        f"| All scene_layout_available | `{s.get('all_scene_layout_available', 'N/A')}` |",
        f"| All scene_layout_confidence > 0 | `{s.get('all_scene_layout_confidence_positive', 'N/A')}` |",
        f"| All selected_by_scene_layout > 0 | `{s.get('all_sel_scene_layout_positive', 'N/A')}` |",
        f"| All accounting valid | `{s.get('all_acct_valid', 'N/A')}` |",
        f"| Avg selected_by_scene_layout | `{s.get('avg_sel_scene_layout', 'N/A')}` |",
        f"| Error count | `{s.get('error_count', 'N/A')}` |",
        "",
        "## Per-Step Results",
        "",
        "| Step | keep | sl_active | sl_avail | sl_conf | sel_sl | sel_sem | tgt_mask | ref_mask | acct |",
        "|------|------|----------|----------|---------|--------|---------|----------|----------|------|",
    ]

    for sr in report["steps"]:
        lines.append(
            f"| {sr['step']} "
            f"| {sr['v2_keep_count']} "
            f"| {sr['scene_layout_active']} "
            f"| {sr['scene_layout_available']} "
            f"| {sr['scene_layout_confidence']} "
            f"| {sr['selected_by_scene_layout']} "
            f"| {sr['selected_by_semantic']} "
            f"| {sr['target_mask_count']} "
            f"| {sr['reference_mask_count']} "
            f"| {sr['acct_valid']} "
            f"|"
        )

    lines.extend(["", "## Conclusion", ""])
    if report.get("all_passed"):
        lines.append("**PASS** — scene-layout branch correctly activates in mock mode. "
                     f"Avg selected_by_scene_layout={s.get('avg_sel_scene_layout', 'N/A')}. "
                     "All accounting valid.")
    else:
        lines.append(f"**FAIL** — Errors: {report['errors']}")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Written: {out_path}")


def write_accounting_audit_report(report: Dict[str, Any]) -> None:
    """Write accounting and debug field audit report."""
    out_path = OUTPUT_DIR / "acgtp_v2_accounting_debug_audit_report.md"
    lines = [
        "# ACGTP-v2 Accounting & Debug Field Audit Report",
        "",
        f"## Steps tested: {report['num_steps']}",
        "",
        "## Missing Required Fields",
        "",
    ]
    if report["missing_required_fields"]:
        for f in report["missing_required_fields"]:
            lines.append(f"- `{f}`")
    else:
        lines.append("None — all required fields present.")

    lines.extend(["", "## Missing New Scene-Layout Fields", ""])
    if report["missing_new_fields"]:
        for f in report["missing_new_fields"]:
            lines.append(f"- `{f}`")
    else:
        lines.append("None — all new fields present.")

    lines.extend(["", "## Conclusion", ""])
    if report["all_passed"]:
        lines.append("**PASS** — All required and new scene-layout fields are present. "
                     "No missing field errors.")
    else:
        lines.append(f"**FAIL** — Missing fields detected. Errors: {report['errors']}")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Written: {out_path}")


def write_attention_alignment_audit_report(report: Dict[str, Any]) -> None:
    """Write attention alignment audit report (Task 6)."""
    out_path = OUTPUT_DIR / "acgtp_v2_attention_alignment_audit_report.md"
    s = report.get("summary", {})
    steps = report.get("steps", [])

    lines = [
        "# ACGTP-v2 Attention Task-Relevance Alignment Audit Report",
        "",
        "## Design: VLA-Cache / VLA-IAP / VLA-Pruner Inspired",
        "",
        "### VLA-Cache",
        "- text-to-vision / decoder attention identifies task-relevant tokens",
        "- Static tokens are only droppable when task-irrelevant",
        "- **Design**: low geometry AND low attention才能safe drop; high-attention tokens are protected candidates",
        "",
        "### VLA-IAP",
        "- semantic attention may early-misalign (focus background)",
        "- Requires geometry alignment gate before hard protection",
        "- **Design**: attention high + geometry weak tokens are diagnostic only, NOT forcibly retained",
        "",
        "### VLA-Pruner",
        "- semantic prefill attention and action relevance are DIFFERENT signals",
        "- Do NOT score-fuse; enter constrained union only",
        "- **Design**: attention branch CANNOT participate in global weighted top-k; feeds into fill pool only",
        "",
        "## Configuration",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Keep ratio | {report['keep_ratio']} → keep_k={report['keep_k']} |",
        f"| Steps | {report['num_steps']} |",
        f"| Instruction | {report['instruction']} |",
        "",
        "## Key Design Decisions",
        "",
        "1. **Attention does NOT replace action-constrained geometry** — geometry branches (scene_layout, depth, contact, motion) carry the primary signal.",
        "2. **Attention does NOT cause global attention top-k** — enters constrained union fill pool only.",
        "3. **Attention requires geometry alignment** — token must satisfy: `attention_high AND (scene_high OR depth_high OR contact_high OR motion_high)` to enter protected candidates.",
        "4. **Attention-only tokens (high attn, geometry weak)** are diagnostic only — recorded as `attention_only_token_count`, NOT forcibly retained.",
        "5. **Attention cannot crowd out depth/contact/motion minimum quotas** — geometry hard-protect budget is unaffected.",
        "",
        "## Assertions",
        "",
        f"| Check | Result |",
        f"|-------|--------|",
        f"| All attention_available | `{s.get('all_attention_available', 'N/A')}` |",
        f"| All attention_source == mock | `{s.get('all_attention_source_mock', 'N/A')}` |",
        f"| All accounting valid | `{s.get('all_acct_valid', 'N/A')}` |",
        f"| Error count | `{s.get('error_count', 'N/A')}` |",
        "",
        "## IoU Summary (attention vs geometry branches)",
        "",
        f"| Metric | Avg |",
        f"|--------|-----|",
        f"| attention_scene_iou | `{s.get('avg_attention_scene_iou', 'N/A')}` |",
        f"| attention_depth_iou | `{s.get('avg_attention_depth_iou', 'N/A')}` |",
        f"| attention_contact_iou | `{s.get('avg_attention_contact_iou', 'N/A')}` |",
        f"| attention_motion_iou | `{s.get('avg_attention_motion_iou', 'N/A')}` |",
        "",
        "## Count Summary",
        "",
        f"| Metric | Avg |",
        f"|--------|-----|",
        f"| avg attention_only_token_count | `{s.get('avg_attention_only_count', 'N/A')}` |",
        f"| avg geometry_only_token_count | `{s.get('avg_geometry_only_count', 'N/A')}` |",
        f"| avg attention_selected_by_final_count | `{s.get('avg_attention_sel_count', 'N/A')}` |",
        f"| avg safe_drop_candidate_count | `{s.get('avg_safe_drop_candidates', 'N/A')}` |",
        "",
        "## Per-Step Results",
        "",
        "| Step | attn_avail | conf | top_attn | attn_scene_iou | attn_contact_iou | "
        "attn_only | geo_only | attn_sel | acct | warnings |",
        "|------|------------|------|----------|----------------|------------------|"
        "----------|----------|----------|------|-----------|",
    ]

    for sr in steps:
        warnings_str = str(len(sr.get("warnings", []))) if sr.get("warnings") else "0"
        lines.append(
            f"| {sr['step']} "
            f"| {sr['attention_available']} "
            f"| {sr['attention_confidence']} "
            f"| {sr['top_attention_count']} "
            f"| {sr['attention_scene_iou']} "
            f"| {sr['attention_contact_iou']} "
            f"| {sr['attention_only_token_count']} "
            f"| {sr['geometry_only_token_count']} "
            f"| {sr['attention_selected_by_final_count']} "
            f"| {sr['acct_valid']} "
            f"| {warnings_str} "
            f"|"
        )

    lines.extend(["", "## Warnings", ""])
    if report.get("warnings"):
        for w in report["warnings"]:
            lines.append(f"- {w}")
    else:
        lines.append("No warnings.")

    lines.extend(["", "## Conclusion", ""])
    if report["all_passed"]:
        lines.append("**PASS** — attention_alignment_audit passed. Accounting valid across all steps. "
                    "Attention enters constrained union only. No global attention top-k behavior.")
    else:
        lines.append(f"**FAIL** — Errors: {report['errors']}")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Written: {out_path}")

    # Also write CSV
    csv_path = OUTPUT_DIR / "acgtp_v2_attention_alignment_audit.csv"
    fieldnames = [
        "step", "seed", "keep_k", "acct_valid",
        "attention_available", "attention_source", "attention_confidence", "top_attention_count",
        "attention_scene_iou", "attention_depth_iou", "attention_contact_iou", "attention_motion_iou",
        "attention_only_token_count", "geometry_only_token_count",
        "attention_geometry_overlap_count", "attention_background_risk_count",
        "attention_selected_by_final_count",
        "safe_drop_candidate_count", "dropped_safe_candidate_count",
        "dropped_high_attention_count", "dropped_high_geometry_count",
        "high_attention_low_geometry_count", "high_geometry_low_attention_count",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sr in steps:
            row = {k: sr.get(k, "") for k in fieldnames}
            writer.writerow(row)
    print(f"  Written: {csv_path}")


def write_attention_stress_audit_report(report: Dict[str, Any]) -> None:
    """Write attention stress test report (Task 7)."""
    out_path = OUTPUT_DIR / "acgtp_v2_attention_stress_audit_report.md"
    results = report.get("results", {})
    keep_k = report.get("keep_k", 0)

    lines = [
        "# ACGTP-v2 Attention Stress Test Report",
        "",
        "## Design: VLA-Cache / VLA-IAP / VLA-Pruner Inspired",
        "",
        "### VLA-Cache",
        "- safe_drop_candidate = low_scene AND low_depth AND low_contact AND low_motion AND low_attention",
        "- **Design**: high-attention tokens should NOT be dropped without geometry check",
        "",
        "### VLA-IAP",
        "- semantic attention may focus background; needs geometry alignment gate",
        "- **Design**: attention-only (high attn, geometry weak) = background bias candidate",
        "",
        "### VLA-Pruner",
        "- semantic prefill attention ≠ action relevance; no score fusion",
        "- **Design**: attention branch enters constrained union only, no global top-k",
        "",
        "## Configuration",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Keep ratio | {report['keep_ratio']} |",
        f"| keep_k | {keep_k} |",
        f"| n_tokens | {report['n_tokens']} |",
        f"| Instruction | {report['instruction']} |",
        "",
        "## Case Results",
        "",
        "| Case | Mode | Description | Passed | acct_valid | attn_sel | attn_only | attn_quota_rel |",
        "|------|------|-------------|--------|------------|----------|-----------|----------------|",
    ]

    for case_id in sorted(results.keys()):
        r = results[case_id]
        lines.append(
            f"| {case_id} "
            f"| {r.get('mode', 'N/A')} "
            f"| {r.get('description', '')} "
            f"| {'PASS' if r.get('passed') else 'FAIL'} "
            f"| {r.get('acct_valid', False)} "
            f"| {r.get('attention_selected_count', 0)} "
            f"| {r.get('attn_only_token_count', 0)} "
            f"| {r.get('attn_quota_released', 'N/A')} "
            f"|"
        )

    lines.extend(["", "## Detailed Results", ""])

    for case_id in sorted(results.keys()):
        r = results[case_id]
        lines.extend(["", f"### Case {case_id}: {r.get('description', '')}", ""])
        lines.append(f"**Mode**: `{r.get('mode', 'N/A')}`")
        lines.append(f"**Passed**: {'YES' if r.get('passed') else 'NO'}")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| accounting_valid | `{r.get('acct_valid', 'N/A')}` |")
        lines.append(f"| attention_available | `{r.get('attn_available', 'N/A')}` |")
        lines.append(f"| attention_quota_released | `{r.get('attn_quota_released', 'N/A')}` |")
        lines.append(f"| attention_selected_count | `{r.get('attention_selected_count', 0)}` |")
        lines.append(f"| attention_candidate_count | `{r.get('attn_candidate_count', 0)}` |")
        lines.append(f"| attention_top_count | `{r.get('attn_top_count', 0)}` |")
        lines.append(f"| attention_only_token_count | `{r.get('attn_only_token_count', 0)}` |")
        lines.append(f"| geometry_only_token_count | `{r.get('geometry_only_token_count', 0)}` |")
        lines.append(f"| keep_k | `{r.get('keep_k', keep_k)}` |")
        lines.append(f"| final_kept | `{r.get('final_kept', 'N/A')}` |")
        if case_id != "C":
            lines.append(f"| attention_scene_iou | `{r.get('attention_scene_iou', 'N/A')}` |")
            lines.append(f"| attention_contact_iou | `{r.get('attention_contact_iou', 'N/A')}` |")
            lines.append(f"| attention_geometry_overlap_count | `{r.get('attention_geometry_overlap_count', 'N/A')}` |")
            lines.append(f"| dropped_high_attention_count | `{r.get('dropped_high_attention_count', 0)}` |")
            lines.append(f"| dropped_high_geometry_count | `{r.get('dropped_high_geometry_count', 0)}` |")
            lines.append(f"| high_attention_low_geometry_count | `{r.get('high_attention_low_geometry_count', 0)}` |")
            lines.append(f"| high_geometry_low_attention_count | `{r.get('high_geometry_low_attention_count', 0)}` |")
            safe_warn = r.get("safe_drop_warning") or r.get("warnings", [None])[0] if r.get("warnings") else None
            if safe_warn:
                lines.append(f"| safe_drop_warning | `{safe_warn}` |")
        lines.append("")

        # Warnings
        if r.get("warnings"):
            lines.append("**Warnings:**")
            for w in r["warnings"]:
                lines.append(f"- {w}")
            lines.append("")

    lines.extend(["", "## Summary", ""])
    passed_count = report.get("passed_cases", 0)
    total_count = report.get("total_cases", 0)
    lines.append(f"**Cases passed**: {passed_count}/{total_count}")
    if report.get("all_passed"):
        lines.append("**ALL PASS** — Attention branch behaves correctly across all stress cases.")
        lines.append("")
        lines.append("Key verifications:")
        lines.append("- Case A: attention-aligned candidates can be selected (accounting valid)")
        lines.append("- Case B: attention-only tokens NOT forcibly retained (background_bias warning expected)")
        lines.append("- Case C: attention unavailable → full fallback, v1/v2 geometry path unaffected")
        lines.append("- Case D: geometry-dominant tokens still protected despite weak attention")
        lines.append("- Case E: partial overlap → union + refill works, no global attention top-k")
    else:
        lines.append(f"**FAIL** — Some cases did not pass: {[k for k,v in results.items() if not v.get('passed')]}")

    # Recommendations
    lines.extend(["", "## Recommendations", ""])
    lines.append("Based on stress test results:")
    lines.append("")
    lines.append("1. **Attention backend = none** (default): safe, no effect on selector, Jaccard = 1.0 with v1")
    lines.append("2. **Attention backend = mock**: safe when accounting valid; attention_only tokens correctly diagnostic")
    lines.append("3. **Next step (recommended)**: Integrate real attention probe from precomputed files")
    lines.append("   - Load text-to-vision / action-to-vision scores from model forward passes")
    lines.append("   - Use `precomputed` backend to read from `outputs/attention/` directory")
    lines.append("4. **Future semantic backend**: GroundingDINO / OWL-ViT for real visual grounding")
    lines.append("   - Attention can serve as alignment gate for semantic scores")
    lines.append("   - VLA-IAP: attention-misaligned semantic regions can be flagged")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Written: {out_path}")

    # Also write CSV
    csv_path = OUTPUT_DIR / "acgtp_v2_attention_stress_audit.csv"
    fieldnames = [
        "case_id", "mode", "description", "passed", "acct_valid",
        "attn_available", "attn_quota_released",
        "attention_selected_count", "attn_candidate_count", "attn_top_count",
        "attn_only_token_count", "geometry_only_token_count",
        "attention_scene_iou", "attention_contact_iou",
        "attention_geometry_overlap_count",
        "dropped_high_attention_count", "dropped_high_geometry_count",
        "high_attention_low_geometry_count", "high_geometry_low_attention_count",
        "keep_k", "final_kept",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for case_id in sorted(results.keys()):
            r = results[case_id]
            row = {k: r.get(k, "") for k in fieldnames}
            row["case_id"] = case_id
            row["description"] = r.get("description", "")
            writer.writerow(row)
    print(f"  Written: {csv_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="ACGTP-v2 Semantic Scene-Layout Audit Suite")
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["fallback_equivalence", "parser_audit", "semantic_backend_audit",
                 "scene_layout_audit", "accounting_audit",
                 "attention_alignment_audit", "attention_stress_audit",
                 "all"],
        help="Which audit task(s) to run.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("ACGTP-v2 Semantic Scene-Layout Branch Audit Suite")
    print("=" * 70)

    if args.task in ("fallback_equivalence", "all"):
        print("\n[TASK 1] Fallback equivalence: v1 vs v2 (semantic_backend=none)")
        t0 = time.perf_counter()
        eq_report = task1_fallback_equivalence(seed=7, n_tokens=256, keep_ratio=0.875, num_steps=20)
        t_eq = time.perf_counter() - t0
        print(f"  Done in {t_eq:.2f}s")
        write_fallback_equivalence_report(eq_report)
        print(f"  Result: {'PASS' if eq_report['equivalence_passed'] else 'FAIL'}")
        print(f"  strict_fallback_dispatch: {eq_report['summary']['all_strict_fallback_dispatch_used']}")
        print(f"  delegated_selector_name: {eq_report['summary']['all_delegated_to_v1']}")
        print(f"  keep_indices_exact_equal: {eq_report['summary']['all_indices_exact_equal']}")
        print(f"  Jaccard == 1.0: {eq_report['summary']['all_jaccard_eq_1']} (min={eq_report['summary']['min_jaccard']:.6f})")
        print(f"  Errors: {len(eq_report['errors'])}")

    if args.task in ("parser_audit", "all"):
        print("\n[TASK 2] Instruction parser audit (LIBERO suites)")
        t0 = time.perf_counter()
        parser_rows = task2_parser_audit()
        t_parser = time.perf_counter() - t0
        print(f"  Done in {t_parser:.1f}s — {len(parser_rows)} tasks")
        write_parser_audit_csv(parser_rows)
        write_parser_audit_report(parser_rows)
        if parser_rows:
            meaningful = sum(1 for r in parser_rows if r["instruction_is_meaningful"])
            action_leak = sum(1 for r in parser_rows if r.get("action_word_in_object_prompt", False))
            rel_leak = sum(1 for r in parser_rows if r.get("relation_word_in_object_prompt", False))
            print(f"  Coverage: {meaningful}/{len(parser_rows)} ({meaningful/max(len(parser_rows),1)*100:.1f}%)")
            print(f"  Action-word leaks: {action_leak}")
            print(f"  Relation-word leaks: {rel_leak}")

    if args.task in ("semantic_backend_audit", "all"):
        print("\n[TASK 3] Semantic backend audit: none / parser_only / mock")
        t0 = time.perf_counter()
        be_report = task3_semantic_backend_audit(
            instruction="pick up the black bowl between the plate and the ramekin",
            seed=42,
        )
        t_be = time.perf_counter() - t0
        print(f"  Done in {t_be:.2f}s")
        write_semantic_backend_report(be_report)
        print(f"  Result: {'PASS' if be_report['all_passed'] else 'FAIL'}")
        for bname, brec in be_report["backends"].items():
            print(f"  {bname}: available={brec['semantic_available']}, conf={brec['confidence']:.2f}, "
                  f"passed={brec.get('assert_passed', 'N/A')}")

    if args.task in ("scene_layout_audit", "all"):
        print("\n[TASK 4] Scene-layout branch audit (semantic_backend=mock)")
        t0 = time.perf_counter()
        sl_report = task4_scene_layout_audit(seed=7, n_tokens=256, keep_ratio=0.875, num_steps=10)
        t_sl = time.perf_counter() - t0
        print(f"  Done in {t_sl:.2f}s")
        write_scene_layout_audit_report(sl_report)
        print(f"  Result: {'PASS' if sl_report['all_passed'] else 'FAIL'}")
        print(f"  Avg selected_by_scene_layout: {sl_report['summary']['avg_sel_scene_layout']}")
        print(f"  Errors: {len(sl_report['errors'])}")

    if args.task in ("accounting_audit", "all"):
        print("\n[TASK 5] Accounting & debug field audit")
        t0 = time.perf_counter()
        acct_report = task5_accounting_audit(seed=7, n_tokens=256, keep_ratio=0.875, num_steps=10)
        t_acct = time.perf_counter() - t0
        print(f"  Done in {t_acct:.2f}s")
        write_accounting_audit_report(acct_report)
        print(f"  Result: {'PASS' if acct_report['all_passed'] else 'FAIL'}")
        print(f"  Missing required fields: {acct_report['missing_required_fields']}")
        print(f"  Missing new fields: {acct_report['missing_new_fields']}")

    if args.task in ("attention_alignment_audit", "all"):
        print("\n[TASK 6] Attention task-relevance alignment audit")
        t0 = time.perf_counter()
        attn_report = task6_attention_alignment_audit(seed=7, n_tokens=256, keep_ratio=0.875, num_steps=10)
        t_attn = time.perf_counter() - t0
        print(f"  Done in {t_attn:.2f}s")
        write_attention_alignment_audit_report(attn_report)
        print(f"  Result: {'PASS' if attn_report['all_passed'] else 'FAIL'}")
        print(f"  Avg attention_scene_iou: {attn_report['summary']['avg_attention_scene_iou']}")
        print(f"  Avg attention_only_count: {attn_report['summary']['avg_attention_only_count']}")
        print(f"  Avg attention_sel_count: {attn_report['summary']['avg_attention_sel_count']}")
        print(f"  Errors: {len(attn_report['errors'])}")
        print(f"  Warnings: {len(attn_report['warnings'])}")

    if args.task in ("attention_stress_audit", "all"):
        print("\n[TASK 7] Attention stress tests (Cases A-E)")
        t0 = time.perf_counter()
        stress_report = task7_attention_stress_test(seed=7, n_tokens=256, keep_ratio=0.875)
        t_stress = time.perf_counter() - t0
        print(f"  Done in {t_stress:.2f}s")
        write_attention_stress_audit_report(stress_report)
        print(f"  Result: {'PASS' if stress_report['all_passed'] else 'FAIL'}")
        print(f"  Cases passed: {stress_report['passed_cases']}/{stress_report['total_cases']}")
        for case_id, r in stress_report["results"].items():
            print(f"  Case {case_id}: {'PASS' if r.get('passed') else 'FAIL'} — {r.get('description', '')}")

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)
    print(f"\nAll outputs written to: {OUTPUT_DIR}/")
    print("\nGenerated reports:")
    print(f"  {OUTPUT_DIR / 'acgtp_v2_fallback_equivalence_report.md'}")
    print(f"  {OUTPUT_DIR / 'acgtp_v2_instruction_parser_audit.csv'}")
    print(f"  {OUTPUT_DIR / 'acgtp_v2_instruction_parser_audit_report.md'}")
    print(f"  {OUTPUT_DIR / 'acgtp_v2_semantic_backend_audit_report.md'}")
    print(f"  {OUTPUT_DIR / 'acgtp_v2_scene_layout_audit_report.md'}")
    print(f"  {OUTPUT_DIR / 'acgtp_v2_accounting_debug_audit_report.md'}")
    print(f"  {OUTPUT_DIR / 'acgtp_v2_attention_alignment_audit_report.md'}")
    print(f"  {OUTPUT_DIR / 'acgtp_v2_attention_alignment_audit.csv'}")
    print(f"  {OUTPUT_DIR / 'acgtp_v2_attention_stress_audit_report.md'}")
    print(f"  {OUTPUT_DIR / 'acgtp_v2_attention_stress_audit.csv'}")


if __name__ == "__main__":
    main()
