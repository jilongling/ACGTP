"""
SemanticLayoutBackend — Pluggable Scene-Layout Semantic Branch for ACGTP-v2.

Provides three backends:
  1. "none"     — always unavailable, triggers v2 fallback (geometric path only).
  2. "parser_only" — returns parsed instruction only, no valid masks/scores.
  3. "mock"     — synthetic token-grid masks/scores for selector verification.

select_acgtp_v2 only depends on SemanticLayoutResult, not on any specific backend.

Usage:
    backend = SemanticLayoutBackend(backend="mock", grid_h=16, grid_w=16, seed=42)
    result = backend.run(instruction="pick up the black bowl between the plate and the ramekin",
                         parsed=parsed_terms)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# SemanticLayoutResult — unified return type for all backends
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SemanticLayoutResult:
    """Unified result structure returned by all semantic backends.

    Attributes
    ----------
    semantic_available : bool
        True when a real visual detector was queried and produced usable masks.
        False for "none" and "parser_only" backends.
    confidence : float
        In [0, 1]. 0.0 = no detector available. >0.8 = real detector response.
    target_mask : np.ndarray
        [N] float array, 1.0 for target-object tokens, 0.0 elsewhere.
    reference_mask : np.ndarray
        [N] float array, 1.0 for reference-object tokens, 0.0 elsewhere.
    relation_mask : np.ndarray
        [N] float array, 1.0 for spatial-relation region tokens, 0.0 elsewhere.
    layout_anchor_mask : np.ndarray
        [N] float array, 1.0 for layout-anchor tokens, 0.0 elsewhere.
    token_scores : np.ndarray
        [N] float array, composite scene-layout score per token (weighted mix).
    debug : dict
        Backend diagnostics, per-category counts, intermediate arrays.
    """

    semantic_available: bool
    confidence: float
    target_mask: np.ndarray
    reference_mask: np.ndarray
    relation_mask: np.ndarray
    layout_anchor_mask: np.ndarray
    token_scores: np.ndarray
    debug: Dict[str, Any] = field(default_factory=dict)

    # Convenience aliases for selector
    @property
    def target_scores(self) -> np.ndarray:
        return self.target_mask

    @property
    def reference_scores(self) -> np.ndarray:
        return self.reference_mask

    @property
    def relation_scores(self) -> np.ndarray:
        return self.relation_mask

    @property
    def anchor_scores(self) -> np.ndarray:
        return self.layout_anchor_mask

    @property
    def semantic_anchor_scores(self) -> np.ndarray:
        return self.token_scores

    def to_dict(self) -> Dict[str, Any]:
        return {
            "semantic_available": self.semantic_available,
            "confidence": self.confidence,
            "target_mask": self.target_mask,
            "reference_mask": self.reference_mask,
            "relation_mask": self.relation_mask,
            "layout_anchor_mask": self.layout_anchor_mask,
            "token_scores": self.token_scores,
            "debug": self.debug,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SemanticLayoutBackend — factory + dispatch
# ─────────────────────────────────────────────────────────────────────────────

class SemanticLayoutBackend:
    """Pluggable semantic scene-layout backend for ACGTP-v2.

    Parameters
    ----------
    backend : str
        One of {"none", "parser_only", "mock"}.
    grid_h, grid_w : int
        Token grid dimensions (default 16x16 → 256 tokens).
    seed : int
        Random seed for mock backend.
    w_target : float
        Weight for target-object tokens in composite score.
    w_reference : float
        Weight for reference-object tokens.
    w_relation : float
        Weight for spatial-relation region tokens.
    w_layout : float
        Weight for layout-anchor tokens.
    """

    SUPPORTED = {"none", "parser_only", "mock"}

    def __init__(
        self,
        backend: str = "none",
        grid_h: int = 16,
        grid_w: int = 16,
        seed: int = 0,
        w_target: float = 1.0,
        w_reference: float = 0.7,
        w_relation: float = 0.5,
        w_layout: float = 0.6,
    ):
        if backend not in self.SUPPORTED:
            raise ValueError(
                f"Unknown backend '{backend}'. Supported: {self.SUPPORTED}"
            )
        self.backend = backend
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.n = grid_h * grid_w
        self.seed = seed
        self.w_target = w_target
        self.w_reference = w_reference
        self.w_relation = w_relation
        self.w_layout = w_layout

    # ── Public API ──────────────────────────────────────────────────────────

    def run(
        self,
        instruction: Optional[str],
        parsed: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SemanticLayoutResult:
        """Compute scene-layout semantic scores for the given instruction.

        Parameters
        ----------
        instruction : str or None
            Raw instruction string (used for debug metadata).
        parsed : dict or None
            Structured parse from ``parse_instruction_terms()``.
            If None, a dummy parse is created.
        **kwargs
            Backend-specific overrides.

        Returns
        -------
        SemanticLayoutResult
        """
        if self.backend == "none":
            return self._run_none(instruction, parsed, **kwargs)
        elif self.backend == "parser_only":
            return self._run_parser_only(instruction, parsed, **kwargs)
        elif self.backend == "mock":
            return self._run_mock(instruction, parsed, **kwargs)
        else:
            raise RuntimeError(f"Unreachable: unknown backend '{self.backend}'")

    def is_semantic_available(self) -> bool:
        """Return True when this backend provides real semantic scores."""
        return self.backend == "mock"

    def __repr__(self) -> str:
        return (
            f"SemanticLayoutBackend(backend={self.backend!r}, "
            f"grid=({self.grid_h}×{self.grid_w}), "
            f"w_target={self.w_target}, w_reference={self.w_reference}, "
            f"w_relation={self.w_relation}, w_layout={self.w_layout})"
        )

    # ── Backend: none ─────────────────────────────────────────────────────────

    def _run_none(
        self,
        instruction: Optional[str],
        parsed: Optional[Dict[str, Any]],
        **kwargs,
    ) -> SemanticLayoutResult:
        """Always returns unavailable / zero scores. Triggers v2 geometric fallback."""
        zeros = np.zeros(self.n, dtype=np.float32)
        return SemanticLayoutResult(
            semantic_available=False,
            confidence=0.0,
            target_mask=zeros.copy(),
            reference_mask=zeros.copy(),
            relation_mask=zeros.copy(),
            layout_anchor_mask=zeros.copy(),
            token_scores=zeros.copy(),
            debug={
                "backend": "none",
                "instruction": instruction or "",
                "parsed_summary": self._summarise_parsed(parsed),
                "note": (
                    "semantic_backend='none': all masks/scores are zero. "
                    "v2 falls back to geometric branches (scene_layout, depth, "
                    "contact_ring, motion_corridor) with full budget."
                ),
            },
        )

    # ── Backend: parser_only ──────────────────────────────────────────────────

    def _run_parser_only(
        self,
        instruction: Optional[str],
        parsed: Optional[Dict[str, Any]],
        **kwargs,
    ) -> SemanticLayoutResult:
        """Returns parsed instruction only. No valid masks/scores — available=False."""
        zeros = np.zeros(self.n, dtype=np.float32)
        summary = self._summarise_parsed(parsed)
        return SemanticLayoutResult(
            semantic_available=False,
            confidence=0.0,
            target_mask=zeros.copy(),
            reference_mask=zeros.copy(),
            relation_mask=zeros.copy(),
            layout_anchor_mask=zeros.copy(),
            token_scores=zeros.copy(),
            debug={
                "backend": "parser_only",
                "instruction": instruction or "",
                "parsed_summary": summary,
                "parsed_full": parsed,
                "note": (
                    "parser_only: no visual detector queried. "
                    "Masks and scores are zero. "
                    "Use for instruction-only diagnostics without grounding."
                ),
            },
        )

    # ── Backend: mock ────────────────────────────────────────────────────────

    def _run_mock(
        self,
        instruction: Optional[str],
        parsed: Optional[Dict[str, Any]],
        **kwargs,
    ) -> SemanticLayoutResult:
        """Construct synthetic target/reference/relation/layout masks for selector testing.

        Layout:
          - Grid is divided into 4 quadrants (each 8×8)
          - Quadrant 0 (top-left):     target_mask = 1.0
          - Quadrant 1 (top-right):    reference_mask = 1.0
          - Quadrant 2 (bottom-left):  relation_mask = 1.0
          - Quadrant 3 (bottom-right): layout_anchor_mask = 1.0

        If parsed terms are provided, their values are injected into debug output.
        """
        rng = kwargs.get("rng", np.random.default_rng(self.seed))
        parsed_summary = self._summarise_parsed(parsed)

        qh, qw = self.grid_h // 2, self.grid_w // 2

        # Target: top-left quadrant (Q0)
        target_mask = np.zeros(self.n, dtype=np.float32)
        for r in range(qh):
            for c in range(qw):
                target_mask[r * self.grid_w + c] = 1.0

        # Reference: top-right quadrant (Q1)
        reference_mask = np.zeros(self.n, dtype=np.float32)
        for r in range(qh):
            for c in range(qw, self.grid_w):
                reference_mask[r * self.grid_w + c] = 1.0

        # Relation: bottom-left quadrant (Q2)
        relation_mask = np.zeros(self.n, dtype=np.float32)
        for r in range(qh, self.grid_h):
            for c in range(qw):
                relation_mask[r * self.grid_w + c] = 1.0

        # Layout anchor: bottom-right quadrant (Q3) + sparse noise
        layout_anchor_mask = np.zeros(self.n, dtype=np.float32)
        for r in range(qh, self.grid_h):
            for c in range(qw, self.grid_w):
                layout_anchor_mask[r * self.grid_w + c] = 1.0
        # Add small noise anchors for more realistic distribution
        n_extra = max(1, self.n // 32)
        extra_idx = rng.choice(self.n, size=n_extra, replace=False)
        layout_anchor_mask[extra_idx] = 0.7

        # Composite token scores: weighted mix
        token_scores = (
            self.w_target * target_mask
            + self.w_reference * reference_mask
            + self.w_relation * relation_mask
            + self.w_layout * layout_anchor_mask
        )

        # Add tiny per-token perturbation for tie-break testing
        perturbation = rng.uniform(-0.001, 0.001, size=self.n).astype(np.float32)
        token_scores = token_scores + perturbation

        # Normalise to [0, 1]
        v = token_scores[token_scores > 0]
        if v.size > 0:
            lo, hi = float(np.min(v)), float(np.max(v))
            if hi - lo > 1e-8:
                token_scores = (token_scores - lo) / (hi - lo)
            else:
                token_scores = np.zeros(self.n, dtype=np.float32)

        target_count = int(np.sum(target_mask > 0.5))
        reference_count = int(np.sum(reference_mask > 0.5))
        relation_count = int(np.sum(relation_mask > 0.5))
        layout_count = int(np.sum(layout_anchor_mask > 0.5))

        return SemanticLayoutResult(
            semantic_available=True,
            confidence=0.85,
            target_mask=target_mask,
            reference_mask=reference_mask,
            relation_mask=relation_mask,
            layout_anchor_mask=layout_anchor_mask,
            token_scores=token_scores,
            debug={
                "backend": "mock",
                "seed": self.seed,
                "instruction": instruction or "",
                "parsed_summary": parsed_summary,
                "grid_shape": [self.grid_h, self.grid_w],
                "weights": {
                    "w_target": self.w_target,
                    "w_reference": self.w_reference,
                    "w_relation": self.w_relation,
                    "w_layout": self.w_layout,
                },
                "quadrant_layout": {
                    "target": {"quadrant": "top-left", "token_count": target_count},
                    "reference": {"quadrant": "top-right", "token_count": reference_count},
                    "relation": {"quadrant": "bottom-left", "token_count": relation_count},
                    "layout_anchor": {"quadrant": "bottom-right+sparse", "token_count": layout_count},
                },
                "note": (
                    "mock backend: synthetic quadrant-based masks. "
                    "Useful for verifying semantic/scene-layout branch token protection "
                    "and accounting without real detector dependency."
                ),
            },
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    def _summarise_parsed(self, parsed: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract a debug-friendly summary from a parse result."""
        if parsed is None:
            return {"has_parse": False}
        raw = parsed.get("raw_parse", {})
        target = parsed.get("target", {})
        return {
            "has_parse": True,
            "target_object": target.get("object", ""),
            "target_attrs": target.get("attributes", []),
            "reference_objects": [r.get("object", "") for r in parsed.get("references", [])],
            "relations": [
                {"type": rel.get("type", ""), "refs": rel.get("references", [])}
                for rel in parsed.get("relations", [])
            ],
            "actions": parsed.get("actions", []),
            "raw_target_terms": raw.get("parsed_target_terms", []),
            "raw_reference_terms": raw.get("parsed_reference_terms", []),
            "raw_relation_terms": raw.get("parsed_relation_terms", []),
            "instruction_is_meaningful": parsed.get("instruction_is_meaningful", False),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────────────────────────────────────

def get_semantic_backend(
    backend: str = "none",
    grid_h: int = 16,
    grid_w: int = 16,
    seed: int = 0,
    w_target: float = 1.0,
    w_reference: float = 0.7,
    w_relation: float = 0.5,
    w_layout: float = 0.6,
) -> SemanticLayoutBackend:
    """Factory function — creates and returns a configured SemanticLayoutBackend."""
    return SemanticLayoutBackend(
        backend=backend,
        grid_h=grid_h,
        grid_w=grid_w,
        seed=seed,
        w_target=w_target,
        w_reference=w_reference,
        w_relation=w_relation,
        w_layout=w_layout,
    )
