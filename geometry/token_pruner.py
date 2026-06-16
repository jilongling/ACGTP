"""
TokenPruner: Applies token pruning to OpenVLA visual tokens.

This module implements visual token pruning for the Prismatic VLM architecture.
The pruning is applied after the vision encoder generates visual tokens and before
they are concatenated with text embeddings for the LLM.

Supported pruning methods:
- none: No pruning (baseline)
- random: Random token pruning
- depth_edge: Prune tokens based on depth edge scores
- depth_edge_fast_diverse: Depth-edge pruning with a small spatial diversity reserve
- gripper_distance: Prune tokens based on distance to gripper
- geometry_rule: Prune tokens based on combined geometry scores

Architecture:
1. Unified prune_visual_tokens() function for all methods
2. TokenPruner class handles configuration and orchestration
3. PruningModelWrapper wraps the VLM to intercept and modify visual tokens

Usage:
    # Functional approach
    result = prune_visual_tokens(
        visual_tokens=patch_embeddings,
        scores=GeometryScoreResult(...),
        keep_ratio=0.5,
        method="geometry_rule",
        seed=42,
    )
    pruned_tokens = result.pruned_visual_tokens

    # Class-based approach
    pruner = TokenPruner(config=PruningConfig(
        method="geometry_rule",
        keep_ratio=0.5,
    ))
    pruned_tokens, result = pruner.prune_tokens(patch_embeddings, scores)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


class PruningMethod(str, Enum):
    """Supported token pruning methods."""

    NONE = "none"
    RANDOM = "random"
    DEPTH_EDGE = "depth_edge"
    DEPTH_EDGE_FAST_DIVERSE = "depth_edge_fast_diverse"
    GRIPPER_DISTANCE = "gripper_distance"
    GEOMETRY_RULE = "geometry_rule"

    @classmethod
    def from_string(cls, s: str) -> "PruningMethod":
        """Parse pruning method from string."""
        s_lower = s.lower().strip()
        for method in cls:
            if method.value == s_lower:
                return method
        raise ValueError(f"Unknown pruning method: {s}. Available: {[m.value for m in cls]}")


@dataclass
class PruningConfig:
    """
    Configuration for token pruning.

    Attributes:
        method: Pruning method to use.
        keep_ratio: Fraction of tokens to keep (0.0 to 1.0).
        seed: Random seed for reproducibility.
        debug: Whether to enable debug logging.
    """

    method: PruningMethod = PruningMethod.NONE
    keep_ratio: float = 1.0
    seed: int = 42
    debug: bool = False

    def __post_init__(self):
        """Validate configuration."""
        if isinstance(self.method, str):
            self.method = PruningMethod.from_string(self.method)

        if not 0.0 < self.keep_ratio <= 1.0:
            logger.warning(
                f"[PruningConfig] keep_ratio={self.keep_ratio} is out of range (0, 1]. "
                f"Clamping to valid range."
            )
            self.keep_ratio = max(0.01, min(1.0, self.keep_ratio))

        if self.method != PruningMethod.NONE and self.keep_ratio >= 1.0:
            logger.warning(
                f"[PruningConfig] keep_ratio=1.0 with method={self.method.value}. "
                f"This is equivalent to 'none'. Setting method to 'none'."
            )
            self.method = PruningMethod.NONE

    def is_enabled(self) -> bool:
        """Check if pruning is enabled."""
        return self.method != PruningMethod.NONE and self.keep_ratio < 1.0


@dataclass
class PruningResult:
    """
    Result of applying token pruning.

    Attributes:
        pruned_visual_tokens: Pruned visual token embeddings [batch, num_kept, hidden_dim].
        keep_indices: Indices of kept tokens (sorted).
        keep_mask: Boolean mask [num_original] indicating kept tokens.
        num_tokens_before: Number of tokens before pruning.
        num_tokens_after: Number of tokens after pruning.
        actual_keep_ratio: Actual ratio of kept tokens.
        timing_ms: Time taken for pruning in milliseconds.
        method: Pruning method used.
    """

    pruned_visual_tokens: torch.Tensor
    keep_indices: np.ndarray
    keep_mask: np.ndarray
    num_tokens_before: int
    num_tokens_after: int
    actual_keep_ratio: float
    timing_ms: float = 0.0
    method: str = "none"
    selection_metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate result."""
        if self.num_tokens_before > 0:
            expected_ratio = self.num_tokens_after / self.num_tokens_before
            if abs(expected_ratio - self.actual_keep_ratio) > 0.01:
                logger.warning(
                    f"[PruningResult] actual_keep_ratio mismatch: "
                    f"computed={expected_ratio:.3f}, stored={self.actual_keep_ratio:.3f}"
                )

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary dict."""
        return {
            "num_tokens_before": self.num_tokens_before,
            "num_tokens_after": self.num_tokens_after,
            "actual_keep_ratio": self.actual_keep_ratio,
            "timing_ms": self.timing_ms,
            "method": self.method,
            "selection_metadata": self.selection_metadata,
        }


@dataclass
class TokenScores:
    """
    Unified score container for geometry-based pruning.

    This dataclass wraps all score components from the geometry expert
    for use by the pruning methods.
    """

    token_scores: np.ndarray
    distance_score: np.ndarray
    direction_score: np.ndarray
    depth_edge_score: np.ndarray
    valid_mask: np.ndarray

    @classmethod
    def from_geometry_result(cls, geometry_result: Any) -> "TokenScores":
        """
        Create TokenScores from GeometryScoreResult.

        Args:
            geometry_result: GeometryScoreResult from RuleBasedGeometryExpert.

        Returns:
            TokenScores instance.
        """
        return cls(
            token_scores=np.asarray(geometry_result.token_scores, dtype=np.float32),
            distance_score=np.asarray(geometry_result.distance_score, dtype=np.float32),
            direction_score=np.asarray(geometry_result.direction_score, dtype=np.float32),
            depth_edge_score=np.asarray(geometry_result.depth_edge_score, dtype=np.float32),
            valid_mask=np.asarray(geometry_result.valid_score_mask, dtype=np.bool_),
        )

    @classmethod
    def from_arrays(
        cls,
        token_scores: np.ndarray,
        distance_score: Optional[np.ndarray] = None,
        direction_score: Optional[np.ndarray] = None,
        depth_edge_score: Optional[np.ndarray] = None,
        valid_mask: Optional[np.ndarray] = None,
    ) -> "TokenScores":
        """Create TokenScores from numpy arrays."""
        n = len(token_scores)
        return cls(
            token_scores=np.asarray(token_scores, dtype=np.float32),
            distance_score=np.asarray(distance_score) if distance_score is not None else np.zeros(n, dtype=np.float32),
            direction_score=np.asarray(direction_score) if direction_score is not None else np.zeros(n, dtype=np.float32),
            depth_edge_score=np.asarray(depth_edge_score) if depth_edge_score is not None else np.zeros(n, dtype=np.float32),
            valid_mask=np.asarray(valid_mask) if valid_mask is not None else np.ones(n, dtype=np.bool_),
        )


# =============================================================================
# Core Pruning Function
# =============================================================================


def prune_visual_tokens(
    visual_tokens: torch.Tensor,
    scores: Optional[Union[TokenScores, np.ndarray, Dict[str, np.ndarray]]],
    keep_ratio: float,
    method: str = "none",
    seed: int = 42,
    debug: bool = False,
) -> PruningResult:
    """
    Unified function to prune visual tokens.

    Args:
        visual_tokens: Visual token embeddings [batch, num_visual_tokens, hidden_dim].
        scores: Token scores for importance-based pruning. Can be:
            - TokenScores: Full scores from geometry expert
            - np.ndarray: Combined token scores [num_visual_tokens]
            - Dict: {"token_scores": ..., "distance_score": ..., etc.}
        keep_ratio: Fraction of tokens to keep (0.0 to 1.0).
        method: Pruning method ("none", "random", "depth_edge",
            "depth_edge_fast_diverse", "gripper_distance", "geometry_rule").
        seed: Random seed for reproducibility (only used for "random" method).
        debug: Enable debug logging.

    Returns:
        PruningResult with pruned tokens and metadata.

    Raises:
        ValueError: If method is unknown or inputs are invalid.
        AssertionError: If batch_size != 1 (currently only batch_size=1 is supported).
    """
    start_time = time.perf_counter()

    # Validate inputs
    if len(visual_tokens.shape) != 3:
        raise ValueError(f"visual_tokens must be 3D [batch, tokens, hidden], got shape {visual_tokens.shape}")

    batch_size, num_tokens, hidden_dim = visual_tokens.shape

    # Batch size check
    if batch_size != 1:
        logger.warning(
            f"[prune_visual_tokens] batch_size={batch_size} != 1. "
            f"Only batch_size=1 is fully supported. Attempting to continue..."
        )

    num_kept = int(num_tokens * keep_ratio)
    num_pruned = num_tokens - num_kept

    # No pruning case
    if method == "none" or keep_ratio >= 1.0:
        keep_indices = np.arange(num_tokens)
        keep_mask = np.ones(num_tokens, dtype=np.bool_)
        pruned_tokens = visual_tokens

        timing_ms = (time.perf_counter() - start_time) * 1000.0
        return PruningResult(
            pruned_visual_tokens=pruned_tokens,
            keep_indices=keep_indices,
            keep_mask=keep_mask,
            num_tokens_before=num_tokens,
            num_tokens_after=num_tokens,
            actual_keep_ratio=1.0,
            timing_ms=timing_ms,
            method="none",
        )

    # Parse scores
    token_scores_np, distance_np, direction_np, depth_edge_np, valid_mask_np = _parse_scores(scores, num_tokens)

    # Compute keep indices based on method
    selection_metadata = None

    if method == "random":
        keep_indices = _random_pruning(num_tokens, num_kept, seed)
    elif method == "depth_edge":
        keep_indices = _score_based_pruning(
            depth_edge_np, num_kept, valid_mask_np, higher_is_better=True, seed=seed
        )
    elif method == "depth_edge_fast_diverse":
        keep_indices, selection_metadata = select_depth_edge_diverse_indices(
            scores=depth_edge_np,
            valid_mask=valid_mask_np,
            keep_total=num_kept,
            depth_quota=max(0, num_kept - min(32, num_kept)),
            grid_size=int(round(num_tokens ** 0.5)),
            cell_grid=4,
        )
    elif method == "gripper_distance":
        # For gripper_distance, lower distance = closer = better
        # distance_score is already computed as exp(-d^2/sigma^2), so higher is closer
        keep_indices = _score_based_pruning(
            distance_np, num_kept, valid_mask_np, higher_is_better=True, seed=seed
        )
    elif method == "geometry_rule":
        keep_indices = _score_based_pruning(
            token_scores_np, num_kept, valid_mask_np, higher_is_better=True, seed=seed
        )
    else:
        raise ValueError(f"Unknown pruning method: {method}")

    # Sort keep_indices to maintain original spatial order
    keep_indices = np.sort(keep_indices)

    # Create keep mask
    keep_mask = np.zeros(num_tokens, dtype=np.bool_)
    keep_mask[keep_indices] = True

    # Apply pruning
    pruned_tokens = visual_tokens[:, keep_indices, :]

    timing_ms = (time.perf_counter() - start_time) * 1000.0
    actual_ratio = num_kept / num_tokens

    if debug:
        logger.debug(
            f"[prune_visual_tokens] method={method}, "
            f"pruned {num_pruned}/{num_tokens} tokens "
            f"(keep_ratio={actual_ratio:.2f}), "
            f"timing={timing_ms:.2f}ms"
        )

    return PruningResult(
        pruned_visual_tokens=pruned_tokens,
        keep_indices=keep_indices,
        keep_mask=keep_mask,
        num_tokens_before=num_tokens,
        num_tokens_after=num_kept,
        actual_keep_ratio=actual_ratio,
        timing_ms=timing_ms,
        method=method,
        selection_metadata=selection_metadata,
    )


def _parse_scores(
    scores: Optional[Union[TokenScores, np.ndarray, Dict[str, np.ndarray]]],
    num_tokens: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse scores into component arrays.

    Returns:
        Tuple of (token_scores, distance, direction, depth_edge, valid_mask).
    """
    if scores is None:
        # Default scores: uniform
        return (
            np.ones(num_tokens, dtype=np.float32),
            np.ones(num_tokens, dtype=np.float32),
            np.zeros(num_tokens, dtype=np.float32),
            np.zeros(num_tokens, dtype=np.float32),
            np.ones(num_tokens, dtype=np.bool_),
        )

    if isinstance(scores, TokenScores):
        return (
            scores.token_scores,
            scores.distance_score,
            scores.direction_score,
            scores.depth_edge_score,
            scores.valid_mask,
        )

    if isinstance(scores, np.ndarray):
        # Combined scores only
        return (
            scores,
            np.zeros(num_tokens, dtype=np.float32),
            np.zeros(num_tokens, dtype=np.float32),
            np.zeros(num_tokens, dtype=np.float32),
            np.ones(num_tokens, dtype=np.bool_),
        )

    if isinstance(scores, dict):
        # Dictionary of scores
        return (
            scores.get("token_scores", np.ones(num_tokens, dtype=np.float32)),
            scores.get("distance_score", np.ones(num_tokens, dtype=np.float32)),
            scores.get("direction_score", np.zeros(num_tokens, dtype=np.float32)),
            scores.get("depth_edge_score", np.zeros(num_tokens, dtype=np.float32)),
            scores.get("valid_mask", np.ones(num_tokens, dtype=np.bool_)),
        )

    raise ValueError(f"Unknown scores type: {type(scores)}")


def _random_pruning(num_tokens: int, num_kept: int, seed: int) -> np.ndarray:
    """Select random tokens to keep."""
    rng = np.random.default_rng(seed)
    indices = np.arange(num_tokens)
    rng.shuffle(indices)
    return np.sort(indices[:num_kept])


def select_depth_edge_diverse_indices(
    scores: Union[np.ndarray, torch.Tensor],
    valid_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    keep_total: int = 192,
    depth_quota: int = 160,
    grid_size: int = 16,
    cell_grid: int = 4,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Select depth-edge tokens with a small spatial diversity reserve.

    The first quota keeps the highest depth-edge tokens. The reserve quota then
    fills under-covered 4x4 spatial cells with the best remaining token in each
    cell, and finally falls back to global score order if needed.
    """
    scores_np = np.asarray(scores.detach().cpu().numpy() if isinstance(scores, torch.Tensor) else scores, dtype=np.float32)
    n = int(scores_np.shape[0])
    if valid_mask is None:
        valid_np = np.ones(n, dtype=np.bool_)
    else:
        valid_np = np.asarray(
            valid_mask.detach().cpu().numpy() if isinstance(valid_mask, torch.Tensor) else valid_mask,
            dtype=np.bool_,
        )

    if n == 0:
        return np.array([], dtype=np.int64), {
            "depth_quota": 0,
            "reserve_quota": 0,
            "reserve_selected": 0,
            "fallback_selected": 0,
        }

    keep_total = int(max(0, min(keep_total, n)))
    depth_quota = int(max(0, min(depth_quota, keep_total)))
    reserve_quota = keep_total - depth_quota
    if grid_size * grid_size != n:
        inferred = int(round(n ** 0.5))
        grid_size = inferred if inferred * inferred == n else grid_size

    adjusted = np.nan_to_num(scores_np, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    adjusted = np.where(valid_np, adjusted, -np.inf)
    order = np.lexsort((np.arange(n), -adjusted))

    depth_selected = order[:depth_quota].astype(np.int64)
    selected = set(int(i) for i in depth_selected)
    reserve_selected: List[int] = []
    fallback_selected = 0

    if reserve_quota > 0 and grid_size > 0 and n == grid_size * grid_size:
        cell_size = max(1, grid_size // max(1, cell_grid))
        cell_infos: List[Tuple[int, int, int, int]] = []
        for cell_r in range(cell_grid):
            for cell_c in range(cell_grid):
                r0 = cell_r * cell_size
                c0 = cell_c * cell_size
                r1 = grid_size if cell_r == cell_grid - 1 else min(grid_size, r0 + cell_size)
                c1 = grid_size if cell_c == cell_grid - 1 else min(grid_size, c0 + cell_size)
                cell_indices = [
                    r * grid_size + c
                    for r in range(r0, r1)
                    for c in range(c0, c1)
                    if (r * grid_size + c) < n
                ]
                covered = sum(1 for idx in cell_indices if idx in selected)
                cell_infos.append((covered, cell_r, cell_c, len(cell_indices)))

        # Under-covered cells first, deterministic row-major tie-break.
        for _, cell_r, cell_c, _ in sorted(cell_infos, key=lambda x: (x[0], x[1], x[2])):
            if len(reserve_selected) >= reserve_quota:
                break
            r0 = cell_r * cell_size
            c0 = cell_c * cell_size
            r1 = grid_size if cell_r == cell_grid - 1 else min(grid_size, r0 + cell_size)
            c1 = grid_size if cell_c == cell_grid - 1 else min(grid_size, c0 + cell_size)
            candidates = [
                r * grid_size + c
                for r in range(r0, r1)
                for c in range(c0, c1)
                if (r * grid_size + c) < n and (r * grid_size + c) not in selected
            ]
            candidates = [idx for idx in candidates if valid_np[idx]]
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
        # Only possible when many tokens are invalid; fill deterministically to
        # keep the requested visual token count exact.
        for idx in range(n):
            if idx not in selected:
                selected.add(idx)
            if len(selected) >= keep_total:
                break

    keep_indices = np.array(sorted(selected), dtype=np.int64)[:keep_total]
    metadata = {
        "depth_quota": int(depth_quota),
        "reserve_quota": int(reserve_quota),
        "depth_selected": int(len(depth_selected)),
        "reserve_selected": int(min(len(reserve_selected), reserve_quota)),
        "fallback_selected": int(fallback_selected),
        "grid_size": int(grid_size),
        "cell_grid": int(cell_grid),
        "final_kept": int(len(keep_indices)),
        "keep_indices_sorted": bool(np.all(keep_indices[:-1] <= keep_indices[1:])) if len(keep_indices) > 1 else True,
    }
    return keep_indices, metadata


def _score_based_pruning(
    scores: np.ndarray,
    num_kept: int,
    valid_mask: np.ndarray,
    higher_is_better: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Select tokens based on importance scores.

    Tokens are selected in order of importance, with invalid tokens deprioritized.
    If there are not enough valid tokens, invalid tokens may be included.

    Args:
        scores: Token importance scores [num_tokens].
        num_kept: Number of tokens to keep.
        valid_mask: Boolean mask indicating valid tokens.
        higher_is_better: If True, keep highest scores; else keep lowest.
        seed: Random seed for tiebreaking.

    Returns:
        Indices of tokens to keep (sorted).
    """
    num_tokens = len(scores)

    # Handle NaN/inf in scores
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)

    # Create scoring for sorting: penalize invalid tokens
    # Valid tokens keep their score, invalid tokens get -inf
    adjusted_scores = np.where(valid_mask, scores, -np.inf)

    if higher_is_better:
        # Sort by adjusted scores (descending), then by original index for stable sort
        sorted_indices = np.lexsort((np.arange(num_tokens), -adjusted_scores))
    else:
        # Sort by adjusted scores (ascending)
        sorted_indices = np.lexsort((np.arange(num_tokens), adjusted_scores))

    # Take top-k
    keep_indices = sorted_indices[:num_kept]

    # If we don't have enough valid tokens, we may need to include some invalid ones
    valid_keep = np.sum(valid_mask[keep_indices])
    if valid_keep < num_kept:
        # Find invalid tokens with highest original scores
        invalid_indices = np.where(~valid_mask)[0]
        invalid_scores = scores[invalid_indices]
        num_invalid_needed = num_kept - valid_keep

        if len(invalid_indices) > 0:
            # Sort invalid by their scores (descending)
            invalid_sorted = invalid_indices[np.argsort(-invalid_scores)]
            invalid_to_add = invalid_sorted[:num_invalid_needed]

            # Combine and re-sort
            keep_indices = np.concatenate([keep_indices[valid_mask[keep_indices]], invalid_to_add])
            keep_indices = np.sort(keep_indices)

    return keep_indices


# =============================================================================
# Token Pruner Class
# =============================================================================


class TokenPruner:
    """
    Token pruner with configuration management.

    This class provides a higher-level interface for token pruning,
    including configuration validation and model wrapping.
    """

    def __init__(self, config: Optional[PruningConfig] = None):
        """
        Initialize the token pruner.

        Args:
            config: Pruning configuration. Uses default (no pruning) if None.
        """
        self.config = config or PruningConfig()
        self._rng = np.random.default_rng(self.config.seed)

        logger.info(
            f"[TokenPruner] Initialized: method={self.config.method.value}, "
            f"keep_ratio={self.config.keep_ratio}, seed={self.config.seed}"
        )

    def prune_tokens(
        self,
        visual_tokens: torch.Tensor,
        scores: Optional[Union[TokenScores, np.ndarray, Dict[str, np.ndarray]]] = None,
    ) -> Tuple[torch.Tensor, PruningResult]:
        """
        Apply pruning to visual tokens.

        Args:
            visual_tokens: Visual token embeddings [batch, num_tokens, hidden_dim].
            scores: Token scores for importance-based pruning.

        Returns:
            Tuple of (pruned_visual_tokens, PruningResult).
        """
        result = prune_visual_tokens(
            visual_tokens=visual_tokens,
            scores=scores,
            keep_ratio=self.config.keep_ratio,
            method=self.config.method.value,
            seed=self.config.seed,
            debug=self.config.debug,
        )
        return result.pruned_visual_tokens, result

    def wrap_model(self, model: torch.nn.Module) -> "PruningModelWrapper":
        """
        Wrap a Prismatic VLM model to apply token pruning during forward pass.

        Args:
            model: The PrismaticForConditionalGeneration model to wrap.

        Returns:
            PruningModelWrapper that applies pruning.
        """
        return PruningModelWrapper(model, self)


class PruningModelWrapper:
    """
    Wrapper around PrismaticForConditionalGeneration that applies token pruning.

    This wrapper hooks into the forward pass to:
    1. Capture visual tokens after the projector
    2. Apply pruning based on configured method
    3. Return pruned tokens

    The wrapping is done by registering a forward hook on the projector module.
    """

    def __init__(self, model: torch.nn.Module, pruner: TokenPruner):
        """
        Initialize the wrapper.

        Args:
            model: The model to wrap.
            pruner: The TokenPruner instance.
        """
        self.model = model
        self.pruner = pruner
        self._hook_handle = None
        self._cached_result: Optional[PruningResult] = None
        self._current_scores: Optional[TokenScores] = None

        # Register forward hook on projector
        self._register_hook()

        logger.info(
            f"[PruningModelWrapper] Wrapped model with pruner: "
            f"method={pruner.config.method.value}, keep_ratio={pruner.config.keep_ratio}"
        )

    def _register_hook(self) -> None:
        """Register a forward hook on the projector module."""
        for name, module in self.model.named_modules():
            if name == "projector":
                self._hook_handle = module.register_forward_hook(self._projector_hook)
                logger.debug(f"[PruningModelWrapper] Hooked projector module")
                return

        logger.warning("[PruningModelWrapper] Could not find projector module to hook")

    def _projector_hook(
        self,
        module: torch.nn.Module,
        inputs: Tuple[torch.Tensor],
        output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward hook on the projector.

        Args:
            module: The projector module.
            inputs: Input tensors to the projector.
            output: Output tensor from the projector.

        Returns:
            Modified output (pruned embeddings) if pruning is enabled.
        """
        # Apply pruning
        result = prune_visual_tokens(
            visual_tokens=output,
            scores=self._current_scores,
            keep_ratio=self.pruner.config.keep_ratio,
            method=self.pruner.config.method.value,
            seed=self.pruner.config.seed,
            debug=self.pruner.config.debug,
        )

        # Cache result for later retrieval
        self._cached_result = result

        return result.pruned_visual_tokens

    def get_pruning_result(self) -> Optional[PruningResult]:
        """Get the most recent pruning result."""
        return self._cached_result

    def set_scores(self, scores: Optional[TokenScores]) -> None:
        """
        Set token scores for importance-based pruning.

        Args:
            scores: TokenScores from geometry expert, or None.
        """
        self._current_scores = scores

    def __del__(self):
        """Clean up hook on deletion."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def unwrap(self) -> torch.nn.Module:
        """
        Remove the pruning wrapper and return the original model.

        Returns:
            The original unwrapped model.
        """
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        logger.info("[PruningModelWrapper] Unwrapped model")
        return self.model


# =============================================================================
# Backward Compatibility
# =============================================================================


def apply_token_pruning(
    projected_patch_embeddings: torch.Tensor,
    keep_ratio: float,
    method: str = "random",
    seed: int = 42,
    scores: Optional[np.ndarray] = None,
    debug: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Backward-compatible wrapper for apply_token_pruning.

    Args:
        projected_patch_embeddings: Visual tokens [batch, num_tokens, hidden_dim].
        keep_ratio: Fraction of tokens to keep.
        method: Pruning method.
        seed: Random seed.
        scores: Token scores (will be wrapped in TokenScores).
        debug: Enable debug logging.

    Returns:
        Tuple of (pruned_embeddings, result_dict).
    """
    parsed_scores = None
    if scores is not None:
        parsed_scores = TokenScores.from_arrays(token_scores=scores)

    result = prune_visual_tokens(
        visual_tokens=projected_patch_embeddings,
        scores=parsed_scores,
        keep_ratio=keep_ratio,
        method=method,
        seed=seed,
        debug=debug,
    )

    return result.pruned_visual_tokens, result.get_summary()


# =============================================================================
# Debug/Test Functions
# =============================================================================


def debug_token_pruner():
    """
    Verify the token pruner correctness.

    Tests:
    1. No pruning (keep_ratio=1.0)
    2. Random pruning 50%
    3. Different keep ratios
    4. depth_edge pruning
    5. gripper_distance pruning
    6. geometry_rule pruning
    7. Invalid token handling
    8. Index sorting preservation
    """
    print("=" * 60)
    print("[TokenPruner] Running debug tests...")
    print("=" * 60)

    # Create test embeddings [batch=1, tokens=256, hidden=1024]
    batch_size, num_tokens, hidden_dim = 1, 256, 1024
    embeddings = torch.randn(batch_size, num_tokens, hidden_dim)

    # Create test scores
    np.random.seed(42)
    token_scores = np.random.rand(num_tokens).astype(np.float32)
    distance_scores = np.random.rand(num_tokens).astype(np.float32)
    direction_scores = np.random.rand(num_tokens).astype(np.float32)
    depth_edge_scores = np.random.rand(num_tokens).astype(np.float32)
    valid_mask = np.ones(num_tokens, dtype=np.bool_)

    # Make some tokens invalid
    valid_mask[50:60] = False

    # Test 1: No pruning
    print("\n[Test 1] No pruning (keep_ratio=1.0)")
    result = prune_visual_tokens(embeddings, None, keep_ratio=1.0, method="none")

    assert result.pruned_visual_tokens.shape == embeddings.shape
    assert result.num_tokens_after == num_tokens
    assert result.num_tokens_before == num_tokens
    assert result.actual_keep_ratio == 1.0
    assert len(result.keep_indices) == num_tokens
    print(f"  [OK] No pruning: shape={result.pruned_visual_tokens.shape}")

    # Test 2: Random pruning 50%
    print("\n[Test 2] Random pruning 50%")
    result = prune_visual_tokens(
        embeddings, None, keep_ratio=0.5, method="random", seed=42
    )

    expected_kept = int(num_tokens * 0.5)
    assert result.pruned_visual_tokens.shape == (batch_size, expected_kept, hidden_dim)
    assert result.num_tokens_after == expected_kept
    assert abs(result.actual_keep_ratio - 0.5) < 0.01
    print(f"  [OK] Random 50%: shape={result.pruned_visual_tokens.shape}")

    # Test 3: Different keep ratios
    print("\n[Test 3] Different keep ratios")
    for ratio in [0.75, 0.5, 0.25]:
        result = prune_visual_tokens(embeddings, None, keep_ratio=ratio, method="random", seed=123)
        expected_kept = int(num_tokens * ratio)
        assert result.num_tokens_after == expected_kept, f"keep_ratio={ratio}: expected {expected_kept}, got {result.num_tokens_after}"
        print(f"  [OK] keep_ratio={ratio}: keep={result.num_tokens_after}/{num_tokens}")

    # Test 4: depth_edge pruning
    print("\n[Test 4] depth_edge pruning")
    scores = TokenScores.from_arrays(
        token_scores=token_scores,
        distance_score=distance_scores,
        direction_score=direction_scores,
        depth_edge_score=depth_edge_scores,
        valid_mask=valid_mask,
    )

    # Make some depth_edge_scores very high
    scores.depth_edge_score[100:120] = 100.0

    result = prune_visual_tokens(
        embeddings, scores, keep_ratio=0.25, method="depth_edge", seed=42
    )

    expected_keep = int(num_tokens * 0.25)
    assert result.num_tokens_after == expected_keep
    # Top depth_edge scores should be in keep_indices
    top_indices = set(np.argsort(scores.depth_edge_score)[-expected_keep:])
    kept_set = set(result.keep_indices)
    print(f"  [OK] depth_edge: kept top {expected_keep} scores")

    # Test 4b: depth_edge_fast_diverse pruning
    print("\n[Test 4b] depth_edge_fast_diverse pruning")
    result = prune_visual_tokens(
        embeddings, scores, keep_ratio=0.75, method="depth_edge_fast_diverse", seed=42
    )
    assert result.num_tokens_after == 192
    assert len(result.keep_indices) == 192
    assert len(np.unique(result.keep_indices)) == 192
    assert np.array_equal(result.keep_indices, np.sort(result.keep_indices))
    meta = result.selection_metadata or {}
    assert meta.get("depth_quota") == 160
    assert meta.get("reserve_quota") == 32
    print(f"  [OK] depth_edge_fast_diverse: {meta}")

    # Test 5: gripper_distance pruning
    print("\n[Test 5] gripper_distance pruning")
    # Make some distance_scores very high (closer to gripper)
    scores.distance_score[50:70] = 100.0

    result = prune_visual_tokens(
        embeddings, scores, keep_ratio=0.25, method="gripper_distance", seed=42
    )

    expected_keep = int(num_tokens * 0.25)
    assert result.num_tokens_after == expected_keep
    print(f"  [OK] gripper_distance: kept {expected_keep} tokens")

    # Test 6: geometry_rule pruning
    print("\n[Test 6] geometry_rule pruning")
    # Make some token_scores very high
    scores.token_scores[150:170] = 100.0

    result = prune_visual_tokens(
        embeddings, scores, keep_ratio=0.25, method="geometry_rule", seed=42
    )

    expected_keep = int(num_tokens * 0.25)
    assert result.num_tokens_after == expected_keep
    print(f"  [OK] geometry_rule: kept {expected_keep} tokens")

    # Test 7: Invalid token handling
    print("\n[Test 7] Invalid token handling")
    scores.valid_mask[0:200] = False  # Only 56 valid tokens
    scores.token_scores[200:] = 100.0  # Invalid tokens have high scores

    result = prune_visual_tokens(
        embeddings, scores, keep_ratio=0.25, method="geometry_rule", seed=42
    )

    # Should prioritize valid tokens
    valid_kept = np.sum(scores.valid_mask[result.keep_indices])
    assert valid_kept >= 50, f"Should keep mostly valid tokens, got {valid_kept}"
    print(f"  [OK] Invalid handling: {valid_kept} valid tokens kept out of {result.num_tokens_after}")

    # Test 8: Index sorting
    print("\n[Test 8] Index sorting preservation")
    scores.valid_mask[:] = True
    scores.token_scores[:] = np.arange(num_tokens, dtype=np.float32)  # Increasing scores

    result = prune_visual_tokens(
        embeddings, scores, keep_ratio=0.5, method="geometry_rule", seed=42
    )

    # Should keep top 50% (indices 128-255), sorted
    assert np.array_equal(result.keep_indices, np.arange(128, 256))
    print(f"  [OK] Index sorting: keep_indices are sorted")

    # Test 9: Reproducibility with seed
    print("\n[Test 9] Reproducibility with seed")
    result1 = prune_visual_tokens(embeddings, None, keep_ratio=0.5, method="random", seed=999)
    result2 = prune_visual_tokens(embeddings, None, keep_ratio=0.5, method="random", seed=999)

    assert np.array_equal(result1.keep_indices, result2.keep_indices)
    print(f"  [OK] Reproducibility: seed=999 produces same indices")

    # Test 10: PruningResult summary
    print("\n[Test 10] PruningResult summary")
    summary = result.get_summary()
    assert "num_tokens_before" in summary
    assert "num_tokens_after" in summary
    assert "actual_keep_ratio" in summary
    assert "method" in summary
    print(f"  [OK] Summary: {summary}")

    print("\n" + "=" * 60)
    print("[TokenPruner] All debug tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    debug_token_pruner()
