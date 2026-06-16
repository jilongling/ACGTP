"""
RuleBasedGeometryExpert: Computes token importance scores based on 3D geometry.

This expert implements a rule-based scoring system for visual tokens using:
1. Distance score: How close the token's 3D point is to the gripper
2. Direction score: Whether the token lies in the gripper movement direction
3. Depth edge score: Tokens with high depth variance (likely edges)

This is a NO-LEARNING approach - all scores are computed from geometric rules.
No MLPs or learned parameters are used.

Score formula:
    score = w_distance * distance_score + w_direction * direction_score + w_edge * depth_edge_score

Where:
    distance_score = exp(-d^2 / sigma_d^2)
    direction_score = max(0, cosine(r, gripper_dir))
    depth_edge_score = normalized_depth_std

Usage:
    expert = RuleBasedGeometryExpert(
        w_distance=0.5,
        w_direction=0.3,
        w_edge=0.2,
        sigma_d=0.15,
    )
    result = expert.compute_scores(
        token_points_base=token_3d_points,
        token_valid_mask=valid_mask,
        depth_std=depth_variance,
        valid_depth_ratio=depth_ratio,
        gripper_pos=gripper_xyz,
        prev_gripper_pos=prev_gripper_xyz,
    )
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class GeometryScoreResult:
    """
    Result of computing geometry-based token importance scores.

    Attributes:
        token_scores: Combined importance scores [N].
        distance_score: Distance-based scores [N] (closer to gripper = higher).
        direction_score: Direction-based scores [N] (in gripper direction = higher).
        depth_edge_score: Depth edge scores [N] (high depth variance = higher).
        valid_score_mask: Boolean mask [N] for tokens with valid scores.
        score_stats: Dictionary with statistics for each score component.
        config: Configuration used for scoring.
    """

    token_scores: np.ndarray
    distance_score: np.ndarray
    direction_score: np.ndarray
    depth_edge_score: np.ndarray
    valid_score_mask: np.ndarray
    score_stats: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate shapes."""
        n = len(self.token_scores)
        for name, arr in [
            ("token_scores", self.token_scores),
            ("distance_score", self.distance_score),
            ("direction_score", self.direction_score),
            ("depth_edge_score", self.depth_edge_score),
            ("valid_score_mask", self.valid_score_mask),
        ]:
            if arr is not None and len(arr) != n:
                logger.warning(
                    f"[GeometryScoreResult] {name} has length {len(arr)} != {n}"
                )

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of the scoring result."""
        valid_count = np.sum(self.valid_score_mask) if self.valid_score_mask is not None else 0
        return {
            "num_tokens": len(self.token_scores),
            "valid_scores": int(valid_count),
            "score_mean": float(np.nanmean(self.token_scores)),
            "score_std": float(np.nanstd(self.token_scores)),
            "score_min": float(np.nanmin(self.token_scores)),
            "score_max": float(np.nanmax(self.token_scores)),
        }


@dataclass
class GeometryExpertConfig:
    """
    Configuration for the RuleBasedGeometryExpert.

    All weights should sum to 1.0 for proper normalization.
    """

    # Score component weights
    w_distance: float = 0.5
    w_direction: float = 0.3
    w_edge: float = 0.2

    # Distance score parameters
    sigma_d: float = 0.15  # Distance sigma in meters

    # Direction score parameters
    direction_threshold: float = 0.001  # Minimum gripper displacement to compute direction

    # Depth edge score parameters
    edge_normalization_percentile: float = 95.0  # Percentile for normalization

    # Validation
    min_valid_ratio: float = 0.1  # Minimum valid_depth_ratio to consider token

    def __post_init__(self):
        """Validate configuration."""
        total = self.w_distance + self.w_direction + self.w_edge
        if abs(total - 1.0) > 0.01:
            logger.warning(
                f"[GeometryExpertConfig] Weights sum to {total}, not 1.0. "
                "Scores may not be properly normalized."
            )


# =============================================================================
# Rule-Based Geometry Expert
# =============================================================================


class RuleBasedGeometryExpert:
    """
    Rule-based geometry expert for computing token importance scores.

    This expert uses geometric rules to score visual tokens based on their
    relationship to the robot gripper in 3D space. No learning is involved.

    Score Components:
    1. Distance Score: Tokens closer to gripper get higher scores
    2. Direction Score: Tokens in gripper movement direction get higher scores
    3. Depth Edge Score: Tokens with high depth variance (likely object edges) get higher scores

    The combined score is a weighted sum of these components.
    """

    def __init__(self, config: Optional[GeometryExpertConfig] = None, debug: bool = False):
        """
        Initialize the expert.

        Args:
            config: Configuration for scoring. Uses default if None.
            debug: Whether to enable debug logging.
        """
        self.config = config or GeometryExpertConfig()
        self.debug = debug

        logger.info(
            f"[RuleBasedGeometryExpert] Initialized with weights: "
            f"distance={self.config.w_distance}, direction={self.config.w_direction}, "
            f"edge={self.config.w_edge}, sigma_d={self.config.sigma_d}m"
        )

    def compute_scores(
        self,
        token_points_base: np.ndarray,
        token_valid_mask: np.ndarray,
        depth_std: np.ndarray,
        valid_depth_ratio: np.ndarray,
        gripper_pos: Optional[np.ndarray],
        prev_gripper_pos: Optional[np.ndarray] = None,
    ) -> GeometryScoreResult:
        """
        Compute geometry-based importance scores for all tokens.

        Args:
            token_points_base: 3D points [N, 3] in base frame (invalid = NaN).
            token_valid_mask: Boolean mask [N] indicating valid tokens.
            depth_std: Depth standard deviation [N] for each token's patch.
            valid_depth_ratio: Fraction [N] of valid depth pixels in patch.
            gripper_pos: Current gripper position [3] in base frame.
            prev_gripper_pos: Previous gripper position [3] in base frame.

        Returns:
            GeometryScoreResult with all score components.
        """
        n_tokens = len(token_valid_mask)

        # Initialize output arrays
        token_scores = np.zeros(n_tokens, dtype=np.float32)
        distance_score = np.zeros(n_tokens, dtype=np.float32)
        direction_score = np.zeros(n_tokens, dtype=np.float32)
        depth_edge_score = np.zeros(n_tokens, dtype=np.float32)
        valid_score_mask = np.zeros(n_tokens, dtype=np.bool_)

        # Build valid score mask
        # Token must have valid 3D point AND sufficient valid depth
        valid_depth_mask = valid_depth_ratio >= self.config.min_valid_ratio
        valid_score_mask = token_valid_mask & valid_depth_mask

        # Compute scores only for valid tokens
        valid_indices = np.where(valid_score_mask)[0]

        if len(valid_indices) == 0:
            logger.warning("[RuleBasedGeometryExpert] No valid tokens for scoring")
            return self._create_empty_result(n_tokens)

        if gripper_pos is None:
            logger.warning("[RuleBasedGeometryExpert] gripper_pos is None, using distance=0 for all")
            return self._create_empty_result(n_tokens)

        # Ensure gripper_pos is numpy array
        gripper_pos = np.asarray(gripper_pos, dtype=np.float32)

        # 1. Compute distance score
        distance_score = self._compute_distance_score(
            token_points_base, valid_indices, gripper_pos
        )

        # 2. Compute direction score
        direction_score = self._compute_direction_score(
            token_points_base, valid_indices, gripper_pos, prev_gripper_pos
        )

        # 3. Compute depth edge score
        depth_edge_score = self._compute_depth_edge_score(
            depth_std, valid_indices, valid_score_mask
        )

        # 4. Combine scores
        token_scores = (
            self.config.w_distance * distance_score +
            self.config.w_direction * direction_score +
            self.config.w_edge * depth_edge_score
        )

        # Set invalid tokens to 0
        token_scores[~valid_score_mask] = 0.0
        distance_score[~valid_score_mask] = 0.0
        direction_score[~valid_score_mask] = 0.0
        depth_edge_score[~valid_score_mask] = 0.0

        # Compute statistics
        score_stats = self._compute_score_stats(
            token_scores, distance_score, direction_score, depth_edge_score, valid_score_mask
        )

        if self.debug:
            logger.debug(
                f"[RuleBasedGeometryExpert] Scored {n_tokens} tokens, "
                f"{np.sum(valid_score_mask)} valid. "
                f"score_mean={score_stats['token_scores_mean']:.4f}, "
                f"score_max={score_stats['token_scores_max']:.4f}"
            )

        return GeometryScoreResult(
            token_scores=token_scores,
            distance_score=distance_score,
            direction_score=direction_score,
            depth_edge_score=depth_edge_score,
            valid_score_mask=valid_score_mask,
            score_stats=score_stats,
            config={
                "w_distance": self.config.w_distance,
                "w_direction": self.config.w_direction,
                "w_edge": self.config.w_edge,
                "sigma_d": self.config.sigma_d,
            },
        )

    def _compute_distance_score(
        self,
        token_points_base: np.ndarray,
        valid_indices: np.ndarray,
        gripper_pos: np.ndarray,
    ) -> np.ndarray:
        """
        Compute distance-based scores.

        Score = exp(-d^2 / sigma_d^2)
        where d is the Euclidean distance from token point to gripper.

        Closer tokens get higher scores (exponential decay with distance).
        """
        n = len(token_points_base)
        scores = np.zeros(n, dtype=np.float32)

        sigma_sq = self.config.sigma_d ** 2

        for idx in valid_indices:
            point = token_points_base[idx]
            if np.any(np.isnan(point)):
                continue

            # Euclidean distance
            d = np.linalg.norm(point - gripper_pos)

            # Gaussian kernel
            scores[idx] = math.exp(-(d ** 2) / sigma_sq)

        return scores

    def _compute_direction_score(
        self,
        token_points_base: np.ndarray,
        valid_indices: np.ndarray,
        gripper_pos: np.ndarray,
        prev_gripper_pos: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Compute direction-based scores.

        Score = max(0, cosine(r, gripper_dir))
        where:
            r = token_point - gripper_pos (vector from gripper to token)
            gripper_dir = gripper_pos - prev_gripper_pos (gripper movement direction)

        Tokens in the gripper's movement direction get higher scores.
        """
        n = len(token_points_base)
        scores = np.zeros(n, dtype=np.float32)

        if prev_gripper_pos is None:
            return scores

        prev_gripper_pos = np.asarray(prev_gripper_pos, dtype=np.float32)
        gripper_dir = gripper_pos - prev_gripper_pos

        # Check if gripper displacement is large enough
        dir_norm = np.linalg.norm(gripper_dir)
        if dir_norm < self.config.direction_threshold:
            return scores

        # Normalize direction
        gripper_dir = gripper_dir / dir_norm

        for idx in valid_indices:
            point = token_points_base[idx]
            if np.any(np.isnan(point)):
                continue

            # Vector from gripper to token
            r = point - gripper_pos
            r_norm = np.linalg.norm(r)

            if r_norm < 1e-6:
                continue

            # Cosine similarity
            cos_sim = np.dot(r, gripper_dir) / r_norm

            # Clamp to [0, 1]
            scores[idx] = max(0.0, cos_sim)

        return scores

    def _compute_depth_edge_score(
        self,
        depth_std: np.ndarray,
        valid_indices: np.ndarray,
        valid_score_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Compute depth edge-based scores.

        Score = normalized_depth_std
        where normalization maps depth_std to [0, 1] using percentile scaling.

        Tokens with high depth variance (likely object edges) get higher scores.
        """
        n = len(depth_std)
        scores = np.zeros(n, dtype=np.float32)

        # Get depth std values for valid tokens
        valid_std_values = depth_std[valid_score_mask]

        if len(valid_std_values) == 0:
            return scores

        # Compute percentile-based normalization
        # Map values to [0, 1] where 1 = highest depth variance
        percentile = self.config.edge_normalization_percentile
        max_std = np.percentile(valid_std_values, percentile)
        max_std = max(max_std, 1e-6)  # Avoid division by zero

        for idx in valid_indices:
            std_val = depth_std[idx]
            if np.isnan(std_val):
                continue

            # Normalize to [0, 1]
            scores[idx] = min(1.0, std_val / max_std)

        return scores

    def _compute_score_stats(
        self,
        token_scores: np.ndarray,
        distance_score: np.ndarray,
        direction_score: np.ndarray,
        depth_edge_score: np.ndarray,
        valid_mask: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute statistics for all score components."""
        valid_scores = token_scores[valid_mask]
        valid_dist = distance_score[valid_mask]
        valid_dir = direction_score[valid_mask]
        valid_edge = depth_edge_score[valid_mask]

        def safe_stats(arr: np.ndarray) -> Dict[str, float]:
            if len(arr) == 0:
                return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
            return {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }

        return {
            "token_scores_mean": float(np.mean(valid_scores)) if len(valid_scores) > 0 else 0.0,
            "token_scores_std": float(np.std(valid_scores)) if len(valid_scores) > 0 else 0.0,
            "token_scores_min": float(np.min(valid_scores)) if len(valid_scores) > 0 else 0.0,
            "token_scores_max": float(np.max(valid_scores)) if len(valid_scores) > 0 else 0.0,
            "num_valid": int(np.sum(valid_mask)),
            "num_invalid": int(np.sum(~valid_mask)),
            "distance_score": safe_stats(valid_dist),
            "direction_score": safe_stats(valid_dir),
            "depth_edge_score": safe_stats(valid_edge),
        }

    def _create_empty_result(self, n_tokens: int) -> GeometryScoreResult:
        """Create an empty result when scoring fails."""
        return GeometryScoreResult(
            token_scores=np.zeros(n_tokens, dtype=np.float32),
            distance_score=np.zeros(n_tokens, dtype=np.float32),
            direction_score=np.zeros(n_tokens, dtype=np.float32),
            depth_edge_score=np.zeros(n_tokens, dtype=np.float32),
            valid_score_mask=np.zeros(n_tokens, dtype=np.bool_),
            score_stats={"error": "No valid tokens or gripper_pos"},
            config={
                "w_distance": self.config.w_distance,
                "w_direction": self.config.w_direction,
                "w_edge": self.config.w_edge,
                "sigma_d": self.config.sigma_d,
            },
        )


# =============================================================================
# Debug/Test Functions
# =============================================================================


def debug_geometry_expert():
    """
    Verify the geometry expert correctness.

    Tests:
    1. Distance score decreases with distance
    2. Direction score detects movement direction
    3. Depth edge score responds to depth variance
    4. Invalid tokens get zero scores
    """
    print("=" * 60)
    print("[RuleBasedGeometryExpert] Running debug tests...")
    print("=" * 60)

    # Create expert with default config
    expert = RuleBasedGeometryExpert(debug=True)
    n_tokens = 256

    # Create test data
    token_points = np.zeros((n_tokens, 3), dtype=np.float32)
    token_valid = np.ones(n_tokens, dtype=np.bool_)
    depth_std = np.ones(n_tokens, dtype=np.float32) * 0.1
    valid_depth_ratio = np.ones(n_tokens, dtype=np.float32)

    # Place gripper at origin
    gripper_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    prev_gripper_pos = np.array([-0.1, 0.0, 0.0], dtype=np.float32)

    # Test 1: Distance score
    print("\n[Test 1] Distance score")
    # Use larger sigma for testing, or tokens very close to gripper
    test_config = GeometryExpertConfig(w_distance=0.5, w_direction=0.3, w_edge=0.2, sigma_d=0.5)
    test_expert = RuleBasedGeometryExpert(config=test_config, debug=False)

    token_points[:, 2] = 1.0  # All tokens at z=1m
    token_points[:, 0] = np.linspace(0, 1.0, n_tokens)  # x from 0 to 1m

    result = test_expert.compute_scores(
        token_points_base=token_points,
        token_valid_mask=token_valid,
        depth_std=depth_std,
        valid_depth_ratio=valid_depth_ratio,
        gripper_pos=gripper_pos,
        prev_gripper_pos=prev_gripper_pos,
    )

    # Check that closer tokens have higher scores
    # With sigma=0.5 and max distance ~1.4m, we should see variation
    print(f"  Distance scores: d0={result.distance_score[0]:.4f}, "
          f"d128={result.distance_score[128]:.4f}, d255={result.distance_score[-1]:.4f}")

    # First token (x=0) is at (0,0,1), distance to (0,0,0) = 1m
    # Token at x=0.5 is at (0.5,0,1), distance ~1.12m
    # Both should have low but decreasing scores
    assert result.distance_score[0] >= result.distance_score[128], \
        "Closer tokens should have higher or equal distance scores"
    assert result.distance_score[0] >= result.distance_score[-1], \
        "Closest token should have highest or equal distance score"
    print(f"  [OK] Distance score decreases with distance")

    # Check that combined score is bounded by weights
    max_possible = test_config.w_distance + test_config.w_direction + test_config.w_edge
    assert result.token_scores[0] <= max_possible + 0.01, \
        f"Max score {result.token_scores[0]} should be <= weights sum {max_possible}"
    print(f"  [OK] Score bounded by weights: max={result.token_scores[0]:.4f}")

    # Test 2: Direction score
    print("\n[Test 2] Direction score")
    # Reset: all tokens at same distance, varying x position
    token_points[:, 0] = np.linspace(-0.2, 0.2, n_tokens)  # x from -0.2 to 0.2
    token_points[:, 1] = 0.0
    token_points[:, 2] = 1.0

    # Gripper moving in +x direction
    gripper_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    prev_gripper_pos = np.array([-0.1, 0.0, 1.0], dtype=np.float32)

    result = expert.compute_scores(
        token_points_base=token_points,
        token_valid_mask=token_valid,
        depth_std=depth_std,
        valid_depth_ratio=valid_depth_ratio,
        gripper_pos=gripper_pos,
        prev_gripper_pos=prev_gripper_pos,
    )

    # Tokens in +x direction (gripper movement) should have higher direction scores
    # Token at x=0.1 is in movement direction
    # Token at x=-0.1 is opposite to movement direction
    assert result.direction_score[192] > result.direction_score[64], \
        "Tokens in movement direction should have higher scores"
    print(f"  [OK] Direction score higher in movement direction: "
          f"fwd={result.direction_score[192]:.4f}, bwd={result.direction_score[64]:.4f}")

    # Test 3: Depth edge score
    print("\n[Test 3] Depth edge score")
    depth_std = np.zeros(n_tokens, dtype=np.float32)
    depth_std[:128] = 0.05  # Low variance
    depth_std[128:] = 0.5   # High variance

    result = expert.compute_scores(
        token_points_base=token_points,
        token_valid_mask=token_valid,
        depth_std=depth_std,
        valid_depth_ratio=valid_depth_ratio,
        gripper_pos=gripper_pos,
        prev_gripper_pos=None,  # No direction score
    )

    # High depth std tokens should have higher edge scores
    assert result.depth_edge_score[200] > result.depth_edge_score[50], \
        "High depth variance should have higher edge scores"
    print(f"  [OK] Depth edge score higher for high variance: "
          f"high={result.depth_edge_score[200]:.4f}, low={result.depth_edge_score[50]:.4f}")

    # Test 4: Invalid tokens
    print("\n[Test 4] Invalid tokens")
    token_valid = np.ones(n_tokens, dtype=np.bool_)
    token_valid[100:150] = False  # Invalid range

    result = expert.compute_scores(
        token_points_base=token_points,
        token_valid_mask=token_valid,
        depth_std=depth_std,
        valid_depth_ratio=valid_depth_ratio,
        gripper_pos=gripper_pos,
        prev_gripper_pos=None,
    )

    # Invalid tokens should have zero scores
    assert np.all(result.token_scores[100:150] == 0.0), \
        "Invalid tokens should have zero scores"
    print(f"  [OK] Invalid tokens have zero scores")

    # Test 5: No prev_gripper_pos
    print("\n[Test 5] No previous gripper position")
    result = expert.compute_scores(
        token_points_base=token_points,
        token_valid_mask=token_valid,
        depth_std=depth_std,
        valid_depth_ratio=valid_depth_ratio,
        gripper_pos=gripper_pos,
        prev_gripper_pos=None,
    )

    # Direction score should be all zeros
    assert np.all(result.direction_score == 0.0), \
        "Direction score should be 0 without prev_gripper_pos"
    print(f"  [OK] Direction score is 0 without prev_gripper_pos")

    # Test 6: Score statistics
    print("\n[Test 6] Score statistics")
    assert "token_scores_mean" in result.score_stats
    assert "distance_score" in result.score_stats
    assert "direction_score" in result.score_stats
    assert "depth_edge_score" in result.score_stats
    print(f"  [OK] Score statistics computed: mean={result.score_stats['token_scores_mean']:.4f}")

    print("\n" + "=" * 60)
    print("[RuleBasedGeometryExpert] All debug tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    debug_geometry_expert()
