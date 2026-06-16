"""Action and motion score signals."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Source: pruning/scores/action_constraint.py
# ---------------------------------------------------------------------------
from typing import Any, Dict, Optional

import numpy as np


def _to_1d(arr: Optional[np.ndarray], n: Optional[int] = None) -> Optional[np.ndarray]:
    if arr is None:
        return None
    out = np.asarray(arr, dtype=np.float32).reshape(-1)
    if n is not None and out.shape[0] != n:
        return None
    return out


def _norm01(arr: Optional[np.ndarray], n: int, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    a = _to_1d(arr, n)
    if a is None:
        return out
    valid = np.ones(n, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool).reshape(-1)
    if valid.shape[0] != n:
        valid = np.ones(n, dtype=bool)
    finite = valid & np.isfinite(a)
    if not np.any(finite):
        return out
    vals = a[finite]
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if hi - lo > 1e-8:
        out[finite] = (a[finite] - lo) / (hi - lo)
    else:
        out[finite] = np.clip(a[finite], 0.0, 1.0)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _scene_component(scene_result: Optional[Dict[str, Any]], key: str, n: int) -> np.ndarray:
    if not scene_result:
        return np.zeros(n, dtype=np.float32)
    return _norm01(scene_result.get(key), n)


def compute_future_action_constraint_scores(
    *,
    scene_layout_scores: Optional[np.ndarray],
    depth_structure_scores: Optional[np.ndarray],
    contact_ring_scores: Optional[np.ndarray],
    motion_corridor_scores: Optional[np.ndarray],
    valid_mask: Optional[np.ndarray],
    robot_self_core_mask: Optional[np.ndarray] = None,
    scene_result: Optional[Dict[str, Any]] = None,
    contact_result: Optional[Dict[str, Any]] = None,
    motion_result: Optional[Dict[str, Any]] = None,
    w_scene: float = 0.30,
    w_depth: float = 0.25,
    w_contact: float = 0.25,
    w_motion: float = 0.20,
    robot_self_penalty: float = 0.35,
) -> Dict[str, Any]:
    """Compute per-token future action-constraint relevance.

    The score is:

        ACR = scene + depth + object-side contact + swept motion risk - self

    Contact and motion are refined as soft OR/mixture signals, not products.
    This keeps contact/collision evidence alive when one noisy branch is weak.
    """
    n = 0
    for arr in (scene_layout_scores, depth_structure_scores, contact_ring_scores, motion_corridor_scores):
        if arr is not None:
            n = max(n, int(np.asarray(arr).size))
    if n <= 0:
        return {
            "action_constraint_scores": np.array([], dtype=np.float32),
            "object_side_contact_scores": np.array([], dtype=np.float32),
            "swept_motion_risk_scores": np.array([], dtype=np.float32),
            "collision_contact_risk_scores": np.array([], dtype=np.float32),
            "action_constraint_valid": False,
            "action_constraint_disabled_reason": "no_scores",
        }

    valid = np.ones(n, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool).reshape(-1)
    if valid.shape[0] != n:
        valid = np.ones(n, dtype=bool)

    scene = _norm01(scene_layout_scores, n, valid)
    depth = _norm01(depth_structure_scores, n, valid)
    contact = _norm01(contact_ring_scores, n, valid)
    motion = _norm01(motion_corridor_scores, n, valid)

    object_component = _scene_component(scene_result, "object_component_scores", n)
    boundary = _scene_component(scene_result, "boundary_scores", n)
    support = _scene_component(scene_result, "support_plane_candidate_scores", n)
    object_boundary = np.maximum(object_component, boundary)

    self_core = _to_1d(robot_self_core_mask, n)
    if self_core is None and contact_result is not None:
        self_core = _to_1d(contact_result.get("robot_self_core_mask"), n)
    if self_core is None:
        self_core_mask = np.zeros(n, dtype=bool)
    else:
        self_core_mask = np.asarray(self_core > 0.5, dtype=bool)

    ring_mask = None
    if contact_result is not None:
        ring_mask = _to_1d(contact_result.get("contact_ring_mask"), n)
    if ring_mask is None:
        ring = (contact > 0.0).astype(np.float32)
    else:
        ring = (ring_mask > 0.5).astype(np.float32)

    # Object-side contact: the ring matters most when it touches object,
    # boundary, or depth-discontinuity evidence. Additive mixture avoids the
    # brittle near * edge collapse.
    contact_gate = np.maximum.reduce([object_boundary, depth, scene])
    object_side_contact = ring * (
        0.45 * contact
        + 0.35 * contact_gate
        + 0.20 * depth
    )
    object_side_contact = np.clip(object_side_contact, 0.0, 1.0).astype(np.float32)
    object_side_contact[self_core_mask] = 0.0

    motion_valid = bool(motion_result.get("motion_corridor_valid", False)) if motion_result else bool(np.any(motion > 0.0))
    if motion_valid:
        swept_motion_risk = (
            0.60 * motion
            + 0.25 * depth
            + 0.15 * np.maximum(object_boundary, support)
        )
    else:
        swept_motion_risk = np.zeros(n, dtype=np.float32)
    swept_motion_risk = np.clip(swept_motion_risk, 0.0, 1.0).astype(np.float32)
    swept_motion_risk[self_core_mask] = 0.0

    collision_contact_risk = np.maximum(object_side_contact, swept_motion_risk)

    weight_sum = max(1e-8, float(w_scene + w_depth + w_contact + w_motion))
    action_constraint = (
        float(w_scene) * scene
        + float(w_depth) * depth
        + float(w_contact) * object_side_contact
        + float(w_motion) * swept_motion_risk
    ) / weight_sum
    action_constraint = action_constraint - float(robot_self_penalty) * self_core_mask.astype(np.float32)
    action_constraint[~valid] = 0.0
    action_constraint = np.clip(action_constraint, 0.0, 1.0).astype(np.float32)

    contact_overlap = int(np.sum((object_side_contact > 0.0) & (object_boundary > 0.0)))
    self_penalty_count = int(np.sum(self_core_mask))

    return {
        "action_constraint_scores": action_constraint,
        "object_side_contact_scores": object_side_contact,
        "swept_motion_risk_scores": swept_motion_risk,
        "collision_contact_risk_scores": collision_contact_risk.astype(np.float32),
        "action_constraint_valid": bool(np.any(action_constraint[valid] > 0.0)) if np.any(valid) else False,
        "action_constraint_disabled_reason": None if np.any(action_constraint[valid] > 0.0) else "all_zero",
        "action_constraint_score_mean": float(np.mean(action_constraint[valid])) if np.any(valid) else 0.0,
        "action_constraint_score_max": float(np.max(action_constraint[valid])) if np.any(valid) else 0.0,
        "object_side_contact_score_mean": float(np.mean(object_side_contact[valid])) if np.any(valid) else 0.0,
        "object_side_contact_score_max": float(np.max(object_side_contact[valid])) if np.any(valid) else 0.0,
        "swept_motion_risk_score_mean": float(np.mean(swept_motion_risk[valid])) if np.any(valid) else 0.0,
        "swept_motion_risk_score_max": float(np.max(swept_motion_risk[valid])) if np.any(valid) else 0.0,
        "collision_contact_risk_score_mean": float(np.mean(collision_contact_risk[valid])) if np.any(valid) else 0.0,
        "collision_contact_risk_score_max": float(np.max(collision_contact_risk[valid])) if np.any(valid) else 0.0,
        "contact_object_overlap_count": contact_overlap,
        "robot_self_penalty_count": self_penalty_count,
    }

# ---------------------------------------------------------------------------
# Source: pruning/scores/motion_corridor.py
# ---------------------------------------------------------------------------
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import numpy as np


class MotionEMABuffer:
    """Exponential moving average buffer for motion direction smoothing.

    Stores recent gripper positions and computes smoothed motion direction.
    """

    def __init__(self, maxlen: int = 5, ema_alpha: float = 0.6) -> None:
        self.maxlen = maxlen
        self.ema_alpha = ema_alpha
        self._positions: Deque[np.ndarray] = deque(maxlen=maxlen)
        self._ema_dir: Optional[np.ndarray] = None
        self._ema_norm: float = 0.0

    def update(self, gripper_pos: np.ndarray) -> None:
        pos = np.asarray(gripper_pos, dtype=np.float64)
        if pos.shape != (3,):
            return
        self._positions.append(pos)

    def get_smoothed_motion(self) -> tuple[Optional[np.ndarray], float]:
        """Returns (smoothed_motion_direction, motion_norm) or (None, 0.0) if insufficient data."""
        if len(self._positions) < 2:
            return None, 0.0

        if self._ema_dir is None or len(self._positions) < 3:
            recent = list(self._positions)
            motion = recent[-1] - recent[0]
            norm = float(np.linalg.norm(motion))
            if norm > 1e-8:
                self._ema_dir = motion / norm
                self._ema_norm = norm
            return self._ema_dir, self._ema_norm

        direction, norm = self._compute_ema_motion()
        return direction, norm

    def _compute_ema_motion(self) -> tuple[Optional[np.ndarray], float]:
        positions = list(self._positions)
        if len(positions) < 2:
            return None, 0.0

        alpha = self.ema_alpha
        ema_dir = None
        ema_norm = 0.0

        motion_0 = positions[-1] - positions[-2]
        norm_0 = float(np.linalg.norm(motion_0))

        if norm_0 > 1e-8:
            dir_0 = motion_0 / norm_0
        else:
            dir_0 = None

        if self._ema_dir is None:
            if dir_0 is not None:
                ema_dir = dir_0
                ema_norm = norm_0
        else:
            if dir_0 is not None:
                cos_sim = float(np.dot(self._ema_dir, dir_0))
                cos_sim = np.clip(cos_sim, -1.0, 1.0)
                aligned = cos_sim > 0.3
                ema_dir = aligned * (alpha * dir_0 + (1 - alpha) * self._ema_dir)
                norm_d = float(np.linalg.norm(ema_dir))
                if norm_d > 1e-8:
                    ema_dir = ema_dir / norm_d
                ema_norm = alpha * norm_0 + (1 - alpha) * self._ema_norm
            else:
                ema_dir = self._ema_dir
                ema_norm = (1 - alpha) * self._ema_norm

        self._ema_dir = ema_dir
        self._ema_norm = ema_norm
        return ema_dir, ema_norm

    def __len__(self) -> int:
        return len(self._positions)

    def reset(self) -> None:
        self._positions.clear()
        self._ema_dir = None
        self._ema_norm = 0.0


def compute_motion_corridor_scores(
    points_robot: np.ndarray,
    gripper_pos: np.ndarray,
    prev_gripper_pos: Optional[np.ndarray],
    depth_edge_scores: Optional[np.ndarray] = None,
    *,
    motion_buffer: Optional[MotionEMABuffer] = None,
    corridor_length_m: float = 0.15,
    corridor_sigma_m: float = 0.06,
    min_motion_norm: float = 1e-4,
    ema_alpha: float = 0.6,
    depth_discontinuity_threshold_m: float = 0.05,
) -> Dict[str, Any]:
    """Compute smoothed motion corridor scores.

    Args:
        points_robot: [N, 3] token positions in robot frame.
        gripper_pos:  [3] current gripper position.
        prev_gripper_pos: [3] previous gripper position (optional).
        depth_edge_scores: [N] depth edge scores for discontinuity detection.
        motion_buffer: MotionEMABuffer for smoothed motion direction (optional).
        corridor_length_m: Length of swept path in meters.
        corridor_sigma_m: Gaussian sigma for corridor width in meters.
        min_motion_norm: Minimum motion norm to consider corridor valid.
        ema_alpha: EMA smoothing factor.
        depth_discontinuity_threshold_m: Threshold for depth discontinuity on path.

    Returns:
        Dict with keys:
          motion_corridor_scores: [N] final motion corridor scores [0, 1].
          motion_corridor_valid: bool indicating whether signal is reliable.
          motion_direction_valid: bool.
          motion_norm_m: float.
          swept_corridor_score_mean: float.
          swept_corridor_score_max: float.
          depth_discontinuity_score_mean: float.
          depth_discontinuity_score_max: float.
          motion_disabled_reason: str or None.
          ema_enabled: bool.
    """
    result: Dict[str, Any] = {
        "motion_corridor_scores": np.array([]),
        "motion_corridor_valid": False,
        "motion_direction_valid": False,
        "motion_norm_m": 0.0,
        "swept_corridor_score_mean": 0.0,
        "swept_corridor_score_max": 0.0,
        "depth_discontinuity_score_mean": 0.0,
        "depth_discontinuity_score_max": 0.0,
        "motion_disabled_reason": None,
        "ema_enabled": False,
        "ema_alpha": float(ema_alpha),
    }

    n = int(points_robot.shape[0]) if points_robot.ndim == 2 else 0
    if n == 0 or gripper_pos is None:
        result["motion_disabled_reason"] = "no_points_or_gripper"
        return result

    p_r = np.asarray(points_robot, dtype=np.float64)
    grip = np.asarray(gripper_pos, dtype=np.float64)

    if p_r.ndim == 1:
        p_r = p_r.reshape(1, -1)
    if p_r.shape[1] != 3:
        result["motion_disabled_reason"] = "invalid_points_shape"
        return result

    # ── Motion direction computation ──────────────────────────────────────────
    motion_valid = False
    motion_dir = None
    motion_norm = 0.0

    if motion_buffer is not None:
        motion_buffer.update(grip)
        motion_dir, motion_norm = motion_buffer.get_smoothed_motion()
        motion_valid = motion_norm > min_motion_norm
    elif prev_gripper_pos is not None:
        prev = np.asarray(prev_gripper_pos, dtype=np.float64)
        if prev.shape == (3,):
            raw_motion = grip - prev
            motion_norm = float(np.linalg.norm(raw_motion))
            if motion_norm > min_motion_norm:
                motion_dir = raw_motion / motion_norm
                motion_valid = True

    result["motion_direction_valid"] = motion_valid
    result["motion_norm_m"] = motion_norm
    result["ema_enabled"] = motion_buffer is not None

    if not motion_valid:
        result["motion_disabled_reason"] = (
            "step_0_or_1" if motion_buffer is not None and len(motion_buffer) < 2
            else "motion_norm_too_small" if motion_norm <= min_motion_norm
            else "no_prev_gripper_pose"
        )
        # Return zero scores, NOT misleading high scores
        result["motion_corridor_scores"] = np.zeros(n, dtype=np.float32)
        return result

    # ── Swept corridor score ──────────────────────────────────────────────────
    offset = p_r - grip
    proj_scalar = np.sum(offset * motion_dir * np.ones((n, 1)), axis=1)  # noqa: N806

    # Distance from swept path (perpendicular to motion direction)
    # For 3D, compute perpendicular distance
    offset_along = proj_scalar
    offset_perp_sq = np.sum(offset ** 2, axis=1) - offset_along ** 2
    offset_perp_sq = np.maximum(offset_perp_sq, 0.0)
    offset_perp = np.sqrt(offset_perp_sq)

    # Tokens on the swept path: projected distance <= corridor_length AND perp dist <= sigma
    corridor_length = corridor_length_m
    corridor_sigma = corridor_sigma_m

    along_dist = np.abs(offset_along)
    swept_along = np.exp(-((along_dist - corridor_length) ** 2) / (2 * corridor_sigma ** 2))
    swept_along[offset_along < 0] *= 0.3  # Behind the gripper gets lower score

    swept_perp = np.exp(-(offset_perp ** 2) / (2 * (corridor_sigma * 2) ** 2))
    swept_corridor = swept_along * swept_perp

    # Also check: is the projected point in front of the gripper?
    in_front = offset_along >= 0
    swept_corridor = swept_corridor * np.where(in_front, 1.0, 0.3)

    swept_corridor = np.clip(swept_corridor, 0.0, 1.0).astype(np.float32)
    result["swept_corridor_score_mean"] = float(np.mean(swept_corridor))
    result["swept_corridor_score_max"] = float(np.max(swept_corridor))

    # ── Depth discontinuity on path ─────────────────────────────────────────
    depth_discontinuity = np.zeros(n, dtype=np.float32)
    if depth_edge_scores is not None:
        de = np.asarray(depth_edge_scores, dtype=np.float32).reshape(-1)
        if de.shape[0] == n:
            depth_discontinuity = np.clip(de, 0.0, 1.0)

    result["depth_discontinuity_score_mean"] = float(np.mean(depth_discontinuity))
    result["depth_discontinuity_score_max"] = float(np.max(depth_discontinuity))

    # ── Final motion_corridor_score: mixture ───────────────────────────────
    # Contact-risk = a * swept_corridor + b * depth_discontinuity
    # Where a, b are implicit weights that produce scores in [0, 1]
    motion_corridor_scores = (
        0.7 * swept_corridor.astype(np.float32)
        + 0.3 * depth_discontinuity
    )
    motion_corridor_scores = np.clip(motion_corridor_scores, 0.0, 1.0).astype(np.float32)

    result["motion_corridor_scores"] = motion_corridor_scores
    result["motion_corridor_valid"] = True
    result["motion_disabled_reason"] = None

    return result


def create_motion_buffer(maxlen: int = 5, ema_alpha: float = 0.6) -> MotionEMABuffer:
    return MotionEMABuffer(maxlen=maxlen, ema_alpha=ema_alpha)
