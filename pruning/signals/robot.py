"""Robot, token geometry, and geometry-cache helpers."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Source: pruning/geometry/robot_state.py
# ---------------------------------------------------------------------------
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


EE_POSITION_KEYS = (
    "ee_position",
    "ee_pos",
    "eef_pos",
    "end_effector_pos",
    "end_effector_position",
    "gripper_pos",
    "gripper_position",
    "robot0_eef_pos",
)
EE_ORIENTATION_KEYS = (
    "ee_orientation",
    "ee_quat",
    "eef_quat",
    "end_effector_quat",
    "gripper_quat",
    "robot0_eef_quat",
    "ee_rot",
    "eef_rot",
    "ee_rotation_matrix",
)
GRIPPER_WIDTH_KEYS = (
    "gripper_width",
    "gripper_qpos",
    "robot0_gripper_qpos",
)
GRIPPER_OPEN_KEYS = (
    "gripper_open",
    "is_gripper_open",
)
ACTION_DELTA_KEYS = (
    "action_delta",
    "delta_action",
    "eef_delta",
    "ee_delta",
)
NESTED_STATE_KEYS = (
    "robot_state",
    "robot_obs",
    "proprio",
    "proprioception",
    "state",
)


@dataclass
class RobotState:
    ee_position: Optional[torch.Tensor] = None
    ee_orientation: Optional[torch.Tensor] = None
    gripper_width: Optional[float] = None
    gripper_open: Optional[bool] = None
    action_delta: Optional[torch.Tensor] = None
    prev_ee_position: Optional[torch.Tensor] = None
    frame: str = "unknown"
    valid: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


def extract_robot_state_from_obs(obs: Dict[str, Any]) -> RobotState:
    """Best-effort robot-state extraction from flat or nested observations."""
    if obs is None or not isinstance(obs, dict):
        return RobotState(valid=False, metadata={"missing_reason": "obs_not_dict"})

    ee_position, pos_key = _find_tensor(obs, EE_POSITION_KEYS, expected_shapes=((3,),))
    ee_orientation, ori_key = _find_tensor(obs, EE_ORIENTATION_KEYS, expected_shapes=((4,), (3, 3)))
    action_delta, action_key = _find_tensor(obs, ACTION_DELTA_KEYS)
    gripper_width, width_key = _find_float(obs, GRIPPER_WIDTH_KEYS)
    gripper_open, open_key = _find_bool(obs, GRIPPER_OPEN_KEYS)

    if gripper_open is None and gripper_width is not None:
        gripper_open = bool(gripper_width > 0.0)
        open_key = f"inferred_from:{width_key}"

    frame = _find_frame(obs) or "unknown"
    valid = ee_position is not None
    metadata = {
        "position_key": pos_key,
        "orientation_key": ori_key,
        "gripper_width_key": width_key,
        "gripper_open_key": open_key,
        "action_delta_key": action_key,
        "searched_position_keys": list(EE_POSITION_KEYS),
        "searched_nested_keys": list(NESTED_STATE_KEYS),
    }
    if not valid:
        metadata["missing_reason"] = "missing_ee_position"

    return RobotState(
        ee_position=ee_position,
        ee_orientation=ee_orientation,
        gripper_width=gripper_width,
        gripper_open=gripper_open,
        action_delta=action_delta,
        frame=frame,
        valid=valid,
        metadata=metadata,
    )


def estimate_motion_direction(
    robot_state: RobotState,
    prev_robot_state: Optional[RobotState] = None,
    prev_action: Optional[Any] = None,
    *,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, bool]:
    """Estimate a unit motion direction from state difference or action delta."""
    if (
        robot_state is not None
        and robot_state.ee_position is not None
        and prev_robot_state is not None
        and prev_robot_state.ee_position is not None
    ):
        delta = robot_state.ee_position - prev_robot_state.ee_position
    elif robot_state is not None and robot_state.prev_ee_position is not None and robot_state.ee_position is not None:
        delta = robot_state.ee_position - robot_state.prev_ee_position
    elif robot_state is not None and robot_state.action_delta is not None and robot_state.action_delta.numel() >= 3:
        delta = robot_state.action_delta.reshape(-1)[:3]
    elif prev_action is not None:
        delta = _as_tensor(prev_action).reshape(-1)[:3]
    else:
        return torch.zeros(3, dtype=torch.float32), False

    delta = delta.to(dtype=torch.float32).reshape(-1)[:3]
    if delta.numel() < 3 or not bool(torch.isfinite(delta).all()):
        return torch.zeros(3, dtype=torch.float32, device=delta.device), False
    norm = torch.linalg.norm(delta)
    if not bool(torch.isfinite(norm)) or float(norm.item()) <= float(eps):
        return torch.zeros(3, dtype=torch.float32, device=delta.device), False
    return delta / norm, True


def transform_robot_state_frame(
    robot_state: RobotState,
    T_target_source: Optional[Any],
    *,
    target_frame: str = "robot",
) -> RobotState:
    """Transform position and matrix orientation into a target frame.

    Quaternion conventions differ across simulators, so 4D quaternion
    orientations are preserved and annotated rather than silently rotated.
    """
    if robot_state is None:
        return RobotState(valid=False, metadata={"missing_reason": "robot_state_none"})
    if T_target_source is None:
        meta = dict(robot_state.metadata)
        meta["transform_applied"] = False
        meta["transform_reason"] = "missing_T_target_source"
        return RobotState(**{**robot_state.__dict__, "metadata": meta})

    T = _as_tensor(T_target_source, dtype=torch.float32)
    if tuple(T.shape) != (4, 4) or not bool(torch.isfinite(T).all()):
        meta = dict(robot_state.metadata)
        meta["transform_applied"] = False
        meta["transform_reason"] = "invalid_T_target_source"
        return RobotState(**{**robot_state.__dict__, "metadata": meta})

    R = T[:3, :3]
    t = T[:3, 3]
    ee_position = None
    prev_ee_position = None
    if robot_state.ee_position is not None:
        pos = robot_state.ee_position.to(dtype=torch.float32, device=T.device).reshape(3)
        ee_position = R @ pos + t
    if robot_state.prev_ee_position is not None:
        prev = robot_state.prev_ee_position.to(dtype=torch.float32, device=T.device).reshape(3)
        prev_ee_position = R @ prev + t

    ee_orientation = robot_state.ee_orientation
    orientation_note = None
    if ee_orientation is not None:
        ori = ee_orientation.to(dtype=torch.float32, device=T.device)
        if tuple(ori.shape) == (3, 3):
            ee_orientation = R @ ori
        elif ori.numel() == 4:
            ee_orientation = ori.reshape(4)
            orientation_note = "quaternion_preserved_convention_unknown"

    action_delta = robot_state.action_delta
    if action_delta is not None and action_delta.numel() >= 3:
        delta = action_delta.to(dtype=torch.float32, device=T.device).reshape(-1)
        xyz = R @ delta[:3]
        action_delta = torch.cat([xyz, delta[3:]], dim=0) if delta.numel() > 3 else xyz

    meta = dict(robot_state.metadata)
    meta.update({
        "transform_applied": True,
        "source_frame": robot_state.frame,
        "target_frame": target_frame,
    })
    if orientation_note is not None:
        meta["orientation_note"] = orientation_note

    return RobotState(
        ee_position=ee_position,
        ee_orientation=ee_orientation,
        gripper_width=robot_state.gripper_width,
        gripper_open=robot_state.gripper_open,
        action_delta=action_delta,
        prev_ee_position=prev_ee_position,
        frame=target_frame,
        valid=robot_state.valid and ee_position is not None,
        metadata=meta,
    )


def _find_tensor(
    obs: Dict[str, Any],
    keys: Tuple[str, ...],
    expected_shapes: Optional[Tuple[Tuple[int, ...], ...]] = None,
) -> Tuple[Optional[torch.Tensor], Optional[str]]:
    for container, prefix in _iter_state_containers(obs):
        for key in keys:
            if key not in container:
                continue
            tensor = _as_valid_tensor(container[key], expected_shapes=expected_shapes)
            if tensor is not None:
                return tensor, f"{prefix}.{key}" if prefix else key
    return None, None


def _find_float(obs: Dict[str, Any], keys: Tuple[str, ...]) -> Tuple[Optional[float], Optional[str]]:
    for container, prefix in _iter_state_containers(obs):
        for key in keys:
            if key not in container:
                continue
            value = container[key]
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if arr.size and np.all(np.isfinite(arr)):
                return float(np.mean(arr)), f"{prefix}.{key}" if prefix else key
    return None, None


def _find_bool(obs: Dict[str, Any], keys: Tuple[str, ...]) -> Tuple[Optional[bool], Optional[str]]:
    for container, prefix in _iter_state_containers(obs):
        for key in keys:
            if key not in container:
                continue
            value = container[key]
            if isinstance(value, bool):
                return value, f"{prefix}.{key}" if prefix else key
            arr = np.asarray(value).reshape(-1)
            if arr.size:
                return bool(arr[0]), f"{prefix}.{key}" if prefix else key
    return None, None


def _find_frame(obs: Dict[str, Any]) -> Optional[str]:
    for container, _ in _iter_state_containers(obs):
        value = container.get("frame") or container.get("robot_state_frame")
        if value is not None:
            return str(value)
    return None


def _iter_state_containers(obs: Dict[str, Any]):
    yield obs, ""
    for key in NESTED_STATE_KEYS:
        value = obs.get(key)
        if isinstance(value, dict):
            yield value, key


def _as_valid_tensor(
    value: Any,
    expected_shapes: Optional[Tuple[Tuple[int, ...], ...]] = None,
) -> Optional[torch.Tensor]:
    if value is None:
        return None
    try:
        tensor = _as_tensor(value, dtype=torch.float32)
    except (TypeError, ValueError):
        return None
    if not bool(torch.isfinite(tensor).all()):
        return None
    if expected_shapes is not None:
        flat = tensor.reshape(-1)
        for shape in expected_shapes:
            if tuple(tensor.shape) == shape:
                return tensor.clone()
            if int(np.prod(shape)) == int(flat.numel()):
                return flat.reshape(shape).clone()
        return None
    return tensor.clone()


def _as_tensor(value: Any, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)

# ---------------------------------------------------------------------------
# Source: pruning/geometry/token_geometry.py
# ---------------------------------------------------------------------------
import math
from typing import Any, Dict, Optional, Tuple

import torch


def infer_token_grid(num_visual_tokens: int) -> Tuple[int, int]:
    """Infer the spatial token grid used by a visual-token sequence.

    For a single square grid, this returns that grid directly, e.g. 256 -> 16x16.
    For repeated square grids, such as 512 tokens that likely represent two
    16x16 encoder streams, this returns the per-encoder spatial grid. Use
    `infer_token_grid_metadata` when the repeat count matters.
    """
    return infer_token_grid_metadata(num_visual_tokens)["grid_shape"]


def infer_token_grid_metadata(
    num_visual_tokens: int,
    preferred_grid: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """Infer token-grid metadata without pretending ambiguous layouts are solved.

    The function prefers a known `preferred_grid` when the token count is an
    integer multiple of that grid. Otherwise it detects square single-grid and
    repeated-square-grid layouts. A 512-token OpenVLA projector output is treated
    as two repeated 16x16 grids, not as one 512-cell rectangular grid.
    """
    n = int(num_visual_tokens)
    if n <= 0:
        raise ValueError(f"num_visual_tokens must be positive, got {num_visual_tokens}")

    if preferred_grid is not None:
        grid_h, grid_w = int(preferred_grid[0]), int(preferred_grid[1])
        tokens_per_grid = grid_h * grid_w
        if grid_h > 0 and grid_w > 0 and tokens_per_grid > 0 and n % tokens_per_grid == 0:
            num_encoders = n // tokens_per_grid
            return _metadata(
                n,
                grid_h,
                grid_w,
                num_encoders,
                "preferred_grid",
                "Using caller-provided token grid; repeated encoders inferred from token count.",
            )

    root = int(math.isqrt(n))
    if root * root == n:
        return _metadata(
            n,
            root,
            root,
            1,
            "single_square_grid",
            "Token count is a perfect square; assuming one spatial grid.",
        )

    repeated = _largest_repeated_square_grid(n)
    if repeated is not None:
        grid_h, grid_w, num_encoders = repeated
        return _metadata(
            n,
            grid_h,
            grid_w,
            num_encoders,
            "repeated_square_grid",
            "Token count is an integer multiple of a square grid. Treating layout as repeated encoder grids; per-token encoder_id is inferred by sequence block.",
        )

    grid_h, grid_w = _closest_factor_grid(n)
    return _metadata(
        n,
        grid_h,
        grid_w,
        1,
        "factor_grid_ambiguous",
        "Could not infer a square or repeated-square visual-token grid. Returning the closest factor grid as a best-effort fallback.",
    )


def build_patch_centers(
    grid_h: int,
    grid_w: int,
    image_h: int,
    image_w: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build patch center pixel coordinates `[u, v]` for a grid.

    Returns:
        Tensor with shape `[grid_h * grid_w, 2]`, where column 0 is horizontal
        pixel coordinate `u` and column 1 is vertical pixel coordinate `v`.
    """
    grid_h = int(grid_h)
    grid_w = int(grid_w)
    image_h = int(image_h)
    image_w = int(image_w)
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError(f"grid shape must be positive, got {(grid_h, grid_w)}")
    if image_h <= 0 or image_w <= 0:
        raise ValueError(f"image shape must be positive, got {(image_h, image_w)}")

    rows = torch.arange(grid_h, device=device, dtype=dtype)
    cols = torch.arange(grid_w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(rows, cols, indexing="ij")
    u = (xx + 0.5) * (float(image_w) / float(grid_w))
    v = (yy + 0.5) * (float(image_h) / float(grid_h))
    return torch.stack([u.reshape(-1), v.reshape(-1)], dim=-1)


def build_token_2d_geometry(
    num_visual_tokens: int,
    image_h: int,
    image_w: int,
    *,
    token_grid_shape: Optional[Tuple[int, int]] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Any]:
    """Build token index, grid coordinate, pixel center, and encoder metadata.

    For repeated encoder grids, e.g. 512 tokens as two 16x16 streams, `grid_xy`
    and `pixel_xy` are repeated per encoder block, and `encoder_id` marks the
    inferred stream. This is sufficient for depth sampling but not a full
    semantic guarantee that the two streams correspond to identical camera views.
    """
    meta = infer_token_grid_metadata(num_visual_tokens, preferred_grid=token_grid_shape)
    n = int(num_visual_tokens)
    grid_h, grid_w = meta["grid_shape"]
    tokens_per_grid = int(meta["tokens_per_grid"])
    num_encoders = int(meta["num_encoders"])

    token_indices = torch.arange(n, device=device, dtype=torch.long)
    base_grid = _build_grid_xy(grid_h, grid_w, device=device, dtype=dtype)
    base_pixels = build_patch_centers(grid_h, grid_w, image_h, image_w, device=device, dtype=dtype)

    if n == tokens_per_grid * num_encoders:
        grid_xy = base_grid.repeat(num_encoders, 1)
        pixel_xy = base_pixels.repeat(num_encoders, 1)
        encoder_id = torch.arange(num_encoders, device=device, dtype=torch.long).repeat_interleave(tokens_per_grid)
    else:
        idx = torch.remainder(token_indices, tokens_per_grid)
        grid_xy = base_grid.index_select(0, idx)
        pixel_xy = base_pixels.index_select(0, idx)
        encoder_id = torch.div(token_indices, tokens_per_grid, rounding_mode="floor")
        meta["mapping_notes"].append(
            "Token count did not exactly match inferred repeated grid; coordinates use modulo grid indexing."
        )

    valid_mask = torch.ones(n, device=device, dtype=torch.bool)
    return {
        "token_indices": token_indices,
        "grid_xy": grid_xy,
        "pixel_xy": pixel_xy,
        "valid_mask": valid_mask,
        "encoder_id": encoder_id,
        "metadata": meta,
    }


def _metadata(
    num_visual_tokens: int,
    grid_h: int,
    grid_w: int,
    num_encoders: int,
    inference_mode: str,
    note: str,
) -> Dict[str, Any]:
    tokens_per_grid = int(grid_h) * int(grid_w)
    return {
        "num_visual_tokens": int(num_visual_tokens),
        "grid_shape": (int(grid_h), int(grid_w)),
        "tokens_per_grid": int(tokens_per_grid),
        "num_encoders": int(num_encoders),
        "inference_mode": inference_mode,
        "is_repeated_grid": int(num_encoders) > 1,
        "mapping_notes": [note],
    }


def _largest_repeated_square_grid(num_tokens: int) -> Optional[Tuple[int, int, int]]:
    root = int(math.isqrt(num_tokens))
    for side in range(root, 1, -1):
        square = side * side
        if num_tokens % square == 0:
            repeats = num_tokens // square
            if repeats > 1:
                return side, side, repeats
    return None


def _closest_factor_grid(num_tokens: int) -> Tuple[int, int]:
    root = int(math.isqrt(num_tokens))
    for h in range(root, 0, -1):
        if num_tokens % h == 0:
            return h, num_tokens // h
    return 1, num_tokens


def _build_grid_xy(
    grid_h: int,
    grid_w: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    rows = torch.arange(int(grid_h), device=device, dtype=dtype)
    cols = torch.arange(int(grid_w), device=device, dtype=dtype)
    yy, xx = torch.meshgrid(rows, cols, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)

# ---------------------------------------------------------------------------
# Source: pruning/geometry/geometry_cache.py
# ---------------------------------------------------------------------------
from typing import Any, Dict, Tuple

import numpy as np


class TokenGeometryCache:
    """Caches 16x16 token centers, grid coordinates, rays, and extrinsics."""

    def __init__(self) -> None:
        self._cache: Dict[Tuple[Any, ...], Dict[str, np.ndarray]] = {}

    def get(
        self,
        depth_shape: Tuple[int, int],
        camera_intrinsics: np.ndarray,
        camera_extrinsics: np.ndarray,
        preprocess_meta: Any,
        token_grid_shape: Tuple[int, int] = (16, 16),
        num_visual_tokens: int = 256,
        projection_mode: str = "current",
    ) -> Dict[str, np.ndarray]:
        key = self._signature(
            depth_shape,
            camera_intrinsics,
            camera_extrinsics,
            preprocess_meta,
            token_grid_shape,
            num_visual_tokens,
            projection_mode,
        )
        if key not in self._cache:
            self._cache[key] = self._build(
                depth_shape,
                camera_intrinsics,
                camera_extrinsics,
                preprocess_meta,
                token_grid_shape,
                num_visual_tokens,
                projection_mode,
            )
        return self._cache[key]

    def sample_depth(
        self,
        depth: np.ndarray,
        cache: Dict[str, np.ndarray],
        *,
        check_zbuffer: bool = True,
    ) -> np.ndarray:
        """Sample depth at token centers via bilinear interpolation.

        Args:
            depth: Full-resolution depth image [H, W] in meters. Must be metric depth,
                   NOT raw robosuite z-buffer. If raw z-buffer is passed, 3D
                   backprojection will be catastrophically wrong (depth ~0.99 interpreted as 0.99m).
        """
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[:, :, 0]

        # Sanity check: warn if depth looks like raw robosuite z-buffer
        if check_zbuffer:
            flat = depth.reshape(-1)
            valid = flat[np.isfinite(flat)]
            if valid.size > 0:
                dmin, dmax, dmean, dstd = float(np.min(valid)), float(np.max(valid)), float(np.mean(valid)), float(np.std(valid))
                if dmin >= 0.90 and dmax <= 1.05 and dstd < 0.08:
                    import warnings
                    warnings.warn(
                        f"[TokenGeometryCache.sample_depth] "
                        f"depth looks like raw robosuite z-buffer (min={dmin:.4f}, max={dmax:.4f}, mean={dmean:.4f}, std={dstd:.4f}). "
                        f"This will produce incorrect 3D points. "
                        f"Expected metric depth in meters (e.g. min~0.5, max~3.0). "
                        f"Call convert_depth_to_metric() before sampling."
                    )

        u = cache["u"]
        v = cache["v"]
        h, w = depth.shape[:2]
        x0 = np.floor(u).astype(np.int32)
        y0 = np.floor(v).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)
        x0 = np.clip(x0, 0, w - 1)
        y0 = np.clip(y0, 0, h - 1)
        wx = (u - x0).astype(np.float32)
        wy = (v - y0).astype(np.float32)
        return (
            depth[y0, x0] * (1.0 - wx) * (1.0 - wy)
            + depth[y0, x1] * wx * (1.0 - wy)
            + depth[y1, x0] * (1.0 - wx) * wy
            + depth[y1, x1] * wx * wy
        ).astype(np.float32)

    def _signature(
        self,
        depth_shape: Tuple[int, int],
        K: np.ndarray,
        T: np.ndarray,
        meta: Any,
        token_grid_shape: Tuple[int, int],
        num_visual_tokens: int,
        projection_mode: str,
    ) -> Tuple[Any, ...]:
        return (
            tuple(depth_shape),
            tuple(np.round(np.asarray(K, dtype=np.float32).reshape(-1), 5).tolist()),
            tuple(np.round(np.asarray(T, dtype=np.float32).reshape(-1), 5).tolist()),
            tuple(getattr(meta, "original_size", depth_shape)),
            tuple(getattr(meta, "processed_size", (224, 224))),
            float(getattr(meta, "crop_scale", 0.9)),
            bool(getattr(meta, "center_crop", True)),
            int(getattr(meta, "patch_size", 14)),
            tuple(token_grid_shape),
            int(num_visual_tokens),
            projection_mode,
        )

    def _build(
        self,
        depth_shape: Tuple[int, int],
        K: np.ndarray,
        T: np.ndarray,
        meta: Any,
        token_grid_shape: Tuple[int, int],
        num_visual_tokens: int,
        projection_mode: str,
    ) -> Dict[str, np.ndarray]:
        depth_h, depth_w = depth_shape
        grid_h, grid_w = token_grid_shape
        tokens_per_grid = grid_h * grid_w

        orig_h, orig_w = getattr(meta, "original_size", depth_shape)
        processed_h, processed_w = getattr(meta, "processed_size", (224, 224))
        patch_size = int(getattr(meta, "patch_size", 14))
        scale_y = float(orig_h) / float(processed_h) if processed_h else 1.0
        scale_x = float(orig_w) / float(processed_w) if processed_w else 1.0

        if bool(getattr(meta, "center_crop", True)):
            crop_scale = float(getattr(meta, "crop_scale", 0.9))
            crop_h = int(orig_h * np.sqrt(crop_scale))
            crop_w = int(orig_w * np.sqrt(crop_scale))
            crop_top = (int(orig_h) - crop_h) // 2
            crop_left = (int(orig_w) - crop_w) // 2
        else:
            crop_top = 0
            crop_left = 0

        rows, cols = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing="ij")
        base_v = np.clip(crop_top + (rows.astype(np.float32) + 0.5) * patch_size * scale_y, 0, depth_h - 1).reshape(-1)
        base_u = np.clip(crop_left + (cols.astype(np.float32) + 0.5) * patch_size * scale_x, 0, depth_w - 1).reshape(-1)
        if num_visual_tokens != tokens_per_grid and num_visual_tokens % tokens_per_grid == 0:
            u = np.tile(base_u, num_visual_tokens // tokens_per_grid)
            v = np.tile(base_v, num_visual_tokens // tokens_per_grid)
        else:
            idx = np.arange(num_visual_tokens) % tokens_per_grid
            u = base_u[idx]
            v = base_v[idx]

        u_project = u.copy()
        v_project = v.copy()
        if projection_mode == "unrotate_pixels_then_original_K":
            u_project = (depth_w - 1) - u_project
            v_project = (depth_h - 1) - v_project

        pixels = np.stack([u_project, v_project, np.ones_like(u_project)], axis=0).astype(np.float32)
        rays = (np.linalg.inv(np.asarray(K, dtype=np.float32)) @ pixels).T.astype(np.float32)
        T = np.asarray(T, dtype=np.float32)
        return {
            "u": u.astype(np.float32),
            "v": v.astype(np.float32),
            "normalized_grid": np.stack(
                [
                    2.0 * u / max(1.0, depth_w - 1) - 1.0,
                    2.0 * v / max(1.0, depth_h - 1) - 1.0,
                ],
                axis=-1,
            ).astype(np.float32),
            "rays": rays,
            "R": T[:3, :3].astype(np.float32),
            "t": T[:3, 3].astype(np.float32),
            "tokens_per_grid": np.array(tokens_per_grid, dtype=np.int32),
        }

# ---------------------------------------------------------------------------
# Source: pruning/geometry/static_scene_cache.py
# ---------------------------------------------------------------------------
from typing import Any, Dict, Tuple

import numpy as np


class ACGTPStaticSceneCache:
    """Cache scene-layout and depth-edge signals when depth is stable."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self._cache: Dict[str, Any] = {}

    def reset(self) -> None:
        self._cache = {}

    @staticmethod
    def _depth_signature(token_depth: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        depth = np.asarray(token_depth, dtype=np.float32).reshape(-1)
        valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
        out = np.zeros_like(depth, dtype=np.float32)
        finite = valid & np.isfinite(depth)
        if not np.any(finite):
            return out
        vals = depth[finite]
        lo = float(np.min(vals))
        hi = float(np.max(vals))
        if hi - lo > 1e-8:
            out[finite] = (depth[finite] - lo) / (hi - lo)
        else:
            out[finite] = np.clip(depth[finite], 0.0, 1.0)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    def lookup(
        self,
        *,
        token_depth: np.ndarray,
        valid_mask: np.ndarray,
        token_grid_shape: Tuple[int, int],
        num_tokens: int,
    ) -> Tuple[bool, Dict[str, Any]]:
        meta: Dict[str, Any] = {
            "acgtp_static_scene_cache_enabled": bool(getattr(self.config, "acgtp_static_scene_cache_enabled", False)),
            "acgtp_static_scene_cache_hit": False,
            "acgtp_static_scene_cache_reason": "disabled",
        }
        if not bool(getattr(self.config, "acgtp_static_scene_cache_enabled", False)):
            return False, meta
        cache = self._cache
        if not cache:
            meta["acgtp_static_scene_cache_reason"] = "empty"
            return False, meta
        if cache.get("num_tokens") != int(num_tokens) or tuple(cache.get("token_grid_shape", ())) != tuple(token_grid_shape):
            meta["acgtp_static_scene_cache_reason"] = "shape_changed"
            return False, meta
        cached_sig = cache.get("depth_signature")
        cached_valid = cache.get("valid_mask")
        if cached_sig is None or cached_valid is None:
            meta["acgtp_static_scene_cache_reason"] = "missing_signature"
            return False, meta
        sig = self._depth_signature(token_depth, valid_mask)
        cached_sig = np.asarray(cached_sig, dtype=np.float32).reshape(-1)
        cached_valid = np.asarray(cached_valid, dtype=bool).reshape(-1)
        valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if cached_sig.shape[0] != sig.shape[0] or cached_valid.shape[0] != valid.shape[0]:
            meta["acgtp_static_scene_cache_reason"] = "signature_shape_changed"
            return False, meta
        overlap = cached_valid | valid
        depth_delta = float(np.mean(np.abs(sig[overlap] - cached_sig[overlap]))) if np.any(overlap) else 1.0
        valid_iou = float(np.sum(cached_valid & valid)) / float(max(1, np.sum(cached_valid | valid)))
        age = int(cache.get("age", 0))
        meta.update({
            "acgtp_static_scene_cache_depth_delta": depth_delta,
            "acgtp_static_scene_cache_valid_iou": valid_iou,
            "acgtp_static_scene_cache_age": age,
        })
        if depth_delta > float(self.config.acgtp_static_scene_cache_depth_delta_threshold):
            meta["acgtp_static_scene_cache_reason"] = "depth_changed"
            return False, meta
        if valid_iou < float(self.config.acgtp_static_scene_cache_valid_iou_threshold):
            meta["acgtp_static_scene_cache_reason"] = "valid_mask_changed"
            return False, meta
        if age >= int(self.config.acgtp_static_scene_cache_max_age):
            meta["acgtp_static_scene_cache_reason"] = "max_age"
            return False, meta
        cache["age"] = age + 1
        meta["acgtp_static_scene_cache_hit"] = True
        meta["acgtp_static_scene_cache_reason"] = "hit"
        meta["edge_scores"] = cache.get("edge_scores")
        meta["scene_result"] = cache.get("scene_result")
        return meta["edge_scores"] is not None and meta["scene_result"] is not None, meta

    def store(
        self,
        *,
        token_depth: np.ndarray,
        valid_mask: np.ndarray,
        token_grid_shape: Tuple[int, int],
        num_tokens: int,
        edge_scores: np.ndarray,
        scene_result: Dict[str, Any],
    ) -> None:
        if not bool(getattr(self.config, "acgtp_static_scene_cache_enabled", False)):
            return
        self._cache = {
            "num_tokens": int(num_tokens),
            "token_grid_shape": tuple(token_grid_shape),
            "depth_signature": self._depth_signature(token_depth, valid_mask),
            "valid_mask": np.asarray(valid_mask, dtype=bool).reshape(-1).copy(),
            "edge_scores": np.asarray(edge_scores, dtype=np.float32).reshape(-1).copy(),
            "scene_result": dict(scene_result),
            "age": 0,
        }

# ---------------------------------------------------------------------------
# Source: pruning/geometry/robot_geometry.py
# ---------------------------------------------------------------------------
from typing import Any, Dict, Optional, Tuple
import time

import numpy as np
import torch

from .spatial import compute_depth_edge_scores


ROBOT_POS_KEYS = ("gripper_pos", "eef_pos", "ee_pos", "end_effector_pos")
ROBOT_POSE_KEYS = ("ee_pose", "eef_pose", "gripper_pose", "end_effector_pose")
NESTED_ROBOT_KEYS = ("robot_state", "robot_obs")
T_ROBOT_CAM_KEYS = ("T_robot_cam", "T_base_cam", "camera_extrinsics", "camera_to_base", "camera_to_robot")
T_CAM_ROBOT_KEYS = ("T_cam_robot", "T_camera_robot", "base_to_camera", "robot_to_camera")


def sample_depth_for_tokens(
    depth_map: Any,
    pixel_xy: Any,
    method: str = "center",
    *,
    local_window: int = 3,
) -> Tuple[Any, Any]:
    """Sample depth values for token center pixels.

    Args:
        depth_map: `[H, W]` or `[H, W, 1]` numpy array / torch tensor.
        pixel_xy: `[N, 2]` coordinates in `[u, v]` order.
        method: `"center"` or `"local_median"`.

    Returns:
        token_depth and valid_depth_mask, preserving numpy versus torch input
        type. Invalid values are NaN/Inf/non-positive depths.
    """
    if torch.is_tensor(depth_map) or torch.is_tensor(pixel_xy):
        return _sample_depth_for_tokens_torch(depth_map, pixel_xy, method, local_window)
    return _sample_depth_for_tokens_numpy(depth_map, pixel_xy, method, local_window)


def backproject_pixels_to_camera(pixel_xy: Any, depth: Any, intrinsics: Any) -> Any:
    """Backproject pixel centers and depth into camera-frame 3D points.

    Formula:

    ```text
    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy
    Z = z
    ```
    """
    if torch.is_tensor(pixel_xy) or torch.is_tensor(depth) or torch.is_tensor(intrinsics):
        pixel_t = _as_torch(pixel_xy)
        depth_t = _as_torch(depth, device=pixel_t.device, dtype=pixel_t.dtype).reshape(-1)
        K_t = _as_torch(intrinsics, device=pixel_t.device, dtype=pixel_t.dtype)
        fx, fy, cx, cy = _intrinsics_components_torch(K_t)
        u = pixel_t.reshape(-1, 2)[:, 0]
        v = pixel_t.reshape(-1, 2)[:, 1]
        x = (u - cx) * depth_t / fx
        y = (v - cy) * depth_t / fy
        points = torch.stack([x, y, depth_t], dim=-1)
        valid = torch.isfinite(points).all(dim=-1) & torch.isfinite(depth_t) & (depth_t > 0)
        points = torch.where(valid[:, None], points, torch.full_like(points, float("nan")))
        return points

    pixel_np = np.asarray(pixel_xy, dtype=np.float32).reshape(-1, 2)
    depth_np = np.asarray(depth, dtype=np.float32).reshape(-1)
    fx, fy, cx, cy = _intrinsics_components_numpy(intrinsics)
    u = pixel_np[:, 0]
    v = pixel_np[:, 1]
    points = np.stack(
        [
            (u - cx) * depth_np / fx,
            (v - cy) * depth_np / fy,
            depth_np,
        ],
        axis=-1,
    ).astype(np.float32)
    valid = np.isfinite(points).all(axis=-1) & np.isfinite(depth_np) & (depth_np > 0.0)
    points[~valid] = np.nan
    return points


def transform_camera_to_robot(points_cam: Any, extrinsics: Optional[Any] = None) -> Tuple[Any, Dict[str, Any]]:
    """Transform camera-frame points with optional `T_robot_cam`.

    If `extrinsics` is `None`, points are returned unchanged and frame metadata
    explicitly marks the result as camera-frame.
    """
    if extrinsics is None:
        return points_cam, {"frame": "camera", "transform_applied": False, "extrinsics_shape": None}

    if torch.is_tensor(points_cam) or torch.is_tensor(extrinsics):
        pts = _as_torch(points_cam)
        T = _as_torch(extrinsics, device=pts.device, dtype=pts.dtype)
        if tuple(T.shape[-2:]) != (4, 4):
            raise ValueError(f"extrinsics must have shape [4, 4], got {tuple(T.shape)}")
        robot = pts @ T[:3, :3].T + T[:3, 3]
        valid = torch.isfinite(pts).all(dim=-1)
        robot = torch.where(valid[:, None], robot, torch.full_like(robot, float("nan")))
        return robot, {"frame": "robot", "transform_applied": True, "extrinsics_shape": tuple(T.shape)}

    pts_np = np.asarray(points_cam, dtype=np.float32).reshape(-1, 3)
    T_np = np.asarray(extrinsics, dtype=np.float32)
    if T_np.shape != (4, 4):
        raise ValueError(f"extrinsics must have shape [4, 4], got {T_np.shape}")
    robot_np = (T_np[:3, :3] @ pts_np.T).T + T_np[:3, 3][None, :]
    valid = np.isfinite(pts_np).all(axis=-1)
    robot_np[~valid] = np.nan
    return robot_np.astype(np.float32), {"frame": "robot", "transform_applied": True, "extrinsics_shape": tuple(T_np.shape)}


def build_token_3d_geometry(
    token_2d_geometry: Dict[str, Any],
    depth_map: Any,
    intrinsics: Any,
    extrinsics: Optional[Any] = None,
    *,
    sample_method: str = "center",
) -> Dict[str, Any]:
    """Build token-level camera/robot 3D metadata from 2D token geometry."""
    pixel_xy = token_2d_geometry["pixel_xy"]
    token_depth, valid_depth_mask = sample_depth_for_tokens(depth_map, pixel_xy, method=sample_method)
    points_cam = backproject_pixels_to_camera(pixel_xy, token_depth, intrinsics)
    points_robot, frame_info = transform_camera_to_robot(points_cam, extrinsics)

    if torch.is_tensor(points_cam) or torch.is_tensor(valid_depth_mask):
        valid_3d_mask = _as_torch(valid_depth_mask, device=points_cam.device, dtype=torch.bool) & torch.isfinite(points_cam).all(dim=-1)
        if torch.is_tensor(points_robot):
            valid_3d_mask = valid_3d_mask & torch.isfinite(points_robot).all(dim=-1)
    else:
        valid_3d_mask = np.asarray(valid_depth_mask, dtype=np.bool_) & np.isfinite(points_cam).all(axis=-1)
        if points_robot is not None:
            valid_3d_mask = valid_3d_mask & np.isfinite(points_robot).all(axis=-1)

    return {
        "points_cam": points_cam,
        "points_robot": points_robot,
        "valid_3d_mask": valid_3d_mask,
        "depth_values": token_depth,
        "frame_info": frame_info,
    }


def compute_robot_geo_scores_v0(
    token_3d_geometry: Dict[str, Any],
    robot_state: Any,
    motion_direction: Optional[Any] = None,
    depth_edge_score: Optional[Any] = None,
    config: Optional[Any] = None,
) -> Dict[str, Any]:
    """Rule-based robot-centric token scoring v0.

    This function is model-agnostic and only produces per-token geometry scores.
    It does not gather tokens or alter OpenVLA action decoding.
    """
    with torch.no_grad():
        points_raw = token_3d_geometry.get("points_robot")
        frame = "robot"
        if points_raw is None:
            points_raw = token_3d_geometry.get("points_cam")
            frame = "camera"
        if points_raw is None:
            n = _infer_score_length(depth_edge_score, token_3d_geometry)
            zeros = torch.zeros(n, dtype=torch.float32)
            return _robot_geo_v0_output(zeros, zeros, zeros, zeros, zeros, zeros, torch.zeros(n, dtype=torch.bool), {
                "geometry_available": False,
                "fallback_reason": "missing_token_points",
                "frame": "unknown",
            })

        points = _as_torch(points_raw, dtype=torch.float32).reshape(-1, 3)
        n = int(points.shape[0])
        valid_mask = token_3d_geometry.get("valid_3d_mask")
        if valid_mask is None:
            valid = torch.isfinite(points).all(dim=-1)
        else:
            valid = _as_torch(valid_mask, device=points.device, dtype=torch.bool).reshape(-1)
            valid = valid & torch.isfinite(points).all(dim=-1)

        grip = _extract_robot_state_position(robot_state, device=points.device)
        if grip is None:
            zeros = torch.zeros(n, dtype=torch.float32, device=points.device)
            return _robot_geo_v0_output(zeros, zeros, zeros, zeros, zeros, zeros, torch.zeros(n, dtype=torch.bool, device=points.device), {
                "geometry_available": False,
                "fallback_reason": "missing_robot_state",
                "frame": frame,
            })

        sigma = max(float(_cfg_value(config, "sigma_near", 0.12)), 1e-6)
        diff = points - grip.reshape(1, 3)
        distances = torch.linalg.norm(diff, dim=-1)
        distance_score = torch.exp(-distances / sigma)
        distance_score = torch.where(valid, distance_score, torch.zeros_like(distance_score))

        direction = _resolve_motion_direction(robot_state, motion_direction, device=points.device)
        if direction is None:
            motion_cone = torch.zeros(n, dtype=torch.float32, device=points.device)
            motion_valid = False
        else:
            r_norm = torch.linalg.norm(diff, dim=-1).clamp_min(1e-8)
            cosine = (diff @ direction.reshape(3)) / r_norm
            motion_cone = torch.clamp(cosine, min=0.0, max=1.0)
            motion_cone = torch.where(valid, motion_cone, torch.zeros_like(motion_cone))
            motion_valid = True

        if depth_edge_score is None:
            edge = torch.zeros(n, dtype=torch.float32, device=points.device)
        else:
            edge = _as_torch(depth_edge_score, device=points.device, dtype=torch.float32).reshape(-1)
            if edge.numel() != n:
                idx = torch.remainder(torch.arange(n, device=points.device), int(edge.numel()))
                edge = edge.index_select(0, idx)
            edge = torch.nan_to_num(edge, nan=0.0, posinf=0.0, neginf=0.0)
            edge = torch.where(valid, edge, torch.zeros_like(edge))

        workspace_score = _workspace_score(points, valid, config)
        edge_norm = _normalize_torch(edge, valid)
        distance_norm = _normalize_torch(distance_score, valid)
        motion_norm = _normalize_torch(motion_cone, valid)
        workspace_norm = _normalize_torch(workspace_score, valid)
        contact_risk = distance_norm * (0.5 + 0.5 * motion_norm) * (0.5 + 0.5 * edge_norm)
        contact_risk = torch.where(valid, contact_risk, torch.zeros_like(contact_risk))

        weights = _robot_geo_v0_weights(config)
        final = (
            weights["distance_to_gripper"] * distance_norm
            + weights["motion_direction"] * motion_norm
            + weights["depth_edge"] * edge_norm
            + weights["workspace"] * workspace_norm
            + weights["contact_risk"] * contact_risk
        )
        final = torch.nan_to_num(final, nan=0.0, posinf=0.0, neginf=0.0)
        final = torch.where(valid, final, torch.zeros_like(final))

        valid_values = final[valid]
        debug = {
            "geometry_available": True,
            "frame": frame,
            "valid_token_ratio": float(valid.float().mean().item()) if n else None,
            "distance_to_gripper_score_mean": _tensor_mean(distance_score, valid),
            "distance_to_gripper_score_max": _tensor_max(distance_score, valid),
            "motion_cone_score_mean": _tensor_mean(motion_cone, valid),
            "motion_cone_score_max": _tensor_max(motion_cone, valid),
            "depth_edge_score_mean": _tensor_mean(edge, valid),
            "depth_edge_score_max": _tensor_max(edge, valid),
            "workspace_score_mean": _tensor_mean(workspace_score, valid),
            "workspace_score_max": _tensor_max(workspace_score, valid),
            "contact_risk_score_mean": _tensor_mean(contact_risk, valid),
            "contact_risk_score_max": _tensor_max(contact_risk, valid),
            "final_score_mean": float(valid_values.mean().item()) if valid_values.numel() else None,
            "final_score_max": float(valid_values.max().item()) if valid_values.numel() else None,
            "final_score_std": float(valid_values.std(unbiased=False).item()) if valid_values.numel() else None,
            "motion_direction_valid": bool(motion_valid),
            "weights": dict(weights),
        }
        return _robot_geo_v0_output(
            final,
            distance_score,
            motion_cone,
            edge,
            workspace_score,
            contact_risk,
            valid,
            debug,
        )


def decide_dynamic_keep_ratio(geo_scores_dict: Dict[str, Any], config: Optional[Any]) -> Dict[str, Any]:
    """Decide a conservative dynamic keep ratio from robot-geometry risk.

    The scheduler is intentionally external to OpenVLA. It only consumes score
    tensors and returns a keep-ratio decision for the pruning hook.

    Conservative defaults (v2 patch):
    - low_risk: 0.75
    - medium_risk: 0.85
    - high_risk: 0.95
    """
    contact = _score_tensor_from_dict(geo_scores_dict, "contact_risk_score", "contact_risk_scores")
    valid = _score_tensor_from_dict(geo_scores_dict, "valid_mask", "valid_3d_mask")
    distance = _score_tensor_from_dict(geo_scores_dict, "distance_to_gripper_score", "distance_scores", "near_scores")
    motion = _score_tensor_from_dict(geo_scores_dict, "motion_cone_score", "motion_scores")

    dynamic_cfg = _cfg_value(config, "dynamic_keep_ratio_config", {}) or {}
    # Conservative defaults: low=0.75, mid=0.85, high=0.95
    min_ratio = float(dynamic_cfg.get("min_keep_ratio", _cfg_value(config, "keep_ratio_far", 0.75)))
    mid_ratio = float(dynamic_cfg.get("mid_keep_ratio", _cfg_value(config, "keep_ratio_mid", 0.85)))
    max_ratio = float(dynamic_cfg.get("max_keep_ratio", _cfg_value(config, "keep_ratio_near", 0.95)))
    contact_threshold = float(dynamic_cfg.get("contact_risk_threshold", 0.5))
    uncertainty_threshold = float(dynamic_cfg.get("uncertainty_threshold", 0.5))

    if contact is None:
        return {
            "keep_ratio": float(max(min_ratio, 0.75)),
            "risk_level": "medium",
            "risk_score": None,
            "reason": "missing_geometry_fallback",
            "component_summary": {
                "num_high_contact_tokens": None,
                "num_valid_3d_tokens": None,
                "motion_direction_valid": None,
                "uncertainty_high": None,
            },
        }

    contact = torch.nan_to_num(contact.reshape(-1).to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
    n = int(contact.numel())
    if valid is None:
        valid_bool = torch.isfinite(contact)
    else:
        valid_bool = valid.reshape(-1).to(dtype=torch.bool)
        if valid_bool.numel() != n:
            idx = torch.remainder(torch.arange(n, device=contact.device), int(valid_bool.numel()))
            valid_bool = valid_bool.index_select(0, idx)
        valid_bool = valid_bool & torch.isfinite(contact)

    num_valid = int(valid_bool.sum().item())
    valid_ratio = num_valid / max(1, n)
    if n == 0 or num_valid == 0:
        return {
            "keep_ratio": float(max(min_ratio, 0.75)),
            "risk_level": "medium",
            "risk_score": None,
            "reason": "missing_valid_3d_tokens_fallback",
            "component_summary": {
                "num_high_contact_tokens": 0,
                "num_valid_3d_tokens": num_valid,
                "valid_3d_token_ratio": float(valid_ratio),
                "motion_direction_valid": None,
                "uncertainty_high": True,
            },
        }

    contact_valid = contact[valid_bool]
    high_contact_mask = contact_valid >= contact_threshold
    num_high_contact = int(high_contact_mask.sum().item())
    max_contact = float(contact_valid.max().item())
    mean_contact = float(contact_valid.mean().item())
    top_k = max(1, int(round(0.1 * contact_valid.numel())))
    top_contact_mean = float(torch.topk(contact_valid, k=min(top_k, contact_valid.numel())).values.mean().item())

    if distance is not None:
        dist_score = torch.nan_to_num(distance.reshape(-1).to(device=contact.device, dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if dist_score.numel() != n:
            idx = torch.remainder(torch.arange(n, device=contact.device), int(dist_score.numel()))
            dist_score = dist_score.index_select(0, idx)
        near_count = int(((dist_score >= uncertainty_threshold) & valid_bool).sum().item())
    else:
        near_count = 0
    uncertainty_high = near_count < max(2, int(0.02 * num_valid))

    if motion is not None:
        motion_score = torch.nan_to_num(motion.reshape(-1).to(device=contact.device, dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        motion_direction_valid = bool(torch.any((motion_score > 1e-6) & valid_bool).item())
    else:
        motion_direction_valid = False

    # Conservative: when motion direction is invalid, keep more tokens
    motion_penalty = 0.0 if motion_direction_valid else 0.10

    # Conservative: when geometry quality is uncertain, keep more tokens
    uncertainty_penalty = 0.0 if not uncertainty_high else 0.10

    # Conservative: compute risk score with bounded penalty
    risk_score = max(0.0, min(1.0,
        0.5 * max_contact + 0.25 * top_contact_mean + 0.15 * mean_contact + motion_penalty + uncertainty_penalty
    ))

    # Conservative: low valid_3d_token_ratio means uncertainty, keep at least mid
    if valid_ratio < 0.3:
        return {
            "keep_ratio": float(max(min_ratio, 0.85)),
            "risk_level": "medium",
            "risk_score": float(risk_score),
            "reason": "low_valid_3d_token_ratio_conservative",
            "component_summary": {
                "num_high_contact_tokens": num_high_contact,
                "num_valid_3d_tokens": num_valid,
                "valid_3d_token_ratio": float(valid_ratio),
                "max_contact_risk": max_contact,
                "mean_contact_risk": mean_contact,
                "top_contact_risk_mean": top_contact_mean,
                "near_token_count": near_count,
                "motion_direction_valid": motion_direction_valid,
                "uncertainty_high": bool(uncertainty_high),
            },
        }

    # Fallback to mid_keep_ratio (0.85) when geometry is uncertain
    if uncertainty_high or not motion_direction_valid:
        return {
            "keep_ratio": float(max(min_ratio, 0.85)),
            "risk_level": "medium",
            "risk_score": float(risk_score),
            "reason": "uncertain_geometry_conservative" if uncertainty_high else "invalid_motion_direction_conservative",
            "component_summary": {
                "num_high_contact_tokens": num_high_contact,
                "num_valid_3d_tokens": num_valid,
                "valid_3d_token_ratio": float(valid_ratio),
                "max_contact_risk": max_contact,
                "mean_contact_risk": mean_contact,
                "top_contact_risk_mean": top_contact_mean,
                "near_token_count": near_count,
                "motion_direction_valid": motion_direction_valid,
                "uncertainty_high": bool(uncertainty_high),
            },
        }

    # Risk-level based keep ratio with conservative thresholds
    if risk_score >= 0.65 or max_contact >= contact_threshold or num_high_contact >= max(1, int(0.03 * num_valid)):
        risk_level = "high"
        keep_ratio = max_ratio
        reason = "high_contact_risk"
    elif risk_score >= 0.35:
        risk_level = "medium"
        keep_ratio = mid_ratio
        reason = "medium_contact_risk"
    else:
        risk_level = "low"
        keep_ratio = min_ratio
        reason = "low_contact_risk"

    return {
        "keep_ratio": float(max(0.75, max(min_ratio, min(1.0, keep_ratio)))),
        "risk_level": risk_level,
        "risk_score": float(risk_score),
        "reason": reason,
        "component_summary": {
            "num_high_contact_tokens": num_high_contact,
            "num_valid_3d_tokens": num_valid,
            "valid_3d_token_ratio": float(valid_ratio),
            "max_contact_risk": max_contact,
            "mean_contact_risk": mean_contact,
            "top_contact_risk_mean": top_contact_mean,
            "near_token_count": near_count,
            "motion_direction_valid": motion_direction_valid,
            "uncertainty_high": bool(uncertainty_high),
        },
    }


def extract_gripper_position(source: Any) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Best-effort extraction of gripper/eef position from objects or dicts."""
    pos, key = _extract_position_direct(source)
    if pos is not None:
        return pos, key

    for nested_key in NESTED_ROBOT_KEYS:
        nested = _get_value(source, nested_key)
        if nested is None:
            continue
        pos, key = _extract_position_direct(nested)
        if pos is not None:
            return pos, f"{nested_key}.{key}"
    return None, None


def extract_robot_camera_transform(source: Any) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Return T_robot_cam, accepting either camera-to-robot or inverse naming."""
    for key in T_ROBOT_CAM_KEYS:
        value = _get_value(source, key)
        T = _as_transform(value)
        if T is not None:
            return T, key

    for key in T_CAM_ROBOT_KEYS:
        value = _get_value(source, key)
        T = _as_transform(value)
        if T is not None:
            try:
                return np.linalg.inv(T).astype(np.float32), f"inv({key})"
            except np.linalg.LinAlgError:
                return None, None
    return None, None


def project_tokens_to_robot(
    token_depth: np.ndarray,
    rays: np.ndarray,
    T_robot_cam: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Sparse 3D mapping for token centers."""
    depth = np.asarray(token_depth, dtype=np.float32).reshape(-1)
    rays_np = np.asarray(rays, dtype=np.float32).reshape(depth.shape[0], 3)
    p_cam = rays_np * depth[:, None]
    p_robot = (np.asarray(T_robot_cam, dtype=np.float32)[:3, :3] @ p_cam.T).T
    p_robot = p_robot + np.asarray(T_robot_cam, dtype=np.float32)[:3, 3][None, :]
    p_robot = p_robot.astype(np.float32)
    p_robot[~np.asarray(valid_mask, dtype=np.bool_)] = np.nan
    return p_robot


def map_depth_tokens_to_robot(
    token_depth: np.ndarray,
    rays: np.ndarray,
    T_robot_cam: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Batch sparse token-depth to robot-frame mapping.

    Returns:
        p_robot: [B, N, 3]
        valid_mask: [B, N]
    """
    depth_np = np.asarray(token_depth, dtype=np.float32)
    was_1d = depth_np.ndim == 1
    if was_1d:
        depth_np = depth_np[None, :]
    if depth_np.ndim != 2:
        raise ValueError(f"token_depth must have shape [B, N] or [N], got {depth_np.shape}")
    rays_np = np.asarray(rays, dtype=np.float32)
    if rays_np.ndim == 2:
        rays_np = np.broadcast_to(rays_np[None, :, :], (depth_np.shape[0], rays_np.shape[0], rays_np.shape[1]))
    if valid_mask is None:
        valid_np = np.isfinite(depth_np) & (depth_np > 0.0)
    else:
        valid_np = np.asarray(valid_mask, dtype=np.bool_)
        if valid_np.ndim == 1:
            valid_np = valid_np[None, :]
    rows = [
        project_tokens_to_robot(depth_np[b], rays_np[b], T_robot_cam, valid_np[b])
        for b in range(depth_np.shape[0])
    ]
    out = np.stack(rows, axis=0).astype(np.float32)
    return (out[0] if was_1d else out), (valid_np[0] if was_1d else valid_np)


def compute_robot_geo_near_scores(
    token_depth: np.ndarray,
    valid_mask: np.ndarray,
    cache: Dict[str, np.ndarray],
    T_robot_cam: np.ndarray,
    gripper_pos: np.ndarray,
    *,
    token_grid_shape: Tuple[int, int] = (16, 16),
    num_visual_tokens: int = 256,
    w_edge: float = 0.8,
    w_near: float = 0.2,
    sigma_near: float = 0.12,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Combine token-level depth-edge and gripper-nearness scores."""
    valid = np.asarray(valid_mask, dtype=np.bool_).reshape(-1)
    edge = compute_depth_edge_scores(
        token_depth,
        valid,
        token_grid_shape=token_grid_shape,
        num_visual_tokens=num_visual_tokens,
    )

    p_robot = project_tokens_to_robot(token_depth, cache["rays"], T_robot_cam, valid)
    grip = np.asarray(gripper_pos, dtype=np.float32).reshape(3)
    distances = np.linalg.norm(p_robot - grip[None, :], axis=1)
    distances = np.where(np.isfinite(distances), distances, np.inf).astype(np.float32)

    sigma = max(float(sigma_near), 1e-6)
    near = np.exp(-(distances * distances) / (2.0 * sigma * sigma)).astype(np.float32)
    near[~valid] = 0.0

    edge_norm = normalize_scores(edge, valid)
    near_norm = normalize_scores(near, valid)
    scores = float(w_edge) * edge_norm + float(w_near) * near_norm
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    scores[~valid] = 0.0

    valid_dist = distances[valid & np.isfinite(distances)]
    stats = {
        "edge_scores": edge,
        "near_scores": near,
        "distances": distances,
        "token_points_robot": p_robot,
        "d_min": float(np.min(valid_dist)) if valid_dist.size else None,
        "mean_near_score": float(np.mean(near[valid])) if np.any(valid) else None,
        "max_near_score": float(np.max(near[valid])) if np.any(valid) else None,
        "depth_edge_score_mean": float(np.mean(edge[valid])) if np.any(valid) else None,
        "geometry_score_mean": float(np.mean(scores[valid])) if np.any(valid) else None,
        "geometry_score_max": float(np.max(scores[valid])) if np.any(valid) else None,
        "geometry_score_std": float(np.std(scores[valid])) if np.any(valid) else None,
    }
    return scores, stats


def compute_robot_geo_corridor_scores(
    token_depth: np.ndarray,
    valid_mask: np.ndarray,
    cache: Dict[str, np.ndarray],
    T_robot_cam: np.ndarray,
    gripper_pos: np.ndarray,
    prev_gripper_pos: Optional[np.ndarray],
    *,
    token_grid_shape: Tuple[int, int] = (16, 16),
    num_visual_tokens: int = 256,
    w_edge: float = 0.7,
    w_near: float = 0.1,
    w_corridor: float = 0.2,
    sigma_near: float = 0.12,
    sigma_corridor: float = 0.08,
    corridor_length: float = 0.12,
    min_motion_norm: float = 1e-4,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Combine edge, near, and forward motion-corridor token scores."""
    scores_near, stats = compute_robot_geo_near_scores(
        token_depth,
        valid_mask,
        cache,
        T_robot_cam,
        gripper_pos,
        token_grid_shape=token_grid_shape,
        num_visual_tokens=num_visual_tokens,
        w_edge=0.0,
        w_near=1.0,
        sigma_near=sigma_near,
    )
    valid = np.asarray(valid_mask, dtype=np.bool_).reshape(-1)
    edge = np.asarray(stats["edge_scores"], dtype=np.float32)
    near = np.asarray(stats["near_scores"], dtype=np.float32)
    p_robot = np.asarray(stats["token_points_robot"], dtype=np.float32)

    grip = np.asarray(gripper_pos, dtype=np.float32).reshape(3)
    prev = None if prev_gripper_pos is None else np.asarray(prev_gripper_pos, dtype=np.float32).reshape(3)
    motion_norm = 0.0
    corridor_active = False
    corridor = np.zeros(valid.shape[0], dtype=np.float32)
    corridor_distances = np.full(valid.shape[0], np.inf, dtype=np.float32)

    if prev is not None and np.all(np.isfinite(prev)):
        motion = grip - prev
        motion_norm = float(np.linalg.norm(motion))
        if motion_norm >= float(min_motion_norm):
            direction = motion / max(motion_norm, 1e-8)
            start = grip
            end = grip + float(corridor_length) * direction
            corridor_distances = point_segment_distances(p_robot, start, end, valid)
            sigma = max(float(sigma_corridor), 1e-6)
            corridor = np.exp(
                -(corridor_distances * corridor_distances) / (2.0 * sigma * sigma)
            ).astype(np.float32)
            corridor[~valid] = 0.0
            corridor_active = True

    edge_norm = normalize_scores(edge, valid)
    near_norm = normalize_scores(near, valid)
    corridor_norm = normalize_scores(corridor, valid)
    scores = (
        float(w_edge) * edge_norm
        + float(w_near) * near_norm
        + float(w_corridor) * corridor_norm
    )
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    scores[~valid] = 0.0

    valid_corridor = corridor_distances[valid & np.isfinite(corridor_distances)]
    stats.update({
        "corridor_scores": corridor,
        "corridor_distances": corridor_distances,
        "motion_norm": motion_norm,
        "corridor_active": bool(corridor_active),
        "corridor_strength_mean": float(np.mean(corridor[valid])) if np.any(valid) else None,
        "d_corridor_min": float(np.min(valid_corridor)) if valid_corridor.size else None,
        "geometry_score_mean": float(np.mean(scores[valid])) if np.any(valid) else None,
        "geometry_score_max": float(np.max(scores[valid])) if np.any(valid) else None,
        "geometry_score_std": float(np.std(scores[valid])) if np.any(valid) else None,
    })
    return scores, stats


def compute_robot_geo_contact_budget_scores(
    token_depth: np.ndarray,
    valid_mask: np.ndarray,
    cache: Dict[str, np.ndarray],
    T_robot_cam: np.ndarray,
    gripper_pos: np.ndarray,
    prev_gripper_pos: Optional[np.ndarray],
    *,
    token_grid_shape: Tuple[int, int] = (16, 16),
    num_visual_tokens: int = 256,
    sigma_near: float = 0.12,
    sigma_corridor: float = 0.08,
    corridor_length: float = 0.12,
    min_motion_norm: float = 1e-4,
    w_near_contact: float = 0.5,
    w_corridor_contact: float = 0.8,
    edge_gate_eps: float = 1e-6,
    detailed_timing: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Edge-gated robot-contact score for contact-budget selection.

    Robot geometry supplements depth-edge tokens instead of competing with them
    directly in one global additive top-k.
    """
    valid = np.asarray(valid_mask, dtype=np.bool_).reshape(-1)
    t0 = time.perf_counter()
    edge = compute_depth_edge_scores(
        token_depth,
        valid,
        token_grid_shape=token_grid_shape,
        num_visual_tokens=num_visual_tokens,
    )
    edge_score_ms = (time.perf_counter() - t0) * 1000.0 if detailed_timing else None
    t0 = time.perf_counter()
    p_robot = project_tokens_to_robot(token_depth, cache["rays"], T_robot_cam, valid)
    robot_mapping_ms = (time.perf_counter() - t0) * 1000.0 if detailed_timing else None
    t0 = time.perf_counter()
    grip = np.asarray(gripper_pos, dtype=np.float32).reshape(3)
    distances = np.linalg.norm(p_robot - grip[None, :], axis=1)
    distances = np.where(np.isfinite(distances), distances, np.inf).astype(np.float32)

    sigma = max(float(sigma_near), 1e-6)
    near = np.exp(-(distances * distances) / (2.0 * sigma * sigma)).astype(np.float32)
    near[~valid] = 0.0
    near_score_ms = (time.perf_counter() - t0) * 1000.0 if detailed_timing else None

    t0 = time.perf_counter()
    prev = None if prev_gripper_pos is None else np.asarray(prev_gripper_pos, dtype=np.float32).reshape(3)
    motion_norm = 0.0
    corridor_active = False
    corridor = np.zeros(valid.shape[0], dtype=np.float32)
    corridor_distances = np.full(valid.shape[0], np.inf, dtype=np.float32)
    if prev is not None and np.all(np.isfinite(prev)):
        motion = grip - prev
        motion_norm = float(np.linalg.norm(motion))
        if motion_norm >= float(min_motion_norm):
            direction = motion / max(motion_norm, 1e-8)
            start = grip
            end = grip + float(corridor_length) * direction
            corridor_distances = point_segment_distances(p_robot, start, end, valid)
            sigma_c = max(float(sigma_corridor), 1e-6)
            corridor = np.exp(
                -(corridor_distances * corridor_distances) / (2.0 * sigma_c * sigma_c)
            ).astype(np.float32)
            corridor[~valid] = 0.0
            corridor_active = True
    corridor_score_ms = (time.perf_counter() - t0) * 1000.0 if detailed_timing else None

    t0 = time.perf_counter()
    edge_norm = normalize_scores(edge, valid)
    near_norm = normalize_scores(near, valid)
    corridor_norm = normalize_scores(corridor, valid)
    # Follow the edge-gated rule: near/corridor can only help where edge is present.
    _ = edge_gate_eps  # kept as a config-visible numerical guard for future probes
    gate = np.where(valid, edge_norm, 0.0).astype(np.float32)
    near_contact = near_norm * gate
    corridor_contact = corridor_norm * gate
    geo_contact = (
        float(w_near_contact) * near_contact
        + float(w_corridor_contact) * corridor_contact
    ).astype(np.float32)
    geo_contact = np.nan_to_num(geo_contact, nan=0.0, posinf=0.0, neginf=0.0)
    geo_contact[~valid] = 0.0
    contact_score_ms = (time.perf_counter() - t0) * 1000.0 if detailed_timing else None

    valid_dist = distances[valid & np.isfinite(distances)]
    valid_corridor = corridor_distances[valid & np.isfinite(corridor_distances)]
    stats = {
        "edge_scores": edge,
        "edge_norm": edge_norm,
        "near_scores": near,
        "near_contact_scores": near_contact,
        "corridor_scores": corridor,
        "corridor_contact_scores": corridor_contact,
        "geo_contact_scores": geo_contact,
        "distances": distances,
        "corridor_distances": corridor_distances,
        "token_points_robot": p_robot,
        "motion_norm": motion_norm,
        "corridor_active": bool(corridor_active),
        "d_min": float(np.min(valid_dist)) if valid_dist.size else None,
        "d_corridor_min": float(np.min(valid_corridor)) if valid_corridor.size else None,
        "mean_near_score": float(np.mean(near[valid])) if np.any(valid) else None,
        "max_near_score": float(np.max(near[valid])) if np.any(valid) else None,
        "corridor_strength_mean": float(np.mean(corridor[valid])) if np.any(valid) else None,
        "depth_edge_score_mean": float(np.mean(edge[valid])) if np.any(valid) else None,
        "edge_score_mean": float(np.mean(edge[valid])) if np.any(valid) else None,
        "near_contact_score_mean": float(np.mean(near_contact[valid])) if np.any(valid) else None,
        "corridor_contact_score_mean": float(np.mean(corridor_contact[valid])) if np.any(valid) else None,
        "geo_contact_score_mean": float(np.mean(geo_contact[valid])) if np.any(valid) else None,
        "geometry_score_mean": float(np.mean(geo_contact[valid])) if np.any(valid) else None,
        "geometry_score_max": float(np.max(geo_contact[valid])) if np.any(valid) else None,
        "geometry_score_std": float(np.std(geo_contact[valid])) if np.any(valid) else None,
    }
    if detailed_timing:
        stats.update({
            "edge_score_ms": edge_score_ms,
            "robot_mapping_ms": robot_mapping_ms,
            "near_score_ms": near_score_ms,
            "corridor_score_ms": corridor_score_ms,
            "contact_score_ms": contact_score_ms,
        })
    return geo_contact, stats


def point_segment_distances(
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    points_np = np.asarray(points, dtype=np.float32)
    start_np = np.asarray(start, dtype=np.float32).reshape(3)
    end_np = np.asarray(end, dtype=np.float32).reshape(3)
    seg = end_np - start_np
    seg_len2 = float(np.dot(seg, seg))
    if seg_len2 <= 1e-12:
        dist = np.linalg.norm(points_np - start_np[None, :], axis=1)
    else:
        t = ((points_np - start_np[None, :]) @ seg) / seg_len2
        t = np.clip(t, 0.0, 1.0).astype(np.float32)
        closest = start_np[None, :] + t[:, None] * seg[None, :]
        dist = np.linalg.norm(points_np - closest, axis=1)
    dist = np.where(np.isfinite(dist), dist, np.inf).astype(np.float32)
    if valid_mask is not None:
        dist[~np.asarray(valid_mask, dtype=np.bool_)] = np.inf
    return dist


def _sample_depth_for_tokens_numpy(
    depth_map: Any,
    pixel_xy: Any,
    method: str,
    local_window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth_map, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    if depth.ndim != 2:
        raise ValueError(f"depth_map must have shape [H, W] or [H, W, 1], got {depth.shape}")
    pixels = np.asarray(pixel_xy, dtype=np.float32).reshape(-1, 2)
    h, w = depth.shape
    u = np.rint(pixels[:, 0]).astype(np.int64)
    v = np.rint(pixels[:, 1]).astype(np.int64)
    in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u = np.clip(u, 0, w - 1)
    v = np.clip(v, 0, h - 1)

    if method == "center":
        token_depth = depth[v, u].astype(np.float32)
    elif method == "local_median":
        radius = max(0, int(local_window) // 2)
        values = []
        for x, y in zip(u, v):
            patch = depth[max(0, y - radius): min(h, y + radius + 1), max(0, x - radius): min(w, x + radius + 1)]
            valid_patch = patch[np.isfinite(patch) & (patch > 0.0)]
            values.append(float(np.median(valid_patch)) if valid_patch.size else np.nan)
        token_depth = np.asarray(values, dtype=np.float32)
    else:
        raise ValueError(f"Unknown depth sampling method: {method}")

    valid = in_bounds & np.isfinite(token_depth) & (token_depth > 0.0)
    token_depth = token_depth.astype(np.float32)
    token_depth[~valid] = np.nan
    return token_depth, valid.astype(np.bool_)


def _sample_depth_for_tokens_torch(
    depth_map: Any,
    pixel_xy: Any,
    method: str,
    local_window: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    depth = _as_torch(depth_map)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    if depth.ndim != 2:
        raise ValueError(f"depth_map must have shape [H, W] or [H, W, 1], got {tuple(depth.shape)}")
    pixels = _as_torch(pixel_xy, device=depth.device, dtype=torch.float32).reshape(-1, 2)
    h, w = int(depth.shape[0]), int(depth.shape[1])
    u = torch.round(pixels[:, 0]).long()
    v = torch.round(pixels[:, 1]).long()
    in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u = torch.clamp(u, 0, w - 1)
    v = torch.clamp(v, 0, h - 1)

    if method == "center":
        token_depth = depth[v, u].to(dtype=torch.float32)
    elif method == "local_median":
        radius = max(0, int(local_window) // 2)
        values = []
        for x_t, y_t in zip(u, v):
            x = int(x_t.item())
            y = int(y_t.item())
            patch = depth[max(0, y - radius): min(h, y + radius + 1), max(0, x - radius): min(w, x + radius + 1)]
            valid_patch = patch[torch.isfinite(patch) & (patch > 0)]
            values.append(torch.median(valid_patch) if valid_patch.numel() else torch.tensor(float("nan"), device=depth.device))
        token_depth = torch.stack(values).to(dtype=torch.float32)
    else:
        raise ValueError(f"Unknown depth sampling method: {method}")

    valid = in_bounds & torch.isfinite(token_depth) & (token_depth > 0)
    token_depth = torch.where(valid, token_depth, torch.full_like(token_depth, float("nan")))
    return token_depth, valid


def _intrinsics_components_numpy(intrinsics: Any) -> Tuple[float, float, float, float]:
    if isinstance(intrinsics, dict):
        return (
            float(intrinsics["fx"]),
            float(intrinsics["fy"]),
            float(intrinsics["cx"]),
            float(intrinsics["cy"]),
        )
    K = np.asarray(intrinsics, dtype=np.float32)
    if K.shape != (3, 3):
        raise ValueError(f"intrinsics must be a dict or [3, 3] matrix, got {K.shape}")
    return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])


def _intrinsics_components_torch(intrinsics: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if tuple(intrinsics.shape) != (3, 3):
        raise ValueError(f"intrinsics must have shape [3, 3], got {tuple(intrinsics.shape)}")
    return intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]


def _as_torch(
    value: Any,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if torch.is_tensor(value):
        tensor = value
        if device is not None or dtype is not None:
            tensor = tensor.to(device=device if device is not None else tensor.device, dtype=dtype if dtype is not None else tensor.dtype)
        return tensor
    return torch.as_tensor(value, device=device, dtype=dtype if dtype is not None else torch.float32)


def normalize_scores(scores: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    scores_np = np.nan_to_num(np.asarray(scores, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    valid = np.ones(scores_np.shape[0], dtype=np.bool_) if valid_mask is None else np.asarray(valid_mask, dtype=np.bool_)
    out = np.zeros_like(scores_np, dtype=np.float32)
    if not np.any(valid):
        return out
    vals = scores_np[valid]
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if hi - lo > 1e-8:
        out[valid] = (vals - lo) / (hi - lo)
    else:
        out[valid] = 0.0
    return out


def _robot_geo_v0_output(
    final_scores: torch.Tensor,
    distance_score: torch.Tensor,
    motion_cone_score: torch.Tensor,
    depth_edge_score: torch.Tensor,
    workspace_score: torch.Tensor,
    contact_risk_score: torch.Tensor,
    valid_mask: torch.Tensor,
    debug_info: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "final_scores": final_scores,
        "distance_to_gripper_score": distance_score,
        "motion_cone_score": motion_cone_score,
        "depth_edge_score": depth_edge_score,
        "workspace_score": workspace_score,
        "contact_risk_score": contact_risk_score,
        "valid_mask": valid_mask,
        "debug_info": debug_info,
    }


def _infer_score_length(depth_edge_score: Optional[Any], token_3d_geometry: Dict[str, Any]) -> int:
    if depth_edge_score is not None:
        return int(_as_torch(depth_edge_score).reshape(-1).shape[0])
    for key in ("valid_3d_mask", "depth_values"):
        if key in token_3d_geometry and token_3d_geometry[key] is not None:
            return int(_as_torch(token_3d_geometry[key]).reshape(-1).shape[0])
    return 0


def _extract_robot_state_position(robot_state: Any, device: torch.device) -> Optional[torch.Tensor]:
    if robot_state is None:
        return None
    if isinstance(robot_state, RobotState):
        value = robot_state.ee_position
    elif isinstance(robot_state, dict):
        value = None
        for key in ("ee_position", "ee_pos", "gripper_pos", "eef_pos"):
            if robot_state.get(key) is not None:
                value = robot_state.get(key)
                break
    else:
        value = getattr(robot_state, "ee_position", None)
    if value is None:
        return None
    tensor = _as_torch(value, device=device, dtype=torch.float32).reshape(-1)
    if tensor.numel() < 3 or not bool(torch.isfinite(tensor[:3]).all()):
        return None
    return tensor[:3]


def _resolve_motion_direction(
    robot_state: Any,
    motion_direction: Optional[Any],
    *,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if motion_direction is not None:
        direction = _as_torch(motion_direction, device=device, dtype=torch.float32).reshape(-1)[:3]
    elif isinstance(robot_state, RobotState) and robot_state.action_delta is not None and robot_state.action_delta.numel() >= 3:
        direction = robot_state.action_delta.to(device=device, dtype=torch.float32).reshape(-1)[:3]
    else:
        return None
    if direction.numel() < 3 or not bool(torch.isfinite(direction).all()):
        return None
    norm = torch.linalg.norm(direction)
    if not bool(torch.isfinite(norm)) or float(norm.item()) <= 1e-8:
        return None
    return direction / norm


def _workspace_score(points: torch.Tensor, valid: torch.Tensor, config: Optional[Any]) -> torch.Tensor:
    bounds = _cfg_value(config, "workspace_bounds", None)
    if bounds is None:
        bounds = ((-2.0, 2.0), (-2.0, 2.0), (-0.5, 2.0))
    try:
        bounds_t = _as_torch(bounds, device=points.device, dtype=points.dtype).reshape(3, 2)
    except Exception:
        bounds_t = _as_torch(((-2.0, 2.0), (-2.0, 2.0), (-0.5, 2.0)), device=points.device, dtype=points.dtype).reshape(3, 2)
    inside = (
        (points[:, 0] >= bounds_t[0, 0]) & (points[:, 0] <= bounds_t[0, 1])
        & (points[:, 1] >= bounds_t[1, 0]) & (points[:, 1] <= bounds_t[1, 1])
        & (points[:, 2] >= bounds_t[2, 0]) & (points[:, 2] <= bounds_t[2, 1])
        & valid
    )
    return inside.to(dtype=torch.float32)


def _normalize_torch(scores: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(scores, dtype=torch.float32)
    valid = valid & torch.isfinite(scores)
    if not bool(valid.any()):
        return out
    vals = scores[valid].to(dtype=torch.float32)
    lo = vals.min()
    hi = vals.max()
    if float((hi - lo).item()) > 1e-8:
        out[valid] = (vals - lo) / (hi - lo)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _robot_geo_v0_weights(config: Optional[Any]) -> Dict[str, float]:
    defaults = {
        "distance_to_gripper": 0.35,
        "motion_direction": 0.25,
        "depth_edge": 0.15,
        "workspace": 0.05,
        "contact_risk": 0.20,
    }
    weights = _cfg_value(config, "geo_score_weights", None)
    if isinstance(weights, dict):
        robot_weight_sum = sum(float(weights.get(k, 0.0) or 0.0) for k in ("distance_to_gripper", "motion_direction", "workspace", "contact_risk"))
        # The stage-1 config defaults are intentionally neutral for existing
        # modes. For explicit rule_v0, treat that neutral depth-edge-only dict as
        # "not configured" so the rule expert remains robot-centric.
        if robot_weight_sum > 1e-8:
            for key in defaults:
                if key in weights:
                    try:
                        defaults[key] = float(weights[key])
                    except (TypeError, ValueError):
                        pass
    total = sum(max(0.0, v) for v in defaults.values())
    if total <= 1e-8:
        return defaults
    return {k: max(0.0, v) / total for k, v in defaults.items()}


def _cfg_value(config: Optional[Any], key: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _tensor_mean(values: torch.Tensor, valid: torch.Tensor) -> Optional[float]:
    mask = valid & torch.isfinite(values)
    if not bool(mask.any()):
        return None
    return float(values[mask].mean().item())


def _tensor_max(values: torch.Tensor, valid: torch.Tensor) -> Optional[float]:
    mask = valid & torch.isfinite(values)
    if not bool(mask.any()):
        return None
    return float(values[mask].max().item())


def _score_tensor_from_dict(geo_scores_dict: Dict[str, Any], *keys: str) -> Optional[torch.Tensor]:
    for key in keys:
        if key in geo_scores_dict and geo_scores_dict[key] is not None:
            value = geo_scores_dict[key]
            if torch.is_tensor(value):
                return value.detach()
            return torch.as_tensor(value, dtype=torch.float32)
    return None


def _extract_position_direct(source: Any) -> Tuple[Optional[np.ndarray], Optional[str]]:
    for key in ROBOT_POS_KEYS:
        value = _get_value(source, key)
        pos = _as_vec3(value)
        if pos is not None:
            return pos, key
    for key in ROBOT_POSE_KEYS:
        value = _get_value(source, key)
        pose = _as_transform(value)
        if pose is not None:
            return pose[:3, 3].astype(np.float32), key
    return None, None


def _get_value(source: Any, key: str) -> Any:
    if source is None:
        return None
    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)


def _as_vec3(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape[0] < 3 or not np.all(np.isfinite(arr[:3])):
        return None
    return arr[:3].astype(np.float32)


def _as_transform(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape == (4, 4) and np.all(np.isfinite(arr)):
        return arr.astype(np.float32)
    flat = arr.reshape(-1)
    if flat.shape[0] == 16 and np.all(np.isfinite(flat)):
        return flat.reshape(4, 4).astype(np.float32)
    return None
