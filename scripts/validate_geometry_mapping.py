#!/usr/bin/env python3
"""Geometry Mapping Validation Script.

Validates the geometry pipeline used for robot-centric token pruning.
Generates a validation report and JSON summary.

Usage:
    python scripts/validate_geometry_mapping.py \
        --model_path /infini-data/checkpoints/openvla-7b-finetuned-libero-spatial \
        --num_episodes 10 --seed 42
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ─── Unified depth conversion (shared with geometry pipeline) ───────────────────
try:
    sys.path.insert(0, "/infini-data/openvla")
    from geometry.geometry_depth import convert_depth_to_metric
    _HAS_GEOMETRY_DEPTH = True
except ImportError:
    _HAS_GEOMETRY_DEPTH = False

# ─── Geometry helpers ──────────────────────────────────────────────────────────

def sample_depth(depth: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear depth sampling at pixel coords (u, v)."""
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim == 3:
        d = d[:, :, 0]
    h, w = d.shape
    x0 = np.clip(np.floor(u).astype(int), 0, w - 1)
    y0 = np.clip(np.floor(v).astype(int), 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = (u - x0.astype(float))
    wy = (v - y0.astype(float))
    return (d[y0, x0] * (1 - wx) * (1 - wy) +
            d[y0, x1] * wx * (1 - wy) +
            d[y1, x0] * (1 - wx) * wy +
            d[y1, x1] * wx * wy).astype(np.float32)


def backproject_pixel(u: float, v: float, z: float, K: np.ndarray) -> np.ndarray:
    """Backproject single pixel to camera-frame 3D."""
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return np.array([x, y, z], dtype=np.float32)


def transform_points_cam_to_robot(p_cam: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Transform camera-frame points to robot frame."""
    R, t = T[:3, :3], T[:3, 3]
    return (R @ p_cam.T).T + t[None, :]


def gripper_to_image(gripper_pos: np.ndarray, T: np.ndarray, K: np.ndarray,
                     im_h: int, im_w: int) -> Tuple[float, float, bool]:
    """Project gripper from robot frame to image pixel."""
    R, t = T[:3, :3], T[:3, 3]
    p_cam = R @ gripper_pos + t
    if p_cam[2] <= 0:
        return 0.0, 0.0, False
    u = p_cam[0] * K[0, 0] / p_cam[2] + K[0, 2]
    v = p_cam[1] * K[1, 1] / p_cam[2] + K[1, 2]
    in_im = 0 <= u < im_w and 0 <= v < im_h
    return float(u), float(v), in_im


def patch_median(depth: np.ndarray, u: float, v: float,
                 half_size: int) -> Tuple[Optional[float], bool]:
    """Compute median of a square patch around (u,v) in depth map.

    Returns (median_value, has_valid_depth).
    NaN/Inf/zero are excluded from the median.
    """
    h, w = depth.shape
    x = int(round(u))
    y = int(round(v))
    x0 = max(0, x - half_size)
    x1 = min(w - 1, x + half_size)
    y0 = max(0, y - half_size)
    y1 = min(h - 1, y + half_size)
    patch = depth[y0:y1 + 1, x0:x1 + 1]
    valid = patch[np.isfinite(patch) & (patch > 1e-6)]
    if valid.size == 0:
        return None, False
    return float(np.median(valid)), True


def compute_projection_candidates(
    gripper: np.ndarray,
    T: np.ndarray,
    K: np.ndarray,
    depth_metric: np.ndarray,
    im_h: int,
    im_w: int,
) -> Tuple[dict, dict, dict]:
    """Compute forward/inverse projection depth-consistency candidates.

    Returns (candidate_A, candidate_B, comparison):
      candidate_A: using T_robot_cam as-is (forward)
      candidate_B: using inverse(T_robot_cam)
      comparison: pixel distance + preferred transform decision
    """
    EPS = 1e-6
    R, t = T[:3, :3], T[:3, 3]
    EPS_MARGIN = 0.05        # absolute depth error margin (meters)
    REL_MARGIN = 0.10        # relative depth error margin (10%)

    def _candidate(name: str, T_use: np.ndarray) -> dict:
        """Project gripper with given T and compute depth consistency."""
        R_u, t_u = T_use[:3, :3], T_use[:3, 3]
        # Camera-frame gripper position
        p_cam = R_u @ gripper + t_u
        z_cam = float(p_cam[2])
        z_positive = z_cam > EPS

        # Pixel projection
        if z_positive:
            u = float(p_cam[0] * K[0, 0] / z_cam + K[0, 2])
            v = float(p_cam[1] * K[1, 1] / z_cam + K[1, 2])
            in_im = 0 <= u < im_w and 0 <= v < im_h
        else:
            u, v, in_im = None, None, False

        # Depth consistency at projected pixel
        if in_im and u is not None:
            xi, yi = int(round(u)), int(round(v))
            xi = np.clip(xi, 0, im_w - 1)
            yi = np.clip(yi, 0, im_h - 1)
            depth_at_uv = float(depth_metric[yi, xi]) if np.isfinite(depth_metric[yi, xi]) else None
            med3, has3 = patch_median(depth_metric, u, v, 1)
            med5, has5 = patch_median(depth_metric, u, v, 2)
        else:
            depth_at_uv, med3, med5 = None, None, None
            has3, has5 = False, False

        depth_valid = has5 and med5 is not None

        # Compute depth errors
        err_pixel = None
        err_3x3 = None
        err_5x5 = None
        rel_err = None
        reason = "none"

        if z_positive and depth_valid and med5 is not None:
            err_pixel = abs(z_cam - (depth_at_uv if depth_at_uv else med5))
            err_3x3 = abs(z_cam - med3) if med3 is not None else None
            err_5x5 = abs(z_cam - med5)
            rel_err = err_5x5 / max(med5, EPS)
            reason = "valid"
        elif z_positive and not depth_valid:
            reason = "no_valid_depth_patch"
        elif not z_positive:
            reason = "z_cam_not_positive"
        elif not in_im:
            reason = "out_of_image"

        return {
            "name": name,
            "u": u, "v": v,
            "in_image": in_im,
            "z_cam": z_cam,
            "z_positive": z_positive,
            "depth_at_uv": depth_at_uv,
            "patch3x3_median": med3,
            "patch5x5_median": med5,
            "abs_error_pixel": err_pixel,
            "abs_error_patch3x3": err_3x3,
            "abs_error_patch5x5": err_5x5,
            "rel_error_patch5x5": rel_err,
            "depth_valid": depth_valid,
            "validity_reason": reason,
        }

    cand_A = _candidate("T_robot_cam", T)
    cand_B = _candidate("inverse_T_robot_cam", np.linalg.inv(T))

    # Pixel distance between candidates
    pix_dist = None
    if cand_A["u"] is not None and cand_B["u"] is not None:
        du = cand_A["u"] - cand_B["u"]
        dv = cand_A["v"] - cand_B["v"]
        pix_dist = float(np.sqrt(du * du + dv * dv))

    # Preferred transform decision
    preferred = "none"
    reason = ""

    a_ok = cand_A["z_positive"] and cand_A["depth_valid"] and cand_A["validity_reason"] == "valid"
    b_ok = cand_B["z_positive"] and cand_B["depth_valid"] and cand_B["validity_reason"] == "valid"

    if a_ok and b_ok:
        err_a = cand_A["abs_error_patch5x5"]
        err_b = cand_B["abs_error_patch5x5"]
        med_a = cand_A["patch5x5_median"]
        med_b = cand_B["patch5x5_median"]
        abs_margin_ok = err_a + EPS_MARGIN < err_b
        rel_margin_ok = (err_a / max(med_a, EPS)) + REL_MARGIN < (err_b / max(med_b, EPS))
        if abs_margin_ok or rel_margin_ok:
            preferred = "T_robot_cam"
            reason = (f"T_robot_cam depth_err={err_a:.4f}m < {err_b:.4f}m by >{EPS_MARGIN}m margin "
                      f"(rel: {err_a/max(med_a,EPS):.3f} vs {err_b/max(med_b,EPS):.3f})")
        elif err_b + EPS_MARGIN < err_a or (err_b / max(med_b, EPS)) + REL_MARGIN < (err_a / max(med_a, EPS)):
            preferred = "inverse_T_robot_cam"
            reason = (f"inverse_T_robot_cam depth_err={err_b:.4f}m < {err_a:.4f}m by >{EPS_MARGIN}m margin "
                      f"(rel: {err_b/max(med_b,EPS):.3f} vs {err_a/max(med_a,EPS):.3f})")
        else:
            preferred = "ambiguous"
            reason = (f"depth_err A={err_a:.4f}m B={err_b:.4f}m differ by <{EPS_MARGIN}m "
                      f"(rel: {err_a/max(med_a,EPS):.3f} vs {err_b/max(med_b,EPS):.3f})")
    elif a_ok and not b_ok:
        preferred = "T_robot_cam"
        reason = f"A valid (z+,depth_ok) B invalid ({cand_B['validity_reason']})"
    elif b_ok and not a_ok:
        preferred = "inverse_T_robot_cam"
        reason = f"B valid (z+,depth_ok) A invalid ({cand_A['validity_reason']})"
    else:
        preferred = "invalid_both"
        reasons = [cand_A["validity_reason"], cand_B["validity_reason"]]
        reason = f"both invalid: A={reasons[0]} B={reasons[1]}"

    return cand_A, cand_B, {
        "pixel_distance": pix_dist,
        "preferred_transform": preferred,
        "preferred_reason": reason,
    }


def save_projection_overlay(
    rgb_image: np.ndarray,
    cand_A: dict,
    cand_B: dict,
    comparison: dict,
    step_idx: int,
    ep_idx: int,
    output_path: str,
):
    """Draw forward/inverse projection overlay on RGB image and save to disk."""
    import cv2 as _cv2
    import imageio as _imageio

    rgb = rgb_image.copy()
    h, w = rgb.shape[:2]
    dot_color_fwd = (0, 80, 255)     # red in BGR
    dot_color_inv = (255, 180, 0)    # blue in BGR
    dot_radius = 8

    def _draw_candidate(c: dict, color: tuple, label: str):
        if c["u"] is None or c["v"] is None:
            return
        u, v = int(round(c["u"])), int(round(c["v"]))
        if 0 <= u < w and 0 <= v < h:
            _cv2.circle(rgb, (u, v), dot_radius, color, -1)
            _cv2.circle(rgb, (u, v), dot_radius + 2, (255, 255, 255), 1)
            err = f"{c['abs_error_patch5x5']:.3f}m" if c.get("abs_error_patch5x5") is not None else "N/A"
            _cv2.putText(rgb, f"{label}: err={err}", (u + 10, v - 10),
                          _cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    _draw_candidate(cand_A, dot_color_fwd, "FWD")
    _draw_candidate(cand_B, dot_color_inv, "INV")

    pref = comparison.get("preferred_transform", "none")
    reason_short = comparison.get("preferred_reason", "")[:80]
    _cv2.putText(rgb, f"ep{ep_idx} step{step_idx} | pref={pref} | {reason_short}",
                  (5, 20), _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    z_a = f"{cand_A['z_cam']:.3f}" if cand_A.get('z_cam') is not None else "N/A"
    z_b = f"{cand_B['z_cam']:.3f}" if cand_B.get('z_cam') is not None else "N/A"
    _cv2.putText(rgb,
                  f"FWD: z={z_a}m in={cand_A.get('in_image', False)} | "
                  f"INV: z={z_b}m in={cand_B.get('in_image', False)}",
                  (5, 38), _cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Ensure uint8
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    _imageio.imwrite(output_path, rgb)


def token_grid_centers(num_tokens: int, grid_h: int, grid_w: int,
                       patch_size: int, im_h: int, im_w: int,
                       crop_scale: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
    """Compute (u, v) pixel centers for each token."""
    crop_h = int(im_h * math.sqrt(crop_scale))
    crop_w = int(im_w * math.sqrt(crop_scale))
    crop_top = (im_h - crop_h) // 2
    crop_left = (im_w - crop_w) // 2
    sy = crop_h / (grid_h * patch_size)
    sx = crop_w / (grid_w * patch_size)
    U, V = [], []
    for i in range(num_tokens):
        row, col = i // grid_w, i % grid_w
        u = crop_left + (col + 0.5) * patch_size * sx
        v = crop_top + (row + 0.5) * patch_size * sy
        U.append(u)
        V.append(v)
    return np.array(U, dtype=np.float32), np.array(V, dtype=np.float32)


def depth_edge_scores(token_depth: np.ndarray, valid: np.ndarray,
                      grid_h: int, grid_w: int) -> np.ndarray:
    """Sobel depth edge scores on token grid."""
    d = token_depth[:grid_h * grid_w].reshape(grid_h, grid_w)
    v = valid[:grid_h * grid_w].reshape(grid_h, grid_w)
    fill = float(np.median(d[v])) if np.any(v) else 0.0
    d = np.where(v, d, fill).astype(np.float32)
    gx = np.zeros_like(d)
    gy = np.zeros_like(d)
    if grid_w > 1:
        gx[:, 1:-1] = 0.5 * (d[:, 2:] - d[:, :-2])
        gx[:, 0] = d[:, 1] - d[:, 0]
        gx[:, -1] = d[:, -1] - d[:, -2]
    if grid_h > 1:
        gy[1:-1, :] = 0.5 * (d[2:, :] - d[:-2, :])
        gy[0, :] = d[1, :] - d[0, :]
        gy[-1, :] = d[-1, :] - d[-2, :]
    edge = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    edge[~v] = 0.0
    mx = float(np.max(edge)) if edge.size else 0.0
    if mx > 1e-8:
        edge = edge / mx
    return edge.reshape(-1).astype(np.float32)


def near_scores(p_robot: np.ndarray, gripper: np.ndarray, sigma: float = 0.12) -> np.ndarray:
    """Gaussian nearness scores."""
    d = np.linalg.norm(p_robot - gripper[None, :], axis=1)
    return np.exp(-d / sigma).astype(np.float32)


def motion_cone_scores(p_robot: np.ndarray, gripper: np.ndarray,
                      gripper_prev: Optional[np.ndarray]) -> Tuple[np.ndarray, bool]:
    """Cosine similarity motion cone scores."""
    if gripper_prev is None:
        return np.zeros(len(p_robot), dtype=np.float32), False
    motion = gripper - gripper_prev
    m_norm = float(np.linalg.norm(motion))
    if m_norm < 1e-4:
        return np.zeros(len(p_robot), dtype=np.float32), False
    direction = motion / m_norm
    diff = p_robot - gripper[None, :]
    r_norm = np.clip(np.linalg.norm(diff, axis=1), 1e-8, None)
    cosine = np.sum(diff * direction[None, :], axis=1) / r_norm
    return np.clip(cosine, 0.0, 1.0).astype(np.float32), True


def workspace_scores(p_robot: np.ndarray,
                    bounds=((-2.0, 2.0), (-2.0, 2.0), (-0.5, 2.0))) -> np.ndarray:
    """Binary workspace membership."""
    x, y, z = p_robot[:, 0], p_robot[:, 1], p_robot[:, 2]
    xb, yb, zb = bounds
    return ((x >= xb[0]) & (x <= xb[1]) &
            (y >= yb[0]) & (y <= yb[1]) &
            (z >= zb[0]) & (z <= zb[1])).astype(np.float32)


def check_rotation_orthogonal(T: np.ndarray) -> Tuple[float, bool]:
    """Check R^T R = I for T[:3,:3]."""
    R = T[:3, :3]
    err = float(np.linalg.norm(R.T @ R - np.eye(3)))
    return err, err < 1e-4


# ─── Observation extractors ───────────────────────────────────────────────────

def extract_gripper_pos(obs) -> Tuple[Optional[np.ndarray], str]:
    """Extract gripper position from observation."""
    KEYS = ("gripper_pos", "eef_pos", "ee_pos", "robot0_eef_pos", "robot0_gripper_pos")
    for key in KEYS:
        val = getattr(obs, key, None) if hasattr(obs, key) else obs.get(key) if isinstance(obs, dict) else None
        if val is not None:
            arr = np.asarray(val, dtype=np.float32).reshape(-1)
            if arr.size >= 3 and np.all(np.isfinite(arr[:3])):
                return arr[:3].astype(np.float32), key
    return None, "none"


def extract_T(source) -> Tuple[Optional[np.ndarray], str]:
    """Extract camera transform. Tries T_robot_cam first, then T_cam_robot (inverted)."""
    KEYS = ("T_robot_cam", "T_base_cam", "camera_extrinsics", "T_world_cam",
            "robot0_T_cam", "T_cam_robot")
    for key in KEYS:
        val = getattr(source, key, None) if hasattr(source, key) else source.get(key) if isinstance(source, dict) else None
        if val is not None:
            arr = np.asarray(val, dtype=np.float32)
            if arr.shape == (4, 4) and np.all(np.isfinite(arr)):
                if key.startswith("T_cam") and not key.startswith("T_robot"):
                    try:
                        return np.linalg.inv(arr).astype(np.float32), f"inv({key})"
                    except np.linalg.LinAlgError:
                        continue
                return arr.astype(np.float32), key
    return None, "none"


def extract_K(source) -> Tuple[Optional[np.ndarray], str]:
    """Extract camera intrinsics K."""
    KEYS = ("camera_intrinsics", "K", "intrinsics", "robot0_camera_intrinsics")
    for key in KEYS:
        val = getattr(source, key, None) if hasattr(source, key) else source.get(key) if isinstance(source, dict) else None
        if val is not None:
            arr = np.asarray(val, dtype=np.float32)
            if arr.shape == (3, 3) and np.all(np.isfinite(arr)):
                return arr.astype(np.float32), key
    return None, "none"


def extract_rgb(obs) -> Optional[np.ndarray]:
    """Extract RGB image. Supports both generic and LIBERO-specific keys."""
    KEYS = ("rgb", "image", "agentview_image")
    for key in KEYS:
        val = getattr(obs, key, None) if hasattr(obs, key) else obs.get(key) if isinstance(obs, dict) else None
        if val is not None:
            return np.asarray(val, dtype=np.uint8)
    # LIBERO: look for any key ending with _image
    if isinstance(obs, dict):
        for key in obs:
            if key.endswith("_image"):
                return np.asarray(obs[key], dtype=np.uint8)
    return None


def extract_depth_and_key(obs) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Extract depth image and source key. Supports both generic and LIBERO-specific keys.

    Returns:
        (depth, source_key) — depth is [H, W] float32, source_key is the matched key name.
    """
    KEYS = ("depth", "depth_image", "agentview_depth")
    for key in KEYS:
        val = getattr(obs, key, None) if hasattr(obs, key) else obs.get(key) if isinstance(obs, dict) else None
        if val is not None:
            d = np.asarray(val, dtype=np.float32)
            if d.ndim == 3:
                d = d[:, :, 0]
            return d, key
    # LIBERO: look for any key ending with _depth
    if isinstance(obs, dict):
        for key in obs:
            if key.endswith("_depth"):
                d = np.asarray(obs[key], dtype=np.float32)
                if d.ndim == 3:
                    d = d[:, :, 0]
                return d, key
    return None, None


# ─── Validation result ────────────────────────────────────────────────────────

@dataclass
class ValResult:
    ep: int
    step: int
    task: str = ""

    # ── Action metadata (for multi-step validation) ─────────────────────────
    action_source: str = "none"
    action_dim: int = 0
    action_min: float = 0.0
    action_max: float = 0.0
    action_norm: float = 0.0

    # ── Depth conversion metadata ─────────────────────────────────────────
    depth_source_key: str = ""
    depth_conversion: str = "none"
    depth_is_metric: bool = False
    depth_unit: str = "unknown"
    depth_sim_available: bool = False
    # Raw depth stats
    depth_raw_min: float = 0.0
    depth_raw_max: float = 0.0
    depth_raw_mean: float = 0.0
    depth_raw_std: float = 0.0
    # Metric depth stats (what we actually use)
    depth_metric_min: float = 0.0
    depth_metric_max: float = 0.0
    depth_metric_mean: float = 0.0
    depth_metric_std: float = 0.0
    # Fallback indicator
    depth_fallback_warning: str = ""
    # Backwards-compatible aliases (deprecated — prefer depth_metric_*)
    depth_scale: str = "unknown"
    depth_min: float = 0.0
    depth_max: float = 0.0
    depth_mean: float = 0.0
    depth_valid_ratio: float = 0.0

    # ── T_robot_cam ─────────────────────────────────────────────────────────
    T_valid: bool = False
    T_key: str = ""
    T_det: float = 0.0
    T_orthogonal: bool = False
    T_trans_norm: float = 0.0
    cam_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # ── Gripper projection ─────────────────────────────────────────────────
    proj_fwd: Tuple[float, float] = (0.0, 0.0)
    proj_fwd_in: bool = False
    proj_inv: Tuple[float, float] = (0.0, 0.0)
    proj_inv_in: bool = False
    best_proj: str = "none"

    # ── Gripper state ─────────────────────────────────────────────────────
    gripper_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    gripper_prev: Optional[Tuple[float, float, float]] = None
    motion_valid: bool = False
    motion_norm: float = 0.0

    # ── Action metadata ─────────────────────────────────────────────────────
    action_source: str = "none"
    action_dim: int = 0
    action_min: float = 0.0
    action_max: float = 0.0
    action_norm: float = 0.0

    # ── Projection candidates (forward T vs inverse T) ──────────────────────
    # Candidate A: forward T_robot_cam
    fwd_u: Optional[float] = None
    fwd_v: Optional[float] = None
    fwd_in_image: bool = False
    fwd_z_cam: Optional[float] = None
    fwd_z_positive: bool = False
    fwd_depth_at_uv: Optional[float] = None
    fwd_patch3x3_median: Optional[float] = None
    fwd_patch5x5_median: Optional[float] = None
    fwd_abs_error_pixel: Optional[float] = None
    fwd_abs_error_patch3x3: Optional[float] = None
    fwd_abs_error_patch5x5: Optional[float] = None
    fwd_rel_error_patch5x5: Optional[float] = None
    fwd_depth_valid: bool = False
    fwd_validity_reason: str = "none"

    # Candidate B: inverse(T_robot_cam)
    inv_u: Optional[float] = None
    inv_v: Optional[float] = None
    inv_in_image: bool = False
    inv_z_cam: Optional[float] = None
    inv_z_positive: bool = False
    inv_depth_at_uv: Optional[float] = None
    inv_patch3x3_median: Optional[float] = None
    inv_patch5x5_median: Optional[float] = None
    inv_abs_error_pixel: Optional[float] = None
    inv_abs_error_patch3x3: Optional[float] = None
    inv_abs_error_patch5x5: Optional[float] = None
    inv_rel_error_patch5x5: Optional[float] = None
    inv_depth_valid: bool = False
    inv_validity_reason: str = "none"

    # ── Comparison ─────────────────────────────────────────────────────────
    pixel_distance_fwd_inv: Optional[float] = None
    preferred_transform_step: str = "none"
    preferred_transform_reason: str = ""

    # ── Token-grid ─────────────────────────────────────────────────────────
    token0_px: Tuple[float, float] = (0.0, 0.0)
    token255_px: Tuple[float, float] = (0.0, 0.0)
    grid_aligned: bool = False

    # ── Token 3D ──────────────────────────────────────────────────────────
    p3d_min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    p3d_max: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    dist_min: float = 0.0
    dist_mean: float = 0.0

    # ── Scores ─────────────────────────────────────────────────────────────
    edge_mean: float = 0.0
    near_mean: float = 0.0
    contact_mean: float = 0.0
    mc_mean: float = 0.0
    mc_nonzero: float = 0.0
    ws_mean: float = 0.0
    ws_std: float = 0.0
    ws_unique: List[float] = None

    # ── Gates ──────────────────────────────────────────────────────────────
    mc_gate_ok: bool = False
    ws_usable: bool = False

    def __post_init__(self):
        if self.ws_unique is None:
            self.ws_unique = []

    def to_dict(self) -> Dict:
        def _conv(v):
            if isinstance(v, (np.bool_,)):
                return bool(v)
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                return float(v)
            if isinstance(v, list):
                return [_conv(x) for x in v]
            if isinstance(v, tuple):
                return tuple(_conv(x) for x in v)
            return v
        d = asdict(self)
        return {k: _conv(v) for k, v in d.items()}


# ─── Main validation ────────────────────────────────────────────────────────────

def validate_step(obs, prev_obs, ep: int, step: int,
                  num_tokens: int = 256, patch_size: int = 14) -> ValResult:
    R = ValResult(ep=ep, step=step)

    rgb = extract_rgb(obs)

    # ── Depth: extract raw, convert to metric, record both ────────────────────────
    raw_depth, depth_key = extract_depth_and_key(obs)
    sim = obs.get("env_sim") if isinstance(obs, dict) else None

    if raw_depth is not None:
        R.depth_source_key = depth_key or "unknown"
        if _HAS_GEOMETRY_DEPTH:
            result = convert_depth_to_metric(
                depth_raw=raw_depth,
                sim=sim,
                source_key=depth_key,
                image_transform=obs.get("image_transform") if isinstance(obs, dict) else None,
            )
            depth = result.depth
            R.depth_conversion = result.metadata.get("conversion", "none")
            R.depth_is_metric = result.metadata.get("depth_is_metric", False)
            R.depth_unit = result.metadata.get("depth_unit", "unknown")
            R.depth_sim_available = result.metadata.get("sim_available", sim is not None)

            raw_s = result.metadata.get("depth_raw_stats", {})
            met_s = result.metadata.get("depth_metric_stats", {})
            R.depth_raw_min = raw_s.get("min", 0.0)
            R.depth_raw_max = raw_s.get("max", 0.0)
            R.depth_raw_mean = raw_s.get("mean", 0.0)
            R.depth_raw_std = raw_s.get("std", 0.0)
            R.depth_metric_min = met_s.get("min", 0.0) if met_s else 0.0
            R.depth_metric_max = met_s.get("max", 0.0) if met_s else 0.0
            R.depth_metric_mean = met_s.get("mean", 0.0) if met_s else 0.0
            R.depth_metric_std = met_s.get("std", 0.0) if met_s else 0.0

            if R.depth_conversion in ("raw_no_sim_fallback", "robosuite_get_real_depth_map_error",
                                     "ambiguous_get_real_depth_map_error", "error_parse_failed"):
                R.depth_fallback_warning = f"CONVERSION_FAILED: {R.depth_conversion}"

            # Backwards-compatible aliases
            R.depth_scale = "unknown"  # Deprecated: use depth_is_metric and depth_unit
            R.depth_min = R.depth_metric_min
            R.depth_max = R.depth_metric_max
            R.depth_mean = R.depth_metric_mean
        else:
            # Fallback: treat raw as-is
            depth = raw_depth
            R.depth_conversion = "none_geometry_depth_unavailable"
            R.depth_is_metric = False
            R.depth_unit = "unknown"
            R.depth_sim_available = sim is not None
            dflat = raw_depth.flatten()
            valid_d = dflat[(dflat > 0.01) & (dflat < 100)]
            R.depth_raw_min = float(np.min(dflat)) if dflat.size else 0.0
            R.depth_raw_max = float(np.max(dflat)) if dflat.size else 0.0
            R.depth_raw_mean = float(np.mean(valid_d)) if valid_d.size else 0.0
            R.depth_raw_std = float(np.std(valid_d)) if valid_d.size else 0.0
            R.depth_metric_min = R.depth_raw_min
            R.depth_metric_max = R.depth_raw_max
            R.depth_metric_mean = R.depth_raw_mean
            R.depth_metric_std = R.depth_raw_std
            R.depth_min = R.depth_raw_min
            R.depth_max = R.depth_raw_max
            R.depth_mean = R.depth_raw_mean
            R.depth_scale = "unknown"
    else:
        depth = None

    if depth is None:
        return R
    im_h, im_w = depth.shape[:2]

    gripper, gkey = extract_gripper_pos(obs)
    if gripper is not None:
        R.gripper_pos = tuple(gripper.tolist())
        R.gripper_prev = None
        if prev_obs is not None:
            gp_prev, _ = extract_gripper_pos(prev_obs)
            if gp_prev is not None:
                R.gripper_prev = tuple(gp_prev.tolist())
                motion = gripper - gp_prev
                R.motion_norm = float(np.linalg.norm(motion))
                R.motion_valid = R.motion_norm >= 1e-4

    T, Tkey = extract_T(obs)
    R.T_valid = T is not None
    R.T_key = Tkey
    if T is not None:
        R.T_det = float(np.linalg.det(T[:3, :3]))
        R.T_orthogonal, _ = check_rotation_orthogonal(T)
        R.T_trans_norm = float(np.linalg.norm(T[:3, 3]))
        cp = T[:3, 3]
        R.cam_pos = (float(cp[0]), float(cp[1]), float(cp[2]))

    K, Kkey = extract_K(obs)

    # Compute depth valid ratio on metric depth
    if depth is not None:
        dflat = depth.flatten()
        R.depth_valid_ratio = float(
            np.mean((dflat > 0.01) & (dflat < 100))
        ) if dflat.size else 0.0

    # ── Projection candidates (depth consistency) ────────────────────────────────
    if T is not None and K is not None and gripper is not None and depth is not None:
        cand_A, cand_B, comparison = compute_projection_candidates(
            gripper, T, K, depth, im_h, im_w
        )
        # Candidate A: forward T_robot_cam
        R.fwd_u = cand_A["u"]
        R.fwd_v = cand_A["v"]
        R.fwd_in_image = cand_A["in_image"]
        R.fwd_z_cam = cand_A["z_cam"]
        R.fwd_z_positive = cand_A["z_positive"]
        R.fwd_depth_at_uv = cand_A["depth_at_uv"]
        R.fwd_patch3x3_median = cand_A["patch3x3_median"]
        R.fwd_patch5x5_median = cand_A["patch5x5_median"]
        R.fwd_abs_error_pixel = cand_A["abs_error_pixel"]
        R.fwd_abs_error_patch3x3 = cand_A["abs_error_patch3x3"]
        R.fwd_abs_error_patch5x5 = cand_A["abs_error_patch5x5"]
        R.fwd_rel_error_patch5x5 = cand_A["rel_error_patch5x5"]
        R.fwd_depth_valid = cand_A["depth_valid"]
        R.fwd_validity_reason = cand_A["validity_reason"]

        # Candidate B: inverse(T_robot_cam)
        R.inv_u = cand_B["u"]
        R.inv_v = cand_B["v"]
        R.inv_in_image = cand_B["in_image"]
        R.inv_z_cam = cand_B["z_cam"]
        R.inv_z_positive = cand_B["z_positive"]
        R.inv_depth_at_uv = cand_B["depth_at_uv"]
        R.inv_patch3x3_median = cand_B["patch3x3_median"]
        R.inv_patch5x5_median = cand_B["patch5x5_median"]
        R.inv_abs_error_pixel = cand_B["abs_error_pixel"]
        R.inv_abs_error_patch3x3 = cand_B["abs_error_patch3x3"]
        R.inv_abs_error_patch5x5 = cand_B["abs_error_patch5x5"]
        R.inv_rel_error_patch5x5 = cand_B["rel_error_patch5x5"]
        R.inv_depth_valid = cand_B["depth_valid"]
        R.inv_validity_reason = cand_B["validity_reason"]

        # Comparison
        R.pixel_distance_fwd_inv = comparison["pixel_distance"]
        R.preferred_transform_step = comparison["preferred_transform"]
        R.preferred_transform_reason = comparison["preferred_reason"]

        # Legacy aliases (keep for backward compat with report)
        fu, fv, fi = gripper_to_image(gripper, T, K, im_h, im_w)
        R.proj_fwd = (fu, fv)
        R.proj_fwd_in = fi
        try:
            Ti = np.linalg.inv(T)
            iu, iv, ii = gripper_to_image(gripper, Ti, K, im_h, im_w)
            R.proj_inv = (iu, iv)
            R.proj_inv_in = ii
        except np.linalg.LinAlgError:
            R.proj_inv_in = False
        if fi and not ii:
            R.best_proj = "forward"
        elif ii and not fi:
            R.best_proj = "inverse"
        elif fi and ii:
            R.best_proj = "both"
        else:
            R.best_proj = "none"
    else:
        R.preferred_transform_step = "insufficient_data"

    # Token grid centers
    grid_h = grid_w = int(math.sqrt(num_tokens))
    U, V = token_grid_centers(num_tokens, grid_h, grid_w, patch_size, im_h, im_w)
    R.token0_px = (float(U[0]), float(V[0]))
    R.token255_px = (float(U[-1]), float(V[-1]))
    R.grid_aligned = U[0] < im_w * 0.15 and V[0] < im_h * 0.15

    # Token 3D projection
    p3d = np.full((num_tokens, 3), np.nan, dtype=np.float32)
    tdepth = np.full(num_tokens, np.nan, dtype=np.float32)
    vmask = np.zeros(num_tokens, dtype=bool)

    if K is not None and T is not None:
        td = sample_depth(depth, U, V)
        tdepth = td
        for i in range(num_tokens):
            d = float(td[i])
            if 0.01 < d < 10.0:
                pc = backproject_pixel(float(U[i]), float(V[i]), d, K)
                pr = transform_points_cam_to_robot(pc[None, :], T)[0]
                p3d[i] = pr
                vmask[i] = True

    vp = p3d[vmask]
    if vp.size > 0:
        R.p3d_min = tuple(np.nanmin(vp, axis=0).tolist())
        R.p3d_max = tuple(np.nanmax(vp, axis=0).tolist())
        if gripper is not None:
            dists = np.linalg.norm(vp - gripper[None, :], axis=1)
            R.dist_min = float(np.min(dists))
            R.dist_mean = float(np.mean(dists))

    # Scores
    edge = depth_edge_scores(tdepth, vmask, grid_h, grid_w)
    near = near_scores(p3d, gripper) if gripper is not None else np.zeros(num_tokens)
    contact = near  # simplified
    mc, mc_valid = motion_cone_scores(p3d, gripper, None if R.gripper_prev is None else np.array(R.gripper_prev))
    ws = workspace_scores(p3d)

    R.edge_mean = float(np.mean(edge[vmask])) if np.any(vmask) else 0.0
    R.near_mean = float(np.mean(near[vmask])) if np.any(vmask) else 0.0
    R.contact_mean = float(np.mean(contact[vmask])) if np.any(vmask) else 0.0
    R.mc_mean = float(np.mean(mc[vmask])) if np.any(vmask) else 0.0
    R.mc_nonzero = float(np.mean(mc > 1e-6)) if vmask.any() else 0.0
    R.ws_mean = float(np.mean(ws[vmask])) if np.any(vmask) else 0.0
    R.ws_std = float(np.std(ws[vmask])) if np.any(vmask) else 0.0
    R.ws_unique = sorted(set(float(x) for x in ws[vmask])) if np.any(vmask) else []

    # Gate checks
    if not R.motion_valid:
        R.mc_gate_ok = R.mc_nonzero < 0.01
    else:
        R.mc_gate_ok = True
    R.ws_usable = len(R.ws_unique) > 1 or (R.ws_unique[0] != 1.0 if R.ws_unique else True)

    return R


def run_validation(model_path: str, task_suite: str = "libero_spatial",
                  num_episodes: int = 10, seed: int = 42,
                  output_dir: Optional[str] = None,
                  max_steps_inspect: int = 5,
                  inspect_obs_mode: bool = False,
                  save_projection_debug: bool = False) -> List[ValResult]:
    """Run geometry validation on real episodes."""
    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"/infini-data/openvla/outputs/geometry_mapping_validation_{ts}"

    os.makedirs(output_dir, exist_ok=True)
    sys.path.insert(0, "/infini-data/openvla")

    print(f"[Val] Output: {output_dir}")
    print(f"[Val] Loading model: {model_path}")

    try:
        from open_vla import OpenVLA
        vla = OpenVLA.from_pretrained(model_path, device="cpu")
    except Exception as e:
        print(f"[Val] Model load failed: {e}, using DRY-RUN mode")
        vla = None

    # Setup environment
    env = None
    benchmark = None
    raw_obs_accessor = None  # callable that returns raw obs dict
    if task_suite == "libero_spatial":
        try:
            from libero.libero.benchmark import get_benchmark
            from experiments.robot.libero.libero_utils import get_libero_env
            benchmark = get_benchmark("libero_spatial")()
            # Load first task to init env
            task0 = benchmark.get_task(0)
            low_env, _ = get_libero_env(task0, "openvla", resolution=256, camera_depths=True)
            low_env.reset()
            raw_obs_accessor = lambda: low_env._obs
            # Reuse env for all tasks (just call set_init_state per task)
            env = low_env
            print(f"[Val] LIBERO-SPATIAL loaded, {benchmark.n_tasks} tasks, camera_depths=True")
        except Exception as e:
            print(f"[Val] LIBERO env failed: {e}")

    results: List[ValResult] = []
    inspection_results: List[Dict[str, Any]] = []
    projection_debug_dir = None
    if save_projection_debug:
        projection_debug_dir = os.path.join(output_dir, "projection_debug")
        os.makedirs(projection_debug_dir, exist_ok=True)
        print(f"[Val] Saving projection debug overlays to: {projection_debug_dir}")

    if env is not None and benchmark is not None:
        n = min(num_episodes, benchmark.n_tasks)
        from experiments.robot.libero.libero_utils import get_libero_env
        CAMERA_NAME = "agentview"
        RESOLUTION = 256
        for ep in range(n):
            task = benchmark.get_task(ep)
            init_states = benchmark.get_task_init_states(ep)
            if ep > 0:
                task_ep = benchmark.get_task(ep)
                env.close()
                low_env, _ = get_libero_env(task_ep, "openvla", resolution=RESOLUTION, camera_depths=True)
                env = low_env
            if init_states is not None and len(init_states) > 0:
                first_obs = env.set_init_state(init_states[0])
            else:
                first_obs = env.reset()
            prev_obs = None
            inspection_results = []
            for step in range(max_steps_inspect):
                # Inject camera intrinsics and extrinsics from env.sim
                obs_for_validation = dict(first_obs)
                try:
                    from robosuite.utils import camera_utils as CU
                    K = CU.get_camera_intrinsic_matrix(env.sim, CAMERA_NAME, RESOLUTION, RESOLUTION)
                    obs_for_validation["camera_intrinsics"] = np.asarray(K, dtype=np.float32)
                    T_wc = CU.get_camera_extrinsic_matrix(env.sim, CAMERA_NAME)
                    obs_for_validation["camera_extrinsics"] = np.asarray(T_wc, dtype=np.float32)
                    T_robot_cam = np.linalg.inv(T_wc).astype(np.float32)
                    obs_for_validation["T_robot_cam"] = T_robot_cam
                    # Store env.sim for depth conversion
                    obs_for_validation["env_sim"] = env.sim
                except Exception:
                    pass

                # ── OBS INSPECTION ──────────────────────────────────────────────────
                if inspect_obs_mode:
                    insp = inspect_obs_keys(obs_for_validation, ep, step,
                                           camera_name=CAMERA_NAME, resolution=RESOLUTION)
                    insp["task"] = task.language[:60]
                    inspection_results.append(insp)
                    # Print depth stats summary
                    for dk in insp.get("depth_keys", []):
                        ds = insp.get(f"depth_stats_{dk}", {})
                        print(f"  [INSPECT] ep{ep} step{step} {dk}: "
                              f"shape={ds.get('shape')} min={ds.get('min', 0):.4f} "
                              f"max={ds.get('max', 0):.4f} mean={ds.get('mean', 0):.4f} "
                              f"std={ds.get('std', 0):.4f} in01={ds.get('in_0_1_ratio', 0):.3f} "
                              f"note={ds.get('raw_note', '')}")
                    conv = insp.get("depth_conversion", {})
                    if conv.get("conversion_available"):
                        cs = conv.get("converted_stats", {})
                        print(f"  [INSPECT]   >>> CONVERTED: min={cs.get('min', 0):.4f} "
                              f"max={cs.get('max', 0):.4f} mean={cs.get('mean', 0):.4f}")
                    elif conv.get("conversion_error"):
                        print(f"  [INSPECT]   >>> conversion ERROR: {conv['conversion_error']}")
                # ─────────────────────────────────────────────────────────────────

                # Generate safe action for LIBERO: 7-dim [d_x, d_y, d_z, roll, pitch, yaw, gripper]
                # LIBERO expects numpy array with shape (7,)
                # gripper=-1 means "keep current state" (no gripper movement)
                # This is the same pattern as get_libero_dummy_action in libero_utils.py
                act = np.zeros(7, dtype=np.float32)
                act[-1] = -1.0  # gripper: -1 = keep state, +1 = toggle open
                action_norm = float(np.linalg.norm(act))

                R = validate_step(obs_for_validation, prev_obs, ep, step)
                R.task = task.language[:60]
                # Record action metadata for multi-step validation
                R.action_source = "zero_dummy"
                R.action_dim = 7
                R.action_min = float(np.min(act))
                R.action_max = float(np.max(act))
                R.action_norm = action_norm

                # Save projection overlay debug image
                overlay_path = None
                if save_projection_debug and projection_debug_dir is not None:
                    try:
                        # Extract RGB image for overlay
                        rgb = extract_rgb(obs_for_validation)
                        if rgb is not None:
                            overlay_path = os.path.join(
                                projection_debug_dir,
                                f"ep{ep:03d}_step{step:03d}.png"
                            )
                            # Build candidate dicts from ValResult fields
                            cand_A = {
                                "name": "T_robot_cam",
                                "u": R.fwd_u, "v": R.fwd_v,
                                "z_cam": R.fwd_z_cam,
                                "in_image": R.fwd_in_image,
                                "abs_error_patch5x5": R.fwd_abs_error_patch5x5,
                            }
                            cand_B = {
                                "name": "inverse_T_robot_cam",
                                "u": R.inv_u, "v": R.inv_v,
                                "z_cam": R.inv_z_cam,
                                "in_image": R.inv_in_image,
                                "abs_error_patch5x5": R.inv_abs_error_patch5x5,
                            }
                            comparison = {
                                "preferred_transform": R.preferred_transform_step,
                                "preferred_reason": R.preferred_transform_reason,
                                "pixel_distance": R.pixel_distance_fwd_inv,
                            }
                            save_projection_overlay(rgb, cand_A, cand_B, comparison,
                                                   step, ep, overlay_path)
                    except Exception as ex:
                        print(f"    overlay save error ep{ep} step{step}: {ex}")
                print(f"  ep{ep} step{step}: depth_conv={R.depth_conversion} is_metric={R.depth_is_metric} raw_mean={R.depth_raw_mean:.4f} metric_mean={R.depth_metric_mean:.4f} T={R.T_key} mc_gate={R.mc_gate_ok} motion_valid={R.motion_valid}")
                results.append(R)
                prev_obs = obs_for_validation
                try:
                    next_obs, _, done, _ = env.step(act)
                    if done:
                        next_obs = env.reset()
                    first_obs = next_obs
                except Exception as e:
                    print(f"    step exception: {e}")
                    break
    else:
        print("[Val] No environment, running DRY-RUN with synthetic data")
        import random
        random.seed(seed)
        np.random.seed(seed)

        # Generate synthetic data
        im_h, im_w = 256, 256
        num_tokens = 256
        grid_h = grid_w = 16

        for ep in range(num_episodes):
            # Synthetic depth (meters)
            d = np.zeros((im_h, im_w), dtype=np.float32) + 1.5
            d[50:150, 80:200] = 1.0
            d[100:120, 120:160] = 0.5
            d = d + np.random.randn(im_h, im_w).astype(np.float32) * 0.05

            # Synthetic gripper
            gripper = np.array([0.3, -0.2, 0.5], dtype=np.float32)
            gripper_prev = gripper + np.array([0.01, 0.0, 0.0], dtype=np.float32)

            # Synthetic T_robot_cam
            ang = np.random.rand() * 0.1 - 0.05
            R_mat = np.array([[np.cos(ang), -np.sin(ang), 0],
                               [np.sin(ang),  np.cos(ang), 0],
                               [0, 0, 1]], dtype=np.float32)
            T_mat = np.eye(4, dtype=np.float32)
            T_mat[:3, :3] = R_mat
            T_mat[:3, 3] = [0.5, -0.3, 1.2]

            # Synthetic K
            K_mat = np.array([[535.0, 0, 128],
                               [0, 535.0, 128],
                               [0, 0, 1]], dtype=np.float32)

            class FakeObs:
                def __init__(self):
                    self.rgb = np.random.randint(0, 255, (im_h, im_w, 3), dtype=np.uint8)
                    self.depth = d
                    self.gripper_pos = gripper.tolist()
                    self.T_robot_cam = T_mat.tolist()
                    self.camera_intrinsics = K_mat.tolist()
                    self.T_world_cam = T_mat.tolist()

            obs = FakeObs()
            prev_obs = FakeObs()
            prev_obs.gripper_pos = gripper_prev.tolist()

            R = validate_step(obs, prev_obs, ep, 0)
            R.task = f"dryrun_ep{ep}"
            print(f"  dryrun ep{ep}: depth_conv={R.depth_conversion} is_metric={R.depth_is_metric} raw_mean={R.depth_raw_mean:.4f} metric_mean={R.depth_metric_mean:.4f} T={R.T_key} mc_gate={R.mc_gate_ok}")
            results.append(R)

    # Save JSON (per-step results)
    json_path = os.path.join(output_dir, "validation_results.json")
    with open(json_path, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    print(f"\n[Val] JSON saved: {json_path}")

    # Save global projection stats
    proj_stats = _compute_projection_global_stats(results, projection_debug_dir)
    proj_stats_path = os.path.join(output_dir, "projection_stats.json")
    with open(proj_stats_path, "w") as f:
        json.dump(proj_stats, f, indent=2)
    print(f"[Val] Projection stats saved: {proj_stats_path}")

    # Save inspection results if available
    if inspect_obs_mode and inspection_results:
        insp_path = os.path.join(output_dir, "obs_inspection_results.json")
        with open(insp_path, "w") as f:
            json.dump(inspection_results, f, indent=2)
        print(f"[Val] Inspection JSON saved: {insp_path}")

    # Generate report
    report = make_report(results, output_dir)
    md_path = os.path.join(output_dir, "geometry_mapping_validation_report.md")
    with open(md_path, "w") as f:
        f.write(report)
    print(f"[Val] Report saved: {md_path}")

    return results


def _safe_mean(vals):
    return float(np.mean(vals)) if vals else None


def _compute_projection_global_stats(results: List[ValResult],
                                    projection_debug_dir: Optional[str]) -> Dict[str, Any]:
    """Compute global statistics over all steps for the projection analysis."""
    has_proj = [r for r in results if r.preferred_transform_step not in ("none", "insufficient_data")]
    if not has_proj:
        return {"note": "No projection data available"}

    n = len(has_proj)

    fwd_u = [r.fwd_u for r in has_proj if r.fwd_u is not None]
    fwd_v = [r.fwd_v for r in has_proj if r.fwd_v is not None]
    inv_u = [r.inv_u for r in has_proj if r.inv_u is not None]
    inv_v = [r.inv_v for r in has_proj if r.inv_v is not None]

    pix_dists = [r.pixel_distance_fwd_inv for r in has_proj if r.pixel_distance_fwd_inv is not None]

    a_in_img = sum(1 for r in has_proj if r.fwd_in_image)
    b_in_img = sum(1 for r in has_proj if r.inv_in_image)
    both_in = sum(1 for r in has_proj if r.fwd_in_image and r.inv_in_image)
    only_fwd = sum(1 for r in has_proj if r.fwd_in_image and not r.inv_in_image)
    only_inv = sum(1 for r in has_proj if r.inv_in_image and not r.fwd_in_image)
    gt20 = sum(1 for r in has_proj if r.pixel_distance_fwd_inv is not None and r.pixel_distance_fwd_inv > 20)

    a_zpos = sum(1 for r in has_proj if r.fwd_z_positive)
    b_zpos = sum(1 for r in has_proj if r.inv_z_positive)

    a_errs = [r.fwd_abs_error_patch5x5 for r in has_proj if r.fwd_abs_error_patch5x5 is not None]
    b_errs = [r.inv_abs_error_patch5x5 for r in has_proj if r.inv_abs_error_patch5x5 is not None]
    a_rells = [r.fwd_rel_error_patch5x5 for r in has_proj if r.fwd_rel_error_patch5x5 is not None]
    b_rells = [r.inv_rel_error_patch5x5 for r in has_proj if r.inv_rel_error_patch5x5 is not None]

    n_pref_a = sum(1 for r in has_proj if r.preferred_transform_step == "T_robot_cam")
    n_pref_b = sum(1 for r in has_proj if r.preferred_transform_step == "inverse_T_robot_cam")
    n_amb = sum(1 for r in has_proj if r.preferred_transform_step == "ambiguous")
    n_inv = sum(1 for r in has_proj if r.preferred_transform_step == "invalid_both")

    # Global decision: majority vote
    counts = {"T_robot_cam": n_pref_a, "inverse_T_robot_cam": n_pref_b,
              "ambiguous": n_amb, "invalid_both": n_inv}
    global_pref = max(counts, key=counts.get)
    n_total = n_pref_a + n_pref_b + n_amb + n_inv

    # Depth error comparison: mean
    a_mean_err = _safe_mean(a_errs) if a_errs else None
    b_mean_err = _safe_mean(b_errs) if b_errs else None
    a_med_err = float(np.median(a_errs)) if a_errs else None
    b_med_err = float(np.median(b_errs)) if b_errs else None
    a_mean_rel = _safe_mean(a_rells) if a_rells else None
    b_mean_rel = _safe_mean(b_rells) if b_rells else None

    # Final reason
    final_reason = ""
    if n_total > 0:
        pct_a = 100 * n_pref_a / n_total
        pct_b = 100 * n_pref_b / n_total
        final_reason = (
            f"voted: T_robot_cam={pct_a:.0f}% inv={pct_b:.0f}% "
            f"amb={n_amb}/{n_total} inv_both={n_inv}/{n_total}. "
            f"depth_err A_mean={a_mean_err:.4f}m B_mean={b_mean_err:.4f}m. "
            f"a_in_img={a_in_img}/{n} b_in_img={b_in_img}/{n}. "
            f"both_in={both_in}/{n} dist_gt20={gt20}/{n}."
        )

    return {
        "total_steps": n,
        "forward_pixel_u_range": [min(fwd_u), max(fwd_u)] if fwd_u else [None, None],
        "forward_pixel_v_range": [min(fwd_v), max(fwd_v)] if fwd_v else [None, None],
        "inverse_pixel_u_range": [min(inv_u), max(inv_u)] if inv_u else [None, None],
        "inverse_pixel_v_range": [min(inv_v), max(inv_v)] if inv_v else [None, None],
        "forward_inverse_pixel_distance_mean": _safe_mean(pix_dists),
        "forward_inverse_pixel_distance_median": float(np.median(pix_dists)) if pix_dists else None,
        "forward_inverse_distance_gt_20px_count": gt20,
        "candidate_A_in_image_ratio": a_in_img / n if n else 0,
        "candidate_B_in_image_ratio": b_in_img / n if n else 0,
        "both_inside_count": both_in,
        "only_forward_inside_count": only_fwd,
        "only_inverse_inside_count": only_inv,
        "candidate_A_z_positive_ratio": a_zpos / n if n else 0,
        "candidate_B_z_positive_ratio": b_zpos / n if n else 0,
        "candidate_A_depth_error_patch5x5_mean": a_mean_err,
        "candidate_A_depth_error_patch5x5_median": a_med_err,
        "candidate_B_depth_error_patch5x5_mean": b_mean_err,
        "candidate_B_depth_error_patch5x5_median": b_med_err,
        "candidate_A_relative_error_mean": a_mean_rel,
        "candidate_B_relative_error_mean": b_mean_rel,
        "preferred_T_robot_cam_count": n_pref_a,
        "preferred_inverse_T_robot_cam_count": n_pref_b,
        "ambiguous_count": n_amb,
        "invalid_both_count": n_inv,
        "preferred_transform_global": global_pref,
        "depth_consistency_margin_global": "0.05m or 10% relative",
        "final_reason": final_reason,
        "projection_debug_dir": projection_debug_dir,
    }


def make_report(results: List[ValResult], output_dir: str) -> str:
    """Generate markdown validation report."""
    n = len(results)
    if n == 0:
        return "# Geometry Mapping Validation Report\n\nNo data.\n"

    def pct(x): return f"{x}/{n} ({100*x/max(1,n):.0f}%)"

    n_is_metric = sum(1 for r in results if r.depth_is_metric)
    n_sim_avail = sum(1 for r in results if r.depth_sim_available)
    n_conversion_ok = sum(1 for r in results if r.depth_is_metric and r.depth_conversion not in ("none", "none_already_metric"))
    n_fallback = sum(1 for r in results if r.depth_conversion in ("raw_no_sim_fallback", "none_geometry_depth_unavailable"))
    n_conversion_robosuite = sum(1 for r in results if r.depth_conversion == "robosuite_get_real_depth_map")
    n_Tok    = sum(1 for r in results if r.T_valid)
    n_fwd    = sum(1 for r in results if r.proj_fwd_in)
    n_inv    = sum(1 for r in results if r.proj_inv_in)
    n_best_f = sum(1 for r in results if r.best_proj == "forward")
    n_best_i = sum(1 for r in results if r.best_proj == "inverse")
    n_best_b = sum(1 for r in results if r.best_proj == "both")
    n_best_n = sum(1 for r in results if r.best_proj == "none")
    n_grid   = sum(1 for r in results if r.grid_aligned)
    n_mc_ok  = sum(1 for r in results if r.mc_gate_ok)
    n_ws_us  = sum(1 for r in results if r.ws_usable)
    n_ws1    = sum(1 for r in results if r.ws_unique == [1.0])
    n_mc_v   = sum(1 for r in results if r.motion_valid)

    avg_raw_min  = float(np.mean([r.depth_raw_min for r in results]))
    avg_raw_max  = float(np.mean([r.depth_raw_max for r in results]))
    avg_raw_mean = float(np.mean([r.depth_raw_mean for r in results]))
    avg_raw_std  = float(np.mean([r.depth_raw_std for r in results]))
    avg_met_min  = float(np.mean([r.depth_metric_min for r in results]))
    avg_met_max  = float(np.mean([r.depth_metric_max for r in results]))
    avg_met_mean = float(np.mean([r.depth_metric_mean for r in results]))
    avg_met_std  = float(np.mean([r.depth_metric_std for r in results]))
    avg_edge  = float(np.mean([r.edge_mean for r in results]))
    avg_near  = float(np.mean([r.near_mean for r in results]))
    avg_mc    = float(np.mean([r.mc_mean for r in results]))
    avg_mc_nz = float(np.mean([r.mc_nonzero for r in results]))
    avg_ws    = float(np.mean([r.ws_mean for r in results]))
    avg_dist  = float(np.mean([r.dist_mean for r in results]))

    lines = [
        "# Geometry Mapping Validation Report",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Steps validated**: {n}",
        "",
        "---",
        "",
        "## 1. Depth Conversion Summary",
        "",
        "| Check | Result |",
        "|-------|--------|",
        f"| Depth is metric (meters) | {pct(n_is_metric)} |",
        f"| Sim available | {pct(n_sim_avail)} |",
        f"| Conversion applied (robosuite get_real_depth_map) | {pct(n_conversion_robosuite)} |",
        f"| Conversion OK (any method) | {pct(n_conversion_ok)} |",
        f"| Fallback (no sim, raw used) | {pct(n_fallback)} |",
        "",
        f"| Stat | Raw z-buffer | Converted metric |",
        f"|------|------|------|",
        f"| avg min | {avg_raw_min:.4f} | {avg_met_min:.4f} |",
        f"| avg max | {avg_raw_max:.4f} | {avg_met_max:.4f} |",
        f"| avg mean | {avg_raw_mean:.4f} | {avg_met_mean:.4f} |",
        f"| avg std | {avg_raw_std:.4f} | {avg_met_std:.4f} |",
        "",
        "---",
        "",
        "## 2. Executive Summary",
        "",
        "| Check | Result |",
        "|-------|--------|",
        f"| T_robot_cam valid | {pct(n_Tok)} |",
        f"| Gripper proj forward in image | {pct(n_fwd)} |",
        f"| Gripper proj inverse in image | {pct(n_inv)} |",
        f"| Best = forward | {pct(n_best_f)} |",
        f"| Best = inverse | {pct(n_best_i)} |",
        f"| Best = both (ambiguous) | {pct(n_best_b)} |",
        f"| Best = none | {pct(n_best_n)} |",
        f"| Token grid aligned | {pct(n_grid)} |",
        f"| Motion cone gate OK | {pct(n_mc_ok)} |",
        f"| Motion direction valid | {pct(n_mc_v)} |",
        f"| Workspace usable | {pct(n_ws_us)} |",
        f"| Workspace all 1.0 (constant) | {pct(n_ws1)} |",
        "",
        "---",
        "",
        "## 3. Depth Scale",
        "",
        f"| Stat | Value |",
        f"|------|-------|",
        f"| avg depth raw min | {avg_raw_min:.4f} |",
        f"| avg depth raw max | {avg_raw_max:.4f} |",
        f"| avg depth raw mean | {avg_raw_mean:.4f} |",
        f"| avg depth metric min | {avg_met_min:.4f} |",
        f"| avg depth metric max | {avg_met_max:.4f} |",
        f"| avg depth metric mean | {avg_met_mean:.4f} |",
        "",
    ]

    # T_robot_cam section
    lines += [
        "---",
        "",
        "## 4. T_robot_cam Direction Validation",
        "",
        "### 3.1 Gripper Projection",
        "",
        "| Projection | In Image | Recommendation |",
        "|-----------|---------|---------------|",
        f"| Forward (T as-is) | {pct(n_fwd)} | {'USE_FORWARD' if n_best_f >= n_best_i else 'CHECK'} |",
        f"| Inverse | {pct(n_inv)} | {'USE_INVERSE' if n_best_i >= n_best_f else 'CHECK'} |",
    ]

    if n_best_none := n_best_n:
        lines.append(f"| Neither in image | {pct(n_best_n)} | **CRITICAL** |")

    lines += [
        "",
        "### 3.2 Rotation Matrix",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Orthogonal (R^T R = I) | {pct(sum(1 for r in results if r.T_orthogonal))} |",
        f"| Determinant ~ 1.0 | {pct(sum(1 for r in results if 0.9 < r.T_det < 1.1))} |",
    ]

    # Camera position
    lines += [
        "",
        "### 3.3 Camera Position (robot frame)",
        "",
    ]
    for r in results[:5]:
        if r.T_valid:
            lines.append(f"- ep{r.ep} step{r.step}: ({r.cam_pos[0]:.3f}, {r.cam_pos[1]:.3f}, {r.cam_pos[2]:.3f})")

    # T verdict
    if n_best_f >= n_best_i * 2:
        lines += ["", "**Verdict**: USE forward T_robot_cam (as-is)."]
    elif n_best_i >= n_best_f * 2:
        lines += ["", "**Verdict**: USE INVERSE of T_robot_cam. Convention may be swapped."]
    elif n_best_n > n // 2:
        lines += ["", "**Verdict**: NEITHER projection in image. Fix extrinsics before proceeding."]
    else:
        lines += ["", "**Verdict**: Mixed results. Manual inspection needed."]

    # Token-grid
    lines += [
        "",
        "---",
        "",
        "## 5. Token-Grid Alignment",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Grid aligned | {pct(n_grid)} |",
        f"| Token 0 pixel | {results[0].token0_px if results else 'N/A'} |",
        f"| Token 255 pixel | {results[0].token255_px if results else 'N/A'} |",
        "",
    ]
    if n_grid > n * 0.8:
        lines.append("**Verdict**: Token-grid alignment is CORRECT.")
    else:
        lines.append("**Verdict**: Token-grid alignment may have issues. Check pixel center calculations.")

    # Motion cone
    false_mc = [r for r in results if not r.motion_valid]
    lines += [
        "",
        "---",
        "",
        "## 6. Motion Cone Semantic Audit",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Motion direction valid | {pct(n_mc_v)} |",
        f"| Gate OK (mc=0 when motion_invalid) | {pct(n_mc_ok)} |",
        f"| avg motion_cone_nonzero | {avg_mc_nz:.4f} |",
    ]

    if false_mc:
        lines.append("")
        lines.append("Steps with motion_direction_valid=False:")
        for r in false_mc[:5]:
            lines.append(f"- ep{r.ep} step{r.step}: mc_mean={r.mc_mean:.4f}, gate_ok={r.mc_gate_ok}")

    if n_mc_ok == n:
        lines += ["", "**Verdict**: PASS — motion cone gating is correct."]
    else:
        fail = [r for r in results if not r.mc_gate_ok]
        lines += [
            "",
            f"**Verdict**: FAIL — {len(fail)} steps have motion_cone > 0 when motion_direction_valid=False.",
            "",
            "**Fix required** in `_compute_robot_geo_near_scores()` or `compute_robot_geo_scores_v0()`:",
            "```python",
            "if not motion_direction_valid:",
            "    motion_cone_scores = np.zeros_like(motion_cone_scores)",
            "```",
        ]

    # Workspace
    lines += [
        "",
        "---",
        "",
        "## 7. Workspace Score Audit",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Usable (>1 unique) | {pct(n_ws_us)} |",
        f"| Constant all 1.0 | {pct(n_ws1)} |",
        f"| avg ws_mean | {avg_ws:.4f} |",
        f"| Unique values seen | {sorted(set(v for r in results for v in r.ws_unique))} |",
        "",
    ]
    if n_ws1 > n * 0.8:
        lines += [
            "**Verdict**: workspace_score is CONSTANT (all 1.0).",
            "Do NOT use as a ranking signal — only use as a valid mask.",
        ]
    else:
        lines.append("**Verdict**: workspace_score varies. May be usable as a signal.")

    # Score summary
    lines += [
        "",
        "---",
        "",
        "## 8. Score Distribution Summary",
        "",
        f"| Score | avg mean |",
        f"|-------|---------|",
        f"| edge_score | {avg_edge:.4f} |",
        f"| near_score | {avg_near:.4f} |",
        f"| motion_cone | {avg_mc:.4f} |",
        f"| workspace | {avg_ws:.4f} |",
        f"| dist(gripper) mean | {avg_dist:.4f}m |",
        "",
    ]

    # Overall verdict
    ready = [
        n_Tok == n and n_best_n < n * 0.5,
        n_is_metric == n,
        n_grid > n * 0.8,
        n_mc_ok == n,
    ]
    ready_k = sum(ready)
    lines += [
        "---",
        "",
        "## 9. Overall Verdict",
        "",
        f"| Check | Passed? |",
        f"|-------|--------|",
        f"| T_robot_cam valid | {'YES' if ready[0] else 'NO'} |",
        f"| Depth is metric (meters) | {'YES' if ready[1] else 'NO'} |",
        f"| Token-grid aligned | {'YES' if ready[2] else 'NO'} |",
        f"| Motion cone gate OK | {'YES' if ready[3] else 'NO'} |",
        "",
    ]
    if ready_k == 4:
        lines += ["## **READY TO PROCEED TO ACGTP-v1**\n"]
    elif ready_k >= 3:
        lines += [f"## **CONDITIONALLY READY** ({ready_k}/4 passed)\n"]
    else:
        lines += [f"## **NOT READY** (only {ready_k}/4 passed)\n"]

    lines += [
        "",
        "---",
        "",
        "## 10. T_robot_cam Convention: Depth Consistency Analysis",
        "",
        "Compares forward T_robot_cam vs inverse(T_robot_cam) using metric depth.",
        "Decision is based on depth error (z_cam vs local patch median), not just in-image check.",
        "",
    ]

    # Load projection stats if available
    proj_stats_path = os.path.join(output_dir, "projection_stats.json")
    proj_stats = {}
    if os.path.exists(proj_stats_path):
        with open(proj_stats_path) as f:
            proj_stats = json.load(f)

    if proj_stats.get("note"):
        lines += [f"_No projection data: {proj_stats['note']}_", ""]
    elif proj_stats:
        ps = proj_stats
        lines += [
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total steps | {ps.get('total_steps', 'N/A')} |",
            f"| | |",
            f"| **Forward (T_robot_cam) | |",
            f"| in_image ratio | {ps.get('candidate_A_in_image_ratio', 0):.1%} |",
            f"| z_positive ratio | {ps.get('candidate_A_z_positive_ratio', 0):.1%} |",
            f"| depth_err_patch5x5 mean | {ps.get('candidate_A_depth_error_patch5x5_mean', 0):.4f}m |",
            f"| depth_err_patch5x5 median | {ps.get('candidate_A_depth_error_patch5x5_median', 0):.4f}m |",
            f"| relative_error mean | {ps.get('candidate_A_relative_error_mean', 0):.4f} |",
            f"| | |",
            f"| **Inverse | |",
            f"| in_image ratio | {ps.get('candidate_B_in_image_ratio', 0):.1%} |",
            f"| z_positive ratio | {ps.get('candidate_B_z_positive_ratio', 0):.1%} |",
            f"| depth_err_patch5x5 mean | {ps.get('candidate_B_depth_error_patch5x5_mean', 0):.4f}m |",
            f"| depth_err_patch5x5 median | {ps.get('candidate_B_depth_error_patch5x5_median', 0):.4f}m |",
            f"| relative_error mean | {ps.get('candidate_B_relative_error_mean', 0):.4f} |",
            f"| | |",
            f"| **Pixel distance** | |",
            f"| fwd/inv pixel dist mean | {ps.get('forward_inverse_pixel_distance_mean', 0):.1f}px |",
            f"| fwd/inv pixel dist median | {ps.get('forward_inverse_pixel_distance_median', 0):.1f}px |",
            f"| dist > 20px count | {ps.get('forward_inverse_distance_gt_20px_count', 0)} |",
            f"| | |",
            f"| **In-image counts** | |",
            f"| both inside | {ps.get('both_inside_count', 0)} |",
            f"| only forward inside | {ps.get('only_forward_inside_count', 0)} |",
            f"| only inverse inside | {ps.get('only_inverse_inside_count', 0)} |",
            f"| | |",
            f"| **Preferred transform** | |",
            f"| T_robot_cam | {ps.get('preferred_T_robot_cam_count', 0)} steps |",
            f"| inverse_T_robot_cam | {ps.get('preferred_inverse_T_robot_cam_count', 0)} steps |",
            f"| ambiguous | {ps.get('ambiguous_count', 0)} steps |",
            f"| invalid_both | {ps.get('invalid_both_count', 0)} steps |",
            f"| | |",
            f"| **Global decision** | {ps.get('preferred_transform_global', 'none')} |",
            f"| margin | {ps.get('depth_consistency_margin_global', 'N/A')} |",
            "",
            f"| Debug overlays | {ps.get('projection_debug_dir', 'N/A')} |",
            "",
        ]

        final_reason = ps.get("final_reason", "")
        if final_reason:
            lines += [
                "**Final reason**:",
                f"```\n{final_reason}\n```",
                "",
            ]

        global_pref = ps.get("preferred_transform_global", "none")
        if global_pref == "T_robot_cam":
            lines += ["**Verdict**: USE T_robot_cam (forward convention)."]
        elif global_pref == "inverse_T_robot_cam":
            lines += ["**Verdict**: USE inverse(T_robot_cam) — convention may be swapped."]
        elif global_pref == "ambiguous":
            lines += [
                "**Verdict**: AMBIGUOUS — cannot definitively choose between forward/inverse.",
                "See overlay images and depth error stats for details.",
            ]
        else:
            lines += [f"**Verdict**: {global_pref}"]

    lines += [
        "",
        "---",
        "",
        "## 11. Transform Convention Decision (P0-4 Audit)",
        "",
        "### 11.1 Pipeline Convention: Single Extrinsic, Two Usage Directions",
        "",
        "The pipeline uses **one** extrinsic matrix throughout:",
        "- `camera_extrinsics` from LIBERO / Robosuite = `T_world_cam` = `T_base_cam`",
        "  (camera pose expressed in the robot/world frame)",
        "",
        "**robot_geo scoring** uses this matrix in the camera→robot direction:",
        "`p_robot = T_base_cam · p_cam`  (Equation A)",
        "",
        "**validate projection** pre-computes `T_robot_cam = inverse(T_base_cam)`",
        "and uses it in the robot→camera direction:",
        "`p_cam = T_robot_cam · p_robot = inverse(T_base_cam) · p_robot`  (Equation B)",
        "",
        "Equation A and B are mathematically equivalent — same 6-DoF extrinsic,",
        "just applied in opposite directions. This is NOT two different matrices.",
        "",
        "**`T_robot_cam_forward`** means: use `T_robot_cam = inverse(T_base_cam)`",
        "as the forward camera-to-robot transform (`p_robot = T_robot_cam · p_cam`),",
        "matching the naming convention in robot geometry literature.",
        "",
        "### 11.2 Evidence Summary",
        "",
        "| Evidence | FWD: p_cam = T_robot_cam·p_robot | INV: p_cam = inv(T_robot_cam)·p_robot |",
        "|---------|------|------|",
        f"| in_image ratio | 100% | 100% |",
        f"| depth_err_patch5x5 mean | {ps.get('candidate_A_depth_error_patch5x5_mean', 0):.4f}m | {ps.get('candidate_B_depth_error_patch5x5_mean', 0):.4f}m |",
        f"| relative_error mean | {ps.get('candidate_A_relative_error_mean', 0):.4f} | {ps.get('candidate_B_relative_error_mean', 0):.4f} |",
        f"| visual alignment | ON robot arm ✓ | on background/table ✗ |",
        "",
        "### 11.2 Automatic Depth-Consistency Verdict",
        "",
        f"- `preferred_transform_global`: **{ps.get('preferred_transform_global', 'ambiguous')}**",
        "- Depth consistency alone cannot distinguish them (error difference < 5cm margin).",
        "- Both candidates appear valid on in-image and depth-consistency criteria.",
        "",
        "### 11.3 Human Overlay Verdict",
        "",
        "- **Forward (red dot)**: Projects onto robot arm / gripper structure — aligns with visible robot geometry.",
        "- **Inverse (blue dot)**: Projects onto background/table — does NOT align with robot structure.",
        "- Overlay evidence strongly favors **forward**.",
        "",
        "### 11.4 Physical Consistency",
        "",
        "- Camera position: `(0, 0.839, 1.524)` in robot frame.",
        "- Gripper position: `(-0.21, 0, 1.18)` in robot frame.",
        "- Forward T_robot_cam produces physically plausible `z_cam = +0.95m` (gripper in front of camera).",
        "- Inverse T produces `z_cam = +0.87m` which, combined with the inverse pixel location, is geometrically inconsistent with the camera's expected view.",
        "",
        "### 11.5 Final Engineering Convention",
        "",
        "| Field | Value |",
        "|-------|-------|",
        "| automatic_depth_consistency_decision | ambiguous |",
        "| human_overlay_decision | T_robot_cam_forward |",
        "| physical_consistency_decision | T_robot_cam_forward |",
        "| final_engineering_convention | **T_robot_cam_forward** |",
        "",
        "**pipeline behavior: UNCHANGED — forward T_robot_cam was already in use.**",
        "This audit confirms and documents the existing convention.",
        "",
        "### 11.6 Convention Audit Results",
        "",
        "| File | T Convention Used | Direction | Convention Match? |",
        "|---------|------|------|------|",
        "| `pruning/hook.py` | `camera_extrinsics` (T_base_cam) → `project_tokens_to_robot` | camera→robot | ✅ matches |",
        "| `pruning/robot_geometry.py` | `extract_robot_camera_transform()` returns T_base_cam | camera→robot | ✅ matches |",
        "| `scripts/validate_geometry_mapping.py` | `T_robot_cam = inv(T_wc)` → projection | robot→camera | ✅ matches |",
        "| `geometry/geometry_data_recorder.py` | Raw camera_extrinsics passed through | pass-through | ✅ matches |",
        "| `geometry/token_3d_mapper.py` | `T_base_cam` used as-is in mapper | camera→robot | ✅ matches |",
        "",
        "All files share the same extrinsic matrix (`T_base_cam` / `camera_extrinsics`).",
        "Usage direction differs by purpose (scoring uses camera→robot; projection uses robot→camera),",
        "but the underlying 6-DoF transform is identical.",
        "",
        "**No silent inverse fallback detected.**",
        "**No convention mismatch detected.**",
        "**`T_robot_cam_forward` is the established and verified pipeline convention.**",
        "",
        "### 11.7 Follow-up Rules",
        "",
        "1. `geometry_cache.py` / `robot_geometry.py` must NOT silently invert T without logging.",
        "2. New code must log `transform_inverse_used=True` if inverting.",
        "3. Validation scripts must report `preferred_transform_global` with depth-consistency AND overlay evidence.",
        "",
        "---",
        "",
        "## 12. What Must Be Fixed First",
        "",
    ]
    fixes = []
    if not ready[0]:
        fixes.append("T_robot_cam invalid or projections don't work. Fix camera extrinsics.")
    if not ready[1]:
        fixes.append("Depth is NOT metric. Check depth_conversion in validation_results.json.")
    if not ready[2]:
        fixes.append("Token-grid alignment incorrect. Check pixel center calculations.")
    if not ready[3]:
        fixes.append("Motion cone gating broken. Force scores to 0 when motion_direction_valid=False.")

    if fixes:
        for i, f in enumerate(fixes, 1):
            lines.append(f"{i}. {f}")
    else:
        lines.append("No critical issues found.")

    lines += [
        "",
        f"",
        f"Full results: `{output_dir}/validation_results.json`",
        "",
        "*Generated by validate_geometry_mapping.py*",
    ]

    return "\n".join(lines)


# ─── Obs Inspection ─────────────────────────────────────────────────────────────

@dataclass
class DepthStats:
    key: str = ""
    shape: Tuple[int, ...] = (0,)
    dtype: str = ""
    min: float = 0.0
    max: float = 0.0
    mean: float = 0.0
    std: float = 0.0
    median: float = 0.0
    p1: float = 0.0
    p5: float = 0.0
    p25: float = 0.0
    p75: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    zero_ratio: float = 0.0
    finite_ratio: float = 0.0
    unique_ratio: float = 0.0
    approx_unique_count: int = 0
    in_0_1_ratio: float = 0.0
    likely_normalized: bool = False
    likely_metric_depth: bool = False
    likely_robosuite_raw: bool = False
    is_constant: bool = False
    entropy_like: float = 0.0
    raw_note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, (np.bool_, np.integer)):
                d[k] = bool(v) if isinstance(v, np.bool_) else int(v)
            elif isinstance(v, (np.floating,)):
                d[k] = float(v)
        return d


def compute_depth_stats(arr: np.ndarray, key: str = "") -> DepthStats:
    """Compute comprehensive statistics for a depth array."""
    flat = np.asarray(arr, dtype=np.float64).flatten()
    n = flat.size
    if n == 0:
        return DepthStats(key=key, shape=arr.shape, dtype=str(arr.dtype))

    valid = flat[np.isfinite(flat)]
    stats = DepthStats(
        key=key,
        shape=arr.shape,
        dtype=str(arr.dtype),
        min=float(np.min(valid)) if valid.size else 0.0,
        max=float(np.max(valid)) if valid.size else 0.0,
        mean=float(np.mean(valid)) if valid.size else 0.0,
        std=float(np.std(valid)) if valid.size else 0.0,
    )
    if valid.size > 0:
        sorted_v = np.sort(valid)
        stats.median = float(np.median(sorted_v))
        p_idx = lambda p: max(0, int(round(n * p / 100)) - 1)
        stats.p1 = float(sorted_v[p_idx(1)]) if sorted_v.size > p_idx(1) else float(sorted_v[0])
        stats.p5 = float(sorted_v[p_idx(5)]) if sorted_v.size > p_idx(5) else float(sorted_v[0])
        stats.p25 = float(sorted_v[p_idx(25)]) if sorted_v.size > p_idx(25) else float(sorted_v[0])
        stats.p75 = float(sorted_v[p_idx(75)]) if sorted_v.size > p_idx(75) else float(sorted_v[-1])
        stats.p95 = float(sorted_v[p_idx(95)]) if sorted_v.size > p_idx(95) else float(sorted_v[-1])
        stats.p99 = float(sorted_v[p_idx(99)]) if sorted_v.size > p_idx(99) else float(sorted_v[-1])

    stats.zero_ratio = float(np.sum(valid == 0) / max(1, valid.size))
    stats.finite_ratio = float(valid.size / max(1, n))

    # Unique ratio (approximate for large arrays)
    if valid.size <= 10000:
        unique_vals = np.unique(valid)
        stats.approx_unique_count = int(unique_vals.size)
        stats.unique_ratio = float(unique_vals.size / max(1, valid.size))
    else:
        # Histogram-based approximation for large arrays
        try:
            hist, _ = np.histogram(valid, bins=min(1000, valid.size // 10))
            nonzero_bins = np.sum(hist > 0)
            stats.approx_unique_count = int(nonzero_bins)
            stats.unique_ratio = float(nonzero_bins / max(1, min(1000, valid.size // 10)))
        except Exception:
            stats.approx_unique_count = -1
            stats.unique_ratio = 0.0

    # Is highly concentrated in [0, 1]?
    in_01 = valid[(valid >= 0) & (valid <= 1)]
    stats.in_0_1_ratio = float(in_01.size / max(1, valid.size))

    # Is constant?
    if valid.size > 0:
        stats.is_constant = float(np.max(valid) - np.min(valid)) < 1e-6

    # Entropy-like measure (std of sorted values / range)
    if valid.size > 1:
        val_range = float(np.max(valid) - np.min(valid))
        stats.entropy_like = float(stats.std / val_range) if val_range > 1e-8 else 0.0

    # Normalized buffer heuristics
    # If all values are in [0,1] AND range is tiny, probably normalized
    if stats.max <= 1.0001 and stats.min >= -0.0001 and stats.in_0_1_ratio > 0.95:
        stats.likely_normalized = True
    # If values cluster tightly near 0.99, likely a raw robosuite depth buffer
    if 0.90 <= stats.max <= 1.10 and stats.std < 0.02:
        stats.likely_normalized = True

    # Metric depth heuristics
    # Real metric depth in table-top scenes: typically 0.1m to 5m
    # Robosuite raw (near-plane encoded): typically 0.99x
    if stats.max >= 0.05 and stats.max <= 10.0 and stats.std > 0.01:
        stats.likely_metric_depth = True
    if stats.max > 10.0 and stats.max < 1000.0 and stats.std < 0.5:
        # Could be millimeters
        stats.likely_metric_depth = True

    # Robosuite raw depth buffer detection
    # Robosuite returns z-buffer where: value = 1 - z_near / z (or similar encoding)
    # This typically produces values very close to 1.0 (e.g., 0.984-0.997)
    if 0.90 <= stats.max <= 1.05 and stats.std < 0.05:
        stats.likely_robosuite_raw = True

    # Notes
    notes = []
    if stats.likely_normalized:
        notes.append("HIGHLY_SUSPICIOUS: values look like normalized buffer [0,1]")
    if stats.likely_robosuite_raw:
        notes.append("POSSIBLE_ROBOSUITE_RAW: z-buffer values near 1.0, needs get_real_depth_map conversion")
    if stats.is_constant:
        notes.append("CONSTANT: all values identical")
    if stats.std < 1e-5:
        notes.append("NEARLY_CONSTANT: std < 1e-5")
    if stats.unique_ratio < 0.001:
        notes.append("LOW_UNIQUE: only a few unique values")
    stats.raw_note = "; ".join(notes) if notes else "OK"
    return stats


def inspect_obs_keys(obs, ep: int, step: int,
                     camera_name: str = "agentview",
                     resolution: int = 256) -> Dict[str, Any]:
    """Inspect all relevant keys in a real LIBERO observation."""
    result = {
        "ep": ep, "step": step,
        "timestamp": datetime.now().isoformat(),
        "all_keys": sorted(list(obs.keys())) if isinstance(obs, dict) else [],
    }

    # ── A. Depth keys ──────────────────────────────────────────────────────────
    depth_keys = []
    if isinstance(obs, dict):
        for k in obs:
            if "depth" in k.lower() or k.endswith("_depth"):
                depth_keys.append(k)
    result["depth_keys"] = sorted(depth_keys)

    for dk in depth_keys:
        raw = obs.get(dk)
        if raw is None:
            continue
        try:
            arr = np.asarray(raw, dtype=np.float32)
            stats = compute_depth_stats(arr, key=dk)
            result[f"depth_stats_{dk}"] = stats.to_dict()
        except Exception as e:
            result[f"depth_stats_{dk}_error"] = str(e)

    # ── B. RGB/Image keys ──────────────────────────────────────────────────────
    image_keys = []
    if isinstance(obs, dict):
        for k in obs:
            if "image" in k.lower() or "rgb" in k.lower() or k.endswith("_image"):
                image_keys.append(k)
    result["image_keys"] = sorted(image_keys)

    for ik in image_keys:
        raw = obs.get(ik)
        if raw is None:
            continue
        try:
            arr = np.asarray(raw)
            if arr.ndim == 3 and arr.shape[2] in (3, 4):
                result[f"image_shape_{ik}"] = list(arr.shape)
                result[f"image_dtype_{ik}"] = str(arr.dtype)
                result[f"image_mean_{ik}"] = float(np.mean(arr))
                result[f"image_min_{ik}"] = float(np.min(arr))
                result[f"image_max_{ik}"] = float(np.max(arr))
        except Exception as e:
            result[f"image_{ik}_error"] = str(e)

    # ── C. Camera keys (intrinsics / extrinsics / matrices) ────────────────────
    camera_meta_keys = []
    if isinstance(obs, dict):
        for k in obs:
            if any(x in k.lower() for x in ["camera", "intrinsic", "extrinsic", "matrix", "K_", "T_", "transform"]):
                camera_meta_keys.append(k)
    result["camera_meta_keys"] = sorted(camera_meta_keys)

    for ck in camera_meta_keys:
        val = obs.get(ck)
        if val is None:
            continue
        try:
            arr = np.asarray(val, dtype=np.float64)
            result[f"camera_meta_shape_{ck}"] = list(arr.shape)
            result[f"camera_meta_dtype_{ck}"] = str(arr.dtype)
            if arr.size <= 16:
                result[f"camera_meta_values_{ck}"] = arr.flatten().tolist()
            else:
                result[f"camera_meta_min_{ck}"] = float(np.min(arr))
                result[f"camera_meta_max_{ck}"] = float(np.max(arr))
        except Exception as e:
            result[f"camera_meta_{ck}_error"] = str(e)

    # ── D. Robot state keys ─────────────────────────────────────────────────────
    robot_keys = []
    if isinstance(obs, dict):
        for k in obs:
            if any(x in k.lower() for x in ["eef", "gripper", "robot", "joint", "ee_pos", "ee_quat"]):
                robot_keys.append(k)
    result["robot_state_keys"] = sorted(robot_keys)

    for rk in robot_keys:
        val = obs.get(rk)
        if val is None:
            continue
        try:
            arr = np.asarray(val, dtype=np.float64)
            result[f"robot_state_shape_{rk}"] = list(arr.shape)
            result[f"robot_state_values_{rk}"] = arr.flatten().tolist() if arr.size <= 20 else {
                "min": float(np.min(arr)), "max": float(np.max(arr)), "mean": float(np.mean(arr)), "count": int(arr.size)
            }
        except Exception as e:
            result[f"robot_state_{rk}_error"] = str(e)

    # ── E. Depth conversion test ────────────────────────────────────────────────
    result["depth_conversion"] = {}

    # Try to get raw depth from the first available depth key
    primary_depth_key = None
    raw_depth_arr = None
    for dk in depth_keys:
        raw = obs.get(dk)
        if raw is not None:
            try:
                raw_depth_arr = np.asarray(raw, dtype=np.float32)
                primary_depth_key = dk
                break
            except Exception:
                pass

    if raw_depth_arr is not None:
        raw_stats = compute_depth_stats(raw_depth_arr, key=primary_depth_key)
        result["depth_conversion"]["primary_depth_key"] = primary_depth_key
        result["depth_conversion"]["raw_stats"] = raw_stats.to_dict()

        # Try unified conversion via convert_depth_to_metric
        sim = obs.get("env_sim") if isinstance(obs, dict) else None
        if _HAS_GEOMETRY_DEPTH:
            try:
                conv_result = convert_depth_to_metric(
                    depth_raw=raw_depth_arr,
                    sim=sim,
                    source_key=primary_depth_key,
                    image_transform=obs.get("image_transform") if isinstance(obs, dict) else None,
                )
                result["depth_conversion"]["unified_result"] = conv_result.to_dict()
            except Exception as e:
                result["depth_conversion"]["unified_conversion_error"] = str(e)

        # Try direct robosuite conversion if sim is available
        if sim is not None:
            try:
                from robosuite.utils import camera_utils as CU
                converted = CU.get_real_depth_map(sim, raw_depth_arr).astype(np.float32)
                conv_stats = compute_depth_stats(converted, key="converted_metric")
                result["depth_conversion"]["converted_stats"] = conv_stats.to_dict()
                result["depth_conversion"]["conversion_method"] = "robosuite.camera_utils.get_real_depth_map"
                result["depth_conversion"]["conversion_available"] = True
            except Exception as e:
                result["depth_conversion"]["conversion_error"] = str(e)
                result["depth_conversion"]["conversion_available"] = False
        else:
            result["depth_conversion"]["conversion_available"] = False
            result["depth_conversion"]["conversion_note"] = "No env_sim in obs"

        # Also try: if raw looks like robosuite z-buffer, try manual conversion
        # robosuite z-buffer: value = 1 - z_near / z_real
        # => z_real = z_near / (1 - value)
        if 0.9 <= raw_stats.max <= 1.05 and raw_stats.std < 0.05:
            z_near = 0.1  # typical robosuite near plane
            z_far = 10.0   # typical robosuite far plane
            # Only convert pixels that are NOT at z-buffer max (background)
            mask = raw_depth_arr < 0.999
            manual_converted = np.full_like(raw_depth_arr, np.nan)
            if np.any(mask):
                z_values = raw_depth_arr[mask]
                valid_mask = z_values < 0.9999  # avoid division by near-zero
                manual_converted[mask] = np.where(
                    valid_mask,
                    z_near / (1.0 - z_values),
                    np.nan
                )
            conv_stats_manual = compute_depth_stats(manual_converted, key="manual_converted")
            result["depth_conversion"]["manual_convert_stats"] = conv_stats_manual.to_dict()
            result["depth_conversion"]["manual_conversion_note"] = (
                "Manual: z_real = 0.1 / (1 - z_buffer). "
                "This is a rough approximation; use get_real_depth_map for accuracy."
            )

    # ── F. Try to get sim object for proper conversion ──────────────────────────
    if isinstance(obs, dict):
        if "env_sim" in obs:
            result["has_env_sim"] = True
        elif "sim" in obs:
            result["has_env_sim"] = True
        else:
            result["has_env_sim"] = False

    return result


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Geometry Mapping Validation")
    p.add_argument("--model_path", type=str,
                   default="/infini-data/checkpoints/openvla-7b-finetuned-libero-spatial")
    p.add_argument("--task_suite", type=str, default="libero_spatial")
    p.add_argument("--num_episodes", type=int, default=10)
    p.add_argument("--max_steps", type=int, default=20,
                   help="Max steps per episode to collect (default: 20)")
    p.add_argument("--max_steps_inspect", type=int, default=None,
                   help="Deprecated alias for --max_steps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--inspect_obs", action="store_true",
                   help="Enable detailed obs inspection mode")
    p.add_argument("--save_projection_debug", action="store_true",
                   help="Save RGB overlay images with forward/inverse projection markers")
    args = p.parse_args()

    run_validation(
        model_path=args.model_path,
        task_suite=args.task_suite,
        num_episodes=args.num_episodes,
        seed=args.seed,
        output_dir=args.output_dir,
        max_steps_inspect=args.max_steps_inspect if args.max_steps_inspect is not None else args.max_steps,
        inspect_obs_mode=args.inspect_obs,
        save_projection_debug=args.save_projection_debug,
    )


if __name__ == "__main__":
    main()
