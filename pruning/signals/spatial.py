"""Spatial score signals: depth edge, contact ring, and scene layout."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Source: pruning/scores/depth_edge.py
# ---------------------------------------------------------------------------
from typing import Tuple

import numpy as np


def compute_valid_depth_mask(
    token_depth: np.ndarray,
    min_depth: float = 1e-6,
    max_depth: float = 10.0,
) -> np.ndarray:
    return np.isfinite(token_depth) & (token_depth > min_depth) & (token_depth < max_depth)


def compute_token_depth_edge(
    token_depth: np.ndarray,
    valid_mask: np.ndarray | None = None,
    token_grid_shape: Tuple[int, int] = (16, 16),
) -> np.ndarray:
    """Batch wrapper returning edge scores with shape [B, N]."""
    depth_np = np.asarray(token_depth, dtype=np.float32)
    was_1d = depth_np.ndim == 1
    if was_1d:
        depth_np = depth_np[None, :]
    if valid_mask is None:
        valid_np = compute_valid_depth_mask(depth_np)
    else:
        valid_np = np.asarray(valid_mask, dtype=np.bool_)
        if valid_np.ndim == 1:
            valid_np = valid_np[None, :]
    rows = [
        compute_depth_edge_scores(
            depth_np[b],
            valid_np[b],
            token_grid_shape=token_grid_shape,
            num_visual_tokens=depth_np.shape[1],
        )
        for b in range(depth_np.shape[0])
    ]
    out = np.stack(rows, axis=0).astype(np.float32)
    return out[0] if was_1d else out


def compute_depth_edge_scores(
    token_depth: np.ndarray,
    valid_mask: np.ndarray,
    token_grid_shape: Tuple[int, int] = (16, 16),
    num_visual_tokens: int = 256,
) -> np.ndarray:
    """Compute normalized finite-difference edge score on the token grid."""
    grid_h, grid_w = token_grid_shape
    tokens_per_grid = grid_h * grid_w
    base_depth = token_depth[:tokens_per_grid].reshape(grid_h, grid_w).astype(np.float32)
    base_valid = valid_mask[:tokens_per_grid].reshape(grid_h, grid_w)
    fill = float(np.median(base_depth[base_valid])) if np.any(base_valid) else 0.0
    d = np.where(base_valid, base_depth, fill).astype(np.float32)

    gx = np.zeros_like(d, dtype=np.float32)
    gy = np.zeros_like(d, dtype=np.float32)
    if grid_w > 1:
        gx[:, 1:-1] = 0.5 * (d[:, 2:] - d[:, :-2])
        gx[:, 0] = d[:, 1] - d[:, 0]
        gx[:, -1] = d[:, -1] - d[:, -2]
    if grid_h > 1:
        gy[1:-1, :] = 0.5 * (d[2:, :] - d[:-2, :])
        gy[0, :] = d[1, :] - d[0, :]
        gy[-1, :] = d[-1, :] - d[-2, :]

    edge = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    edge[~base_valid] = 0.0
    max_edge = float(np.max(edge)) if edge.size else 0.0
    if max_edge > 1e-8:
        edge = edge / max_edge
    else:
        edge.fill(0.0)

    flat = edge.reshape(-1)
    if num_visual_tokens != tokens_per_grid and num_visual_tokens % tokens_per_grid == 0:
        return np.tile(flat, num_visual_tokens // tokens_per_grid).astype(np.float32)
    return flat[np.arange(num_visual_tokens) % tokens_per_grid].astype(np.float32)

# ---------------------------------------------------------------------------
# Source: pruning/scores/contact_ring.py
# ---------------------------------------------------------------------------
from typing import Any, Dict, Optional

import numpy as np


def compute_contact_ring_scores(
    token_u: Optional[np.ndarray],
    token_v: Optional[np.ndarray],
    gripper_pixel: Optional[np.ndarray],
    near_scores: Optional[np.ndarray] = None,
    *,
    self_core_radius_px: float = 16.0,
    contact_ring_inner_px: float = 24.0,
    contact_ring_outer_px: float = 48.0,
    contact_requires_edge_or_object: bool = True,
    depth_edge_scores: Optional[np.ndarray] = None,
    depth_edge_gate_percentile: float = 60.0,
) -> Dict[str, Any]:
    """Compute self-filtered contact ring scores.

    Tokens are classified into three zones:
      - robot_self_core:   within self_core_radius_px of gripper projection
      - contact_ring:      between inner and outer radius, gate-passing tokens
      - outside_ring:      all other tokens (score = 0)

    Args:
        token_u, token_v:   [N] pixel coordinates for each token.
        gripper_pixel:      [2] gripper projection in pixels (u, v).
        near_scores:        [N] optional near-to-gripper scores for scoring.
        self_core_radius_px: Core suppression radius in pixels.
        contact_ring_inner_px: Inner ring boundary.
        contact_ring_outer_px: Outer ring boundary.
        contact_requires_edge_or_object: If True, ring tokens must pass depth-edge gate.
        depth_edge_scores:  [N] depth edge scores for the gate.
        depth_edge_gate_percentile: Percentile threshold for the edge gate.

    Returns:
        Dict with keys:
          contact_ring_scores:  [N] final contact ring scores [0, 1].
          robot_self_core_mask: [N] bool mask of self-core tokens.
          contact_ring_mask:     [N] bool mask of in-ring tokens.
          contact_ring_token_count: int.
          contact_ring_gated_token_count: int (ring tokens that pass gate).
          gripper_pixel_u: float.
          gripper_pixel_v: float.
          gripper_in_bounds: bool.
          self_core_token_count: int.
    """
    result: Dict[str, Any] = {
        "contact_ring_scores": np.array([]),
        "robot_self_core_mask": np.array([]),
        "contact_ring_mask": np.array([]),
        "contact_ring_token_count": 0,
        "contact_ring_gated_token_count": 0,
        "gripper_pixel_u": None,
        "gripper_pixel_v": None,
        "gripper_in_bounds": False,
        "self_core_token_count": 0,
    }

    if token_u is None or token_v is None:
        return result

    u = np.asarray(token_u, dtype=np.float32).reshape(-1)
    v = np.asarray(token_v, dtype=np.float32).reshape(-1)
    n = u.size

    if n == 0:
        return result

    contact_ring_scores = np.zeros(n, dtype=np.float32)
    self_core_mask = np.zeros(n, dtype=np.float32)
    contact_ring_mask = np.zeros(n, dtype=np.float32)

    result["contact_ring_scores"] = contact_ring_scores
    result["robot_self_core_mask"] = self_core_mask
    result["contact_ring_mask"] = contact_ring_mask

    if gripper_pixel is None:
        return result

    gx = float(gripper_pixel[0])
    gy = float(gripper_pixel[1])

    u_max = float(np.max(u)) if n > 0 else 0.0
    v_max = float(np.max(v)) if n > 0 else 0.0
    in_bounds = (0 <= gx <= u_max) and (0 <= gy <= v_max)
    result["gripper_pixel_u"] = gx
    result["gripper_pixel_v"] = gy
    result["gripper_in_bounds"] = in_bounds

    dist = np.sqrt((u - gx) ** 2 + (v - gy) ** 2)

    # Zone 1: robot-self-core (suppressed)
    self_core_mask = (dist <= self_core_radius_px).astype(np.float32)
    result["self_core_token_count"] = int(np.sum(self_core_mask > 0.5))

    # Zone 2: contact ring (between inner and outer radius, NOT including self-core)
    in_ring = ((dist > contact_ring_inner_px) & (dist <= contact_ring_outer_px)).astype(np.float32)
    contact_ring_mask = in_ring.copy()
    result["contact_ring_mask"] = contact_ring_mask
    result["contact_ring_token_count"] = int(np.sum(in_ring > 0.5))

    # Zone 3: gate for contact ring
    # Tokens in the ring get their score only if they pass the depth-edge gate.
    # near_gripper tokens that are NOT on edges/objects get PENALIZED, not included.
    if contact_requires_edge_or_object and depth_edge_scores is not None:
        de = np.asarray(depth_edge_scores, dtype=np.float32).reshape(-1)
        if de.shape[0] == n and np.any(de > 0.0):
            valid_de = de[np.isfinite(de) & (de > 0.0)]
            if valid_de.size > 0:
                gate_threshold = float(np.percentile(valid_de, depth_edge_gate_percentile))
            else:
                gate_threshold = 0.0
            # Apply gate: ring tokens with edge score >= threshold pass
            gate_pass = (de >= gate_threshold) & (in_ring > 0.5)
            result["contact_ring_gated_token_count"] = int(np.sum(gate_pass))

            # Score: ring tokens that pass the gate get near_score * 1.0
            # ring tokens that don't pass get 0.0
            if near_scores is not None:
                ns = np.asarray(near_scores, dtype=np.float32).reshape(-1)
                if ns.shape[0] == n:
                    contact_ring_scores = np.where(gate_pass, np.clip(ns, 0.0, 1.0), 0.0)
                else:
                    contact_ring_scores = np.where(gate_pass, 1.0, 0.0)
            else:
                contact_ring_scores = np.where(gate_pass, 1.0, 0.0)
        else:
            # No valid depth edge scores: apply soft ring score (no gate)
            result["contact_ring_gated_token_count"] = 0
            if near_scores is not None:
                ns = np.asarray(near_scores, dtype=np.float32).reshape(-1)
                if ns.shape[0] == n:
                    contact_ring_scores = np.where(in_ring > 0.5, np.clip(ns, 0.0, 1.0) * 0.3, 0.0)
                else:
                    contact_ring_scores = np.where(in_ring > 0.5, 0.3, 0.0)
            else:
                contact_ring_scores = np.where(in_ring > 0.5, 0.3, 0.0)
    else:
        # Gate disabled: use near_scores in ring
        result["contact_ring_gated_token_count"] = result["contact_ring_token_count"]
        if near_scores is not None:
            ns = np.asarray(near_scores, dtype=np.float32).reshape(-1)
            if ns.shape[0] == n:
                contact_ring_scores = np.where(in_ring > 0.5, np.clip(ns, 0.0, 1.0), 0.0)
            else:
                contact_ring_scores = np.where(in_ring > 0.5, 1.0, 0.0)
        else:
            contact_ring_scores = np.where(in_ring > 0.5, 1.0, 0.0)

    # Zero out self-core tokens regardless of gate result
    contact_ring_scores[self_core_mask > 0.5] = 0.0

    # Also zero out tokens beyond outer ring
    contact_ring_scores[dist > contact_ring_outer_px] = 0.0
    contact_ring_scores[in_ring < 0.5] = 0.0

    result["contact_ring_scores"] = contact_ring_scores
    result["robot_self_core_mask"] = self_core_mask
    result["contact_ring_mask"] = contact_ring_mask

    return result

# ---------------------------------------------------------------------------
# Source: pruning/scores/scene_layout.py
# ---------------------------------------------------------------------------
import time as _time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _connected_components_4n(
    mask: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    min_size: int = 5,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Label 4-connected components in a boolean mask.

    Returns (labels, scores, num_components) where:
      labels[i] = component id (0..n-1) or -1 (no component)
      scores[i] = component-size score (normalized to [0,1])
    """
    n = len(mask)
    labels = np.full(n, -1, dtype=np.int32)
    scores = np.zeros(n, dtype=np.float32)
    if not np.any(mask):
        return labels, scores, 0

    coord_to_idx = {}
    mask_indices = np.where(mask)[0]
    for idx in mask_indices:
        coord_to_idx[(int(u[idx]), int(v[idx]))] = idx

    visited = set()
    comp_id = 0
    sizes = []
    comp_members: List[List[int]] = []

    for start_idx in mask_indices:
        coord = (int(u[start_idx]), int(v[start_idx]))
        if coord in visited:
            continue
        members = []
        queue = [coord]
        visited.add(coord)
        while queue:
            cx, cy = queue.pop()
            ci = coord_to_idx.get((cx, cy))
            if ci is not None and ci not in visited:
                visited.add((cx, cy))
                members.append(ci)
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nc = (cx + dx, cy + dy)
                    if nc in coord_to_idx and nc not in visited:
                        visited.add(nc)
                        queue.append(nc)

        if len(members) >= min_size:
            comp_members.append(members)
            sizes.append(len(members))
            comp_id += 1

    if not sizes:
        return labels, scores, 0

    max_size = max(sizes)
    for cid, members in enumerate(comp_members):
        for mi in members:
            labels[mi] = cid
            scores[mi] = float(len(members)) / float(max_size)

    return labels, scores, len(comp_members)


def _object_contour_mask(
    component_labels: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """Build a boolean mask of tokens on the boundary (contour) of object components."""
    n = len(component_labels)
    contour = np.zeros(n, dtype=bool)
    if not np.any(component_labels >= 0):
        return contour

    label_to_coords: Dict[int, set] = {}
    for idx in range(n):
        lid = component_labels[idx]
        if lid >= 0:
            if lid not in label_to_coords:
                label_to_coords[lid] = set()
            label_to_coords[lid].add((int(u[idx]), int(v[idx])))

    for idx in range(n):
        lid = component_labels[idx]
        if lid < 0:
            continue
        cx, cy = int(u[idx]), int(v[idx])
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (cx + dx, cy + dy)
            if neighbor not in label_to_coords.get(lid, set()):
                contour[idx] = True
                break

    return contour


def _estimate_support_plane_depth(
    td: np.ndarray,
    in_support: np.ndarray,
) -> Tuple[float, float]:
    """Estimate dominant support plane depth via histogram peak.

    Returns (plane_depth, plane_std) in meters.
    Falls back to median of in_support tokens if histogram fails.
    """
    support_depths = td[in_support]
    if support_depths.size == 0:
        return 0.65, 0.05

    try:
        counts, bin_edges = np.histogram(support_depths, bins=32)
        if np.sum(counts) == 0:
            return float(np.median(support_depths)), 0.05
        peak_bin = int(np.argmax(counts))
        plane_depth = (bin_edges[peak_bin] + bin_edges[peak_bin + 1]) * 0.5
        plane_std = float(np.std(support_depths))
        return plane_depth, max(plane_std, 0.02)
    except Exception:
        return float(np.median(support_depths)), 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_scene_layout_scores(
    token_depth: np.ndarray,
    valid_mask: np.ndarray,
    token_u: Optional[np.ndarray] = None,
    token_v: Optional[np.ndarray] = None,
    *,
    support_depth_min: float = 0.3,
    support_depth_max: float = 2.0,
    boundary_threshold: float = 0.05,
    depth_edge_scores: Optional[np.ndarray] = None,
    object_min_area_tokens: int = 5,
    object_height_residual_threshold: float = 0.04,
    grid_h: int = 16,
    grid_w: int = 16,
    support_plane_cap_ratio: float = 0.30,
    boundary_cap_ratio: float = 0.35,
    support_plane_priority_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Compute scene-layout constraint scores from depth distribution.

    Identifies regions by depth:
      - Tabletop / support plane:  depth in [support_depth_min, support_depth_max],
        capped to ``support_plane_cap_ratio`` of total tokens, preferentially
        keeping anchors near objects/boundaries/center.
      - Object-like components:    depth-residual connected components above
        the estimated support plane surface.
      - Boundary / obstacle:        object component contours merged with
        depth-edge fallback.
      - Uncovered scene:            tokens on support plane not yet covered.

    Args:
        token_depth:  [N] token depth values in meters (metric).
        valid_mask:   [N] boolean mask of valid tokens.
        token_u, token_v: [N] integer pixel grid coordinates (optional).
        support_depth_min: Lower depth bound for tabletop / workspace (meters).
        support_depth_max: Upper depth bound for tabletop / workspace.
        boundary_threshold: Depth gradient threshold for boundary tokens.
        object_min_area_tokens: Minimum token count for an object component.
        object_height_residual_threshold: Height above support plane to
            classify as object (meters).
        grid_h, grid_w: Token grid dimensions.
        support_plane_cap_ratio: Maximum fraction of total tokens that can be
            selected as support_plane candidates (default 0.30 = 30%).
        support_plane_priority_weights: Dict with weights for support_plane
            candidate prioritization. Keys: "near_object", "near_boundary",
            "near_depth_center". Values: float weights.

    Returns:
        Dict with keys:
          scene_layout_scores:        [N] composite scene layout score [0, 1].
          support_plane_scores:       [N] support plane proximity score [0, 1].
          support_plane_candidate_scores: [N] per-token support plane candidate score.
          object_component_scores:    [N] object/component score [0, 1].
          boundary_scores:           [N] depth boundary score [0, 1].
          scene_fill_candidates:     [N] bool mask of scene-relevant fill tokens.
          component_ids:             [N] component label per token (-1 = none).
          num_components:           int, number of detected object components.
          support_plane_token_count: int (all in-range tokens, NOT capped).
          support_plane_candidate_count: int (capped candidate count).
          object_component_token_count: int.
          boundary_token_count:      int.
          # New P6 diagnostic fields
          support_plane_cap_ratio:   float (the cap that was applied).
          support_plane_cap_used:    bool (True if cap was applied).
          support_plane_fallback_used: bool (True if fallback was needed).
          support_plane_fallback_reason: str or None.
          object_component_fallback_used: bool.
          object_component_fallback_reason: str or None.
          boundary_fallback_used:    bool.
          boundary_fallback_reason:  str or None.
          object_component_num_components: int.
          boundary_from_object_count: int (boundary tokens from object contours).
          boundary_from_depth_count: int (boundary tokens from depth edges).
          # Timing (populated by caller)
          _timing_ns: int (wall time in nanoseconds, for caller to report).
    """
    t0 = _time.perf_counter_ns()

    n = int(np.prod(token_depth.shape)) if token_depth.ndim == 1 else int(token_depth.size)
    td = np.asarray(token_depth, dtype=np.float32).reshape(-1)
    vm = np.asarray(valid_mask, dtype=bool).reshape(-1)

    result: Dict[str, Any] = {
        "scene_layout_scores": np.zeros(n, dtype=np.float32),
        "support_plane_scores": np.zeros(n, dtype=np.float32),
        "support_plane_candidate_scores": np.zeros(n, dtype=np.float32),
        "object_component_scores": np.zeros(n, dtype=np.float32),
        "boundary_scores": np.zeros(n, dtype=np.float32),
        "scene_fill_candidates": np.zeros(n, dtype=np.float32),
        "component_ids": np.full(n, -1, dtype=np.int32),
        "num_components": 0,
        "support_plane_token_count": 0,
        "support_plane_candidate_count": 0,
        "object_component_token_count": 0,
        "boundary_token_count": 0,
        # P6 diagnostics
        "support_plane_cap_ratio": float(support_plane_cap_ratio),
        "support_plane_cap_used": False,
        "support_plane_fallback_used": False,
        "support_plane_fallback_reason": None,
        "boundary_cap_ratio": float(boundary_cap_ratio),
        "boundary_cap_used": False,
        "object_component_fallback_used": False,
        "object_component_fallback_reason": None,
        "boundary_fallback_used": False,
        "boundary_fallback_reason": None,
        "object_component_num_components": 0,
        "boundary_from_object_count": 0,
        "boundary_from_depth_count": 0,
        "_timing_ns": 0,
    }

    if n == 0 or td.size == 0:
        result["_timing_ns"] = int((_time.perf_counter_ns() - t0))
        return result

    finite_mask = np.isfinite(td) & (td > 0.0)
    valid_tokens = vm & finite_mask
    if not np.any(valid_tokens):
        result["_timing_ns"] = int((_time.perf_counter_ns() - t0))
        return result

    depth_lo = float(np.min(td[valid_tokens]))
    depth_hi = float(np.max(td[valid_tokens]))
    depth_range = max(depth_hi - depth_lo, 1e-6)
    depth_center = (depth_lo + depth_hi) * 0.5

    # ── 1. Support plane: all in-range tokens ────────────────────────────────
    # (candidate scoring + capping applied separately below)
    support_scores = np.zeros(n, dtype=np.float32)
    in_support = valid_tokens & (td >= support_depth_min) & (td <= support_depth_max)
    if np.any(in_support):
        denom = max(support_depth_max - support_depth_min, 1e-6)
        center_depth = (support_depth_min + support_depth_max) * 0.5
        dist_from_center = np.abs(td - center_depth)
        support_scores[in_support] = np.clip(
            1.0 - dist_from_center[in_support] / (denom * 0.5), 0.0, 1.0
        )
    result["support_plane_scores"] = support_scores
    result["support_plane_token_count"] = int(np.sum(in_support))

    # ── 2. Depth boundary (raw) ──────────────────────────────────────────────
    raw_boundary_scores = np.zeros(n, dtype=np.float32)
    has_coords = token_u is not None and token_v is not None and np.any(valid_tokens)

    if has_coords:
        u_arr = np.asarray(token_u, dtype=np.int32).reshape(-1)
        v_arr = np.asarray(token_v, dtype=np.int32).reshape(-1)

        u_max = int(np.max(u_arr[valid_tokens])) if np.any(valid_tokens) else grid_w - 1
        v_max = int(np.max(v_arr[valid_tokens])) if np.any(valid_tokens) else grid_h - 1

        val_to_coord: Dict[Tuple[int, int], Tuple[int, float]] = {}
        for vi in np.where(valid_tokens)[0]:
            val_to_coord[(int(u_arr[vi]), int(v_arr[vi]))] = (vi, float(td[vi]))

        grad_h = np.zeros(n, dtype=np.float32)
        grad_v = np.zeros(n, dtype=np.float32)

        for (ui, vi), (vi_idx, di) in val_to_coord.items():
            if ui > 0:
                lk = (ui - 1, vi)
                if lk in val_to_coord:
                    _, dl = val_to_coord[lk]
                    grad_h[vi_idx] = abs(float(di) - float(dl))
            if ui < u_max:
                rk = (ui + 1, vi)
                if rk in val_to_coord:
                    _, dr = val_to_coord[rk]
                    grad_h[vi_idx] = max(grad_h[vi_idx], abs(float(di) - float(dr)))
            if vi > 0:
                uk = (ui, vi - 1)
                if uk in val_to_coord:
                    _, dup = val_to_coord[uk]
                    grad_v[vi_idx] = abs(float(di) - float(dup))
            if vi < v_max:
                dk = (ui, vi + 1)
                if dk in val_to_coord:
                    _, dd = val_to_coord[dk]
                    grad_v[vi_idx] = max(grad_v[vi_idx], abs(float(di) - float(dd)))

        grad_mag = np.sqrt(grad_h ** 2 + grad_v ** 2)
        if np.any(grad_mag > boundary_threshold):
            boundary_lo = boundary_threshold
            boundary_hi = float(np.percentile(grad_mag[grad_mag > boundary_threshold], 95))
            boundary_hi = max(boundary_hi, boundary_lo + 1e-6)
            raw_boundary_scores = np.clip(
                (grad_mag - boundary_lo) / (boundary_hi - boundary_lo), 0.0, 1.0
            )
            raw_boundary_scores[grad_mag <= boundary_threshold] = 0.0

    if depth_edge_scores is not None:
        de = np.asarray(depth_edge_scores, dtype=np.float32).reshape(-1)
        if de.shape[0] == n and np.any(np.isfinite(de) & (de > 0.0)):
            de = np.nan_to_num(de, nan=0.0, posinf=0.0, neginf=0.0)
            de = np.clip(de, 0.0, None)
            de_hi = float(np.max(de[valid_tokens])) if np.any(valid_tokens) else 0.0
            if de_hi > 1e-8:
                de = np.clip(de / de_hi, 0.0, 1.0)
                raw_boundary_scores = np.maximum(raw_boundary_scores, de)

    # ── 3. Object proposal: depth-residual + connected components ────────────
    component_scores = np.zeros(n, dtype=np.float32)
    component_ids = np.full(n, -1, dtype=np.int32)
    num_components = 0
    object_fallback_used = False
    object_fallback_reason = None
    obj_contour_mask = np.zeros(n, dtype=bool)

    plane_depth, plane_std = _estimate_support_plane_depth(td, in_support)
    # The support band can include object pixels and background, so its raw
    # standard deviation may be too large for tabletop objects. Cap the adaptive
    # threshold to keep object proposals sensitive while preserving the
    # configured minimum residual.
    residual_threshold = max(object_height_residual_threshold, min(plane_std * 1.5, 0.12))

    in_support_indices = np.where(in_support)[0]

    if has_coords and len(in_support_indices) >= object_min_area_tokens:
        u_sp = np.asarray(token_u, dtype=np.int32).reshape(-1)
        v_sp = np.asarray(token_v, dtype=np.int32).reshape(-1)

        # In camera-depth images, objects above a support surface are usually
        # closer to the camera, so they have smaller metric depth than the
        # estimated support plane. Fall back to absolute residual only when
        # the closer-than-plane proposal is too sparse.
        closer_residual = plane_depth - td
        above_plane_mask = in_support & (closer_residual > residual_threshold)
        if np.sum(above_plane_mask) < object_min_area_tokens:
            abs_residual = np.abs(td - plane_depth)
            above_plane_mask = in_support & (abs_residual > residual_threshold)

        if np.sum(above_plane_mask) >= object_min_area_tokens:
            comp_labels, comp_scores, num_comps = _connected_components_4n(
                mask=above_plane_mask,
                u=u_sp,
                v=v_sp,
                min_size=object_min_area_tokens,
            )
            if num_comps > 0:
                component_ids = comp_labels
                component_scores = comp_scores
                num_components = num_comps

                max_support_size = float(np.sum(in_support))
                for idx in range(n):
                    if comp_labels[idx] >= 0:
                        comp_size = float(np.sum(comp_labels == comp_labels[idx]))
                        size_factor = min(comp_size / max(5.0, 0.05 * max_support_size), 1.0)
                        component_scores[idx] = float(comp_scores[idx] * size_factor)

                obj_contour_mask = _object_contour_mask(comp_labels, u_sp, v_sp)
            else:
                object_fallback_used = True
                object_fallback_reason = "no_above_plane_components"
        else:
            object_fallback_used = True
            object_fallback_reason = (
                f"insufficient_above_plane_tokens"
                f"(have={int(np.sum(above_plane_mask))}, need={object_min_area_tokens})"
            )
    else:
        object_fallback_used = True
        object_fallback_reason = (
            "no_coords" if not has_coords else
            f"insufficient_support_tokens({len(in_support_indices)}<{object_min_area_tokens})"
        )

    result["object_component_scores"] = component_scores
    result["component_ids"] = component_ids
    result["object_component_token_count"] = int(np.sum(component_scores > 0.0))
    result["object_component_num_components"] = num_components
    result["object_component_fallback_used"] = object_fallback_used
    result["object_component_fallback_reason"] = object_fallback_reason

    # ── 4. Boundary: object contours + depth edge fallback ───────────────────
    boundary_scores = np.zeros(n, dtype=np.float32)
    boundary_from_object_count = 0
    boundary_from_depth_count = 0
    boundary_fallback_used = False
    boundary_fallback_reason = None

    if np.any(obj_contour_mask):
        boundary_scores[obj_contour_mask] = 1.0
        boundary_from_object_count = int(np.sum(obj_contour_mask))

    if has_coords:
        if np.any(raw_boundary_scores > 0):
            above_thresh = raw_boundary_scores > 0.0
            new_boundary = above_thresh & ~obj_contour_mask
            if np.any(new_boundary):
                boundary_scores[new_boundary] = np.maximum(
                    boundary_scores[new_boundary], raw_boundary_scores[new_boundary]
                )
                boundary_from_depth_count = int(np.sum(new_boundary))
        else:
            if boundary_from_object_count == 0:
                boundary_fallback_used = True
                boundary_fallback_reason = "no_depth_edges_detected"
    else:
        if boundary_from_object_count == 0:
            boundary_fallback_used = True
            boundary_fallback_reason = "no_coords_for_depth_boundary"

    # Cap noisy depth-edge boundaries so scene layout remains a sparse
    # constraint set rather than an all-image fill mask. Object contours get
    # priority through their higher score above.
    boundary_cap_k = max(1, int(round(n * max(0.01, min(float(boundary_cap_ratio), 1.0)))))
    boundary_mask = boundary_scores > 0.0
    if int(np.sum(boundary_mask)) > boundary_cap_k:
        result["boundary_cap_used"] = True
        boundary_order = np.argsort(-boundary_scores)
        keep_boundary = np.zeros(n, dtype=bool)
        kept = 0
        for idx in boundary_order:
            idx_i = int(idx)
            if not boundary_mask[idx_i]:
                break
            keep_boundary[idx_i] = True
            kept += 1
            if kept >= boundary_cap_k:
                break
        boundary_scores[~keep_boundary] = 0.0
        boundary_mask = keep_boundary

    boundary_from_object_count = int(np.sum(boundary_mask & obj_contour_mask))
    boundary_from_depth_count = int(np.sum(boundary_mask & ~obj_contour_mask))

    result["boundary_scores"] = boundary_scores
    result["boundary_token_count"] = int(np.sum(boundary_mask))
    result["boundary_from_object_count"] = boundary_from_object_count
    result["boundary_from_depth_count"] = boundary_from_depth_count
    result["boundary_fallback_used"] = boundary_fallback_used
    result["boundary_fallback_reason"] = boundary_fallback_reason

    # ── 5. Support plane candidate scoring + capping ─────────────────────────
    # Build priority scores for support_plane candidates:
    #   - near object components (+1.0)
    #   - near boundary (+0.8)
    #   - near depth center (+0.4)
    w_obj = 1.0
    w_bnd = 0.8
    w_ctr = 0.4
    if support_plane_priority_weights:
        w_obj = float(support_plane_priority_weights.get("near_object", w_obj))
        w_bnd = float(support_plane_priority_weights.get("near_boundary", w_bnd))
        w_ctr = float(support_plane_priority_weights.get("near_depth_center", w_ctr))

    sp_candidate_scores = np.zeros(n, dtype=np.float32)
    in_sp_candidates = np.zeros(n, dtype=bool)

    if np.any(in_support) and has_coords:
        u_n = np.asarray(token_u, dtype=np.int32).reshape(-1)
        v_n = np.asarray(token_v, dtype=np.int32).reshape(-1)

        if np.any(component_scores > 0):
            comp_u = u_n[component_scores > 0]
            comp_v = v_n[component_scores > 0]
            comp_center_u = float(np.mean(comp_u)) if len(comp_u) > 0 else float(grid_w) * 0.5
            comp_center_v = float(np.mean(comp_v)) if len(comp_v) > 0 else float(grid_h) * 0.5
        else:
            comp_center_u = float(grid_w) * 0.5
            comp_center_v = float(grid_h) * 0.5

        if np.any(boundary_scores > 0):
            bnd_u = u_n[boundary_scores > 0]
            bnd_v = v_n[boundary_scores > 0]
            bnd_center_u = float(np.mean(bnd_u)) if len(bnd_u) > 0 else float(grid_w) * 0.5
            bnd_center_v = float(np.mean(bnd_v)) if len(bnd_v) > 0 else float(grid_h) * 0.5
        else:
            bnd_center_u = float(grid_w) * 0.5
            bnd_center_v = float(grid_h) * 0.5

        u_sp_all = u_n[in_support]
        v_sp_all = v_n[in_support]
        grid_diag = float(np.sqrt(grid_w ** 2 + grid_h ** 2))

        dist_to_obj = np.sqrt(
            (u_sp_all - comp_center_u) ** 2 + (v_sp_all - comp_center_v) ** 2
        )
        dist_to_bnd = np.sqrt(
            (u_sp_all - bnd_center_u) ** 2 + (v_sp_all - bnd_center_v) ** 2
        )
        dist_to_ctr = np.sqrt(
            (u_sp_all - float(grid_w) * 0.5) ** 2
            + (v_sp_all - float(grid_h) * 0.5) ** 2
        )

        norm_dist_obj = np.clip(dist_to_obj / (grid_diag * 0.5), 0.0, 1.0)
        norm_dist_bnd = np.clip(dist_to_bnd / (grid_diag * 0.5), 0.0, 1.0)
        norm_dist_ctr = np.clip(dist_to_ctr / (grid_diag * 0.5), 0.0, 1.0)

        near_obj_score = (1.0 - norm_dist_obj) * w_obj
        near_bnd_score = (1.0 - norm_dist_bnd) * w_bnd
        near_ctr_score = (1.0 - norm_dist_ctr) * w_ctr

        sp_priority = near_obj_score + near_bnd_score + near_ctr_score
        sp_priority = np.clip(sp_priority / max(np.max(sp_priority), 1e-6), 0.0, 1.0)

        combined = 0.4 * support_scores[in_support] + 0.6 * sp_priority
        combined = np.clip(combined, 0.0, 1.0).astype(np.float32)
        sp_candidate_scores[in_support] = combined
    else:
        sp_candidate_scores[in_support] = np.clip(support_scores[in_support], 0.0, 1.0)

    # Apply cap: only top-k by sp_candidate_scores become support_plane candidates
    cap_k = max(1, int(round(n * support_plane_cap_ratio)))
    total_sp = int(np.sum(in_support))
    cap_applied = total_sp > cap_k

    if cap_applied:
        result["support_plane_cap_used"] = True

        support_indices = np.where(in_support)[0]
        topk_order = support_indices[np.argsort(-sp_candidate_scores[support_indices])]
        topk_indices = topk_order[:cap_k]
        in_sp_candidates = np.zeros(n, dtype=bool)
        in_sp_candidates[topk_indices] = True
        sp_candidate_scores[~in_sp_candidates] = 0.0
    else:
        in_sp_candidates = in_support.copy()

    result["support_plane_candidate_scores"] = sp_candidate_scores
    result["support_plane_candidate_count"] = int(np.sum(in_sp_candidates))

    # Update scene_layout_scores with SP candidates only (not blanket fill)
    # Use in_sp_candidates as the support_plane mask for scoring
    sp_mask = in_sp_candidates

    # ── 6. Composite scene_layout_score ─────────────────────────────────────
    scene_layout = (
        0.4 * np.clip(sp_candidate_scores, 0.0, 1.0)
        + 0.35 * np.clip(component_scores, 0.0, 1.0)
        + 0.25 * np.clip(boundary_scores, 0.0, 1.0)
    )
    scene_layout = np.clip(scene_layout, 0.0, 1.0).astype(np.float32)
    scene_layout[~valid_tokens] = 0.0
    result["scene_layout_scores"] = scene_layout

    # ── 7. Scene fill candidates ─────────────────────────────────────────────
    valid_depth = (td > 0.1) & (td < 5.0)
    scene_relevant = (
        (sp_candidate_scores > 0.0)
        | (component_scores > 0.0)
        | (boundary_scores > 0.0)
    )
    fill_candidates = valid_tokens & valid_depth & (
        scene_relevant
    )
    result["scene_fill_candidates"] = fill_candidates.astype(np.float32)

    result["_timing_ns"] = int((_time.perf_counter_ns() - t0))
    return result


def compute_scene_layout_selected_counts(
    keep_indices: np.ndarray,
    scene_result: Dict[str, Any],
    n_total: int,
) -> Dict[str, Any]:
    """Count how many kept tokens belong to each scene-layout component category.

    This function should be called AFTER the token selection has been made.
    It attributes each kept token to one of:
      - support_plane:  tokens whose depth falls in the support plane range
                        AND were selected as candidates (not the full blanket)
      - object_component: tokens that are part of a detected object-like component
      - boundary:        tokens that are on a depth discontinuity / edge
      - relation:        DEPRECATED — renamed to "residual fill" below.

    NOTE on "relation": the tokens counted here are NOT semantic spatial relations
    (e.g. "bowl is between plate and mug"). They are geometry-only leftovers:
    tokens that are scene-relevant (scene_layout_score > 0) but NOT in any of
    support_plane / object_component / boundary. These are named "residual fill"
    to avoid semantic misleading. True semantic relation tokens are tracked by the
    ACGTP-v2 semantic_anchors module when a visual detector is available.

    The corrected field names (acgtp_scene_selected_residual_fill_count, etc.)
    are populated by compute_scene_layout_selected_counts_v2() or by the v2
    selector. This function retains the old "relation" names for backward
    compatibility only.

    The ``scene_result`` dict should be the return value of ``compute_scene_layout_scores``.
    The ``keep_indices`` should be the numpy array of selected token indices.
    The ``n_total`` should be the total number of tokens (n).

    Args:
        keep_indices: [K] array of selected token indices.
        scene_result: Return value of ``compute_scene_layout_scores``.
        n_total: Total number of tokens (should match scene_result arrays).

    Returns:
        Dict with keys:
          acgtp_scene_selected_support_plane_count:  int.
          acgtp_scene_selected_object_component_count: int.
          acgtp_scene_selected_boundary_count:       int.
          acgtp_scene_selected_relation_count:       int or None (if not computable).
          acgtp_scene_support_plane_selected_ratio:   float (0.0 if sp_count == 0).
          acgtp_scene_relation_token_count:          total relation tokens in grid (or None).
          acgtp_scene_relation_token_count_computed:  bool indicating whether relation was derived.
    """
    result: Dict[str, Any] = {
        "acgtp_scene_selected_support_plane_count": 0,
        "acgtp_scene_selected_object_component_count": 0,
        "acgtp_scene_selected_boundary_count": 0,
        "acgtp_scene_selected_relation_count": None,
        "acgtp_scene_support_plane_selected_ratio": 0.0,
        "acgtp_scene_relation_token_count": None,
        "acgtp_scene_relation_token_count_computed": False,
    }

    if keep_indices is None or len(keep_indices) == 0:
        return result

    ki = np.asarray(keep_indices, dtype=np.intp).reshape(-1)
    n = n_total

    # Extract component masks from scene_result
    support_scores = scene_result.get("support_plane_scores")
    sp_cand_scores = scene_result.get("support_plane_candidate_scores")
    object_scores = scene_result.get("object_component_scores")
    boundary_scores = scene_result.get("boundary_scores")
    scene_layout_scores = scene_result.get("scene_layout_scores")

    for arr in (support_scores, object_scores, boundary_scores, scene_layout_scores):
        if arr is not None:
            try:
                arr_n = int(np.asarray(arr, dtype=object).size)
                n = max(n, arr_n)
            except (TypeError, ValueError):
                pass

    # Build per-token category booleans
    # Use support_plane_candidate_scores for SP classification (capped candidates)
    is_support = np.zeros(n, dtype=bool)
    is_object = np.zeros(n, dtype=bool)
    is_boundary = np.zeros(n, dtype=bool)

    if sp_cand_scores is not None:
        s = np.asarray(sp_cand_scores, dtype=np.float32).reshape(-1)
        if s.shape[0] == n:
            is_support = s > 0.0
    elif support_scores is not None:
        s = np.asarray(support_scores, dtype=np.float32).reshape(-1)
        if s.shape[0] == n:
            is_support = s > 0.0

    if object_scores is not None:
        o = np.asarray(object_scores, dtype=np.float32).reshape(-1)
        if o.shape[0] == n:
            is_object = o > 0.0

    if boundary_scores is not None:
        b = np.asarray(boundary_scores, dtype=np.float32).reshape(-1)
        if b.shape[0] == n:
            is_boundary = b > 0.0

    # Relation tokens: tokens that have scene relevance but are NOT in any specific category
    is_relation = np.zeros(n, dtype=bool)
    if scene_layout_scores is not None:
        sl = np.asarray(scene_layout_scores, dtype=np.float32).reshape(-1)
        if sl.shape[0] == n:
            scene_relevant = sl > 0.0
            is_relation = scene_relevant & ~is_support & ~is_object & ~is_boundary

    # Ensure keep_indices are within bounds
    valid_ki_mask = (ki >= 0) & (ki < n)
    ki_valid = ki[valid_ki_mask]

    result["acgtp_scene_selected_support_plane_count"] = int(np.sum(is_support[ki_valid]))
    result["acgtp_scene_selected_object_component_count"] = int(np.sum(is_object[ki_valid]))
    result["acgtp_scene_selected_boundary_count"] = int(np.sum(is_boundary[ki_valid]))

    if np.any(is_relation):
        result["acgtp_scene_selected_relation_count"] = int(np.sum(is_relation[ki_valid]))
        result["acgtp_scene_relation_token_count"] = int(np.sum(is_relation))
        result["acgtp_scene_relation_token_count_computed"] = True

    # Ratio of selected support_plane tokens
    total_sp = int(np.sum(is_support))
    if total_sp > 0:
        selected_sp = result["acgtp_scene_selected_support_plane_count"]
        result["acgtp_scene_support_plane_selected_ratio"] = float(selected_sp) / float(total_sp)

    return result
