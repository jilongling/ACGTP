"""Pruning visualization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


def save_pruning_visualization(
    *,
    output_dir: str,
    method: str,
    episode_id: int,
    step_id: int,
    rgb: Optional[np.ndarray],
    depth: Optional[np.ndarray],
    token_u: Optional[np.ndarray],
    token_v: Optional[np.ndarray],
    keep_indices: Sequence[int],
    score_maps: Dict[str, Optional[np.ndarray]],
    selection_masks: Optional[Dict[str, Sequence[int]]] = None,
    token_grid_shape: Tuple[int, int] = (16, 16),
) -> str:
    """Save paper-style per-step pruning visualizations."""
    out = Path(output_dir)
    if method == "robot_geo_contact_budget" and out.parent.name:
        run_name = out.parent.name
        base = out.parent.parent / "contact_budget_diagnostics" / run_name
    else:
        base = out / "pruning_visualizations"
    step_dir = base / str(method) / f"episode_{int(episode_id):04d}" / f"step_{int(step_id):04d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    grid_h, grid_w = token_grid_shape
    num_grid_tokens = grid_h * grid_w
    keep_mask = np.zeros(num_grid_tokens, dtype=np.float32)
    keep = np.asarray(list(keep_indices), dtype=np.int64)
    keep = keep[(keep >= 0) & (keep < num_grid_tokens)]
    keep_mask[keep] = 1.0

    if rgb is not None:
        rgb_img = _to_rgb_image(rgb)
        rgb_img.save(step_dir / "rgb.png")
    else:
        rgb_img = None

    if depth is not None:
        _heatmap_image(_normalize_image(np.asarray(depth).squeeze())).save(step_dir / "depth.png")

    _save_score(step_dir / "depth_edge_score.png", score_maps.get("depth_edge_score"), token_grid_shape)
    _save_score(step_dir / "gripper_distance_score.png", score_maps.get("gripper_distance_score"), token_grid_shape)
    near_score = score_maps.get("near_score")
    if near_score is None:
        near_score = score_maps.get("gripper_distance_score")
    _save_score(step_dir / "near_score.png", near_score, token_grid_shape)
    _save_score(step_dir / "motion_corridor_score.png", score_maps.get("motion_corridor_score"), token_grid_shape)
    corridor_score = score_maps.get("corridor_score")
    if corridor_score is None:
        corridor_score = score_maps.get("motion_corridor_score")
    _save_score(step_dir / "corridor_score.png", corridor_score, token_grid_shape)
    _save_score(step_dir / "near_contact_score.png", score_maps.get("near_contact_score"), token_grid_shape)
    _save_score(step_dir / "corridor_contact_score.png", score_maps.get("corridor_contact_score"), token_grid_shape)
    _save_score(step_dir / "geo_contact_score.png", score_maps.get("geo_contact_score"), token_grid_shape)
    _save_score(step_dir / "final_geometry_score.png", score_maps.get("final_geometry_score"), token_grid_shape)
    _binary_image(keep_mask.reshape(grid_h, grid_w)).save(step_dir / "final_keep_mask.png")
    selected_masks = {}
    if selection_masks:
        for name, indices in selection_masks.items():
            mask = np.zeros(num_grid_tokens, dtype=np.float32)
            idx = np.asarray(list(indices), dtype=np.int64)
            idx = idx[(idx >= 0) & (idx < num_grid_tokens)]
            mask[idx] = 1.0
            selected_masks[name] = mask
            _binary_image(mask.reshape(grid_h, grid_w)).save(step_dir / f"selected_by_{name}_mask.png")

    if rgb_img is not None and token_u is not None and token_v is not None:
        overlay = rgb_img.copy()
        draw = ImageDraw.Draw(overlay)
        u = np.asarray(token_u).reshape(-1)[:num_grid_tokens]
        v = np.asarray(token_v).reshape(-1)[:num_grid_tokens]
        kept = set(int(i) for i in keep.tolist())
        edge_sel = set(int(i) for i in np.where(selected_masks.get("edge", np.zeros(num_grid_tokens)) > 0.5)[0].tolist())
        geo_sel = set(int(i) for i in np.where(selected_masks.get("geo", np.zeros(num_grid_tokens)) > 0.5)[0].tolist())
        diverse_sel = set(int(i) for i in np.where(selected_masks.get("diverse", np.zeros(num_grid_tokens)) > 0.5)[0].tolist())
        for idx in range(min(num_grid_tokens, u.shape[0], v.shape[0])):
            if idx in edge_sel:
                color = (0, 180, 255)
            elif idx in geo_sel:
                color = (255, 60, 220)
            elif idx in diverse_sel:
                color = (255, 210, 0)
            else:
                color = (0, 255, 80) if idx in kept else (255, 80, 40)
            r = 2 if idx in kept else 1
            x, y = float(u[idx]), float(v[idx])
            draw.ellipse((x - r, y - r, x + r, y + r), outline=color, fill=color if idx in kept else None)
        overlay.save(step_dir / "overlay_kept_token_centers.png")
        if selection_masks:
            overlay.save(step_dir / "overlay_contact_budget_sources.png")

    metadata = {
        "method": method,
        "episode_id": int(episode_id),
        "step_id": int(step_id),
        "token_grid_shape": [int(grid_h), int(grid_w)],
        "num_keep_indices": int(len(keep_indices)),
        "num_grid_keep": int(np.sum(keep_mask)),
        "selected_by_edge_count": int(np.sum(selected_masks.get("edge", np.zeros(num_grid_tokens)))) if selected_masks else None,
        "selected_by_geo_count": int(np.sum(selected_masks.get("geo", np.zeros(num_grid_tokens)))) if selection_masks else None,
        "selected_by_diverse_count": int(np.sum(selected_masks.get("diverse", np.zeros(num_grid_tokens)))) if selection_masks else None,
    }
    with open(step_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    return str(step_dir)


def save_geo_debug_visualization(
    *,
    enabled: bool,
    output_dir: str,
    method: str,
    episode_id: int,
    step_id: int,
    keep_indices: Sequence[int],
    score_maps: Dict[str, Optional[np.ndarray]],
    dynamic_info: Optional[Dict[str, Any]] = None,
    rgb: Optional[np.ndarray] = None,
    token_u: Optional[np.ndarray] = None,
    token_v: Optional[np.ndarray] = None,
    token_grid_shape: Tuple[int, int] = (16, 16),
    topk: int = 8,
) -> Optional[str]:
    """Save lightweight geometry-expert debug visualizations.

    The caller is expected to gate this behind `enable_geo_debug`; this function
    also accepts `enabled=False` for unit tests and defensive no-op use.
    """
    if not enabled:
        return None

    out = Path(output_dir)
    step_dir = out / "geo_debug" / str(method) / f"episode_{int(episode_id):04d}" / f"step_{int(step_id):04d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    grid_h, grid_w = token_grid_shape
    num_grid_tokens = grid_h * grid_w
    keep_mask = np.zeros(num_grid_tokens, dtype=np.float32)
    keep = np.asarray(list(keep_indices), dtype=np.int64)
    keep = keep[(keep >= 0) & (keep < num_grid_tokens)]
    keep_mask[keep] = 1.0

    _binary_image(keep_mask.reshape(grid_h, grid_w)).save(step_dir / "final_keep_mask.png")
    _save_score(step_dir / "distance_to_gripper_score.png", score_maps.get("distance_to_gripper_score"), token_grid_shape)
    _save_score(step_dir / "motion_cone_score.png", score_maps.get("motion_cone_score"), token_grid_shape)
    _save_score(step_dir / "contact_risk_score.png", score_maps.get("contact_risk_score"), token_grid_shape)
    _save_score(step_dir / "depth_edge_score.png", score_maps.get("depth_edge_score"), token_grid_shape)

    info = dict(dynamic_info or {})
    _text_image({
        "method": method,
        "episode_id": int(episode_id),
        "step_id": int(step_id),
        "dynamic_keep_ratio": info.get("dynamic_keep_ratio"),
        "risk_level": info.get("risk_level"),
        "risk_score": info.get("risk_score"),
        "reason": info.get("reason"),
        "num_high_contact_tokens": info.get("num_high_contact_tokens"),
        "num_valid_3d_tokens": info.get("num_valid_3d_tokens"),
    }).save(step_dir / "dynamic_info.png")

    if rgb is not None:
        rgb_img = _to_rgb_image(rgb)
        rgb_img.save(step_dir / "rgb.png")
        overlay = rgb_img.copy()
        draw = ImageDraw.Draw(overlay, "RGBA")
        if token_u is not None and token_v is not None:
            u = np.asarray(token_u).reshape(-1)[:num_grid_tokens]
            v = np.asarray(token_v).reshape(-1)[:num_grid_tokens]
            kept = set(int(i) for i in keep.tolist())
            for idx in range(min(num_grid_tokens, u.shape[0], v.shape[0])):
                x, y = float(u[idx]), float(v[idx])
                if idx in kept:
                    draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(0, 255, 80, 210), outline=(0, 80, 20, 255))
                else:
                    draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(255, 60, 40, 120))
        else:
            _draw_grid_overlay(draw, overlay.size, keep_mask.reshape(grid_h, grid_w))
        overlay.save(step_dir / "rgb_keep_overlay.png")

    # Build top-k per score map
    score_topk = {}
    gh, gw = token_grid_shape
    for name, arr in score_maps.items():
        if arr is not None:
            flat = np.nan_to_num(np.asarray(arr, dtype=np.float32).reshape(-1), nan=0.0)
            topk_indices = np.argsort(-flat)[:topk]
            topk_grid = [(int(idx // gw), int(idx % gw)) for idx in topk_indices]
            score_topk[name] = {
                "indices": [int(i) for i in topk_indices],
                "grid_coords": topk_grid,
                "scores": [float(flat[i]) for i in topk_indices],
            }
            if token_u is not None and token_v is not None:
                u_all = np.asarray(token_u).reshape(-1)
                v_all = np.asarray(token_v).reshape(-1)
                score_topk[name]["pixel_coords"] = [
                    (float(u_all[i]), float(v_all[i])) for i in topk_indices if i < len(u_all)
                ]

    metadata = {
        "method": method,
        "episode_id": int(episode_id),
        "step_id": int(step_id),
        "token_grid_shape": [int(grid_h), int(grid_w)],
        "num_keep_indices": int(len(keep_indices)),
        "num_grid_keep": int(np.sum(keep_mask)),
        "score_topk": score_topk,
        "topk_scores": {
            "distance_to_gripper": score_topk.get("distance_to_gripper_score", {}).get("scores", []),
            "motion_cone": score_topk.get("motion_cone_score", {}).get("scores", []),
            "contact_risk": score_topk.get("contact_risk_score", {}).get("scores", []),
            "depth_edge": score_topk.get("depth_edge_score", {}).get("scores", []),
            "final": score_topk.get("final_geometry_score", {}).get("scores", []),
        },
        "topk_indices": {
            "distance_to_gripper": score_topk.get("distance_to_gripper_score", {}).get("indices", []),
            "motion_cone": score_topk.get("motion_cone_score", {}).get("indices", []),
            "contact_risk": score_topk.get("contact_risk_score", {}).get("indices", []),
            "depth_edge": score_topk.get("depth_edge_score", {}).get("indices", []),
            "final": score_topk.get("final_geometry_score", {}).get("indices", []),
        },
        "dynamic_info": info,
        "saved_files": [
            "final_keep_mask.png",
            "distance_to_gripper_score.png",
            "motion_cone_score.png",
            "contact_risk_score.png",
            "depth_edge_score.png",
            "dynamic_info.png",
        ],
    }
    with open(step_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    return str(step_dir)


def _save_score(path: Path, scores: Optional[np.ndarray], token_grid_shape: Tuple[int, int]) -> None:
    grid_h, grid_w = token_grid_shape
    if scores is None:
        arr = np.zeros((grid_h, grid_w), dtype=np.float32)
    else:
        flat = np.nan_to_num(np.asarray(scores, dtype=np.float32).reshape(-1), nan=0.0)
        tokens = grid_h * grid_w
        if flat.shape[0] < tokens:
            padded = np.zeros(tokens, dtype=np.float32)
            padded[: flat.shape[0]] = flat
            flat = padded
        arr = flat[:tokens].reshape(grid_h, grid_w)
    _heatmap_image(_normalize_image(arr)).save(path)


def _to_rgb_image(rgb: np.ndarray) -> Image.Image:
    arr = np.asarray(rgb)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def _normalize_image(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi - lo <= 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - lo) / (hi - lo)


def _heatmap_image(norm: np.ndarray) -> Image.Image:
    x = np.clip(norm, 0.0, 1.0)
    r = np.clip(1.5 * x, 0, 1)
    g = np.clip(1.5 - np.abs(2.0 * x - 1.0) * 1.5, 0, 1)
    b = np.clip(1.5 * (1.0 - x), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8)).resize((320, 320), Image.Resampling.NEAREST)


def _binary_image(mask: np.ndarray) -> Image.Image:
    x = (np.asarray(mask, dtype=np.float32) > 0.5).astype(np.uint8) * 255
    rgb = np.stack([x, x, x], axis=-1)
    return Image.fromarray(rgb).resize((320, 320), Image.Resampling.NEAREST)


def _text_image(info: Dict[str, Any]) -> Image.Image:
    img = Image.new("RGB", (640, 260), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    y = 12
    for key, value in info.items():
        draw.text((12, y), f"{key}: {value}", fill=(0, 0, 0))
        y += 24
    return img


def _draw_grid_overlay(draw: ImageDraw.ImageDraw, size: Tuple[int, int], keep_mask: np.ndarray) -> None:
    width, height = size
    grid_h, grid_w = keep_mask.shape
    cell_w = width / float(grid_w)
    cell_h = height / float(grid_h)
    for r in range(grid_h):
        for c in range(grid_w):
            if keep_mask[r, c] <= 0.5:
                continue
            x0, y0 = c * cell_w, r * cell_h
            x1, y1 = (c + 1) * cell_w, (r + 1) * cell_h
            draw.rectangle((x0, y0, x1, y1), outline=(0, 255, 80, 220), fill=(0, 255, 80, 45))


# =============================================================================
# P1: Token selection debug visualization
# =============================================================================


def save_token_selection_debug_visualization(
    *,
    output_dir: str,
    method: str,
    episode_id: int,
    step_id: int,
    rgb: Optional[np.ndarray],
    token_u: Optional[np.ndarray],
    token_v: Optional[np.ndarray],
    keep_indices: Sequence[int],
    depth_edge_scores: Optional[np.ndarray],
    robot_geo_scores: Optional[np.ndarray],
    gripper_pixel: Optional[np.ndarray],
    token_grid_shape: Tuple[int, int] = (16, 16),
) -> str:
    """P1: Save token selection debug overlay showing depth_edge vs robot_geo top-k overlap.

    Colors:
      - Final selected tokens:       green  (0, 255, 80)
      - Depth edge top-k tokens:   yellow (255, 255, 0)
      - Hybrid/final-score top-k:  red    (255, 0, 0)  (legacy: robot_geo_topk)
      - Overlap tokens:             cyan   (0, 255, 255)
      - Gripper projection:         blue   (0, 120, 255)

    Each image is saved with task id, episode id, step id, method name in filename.
    """
    out = Path(output_dir)
    step_dir = out / "token_selection_debug" / str(method) / f"episode_{int(episode_id):04d}" / f"step_{int(step_id):04d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    grid_h, grid_w = token_grid_shape
    n_grid = grid_h * grid_w

    if rgb is not None:
        rgb_img = _to_rgb_image(rgb)
    else:
        rgb_img = Image.new("RGB", (640, 480), color=(40, 40, 40))

    idx = np.asarray(list(keep_indices), dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < n_grid)]
    keep_set = set(int(i) for i in idx.tolist())
    k_keep = len(keep_set)

    # Depth edge top-k (80 tokens for 192-keep scenario)
    edge_topk = set()
    if depth_edge_scores is not None:
        edge_flat = np.nan_to_num(np.asarray(depth_edge_scores, dtype=np.float32).reshape(-1), nan=-np.inf)
        k_edge_topk = max(1, min(int(round(k_keep * 0.80)), n_grid))
        edge_order = np.argsort(-edge_flat)
        for i in edge_order[:k_edge_topk]:
            if 0 <= i < n_grid:
                edge_topk.add(int(i))

    # Hybrid/final-score top-k (legacy: robot_geo_topk)
    geo_topk = set()
    if robot_geo_scores is not None:
        geo_flat = np.nan_to_num(np.asarray(robot_geo_scores, dtype=np.float32).reshape(-1), nan=-np.inf)
        k_geo_topk = max(1, min(int(round(k_keep * 0.80)), n_grid))
        geo_order = np.argsort(-geo_flat)
        for i in geo_order[:k_geo_topk]:
            if 0 <= i < n_grid:
                geo_topk.add(int(i))

    overlap = edge_topk & geo_topk
    edge_only = edge_topk - geo_topk
    geo_only = geo_topk - edge_topk

    overlay = rgb_img.copy()
    overlay = overlay.convert("RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")

    if token_u is not None and token_v is not None:
        u_arr = np.asarray(token_u, dtype=np.float32).reshape(-1)
        v_arr = np.asarray(token_v, dtype=np.float32).reshape(-1)

        # Draw all token centers as faint dots first
        for i in range(min(n_grid, u_arr.size, v_arr.size)):
            x, y = float(u_arr[i]), float(v_arr[i])
            if 0 <= i < n_grid:
                if i in overlap:
                    color = (0, 255, 255, 220)
                    r = 4
                elif i in edge_only:
                    color = (255, 255, 0, 200)
                    r = 3
                elif i in geo_only:
                    color = (255, 0, 0, 200)
                    r = 3
                elif i in keep_set:
                    color = (0, 255, 80, 220)
                    r = 3
                else:
                    color = (180, 180, 180, 80)
                    r = 1
                draw.ellipse((x - r, y - r, x + r, y + r), fill=color[:3] + (color[3],), outline=color[:3] + (255,))

    # Draw gripper projection point
    if gripper_pixel is not None:
        gx, gy = float(gripper_pixel[0]), float(gripper_pixel[1])
        r = 6
        draw.ellipse((gx - r, gy - r, gx + r, gy + r), fill=(0, 120, 255), outline=(0, 60, 200, 255))
        draw.text((gx + 8, gy - 12), "gripper", fill=(0, 120, 255))

    overlay_resized = overlay.resize((640, 480), Image.Resampling.LANCZOS)
    overlay_rgb = overlay_resized.convert("RGB")

    # Compose filename with metadata
    filename = f"ep{episode_id:04d}_st{step_id:04d}_{method}.png"
    try:
        overlay_rgb.save(step_dir / filename)
    except Exception as exc:
        print(f"[VIS] Failed to save overlay PNG: {exc}")
        overlay_resized.save(step_dir / filename)

    # Save legend image
    legend = _make_legend()
    legend.save(step_dir / "legend.png")

    # Save metadata
    metadata = {
        "method": method,
        "episode_id": int(episode_id),
        "step_id": int(step_id),
        "token_grid_shape": [int(grid_h), int(grid_w)],
        "num_selected_tokens": k_keep,
        "depth_edge_topk_count": len(edge_topk),
        "robot_geo_topk_count": len(geo_topk),
        "overlap_count": len(overlap),
        "edge_only_count": len(edge_only),
        "geo_only_count": len(geo_only),
        "overlap_ratio": len(overlap) / max(1, len(edge_topk | geo_topk)),
        "field_semantics": {
            "robot_geo_topk_*": "LEGACY/MISNAMED: top-k over hybrid_final_scores (preferred) or final_scores; NOT pure robot geometry",
        },
        "colors": {
            "final_selected": "green",
            "depth_edge_topk": "yellow",
            "robot_geo_topk": "red (legacy: hybrid/final-score top-k)",
            "overlap": "cyan",
            "gripper_projection": "blue",
        },
    }
    with open(step_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return str(step_dir)


def _make_legend() -> Image.Image:
    """Create a small legend image explaining the color coding."""
    img = Image.new("RGB", (520, 160), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    entries = [
        ((10, 10, 30, 30), (0, 255, 80), "Final selected tokens"),
        ((10, 45, 30, 65), (255, 255, 0), "Depth-edge top-k"),
        ((10, 80, 30, 100), (255, 0, 0), "Hybrid/final-score top-k (legacy: robot_geo_topk)"),
        ((10, 115, 30, 135), (0, 255, 255), "Overlap (both)"),
        ((300, 10, 320, 30), (0, 120, 255), "Gripper projection"),
    ]
    for (x0, y0, x1, y1), color, label in entries:
        draw.rectangle((x0, y0, x1, y1), fill=color, outline=(0, 0, 0))
        draw.text((x1 + 8, y0), label, fill=(0, 0, 0))
    return img


def save_token_selection_debug_with_dropped(
    *,
    output_dir: str,
    method: str,
    episode_id: int,
    step_id: int,
    rgb: Optional[np.ndarray],
    token_u: Optional[np.ndarray],
    token_v: Optional[np.ndarray],
    keep_indices: Sequence[int],
    depth_edge_topk_indices: Optional[Sequence[int]] = None,
    depth_edge_dropped_indices: Optional[Sequence[int]] = None,
    robot_geo_topk_indices: Optional[Sequence[int]] = None,
    robot_geo_dropped_indices: Optional[Sequence[int]] = None,
    reserved_edge_indices: Optional[Sequence[int]] = None,
    non_reserved_edge_dropped_indices: Optional[Sequence[int]] = None,
    # P11.3: Branch attribution sets
    depth_branch_indices: Optional[Sequence[int]] = None,
    hybrid_branch_indices: Optional[Sequence[int]] = None,
    fill_branch_indices: Optional[Sequence[int]] = None,
    gripper_pixel: Optional[np.ndarray] = None,
    gripper_neighborhood_r: float = 20.0,
    token_grid_shape: Tuple[int, int] = (16, 16),
    selection_meta: Optional[Dict[str, Any]] = None,
    reconstruction_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """P8.3 + P11.3: Enhanced token selection overlay showing kept AND dropped categories
    plus P11 branch attribution.

    This is visualization-only. It does NOT change selection logic, scores, or
    keep_indices. It only visualizes token sets that are already computed and
    stored in diagnostics.

    Color semantics (priority order, highest drawn last):
      - Final kept tokens:               green  (0, 255, 80)
      - Depth edge top-k kept:          yellow (255, 255, 0)
      - Depth edge top-k dropped:       orange (255, 140, 0), ring only
      - Hybrid/final-score top-k kept: red    (255, 0, 0)
      - Hybrid/final-score top-k dropped: magenta (255, 0, 255), ring only
      - Reserved edge kept:             cyan outline (0, 255, 255)
      - Non-reserved edge dropped:      blue outline (0, 100, 255)
      - P11 Depth branch:              cyan-sky (0, 200, 200)
      - P11 Hybrid branch:             orange (255, 160, 0)
      - P11 Fill branch:              purple (180, 100, 255)
      - Gripper point:               blue dot (0, 120, 255)
      - Gripper neighborhood:        blue circle, transparent fill
    """
    out = Path(output_dir)
    step_dir = (
        out
        / "token_selection_debug_dropped"
        / str(method)
        / f"episode_{int(episode_id):04d}"
        / f"step_{int(step_id):04d}"
    )
    step_dir.mkdir(parents=True, exist_ok=True)

    grid_h, grid_w = token_grid_shape
    n_grid = grid_h * grid_w

    if rgb is not None:
        rgb_img = _to_rgb_image(rgb)
    else:
        rgb_img = Image.new("RGB", (640, 480), color=(40, 40, 40))

    idx = np.asarray(list(keep_indices), dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < n_grid)]
    keep_set = set(int(i) for i in idx.tolist())

    all_indices = set(range(n_grid))
    dropped_set = all_indices - keep_set

    # Build per-category sets
    de_topk = set(int(i) for i in depth_edge_topk_indices) if depth_edge_topk_indices is not None else set()
    de_dropped = set(int(i) for i in depth_edge_dropped_indices) if depth_edge_dropped_indices is not None else set()
    rg_topk = set(int(i) for i in robot_geo_topk_indices) if robot_geo_topk_indices is not None else set()
    rg_dropped = set(int(i) for i in robot_geo_dropped_indices) if robot_geo_dropped_indices is not None else set()
    res_edge = set(int(i) for i in reserved_edge_indices) if reserved_edge_indices is not None else set()
    nres_edge_drop = set(int(i) for i in non_reserved_edge_dropped_indices) if non_reserved_edge_dropped_indices is not None else set()

    # P11.3 branch attribution sets. Non-branch-budget methods pass None/empty
    # and should record null/0-safe metadata without breaking visualization.
    depth_branch_set = set(int(i) for i in depth_branch_indices) if depth_branch_indices is not None else set()
    hybrid_branch_set = set(int(i) for i in hybrid_branch_indices) if hybrid_branch_indices is not None else set()
    fill_branch_set = set(int(i) for i in fill_branch_indices) if fill_branch_indices is not None else set()
    depth_branch_set = {i for i in depth_branch_set if 0 <= i < n_grid}
    hybrid_branch_set = {i for i in hybrid_branch_set if 0 <= i < n_grid}
    fill_branch_set = {i for i in fill_branch_set if 0 <= i < n_grid}

    # Compute actual intersection sets
    de_topk_dropped = de_topk - keep_set
    rg_topk_dropped = rg_topk - keep_set

    overlay = rgb_img.copy()
    overlay = overlay.convert("RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")

    if token_u is not None and token_v is not None:
        u_arr = np.asarray(token_u, dtype=np.float32).reshape(-1)
        v_arr = np.asarray(token_v, dtype=np.float32).reshape(-1)

        # Priority: non-res-edge-drop > de-topk-dropped > rg-topk-dropped > de-topk-kept > rg-topk-kept > kept > dropped
        for i in range(min(n_grid, u_arr.size, v_arr.size)):
            x, y = float(u_arr[i]), float(v_arr[i])
            color: Tuple[int, int, int, int]
            fill: Optional[Tuple[int, int, int, int]]

            if i in nres_edge_drop:
                color = (0, 100, 255, 255)
                fill = None
                r = 4
            elif i in de_topk_dropped:
                color = (255, 140, 0, 255)
                fill = (255, 140, 0, 0)
                r = 4
            elif i in rg_topk_dropped:
                color = (255, 0, 255, 255)
                fill = (255, 0, 255, 0)
                r = 4
            elif i in (de_topk & keep_set):
                color = (255, 255, 0, 220)
                fill = (255, 255, 0, 220)
                r = 3
            elif i in (rg_topk & keep_set):
                color = (255, 0, 0, 220)
                fill = (255, 0, 0, 220)
                r = 3
            elif i in depth_branch_set:
                color = (0, 200, 200, 230)
                fill = (0, 200, 200, 210)
                r = 3
            elif i in hybrid_branch_set:
                color = (255, 160, 0, 230)
                fill = (255, 160, 0, 210)
                r = 3
            elif i in fill_branch_set:
                color = (180, 100, 255, 230)
                fill = (180, 100, 255, 200)
                r = 3
            elif i in keep_set:
                color = (0, 255, 80, 220)
                fill = (0, 255, 80, 220)
                r = 3
            else:
                color = (160, 160, 160, 100)
                fill = (160, 160, 160, 60)
                r = 1

            if fill is not None:
                draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=color)
            else:
                draw.ellipse((x - r, y - r, x + r, y + r), outline=color)

    # Draw gripper neighborhood and point
    if gripper_pixel is not None:
        gx, gy = float(gripper_pixel[0]), float(gripper_pixel[1])
        nr = int(gripper_neighborhood_r)
        draw.ellipse((gx - nr, gy - nr, gx + nr, gy + nr),
                     fill=(0, 120, 255, 40), outline=(0, 80, 200, 200))
        draw.ellipse((gx - 4, gy - 4, gx + 4, gy + 4),
                     fill=(0, 120, 255), outline=(0, 60, 200))
        draw.text((gx + 8, gy - 10), "gripper", fill=(0, 120, 255))

    overlay_resized = overlay.resize((640, 480), Image.Resampling.LANCZOS)
    overlay_rgb = overlay_resized.convert("RGB")

    filename = f"ep{episode_id:04d}_st{step_id:04d}_{method}_dropped.png"
    try:
        overlay_rgb.save(step_dir / filename)
    except Exception as exc:
        print(f"[VIS] Failed to save overlay PNG: {exc}")
        overlay_resized.save(step_dir / filename)

    # Save enhanced legend
    legend = _make_legend_dropped()
    legend.save(step_dir / "legend_dropped.png")

    # Save metadata
    # P8.3: Build method-aware metadata
    # For depth_edge_fast: hybrid_final_scores == edge_scores, so there is no distinct
    # hybrid/final-score top-k. We mark it null to avoid misleading the human reviewer.
    is_depth_edge_fast = method.startswith("depth_edge_fast")
    is_edge_reserve = "edge_reserve" in method

    # Resolve authoritative counts from selection_meta if available
    sm = selection_meta if selection_meta is not None else {}
    rm = reconstruction_metadata if reconstruction_metadata is not None else {}

    # depth_edge_topk_count: prefer authoritative from selection_meta
    if is_depth_edge_fast:
        de_topk_count = len(de_topk)
        de_topk_dropped_count = len(de_topk_dropped)
        hybrid_final_score_topk_count = None  # not applicable for depth_edge_fast
        hybrid_final_score_topk_dropped_count = None
    else:
        de_topk_count = sm.get("depth_edge_topk_count") if sm else None
        if de_topk_count is None:
            de_topk_count = len(de_topk)
        de_topk_dropped_count = sm.get("depth_edge_topk_dropped_count") if sm else None
        if de_topk_dropped_count is None:
            de_topk_dropped_count = len(de_topk_dropped)
        hybrid_final_score_topk_count = sm.get("robot_geo_topk_count") if sm else None
        if hybrid_final_score_topk_count is None:
            hybrid_final_score_topk_count = len(rg_topk) if rg_topk else None
        hybrid_final_score_topk_dropped_count = sm.get("robot_geo_topk_dropped_count") if sm else None
        if hybrid_final_score_topk_dropped_count is None:
            hybrid_final_score_topk_dropped_count = len(rg_topk_dropped) if rg_topk_dropped else None

    # reserved_edge_count: prefer authoritative from selection_meta
    if is_edge_reserve:
        res_count = sm.get("reserved_edge_topk_count") if sm else None
        if res_count is None:
            res_count = len(res_edge)
        nres_drop_count = sm.get("non_reserved_edge_dropped_count") if sm else None
        if nres_drop_count is None:
            nres_drop_count = len(nres_edge_drop)
    else:
        res_count = 0
        nres_drop_count = 0

    metadata = {
        "method": method,
        "episode_id": int(episode_id),
        "step_id": int(step_id),
        "token_grid_shape": [int(grid_h), int(grid_w)],
        "num_selected_tokens": len(keep_set),
        "num_dropped_tokens": len(dropped_set),
        "depth_edge_topk_count": de_topk_count,
        "depth_edge_topk_dropped_count": de_topk_dropped_count,
        # For depth_edge_fast: null (not applicable). For hybrid/edge_reserve: real count.
        "hybrid_final_score_topk_count": hybrid_final_score_topk_count,
        "hybrid_final_score_topk_dropped_count": hybrid_final_score_topk_dropped_count,
        # Legacy field: kept for backwards compat but now annotated with applicability
        "robot_geo_topk_count": None if is_depth_edge_fast else (hybrid_final_score_topk_count or len(rg_topk)),
        "robot_geo_topk_dropped_count": None if is_depth_edge_fast else (hybrid_final_score_topk_dropped_count or len(rg_topk_dropped)),
        "reserved_edge_count": res_count,
        "non_reserved_edge_dropped_count": nres_drop_count,
        # P11.3: Branch attribution counts (exact match to step_metrics.csv)
        "num_depth_branch_tokens": len(depth_branch_set),
        "num_hybrid_branch_tokens": len(hybrid_branch_set),
        "num_fill_branch_tokens": len(fill_branch_set),
        "depth_branch_indices": sorted(depth_branch_set),
        "hybrid_branch_indices": sorted(hybrid_branch_set),
        "fill_branch_indices": sorted(fill_branch_set),
        # P11.3: DE top-k attribution by branch
        "de_topk_kept_by_depth_branch": len(de_topk & depth_branch_set),
        "de_topk_kept_by_hybrid_branch": len(de_topk & hybrid_branch_set),
        "de_topk_kept_by_fill_branch": len(de_topk & fill_branch_set),
        "de_topk_dropped": len(de_topk - keep_set),
        "colors": {
            "final_kept": "green",
            "final_dropped": "transparent_grey",
            "depth_edge_topk_kept": "yellow",
            "depth_edge_topk_dropped": "orange",
            "hybrid_final_score_topk_kept": "red (legacy: robot_geo_topk_kept)",
            "hybrid_final_score_topk_dropped": "magenta (legacy: robot_geo_topk_dropped)",
            "reserved_edge_kept": "cyan_outline",
            "non_reserved_edge_dropped": "blue_outline",
            "gripper_projection": "blue",
            "gripper_neighborhood": "blue_circle",
        },
        "field_semantics": {
            "hybrid_final_score_topk_count/kept/dropped": (
                "Top-k count by hybrid/final-score signal. "
                "For depth_edge_fast: null (not applicable; hybrid_final_scores == edge_scores). "
                "For hybrid/edge_reserve: distinct weighted combination signal."
            ),
            "robot_geo_topk_count/kept/dropped": (
                "LEGACY name for hybrid_final_score top-k. "
                "DEPRECATED — prefer hybrid_final_score_topk_* fields. "
                "For depth_edge_fast: null (not applicable)."
            ),
            "depth_edge_topk_count/kept/dropped": "Top-k count by depth-edge gradient score (pure edge signal).",
            "reserved_edge_count": "Number of edge-reserve tokens. For non-edge_reserve methods: 0.",
            "non_reserved_edge_dropped_count": "Non-reserve DE tokens that were dropped. For non-edge_reserve: 0.",
        },
        "method_type": "depth_edge_fast" if is_depth_edge_fast else ("edge_reserve" if is_edge_reserve else "hybrid"),
        "is_depth_edge_fast": is_depth_edge_fast,
        "is_edge_reserve": is_edge_reserve,
        "reconstruction_metadata": rm,
    }
    with open(step_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return str(step_dir)


def _make_legend_dropped() -> Image.Image:
    """Create a legend for the enhanced dropped-token visualization."""
    img = Image.new("RGB", (640, 220), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    entries = [
        ((10, 10, 30, 30), (0, 255, 80), "Final kept tokens"),
        ((10, 40, 30, 60), (160, 160, 160), "Final dropped tokens"),
        ((10, 70, 30, 90), (255, 255, 0), "DE top-k kept"),
        ((10, 100, 30, 120), (255, 140, 0), "DE top-k dropped (orange ring)"),
        ((10, 130, 30, 150), (255, 0, 0), "Hybrid/final-score top-k kept (legacy: robot_geo_topk)"),
        ((10, 160, 30, 180), (255, 0, 255), "Hybrid/final-score top-k dropped (magenta ring)"),
        ((320, 10, 340, 30), (0, 100, 255), "Non-res edge dropped (blue outline)"),
        ((320, 40, 340, 60), (0, 255, 255), "Reserved edge kept (cyan outline)"),
        ((320, 70, 340, 90), (0, 120, 255), "Gripper point"),
        ((320, 100, 340, 120), (0, 120, 255), "Gripper neighborhood"),
    ]
    for (x0, y0, x1, y1), color, label in entries:
        draw.rectangle((x0, y0, x1, y1), fill=color, outline=(0, 0, 0))
        draw.text((x1 + 8, y0), label, fill=(0, 0, 0))
    return img
