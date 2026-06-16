#!/usr/bin/env python3
"""Read-only probe for functional-token competition in ACGTP.

The probe captures the internal pruning plan for one LIBERO observation, then
analyzes whether scene-layout, contact/interaction, and motion-corridor tokens
compete under single-score/global-top-k selection. It does not modify model
weights, pruning code, or runtime selection.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


def _norm01(arr: Optional[np.ndarray], valid: np.ndarray) -> np.ndarray:
    if arr is None:
        return np.zeros_like(valid, dtype=np.float64)
    out = np.asarray(arr, dtype=np.float64).reshape(-1)
    if out.shape[0] != valid.shape[0]:
        fixed = np.zeros_like(valid, dtype=np.float64)
        n = min(fixed.shape[0], out.shape[0])
        fixed[:n] = out[:n]
        out = fixed
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    if not bool(valid.any()):
        return np.clip(out, 0.0, 1.0)
    vals = out[valid]
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if hi - lo <= 1e-8:
        res = np.zeros_like(out, dtype=np.float64)
        res[valid] = 1.0 if hi > 1e-8 else 0.0
        return res
    res = (out - lo) / (hi - lo)
    res[~valid] = 0.0
    return np.clip(res, 0.0, 1.0)


def _topk(vec: np.ndarray, k: int, valid: Optional[np.ndarray] = None) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    if valid is None:
        valid = np.ones(vec.shape[0], dtype=bool)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    k = max(0, min(int(k), int(valid.sum())))
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    score = vec.copy()
    score[~valid] = -np.inf
    order = np.argsort(-score, kind="stable")
    return order[:k].astype(np.int64)


def _mask(indices: Iterable[int], n: int) -> np.ndarray:
    out = np.zeros(int(n), dtype=bool)
    for idx in indices:
        i = int(idx)
        if 0 <= i < n:
            out[i] = True
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum()) / float(union)


def _coverage(keep: np.ndarray, ref: np.ndarray) -> float:
    keep = np.asarray(keep, dtype=bool)
    ref = np.asarray(ref, dtype=bool)
    denom = int(ref.sum())
    if denom <= 0:
        return 1.0
    return float(np.logical_and(keep, ref).sum()) / float(denom)


def _owner_counts(indices: Iterable[int], scores: Dict[str, np.ndarray]) -> Dict[str, int]:
    names = list(scores)
    counts = {name: 0 for name in names}
    counts["tie_or_zero"] = 0
    for raw in indices:
        idx = int(raw)
        vals = [(name, float(scores[name][idx])) for name in names]
        best = max(v for _, v in vals) if vals else 0.0
        if best <= 1e-8:
            counts["tie_or_zero"] += 1
            continue
        winners = [name for name, value in vals if abs(value - best) <= 1e-8]
        if len(winners) != 1:
            counts["tie_or_zero"] += 1
        else:
            counts[winners[0]] += 1
    return counts


def _grid_stats(indices: Iterable[int], n: int) -> Dict[str, Any]:
    idx = np.asarray(list(indices), dtype=np.int64)
    if idx.size == 0:
        return {"count": 0}
    side = int(round(np.sqrt(int(n))))
    if side * side != int(n):
        return {"count": int(idx.size)}
    rows = idx // side
    cols = idx % side
    center = (side - 1) / 2.0
    dist = np.sqrt((rows - center) ** 2 + (cols - center) ** 2)
    return {
        "count": int(idx.size),
        "row_mean": round(float(rows.mean()), 4),
        "col_mean": round(float(cols.mean()), 4),
        "center_distance_mean": round(float(dist.mean()), 4),
    }


def _allocate(total: int, ratios: Dict[str, float], available: Dict[str, bool]) -> Dict[str, int]:
    total = max(0, int(total))
    active = {
        name: max(0.0, float(ratio))
        for name, ratio in ratios.items()
        if max(0.0, float(ratio)) > 0.0 and bool(available.get(name, True))
    }
    if total <= 0 or not active:
        return {name: 0 for name in ratios}
    weight_sum = sum(active.values())
    raw = {name: total * weight / weight_sum for name, weight in active.items()}
    out = {name: int(np.floor(value)) for name, value in raw.items()}
    used = sum(out.values())
    for _, name in sorted(((raw[name] - np.floor(raw[name]), name) for name in raw), reverse=True):
        if used >= total:
            break
        out[name] += 1
        used += 1
    for name in ratios:
        out.setdefault(name, 0)
    return out


def _simulate_quota(
    *,
    target_k: int,
    valid: np.ndarray,
    protect: np.ndarray,
    scores: Dict[str, np.ndarray],
    ratios: Dict[str, float],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    n = int(valid.shape[0])
    selected: Dict[int, str] = {}

    def add(indices: Iterable[int], owner: str, limit: Optional[int] = None) -> None:
        added = 0
        for raw in indices:
            idx = int(raw)
            if idx < 0 or idx >= n or idx in selected:
                continue
            selected[idx] = owner
            added += 1
            if limit is not None and added >= int(limit):
                break
            if len(selected) >= target_k:
                break

    protected_idx = np.flatnonzero(protect & valid)
    if protected_idx.size > target_k:
        target_k = int(protected_idx.size)
    add(protected_idx.tolist(), "hard_protect")
    remaining = max(0, int(target_k) - len(selected))
    available = {name: bool(np.any(scores[name] > 0.0)) for name in ratios}
    quotas = _allocate(remaining, ratios, available)

    for name in ("layout", "contact", "motion", "semantic", "action"):
        quota = int(quotas.get(name, 0))
        if quota <= 0 or name not in scores:
            continue
        add(_topk(scores[name], int(valid.sum()), valid).tolist(), name, limit=quota)

    fill_score = scores.get("fill")
    if fill_score is None:
        fill_score = np.maximum.reduce([scores[k] for k in ("layout", "contact", "motion") if k in scores])
    if len(selected) < target_k:
        add(_topk(fill_score, target_k, valid).tolist(), "fill")
    if len(selected) < target_k:
        add(np.flatnonzero(valid).tolist(), "fallback")

    keep = np.array(sorted(selected), dtype=np.int64)
    owner_counts: Dict[str, int] = {}
    for owner in selected.values():
        owner_counts[owner] = owner_counts.get(owner, 0) + 1
    return keep, {
        "target_k": int(target_k),
        "quotas": quotas,
        "owner_counts": owner_counts,
        "protected_count": int(protected_idx.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="task_0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max_wait", type=int, default=10)
    parser.add_argument("--prune_layer", type=int, default=2)
    parser.add_argument("--keep_ratio", type=float, default=0.50)
    parser.add_argument("--output_dir", type=str, default="/infini-data/openvla/outputs/function_competition_probe")
    parser.add_argument("--layout_ratio", type=float, default=0.30)
    parser.add_argument("--contact_ratio", type=float, default=0.20)
    parser.add_argument("--motion_ratio", type=float, default=0.15)
    parser.add_argument("--semantic_ratio", type=float, default=0.12)
    parser.add_argument("--action_ratio", type=float, default=0.08)
    parser.add_argument("--fill_ratio", type=float, default=0.15)
    args = parser.parse_args()

    os.chdir("/infini-data/openvla")
    import scripts.eval_openvla_baseline as ev
    from geometry import GeometryDataRecorder
    from pruning.hook import VisualTokenPruningHook
    from pruning import internal_pruning as ip
    from scripts.probe_pruning_compute_reality import _collect_geometry

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg: Dict[str, Any] = {
        "model_path": "/infini-data/checkpoints/openvla-7b-finetuned-libero-spatial",
        "task_suite": "libero_spatial",
        "resolution": 256,
        "num_steps_wait": args.max_wait,
        "seed": args.seed,
        "device": "cuda",
        "precision": "bfloat16",
        "use_flash_attention": True,
        "center_crop": True,
        "vision_patch_size": 14,
        "geometry_camera_name": "agentview",
    }
    cfg = dict(base_cfg)
    cfg.update({
        "pruning_strategy": "robot_geo_acgtp_v2",
        "pruning_mode": "robot_geo_acgtp_v2",
        "pruning_method": "robot_geo_acgtp_v2",
        "pruning_enabled": True,
        "geometry_enabled": True,
        "keep_ratio": float(args.keep_ratio),
        "acgtp_compression_backend": "internal",
        "acgtp_internal_pruning_enabled": True,
        "acgtp_internal_prune_layer": int(args.prune_layer),
        "acgtp_internal_selection_mode": "geo_guarded",
        "acgtp_internal_attention_enabled": True,
        "acgtp_fast_selector_enabled": True,
        "acgtp_full_diagnostics_enabled": False,
        "acgtp_dynamic_enabled": False,
        "acgtp_history_enabled": False,
        "fallback_strategy": "no_pruning",
    })

    print("[competition] loading OpenVLA")
    model, processor = ev.load_model_and_processor(base_cfg)
    model.eval()
    env = ev.LIBEROEnvAdapter(
        task_suite_name="libero_spatial",
        resolution=256,
        num_steps_wait=args.max_wait,
        enable_depth=True,
        camera_name="agentview",
        geometry_debug=False,
    )

    captured: Dict[str, Any] = {}
    results: Dict[str, Any] = {
        "task": args.task,
        "seed": args.seed,
        "prune_layer": args.prune_layer,
        "keep_ratio": args.keep_ratio,
    }

    try:
        env.reset(args.task, args.seed, trial_idx=0)
        for _ in range(args.max_wait):
            env.step([0, 0, 0, 0, 0, 0, -1])
        observation = env.get_observation()
        task_description = env.get_task_description()

        recorder = GeometryDataRecorder(enabled=True, debug=False)
        recorder.reset(episode_id=0, task_name=args.task)
        _collect_geometry(ev, env, recorder, observation, step_id=args.max_wait)

        hook = VisualTokenPruningHook(cfg=cfg, geometry_recorder=recorder, visualizer=None)
        assert hook.attach_to_model(model), "hook attach failed"

        backbone = model.language_model.model
        backend = getattr(backbone, "_acgtp_internal_backend", None)
        assert backend is not None, "internal backend unavailable"
        orig_resolve = backend.resolve_visual_keep_indices

        def _wrapped_resolve(plan, **kw):
            keep, info = orig_resolve(plan, **kw)
            captured["geometry_payload"] = dict(getattr(plan, "geometry_payload", {}) or {})
            captured["visual_keep_indices"] = keep.detach().cpu().numpy().astype(np.int64)
            captured["fusion_info"] = dict(info or {})
            captured["target_keep_ratio"] = float(getattr(plan, "target_keep_ratio", args.keep_ratio) or args.keep_ratio)
            captured["original_visual_tokens"] = int(getattr(plan, "original_visual_tokens", 256))
            return keep, info

        backend.resolve_visual_keep_indices = _wrapped_resolve  # type: ignore[assignment]
        try:
            with torch.no_grad():
                _action, _stats = ev.predict_action(
                    model=model,
                    processor=processor,
                    obs=observation,
                    task_description=task_description,
                    unnorm_key="libero_spatial",
                    cfg=cfg,
                    cuda_timer=ev.CUDATimer(),
                    visual_token_counter=ev.VisualTokenCounter(),
                    pruning_metrics_hook=None,
                    geometry_pruning_hook=hook,
                )
        finally:
            backend.resolve_visual_keep_indices = orig_resolve  # type: ignore[assignment]

        info = ip.get_acgtp_internal_pruning_info(model) or {}
        payload = captured.get("geometry_payload", {}) or {}
        n = int(captured.get("original_visual_tokens", info.get("original_visual_tokens", 256)))
        valid = np.asarray(payload.get("valid_mask", np.ones(n, dtype=bool)), dtype=bool).reshape(-1)[:n]
        if valid.shape[0] != n:
            valid = np.ones(n, dtype=bool)

        scene = _norm01(payload.get("scene_scores"), valid)
        depth = _norm01(payload.get("depth_scores"), valid)
        contact = _norm01(payload.get("contact_scores"), valid)
        action_constraint = _norm01(payload.get("action_constraint_scores"), valid)
        motion = _norm01(payload.get("motion_scores"), valid) if bool(payload.get("motion_corridor_valid", False)) else np.zeros(n)
        geo_soft = _norm01(payload.get("geo_soft_score"), valid)
        protect = np.asarray(payload.get("geo_protect_mask", np.zeros(n, dtype=bool)), dtype=bool).reshape(-1)[:n]
        if protect.shape[0] != n:
            protect = np.zeros(n, dtype=bool)

        explicit_layout = payload.get("layout_score")
        explicit_contact = payload.get("contact_score")
        explicit_motion = payload.get("motion_score")
        layout = _norm01(explicit_layout, valid) if explicit_layout is not None else np.maximum(scene, depth)
        contact_fn = _norm01(explicit_contact, valid) if explicit_contact is not None else np.maximum(contact, action_constraint)
        motion_fn = _norm01(explicit_motion, valid) if explicit_motion is not None else motion
        fill = np.maximum.reduce([layout, contact_fn, motion_fn, geo_soft])
        semantic = np.zeros(n, dtype=np.float64)
        action_attn = np.zeros(n, dtype=np.float64)

        target_k = max(1, min(n, int(round(n * float(args.keep_ratio)))))
        current_keep = np.asarray(captured.get("visual_keep_indices", []), dtype=np.int64).reshape(-1)
        current_mask = _mask(current_keep, n)
        protect_mask = protect & valid
        scores = {
            "layout": layout,
            "contact": contact_fn,
            "motion": motion_fn,
            "semantic": semantic,
            "action": action_attn,
            "fill": fill,
        }

        branch_masks: Dict[str, np.ndarray] = {}
        for name in ("layout", "contact", "motion"):
            branch_masks[name] = _mask(_topk(scores[name], target_k, valid), n)
        global_score = fill
        global_top = _topk(global_score, target_k, valid)
        global_mask = _mask(global_top, n)

        ratios = {
            "layout": float(args.layout_ratio),
            "contact": float(args.contact_ratio),
            "motion": float(args.motion_ratio),
            "semantic": float(args.semantic_ratio),
            "action": float(args.action_ratio),
            "fill": float(args.fill_ratio),
        }
        quota_keep, quota_info = _simulate_quota(
            target_k=target_k,
            valid=valid,
            protect=protect_mask,
            scores=scores,
            ratios=ratios,
        )
        quota_mask = _mask(quota_keep, n)

        results.update({
            "runtime_internal_attention_source": info.get("internal_attention_source"),
            "runtime_attention_available": info.get("internal_attention_available"),
            "branch_score_source": {
                "layout": "payload.layout_score" if explicit_layout is not None else "max(scene_scores, depth_scores)",
                "contact": "payload.contact_score" if explicit_contact is not None else "max(contact_scores, action_constraint_scores)",
                "motion": "payload.motion_score" if explicit_motion is not None else "motion_scores",
            },
            "target_k": int(target_k),
            "current_keep_count": int(current_keep.size),
            "valid_count": int(valid.sum()),
            "geo_protect_count": int(protect_mask.sum()),
            "current_geo_protect_coverage": round(_coverage(current_mask, protect_mask), 4),
            "global_topk_geo_protect_coverage": round(_coverage(global_mask, protect_mask), 4),
            "quota_sim_geo_protect_coverage": round(_coverage(quota_mask, protect_mask), 4),
            "branch_topk_overlap_iou": {
                "layout__contact": round(_iou(branch_masks["layout"], branch_masks["contact"]), 4),
                "layout__motion": round(_iou(branch_masks["layout"], branch_masks["motion"]), 4),
                "contact__motion": round(_iou(branch_masks["contact"], branch_masks["motion"]), 4),
            },
            "current_branch_topk_coverage": {
                name: round(_coverage(current_mask, mask), 4)
                for name, mask in branch_masks.items()
            },
            "global_topk_branch_topk_coverage": {
                name: round(_coverage(global_mask, mask), 4)
                for name, mask in branch_masks.items()
            },
            "quota_sim_branch_topk_coverage": {
                name: round(_coverage(quota_mask, mask), 4)
                for name, mask in branch_masks.items()
            },
            "global_topk_owner_counts": _owner_counts(global_top, {
                "layout": layout,
                "contact": contact_fn,
                "motion": motion_fn,
            }),
            "current_owner_counts": _owner_counts(current_keep, {
                "layout": layout,
                "contact": contact_fn,
                "motion": motion_fn,
            }),
            "quota_simulation": quota_info,
            "grid_stats": {
                "current": _grid_stats(current_keep, n),
                "global_topk": _grid_stats(global_top, n),
                "quota_sim": _grid_stats(quota_keep, n),
                "geo_protect": _grid_stats(np.flatnonzero(protect_mask), n),
            },
            "fusion_info_subset": {
                key: captured.get("fusion_info", {}).get(key)
                for key in (
                    "internal_selected_by_geo_count",
                    "internal_functional_quota_enabled",
                    "internal_quota_layout_k",
                    "internal_quota_contact_k",
                    "internal_quota_motion_k",
                    "internal_quota_fill_k",
                    "internal_selected_by_layout_count",
                    "internal_selected_by_contact_count",
                    "internal_selected_by_motion_count",
                    "internal_selected_by_semantic_attention_count",
                    "internal_selected_by_historical_attention_count",
                    "internal_selected_by_fill_count",
                    "internal_selected_by_fallback_count",
                    "internal_pruned_geo_critical_count",
                    "internal_dynamic_risk_level",
                    "internal_geo_attention_iou",
                )
            },
        })

        out_path = out_dir / "functional_competition_probe.json"
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print("[competition] RESULT")
        print(json.dumps(results, indent=2))
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
