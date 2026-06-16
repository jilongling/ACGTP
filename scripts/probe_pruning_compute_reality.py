#!/usr/bin/env python3
"""Probe whether projector-level pruning actually reduces model compute.

This is not a success-rate evaluation. It reuses one LIBERO observation and
runs real OpenVLA inference multiple times with simple token-retention modes.
The goal is to separate "token selection quality" from "does the model truly
compute less after receiving fewer visual tokens?".
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def _mean(values: List[float]) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None


def _median(values: List[float]) -> Optional[float]:
    return float(statistics.median(values)) if values else None


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        out = float(value)
        return out if np.isfinite(out) else None
    except Exception:
        return None


def _shape(x: Any) -> Optional[List[int]]:
    return list(x.shape) if hasattr(x, "shape") else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return _shape(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _find_module(model: Any, module_name: str) -> Any:
    for name, module in model.named_modules():
        if name == module_name or name.endswith("." + module_name):
            return module
    return None


def _sync_if_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _attach_lm_call_timing_hooks(model: Any, calls: List[Dict[str, Any]]):
    language_model = _find_module(model, "language_model")
    if language_model is None:
        return []

    stack: List[Dict[str, Any]] = []

    def pre_hook(_, args, kwargs):
        inputs_embeds = kwargs.get("inputs_embeds")
        input_ids = kwargs.get("input_ids")
        position_ids = kwargs.get("position_ids")
        attention_mask = kwargs.get("attention_mask")
        inputs_shape = _shape(inputs_embeds)
        input_ids_shape = _shape(input_ids)
        if inputs_shape is not None:
            seq_len = int(inputs_shape[1]) if len(inputs_shape) > 1 else None
            call_type = "prefill" if seq_len and seq_len > 1 else "decode_inputs_embeds"
        elif input_ids_shape is not None:
            seq_len = int(input_ids_shape[-1]) if input_ids_shape else None
            call_type = "decode"
        else:
            seq_len = None
            call_type = "unknown"
        _sync_if_cuda()
        rec = {
            "call_index": len(calls),
            "call_type": call_type,
            "seq_len": seq_len,
            "inputs_embeds": inputs_shape,
            "input_ids": input_ids_shape,
            "position_ids": _shape(position_ids),
            "attention_mask": _shape(attention_mask),
            "_start": time.perf_counter(),
        }
        stack.append(rec)
        calls.append(rec)

    def post_hook(_, __, output):
        _sync_if_cuda()
        rec = stack.pop() if stack else (calls[-1] if calls else {})
        rec["elapsed_ms"] = (time.perf_counter() - float(rec.get("_start", time.perf_counter()))) * 1000.0
        rec.pop("_start", None)

    try:
        pre_handle = language_model.register_forward_pre_hook(pre_hook, with_kwargs=True)
    except TypeError:
        return []
    post_handle = language_model.register_forward_hook(post_hook)
    return [pre_handle, post_handle]


def _collect_geometry(ev: Any, env: Any, recorder: Any, observation: Dict[str, Any], step_id: int) -> None:
    raw_env_obs = env.get_geometry_raw_obs() if hasattr(env, "get_geometry_raw_obs") else {}
    camera_intrinsics = env.get_camera_intrinsics()
    camera_extrinsics = env.get_camera_extrinsics()
    ee_pose = env.get_ee_pose()
    if camera_intrinsics is not None:
        raw_env_obs.setdefault("camera_intrinsics", camera_intrinsics)
    if camera_extrinsics is not None:
        raw_env_obs.setdefault("camera_extrinsics", camera_extrinsics)
    if ee_pose is not None:
        raw_env_obs.setdefault("ee_pose", ee_pose)
    recorder.collect_step(
        rgb=observation.get("full_image"),
        obs=observation,
        action=None,
        step_id=step_id,
        raw_env_obs=raw_env_obs,
        current_ee_pose=ee_pose,
    )


def _summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    fields = [
        "num_llm_visual_tokens",
        "num_visual_tokens_kept",
        "num_visual_tokens_original",
        "position_preserve_prefill_seq_len",
        "position_preserve_original_seq_len",
        "lm_prefill_seq_len_observed",
        "lm_prefill_time_ms_observed",
        "lm_decode_time_ms_observed",
        "lm_decode_calls_observed",
        "lm_total_call_time_ms_observed",
        "internal_uniform_original_seq_length",
        "internal_uniform_kept_seq_length",
        "internal_uniform_pruned_seq_length",
        "internal_uniform_original_visual_tokens",
        "internal_uniform_kept_visual_tokens",
        "internal_uniform_pruned_visual_tokens",
        "internal_uniform_pruning_layer",
        "internal_original_seq_length",
        "internal_kept_seq_length",
        "internal_pruned_seq_length",
        "internal_original_visual_tokens",
        "internal_kept_visual_tokens",
        "internal_pruned_visual_tokens",
        "internal_pruning_layer",
        "internal_first_short_layer",
        "internal_post_prune_layer_count",
        "internal_post_prune_layer_ratio",
        "internal_kv_cache_token_reduction_ratio",
        "internal_selection_mode",
        "internal_attention_available",
        "internal_geo_attention_iou",
        "internal_attention_dropped_geo_count",
        "internal_pruned_geo_critical_count",
        "internal_geo_protected_count",
        "internal_geo_explicit_protected_count",
        "internal_geo_explicit_protected_kept_count",
        "internal_dynamic_risk",
        "internal_dynamic_keep_ratio",
        "internal_selected_by_geo_count",
        "internal_selected_by_semantic_attention_count",
        "internal_selected_by_historical_attention_count",
        "internal_selected_by_fill_count",
        "internal_selected_by_fallback_count",
        "compression_backend",
        "llm_forward_time_ms",
        "total_model_forward_time_ms",
        "model_forward_ms",
        "cuda_latency_ms",
        "hook_total_ms",
        "hook_total_time_ms",
        "selector_total_ms",
        "selection_ms",
        "total_wall_ms",
    ]
    out: Dict[str, Any] = {"num_samples": len(rows)}
    for field in fields:
        vals = [_as_float(row.get(field)) for row in rows]
        vals = [v for v in vals if v is not None]
        if vals:
            out[f"{field}_mean"] = _mean(vals)
            out[f"{field}_median"] = _median(vals)
            out[f"{field}_first"] = vals[0]
    return out


def _mode_cfg(base_cfg: Dict[str, Any], mode: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(base_cfg)
    strategy = mode["strategy"]
    cfg.update(
        {
            "pruning_strategy": strategy,
            "pruning_mode": strategy,
            "pruning_method": strategy,
            "pruning_enabled": bool(strategy != "none"),
            "geometry_enabled": bool(mode.get("geometry_enabled", False)),
            "keep_ratio": float(mode.get("keep_ratio", 1.0)),
            "save_pruning_vis": False,
            "save_pruning_debug": False,
            "save_token_selection_debug": False,
            "save_geometry_vis": False,
            "enable_geo_debug": False,
            "log_step_metrics": False,
            "use_wandb": False,
            "fallback_strategy": "no_pruning",
            "pruning_fallback_strategy": "no_pruning",
            "acgtp_position_preserve_enabled": True,
            "acgtp_compression_backend": str(mode.get("compression_backend", "projector")),
            "acgtp_internal_pruning_enabled": bool(mode.get("compression_backend") == "internal"),
            "acgtp_internal_prune_layer": int(mode.get("internal_prune_layer", 2)),
            "acgtp_internal_fail_on_backend_error": True,
            "acgtp_internal_allow_projector_fallback": False,
            "acgtp_internal_selection_mode": str(mode.get("internal_selection_mode", "geo_guarded")),
            "acgtp_internal_attention_enabled": bool(mode.get("internal_attention_enabled", True)),
            "acgtp_internal_attention_budget_ratio": float(mode.get("internal_attention_budget_ratio", 0.20)),
            "acgtp_internal_history_budget_ratio": float(mode.get("internal_history_budget_ratio", 0.15)),
            "acgtp_internal_risk_adaptive_enabled": bool(mode.get("internal_risk_adaptive", False)),
            "acgtp_internal_capture_decode_attention": bool(mode.get("capture_decode_attention", False)),
        }
    )
    if strategy == "robot_geo_acgtp_v2":
        cfg.update(
            {
                "geometry_enabled": True,
                "acgtp_fast_selector_enabled": True,
                "acgtp_full_diagnostics_enabled": False,
                "acgtp_dynamic_enabled": bool(mode.get("dynamic", True)),
                "acgtp_history_enabled": bool(mode.get("history", False)),
                "acgtp_v2_semantic_enabled": False,
                "acgtp_v2_semantic_backend": "none",
            }
        )
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--task", type=str, default="task_0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max_wait", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--project_root",
        type=str,
        default=None,
        help="Project root to import eval/pruning modules from. Defaults to this script's parent project.",
    )
    parser.add_argument("--include_acgtp", action="store_true")
    parser.add_argument("--include_internal_acgtp", action="store_true")
    parser.add_argument("--include_internal_uniform", action="store_true")
    parser.add_argument("--internal_prune_layer", type=int, default=2)
    parser.add_argument("--internal_sweep", action="store_true", help="Sweep internal prune layers 1/2/3 and visual keeps 64/96/128.")
    parser.add_argument(
        "--geo_guarded_sweep",
        action="store_true",
        help="Sweep internal geo_guarded retention at rho=1.0/0.65/0.50/0.35 for the prefill-vs-retention curve.",
    )
    parser.add_argument(
        "--geo_guarded_ratios",
        type=str,
        default="1.0,0.65,0.50,0.35",
        help="Comma-separated keep ratios for --geo_guarded_sweep.",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve() if args.project_root else Path(__file__).resolve().parents[1]
    os.chdir(project_root)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    import scripts.eval_openvla_baseline as ev
    from geometry import GeometryDataRecorder
    try:
        from pruning.internal.backend import get_acgtp_internal_pruning_info
    except ImportError:
        from pruning.internal_pruning import get_acgtp_internal_pruning_info
    try:
        from pruning.internal.uniform import (
            disable_internal_uniform_pruning,
            enable_internal_uniform_pruning,
            get_internal_uniform_info,
        )
    except ImportError:
        from pruning.internal_uniform_pruning import (
            disable_internal_uniform_pruning,
            enable_internal_uniform_pruning,
            get_internal_uniform_info,
        )
    from pruning.hook import VisualTokenPruningHook

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir or (project_root / "outputs" / f"pruning_compute_reality_{timestamp}"))
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

    modes: List[Dict[str, Any]] = [
        {"label": "none@1.00", "strategy": "none", "keep_ratio": 1.0},
        {"label": "uniform_grid@0.75", "strategy": "uniform_grid", "keep_ratio": 0.75},
        {"label": "uniform_grid@0.60", "strategy": "uniform_grid", "keep_ratio": 0.60},
        {"label": "uniform_grid@0.50", "strategy": "uniform_grid", "keep_ratio": 0.50},
        {"label": "uniform_grid@0.40", "strategy": "uniform_grid", "keep_ratio": 0.40},
    ]
    if args.include_acgtp:
        modes.extend(
            [
                {
                    "label": "projector_acgtp@0.60",
                    "strategy": "robot_geo_acgtp_v2",
                    "keep_ratio": 0.60,
                    "geometry_enabled": True,
                    "dynamic": False,
                    "history": False,
                    "compression_backend": "projector",
                },
                {
                    "label": "projector_acgtp@0.50",
                    "strategy": "robot_geo_acgtp_v2",
                    "keep_ratio": 0.50,
                    "geometry_enabled": True,
                    "dynamic": False,
                    "history": False,
                    "compression_backend": "projector",
                },
            ]
        )
    if args.include_internal_acgtp:
        if args.internal_sweep:
            for layer in (1, 2, 3):
                for keep_tokens in (64, 96, 128):
                    ratio = keep_tokens / 256.0
                    modes.append(
                        {
                            "label": f"internal_acgtp_geometry_only_L{layer}_K{keep_tokens}",
                            "strategy": "robot_geo_acgtp_v2",
                            "keep_ratio": ratio,
                            "geometry_enabled": True,
                            "dynamic": False,
                            "history": False,
                            "compression_backend": "internal",
                            "internal_prune_layer": layer,
                            "internal_selection_mode": "geometry_only",
                        }
                    )
        else:
            modes.extend(
                [
                    {
                    "label": "internal_acgtp_geometry_only@0.60",
                    "strategy": "robot_geo_acgtp_v2",
                    "keep_ratio": 0.60,
                    "geometry_enabled": True,
                    "dynamic": False,
                    "history": False,
                    "compression_backend": "internal",
                    "internal_prune_layer": args.internal_prune_layer,
                    "internal_selection_mode": "geometry_only",
                },
                {
                    "label": "internal_acgtp_geometry_only@0.50",
                    "strategy": "robot_geo_acgtp_v2",
                    "keep_ratio": 0.50,
                    "geometry_enabled": True,
                    "dynamic": False,
                    "history": False,
                    "compression_backend": "internal",
                    "internal_prune_layer": args.internal_prune_layer,
                    "internal_selection_mode": "geometry_only",
                },
                {
                    "label": "internal_acgtp_geo_guarded@0.50",
                    "strategy": "robot_geo_acgtp_v2",
                    "keep_ratio": 0.50,
                    "geometry_enabled": True,
                    "dynamic": False,
                    "history": False,
                    "compression_backend": "internal",
                    "internal_prune_layer": args.internal_prune_layer,
                    "internal_selection_mode": "geo_guarded",
                },
                {
                    "label": "internal_acgtp_dynamic@0.50",
                    "strategy": "robot_geo_acgtp_v2",
                    "keep_ratio": 0.50,
                    "geometry_enabled": True,
                    "dynamic": False,
                    "history": False,
                    "compression_backend": "internal",
                    "internal_prune_layer": args.internal_prune_layer,
                    "internal_selection_mode": "dynamic",
                    "internal_risk_adaptive": True,
                },
                ]
            )
    if args.include_internal_uniform:
        if args.internal_sweep:
            for layer in (1, 2, 3):
                for keep_tokens in (64, 96, 128):
                    modes.append(
                        {
                            "label": f"internal_uniform_L{layer}_K{keep_tokens}",
                            "strategy": "none",
                            "keep_ratio": 1.0,
                            "internal_uniform": True,
                            "internal_keep_ratio": keep_tokens / 256.0,
                            "internal_keep_tokens": keep_tokens,
                            "internal_prune_layer": layer,
                        }
                    )
        else:
            modes.extend(
                [
                    {
                        "label": "internal_uniform@0.75",
                        "strategy": "none",
                        "keep_ratio": 1.0,
                        "internal_uniform": True,
                        "internal_keep_ratio": 0.75,
                    },
                    {
                        "label": "internal_uniform@0.60",
                        "strategy": "none",
                        "keep_ratio": 1.0,
                        "internal_uniform": True,
                        "internal_keep_ratio": 0.60,
                    },
                    {
                        "label": "internal_uniform@0.50",
                        "strategy": "none",
                        "keep_ratio": 1.0,
                        "internal_uniform": True,
                        "internal_keep_ratio": 0.50,
                    },
                    {
                        "label": "internal_uniform@0.40",
                        "strategy": "none",
                        "keep_ratio": 1.0,
                        "internal_uniform": True,
                        "internal_keep_ratio": 0.40,
                    },
                ]
            )

    if args.geo_guarded_sweep:
        # Prefill-vs-retention curve on the main surface (internal geo_guarded).
        # rho=1.0 is the in-backend reference: pruning is requested but keeps all
        # tokens, so it isolates the backend/hook overhead from token reduction.
        try:
            ratios = [float(x) for x in str(args.geo_guarded_ratios).split(",") if x.strip()]
        except ValueError:
            ratios = [1.0, 0.65, 0.50, 0.35]
        for ratio in ratios:
            modes.append(
                {
                    "label": f"internal_geo_guarded@{ratio:.2f}",
                    "strategy": "robot_geo_acgtp_v2",
                    "keep_ratio": float(ratio),
                    "geometry_enabled": True,
                    "dynamic": False,
                    "history": False,
                    "compression_backend": "internal",
                    "internal_prune_layer": args.internal_prune_layer,
                    "internal_selection_mode": "geo_guarded",
                }
            )

    print("[probe] loading OpenVLA")
    model, processor = ev.load_model_and_processor(base_cfg)
    install_decode_patch = getattr(ev, "_install_action_decode_runtime_patch", None)
    if callable(install_decode_patch):
        installed = bool(install_decode_patch(model))
        print(f"[probe] decode runtime patch installed={installed}")
    else:
        print("[probe] decode runtime patch unavailable; predict_action may not accept decode timing kwargs")
    model.eval()
    print(f"[probe] loaded model={type(model).__name__}")

    env = ev.LIBEROEnvAdapter(
        task_suite_name="libero_spatial",
        resolution=256,
        num_steps_wait=args.max_wait,
        enable_depth=True,
        camera_name="agentview",
        geometry_debug=False,
    )
    try:
        env.reset(args.task, args.seed, trial_idx=0)
        for _ in range(args.max_wait):
            env.step([0, 0, 0, 0, 0, 0, -1])
        observation = env.get_observation()
        task_description = env.get_task_description()

        all_results: Dict[str, Any] = {
            "output_dir": str(out_dir),
            "task": args.task,
            "seed": args.seed,
            "iters": args.iters,
            "warmup": args.warmup,
            "modes": {},
        }
        csv_rows: List[Dict[str, Any]] = []

        for mode in modes:
            label = mode["label"]
            cfg = _mode_cfg(base_cfg, mode)
            print(f"[probe] mode={label}")

            recorder = None
            if cfg.get("geometry_enabled", False):
                recorder = GeometryDataRecorder(enabled=True, debug=False)
                recorder.reset(episode_id=0, task_name=args.task)
                _collect_geometry(ev, env, recorder, observation, step_id=args.max_wait)

            pruning_hook = None
            pruning_metrics_hook = None
            internal_uniform_enabled = False
            visual_token_counter = ev.VisualTokenCounter()
            visual_token_counter.attach_to_model(model)
            cuda_timer = ev.CUDATimer()
            rows: List[Dict[str, Any]] = []

            try:
                if mode.get("internal_uniform", False):
                    internal_uniform_enabled = enable_internal_uniform_pruning(
                        model,
                        keep_ratio=float(mode.get("internal_keep_ratio", 0.5)),
                        prune_layer=int(mode.get("internal_prune_layer", args.internal_prune_layer)),
                        image_token_start_index=1,
                        image_token_length=256,
                    )
                    if not internal_uniform_enabled:
                        raise RuntimeError("internal uniform pruning could not patch language model")
                if cfg.get("pruning_enabled", False) or cfg.get("geometry_enabled", False):
                    pruning_hook = VisualTokenPruningHook(cfg=cfg, geometry_recorder=recorder, visualizer=None)
                    attached = pruning_hook.attach_to_model(model)
                    if not attached:
                        raise RuntimeError("VisualTokenPruningHook could not attach to projector")
                if cfg.get("pruning_enabled", False):
                    pruning_metrics_hook = ev.PruningMetricsHook()
                    pruning_metrics_hook.attach_to_model(model)

                total_calls = int(args.warmup + args.iters)
                for i in range(total_calls):
                    lm_calls: List[Dict[str, Any]] = []
                    lm_handles = _attach_lm_call_timing_hooks(model, lm_calls)
                    wall_start = time.perf_counter()
                    action, step_stats = ev.predict_action(
                        model=model,
                        processor=processor,
                        obs=observation,
                        task_description=task_description,
                        unnorm_key="libero_spatial",
                        cfg=cfg,
                        cuda_timer=cuda_timer,
                        visual_token_counter=visual_token_counter,
                        pruning_metrics_hook=pruning_metrics_hook,
                        geometry_pruning_hook=pruning_hook,
                    )
                    total_wall_ms = (time.perf_counter() - wall_start) * 1000.0
                    for handle in lm_handles:
                        handle.remove()
                    if i < args.warmup:
                        continue

                    multimodal = next((call for call in lm_calls if call.get("inputs_embeds") is not None), {})
                    inputs_shape = multimodal.get("inputs_embeds")
                    prefill_calls = [call for call in lm_calls if call.get("call_type") == "prefill"]
                    decode_calls = [call for call in lm_calls if str(call.get("call_type", "")).startswith("decode")]
                    prefill_ms = sum(float(call.get("elapsed_ms") or 0.0) for call in prefill_calls)
                    decode_ms = sum(float(call.get("elapsed_ms") or 0.0) for call in decode_calls)
                    total_lm_call_ms = sum(float(call.get("elapsed_ms") or 0.0) for call in lm_calls)
                    row = {
                        "label": label,
                        "strategy": cfg.get("pruning_strategy"),
                        "keep_ratio": cfg.get("keep_ratio"),
                        "iter": i - args.warmup,
                        "total_wall_ms": total_wall_ms,
                        "lm_inputs_embeds_shape": json.dumps(inputs_shape),
                        "lm_prefill_seq_len_observed": inputs_shape[1] if inputs_shape and len(inputs_shape) > 1 else None,
                        "lm_call_count": len(lm_calls),
                        "lm_prefill_calls_observed": len(prefill_calls),
                        "lm_decode_calls_observed": len(decode_calls),
                        "lm_prefill_time_ms_observed": prefill_ms,
                        "lm_decode_time_ms_observed": decode_ms,
                        "lm_total_call_time_ms_observed": total_lm_call_ms,
                        "lm_calls": json.dumps(_jsonable(lm_calls)),
                    }
                    row.update({k: _jsonable(v) for k, v in step_stats.items()})
                    row["compression_backend"] = mode.get("compression_backend", cfg.get("acgtp_compression_backend", "projector"))
                    acgtp_internal_info = get_acgtp_internal_pruning_info(model) if not mode.get("internal_uniform", False) else {}
                    if acgtp_internal_info:
                        row.update({k: _jsonable(v) for k, v in acgtp_internal_info.items() if k.startswith("internal_")})
                        row.update(
                            {
                                "compression_backend": "internal",
                                "internal_pruning_requested": True,
                                "internal_pruning_applied": acgtp_internal_info.get("applied"),
                                "internal_pruning_layer": acgtp_internal_info.get("pruning_layer", acgtp_internal_info.get("requested_prune_layer")),
                                "internal_original_seq_length": acgtp_internal_info.get("original_seq_length"),
                                "internal_kept_seq_length": acgtp_internal_info.get("kept_seq_length"),
                                "internal_pruned_seq_length": acgtp_internal_info.get("pruned_seq_length"),
                                "internal_original_visual_tokens": acgtp_internal_info.get("original_visual_tokens"),
                                "internal_kept_visual_tokens": acgtp_internal_info.get("kept_visual_tokens"),
                                "internal_pruned_visual_tokens": acgtp_internal_info.get("pruned_visual_tokens"),
                                "internal_decode_calls": acgtp_internal_info.get("decode_calls"),
                                "internal_decode_cache_consistent": acgtp_internal_info.get("decode_cache_consistent"),
                            }
                        )
                    internal_info = get_internal_uniform_info(model)
                    if internal_info:
                        row.update(
                            {
                                "internal_uniform_enabled": True,
                                "internal_uniform_applied": internal_info.get("applied"),
                                "internal_uniform_pruning_layer": internal_info.get("pruning_layer"),
                                "internal_uniform_original_seq_length": internal_info.get("original_seq_length"),
                                "internal_uniform_kept_seq_length": internal_info.get("kept_seq_length"),
                                "internal_uniform_pruned_seq_length": internal_info.get("pruned_seq_length"),
                                "internal_uniform_original_visual_tokens": internal_info.get("original_visual_tokens"),
                                "internal_uniform_kept_visual_tokens": internal_info.get("kept_visual_tokens"),
                                "internal_uniform_pruned_visual_tokens": internal_info.get("pruned_visual_tokens"),
                                "internal_uniform_keep_ratio": internal_info.get("keep_ratio"),
                            }
                        )
                    rows.append(row)
                    csv_rows.append(row)
            finally:
                if pruning_metrics_hook is not None:
                    pruning_metrics_hook.detach()
                if pruning_hook is not None:
                    pruning_hook.detach()
                if internal_uniform_enabled:
                    disable_internal_uniform_pruning(model)
                visual_token_counter.detach()

            summary = _summarize_rows(rows)
            summary["mode"] = mode
            all_results["modes"][label] = {
                "summary": summary,
                "rows": rows,
            }
            print(
                "[probe]",
                label,
                "visual=",
                round(summary.get("num_llm_visual_tokens_mean", -1), 2)
                if summary.get("num_llm_visual_tokens_mean") is not None
                else "NA",
                "prefill=",
                round(summary.get("position_preserve_prefill_seq_len_mean", -1), 2)
                if summary.get("position_preserve_prefill_seq_len_mean") is not None
                else "NA",
                "llm_ms=",
                round(summary.get("llm_forward_time_ms_mean", -1), 2)
                if summary.get("llm_forward_time_ms_mean") is not None
                else "NA",
                "hook_ms=",
                round(summary.get("hook_total_ms_mean", -1), 2)
                if summary.get("hook_total_ms_mean") is not None
                else "NA",
                "wall_ms=",
                round(summary.get("total_wall_ms_mean", -1), 2)
                if summary.get("total_wall_ms_mean") is not None
                else "NA",
            )

        json_path = out_dir / "pruning_compute_reality.json"
        json_path.write_text(json.dumps(_jsonable(all_results), indent=2), encoding="utf-8")

        fieldnames = sorted({key for row in csv_rows for key in row.keys()})
        csv_path = out_dir / "pruning_compute_reality.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

        report_path = out_dir / "pruning_compute_reality_report.md"
        baseline = all_results["modes"].get("none@1.00", {}).get("summary", {})
        base_llm = baseline.get("llm_forward_time_ms_mean")
        base_wall = baseline.get("total_wall_ms_mean")
        base_cuda = baseline.get("cuda_latency_ms_mean")
        lines = [
            "# Pruning Compute Reality Probe",
            "",
            "This probe reuses one LIBERO observation and measures real OpenVLA inference.",
            "",
            "| Mode | Visual Tokens | Prefill Seq | LLM ms | Hook ms | CUDA ms | Wall ms | LLM Speedup | Wall Speedup |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for label, data in all_results["modes"].items():
            s = data["summary"]
            llm = s.get("llm_forward_time_ms_mean")
            wall = s.get("total_wall_ms_mean")
            cuda = s.get("cuda_latency_ms_mean")
            hook = s.get("hook_total_ms_mean") or s.get("hook_total_time_ms_mean")
            prefill = (
                s.get("position_preserve_prefill_seq_len_mean")
                or s.get("lm_prefill_seq_len_observed_mean")
            )
            llm_speed = base_llm / llm if base_llm and llm else None
            wall_speed = base_wall / wall if base_wall and wall else None
            lines.append(
                "| {label} | {vis:.1f} | {prefill} | {llm} | {hook} | {cuda} | {wall} | {llm_speed} | {wall_speed} |".format(
                    label=label,
                    vis=s.get("num_llm_visual_tokens_mean") or 0.0,
                    prefill=f"{prefill:.1f}" if prefill is not None else "N/A",
                    llm=f"{llm:.2f}" if llm is not None else "N/A",
                    hook=f"{hook:.2f}" if hook is not None else "N/A",
                    cuda=f"{cuda:.2f}" if cuda is not None else "N/A",
                    wall=f"{wall:.2f}" if wall is not None else "N/A",
                    llm_speed=f"{llm_speed:.3f}x" if llm_speed else "N/A",
                    wall_speed=f"{wall_speed:.3f}x" if wall_speed else "N/A",
                )
            )
        lines.extend(
            [
                "",
                "Per-call LLM timing:",
                "",
                "| Mode | Prefill ms | Decode ms | Decode Calls | Total Observed LM ms |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for label, data in all_results["modes"].items():
            s = data["summary"]
            lines.append(
                "| {label} | {prefill_ms} | {decode_ms} | {decode_calls} | {total_ms} |".format(
                    label=label,
                    prefill_ms=(
                        f"{s.get('lm_prefill_time_ms_observed_mean'):.2f}"
                        if s.get("lm_prefill_time_ms_observed_mean") is not None
                        else "N/A"
                    ),
                    decode_ms=(
                        f"{s.get('lm_decode_time_ms_observed_mean'):.2f}"
                        if s.get("lm_decode_time_ms_observed_mean") is not None
                        else "N/A"
                    ),
                    decode_calls=(
                        f"{s.get('lm_decode_calls_observed_mean'):.1f}"
                        if s.get("lm_decode_calls_observed_mean") is not None
                        else "N/A"
                    ),
                    total_ms=(
                        f"{s.get('lm_total_call_time_ms_observed_mean'):.2f}"
                        if s.get("lm_total_call_time_ms_observed_mean") is not None
                        else "N/A"
                    ),
                )
            )
        lines.extend(
            [
                "",
                "Internal pruning:",
                "",
                "| Mode | Backend | ACGTP Visual Kept/Orig | ACGTP Visual Retain | ACGTP Seq Kept/Orig | ACGTP Pruned Seq | Uniform Original Seq | Uniform Kept Seq | Uniform Pruned Seq |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for label, data in all_results["modes"].items():
            s = data["summary"]
            a_vis_orig = s.get("internal_original_visual_tokens_mean")
            a_vis_kept = s.get("internal_kept_visual_tokens_mean")
            a_vis_ret = (a_vis_kept / a_vis_orig) if a_vis_orig and a_vis_kept is not None else None
            lines.append(
                "| {label} | {backend} | {a_vis} | {a_ret} | {a_seq} | {a_pruned} | {u_orig} | {u_kept} | {u_pruned} |".format(
                    label=label,
                    backend=s.get("mode", {}).get("compression_backend", "projector"),
                    a_vis=(
                        f"{a_vis_kept:.1f}/{a_vis_orig:.1f}"
                        if a_vis_orig is not None and a_vis_kept is not None
                        else "N/A"
                    ),
                    a_ret=f"{100.0 * a_vis_ret:.1f}%" if a_vis_ret is not None else "N/A",
                    a_seq=(
                        f"{s.get('internal_kept_seq_length_mean'):.1f}/{s.get('internal_original_seq_length_mean'):.1f}"
                        if s.get("internal_kept_seq_length_mean") is not None and s.get("internal_original_seq_length_mean") is not None
                        else "N/A"
                    ),
                    a_pruned=(
                        f"{s.get('internal_pruned_seq_length_mean'):.1f}"
                        if s.get("internal_pruned_seq_length_mean") is not None
                        else "N/A"
                    ),
                    u_orig=(
                        f"{s.get('internal_uniform_original_seq_length_mean'):.1f}"
                        if s.get("internal_uniform_original_seq_length_mean") is not None
                        else "N/A"
                    ),
                    u_kept=(
                        f"{s.get('internal_uniform_kept_seq_length_mean'):.1f}"
                        if s.get("internal_uniform_kept_seq_length_mean") is not None
                        else "N/A"
                    ),
                    u_pruned=(
                        f"{s.get('internal_uniform_pruned_seq_length_mean'):.1f}"
                        if s.get("internal_uniform_pruned_seq_length_mean") is not None
                        else "N/A"
                    ),
                )
            )
        lines.extend(
            [
                "",
                "Internal sweep speed summary:",
                "",
                "| Mode | Backend | Layer | Visual Retain | CUDA Speedup | Wall Speedup | Hook ms | Selector ms |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for label, data in all_results["modes"].items():
            s = data["summary"]
            mode_cfg = s.get("mode", {})
            cuda = s.get("cuda_latency_ms_mean")
            wall = s.get("total_wall_ms_mean")
            cuda_speed = base_cuda / cuda if base_cuda and cuda else None
            wall_speed = base_wall / wall if base_wall and wall else None
            vis_orig = s.get("internal_original_visual_tokens_mean") or s.get("internal_uniform_original_visual_tokens_mean")
            vis_kept = s.get("internal_kept_visual_tokens_mean") or s.get("internal_uniform_kept_visual_tokens_mean")
            vis_ret = (vis_kept / vis_orig) if vis_orig and vis_kept is not None else None
            layer = mode_cfg.get("internal_prune_layer") or s.get("internal_pruning_layer_mean") or s.get("internal_uniform_pruning_layer_mean")
            lines.append(
                "| {label} | {backend} | {layer} | {vis_ret} | {cuda_speed} | {wall_speed} | {hook} | {selector} |".format(
                    label=label,
                    backend=mode_cfg.get("compression_backend", "internal_uniform" if mode_cfg.get("internal_uniform") else "projector"),
                    layer=layer if layer is not None else "N/A",
                    vis_ret=f"{100.0 * vis_ret:.1f}%" if vis_ret is not None else "N/A",
                    cuda_speed=f"{cuda_speed:.3f}x" if cuda_speed else "N/A",
                    wall_speed=f"{wall_speed:.3f}x" if wall_speed else "N/A",
                    hook=f"{(s.get('hook_total_ms_mean') or s.get('hook_total_time_ms_mean')):.2f}" if (s.get('hook_total_ms_mean') or s.get('hook_total_time_ms_mean')) is not None else "N/A",
                    selector=f"{s.get('selection_ms_mean'):.2f}" if s.get("selection_ms_mean") is not None else "N/A",
                )
            )
        lines.extend(
            [
                "",
                "Interpretation:",
                "",
                "- If visual tokens and prefill sequence shrink but LLM/CUDA/wall barely improve, projector-level pruning is not saving enough real compute.",
                "- If LLM improves but wall does not, hook/selector/preprocess/env overhead is hiding the gain.",
                "- If uniform_grid improves but ACGTP does not, ACGTP selection overhead is too high.",
                "- If even uniform_grid@0.40 does not improve meaningfully, the next engineering target should be model-internal pruning.",
                "",
                f"Raw JSON: `{json_path}`",
                f"Raw CSV: `{csv_path}`",
            ]
        )
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[probe] wrote {report_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
