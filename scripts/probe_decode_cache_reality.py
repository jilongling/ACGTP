#!/usr/bin/env python3
"""Decisive verification: does internal pruning actually shrink the KV cache
that decode reads, and is decode the dominant cost?

Loads OpenVLA, runs ONE predict_action with internal geo_guarded pruning, and
instruments per-layer KV cache lengths after prefill + the context length seen
by each decode step. Read-only w.r.t. the model.
"""
import os, time
os.chdir("/infini-data/openvla")
import numpy as np
import torch

import scripts.eval_openvla_baseline as ev
from geometry import GeometryDataRecorder
from pruning.hook import VisualTokenPruningHook
from pruning.internal_pruning import get_acgtp_internal_pruning_info

CKPT = "/infini-data/checkpoints/openvla-7b-finetuned-libero-spatial"


def per_layer_cache_lengths(past):
    out = []
    if past is None:
        return out
    if hasattr(past, "key_cache"):
        for k in past.key_cache:
            out.append(int(k.shape[-2]) if hasattr(k, "shape") else None)
        return out
    if isinstance(past, (tuple, list)):
        for layer in past:
            if isinstance(layer, (tuple, list)) and layer and hasattr(layer[0], "shape"):
                out.append(int(layer[0].shape[-2]))
    return out


def main():
    base_cfg = {
        "model_path": CKPT,
        "task_suite": "libero_spatial",
        "resolution": 256,
        "num_steps_wait": 10,
        "seed": 7,
        "device": "cuda",
        "precision": "bfloat16",
        "use_flash_attention": True,
        "center_crop": True,
        "vision_patch_size": 14,
        "geometry_camera_name": "agentview",
    }
    print("[probe] loading model")
    model, processor = ev.load_model_and_processor(base_cfg)
    model.eval()
    env = ev.LIBEROEnvAdapter(
        task_suite_name="libero_spatial", resolution=256, num_steps_wait=10,
        enable_depth=True, camera_name="agentview", geometry_debug=False,
    )
    env.reset("task_0", 7, trial_idx=0)
    for _ in range(10):
        env.step([0, 0, 0, 0, 0, 0, -1])
    observation = env.get_observation()
    task_description = env.get_task_description()
    raw = env.get_geometry_raw_obs() if hasattr(env, "get_geometry_raw_obs") else {}
    ci, ce, ee = env.get_camera_intrinsics(), env.get_camera_extrinsics(), env.get_ee_pose()
    for k, v in (("camera_intrinsics", ci), ("camera_extrinsics", ce), ("ee_pose", ee)):
        if v is not None:
            raw.setdefault(k, v)

    def run(label, cfg):
        recorder = None
        if cfg.get("geometry_enabled", False):
            recorder = GeometryDataRecorder(enabled=True, debug=False)
            recorder.reset(episode_id=0, task_name="task_0")
            recorder.collect_step(rgb=observation.get("full_image"), obs=observation,
                                  action=None, step_id=0, raw_env_obs=raw, current_ee_pose=ee)
        # instrument language_model forward to log call type + cache lengths
        lm = None
        for name, mod in model.named_modules():
            if name == "language_model" or name.endswith(".language_model"):
                lm = mod; break
        calls = []

        def pre(_, args, kwargs):
            ie = kwargs.get("inputs_embeds"); ii = kwargs.get("input_ids")
            pkv = kwargs.get("past_key_values")
            seq = int(ie.shape[1]) if ie is not None else (int(ii.shape[-1]) if ii is not None else None)
            ct = "prefill" if (seq and seq > 1) else "decode"
            calls.append({"type": ct, "seq": seq, "in_cache": per_layer_cache_lengths(pkv)})

        h = lm.register_forward_pre_hook(pre, with_kwargs=True)
        hook = None
        if cfg.get("pruning_enabled") or cfg.get("geometry_enabled"):
            hook = VisualTokenPruningHook(cfg=cfg, geometry_recorder=recorder, visualizer=None)
            hook.attach_to_model(model)
        try:
            t0 = time.perf_counter()
            ev.predict_action(model=model, processor=processor, obs=observation,
                              task_description=task_description, unnorm_key="libero_spatial", cfg=cfg,
                              geometry_pruning_hook=hook)
            dt = (time.perf_counter() - t0) * 1000.0
        finally:
            h.remove()
            if hook is not None:
                hook.detach()
        info = get_acgtp_internal_pruning_info(model) or {}
        print(f"\n===== {label} =====  total_predict_ms={dt:.1f}")
        print(f"  internal applied={info.get('applied')} prune_layer={info.get('pruning_layer')} "
              f"kept_seq={info.get('kept_seq_length')} orig_visual={info.get('original_visual_tokens')} "
              f"kept_visual={info.get('kept_visual_tokens')} prefill_cache_seq={info.get('prefill_cache_seq_length')}")
        prefills = [c for c in calls if c["type"] == "prefill"]
        decodes = [c for c in calls if c["type"] == "decode"]
        print(f"  LM calls: prefill={len(prefills)} decode={len(decodes)}")
        if prefills:
            pc = prefills[0]
            print(f"  prefill[0] seq={pc['seq']} in_cache_per_layer(min/max)="
                  f"{(min(pc['in_cache']),max(pc['in_cache'])) if pc['in_cache'] else None}")
        if decodes:
            d0 = decodes[0]
            cl = d0["in_cache"]
            print(f"  decode[0] seq={d0['seq']} in_cache_per_layer="
                  f"{(min(cl),max(cl)) if cl else None} n_layers={len(cl)}")
            if cl and len(set(cl)) > 1:
                print(f"  >>> CACHE LENGTH INCONSISTENT across layers: distinct={sorted(set(cl))}")
            elif cl:
                print(f"  >>> decode reads UNIFORM cache length {cl[0]} for all {len(cl)} layers")
        if hook is not None:
            hook.detach()

    base = dict(base_cfg)
    base.update({"pruning_strategy": "none", "pruning_enabled": False, "geometry_enabled": False,
                 "keep_ratio": 1.0, "log_step_metrics": False, "use_wandb": False})
    run("baseline_none", dict(base))

    gg = dict(base_cfg)
    gg.update({
        "pruning_strategy": "robot_geo_acgtp_v2", "pruning_enabled": True, "geometry_enabled": True,
        "keep_ratio": 0.50, "log_step_metrics": False, "use_wandb": False,
        "acgtp_compression_backend": "internal", "acgtp_internal_pruning_enabled": True,
        "acgtp_internal_prune_layer": 2, "acgtp_internal_selection_mode": "geo_guarded",
        "acgtp_internal_fail_on_backend_error": True, "acgtp_internal_allow_projector_fallback": False,
        "acgtp_dynamic_enabled": False, "acgtp_fast_selector_enabled": True,
    })
    run("internal_geo_guarded@0.50", dict(gg))
    env.close()


if __name__ == "__main__":
    main()

