"""Read-only diagnostic: real materialized LLM attention vs QK-proxy vs geometry.

Gap-1 verification. The internal backend currently selects visual tokens using a
QK-proxy text->vision relevance (because the model runs FlashAttention, which does
not materialize attention probabilities). The ACGTP design constitution (layer 2)
calls for the TRUE LLM attention. This probe quantifies how much the proxy differs
from the real materialized attention, on a single LIBERO observation, WITHOUT
touching any core pruning logic.

What it does, all in-process and read-only w.r.t. repo files:
  1. Load base OpenVLA, grab one observation, build geometry payload.
  2. Run prefill up to prune_layer once.
  3. At prune_layer, compute three text->vision importance vectors over the 256
     visual tokens:
       (a) QK-proxy        — pruning.internal_pruning._qk_text_to_visual_attention
       (b) REAL attention  — temporarily swap that layer's self_attn to an eager
                             LlamaAttention (copying weights) and read output_attentions
       (c) geometry-only   — the geo_score the backend would use
  4. Compare: top-k IoU between (a)/(b), (b)/(c), (a)/(c); rank correlation;
     and how each ranks the explicit geo_protect tokens (success-proxy).

No core file is modified. The eager swap is on a deepcopy of the layer, scoped to
this process.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def _topk_set(vec: np.ndarray, k: int) -> set:
    if vec is None or k <= 0:
        return set()
    k = min(k, int(vec.shape[0]))
    return set(np.argsort(-vec)[:k].tolist())


def _iou(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or a.shape != b.shape or a.size < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    denom = (np.sqrt((ra ** 2).sum()) * np.sqrt((rb ** 2).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="task_0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max_wait", type=int, default=10)
    parser.add_argument("--prune_layer", type=int, default=2)
    parser.add_argument("--keep_ratio", type=float, default=0.50)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    os.chdir("/infini-data/openvla")
    import scripts.eval_openvla_baseline as ev
    from geometry import GeometryDataRecorder
    from pruning.hook import VisualTokenPruningHook
    from pruning import internal_pruning as ip
    from transformers.models.llama.modeling_llama import LlamaAttention

    out_dir = Path(args.output_dir or "/infini-data/openvla/outputs/attention_proxy_gap")
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
    # ACGTP internal geo_guarded config (matches the swept main surface).
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

    print("[gap] loading OpenVLA")
    model, processor = ev.load_model_and_processor(base_cfg)
    model.eval()
    print(f"[gap] loaded {type(model).__name__}")

    env = ev.LIBEROEnvAdapter(
        task_suite_name="libero_spatial", resolution=256,
        num_steps_wait=args.max_wait, enable_depth=True,
        camera_name="agentview", geometry_debug=False,
    )
    results: Dict[str, Any] = {"task": args.task, "seed": args.seed, "prune_layer": args.prune_layer, "keep_ratio": args.keep_ratio}
    try:
        env.reset(args.task, args.seed, trial_idx=0)
        for _ in range(args.max_wait):
            env.step([0, 0, 0, 0, 0, 0, -1])
        observation = env.get_observation()
        task_description = env.get_task_description()

        recorder = GeometryDataRecorder(enabled=True, debug=False)
        recorder.reset(episode_id=0, task_name=args.task)
        # collect geometry for this frame (same call the probe uses)
        from scripts.probe_pruning_compute_reality import _collect_geometry
        _collect_geometry(ev, env, recorder, observation, step_id=args.max_wait)

        hook = VisualTokenPruningHook(cfg=cfg, geometry_recorder=recorder, visualizer=None)
        assert hook.attach_to_model(model), "hook attach failed"

        backbone = model.language_model.model
        target_layer = backbone.layers[int(args.prune_layer)]
        backend = getattr(backbone, "_acgtp_internal_backend", None)

        # Capture inputs_embeds + the hidden_states entering prune_layer, plus the
        # geo payload the backend builds. We do this by running predict_action once
        # with output capture taps. The internal backend already runs prefill; we
        # add forward-pre/forward hooks on the prune layer to grab its input and to
        # force a REAL eager attention pass on a cloned layer.
        captured: Dict[str, Any] = {}

        def _capture_layer_input(module, inputs, kwargs):
            # decoder layer forward(hidden_states, attention_mask=..., position_ids=..., ...)
            # Only capture the PREFILL step (seq_len > 1); decode steps have
            # seq_len == 1 and would overwrite the prefill capture.
            hs = inputs[0] if inputs else kwargs.get("hidden_states")
            if hs is None or not torch.is_tensor(hs) or int(hs.shape[1]) <= 1:
                return None
            if "hidden_states" in captured:
                return None  # keep the first prefill capture only
            captured["hidden_states"] = hs.detach()
            captured["attention_mask"] = kwargs.get("attention_mask")
            captured["position_ids"] = kwargs.get("position_ids")
            captured["cache_position"] = kwargs.get("cache_position")
            return None

        # Capture the InternalPruningPlan the backend uses (carries the full
        # geometry payload: scene/depth/contact/motion/action + geo_protect_mask
        # + geo_soft_score) by wrapping resolve_visual_keep_indices in-process.
        # This monkeypatch is scoped to this probe run; no file is modified.
        orig_resolve = backend.resolve_visual_keep_indices if backend is not None else None

        def _wrapped_resolve(plan, **kw):
            captured["geometry_payload"] = dict(getattr(plan, "geometry_payload", {}) or {})
            return orig_resolve(plan, **kw)

        if backend is not None and orig_resolve is not None:
            backend.resolve_visual_keep_indices = _wrapped_resolve  # type: ignore[assignment]

        h = target_layer.register_forward_pre_hook(_capture_layer_input, with_kwargs=True)
        try:
            with torch.no_grad():
                _action, _stats = ev.predict_action(
                    model=model, processor=processor, obs=observation,
                    task_description=task_description, unnorm_key="libero_spatial",
                    cfg=cfg, cuda_timer=ev.CUDATimer(),
                    visual_token_counter=ev.VisualTokenCounter(),
                    pruning_metrics_hook=None, geometry_pruning_hook=hook,
                )
        finally:
            h.remove()
            if backend is not None and orig_resolve is not None:
                backend.resolve_visual_keep_indices = orig_resolve  # type: ignore[assignment]

        info = ip.get_acgtp_internal_pruning_info(model) or {}
        image_start = int(info.get("image_token_start_index", 1))
        image_len = int(info.get("original_visual_tokens", 256))
        results["image_start"] = image_start
        results["image_len"] = image_len
        results["internal_attention_source_runtime"] = info.get("internal_attention_source")

        hs = captured.get("hidden_states")
        if hs is None:
            raise RuntimeError("failed to capture prune-layer hidden states")
        hs = hs.to(dtype=torch.float32) if hs.dtype != torch.float32 else hs
        seq_len = int(hs.shape[1])
        results["seq_len_at_prune_layer"] = seq_len

        # (a) QK-proxy — the exact function the backend uses
        qk_vec = ip._qk_text_to_visual_attention(
            target_layer, hs.to(next(target_layer.parameters()).dtype),
            image_start=image_start, image_len=image_len,
        )
        qk = qk_vec.detach().cpu().numpy() if qk_vec is not None else None

        # (b) REAL materialized attention — clone the layer's self_attn as eager
        real_vec = None
        try:
            attn_cfg = copy.deepcopy(target_layer.self_attn.config)
            attn_cfg._attn_implementation = "eager"
            eager = LlamaAttention(attn_cfg, layer_idx=int(args.prune_layer)).to(
                device=hs.device, dtype=next(target_layer.parameters()).dtype
            )
            eager.load_state_dict(target_layer.self_attn.state_dict(), strict=False)
            eager.eval()
            am = captured.get("attention_mask")
            pos = captured.get("position_ids")
            with torch.no_grad():
                attn_out = eager(
                    hidden_states=hs.to(next(target_layer.parameters()).dtype),
                    attention_mask=am,
                    position_ids=pos,
                    output_attentions=True,
                    use_cache=False,
                )
            attn_w = attn_out[1] if isinstance(attn_out, (tuple, list)) and len(attn_out) > 1 else None
            real_v, real_avail, real_conf = ip._text_to_visual_attention(
                attn_w, image_start=image_start, image_len=image_len, seq_len=seq_len, device=hs.device,
            )
            results["real_attention_available"] = bool(real_avail)
            results["real_attention_confidence"] = float(real_conf)
            if real_avail:
                real_vec = real_v.detach().cpu().numpy()
        except Exception as e:  # noqa: BLE001
            results["real_attention_error"] = repr(e)

        # (c) geometry-only score + explicit geo_protect mask, from the captured
        # plan payload (exactly what the backend used this step).
        payload = captured.get("geometry_payload", {}) or {}
        geo = None
        protect = None
        if isinstance(payload, dict):
            gp = payload.get("geo_soft_score")
            if gp is not None:
                geo = np.asarray(gp, dtype=np.float64).reshape(-1)[:image_len]
            pm = payload.get("geo_protect_mask")
            if pm is not None:
                protect = np.asarray(pm).astype(bool).reshape(-1)[:image_len]
        # If geo_soft_score is absent, reconstruct the backend's geo_score from the
        # per-branch scores (max of action/scene/depth/contact/motion), matching
        # internal_pruning.resolve_visual_keep_indices.
        if geo is None and isinstance(payload, dict):
            def _g(key):
                v = payload.get(key)
                return np.asarray(v, dtype=np.float64).reshape(-1)[:image_len] if v is not None else None
            comps = [_g(k) for k in ("action_constraint_scores", "scene_scores", "depth_scores", "contact_scores", "motion_scores")]
            comps = [c for c in comps if c is not None and c.shape[0] == image_len]
            if comps:
                geo = np.maximum.reduce(comps)

        k = max(1, int(round(image_len * float(args.keep_ratio))))
        results["topk_k"] = k

        sets = {}
        if qk is not None: sets["qk_proxy"] = _topk_set(qk, k)
        if real_vec is not None: sets["real_attn"] = _topk_set(real_vec, k)
        if geo is not None: sets["geometry"] = _topk_set(geo, k)

        results["topk_iou"] = {}
        keys = list(sets.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                results["topk_iou"][f"{keys[i]}__vs__{keys[j]}"] = round(_iou(sets[keys[i]], sets[keys[j]]), 4)

        results["spearman"] = {}
        if qk is not None and real_vec is not None:
            results["spearman"]["qk_proxy__vs__real_attn"] = round(_spearman(qk, real_vec), 4)
        if geo is not None and real_vec is not None:
            results["spearman"]["geometry__vs__real_attn"] = round(_spearman(geo, real_vec), 4)
        if geo is not None and qk is not None:
            results["spearman"]["geometry__vs__qk_proxy"] = round(_spearman(geo, qk), 4)

        # Success-proxy: do the proxy / real attention rank the geo_protect tokens
        # highly? (Low = the signal would fight the geometry hard-protect.)
        if protect is not None and protect.any():
            pidx = set(np.flatnonzero(protect).tolist())
            results["geo_protect_count"] = int(protect.sum())
            cov = {}
            for name, vec in (("qk_proxy", qk), ("real_attn", real_vec), ("geometry", geo)):
                if vec is None:
                    continue
                tk = _topk_set(vec, k)
                cov[name] = round(len(tk & pidx) / max(1, len(pidx)), 4)
            results["geo_protect_topk_coverage"] = cov

        (out_dir / "attention_proxy_gap.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print("[gap] RESULT")
        print(json.dumps(results, indent=2))
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
