#!/usr/bin/env python3
"""Build compact, paper-aligned benchmark metrics for OpenVLA pruning runs.

This script is read-only with respect to inference artifacts. It reads each
strategy directory's summary.json / episode_metrics.csv / step_metrics.csv and
writes a compact benchmark layer:

  - <root>/<prefix>_comparison.csv
  - <root>/<prefix>_metrics.json
  - <root>/<prefix>_report.md
  - <method>/benchmark_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


BASELINE_NAMES = {"none", "baseline", "baseline_none", "baseline_none_keep100"}


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def as_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        value_f = float(value)
        return value_f if math.isfinite(value_f) else None
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null", "n/a"}:
        return None
    try:
        value_f = float(text)
    except ValueError:
        return None
    return value_f if math.isfinite(value_f) else None


def positive_timing_ms(value: Any) -> Optional[float]:
    value_f = as_float(value)
    if value_f is None or value_f <= 0.0:
        return None
    return value_f


def as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text in {"none", "nan", "null", "n/a"}:
        return None
    if text in {"true", "1", "yes", "y", "success"}:
        return True
    if text in {"false", "0", "no", "n", "failure"}:
        return False
    return None


def values(rows: Iterable[Dict[str, str]], field: str) -> List[float]:
    out: List[float] = []
    for row in rows:
        value = as_float(row.get(field))
        if value is not None:
            out.append(value)
    return out


def bool_values(rows: Iterable[Dict[str, str]], field: str) -> List[bool]:
    out: List[bool] = []
    for row in rows:
        value = as_bool(row.get(field))
        if value is not None:
            out.append(value)
    return out


def all_true_or_none(rows: Iterable[Dict[str, str]], field: str, fallback: Any = None) -> Optional[bool]:
    vals = bool_values(rows, field)
    if vals:
        return all(vals)
    return as_bool(fallback)


def mean(vals: Iterable[float]) -> Optional[float]:
    vals = list(vals)
    return sum(vals) / len(vals) if vals else None


def percentile(vals: Iterable[float], q: float) -> Optional[float]:
    vals = sorted(vals)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def first_mean(rows: List[Dict[str, str]], fields: Iterable[str], fallback: Any = None) -> Optional[float]:
    for field in fields:
        vals = values(rows, field)
        if vals:
            return mean(vals)
    return as_float(fallback)


def select_primary_latency(
    latency_scope: str,
    *,
    wall_ms: Optional[float],
    model_ms: Optional[float],
    llm_total_ms: Optional[float],
) -> Dict[str, Any]:
    """Pick the paper-facing latency axis without overwriting raw wall-clock."""

    scope = str(latency_scope or "llm_only").strip().lower()
    if scope == "wall":
        return {
            "scope": "wall",
            "value_ms": wall_ms,
            "source": "mean_step_wall_time_ms_step_records",
            "control_frequency_source": "1000 / mean_step_wall_time_ms_step_records",
            "description": "inference-step wall time including model forward, hook, action postprocess, and env.step; excludes settling/no-op steps",
            "includes_env_step": True,
        }
    if scope == "model_forward":
        return {
            "scope": "model_forward",
            "value_ms": model_ms,
            "source": "mean_model_forward_time_ms",
            "control_frequency_source": "1000 / mean_model_forward_time_ms",
            "description": "model forward only; excludes env.step and action postprocess, includes non-LLM model modules and pruning hook when logged inside model forward",
            "includes_env_step": False,
        }
    if scope == "llm_only":
        return {
            "scope": "llm_only",
            "value_ms": llm_total_ms,
            "source": "mean_llm_forward_time_ms",
            "control_frequency_source": "1000 / mean_llm_forward_time_ms",
            "description": "pure language_model forward hook total; excludes vision encoder, projector, pruning hook, action postprocess, and env.step",
            "includes_env_step": False,
        }
    raise SystemExit(f"Unsupported latency_scope={latency_scope!r}; choose wall, model_forward, or llm_only")


def safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return a / b


def _row_float(row: Dict[str, str], field: str) -> Optional[float]:
    return as_float(row.get(field))


def _llm_layer_flops(seq_len: float, hidden: float, intermediate: float) -> float:
    """VLA-Pruner-style transformer-block FLOPs formula.

    C(n) = 4 n d^2 + 2 n^2 d + 2 n d m.
    This is analytic model FLOPs for LLM blocks, not a hardware profiler trace.
    """

    n = float(seq_len)
    d = float(hidden)
    m = float(intermediate)
    return 4.0 * n * d * d + 2.0 * n * n * d + 2.0 * n * d * m


def _llm_decode_layer_flops(query_tokens: float, context_tokens: float, hidden: float, intermediate: float) -> float:
    q = float(query_tokens)
    k = float(context_tokens)
    d = float(hidden)
    m = float(intermediate)
    return 4.0 * q * d * d + 2.0 * k * d + 2.0 * q * d * m


def analytic_llm_flops(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    prefill_vals: List[float] = []
    decode_vals: List[float] = []
    total_vals: List[float] = []
    used_default_config = False
    for row in rows:
        layers = _row_float(row, "llm_num_hidden_layers")
        hidden = _row_float(row, "llm_hidden_size")
        intermediate = _row_float(row, "llm_intermediate_size")
        if layers is None:
            layers = _row_float(row, "internal_total_layers")
        # Compatibility fallback for old OpenVLA-7B logs. New runs should log
        # these values directly from the model config.
        if layers is None or hidden is None or intermediate is None:
            layers = layers or 32.0
            hidden = hidden or 4096.0
            intermediate = intermediate or 11008.0
            used_default_config = True
        prefill_seq = _row_float(row, "llm_prefill_seq_len")
        if prefill_seq is None:
            prefill_seq = _row_float(row, "internal_original_seq_length")
        if prefill_seq is None:
            continue
        kept_seq = _row_float(row, "internal_kept_seq_length") or prefill_seq
        prune_layer = _row_float(row, "internal_pruning_layer")
        internal_applied = str(row.get("internal_pruning_applied", "")).strip().lower() in {"true", "1", "yes"}
        if internal_applied and prune_layer is not None and kept_seq < prefill_seq:
            full_layers = max(0.0, min(float(layers), float(prune_layer) + 1.0))
            pruned_layers = max(0.0, float(layers) - full_layers)
            prefill = full_layers * _llm_layer_flops(prefill_seq, hidden, intermediate)
            prefill += pruned_layers * _llm_layer_flops(kept_seq, hidden, intermediate)
        else:
            prefill = float(layers) * _llm_layer_flops(prefill_seq, hidden, intermediate)
        prefill_vals.append(prefill)

        q_tokens = _row_float(row, "llm_decode_query_tokens")
        ctx_tokens = _row_float(row, "llm_decode_context_tokens")
        decode = None
        if q_tokens is not None and ctx_tokens is not None:
            decode = float(layers) * _llm_decode_layer_flops(q_tokens, ctx_tokens, hidden, intermediate)
            decode_vals.append(decode)
        total_vals.append(prefill + (decode or 0.0))

    return {
        "available": bool(prefill_vals),
        "prefill_flops_per_step": mean(prefill_vals),
        "decode_flops_per_step": mean(decode_vals),
        "total_flops_per_step": mean(total_vals),
        "method": (
            "analytic_llm_transformer_flops_from_logged_seq_lengths_and_model_config"
            if prefill_vals and not used_default_config
            else "analytic_llm_transformer_flops_from_logged_seq_lengths_with_openvla7b_default_config"
            if prefill_vals
            else "unavailable_missing_logged_seq_lengths"
        ),
        "uses_default_openvla7b_config": bool(used_default_config and prefill_vals),
        "scope": "LLM transformer blocks only; excludes vision encoder, projector, hook, action postprocess, env",
    }



def profiler_flops(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    vals = values(rows, "profiler_exact_flops")
    profiled_count = sum(1 for row in rows if as_bool(row.get("profiler_exact_flops_profiled")) is True)
    available_count = sum(1 for row in rows if as_bool(row.get("profiler_exact_flops_available")) is True and as_float(row.get("profiler_exact_flops")) is not None)
    source = next((str(row.get("profiler_exact_flops_source")) for row in rows if row.get("profiler_exact_flops_source")), None)
    scope = next((str(row.get("profiler_exact_flops_scope")) for row in rows if row.get("profiler_exact_flops_scope")), None)
    return {
        "available": bool(vals),
        "flops_per_profiled_step": mean(vals),
        "profiled_step_count": profiled_count,
        "available_step_count": available_count,
        "source": source,
        "scope": scope,
        "unavailable_reason": None if vals else "No torch.profiler FLOP sample is present in step_metrics.csv. Run with --enable_torch_profiler_flops true for a dedicated FLOP smoke run.",
    }


def ratio_from_rows(rows: List[Dict[str, str]], kept_fields: Iterable[str], orig_fields: Iterable[str]) -> Optional[float]:
    ratios: List[float] = []
    for row in rows:
        kept = first_mean([row], kept_fields)
        orig = first_mean([row], orig_fields)
        ratio = safe_div(kept, orig)
        if ratio is not None:
            ratios.append(ratio)
    return mean(ratios)


def counter(rows: List[Dict[str, str]], field: str) -> Dict[str, int]:
    c: Counter[str] = Counter()
    for row in rows:
        value = str(row.get(field, "")).strip()
        if value:
            c[value] += 1
    return dict(c)


def method_dirs(root: Path, selected: Optional[List[str]]) -> List[Path]:
    dirs = [p for p in root.iterdir() if p.is_dir() and (p / "summary.json").exists()]
    if selected:
        wanted = set(selected)
        dirs = [p for p in dirs if p.name in wanted]

    def key(path: Path) -> Tuple[int, str]:
        name = path.name.lower()
        if name in BASELINE_NAMES:
            return (0, name)
        if "functional_quota_static" in name:
            return (1, name)
        if "legacy" in name:
            return (2, name)
        return (3, name)

    return sorted(dirs, key=key)



def episode_pair_key(row: Dict[str, str], strict: bool = False) -> Optional[str]:
    task = row.get("task_name") or f"task_{row.get('task_id', '')}".strip("_")
    if strict:
        strict_key = str(row.get("strict_pairing_key") or "").strip()
        if strict_key:
            return strict_key
        seed = str(row.get("seed") or "").strip()
        init_idx = str(row.get("initial_state_index") or "").strip()
        init_hash = str(row.get("initial_state_hash") or "").strip()
        if task and seed and init_idx and init_hash:
            return f"{task}|seed_{seed}|init_{init_idx}|hash_{init_hash}"
        return None
    trial = row.get("trial_idx") or row.get("trial") or row.get("episode_id") or "0"
    return f"{task}|trial_{trial}"


def compact_episode_pairs(rows: List[Dict[str, str]], strict: bool = False) -> Dict[str, Dict[str, Any]]:
    pairs: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = episode_pair_key(row, strict=strict)
        if not key:
            continue
        pairs[key] = {
            "success": as_bool(row.get("success")),
            "timeout": as_bool(row.get("timeout")),
            "steps": as_float(row.get("num_steps")),
            "wall_ms": as_float(row.get("mean_step_wall_time_ms")),
            "model_ms": as_float(row.get("mean_model_forward_time_ms")),
            "cuda_ms": as_float(row.get("mean_cuda_latency_ms")),
            "llm_ms": as_float(row.get("mean_llm_forward_time_ms")),
            "seed": row.get("seed"),
            "initial_state_index": row.get("initial_state_index"),
            "initial_state_hash": row.get("initial_state_hash"),
        }
    return pairs


def paired_delta(baseline_pairs: Dict[str, Dict[str, Any]], method_pairs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    keys = sorted(set(baseline_pairs) & set(method_pairs))
    out: Dict[str, Any] = {"paired_n": len(keys), "paired_keys": keys}
    for name in ["wall_ms", "model_ms", "cuda_ms", "llm_ms"]:
        base_vals: List[float] = []
        method_vals: List[float] = []
        deltas: List[float] = []
        for key in keys:
            b = baseline_pairs[key].get(name)
            m = method_pairs[key].get(name)
            if b is None or m is None:
                continue
            base_vals.append(float(b))
            method_vals.append(float(m))
            deltas.append(float(b) - float(m))
        short = name.replace("_ms", "")
        out[f"paired_{short}_saved_ms"] = mean(deltas)
        out[f"paired_{short}_speedup"] = safe_div(mean(base_vals), mean(method_vals))
    success_delta_count = 0
    comparable_success = 0
    for key in keys:
        b_ok = baseline_pairs[key].get("success")
        m_ok = method_pairs[key].get("success")
        if b_ok is None or m_ok is None:
            continue
        comparable_success += 1
        success_delta_count += int(bool(m_ok)) - int(bool(b_ok))
    out["paired_success_n"] = comparable_success
    out["paired_success_delta_count"] = success_delta_count
    return out

def success_from_episodes(rows: List[Dict[str, str]], summary: Dict[str, Any]) -> Tuple[int, int, Dict[str, Tuple[int, int]]]:
    successes = 0
    total = 0
    per_task: Dict[str, List[int]] = {}
    for row in rows:
        ok = as_bool(row.get("success"))
        if ok is None:
            continue
        task = row.get("task_name") or f"task_{row.get('task_id', '')}".strip("_")
        total += 1
        successes += int(ok)
        per_task.setdefault(task, [0, 0])
        per_task[task][0] += int(ok)
        per_task[task][1] += 1
    if total == 0:
        successes = int(summary.get("num_successes") or 0)
        total = int(summary.get("num_episodes") or 0)
    return successes, total, {k: (v[0], v[1]) for k, v in per_task.items()}


def fmt_ms(v: Optional[float]) -> str:
    return "N/A" if v is None else f"{v:.2f}"


def fmt_ratio(v: Optional[float]) -> str:
    return "N/A" if v is None else f"{v:.3f}x"


def fmt_pct(v: Optional[float]) -> str:
    return "N/A" if v is None else f"{100.0 * v:.1f}%"


def fmt_int(v: Optional[float]) -> str:
    return "N/A" if v is None else str(int(round(v)))


def compact_method(method_dir: Path, latency_scope: str = "llm_only") -> Dict[str, Any]:
    summary = read_json(method_dir / "summary.json")
    ep_rows = read_csv(method_dir / "episode_metrics.csv")
    step_rows = read_csv(method_dir / "step_metrics.csv")
    succ, total, per_task = success_from_episodes(ep_rows, summary)

    backend = str(summary.get("compression_backend") or "").strip().lower()
    if not backend:
        backend = "internal" if first_mean(step_rows, ["internal_kept_visual_tokens"]) is not None else "projector"

    wall_episode_ms = first_mean(ep_rows, ["mean_step_wall_time_ms"], summary.get("mean_step_wall_time_ms"))
    wall_global_ms = first_mean(
        step_rows,
        ["total_step_wall_time_ms", "end_to_end_step_wall_time_ms"],
        summary.get("mean_step_wall_time_ms_global"),
    )
    # The per-episode wall mean can include settling/no-op steps before model
    # inference. For pruning speedups, use logged inference-step wall records
    # whenever they exist.
    raw_wall_ms = wall_global_ms if wall_global_ms is not None else wall_episode_ms
    model_ms = first_mean(
        step_rows,
        ["total_model_forward_time_ms", "model_forward_ms"],
        summary.get("mean_model_forward_time_ms"),
    )
    cuda_ms = first_mean(step_rows, ["cuda_latency_ms"], summary.get("mean_cuda_latency_ms"))
    llm_total_ms = positive_timing_ms(first_mean(step_rows, ["llm_forward_time_ms"], summary.get("mean_llm_forward_time_ms")))
    primary_latency = select_primary_latency(
        latency_scope,
        wall_ms=raw_wall_ms,
        model_ms=model_ms,
        llm_total_ms=llm_total_ms,
    )
    llm_prefill_ms = positive_timing_ms(
        first_mean(
            step_rows,
            ["llm_prefill_time_ms", "lm_prefill_time_ms_observed", "prefill_time_ms"],
            summary.get("mean_llm_prefill_time_ms") or summary.get("mean_lm_prefill_time_ms_observed"),
        )
    )
    llm_decode_ms = positive_timing_ms(
        first_mean(
            step_rows,
            ["llm_decode_time_ms", "lm_decode_time_ms_observed", "decode_time_ms"],
            summary.get("mean_llm_decode_time_ms") or summary.get("mean_lm_decode_time_ms_observed"),
        )
    )
    hook_vals = values(step_rows, "hook_total_time_ms") or values(step_rows, "hook_total_ms")
    selector_vals = values(step_rows, "selection_ms") or values(step_rows, "pruning_time_ms") or values(step_rows, "topk_pruning_ms")
    action_decode_ms = first_mean(step_rows, ["action_decode_time_ms"], summary.get("mean_action_decode_time_ms"))
    # Older runs wrote 0.0 when action-decode timing was not instrumented.
    if action_decode_ms == 0.0:
        action_decode_ms = None

    internal_kept = first_mean(step_rows, ["internal_kept_visual_tokens"], summary.get("mean_internal_kept_visual_tokens"))
    internal_orig = first_mean(step_rows, ["internal_original_visual_tokens"], summary.get("mean_internal_original_visual_tokens"))
    internal_seq_kept = first_mean(step_rows, ["internal_kept_seq_length"], summary.get("mean_internal_kept_seq_length"))
    internal_seq_orig = first_mean(step_rows, ["internal_original_seq_length"], summary.get("mean_internal_original_seq_length"))
    internal_retention = ratio_from_rows(step_rows, ["internal_kept_visual_tokens"], ["internal_original_visual_tokens"])
    if internal_retention is None:
        internal_retention = safe_div(internal_kept, internal_orig)
    internal_seq_retention = safe_div(internal_seq_kept, internal_seq_orig)

    projector_kept = first_mean(step_rows, ["num_projector_visual_tokens_kept"], summary.get("mean_projector_visual_tokens_kept"))
    projector_orig = first_mean(step_rows, ["num_projector_visual_tokens_original"], summary.get("mean_projector_visual_tokens_original"))
    projector_retention = ratio_from_rows(step_rows, ["num_projector_visual_tokens_kept"], ["num_projector_visual_tokens_original"])
    if projector_retention is None:
        projector_retention = safe_div(projector_kept, projector_orig)

    generic_kept = first_mean(step_rows, ["num_visual_tokens_kept", "num_visual_tokens_kept_total"], summary.get("num_visual_tokens_kept_mean"))
    generic_orig = first_mean(step_rows, ["num_visual_tokens_original", "num_visual_tokens_original_total"], summary.get("num_visual_tokens_original_mean"))
    generic_retention = ratio_from_rows(
        step_rows,
        ["num_visual_tokens_kept", "num_visual_tokens_kept_total"],
        ["num_visual_tokens_original", "num_visual_tokens_original_total"],
    )
    if generic_retention is None:
        generic_retention = safe_div(generic_kept, generic_orig)

    effective_retention = internal_retention if backend == "internal" and internal_retention is not None else generic_retention
    effective_visual_tokens = internal_kept if backend == "internal" and internal_kept is not None else generic_kept
    original_visual_tokens = internal_orig if backend == "internal" and internal_orig is not None else generic_orig

    prune_layer = first_mean(step_rows, ["internal_pruning_layer"], summary.get("mean_internal_pruning_layer"))
    total_layers = first_mean(
        step_rows,
        ["llm_num_hidden_layers", "num_hidden_layers", "internal_total_layers"],
        summary.get("llm_num_hidden_layers"),
    )
    if backend == "internal" and prune_layer is not None:
        total_layers = total_layers or 32.0
        benefit_frac = max(0.0, (total_layers - 1.0 - prune_layer) / total_layers)
    else:
        benefit_frac = 1.0

    if internal_seq_retention is not None:
        attn_ratio_upper = internal_seq_retention ** 2
        linear_ratio_upper = internal_seq_retention
        effective_attn_ratio = benefit_frac * attn_ratio_upper + (1.0 - benefit_frac)
        effective_linear_ratio = benefit_frac * linear_ratio_upper + (1.0 - benefit_frac)
    else:
        attn_ratio_upper = linear_ratio_upper = effective_attn_ratio = effective_linear_ratio = None

    analytic_flops = analytic_llm_flops(step_rows)
    profiler_flops_stats = profiler_flops(step_rows)
    visual_tokens_at_llm_entry = first_mean(step_rows, ["visual_tokens_at_llm_entry"], summary.get("mean_visual_tokens_at_llm_entry"))
    visual_tokens_after_internal_prune = first_mean(
        step_rows,
        ["visual_tokens_after_internal_prune"],
        summary.get("mean_visual_tokens_after_internal_prune"),
    )
    internal_first_short_layer = first_mean(step_rows, ["internal_first_short_layer"], summary.get("mean_internal_first_short_layer"))
    internal_shortened_layer_count = first_mean(
        step_rows,
        ["internal_shortened_layer_count"],
        summary.get("mean_internal_shortened_layer_count"),
    )
    internal_post_prune_layer_count = first_mean(
        step_rows,
        ["internal_post_prune_layer_count"],
        summary.get("mean_internal_post_prune_layer_count"),
    )
    internal_post_prune_layer_ratio = first_mean(
        step_rows,
        ["internal_post_prune_layer_ratio"],
        summary.get("mean_internal_post_prune_layer_ratio"),
    )
    internal_kv_cache_short_length_layer_count = first_mean(
        step_rows,
        ["internal_kv_cache_short_length_layer_count"],
        summary.get("mean_internal_kv_cache_short_length_layer_count"),
    )
    internal_kv_cache_token_reduction_ratio = first_mean(
        step_rows,
        ["internal_kv_cache_token_reduction_ratio"],
        summary.get("mean_internal_kv_cache_token_reduction_ratio"),
    )
    internal_kv_cache_mean_seq_len = first_mean(
        step_rows,
        ["internal_kv_cache_mean_seq_len"],
        summary.get("mean_internal_kv_cache_mean_seq_len"),
    )
    decode_effective_kv_tokens = first_mean(
        step_rows,
        ["decode_effective_kv_tokens_mean"],
        summary.get("mean_decode_effective_kv_tokens"),
    )

    branch_fields = [
        "internal_quota_hard_k",
        "internal_quota_layout_k",
        "internal_quota_contact_k",
        "internal_quota_motion_k",
        "internal_quota_semantic_attention_k",
        "internal_quota_historical_attention_k",
        "internal_quota_fill_k",
        "internal_selected_by_geo_count",
        "internal_selected_by_layout_count",
        "internal_selected_by_contact_count",
        "internal_selected_by_motion_count",
        "internal_selected_by_semantic_attention_count",
        "internal_selected_by_historical_attention_count",
        "internal_selected_by_fill_count",
        "internal_selected_by_fallback_count",
        "internal_unique_geo_count",
        "internal_unique_layout_count",
        "internal_unique_contact_count",
        "internal_unique_motion_count",
        "internal_unique_semantic_attention_count",
        "internal_unique_historical_attention_count",
        "internal_unique_fill_count",
        "internal_unique_fallback_count",
        "internal_branch_selected_sum",
        "internal_branch_unique_sum",
        "internal_branch_overlap_count",
        "internal_branch_overlap_ratio",
        "internal_branch_unique_ratio",
        "internal_branch_sum_equals_kept",
        "internal_branch_accounting_valid",
        "internal_pruned_geo_critical_count",
        "internal_geo_protected_count",
    ]
    branch = {field: first_mean(step_rows, [field], summary.get("mean_" + field)) for field in branch_fields}
    for field in ("internal_branch_sum_equals_kept", "internal_branch_accounting_valid"):
        branch[field] = all_true_or_none(step_rows, field, summary.get(field) or summary.get("mean_" + field))
    fallback_count = sum(1 for row in step_rows if as_bool(row.get("fallback_used")))
    n_steps = len(step_rows)
    latency_plan_cache_enabled_steps = sum(
        1 for row in step_rows if as_bool(row.get("acgtp_latency_plan_cache_enabled"))
    )
    latency_plan_cache_hit_steps = sum(
        1 for row in step_rows if as_bool(row.get("acgtp_latency_plan_cache_hit"))
    )

    return {
        "label": method_dir.name,
        "method_dir": str(method_dir),
        "quality": {
            "successes": succ,
            "episodes": total,
            "success_rate": safe_div(float(succ), float(total)),
            "timeout_rate": safe_div(float(sum(1 for r in ep_rows if as_bool(r.get("timeout")))), float(total)) if total else None,
            "mean_episode_steps_all": first_mean(ep_rows, ["num_steps"], summary.get("mean_episode_steps_all")),
            "mean_episode_steps_success": mean(
                [
                    as_float(r.get("num_steps"))
                    for r in ep_rows
                    if as_bool(r.get("success")) and as_float(r.get("num_steps")) is not None
                ]
            ),
            "per_task": {
                k: {"successes": v[0], "episodes": v[1], "success_rate": safe_div(float(v[0]), float(v[1]))}
                for k, v in per_task.items()
            },
        },
        "latency_ms": {
            "wall_step_primary": primary_latency["value_ms"],
            "primary_latency": primary_latency["value_ms"],
            "primary_latency_scope": primary_latency["scope"],
            "primary_latency_source": primary_latency["source"],
            "primary_latency_description": primary_latency["description"],
            "primary_latency_includes_env_step": primary_latency["includes_env_step"],
            "primary_control_frequency_source": primary_latency["control_frequency_source"],
            "raw_wall_step_primary": raw_wall_ms,
            "wall_step_episode_avg": wall_episode_ms,
            "wall_step_global": wall_global_ms,
            "model_forward": model_ms,
            "cuda_core": cuda_ms,
            "vision_encoder": first_mean(step_rows, ["vision_encoder_time_ms"], summary.get("mean_vision_encoder_time_ms")),
            "projector": first_mean(step_rows, ["projector_time_ms"], summary.get("mean_projector_time_ms")),
            "llm_total": llm_total_ms,
            "llm_prefill": llm_prefill_ms,
            "llm_decode": llm_decode_ms,
            "action_decode": action_decode_ms,
            "env_step": first_mean(step_rows, ["env_step_time_ms"], None),
            "hook_total": mean(hook_vals),
            "hook_p50": percentile(hook_vals, 0.50),
            "hook_p95": percentile(hook_vals, 0.95),
            "selector": mean(selector_vals),
            "score_compute": first_mean(
                step_rows,
                ["score_compute_ms", "geometry_score_time_ms", "token_scoring_time_ms"],
                summary.get("mean_score_compute_ms"),
            ),
            "token_mapping": first_mean(step_rows, ["token_mapping_time_ms"], summary.get("mean_token_mapping_time_ms")),
            "depth_sample": first_mean(step_rows, ["depth_sample_ms", "depth_sampling_ms"], summary.get("mean_depth_sample_ms")),
            "depth_edge_score": first_mean(step_rows, ["depth_edge_score_ms"], summary.get("mean_depth_edge_score_ms")),
            "gather": first_mean(step_rows, ["gather_ms"], summary.get("mean_gather_ms")),
            "latency_plan_cache_lookup": first_mean(step_rows, ["acgtp_latency_plan_cache_lookup_ms"], None),
        },
        "tokens": {
            "compression_backend": backend,
            "original_visual_tokens": original_visual_tokens,
            "effective_visual_tokens": effective_visual_tokens,
            "effective_visual_retention": effective_retention,
            "projector_visual_retention": projector_retention,
            "internal_visual_retention": internal_retention,
            "internal_seq_original": internal_seq_orig,
            "internal_seq_kept": internal_seq_kept,
            "internal_seq_retention": internal_seq_retention,
            "internal_seq_reduction": None if internal_seq_retention is None else 1.0 - internal_seq_retention,
            "visual_tokens_at_llm_entry": visual_tokens_at_llm_entry,
            "visual_tokens_after_internal_prune": visual_tokens_after_internal_prune,
            "internal_first_short_layer": internal_first_short_layer,
            "internal_shortened_layer_count": internal_shortened_layer_count,
            "internal_post_prune_layer_count": internal_post_prune_layer_count,
            "internal_post_prune_layer_ratio": internal_post_prune_layer_ratio,
            "internal_kv_cache_short_length_layer_count": internal_kv_cache_short_length_layer_count,
            "internal_kv_cache_token_reduction_ratio": internal_kv_cache_token_reduction_ratio,
            "internal_kv_cache_mean_seq_len": internal_kv_cache_mean_seq_len,
            "decode_effective_kv_tokens": decode_effective_kv_tokens,
        },
        "flops_estimate": {
            "basis": "sequence_length_proxy_internal_layer_weighted",
            "internal_pruning_layer": prune_layer,
            "llm_total_layers": total_layers,
            "benefiting_layer_fraction": benefit_frac,
            "attention_flop_ratio_upper_bound": attn_ratio_upper,
            "attention_speedup_upper_bound": safe_div(1.0, attn_ratio_upper),
            "linear_flop_ratio_upper_bound": linear_ratio_upper,
            "linear_speedup_upper_bound": safe_div(1.0, linear_ratio_upper),
            "internal_effective_attention_flop_ratio": effective_attn_ratio,
            "internal_effective_attention_speedup": safe_div(1.0, effective_attn_ratio),
            "internal_effective_linear_flop_ratio": effective_linear_ratio,
            "internal_effective_linear_speedup": safe_div(1.0, effective_linear_ratio),
            "analytic_llm_flops_available": analytic_flops["available"],
            "analytic_llm_prefill_flops_per_step": analytic_flops["prefill_flops_per_step"],
            "analytic_llm_decode_flops_per_step": analytic_flops["decode_flops_per_step"],
            "analytic_llm_total_flops_per_step": analytic_flops["total_flops_per_step"],
            "analytic_llm_flops_method": analytic_flops["method"],
            "analytic_llm_flops_scope": analytic_flops["scope"],
            "analytic_llm_flops_uses_default_openvla7b_config": analytic_flops["uses_default_openvla7b_config"],
            "profiler_exact_flops_available": profiler_flops_stats["available"],
            "profiler_exact_flops_per_step": profiler_flops_stats["flops_per_profiled_step"],
            "profiler_exact_flops_profiled_step_count": profiler_flops_stats["profiled_step_count"],
            "profiler_exact_flops_available_step_count": profiler_flops_stats["available_step_count"],
            "profiler_exact_flops_source": profiler_flops_stats["source"],
            "profiler_exact_flops_scope": profiler_flops_stats["scope"],
            "profiler_exact_flops_unavailable_reason": profiler_flops_stats["unavailable_reason"],
        },
        "memory": {
            "mean_gpu_memory_mb": first_mean(ep_rows, ["max_gpu_memory_mb"], summary.get("mean_gpu_memory_mb")),
            "max_gpu_memory_mb": max(values(ep_rows, "max_gpu_memory_mb") or [as_float(summary.get("max_gpu_memory_mb")) or 0.0]),
            "max_gpu_memory_reserved_mb": max(
                values(ep_rows, "max_gpu_memory_reserved_mb")
                or [as_float(summary.get("max_gpu_memory_reserved_mb")) or 0.0]
            ),
        },
        "branch_attribution": branch,
        "diagnostics": {
            "n_step_records": n_steps,
            "prefill_decode_split_available": llm_prefill_ms is not None and llm_decode_ms is not None,
            "branch_attribution_available": any(v is not None for v in branch.values()),
            "fallback_steps": fallback_count,
            "fallback_rate": safe_div(float(fallback_count), float(n_steps)) if n_steps else None,
            "latency_plan_cache_enabled_steps": latency_plan_cache_enabled_steps,
            "latency_plan_cache_hit_steps": latency_plan_cache_hit_steps,
            "latency_plan_cache_hit_rate": safe_div(
                float(latency_plan_cache_hit_steps),
                float(latency_plan_cache_enabled_steps),
            ) if latency_plan_cache_enabled_steps else None,
            "selector_counts": counter(step_rows, "selector_function_name"),
            "fallback_reason_counts": counter(step_rows, "fallback_reason"),
            "episode_pair_metrics": compact_episode_pairs(ep_rows, strict=False),
            "strict_episode_pair_metrics": compact_episode_pairs(ep_rows, strict=True),
            "timing_fields_missing": [
                name for name, value in {
                    "llm_prefill": llm_prefill_ms,
                    "llm_decode": llm_decode_ms,
                    "action_decode": action_decode_ms,
                    "env_step": first_mean(step_rows, ["env_step_time_ms"], None),
                }.items() if value is None
            ],
        },
    }


def add_baseline_deltas(methods: List[Dict[str, Any]], baseline_label: Optional[str]) -> Dict[str, Any]:
    if baseline_label:
        baseline = next((m for m in methods if m["label"] == baseline_label), None)
        if baseline is None:
            raise SystemExit(f"Baseline {baseline_label!r} not found. Available: {', '.join(m['label'] for m in methods)}")
    else:
        baseline = next((m for m in methods if m["label"].lower() in BASELINE_NAMES), methods[0])

    bq = baseline["quality"]
    bl = baseline["latency_ms"]
    for m in methods:
        l = m["latency_ms"]
        q = m["quality"]
        d = {
            "success_delta": None if q["success_rate"] is None or bq["success_rate"] is None else q["success_rate"] - bq["success_rate"],
            "quality_preserved": None,
            "wall_speedup": safe_div(bl["wall_step_primary"], l["wall_step_primary"]),
            "primary_latency_speedup": safe_div(bl["primary_latency"], l["primary_latency"]),
            "wall_step_record_speedup": safe_div(bl["wall_step_global"], l["wall_step_global"]),
            "raw_wall_speedup": safe_div(bl["raw_wall_step_primary"], l["raw_wall_step_primary"]),
            "model_speedup": safe_div(bl["model_forward"], l["model_forward"]),
            "cuda_speedup": safe_div(bl["cuda_core"], l["cuda_core"]),
            "llm_speedup": safe_div(bl["llm_total"], l["llm_total"]),
            "wall_saved_ms": None if bl["wall_step_primary"] is None or l["wall_step_primary"] is None else bl["wall_step_primary"] - l["wall_step_primary"],
            "primary_latency_saved_ms": None if bl["primary_latency"] is None or l["primary_latency"] is None else bl["primary_latency"] - l["primary_latency"],
            "wall_step_record_saved_ms": None if bl["wall_step_global"] is None or l["wall_step_global"] is None else bl["wall_step_global"] - l["wall_step_global"],
            "raw_wall_saved_ms": None if bl["raw_wall_step_primary"] is None or l["raw_wall_step_primary"] is None else bl["raw_wall_step_primary"] - l["raw_wall_step_primary"],
            "model_saved_ms": None if bl["model_forward"] is None or l["model_forward"] is None else bl["model_forward"] - l["model_forward"],
            "llm_saved_ms": None if bl["llm_total"] is None or l["llm_total"] is None else bl["llm_total"] - l["llm_total"],
            "prefill_speedup": safe_div(bl["llm_prefill"], l["llm_prefill"]),
            "decode_speedup": safe_div(bl["llm_decode"], l["llm_decode"]),
        }
        if d["success_delta"] is not None:
            tol = max(0.05, 1.0 / q["episodes"]) if q["episodes"] else 0.05
            d["quality_preserved"] = d["success_delta"] >= -tol
        if d["llm_saved_ms"] is not None and d["llm_saved_ms"] > 0:
            d["hook_pct_of_llm_saved"] = safe_div(l["hook_total"], d["llm_saved_ms"])
            d["selector_pct_of_llm_saved"] = safe_div(l["selector"], d["llm_saved_ms"])
        else:
            d["hook_pct_of_llm_saved"] = None
            d["selector_pct_of_llm_saved"] = None
        d["net_llm_saved_after_hook_ms"] = None if d["llm_saved_ms"] is None else d["llm_saved_ms"] - (l["hook_total"] or 0.0)
        d["prefill_latency_reduction_ms"] = (
            None if bl["llm_prefill"] is None or l["llm_prefill"] is None else bl["llm_prefill"] - l["llm_prefill"]
        )
        d["gpu_memory_reduction_percent_vs_baseline"] = None
        base_mem = baseline.get("memory", {}).get("mean_gpu_memory_mb")
        mem = m.get("memory", {}).get("mean_gpu_memory_mb")
        if base_mem is not None and mem is not None and base_mem > 0:
            d["gpu_memory_reduction_percent_vs_baseline"] = 100.0 * (base_mem - mem) / base_mem
        base_flops = baseline.get("flops_estimate", {}).get("analytic_llm_total_flops_per_step")
        method_flops = m.get("flops_estimate", {}).get("analytic_llm_total_flops_per_step")
        d["analytic_llm_flops_speedup_vs_baseline"] = safe_div(base_flops, method_flops)
        d["analytic_llm_flops_reduction_percent_vs_baseline"] = (
            None if base_flops is None or method_flops is None or base_flops == 0
            else 100.0 * (base_flops - method_flops) / base_flops
        )
        base_prof_flops = baseline.get("flops_estimate", {}).get("profiler_exact_flops_per_step")
        method_prof_flops = m.get("flops_estimate", {}).get("profiler_exact_flops_per_step")
        d["profiler_exact_flops_speedup_vs_baseline"] = safe_div(base_prof_flops, method_prof_flops)
        d["profiler_exact_flops_reduction_percent_vs_baseline"] = (
            None if base_prof_flops is None or method_prof_flops is None or base_prof_flops == 0
            else 100.0 * (base_prof_flops - method_prof_flops) / base_prof_flops
        )
        d["paired"] = paired_delta(
            baseline.get("diagnostics", {}).get("episode_pair_metrics", {}),
            m.get("diagnostics", {}).get("episode_pair_metrics", {}),
        )
        d["strict_paired"] = paired_delta(
            baseline.get("diagnostics", {}).get("strict_episode_pair_metrics", {}),
            m.get("diagnostics", {}).get("strict_episode_pair_metrics", {}),
        )
        for paired_blob in (d["paired"], d["strict_paired"]):
            scope = l.get("primary_latency_scope")
            if scope == "llm_only":
                paired_blob["paired_primary_speedup"] = paired_blob.get("paired_llm_speedup")
                paired_blob["paired_primary_saved_ms"] = paired_blob.get("paired_llm_saved_ms")
            elif scope == "model_forward":
                paired_blob["paired_primary_speedup"] = paired_blob.get("paired_model_speedup")
                paired_blob["paired_primary_saved_ms"] = paired_blob.get("paired_model_saved_ms")
            else:
                paired_blob["paired_primary_speedup"] = paired_blob.get("paired_wall_speedup")
                paired_blob["paired_primary_saved_ms"] = paired_blob.get("paired_wall_saved_ms")
        m["baseline_delta"] = d
        m["verdict"] = verdict(m, baseline)
    return baseline


def verdict(method: Dict[str, Any], baseline: Dict[str, Any]) -> str:
    if method is baseline:
        return "BASELINE"
    d = method["baseline_delta"]
    latency_scope = method.get("latency_ms", {}).get("primary_latency_scope", "wall")
    if d["quality_preserved"] is False:
        return "QUALITY_REGRESSION"
    if d["wall_speedup"] is not None and d["wall_speedup"] >= 1.02:
        return "END_TO_END_SPEEDUP" if latency_scope == "wall" else "PRIMARY_LATENCY_SPEEDUP"
    if d["llm_saved_ms"] is not None and d["llm_saved_ms"] > 0:
        if method["diagnostics"]["prefill_decode_split_available"]:
            return "MECHANISM_TREND_BUT_E2E_UNPROVEN"
        return "LLM_TOTAL_TREND_BUT_PREFILL_SPLIT_MISSING"
    return "NO_OBSERVED_LLM_GAIN"


def add_cross_method_warnings(methods: List[Dict[str, Any]]) -> None:
    groups: Dict[Tuple[str, Optional[int]], List[Dict[str, Any]]] = defaultdict(list)
    for m in methods:
        retention = m["tokens"].get("effective_visual_retention")
        rounded = None if retention is None else int(round(retention * 1000))
        groups[(m["tokens"].get("compression_backend") or "", rounded)].append(m)
    for group in groups.values():
        llm_saved = [m["baseline_delta"].get("llm_saved_ms") for m in group if m["baseline_delta"].get("llm_saved_ms") is not None]
        spread = (max(llm_saved) - min(llm_saved)) if len(llm_saved) >= 2 else None
        for m in group:
            warnings: List[str] = []
            if not m["diagnostics"]["prefill_decode_split_available"]:
                warnings.append("prefill_decode_split_missing")
            latency_scope = m.get("latency_ms", {}).get("primary_latency_scope", "wall")
            if m["baseline_delta"].get("wall_speedup") is not None and m["baseline_delta"]["wall_speedup"] < 1.0:
                warnings.append(f"{latency_scope}_latency_not_accelerated")
            if m["baseline_delta"].get("raw_wall_speedup") is not None and m["baseline_delta"]["raw_wall_speedup"] < 1.0:
                warnings.append("raw_wall_clock_not_accelerated")
            wall_ep = m.get("latency_ms", {}).get("wall_step_episode_avg")
            wall_step = m.get("latency_ms", {}).get("wall_step_global")
            if wall_ep is not None and wall_step is not None and wall_ep > 0:
                rel_gap = abs(wall_step - wall_ep) / wall_ep
                if rel_gap >= 0.05:
                    warnings.append(f"wall_timing_scope_gap_{100.0 * rel_gap:.1f}pct")
            if m["baseline_delta"].get("hook_pct_of_llm_saved") is not None and m["baseline_delta"]["hook_pct_of_llm_saved"] >= 0.5:
                warnings.append("hook_consumes_large_fraction_of_llm_saving")
            if spread is not None and spread >= 5.0:
                warnings.append(f"same_retention_llm_saved_spread_{spread:.2f}ms")
            if m["diagnostics"].get("fallback_rate") not in (None, 0.0):
                warnings.append("fallback_used")
            if (m.get("flops_estimate", {}).get("profiler_exact_flops_profiled_step_count") or 0) > 0:
                warnings.append("torch_profiler_flops_enabled_latency_overhead_possible")
            strict_paired_n = (m.get("baseline_delta", {}).get("strict_paired") or {}).get("paired_n")
            if strict_paired_n is None or strict_paired_n <= 0:
                warnings.append("strict_paired_comparison_missing")
            episodes = m.get("quality", {}).get("episodes") or 0
            if episodes < 3:
                warnings.append("sample_size_lt3_smoke_only")
            paired_n = (m.get("baseline_delta", {}).get("paired") or {}).get("paired_n")
            if paired_n is None or paired_n <= 0:
                warnings.append("paired_comparison_missing")
            elif paired_n < 3:
                warnings.append("paired_n_lt3")
            success_delta = m.get("baseline_delta", {}).get("success_delta")
            if success_delta is not None and success_delta < 0 and episodes:
                approx_drop = int(round(abs(success_delta) * episodes))
                warnings.append(f"success_drop_{approx_drop}_episode")
            if "prefill_decode_split_missing" in warnings:
                confidence = "LOW_NO_PREFILL_DECODE_SPLIT"
            elif episodes < 3 or (paired_n is not None and paired_n < 3):
                confidence = "LOW_SMALL_SAMPLE"
            elif any(w.startswith("same_retention_llm_saved_spread") for w in warnings):
                confidence = "MEDIUM_TIMING_NOISY"
            else:
                confidence = "MEDIUM"
            m["diagnostics"]["measurement_confidence"] = confidence
            m["diagnostics"]["benchmark_warnings"] = warnings


def latency_reduction_percent(saved_ms: Optional[float], baseline_ms: Optional[float]) -> Optional[float]:
    if saved_ms is None or baseline_ms is None or baseline_ms == 0:
        return None
    return 100.0 * saved_ms / baseline_ms


def slowdown_percent(speedup: Optional[float]) -> Optional[float]:
    if speedup is None or speedup >= 1.0 or speedup <= 0.0:
        return 0.0 if speedup is not None and speedup >= 1.0 else None
    return 100.0 * ((1.0 / speedup) - 1.0)


def method_core_summary(m: Dict[str, Any]) -> Dict[str, Any]:
    """Return the compact per-strategy benchmark_summary.json payload.

    Keep this intentionally small and paper-facing. The full raw aggregate still
    lives in summary.json and step_metrics.csv.
    """
    b = m["baseline_delta"]
    t = m["tokens"]
    l = m["latency_ms"]
    q = m["quality"]
    f = m["flops_estimate"]
    mem = m["memory"]
    br = m["branch_attribution"]
    warnings = m["diagnostics"].get("benchmark_warnings", [])
    branch_selected = {
        "hard_geo": br.get("internal_selected_by_geo_count") if br.get("internal_selected_by_geo_count") is not None else br.get("internal_quota_hard_k"),
        "layout": br.get("internal_selected_by_layout_count"),
        "contact": br.get("internal_selected_by_contact_count"),
        "motion": br.get("internal_selected_by_motion_count"),
        "semantic": br.get("internal_selected_by_semantic_attention_count"),
        "action": br.get("internal_selected_by_historical_attention_count"),
        "fill": br.get("internal_selected_by_fill_count"),
    }
    branch_unique = {
        "hard_geo": br.get("internal_unique_geo_count"),
        "layout": br.get("internal_unique_layout_count"),
        "contact": br.get("internal_unique_contact_count"),
        "motion": br.get("internal_unique_motion_count"),
        "semantic": br.get("internal_unique_semantic_attention_count"),
        "action": br.get("internal_unique_historical_attention_count"),
        "fill": br.get("internal_unique_fill_count"),
        "fallback": br.get("internal_unique_fallback_count"),
    }
    branch_accounting_available = any(v is not None for v in branch_selected.values())
    branch_unique_available = any(v is not None for v in branch_unique.values())
    branch_accounting_valid = br.get("internal_branch_accounting_valid")
    if branch_accounting_valid is None and branch_accounting_available:
        branch_accounting_valid = False
    branch_reason = None
    if branch_accounting_available and not branch_unique_available:
        branch_reason = "selected branch counts are logged, but unique/overlap/sum accounting is not logged by selector/hook"
    elif not branch_accounting_available:
        branch_reason = "not logged by selector/hook"

    retention = t["effective_visual_retention"]
    token_pruning_ratio = None if retention is None else 1.0 - retention
    compression_ratio = safe_div(t["original_visual_tokens"], t["effective_visual_tokens"])
    primary_ms = l["wall_step_primary"]
    raw_wall_ms = l.get("raw_wall_step_primary")
    control_hz = safe_div(1000.0, primary_ms)

    return {
        "strategy": m["label"],
        "verdict": m["verdict"],
        "num_episodes": q["episodes"],
        "num_successes": q["successes"],
        "overall_success_rate": q["success_rate"],
        "mean_episode_steps_success_only": q.get("mean_episode_steps_success"),
        "num_visual_tokens_original": t["original_visual_tokens"],
        "num_visual_tokens_kept": t["effective_visual_tokens"],
        "effective_visual_tokens_for_llm": t["effective_visual_tokens"],
        "visual_tokens_at_llm_entry": t.get("visual_tokens_at_llm_entry"),
        "visual_tokens_after_internal_prune": t.get("visual_tokens_after_internal_prune"),
        "internal_first_short_layer": t.get("internal_first_short_layer"),
        "internal_post_prune_layer_count": t.get("internal_post_prune_layer_count"),
        "internal_post_prune_layer_ratio": t.get("internal_post_prune_layer_ratio"),
        "internal_kv_cache_short_length_layer_count": t.get("internal_kv_cache_short_length_layer_count"),
        "internal_kv_cache_token_reduction_ratio": t.get("internal_kv_cache_token_reduction_ratio"),
        "internal_kv_cache_mean_seq_len": t.get("internal_kv_cache_mean_seq_len"),
        "decode_effective_kv_tokens": t.get("decode_effective_kv_tokens"),
        "token_retention_ratio": retention,
        "token_pruning_ratio": token_pruning_ratio,
        "compression_ratio": compression_ratio,
        "mean_llm_forward_time_ms": l["llm_total"],
        "llm_timing_scope": "language_model_forward_hook_total",
        "prefill_decode_split_available": m["diagnostics"]["prefill_decode_split_available"],
        "mean_llm_prefill_time_ms": l["llm_prefill"],
        "mean_llm_decode_time_ms": l["llm_decode"],
        "mean_hook_total_ms": l["hook_total"],
        "mean_selection_ms": l["selector"],
        "selection_included_in_hook_total": bool(l["hook_total"] is not None and l["selector"] is not None),
        "hook_overhead_ratio_of_model_forward": safe_div(l["hook_total"], l["model_forward"]),
        "hook_overhead_ratio_of_wall_step": safe_div(l["hook_total"], raw_wall_ms),
        "hook_overhead_ratio_of_primary_latency": safe_div(l["hook_total"], primary_ms),
        "mean_model_forward_time_ms": l["model_forward"],
        "primary_latency_scope": l["primary_latency_scope"],
        "primary_latency_ms": primary_ms,
        "primary_latency_source": l["primary_latency_source"],
        "primary_latency_description": l["primary_latency_description"],
        "mean_step_wall_time_ms": primary_ms,
        "mean_step_wall_time_ms_raw": raw_wall_ms,
        "mean_step_wall_time_ms_episode_avg": l["wall_step_episode_avg"],
        "mean_step_wall_time_ms_step_records": l["wall_step_global"],
        "mean_step_wall_time_ms_step_records_raw": l["wall_step_global"],
        "wall_timing_scope": f"primary_latency_scope={l['primary_latency_scope']}; raw wall preserved in *_raw fields",
        "control_frequency_hz": control_hz,
        "control_frequency_source": l["primary_control_frequency_source"],
        "control_frequency_scope": l["primary_latency_description"],
        "control_frequency_includes_env_step": bool(l["primary_latency_includes_env_step"]),
        "mean_gpu_memory_mb": mem["mean_gpu_memory_mb"],
        "max_gpu_memory_mb": mem["max_gpu_memory_mb"],
        "token_compute_proxy_reduction_percent": None if token_pruning_ratio is None else 100.0 * token_pruning_ratio,
        "estimated_flop_reduction_percent": b.get("analytic_llm_flops_reduction_percent_vs_baseline"),
        "flops_estimation_method": f.get("analytic_llm_flops_method"),
        "flops_estimation_available": bool(f.get("analytic_llm_flops_available")),
        "flops_estimation_scope": f.get("analytic_llm_flops_scope"),
        "analytic_llm_prefill_flops_per_step": f.get("analytic_llm_prefill_flops_per_step"),
        "analytic_llm_decode_flops_per_step": f.get("analytic_llm_decode_flops_per_step"),
        "analytic_llm_total_flops_per_step": f.get("analytic_llm_total_flops_per_step"),
        "analytic_llm_flops_speedup_vs_baseline": b.get("analytic_llm_flops_speedup_vs_baseline"),
        "analytic_llm_flops_reduction_percent_vs_baseline": b.get("analytic_llm_flops_reduction_percent_vs_baseline"),
        "analytic_llm_flops_uses_default_openvla7b_config": f.get("analytic_llm_flops_uses_default_openvla7b_config"),
        "profiler_exact_flops_available": bool(f.get("profiler_exact_flops_available")),
        "profiler_exact_flops_per_step": f.get("profiler_exact_flops_per_step"),
        "profiler_exact_flops_profiled_step_count": f.get("profiler_exact_flops_profiled_step_count"),
        "profiler_exact_flops_available_step_count": f.get("profiler_exact_flops_available_step_count"),
        "profiler_exact_flops_source": f.get("profiler_exact_flops_source"),
        "profiler_exact_flops_scope": f.get("profiler_exact_flops_scope"),
        "profiler_exact_flops_speedup_vs_baseline": b.get("profiler_exact_flops_speedup_vs_baseline"),
        "profiler_exact_flops_reduction_percent_vs_baseline": b.get("profiler_exact_flops_reduction_percent_vs_baseline"),
        "profiler_exact_flops_unavailable_reason": f.get("profiler_exact_flops_unavailable_reason"),
        "token_compute_proxy_ratio": retention,
        "branch_accounting_available": branch_accounting_available,
        "branch_accounting_valid": branch_accounting_valid,
        "branch_accounting_unavailable_reason": branch_reason,
        "branch_unique_overlap_requires_rerun": not branch_unique_available,
        "branch_unique_overlap_requires_instrumentation": not branch_unique_available,
        "branch_selected_counts": branch_selected,
        "branch_unique_counts": branch_unique,
        "branch_unique_counts_available": branch_unique_available,
        "branch_selected_sum": br.get("internal_branch_selected_sum"),
        "branch_unique_sum": br.get("internal_branch_unique_sum"),
        "branch_overlap_count": br.get("internal_branch_overlap_count"),
        "branch_overlap_ratio": br.get("internal_branch_overlap_ratio"),
        "branch_unique_ratio": br.get("internal_branch_unique_ratio"),
        "branch_sum_equals_kept": br.get("internal_branch_sum_equals_kept"),
        "cache_enabled": False,
        "cache_method_applicable": False,
        "cache_hit_rate": None,
        "cache_reuse_ratio": None,
        "cache_not_applicable_reason": "ACGTP is pruning, not KV/token cache",
        "prefill_decode_requires_rerun": not m["diagnostics"]["prefill_decode_split_available"],
        "prefill_decode_requires_instrumentation": not m["diagnostics"]["prefill_decode_split_available"],
        "prefill_decode_unavailable_reason": (
            None
            if m["diagnostics"]["prefill_decode_split_available"]
            else "model.predict_action currently exposes only language_model total forward timing"
        ),
        "measurement_confidence": m["diagnostics"].get("measurement_confidence"),
        "diagnostic_warnings": warnings,
        "diagnostic_warning_count": len(warnings),
        "baseline_delta": {
            "llm_speedup_vs_baseline": b["llm_speedup"],
            "llm_latency_reduction_ms": b["llm_saved_ms"],
            "net_saved_ms_after_hook": b["net_llm_saved_after_hook_ms"],
            "hook_eats_llm_saving_ratio": b["hook_pct_of_llm_saved"],
            "model_speedup_vs_baseline": b["model_speedup"],
            "model_latency_reduction_ms": b["model_saved_ms"],
            "primary_latency_scope": l["primary_latency_scope"],
            "primary_latency_speedup_vs_baseline": b["primary_latency_speedup"],
            "primary_latency_reduction_ms": b["primary_latency_saved_ms"],
            "wall_speedup_vs_baseline": b["wall_speedup"],
            "wall_latency_reduction_ms": b["wall_saved_ms"],
            "wall_step_record_speedup_vs_baseline": b["wall_step_record_speedup"],
            "wall_step_record_latency_reduction_ms": b["wall_step_record_saved_ms"],
            "raw_wall_speedup_vs_baseline": b["raw_wall_speedup"],
            "raw_wall_latency_reduction_ms": b["raw_wall_saved_ms"],
            "wall_is_faster_than_baseline": None if b["wall_speedup"] is None else b["wall_speedup"] > 1.0,
            "wall_slowdown_percent": slowdown_percent(b["wall_speedup"]),
            "paired_n": (b.get("paired") or {}).get("paired_n"),
            "paired_wall_speedup": (b.get("paired") or {}).get("paired_wall_speedup"),
            "paired_model_speedup": (b.get("paired") or {}).get("paired_model_speedup"),
            "paired_llm_saved_ms": (b.get("paired") or {}).get("paired_llm_saved_ms"),
            "paired_success_delta": (b.get("paired") or {}).get("paired_success_delta_count"),
            "strict_paired_n": (b.get("strict_paired") or {}).get("paired_n"),
            "strict_paired_wall_speedup": (b.get("strict_paired") or {}).get("paired_wall_speedup"),
            "strict_paired_model_speedup": (b.get("strict_paired") or {}).get("paired_model_speedup"),
            "strict_paired_llm_saved_ms": (b.get("strict_paired") or {}).get("paired_llm_saved_ms"),
            "strict_paired_success_delta": (b.get("strict_paired") or {}).get("paired_success_delta_count"),
        },
    }


def write_method_summaries(methods: List[Dict[str, Any]]) -> None:
    for m in methods:
        path = Path(m["method_dir"]) / "benchmark_summary.json"
        path.write_text(json.dumps(method_core_summary(m), indent=2, ensure_ascii=False), encoding="utf-8")


def flatten_row(m: Dict[str, Any]) -> Dict[str, Any]:
    b = m["baseline_delta"]
    t = m["tokens"]
    l = m["latency_ms"]
    q = m["quality"]
    br = m["branch_attribution"]
    summary = method_core_summary(m)
    base_success = q["success_rate"] - b["success_delta"] if b["success_delta"] is not None and q["success_rate"] is not None else None
    success_rate_drop_abs = None if b["success_delta"] is None else max(0.0, -b["success_delta"])
    success_rate_drop_rel = safe_div(success_rate_drop_abs, base_success)
    quality_preservation_ratio = safe_div(q["success_rate"], base_success)
    wall_slowdown = slowdown_percent(b["wall_speedup"])
    raw_wall_slowdown = slowdown_percent(b.get("raw_wall_speedup"))
    control_hz = summary["control_frequency_hz"]
    # baseline control frequency = method frequency / speedup, so speedup follows the selected primary latency scope.
    control_speedup = b["wall_speedup"]
    token_pruning_ratio = summary["token_pruning_ratio"]
    memory_reduction = b.get("gpu_memory_reduction_percent_vs_baseline")
    paired = b.get("paired") or {}
    strict_paired = b.get("strict_paired") or {}
    return {
        "strategy": m["label"],
        "verdict": m["verdict"],
        "num_episodes": q["episodes"],
        "success_count": q["successes"],
        "success_rate": q["success_rate"],
        "quality_preservation_ratio": quality_preservation_ratio,
        "success_rate_drop_abs": success_rate_drop_abs,
        "success_rate_drop_rel": success_rate_drop_rel,
        "quality_preserved": b["quality_preserved"],
        "mean_episode_steps_success_only": q.get("mean_episode_steps_success"),
        "tokens_original": t["original_visual_tokens"],
        "tokens_kept": t["effective_visual_tokens"],
        "tokens_pruned": None if t["original_visual_tokens"] is None or t["effective_visual_tokens"] is None else t["original_visual_tokens"] - t["effective_visual_tokens"],
        "effective_visual_tokens_for_llm": t["effective_visual_tokens"],
        "visual_tokens_at_llm_entry": t.get("visual_tokens_at_llm_entry"),
        "visual_tokens_after_internal_prune": t.get("visual_tokens_after_internal_prune"),
        "internal_first_short_layer": t.get("internal_first_short_layer"),
        "internal_post_prune_layer_count": t.get("internal_post_prune_layer_count"),
        "internal_post_prune_layer_ratio": t.get("internal_post_prune_layer_ratio"),
        "internal_kv_cache_short_length_layer_count": t.get("internal_kv_cache_short_length_layer_count"),
        "internal_kv_cache_token_reduction_ratio": t.get("internal_kv_cache_token_reduction_ratio"),
        "internal_kv_cache_mean_seq_len": t.get("internal_kv_cache_mean_seq_len"),
        "decode_effective_kv_tokens": t.get("decode_effective_kv_tokens"),
        "token_retention_ratio": t["effective_visual_retention"],
        "token_pruning_ratio": token_pruning_ratio,
        "compression_ratio": summary["compression_ratio"],
        "mean_llm_forward_time_ms": l["llm_total"],
        "llm_timing_scope": summary["llm_timing_scope"],
        "llm_speedup_vs_baseline": b["llm_speedup"],
        "llm_latency_reduction_ms": b["llm_saved_ms"],
        "llm_saved_ms_vs_baseline": b["llm_saved_ms"],
        "mean_llm_prefill_time_ms": l["llm_prefill"],
        "mean_llm_decode_time_ms": l["llm_decode"],
        "prefill_speedup_vs_baseline": b["prefill_speedup"],
        "prefill_latency_reduction_ms": b["prefill_latency_reduction_ms"],
        "prefill_decode_split_available": m["diagnostics"]["prefill_decode_split_available"],
        "prefill_decode_requires_instrumentation": summary["prefill_decode_requires_instrumentation"],
        "hook_total_ms": l["hook_total"],
        "selection_ms": l["selector"],
        "selection_included_in_hook_total": summary["selection_included_in_hook_total"],
        "latency_plan_cache_hit_rate": m["diagnostics"].get("latency_plan_cache_hit_rate"),
        "latency_plan_cache_hit_steps": m["diagnostics"].get("latency_plan_cache_hit_steps"),
        "latency_plan_cache_enabled_steps": m["diagnostics"].get("latency_plan_cache_enabled_steps"),
        "latency_plan_cache_lookup_ms": l.get("latency_plan_cache_lookup"),
        "net_saved_ms_after_hook": b["net_llm_saved_after_hook_ms"],
        "hook_eats_llm_saving_ratio": b["hook_pct_of_llm_saved"],
        "hook_overhead_ratio_of_model_forward": summary["hook_overhead_ratio_of_model_forward"],
        "hook_overhead_ratio_of_wall_step": summary["hook_overhead_ratio_of_wall_step"],
        "hook_overhead_ratio_of_primary_latency": summary["hook_overhead_ratio_of_primary_latency"],
        "mean_model_forward_time_ms": l["model_forward"],
        "model_speedup_vs_baseline": b["model_speedup"],
        "model_latency_reduction_ms": b["model_saved_ms"],
        "primary_latency_scope": summary["primary_latency_scope"],
        "primary_latency_ms": summary["primary_latency_ms"],
        "primary_latency_source": summary["primary_latency_source"],
        "primary_latency_speedup_vs_baseline": b["primary_latency_speedup"],
        "primary_latency_reduction_ms": b["primary_latency_saved_ms"],
        "mean_step_wall_time_ms": l["wall_step_primary"],
        "mean_step_wall_time_ms_raw": summary["mean_step_wall_time_ms_raw"],
        "mean_step_wall_time_ms_episode_avg": l["wall_step_episode_avg"],
        "mean_step_wall_time_ms_step_records": l["wall_step_global"],
        "wall_speedup_vs_baseline": b["wall_speedup"],
        "wall_latency_reduction_ms": b["wall_saved_ms"],
        "wall_step_record_speedup_vs_baseline": b["wall_step_record_speedup"],
        "wall_step_record_latency_reduction_ms": b["wall_step_record_saved_ms"],
        "raw_wall_speedup_vs_baseline": b["raw_wall_speedup"],
        "raw_wall_latency_reduction_ms": b["raw_wall_saved_ms"],
        "wall_is_faster_than_baseline": None if b["wall_speedup"] is None else b["wall_speedup"] > 1.0,
        "wall_slowdown_percent": wall_slowdown,
        "raw_wall_slowdown_percent": raw_wall_slowdown,
        "control_frequency_hz": control_hz,
        "control_frequency_source": summary["control_frequency_source"],
        "control_frequency_scope": summary["control_frequency_scope"],
        "control_frequency_includes_env_step": summary["control_frequency_includes_env_step"],
        "control_frequency_speedup_vs_baseline": control_speedup,
        "token_compute_proxy_reduction_percent": summary["token_compute_proxy_reduction_percent"],
        "token_compute_proxy_ratio": summary["token_compute_proxy_ratio"],
        "estimated_flop_reduction_percent": summary["estimated_flop_reduction_percent"],
        "flops_estimation_method": summary["flops_estimation_method"],
        "flops_estimation_available": summary["flops_estimation_available"],
        "flops_estimation_scope": summary["flops_estimation_scope"],
        "profiler_exact_flops_available": summary["profiler_exact_flops_available"],
        "profiler_exact_flops_per_step": summary["profiler_exact_flops_per_step"],
        "profiler_exact_flops_profiled_step_count": summary["profiler_exact_flops_profiled_step_count"],
        "profiler_exact_flops_available_step_count": summary["profiler_exact_flops_available_step_count"],
        "profiler_exact_flops_source": summary["profiler_exact_flops_source"],
        "profiler_exact_flops_scope": summary["profiler_exact_flops_scope"],
        "profiler_exact_flops_speedup_vs_baseline": summary["profiler_exact_flops_speedup_vs_baseline"],
        "profiler_exact_flops_reduction_percent_vs_baseline": summary["profiler_exact_flops_reduction_percent_vs_baseline"],
        "profiler_exact_flops_unavailable_reason": summary["profiler_exact_flops_unavailable_reason"],
        "analytic_llm_total_flops_per_step": summary["analytic_llm_total_flops_per_step"],
        "analytic_llm_flops_speedup_vs_baseline": summary["analytic_llm_flops_speedup_vs_baseline"],
        "analytic_llm_flops_reduction_percent_vs_baseline": summary["analytic_llm_flops_reduction_percent_vs_baseline"],
        "mean_gpu_memory_mb": m["memory"]["mean_gpu_memory_mb"],
        "max_gpu_memory_mb": m["memory"]["max_gpu_memory_mb"],
        "gpu_memory_reduction_percent_vs_baseline": memory_reduction,
        "backend": t["compression_backend"],
        "internal_seq_retention": t["internal_seq_retention"],
        "geo_critical_pruned": br.get("internal_pruned_geo_critical_count"),
        "selected_hard_geo": br.get("internal_selected_by_geo_count") if br.get("internal_selected_by_geo_count") is not None else br.get("internal_quota_hard_k"),
        "selected_layout": br.get("internal_selected_by_layout_count"),
        "selected_contact": br.get("internal_selected_by_contact_count"),
        "selected_motion": br.get("internal_selected_by_motion_count"),
        "selected_semantic": br.get("internal_selected_by_semantic_attention_count"),
        "selected_action": br.get("internal_selected_by_historical_attention_count"),
        "selected_fill": br.get("internal_selected_by_fill_count"),
        "unique_hard_geo": br.get("internal_unique_geo_count"),
        "unique_layout": br.get("internal_unique_layout_count"),
        "unique_contact": br.get("internal_unique_contact_count"),
        "unique_motion": br.get("internal_unique_motion_count"),
        "unique_semantic": br.get("internal_unique_semantic_attention_count"),
        "unique_action": br.get("internal_unique_historical_attention_count"),
        "unique_fill": br.get("internal_unique_fill_count"),
        "branch_overlap_ratio": summary["branch_overlap_ratio"],
        "branch_unique_ratio": summary["branch_unique_ratio"],
        "branch_sum_equals_kept": summary["branch_sum_equals_kept"],
        "branch_accounting_available": summary["branch_accounting_available"],
        "branch_accounting_valid": summary["branch_accounting_valid"],
        "branch_accounting_unavailable_reason": summary["branch_accounting_unavailable_reason"],
        "branch_unique_overlap_requires_instrumentation": summary["branch_unique_overlap_requires_instrumentation"],
        "paired_n": paired.get("paired_n"),
        "paired_key_scope": "task_name+trial_idx",
        "paired_primary_speedup": paired.get("paired_primary_speedup"),
        "paired_primary_saved_ms": paired.get("paired_primary_saved_ms"),
        "paired_wall_speedup": paired.get("paired_wall_speedup"),
        "paired_model_speedup": paired.get("paired_model_speedup"),
        "paired_llm_saved_ms": paired.get("paired_llm_saved_ms"),
        "paired_success_delta": paired.get("paired_success_delta_count"),
        "paired_comparison_valid": (paired.get("paired_n") or 0) >= 3,
        "strict_paired_n": strict_paired.get("paired_n"),
        "strict_paired_key_scope": "task_name+seed+initial_state_index+initial_state_hash",
        "strict_paired_primary_speedup": strict_paired.get("paired_primary_speedup"),
        "strict_paired_primary_saved_ms": strict_paired.get("paired_primary_saved_ms"),
        "strict_paired_wall_speedup": strict_paired.get("paired_wall_speedup"),
        "strict_paired_model_speedup": strict_paired.get("paired_model_speedup"),
        "strict_paired_llm_saved_ms": strict_paired.get("paired_llm_saved_ms"),
        "strict_paired_success_delta": strict_paired.get("paired_success_delta_count"),
        "strict_paired_comparison_valid": (strict_paired.get("paired_n") or 0) >= 3,
        "measurement_confidence": m["diagnostics"].get("measurement_confidence"),
        "diagnostic_warning_count": summary["diagnostic_warning_count"],
        "warnings": ";".join(m["diagnostics"].get("benchmark_warnings", [])),
    }


def write_csv_report(methods: List[Dict[str, Any]], path: Path) -> None:
    rows = [flatten_row(m) for m in methods]
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(methods: List[Dict[str, Any]], baseline: Dict[str, Any], path: Path) -> None:
    lines: List[str] = []
    any_split = any(m["diagnostics"].get("prefill_decode_split_available") for m in methods)
    any_branch_unique = any(
        m["branch_attribution"].get("internal_branch_unique_sum") is not None
        for m in methods
    )
    any_profiler_flops = any(m["flops_estimate"].get("profiler_exact_flops_available") for m in methods)
    primary_scope = methods[0].get("latency_ms", {}).get("primary_latency_scope", "wall") if methods else "wall"
    primary_desc = methods[0].get("latency_ms", {}).get("primary_latency_description", "") if methods else ""
    lines.append("# Compact Benchmark Report\n")
    lines.append(f"Baseline: `{baseline['label']}`\n")
    lines.append(f"Primary latency scope: `{primary_scope}`. {primary_desc}\n")
    lines.append("Raw end-to-end wall-clock is preserved separately in `*_raw` / `Raw Wall` fields.\n")
    lines.append("## Main Takeaways\n")
    for m in methods:
        if m is baseline:
            continue
        d = m["baseline_delta"]
        lines.append(
            f"- **{m['label']}**: {m['verdict']}; success "
            f"{m['quality']['successes']}/{m['quality']['episodes']} "
            f"({fmt_pct(m['quality']['success_rate'])}, delta {fmt_pct(d['success_delta'])}); "
            f"primary latency {fmt_ratio(d['wall_speedup'])}; raw wall {fmt_ratio(d.get('raw_wall_speedup'))}; "
            f"model {fmt_ratio(d['model_speedup'])}; "
            f"LLM saved {fmt_ms(d['llm_saved_ms'])} ms; hook {fmt_ms(m['latency_ms']['hook_total'])} ms; "
            f"plan-cache hit {fmt_pct(m['diagnostics'].get('latency_plan_cache_hit_rate'))}; "
            f"confidence {m['diagnostics'].get('measurement_confidence')}; "
            f"warnings: {', '.join(m['diagnostics'].get('benchmark_warnings', [])) or 'none'}."
        )
    lines.append("")

    main = next((m for m in methods if m["label"] == "functional_quota_static_050"), None)
    if main is not None:
        s = method_core_summary(main)
        d = main["baseline_delta"]
        prefill_delta = d.get("prefill_latency_reduction_ms")
        prefill_available = bool(s["prefill_decode_split_available"])
        if prefill_available:
            prefill_status = "YES" if (prefill_delta or 0.0) > 0.0 else "NO"
            prefill_evidence = f"prefill saved {fmt_ms(prefill_delta)} ms"
            prefill_claim = "measured, smoke only" if (prefill_delta or 0.0) > 0.0 else "measured but no decrease in this run"
        else:
            prefill_status = "UNAVAILABLE"
            prefill_evidence = "split missing"
            prefill_claim = "cannot claim prefill acceleration"
        lines.append("## Mechanism Chain Status\n")
        lines.append("| Link | Current status | Evidence | Paper claim status |")
        lines.append("|---|---|---|---|")
        lines.append(
            f"| visual tokens entering LLM decrease | YES | "
            f"{fmt_int(s['num_visual_tokens_original'])} -> {fmt_int(s['effective_visual_tokens_for_llm'])}, "
            f"retention {fmt_pct(s['token_retention_ratio'])} | usable |"
        )
        lines.append(
            f"| LLM total decreases | {'YES' if (d.get('llm_saved_ms') or 0) > 0 else 'NO'} | "
            f"LLM saved {fmt_ms(d.get('llm_saved_ms'))} ms | trend only |"
        )
        lines.append(
            f"| LLM prefill decreases | {prefill_status} | "
            f"{prefill_evidence}, split available = {s['prefill_decode_split_available']} | {prefill_claim} |"
        )
        lines.append(
            f"| hook overhead is controlled | {'YES' if (s.get('mean_hook_total_ms') or 999) < 1.0 else 'NO'} | "
            f"hook {fmt_ms(s.get('mean_hook_total_ms'))} ms, net LLM-hook {fmt_ms(d.get('net_llm_saved_after_hook_ms'))} ms | bottleneck |"
        )
        lines.append(
            f"| model forward decreases | {'YES' if (d.get('model_saved_ms') or 0) > 0 else 'NO'} | "
            f"model speed {fmt_ratio(d.get('model_speedup'))} | weak evidence |"
        )
        lines.append(
            f"| primary latency decreases | {'YES' if (d.get('wall_speedup') or 0) > 1.0 else 'NO'} | "
            f"{primary_scope} speed {fmt_ratio(d.get('wall_speedup'))}; raw wall speed {fmt_ratio(d.get('raw_wall_speedup'))} | "
            "use raw wall for end-to-end claims |"
        )
        lines.append(
            f"| success preserved | {'YES' if d.get('success_delta', 0) >= 0 else 'RISK'} | "
            f"success {main['quality']['successes']}/{main['quality']['episodes']} | usable in this small run |"
        )
        lines.append("")

    lines.append("## Metric Audit Matrix\n")
    lines.append("| Metric group | Status | Source | Use in paper | Notes |")
    lines.append("|---|---|---|---|---|")
    lines.append("| token compression | REAL_LOGGED / DERIVED_VALID | step_metrics.csv + offline formula | yes | kept/original/retention/compression are computed from logged token counts |")
    lines.append("| LLM total timing | REAL_LOGGED | language_model forward hook in step_metrics.csv | yes, with timing-scope caveat | total language-model forward time, including prefill and decode calls |")
    if any_split:
        lines.append("| LLM prefill/decode | REAL_LOGGED | language_model forward pre/post hooks classified by seq/cache shape | yes, smoke/formal-run dependent | measured timing split is available in these logs |")
    else:
        lines.append("| LLM prefill/decode | UNAVAILABLE_NEEDS_INSTRUMENTATION | reserved columns only | no | columns exist but are empty; report keeps `prefill_decode_split_missing` |")
    lines.append("| hook overhead | REAL_LOGGED | hook_total_time_ms in step_metrics.csv | yes | selection is included inside hook_total; do not add twice |")
    lines.append("| model/primary/raw-wall speedup | DERIVED_VALID | episode/step metrics + baseline comparison | yes, with confidence warnings | speedup < 1 is slowdown; raw wall remains the end-to-end metric |")
    lines.append("| quality preservation | REAL_LOGGED / DERIVED_VALID | episode_metrics.csv | yes | success and steps are reported with baseline deltas |")
    lines.append("| paired timing | DERIVED_VALID | task_name + trial_idx episode pairing | diagnostic | task-trial paired, not strict seed/init-state proof |")
    lines.append("| strict paired timing | REAL_LOGGED / DERIVED_VALID | seed + initial_state_index/hash in episode_metrics.csv | diagnostic | strict pairing is available only for reruns after metadata instrumentation |")
    if any_profiler_flops:
        lines.append("| profiler FLOPs | PROFILER_LOGGED | torch.profiler.with_flops sampled steps | diagnostic | operator FLOPs for profiled model.predict_action steps; profiling adds latency overhead |")
    else:
        lines.append("| profiler FLOPs | UNAVAILABLE_NEEDS_RERUN | optional torch.profiler.with_flops columns | no | run a dedicated FLOP smoke with --enable_torch_profiler_flops true |")
    lines.append("| analytic FLOPs | ANALYTIC_LLM_BLOCKS | logged seq lengths + model config | yes for analytic LLM-block FLOPs | not a torch.profiler/Nsight hardware trace |")
    lines.append("| memory | REAL_LOGGED | episode_metrics.csv | yes | do not claim reduction unless measured |")
    lines.append("| cache | NOT_APPLICABLE | method design | no | ACGTP is pruning, not cache |")
    lines.append("| branch selected counts | REAL_LOGGED | step_metrics.csv internal_selected_by_* | mechanism diagnostic | selected counts may overlap across branches |")
    if any_branch_unique:
        lines.append("| branch unique/overlap | REAL_LOGGED | step_metrics.csv internal_unique_* and overlap fields | mechanism diagnostic | available for runs after attribution instrumentation |")
    else:
        lines.append("| branch unique/overlap | UNAVAILABLE_NEEDS_INSTRUMENTATION | not present | no | requires selector/hook attribution logging and rerun |")
    lines.append("")

    lines.append("## Paper-Aligned Comparison\n")
    lines.append("| Method | Verdict | SR | SR Delta | Retain | Seq Retain | Primary | Raw Wall | Model | LLM Saved | Hook | Net LLM-Hook | Eff Attn Speed | Warnings |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for m in methods:
        d = m["baseline_delta"]
        t = m["tokens"]
        f = m["flops_estimate"]
        lines.append(
            "| "
            f"{m['label']} | {m['verdict']} | {m['quality']['successes']}/{m['quality']['episodes']} | "
            f"{fmt_pct(d['success_delta'])} | {fmt_pct(t['effective_visual_retention'])} | "
            f"{fmt_pct(t['internal_seq_retention'])} | {fmt_ratio(d['wall_speedup'])} | "
            f"{fmt_ratio(d.get('raw_wall_speedup'))} | {fmt_ratio(d['model_speedup'])} | {fmt_ms(d['llm_saved_ms'])} | "
            f"{fmt_ms(m['latency_ms']['hook_total'])} | {fmt_ms(d['net_llm_saved_after_hook_ms'])} | "
            f"{fmt_ratio(f['internal_effective_attention_speedup'])} | "
            f"{', '.join(m['diagnostics'].get('benchmark_warnings', [])) or ''} |"
        )
    lines.append("")

    lines.append("## Paired Timing\n")
    lines.append("Pairs use matching `task_name` + `trial_idx` between a strategy and the baseline. Strict pairs additionally require matching seed and LIBERO initial-state hash when those fields are present.\n")
    lines.append("| Method | Paired N | Paired Primary | Paired Raw Wall | Paired Model | Paired LLM Saved | Paired Success Delta | Confidence |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for m in methods:
        paired = (m["baseline_delta"].get("paired") or {})
        lines.append(
            "| "
            f"{m['label']} | {paired.get('paired_n', 0)} | "
            f"{fmt_ratio(paired.get('paired_primary_speedup'))} | {fmt_ratio(paired.get('paired_wall_speedup'))} | "
            f"{fmt_ratio(paired.get('paired_model_speedup'))} | "
            f"{fmt_ms(paired.get('paired_llm_saved_ms'))} | "
            f"{paired.get('paired_success_delta_count', 'N/A')} | "
            f"{m['diagnostics'].get('measurement_confidence') or 'N/A'} |"
        )
    lines.append("")

    lines.append("### Strict Paired Timing\n")
    lines.append("| Method | Strict Paired N | Strict Primary | Strict Raw Wall | Strict Model | Strict LLM Saved | Strict Success Delta |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for m in methods:
        strict_paired = (m["baseline_delta"].get("strict_paired") or {})
        lines.append(
            "| "
            f"{m['label']} | {strict_paired.get('paired_n', 0)} | "
            f"{fmt_ratio(strict_paired.get('paired_primary_speedup'))} | {fmt_ratio(strict_paired.get('paired_wall_speedup'))} | "
            f"{fmt_ratio(strict_paired.get('paired_model_speedup'))} | "
            f"{fmt_ms(strict_paired.get('paired_llm_saved_ms'))} | "
            f"{strict_paired.get('paired_success_delta_count', 'N/A')} |"
        )
    lines.append("")

    lines.append("## Timing Coverage\n")
    lines.append("| Method | Vision | Projector | LLM Total | Prefill | Decode | Hook | Selector | Plan Cache Hit | Env Step | Primary | Raw Wall | Wall Ep Avg | Wall Step Records | Split Available |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for m in methods:
        l = m["latency_ms"]
        lines.append(
            "| "
            f"{m['label']} | {fmt_ms(l['vision_encoder'])} | {fmt_ms(l['projector'])} | "
            f"{fmt_ms(l['llm_total'])} | {fmt_ms(l['llm_prefill'])} | {fmt_ms(l['llm_decode'])} | "
            f"{fmt_ms(l['hook_total'])} | {fmt_ms(l['selector'])} | "
            f"{fmt_pct(m['diagnostics'].get('latency_plan_cache_hit_rate'))} | {fmt_ms(l['env_step'])} | "
            f"{fmt_ms(l['wall_step_primary'])} | {fmt_ms(l.get('raw_wall_step_primary'))} | "
            f"{fmt_ms(l['wall_step_episode_avg'])} | {fmt_ms(l['wall_step_global'])} | "
            f"{m['diagnostics']['prefill_decode_split_available']} |"
        )
    lines.append("")

    lines.append("## LLM Internal Graph Trace\n")
    lines.append("These fields verify whether pruning shortened the actual LLM computation graph, not just the selector output. For internal pruning, full visual tokens enter early layers, then hidden states and KV cache should become shorter after the prune layer.\n")
    lines.append("| Method | Entry Vis | Post-Prune Vis | Prune Layer | First Short Layer | Post-Prune Layers | Short KV Layers | KV Reduction | Decode KV Mean |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in methods:
        t = m["tokens"]
        f = m["flops_estimate"]
        lines.append(
            "| "
            f"{m['label']} | {fmt_int(t.get('visual_tokens_at_llm_entry'))} | "
            f"{fmt_int(t.get('visual_tokens_after_internal_prune'))} | "
            f"{fmt_int(f.get('internal_pruning_layer'))} | "
            f"{fmt_int(t.get('internal_first_short_layer'))} | "
            f"{fmt_int(t.get('internal_post_prune_layer_count'))} ({fmt_pct(t.get('internal_post_prune_layer_ratio'))}) | "
            f"{fmt_int(t.get('internal_kv_cache_short_length_layer_count'))} | "
            f"{fmt_pct(t.get('internal_kv_cache_token_reduction_ratio'))} | "
            f"{fmt_int(t.get('decode_effective_kv_tokens'))} |"
        )
    lines.append("")

    lines.append("## Control Frequency\n")
    lines.append(f"`control_frequency_hz` is computed as 1000 / primary latency. Current primary latency scope is `{primary_scope}`. If the scope is `llm_only`, this is an LLM-only equivalent frequency, not a real robot end-to-end control-loop frequency. Raw wall-clock remains available for end-to-end frequency claims.\n")
    lines.append("| Method | Control Freq. | Speed vs Baseline | Source |")
    lines.append("|---|---:|---:|---|")
    for m in methods:
        summary = method_core_summary(m)
        hz_cell = f"{summary.get('control_frequency_hz'):.2f} Hz" if summary.get('control_frequency_hz') is not None else "N/A"
        lines.append(
            "| "
            f"{m['label']} | {hz_cell} | "
            f"{fmt_ratio(m['baseline_delta'].get('wall_speedup'))} | "
            f"{summary.get('control_frequency_source')} |"
        )
    lines.append("")

    lines.append("## Branch Attribution\n")
    lines.append("Selected counts are candidate tokens that survived final keep and may overlap. Unique counts are ownership-attributed final tokens, so their sum should equal kept tokens when accounting is valid.\n")
    lines.append("| Method | Hard | Layout | Contact | Motion | Semantic | Hist-Act | Fill | Geo Critical Pruned |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in methods:
        b = m["branch_attribution"]
        lines.append(
            "| "
            f"{m['label']} | {fmt_int(b.get('internal_selected_by_geo_count') if b.get('internal_selected_by_geo_count') is not None else b.get('internal_quota_hard_k'))} | "
            f"{fmt_int(b.get('internal_selected_by_layout_count'))} | "
            f"{fmt_int(b.get('internal_selected_by_contact_count'))} | "
            f"{fmt_int(b.get('internal_selected_by_motion_count'))} | "
            f"{fmt_int(b.get('internal_selected_by_semantic_attention_count'))} | "
            f"{fmt_int(b.get('internal_selected_by_historical_attention_count'))} | "
            f"{fmt_int(b.get('internal_selected_by_fill_count'))} | "
            f"{fmt_int(b.get('internal_pruned_geo_critical_count'))} |"
        )
    lines.append("")

    if any_branch_unique:
        lines.append("## Branch Unique / Overlap\n")
        lines.append("| Method | Unique Hard | Unique Layout | Unique Contact | Unique Motion | Unique Semantic | Unique Hist-Act | Unique Fill | Unique Sum | Overlap Ratio | Accounting Valid |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for m in methods:
            b = m["branch_attribution"]
            lines.append(
                "| "
                f"{m['label']} | {fmt_int(b.get('internal_unique_geo_count'))} | "
                f"{fmt_int(b.get('internal_unique_layout_count'))} | "
                f"{fmt_int(b.get('internal_unique_contact_count'))} | "
                f"{fmt_int(b.get('internal_unique_motion_count'))} | "
                f"{fmt_int(b.get('internal_unique_semantic_attention_count'))} | "
                f"{fmt_int(b.get('internal_unique_historical_attention_count'))} | "
                f"{fmt_int(b.get('internal_unique_fill_count'))} | "
                f"{fmt_int(b.get('internal_branch_unique_sum'))} | "
                f"{fmt_pct(b.get('internal_branch_overlap_ratio'))} | "
                f"{b.get('internal_branch_accounting_valid')} |"
            )
        lines.append("")

    lines.append("## Interpretation Guardrails\n")
    lines.append("- `summary.json` remains the full raw aggregate. Use `benchmark_summary.json` and this report for paper-facing analysis.")
    lines.append("- `benchmark_comparison.csv` is intentionally compact: it keeps the mechanism chain and audit flags, while raw per-step diagnostics stay in `step_metrics.csv`.")
    lines.append("- `mean_step_wall_time_ms` and `wall_speedup_vs_baseline` follow the selected primary latency scope for paper-facing tables; with the default `llm_only` scope they are pure LLM total time.")
    lines.append("- Raw end-to-end wall-clock is preserved in `mean_step_wall_time_ms_raw`, `raw_wall_speedup_vs_baseline`, and `Raw Wall` report columns.")
    lines.append("- If `prefill_decode_split_available=False`, LLM speedup is only an LLM-total trend, not a clean prefill/decode attribution.")
    lines.append("- For internal pruning, effective FLOP estimates account for the fact that layers before the prune point still run the full sequence.")
    lines.append("- FLOPs include analytic LLM transformer-block estimates by default. Optional torch.profiler FLOPs are reported only when sampled and should be collected in dedicated FLOP runs because profiling adds timing overhead.")
    lines.append("- Do not claim stable end-to-end acceleration unless quality is preserved and raw wall speedup is consistently above 1.0 across larger paired runs.\n")
    lines.append("## Minimal Instrumentation Still Needed\n")
    needed_idx = 1
    if not any_split:
        lines.append(f"{needed_idx}. **Prefill/decode split**: add timing probes only; do not change prompt, tokens, action output, or generation logic.")
        needed_idx += 1
    if not any_branch_unique:
        lines.append(f"{needed_idx}. **Branch unique/overlap attribution**: make selector/hook log per-branch unique counts, overlap ratios, and `branch_sum_equals_kept`; do not infer these offline.")
        needed_idx += 1
    strict_available = any((m.get("baseline_delta", {}).get("strict_paired") or {}).get("paired_n", 0) > 0 for m in methods)
    if not strict_available:
        lines.append(f"{needed_idx}. **Strict pairing rerun**: new runs log seed and initial-state hash; rerun to upgrade old task-trial pairs to strict pairs.")
        needed_idx += 1
    if not any_profiler_flops:
        lines.append(f"{needed_idx}. **Profiler FLOPs smoke**: run with `--enable_torch_profiler_flops true --torch_profiler_flops_max_steps 1` to populate profiler FLOP fields. Use a separate run because torch.profiler changes latency.\n")
    else:
        lines.append(f"{needed_idx}. **Profiler FLOPs caveat**: sampled torch.profiler FLOPs are present; keep latency claims from non-profiled runs.\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def build(
    root: Path,
    selected: Optional[List[str]],
    baseline_label: Optional[str],
    prefix: str,
    latency_scope: str = "llm_only",
) -> Dict[str, Any]:
    dirs = method_dirs(root, selected)
    if not dirs:
        raise SystemExit(f"No method directories found under {root}")
    methods = [compact_method(d, latency_scope=latency_scope) for d in dirs]
    baseline = add_baseline_deltas(methods, baseline_label)
    add_cross_method_warnings(methods)
    write_method_summaries(methods)
    compact_methods = [method_core_summary(m) for m in methods]
    report = {
        "root": str(root.resolve()),
        "baseline": baseline["label"],
        "methods": compact_methods,
        "comparison_rows": [flatten_row(m) for m in methods],
        "notes": {
            "raw_full_summary": "Each strategy directory still contains the full summary.json and step_metrics.csv.",
            "primary_latency_scope": latency_scope,
            "primary_latency": "Paper-facing latency and control_frequency_hz use the selected primary latency scope; raw wall-clock remains in *_raw fields.",
            "prefill_decode": "If prefill_decode_split_available is false, LLM timing is total forward only.",
            "flops": "Analytic LLM transformer-block FLOPs are computed from logged sequence lengths and model config; optional torch.profiler operator FLOPs are reported only for sampled profiler runs.",
        },
    }
    (root / f"{prefix}_metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv_report(methods, root / f"{prefix}_comparison.csv")
    write_markdown(methods, baseline, root / f"{prefix}_report.md")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", "--input_dir", dest="root", required=True, type=Path)
    parser.add_argument("--method", action="append", default=None)
    parser.add_argument("--baseline", "--baseline_strategy", dest="baseline", default=None)
    parser.add_argument("--prefix", default="benchmark")
    parser.add_argument(
        "--latency_scope",
        choices=["llm_only", "model_forward", "wall"],
        default="llm_only",
        help=(
            "Primary latency axis for paper-facing wall/control-frequency fields. "
            "Default llm_only maps mean_step_wall_time_ms to mean_llm_forward_time_ms "
            "while preserving raw wall-clock in *_raw fields."
        ),
    )
    args = parser.parse_args()
    report = build(args.root.resolve(), args.method, args.baseline, args.prefix, args.latency_scope)
    root = Path(report["root"])
    print(f"Wrote {root / (args.prefix + '_metrics.json')}")
    print(f"Wrote {root / (args.prefix + '_comparison.csv')}")
    print(f"Wrote {root / (args.prefix + '_report.md')}")
    print("Wrote per-method benchmark_summary.json files")


if __name__ == "__main__":
    main()
