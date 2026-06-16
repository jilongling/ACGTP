"""Build a direct performance report for pruning comparison runs.

The report is read-only with respect to inference outputs. It expects a root
directory containing one subdirectory per strategy, each with summary.json,
episode_metrics.csv, and step_metrics.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from pruning.strategy_registry import is_main_experiment_method_label
except Exception:  # Keep report builder usable on archived outputs without imports.
    def is_main_experiment_method_label(label: str) -> bool:
        text = str(label or "").strip()
        return (
            text in {"none", "baseline", "baseline_none", "baseline_none_keep100"}
            or text.startswith("projector_acgtp_legacy")
            or text.startswith("internal_acgtp_geometry_only")
            or text.startswith("internal_acgtp_geo_guarded")
            or text.startswith("internal_acgtp_dynamic")
            or text.startswith("internal_geometry_only")
            or text.startswith("internal_geo_guarded")
            or text.startswith("internal_dynamic")
        )


DEFAULT_LABELS = {
    "none": "none",
    "baseline": "none",
    "depth_edge_fast_075": "depth_edge_fast@0.75",
    "acgtp_dynamic_fast_075": "ACGTP dynamic-fast@0.75",
    "acgtp_dynamic_history_fast_075": "ACGTP dynamic-history-fast@0.75",
}


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in {"none", "nan", "null", "n/a"}:
        return None
    try:
        value_f = float(text)
    except ValueError:
        return None
    return value_f if math.isfinite(value_f) else None


def _as_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", ""}:
        return False
    return None


def _values(rows: Iterable[Dict[str, str]], field: str) -> List[float]:
    vals: List[float] = []
    for row in rows:
        value = _as_float(row.get(field))
        if value is not None:
            vals.append(value)
    return vals


def _mean(vals: Iterable[float]) -> Optional[float]:
    vals = list(vals)
    return sum(vals) / len(vals) if vals else None


def _percentile(vals: Iterable[float], q: float) -> Optional[float]:
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


def _first_mean(rows: List[Dict[str, str]], fields: Iterable[str], fallback: Any = None) -> Optional[float]:
    for field in fields:
        vals = _values(rows, field)
        if vals:
            return _mean(vals)
    return _as_float(fallback)


def _counter(rows: List[Dict[str, str]], field: str) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        text = str(row.get(field, "")).strip()
        if text:
            counts[text] += 1
    return dict(counts)


def _success_from_episodes(rows: List[Dict[str, str]]) -> Tuple[int, int, Dict[str, Tuple[int, int]]]:
    per_task: Dict[str, List[int]] = {}
    successes = 0
    total = 0
    for row in rows:
        task = row.get("task_name") or f"task_{row.get('task_id', '')}".strip("_")
        ok = _as_bool(row.get("success"))
        if ok is None:
            continue
        total += 1
        successes += int(ok)
        if task not in per_task:
            per_task[task] = [0, 0]
        per_task[task][0] += int(ok)
        per_task[task][1] += 1
    return successes, total, {k: (v[0], v[1]) for k, v in per_task.items()}


def _method_dirs(root: Path, selected: Optional[List[str]]) -> List[Path]:
    dirs = [
        p for p in root.iterdir()
        if p.is_dir() and (p / "summary.json").exists() and (p / "step_metrics.csv").exists()
    ]
    if selected:
        wanted = set(selected)
        dirs = [p for p in dirs if p.name in wanted]

    def key(path: Path) -> Tuple[int, str]:
        name = path.name.lower()
        if name in {"none", "baseline", "baseline_none_keep100"}:
            return (0, name)
        if "depth_edge" in name:
            return (1, name)
        if "acgtp" in name:
            return (2, name)
        return (3, name)

    return sorted(dirs, key=key)


def _fmt_ms(v: Optional[float]) -> str:
    return "N/A" if v is None else f"{v:.2f}"


def _fmt_ratio(v: Optional[float]) -> str:
    return "N/A" if v is None else f"{v:.3f}x"


def _fmt_pct(v: Optional[float]) -> str:
    return "N/A" if v is None else f"{100.0 * v:.1f}%"


def _fmt_int(v: Optional[float]) -> str:
    return "N/A" if v is None else str(int(round(v)))


def _safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return a / b


def _row_ratio(row: Dict[str, Any], kept_fields: Iterable[str], original_fields: Iterable[str]) -> Optional[float]:
    kept = _first_mean([row], kept_fields)
    original = _first_mean([row], original_fields)
    return _safe_div(kept, original)


def _mean_ratio(rows: List[Dict[str, str]], kept_fields: Iterable[str], original_fields: Iterable[str]) -> Optional[float]:
    vals: List[float] = []
    for row in rows:
        ratio = _row_ratio(row, kept_fields, original_fields)
        if ratio is not None:
            vals.append(ratio)
    return _mean(vals)


def _first_nonempty(rows: List[Dict[str, str]], field: str) -> Optional[str]:
    for row in rows:
        value = row.get(field)
        if value not in (None, ""):
            return str(value)
    return None


def _label_for(name: str, labels: Dict[str, str]) -> str:
    if name in labels:
        return labels[name]
    return DEFAULT_LABELS.get(name, name)


def build_report(
    root: Path,
    method_names: Optional[List[str]],
    labels: Dict[str, str],
    baseline_method: Optional[str] = None,
) -> Dict[str, Any]:
    methods: List[Dict[str, Any]] = []
    per_task_by_label: Dict[str, Dict[str, Tuple[int, int]]] = {}
    dirs = _method_dirs(root, method_names)
    if not dirs:
        raise SystemExit(f"No method directories found under {root}")

    for method_dir in dirs:
        summary = _read_json(method_dir / "summary.json")
        step_rows = _read_csv(method_dir / "step_metrics.csv")
        ep_rows = _read_csv(method_dir / "episode_metrics.csv")
        succ, total, per_task = _success_from_episodes(ep_rows)
        if total <= 0:
            succ = int(summary.get("num_successes") or 0)
            total = int(summary.get("num_episodes") or summary.get("num_trials") or 0)

        kept = _first_mean(step_rows, ["num_visual_tokens_kept", "num_visual_tokens_kept_total"], summary.get("mean_visual_tokens_kept"))
        orig = _first_mean(step_rows, ["num_visual_tokens_original", "num_visual_tokens_original_total"], summary.get("mean_num_visual_tokens_original"))
        compression_backend = (_first_nonempty(step_rows, "compression_backend") or str(summary.get("compression_backend") or "projector")).strip().lower()

        projector_kept = _first_mean(step_rows, ["num_projector_visual_tokens_kept"], summary.get("mean_projector_visual_tokens_kept"))
        projector_orig = _first_mean(step_rows, ["num_projector_visual_tokens_original"], summary.get("mean_projector_visual_tokens_original"))
        projector_retention = _mean_ratio(step_rows, ["num_projector_visual_tokens_kept"], ["num_projector_visual_tokens_original"])
        if projector_retention is None:
            projector_retention = _safe_div(projector_kept, projector_orig)
        if projector_retention is None:
            projector_retention = _mean_ratio(step_rows, ["num_visual_tokens_kept", "num_visual_tokens_kept_total"], ["num_visual_tokens_original", "num_visual_tokens_original_total"])
            projector_kept = kept if projector_kept is None else projector_kept
            projector_orig = orig if projector_orig is None else projector_orig

        internal_kept = _first_mean(step_rows, ["internal_kept_visual_tokens"], summary.get("mean_internal_kept_visual_tokens"))
        internal_orig = _first_mean(step_rows, ["internal_original_visual_tokens"], summary.get("mean_internal_original_visual_tokens"))
        internal_pruned = _first_mean(step_rows, ["internal_pruned_visual_tokens"], summary.get("mean_internal_pruned_visual_tokens"))
        internal_seq_orig = _first_mean(step_rows, ["internal_original_seq_length"], summary.get("mean_internal_original_seq_length"))
        internal_seq_kept = _first_mean(step_rows, ["internal_kept_seq_length"], summary.get("mean_internal_kept_seq_length"))
        internal_seq_pruned = _first_mean(step_rows, ["internal_pruned_seq_length"], summary.get("mean_internal_pruned_seq_length"))
        internal_retention = _mean_ratio(step_rows, ["internal_kept_visual_tokens"], ["internal_original_visual_tokens"])
        if internal_retention is None:
            internal_retention = _safe_div(internal_kept, internal_orig)

        generic_retention = _mean_ratio(step_rows, ["num_visual_tokens_kept", "num_visual_tokens_kept_total"], ["num_visual_tokens_original", "num_visual_tokens_original_total"])
        if generic_retention is None:
            generic_retention = _safe_div(kept, orig)
        if compression_backend == "internal" and internal_retention is not None:
            retention = internal_retention
            effective_kept = internal_kept
            effective_orig = internal_orig
        else:
            retention = generic_retention
            effective_kept = kept
            effective_orig = orig
        effective_visual_tokens_for_llm = _first_mean(step_rows, ["effective_visual_tokens_for_llm"], summary.get("effective_visual_tokens_for_llm_mean"))
        if effective_visual_tokens_for_llm is None:
            effective_visual_tokens_for_llm = effective_kept
        retention_warning = None
        if compression_backend == "internal" and internal_retention is None:
            retention_warning = "internal_backend_missing_internal_retention_fields"

        hook_vals = _values(step_rows, "hook_total_time_ms") or _values(step_rows, "hook_total_ms")
        selector_vals = (
            _values(step_rows, "selection_ms")
            or _values(step_rows, "pruning_time_ms")
            or _values(step_rows, "topk_pruning_ms")
        )

        fallback_steps = 0
        for row in step_rows:
            if _as_bool(row.get("fallback_used")):
                fallback_steps += 1

        label = _label_for(method_dir.name, labels)
        main_surface = is_main_experiment_method_label(method_dir.name) or is_main_experiment_method_label(label)
        row = {
            "method_dir": str(method_dir),
            "method": method_dir.name,
            "label": label,
            "main_experiment_surface": main_surface,
            "audit_probe_only": not main_surface,
            "successes": succ,
            "episodes": total,
            "success_rate": succ / total if total else None,
            "n_steps": len(step_rows),
            "compression_backend": compression_backend,
            "kept_mean": kept,
            "original_mean": orig,
            "projector_kept_mean": projector_kept,
            "projector_original_mean": projector_orig,
            "projector_retention": projector_retention,
            "internal_kept_mean": internal_kept,
            "internal_original_mean": internal_orig,
            "internal_pruned_mean": internal_pruned,
            "internal_seq_original_mean": internal_seq_orig,
            "internal_seq_kept_mean": internal_seq_kept,
            "internal_seq_pruned_mean": internal_seq_pruned,
            "internal_retention": internal_retention,
            "effective_kept_mean": effective_kept,
            "effective_original_mean": effective_orig,
            "effective_visual_tokens_for_llm": effective_visual_tokens_for_llm,
            "effective_retention": retention,
            "retention": retention,
            "retention_warning": retention_warning,
            "cuda_ms": _first_mean(step_rows, ["cuda_latency_ms"], summary.get("mean_cuda_latency_ms")),
            "model_ms": _first_mean(step_rows, ["total_model_forward_time_ms", "model_forward_ms"], summary.get("mean_model_forward_time_ms")),
            # Use the episode-level wall average when available. It matches the
            # run summary and avoids overweighting long episodes in the wall
            # metric, while component timings below remain step-weighted.
            "wall_ms": (
                _first_mean(ep_rows, ["mean_step_wall_time_ms", "end_to_end_step_wall_time_ms"], summary.get("mean_step_wall_time_ms"))
                or _first_mean(step_rows, ["total_step_wall_time_ms", "end_to_end_step_wall_time_ms"], summary.get("mean_step_wall_time_ms"))
            ),
            "vision_ms": _first_mean(step_rows, ["vision_encoder_time_ms"], summary.get("mean_vision_encoder_time_ms")),
            "projector_ms": _first_mean(step_rows, ["projector_time_ms"], summary.get("mean_projector_time_ms")),
            "llm_ms": _first_mean(step_rows, ["llm_forward_time_ms"], summary.get("mean_llm_forward_time_ms")),
            "lm_prefill_ms": _first_mean(step_rows, ["lm_prefill_time_ms_observed", "prefill_time_ms"], summary.get("mean_lm_prefill_time_ms_observed")),
            "lm_decode_observed_ms": _first_mean(step_rows, ["lm_decode_time_ms_observed", "decode_time_ms"], summary.get("mean_lm_decode_time_ms_observed")),
            "decode_ms": _first_mean(step_rows, ["action_decode_time_ms"], summary.get("mean_action_decode_time_ms")),
            # Mechanism-evidence inputs (paper-aligned). The internal backend prunes
            # at LLM layer K, so layers 0..K still run the full sequence: an honest
            # internal FLOP estimate must weight the saving by the benefiting-layer
            # fraction, not assume all layers shrink. We read the prune layer and the
            # geo-critical guard count to support that and the verdict logic below.
            "internal_pruning_layer": _first_mean(step_rows, ["internal_pruning_layer"], summary.get("mean_internal_pruning_layer")),
            "internal_pruned_geo_critical_count": _first_mean(
                step_rows, ["internal_pruned_geo_critical_count"], summary.get("mean_internal_pruned_geo_critical_count")
            ),
            "internal_quota_layout_k": _first_mean(step_rows, ["internal_quota_layout_k"], summary.get("mean_internal_quota_layout_k")),
            "internal_quota_contact_k": _first_mean(step_rows, ["internal_quota_contact_k"], summary.get("mean_internal_quota_contact_k")),
            "internal_quota_motion_k": _first_mean(step_rows, ["internal_quota_motion_k"], summary.get("mean_internal_quota_motion_k")),
            "internal_quota_semantic_attention_k": _first_mean(
                step_rows, ["internal_quota_semantic_attention_k"], summary.get("mean_internal_quota_semantic_attention_k")
            ),
            "internal_quota_historical_attention_k": _first_mean(
                step_rows, ["internal_quota_historical_attention_k"], summary.get("mean_internal_quota_historical_attention_k")
            ),
            "internal_quota_fill_k": _first_mean(step_rows, ["internal_quota_fill_k"], summary.get("mean_internal_quota_fill_k")),
            "internal_selected_by_layout_count": _first_mean(
                step_rows, ["internal_selected_by_layout_count"], summary.get("mean_internal_selected_by_layout_count")
            ),
            "internal_selected_by_contact_count": _first_mean(
                step_rows, ["internal_selected_by_contact_count"], summary.get("mean_internal_selected_by_contact_count")
            ),
            "internal_selected_by_motion_count": _first_mean(
                step_rows, ["internal_selected_by_motion_count"], summary.get("mean_internal_selected_by_motion_count")
            ),
            "internal_selected_by_semantic_attention_count": _first_mean(
                step_rows,
                ["internal_selected_by_semantic_attention_count"],
                summary.get("mean_internal_selected_by_semantic_attention_count"),
            ),
            "internal_selected_by_historical_attention_count": _first_mean(
                step_rows,
                ["internal_selected_by_historical_attention_count"],
                summary.get("mean_internal_selected_by_historical_attention_count"),
            ),
            "internal_selected_by_fill_count": _first_mean(
                step_rows, ["internal_selected_by_fill_count"], summary.get("mean_internal_selected_by_fill_count")
            ),
            "llm_total_layers": _first_mean(
                step_rows, ["llm_num_hidden_layers", "num_hidden_layers", "internal_total_layers"], summary.get("llm_num_hidden_layers")
            ),
            "hook_ms": _mean(hook_vals),
            "hook_p50_ms": _percentile(hook_vals, 0.50),
            "hook_p95_ms": _percentile(hook_vals, 0.95),
            "selector_ms": _mean(selector_vals),
            "selector_p50_ms": _percentile(selector_vals, 0.50),
            "score_ms": _first_mean(step_rows, ["score_compute_ms", "geometry_score_time_ms", "token_scoring_time_ms"], summary.get("mean_score_compute_ms")),
            "mapping_ms": _first_mean(step_rows, ["token_mapping_time_ms"], summary.get("mean_token_mapping_ms")),
            "gather_ms": _first_mean(step_rows, ["gather_ms"], summary.get("mean_gather_ms")),
            "depth_edge_score_ms": _first_mean(step_rows, ["depth_edge_score_ms"], summary.get("mean_depth_edge_score_ms")),
            "action_constraint_ms": _first_mean(step_rows, ["acgtp_action_constraint_ms"], summary.get("mean_acgtp_action_constraint_ms")),
            "fallback_steps": fallback_steps,
            "fallback_rate": fallback_steps / len(step_rows) if step_rows else 0.0,
            "selector_counts": _counter(step_rows, "selector_function_name"),
            "fallback_reason_counts": _counter(step_rows, "fallback_reason"),
            "dynamic_phase_counts": _counter(step_rows, "dynamic_phase"),
            "history_counts": _counter(step_rows, "acgtp_history_enabled") or _counter(step_rows, "score_ema_enabled"),
        }
        methods.append(row)
        per_task_by_label[label] = per_task

    if baseline_method:
        baseline_key = baseline_method.strip()
        baseline = next(
            (m for m in methods if m["method"] == baseline_key or m["label"] == baseline_key),
            None,
        )
        if baseline is None:
            available = ", ".join(m["method"] for m in methods)
            raise SystemExit(f"Baseline method not found: {baseline_key}. Available: {available}")
    else:
        baseline = next((m for m in methods if m["method"].lower() in {"none", "baseline", "baseline_none_keep100"}), methods[0])
    for row in methods:
        row["success_delta_vs_baseline"] = (
            None
            if row["success_rate"] is None or baseline["success_rate"] is None
            else row["success_rate"] - baseline["success_rate"]
        )
        for field in ["cuda_ms", "model_ms", "wall_ms", "llm_ms", "lm_prefill_ms", "lm_decode_observed_ms"]:
            base_value = baseline.get(field)
            value = row.get(field)
            prefix = field.replace("_ms", "")
            # Normalize the two LM-component prefixes so the derived keys read as
            # prefill_* / decode_*, matching the paper-aligned naming.
            if prefix == "lm_prefill":
                prefix = "prefill"
            elif prefix == "lm_decode_observed":
                prefix = "decode"
            row[f"{prefix}_speedup_vs_baseline"] = _safe_div(base_value, value)
            row[f"{prefix}_saved_ms_vs_baseline"] = None if base_value is None or value is None else base_value - value

        llm_saved = row.get("llm_saved_ms_vs_baseline")
        model_saved = row.get("model_saved_ms_vs_baseline")
        hook = row.get("hook_ms") or 0.0
        selector = row.get("selector_ms") or 0.0
        row["hook_pct_of_llm_saved"] = _safe_div(hook, llm_saved)
        row["selector_pct_of_llm_saved"] = _safe_div(selector, llm_saved)
        row["surviving_model_gain_pct_of_llm_saved"] = _safe_div(model_saved, llm_saved)

        # ── Paper-aligned mechanism-evidence metrics ────────────────────────
        # Category 4 in VLA-ADP / VLA-IAP / VLA-Pruner / VLA-Cache / DepthCache:
        # show *why* fewer tokens should be faster (sequence/FLOP reduction) and
        # how much of that theoretical gain actually reaches wall-clock.
        prefill_saved = row.get("prefill_saved_ms_vs_baseline")
        prefill_ms = row.get("lm_prefill_ms")
        decode_ms = row.get("lm_decode_observed_ms")
        llm_ms = row.get("llm_ms")
        wall_saved = row.get("wall_saved_ms_vs_baseline")

        row["prefill_share_of_llm"] = _safe_div(prefill_ms, llm_ms)
        row["decode_share_of_llm"] = _safe_div(decode_ms, llm_ms)

        # Sequence-length reduction at the internal prune point.
        seq_kept = row.get("internal_seq_kept_mean")
        seq_orig = row.get("internal_seq_original_mean")
        internal_seq_retention = _safe_div(seq_kept, seq_orig)
        row["internal_seq_retention"] = internal_seq_retention
        row["internal_seq_reduction_pct"] = (
            None if internal_seq_retention is None else 1.0 - internal_seq_retention
        )

        # Naive theoretical speedup, assuming ALL layers run the shortened
        # sequence (this is the upper bound used by projector/pre-layer-0 work
        # such as VLA-IAP / VLA-ADP / DepthCache).
        seq_ratio = internal_seq_retention
        row["theoretical_attention_flop_ratio"] = (
            None if seq_ratio is None else seq_ratio ** 2
        )
        row["theoretical_attention_speedup"] = _safe_div(
            1.0, row["theoretical_attention_flop_ratio"]
        )
        row["theoretical_linear_flop_ratio"] = seq_ratio
        row["theoretical_linear_speedup"] = _safe_div(1.0, seq_ratio)

        # Internal-backend honest correction: pruning happens AFTER layer K, so
        # layers 0..K still run the full sequence. Effective per-token saving is
        # weighted by the benefiting-layer fraction (L-1-K)/L. This is the number
        # that should be compared to wall — the naive one above is an upper bound.
        prune_layer = row.get("internal_pruning_layer")
        total_layers = row.get("llm_total_layers")
        backend = (row.get("compression_backend") or "").strip().lower()
        if backend == "internal" and prune_layer is not None and total_layers and total_layers > 0:
            benefit_frac = max(0.0, (total_layers - 1.0 - prune_layer) / total_layers)
        elif backend == "internal" and prune_layer is not None:
            # Fall back to the OpenVLA Llama-7B depth (32) when the column is
            # absent in older runs, so the honest estimate is still populated.
            benefit_frac = max(0.0, (32.0 - 1.0 - prune_layer) / 32.0)
        else:
            # Projector / pre-layer-0 pruning: every layer benefits.
            benefit_frac = 1.0
        row["internal_benefiting_layer_fraction"] = benefit_frac
        if seq_ratio is not None:
            attn_eff = benefit_frac * (seq_ratio ** 2) + (1.0 - benefit_frac) * 1.0
            lin_eff = benefit_frac * seq_ratio + (1.0 - benefit_frac) * 1.0
            row["internal_effective_attention_flop_ratio"] = attn_eff
            row["internal_effective_attention_speedup"] = _safe_div(1.0, attn_eff)
            row["internal_effective_linear_speedup"] = _safe_div(1.0, lin_eff)
        else:
            row["internal_effective_attention_flop_ratio"] = None
            row["internal_effective_attention_speedup"] = None
            row["internal_effective_linear_speedup"] = None

        # How much of the prefill saving survives to LLM / wall — the
        # theory-vs-wall gap that VLA-Pruner reports explicitly.
        row["wall_gain_vs_prefill_gain_ratio"] = _safe_div(wall_saved, prefill_saved)
        row["llm_gain_vs_prefill_gain_ratio"] = _safe_div(llm_saved, prefill_saved)

        # ── Layered verdict (paper-aligned) ─────────────────────────────────
        # End-to-end wall/CUDA speedup on autoregressive base OpenVLA is decode-
        # bound and is NOT the sole gate. We separate: quality (SR + geo guard),
        # mechanism (prefill gain), and end-to-end (wall gain).
        success_delta = row.get("success_delta_vs_baseline")
        prefill_speedup = row.get("prefill_speedup_vs_baseline")  # None when unmeasured
        wall_speedup = row.get("wall_speedup_vs_baseline") or 0.0
        cuda_speedup = row.get("cuda_speedup_vs_baseline") or 0.0
        geo_critical = row.get("internal_pruned_geo_critical_count")
        # quality threshold: allow up to 1/episodes (≈ one failed episode) drop,
        # floored at 5%, mirroring the ≤1% SR-drop convention in the papers
        # while staying meaningful on small episode counts.
        n_ep = row.get("episodes") or 0
        quality_tol = max(0.05, (1.0 / n_ep) if n_ep else 0.05)
        geo_guard_ok = (geo_critical is None) or (geo_critical <= 0.0)

        if row is baseline:
            verdict = "BASELINE"
        elif success_delta is not None and success_delta < -quality_tol:
            verdict = "QUALITY_REGRESSION"
        elif wall_speedup > 1.02:
            verdict = "END_TO_END_SPEEDUP"
        elif prefill_speedup is None:
            # Prefill/decode split was not measured in this run (e.g. plain
            # eval_openvla_baseline rather than the prefill-splitting probe).
            # Do NOT condemn the method on an unmeasured mechanism metric — report
            # quality status and point at the probe for the mechanism evidence.
            if geo_guard_ok:
                verdict = "QUALITY_OK_MECHANISM_UNMEASURED"
            else:
                verdict = "GEO_GUARD_BREACH_MECHANISM_UNMEASURED"
        elif prefill_speedup <= 1.0:
            verdict = "NO_PREFILL_GAIN"
        elif geo_guard_ok:
            # Mechanism works (prefill shrank, quality held, guard intact) but
            # the autoregressive decode floor caps end-to-end on base OpenVLA.
            verdict = "MECHANISM_VALID_BUT_END_TO_END_DECODE_BOUND"
        else:
            verdict = "MECHANISM_PARTIAL: prefill gain without geo guard"
        row["verdict"] = verdict
        # Keep the legacy coarse verdict so downstream/eyeballing stays stable.
        if row is baseline:
            row["legacy_verdict"] = "BASELINE"
        elif success_delta is not None and success_delta < -0.05:
            row["legacy_verdict"] = "FAIL: success drop"
        elif cuda_speedup < 1.02:
            row["legacy_verdict"] = "WEAK: little net CUDA gain"
        elif wall_speedup < 1.00:
            row["legacy_verdict"] = "WEAK: wall time regressed"
        elif cuda_speedup < 1.10:
            row["legacy_verdict"] = "MILD: speedup is real but small"
        else:
            row["legacy_verdict"] = "PASS: useful net speedup"

    tasks = sorted({task for table in per_task_by_label.values() for task in table})
    return {
        "root": str(root.resolve()),
        "baseline_label": baseline["label"],
        "methods": methods,
        "per_task": per_task_by_label,
        "tasks": tasks,
    }


def _selector_name(row: Dict[str, Any]) -> str:
    counts = row.get("selector_counts") or {}
    if not counts:
        return "N/A"
    return ", ".join(f"{k}({v})" for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3])


def _history_name(row: Dict[str, Any]) -> str:
    counts = row.get("history_counts") or {}
    if not counts:
        return "N/A"
    return ", ".join(f"{k}({v})" for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3])


def _write_paper_aligned_sections(lines: List[str], methods: List[Dict[str, Any]], baseline: Dict[str, Any]) -> None:
    """Emit the three paper-aligned markdown sections + the alignment table.

    Aligns the report with the four metric categories common to VLA-ADP,
    VLA-IAP, VLA-Pruner, VLA-Cache and DepthCache: (1) retention, (2) success
    rate, (3) end-to-end speedup, (4) mechanism evidence (prefill latency,
    sequence/FLOP reduction, theory-vs-wall gap).
    """
    lines.append("## Paper-aligned Metrics\n")
    lines.append(
        "This report aligns with the evaluation conventions of VLA-ADP, VLA-IAP, "
        "VLA-Pruner, VLA-Cache and DepthCache, which jointly report four metric "
        "categories rather than a single end-to-end speedup number:\n"
    )
    lines.append("1. **Retention** — visual-token keep ratio (`retention`, `internal_seq_retention`).")
    lines.append("2. **Success rate** — task SR and its delta vs baseline (`success_rate`, `success_delta_vs_baseline`).")
    lines.append("3. **Speedup** — end-to-end wall/CUDA (`wall_speedup_vs_baseline`, `cuda_speedup_vs_baseline`).")
    lines.append(
        "4. **Mechanism evidence** — prefill latency and theoretical FLOP/sequence "
        "reduction (`prefill_speedup_vs_baseline`, `theoretical_attention_speedup`, "
        "`theoretical_linear_speedup`), used to show *why* fewer tokens should be faster.\n"
    )
    lines.append(
        "> On autoregressive base OpenVLA, end-to-end wall/CUDA speedup is decode-bound "
        "and is treated as an **auxiliary** metric, not the sole pass/fail gate. "
        "The verdict separates quality (SR + geo guard), mechanism (prefill gain), and "
        "end-to-end (wall gain). See *Prefill-vs-Decode Breakdown* and *Theory-vs-Wall Gap*.\n"
    )
    lines.append(
        "| Method | Backend | SR | SR Δ | Retain | Seq Retain | Prefill ms | Prefill speed | Decode ms | Decode share | LLM ms | Wall ms | Wall speed | Theo attn speed | Theo lin speed | Verdict |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in methods:
        lines.append(
            "| "
            f"{row['label']} | {row.get('compression_backend') or 'unknown'} | "
            f"{_fmt_pct(row.get('success_rate'))} | {_fmt_pct(row.get('success_delta_vs_baseline'))} | "
            f"{_fmt_pct(row.get('retention'))} | {_fmt_pct(row.get('internal_seq_retention'))} | "
            f"{_fmt_ms(row.get('lm_prefill_ms'))} | {_fmt_ratio(row.get('prefill_speedup_vs_baseline'))} | "
            f"{_fmt_ms(row.get('lm_decode_observed_ms'))} | {_fmt_pct(row.get('decode_share_of_llm'))} | "
            f"{_fmt_ms(row.get('llm_ms'))} | {_fmt_ms(row.get('wall_ms'))} | {_fmt_ratio(row.get('wall_speedup_vs_baseline'))} | "
            f"{_fmt_ratio(row.get('theoretical_attention_speedup'))} | {_fmt_ratio(row.get('theoretical_linear_speedup'))} | "
            f"{row.get('verdict')} |"
        )
    lines.append("")

    lines.append("## Prefill-vs-Decode Breakdown\n")
    lines.append(
        "Splits LLM time into prefill (where visual-token pruning can help) and "
        "autoregressive decode (memory-bandwidth-bound on the 7B weights, unaffected "
        "by shorter visual context). A high `decode_share` confirms the model is "
        "decode-bound, which is why end-to-end speedup is capped on base OpenVLA.\n"
    )
    lines.append(
        "| Method | Backend | Prefill ms | Decode ms | Prefill share | Decode share | Prefill speed | Decode bound? |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for row in methods:
        decode_share = row.get("decode_share_of_llm")
        decode_bound = "N/A" if decode_share is None else ("YES" if decode_share >= 0.5 else "no")
        lines.append(
            "| "
            f"{row['label']} | {row.get('compression_backend') or 'unknown'} | "
            f"{_fmt_ms(row.get('lm_prefill_ms'))} | {_fmt_ms(row.get('lm_decode_observed_ms'))} | "
            f"{_fmt_pct(row.get('prefill_share_of_llm'))} | {_fmt_pct(decode_share)} | "
            f"{_fmt_ratio(row.get('prefill_speedup_vs_baseline'))} | {decode_bound} |"
        )
    lines.append("")

    lines.append("## Theory-vs-Wall Gap\n")
    lines.append(
        "Compares the sequence/FLOP reduction (what pruning *should* buy) against the "
        "wall-clock speedup actually observed. If theoretical speedup is much larger "
        "than wall speedup, the bottleneck is **decode / hook overhead / memory "
        "bandwidth**, NOT the token-selection quality. This is the same FLOP-vs-wall "
        "gap VLA-Pruner reports (FLOP cut to ~30% → theoretical ~3.3x, but wall only ~1.83x "
        "on base OpenVLA).\n"
    )
    lines.append(
        "> For the **internal** backend, pruning happens after LLM layer K, so layers "
        "0..K still run the full sequence. The naive `theoretical_*_speedup` columns assume "
        "all layers shrink (an upper bound, valid for pre-layer-0 / projector pruning); the "
        "`internal_effective_*` columns weight the saving by the benefiting-layer fraction "
        "`(L-1-K)/L` and are the honest internal estimate to compare against wall.\n"
    )
    lines.append(
        "| Method | Backend | Prune layer | Benefit frac | Seq Retain | Seq Reduct | Theo attn speed (UB) | Theo lin speed (UB) | Internal-eff attn speed | Wall speed | LLM gain/Prefill gain | Wall gain/Prefill gain |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in methods:
        lines.append(
            "| "
            f"{row['label']} | {row.get('compression_backend') or 'unknown'} | "
            f"{_fmt_int(row.get('internal_pruning_layer'))} | {_fmt_pct(row.get('internal_benefiting_layer_fraction'))} | "
            f"{_fmt_pct(row.get('internal_seq_retention'))} | {_fmt_pct(row.get('internal_seq_reduction_pct'))} | "
            f"{_fmt_ratio(row.get('theoretical_attention_speedup'))} | {_fmt_ratio(row.get('theoretical_linear_speedup'))} | "
            f"{_fmt_ratio(row.get('internal_effective_attention_speedup'))} | {_fmt_ratio(row.get('wall_speedup_vs_baseline'))} | "
            f"{_fmt_ratio(row.get('llm_gain_vs_prefill_gain_ratio'))} | {_fmt_ratio(row.get('wall_gain_vs_prefill_gain_ratio'))} |"
        )
    lines.append("")


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    methods = report["methods"]
    baseline = next(m for m in methods if m["label"] == report["baseline_label"])
    lines: List[str] = []
    lines.append("# Performance Report\n")
    lines.append(f"Root: `{report['root']}`\n")
    lines.append(f"Baseline: `{baseline['label']}`\n")
    audit_only = [row for row in methods if row.get("audit_probe_only")]
    if audit_only:
        lines.append(
            "Audit/probe-only methods present: "
            + ", ".join(f"`{row['label']}`" for row in audit_only)
            + ". They are excluded from the formal convergence surface unless explicitly requested.\n"
        )
    lines.append("## Read This First\n")

    for row in methods:
        if row is baseline:
            continue
        lines.append(
            "- "
            f"**{row['label']}**: {row['verdict']}; "
            f"success {row['successes']}/{row['episodes']} ({_fmt_pct(row['success_rate'])}, "
            f"delta {_fmt_pct(row['success_delta_vs_baseline'])}); "
            f"prefill {_fmt_ratio(row.get('prefill_speedup_vs_baseline'))} "
            f"(decode share {_fmt_pct(row.get('decode_share_of_llm'))}); "
            f"wall {_fmt_ratio(row['wall_speedup_vs_baseline'])}, "
            f"CUDA {_fmt_ratio(row['cuda_speedup_vs_baseline'])}; "
            f"LLM saved {_fmt_ms(row['llm_saved_ms_vs_baseline'])} ms, "
            f"hook {_fmt_ms(row['hook_ms'])} ms, selector {_fmt_ms(row['selector_ms'])} ms"
            f"{'; warning ' + row['retention_warning'] if row.get('retention_warning') else ''}."
        )
    lines.append("")

    lines.append("## Net Performance\n")
    lines.append("| Strategy | Backend | Verdict | Success | Effective Retain | Effective Visual | Projector Retain | Internal Retain | CUDA ms | CUDA speed | Wall ms | Wall speed | LLM saved | Hook ms | Selector ms | Model saved |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in methods:
        lines.append(
            "| "
            f"{row['label']} | {row.get('compression_backend') or 'unknown'} | {row['verdict']} | "
            f"{row['successes']}/{row['episodes']} ({_fmt_pct(row['success_rate'])}) | "
            f"{_fmt_pct(row['effective_retention'])} | {_fmt_int(row['effective_visual_tokens_for_llm'])} | "
            f"{_fmt_pct(row['projector_retention'])} | {_fmt_pct(row['internal_retention'])} | "
            f"{_fmt_ms(row['cuda_ms'])} | {_fmt_ratio(row['cuda_speedup_vs_baseline'])} | "
            f"{_fmt_ms(row['wall_ms'])} | {_fmt_ratio(row['wall_speedup_vs_baseline'])} | "
            f"{_fmt_ms(row['llm_saved_ms_vs_baseline'])} | {_fmt_ms(row['hook_ms'])} | "
            f"{_fmt_ms(row['selector_ms'])} | {_fmt_ms(row['model_saved_ms_vs_baseline'])} |"
        )
    lines.append("")

    _write_paper_aligned_sections(lines, methods, baseline)

    lines.append("## Internal Sequence / Retention Semantics\n")
    lines.append("| Strategy | Backend | Effective Visual | Internal Visual Kept/Orig | Internal Retain | Internal Seq Kept/Orig | Projector Retain | Warning |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for row in methods:
        internal_visual = "N/A" if row.get("internal_kept_mean") is None or row.get("internal_original_mean") is None else f"{_fmt_int(row['internal_kept_mean'])}/{_fmt_int(row['internal_original_mean'])}"
        internal_seq = "N/A" if row.get("internal_seq_kept_mean") is None or row.get("internal_seq_original_mean") is None else f"{_fmt_int(row['internal_seq_kept_mean'])}/{_fmt_int(row['internal_seq_original_mean'])}"
        lines.append(
            "| "
            f"{row['label']} | {row.get('compression_backend') or 'unknown'} | "
            f"{_fmt_int(row.get('effective_visual_tokens_for_llm'))} | {internal_visual} | "
            f"{_fmt_pct(row.get('internal_retention'))} | {internal_seq} | "
            f"{_fmt_pct(row.get('projector_retention'))} | {row.get('retention_warning') or ''} |"
        )
    lines.append("")

    lines.append("## Timing Breakdown\n")
    lines.append("| Strategy | Vision | Projector | LLM | Hook total | Hook p50/p95 | Score | Selector | Gather | CUDA total | Model total | Wall total |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in methods:
        hook_tail = "N/A" if row.get("hook_p50_ms") is None else f"{row['hook_p50_ms']:.2f}/{row['hook_p95_ms']:.2f}"
        lines.append(
            "| "
            f"{row['label']} | {_fmt_ms(row['vision_ms'])} | {_fmt_ms(row['projector_ms'])} | "
            f"{_fmt_ms(row['llm_ms'])} | {_fmt_ms(row['hook_ms'])} | {hook_tail} | "
            f"{_fmt_ms(row['score_ms'])} | {_fmt_ms(row['selector_ms'])} | {_fmt_ms(row['gather_ms'])} | "
            f"{_fmt_ms(row['cuda_ms'])} | {_fmt_ms(row['model_ms'])} | {_fmt_ms(row['wall_ms'])} |"
        )
    lines.append("")

    lines.append("## Bottleneck Accounting\n")
    lines.append("| Strategy | LLM saved | Hook / LLM saved | Selector / LLM saved | Surviving model gain | Fallback | Selector | History |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---|")
    for row in methods:
        lines.append(
            "| "
            f"{row['label']} | {_fmt_ms(row['llm_saved_ms_vs_baseline'])} | "
            f"{_fmt_pct(row['hook_pct_of_llm_saved'])} | {_fmt_pct(row['selector_pct_of_llm_saved'])} | "
            f"{_fmt_pct(row['surviving_model_gain_pct_of_llm_saved'])} | "
            f"{row['fallback_steps']}/{row['n_steps']} ({_fmt_pct(row['fallback_rate'])}) | "
            f"{_selector_name(row)} | {_history_name(row)} |"
        )
    lines.append("")

    lines.append("## Per-Task Success\n")
    header = "| Task | " + " | ".join(row["label"] for row in methods) + " |"
    sep = "|---|" + "|".join("---:" for _ in methods) + "|"
    lines.append(header)
    lines.append(sep)
    baseline_tasks = report["per_task"].get(baseline["label"], {})
    for task in report["tasks"]:
        cells = [task]
        base_success = baseline_tasks.get(task, (0, 0))[0]
        for row in methods:
            succ, total = report["per_task"].get(row["label"], {}).get(task, (0, 0))
            delta = succ - base_success if row is not baseline else 0
            suffix = "" if row is baseline or delta == 0 else f" ({delta:+d})"
            cells.append(f"{succ}/{total}{suffix}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Optimization Targets\n")
    for row in methods:
        if row is baseline:
            continue
        hints: List[str] = []
        if (row.get("hook_pct_of_llm_saved") or 0.0) > 0.5:
            hints.append("reduce hook below 50% of LLM saving")
        if (row.get("selector_ms") or 0.0) > 2.0:
            hints.append("reduce selector/pruning below 2 ms")
        if (row.get("cuda_speedup_vs_baseline") or 0.0) < 1.10:
            hints.append("test lower retention or remove overhead before claiming speedup")
        if row.get("success_delta_vs_baseline") is not None and row["success_delta_vs_baseline"] < 0:
            hints.append("inspect tasks with negative per-task delta")
        if not hints:
            hints.append("keep this as a candidate operating point")
        lines.append(f"- **{row['label']}**: " + "; ".join(hints) + ".")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(report: Dict[str, Any], path: Path) -> None:
    fields = [
        "label", "compression_backend", "main_experiment_surface", "audit_probe_only", "verdict", "successes", "episodes", "success_rate",
        "success_delta_vs_baseline", "retention", "effective_retention",
        "projector_retention", "internal_retention", "effective_visual_tokens_for_llm",
        "kept_mean", "original_mean", "effective_kept_mean", "effective_original_mean",
        "projector_kept_mean", "projector_original_mean", "internal_kept_mean", "internal_original_mean",
        "internal_pruned_mean", "internal_seq_original_mean", "internal_seq_kept_mean", "internal_seq_pruned_mean",
        "retention_warning",
        "cuda_ms", "cuda_speedup_vs_baseline", "wall_ms", "wall_speedup_vs_baseline",
        "model_ms", "model_speedup_vs_baseline", "llm_ms", "llm_saved_ms_vs_baseline",
        "hook_ms", "selector_ms", "score_ms", "gather_ms",
        "hook_pct_of_llm_saved", "selector_pct_of_llm_saved",
        "surviving_model_gain_pct_of_llm_saved", "fallback_steps", "n_steps",
        # Paper-aligned mechanism-evidence fields (additive; legacy fields above
        # are unchanged for downstream compatibility).
        "legacy_verdict",
        "lm_prefill_ms", "prefill_speedup_vs_baseline", "prefill_saved_ms_vs_baseline",
        "lm_decode_observed_ms", "decode_speedup_vs_baseline", "decode_saved_ms_vs_baseline",
        "prefill_share_of_llm", "decode_share_of_llm",
        "internal_pruning_layer", "internal_pruned_geo_critical_count", "llm_total_layers",
        "internal_quota_layout_k", "internal_quota_contact_k", "internal_quota_motion_k",
        "internal_quota_semantic_attention_k", "internal_quota_historical_attention_k", "internal_quota_fill_k",
        "internal_selected_by_layout_count", "internal_selected_by_contact_count", "internal_selected_by_motion_count",
        "internal_selected_by_semantic_attention_count", "internal_selected_by_historical_attention_count",
        "internal_selected_by_fill_count",
        "internal_seq_retention", "internal_seq_reduction_pct",
        "theoretical_attention_flop_ratio", "theoretical_attention_speedup",
        "theoretical_linear_flop_ratio", "theoretical_linear_speedup",
        "internal_benefiting_layer_fraction", "internal_effective_attention_flop_ratio",
        "internal_effective_attention_speedup", "internal_effective_linear_speedup",
        "wall_gain_vs_prefill_gain_ratio", "llm_gain_vs_prefill_gain_ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in report["methods"]:
            writer.writerow({field: row.get(field) for field in fields})


def parse_labels(values: List[str]) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"Invalid --label {value!r}; expected method_dir=Display Label")
        key, label = value.split("=", 1)
        labels[key.strip()] = label.strip()
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--method", action="append", default=None, help="Method directory name to include. May be repeated.")
    parser.add_argument("--baseline", default=None, help="Method directory name or display label to use as baseline.")
    parser.add_argument("--label", action="append", default=[], help="Override display label: method_dir=Display Label")
    parser.add_argument("--prefix", default="performance_report", help="Output filename prefix under --root.")
    args = parser.parse_args()

    root = args.root.resolve()
    report = build_report(root, args.method, parse_labels(args.label), baseline_method=args.baseline)
    json_path = root / f"{args.prefix}.json"
    md_path = root / f"{args.prefix}.md"
    csv_path = root / f"{args.prefix}.csv"

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(report, md_path)
    write_csv(report, csv_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
