"""Build paired latency-attribution diagnostics for visual token pruning runs."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PAIR_FIELDS = ("task_name", "episode_id", "step_id")
SOURCE_SCAN_FILES = (
    "pruning/methods/functional_quota.py",
    "pruning/internal/backend.py",
    "pruning/runtime/fast.py",
    "pruning/runtime/diagnostics.py",
    "scripts/eval_openvla_baseline.py",
)
SOURCE_PATTERNS = (
    ".cpu(",
    ".numpy(",
    ".item(",
    "tolist(",
    "json.dump",
    "cv2.imwrite",
    "matplotlib",
    "torch.cuda.synchronize",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--baseline", default="baseline_none")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["projector_acgtp_legacy_050", "functional_quota_static_050"],
    )
    parser.add_argument("--report_path", type=Path, default=None)
    parser.add_argument("--paired_csv", type=Path, default=None)
    return parser.parse_args()


def _to_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _first_float(row: Dict[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        value = _to_float(row.get(key))
        if value is not None:
            return value
    return None


def _first_positive_float(row: Dict[str, Any], keys: Sequence[str]) -> Optional[float]:
    value = _first_float(row, keys)
    return value if value is not None and value > 0.0 else None


def _first_value(row: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return mean(vals) if vals else None


def _fmt(value: Optional[float], digits: int = 2) -> str:
    return "unavailable" if value is None else f"{value:.{digits}f}"


def _step_metrics_path(output_root: Path, label: str) -> Path:
    direct = output_root / label / "step_metrics.csv"
    if direct.exists():
        return direct
    nested = list(output_root.glob(f"**/{label}/step_metrics.csv"))
    if nested:
        return nested[0]
    raise FileNotFoundError(f"step_metrics.csv not found for {label!r} under {output_root}")


def read_steps(output_root: Path, label: str) -> List[Dict[str, Any]]:
    path = _step_metrics_path(output_root, label)
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["_method_label"] = label
        row["_step_metrics_path"] = str(path)
    return rows


def pair_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    episode = row.get("episode_id", row.get("episode_idx", ""))
    step = row.get("step_id", row.get("step_idx", ""))
    return (str(row.get("task_name", "")), str(episode), str(step))


def index_rows(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in rows:
        out.setdefault(pair_key(row), row)
    return out


def row_model_forward_ms(row: Dict[str, Any]) -> Optional[float]:
    return _first_float(row, ["model_forward_total_ms", "total_model_forward_time_ms", "model_forward_ms"])


def row_wall_step_ms(row: Dict[str, Any]) -> Optional[float]:
    return _first_float(row, ["wall_step_ms", "total_step_wall_time_ms", "end_to_end_step_wall_time_ms"])


def row_llm_total_ms(row: Dict[str, Any]) -> Optional[float]:
    value = _first_positive_float(row, ["llm_total_ms", "llm_forward_time_ms"])
    if value is not None:
        return value
    prefill = _first_positive_float(row, ["llm_prefill_ms", "llm_prefill_time_ms"])
    decode = _first_positive_float(row, ["llm_decode_ms", "llm_decode_time_ms"])
    if prefill is not None or decode is not None:
        return float(prefill or 0.0) + float(decode or 0.0)
    return None


def row_selector_ms(row: Dict[str, Any]) -> Optional[float]:
    return _first_positive_float(row, ["selector_total_ms", "internal_selector_total_ms", "selection_ms", "hook_total_ms"])


def build_pairs(
    baseline_rows: List[Dict[str, Any]],
    method_rows: List[Dict[str, Any]],
    method: str,
) -> List[Dict[str, Any]]:
    baseline = index_rows(baseline_rows)
    pairs = []
    for row in method_rows:
        key = pair_key(row)
        base = baseline.get(key)
        if base is None:
            continue
        base_model = row_model_forward_ms(base)
        method_model = row_model_forward_ms(row)
        base_llm = row_llm_total_ms(base)
        method_llm = row_llm_total_ms(row)
        selector = row_selector_ms(row)
        paired = {
            "method": method,
            "task_name": key[0],
            "episode_id": key[1],
            "step_id": key[2],
            "baseline_model_forward_ms": base_model,
            "method_model_forward_ms": method_model,
            "paired_model_forward_delta_ms": (
                method_model - base_model if base_model is not None and method_model is not None else None
            ),
            "baseline_llm_total_ms": base_llm,
            "method_llm_total_ms": method_llm,
            "paired_llm_total_delta_ms": (
                method_llm - base_llm if base_llm is not None and method_llm is not None else None
            ),
            "paired_selector_overhead_ms": selector,
            "paired_net_saved_ms": (
                base_llm - method_llm - float(selector or 0.0)
                if base_llm is not None and method_llm is not None
                else None
            ),
            "baseline_wall_step_ms": row_wall_step_ms(base),
            "method_wall_step_ms": row_wall_step_ms(row),
        }
        pairs.append(paired)
    return pairs


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "n_steps": len(rows),
        "model_forward_ms": _mean(row_model_forward_ms(r) for r in rows),
        "wall_step_ms": _mean(row_wall_step_ms(r) for r in rows),
        "llm_total_ms": _mean(row_llm_total_ms(r) for r in rows),
        "selector_total_ms": _mean(row_selector_ms(r) for r in rows),
        "visual_before": _mean(
            _first_float(r, ["visual_token_count_before_prune", "original_visual_tokens", "num_visual_tokens_original"])
            for r in rows
        ),
        "visual_after": _mean(
            _first_float(r, ["visual_token_count_after_prune", "kept_visual_tokens", "effective_visual_tokens_for_llm"])
            for r in rows
        ),
        "llm_input_seq_len": _mean(_first_float(r, ["llm_input_seq_len", "multimodal_seq_len"]) for r in rows),
        "llm_seq_after_prune": _mean(_first_float(r, ["llm_input_seq_len_after_prune"]) for r in rows),
        "decoder_first_layer_seq_len": _mean(_first_float(r, ["decoder_first_layer_seq_len"]) for r in rows),
        "score_depth_edge_ms": _mean(_first_float(r, ["score_depth_edge_ms", "internal_score_depth_edge_ms"]) for r in rows),
        "score_layout_ms": _mean(_first_float(r, ["score_layout_ms", "internal_score_layout_ms"]) for r in rows),
        "score_contact_ms": _mean(_first_float(r, ["score_contact_ms", "internal_score_contact_ms"]) for r in rows),
        "score_motion_ms": _mean(_first_float(r, ["score_motion_ms", "internal_score_motion_ms"]) for r in rows),
        "quota_alloc_ms": _mean(_first_float(r, ["quota_alloc_ms", "internal_quota_alloc_ms"]) for r in rows),
        "quota_merge_ms": _mean(_first_float(r, ["quota_merge_ms", "internal_quota_merge_ms"]) for r in rows),
        "debug_record_ms": _mean(_first_float(r, ["debug_record_ms", "internal_debug_record_ms"]) for r in rows),
        "apply_prune_ms": _mean(_first_float(r, ["apply_prune_ms", "internal_apply_prune_ms"]) for r in rows),
        "attn_implementation_counts": Counter(str(_first_value(r, ["attn_implementation"]) or "") for r in rows),
        "output_attentions_true": sum(1 for r in rows if _to_bool(_first_value(r, ["output_attentions"])) is True),
        "output_attentions_effective_true": sum(
            1 for r in rows if _to_bool(_first_value(r, ["output_attentions_effective"])) is True
        ),
        "latency_contamination_attention_true": sum(
            1 for r in rows if _to_bool(_first_value(r, ["latency_contamination_attention"])) is True
        ),
        "latency_contamination_debug_true": sum(
            1 for r in rows if _to_bool(_first_value(r, ["latency_contamination_debug"])) is True
        ),
    }


def summarize_pairs(pairs: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    return {
        "n_pairs": float(len(pairs)),
        "paired_model_forward_delta_ms": _mean(_to_float(p.get("paired_model_forward_delta_ms")) for p in pairs),
        "paired_llm_total_delta_ms": _mean(_to_float(p.get("paired_llm_total_delta_ms")) for p in pairs),
        "paired_selector_overhead_ms": _mean(_to_float(p.get("paired_selector_overhead_ms")) for p in pairs),
        "paired_net_saved_ms": _mean(_to_float(p.get("paired_net_saved_ms")) for p in pairs),
    }


def scan_sources(project_root: Path) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for rel in SOURCE_SCAN_FILES:
        path = project_root / rel
        if not path.exists():
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line_no, line in enumerate(lines, start=1):
            for pattern in SOURCE_PATTERNS:
                if pattern in line:
                    if rel.endswith("functional_quota.py") and pattern in {".cpu(", "tolist(", ".item("}:
                        category = "selector_core_scalar_or_index_transfer"
                    elif rel.endswith("backend.py") and pattern in {".cpu(", "tolist(", ".item("}:
                        category = "internal_debug_history_or_scalar_audit"
                    elif pattern == "torch.cuda.synchronize":
                        category = "timing_hook_synchronization"
                    elif pattern in {"cv2.imwrite", "matplotlib", "json.dump"}:
                        category = "debug_or_visualization_output"
                    else:
                        category = "diagnostic_or_unknown"
                    hits.append(
                        {
                            "file": rel,
                            "line": line_no,
                            "pattern": pattern,
                            "category": category,
                        }
                    )
    return hits


def write_paired_csv(path: Path, pairs_by_method: Dict[str, List[Dict[str, Any]]]) -> None:
    rows = [row for rows in pairs_by_method.values() for row in rows]
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_report(
    output_root: Path,
    baseline_label: str,
    rows_by_method: Dict[str, List[Dict[str, Any]]],
    pairs_by_method: Dict[str, List[Dict[str, Any]]],
    source_hits: List[Dict[str, Any]],
    paired_csv: Optional[Path],
) -> str:
    summaries = {label: summarize_rows(rows) for label, rows in rows_by_method.items()}
    pair_summaries = {label: summarize_pairs(rows) for label, rows in pairs_by_method.items()}
    lines: List[str] = []
    lines.append("# Visual Token Pruning Latency Attribution")
    lines.append("")
    lines.append(f"- output_root: `{output_root}`")
    lines.append(f"- baseline: `{baseline_label}`")
    if paired_csv is not None:
        lines.append(f"- paired_csv: `{paired_csv}`")
    lines.append("")
    lines.append("## Method Summary")
    lines.append("")
    lines.append(
        "| method | steps | visual before | visual after | llm input seq | seq after prune | decoder first layer | model ms | llm ms | selector ms | debug ms | apply ms |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, summary in summaries.items():
        lines.append(
            "| {label} | {steps} | {vb} | {va} | {llm_in} | {seq_after} | {dec0} | {model} | {llm} | {selector} | {debug} | {apply} |".format(
                label=label,
                steps=int(summary["n_steps"]),
                vb=_fmt(summary["visual_before"]),
                va=_fmt(summary["visual_after"]),
                llm_in=_fmt(summary["llm_input_seq_len"]),
                seq_after=_fmt(summary["llm_seq_after_prune"]),
                dec0=_fmt(summary["decoder_first_layer_seq_len"]),
                model=_fmt(summary["model_forward_ms"]),
                llm=_fmt(summary["llm_total_ms"]),
                selector=_fmt(summary["selector_total_ms"]),
                debug=_fmt(summary["debug_record_ms"]),
                apply=_fmt(summary["apply_prune_ms"]),
            )
        )
    lines.append("")
    lines.append("## Paired Timing")
    lines.append("")
    lines.append("| method | pairs | model delta ms | llm delta ms | selector overhead ms | net saved ms |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for label, summary in pair_summaries.items():
        lines.append(
            "| {label} | {pairs} | {model_delta} | {llm_delta} | {selector} | {net} |".format(
                label=label,
                pairs=int(summary["n_pairs"] or 0),
                model_delta=_fmt(summary["paired_model_forward_delta_ms"]),
                llm_delta=_fmt(summary["paired_llm_total_delta_ms"]),
                selector=_fmt(summary["paired_selector_overhead_ms"]),
                net=_fmt(summary["paired_net_saved_ms"]),
            )
        )
    lines.append("")
    lines.append("## Attention Backend Audit")
    lines.append("")
    for label, summary in summaries.items():
        attn_counts = ", ".join(f"{k or 'unknown'}={v}" for k, v in summary["attn_implementation_counts"].items())
        lines.append(
            f"- `{label}`: attn_implementation {{{attn_counts}}}; "
            f"output_attentions_true={summary['output_attentions_true']}; "
            f"output_attentions_effective_true={summary['output_attentions_effective_true']}; "
            f"attention_contamination_steps={summary['latency_contamination_attention_true']}; "
            f"debug_contamination_steps={summary['latency_contamination_debug_true']}."
        )
    lines.append("")
    lines.append("## Source Contamination Scan")
    lines.append("")
    source_counts = Counter(hit["category"] for hit in source_hits)
    if source_counts:
        for category, count in sorted(source_counts.items()):
            lines.append(f"- {category}: {count}")
    else:
        lines.append("- No matching source patterns found.")
    lines.append("")
    lines.append("Representative hits:")
    for hit in source_hits[:20]:
        lines.append(f"- `{hit['file']}:{hit['line']}` `{hit['pattern']}` -> {hit['category']}")
    lines.append("")
    lines.append("## Diagnostic Answers")
    lines.append("")
    baseline = summaries.get(baseline_label, {})
    keep_methods = [m for m in summaries if m != baseline_label]
    baseline_seq = baseline.get("llm_seq_after_prune") or baseline.get("llm_input_seq_len")
    seq_answers = []
    for label in keep_methods:
        summary = summaries[label]
        seq_after = summary.get("llm_seq_after_prune")
        dec0 = summary.get("decoder_first_layer_seq_len")
        llm_in = summary.get("llm_input_seq_len")
        seq_answers.append(
            f"`{label}` llm_input={_fmt(llm_in)}, decoder_first_layer={_fmt(dec0)}, seq_after_prune={_fmt(seq_after)}"
        )
    lines.append(
        "1. keep050 decoder sequence: baseline is "
        f"{_fmt(baseline_seq)}; " + "; ".join(seq_answers) + "."
    )
    functional = summaries.get("functional_quota_static_050")
    if functional:
        overheads = {
            "selector": functional.get("selector_total_ms"),
            "debug": functional.get("debug_record_ms"),
            "quota_alloc": functional.get("quota_alloc_ms"),
            "apply_prune": functional.get("apply_prune_ms"),
        }
        dominant = max((k for k, v in overheads.items() if isinstance(v, (int, float))), key=lambda k: float(overheads[k]), default="unavailable")
        lines.append(
            "2. functional quota overhead: dominant measured bucket is "
            f"`{dominant}`; selector={_fmt(functional.get('selector_total_ms'))}, "
            f"quota_alloc={_fmt(functional.get('quota_alloc_ms'))}, debug={_fmt(functional.get('debug_record_ms'))}, "
            f"apply={_fmt(functional.get('apply_prune_ms'))}."
        )
    else:
        lines.append("2. functional quota overhead: unavailable because method rows were not found.")
    legacy = pair_summaries.get("projector_acgtp_legacy_050")
    if legacy:
        lines.append(
            "3. legacy_050 speedup: paired model delta is "
            f"{_fmt(legacy.get('paired_model_forward_delta_ms'))} ms and paired net saved is "
            f"{_fmt(legacy.get('paired_net_saved_ms'))} ms; small gains usually mean most model_forward time is outside prunable LLM prefill or selector/apply overhead eats savings."
        )
    else:
        lines.append("3. legacy_050 speedup: unavailable because paired rows were not found.")
    lines.append(
        "4. model_forward scope: `model_forward_total_ms` is the full `model.predict_action` timer, so it includes vision encoder, projector, LLM, action decode, and pruning hooks when present."
    )
    lines.append(
        "5. next step: use diagnostic profile for llm_prefill/decode attribution, latency profile for clean wall/model timing, and optimize only after paired_net_saved_ms identifies whether selector/debug/attention is the bottleneck."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    labels = [args.baseline] + [m for m in args.methods if m != args.baseline]
    rows_by_method = {label: read_steps(output_root, label) for label in labels}
    baseline_rows = rows_by_method[args.baseline]
    pairs_by_method = {
        label: build_pairs(baseline_rows, rows_by_method[label], label)
        for label in labels
        if label != args.baseline
    }
    report_path = args.report_path or (output_root / "latency_attribution_report.md")
    paired_csv = args.paired_csv or (output_root / "paired_latency_attribution.csv")
    write_paired_csv(paired_csv, pairs_by_method)
    source_hits = scan_sources(Path(__file__).resolve().parents[1])
    report = render_report(output_root, args.baseline, rows_by_method, pairs_by_method, source_hits, paired_csv)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote {report_path}")
    print(f"Wrote {paired_csv}")


if __name__ == "__main__":
    main()
