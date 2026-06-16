"""
compare_results.py

Post-processing script to compare two evaluation summaries and compute relative metrics.

Usage:
    python scripts/compare_results.py \
        --baseline outputs/baseline_run/summary.json \
        --method outputs/method_run/summary.json \
        --output outputs/comparison_report.md

Input:
    - baseline summary.json (typically from the unmodified OpenVLA baseline)
    - method summary.json (from a token-pruning or other optimization method)

Computed relative metrics:
    - speedup_model_forward = baseline_fw_time / method_fw_time
    - speedup_step_wall = baseline_step_wall / method_step_wall
    - speedup_cuda = baseline_cuda / method_cuda
    - latency_reduction_percent = (1 - method / baseline) * 100
    - relative_success_rate = method_success_rate / baseline_success_rate
    - performance_preservation_ratio = relative_success_rate * 100
    - success_drop_percent = (1 - relative_success_rate) * 100
    - flop_ratio = method_flops_T / baseline_flops_T
    - flop_reduction_percent = (1 - method_flops / baseline_flops) * 100
    - token_retention_ratio = method_tokens_kept / baseline_tokens_original
    - token_pruning_ratio = 1 - token_retention_ratio

The script does NOT modify either input file.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def load_summary(path: str) -> Dict[str, Any]:
    """Load and return a summary JSON file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")
    with open(p) as f:
        data = json.load(f)
    return data


def safe_div(a: float, b: float, default: Optional[float] = None) -> Optional[float]:
    """Safely divide a / b, returning default if b is 0 or None."""
    if a is None or b is None or b == 0:
        return default
    return a / b


def safe_pct(a: float, b: float, default: Optional[float] = None) -> Optional[float]:
    """Compute (1 - a/b) * 100, returning default if b is 0 or None."""
    ratio = safe_div(a, b)
    if ratio is None:
        return default
    return (1 - ratio) * 100


def compute_comparison(baseline: Dict[str, Any], method: Dict[str, Any]) -> Dict[str, Any]:
    """Compute relative metrics between baseline and method summaries."""

    comparison = {
        "baseline_checkpoint": baseline.get("checkpoint", "unknown"),
        "method_checkpoint": method.get("checkpoint", "unknown"),
        "baseline_task_suite": baseline.get("task_suite", "unknown"),
        "method_task_suite": method.get("task_suite", "unknown"),
    }

    # --- Success rates ---
    base_sr = baseline.get("overall_success_rate", 0.0)
    meth_sr = method.get("overall_success_rate", 0.0)
    comparison["baseline_success_rate"] = base_sr
    comparison["method_success_rate"] = meth_sr
    comparison["relative_success_rate"] = safe_div(meth_sr, base_sr)
    comparison["performance_preservation_ratio"] = (
        comparison["relative_success_rate"] * 100
        if comparison["relative_success_rate"] is not None else None
    )
    comparison["success_drop_percent"] = safe_pct(meth_sr, base_sr)

    # --- Model forward time ---
    base_fw = baseline.get("mean_model_forward_time_ms", 0.0)
    meth_fw = method.get("mean_model_forward_time_ms", 0.0)
    comparison["baseline_mean_model_forward_time_ms"] = base_fw
    comparison["method_mean_model_forward_time_ms"] = meth_fw
    comparison["speedup_model_forward"] = safe_div(base_fw, meth_fw)
    comparison["model_forward_latency_reduction_percent"] = safe_pct(meth_fw, base_fw)

    # --- Step wall time ---
    base_sw = baseline.get("mean_step_wall_time_ms", 0.0)
    meth_sw = method.get("mean_step_wall_time_ms", 0.0)
    comparison["baseline_mean_step_wall_time_ms"] = base_sw
    comparison["method_mean_step_wall_time_ms"] = meth_sw
    comparison["speedup_step_wall"] = safe_div(base_sw, meth_sw)
    comparison["step_wall_latency_reduction_percent"] = safe_pct(meth_sw, base_sw)

    # --- CUDA latency ---
    base_cuda = baseline.get("mean_cuda_latency_ms")
    meth_cuda = method.get("mean_cuda_latency_ms")
    comparison["baseline_mean_cuda_latency_ms"] = base_cuda
    comparison["method_mean_cuda_latency_ms"] = meth_cuda
    comparison["speedup_cuda"] = safe_div(base_cuda, meth_cuda)
    comparison["cuda_latency_reduction_percent"] = safe_pct(meth_cuda, base_cuda)

    # --- Overall speedup (use model forward as primary) ---
    comparison["speedup"] = comparison["speedup_model_forward"]
    comparison["latency_reduction_percent"] = comparison["model_forward_latency_reduction_percent"]

    # --- Control frequency ---
    base_freq = baseline.get("control_frequency_hz")
    meth_freq = method.get("control_frequency_hz")
    comparison["baseline_control_frequency_hz"] = base_freq
    comparison["method_control_frequency_hz"] = meth_freq
    comparison["frequency_speedup"] = safe_div(meth_freq, base_freq)

    # --- FLOPs ---
    base_flops = baseline.get("flops_T")
    meth_flops = method.get("flops_T")
    comparison["baseline_flops_T"] = base_flops
    comparison["method_flops_T"] = meth_flops
    comparison["flop_ratio"] = safe_div(meth_flops, base_flops)
    comparison["flop_reduction_percent"] = safe_pct(meth_flops, base_flops)

    # --- Visual tokens ---
    base_orig = baseline.get("num_visual_tokens_original")
    meth_kept = method.get("num_visual_tokens_kept")
    comparison["baseline_num_visual_tokens_original"] = base_orig
    comparison["method_num_visual_tokens_kept"] = meth_kept
    comparison["token_retention_ratio"] = safe_div(meth_kept, base_orig)
    comparison["token_pruning_ratio"] = (
        1 - comparison["token_retention_ratio"]
        if comparison["token_retention_ratio"] is not None else None
    )

    # --- Speedup of LLM / Vision / Projector (if available) ---
    for key, label in [
        ("mean_vision_encoder_time_ms", "vision_encoder"),
        ("mean_llm_forward_time_ms", "llm_forward"),
        ("mean_projector_time_ms", "projector"),
    ]:
        b = baseline.get(key)
        m = method.get(key)
        comparison[f"baseline_{key}"] = b
        comparison[f"method_{key}"] = m
        comparison[f"speedup_{label}"] = safe_div(b, m)

    # --- Memory ---
    comparison["baseline_max_gpu_memory_mb"] = baseline.get("max_gpu_memory_mb", 0.0)
    comparison["method_max_gpu_memory_mb"] = method.get("max_gpu_memory_mb", 0.0)

    # --- Token pruning config ---
    comparison["baseline_pruning_method"] = baseline.get("pruning_method", "none")
    comparison["method_pruning_method"] = method.get("pruning_method", "none")
    comparison["baseline_keep_ratio"] = baseline.get("keep_ratio", 1.0)
    comparison["method_keep_ratio"] = method.get("keep_ratio", 1.0)

    # --- Action smoothness (preservation) ---
    base_jerk = baseline.get("mean_jerk_action", 0.0)
    meth_jerk = method.get("mean_jerk_action", 0.0)
    comparison["baseline_mean_jerk_action"] = base_jerk
    comparison["method_mean_jerk_action"] = meth_jerk
    comparison["jerk_change_percent"] = safe_pct(meth_jerk, base_jerk)

    return comparison


def format_number(val: Optional[float], suffix: str = "", precision: int = 2) -> str:
    """Format a number with a suffix, or return 'N/A'."""
    if val is None:
        return "N/A"
    return f"{val:.{precision}f}{suffix}"


def print_comparison(comparison: Dict[str, Any]) -> None:
    """Print a human-readable comparison report."""
    print("=" * 70)
    print("COMPARISON REPORT: Baseline vs Method")
    print("=" * 70)
    print(f"  Baseline: {comparison.get('baseline_checkpoint', 'N/A')}")
    print(f"  Method:   {comparison.get('method_checkpoint', 'N/A')}")
    print(f"  Task suite: baseline={comparison.get('baseline_task_suite', 'N/A')}, "
          f"method={comparison.get('method_task_suite', 'N/A')}")
    print()
    print("--- Success Rate ---")
    print(f"  Baseline:      {format_number(comparison.get('baseline_success_rate'), '%', 1)}")
    print(f"  Method:        {format_number(comparison.get('method_success_rate'), '%', 1)}")
    print(f"  Relative SR:   {format_number(comparison.get('relative_success_rate'))}")
    print(f"  Preservation:  {format_number(comparison.get('performance_preservation_ratio'), '%', 1)}")
    print(f"  Drop:          {format_number(comparison.get('success_drop_percent'), '%', 1)}")
    print()
    print("--- Timing ---")
    print(f"  Model Forward (ms): baseline={format_number(comparison.get('baseline_mean_model_forward_time_ms'))}, "
          f"method={format_number(comparison.get('method_mean_model_forward_time_ms'))}, "
          f"speedup={format_number(comparison.get('speedup_model_forward'))}x")
    print(f"  Step Wall (ms):     baseline={format_number(comparison.get('baseline_mean_step_wall_time_ms'))}, "
          f"method={format_number(comparison.get('method_mean_step_wall_time_ms'))}, "
          f"speedup={format_number(comparison.get('speedup_step_wall'))}x")
    cuda_base = comparison.get("baseline_mean_cuda_latency_ms")
    cuda_meth = comparison.get("method_mean_cuda_latency_ms")
    print(f"  CUDA Latency (ms):  baseline={format_number(cuda_base)}, "
          f"method={format_number(cuda_meth)}, "
          f"speedup={format_number(comparison.get('speedup_cuda'))}x")
    print(f"  Control Freq (Hz):  baseline={format_number(comparison.get('baseline_control_frequency_hz'))}, "
          f"method={format_number(comparison.get('method_control_frequency_hz'))}, "
          f"speedup={format_number(comparison.get('frequency_speedup'))}x")
    print()
    print("--- FLOPs ---")
    print(f"  Baseline: {comparison.get('baseline_flops_T', 'N/A')} T")
    print(f"  Method:   {comparison.get('method_flops_T', 'N/A')} T")
    print(f"  Ratio:    {format_number(comparison.get('flop_ratio'))}")
    print(f"  Reduction: {format_number(comparison.get('flop_reduction_percent'), '%', 1)}")
    print()
    print("--- Visual Tokens ---")
    print(f"  Baseline original: {comparison.get('baseline_num_visual_tokens_original', 'N/A')}")
    print(f"  Method kept:       {comparison.get('method_num_visual_tokens_kept', 'N/A')}")
    print(f"  Retention ratio:   {format_number(comparison.get('token_retention_ratio'))}")
    print(f"  Pruning ratio:     {format_number(comparison.get('token_pruning_ratio'))}")
    print()
    print("--- GPU Memory ---")
    print(f"  Baseline: {format_number(comparison.get('baseline_max_gpu_memory_mb'), ' MB')}")
    print(f"  Method:   {format_number(comparison.get('method_max_gpu_memory_mb'), ' MB')}")
    print()
    print("--- Sub-Module Speedups ---")
    for label in ["vision_encoder", "llm_forward", "projector"]:
        b = comparison.get(f"baseline_mean_{label}_time_ms")
        m = comparison.get(f"method_mean_{label}_time_ms")
        s = comparison.get(f"speedup_{label}")
        print(f"  {label}: baseline={format_number(b)}ms, method={format_number(m)}ms, "
              f"speedup={format_number(s)}x")
    print("=" * 70)


def write_json_report(comparison: Dict[str, Any], output_path: str) -> None:
    """Save comparison as JSON."""
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"JSON report saved to: {output_path}")


def write_markdown_report(comparison: Dict[str, Any], output_path: str) -> None:
    """Save comparison as Markdown."""
    with open(output_path, "w") as f:
        f.write("# Evaluation Comparison Report\n\n")
        f.write(f"**Baseline**: `{comparison.get('baseline_checkpoint', 'N/A')}`  \n")
        f.write(f"**Method**: `{comparison.get('method_checkpoint', 'N/A')}`  \n")
        f.write(f"**Task Suite**: baseline={comparison.get('baseline_task_suite')}, "
                f"method={comparison.get('method_task_suite')}  \n\n")

        f.write("## Success Rate\n\n")
        f.write("| Metric | Baseline | Method | Relative |\n")
        f.write("|--------|----------|--------|----------|\n")
        f.write(f"| Success Rate | {comparison.get('baseline_success_rate', 'N/A'):.1%} | "
                f"{comparison.get('method_success_rate', 'N/A'):.1%} | "
                f"{comparison.get('relative_success_rate', 'N/A')} |\n")
        f.write(f"| Preservation Ratio | - | - | "
                f"{comparison.get('performance_preservation_ratio', 'N/A'):.1f}% |\n")
        f.write(f"| Success Drop | - | - | "
                f"{comparison.get('success_drop_percent', 'N/A'):.1f}% |\n\n")

        f.write("## Timing\n\n")
        f.write("| Metric | Baseline (ms) | Method (ms) | Speedup | Reduction |\n")
        f.write("|--------|--------------|-------------|---------|-----------|\n")
        f.write(f"| Model Forward | {comparison.get('baseline_mean_model_forward_time_ms', 'N/A')} | "
                f"{comparison.get('method_mean_model_forward_time_ms', 'N/A')} | "
                f"{comparison.get('speedup_model_forward', 'N/A')}x | "
                f"{comparison.get('model_forward_latency_reduction_percent', 'N/A'):.1f}% |\n")
        f.write(f"| Step Wall | {comparison.get('baseline_mean_step_wall_time_ms', 'N/A')} | "
                f"{comparison.get('method_mean_step_wall_time_ms', 'N/A')} | "
                f"{comparison.get('speedup_step_wall', 'N/A')}x | "
                f"{comparison.get('step_wall_latency_reduction_percent', 'N/A'):.1f}% |\n")
        f.write(f"| CUDA Latency | {comparison.get('baseline_mean_cuda_latency_ms', 'N/A')} | "
                f"{comparison.get('method_mean_cuda_latency_ms', 'N/A')} | "
                f"{comparison.get('speedup_cuda', 'N/A')}x | "
                f"{comparison.get('cuda_latency_reduction_percent', 'N/A'):.1f}% |\n")
        f.write(f"| Control Freq (Hz) | {comparison.get('baseline_control_frequency_hz', 'N/A')} | "
                f"{comparison.get('method_control_frequency_hz', 'N/A')} | "
                f"{comparison.get('frequency_speedup', 'N/A')}x | - |\n\n")

        f.write("## FLOPs\n\n")
        f.write(f"- Baseline: {comparison.get('baseline_flops_T', 'N/A')} T\n")
        f.write(f"- Method: {comparison.get('method_flops_T', 'N/A')} T\n")
        f.write(f"- Ratio: {comparison.get('flop_ratio', 'N/A')}\n")
        f.write(f"- Reduction: {comparison.get('flop_reduction_percent', 'N/A'):.1f}%\n\n")

        f.write("## Visual Tokens\n\n")
        f.write(f"- Baseline original: {comparison.get('baseline_num_visual_tokens_original', 'N/A')}\n")
        f.write(f"- Method kept: {comparison.get('method_num_visual_tokens_kept', 'N/A')}\n")
        f.write(f"- Retention ratio: {comparison.get('token_retention_ratio', 'N/A')}\n")
        f.write(f"- Pruning ratio: {comparison.get('token_pruning_ratio', 'N/A')}\n\n")

        f.write("## GPU Memory\n\n")
        f.write(f"- Baseline: {comparison.get('baseline_max_gpu_memory_mb', 'N/A'):.0f} MB\n")
        f.write(f"- Method: {comparison.get('method_max_gpu_memory_mb', 'N/A'):.0f} MB\n\n")

        f.write("## Sub-Module Speedups\n\n")
        f.write("| Module | Baseline (ms) | Method (ms) | Speedup |\n")
        f.write("|--------|--------------|-------------|---------|\n")
        for label in ["vision_encoder", "llm_forward", "projector"]:
            b = comparison.get(f"baseline_mean_{label}_time_ms")
            m = comparison.get(f"method_mean_{label}_time_ms")
            s = comparison.get(f"speedup_{label}")
            f.write(f"| {label} | {b} | {m} | {s}x |\n")

    print(f"Markdown report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two evaluation summaries and compute relative metrics."
    )
    parser.add_argument(
        "--baseline", "-b", type=str, required=True,
        help="Path to baseline summary.json"
    )
    parser.add_argument(
        "--method", "-m", type=str, required=True,
        help="Path to method (e.g., token-pruned) summary.json"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output path for JSON report (optional)"
    )
    parser.add_argument(
        "--format", "-f", type=str, choices=["json", "markdown", "both"],
        default="both",
        help="Output format: json, markdown, or both (default: both)"
    )

    args = parser.parse_args()

    print(f"Loading baseline: {args.baseline}")
    baseline = load_summary(args.baseline)

    print(f"Loading method: {args.method}")
    method = load_summary(args.method)

    comparison = compute_comparison(baseline, method)
    print_comparison(comparison)

    if args.output:
        base_path = Path(args.output)
        if args.format in ("json", "both"):
            json_path = base_path.with_suffix(".json") if base_path.suffix else Path(str(base_path) + ".json")
            write_json_report(comparison, str(json_path))
        if args.format in ("markdown", "both"):
            md_path = base_path.with_suffix(".md") if base_path.suffix else Path(str(base_path) + ".md")
            write_markdown_report(comparison, str(md_path))
    elif args.format == "json":
        print("\n--- JSON Output ---")
        print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
