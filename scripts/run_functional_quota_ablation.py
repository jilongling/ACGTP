#!/usr/bin/env python3
"""Run ACGTP functional-quota ablations.

This is a thin runner around eval_openvla_baseline.py. It keeps the formal core
surface intact while making the new execution-function-aware allocation easy to
probe with no-layout/no-contact/no-motion/no-semantic/global-legacy variants.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pruning.method_profiles import FUNCTIONAL_QUOTA_METHOD_LABELS, method_cli_args, method_profile_labels


PYBIN = "/infini-data/miniconda3/envs/openvla/bin/python"
CKPT = "/infini-data/checkpoints/openvla-7b-finetuned-libero-spatial"
ROOT = PROJECT_ROOT


METHODS = {label: method_cli_args(label) for label in method_profile_labels()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--num_tasks", type=int, default=3)
    ap.add_argument("--num_episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--methods", type=str, default="baseline_none,functional_quota_static_050,functional_no_layout_050,functional_no_contact_050,functional_no_motion_050,legacy_geo_guarded_quota_050")
    ap.add_argument("--build_report", type=lambda x: x.lower() == "true", default=True)
    ap.add_argument(
        "--timing_profile",
        choices=["diagnostic", "latency", "both"],
        default="diagnostic",
        help="diagnostic records module/prefill/decode timings; latency disables timing hooks for cleaner wall-clock; both writes diagnostic/ and latency/ subruns",
    )
    ap.add_argument("--enable_torch_profiler_flops", type=lambda x: x.lower() == "true", default=False)
    ap.add_argument("--torch_profiler_flops_warmup_steps", type=int, default=0)
    ap.add_argument("--torch_profiler_flops_max_steps", type=int, default=1)
    ap.add_argument(
        "--latency_plan_cache",
        type=lambda x: x.lower() == "true",
        default=False,
        help=(
            "Enable the latency-profile pruning-plan reuse cache. Keep disabled "
            "for clean strategy latency comparisons; enable only for a separate "
            "runtime-cache optimization run."
        ),
    )
    ap.add_argument("--latency_plan_cache_max_age", type=int, default=20)
    ap.add_argument("--latency_plan_cache_depth_delta_threshold", type=float, default=0.120)
    ap.add_argument("--latency_plan_cache_gripper_delta_threshold", type=float, default=0.150)
    args = ap.parse_args()

    wanted = [m.strip() for m in args.methods.split(",") if m.strip()]
    profiles = ["diagnostic", "latency"] if args.timing_profile == "both" else [args.timing_profile]

    for profile in profiles:
        out_root = Path(args.output_root) / profile if args.timing_profile == "both" else Path(args.output_root)
        out_root.mkdir(parents=True, exist_ok=True)
        profile_args = [
            "--timing_profile", profile,
            "--measure_submodule_timing", "true" if profile == "diagnostic" else "false",
            "--measure_llm_split_timing", "true" if profile == "diagnostic" else "false",
            "--acgtp_runtime_mode", "fast",
            "--acgtp_full_diagnostics_enabled", "false",
            "--acgtp_latency_plan_cache_enabled", "true" if profile == "latency" and args.latency_plan_cache else "false",
            "--acgtp_latency_plan_cache_max_age", str(args.latency_plan_cache_max_age),
            "--acgtp_latency_plan_cache_depth_delta_threshold", str(args.latency_plan_cache_depth_delta_threshold),
            "--acgtp_latency_plan_cache_gripper_delta_threshold", str(args.latency_plan_cache_gripper_delta_threshold),
        ]

        for label in wanted:
            if label not in METHODS:
                raise SystemExit(f"Unknown method {label!r}. Available: {', '.join(sorted(METHODS))}")
            save_dir = out_root / label
            save_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                PYBIN,
                "scripts/eval_openvla_baseline.py",
                "--model_path", CKPT,
                "--task_suite", "libero_spatial",
                "--num_tasks", str(args.num_tasks),
                "--num_episodes", str(args.num_episodes),
                "--seed", str(args.seed),
                "--save_video", "false",
                "--log_step_metrics", "true",
                "--save_dir", str(save_dir),
            ]
            cmd += profile_args
            if args.enable_torch_profiler_flops:
                cmd += [
                    "--enable_torch_profiler_flops", "true",
                    "--torch_profiler_flops_warmup_steps", str(args.torch_profiler_flops_warmup_steps),
                    "--torch_profiler_flops_max_steps", str(args.torch_profiler_flops_max_steps),
                ]
            if args.max_steps is not None:
                cmd += ["--max_steps", str(args.max_steps)]
            cmd += METHODS[label]
            print(f"\n===== {profile}:{label} =====\n{' '.join(cmd)}", flush=True)
            rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
            print(f"[done] {profile}:{label} rc={rc}", flush=True)
            if rc != 0:
                raise SystemExit(rc)

        if args.build_report:
            report_cmd = [
                PYBIN,
                "scripts/build_performance_report.py",
                "--root", str(out_root),
                "--baseline", "baseline_none",
                "--prefix", "functional_quota_ablation_report",
            ]
            print(f"\n===== {profile}:report =====\n{' '.join(report_cmd)}", flush=True)
            rc = subprocess.run(report_cmd, cwd=str(ROOT)).returncode
            print(f"[report:{profile}] rc={rc}", flush=True)
            if rc != 0:
                raise SystemExit(rc)


            benchmark_cmd = [
                PYBIN,
                "scripts/build_benchmark_metrics.py",
                "--root", str(out_root),
                "--baseline", "baseline_none",
                "--prefix", "benchmark",
                "--latency_scope", "wall" if profile == "latency" else "llm_only",
            ]
            print(f"\n===== {profile}:compact benchmark =====\n{' '.join(benchmark_cmd)}", flush=True)
            rc = subprocess.run(benchmark_cmd, cwd=str(ROOT)).returncode
            print(f"[compact benchmark:{profile}] rc={rc}", flush=True)
            if rc != 0:
                raise SystemExit(rc)


if __name__ == "__main__":
    main()
