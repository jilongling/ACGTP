#!/usr/bin/env python3
"""Small LIBERO success+speed validation for the converged ACGTP surface.

Runs the core comparison (baseline, projector legacy, internal geo_guarded,
internal dynamic) on a few tasks/trials, each into its own subdir, then leaves
results for build_performance_report.py. Read-only w.r.t. the model.
"""

import argparse
import subprocess
import sys
from pathlib import Path

PYBIN = "/infini-data/miniconda3/envs/openvla/bin/python"
CKPT = "/infini-data/checkpoints/openvla-7b-finetuned-libero-spatial"
ROOT = Path("/infini-data/openvla")
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pruning.method_profiles import CORE_SURFACE_METHOD_LABELS, method_cli_args

# label -> extra CLI flags (all share strategy/geometry; backend differs)
METHODS = {label: method_cli_args(label) for label in CORE_SURFACE_METHOD_LABELS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--num_tasks", type=int, default=3)
    ap.add_argument("--num_episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--methods", type=str, default=",".join(METHODS))
    args = ap.parse_args()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    wanted = [m.strip() for m in args.methods.split(",") if m.strip()]

    for label in wanted:
        if label not in METHODS:
            print(f"[skip] unknown method {label}")
            continue
        save_dir = out_root / label
        save_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            PYBIN, "scripts/eval_openvla_baseline.py",
            "--model_path", CKPT,
            "--task_suite", "libero_spatial",
            "--num_tasks", str(args.num_tasks),
            "--num_episodes", str(args.num_episodes),
            "--seed", str(args.seed),
            "--save_video", "false",
            "--log_step_metrics", "true",
            "--save_dir", str(save_dir),
        ]
        if args.max_steps is not None:
            cmd += ["--max_steps", str(args.max_steps)]
        cmd += METHODS[label]
        print(f"\n===== {label} =====\n{' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
        print(f"[done] {label} rc={rc}", flush=True)
        if rc != 0:
            print(f"[warn] {label} exited non-zero", flush=True)

    print("\n[all done] build report with:")
    print(f"  {PYBIN} scripts/build_performance_report.py --root {out_root} --baseline baseline_none")


if __name__ == "__main__":
    main()
