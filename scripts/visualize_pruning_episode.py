"""Run a focused episode/step pruning visualization.

This is a thin wrapper around eval_openvla_baseline.py with pruning
visualization enabled. It does not alter model/action code.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save pruning visualization for one episode/step")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--method", type=str, default="robot_geo_corridor")
    parser.add_argument("--task_suite", type=str, default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--keep_ratio", type=float, default=0.75)
    parser.add_argument("--max_steps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output_dir or PROJECT_ROOT / "outputs" / "pruning_visualizations" / f"{args.method}_{time.strftime('%Y%m%d_%H%M%S')}"
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "eval_openvla_baseline.py"),
        "--model_path",
        str(args.checkpoint),
        "--task_suite",
        args.task_suite,
        "--num_tasks",
        str(args.task_id + 1),
        "--num_episodes",
        "1",
        "--seed",
        str(args.seed),
        "--save_dir",
        str(out),
        "--save_video",
        "false",
        "--log_step_metrics",
        "true",
        "--use_wandb",
        "false",
        "--pruning_strategy",
        args.method,
        "--pruning_enabled",
        "true",
        "--geometry_enabled",
        "true",
        "--keep_ratio",
        str(args.keep_ratio),
        "--save_pruning_vis",
        "true",
        "--pruning_vis_episode",
        str(args.episode),
        "--pruning_vis_step",
        str(args.step),
        "--save_geometry_vis",
        "false",
    ]
    if args.method == "robot_geo_dynamic":
        cmd[cmd.index("--keep_ratio") + 1] = "1.0"
    if args.max_steps is not None:
        cmd += ["--max_steps", str(args.max_steps)]
    print(" ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    print(f"Visualization root: {out / 'pruning_visualizations'}")


if __name__ == "__main__":
    main()
