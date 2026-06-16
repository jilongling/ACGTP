"""Unified runner for OpenVLA external pruning ablations.

The runner launches the existing evaluation entrypoint with method-specific CLI
flags. It does not import or modify model/action code.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pruning.strategy_registry import is_legacy_strategy

DEFAULT_METHODS = [
    "baseline_none_keep100",
    "random_keep075",
    "depth_edge_fast_keep075",
    "contact_budget_keep075",
    "robot_geo_rule_v0_keep075",
    "robot_geo_dynamic_v0",
    "robot_geo_temporal_v0",
    "robot_geo_hybrid_v0_keep075",
]


METHOD_CONFIGS: Dict[str, Dict[str, Any]] = {
    "baseline_none_keep100": {
        "pruning_strategy": "none",
        "pruning_enabled": "false",
        "geometry_enabled": "false",
        "keep_ratio": 1.0,
    },
    "random_keep075": {
        "pruning_strategy": "random",
        "pruning_enabled": "true",
        "geometry_enabled": "false",
        "keep_ratio": 0.75,
    },
    "uniform_grid_keep075": {
        "pruning_strategy": "uniform_grid",
        "pruning_enabled": "true",
        "geometry_enabled": "false",
        "keep_ratio": 0.75,
    },
    "depth_edge_fast_keep075": {
        "pruning_strategy": "depth_edge_fast",
        "pruning_enabled": "true",
        "geometry_enabled": "true",
        "keep_ratio": 0.75,
    },
    "depth_edge_fast_diverse_keep075": {
        "pruning_strategy": "depth_edge_fast_diverse",
        "pruning_enabled": "true",
        "geometry_enabled": "true",
        "keep_ratio": 0.75,
    },
    "robot_geo_near_keep075": {
        "pruning_strategy": "robot_geo_near",
        "pruning_enabled": "true",
        "geometry_enabled": "true",
        "keep_ratio": 0.75,
    },
    "robot_geo_corridor_keep075": {
        "pruning_strategy": "robot_geo_corridor",
        "pruning_enabled": "true",
        "geometry_enabled": "true",
        "keep_ratio": 0.75,
    },
    "robot_geo_contact_budget_keep075": {
        "pruning_strategy": "robot_geo_contact_budget",
        "pruning_enabled": "true",
        "geometry_enabled": "true",
        "keep_ratio": 0.75,
    },
    "contact_budget_keep075": {
        "pruning_strategy": "robot_geo_contact_budget",
        "pruning_enabled": "true",
        "geometry_enabled": "true",
        "keep_ratio": 0.75,
    },
    "robot_geo_rule_v0_keep090": {
        "pruning_strategy": "robot_geo_rule_v0",
        "pruning_enabled": "true",
        "geometry_enabled": "true",
        "keep_ratio": 0.90,
        "enable_robot_geo_expert": "true",
        "robot_geo_mode": "rule_v0",
    },
    "robot_geo_dynamic_v0": {
        "pruning_strategy": "robot_geo_dynamic_v0",
        "pruning_enabled": "true",
        "geometry_enabled": "true",
        "keep_ratio": 1.0,
        "enable_robot_geo_expert": "true",
        "robot_geo_mode": "dynamic_rule_v0",
        "enable_dynamic_keep_ratio": "true",
    },
    "robot_geo_rule_v0_keep085": {
        "pruning_strategy": "robot_geo_rule_v0",
        "pruning_enabled": "true",
        "geometry_enabled": "true",
        "keep_ratio": 0.85,
        "enable_robot_geo_expert": "true",
        "robot_geo_mode": "rule_v0",
    },
    "robot_geo_rule_v0_keep075": {
        "pruning_strategy": "robot_geo_rule_v0",
        "pruning_enabled": "true",
        "geometry_enabled": "true",
        "keep_ratio": 0.75,
        "enable_robot_geo_expert": "true",
        "robot_geo_mode": "rule_v0",
    },
    "robot_geo_rule_v0_keep065": {
        "pruning_strategy": "robot_geo_rule_v0",
        "pruning_enabled": "true",
        "geometry_enabled": "true",
        "keep_ratio": 0.65,
        "enable_robot_geo_expert": "true",
        "robot_geo_mode": "rule_v0",
    },
    "robot_geo_temporal_v0": {
        "pruning_strategy": "robot_geo_temporal_v0",
        "pruning_enabled": "true",
        "geometry_enabled": "true",
        "keep_ratio": 1.0,
        "enable_robot_geo_expert": "true",
        "robot_geo_mode": "temporal_rule_v0",
        "enable_dynamic_keep_ratio": "true",
    },
    "robot_geo_hybrid_v0_keep075": {
        "pruning_strategy": "robot_geo_hybrid_v0",
        "pruning_enabled": "true",
        "geometry_enabled": "true",
        "keep_ratio": 1.0,
        "enable_robot_geo_expert": "true",
        "robot_geo_mode": "hybrid_v0",
        "enable_dynamic_keep_ratio": "true",
    },
    "contact_budget_160_8_24": {
        "pruning_strategy": "robot_geo_contact_budget",
        "pruning_enabled": "true",
        "geometry_enabled": "true",
        "keep_ratio": 0.75,
        "contact_budget_edge_ratio": 160 / 192,
        "contact_budget_geo_ratio": 8 / 192,
        "contact_budget_diverse_ratio": 24 / 192,
    },
    "contact_budget_152_16_24": {
        "pruning_strategy": "robot_geo_contact_budget",
        "pruning_enabled": "true",
        "geometry_enabled": "true",
        "keep_ratio": 0.75,
        "contact_budget_edge_ratio": 152 / 192,
        "contact_budget_geo_ratio": 16 / 192,
        "contact_budget_diverse_ratio": 24 / 192,
    },
    "contact_budget_144_24_24": {
        "pruning_strategy": "robot_geo_contact_budget",
        "pruning_enabled": "true",
        "geometry_enabled": "true",
        "keep_ratio": 0.75,
        "contact_budget_edge_ratio": 144 / 192,
        "contact_budget_geo_ratio": 24 / 192,
        "contact_budget_diverse_ratio": 24 / 192,
    },
    "contact_budget_136_32_24": {
        "pruning_strategy": "robot_geo_contact_budget",
        "pruning_enabled": "true",
        "geometry_enabled": "true",
        "keep_ratio": 0.75,
        "contact_budget_edge_ratio": 136 / 192,
        "contact_budget_geo_ratio": 32 / 192,
        "contact_budget_diverse_ratio": 24 / 192,
    },
    "robot_geo_rule_v0_keep065": {
        "pruning_strategy": "robot_geo_rule_v0",
        "pruning_enabled": "true",
        "geometry_enabled": "true",
        "keep_ratio": 0.65,
        "enable_robot_geo_expert": "true",
        "robot_geo_mode": "rule_v0",
    },
    "robot_geo_rule_v0_keep090": {
        "pruning_strategy": "robot_geo_rule_v0",
        "pruning_enabled": "true",
        "geometry_enabled": "true",
        "keep_ratio": 0.90,
        "enable_robot_geo_expert": "true",
        "robot_geo_mode": "rule_v0",
    },
    "robot_geo_rule_v0_keep085": {
        "pruning_strategy": "robot_geo_rule_v0",
        "pruning_enabled": "true",
        "geometry_enabled": "true",
        "keep_ratio": 0.85,
        "enable_robot_geo_expert": "true",
        "robot_geo_mode": "rule_v0",
    },
    "robot_geo_distance_only_keep075": {
        "pruning_strategy": "robot_geo_rule_v0",
        "pruning_enabled": "true",
        "geometry_enabled": "true",
        "keep_ratio": 0.75,
        "enable_robot_geo_expert": "true",
        "robot_geo_mode": "rule_v0",
        "geo_score_weights": {
            "distance_to_gripper": 1.0,
            "motion_direction": 0.0,
            "depth_edge": 0.0,
            "workspace": 0.0,
            "contact_risk": 0.0,
        },
    },
    "robot_geo_motion_only_keep075": {
        "pruning_strategy": "robot_geo_rule_v0",
        "pruning_enabled": "true",
        "geometry_enabled": "true",
        "keep_ratio": 0.75,
        "enable_robot_geo_expert": "true",
        "robot_geo_mode": "rule_v0",
        "geo_score_weights": {
            "distance_to_gripper": 0.0,
            "motion_direction": 1.0,
            "depth_edge": 0.0,
            "workspace": 0.0,
            "contact_risk": 0.0,
        },
    },
    "robot_geo_contact_only_keep075": {
        "pruning_strategy": "robot_geo_rule_v0",
        "pruning_enabled": "true",
        "geometry_enabled": "true",
        "keep_ratio": 0.75,
        "enable_robot_geo_expert": "true",
        "robot_geo_mode": "rule_v0",
        "geo_score_weights": {
            "distance_to_gripper": 0.0,
            "motion_direction": 0.0,
            "depth_edge": 0.0,
            "workspace": 0.0,
            "contact_risk": 1.0,
        },
    },
    "robot_geo_depth_edge_only_keep075": {
        "pruning_strategy": "robot_geo_rule_v0",
        "pruning_enabled": "true",
        "geometry_enabled": "true",
        "keep_ratio": 0.75,
        "enable_robot_geo_expert": "true",
        "robot_geo_mode": "rule_v0",
        "geo_score_weights": {
            "distance_to_gripper": 0.0,
            "motion_direction": 0.0,
            "depth_edge": 1.0,
            "workspace": 0.0,
            "contact_risk": 0.0,
        },
    },
    "robot_geo_distance_motion_keep075": {
        "pruning_strategy": "robot_geo_rule_v0",
        "pruning_enabled": "true",
        "geometry_enabled": "true",
        "keep_ratio": 0.75,
        "enable_robot_geo_expert": "true",
        "robot_geo_mode": "rule_v0",
        "geo_score_weights": {
            "distance_to_gripper": 0.5,
            "motion_direction": 0.5,
            "depth_edge": 0.0,
            "workspace": 0.0,
            "contact_risk": 0.0,
        },
    },
    "robot_geo_distance_contact_keep075": {
        "pruning_strategy": "robot_geo_rule_v0",
        "pruning_enabled": "true",
        "geometry_enabled": "true",
        "keep_ratio": 0.75,
        "enable_robot_geo_expert": "true",
        "robot_geo_mode": "rule_v0",
        "geo_score_weights": {
            "distance_to_gripper": 0.5,
            "motion_direction": 0.0,
            "depth_edge": 0.0,
            "workspace": 0.0,
            "contact_risk": 0.5,
        },
    },
    "robot_geo_full_keep075": {
        "pruning_strategy": "robot_geo_rule_v0",
        "pruning_enabled": "true",
        "geometry_enabled": "true",
        "keep_ratio": 0.75,
        "enable_robot_geo_expert": "true",
        "robot_geo_mode": "rule_v0",
        "geo_score_weights": {
            "distance_to_gripper": 0.35,
            "motion_direction": 0.25,
            "depth_edge": 0.15,
            "workspace": 0.05,
            "contact_risk": 0.20,
        },
    },
    "robot_geo_hybrid_v0_keep075": {
        "pruning_strategy": "robot_geo_hybrid_v0",
        "pruning_enabled": "true",
        "geometry_enabled": "true",
        "keep_ratio": 0.75,
        "enable_robot_geo_expert": "true",
        "robot_geo_mode": "rule_v0",
    },
    "robot_geo_hybrid_dynamic_v0": {
        "pruning_strategy": "robot_geo_hybrid_dynamic_v0",
        "pruning_enabled": "true",
        "geometry_enabled": "true",
        "keep_ratio": 0.75,
        "enable_robot_geo_expert": "true",
        "robot_geo_mode": "rule_v0",
    },
    "robot_geo_hybrid_v1_keep075": {
        "pruning_strategy": "robot_geo_hybrid_v1",
        "pruning_enabled": "true",
        "geometry_enabled": "true",
        "keep_ratio": 0.75,
        "enable_robot_geo_expert": "true",
        "hybrid_v1_weights": {
            "w_edge": 0.45,
            "w_near": 0.20,
            "w_contact": 0.20,
            "w_corr": 0.10,
            "w_diverse": 0.05,
        },
    },
    "robot_geo_hybrid_v1_keep085": {
        "pruning_strategy": "robot_geo_hybrid_v1",
        "pruning_enabled": "true",
        "geometry_enabled": "true",
        "keep_ratio": 0.85,
        "enable_robot_geo_expert": "true",
        "hybrid_v1_weights": {
            "w_edge": 0.45,
            "w_near": 0.20,
            "w_contact": 0.20,
            "w_corr": 0.10,
            "w_diverse": 0.05,
        },
    },
    "robot_geo_hybrid_temporal_v1": {
        "pruning_strategy": "robot_geo_hybrid_temporal_v1",
        "pruning_enabled": "true",
        "geometry_enabled": "true",
        "keep_ratio": 0.75,
        "enable_robot_geo_expert": "true",
        "hybrid_v1_weights": {
            "w_edge": 0.45,
            "w_near": 0.20,
            "w_contact": 0.20,
            "w_corr": 0.10,
            "w_diverse": 0.05,
        },
        "temporal_adaptive_threshold_min": 0.08,
        "temporal_adaptive_threshold_percentile": 85.0,
        "temporal_interaction_lock_conservative_ratio": 0.90,
        "temporal_history_length": 5,
        "temporal_lock_min_frames": 2,
    },
    # P5: Targeted edge-reserve ablation for squeeze hypothesis validation
    # NOTE: robot_geo_hybrid_temporal_edge_reserve_v1 uses ADJUSTED weights (w_edge=0.35)
    # as a side effect of the edge_reserve implementation — this is a CONFOUND.
    # See robot_geo_hybrid_temporal_edge_reserve_adjusted_v1 (same thing, renamed for clarity).
    "robot_geo_hybrid_temporal_edge_reserve_v1": {
        "pruning_strategy": "robot_geo_hybrid_temporal_edge_reserve_v1",
        "pruning_enabled": "true",
        "geometry_enabled": "true",
        "keep_ratio": 0.75,
        "enable_robot_geo_expert": "true",
        "edge_reserve_ratio": 0.40,  # = round(0.40 * 192) = 77 tokens; clean ratio-based config
        "hybrid_v1_weights": {
            "w_edge": 0.45,
            "w_near": 0.20,
            "w_contact": 0.20,
            "w_corr": 0.10,
            "w_diverse": 0.05,
        },
        "temporal_adaptive_threshold_min": 0.08,
        "temporal_adaptive_threshold_percentile": 85.0,
        "temporal_interaction_lock_conservative_ratio": 0.90,
        "temporal_history_length": 5,
        "temporal_lock_min_frames": 2,
    },
    # P5-fix: Controlled ablation — weight adjustment only (no edge reserve)
    # Tests if the success improvement from edge_reserve_v1 is due to the weight change
    # (w_edge=0.35 vs original 0.45) rather than the edge reservation itself.
    "robot_geo_hybrid_temporal_weight_adjust_v1": {
        "pruning_strategy": "robot_geo_hybrid_temporal_v1",
        "pruning_enabled": "true",
        "geometry_enabled": "true",
        "keep_ratio": 0.75,
        "enable_robot_geo_expert": "true",
        "edge_reserve_ratio": 0.0,  # no edge reserve — tests weight change only
        "hybrid_v1_weights": {
            "w_edge": 0.35,  # ADJUSTED: down from 0.45 — this is the only change vs baseline
            "w_near": 0.25,  # ADJUSTED: up from 0.20
            "w_contact": 0.25,  # ADJUSTED: up from 0.20
            "w_corr": 0.10,  # unchanged
            "w_diverse": 0.05,  # unchanged
        },
        "temporal_adaptive_threshold_min": 0.08,
        "temporal_adaptive_threshold_percentile": 85.0,
        "temporal_interaction_lock_conservative_ratio": 0.90,
        "temporal_history_length": 5,
        "temporal_lock_min_frames": 2,
    },
    # P5-fix: Controlled ablation — edge reserve with ORIGINAL weights (fixed implementation)
    # Tests edge_reserve mechanism in isolation, without weight confound.
    # Uses edge_reserve strategy but with original (0.45/0.20/0.20) weights.
    "robot_geo_hybrid_temporal_edge_reserve_fixed_v1": {
        "pruning_strategy": "robot_geo_hybrid_temporal_edge_reserve_v1",
        "pruning_enabled": "true",
        "geometry_enabled": "true",
        "keep_ratio": 0.75,
        "enable_robot_geo_expert": "true",
        "edge_reserve_ratio": 0.40,  # same ratio as original edge_reserve_v1
        "hybrid_v1_weights": {
            "w_edge": 0.45,  # ORIGINAL: unchanged from baseline
            "w_near": 0.20,  # ORIGINAL
            "w_contact": 0.20,  # ORIGINAL
            "w_corr": 0.10,
            "w_diverse": 0.05,
        },
        "temporal_adaptive_threshold_min": 0.08,
        "temporal_adaptive_threshold_percentile": 85.0,
        "temporal_interaction_lock_conservative_ratio": 0.90,
        "temporal_history_length": 5,
        "temporal_lock_min_frames": 2,
    },
    # P5-fix: edge_reserve with ADJUSTED weights — same as original edge_reserve_v1, renamed
    # Confirms that the success improvement from the original (broken) run was due to
    # weight adjustment, and validates that the fixed implementation still recovers it.
    "robot_geo_hybrid_temporal_edge_reserve_adjusted_v1": {
        "pruning_strategy": "robot_geo_hybrid_temporal_edge_reserve_v1",
        "pruning_enabled": "true",
        "geometry_enabled": "true",
        "keep_ratio": 0.75,
        "enable_robot_geo_expert": "true",
        "edge_reserve_ratio": 0.40,
        "hybrid_v1_weights": {
            "w_edge": 0.35,  # ADJUSTED: down from 0.45
            "w_near": 0.25,  # ADJUSTED: up from 0.20
            "w_contact": 0.25,  # ADJUSTED: up from 0.20
            "w_corr": 0.10,
            "w_diverse": 0.05,
        },
        "temporal_adaptive_threshold_min": 0.08,
        "temporal_adaptive_threshold_percentile": 85.0,
        "temporal_interaction_lock_conservative_ratio": 0.90,
        "temporal_history_length": 5,
        "temporal_lock_min_frames": 2,
    },
    "contact_budget_144_16_32": {
        "pruning_strategy": "robot_geo_contact_budget",
        "pruning_enabled": "true",
        "geometry_enabled": "true",
        "keep_ratio": 0.75,
        "contact_budget_edge_ratio": 144 / 192,
        "contact_budget_geo_ratio": 16 / 192,
        "contact_budget_diverse_ratio": 32 / 192,
    },
    # P11: branch_budget_v0 — explicit depth/hybrid/fill branch budgets
    # Uses same score path as robot_geo_hybrid_temporal_edge_reserve_v1 (hybrid_v1 weights)
    "robot_geo_branch_budget_v0": {
        "pruning_strategy": "robot_geo_branch_budget_v0",
        "pruning_enabled": "true",
        "geometry_enabled": "true",
        "keep_ratio": 0.75,
        "enable_robot_geo_expert": "true",
        "branch_budget_depth_tokens": 65,
        "branch_budget_hybrid_tokens": 80,
        "hybrid_v1_weights": {
            "w_edge": 0.45,
            "w_near": 0.20,
            "w_contact": 0.20,
            "w_corr": 0.10,
            "w_diverse": 0.05,
        },
        "temporal_adaptive_threshold_min": 0.08,
        "temporal_adaptive_threshold_percentile": 85.0,
        "temporal_interaction_lock_conservative_ratio": 0.90,
        "temporal_history_length": 5,
        "temporal_lock_min_frames": 2,
    },
}


def build_eval_command(args: argparse.Namespace, method: str, method_dir: Path) -> List[str]:
    cfg = METHOD_CONFIGS[method]
    # Use the openvla conda env Python which has libero available
    python_exe = "/infini-data/miniconda3/envs/openvla/bin/python"
    cmd = [
        python_exe,
        str(SCRIPT_DIR / "eval_openvla_baseline.py"),
        "--model_path",
        str(args.checkpoint),
        "--task_suite",
        args.task_suite,
        "--num_episodes",
        str(args.num_trials_per_task),
        "--seed",
        str(args.seed),
        "--save_dir",
        str(method_dir),
        "--save_video",
        str(args.save_video).lower(),
        "--log_step_metrics",
        "true",
        "--use_wandb",
        str(args.use_wandb).lower(),
        "--pruning_strategy",
        str(cfg["pruning_strategy"]),
        "--pruning_enabled",
        str(cfg["pruning_enabled"]),
        "--geometry_enabled",
        str(cfg["geometry_enabled"]),
        "--keep_ratio",
        str(cfg["keep_ratio"]),
        "--save_geometry_vis",
        "false",
        "--geometry_debug",
        "false",
    ]
    if args.dry_run is not None:
        cmd += ["--dry_run", str(args.dry_run).lower()]
    if is_legacy_strategy(str(cfg["pruning_strategy"])):
        cmd += ["--allow_legacy_strategy", "true"]
    if args.num_tasks is not None:
        cmd += ["--num_tasks", str(args.num_tasks)]
    if args.max_steps is not None:
        cmd += ["--max_steps", str(args.max_steps)]
    if args.num_steps_wait is not None:
        cmd += ["--num_steps_wait", str(args.num_steps_wait)]
    if args.device is not None:
        cmd += ["--device", args.device]
    if args.precision is not None:
        cmd += ["--precision", args.precision]
    if args.use_flash_attention is not None:
        cmd += ["--use_flash_attention", str(args.use_flash_attention).lower()]
    if args.save_pruning_debug is not None:
        cmd += ["--save_pruning_debug", str(args.save_pruning_debug).lower()]
    if args.debug_tasks is not None:
        cmd += ["--debug_tasks", str(args.debug_tasks)]
    if getattr(args, "save_token_selection_debug", None) is not None:
        cmd += ["--save_token_selection_debug", str(args.save_token_selection_debug).lower()]
    for key in (
        "contact_budget_edge_ratio",
        "contact_budget_geo_ratio",
        "contact_budget_diverse_ratio",
        "w_near_contact",
        "w_corridor_contact",
        "edge_gate_eps",
        "detailed_pruning_timing",
        "hybrid_v1_weights",
        "temporal_adaptive_threshold_min",
        "temporal_adaptive_threshold_percentile",
        "temporal_interaction_lock_conservative_ratio",
        "temporal_history_length",
        "temporal_lock_min_frames",
        "temporal_ema_alpha",
        "temporal_contact_risk_threshold",
        "edge_reserve_ratio",  # P5-fix: was missing — prevents edge_reserve_k from being computed
        "edge_reserve_k",
        "branch_budget_depth_tokens",  # P11: branch_budget_v0 config
        "branch_budget_hybrid_tokens",  # P11: branch_budget_v0 config
    ):
        if key in cfg:
            value = cfg[key]
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            cmd += [f"--{key}", str(value).lower()]
    for key in (
        "enable_robot_geo_expert",
        "robot_geo_mode",
        "enable_depth_token_mapping",
        "enable_robot_state_adapter",
        "enable_dynamic_keep_ratio",
        "enable_geo_debug",
        "max_debug_frames",
        "geo_debug_interval",
        "temporal_history_length",
        "temporal_lock_min_frames",
        "temporal_contact_risk_threshold",
        "temporal_stability_threshold",
        "temporal_motion_cos_threshold",
        "temporal_ema_alpha",
        "geo_score_weights",
        "dynamic_keep_ratio_config",
    ):
        value = cfg.get(key, getattr(args, key, None))
        if value is not None:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            cmd += [f"--{key}", str(value).lower()]
    return cmd


def postprocess_method_dir(method_dir: Path, method: str, command: List[str]) -> None:
    config = {
        "method": method,
        "command": command,
        "output_dir": str(method_dir),
    }
    with open(method_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    episodes_src = method_dir / "episodes.jsonl"
    episodes_dst = method_dir / "per_episode.jsonl"
    if episodes_src.exists():
        episodes_dst.write_text(episodes_src.read_text())

    step_csv = method_dir / "step_metrics.csv"
    step_jsonl = method_dir / "per_step_metrics.jsonl"
    if step_csv.exists():
        with open(step_csv, newline="") as src, open(step_jsonl, "w") as dst:
            for row in csv.DictReader(src):
                dst.write(json.dumps(_coerce_row(row)) + "\n")


def _coerce_row(row: Dict[str, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in row.items():
        if value == "":
            out[key] = None
            continue
        if value in ("True", "False"):
            out[key] = value == "True"
            continue
        try:
            if "." in value or "e" in value.lower():
                out[key] = float(value)
            else:
                out[key] = int(value)
        except ValueError:
            out[key] = value
    return out


def parse_methods(value: str) -> List[str]:
    if value.strip().lower() == "all":
        return list(DEFAULT_METHODS)
    methods = [m.strip() for m in value.split(",") if m.strip()]
    unknown = [m for m in methods if m not in METHOD_CONFIGS]
    if unknown:
        raise ValueError(f"Unknown method(s): {unknown}. Supported: {sorted(METHOD_CONFIGS)}")
    return methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenVLA pruning ablation methods")
    parser.add_argument("--checkpoint", type=Path, required=True, help="OpenVLA checkpoint path")
    parser.add_argument("--task_suite", type=str, default="libero_spatial")
    parser.add_argument("--num_trials_per_task", type=int, default=3)
    parser.add_argument("--num_tasks", type=int, default=None)
    parser.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--num_steps_wait", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--precision", type=str, default=None)
    parser.add_argument("--use_flash_attention", type=lambda x: x.lower() == "true", default=None)
    parser.add_argument("--save_video", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--use_wandb", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--dry_run", type=lambda x: x.lower() == "true", default=None)
    parser.add_argument("--save_pruning_debug", type=lambda x: x.lower() == "true", default=None)
    parser.add_argument("--debug_tasks", type=str, default=None)
    parser.add_argument("--save_token_selection_debug", type=lambda x: x.lower() == "true", default=None)
    parser.add_argument("--enable_robot_geo_expert", type=lambda x: x.lower() == "true", default=None)
    parser.add_argument("--robot_geo_mode", type=str, default=None,
                        choices=["off", "rule_v0", "dynamic_rule_v0", "temporal_rule_v0"])
    parser.add_argument("--enable_depth_token_mapping", type=lambda x: x.lower() == "true", default=None)
    parser.add_argument("--enable_robot_state_adapter", type=lambda x: x.lower() == "true", default=None)
    parser.add_argument("--enable_dynamic_keep_ratio", type=lambda x: x.lower() == "true", default=None)
    parser.add_argument("--enable_geo_debug", type=lambda x: x.lower() == "true", default=None)
    parser.add_argument("--max_debug_frames", type=int, default=None)
    parser.add_argument("--geo_debug_interval", type=int, default=None)
    parser.add_argument("--temporal_history_length", type=int, default=None)
    parser.add_argument("--temporal_lock_min_frames", type=int, default=None)
    parser.add_argument("--temporal_contact_risk_threshold", type=float, default=None)
    parser.add_argument("--temporal_stability_threshold", type=float, default=None)
    parser.add_argument("--temporal_motion_cos_threshold", type=float, default=None)
    parser.add_argument("--temporal_ema_alpha", type=float, default=None)
    parser.add_argument("--geo_score_weights", type=json.loads, default=None)
    parser.add_argument("--dynamic_keep_ratio_config", type=json.loads, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.methods)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    root = args.output_root or PROJECT_ROOT / "outputs" / f"pruning_ablation_{timestamp}"
    root.mkdir(parents=True, exist_ok=True)

    for method in methods:
        method_dir = root / method
        summary_path = method_dir / "summary.json"
        if args.skip_existing and summary_path.exists():
            print(f"[SKIP] {method}: {summary_path} exists")
            continue
        method_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_eval_command(args, method, method_dir)
        print(f"[RUN] {method}")
        print(" ".join(str(part) for part in cmd))
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        postprocess_method_dir(method_dir, method, cmd)

    rollup_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "rollup_pruning_results.py"),
        "--root",
        str(root),
        "--methods",
        ",".join(methods),
    ]
    subprocess.run(rollup_cmd, cwd=PROJECT_ROOT, check=True)
    print(f"[DONE] Rollup: {root / 'rollup_summary.csv'}")


if __name__ == "__main__":
    main()
