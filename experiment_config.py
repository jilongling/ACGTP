"""
experiment_config.py

Experiment configuration for geometry-guided visual token pruning ablation studies.

Supports:
- Baseline (no pruning)
- Random pruning
- Depth edge pruning
- Gripper distance pruning
- Geometry rule pruning

With various keep ratios: 1.0, 0.75, 0.5, 0.25

Usage:
    # Create from preset
    cfg = ExperimentConfig.from_preset("baseline")

    # Create custom
    cfg = ExperimentConfig(
        geometry_enabled=True,
        pruning_enabled=True,
        pruning_method="geometry_rule",
        keep_ratio=0.5,
    )

    # Get output directory
    output_dir = cfg.get_output_dir(task_suite="libero_spatial")
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================


class PruningMethod(str, Enum):
    """Supported pruning methods."""

    NONE = "none"
    RANDOM = "random"
    DEPTH_EDGE = "depth_edge"
    GRIPPER_DISTANCE = "gripper_distance"
    GEOMETRY_RULE = "geometry_rule"

    @classmethod
    def from_string(cls, s: str) -> "PruningMethod":
        """Parse pruning method from string."""
        s_lower = s.lower().strip()
        for method in cls:
            if method.value == s_lower:
                return method
        raise ValueError(f"Unknown pruning method: {s}. Available: {[m.value for m in cls]}")

    @property
    def display_name(self) -> str:
        """Human-readable name."""
        names = {
            "none": "baseline",
            "random": "random",
            "depth_edge": "depth_edge",
            "gripper_distance": "gripper_dist",
            "geometry_rule": "geometry_rule",
        }
        return names.get(self.value, self.value)


class ExperimentPreset(str, Enum):
    """Predefined experiment presets."""

    BASELINE = "baseline"
    RANDOM_75 = "random_0.75"
    RANDOM_50 = "random_0.5"
    RANDOM_25 = "random_0.25"
    DEPTH_EDGE_75 = "depth_edge_0.75"
    DEPTH_EDGE_50 = "depth_edge_0.5"
    DEPTH_EDGE_25 = "depth_edge_0.25"
    GRIPPER_DIST_75 = "gripper_dist_0.75"
    GRIPPER_DIST_50 = "gripper_dist_0.5"
    GRIPPER_DIST_25 = "gripper_dist_0.25"
    GEOMETRY_RULE_75 = "geometry_rule_0.75"
    GEOMETRY_RULE_50 = "geometry_rule_0.5"
    GEOMETRY_RULE_25 = "geometry_rule_0.25"


# Valid keep ratios
VALID_KEEP_RATIOS = [1.0, 0.75, 0.5, 0.25]

# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class GeometryScoreWeights:
    """Weights for combining geometry score components."""

    distance: float = 0.5
    direction: float = 0.3
    depth_edge: float = 0.2

    def __post_init__(self):
        """Validate weights sum to 1.0."""
        total = self.distance + self.direction + self.depth_edge
        if abs(total - 1.0) > 0.01:
            logger.warning(
                f"[GeometryScoreWeights] Weights sum to {total:.2f}, not 1.0. "
                f"Normalizing..."
            )
            self.distance /= total
            self.direction /= total
            self.depth_edge /= total

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "distance": self.distance,
            "direction": self.direction,
            "depth_edge": self.depth_edge,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "GeometryScoreWeights":
        """Create from dictionary."""
        return cls(
            distance=d.get("distance", 0.5),
            direction=d.get("direction", 0.3),
            depth_edge=d.get("depth_edge", 0.2),
        )


@dataclass
class VisualizationConfig:
    """Configuration for geometry visualization."""

    enabled: bool = False
    vis_interval: int = 10
    max_vis_per_episode: int = 5
    output_dir: str = "logs/geometry_vis"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ExperimentConfig:
    """
    Complete configuration for a geometry-guided pruning experiment.

    Attributes:
        geometry_enabled: Enable geometry data collection and scoring.
        pruning_enabled: Enable token pruning.
        pruning_method: Pruning method to use.
        keep_ratio: Fraction of tokens to keep (0.0 to 1.0).

        geometry_sigma_d: Sigma for distance scoring (meters).
        geometry_score_weights: Weights for combining score components.

        visualization: Visualization configuration.
        debug: Enable debug mode (fewer tasks/episodes).

        seed: Random seed for reproducibility.
    """

    # Geometry settings
    geometry_enabled: bool = False

    # Pruning settings
    pruning_enabled: bool = False
    pruning_method: str = "none"
    keep_ratio: float = 1.0

    # Geometry expert settings
    geometry_sigma_d: float = 0.5
    geometry_score_weights: GeometryScoreWeights = field(
        default_factory=lambda: GeometryScoreWeights(
            distance=0.5,
            direction=0.3,
            depth_edge=0.2,
        )
    )

    # Visualization settings
    visualization: VisualizationConfig = field(
        default_factory=lambda: VisualizationConfig(
            enabled=False,
            vis_interval=10,
            max_vis_per_episode=5,
        )
    )

    # Debug settings
    debug: bool = False
    debug_num_tasks: int = 2
    debug_num_episodes: int = 1

    # General settings
    seed: int = 42
    save_dir: str = "outputs/ablation"

    def __post_init__(self):
        """Validate configuration."""
        # Validate pruning method
        try:
            self.pruning_method = PruningMethod.from_string(self.pruning_method).value
        except ValueError:
            pass  # Will be caught later

        # Validate keep ratio
        if self.keep_ratio not in VALID_KEEP_RATIOS:
            logger.warning(
                f"[ExperimentConfig] keep_ratio={self.keep_ratio} not in {VALID_KEEP_RATIOS}. "
                f"Clamping to nearest valid value."
            )
            self.keep_ratio = min(VALID_KEEP_RATIOS, key=lambda x: abs(x - self.keep_ratio))

        # Validate consistency
        if self.pruning_method == "none" and self.pruning_enabled:
            logger.warning(
                "[ExperimentConfig] pruning_method='none' but pruning_enabled=True. "
                "Setting pruning_enabled=False."
            )
            self.pruning_enabled = False

        # geometry_rule, depth_edge, gripper_distance require geometry_enabled
        geometry_required_methods = ["depth_edge", "gripper_distance", "geometry_rule"]
        if self.pruning_method in geometry_required_methods and not self.geometry_enabled:
            logger.warning(
                f"[ExperimentConfig] pruning_method='{self.pruning_method}' requires geometry_enabled=True. "
                "Setting geometry_enabled=True."
            )
            self.geometry_enabled = True

        if not self.pruning_enabled and self.keep_ratio < 1.0:
            logger.warning(
                "[ExperimentConfig] pruning_enabled=False but keep_ratio<1.0. "
                "Setting keep_ratio=1.0."
            )
            self.keep_ratio = 1.0

        # Set visualization based on settings
        if self.geometry_enabled and not self.visualization.enabled:
            # Enable visualization when geometry is enabled
            self.visualization.enabled = self.save_geometry_vis if hasattr(self, "save_geometry_vis") else False

    @property
    def method_short(self) -> str:
        """Short method name for file naming."""
        if self.pruning_method == "none":
            return "baseline"
        return f"{PruningMethod.from_string(self.pruning_method).display_name}_{self.keep_ratio}"

    def is_baseline(self) -> bool:
        """Check if this is a baseline configuration."""
        return not self.pruning_enabled and self.pruning_method == "none"

    def get_output_dir(self, task_suite: str) -> str:
        """
        Get output directory for this experiment.

        Args:
            task_suite: Name of the task suite (e.g., "libero_spatial").

        Returns:
            Output directory path.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"EVAL-{task_suite}-{self.method_short}-{timestamp}"
        return os.path.join(self.save_dir, dir_name)

    def get_experiment_name(self, task_suite: str) -> str:
        """
        Get experiment name for logging.

        Args:
            task_suite: Name of the task suite.

        Returns:
            Experiment name string.
        """
        parts = [
            f"geometry={'on' if self.geometry_enabled else 'off'}",
            f"pruning={self.pruning_method}",
            f"keep={self.keep_ratio}",
        ]
        if self.debug:
            parts.append("debug")
        return f"[{'/'.join(parts)}]"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "geometry_enabled": self.geometry_enabled,
            "pruning_enabled": self.pruning_enabled,
            "pruning_method": self.pruning_method,
            "keep_ratio": self.keep_ratio,
            "geometry_sigma_d": self.geometry_sigma_d,
            "geometry_score_weights": self.geometry_score_weights.to_dict(),
            "visualization": self.visualization.to_dict(),
            "debug": self.debug,
            "debug_num_tasks": self.debug_num_tasks if self.debug else None,
            "debug_num_episodes": self.debug_num_episodes if self.debug else None,
            "seed": self.seed,
            "save_dir": self.save_dir,
            "experiment_type": "baseline" if self.is_baseline() else f"pruning_{self.pruning_method}",
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        """Create from dictionary."""
        weights = GeometryScoreWeights.from_dict(
            d.get("geometry_score_weights", {})
        )
        vis = VisualizationConfig(**d.get("visualization", {}))

        return cls(
            geometry_enabled=d.get("geometry_enabled", False),
            pruning_enabled=d.get("pruning_enabled", False),
            pruning_method=d.get("pruning_method", "none"),
            keep_ratio=d.get("keep_ratio", 1.0),
            geometry_sigma_d=d.get("geometry_sigma_d", 0.5),
            geometry_score_weights=weights,
            visualization=vis,
            debug=d.get("debug", False),
            debug_num_tasks=d.get("debug_num_tasks", 2),
            debug_num_episodes=d.get("debug_num_episodes", 1),
            seed=d.get("seed", 42),
            save_dir=d.get("save_dir", "outputs/ablation"),
        )

    @classmethod
    def from_preset(cls, preset: str) -> "ExperimentConfig":
        """
        Create configuration from preset.

        Args:
            preset: Preset name (e.g., "baseline", "random_0.5", "geometry_rule_0.25").

        Returns:
            ExperimentConfig instance.
        """
        preset_lower = preset.lower().strip()

        # Parse preset
        if preset_lower == "baseline":
            return cls(
                geometry_enabled=False,
                pruning_enabled=False,
                pruning_method="none",
                keep_ratio=1.0,
            )

        # Parse method_ratio presets
        parts = preset_lower.rsplit("_", 1)
        if len(parts) == 2:
            method, ratio_str = parts
            try:
                ratio = float(ratio_str)
            except ValueError:
                raise ValueError(f"Invalid preset: {preset}. Ratio must be float.")

            # Map display names to method names
            method_map = {
                "random": "random",
                "depth_edge": "depth_edge",
                "depthedge": "depth_edge",
                "gripper_dist": "gripper_distance",
                "gripperdist": "gripper_distance",
                "geometry_rule": "geometry_rule",
                "geometryrule": "geometry_rule",
            }

            method_name = method_map.get(method, method)
            if method_name not in [m.value for m in PruningMethod]:
                raise ValueError(f"Unknown method in preset: {preset}")

            # Determine if geometry is needed
            # geometry_rule, depth_edge, gripper_distance require geometry
            # random does NOT require geometry
            geometry_enabled = method_name not in ["random", "none"]

            return cls(
                geometry_enabled=geometry_enabled,
                pruning_enabled=method_name != "none",
                pruning_method=method_name,
                keep_ratio=ratio,
            )

        raise ValueError(
            f"Invalid preset: {preset}. "
            f"Use 'baseline' or 'method_ratio' (e.g., 'geometry_rule_0.5')."
        )

    @classmethod
    def from_preset_enum(cls, preset: ExperimentPreset) -> "ExperimentConfig":
        """Create from ExperimentPreset enum."""
        return cls.from_preset(preset.value)


# =============================================================================
# Experiment Suite Manager
# =============================================================================


class ExperimentSuite:
    """
    Manages a suite of ablation experiments.

    Usage:
        suite = ExperimentSuite(
            task_suite="libero_spatial",
            presets=["baseline", "random_0.5", "geometry_rule_0.5"],
        )
        for cfg in suite.configs:
            run_experiment(cfg)
    """

    def __init__(
        self,
        task_suite: str,
        presets: Optional[List[str]] = None,
        methods: Optional[List[str]] = None,
        ratios: Optional[List[float]] = None,
        include_baseline: bool = True,
        debug: bool = False,
        base_config: Optional[ExperimentConfig] = None,
    ):
        """
        Initialize experiment suite.

        Args:
            task_suite: Name of the task suite.
            presets: List of preset names to run.
            methods: List of pruning methods to run (mutually exclusive with presets).
            ratios: List of keep ratios to run.
            include_baseline: Include baseline experiment.
            debug: Enable debug mode for all experiments.
            base_config: Base configuration to extend.
        """
        self.task_suite = task_suite
        self.debug = debug
        self.base_config = base_config or ExperimentConfig()

        # Generate configs
        self.configs = self._generate_configs(presets, methods, ratios, include_baseline)

        logger.info(
            f"[ExperimentSuite] Created suite with {len(self.configs)} experiments "
            f"for task_suite={task_suite}"
        )

    def _generate_configs(
        self,
        presets: Optional[List[str]],
        methods: Optional[List[str]],
        ratios: Optional[List[float]],
        include_baseline: bool,
    ) -> List[ExperimentConfig]:
        """Generate list of experiment configurations."""
        configs = []

        if presets:
            # Use presets directly
            for preset in presets:
                cfg = ExperimentConfig.from_preset(preset)
                if self.debug:
                    cfg.debug = True
                cfg.save_dir = self.base_config.save_dir
                configs.append(cfg)
        else:
            # Generate from methods and ratios
            methods = methods or ["random", "depth_edge", "gripper_distance", "geometry_rule"]
            ratios = ratios or [0.75, 0.5, 0.25]

            # Add baseline first if requested
            if include_baseline:
                cfg = ExperimentConfig.from_preset("baseline")
                if self.debug:
                    cfg.debug = True
                cfg.save_dir = self.base_config.save_dir
                configs.append(cfg)

            # Add pruning configs
            for method in methods:
                for ratio in ratios:
                    preset = f"{method}_{ratio}"
                    cfg = ExperimentConfig.from_preset(preset)
                    if self.debug:
                        cfg.debug = True
                    cfg.save_dir = self.base_config.save_dir
                    configs.append(cfg)

        return configs

    def __iter__(self):
        """Iterate over experiment configurations."""
        return iter(self.configs)

    def __len__(self):
        """Return number of experiments."""
        return len(self.configs)

    def save_suite_config(self, output_dir: str) -> str:
        """
        Save suite configuration to JSON.

        Args:
            output_dir: Directory to save the config.

        Returns:
            Path to saved config file.
        """
        output_path = Path(output_dir) / "experiment_suite.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        suite_config = {
            "task_suite": self.task_suite,
            "num_experiments": len(self.configs),
            "debug": self.debug,
            "experiments": [
                {**cfg.to_dict(), "output_dir": cfg.get_output_dir(self.task_suite)}
                for cfg in self.configs
            ],
        }

        with open(output_path, "w") as f:
            json.dump(suite_config, f, indent=2)

        logger.info(f"[ExperimentSuite] Saved suite config to {output_path}")
        return str(output_path)


# =============================================================================
# Predefined Experiment Groups
# =============================================================================


def get_baseline_config() -> ExperimentConfig:
    """Get baseline configuration (Group A)."""
    return ExperimentConfig(
        geometry_enabled=False,
        pruning_enabled=False,
        pruning_method="none",
        keep_ratio=1.0,
    )


def get_random_pruning_configs(ratios: List[float] = None) -> List[ExperimentConfig]:
    """Get random pruning configurations (Group B)."""
    ratios = ratios or [0.75, 0.5, 0.25]
    return [
        ExperimentConfig(
            geometry_enabled=False,
            pruning_enabled=True,
            pruning_method="random",
            keep_ratio=r,
        )
        for r in ratios
    ]


def get_depth_edge_configs(ratios: List[float] = None) -> List[ExperimentConfig]:
    """Get depth edge pruning configurations (Group C)."""
    ratios = ratios or [0.75, 0.5, 0.25]
    return [
        ExperimentConfig(
            geometry_enabled=True,
            pruning_enabled=True,
            pruning_method="depth_edge",
            keep_ratio=r,
        )
        for r in ratios
    ]


def get_gripper_distance_configs(ratios: List[float] = None) -> List[ExperimentConfig]:
    """Get gripper distance pruning configurations (Group D)."""
    ratios = ratios or [0.75, 0.5, 0.25]
    return [
        ExperimentConfig(
            geometry_enabled=True,
            pruning_enabled=True,
            pruning_method="gripper_distance",
            keep_ratio=r,
        )
        for r in ratios
    ]


def get_geometry_rule_configs(ratios: List[float] = None) -> List[ExperimentConfig]:
    """Get geometry rule pruning configurations (Group E)."""
    ratios = ratios or [0.75, 0.5, 0.25]
    return [
        ExperimentConfig(
            geometry_enabled=True,
            pruning_enabled=True,
            pruning_method="geometry_rule",
            keep_ratio=r,
        )
        for r in ratios
    ]


def get_all_ablation_configs(
    ratios: List[float] = None,
    include_baseline: bool = True,
) -> List[ExperimentConfig]:
    """Get all ablation experiment configurations."""
    ratios = ratios or [0.75, 0.5, 0.25]
    configs = []

    if include_baseline:
        configs.append(get_baseline_config())

    configs.extend(get_random_pruning_configs(ratios))
    configs.extend(get_depth_edge_configs(ratios))
    configs.extend(get_gripper_distance_configs(ratios))
    configs.extend(get_geometry_rule_configs(ratios))

    return configs


# =============================================================================
# CLI Helper
# =============================================================================


def create_parser():
    """Create argument parser for experiment configuration."""
    import argparse

    parser = argparse.ArgumentParser(description="Geometry-guided pruning experiments")

    # Experiment selection
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Experiment preset (e.g., 'baseline', 'random_0.5', 'geometry_rule_0.25')",
    )
    parser.add_argument(
        "--experiment-group",
        type=str,
        choices=["A", "B", "C", "D", "E", "all"],
        default=None,
        help="Predefined experiment group",
    )

    # Geometry settings
    parser.add_argument("--geometry-enabled", action="store_true", default=None)
    parser.add_argument("--no-geometry", action="store_true", default=None)

    # Pruning settings
    parser.add_argument("--pruning-enabled", action="store_true", default=None)
    parser.add_argument("--no-pruning", action="store_true", default=None)
    parser.add_argument(
        "--pruning-method",
        type=str,
        choices=["none", "random", "depth_edge", "gripper_distance", "geometry_rule"],
        default=None,
    )
    parser.add_argument(
        "--keep-ratio",
        type=float,
        choices=VALID_KEEP_RATIOS,
        default=None,
    )

    # Debug
    parser.add_argument("--debug", action="store_true", default=False)

    # Output
    parser.add_argument("--save-dir", type=str, default="outputs/ablation")

    return parser


def parse_args_to_config(args) -> ExperimentConfig:
    """Parse command-line arguments to ExperimentConfig."""
    # Use preset if provided
    if args.preset:
        cfg = ExperimentConfig.from_preset(args.preset)
    else:
        # Build from arguments
        cfg = ExperimentConfig()

        if args.geometry_enabled is not None:
            cfg.geometry_enabled = args.geometry_enabled
        if args.no_geometry is not None:
            cfg.geometry_enabled = not args.no_geometry

        if args.pruning_enabled is not None:
            cfg.pruning_enabled = args.pruning_enabled
        if args.no_pruning is not None:
            cfg.pruning_enabled = not args.no_pruning

        if args.pruning_method is not None:
            cfg.pruning_method = args.pruning_method

        if args.keep_ratio is not None:
            cfg.keep_ratio = args.keep_ratio

    # Apply common settings
    cfg.debug = args.debug
    cfg.save_dir = args.save_dir

    return cfg


# =============================================================================
# Debug/Test Functions
# =============================================================================


def debug_experiment_config():
    """Verify experiment configuration."""
    print("=" * 60)
    print("[ExperimentConfig] Running debug tests...")
    print("=" * 60)

    # Test 1: Baseline preset
    print("\n[Test 1] Baseline preset")
    cfg = ExperimentConfig.from_preset("baseline")
    assert cfg.geometry_enabled == False
    assert cfg.pruning_enabled == False
    assert cfg.pruning_method == "none"
    assert cfg.keep_ratio == 1.0
    print(f"  [OK] Baseline: {cfg.to_dict()}")

    # Test 2: Random pruning presets
    print("\n[Test 2] Random pruning presets")
    for ratio in [0.75, 0.5, 0.25]:
        cfg = ExperimentConfig.from_preset(f"random_{ratio}")
        assert cfg.geometry_enabled == False
        assert cfg.pruning_enabled == True
        assert cfg.pruning_method == "random"
        assert cfg.keep_ratio == ratio
        print(f"  [OK] random_{ratio}")

    # Test 3: Geometry rule presets
    print("\n[Test 3] Geometry rule presets")
    for ratio in [0.75, 0.5, 0.25]:
        cfg = ExperimentConfig.from_preset(f"geometry_rule_{ratio}")
        assert cfg.geometry_enabled == True
        assert cfg.pruning_enabled == True
        assert cfg.pruning_method == "geometry_rule"
        assert cfg.keep_ratio == ratio
        print(f"  [OK] geometry_rule_{ratio}")

    # Test 4: Output directory naming
    print("\n[Test 4] Output directory naming")
    cfg = ExperimentConfig.from_preset("geometry_rule_0.5")
    output_dir = cfg.get_output_dir("libero_spatial")
    assert "libero_spatial" in output_dir
    assert "geometry_rule" in output_dir or "geometry-rule" in output_dir.replace("_", "-")
    assert "0.5" in output_dir
    print(f"  [OK] Output dir: {output_dir}")

    # Test 5: Experiment suite
    print("\n[Test 5] Experiment suite")
    suite = ExperimentSuite(
        task_suite="libero_spatial",
        methods=["random", "geometry_rule"],
        ratios=[0.5, 0.25],
        include_baseline=True,
    )
    assert len(suite) == 5  # 1 baseline + 2 methods * 2 ratios
    print(f"  [OK] Suite has {len(suite)} experiments")

    # Test 6: get_all_ablation_configs
    print("\n[Test 6] All ablation configs")
    configs = get_all_ablation_configs(ratios=[0.5])
    assert len(configs) == 5  # 1 baseline + 4 methods * 1 ratio
    print(f"  [OK] Ablation configs: {len(configs)} experiments")

    # Test 7: JSON serialization
    print("\n[Test 7] JSON serialization")
    cfg = ExperimentConfig.from_preset("geometry_rule_0.5")
    d = cfg.to_dict()
    cfg2 = ExperimentConfig.from_dict(d)
    assert cfg2.pruning_method == cfg.pruning_method
    assert cfg2.keep_ratio == cfg.keep_ratio
    print(f"  [OK] JSON roundtrip works")

    # Test 8: method_short property
    print("\n[Test 8] method_short property")
    for method in ["baseline", "random_0.5", "geometry_rule_0.25"]:
        cfg = ExperimentConfig.from_preset(method)
        print(f"  {method} -> {cfg.method_short}")

    print("\n" + "=" * 60)
    print("[ExperimentConfig] All debug tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    debug_experiment_config()
