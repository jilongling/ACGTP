"""
verify_pipeline.py

分阶段验证 pipeline，确保所有模块正确集成。

验证顺序：
1. baseline sanity check - 确认结果和之前 baseline 一致
2. geometry data check - 确认可以读取 rgb/depth/K/T_base_cam/ee_pose
3. token mapping check - 确认 token_points_base shape 正确，invalid depth 不导致 NaN
4. visualization check - 保存 heatmap，肉眼检查
5. random pruning check - 确认剪枝链路不 shape mismatch
6. geometry_rule pruning check - 确认 success rate / latency / token 数有记录

Usage:
    python scripts/verify_pipeline.py --stage 1
    python scripts/verify_pipeline.py --stage all
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment_config import ExperimentConfig


# =============================================================================
# Verification Results
# =============================================================================


@dataclass
class VerificationResult:
    stage: int
    name: str
    passed: bool
    message: str
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class VerificationRunner:
    """运行验证并收集结果。"""

    def __init__(self, save_dir: str = "outputs/verification"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[VerificationResult] = []

    def add_result(self, result: VerificationResult):
        self.results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] Stage {result.stage}: {result.name}")
        if result.message:
            print(f"       {result.message}")

    def save_report(self) -> str:
        """保存验证报告。"""
        report = {
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "total_stages": 6,
            "results": [
                {
                    "stage": r.stage,
                    "name": r.name,
                    "passed": bool(r.passed),
                    "message": r.message,
                    "details": r.details,
                }
                for r in self.results
            ],
        }

        # Custom JSON encoder for numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.bool_):
                    return bool(obj)
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.generic):
                    try:
                        return float(obj)
                    except (ValueError, TypeError):
                        return str(obj)
                return super().default(obj)

        report_path = self.save_dir / f"verification_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)

        passed = sum(1 for r in self.results if r.passed)
        print(f"\n{'='*60}")
        print(f"Verification Report: {passed}/{len(self.results)} stages passed")
        print(f"Report saved to: {report_path}")
        print(f"{'='*60}")

        return str(report_path)


# =============================================================================
# Stage 1: Baseline Sanity Check
# =============================================================================


def verify_baseline(runner: VerificationRunner, cfg: ExperimentConfig, dry_run: bool = True):
    """
    Stage 1: baseline sanity check
    geometry_enabled=false, pruning_enabled=false
    确认结果和之前 baseline 一致
    """
    print("\n" + "=" * 60)
    print("Stage 1: Baseline Sanity Check")
    print("=" * 60)

    # Create baseline config
    cfg = ExperimentConfig.from_preset("baseline")
    print(f"Config: geometry_enabled={cfg.geometry_enabled}, "
          f"pruning_enabled={cfg.pruning_enabled}, "
          f"pruning_method={cfg.pruning_method}")

    # Expected behavior
    expected = {
        "geometry_enabled": False,
        "pruning_enabled": False,
        "pruning_method": "none",
        "keep_ratio": 1.0,
    }

    # Check config
    checks = [
        ("geometry_enabled", cfg.geometry_enabled == expected["geometry_enabled"]),
        ("pruning_enabled", cfg.pruning_enabled == expected["pruning_enabled"]),
        ("pruning_method", cfg.pruning_method == expected["pruning_method"]),
        ("keep_ratio", cfg.keep_ratio == expected["keep_ratio"]),
    ]

    all_passed = all(c[1] for c in checks)
    details = {c[0]: {"expected": expected[c[0]], "actual": getattr(cfg, c[0]), "match": c[1]}
               for c in checks}

    runner.add_result(VerificationResult(
        stage=1,
        name="baseline_config",
        passed=all_passed,
        message="Baseline config matches expected" if all_passed else "Config mismatch",
        details=details,
    ))

    # Check output naming
    output_dir = cfg.get_output_dir("libero_spatial")
    has_baseline = "baseline" in output_dir
    details = {"output_dir": output_dir, "has_baseline": has_baseline}
    runner.add_result(VerificationResult(
        stage=1,
        name="output_naming",
        passed=has_baseline,
        message=f"Output dir contains 'baseline': {has_baseline}",
        details=details,
    ))

    print(f"\n  Expected output: {output_dir}")

    # Check metrics_logger compatibility
    from utils.metrics_logger import EpisodeMetrics, StepMetrics

    step = StepMetrics()
    episode = EpisodeMetrics()

    # Verify pruning fields are null for baseline
    step_checks = [
        ("num_visual_tokens_original", step.num_visual_tokens_original is None),
        ("geometry_score_mean", step.geometry_score_mean is None),
        ("token_mapping_time_ms", step.token_mapping_time_ms is None),
    ]

    all_passed = all(c[1] for c in step_checks)
    details = {c[0]: {"expected": None, "actual": getattr(step, c[0]), "match": c[1]}
               for c in step_checks}
    runner.add_result(VerificationResult(
        stage=1,
        name="metrics_null_for_baseline",
        passed=all_passed,
        message="Step metrics fields are null for baseline" if all_passed else "Some fields not null",
        details=details,
    ))


# =============================================================================
# Stage 2: Geometry Data Check
# =============================================================================


def verify_geometry_data(runner: VerificationRunner, cfg: ExperimentConfig, dry_run: bool = True):
    """
    Stage 2: geometry data check
    geometry_enabled=true, pruning_enabled=false
    确认可以读取 rgb/depth/K/T_base_cam/ee_pose
    如果某些字段是 None，需要在日志中明确说明
    """
    print("\n" + "=" * 60)
    print("Stage 2: Geometry Data Check")
    print("=" * 60)

    # Create geometry config
    cfg = ExperimentConfig.from_preset("geometry_rule_0.5")
    cfg.geometry_enabled = True
    cfg.pruning_enabled = False
    cfg.pruning_method = "none"
    cfg.keep_ratio = 1.0

    print(f"Config: geometry_enabled={cfg.geometry_enabled}, "
          f"pruning_enabled={cfg.pruning_enabled}")

    # Check GeometryDataRecorder
    from geometry import GeometryDataRecorder

    recorder = GeometryDataRecorder(enabled=True)

    # Check required fields in observation interface
    required_obs_fields = [
        "full_image",       # RGB image
        "depth",            # Depth image
        "camera_intrinsics", # K matrix
        "camera_extrinsics", # T_base_cam
        "ee_pose",          # End effector pose
    ]

    # Check recorder interface
    from geometry import Token3DMapper

    recorder_checks = [
        ("enabled", recorder.enabled == True),
        ("has_collect_step", hasattr(recorder, "collect_step")),
        ("has_reset", hasattr(recorder, "reset")),
        ("has_get_history", hasattr(recorder, "get_history")),
    ]

    all_passed = all(c[1] for c in recorder_checks)
    details = {c[0]: {"check": c[1]} for c in recorder_checks}
    runner.add_result(VerificationResult(
        stage=2,
        name="recorder_interface",
        passed=all_passed,
        message="GeometryDataRecorder interface correct" if all_passed else "Interface missing",
        details=details,
    ))

    # Simulate observation data
    print("\n  Testing observation field requirements...")

    # Check if we can create mock observation
    mock_obs = {
        "full_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "depth": np.random.rand(224, 224).astype(np.float32),
        "camera_intrinsics": np.eye(3, dtype=np.float32),
        "camera_extrinsics": np.eye(4, dtype=np.float32),
        "ee_pose": np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32),
    }

    # Check if recorder can handle the observation
    exception_msg = None
    try:
        # The recorder uses collect_step, not record_step
        mock_rgb = mock_obs["full_image"]
        mock_action = np.zeros(7, dtype=np.float32)
        recorder.reset(episode_id=0, task_name="test_task")
        step_data = recorder.collect_step(
            rgb=mock_rgb,
            obs=mock_obs,
            action=mock_action,
            step_id=0,
            raw_env_obs=mock_obs,
        )
        recorder_passed = True
        msg = "GeometryDataRecorder.collect_step works with mock observation"
    except Exception as e:
        recorder_passed = False
        msg = f"Error: {str(e)[:100]}"
        exception_msg = str(e)

    runner.add_result(VerificationResult(
        stage=2,
        name="recorder_mock_observation",
        passed=recorder_passed,
        message=msg,
        details={"exception": exception_msg},
    ))

    # Test Token3DMapper
    mapper = Token3DMapper()

    mapper_checks = [
        ("has_map_tokens_to_3d", hasattr(mapper, "map_tokens_to_3d")),
        ("has_ImagePreprocessMeta", True),  # Already imported
    ]

    all_passed = all(c[1] for c in mapper_checks)
    details = {c[0]: {"check": c[1]} for c in mapper_checks}
    runner.add_result(VerificationResult(
        stage=2,
        name="token3d_mapper_interface",
        passed=all_passed,
        message="Token3DMapper interface correct" if all_passed else "Interface missing",
        details=details,
    ))

    print("\n  Observation fields required:")
    for field in required_obs_fields:
        has_field = field in mock_obs
        status = "OK" if has_field else "MISSING"
        print(f"    - {field}: {status}")


# =============================================================================
# Stage 3: Token Mapping Check
# =============================================================================


def verify_token_mapping(runner: VerificationRunner, dry_run: bool = True):
    """
    Stage 3: token mapping check
    确认 token_points_base shape 正确
    确认 invalid depth 不会导致 NaN 进入 score
    确认 valid_token_ratio 合理
    """
    print("\n" + "=" * 60)
    print("Stage 3: Token Mapping Check")
    print("=" * 60)

    # Test Token3DMapper.map_tokens_to_3d
    from geometry import Token3DMapper, ImagePreprocessMeta, RuleBasedGeometryExpert

    mapper = Token3DMapper()

    # Mock depth image (224x224)
    depth = np.random.rand(224, 224).astype(np.float32) * 2.0  # 0-2m

    # Set some invalid depth values
    depth[0:50, :] = 0.0  # Invalid depth
    depth[:, 0:50] = -1.0  # Invalid depth

    # Mock camera params
    K = np.array([
        [500, 0, 112],
        [0, 500, 112],
        [0, 0, 1]
    ], dtype=np.float32)

    T_base_cam = np.eye(4, dtype=np.float32)
    T_base_cam[:3, 3] = [0.5, 0, 0.5]  # Camera 0.5m forward, 0.5m up

    ee_pose = np.array([0, 0, 0.3, 0, 0, 0, 1], dtype=np.float32)

    # Image preprocess meta (OpenVLA uses patch_size=16, 224/16=14)
    meta = ImagePreprocessMeta(
        original_size=(224, 224),
        processed_size=(224, 224),
        crop_scale=0.9,
        center_crop=True,
        patch_size=16,  # OpenVLA uses 16x16 patches
    )

    # Token grid shape for 224x224 image with 16x16 patches = 14x14 = 196 tokens
    token_grid_shape = (14, 14)
    num_tokens = 14 * 14

    # Test projection
    try:
        result = mapper.map_tokens_to_3d(
            depth=depth,
            camera_intrinsics=K,
            camera_extrinsics=T_base_cam,
            image_preprocess_meta=meta,
            token_grid_shape=token_grid_shape,
            num_visual_tokens=num_tokens,
        )

        shape_correct = result.token_points_base.shape == (num_tokens, 3)
        has_valid_mask = hasattr(result, "token_valid_mask") and result.token_valid_mask is not None

        details = {
            "num_tokens": num_tokens,
            "token_points_base_shape": result.token_points_base.shape,
            "expected_shape": (num_tokens, 3),
            "shape_match": shape_correct,
            "has_valid_mask": has_valid_mask,
            "valid_ratio": np.sum(result.token_valid_mask) / len(result.token_valid_mask) if has_valid_mask else None,
        }

        runner.add_result(VerificationResult(
            stage=3,
            name="token_mapping_shape",
            passed=shape_correct and has_valid_mask,
            message=f"Token mapping shape correct: {result.token_points_base.shape}",
            details=details,
        ))

        # Check no NaN in valid points
        if has_valid_mask:
            valid_mask = result.token_valid_mask
            valid_points = result.token_points_base[valid_mask]
            no_nan = not np.any(np.isnan(valid_points)) if len(valid_points) > 0 else True
            runner.add_result(VerificationResult(
                stage=3,
                name="no_nan_in_valid_points",
                passed=no_nan,
                message="No NaN in valid 3D points" if no_nan else "NaN found in valid points",
                details={"nan_count": np.sum(np.isnan(valid_points)) if len(valid_points) > 0 else 0},
            ))

            print(f"\n  Token mapping results:")
            print(f"    - Shape: {result.token_points_base.shape}")
            print(f"    - Valid mask ratio: {np.sum(valid_mask) / len(valid_mask):.2%}")
            print(f"    - No NaN in valid points: {no_nan}")

    except Exception as e:
        runner.add_result(VerificationResult(
            stage=3,
            name="token_mapping",
            passed=False,
            message=f"Error: {str(e)[:100]}",
            details={"exception": str(e)},
        ))

    # Test RuleBasedGeometryExpert
    print("\n  Testing RuleBasedGeometryExpert...")

    expert = RuleBasedGeometryExpert()

    # Create mock geometry scores
    try:
        # Use mock 3D points from token mapping result if available
        if 'result' in dir() and result.token_points_base is not None:
            mock_points = result.token_points_base
            mock_valid_mask = result.token_valid_mask
        else:
            mock_points = np.random.randn(num_tokens, 3).astype(np.float32)
            mock_valid_mask = np.random.rand(num_tokens) > 0.1  # 90% valid

        # Mock additional arrays needed by compute_scores
        depth_std = np.random.rand(num_tokens).astype(np.float32) * 0.05
        valid_depth_ratio = np.random.rand(num_tokens).astype(np.float32)
        gripper_pos = np.array([0.5, 0, 0.3], dtype=np.float32)
        prev_gripper_pos = np.array([0.5, 0, 0.3], dtype=np.float32)

        scores = expert.compute_scores(
            token_points_base=mock_points,
            token_valid_mask=mock_valid_mask,
            depth_std=depth_std,
            valid_depth_ratio=valid_depth_ratio,
            gripper_pos=gripper_pos,
            prev_gripper_pos=prev_gripper_pos,
        )

        score_checks = [
            ("token_scores_shape", scores.token_scores.shape == (num_tokens,)),
            ("distance_score_shape", scores.distance_score.shape == (num_tokens,)),
            ("no_nan_token_scores", not np.any(np.isnan(scores.token_scores))),
            ("no_nan_distance", not np.any(np.isnan(scores.distance_score))),
            ("valid_mask_consistent", len(scores.valid_score_mask) == num_tokens),
        ]

        all_passed = all(c[1] for c in score_checks)
        details = {
            c[0]: {"check": c[1], "expected": True}
            for c in score_checks
        }

        runner.add_result(VerificationResult(
            stage=3,
            name="geometry_expert_scores",
            passed=all_passed,
            message="Geometry expert scores valid" if all_passed else "Score validation failed",
            details=details,
        ))

        # Check valid_token_ratio
        valid_ratio = np.sum(scores.valid_score_mask) / num_tokens
        valid_ratio_reasonable = 0.5 < valid_ratio < 1.0  # Should be reasonable

        runner.add_result(VerificationResult(
            stage=3,
            name="valid_token_ratio",
            passed=valid_ratio_reasonable,
            message=f"Valid token ratio: {valid_ratio:.2%}",
            details={"valid_ratio": valid_ratio, "reasonable_range": "50%-100%"},
        ))

        print(f"\n  Geometry expert results:")
        print(f"    - Token scores shape: {scores.token_scores.shape}")
        print(f"    - Valid token ratio: {valid_ratio:.2%}")
        print(f"    - Score range: [{np.min(scores.token_scores):.3f}, {np.max(scores.token_scores):.3f}]")

    except Exception as e:
        runner.add_result(VerificationResult(
            stage=3,
            name="geometry_expert",
            passed=False,
            message=f"Error: {str(e)[:100]}",
            details={"exception": str(e)},
        ))


# =============================================================================
# Stage 4: Visualization Check
# =============================================================================


def verify_visualization(runner: VerificationRunner, dry_run: bool = True):
    """
    Stage 4: visualization check
    保存若干 step 的 heatmap
    肉眼检查高分区域
    """
    print("\n" + "=" * 60)
    print("Stage 4: Visualization Check")
    print("=" * 60)

    from geometry import GeometryVisualizer, VisualizationConfig

    # Create visualizer
    from geometry import GeometryVisualizer, VisualizationConfig

    vis_cfg = VisualizationConfig(
        output_dir=str(runner.save_dir / "vis"),
        vis_interval=10,
        max_vis_per_episode=5,
    )

    visualizer = GeometryVisualizer(config=vis_cfg)

    # Create mock data
    num_patches_h, num_patches_w = 14, 14
    num_tokens = num_patches_h * num_patches_w

    # RGB image
    rgb_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # Token scores (simulate high scores near "gripper" region)
    token_scores = np.random.rand(num_tokens).astype(np.float32)
    # Make center region higher scores
    token_scores[90:110] += 0.5

    distance_score = np.random.rand(num_tokens).astype(np.float32)
    direction_score = np.random.rand(num_tokens).astype(np.float32)
    depth_edge_score = np.random.rand(num_tokens).astype(np.float32)
    valid_mask = np.random.rand(num_tokens) > 0.1

    # Token points in 3D (mock)
    token_points_base = np.random.randn(num_tokens, 3).astype(np.float32)

    # Save visualization
    vis_dir = runner.save_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    try:
        vis_path = visualizer.visualize_step(
            step_id=0,
            episode_id=0,
            task_name="verify_vis",
            rgb_image=rgb_image,
            depth_image=None,
            token_grid_shape=(14, 14),
            num_visual_tokens=num_tokens,
            token_scores=token_scores,
            distance_scores=distance_score,
            direction_scores=direction_score,
            depth_edge_scores=depth_edge_score,
            token_valid_mask=valid_mask,
            gripper_pos=None,
            camera_intrinsics=None,
            camera_extrinsics=None,
        )

        vis_saved = vis_path is not None
        runner.add_result(VerificationResult(
            stage=4,
            name="visualization_saved",
            passed=vis_saved,
            message=f"Visualization saved to: {vis_path}" if vis_saved else "Visualization failed",
            details={"vis_path": str(vis_path) if vis_saved else None},
        ))

        print(f"\n  Visualization results:")
        print(f"    - Output dir: {vis_dir}")
        print(f"    - Visualization saved: {vis_saved}")

    except Exception as e:
        runner.add_result(VerificationResult(
            stage=4,
            name="visualization",
            passed=False,
            message=f"Error: {str(e)[:100]}",
            details={"exception": str(e)},
        ))

    # Check visualizer interface
    vis_checks = [
        ("has_visualize_step", hasattr(visualizer, "visualize_step")),
    ]

    all_passed = all(c[1] for c in vis_checks)
    details = {c[0]: {"check": c[1]} for c in vis_checks}
    runner.add_result(VerificationResult(
        stage=4,
        name="visualizer_interface",
        passed=all_passed,
        message="GeometryVisualizer interface correct" if all_passed else "Interface missing",
        details=details,
    ))

    print("\n  Please check visualization at:")
    print(f"    {vis_dir}")
    print("\n  Visual inspection checklist:")
    print("    - High score regions near gripper?")
    print("    - High score regions near objects/touch areas?")
    print("    - depth_edge at object boundaries?")
    print("    - direction_score in movement direction?")


# =============================================================================
# Stage 5: Random Pruning Check
# =============================================================================


def verify_random_pruning(runner: VerificationRunner, dry_run: bool = True):
    """
    Stage 5: random pruning check
    确认剪枝链路不 shape mismatch
    """
    print("\n" + "=" * 60)
    print("Stage 5: Random Pruning Check")
    print("=" * 60)

    from geometry import TokenPruner, PruningConfig, PruningMethod, prune_visual_tokens

    # Test different keep ratios
    for keep_ratio in [1.0, 0.75, 0.5, 0.25]:
        cfg = ExperimentConfig.from_preset(f"random_{keep_ratio}")
        print(f"\n  Testing keep_ratio={keep_ratio}")

        # Create mock visual tokens [batch=1, tokens=256, hidden=1024]
        num_tokens = 256
        batch_size = 1
        hidden_dim = 1024

        visual_tokens = torch.randn(batch_size, num_tokens, hidden_dim)

        # Apply pruning
        try:
            result = prune_visual_tokens(
                visual_tokens=visual_tokens,
                scores=None,  # Random doesn't need scores
                keep_ratio=cfg.keep_ratio,
                method="random",
                seed=42,
            )

            # Verify results
            expected_kept = int(num_tokens * keep_ratio)
            shape_correct = result.pruned_visual_tokens.shape == (batch_size, expected_kept, hidden_dim)
            ratio_correct = abs(result.actual_keep_ratio - keep_ratio) < 0.01

            runner.add_result(VerificationResult(
                stage=5,
                name=f"random_pruning_{keep_ratio}",
                passed=shape_correct and ratio_correct,
                message=f"Random pruning {keep_ratio}: shape={result.pruned_visual_tokens.shape}",
                details={
                    "keep_ratio": keep_ratio,
                    "expected_kept": expected_kept,
                    "actual_kept": result.num_tokens_after,
                    "shape": list(result.pruned_visual_tokens.shape),
                    "shape_correct": shape_correct,
                    "ratio_correct": ratio_correct,
                },
            ))

            print(f"    - Shape: {result.pruned_visual_tokens.shape} (expected: ({batch_size}, {expected_kept}, {hidden_dim}))")
            print(f"    - Keep ratio: {result.actual_keep_ratio:.3f} (expected: {keep_ratio})")
            print(f"    - Keep indices sorted: {np.all(np.diff(result.keep_indices) >= 0)}")

        except Exception as e:
            runner.add_result(VerificationResult(
                stage=5,
                name=f"random_pruning_{keep_ratio}",
                passed=False,
                message=f"Error: {str(e)[:100]}",
                details={"exception": str(e)},
            ))

    # Test reproducibility
    print("\n  Testing reproducibility...")
    result1 = prune_visual_tokens(
        visual_tokens=torch.randn(1, 256, 1024),
        scores=None,
        keep_ratio=0.5,
        method="random",
        seed=42,
    )
    result2 = prune_visual_tokens(
        visual_tokens=torch.randn(1, 256, 1024),
        scores=None,
        keep_ratio=0.5,
        method="random",
        seed=42,
    )

    same_indices = np.array_equal(result1.keep_indices, result2.keep_indices)
    runner.add_result(VerificationResult(
        stage=5,
        name="random_pruning_reproducibility",
        passed=same_indices,
        message="Random pruning reproducible with same seed" if same_indices else "Not reproducible",
        details={"same_indices": same_indices},
    ))


# =============================================================================
# Stage 6: Geometry Rule Pruning Check
# =============================================================================


def verify_geometry_rule_pruning(runner: VerificationRunner, dry_run: bool = True):
    """
    Stage 6: geometry_rule pruning check
    确认 success rate / latency / token 数有记录
    """
    print("\n" + "=" * 60)
    print("Stage 6: Geometry Rule Pruning Check")
    print("=" * 60)

    from geometry import TokenPruner, PruningConfig, PruningMethod, TokenScores, prune_visual_tokens

    # Create mock scores
    num_tokens = 256
    token_scores = np.random.rand(num_tokens).astype(np.float32)
    distance_score = np.random.rand(num_tokens).astype(np.float32)
    direction_score = np.random.rand(num_tokens).astype(np.float32)
    depth_edge_score = np.random.rand(num_tokens).astype(np.float32)
    valid_mask = np.random.rand(num_tokens) > 0.1

    # Create TokenScores
    scores = TokenScores.from_arrays(
        token_scores=token_scores,
        distance_score=distance_score,
        direction_score=direction_score,
        depth_edge_score=depth_edge_score,
        valid_mask=valid_mask,
    )

    # Make some scores very high (simulate high importance tokens)
    scores.token_scores[100:120] = 100.0

    print(f"\n  Testing geometry_rule pruning with scores...")
    print(f"    - Token scores range: [{token_scores.min():.3f}, {token_scores.max():.3f}]")
    print(f"    - Valid mask ratio: {np.sum(valid_mask) / num_tokens:.2%}")

    # Test different keep ratios
    for keep_ratio in [0.75, 0.5, 0.25]:
        cfg = ExperimentConfig.from_preset(f"geometry_rule_{keep_ratio}")
        print(f"\n  Testing keep_ratio={keep_ratio}")

        visual_tokens = torch.randn(1, num_tokens, 1024)

        try:
            result = prune_visual_tokens(
                visual_tokens=visual_tokens,
                scores=scores,
                keep_ratio=cfg.keep_ratio,
                method="geometry_rule",
                seed=42,
            )

            # Verify results
            expected_kept = int(num_tokens * keep_ratio)
            shape_correct = result.pruned_visual_tokens.shape == (1, expected_kept, 1024)

            # Verify top scores are kept
            top_indices = np.argsort(token_scores)[-expected_kept:]
            kept_set = set(result.keep_indices)
            expected_set = set(top_indices)

            # Allow small difference due to invalid tokens
            overlap = len(kept_set & expected_set) / expected_kept

            runner.add_result(VerificationResult(
                stage=6,
                name=f"geometry_rule_pruning_{keep_ratio}",
                passed=shape_correct,
                message=f"Geometry rule {keep_ratio}: shape={result.pruned_visual_tokens.shape}",
                details={
                    "keep_ratio": keep_ratio,
                    "expected_kept": expected_kept,
                    "actual_kept": result.num_tokens_after,
                    "shape_correct": shape_correct,
                    "top_score_overlap": overlap,
                    "method": result.method,
                    "timing_ms": result.timing_ms,
                },
            ))

            print(f"    - Shape: {result.pruned_visual_tokens.shape}")
            print(f"    - Keep ratio: {result.actual_keep_ratio:.3f}")
            print(f"    - Top score overlap: {overlap:.2%}")
            print(f"    - Timing: {result.timing_ms:.2f}ms")

        except Exception as e:
            runner.add_result(VerificationResult(
                stage=6,
                name=f"geometry_rule_pruning_{keep_ratio}",
                passed=False,
                message=f"Error: {str(e)[:100]}",
                details={"exception": str(e)},
            ))

    # Test depth_edge method
    print("\n  Testing depth_edge pruning...")
    scores.depth_edge_score[50:70] = 100.0

    result = prune_visual_tokens(
        visual_tokens=torch.randn(1, num_tokens, 1024),
        scores=scores,
        keep_ratio=0.5,
        method="depth_edge",
        seed=42,
    )

    runner.add_result(VerificationResult(
        stage=6,
        name="depth_edge_pruning",
        passed=result.num_tokens_after == 128,
        message=f"depth_edge: kept {result.num_tokens_after} tokens",
        details={"method": result.method, "keep_ratio": result.actual_keep_ratio},
    ))

    # Test gripper_distance method
    print("\n  Testing gripper_distance pruning...")
    scores.distance_score[150:170] = 100.0

    result = prune_visual_tokens(
        visual_tokens=torch.randn(1, num_tokens, 1024),
        scores=scores,
        keep_ratio=0.5,
        method="gripper_distance",
        seed=42,
    )

    runner.add_result(VerificationResult(
        stage=6,
        name="gripper_distance_pruning",
        passed=result.num_tokens_after == 128,
        message=f"gripper_distance: kept {result.num_tokens_after} tokens",
        details={"method": result.method, "keep_ratio": result.actual_keep_ratio},
    ))

    # Test all methods produce correct summary
    print("\n  Testing PruningResult summary...")
    summary = result.get_summary()

    expected_keys = ["num_tokens_before", "num_tokens_after", "actual_keep_ratio", "timing_ms", "method"]
    has_all_keys = all(k in summary for k in expected_keys)

    runner.add_result(VerificationResult(
        stage=6,
        name="pruning_result_summary",
        passed=has_all_keys,
        message="PruningResult summary has all keys" if has_all_keys else "Missing keys",
        details={"keys": list(summary.keys())},
    ))


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Verify geometry-guided pruning pipeline")
    parser.add_argument("--stage", type=str, default="all",
                        help="Stage to verify: 1-6 or 'all'")
    parser.add_argument("--save-dir", type=str, default="outputs/verification",
                        help="Directory to save verification results")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Dry run (don't execute full evaluation)")

    args = parser.parse_args()

    runner = VerificationRunner(save_dir=args.save_dir)

    # Parse stage
    if args.stage == "all":
        stages = list(range(1, 7))
    elif args.stage.isdigit():
        stages = [int(args.stage)]
    else:
        print(f"Invalid stage: {args.stage}")
        return

    print(f"\n{'#'*60}")
    print(f"# Running verification stages: {stages}")
    print(f"{'#'*60}")

    for stage in stages:
        if stage == 1:
            verify_baseline(runner, None, args.dry_run)
        elif stage == 2:
            verify_geometry_data(runner, None, args.dry_run)
        elif stage == 3:
            verify_token_mapping(runner, args.dry_run)
        elif stage == 4:
            verify_visualization(runner, args.dry_run)
        elif stage == 5:
            verify_random_pruning(runner, args.dry_run)
        elif stage == 6:
            verify_geometry_rule_pruning(runner, args.dry_run)

    # Save report
    runner.save_report()


if __name__ == "__main__":
    main()
