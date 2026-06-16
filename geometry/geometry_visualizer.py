"""
GeometryVisualizer: Visualizes token scores as heatmaps for analysis.

This module generates visualization outputs for geometry-based token analysis:
1. RGB image with gripper projection
2. Depth image
3. Token score heatmaps (overall, distance, direction, depth_edge)
4. Token valid mask heatmap
5. (Future) Keep mask heatmap after pruning

Output is saved as a grid of subplots to a single PNG file per step.

Visualization Pipeline:
1. Convert token-level scores to 2D heatmaps matching token_grid_shape
2. Optionally project gripper position onto RGB image
3. Arrange all heatmaps in a grid layout
4. Save to file with proper annotations

Usage:
    visualizer = GeometryVisualizer(
        output_dir="logs/geometry_vis",
        vis_interval=10,  # Save every 10 steps
        max_vis_per_episode=5,
    )
    visualizer.visualize_step(
        step_id=0,
        episode_id=0,
        task_name="task_0",
        rgb_image=rgb,
        depth_image=depth,
        mapping_result=mapping_result,
        score_result=score_result,
        camera_intrinsics=K,
        gripper_pos=gripper_xyz,
    )
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Try to import visualization libraries
try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("[GeometryVisualizer] OpenCV (cv2) not available. Gripper projection disabled.")

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False
    logger.warning("[GeometryVisualizer] Matplotlib not available. Visualization disabled.")


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class VisualizationConfig:
    """Configuration for geometry visualization."""

    output_dir: str = "logs/geometry_vis"
    vis_interval: int = 10  # Save every N steps
    max_vis_per_episode: int = 5  # Maximum visualizations per episode
    figure_size: Tuple[int, int] = (16, 12)  # Figure size (width, height)
    colormap: str = "jet"  # Matplotlib colormap for heatmaps
    dpi: int = 100  # DPI for saved figures
    save_raw_images: bool = True  # Also save raw RGB/depth images
    annotate_scores: bool = True  # Add colorbar annotations
    grid_layout: str = "auto"  # Layout: "auto", "2x3", "3x2", "custom"

    def __post_init__(self):
        """Validate configuration."""
        if self.vis_interval < 1:
            self.vis_interval = 1
        if self.max_vis_per_episode < 1:
            self.max_vis_per_episode = 1


# =============================================================================
# Geometry Visualizer
# =============================================================================


class GeometryVisualizer:
    """
    Visualizes geometry-based token scores as heatmaps.

    This visualizer creates PNG files showing:
    - RGB image with gripper projection (if available)
    - Depth image
    - Combined token score heatmap
    - Individual score component heatmaps
    - Token validity mask

    Saved to: {output_dir}/{task_name}/episode_{episode_id}/step_{step_id}.png
    """

    def __init__(
        self,
        config: Optional[VisualizationConfig] = None,
        enabled: bool = True,
        debug: bool = False,
    ):
        """
        Initialize the visualizer.

        Args:
            config: Visualization configuration. Uses default if None.
            enabled: Whether visualization is enabled.
            debug: Whether to enable debug logging.
        """
        self.config = config or VisualizationConfig()
        self.enabled = enabled
        self.debug = debug

        # Track visualization count per episode
        self._vis_counts: Dict[int, int] = {}  # episode_id -> count

        if not self.enabled:
            logger.info("[GeometryVisualizer] Disabled (no-op mode)")
            return

        if not MPL_AVAILABLE:
            logger.warning("[GeometryVisualizer] Matplotlib not available. Visualization disabled.")
            self.enabled = False
            return

        # Create output directory
        self._output_dir = Path(self.config.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"[GeometryVisualizer] Initialized: output_dir={self._output_dir}, "
            f"vis_interval={self.config.vis_interval}, "
            f"max_vis_per_episode={self.config.max_vis_per_episode}"
        )

    def should_visualize(self, step_id: int, episode_id: int) -> bool:
        """
        Determine whether to visualize the current step.

        Args:
            step_id: Current step ID.
            episode_id: Current episode ID.

        Returns:
            True if visualization should be performed.
        """
        if not self.enabled:
            return False

        # Check interval
        if step_id % self.config.vis_interval != 0:
            return False

        # Check max per episode
        current_count = self._vis_counts.get(episode_id, 0)
        if current_count >= self.config.max_vis_per_episode:
            return False

        return True

    def visualize_step(
        self,
        step_id: int,
        episode_id: int,
        task_name: str,
        rgb_image: np.ndarray,
        depth_image: Optional[np.ndarray],
        token_grid_shape: Tuple[int, int],
        num_visual_tokens: int,
        token_scores: Optional[np.ndarray] = None,
        distance_scores: Optional[np.ndarray] = None,
        direction_scores: Optional[np.ndarray] = None,
        depth_edge_scores: Optional[np.ndarray] = None,
        token_valid_mask: Optional[np.ndarray] = None,
        keep_mask: Optional[np.ndarray] = None,
        gripper_pos: Optional[np.ndarray] = None,
        camera_intrinsics: Optional[np.ndarray] = None,
        camera_extrinsics: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Visualize token scores for a single step.

        Args:
            step_id: Current step ID.
            episode_id: Current episode ID.
            task_name: Name of the current task.
            rgb_image: RGB image (H x W x 3), uint8.
            depth_image: Depth image (H x W), float32 in meters.
            token_grid_shape: Shape (H_patches, W_patches) of the token grid.
            num_visual_tokens: Total number of visual tokens.
            token_scores: Combined token scores [N].
            distance_scores: Distance-based scores [N].
            direction_scores: Direction-based scores [N].
            depth_edge_scores: Depth edge scores [N].
            token_valid_mask: Boolean mask [N] for valid tokens.
            gripper_pos: Gripper position [3] in base frame.
            camera_intrinsics: 3x3 camera intrinsics matrix K.
            camera_extrinsics: 4x4 camera-to-base transform T_base_cam.
            metadata: Additional metadata to display.

        Returns:
            Path to saved visualization file, or None if visualization was skipped/failed.
        """
        if not self.should_visualize(step_id, episode_id):
            return None

        try:
            return self._visualize_impl(
                step_id=step_id,
                episode_id=episode_id,
                task_name=task_name,
                rgb_image=rgb_image,
                depth_image=depth_image,
                token_grid_shape=token_grid_shape,
                num_visual_tokens=num_visual_tokens,
                token_scores=token_scores,
                distance_scores=distance_scores,
                direction_scores=direction_scores,
                depth_edge_scores=depth_edge_scores,
                token_valid_mask=token_valid_mask,
                keep_mask=keep_mask,
                gripper_pos=gripper_pos,
                camera_intrinsics=camera_intrinsics,
                camera_extrinsics=camera_extrinsics,
                metadata=metadata,
            )
        except Exception as e:
            logger.warning(f"[GeometryVisualizer] Visualization failed: {e}")
            if self.debug:
                import traceback

                traceback.print_exc()
            return None

    def _visualize_impl(
        self,
        step_id: int,
        episode_id: int,
        task_name: str,
        rgb_image: np.ndarray,
        depth_image: Optional[np.ndarray],
        token_grid_shape: Tuple[int, int],
        num_visual_tokens: int,
        token_scores: Optional[np.ndarray],
        distance_scores: Optional[np.ndarray],
        direction_scores: Optional[np.ndarray],
        depth_edge_scores: Optional[np.ndarray],
        token_valid_mask: Optional[np.ndarray],
        keep_mask: Optional[np.ndarray],
        gripper_pos: Optional[np.ndarray],
        camera_intrinsics: Optional[np.ndarray],
        camera_extrinsics: Optional[np.ndarray],
        metadata: Optional[Dict[str, Any]],
    ) -> str:
        """Internal implementation of visualization."""
        import matplotlib.pyplot as plt

        # Update visualization count
        self._vis_counts[episode_id] = self._vis_counts.get(episode_id, 0) + 1

        # Create output directory
        task_dir = self._output_dir / task_name
        episode_dir = task_dir / f"episode_{episode_id:04d}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        # Determine which components to visualize
        components = self._prepare_components(
            rgb_image=rgb_image,
            depth_image=depth_image,
            token_grid_shape=token_grid_shape,
            num_visual_tokens=num_visual_tokens,
            token_scores=token_scores,
            distance_scores=distance_scores,
            direction_scores=direction_scores,
            depth_edge_scores=depth_edge_scores,
            token_valid_mask=token_valid_mask,
            keep_mask=keep_mask,
        )

        # Create figure
        n_components = len(components)
        n_cols = 3
        n_rows = (n_components + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        if n_components == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # Plot each component
        for idx, (title, data, cmap, vmin, vmax) in enumerate(components):
            ax = axes[idx]
            if data is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=12)
                ax.set_title(title)
                ax.axis("off")
                continue

            # Handle 2D vs 3D data
            if len(data.shape) == 3 and data.shape[2] == 3:
                # RGB image
                if data.dtype != np.uint8:
                    data = (np.clip(data, 0, 1) * 255).astype(np.uint8)
                ax.imshow(data)
            else:
                # 2D heatmap
                im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
                if self.config.annotate_scores and data.dtype in [np.float32, np.float64]:
                    ax.text(
                        0.02, 0.98, f"mean={np.nanmean(data):.3f}",
                        transform=ax.transAxes, fontsize=8, va="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                    )
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax.set_title(title, fontsize=10)
            ax.axis("off")

        # Hide unused axes
        for idx in range(n_components, len(axes)):
            axes[idx].axis("off")

        # Add metadata as figure title
        metadata_text = self._format_metadata(metadata)
        if metadata_text:
            fig.suptitle(metadata_text, fontsize=10, y=1.02)

        plt.tight_layout()

        # Save figure
        output_path = episode_dir / f"step_{step_id:04d}.png"
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)

        if self.debug:
            logger.debug(f"[GeometryVisualizer] Saved: {output_path}")

        return str(output_path)

    def _prepare_components(
        self,
        rgb_image: np.ndarray,
        depth_image: Optional[np.ndarray],
        token_grid_shape: Tuple[int, int],
        num_visual_tokens: int,
        token_scores: Optional[np.ndarray],
        distance_scores: Optional[np.ndarray],
        direction_scores: Optional[np.ndarray],
        depth_edge_scores: Optional[np.ndarray],
        token_valid_mask: Optional[np.ndarray],
        keep_mask: Optional[np.ndarray],
    ) -> List[Tuple[str, Optional[np.ndarray], str, float, float]]:
        """Prepare visualization components."""
        components = []

        # 1. RGB image (optional gripper projection)
        rgb_display = rgb_image.copy()
        if self.debug and CV2_AVAILABLE:
            # Could add gripper projection here
            pass
        components.append(("RGB Image", rgb_display, "rgb", 0, 255))

        # 2. Depth image
        if depth_image is not None:
            depth_display = self._prepare_depth_image(depth_image)
            components.append(("Depth Image", depth_display, "viridis", None, None))
        else:
            components.append(("Depth Image", None, "viridis", 0, 1))

        # 3. Combined token score
        if token_scores is not None:
            score_heatmap = self._tokens_to_heatmap(token_scores, token_grid_shape, num_visual_tokens)
            components.append(("Combined Score", score_heatmap, self.config.colormap, 0, 1))
        else:
            components.append(("Combined Score", None, self.config.colormap, 0, 1))

        # 4. Distance score
        if distance_scores is not None:
            dist_heatmap = self._tokens_to_heatmap(distance_scores, token_grid_shape, num_visual_tokens)
            components.append(("Distance Score", dist_heatmap, self.config.colormap, 0, 1))
        else:
            components.append(("Distance Score", None, self.config.colormap, 0, 1))

        # 5. Direction score
        if direction_scores is not None:
            dir_heatmap = self._tokens_to_heatmap(direction_scores, token_grid_shape, num_visual_tokens)
            components.append(("Direction Score", dir_heatmap, self.config.colormap, 0, 1))
        else:
            components.append(("Direction Score", None, self.config.colormap, 0, 1))

        # 6. Depth edge score
        if depth_edge_scores is not None:
            edge_heatmap = self._tokens_to_heatmap(depth_edge_scores, token_grid_shape, num_visual_tokens)
            components.append(("Depth Edge Score", edge_heatmap, self.config.colormap, 0, 1))
        else:
            components.append(("Depth Edge Score", None, self.config.colormap, 0, 1))

        # 7. Valid mask
        if token_valid_mask is not None:
            mask_heatmap = self._tokens_to_heatmap(
                token_valid_mask.astype(np.float32),
                token_grid_shape,
                num_visual_tokens,
            )
            components.append(("Valid Mask", mask_heatmap, "gray", 0, 1))
        else:
            components.append(("Valid Mask", None, "gray", 0, 1))

        # 8. Keep mask after pruning
        if keep_mask is not None:
            keep_heatmap = self._tokens_to_heatmap(
                keep_mask.astype(np.float32),
                token_grid_shape,
                num_visual_tokens,
            )
            components.append(("Keep Mask", keep_heatmap, "gray", 0, 1))

        return components

    def _tokens_to_heatmap(
        self,
        token_values: np.ndarray,
        token_grid_shape: Tuple[int, int],
        num_visual_tokens: int,
    ) -> Optional[np.ndarray]:
        """
        Convert token values to 2D heatmap.

        Args:
            token_values: Token values [N].
            token_grid_shape: Shape (H, W) of the token grid.
            num_visual_tokens: Total number of tokens.

        Returns:
            2D heatmap [H, W], or None if conversion fails.
        """
        tokens_per_grid = token_grid_shape[0] * token_grid_shape[1]

        # Check for multi-encoder case
        if num_visual_tokens > tokens_per_grid:
            # Multiple encoder grids - for now, visualize first grid
            # or average if token count is multiple
            n_encoders = num_visual_tokens // tokens_per_grid
            if self.debug:
                logger.debug(
                    f"[GeometryVisualizer] Multi-encoder: {num_visual_tokens} tokens, "
                    f"grid={token_grid_shape}, visualizing first encoder (n={n_encoders})"
                )
            token_values = token_values[:tokens_per_grid]

        # Handle shape mismatch
        if len(token_values) != tokens_per_grid:
            logger.warning(
                f"[GeometryVisualizer] Token count mismatch: {len(token_values)} != {tokens_per_grid}"
            )
            # Try to reshape anyway, or pad
            if len(token_values) < tokens_per_grid:
                padded = np.full(tokens_per_grid, np.nan)
                padded[: len(token_values)] = token_values
                token_values = padded
            else:
                token_values = token_values[:tokens_per_grid]

        # Reshape to 2D grid
        try:
            heatmap = token_values.reshape(token_grid_shape)
            return heatmap
        except Exception as e:
            logger.warning(f"[GeometryVisualizer] Failed to reshape tokens to heatmap: {e}")
            return None

    def _prepare_depth_image(self, depth: np.ndarray) -> np.ndarray:
        """Prepare depth image for visualization."""
        depth = np.asarray(depth, dtype=np.float32)

        # Handle invalid depth
        valid_mask = (depth > 0) & (depth < 10.0)
        if not np.any(valid_mask):
            return np.zeros_like(depth)

        # Normalize to [0, 1]
        d_min = np.percentile(depth[valid_mask], 5)
        d_max = np.percentile(depth[valid_mask], 95)
        if d_max <= d_min:
            d_max = d_min + 0.1

        depth_normalized = np.clip((depth - d_min) / (d_max - d_min), 0, 1)
        depth_normalized[~valid_mask] = 0

        return depth_normalized

    def _format_metadata(self, metadata: Optional[Dict[str, Any]]) -> str:
        """Format metadata for display."""
        if not metadata:
            return ""

        parts = []
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, float):
                parts.append(f"{key}={value:.3f}")
            elif isinstance(value, int):
                parts.append(f"{key}={value}")
            elif isinstance(value, str):
                parts.append(f"{key}={value}")
            else:
                parts.append(f"{key}={value}")

        return " | ".join(parts)

    def reset_episode(self, episode_id: int) -> None:
        """Reset visualization count for a new episode."""
        self._vis_counts[episode_id] = 0

    def get_output_dir(self) -> Path:
        """Get the visualization output directory."""
        return self._output_dir


# =============================================================================
# Debug/Test Functions
# =============================================================================


def debug_visualization():
    """
    Verify the visualization pipeline.

    Creates synthetic data and verifies:
    1. Heatmap conversion works
    2. Multi-encoder case handled
    3. Output file created
    """
    print("=" * 60)
    print("[GeometryVisualizer] Running debug tests...")
    print("=" * 60)

    if not MPL_AVAILABLE:
        print("[SKIP] Matplotlib not available")
        return

    import tempfile

    # Create visualizer with temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        config = VisualizationConfig(
            output_dir=tmpdir,
            vis_interval=10,  # Save every 10 steps
            max_vis_per_episode=3,
        )
        visualizer = GeometryVisualizer(config=config, enabled=True, debug=True)

        # Create synthetic data
        n_tokens = 256
        grid_shape = (16, 16)
        rgb = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        depth = np.random.rand(224, 224).astype(np.float32) * 3

        token_scores = np.random.rand(n_tokens).astype(np.float32)
        distance_scores = np.random.rand(n_tokens).astype(np.float32)
        direction_scores = np.random.rand(n_tokens).astype(np.float32)
        depth_edge_scores = np.random.rand(n_tokens).astype(np.float32)
        valid_mask = np.ones(n_tokens, dtype=np.bool_)

        # Test 1: Single step visualization
        print("\n[Test 1] Single step visualization")
        output_path = visualizer.visualize_step(
            step_id=0,
            episode_id=0,
            task_name="test_task",
            rgb_image=rgb,
            depth_image=depth,
            token_grid_shape=grid_shape,
            num_visual_tokens=n_tokens,
            token_scores=token_scores,
            distance_scores=distance_scores,
            direction_scores=direction_scores,
            depth_edge_scores=depth_edge_scores,
            token_valid_mask=valid_mask,
            metadata={"episode": 0, "step": 0, "score_mean": float(np.mean(token_scores))},
        )

        assert output_path is not None, "Output path should not be None"
        assert os.path.exists(output_path), f"Output file should exist: {output_path}"
        print(f"  [OK] Visualization saved: {output_path}")

        # Test 2: Interval check
        print("\n[Test 2] Interval check")
        # Reset counts and use a different episode
        visualizer._vis_counts[2] = 0
        # With vis_interval=10, step 0 should visualize, step 5 should not
        should_vis_0 = visualizer.should_visualize(step_id=0, episode_id=2)
        should_vis_5 = visualizer.should_visualize(step_id=5, episode_id=2)
        should_vis_10 = visualizer.should_visualize(step_id=10, episode_id=2)
        assert should_vis_0, "Step 0 should be visualized with interval=10"
        assert not should_vis_5, "Step 5 should not be visualized with interval=10"
        assert should_vis_10, "Step 10 should be visualized with interval=10"
        print(f"  [OK] Interval check passed: step_0={should_vis_0}, step_5={should_vis_5}, step_10={should_vis_10}")

        # Test 3: Max per episode check
        print("\n[Test 3] Max per episode check")
        # Episode 0 already has 1 visualization from Test 1 (at step 0)
        # Set to max (3) and verify no more visualizations
        visualizer._vis_counts[0] = 3  # At limit
        should_vis = visualizer.should_visualize(step_id=20, episode_id=0)  # 20 % 10 == 0
        assert not should_vis, "Should not visualize after max reached"
        print(f"  [OK] Max per episode check passed")

        # Test 4: Multi-encoder case (use episode 4)
        print("\n[Test 4] Multi-encoder visualization")
        visualizer._vis_counts[4] = 0
        output_path = visualizer.visualize_step(
            step_id=0,
            episode_id=4,
            task_name="test_task",
            rgb_image=rgb,
            depth_image=depth,
            token_grid_shape=(16, 16),
            num_visual_tokens=512,  # 2 encoders
            token_scores=np.random.rand(512).astype(np.float32),
            distance_scores=distance_scores,  # Only 256 tokens
            metadata={"multi_encoder": True},
        )
        assert output_path is not None, "Multi-encoder should still visualize"
        print(f"  [OK] Multi-encoder visualization: {output_path}")

        # Test 5: Heatmap conversion
        print("\n[Test 5] Heatmap conversion")
        scores = np.random.rand(256).astype(np.float32)
        heatmap = visualizer._tokens_to_heatmap(scores, (16, 16), 256)
        assert heatmap is not None, "Heatmap should not be None"
        assert heatmap.shape == (16, 16), f"Heatmap shape mismatch: {heatmap.shape}"
        print(f"  [OK] Heatmap shape: {heatmap.shape}")

        # Test 6: No visualization when disabled
        print("\n[Test 6] Disabled visualization")
        disabled_vis = GeometryVisualizer(enabled=False)
        output_path = disabled_vis.visualize_step(
            step_id=0,
            episode_id=0,
            task_name="test_task",
            rgb_image=rgb,
            depth_image=depth,
            token_grid_shape=grid_shape,
            num_visual_tokens=n_tokens,
        )
        assert output_path is None, "Disabled visualizer should return None"
        print(f"  [OK] Disabled visualizer returns None")

        print("\n" + "=" * 60)
        print("[GeometryVisualizer] All debug tests PASSED!")
        print("=" * 60)


if __name__ == "__main__":
    debug_visualization()
