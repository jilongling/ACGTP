"""
Token3DMapper: Maps 2D visual tokens to 3D base coordinates.

This module projects visual tokens from OpenVLA's ViT backbone to 3D coordinates
in the robot base frame using depth images and camera parameters.

Core algorithm:
1. For each visual token, identify its corresponding 2D patch in the image
2. Extract depth values from the patch (using median for robustness)
3. Backproject 2D + depth to 3D camera coordinates
4. Transform camera coordinates to robot base coordinates

Key design principles:
1. No hardcoded token counts - read from visual_tokens.shape[1]
2. Token grid shape inferred from token count, not hardcoded
3. Handle multi-encoder fused models (DinoSigLIP) by mapping all tokens
4. Invalid depth tokens get NaN coordinates and False in valid_mask
5. Pure numpy implementation, no external geometry libraries
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class ImagePreprocessMeta:
    """
    Metadata describing the image preprocessing pipeline applied to the RGB image
    before it was fed to the vision encoder.

    This information is critical for correctly mapping visual tokens back to
    their corresponding 2D pixel positions in the original depth image.

    Attributes:
        original_size: Original image dimensions (H, W) before any preprocessing.
        processed_size: Image dimensions (H, W) after resize but before center crop.
        crop_scale: Center crop scale factor (e.g., 0.9 means 90% area is kept).
        center_crop: Whether center crop is applied after resize.
        patch_size: ViT patch size in pixels (default 14 for OpenVLA).
        is_fused: Whether this is a fused multi-encoder model (e.g., DinoSigLIP).
        num_encoders: Number of vision encoders (1 for single, 2 for fused).
    """

    original_size: Tuple[int, int] = (256, 256)  # LIBERO default
    processed_size: Tuple[int, int] = (224, 224)  # Model input size
    crop_scale: float = 0.9  # Center crop scale
    center_crop: bool = True  # Whether center crop is applied
    patch_size: int = 14  # ViT patch size for OpenVLA
    is_fused: bool = False  # DinoSigLIP uses 2 encoders
    num_encoders: int = 1  # Number of vision encoders

    def __post_init__(self):
        """Validate and compute derived fields."""
        if self.is_fused and self.num_encoders == 1:
            self.num_encoders = 2  # Assume fused = 2 encoders

    def get_grid_shape(self) -> Tuple[int, int]:
        """Compute the token grid shape from processed image size and patch size.

        Returns:
            (H_patches, W_patches) grid dimensions.
        """
        h, w = self.processed_size
        return (h // self.patch_size, w // self.patch_size)

    def get_tokens_per_encoder(self) -> int:
        """Compute number of tokens per encoder.

        Returns:
            Number of tokens per vision encoder.
        """
        h_patches, w_patches = self.get_grid_shape()
        return h_patches * w_patches


@dataclass
class Token3DMappingResult:
    """
    Result of mapping visual tokens to 3D coordinates in the robot base frame.

    Attributes:
        token_points_base: 3D coordinates [num_visual_tokens, 3] in base frame.
                           Invalid tokens have NaN coordinates.
        token_valid_mask: Boolean mask [num_visual_tokens] indicating valid tokens.
        depth_median: Median depth [num_visual_tokens] of each token's patch.
        depth_std: Depth standard deviation [num_visual_tokens] of each token's patch.
        valid_depth_ratio: Fraction [num_visual_tokens] of valid depth pixels in patch.
        token_grid_shape: Shape (H, W) of the token grid per encoder.
        num_visual_tokens: Total number of visual tokens.
        mapping_notes: Dictionary with metadata and warnings about the mapping.
    """

    token_points_base: np.ndarray
    token_valid_mask: np.ndarray
    depth_median: np.ndarray
    depth_std: np.ndarray
    valid_depth_ratio: np.ndarray
    token_grid_shape: Tuple[int, int]
    num_visual_tokens: int
    mapping_notes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate shapes."""
        n = self.num_visual_tokens
        expected_arrays = [
            (self.token_points_base, (n, 3)),
            (self.token_valid_mask, (n,)),
            (self.depth_median, (n,)),
            (self.depth_std, (n,)),
            (self.valid_depth_ratio, (n,)),
        ]
        for arr, expected_shape in expected_arrays:
            if arr is not None and arr.shape != expected_shape:
                logger.warning(
                    f"[Token3DMappingResult] Shape mismatch: {arr.shape} != {expected_shape}"
                )

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of the mapping result."""
        valid_count = np.sum(self.token_valid_mask) if self.token_valid_mask is not None else 0
        return {
            "num_visual_tokens": self.num_visual_tokens,
            "valid_tokens": int(valid_count),
            "invalid_tokens": self.num_visual_tokens - int(valid_count),
            "valid_ratio": float(valid_count / self.num_visual_tokens) if self.num_visual_tokens > 0 else 0.0,
            "token_grid_shape": self.token_grid_shape,
            "has_warnings": len(self.mapping_notes) > 0,
        }


# =============================================================================
# Token 3D Mapper
# =============================================================================


class Token3DMapper:
    """
    Maps visual tokens from 2D image patches to 3D coordinates in robot base frame.

    This mapper implements the 2D-to-3D projection pipeline:
    1. Extract depth statistics (median, std, valid ratio) for each token's patch
    2. Backproject the median depth point to camera coordinates
    3. Transform camera coordinates to base frame using extrinsics

    The mapping handles:
    - Single encoder models (e.g., DINOv2, CLIP)
    - Fused multi-encoder models (e.g., DinoSigLIP with 2 encoders)
    - Center crop and resize preprocessing
    - Invalid depth values (marked with NaN and False mask)

    Usage:
        mapper = Token3DMapper()
        result = mapper.map_tokens_to_3d(
            depth=depth_image,
            camera_intrinsics=K,
            camera_extrinsics=T_base_cam,
            image_preprocess_meta=preprocess_meta,
            token_grid_shape=(16, 16),
            num_visual_tokens=256,
        )
    """

    def __init__(
        self,
        min_valid_depth_ratio: float = 0.1,
        max_depth: float = 10.0,
        projection_mode: str = "current",
    ):
        """
        Initialize the Token3DMapper.

        Args:
            min_valid_depth_ratio: Minimum fraction of valid depth pixels in a patch
                                   to consider the token valid (default 0.1 = 10%).
            max_depth: Maximum valid depth in meters (default 10.0m).
            projection_mode: Debug-only pixel/K convention switch. "current"
                preserves existing behavior. "rotated_pixels_with_original_K"
                uses rotated image coordinates directly. "unrotate_pixels_then_original_K"
                maps rotated image coordinates back to the original camera pixel frame
                before applying K.
        """
        self.min_valid_depth_ratio = min_valid_depth_ratio
        self.max_depth = max_depth
        self.projection_mode = projection_mode

    def map_tokens_to_3d(
        self,
        depth: np.ndarray,
        camera_intrinsics: np.ndarray,
        camera_extrinsics: np.ndarray,
        image_preprocess_meta: ImagePreprocessMeta,
        token_grid_shape: Tuple[int, int],
        num_visual_tokens: int,
    ) -> Token3DMappingResult:
        """
        Map all visual tokens to 3D coordinates in the robot base frame.

        Args:
            depth: Depth image (H x W) in meters, float32.
            camera_intrinsics: 3x3 camera intrinsics matrix K.
            camera_extrinsics: 4x4 camera-to-base transform T_base_cam.
            image_preprocess_meta: Metadata describing image preprocessing.
            token_grid_shape: Shape (H_patches, W_patches) of the token grid.
            num_visual_tokens: Total number of visual tokens.

        Returns:
            Token3DMappingResult with 3D coordinates and depth statistics.
        """
        mapping_notes: Dict[str, Any] = {}

        # Validate inputs
        if depth is None:
            logger.warning("[Token3DMapper] depth is None, returning empty result")
            return self._create_empty_result(num_visual_tokens, token_grid_shape, mapping_notes)

        if camera_intrinsics is None or camera_extrinsics is None:
            logger.warning("[Token3DMapper] camera parameters are None, returning empty result")
            return self._create_empty_result(num_visual_tokens, token_grid_shape, mapping_notes)

        # Ensure arrays are float32
        depth = np.asarray(depth, dtype=np.float32)
        K = np.asarray(camera_intrinsics, dtype=np.float32)
        T = np.asarray(camera_extrinsics, dtype=np.float32)

        # Check shape compatibility
        tokens_per_encoder = token_grid_shape[0] * token_grid_shape[1]

        if num_visual_tokens != tokens_per_encoder:
            if tokens_per_encoder > 0 and num_visual_tokens % tokens_per_encoder == 0:
                num_encoder_grids = num_visual_tokens // tokens_per_encoder
                repeat_strategy = "modulo_grid_index"
            else:
                num_encoder_grids = None
                repeat_strategy = "clipped_mismatch"
            mapping_notes["multi_encoder"] = True
            mapping_notes["tokens_per_encoder"] = tokens_per_encoder
            mapping_notes["num_encoder_grids"] = num_encoder_grids
            mapping_notes["repeat_strategy"] = repeat_strategy
            mapping_notes["warning"] = (
                f"num_visual_tokens={num_visual_tokens} != grid_size={tokens_per_encoder}. "
                f"This may indicate a fused multi-encoder model. "
                f"Geometry is repeated per encoder grid when divisible."
            )
            logger.warning(f"[Token3DMapper] {mapping_notes['warning']}")

        # Allocate output arrays
        token_points_base = np.full((num_visual_tokens, 3), np.nan, dtype=np.float32)
        token_valid_mask = np.zeros(num_visual_tokens, dtype=np.bool_)
        depth_median = np.full(num_visual_tokens, np.nan, dtype=np.float32)
        depth_std = np.full(num_visual_tokens, np.nan, dtype=np.float32)
        valid_depth_ratio = np.zeros(num_visual_tokens, dtype=np.float32)

        # Compute depth image scale factor (depth to original image)
        depth_h, depth_w = depth.shape[:2]
        orig_h, orig_w = image_preprocess_meta.original_size
        processed_h, processed_w = image_preprocess_meta.processed_size

        # Scale factor: how much to scale depth coordinates to match original image
        scale_y = orig_h / processed_h if processed_h > 0 else 1.0
        scale_x = orig_w / processed_w if processed_w > 0 else 1.0

        # Handle center crop if applicable
        if image_preprocess_meta.center_crop:
            crop_scale = image_preprocess_meta.crop_scale
            crop_h = int(orig_h * np.sqrt(crop_scale))
            crop_w = int(orig_w * np.sqrt(crop_scale))
            crop_top = (orig_h - crop_h) // 2
            crop_left = (orig_w - crop_w) // 2
        else:
            crop_h, crop_w = orig_h, orig_w
            crop_top, crop_left = 0, 0

        mapping_notes["depth_scale"] = {"scale_x": scale_x, "scale_y": scale_y}
        mapping_notes["crop_applied"] = image_preprocess_meta.center_crop
        mapping_notes["projection_mode"] = self.projection_mode

        # Extract rotation and translation from extrinsics
        R = T[:3, :3]
        t = T[:3, 3]

        # Process each token
        for token_idx in range(num_visual_tokens):
            # Compute token's position in the per-encoder grid. For fused
            # encoders, each encoder contributes one grid, so indices wrap
            # instead of drifting outside the image.
            grid_token_idx = token_idx % tokens_per_encoder if tokens_per_encoder > 0 else token_idx
            row = grid_token_idx // token_grid_shape[1]
            col = grid_token_idx % token_grid_shape[1]

            # Compute patch bounds in processed image coordinates
            patch_size = image_preprocess_meta.patch_size
            proc_v_min = row * patch_size
            proc_v_max = (row + 1) * patch_size
            proc_u_min = col * patch_size
            proc_u_max = (col + 1) * patch_size

            # Map to original image coordinates (with crop)
            orig_v_min = int(crop_top + proc_v_min * scale_y)
            orig_v_max = int(crop_top + proc_v_max * scale_y)
            orig_u_min = int(crop_left + proc_u_min * scale_x)
            orig_u_max = int(crop_left + proc_u_max * scale_x)

            # Clip to depth image bounds
            orig_v_min = max(0, min(orig_v_min, depth_h - 1))
            orig_v_max = max(0, min(orig_v_max, depth_h))
            orig_u_min = max(0, min(orig_u_min, depth_w - 1))
            orig_u_max = max(0, min(orig_u_max, depth_w))

            # Extract depth patch
            depth_patch = depth[orig_v_min:orig_v_max, orig_u_min:orig_u_max]

            # Compute depth statistics
            median, std, ratio, is_valid = self._compute_patch_depth_stats(depth_patch)

            depth_median[token_idx] = median
            depth_std[token_idx] = std if is_valid else np.nan
            valid_depth_ratio[token_idx] = ratio
            token_valid_mask[token_idx] = is_valid

            if not is_valid:
                mapping_notes.setdefault("invalid_tokens", []).append(token_idx)
                continue

            # Compute patch center in pixel coordinates
            u_center = (orig_u_min + orig_u_max) / 2.0
            v_center = (orig_v_min + orig_v_max) / 2.0
            u_project = u_center
            v_project = v_center

            if self.projection_mode == "unrotate_pixels_then_original_K":
                u_project = (depth_w - 1) - u_center
                v_project = (depth_h - 1) - v_center
            elif self.projection_mode in ("current", "rotated_pixels_with_original_K"):
                pass
            else:
                mapping_notes.setdefault("warnings", []).append(
                    f"Unknown projection_mode={self.projection_mode}; using current pixel convention."
                )

            # Backproject to camera coordinates
            p_cam = self._backproject_pixel(u_project, v_project, median, K)

            # Transform to base coordinates
            p_base = R @ p_cam + t

            token_points_base[token_idx] = p_base

        # Add summary to mapping notes
        mapping_notes["num_invalid"] = int(np.sum(~token_valid_mask))
        mapping_notes["num_valid"] = int(np.sum(token_valid_mask))
        mapping_notes["valid_ratio"] = float(np.mean(token_valid_mask))

        logger.debug(
            f"[Token3DMapper] Mapped {num_visual_tokens} tokens: "
            f"valid={mapping_notes['num_valid']}, invalid={mapping_notes['num_invalid']}"
        )

        return Token3DMappingResult(
            token_points_base=token_points_base,
            token_valid_mask=token_valid_mask,
            depth_median=depth_median,
            depth_std=depth_std,
            valid_depth_ratio=valid_depth_ratio,
            token_grid_shape=token_grid_shape,
            num_visual_tokens=num_visual_tokens,
            mapping_notes=mapping_notes,
        )

    def _backproject_pixel(
        self, u: float, v: float, z: float, K: np.ndarray
    ) -> np.ndarray:
        """
        Backproject a 2D pixel with depth to 3D camera coordinates.

        Args:
            u: Pixel x coordinate.
            v: Pixel y coordinate.
            z: Depth in meters.
            K: 3x3 camera intrinsics matrix.

        Returns:
            3D point [X, Y, Z] in camera frame.
        """
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        X_cam = (u - cx) * z / fx
        Y_cam = (v - cy) * z / fy
        Z_cam = z

        return np.array([X_cam, Y_cam, Z_cam], dtype=np.float32)

    def _compute_patch_depth_stats(
        self, depth_patch: np.ndarray
    ) -> Tuple[float, float, float, bool]:
        """
        Compute depth statistics for a token's image patch.

        Args:
            depth_patch: 2D array of depth values in the patch.

        Returns:
            Tuple of (median, std, valid_ratio, is_valid).
            - median: Median depth value (NaN if invalid).
            - std: Standard deviation (NaN if invalid).
            - valid_ratio: Fraction of valid depth pixels.
            - is_valid: Whether the patch has enough valid depth.
        """
        if depth_patch.size == 0:
            return np.nan, np.nan, 0.0, False

        # Define valid depth range
        valid_mask = (depth_patch > 0) & (depth_patch < self.max_depth)
        valid_depths = depth_patch[valid_mask]
        valid_ratio = valid_mask.sum() / depth_patch.size

        # Check minimum valid ratio threshold
        if valid_ratio < self.min_valid_depth_ratio:
            return np.nan, np.nan, float(valid_ratio), False

        if len(valid_depths) == 0:
            return np.nan, np.nan, float(valid_ratio), False

        # Compute statistics
        median = float(np.median(valid_depths))
        std = float(np.std(valid_depths)) if len(valid_depths) > 1 else 0.0

        return median, std, float(valid_ratio), True

    def _create_empty_result(
        self,
        num_visual_tokens: int,
        token_grid_shape: Tuple[int, int],
        mapping_notes: Dict[str, Any],
    ) -> Token3DMappingResult:
        """Create an empty result with all invalid tokens."""
        mapping_notes["empty_result"] = True
        mapping_notes["reason"] = "Input data was None or invalid"

        return Token3DMappingResult(
            token_points_base=np.full((num_visual_tokens, 3), np.nan, dtype=np.float32),
            token_valid_mask=np.zeros(num_visual_tokens, dtype=np.bool_),
            depth_median=np.full(num_visual_tokens, np.nan, dtype=np.float32),
            depth_std=np.full(num_visual_tokens, np.nan, dtype=np.float32),
            valid_depth_ratio=np.zeros(num_visual_tokens, dtype=np.float32),
            token_grid_shape=token_grid_shape,
            num_visual_tokens=num_visual_tokens,
            mapping_notes=mapping_notes,
        )


# =============================================================================
# Utility Functions
# =============================================================================


def create_default_preprocess_meta(
    original_size: Tuple[int, int] = (256, 256),
    processed_size: Tuple[int, int] = (224, 224),
    center_crop: bool = True,
    patch_size: int = 14,
    is_fused: bool = False,
    num_encoders: int = 1,
) -> ImagePreprocessMeta:
    """
    Create a default ImagePreprocessMeta based on OpenVLA/LIBERO settings.

    Args:
        original_size: Original camera image size (default: 256x256 for LIBERO).
        processed_size: Size after resize (default: 224x224 for OpenVLA).
        center_crop: Whether center crop is applied (default: True).
        patch_size: ViT patch size (default: 14 for OpenVLA ViT-L).
        is_fused: Whether using fused multi-encoder (default: False).
        num_encoders: Number of encoders (default: 1).

    Returns:
        ImagePreprocessMeta instance.
    """
    return ImagePreprocessMeta(
        original_size=original_size,
        processed_size=processed_size,
        crop_scale=0.9,
        center_crop=center_crop,
        patch_size=patch_size,
        is_fused=is_fused,
        num_encoders=num_encoders,
    )


def infer_token_grid_from_count(
    num_visual_tokens: int, patch_size: int = 14
) -> Tuple[int, int, int]:
    """
    Infer token grid shape from token count.

    Assumes square grid (e.g., 16x16 = 256 tokens for 224x224 image).

    Args:
        num_visual_tokens: Total number of visual tokens.
        patch_size: ViT patch size (default: 14).

    Returns:
        Tuple of (grid_h, grid_w, tokens_per_encoder).
    """
    import math

    # For 224x224 with patch_size=14: grid is 16x16 = 256 tokens
    # For 384x384 with patch_size=14: grid is 27x27 = 729 tokens
    # Assume processed_size = grid_size * patch_size
    grid_size = int(math.sqrt(num_visual_tokens))
    if grid_size * grid_size != num_visual_tokens:
        # Non-square grid, need more inference
        logger.warning(
            f"[Token3DMapper] num_visual_tokens={num_visual_tokens} is not a perfect square. "
            f"Assuming square grid of {grid_size}x{grid_size}."
        )

    processed_size = grid_size * patch_size
    return (grid_size, grid_size, num_visual_tokens)


# =============================================================================
# Debug/Test Functions
# =============================================================================


def debug_token_3d_mapping():
    """
    Verify the 2D to 3D mapping correctness.

    This test creates synthetic data and verifies:
    1. Output shapes are correct
    2. Center token maps to expected position
    3. Invalid depth tokens get NaN coordinates
    """
    print("=" * 60)
    print("[Token3DMapper] Running debug tests...")
    print("=" * 60)

    mapper = Token3DMapper(min_valid_depth_ratio=0.1, max_depth=10.0)

    # Test 1: Uniform depth image
    print("\n[Test 1] Uniform depth image (depth=1.0m)")
    depth_1m = np.ones((224, 224), dtype=np.float32) * 1.0

    # Camera intrinsics (focal=200, center=112)
    K = np.array([
        [200.0, 0.0, 112.0],
        [0.0, 200.0, 112.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    # Camera extrinsics (camera at (0, 0, 1) in base frame)
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = [0.0, 0.0, 1.0]

    # Preprocess meta
    meta = ImagePreprocessMeta(
        original_size=(224, 224),
        processed_size=(224, 224),
        center_crop=False,
        patch_size=14,
    )

    result = mapper.map_tokens_to_3d(
        depth=depth_1m,
        camera_intrinsics=K,
        camera_extrinsics=T,
        image_preprocess_meta=meta,
        token_grid_shape=(16, 16),
        num_visual_tokens=256,
    )

    # Verify shapes
    assert result.token_points_base.shape == (256, 3), f"Shape mismatch: {result.token_points_base.shape}"
    assert result.token_valid_mask.shape == (256,), f"Mask shape mismatch: {result.token_valid_mask.shape}"
    assert result.num_visual_tokens == 256
    assert result.token_grid_shape == (16, 16)
    print(f"  [OK] Shapes correct: {result.token_points_base.shape}")

    # Verify all tokens valid
    assert np.all(result.token_valid_mask), "Not all tokens valid with uniform depth"
    print(f"  [OK] All tokens valid")

    # Verify projection consistency: center token (pixel near 112, 112) with depth=1.0
    # For camera at (0,0,1), focal=200, depth=1.0:
    # - Pixel (112, 112) is at optical center
    # - Backprojection: (112-112)*1/200 = 0, (112-112)*1/200 = 0
    # - Camera center: (0, 0, 1) + (0, 0, 1) = (0, 0, 2)
    # Token index 136 = row 8, col 8 (center of 16x16 grid)
    center_token = result.token_points_base[136]
    expected = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    error = np.linalg.norm(center_token - expected)
    assert error < 0.1, f"Center projection error: {error} (got {center_token}, expected {expected})"
    print(f"  [OK] Center token (index 136) correct: {center_token} (error={error:.4f})")

    # Verify depth values are correctly extracted
    assert np.allclose(result.depth_median, 1.0, atol=0.01), "Depth median should be 1.0"
    print(f"  [OK] Depth values correct: median={result.depth_median[0]:.3f}")

    # Verify all tokens have same depth (uniform depth image)
    assert np.allclose(result.depth_median, result.depth_median[0]), "All tokens should have same depth"
    print(f"  [OK] Uniform depth across all tokens")

    # Test 2: Invalid depth
    print("\n[Test 2] Invalid depth (all zeros)")
    depth_zero = np.zeros((224, 224), dtype=np.float32)

    result2 = mapper.map_tokens_to_3d(
        depth=depth_zero,
        camera_intrinsics=K,
        camera_extrinsics=T,
        image_preprocess_meta=meta,
        token_grid_shape=(16, 16),
        num_visual_tokens=256,
    )

    assert np.all(~result2.token_valid_mask), "No tokens should be valid with zero depth"
    assert np.all(np.isnan(result2.token_points_base)), "All coordinates should be NaN"
    print(f"  [OK] All tokens invalid with zero depth")

    # Test 3: Mixed depth
    print("\n[Test 3] Mixed depth (half valid)")
    depth_mixed = np.zeros((224, 224), dtype=np.float32)
    depth_mixed[112:, :] = 2.0  # Bottom half has valid depth

    result3 = mapper.map_tokens_to_3d(
        depth=depth_mixed,
        camera_intrinsics=K,
        camera_extrinsics=T,
        image_preprocess_meta=meta,
        token_grid_shape=(16, 16),
        num_visual_tokens=256,
    )

    valid_count = np.sum(result3.token_valid_mask)
    invalid_count = np.sum(~result3.token_valid_mask)
    print(f"  Valid tokens: {valid_count}, Invalid tokens: {invalid_count}")
    assert valid_count > 0, "Should have some valid tokens"
    assert invalid_count > 0, "Should have some invalid tokens"
    print(f"  [OK] Mixed depth handled correctly")

    # Test 4: Multi-encoder (512 tokens)
    print("\n[Test 4] Multi-encoder model (512 tokens)")
    result4 = mapper.map_tokens_to_3d(
        depth=depth_1m,
        camera_intrinsics=K,
        camera_extrinsics=T,
        image_preprocess_meta=meta,
        token_grid_shape=(16, 16),
        num_visual_tokens=512,
    )

    assert result4.num_visual_tokens == 512
    assert result4.mapping_notes.get("multi_encoder", False)
    print(f"  [OK] Multi-encoder handled: {result4.mapping_notes.get('warning', '')[:60]}...")

    # Test 5: Depth unavailable
    print("\n[Test 5] Depth unavailable (None)")
    result5 = mapper.map_tokens_to_3d(
        depth=None,
        camera_intrinsics=K,
        camera_extrinsics=T,
        image_preprocess_meta=meta,
        token_grid_shape=(16, 16),
        num_visual_tokens=256,
    )

    assert result5.mapping_notes.get("empty_result", False)
    assert np.all(~result5.token_valid_mask)
    print(f"  [OK] None depth handled gracefully")

    print("\n" + "=" * 60)
    print("[Token3DMapper] All debug tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    debug_token_3d_mapping()
