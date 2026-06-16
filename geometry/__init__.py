"""Geometry module for visual token geometric analysis.

This module provides tools for collecting geometric information (RGB-D, camera parameters,
end-effector pose, gripper position) needed for geometry-guided visual token pruning.

Architecture:
- geometry_data_recorder.py: Data collection during evaluation
- token_3d_mapper.py: 2D patch to 3D base coordinate projection
- rule_based_geometry_expert.py: Rule-based token scoring based on 3D geometry
- geometry_visualizer.py: Score and mask visualization
- token_pruner.py: Token pruning integration
"""

from geometry.geometry_data_recorder import GeometryDataRecorder, GeometryStepData
from geometry.geometry_depth import convert_depth_to_metric, DepthConversionResult
from geometry.token_3d_mapper import (
    Token3DMapper,
    Token3DMappingResult,
    ImagePreprocessMeta,
)
from geometry.rule_based_geometry_expert import (
    RuleBasedGeometryExpert,
    GeometryScoreResult,
    GeometryExpertConfig,
)
from geometry.geometry_visualizer import (
    GeometryVisualizer,
    VisualizationConfig,
)
from geometry.token_pruner import (
    TokenPruner,
    PruningConfig,
    PruningResult,
    PruningMethod,
    TokenScores,
    prune_visual_tokens,
    apply_token_pruning,
)

__all__ = [
    # Data collection
    "GeometryDataRecorder",
    "GeometryStepData",
    # Depth conversion
    "convert_depth_to_metric",
    "DepthConversionResult",
    # 3D mapping
    "Token3DMapper",
    "Token3DMappingResult",
    "ImagePreprocessMeta",
    # Scoring
    "RuleBasedGeometryExpert",
    "GeometryScoreResult",
    "GeometryExpertConfig",
    # Visualization
    "GeometryVisualizer",
    "VisualizationConfig",
    # Pruning
    "TokenPruner",
    "PruningConfig",
    "PruningResult",
    "PruningMethod",
    "TokenScores",
    "prune_visual_tokens",
    "apply_token_pruning",
]
