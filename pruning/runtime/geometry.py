"""Geometry scoring methods for VisualTokenPruningHook."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from ..signals.spatial import compute_depth_edge_scores, compute_valid_depth_mask
from ..signals.action import compute_future_action_constraint_scores
from ..signals.spatial import compute_contact_ring_scores
from ..signals.action import compute_motion_corridor_scores, create_motion_buffer
from ..signals.robot import (
    compute_robot_geo_contact_budget_scores,
    compute_robot_geo_corridor_scores,
    compute_robot_geo_near_scores,
    compute_robot_geo_scores_v0,
    decide_dynamic_keep_ratio,
    extract_gripper_position,
    extract_robot_camera_transform,
    project_tokens_to_robot,
)
from ..signals.robot import RobotState
from ..signals.spatial import compute_scene_layout_scores
from ..strategy_registry import BRANCH_MIXTURE_SCORE_STRATEGIES
from ..core.utils import (
    _append_score_stats,
    _arr_stats,
    _arr_to_str,
    _build_hybrid_final_scores,
    _norm_hybrid_component,
)


class HookGeometryMixin:
    def _compute_depth_edge_scores(self, num_tokens: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any], Optional[str]]:
        latest = self.geometry_recorder.get_latest() if self.geometry_recorder is not None else None
        if latest is None:
            return None, None, {}, "missing_geometry"
        if latest.depth is None:
            return None, None, {}, "missing_depth"

        # Propagate depth conversion metadata
        edge_meta: Dict[str, Any] = {}
        if getattr(latest, "depth_metadata", None) is not None:
            dm = latest.depth_metadata
            edge_meta["depth_source_key"] = dm.get("source_key")
            edge_meta["depth_conversion"] = dm.get("conversion")
            edge_meta["depth_is_metric"] = dm.get("depth_is_metric")
            edge_meta["depth_unit"] = dm.get("depth_unit")
            edge_meta["depth_sim_available"] = dm.get("sim_available")
            raw_s = dm.get("depth_raw_stats", {})
            met_s = dm.get("depth_metric_stats", {})
            if raw_s:
                edge_meta["depth_raw_min"] = raw_s.get("min")
                edge_meta["depth_raw_max"] = raw_s.get("max")
                edge_meta["depth_raw_mean"] = raw_s.get("mean")
                edge_meta["depth_raw_std"] = raw_s.get("std")
            if met_s:
                edge_meta["depth_metric_min"] = met_s.get("min")
                edge_meta["depth_metric_max"] = met_s.get("max")
                edge_meta["depth_metric_mean"] = met_s.get("mean")
                edge_meta["depth_metric_std"] = met_s.get("std")

        K = latest.camera_intrinsics if latest.camera_intrinsics is not None else np.eye(3, dtype=np.float32)
        T = latest.camera_extrinsics if latest.camera_extrinsics is not None else np.eye(4, dtype=np.float32)

        meta = self._latest_preprocess_meta
        if meta is None:
            from geometry.token_3d_mapper import create_default_preprocess_meta

            meta = create_default_preprocess_meta(
                original_size=latest.rgb.shape[:2] if latest.rgb is not None else (256, 256),
                processed_size=(224, 224),
                center_crop=bool(self.cfg.get("center_crop", True)),
            )

        token_grid_shape = self._latest_token_grid_shape or self.config.token_grid_shape
        depth = np.asarray(latest.depth, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[:, :, 0]

        mapping_start = time.perf_counter()
        cache = self._cache.get(
            depth.shape[:2],
            K,
            T,
            meta,
            token_grid_shape=token_grid_shape,
            num_visual_tokens=num_tokens,
            projection_mode=str(self.cfg.get("projection_mode", "current")),
        )
        sampling_start = time.perf_counter()
        token_depth = self._cache.sample_depth(depth, cache)
        depth_sampling_ms = (time.perf_counter() - sampling_start) * 1000.0
        valid_mask = compute_valid_depth_mask(token_depth)
        valid_ratio = float(np.mean(valid_mask)) if valid_mask.size else 0.0
        token_mapping_ms = (time.perf_counter() - mapping_start) * 1000.0

        if valid_ratio < self.config.min_valid_token_ratio:
            return None, valid_mask, {
                "token_mapping_ms": token_mapping_ms,
                "depth_sampling_ms": depth_sampling_ms,
                "valid_token_ratio": valid_ratio,
            }, "invalid_depth_ratio"

        score_start = time.perf_counter()
        token_depth = np.nan_to_num(token_depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        scores = compute_depth_edge_scores(
            token_depth,
            valid_mask,
            token_grid_shape=token_grid_shape,
            num_visual_tokens=num_tokens,
        )
        score_compute_ms = (time.perf_counter() - score_start) * 1000.0
        result_meta = {
            "token_mapping_ms": token_mapping_ms,
            "depth_sampling_ms": depth_sampling_ms,
            "score_compute_ms": score_compute_ms,
            "valid_token_ratio": valid_ratio,
            "edge_scores": scores,
            "cache": cache,
            "score_mean": float(np.mean(scores)) if scores.size else 0.0,
            "score_max": float(np.max(scores)) if scores.size else 0.0,
            "score_std": float(np.std(scores)) if scores.size else 0.0,
        }
        result_meta.update(edge_meta)
        return scores, valid_mask, result_meta, None

    def _compute_robot_geo_near_scores(self, num_tokens: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any], Optional[str]]:
        latest = self.geometry_recorder.get_latest() if self.geometry_recorder is not None else None
        if latest is None:
            return None, None, {"geometry_available": False}, "missing_geometry"
        if latest.depth is None:
            return None, None, {"geometry_available": False}, "missing_depth"

        gripper_pos, gripper_key = extract_gripper_position(latest)
        T_robot_cam, transform_key = extract_robot_camera_transform(latest)
        camera_available = latest.camera_intrinsics is not None and T_robot_cam is not None
        robot_available = gripper_pos is not None

        base_metrics: Dict[str, Any] = {
            "geometry_available": bool(camera_available and robot_available),
            "robot_state_available": bool(robot_available),
            "camera_available": bool(camera_available),
            "gripper_source": gripper_key,
            "camera_transform_source": transform_key,
            # Transform convention audit (P0-4):
            # T_robot_cam_forward convention: p_robot = T_robot_cam · p_cam
            # In this pipeline, T_robot_cam = camera_extrinsics = T_base_cam = T_world_cam
            # (single extrinsic matrix, used in camera→robot direction in project_tokens_to_robot)
            # LIBERO: robot_base == world frame, CU.get_camera_extrinsic_matrix returns T_world_cam.
            # This matrix is used AS-IS (not inverted) in project_tokens_to_robot.
            # The same matrix is pre-inverted to T_robot_cam in validate_geometry_mapping.py
            # for the inverse (robot→camera) projection direction — same extrinsic, opposite usage.
            "transform_convention": "T_robot_cam_forward",
            "transform_inverse_used": False,  # T_base_cam used as camera→robot (not inverted)
            # transform_key = key name found by extract_robot_camera_transform()
            # e.g. "camera_extrinsics" (T_base_cam/T_world_cam), "T_robot_cam", etc.
            "transform_source": transform_key,
            "transform_convention_verified": True,
            "transform_convention_evidence": (
                "P0-3 overlay: forward aligns with gripper/eef; "
                "inverse falls on background; depth consistency ambiguous but not contradictory; "
                "physical z_cam positive supports forward"
            ),
        }

        # Propagate depth conversion metadata from GeometryStepData into metrics
        if getattr(latest, "depth_metadata", None) is not None:
            dm = latest.depth_metadata
            base_metrics["depth_source_key"] = dm.get("source_key")
            base_metrics["depth_conversion"] = dm.get("conversion")
            base_metrics["depth_is_metric"] = dm.get("depth_is_metric")
            base_metrics["depth_unit"] = dm.get("depth_unit")
            base_metrics["depth_sim_available"] = dm.get("sim_available")
            raw_s = dm.get("depth_raw_stats", {})
            met_s = dm.get("depth_metric_stats", {})
            if raw_s:
                base_metrics["depth_raw_min"] = raw_s.get("min")
                base_metrics["depth_raw_max"] = raw_s.get("max")
                base_metrics["depth_raw_mean"] = raw_s.get("mean")
                base_metrics["depth_raw_std"] = raw_s.get("std")
            if met_s:
                base_metrics["depth_metric_min"] = met_s.get("min")
                base_metrics["depth_metric_max"] = met_s.get("max")
                base_metrics["depth_metric_mean"] = met_s.get("mean")
                base_metrics["depth_metric_std"] = met_s.get("std")

        if not robot_available:
            scores, valid_mask, depth_metrics, depth_fallback = self._compute_depth_edge_scores(num_tokens)
            base_metrics.update(depth_metrics)
            # P0-4: no geometry available, so transform fields are not meaningful
            base_metrics["transform_convention"] = None
            base_metrics["transform_inverse_used"] = None
            base_metrics["transform_source"] = None
            base_metrics["transform_convention_verified"] = None
            base_metrics["transform_convention_evidence"] = None
            return scores, valid_mask, base_metrics, depth_fallback or "missing_robot_state"
        if not camera_available:
            scores, valid_mask, depth_metrics, depth_fallback = self._compute_depth_edge_scores(num_tokens)
            base_metrics.update(depth_metrics)
            base_metrics["transform_convention"] = None
            base_metrics["transform_inverse_used"] = None
            base_metrics["transform_source"] = None
            base_metrics["transform_convention_verified"] = None
            base_metrics["transform_convention_evidence"] = None
            if latest.camera_intrinsics is None:
                reason = "missing_camera_intrinsics"
            elif T_robot_cam is None:
                reason = "missing_camera_extrinsics"
            else:
                reason = "missing_camera"
            return scores, valid_mask, base_metrics, depth_fallback or reason

        meta = self._latest_preprocess_meta
        if meta is None:
            from geometry.token_3d_mapper import create_default_preprocess_meta

            meta = create_default_preprocess_meta(
                original_size=latest.rgb.shape[:2] if latest.rgb is not None else (256, 256),
                processed_size=(224, 224),
                center_crop=bool(self.cfg.get("center_crop", True)),
            )

        token_grid_shape = self._latest_token_grid_shape or self.config.token_grid_shape
        depth = np.asarray(latest.depth, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[:, :, 0]

        mapping_start = time.perf_counter()
        cache = self._cache.get(
            depth.shape[:2],
            latest.camera_intrinsics,
            T_robot_cam,
            meta,
            token_grid_shape=token_grid_shape,
            num_visual_tokens=num_tokens,
            projection_mode=str(self.cfg.get("projection_mode", "current")),
        )
        sampling_start = time.perf_counter()
        token_depth = self._cache.sample_depth(depth, cache)
        depth_sampling_ms = (time.perf_counter() - sampling_start) * 1000.0
        valid_mask = compute_valid_depth_mask(
            token_depth,
            min_depth=self.config.min_depth,
            max_depth=self.config.max_depth,
        )
        valid_ratio = float(np.mean(valid_mask)) if valid_mask.size else 0.0
        token_mapping_ms = (time.perf_counter() - mapping_start) * 1000.0

        if valid_ratio < self.config.min_valid_token_ratio:
            base_metrics.update({
                "token_mapping_ms": token_mapping_ms,
                "depth_sampling_ms": depth_sampling_ms,
                "valid_token_ratio": valid_ratio,
            })
            return None, valid_mask, base_metrics, "invalid_depth_ratio"

        token_depth = np.nan_to_num(token_depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        score_start = time.perf_counter()
        prev_gripper_pos = self._get_previous_gripper_pos(latest)
        # Initialize for all paths; specific branches may redefine these
        rule_workspace_scores = None
        rule_contact_scores = None
        rule_motion_cone_scores = None
        rule_near_scores = None
        edge_scores = None
        if self.config.strategy in ("robot_geo_rule_v0", "robot_geo_dynamic_v0", "robot_geo_temporal_v0"):
            p_robot = project_tokens_to_robot(token_depth, cache["rays"], T_robot_cam, valid_mask)
            edge_scores = compute_depth_edge_scores(
                token_depth,
                valid_mask,
                token_grid_shape=token_grid_shape,
                num_visual_tokens=num_tokens,
            )
            prev_tensor = None if prev_gripper_pos is None else torch.as_tensor(prev_gripper_pos, dtype=torch.float32)
            motion_direction = None
            if prev_gripper_pos is not None:
                motion_direction = torch.as_tensor(gripper_pos - prev_gripper_pos, dtype=torch.float32)
            rule_state = RobotState(
                ee_position=torch.as_tensor(gripper_pos, dtype=torch.float32),
                prev_ee_position=prev_tensor,
                frame="robot",
                valid=True,
                metadata={"source": gripper_key},
            )
            rule_result = compute_robot_geo_scores_v0(
                {
                    "points_robot": torch.as_tensor(p_robot, dtype=torch.float32),
                    "valid_3d_mask": torch.as_tensor(valid_mask, dtype=torch.bool),
                },
                rule_state,
                motion_direction=motion_direction,
                depth_edge_score=torch.as_tensor(edge_scores, dtype=torch.float32),
                config=self.config,
            )
            scores = rule_result["final_scores"].detach().cpu().numpy().astype(np.float32)
            debug = rule_result.get("debug_info", {})
            score_stats = {
                "token_points_robot": p_robot,
                "edge_scores": edge_scores,
                "near_scores": rule_result["distance_to_gripper_score"].detach().cpu().numpy().astype(np.float32),
                "distances": np.linalg.norm(p_robot - gripper_pos[None, :], axis=1).astype(np.float32),
                "motion_direction": None if motion_direction is None else motion_direction.detach().cpu().numpy().astype(np.float32),
                "rule_v0_motion_cone_scores": rule_result["motion_cone_score"].detach().cpu().numpy().astype(np.float32),
                "rule_v0_workspace_scores": rule_result["workspace_score"].detach().cpu().numpy().astype(np.float32),
                "rule_v0_contact_risk_scores": rule_result["contact_risk_score"].detach().cpu().numpy().astype(np.float32),
                "motion_norm": float(np.linalg.norm(gripper_pos - prev_gripper_pos)) if prev_gripper_pos is not None else None,
                "corridor_active": bool(debug.get("motion_direction_valid", False)),
                "d_min": float(np.nanmin(np.linalg.norm(p_robot[valid_mask] - gripper_pos[None, :], axis=1))) if np.any(valid_mask) else None,
                "depth_edge_score_mean": debug.get("depth_edge_score_mean"),
                "edge_score_mean": debug.get("depth_edge_score_mean"),
                "mean_near_score": debug.get("distance_to_gripper_score_mean"),
                "max_near_score": debug.get("distance_to_gripper_score_max"),
                "motion_cone_score_mean": debug.get("motion_cone_score_mean"),
                "motion_cone_score_max": debug.get("motion_cone_score_max"),
                "workspace_score_mean": debug.get("workspace_score_mean"),
                "workspace_score_max": debug.get("workspace_score_max"),
                "contact_risk_score_mean": debug.get("contact_risk_score_mean"),
                "contact_risk_score_max": debug.get("contact_risk_score_max"),
                "geometry_score_mean": debug.get("final_score_mean"),
                "geometry_score_max": debug.get("final_score_max"),
                "geometry_score_std": debug.get("final_score_std"),
            }
            # Append full distribution stats for each component
            _append_score_stats(score_stats, "edge_scores", edge_scores, valid_mask)
            _append_score_stats(score_stats, "distance_scores", rule_result["distance_to_gripper_score"].detach().cpu().numpy().astype(np.float32), valid_mask)
            _append_score_stats(score_stats, "motion_cone_scores", rule_result["motion_cone_score"].detach().cpu().numpy().astype(np.float32), valid_mask)
            _append_score_stats(score_stats, "workspace_scores", rule_result["workspace_score"].detach().cpu().numpy().astype(np.float32), valid_mask)
            _append_score_stats(score_stats, "contact_risk_scores", rule_result["contact_risk_score"].detach().cpu().numpy().astype(np.float32), valid_mask)
            _append_score_stats(score_stats, "final_scores", scores, valid_mask)
            if self.config.strategy in ("robot_geo_dynamic_v0", "robot_geo_temporal_v0"):
                dynamic_decision = decide_dynamic_keep_ratio(
                    {
                        "contact_risk_score": rule_result["contact_risk_score"],
                        "distance_to_gripper_score": rule_result["distance_to_gripper_score"],
                        "motion_cone_score": rule_result["motion_cone_score"],
                        "valid_mask": rule_result["valid_mask"],
                    },
                    self.config,
                )
                summary = dynamic_decision.get("component_summary", {}) or {}
                score_stats.update({
                    "dynamic_decision": dynamic_decision,
                    "geo_risk_level": dynamic_decision.get("risk_level"),
                    "geo_risk_score": dynamic_decision.get("risk_score"),
                    "dynamic_keep_reason": dynamic_decision.get("reason"),
                    "num_high_contact_tokens": summary.get("num_high_contact_tokens"),
                    "num_valid_3d_tokens": summary.get("num_valid_3d_tokens"),
                })
        elif self.config.strategy == "robot_geo_contact_budget":
            scores, score_stats = compute_robot_geo_contact_budget_scores(
                token_depth,
                valid_mask,
                cache,
                T_robot_cam,
                gripper_pos,
                prev_gripper_pos,
                token_grid_shape=token_grid_shape,
                num_visual_tokens=num_tokens,
                sigma_near=self.config.sigma_near,
                sigma_corridor=self.config.sigma_corridor,
                corridor_length=self.config.corridor_length,
                min_motion_norm=self.config.min_motion_norm,
                w_near_contact=self.config.w_near_contact,
                w_corridor_contact=self.config.w_corridor_contact,
                edge_gate_eps=self.config.edge_gate_eps,
                detailed_timing=self.config.detailed_pruning_timing,
            )
        elif self.config.strategy in ("robot_geo_corridor", "robot_geo_dynamic"):
            scores, score_stats = compute_robot_geo_corridor_scores(
                token_depth,
                valid_mask,
                cache,
                T_robot_cam,
                gripper_pos,
                prev_gripper_pos,
                token_grid_shape=token_grid_shape,
                num_visual_tokens=num_tokens,
                w_edge=self.config.w_edge,
                w_near=self.config.w_near,
                w_corridor=self.config.w_corridor,
                sigma_near=self.config.sigma_near,
                sigma_corridor=self.config.sigma_corridor,
                corridor_length=self.config.corridor_length,
                min_motion_norm=self.config.min_motion_norm,
            )
        elif self.config.strategy in BRANCH_MIXTURE_SCORE_STRATEGIES:
            # P1-1 Fix: compute contact_risk, motion_cone, workspace scores via
            # compute_robot_geo_scores_v0 (same logic as robot_geo_rule_v0 branch)
            # so that select_hybrid_v1 receives non-None contact/corridor scores.
            p_robot = project_tokens_to_robot(token_depth, cache["rays"], T_robot_cam, valid_mask)
            edge_scores = compute_depth_edge_scores(
                token_depth,
                valid_mask,
                token_grid_shape=token_grid_shape,
                num_visual_tokens=num_tokens,
            )
            prev_tensor = None if prev_gripper_pos is None else torch.as_tensor(prev_gripper_pos, dtype=torch.float32)
            motion_direction = None
            if prev_gripper_pos is not None:
                motion_direction = torch.as_tensor(gripper_pos - prev_gripper_pos, dtype=torch.float32)
            rule_state = RobotState(
                ee_position=torch.as_tensor(gripper_pos, dtype=torch.float32),
                prev_ee_position=prev_tensor,
                frame="robot",
                valid=True,
                metadata={"source": gripper_key},
            )
            rule_result = compute_robot_geo_scores_v0(
                {
                    "points_robot": torch.as_tensor(p_robot, dtype=torch.float32),
                    "valid_3d_mask": torch.as_tensor(valid_mask, dtype=torch.bool),
                },
                rule_state,
                motion_direction=motion_direction,
                depth_edge_score=torch.as_tensor(edge_scores, dtype=torch.float32),
                config=self.config,
            )
            # P1-1: This is the computed contact / corridor / workspace scores that were missing.
            rule_contact_scores = rule_result["contact_risk_score"].detach().cpu().numpy().astype(np.float32)
            rule_motion_cone_scores = rule_result["motion_cone_score"].detach().cpu().numpy().astype(np.float32)
            rule_workspace_scores = rule_result["workspace_score"].detach().cpu().numpy().astype(np.float32)
            rule_near_scores = rule_result["distance_to_gripper_score"].detach().cpu().numpy().astype(np.float32)

            # Build score_stats using the same pattern as robot_geo_rule_v0
            debug = rule_result.get("debug_info", {})
            scores = rule_result["final_scores"].detach().cpu().numpy().astype(np.float32)
            score_stats = {
                "token_points_robot": p_robot,
                "edge_scores": edge_scores,
                "near_scores": rule_near_scores,
                "distances": np.linalg.norm(p_robot - gripper_pos[None, :], axis=1).astype(np.float32),
                "motion_direction": None if motion_direction is None else motion_direction.detach().cpu().numpy().astype(np.float32),
                "rule_v0_motion_cone_scores": rule_motion_cone_scores,
                "rule_v0_workspace_scores": rule_workspace_scores,
                "rule_v0_contact_risk_scores": rule_contact_scores,
                "motion_norm": float(np.linalg.norm(gripper_pos - prev_gripper_pos)) if prev_gripper_pos is not None else None,
                "corridor_active": bool(debug.get("motion_direction_valid", False)),
                "d_min": float(np.nanmin(np.linalg.norm(p_robot[valid_mask] - gripper_pos[None, :], axis=1))) if np.any(valid_mask) else None,
                "depth_edge_score_mean": debug.get("depth_edge_score_mean"),
                "edge_score_mean": debug.get("depth_edge_score_mean"),
                "mean_near_score": debug.get("distance_to_gripper_score_mean"),
                "max_near_score": debug.get("distance_to_gripper_score_max"),
                "motion_cone_score_mean": debug.get("motion_cone_score_mean"),
                "motion_cone_score_max": debug.get("motion_cone_score_max"),
                "workspace_score_mean": debug.get("workspace_score_mean"),
                "workspace_score_max": debug.get("workspace_score_max"),
                "contact_risk_score_mean": debug.get("contact_risk_score_mean"),
                "contact_risk_score_max": debug.get("contact_risk_score_max"),
                "geometry_score_mean": debug.get("final_score_mean"),
                "geometry_score_max": debug.get("final_score_max"),
                "geometry_score_std": debug.get("final_score_std"),
                # P1-1: Record is_none flags for all score components
                "contact_risk_scores_is_none": False,
                "motion_cone_scores_is_none": False,
                "workspace_scores_is_none": False,
                # P1-1: temporal_v1 semantics fix
                "temporal_enabled": False,
                "ema_enabled": False,
                "interaction_lock_triggered": False,
                "interaction_lock_reason": "insufficient_history",
            }
            # Append full distribution stats for each component (same as rule_v0)
            _append_score_stats(score_stats, "edge_scores", edge_scores, valid_mask)
            _append_score_stats(score_stats, "distance_scores", rule_near_scores, valid_mask)
            _append_score_stats(score_stats, "motion_cone_scores", rule_motion_cone_scores, valid_mask)
            _append_score_stats(score_stats, "workspace_scores", rule_workspace_scores, valid_mask)
            _append_score_stats(score_stats, "contact_risk_scores", rule_contact_scores, valid_mask)
            _append_score_stats(score_stats, "final_scores", scores, valid_mask)
            # P1-1: Add nonzero_ratio for each component
            for _key, _arr in [
                ("contact_risk", rule_contact_scores),
                ("motion_cone", rule_motion_cone_scores),
                ("workspace", rule_workspace_scores),
            ]:
                arr_valid = np.asarray(_arr, dtype=np.float32).reshape(-1)
                arr_v = arr_valid[np.isfinite(arr_valid) & valid_mask]
                score_stats[f"{_key}_nonzero_ratio"] = float(np.mean(arr_v > 1e-6)) if arr_v.size > 0 else 0.0

            # ── P15: ACGTP-v1 score branches — use dedicated modules ───────────────
            # P6: Per-branch timing for bottleneck analysis
            import time as _hook_time
            _t_scene_start = _hook_time.perf_counter()
            # 1. Scene layout: tabletop/support plane + object components + boundaries
            _scene_result = compute_scene_layout_scores(
                token_depth=token_depth,
                valid_mask=valid_mask,
                token_u=cache.get("u"),
                token_v=cache.get("v"),
                support_depth_min=float(self.config.acgtp_scene_support_depth_min),
                support_depth_max=float(self.config.acgtp_scene_support_depth_max),
                depth_edge_scores=edge_scores,
                object_min_area_tokens=int(self.config.acgtp_scene_object_min_area_tokens),
                object_height_residual_threshold=float(self.config.acgtp_scene_object_height_residual_threshold),
                grid_h=token_grid_shape[0],
                grid_w=token_grid_shape[1],
                support_plane_cap_ratio=float(self.config.acgtp_scene_support_plane_cap_ratio),
            )
            _t_scene_ms = (_hook_time.perf_counter() - _t_scene_start) * 1000.0
            _scene_layout_scores = _scene_result["scene_layout_scores"]
            _constrained_fill_mask = _scene_result["scene_fill_candidates"]
            _t_contact_start = _hook_time.perf_counter()
            # 2. Self-filtered contact ring: excludes self-core, gates on depth_edge
            # Compute gripper pixel projection directly (robot_metrics not in scope here)
            _gripper_pixel = self._project_gripper_to_pixel(gripper_pos, latest, T_robot_cam)
            _contact_result = compute_contact_ring_scores(
                token_u=cache.get("u"),
                token_v=cache.get("v"),
                gripper_pixel=_gripper_pixel,
                near_scores=rule_near_scores,
                self_core_radius_px=float(self.config.acgtp_self_core_radius_px),
                contact_ring_inner_px=float(self.config.acgtp_contact_ring_inner_px),
                contact_ring_outer_px=float(self.config.acgtp_contact_ring_outer_px),
                contact_requires_edge_or_object=bool(self.config.acgtp_contact_requires_edge_or_object),
                depth_edge_scores=edge_scores,
            )
            _t_contact_ms = (_hook_time.perf_counter() - _t_contact_start) * 1000.0
            _contact_ring_scores = _contact_result["contact_ring_scores"]
            _self_core_mask = _contact_result["robot_self_core_mask"]
            _t_motion_start = _hook_time.perf_counter()
            # 3. Motion corridor: smoothed swept path with EMA-smoothing
            # Bootstrap motion buffer BEFORE first corridor computation
            # so that the first gripper position enters the buffer immediately
            if self._motion_buffer is None and gripper_pos is not None:
                self._motion_buffer = create_motion_buffer(
                    maxlen=5,
                    ema_alpha=float(self.config.acgtp_motion_ema_alpha),
                )
            _motion_result = compute_motion_corridor_scores(
                points_robot=np.asarray(p_robot, dtype=np.float64),
                gripper_pos=np.asarray(gripper_pos, dtype=np.float64),
                prev_gripper_pos=np.asarray(prev_gripper_pos, dtype=np.float64) if prev_gripper_pos is not None else None,
                depth_edge_scores=edge_scores,
                motion_buffer=self._motion_buffer,
                corridor_length_m=float(self.config.acgtp_motion_corridor_length_m),
                corridor_sigma_m=float(self.config.acgtp_motion_sigma_m),
                min_motion_norm=1e-4,
                ema_alpha=float(self.config.acgtp_motion_ema_alpha),
            )
            _t_motion_ms = (_hook_time.perf_counter() - _t_motion_start) * 1000.0
            _motion_corridor_scores = _motion_result["motion_corridor_scores"]
            _motion_valid = _motion_result["motion_corridor_valid"]

            _t_acr_start = _hook_time.perf_counter()
            _acr_result = compute_future_action_constraint_scores(
                scene_layout_scores=_scene_layout_scores,
                depth_structure_scores=edge_scores,
                contact_ring_scores=_contact_ring_scores,
                motion_corridor_scores=_motion_corridor_scores,
                valid_mask=valid_mask,
                robot_self_core_mask=_self_core_mask,
                scene_result=_scene_result,
                contact_result=_contact_result,
                motion_result=_motion_result,
                w_scene=float(self.config.acgtp_w_scene_layout),
                w_depth=float(self.config.acgtp_w_depth_structure),
                w_contact=float(self.config.acgtp_w_contact_ring),
                w_motion=float(self.config.acgtp_w_motion_corridor),
            )
            _t_acr_ms = (_hook_time.perf_counter() - _t_acr_start) * 1000.0
            _action_constraint_scores = _acr_result["action_constraint_scores"]
            _object_side_contact_scores = _acr_result["object_side_contact_scores"]
            _swept_motion_risk_scores = _acr_result["swept_motion_risk_scores"]

            # Add ACGTP-v1 scores and module results to score_stats for the selector dispatch
            score_stats["acgtp_scene_layout_scores"] = _scene_layout_scores
            score_stats["acgtp_contact_ring_scores"] = _object_side_contact_scores
            score_stats["acgtp_raw_contact_ring_scores"] = _contact_ring_scores
            score_stats["acgtp_motion_corridor_scores"] = _swept_motion_risk_scores
            score_stats["acgtp_raw_motion_corridor_scores"] = _motion_corridor_scores
            score_stats["acgtp_action_constraint_scores"] = _action_constraint_scores
            score_stats["acgtp_motion_corridor_valid"] = _motion_valid
            score_stats["acgtp_self_core_mask"] = _self_core_mask
            score_stats["acgtp_constrained_fill_mask"] = _constrained_fill_mask
            # Gripper pixel projection for contact_ring and diagnostics
            score_stats["gripper_pixel"] = _gripper_pixel
            # Full contact ring result for diagnostics
            score_stats["acgtp_contact_ring_result"] = _contact_result
            # Full motion corridor result for diagnostics
            score_stats["acgtp_motion_corridor_result"] = _motion_result
            score_stats["acgtp_action_constraint_result"] = _acr_result
            # Full scene layout result for diagnostics
            score_stats["acgtp_scene_layout_result"] = _scene_result
            # P6: per-branch timing
            score_stats["acgtp_scene_layout_ms"] = _t_scene_ms
            score_stats["acgtp_contact_ring_ms"] = _t_contact_ms
            score_stats["acgtp_motion_corridor_ms"] = _t_motion_ms
            score_stats["acgtp_action_constraint_ms"] = _t_acr_ms
        else:
            scores, score_stats = compute_robot_geo_near_scores(
                token_depth,
                valid_mask,
                cache,
                T_robot_cam,
                gripper_pos,
                token_grid_shape=token_grid_shape,
                num_visual_tokens=num_tokens,
                w_edge=self.config.w_edge,
                w_near=self.config.w_near,
                sigma_near=self.config.sigma_near,
            )
        self._update_previous_gripper_pos(latest, gripper_pos)
        score_compute_ms = (time.perf_counter() - score_start) * 1000.0
        if not np.all(np.isfinite(np.asarray(scores, dtype=np.float32))):
            base_metrics.update({
                "token_mapping_ms": token_mapping_ms,
                "depth_sampling_ms": depth_sampling_ms,
                "valid_token_ratio": valid_ratio,
            })
            if "score_stats" in dir():
                base_metrics.update(score_stats)
            scores, valid_mask, depth_metrics, depth_fallback = self._compute_depth_edge_scores(num_tokens)
            base_metrics.update(depth_metrics)
            # P0-4: NaN geometry score, transform fields not reliable in error path
            base_metrics["transform_convention"] = None
            base_metrics["transform_inverse_used"] = None
            base_metrics["transform_source"] = None
            base_metrics["transform_convention_verified"] = None
            base_metrics["transform_convention_evidence"] = None
            return scores, valid_mask, base_metrics, depth_fallback or "geometry_score_nan"
        # The robot helper does projection and score fusion in one vectorized block.
        token_xyz_projection_ms = None
        score_fusion_ms = score_compute_ms

        base_metrics.update({
            # P0-4: Transform convention (must be re-included here since the full update()
            # below would otherwise drop fields set at the top of _compute_robot_geo_scores)
            "transform_convention": "T_robot_cam_forward",
            "transform_inverse_used": False,
            "transform_source": transform_key,
            "transform_convention_verified": True,
            "transform_convention_evidence": (
                "P0-3 overlay: forward aligns with gripper/eef; "
                "inverse falls on background; depth consistency ambiguous but not contradictory; "
                "physical z_cam positive supports forward"
            ),
            "token_mapping_ms": token_mapping_ms,
            "depth_sampling_ms": depth_sampling_ms,
            "score_compute_ms": score_compute_ms,
            "depth_edge_score_ms": score_compute_ms,
            "token_xyz_projection_ms": token_xyz_projection_ms,
            "score_fusion_ms": score_fusion_ms,
            "valid_token_ratio": valid_ratio,
            "gripper_pos": gripper_pos,
            "gripper_pixel": self._project_gripper_to_pixel(gripper_pos, latest, T_robot_cam),
            "token_points_robot": score_stats.get("token_points_robot"),
            "edge_scores": score_stats.get("edge_scores"),
            "near_scores": score_stats.get("near_scores"),
            "distances": score_stats.get("distances"),
            "d_min": score_stats.get("d_min"),
            "mean_near_score": score_stats.get("mean_near_score"),
            "max_near_score": score_stats.get("max_near_score"),
            "motion_norm": score_stats.get("motion_norm"),
            "corridor_strength_mean": score_stats.get("corridor_strength_mean"),
            "corridor_active": score_stats.get("corridor_active"),
            "d_corridor_min": score_stats.get("d_corridor_min"),
            "corridor_distances": score_stats.get("corridor_distances"),
            "corridor_scores": score_stats.get("corridor_scores"),
            "near_contact_scores": score_stats.get("near_contact_scores"),
            "corridor_contact_scores": score_stats.get("corridor_contact_scores"),
            "geo_contact_scores": score_stats.get("geo_contact_scores"),
            "cache": cache,
            "depth_edge_score_mean": score_stats.get("depth_edge_score_mean"),
            "edge_score_mean": score_stats.get("edge_score_mean"),
            "motion_cone_score_mean": score_stats.get("motion_cone_score_mean"),
            "motion_cone_score_max": score_stats.get("motion_cone_score_max"),
            "workspace_score_mean": score_stats.get("workspace_score_mean"),
            "workspace_score_std": score_stats.get("workspace_score_std"),
            "workspace_score_min": score_stats.get("workspace_score_min"),
            "workspace_score_p50": score_stats.get("workspace_score_p50"),
            "workspace_score_p90": score_stats.get("workspace_score_p90"),
            "workspace_score_max": score_stats.get("workspace_score_max"),
            # Plural keys: from _append_score_stats (key.replace("_scores", "_score"))
            "workspace_scores_mean": score_stats.get("workspace_scores_mean"),
            "workspace_scores_std": score_stats.get("workspace_scores_std"),
            "workspace_scores_max": score_stats.get("workspace_scores_max"),
            "contact_risk_score_mean": score_stats.get("contact_risk_score_mean"),
            "contact_risk_score_max": score_stats.get("contact_risk_score_max"),
            "dynamic_decision": score_stats.get("dynamic_decision"),
            "geo_risk_level": score_stats.get("geo_risk_level"),
            "geo_risk_score": score_stats.get("geo_risk_score"),
            "dynamic_keep_reason": score_stats.get("dynamic_keep_reason"),
            "num_high_contact_tokens": score_stats.get("num_high_contact_tokens"),
            "num_valid_3d_tokens": score_stats.get("num_valid_3d_tokens"),
            "near_contact_score_mean": score_stats.get("near_contact_score_mean"),
            "corridor_contact_score_mean": score_stats.get("corridor_contact_score_mean"),
            "geo_contact_score_mean": score_stats.get("geo_contact_score_mean"),
            "geometry_score_mean": score_stats.get("geometry_score_mean"),
            "geometry_score_max": score_stats.get("geometry_score_max"),
            "geometry_score_std": score_stats.get("geometry_score_std"),
            # New diagnostic fields: depth stats
            "depth_min": float(np.nanmin(token_depth)) if token_depth is not None and token_depth.size > 0 else None,
            "depth_max": float(np.nanmax(token_depth)) if token_depth is not None and token_depth.size > 0 else None,
            "depth_mean": float(np.nanmean(token_depth)) if token_depth is not None and token_depth.size > 0 else None,
            # 3D token geometry in robot frame
            "points_robot_min_xyz": _arr_stats(score_stats.get("token_points_robot"), "min"),
            "points_robot_max_xyz": _arr_stats(score_stats.get("token_points_robot"), "max"),
            "points_robot_mean_xyz": _arr_stats(score_stats.get("token_points_robot"), "mean"),
            "points_robot_std_xyz": _arr_stats(score_stats.get("token_points_robot"), "std"),
            # Camera frame geometry — not computed by robot geometry pipeline
            "points_cam_min_xyz": None,
            "points_cam_max_xyz": None,
            "points_cam_available": False,
            "points_cam_unavailable_reason": "camera_frame_not_connected_in_robot_geo_pipeline",
            "extrinsics_available": bool(T_robot_cam is not None),
            "intrinsics_available": bool(latest.camera_intrinsics is not None),
            "camera_frame_name": str(latest.camera_name) if hasattr(latest, "camera_name") and latest.camera_name else None,
            "geometry_frame_name": "robot",
            # Robot state / gripper
            "ee_position": _arr_to_str(gripper_pos) if gripper_pos is not None else None,
            "robot_state_valid": bool(gripper_pos is not None),
            "motion_direction_valid": bool(score_stats.get("corridor_active", False)),
            "motion_direction_xyz": None,
            "distance_to_gripper_min": score_stats.get("d_min"),
            "distance_to_gripper_mean": float(np.nanmean(score_stats.get("distances"))) if score_stats.get("distances") is not None else None,
            "distance_to_gripper_max": float(np.nanmax(score_stats.get("distances"))) if score_stats.get("distances") is not None else None,
            # Score component distribution stats (all tokens)
            "depth_edge_score_mean": score_stats.get("depth_edge_score_mean"),
            "depth_edge_score_std": score_stats.get("depth_edge_score_std"),
            "depth_edge_score_min": score_stats.get("depth_edge_score_min"),
            "depth_edge_score_p50": score_stats.get("depth_edge_score_p50"),
            "depth_edge_score_p90": score_stats.get("depth_edge_score_p90"),
            "depth_edge_score_max": score_stats.get("depth_edge_score_max"),
            "depth_edge_score_positive_ratio": score_stats.get("depth_edge_score_positive_ratio"),
            "distance_score_mean": score_stats.get("distance_score_mean"),
            "distance_score_std": score_stats.get("distance_score_std"),
            "distance_score_min": score_stats.get("distance_score_min"),
            "distance_score_p50": score_stats.get("distance_score_p50"),
            "distance_score_p90": score_stats.get("distance_score_p90"),
            "distance_score_max": score_stats.get("distance_score_max"),
            "motion_cone_score_mean": score_stats.get("motion_cone_score_mean"),
            "motion_cone_score_std": score_stats.get("motion_cone_score_std"),
            "motion_cone_score_min": score_stats.get("motion_cone_score_min"),
            "motion_cone_score_p50": score_stats.get("motion_cone_score_p50"),
            "motion_cone_score_p90": score_stats.get("motion_cone_score_p90"),
            "motion_cone_score_max": score_stats.get("motion_cone_score_max"),
            "motion_cone_score_positive_ratio": score_stats.get("motion_cone_score_positive_ratio"),
            "motion_cone_score_zero_ratio": score_stats.get("motion_cone_score_zero_ratio"),
            "motion_dir_norm_mean": score_stats.get("motion_dir_norm_mean"),
            "motion_dir_norm_min": score_stats.get("motion_dir_norm_min"),
            "motion_dir_norm_max": score_stats.get("motion_dir_norm_max"),
            "workspace_score_mean": score_stats.get("workspace_score_mean"),
            "workspace_score_std": score_stats.get("workspace_score_std"),
            "workspace_score_max": score_stats.get("workspace_score_max"),
            "contact_risk_score_mean": score_stats.get("contact_risk_score_mean"),
            "contact_risk_score_std": score_stats.get("contact_risk_score_std"),
            "contact_risk_score_min": score_stats.get("contact_risk_score_min"),
            "contact_risk_score_p50": score_stats.get("contact_risk_score_p50"),
            "contact_risk_score_p90": score_stats.get("contact_risk_score_p90"),
            "contact_risk_score_max": score_stats.get("contact_risk_score_max"),
            "final_geometry_score_mean": score_stats.get("geometry_score_mean"),
            "final_geometry_score_std": score_stats.get("geometry_score_std"),
            "final_geometry_score_min": score_stats.get("geometry_score_min"),
            "final_geometry_score_p50": score_stats.get("geometry_score_p50"),
            "final_geometry_score_p90": score_stats.get("geometry_score_p90"),
            "final_geometry_score_max": score_stats.get("geometry_score_max"),
            # P1-1: is_none flags for score components (populated by hybrid_v1 branch)
            "contact_risk_scores_is_none": score_stats.get("contact_risk_scores_is_none", True),
            "motion_cone_scores_is_none": score_stats.get("motion_cone_scores_is_none", True),
            "workspace_scores_is_none": score_stats.get("workspace_scores_is_none", True),
            "contact_nonzero_ratio": score_stats.get("contact_risk_nonzero_ratio", 0.0),
            "motion_cone_nonzero_ratio": score_stats.get("motion_cone_nonzero_ratio", 0.0),
            "workspace_nonzero_ratio": score_stats.get("workspace_nonzero_ratio", 0.0),
            # P1-1: Raw score arrays for token selection attribution
            "workspace_scores": rule_workspace_scores,
            # P1-1: temporal_v1 semantics fix
            "temporal_enabled": score_stats.get("temporal_enabled", False),
            "ema_enabled": score_stats.get("ema_enabled", False),
            "interaction_lock_triggered": score_stats.get("interaction_lock_triggered", False),
            "interaction_lock_reason": score_stats.get("interaction_lock_reason", None),
            # P1: token UV coordinates for spatial distribution diagnostics
            "token_u": cache.get("u"),
            "token_v": cache.get("v"),
            "token_grid_shape": token_grid_shape,
            # P1: gripper pixel projection for near-gripper distance diagnostics
            "gripper_pixel": self._project_gripper_to_pixel(gripper_pos, latest, T_robot_cam),
            # P1: Hybrid v1 scores — precompute for selection attribution
            # NOTE: the following arrays are only in scope inside this elif block
            "hybrid_edge_norm": _norm_hybrid_component(edge_scores, valid_mask),
            "hybrid_near_norm": _norm_hybrid_component(rule_near_scores, valid_mask),
            "hybrid_contact_norm": _norm_hybrid_component(rule_contact_scores, valid_mask),
            "hybrid_corr_norm": _norm_hybrid_component(rule_motion_cone_scores, valid_mask),
            "hybrid_final_scores": _build_hybrid_final_scores(
                edge_scores, rule_near_scores, rule_contact_scores,
                rule_motion_cone_scores, valid_mask,
                self.config.hybrid_v1_weights,
            ),
        })
        # Merge ACGTP-v1 score_stats into base_metrics so gripper_pixel and module
        # results flow through to the metrics recording
        base_metrics.update(score_stats)
        if self.config.detailed_pruning_timing:
            base_metrics.update({
                "edge_score_ms": score_stats.get("edge_score_ms"),
                "robot_mapping_ms": score_stats.get("robot_mapping_ms"),
                "near_score_ms": score_stats.get("near_score_ms"),
                "corridor_score_ms": score_stats.get("corridor_score_ms"),
                "contact_score_ms": score_stats.get("contact_score_ms"),
            })
        return scores, valid_mask, base_metrics, None

    def _project_gripper_to_pixel(
        self,
        gripper_pos_robot: Optional[np.ndarray],
        latest: Any,
        T_robot_cam: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """Project gripper position (robot frame) to pixel coordinates (u, v).

        Returns None if intrinsics are unavailable.
        """
        if gripper_pos_robot is None or T_robot_cam is None:
            return None
        try:
            K = getattr(latest, "camera_intrinsics", None)
            if K is None:
                return None
            K_arr = np.asarray(K, dtype=np.float32)
            if K_arr.shape != (3, 3):
                return None
            T_cam = np.linalg.inv(np.asarray(T_robot_cam, dtype=np.float32))
            p_robot = np.asarray(gripper_pos_robot, dtype=np.float32).reshape(3)
            p_cam = T_cam[:3, :3] @ p_robot + T_cam[:3, 3]
            if p_cam[2] <= 0:
                return None
            fx, fy, cx, cy = K_arr[0, 0], K_arr[1, 1], K_arr[0, 2], K_arr[1, 2]
            u = float(fx * p_cam[0] / p_cam[2] + cx)
            v = float(fy * p_cam[1] / p_cam[2] + cy)
            return np.array([u, v], dtype=np.float32)
        except Exception:
            return None

    def _apply_self_mask(
        self,
        scores: np.ndarray,
        gripper_pixel: Optional[np.ndarray],
        token_u: Optional[np.ndarray],
        token_v: Optional[np.ndarray],
        core_radius_px: float,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Apply minimum 2D robot-self mask: penalize tokens within core_radius_px of gripper projection.

        Args:
            scores: Token scores array [N].
            gripper_pixel: (u, v) pixel coordinates of gripper projection, or None.
            token_u: [N] token pixel u-coordinates, or None.
            token_v: [N] token pixel v-coordinates, or None.
            core_radius_px: Pixel radius for self-core mask.

        Returns:
            Tuple of (masked_scores, self_mask, diagnostics_dict).
            self_mask[i] = True if token i is in self-core region.
        """
        diagnostics: Dict[str, Any] = {}
        diagnostics["self_mask_available"] = False
        diagnostics["gripper_pixel_in_bounds"] = False
        diagnostics["gripper_projection_valid"] = gripper_pixel is not None
        diagnostics["gripper_pixel_u"] = float(gripper_pixel[0]) if gripper_pixel is not None else None
        diagnostics["gripper_pixel_v"] = float(gripper_pixel[1]) if gripper_pixel is not None else None
        diagnostics["self_mask_core_radius_px"] = float(core_radius_px)

        if gripper_pixel is None or token_u is None or token_v is None:
            return scores, np.zeros(len(scores), dtype=bool), diagnostics

        gx, gy = float(gripper_pixel[0]), float(gripper_pixel[1])

        # Check bounds (allow margin of 1 for edge cases)
        img_size = 256
        diagnostics["gripper_pixel_in_bounds"] = (
            0 <= gx < img_size and 0 <= gy < img_size
        )

        if not diagnostics["gripper_pixel_in_bounds"]:
            return scores, np.zeros(len(scores), dtype=bool), diagnostics

        u_arr = np.asarray(token_u, dtype=np.float32).reshape(-1)
        v_arr = np.asarray(token_v, dtype=np.float32).reshape(-1)

        if u_arr.size != len(scores) or v_arr.size != len(scores):
            return scores, np.zeros(len(scores), dtype=bool), diagnostics

        r2 = float(core_radius_px) ** 2
        dist2 = (u_arr - gx) ** 2 + (v_arr - gy) ** 2
        self_mask = dist2 <= r2

        n = len(scores)
        diagnostics["self_mask_available"] = True
        diagnostics["self_mask_token_count"] = int(np.sum(self_mask))
        diagnostics["self_mask_token_ratio"] = float(np.sum(self_mask) / n) if n > 0 else 0.0

        masked_scores = np.asarray(scores, dtype=np.float32).reshape(-1).copy()

        # Record pre-mask near_score statistics
        diagnostics["near_score_mean_before_self_mask"] = float(np.mean(masked_scores))
        if np.any(self_mask):
            valid_self = self_mask & np.isfinite(masked_scores)
            diagnostics["near_score_self_region_mean"] = float(np.mean(masked_scores[valid_self])) if np.any(valid_self) else None
        valid_nonself = (~self_mask) & np.isfinite(masked_scores)
        diagnostics["near_score_nonself_region_mean"] = float(np.mean(masked_scores[valid_nonself])) if np.any(valid_nonself) else None

        # P14-B: Apply penalty ONLY to self-core tokens (do NOT delete them, just reduce their score).
        # penalty=0.0 is a no-op (scores unchanged). With penalty=0.5, self-core scores are halved.
        if self.config.robot_self_mask_penalty > 0.0:
            masked_scores[self_mask] *= (1.0 - self.config.robot_self_mask_penalty)

        diagnostics["near_score_mean_after_self_mask"] = float(np.mean(masked_scores))

        return masked_scores, self_mask, diagnostics

    def _get_previous_gripper_pos(self, latest: Any) -> Optional[np.ndarray]:
        episode_id = getattr(latest, "episode_id", None)
        step_id = getattr(latest, "step_id", None)
        if episode_id is not None and episode_id != self._prev_episode_id:
            self._prev_episode_id = int(episode_id)
            self._prev_gripper_pos = None
        if step_id == 0:
            self._prev_gripper_pos = None
        return None if self._prev_gripper_pos is None else self._prev_gripper_pos.copy()

    def _update_previous_gripper_pos(self, latest: Any, gripper_pos: np.ndarray) -> None:
        episode_id = getattr(latest, "episode_id", None)
        if episode_id is not None:
            self._prev_episode_id = int(episode_id)
        self._prev_gripper_pos = np.asarray(gripper_pos, dtype=np.float32).reshape(3).copy()
