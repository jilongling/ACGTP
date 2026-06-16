"""Projector-output hook for external visual-token pruning."""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .config import PruningHookConfig
from .depth_edge import compute_depth_edge_scores, compute_valid_depth_mask
from .geometry_cache import TokenGeometryCache
from .internal_pruning import (
    disable_acgtp_internal_pruning,
    enable_acgtp_internal_pruning,
)
from .metrics import HookMetrics, HookTiming
from .post_pruning import PostPruningStateManager
from .static_scene_cache import ACGTPStaticSceneCache
from .strategy_registry import (
    ACGTP_STRATEGIES,
    BRANCH_MIXTURE_SCORE_STRATEGIES,
    DYNAMIC_MID_KEEP_STRATEGIES,
    EARLY_GEOMETRY_FALLBACK_STRATEGIES,
    EDGE_RESERVE_LEGACY_STRATEGIES,
    ROBOT_STATE_REQUIRED_LEGACY_STRATEGIES,
    SELF_HANDLED_SELECTOR_STRATEGIES,
    ROBOT_GEO_SCORE_STRATEGIES,
    TOKEN_SELECTION_DEBUG_STRATEGIES,
)
from .robot_geometry import (
    compute_robot_geo_scores_v0,
    compute_robot_geo_contact_budget_scores,
    compute_robot_geo_corridor_scores,
    compute_robot_geo_near_scores,
    decide_dynamic_keep_ratio,
    extract_gripper_position,
    extract_robot_camera_transform,
    project_tokens_to_robot,
)
from .robot_state import RobotState
from .scheduler import compute_dynamic_keep_ratio
from .selector import (
    finalize_selection_debug_info,
    select_keep_indices,
    select_tokens_contact_budget,
    select_hybrid_quota_union,
    select_hybrid_quota_v2,
    select_hybrid_v1,
    select_hybrid_v1_edge_reserve,
    select_hybrid_budget_v2,
    select_branch_budget_v0,
    select_acgtp_v1,
    select_acgtp_v2_fast,
    select_acgtp_v2,
    validate_keep_indices,
)
from .temporal_geometry import GeometryHistoryBuffer
from .visualization import save_geo_debug_visualization, save_pruning_visualization

# P15: ACGTP-v1 new modules
from .scene_layout import compute_scene_layout_scores
from .contact_ring import compute_contact_ring_scores
from .motion_corridor import compute_motion_corridor_scores, create_motion_buffer, MotionEMABuffer
from .action_constraint import compute_future_action_constraint_scores
from .acgtp_dynamic_controller import decide_acgtp_dynamic_budget
from .acgtp_history import ACGTPHistoryBuffer

# P16: ACGTP-v2 Task-Semantic Anchor module
from .semantic_anchors import compute_task_semantic_anchors, parse_instruction_terms


class VisualTokenPruningHook:
    """Prunes only projector output visual tokens before LLM input construction."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        geometry_recorder: Optional[Any] = None,
        visualizer: Optional[Any] = None,
    ) -> None:
        self.cfg = cfg
        self.config = PruningHookConfig.from_eval_cfg(cfg)
        self.geometry_recorder = geometry_recorder
        self.visualizer = visualizer
        self._hook_handle = None
        self._latest_stats: Dict[str, Any] = {}
        self._latest_preprocess_meta: Optional[Any] = None
        self._latest_token_grid_shape: Optional[Tuple[int, int]] = None
        self._cache = TokenGeometryCache()
        self._prev_gripper_pos: Optional[np.ndarray] = None
        self._prev_episode_id: Optional[int] = None
        self._hook_episode_id: Optional[int] = None
        self._hook_step_counter: int = 0
        self._geo_debug_frames_saved: int = 0
        self._dropped_overlay_frames_saved: int = 0
        self._temporal_history = GeometryHistoryBuffer(maxlen=self.config.temporal_history_length)
        # P15: ACGTP-v1 motion EMA buffer for smoothed motion corridor
        self._motion_buffer: Optional[MotionEMABuffer] = None
        # Step 5: lightweight phase hysteresis state for ACGTP dynamic pruning.
        self._acgtp_dynamic_state: Dict[str, Any] = {}
        self._static_scene_cache = ACGTPStaticSceneCache(self.config)
        self._acgtp_latency_plan_cache: Optional[Dict[str, Any]] = None
        self._acgtp_history = ACGTPHistoryBuffer(
            maxlen=self.config.acgtp_history_length,
            scene_current_weight=self.config.acgtp_history_scene_ema_alpha,
            depth_current_weight=self.config.acgtp_history_depth_ema_alpha,
            contact_current_weight=self.config.acgtp_history_contact_ema_alpha,
            motion_current_weight=self.config.acgtp_history_motion_ema_alpha,
            action_current_weight=self.config.acgtp_history_action_ema_alpha,
            depth_change_threshold=self.config.acgtp_history_depth_change_threshold,
            keep_iou_threshold=self.config.acgtp_history_keep_iou_threshold,
            motion_stability_threshold=self.config.acgtp_history_motion_stability_threshold,
        )
        self._acgtp_attention_history = deque(maxlen=max(1, int(self.config.acgtp_attention_history_length)))
        self._lm_pre_hook_handle = None
        self._post_pruning = PostPruningStateManager(
            config=self.config,
            update_stats=self._update_latest_position_stats,
        )
        self._internal_backend = None

    def set_preprocess_meta(self, meta: Any, token_grid_shape: Optional[Tuple[int, int]]) -> None:
        self._latest_preprocess_meta = meta
        self._latest_token_grid_shape = token_grid_shape

    def reset_step(self) -> None:
        self._latest_stats = {}
        self._last_selector_exception: Optional[Exception] = None
        self._last_selector_name: Optional[str] = None
        self._post_pruning.reset()

    @staticmethod
    def _parse_acgtp_ablate_branches(raw: Any) -> set:
        if raw is None:
            return set()
        if isinstance(raw, (list, tuple, set)):
            pieces = raw
        else:
            text = str(raw).strip().lower()
            if text in ("", "none", "off", "false", "0"):
                return set()
            pieces = text.replace(";", ",").replace("|", ",").split(",")

        aliases = {
            "scene": "scene",
            "scene_layout": "scene",
            "layout": "scene",
            "depth": "depth",
            "structure": "depth",
            "depth_structure": "depth",
            "contact": "contact",
            "contact_ring": "contact",
            "motion": "motion",
            "motion_corridor": "motion",
            "fill": "fill",
            "constrained_fill": "fill",
        }
        out = set()
        for item in pieces:
            key = str(item).strip().lower().replace("-", "_")
            if key in ("", "none", "off", "false", "0"):
                continue
            if key == "all":
                out.update(("scene", "depth", "contact", "motion", "fill"))
                continue
            mapped = aliases.get(key)
            if mapped is not None:
                out.add(mapped)
        return out

    def get_latest_stats(self) -> Dict[str, Any]:
        stats = dict(self._latest_stats)
        if self._internal_backend is not None:
            stats.update(self._internal_info_to_stats(self._internal_backend.stats()))
        return stats

    def _update_latest_position_stats(self, **updates: Any) -> None:
        if not isinstance(self._latest_stats, dict):
            self._latest_stats = {}
        self._latest_stats.update(updates)

    def _prepare_position_preserve_info(
        self,
        *,
        keep_indices_np: np.ndarray,
        num_tokens: int,
        metrics: HookMetrics,
    ) -> None:
        self._post_pruning.prepare_position_preserve_info(
            keep_indices_np=keep_indices_np,
            num_tokens=num_tokens,
            metrics=metrics,
        )

    def _language_model_pre_hook(self, module: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
        return self._post_pruning.language_model_pre_hook(module, args, kwargs)

    def _default_keep_ratio_source(self) -> str:
        if self.config.strategy == "robot_geo_acgtp_v2" and bool(getattr(self.config, "acgtp_dynamic_enabled", False)):
            return "acgtp_dynamic_controller"
        return "dynamic_mid_keep_ratio" if self.config.strategy in DYNAMIC_MID_KEEP_STRATEGIES else "cli_keep_ratio"

    def _compression_backend(self) -> str:
        backend = str(getattr(self.config, "acgtp_compression_backend", "projector") or "projector").strip().lower()
        if bool(getattr(self.config, "acgtp_internal_pruning_enabled", False)):
            backend = "internal"
        return "internal" if backend == "internal" else "projector"

    def _internal_pruning_requested(self) -> bool:
        return self._compression_backend() == "internal"

    @staticmethod
    def _internal_info_to_stats(info: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(info, dict) or not info:
            return {}
        return {
            "compression_backend": "internal",
            "projector_pruning_applied": False,
            "internal_pruning_requested": True,
            "internal_pruning_plan_ready": info.get("plan_ready"),
            "internal_pruning_applied": info.get("applied"),
            "internal_pruning_layer": info.get("pruning_layer", info.get("requested_prune_layer")),
            "internal_pruning_disabled_reason": info.get("disabled_reason"),
            "internal_original_seq_length": info.get("original_seq_length"),
            "internal_kept_seq_length": info.get("kept_seq_length"),
            "internal_pruned_seq_length": info.get("pruned_seq_length"),
            "internal_original_visual_tokens": info.get("original_visual_tokens", info.get("image_token_length")),
            "internal_kept_visual_tokens": info.get("kept_visual_tokens"),
            "internal_pruned_visual_tokens": info.get("pruned_visual_tokens"),
            "internal_decode_calls": info.get("decode_calls"),
            "internal_decode_cache_consistent": info.get("decode_cache_consistent"),
            "internal_selection_mode": info.get("internal_selection_mode"),
            "internal_attention_enabled": info.get("internal_attention_enabled"),
            "internal_attention_available": info.get("internal_attention_available"),
            "internal_attention_source": info.get("internal_attention_source"),
            "internal_attention_confidence": info.get("internal_attention_confidence"),
            "internal_historical_action_attention_available": info.get("internal_historical_action_attention_available"),
            "internal_historical_action_attention_source": info.get("internal_historical_action_attention_source"),
            "internal_geo_attention_iou": info.get("internal_geo_attention_iou"),
            "internal_high_geometry_low_attention_count": info.get("internal_high_geometry_low_attention_count"),
            "internal_high_attention_low_geometry_count": info.get("internal_high_attention_low_geometry_count"),
            "internal_attention_dropped_geo_count": info.get("internal_attention_dropped_geo_count"),
            "internal_pruned_geo_critical_count": info.get("internal_pruned_geo_critical_count"),
            "internal_geo_protected_count": info.get("internal_geo_protected_count"),
            "internal_geo_explicit_protected_count": info.get("internal_geo_explicit_protected_count"),
            "internal_geo_explicit_protected_kept_count": info.get("internal_geo_explicit_protected_kept_count"),
            "internal_budget_raised_for_geo_protection": info.get("internal_budget_raised_for_geo_protection"),
            "internal_dynamic_risk": info.get("internal_dynamic_risk"),
            "internal_dynamic_risk_level": info.get("internal_dynamic_risk_level"),
            "internal_dynamic_keep_ratio": info.get("internal_dynamic_keep_ratio"),
            "internal_dynamic_keep_k": info.get("internal_dynamic_keep_k"),
            "internal_quota_hard_k": info.get("internal_quota_hard_k"),
            "internal_quota_semantic_attention_k": info.get("internal_quota_semantic_attention_k"),
            "internal_quota_historical_attention_k": info.get("internal_quota_historical_attention_k"),
            "internal_functional_quota_enabled": info.get("internal_functional_quota_enabled"),
            "internal_quota_layout_k": info.get("internal_quota_layout_k"),
            "internal_quota_contact_k": info.get("internal_quota_contact_k"),
            "internal_quota_motion_k": info.get("internal_quota_motion_k"),
            "internal_quota_fill_k": info.get("internal_quota_fill_k"),
            "internal_selected_by_geo_count": info.get("internal_selected_by_geo_count"),
            "internal_selected_by_layout_count": info.get("internal_selected_by_layout_count"),
            "internal_selected_by_contact_count": info.get("internal_selected_by_contact_count"),
            "internal_selected_by_motion_count": info.get("internal_selected_by_motion_count"),
            "internal_selected_by_semantic_attention_count": info.get("internal_selected_by_semantic_attention_count"),
            "internal_selected_by_historical_attention_count": info.get("internal_selected_by_historical_attention_count"),
            "internal_selected_by_fill_count": info.get("internal_selected_by_fill_count"),
            "internal_selected_by_fallback_count": info.get("internal_selected_by_fallback_count"),
            "internal_unique_geo_count": info.get("internal_unique_geo_count"),
            "internal_unique_layout_count": info.get("internal_unique_layout_count"),
            "internal_unique_contact_count": info.get("internal_unique_contact_count"),
            "internal_unique_motion_count": info.get("internal_unique_motion_count"),
            "internal_unique_semantic_attention_count": info.get("internal_unique_semantic_attention_count"),
            "internal_unique_historical_attention_count": info.get("internal_unique_historical_attention_count"),
            "internal_unique_fill_count": info.get("internal_unique_fill_count"),
            "internal_unique_fallback_count": info.get("internal_unique_fallback_count"),
            "internal_branch_selected_sum": info.get("internal_branch_selected_sum"),
            "internal_branch_unique_sum": info.get("internal_branch_unique_sum"),
            "internal_branch_overlap_count": info.get("internal_branch_overlap_count"),
            "internal_branch_overlap_ratio": info.get("internal_branch_overlap_ratio"),
            "internal_branch_unique_ratio": info.get("internal_branch_unique_ratio"),
            "internal_branch_sum_equals_kept": info.get("internal_branch_sum_equals_kept"),
            "internal_branch_accounting_valid": info.get("internal_branch_accounting_valid"),
            "internal_fallback_added_count": info.get("internal_fallback_added_count"),
            "internal_attention_requires_geometry_alignment": info.get("internal_attention_requires_geometry_alignment"),
        }

    def _prepare_internal_pruning_plan(
        self,
        *,
        keep_indices_np: np.ndarray,
        num_tokens: int,
        metrics: HookMetrics,
        selection_meta: Dict[str, Any],
        geometry_payload: Optional[Dict[str, Any]] = None,
        target_keep_ratio: Optional[float] = None,
        quota_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._post_pruning.prepare_internal_pruning_plan(
            backend=self._internal_backend,
            keep_indices_np=keep_indices_np,
            num_tokens=int(num_tokens),
            metrics=metrics,
            selection_meta=selection_meta,
            source=str(selection_meta.get("selector_name") or selection_meta.get("selection_strategy_name") or self.config.strategy),
            geometry_payload=geometry_payload,
            target_keep_ratio=target_keep_ratio,
            quota_config=quota_config,
        )

    def _runtime_mode(self) -> str:
        mode = str(getattr(self.config, "acgtp_runtime_mode", "fast") or "fast").strip().lower()
        return mode if mode in {"fast", "debug", "audit"} else "fast"

    def _is_fast_runtime(self) -> bool:
        return self._runtime_mode() == "fast"

    def _is_audit_runtime(self) -> bool:
        return self._runtime_mode() == "audit" or bool(getattr(self.config, "acgtp_full_diagnostics_enabled", False))

    def _latency_plan_cache_enabled(self) -> bool:
        return (
            self._is_fast_runtime()
            and not self._is_audit_runtime()
            and bool(getattr(self.config, "acgtp_latency_plan_cache_enabled", False))
        )

    @staticmethod
    def _clone_latency_cache_value(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, dict):
            return {k: VisualTokenPruningHook._clone_latency_cache_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [VisualTokenPruningHook._clone_latency_cache_value(v) for v in value]
        if isinstance(value, tuple):
            return tuple(VisualTokenPruningHook._clone_latency_cache_value(v) for v in value)
        return value

    def _reset_latency_plan_cache(self) -> None:
        self._acgtp_latency_plan_cache = None

    def _latency_depth_probe(self, depth: np.ndarray) -> np.ndarray:
        arr = np.asarray(depth, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        if arr.size == 0:
            return np.zeros((0,), dtype=np.float32)
        stride_y = max(1, int(arr.shape[0]) // 16)
        stride_x = max(1, int(arr.shape[1]) // 16)
        probe = arr[::stride_y, ::stride_x][:16, :16]
        return np.nan_to_num(probe, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=True)

    def _lookup_latency_plan_cache(
        self,
        *,
        depth_probe: Optional[np.ndarray],
        grip: np.ndarray,
        num_tokens: int,
        keep_count: int,
        token_grid_shape: Tuple[int, int],
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        start = time.perf_counter()
        enabled = self._latency_plan_cache_enabled()
        meta: Dict[str, Any] = {
            "acgtp_latency_plan_cache_enabled": enabled,
            "acgtp_latency_plan_cache_hit": False,
            "acgtp_latency_plan_cache_reason": "disabled" if not enabled else None,
            "acgtp_latency_plan_cache_age": None,
            "acgtp_latency_plan_cache_depth_delta": None,
            "acgtp_latency_plan_cache_gripper_delta": None,
            "acgtp_latency_plan_cache_lookup_ms": None,
            "acgtp_latency_plan_cache_keep_count": None,
        }
        if not enabled:
            meta["acgtp_latency_plan_cache_lookup_ms"] = (time.perf_counter() - start) * 1000.0
            return None, meta

        cached = self._acgtp_latency_plan_cache
        if not cached:
            meta["acgtp_latency_plan_cache_reason"] = "empty"
            meta["acgtp_latency_plan_cache_lookup_ms"] = (time.perf_counter() - start) * 1000.0
            return None, meta

        age = max(0, int(self._hook_step_counter) - int(cached.get("step_counter", 0)))
        meta["acgtp_latency_plan_cache_age"] = age
        max_age = int(getattr(self.config, "acgtp_latency_plan_cache_max_age", 4))
        if max_age <= 0 or age <= 0 or age > max_age:
            meta["acgtp_latency_plan_cache_reason"] = "age"
            meta["acgtp_latency_plan_cache_lookup_ms"] = (time.perf_counter() - start) * 1000.0
            return None, meta
        if int(cached.get("num_tokens", -1)) != int(num_tokens) or int(cached.get("keep_count", -1)) != int(keep_count):
            meta["acgtp_latency_plan_cache_reason"] = "shape_or_keep"
            meta["acgtp_latency_plan_cache_lookup_ms"] = (time.perf_counter() - start) * 1000.0
            return None, meta
        if tuple(cached.get("token_grid_shape", ())) != tuple(token_grid_shape):
            meta["acgtp_latency_plan_cache_reason"] = "grid"
            meta["acgtp_latency_plan_cache_lookup_ms"] = (time.perf_counter() - start) * 1000.0
            return None, meta

        cached_probe = cached.get("depth_probe")
        if depth_probe is None or cached_probe is None or np.asarray(depth_probe).shape != np.asarray(cached_probe).shape:
            meta["acgtp_latency_plan_cache_reason"] = "depth_probe"
            meta["acgtp_latency_plan_cache_lookup_ms"] = (time.perf_counter() - start) * 1000.0
            return None, meta
        depth_delta = float(np.mean(np.abs(np.asarray(depth_probe, dtype=np.float32) - np.asarray(cached_probe, dtype=np.float32))))
        meta["acgtp_latency_plan_cache_depth_delta"] = depth_delta
        if depth_delta > float(getattr(self.config, "acgtp_latency_plan_cache_depth_delta_threshold", 0.030)):
            meta["acgtp_latency_plan_cache_reason"] = "depth_delta"
            meta["acgtp_latency_plan_cache_lookup_ms"] = (time.perf_counter() - start) * 1000.0
            return None, meta

        cached_grip = cached.get("gripper_pos")
        if cached_grip is None:
            meta["acgtp_latency_plan_cache_reason"] = "gripper_missing"
            meta["acgtp_latency_plan_cache_lookup_ms"] = (time.perf_counter() - start) * 1000.0
            return None, meta
        grip_delta = float(np.linalg.norm(np.asarray(grip, dtype=np.float32).reshape(3) - np.asarray(cached_grip, dtype=np.float32).reshape(3)))
        meta["acgtp_latency_plan_cache_gripper_delta"] = grip_delta
        if grip_delta > float(getattr(self.config, "acgtp_latency_plan_cache_gripper_delta_threshold", 0.080)):
            meta["acgtp_latency_plan_cache_reason"] = "gripper_delta"
            meta["acgtp_latency_plan_cache_lookup_ms"] = (time.perf_counter() - start) * 1000.0
            return None, meta

        meta["acgtp_latency_plan_cache_hit"] = True
        meta["acgtp_latency_plan_cache_reason"] = "hit"
        meta["acgtp_latency_plan_cache_keep_count"] = int(cached.get("keep_indices", np.empty(0)).size)
        meta["acgtp_latency_plan_cache_lookup_ms"] = (time.perf_counter() - start) * 1000.0
        return cached, meta

    def _store_latency_plan_cache(
        self,
        *,
        depth_probe: Optional[np.ndarray],
        grip: np.ndarray,
        num_tokens: int,
        keep_count: int,
        token_grid_shape: Tuple[int, int],
        keep_indices_np: np.ndarray,
        selection_meta: Dict[str, Any],
        geometry_payload: Optional[Dict[str, Any]],
        robot_metrics: Dict[str, Any],
        valid_ratio: Optional[float],
        depth_source_key: Optional[Any],
    ) -> None:
        if not self._latency_plan_cache_enabled() or depth_probe is None:
            self._reset_latency_plan_cache()
            return
        self._acgtp_latency_plan_cache = {
            "step_counter": int(self._hook_step_counter),
            "depth_probe": np.asarray(depth_probe, dtype=np.float32).copy(),
            "gripper_pos": np.asarray(grip, dtype=np.float32).reshape(3).copy(),
            "num_tokens": int(num_tokens),
            "keep_count": int(keep_count),
            "token_grid_shape": tuple(token_grid_shape),
            "keep_indices": np.asarray(keep_indices_np, dtype=np.int64).reshape(-1).copy(),
            "selection_meta": self._clone_latency_cache_value(selection_meta),
            "geometry_payload": self._clone_latency_cache_value(geometry_payload),
            "robot_metrics": self._clone_latency_cache_value(robot_metrics),
            "valid_ratio": valid_ratio,
            "depth_source_key": depth_source_key,
        }

    def _acgtp_fast_runtime_enabled(self) -> bool:
        """Return True when ACGTP-v2 should use the lean rollout hot path."""
        semantic_active = (
            bool(getattr(self.config, "acgtp_v2_semantic_enabled", False))
            and str(getattr(self.config, "acgtp_v2_semantic_backend", "none")) != "none"
        )
        debug_visual = any(
            bool(self.cfg.get(key, False))
            for key in (
                "save_pruning_vis",
                "save_pruning_debug",
                "save_token_selection_debug",
            )
        )
        return (
            self.config.strategy == "robot_geo_acgtp_v2"
            and self._is_fast_runtime()
            and bool(getattr(self.config, "acgtp_fast_selector_enabled", True))
            and not bool(getattr(self.config, "acgtp_full_diagnostics_enabled", False))
            and not semantic_active
            and not debug_visual
            and not bool(getattr(self.config, "enable_geo_debug", False))
        )

    def _finalize_acgtp_fast_runtime(
        self,
        *,
        visual_tokens: torch.Tensor,
        metrics: HookMetrics,
        keep_indices_np: np.ndarray,
        selection_meta: Dict[str, Any],
        robot_metrics: Dict[str, Any],
        num_tokens: int,
        geometry_payload: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, HookMetrics]:
        """Minimal ACGTP-v2 hot-path finalization.

        This deliberately skips full attribution/accounting recomputation. The
        selector already returns lightweight branch counts, while detailed
        audits remain available through acgtp_full_diagnostics_enabled=True.
        """
        idx = np.asarray(keep_indices_np, dtype=np.int64).reshape(-1)
        idx = idx[(idx >= 0) & (idx < int(num_tokens))]
        if idx.size == 0:
            idx = np.arange(int(num_tokens), dtype=np.int64)
            selection_meta["fallback_used"] = True
            selection_meta["fallback_reason"] = selection_meta.get("fallback_reason") or "fast_runtime_empty_indices"
        else:
            idx = np.sort(np.unique(idx)).astype(np.int64, copy=False)

        selector_ms = selection_meta.get("selection_time_ms")
        try:
            selector_ms = float(selector_ms) if selector_ms is not None else None
        except Exception:
            selector_ms = None
        metrics.timing.selection_ms = selector_ms
        metrics.timing.pruning_time_ms = selector_ms
        metrics.timing.topk_pruning_ms = selector_ms

        internal_mode = self._internal_pruning_requested()
        if internal_mode:
            self._prepare_internal_pruning_plan(
                keep_indices_np=idx,
                num_tokens=int(num_tokens),
                metrics=metrics,
                selection_meta=selection_meta,
                geometry_payload=geometry_payload,
                target_keep_ratio=float(selection_meta.get("acgtp_dynamic_keep_ratio") or selection_meta.get("requested_keep_ratio") or (idx.size / max(1, int(num_tokens)))),
                quota_config=(geometry_payload or {}).get("quota_config") if isinstance(geometry_payload, dict) else None,
            )
            metrics.timing.gather_ms = 0.0
            pruned = visual_tokens
            kept = int(idx.size)
        else:
            gather_start = time.perf_counter()
            keep_indices = torch.as_tensor(idx, dtype=torch.long, device=visual_tokens.device)
            pruned = visual_tokens.index_select(dim=1, index=keep_indices)
            metrics.timing.gather_ms = (time.perf_counter() - gather_start) * 1000.0
            kept = int(pruned.shape[1])
            metrics.compression_backend = "projector"
            metrics.projector_pruning_applied = bool(kept < int(num_tokens))
            metrics.internal_pruning_requested = False
            metrics.internal_pruning_plan_ready = False
            metrics.internal_pruning_applied = False

        actual_ratio = kept / int(num_tokens) if num_tokens else 1.0
        metrics.num_visual_tokens_kept = kept
        metrics.num_visual_tokens_pruned = int(num_tokens) - kept
        metrics.num_visual_tokens_dropped = int(num_tokens) - kept
        metrics.num_visual_tokens_original_total = int(num_tokens)
        metrics.num_visual_tokens_kept_total = kept
        metrics.selected_token_count = kept
        metrics.dropped_token_count = int(num_tokens) - kept
        metrics.selected_token_ratio = actual_ratio
        metrics.dropped_token_ratio = 1.0 - actual_ratio
        metrics.retention_ratio = actual_ratio
        metrics.keep_ratio = actual_ratio
        metrics.actual_keep_ratio = actual_ratio
        metrics.keep_ratio_actual = actual_ratio
        metrics.retention_actual = actual_ratio
        metrics.actual_retention_ratio = actual_ratio
        metrics.effective_keep_count = kept
        metrics.original_token_count = int(num_tokens)
        metrics.protected_token_ratio = actual_ratio
        metrics.pruned_token_ratio = 1.0 - actual_ratio
        metrics.keep_indices_count = kept
        metrics.keep_indices_sorted = True
        metrics.keep_indices_unique = True
        metrics.keep_indices_out_of_bounds = False
        metrics.duplicate_indices_count = 0
        metrics.no_duplicate_final_indices = True
        metrics.final_indices_sorted = True
        metrics.selected_token_count_equals_kept = True
        metrics.retention_ratio_valid = True
        if not internal_mode:
            self._prepare_position_preserve_info(
                keep_indices_np=idx,
                num_tokens=int(num_tokens),
                metrics=metrics,
            )

        metrics.selector_name = selection_meta.get("selector_name", "select_acgtp_v2_fast")
        metrics.selector_function_name = selection_meta.get("selector_function_name", "select_acgtp_v2_fast")
        metrics.selection_strategy_name = selection_meta.get("selection_strategy_name", self.config.strategy)
        metrics.selection_stage_name = selection_meta.get("selection_stage_name", "acgtp_v2_fast_runtime")
        metrics.selector_success = True
        metrics.requested_pruning_strategy = self.config.strategy
        metrics.effective_pruning_strategy = self.config.strategy
        metrics.keep_indices_source = metrics.selector_function_name
        metrics.keep_ratio_source = metrics.keep_ratio_source or "acgtp_dynamic_controller"
        metrics.keep_ratio_requested = selection_meta.get("requested_keep_ratio", metrics.requested_keep_ratio)

        metrics.fallback_used = bool(selection_meta.get("fallback_used", False))
        metrics.fallback_reason = selection_meta.get("fallback_reason")
        metrics.acgtp_v2 = True
        metrics.acgtp_v1 = False
        metrics.acgtp_selector_version = selection_meta.get("acgtp_selector_version")
        metrics.acgtp_quota_policy = selection_meta.get("acgtp_quota_policy")
        metrics.acgtp_fill_policy = selection_meta.get("acgtp_fill_policy")
        metrics.acgtp_fallback_used = bool(selection_meta.get("acgtp_fallback_used", False))
        metrics.acgtp_fallback_reason = selection_meta.get("acgtp_fallback_reason")
        metrics.acgtp_branch_accounting_valid = selection_meta.get("acgtp_branch_accounting_valid")
        metrics.acgtp_branch_sum = selection_meta.get("acgtp_branch_sum")
        metrics.acgtp_branch_sum_error = selection_meta.get("acgtp_branch_sum_error")
        metrics.branch_accounting_valid = selection_meta.get("branch_accounting_valid")
        metrics.branch_sum_equals_kept = selection_meta.get("branch_sum_equals_kept")
        metrics.selected_by_scene_layout_count = selection_meta.get("selected_by_scene_layout_count")
        metrics.selected_by_depth_structure_count = selection_meta.get("selected_by_depth_structure_count")
        metrics.selected_by_contact_ring_count = selection_meta.get("selected_by_contact_ring_count")
        metrics.selected_by_motion_corridor_count = selection_meta.get("selected_by_motion_corridor_count")
        metrics.selected_by_constrained_fill_count = selection_meta.get("selected_by_constrained_fill_count")
        metrics.selected_by_acgtp_fallback_count = selection_meta.get("selected_by_acgtp_fallback_count")
        metrics.acgtp_actual_keep_ratio = actual_ratio
        metrics.acgtp_final_kept = kept
        metrics.acgtp_expected_kept = selection_meta.get("expected_kept", kept)
        metrics.acgtp_action_constraint_ms = robot_metrics.get("acgtp_action_constraint_ms")
        metrics.acgtp_motion_corridor_valid = robot_metrics.get("acgtp_motion_corridor_valid")
        metrics.acgtp_scene_fill_candidate_count = selection_meta.get("acgtp_scene_fill_candidate_count")
        metrics.acgtp_scene_fill_candidate_ratio = selection_meta.get("acgtp_scene_fill_candidate_ratio")
        metrics.acgtp_coverage_fill_candidate_count = selection_meta.get("acgtp_coverage_fill_candidate_count")
        metrics.acgtp_coverage_fill_candidate_ratio = selection_meta.get("acgtp_coverage_fill_candidate_ratio")
        metrics.acgtp_latency_plan_cache_enabled = selection_meta.get("acgtp_latency_plan_cache_enabled")
        metrics.acgtp_latency_plan_cache_hit = selection_meta.get("acgtp_latency_plan_cache_hit")
        metrics.acgtp_latency_plan_cache_reason = selection_meta.get("acgtp_latency_plan_cache_reason")
        metrics.acgtp_latency_plan_cache_age = selection_meta.get("acgtp_latency_plan_cache_age")
        metrics.acgtp_latency_plan_cache_depth_delta = selection_meta.get("acgtp_latency_plan_cache_depth_delta")
        metrics.acgtp_latency_plan_cache_gripper_delta = selection_meta.get("acgtp_latency_plan_cache_gripper_delta")
        metrics.acgtp_latency_plan_cache_lookup_ms = selection_meta.get("acgtp_latency_plan_cache_lookup_ms")
        metrics.acgtp_latency_plan_cache_keep_count = selection_meta.get("acgtp_latency_plan_cache_keep_count")

        if self._is_audit_runtime():
            for key in (
                "acgtp_dynamic_enabled",
                "acgtp_dynamic_phase",
                "acgtp_dynamic_phase_schedule",
                "acgtp_dynamic_candidate_phase",
                "acgtp_dynamic_previous_phase",
                "acgtp_dynamic_hysteresis_state",
                "acgtp_dynamic_risk",
                "acgtp_dynamic_confidence",
                "acgtp_dynamic_keep_ratio",
                "acgtp_dynamic_keep_k",
                "acgtp_dynamic_base_keep_ratio",
                "acgtp_dynamic_raw_keep_ratio",
                "acgtp_dynamic_phase_min_keep_ratio",
                "acgtp_dynamic_phase_max_keep_ratio",
                "acgtp_dynamic_lock_strength",
                "acgtp_dynamic_uncertainty_boost",
                "acgtp_dynamic_risk_boost",
                "acgtp_dynamic_prune_gain",
                "acgtp_dynamic_keep_reason",
                "acgtp_dynamic_layout_motion_alignment",
                "acgtp_dynamic_binary_alignment",
                "acgtp_dynamic_contact_phase_gate",
                "acgtp_dynamic_contact_peak",
                "acgtp_dynamic_contact_mean",
                "acgtp_dynamic_contact_ratio",
                "acgtp_dynamic_motion_peak",
                "acgtp_dynamic_motion_mean",
                "acgtp_dynamic_motion_ratio",
                "acgtp_dynamic_physical_ratio",
                "acgtp_dynamic_high_contact",
                "acgtp_dynamic_high_contact_coverage",
                "acgtp_dynamic_high_contact_legacy",
                "acgtp_dynamic_shadow_contact_guard",
                "acgtp_dynamic_high_motion",
                "acgtp_dynamic_strong_layout",
                "acgtp_dynamic_action_peak",
                "acgtp_dynamic_action_mean",
                "acgtp_dynamic_depth_valid_ratio",
                "acgtp_dynamic_fill_candidate_count",
                "acgtp_dynamic_fill_candidate_ratio",
                "acgtp_dynamic_candidate_gap_count",
                "acgtp_dynamic_candidate_gap_ratio",
                "acgtp_dynamic_candidate_clamped",
                "acgtp_dynamic_scene_weight",
                "acgtp_dynamic_depth_weight",
                "acgtp_dynamic_contact_weight",
                "acgtp_dynamic_motion_weight",
                "acgtp_dynamic_hard_protect_ratio",
                "acgtp_dynamic_branch_floor_enabled",
                "acgtp_dynamic_min_scene_tokens",
                "acgtp_dynamic_min_depth_tokens",
                "acgtp_dynamic_min_contact_tokens",
                "acgtp_dynamic_min_motion_tokens",
                "acgtp_dynamic_fill_cap_ratio",
                "acgtp_dynamic_fill_cap_tokens",
                "acgtp_dynamic_budget_vector",
                "acgtp_min_scene_tokens",
                "acgtp_min_depth_tokens",
                "acgtp_min_contact_tokens",
                "acgtp_min_motion_tokens",
                "acgtp_constrained_fill_cap_tokens",
                "acgtp_constrained_fill_cap_applied",
                "acgtp_static_scene_cache_enabled",
                "acgtp_static_scene_cache_hit",
                "acgtp_static_scene_cache_reason",
                "acgtp_static_scene_cache_depth_delta",
                "acgtp_static_scene_cache_valid_iou",
                "acgtp_static_scene_cache_age",
                "acgtp_ablate_enabled",
                "acgtp_ablate_branches",
                "acgtp_ablate_scene",
                "acgtp_ablate_depth",
                "acgtp_ablate_contact",
                "acgtp_ablate_motion",
                "acgtp_ablate_fill",
                "attention_backend",
                "attention_source",
                "attention_available",
                "attention_confidence",
                "attention_quota_released",
                "selected_by_attention_count",
                "attention_only_token_count",
                "attention_selected_by_final_count",
                "attention_top_count",
                "safe_drop_candidate_count",
                "high_attention_low_geometry_count",
                "high_geometry_low_attention_count",
            ):
                if hasattr(metrics, key):
                    setattr(metrics, key, selection_meta.get(key))
        metrics.dynamic_enabled = bool(selection_meta.get("acgtp_dynamic_enabled", metrics.dynamic_enabled))
        metrics.dynamic_phase = selection_meta.get("acgtp_dynamic_phase", metrics.dynamic_phase)
        metrics.dynamic_keep_ratio = selection_meta.get("acgtp_dynamic_keep_ratio", metrics.dynamic_keep_ratio)
        metrics.dynamic_keep_k = selection_meta.get("acgtp_dynamic_keep_k", metrics.dynamic_keep_k)
        metrics.dynamic_keep_reason = selection_meta.get("acgtp_dynamic_keep_reason", metrics.dynamic_keep_reason)
        metrics.geo_risk_score = selection_meta.get("acgtp_dynamic_risk", metrics.geo_risk_score)
        metrics.geo_risk_level = selection_meta.get("acgtp_dynamic_phase", metrics.geo_risk_level)

        if bool(getattr(self.config, "acgtp_history_enabled", False)) and not metrics.fallback_used:
            hist_update = self._acgtp_history.update_after_selection(
                keep_indices=idx,
                num_tokens=num_tokens,
                dynamic_decision=selection_meta,
                gripper_pos=robot_metrics.get("gripper_pos"),
            )
            metrics.acgtp_history_enabled = True
            metrics.acgtp_history_length_after_update = hist_update.get("acgtp_history_length_after_update")
            metrics.acgtp_history_keep_mask_iou = hist_update.get("acgtp_history_keep_mask_iou")
            metrics.acgtp_history_phase_switch = hist_update.get("acgtp_history_phase_switch")
            metrics.acgtp_history_force_conservative_next = hist_update.get("acgtp_history_force_conservative_next")
            metrics.acgtp_history_force_conservative_reason = hist_update.get("acgtp_history_force_conservative_reason")
        else:
            metrics.acgtp_history_enabled = False
            metrics.score_ema_enabled = False

        metrics.pruning_result = {
            "num_tokens_before": int(num_tokens),
            "num_tokens_after": kept,
            "actual_keep_ratio": actual_ratio,
            "method": self.config.strategy,
            "effective_strategy": self.config.strategy,
            "fast_runtime": True,
            "compression_backend": metrics.compression_backend,
            "projector_pruning_applied": metrics.projector_pruning_applied,
            "internal_pruning_requested": metrics.internal_pruning_requested,
            "internal_pruning_plan_ready": metrics.internal_pruning_plan_ready,
            "selector_function_name": metrics.selector_function_name,
            "fallback_used": metrics.fallback_used,
            "fallback_reason": metrics.fallback_reason,
            "dynamic_phase": metrics.dynamic_phase,
            "dynamic_keep_k": metrics.dynamic_keep_k,
            "geo_risk_score": metrics.geo_risk_score,
            "keep_ratio_source": metrics.keep_ratio_source,
            "requested_keep_ratio": metrics.requested_keep_ratio,
        }
        self._last_selector_exception = None
        self._last_selector_name = None
        return pruned, metrics

    @staticmethod
    def _norm01_np(arr: Optional[np.ndarray], n: int, valid: np.ndarray) -> np.ndarray:
        out = np.zeros(n, dtype=np.float32)
        if arr is None:
            return out
        try:
            flat = np.asarray(arr, dtype=np.float32).reshape(-1)
        except Exception:
            return out
        if flat.shape[0] != n:
            return out
        finite = valid & np.isfinite(flat)
        if not np.any(finite):
            return out
        vals = flat[finite]
        lo = float(np.min(vals))
        hi = float(np.max(vals))
        if hi - lo > 1e-8:
            out[finite] = (flat[finite] - lo) / (hi - lo)
        else:
            out[finite] = np.clip(flat[finite], 0.0, 1.0)
        out[~valid] = 0.0
        return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

    @staticmethod
    def _topk_np(score: np.ndarray, k: int, eligible: np.ndarray) -> np.ndarray:
        if k <= 0:
            return np.asarray([], dtype=np.int64)
        arr = np.asarray(score, dtype=np.float32).reshape(-1)
        mask = np.asarray(eligible, dtype=bool).reshape(-1) & np.isfinite(arr) & (arr > 0.0)
        cand = np.flatnonzero(mask)
        if cand.size == 0:
            return np.asarray([], dtype=np.int64)
        if cand.size > k:
            cand = cand[np.argpartition(-arr[cand], k - 1)[:k]]
        order = np.lexsort((cand, -arr[cand]))
        return cand[order].astype(np.int64, copy=False)

    def _attention_history_ema(self, n: int) -> Optional[np.ndarray]:
        if not self._acgtp_attention_history:
            return None
        decay = float(getattr(self.config, "acgtp_attention_history_decay", 0.8))
        acc = np.zeros(n, dtype=np.float32)
        weight_sum = 0.0
        for i, item in enumerate(reversed(self._acgtp_attention_history)):
            arr = np.asarray(item, dtype=np.float32).reshape(-1)
            if arr.shape[0] != n:
                continue
            w = float(decay ** i)
            acc += w * arr
            weight_sum += w
        if weight_sum <= 1e-8:
            return None
        return np.clip(acc / weight_sum, 0.0, 1.0).astype(np.float32, copy=False)

    def _redundancy_filter_attention_candidates(
        self,
        *,
        visual_tokens: torch.Tensor,
        candidate_idx: np.ndarray,
        relevance_score: np.ndarray,
        budget: int,
    ) -> np.ndarray:
        idx = np.asarray(candidate_idx, dtype=np.int64).reshape(-1)
        if idx.size <= budget or budget <= 0 or not bool(getattr(self.config, "acgtp_attention_redundancy_filter_enabled", True)):
            return idx[:max(0, budget)]
        try:
            score = np.asarray(relevance_score, dtype=np.float32).reshape(-1)
            if idx.size > 64:
                order = np.lexsort((idx, -score[idx]))
                idx = idx[order[:64]]
            token_idx = torch.as_tensor(idx, dtype=torch.long, device=visual_tokens.device)
            feats = visual_tokens.detach()[0, token_idx].to(dtype=torch.float32).cpu().numpy()
            feats = feats / np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-6)
            sim = np.clip(feats @ feats.T, -1.0, 1.0)
            dist = 1.0 - sim
            rel_np = score[idx]
            rel_np = rel_np - float(np.min(rel_np))
            rel_np = rel_np / max(float(np.max(rel_np)), 1e-6)
            redundancy_weight = float(getattr(self.config, "acgtp_attention_redundancy_weight", 0.35))
            redundancy_weight = max(0.0, min(1.0, redundancy_weight))
            selected: List[int] = [int(np.argmax(rel_np))]
            blocked = np.zeros(idx.size, dtype=bool)
            blocked[selected[0]] = True
            while len(selected) < budget:
                min_dist = np.min(dist[:, np.asarray(selected, dtype=np.int64)], axis=1)
                combined = (1.0 - redundancy_weight) * rel_np + redundancy_weight * min_dist
                combined[blocked] = -1.0
                nxt = int(np.argmax(combined))
                if bool(blocked[nxt]):
                    break
                selected.append(nxt)
                blocked[nxt] = True
            return idx[np.asarray(selected, dtype=np.int64)]
        except Exception:
            score = np.asarray(relevance_score, dtype=np.float32).reshape(-1)
            order = np.lexsort((idx, -score[idx]))
            return idx[order[:budget]]

    def _build_acgtp_attention_guide(
        self,
        *,
        visual_tokens: torch.Tensor,
        scene_scores: Optional[np.ndarray],
        depth_scores: Optional[np.ndarray],
        contact_scores: Optional[np.ndarray],
        motion_scores: Optional[np.ndarray],
        action_scores: Optional[np.ndarray],
        valid_mask: np.ndarray,
        keep_count: int,
    ) -> Dict[str, Any]:
        n = int(valid_mask.shape[0])
        disabled = {
            "enabled": False,
            "score": None,
            "mask": None,
            "meta": {
                "attention_backend": "none",
                "attention_source": "none",
                "attention_available": False,
                "attention_confidence": 0.0,
                "attention_quota_released": True,
                "attention_top_count": 0,
            },
        }
        if not bool(getattr(self.config, "acgtp_attention_guidance_enabled", False)) or n <= 0:
            return disabled

        valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
        scene = self._norm01_np(scene_scores, n, valid)
        depth = self._norm01_np(depth_scores, n, valid)
        contact = self._norm01_np(contact_scores, n, valid)
        motion = self._norm01_np(motion_scores, n, valid)
        action = self._norm01_np(action_scores, n, valid)

        source = str(getattr(self.config, "acgtp_attention_guidance_source", "action_proxy")).strip().lower()
        geometry_proxy = 0.30 * scene + 0.35 * depth + 0.15 * contact + 0.20 * motion
        if source == "geometry_proxy" or float(np.max(action[valid])) <= 1e-8:
            current = geometry_proxy
            source = "geometry_proxy"
        else:
            current = 0.70 * action + 0.30 * geometry_proxy
            source = "action_proxy"
        current = np.clip(current, 0.0, 1.0).astype(np.float32, copy=False)
        historical = self._attention_history_ema(n)
        budget_ratio = float(getattr(self.config, "acgtp_attention_budget_ratio", 0.12))
        budget = max(1, min(int(keep_count), int(round(float(keep_count) * budget_ratio))))

        current_top = self._topk_np(current, budget, valid)
        if historical is not None:
            historical_top = self._topk_np(historical, budget, valid)
            combined = np.maximum(current, historical)
            candidate_idx = np.unique(np.concatenate([current_top, historical_top])).astype(np.int64)
            guide_source = f"{source}+historical_union"
        else:
            combined = current
            candidate_idx = current_top
            guide_source = f"{source}+current"

        if candidate_idx.size > budget:
            candidate_idx = self._redundancy_filter_attention_candidates(
                visual_tokens=visual_tokens,
                candidate_idx=candidate_idx,
                relevance_score=combined,
                budget=budget,
            )

        mask = np.zeros(n, dtype=np.float32)
        mask[candidate_idx] = 1.0
        self._acgtp_attention_history.append(current.copy())
        confidence = float(np.mean(combined[candidate_idx])) if candidate_idx.size else 0.0
        return {
            "enabled": True,
            "score": combined.astype(np.float32, copy=False),
            "mask": mask,
            "meta": {
                "attention_backend": "proxy",
                "attention_source": guide_source,
                "attention_available": bool(candidate_idx.size > 0),
                "attention_confidence": confidence,
                "attention_quota_released": not bool(candidate_idx.size > 0),
                "attention_top_count": int(candidate_idx.size),
                "acgtp_attention_history_available": historical is not None,
                "acgtp_attention_current_top_count": int(current_top.size),
                "acgtp_attention_union_candidate_count": int(candidate_idx.size),
                "acgtp_attention_redundancy_filter_enabled": bool(getattr(self.config, "acgtp_attention_redundancy_filter_enabled", True)),
            },
        }

    def _build_internal_geometry_payload(
        self,
        *,
        scene_scores: np.ndarray,
        depth_scores: np.ndarray,
        contact_scores: np.ndarray,
        motion_scores: np.ndarray,
        action_scores: np.ndarray,
        valid_mask: np.ndarray,
        fill_mask: np.ndarray,
        keep_count: int,
        num_tokens: int,
        motion_valid: bool,
        dyn_decision: Dict[str, Any],
        hard_ratio: float,
        w_scene: float,
        w_depth: float,
        w_contact: float,
        w_motion: float,
    ) -> Dict[str, Any]:
        """Package geometry prior for LLM-internal fusion.

        The projector hook keeps doing cheap RGB-D/robot geometry extraction,
        but in internal mode the final keep decision is made after shallow LLM
        attention is available.
        """

        requested_ratio = float(keep_count) / float(max(1, int(num_tokens)))
        mode = str(getattr(self.config, "acgtp_internal_selection_mode", "geo_guarded") or "geo_guarded").strip().lower()
        quota_config = {
            "selection_mode": mode,
            "attention_enabled": bool(getattr(self.config, "acgtp_internal_attention_enabled", True)),
            "semantic_attention_ratio": float(getattr(self.config, "acgtp_internal_attention_budget_ratio", getattr(self.config, "acgtp_attention_budget_ratio", 0.12))),
            "historical_attention_ratio": float(getattr(self.config, "acgtp_internal_history_budget_ratio", 0.15)),
            "attention_requires_geometry_alignment": bool(getattr(self.config, "acgtp_attention_requires_geometry_alignment", True)),
            "hard_protect_ratio": float(hard_ratio),
            "functional_quota_enabled": bool(getattr(self.config, "acgtp_internal_functional_quota_enabled", True)),
            "layout_quota_ratio": float(getattr(self.config, "acgtp_internal_layout_quota_ratio", 0.30)),
            "contact_quota_ratio": float(getattr(self.config, "acgtp_internal_contact_quota_ratio", 0.20)),
            "motion_quota_ratio": float(getattr(self.config, "acgtp_internal_motion_quota_ratio", 0.15)),
            "semantic_quota_ratio": float(getattr(self.config, "acgtp_internal_semantic_quota_ratio", 0.12)),
            "action_quota_ratio": float(getattr(self.config, "acgtp_internal_action_quota_ratio", 0.08)),
            "fill_quota_ratio": float(getattr(self.config, "acgtp_internal_fill_quota_ratio", 0.15)),
            "w_scene": float(w_scene),
            "w_depth": float(w_depth),
            "w_contact": float(w_contact),
            "w_motion": float(w_motion),
            "history_length": int(getattr(self.config, "acgtp_attention_history_length", 3)),
            "risk_adaptive_enabled": bool(getattr(self.config, "acgtp_internal_risk_adaptive_enabled", False)),
            "high_risk_keep_ratio": float(getattr(self.config, "acgtp_internal_high_risk_keep_ratio", 0.85)),
            "medium_risk_keep_ratio": float(getattr(self.config, "acgtp_internal_medium_risk_keep_ratio", 0.55)),
            "low_risk_keep_ratio": float(getattr(self.config, "acgtp_internal_low_risk_keep_ratio", 0.40)),
            "risk_coverage_weight": float(getattr(self.config, "acgtp_internal_risk_coverage_weight", 3.0)),
            "risk_mean_weight": float(getattr(self.config, "acgtp_internal_risk_mean_weight", 1.5)),
            "risk_peak_weight": float(getattr(self.config, "acgtp_internal_risk_peak_weight", 0.15)),
            "risk_physical_weight": float(getattr(self.config, "acgtp_internal_risk_physical_weight", 0.85)),
            "risk_depth_weight": float(getattr(self.config, "acgtp_internal_risk_depth_weight", 0.15)),
            "risk_disagreement_gate": float(getattr(self.config, "acgtp_internal_risk_disagreement_gate", 0.45)),
            "risk_disagreement_max_bonus": float(getattr(self.config, "acgtp_internal_risk_disagreement_max_bonus", 0.10)),
            "risk_high_threshold": float(getattr(self.config, "acgtp_internal_risk_high_threshold", 0.65)),
            "risk_medium_threshold": float(getattr(self.config, "acgtp_internal_risk_medium_threshold", 0.35)),
            "capture_decode_attention": bool(getattr(self.config, "acgtp_internal_capture_decode_attention", False)),
        }

        # Explicit physical hard-protect mask + soft ranking score (final-design
        # P_geo / geo_soft_score). The hard mask marks tokens that geometrically
        # constrain the robot's future action: high contact-ring, high future
        # action-constraint, strong depth/object boundary, or motion corridor.
        # The internal backend must keep these (raising budget if needed) and the
        # ordinary redundancy filter must not remove them.
        valid_arr = np.asarray(valid_mask, dtype=bool).reshape(-1)
        n = int(valid_arr.shape[0]) if valid_arr.size else int(num_tokens)

        def _norm(arr: np.ndarray) -> np.ndarray:
            out = self._norm01_np(arr, n, valid_arr)
            return np.asarray(out, dtype=np.float32).reshape(-1)

        contact_n = _norm(contact_scores)
        action_n = _norm(action_scores)
        depth_n = _norm(depth_scores)
        scene_n = _norm(scene_scores)
        motion_n = _norm(motion_scores) if motion_valid else np.zeros(n, dtype=np.float32)
        # Soft ranking score: action-constraint led, geometry-supported. Only for
        # fill/ordering, never the sole life-or-death signal.
        geo_soft_score = np.clip(
            np.maximum.reduce([
                action_n,
                0.85 * contact_n,
                0.70 * depth_n,
                0.60 * scene_n,
                0.55 * motion_n,
            ]),
            0.0,
            1.0,
        ).astype(np.float32)
        # Execution-function branches for the internal allocator. These keep the
        # paper-level roles explicit instead of forcing all geometry into one
        # global salience score.
        layout_score = np.clip(np.maximum(scene_n, depth_n), 0.0, 1.0).astype(np.float32)
        contact_score = np.clip(np.maximum(contact_n, action_n), 0.0, 1.0).astype(np.float32)
        motion_score = np.clip(motion_n, 0.0, 1.0).astype(np.float32)
        # Hard protect: physically action-constraining tokens above a high
        # quantile of the strongest constraint evidence (contact/action/motion),
        # plus object/support boundaries (depth) that coincide with constraint.
        constraint_evidence = np.maximum.reduce([contact_n, action_n, motion_n]).astype(np.float32)
        q = float(getattr(self.config, "acgtp_internal_geo_protect_quantile", 0.80))
        geo_protect_mask = np.zeros(n, dtype=bool)
        if valid_arr.any():
            ev_valid = constraint_evidence[valid_arr]
            ev_valid = ev_valid[np.isfinite(ev_valid)]
            if ev_valid.size and float(np.max(ev_valid)) > 1e-6:
                thr = float(np.quantile(ev_valid, q))
                thr = max(thr, 1e-6)
                geo_protect_mask = valid_arr & (constraint_evidence >= thr)
        # Cap the hard set so it can never freeze the whole budget; keep the
        # strongest-evidence tokens if the quantile selected too many.
        max_ratio = float(getattr(self.config, "acgtp_internal_geo_protect_max_ratio", 0.50))
        max_hard = int(max(1, round(float(n) * max_ratio)))
        if int(geo_protect_mask.sum()) > max_hard:
            prot_idx = np.flatnonzero(geo_protect_mask)
            order = prot_idx[np.argsort(-constraint_evidence[prot_idx], kind="stable")]
            geo_protect_mask = np.zeros(n, dtype=bool)
            geo_protect_mask[order[:max_hard]] = True

        return {
            "scene_scores": np.asarray(scene_scores, dtype=np.float32).reshape(-1).copy(),
            "depth_scores": np.asarray(depth_scores, dtype=np.float32).reshape(-1).copy(),
            "contact_scores": np.asarray(contact_scores, dtype=np.float32).reshape(-1).copy(),
            "motion_scores": np.asarray(motion_scores, dtype=np.float32).reshape(-1).copy(),
            "action_constraint_scores": np.asarray(action_scores, dtype=np.float32).reshape(-1).copy(),
            "geo_protect_mask": geo_protect_mask.copy(),
            "geo_soft_score": geo_soft_score.copy(),
            "layout_score": layout_score.copy(),
            "contact_score": contact_score.copy(),
            "motion_score": motion_score.copy(),
            "valid_mask": np.asarray(valid_mask, dtype=bool).reshape(-1).copy(),
            "constrained_fill_mask": np.asarray(fill_mask, dtype=bool).reshape(-1).copy(),
            "motion_corridor_valid": bool(motion_valid),
            "target_keep_ratio": requested_ratio,
            "target_keep_k": int(keep_count),
            "quota_config": quota_config,
            "dynamic_decision": {k: v for k, v in dict(dyn_decision or {}).items() if not str(k).endswith("_scores")},
        }

    def _run_acgtp_fast_runtime(
        self,
        visual_tokens: torch.Tensor,
        metrics: HookMetrics,
        *,
        keep_count: int,
        num_tokens: int,
    ) -> Tuple[torch.Tensor, HookMetrics]:
        """Lean ACGTP-v2 runtime used for real rollout timing.

        The legacy path below remains the diagnostic/audit path. This hot path
        avoids torch-based rule-score construction, selected-token attribution,
        overlay metadata, and full branch audit recomputation.
        """
        latest = self.geometry_recorder.get_latest() if self.geometry_recorder is not None else None

        def _fallback(reason: str) -> Tuple[torch.Tensor, HookMetrics]:
            self._reset_latency_plan_cache()
            idx = np.arange(int(num_tokens), dtype=np.int64)
            meta = {
                "selector_name": "acgtp_fast_input_fallback",
                "selector_function_name": "acgtp_fast_input_fallback",
                "selection_strategy_name": "robot_geo_acgtp_v2",
                "selection_stage_name": "acgtp_v2_fast_input_fallback",
                "fallback_used": True,
                "fallback_reason": reason,
                "acgtp_fallback_used": True,
                "acgtp_fallback_reason": reason,
                "acgtp_branch_accounting_valid": True,
                "acgtp_branch_sum": int(num_tokens),
                "acgtp_branch_sum_error": 0,
                "selected_by_scene_layout_count": 0,
                "selected_by_depth_structure_count": 0,
                "selected_by_contact_ring_count": 0,
                "selected_by_motion_corridor_count": 0,
                "selected_by_constrained_fill_count": 0,
                "selected_by_acgtp_fallback_count": int(num_tokens),
                "expected_kept": int(num_tokens),
            }
            metrics.fallback_used = True
            metrics.fallback_reason = reason
            metrics.keep_ratio_source = "fallback"
            return self._finalize_acgtp_fast_runtime(
                visual_tokens=visual_tokens,
                metrics=metrics,
                keep_indices_np=idx,
                selection_meta=meta,
                robot_metrics={"acgtp_motion_corridor_valid": False},
                num_tokens=num_tokens,
            )

        if latest is None:
            return _fallback("missing_geometry")
        if latest.depth is None:
            return _fallback("missing_depth")

        gripper_pos, gripper_key = extract_gripper_position(latest)
        T_robot_cam, transform_key = extract_robot_camera_transform(latest)
        if gripper_pos is None:
            return _fallback("missing_robot_state")
        if latest.camera_intrinsics is None or T_robot_cam is None:
            return _fallback("missing_camera")

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
        grip = np.asarray(gripper_pos, dtype=np.float32).reshape(3)
        depth_source_key = getattr(latest, "depth_metadata", {}) and latest.depth_metadata.get("source_key")
        plan_cache_probe = self._latency_depth_probe(depth) if self._latency_plan_cache_enabled() else None
        cached_plan, plan_cache_meta = self._lookup_latency_plan_cache(
            depth_probe=plan_cache_probe,
            grip=grip,
            num_tokens=num_tokens,
            keep_count=keep_count,
            token_grid_shape=token_grid_shape,
        )
        if cached_plan is not None and bool(plan_cache_meta.get("acgtp_latency_plan_cache_hit")):
            selection_meta = self._clone_latency_cache_value(cached_plan.get("selection_meta") or {})
            selection_meta.update(plan_cache_meta)
            selection_meta["selector_name"] = "select_acgtp_v2_fast_cached"
            selection_meta["selector_function_name"] = "select_acgtp_v2_fast_cached"
            selection_meta["selection_stage_name"] = "acgtp_v2_latency_plan_cache"
            selection_meta["selection_time_ms"] = 0.0

            metrics.timing.token_mapping_ms = 0.0
            metrics.timing.depth_sampling_ms = 0.0
            metrics.timing.depth_sample_ms = 0.0
            metrics.timing.score_compute_ms = 0.0
            metrics.timing.depth_edge_score_ms = 0.0
            metrics.timing.score_fusion_ms = 0.0
            metrics.valid_token_ratio = cached_plan.get("valid_ratio")
            metrics.depth_valid_ratio = cached_plan.get("valid_ratio")
            metrics.depth_source_key = cached_plan.get("depth_source_key")
            metrics.geometry_available = True
            metrics.robot_state_available = True
            metrics.camera_available = True
            metrics.acgtp_scene_layout_ms = 0.0
            metrics.acgtp_contact_ring_ms = 0.0
            metrics.acgtp_motion_corridor_ms = 0.0
            metrics.acgtp_action_constraint_ms = 0.0
            metrics.acgtp_static_scene_cache_enabled = self.config.acgtp_static_scene_cache_enabled
            metrics.acgtp_static_scene_cache_hit = None
            metrics.acgtp_static_scene_cache_reason = "skipped_by_latency_plan_cache"
            self._update_previous_gripper_pos(latest, grip)

            cached_robot_metrics = self._clone_latency_cache_value(cached_plan.get("robot_metrics") or {})
            cached_robot_metrics["gripper_pos"] = grip
            cached_robot_metrics["acgtp_action_constraint_ms"] = 0.0
            return self._finalize_acgtp_fast_runtime(
                visual_tokens=visual_tokens,
                metrics=metrics,
                keep_indices_np=np.asarray(cached_plan.get("keep_indices"), dtype=np.int64),
                selection_meta=selection_meta,
                robot_metrics=cached_robot_metrics,
                num_tokens=num_tokens,
                geometry_payload=self._clone_latency_cache_value(cached_plan.get("geometry_payload")),
            )

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
        sample_start = time.perf_counter()
        token_depth = self._cache.sample_depth(depth, cache, check_zbuffer=self._is_audit_runtime())
        depth_sampling_ms = (time.perf_counter() - sample_start) * 1000.0
        valid_mask = compute_valid_depth_mask(
            token_depth,
            min_depth=self.config.min_depth,
            max_depth=self.config.max_depth,
        )
        valid_ratio = float(np.mean(valid_mask)) if valid_mask.size else 0.0
        token_mapping_ms = (time.perf_counter() - mapping_start) * 1000.0
        metrics.timing.token_mapping_ms = token_mapping_ms
        metrics.timing.depth_sampling_ms = depth_sampling_ms
        metrics.timing.depth_sample_ms = depth_sampling_ms
        metrics.valid_token_ratio = valid_ratio
        metrics.depth_valid_ratio = valid_ratio
        metrics.depth_source_key = depth_source_key

        if valid_ratio < self.config.min_valid_token_ratio:
            return _fallback("invalid_depth_ratio")

        score_start = time.perf_counter()
        token_depth = np.nan_to_num(token_depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        p_robot = project_tokens_to_robot(token_depth, cache["rays"], T_robot_cam, valid_mask)
        distances = np.linalg.norm(p_robot - grip[None, :], axis=1)
        distances = np.where(np.isfinite(distances), distances, np.inf).astype(np.float32)
        sigma = max(float(self.config.sigma_near), 1e-6)
        near_scores = np.exp(-(distances * distances) / (2.0 * sigma * sigma)).astype(np.float32)
        near_scores[~valid_mask] = 0.0
        cache_hit, scene_cache_meta = self._static_scene_cache.lookup(
            token_depth=token_depth,
            valid_mask=valid_mask,
            token_grid_shape=token_grid_shape,
            num_tokens=num_tokens,
        )
        if cache_hit:
            edge_scores = np.asarray(scene_cache_meta["edge_scores"], dtype=np.float32).reshape(-1)
            scene_result = dict(scene_cache_meta["scene_result"])
            scene_ms = 0.0
        else:
            edge_scores = compute_depth_edge_scores(
                token_depth,
                valid_mask,
                token_grid_shape=token_grid_shape,
                num_visual_tokens=num_tokens,
            )

            scene_start = time.perf_counter()
            scene_result = compute_scene_layout_scores(
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
            scene_ms = (time.perf_counter() - scene_start) * 1000.0
            self._static_scene_cache.store(
                token_depth=token_depth,
                valid_mask=valid_mask,
                token_grid_shape=token_grid_shape,
                num_tokens=num_tokens,
                edge_scores=edge_scores,
                scene_result=scene_result,
            )
        scene_scores = scene_result["scene_layout_scores"]
        fill_mask = scene_result["scene_fill_candidates"]

        contact_start = time.perf_counter()
        gripper_pixel = self._project_gripper_to_pixel(grip, latest, T_robot_cam)
        contact_result = compute_contact_ring_scores(
            token_u=cache.get("u"),
            token_v=cache.get("v"),
            gripper_pixel=gripper_pixel,
            near_scores=near_scores,
            self_core_radius_px=float(self.config.acgtp_self_core_radius_px),
            contact_ring_inner_px=float(self.config.acgtp_contact_ring_inner_px),
            contact_ring_outer_px=float(self.config.acgtp_contact_ring_outer_px),
            contact_requires_edge_or_object=bool(self.config.acgtp_contact_requires_edge_or_object),
            depth_edge_scores=edge_scores,
        )
        contact_scores = contact_result["contact_ring_scores"]
        self_core = contact_result["robot_self_core_mask"]
        contact_ms = (time.perf_counter() - contact_start) * 1000.0

        motion_start = time.perf_counter()
        prev_gripper_pos = self._get_previous_gripper_pos(latest)
        if self._motion_buffer is None and gripper_pos is not None:
            self._motion_buffer = create_motion_buffer(
                maxlen=5,
                ema_alpha=float(self.config.acgtp_motion_ema_alpha),
            )
        motion_result = compute_motion_corridor_scores(
            points_robot=np.asarray(p_robot, dtype=np.float64),
            gripper_pos=np.asarray(grip, dtype=np.float64),
            prev_gripper_pos=np.asarray(prev_gripper_pos, dtype=np.float64) if prev_gripper_pos is not None else None,
            depth_edge_scores=edge_scores,
            motion_buffer=self._motion_buffer,
            corridor_length_m=float(self.config.acgtp_motion_corridor_length_m),
            corridor_sigma_m=float(self.config.acgtp_motion_sigma_m),
            min_motion_norm=1e-4,
            ema_alpha=float(self.config.acgtp_motion_ema_alpha),
        )
        motion_scores = motion_result["motion_corridor_scores"]
        motion_valid = bool(motion_result["motion_corridor_valid"])
        motion_ms = (time.perf_counter() - motion_start) * 1000.0

        ablated = self._parse_acgtp_ablate_branches(getattr(self.config, "acgtp_ablate_branches", ""))
        ablation_meta: Dict[str, Any] = {
            "acgtp_ablate_enabled": bool(ablated),
            "acgtp_ablate_branches": ",".join(sorted(ablated)),
            "acgtp_ablate_scene": "scene" in ablated,
            "acgtp_ablate_depth": "depth" in ablated,
            "acgtp_ablate_contact": "contact" in ablated,
            "acgtp_ablate_motion": "motion" in ablated,
            "acgtp_ablate_fill": "fill" in ablated,
        }
        if ablated:
            zeros = np.zeros(num_tokens, dtype=np.float32)
            if "scene" in ablated:
                scene_result = dict(scene_result)
                for key in (
                    "scene_layout_scores",
                    "support_plane_scores",
                    "support_plane_candidate_scores",
                    "object_component_scores",
                    "boundary_scores",
                    "scene_fill_candidates",
                ):
                    value = scene_result.get(key)
                    if value is not None and np.asarray(value).reshape(-1).size == num_tokens:
                        scene_result[key] = np.zeros_like(np.asarray(value, dtype=np.float32).reshape(-1))
                scene_scores = zeros.copy()
                fill_mask = zeros.copy()
            if "depth" in ablated:
                edge_scores = zeros.copy()
            if "contact" in ablated:
                contact_result = dict(contact_result)
                contact_scores = zeros.copy()
                contact_result["contact_ring_scores"] = contact_scores
                contact_result["contact_ring_mask"] = np.zeros(num_tokens, dtype=bool)
            if "motion" in ablated:
                motion_result = dict(motion_result)
                motion_scores = zeros.copy()
                motion_result["motion_corridor_scores"] = motion_scores
                motion_result["motion_corridor_valid"] = False
                motion_result["motion_disabled_reason"] = "acgtp_branch_ablation"
                motion_valid = False
            if "fill" in ablated:
                fill_mask = zeros.copy()

        acr_start = time.perf_counter()
        acr_result = compute_future_action_constraint_scores(
            scene_layout_scores=scene_scores,
            depth_structure_scores=edge_scores,
            contact_ring_scores=contact_scores,
            motion_corridor_scores=motion_scores,
            valid_mask=valid_mask,
            robot_self_core_mask=self_core,
            scene_result=scene_result,
            contact_result=contact_result,
            motion_result=motion_result,
            w_scene=float(self.config.acgtp_w_scene_layout),
            w_depth=float(self.config.acgtp_w_depth_structure),
            w_contact=float(self.config.acgtp_w_contact_ring),
            w_motion=float(self.config.acgtp_w_motion_corridor),
        )
        action_scores = acr_result["action_constraint_scores"]
        acr_ms = (time.perf_counter() - acr_start) * 1000.0
        score_ms = (time.perf_counter() - score_start) * 1000.0

        self._update_previous_gripper_pos(latest, grip)

        metrics.timing.score_compute_ms = score_ms
        metrics.timing.depth_edge_score_ms = score_ms
        metrics.timing.score_fusion_ms = score_ms
        metrics.acgtp_scene_layout_ms = scene_ms
        metrics.acgtp_static_scene_cache_enabled = scene_cache_meta.get("acgtp_static_scene_cache_enabled")
        metrics.acgtp_static_scene_cache_hit = scene_cache_meta.get("acgtp_static_scene_cache_hit")
        metrics.acgtp_static_scene_cache_reason = scene_cache_meta.get("acgtp_static_scene_cache_reason")
        metrics.acgtp_static_scene_cache_depth_delta = scene_cache_meta.get("acgtp_static_scene_cache_depth_delta")
        metrics.acgtp_static_scene_cache_valid_iou = scene_cache_meta.get("acgtp_static_scene_cache_valid_iou")
        metrics.acgtp_static_scene_cache_age = scene_cache_meta.get("acgtp_static_scene_cache_age")
        metrics.acgtp_contact_ring_ms = contact_ms
        metrics.acgtp_motion_corridor_ms = motion_ms
        metrics.acgtp_action_constraint_ms = acr_ms
        metrics.geometry_available = True
        metrics.robot_state_available = True
        metrics.camera_available = True
        metrics.depth_edge_score_mean = float(np.mean(edge_scores[valid_mask])) if np.any(valid_mask) else None
        metrics.mean_near_score = float(np.mean(near_scores[valid_mask])) if np.any(valid_mask) else None
        metrics.max_near_score = float(np.max(near_scores[valid_mask])) if np.any(valid_mask) else None
        metrics.motion_norm = motion_result.get("motion_norm_m")
        metrics.motion_direction_valid = motion_valid
        metrics.distance_to_gripper_min = float(np.min(distances[valid_mask])) if np.any(valid_mask) else None
        metrics.distance_to_gripper_mean = float(np.mean(distances[valid_mask])) if np.any(valid_mask) else None
        metrics.distance_to_gripper_max = float(np.max(distances[valid_mask])) if np.any(valid_mask) else None
        metrics.transform_source = transform_key
        metrics.gripper_pixel_u = float(gripper_pixel[0]) if gripper_pixel is not None else None
        metrics.gripper_pixel_v = float(gripper_pixel[1]) if gripper_pixel is not None else None
        metrics.gripper_in_bounds = contact_result.get("gripper_in_bounds")

        hist_decision: Dict[str, Any] = {"acgtp_history_enabled": False}
        if bool(getattr(self.config, "acgtp_history_enabled", False)):
            try:
                hist_decision = self._acgtp_history.prepare_step(
                    scene_scores=scene_scores,
                    depth_scores=edge_scores,
                    contact_scores=contact_scores,
                    motion_scores=motion_scores,
                    action_scores=action_scores,
                    valid_mask=valid_mask,
                    num_tokens=num_tokens,
                    gripper_pos=grip,
                    depth_valid_ratio=valid_ratio,
                )
                scene_scores = hist_decision.get("scene_scores", scene_scores)
                edge_scores = hist_decision.get("depth_scores", edge_scores)
                contact_scores = hist_decision.get("contact_scores", contact_scores)
                motion_scores = hist_decision.get("motion_scores", motion_scores)
                action_scores = hist_decision.get("action_scores", action_scores)
            except Exception as hist_exc:
                hist_decision = {
                    "acgtp_history_enabled": False,
                    "acgtp_history_disabled_reason": f"history_error:{type(hist_exc).__name__}:{str(hist_exc)[:120]}",
                }

        dyn_decision: Dict[str, Any] = {"acgtp_dynamic_enabled": False}
        w_scene = float(self.config.acgtp_w_scene_layout)
        w_depth = float(self.config.acgtp_w_depth_structure)
        w_contact = float(self.config.acgtp_w_contact_ring)
        w_motion = float(self.config.acgtp_w_motion_corridor)
        hard_ratio = float(self.config.acgtp_hard_protect_ratio)
        if "scene" in ablated:
            w_scene = 0.0
        if "depth" in ablated:
            w_depth = 0.0
        if "contact" in ablated:
            w_contact = 0.0
        if "motion" in ablated:
            w_motion = 0.0
        if bool(getattr(self.config, "acgtp_dynamic_enabled", False)):
            try:
                dyn_decision = decide_acgtp_dynamic_budget(
                    scene_layout_scores=scene_scores,
                    depth_structure_scores=edge_scores,
                    contact_ring_scores=contact_scores,
                    motion_corridor_scores=motion_scores,
                    action_constraint_scores=action_scores,
                    valid_mask=valid_mask,
                    constrained_fill_mask=fill_mask,
                    num_tokens=num_tokens,
                    base_keep_ratio=float(self.config.keep_ratio),
                    previous_state=self._acgtp_dynamic_state,
                    motion_corridor_valid=motion_valid,
                    motion_norm_m=motion_result.get("motion_norm_m"),
                    depth_valid_ratio=valid_ratio,
                    min_keep_ratio=min(
                        float(self.config.acgtp_dynamic_max_keep_ratio),
                        max(
                            float(self.config.acgtp_dynamic_min_keep_ratio),
                            (
                                float(self.config.keep_ratio)
                                if not bool(getattr(self.config, "acgtp_dynamic_allow_below_base_keep_ratio", False))
                                else float(self.config.acgtp_dynamic_min_keep_ratio)
                            ),
                        )
                        + (
                            float(self.config.acgtp_history_conservative_keep_boost)
                            if bool(hist_decision.get("acgtp_history_conservative_mode")) else 0.0
                        ),
                    ),
                    max_keep_ratio=float(self.config.acgtp_dynamic_max_keep_ratio),
                    risk_boost_scale=float(self.config.acgtp_dynamic_risk_boost_scale),
                    confidence_prune_scale=float(self.config.acgtp_dynamic_confidence_prune_scale),
                    contact_phase_gate=str(getattr(self.config, "acgtp_dynamic_contact_phase_gate", "legacy_peak")),
                    phase_schedule=str(getattr(self.config, "acgtp_dynamic_phase_schedule", "legacy")),
                    branch_floor_enabled=bool(getattr(self.config, "acgtp_dynamic_branch_floor_enabled", False)),
                    fill_cap_ratio=float(getattr(self.config, "acgtp_constrained_fill_max_ratio", 1.0)),
                    respect_phase_min_on_candidate_gap=bool(getattr(self.config, "acgtp_dynamic_respect_phase_min_on_candidate_gap", False)),
                    shadow_contact_guard_enabled=bool(getattr(self.config, "acgtp_dynamic_shadow_contact_guard_enabled", False)),
                    shadow_contact_depth_weight_floor=float(getattr(self.config, "acgtp_dynamic_shadow_contact_depth_weight_floor", 0.30)),
                    shadow_contact_contact_weight_floor=float(getattr(self.config, "acgtp_dynamic_shadow_contact_contact_weight_floor", 0.24)),
                    shadow_contact_hard_ratio_floor=float(getattr(self.config, "acgtp_dynamic_shadow_contact_hard_ratio_floor", 0.70)),
                )
                state = dyn_decision.pop("_state", None)
                if isinstance(state, dict):
                    self._acgtp_dynamic_state = state
                keep_count = max(1, min(num_tokens, int(dyn_decision.get("acgtp_dynamic_keep_k", keep_count))))
                w_scene = float(dyn_decision.get("acgtp_dynamic_scene_weight", w_scene))
                w_depth = float(dyn_decision.get("acgtp_dynamic_depth_weight", w_depth))
                w_contact = float(dyn_decision.get("acgtp_dynamic_contact_weight", w_contact))
                w_motion = float(dyn_decision.get("acgtp_dynamic_motion_weight", w_motion))
                hard_ratio = float(dyn_decision.get("acgtp_dynamic_hard_protect_ratio", hard_ratio))
                if "scene" in ablated:
                    w_scene = 0.0
                if "depth" in ablated:
                    w_depth = 0.0
                if "contact" in ablated:
                    w_contact = 0.0
                if "motion" in ablated:
                    w_motion = 0.0
                metrics.keep_ratio_source = "acgtp_dynamic_controller"
            except Exception as dyn_exc:
                dyn_decision = {
                    "acgtp_dynamic_enabled": False,
                    "acgtp_dynamic_disabled_reason": f"controller_error:{type(dyn_exc).__name__}:{str(dyn_exc)[:120]}",
                }

        if self._internal_pruning_requested():
            attention_guide = {
                "enabled": False,
                "score": None,
                "mask": None,
                "meta": {
                    "attention_backend": "internal_llm",
                    "attention_source": "disabled_in_hook_internal_backend_uses_true_attention",
                    "attention_available": False,
                    "attention_confidence": 0.0,
                    "attention_quota_released": True,
                    "attention_top_count": 0,
                },
            }
        else:
            attention_guide = self._build_acgtp_attention_guide(
                visual_tokens=visual_tokens,
                scene_scores=scene_scores,
                depth_scores=edge_scores,
                contact_scores=contact_scores,
                motion_scores=motion_scores,
                action_scores=action_scores,
                valid_mask=valid_mask,
                keep_count=keep_count,
            )

        keep_indices_np, selection_meta = select_acgtp_v2_fast(
            scene_layout_scores=scene_scores,
            depth_edge_scores=edge_scores,
            contact_ring_scores=contact_scores,
            motion_corridor_scores=motion_scores,
            valid_mask=valid_mask,
            keep_k=keep_count,
            constrained_fill_mask=fill_mask,
            token_u=cache.get("u"),
            token_v=cache.get("v"),
            grid_h=token_grid_shape[0],
            grid_w=token_grid_shape[1],
            w_scene_layout=w_scene,
            w_depth_structure=w_depth,
            w_contact_ring=w_contact,
            w_motion_corridor=w_motion,
            w_semantic=0.20,
            hard_protect_ratio=hard_ratio,
            motion_corridor_valid=motion_valid,
            self_core_mask=self_core,
            contact_ring_inner_px=float(self.config.acgtp_contact_ring_inner_px),
            contact_ring_outer_px=float(self.config.acgtp_contact_ring_outer_px),
            contact_requires_edge_or_object=bool(self.config.acgtp_contact_requires_edge_or_object),
            depth_edge_score_for_gate=edge_scores,
            action_constraint_scores=action_scores,
            semantic_enabled=False,
            semantic_backend="none",
            semantic_unavailable=True,
            support_plane_cap_ratio=float(self.config.acgtp_scene_support_plane_cap_ratio),
            acgtp_attention_enabled=bool(attention_guide.get("enabled", False)),
            acgtp_attention_backend=str(attention_guide.get("meta", {}).get("attention_backend", "none")),
            acgtp_attention_task_relevance_score=attention_guide.get("score"),
            acgtp_attention_task_relevance_mask=attention_guide.get("mask"),
            acgtp_attention_source=str(attention_guide.get("meta", {}).get("attention_source", "none")),
            acgtp_attention_available=bool(attention_guide.get("meta", {}).get("attention_available", False)),
            acgtp_attention_confidence=float(attention_guide.get("meta", {}).get("attention_confidence", 0.0)),
            acgtp_attention_budget_ratio=float(getattr(self.config, "acgtp_attention_budget_ratio", 0.12)),
            acgtp_attention_requires_geometry_alignment=bool(getattr(self.config, "acgtp_attention_requires_geometry_alignment", True)),
            min_scene_tokens=int(dyn_decision.get("acgtp_dynamic_min_scene_tokens", 0) or 0),
            min_depth_tokens=int(dyn_decision.get("acgtp_dynamic_min_depth_tokens", 0) or 0),
            min_contact_tokens=int(dyn_decision.get("acgtp_dynamic_min_contact_tokens", 0) or 0),
            min_motion_tokens=int(dyn_decision.get("acgtp_dynamic_min_motion_tokens", 0) or 0),
            constrained_fill_max_tokens=(
                int(dyn_decision.get("acgtp_dynamic_fill_cap_tokens"))
                if dyn_decision.get("acgtp_dynamic_fill_cap_tokens") is not None
                else None
            ),
            minimal_metadata=True,
        )
        selection_meta.update(attention_guide.get("meta", {}))
        selection_meta.update({k: v for k, v in hist_decision.items() if not k.endswith("_scores") and k not in ("scene_scores", "depth_scores", "contact_scores", "motion_scores", "action_scores")})
        selection_meta.update(dyn_decision)
        selection_meta.update(ablation_meta)
        selection_meta.update(scene_cache_meta)
        selection_meta.update(plan_cache_meta)

        robot_metrics = {
            "gripper_pos": grip,
            "acgtp_action_constraint_ms": acr_ms,
            "acgtp_motion_corridor_valid": motion_valid,
        }
        geometry_payload = self._build_internal_geometry_payload(
            scene_scores=scene_scores,
            depth_scores=edge_scores,
            contact_scores=contact_scores,
            motion_scores=motion_scores,
            action_scores=action_scores,
            valid_mask=valid_mask,
            fill_mask=fill_mask,
            keep_count=keep_count,
            num_tokens=num_tokens,
            motion_valid=motion_valid,
            dyn_decision=dyn_decision,
            hard_ratio=hard_ratio,
            w_scene=w_scene,
            w_depth=w_depth,
            w_contact=w_contact,
            w_motion=w_motion,
        )
        self._store_latency_plan_cache(
            depth_probe=plan_cache_probe,
            grip=grip,
            num_tokens=num_tokens,
            keep_count=keep_count,
            token_grid_shape=token_grid_shape,
            keep_indices_np=keep_indices_np,
            selection_meta=selection_meta,
            geometry_payload=geometry_payload,
            robot_metrics=robot_metrics,
            valid_ratio=valid_ratio,
            depth_source_key=depth_source_key,
        )
        return self._finalize_acgtp_fast_runtime(
            visual_tokens=visual_tokens,
            metrics=metrics,
            keep_indices_np=keep_indices_np,
            selection_meta=selection_meta,
            robot_metrics=robot_metrics,
            num_tokens=num_tokens,
            geometry_payload=geometry_payload,
        )

    def attach_to_model(self, model: Any) -> bool:
        projector_attached = False
        for name, module in model.named_modules():
            if name == "projector":
                self._hook_handle = module.register_forward_hook(self._projector_hook)
                projector_attached = True
                break
        language_model = getattr(model, "language_model", None)
        if self._internal_pruning_requested():
            try:
                self._internal_backend = enable_acgtp_internal_pruning(
                    model,
                    prune_layer=int(getattr(self.config, "acgtp_internal_prune_layer", 2)),
                    image_token_start_index=1,
                    image_token_length=256,
                    fail_on_error=bool(getattr(self.config, "acgtp_internal_fail_on_backend_error", True)),
                )
            except Exception as exc:
                self._internal_backend = None
                self._update_latest_position_stats(
                    compression_backend="internal",
                    internal_pruning_requested=True,
                    internal_backend_error=f"{type(exc).__name__}: {exc}",
                    internal_pruning_plan_ready=False,
                    internal_pruning_applied=False,
                )
                if bool(getattr(self.config, "acgtp_internal_fail_on_backend_error", True)):
                    raise
            if self._internal_backend is None:
                self._update_latest_position_stats(
                    compression_backend="internal",
                    internal_pruning_requested=True,
                    internal_backend_error="backend_attach_failed",
                    internal_pruning_plan_ready=False,
                    internal_pruning_applied=False,
                )
                if bool(getattr(self.config, "acgtp_internal_fail_on_backend_error", True)):
                    raise RuntimeError("ACGTP internal pruning backend could not attach to language_model.model")
                if not bool(getattr(self.config, "acgtp_internal_allow_projector_fallback", False)):
                    return False
                self.config.acgtp_compression_backend = "projector"
                self.config.acgtp_internal_pruning_enabled = False
        if language_model is not None and not self._internal_pruning_requested():
            try:
                self._lm_pre_hook_handle = language_model.register_forward_pre_hook(
                    self._language_model_pre_hook,
                    with_kwargs=True,
                )
            except TypeError:
                self._lm_pre_hook_handle = None
                self._update_latest_position_stats(
                    position_preserve_enabled=bool(getattr(self.config, "acgtp_position_preserve_enabled", True)),
                    position_preserve_applied=False,
                    position_preserve_reason="forward_pre_hook_with_kwargs_unavailable",
                )
        return projector_attached

    def detach(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        if self._lm_pre_hook_handle is not None:
            self._lm_pre_hook_handle.remove()
            self._lm_pre_hook_handle = None
        if self._internal_backend is not None:
            disable_acgtp_internal_pruning(getattr(self._internal_backend, "model", None))
            self._internal_backend = None

    def _projector_hook(self, module: Any, inputs: Tuple[torch.Tensor], output: torch.Tensor) -> torch.Tensor:
        start_total = time.perf_counter()
        with torch.no_grad():
            pruned, metrics = self._run(output)
        metrics.timing.hook_total_ms = (time.perf_counter() - start_total) * 1000.0
        if self._acgtp_fast_runtime_enabled() and hasattr(metrics, "to_fast_eval_stats"):
            self._latest_stats = metrics.to_fast_eval_stats()
        else:
            self._latest_stats = metrics.to_eval_stats()
        self._latest_stats["acgtp_runtime_mode"] = self._runtime_mode()
        return pruned

    def _run(self, visual_tokens: torch.Tensor) -> Tuple[torch.Tensor, HookMetrics]:
        num_tokens = int(visual_tokens.shape[1])
        keep_count = self.config.keep_count(num_tokens)
        metrics = HookMetrics(
            num_visual_tokens_original=num_tokens,
            num_visual_tokens_kept=num_tokens,
            num_visual_tokens_pruned=0,
            keep_ratio=1.0,
            requested_keep_ratio=float(self.config.keep_ratio),
            keep_ratio_source=self._default_keep_ratio_source(),
            effective_keep_count=keep_count,
            original_token_count=num_tokens,
            pruning_strategy=self.config.strategy,
            pruning_method=self.config.strategy,
            protected_token_ratio=1.0,
            pruned_token_ratio=0.0,
            timing=HookTiming(),
        )

        _skip_postprocess_diag = False
        if not self.config.enabled:
            metrics.keep_indices_sorted = True
            metrics.keep_indices_unique = True
            metrics.keep_indices_out_of_bounds = False
            metrics.duplicate_indices_count = 0
            metrics.effective_keep_count = num_tokens
            metrics.original_token_count = num_tokens
            metrics.requested_keep_ratio = float(self.config.keep_ratio)
            metrics.actual_keep_ratio = 1.0
            metrics.keep_ratio_requested = float(self.config.keep_ratio)
            metrics.keep_ratio_actual = 1.0
            metrics.retention_actual = 1.0
            metrics.actual_retention_ratio = 1.0
            metrics.num_visual_tokens_dropped = 0
            metrics.num_visual_tokens_original_total = num_tokens
            metrics.num_visual_tokens_kept_total = num_tokens
            metrics.keep_ratio_source = "cli_keep_ratio"
            metrics.selector_name = "none"
            metrics.selection_strategy_name = "none"
            metrics.selection_stage_name = "disabled_no_pruning"
            metrics.selector_success = None
            metrics.fallback_used = False
            metrics.fallback_reason = None
            self._prepare_position_preserve_info(
                keep_indices_np=np.arange(num_tokens, dtype=np.int64),
                num_tokens=int(num_tokens),
                metrics=metrics,
            )
            return visual_tokens, metrics
        latest_for_warmup = self.geometry_recorder.get_latest() if self.geometry_recorder is not None else None
        metrics.is_warmup_step = self._mark_warmup_step(latest_for_warmup)

        # ── 1. Uniform init of ALL intermediate state ───────────────────────
        scores: Optional[np.ndarray] = None
        valid_mask: Optional[np.ndarray] = None
        fallback_reason: Optional[str] = None
        keep_indices_np: Optional[np.ndarray] = None
        selection_meta: Dict[str, Any] = {}
        robot_metrics: Dict[str, Any] = {}

        if self._acgtp_fast_runtime_enabled():
            return self._run_acgtp_fast_runtime(
                visual_tokens,
                metrics,
                keep_count=keep_count,
                num_tokens=num_tokens,
            )

        # P8: Determine which aux metrics are active for this strategy.
        # Used by visualization and geo_debug — avoids the broken
        #   robot_metrics if "robot_metrics" in locals() else depth_metrics ...
        # pattern where robot_metrics={} is always in locals() even for depth_edge_fast.
        if self.config.strategy in ("depth_edge_fast", "depth_edge_fast_diverse"):
            scores, valid_mask, depth_metrics, fallback_reason = self._compute_depth_edge_scores(num_tokens)
            metrics.timing.token_mapping_ms = depth_metrics.get("token_mapping_ms")
            metrics.timing.depth_sampling_ms = depth_metrics.get("depth_sampling_ms")
            metrics.timing.depth_sample_ms = metrics.timing.depth_sampling_ms
            metrics.timing.score_compute_ms = depth_metrics.get("score_compute_ms")
            metrics.timing.depth_edge_score_ms = metrics.timing.score_compute_ms
            metrics.valid_token_ratio = depth_metrics.get("valid_token_ratio")
            metrics.geometry_score_mean = depth_metrics.get("score_mean")
            metrics.geometry_score_max = depth_metrics.get("score_max")
            metrics.geometry_score_std = depth_metrics.get("score_std")
            metrics.depth_edge_score_mean = depth_metrics.get("score_mean")
            depth_metrics.setdefault("edge_scores", scores)
            depth_metrics.setdefault("final_scores", scores)
            depth_metrics.setdefault("hybrid_final_scores", scores)
            # P1-1: Add token_u, token_v, gripper_pixel for attribution if available from cache
            if "u" in depth_metrics.get("cache", {}):
                depth_metrics.setdefault("token_u", depth_metrics["cache"]["u"])
            if "v" in depth_metrics.get("cache", {}):
                depth_metrics.setdefault("token_v", depth_metrics["cache"]["v"])
            depth_metrics.setdefault("token_grid_shape", self.config.token_grid_shape)
            # P0: depth conversion metadata
            metrics.depth_source_key = depth_metrics.get("depth_source_key")
            metrics.depth_conversion = depth_metrics.get("depth_conversion")
            metrics.depth_is_metric = depth_metrics.get("depth_is_metric")
            metrics.depth_unit = depth_metrics.get("depth_unit")
            metrics.depth_sim_available = depth_metrics.get("depth_sim_available")
            metrics.depth_raw_min = depth_metrics.get("depth_raw_min")
            metrics.depth_raw_max = depth_metrics.get("depth_raw_max")
            metrics.depth_raw_mean = depth_metrics.get("depth_raw_mean")
            metrics.depth_raw_std = depth_metrics.get("depth_raw_std")
            metrics.depth_metric_min = depth_metrics.get("depth_metric_min")
            metrics.depth_metric_max = depth_metrics.get("depth_metric_max")
            metrics.depth_metric_mean = depth_metrics.get("depth_metric_mean")
            metrics.depth_metric_std = depth_metrics.get("depth_metric_std")
        elif self.config.strategy in ROBOT_GEO_SCORE_STRATEGIES:
            scores, valid_mask, robot_metrics, fallback_reason = self._compute_robot_geo_near_scores(num_tokens)
            metrics.timing.token_mapping_ms = robot_metrics.get("token_mapping_ms")
            metrics.timing.depth_sampling_ms = robot_metrics.get("depth_sampling_ms")
            metrics.timing.depth_sample_ms = metrics.timing.depth_sampling_ms
            metrics.timing.score_compute_ms = robot_metrics.get("score_compute_ms")
            metrics.timing.depth_edge_score_ms = robot_metrics.get("depth_edge_score_ms")
            metrics.timing.token_xyz_projection_ms = robot_metrics.get("token_xyz_projection_ms")
            metrics.timing.score_fusion_ms = robot_metrics.get("score_fusion_ms")
            metrics.timing.edge_score_ms = robot_metrics.get("edge_score_ms")
            metrics.timing.robot_mapping_ms = robot_metrics.get("robot_mapping_ms")
            metrics.timing.near_score_ms = robot_metrics.get("near_score_ms")
            metrics.timing.corridor_score_ms = robot_metrics.get("corridor_score_ms")
            metrics.timing.contact_score_ms = robot_metrics.get("contact_score_ms")
            metrics.geometry_available = robot_metrics.get("geometry_available")
            metrics.robot_state_available = robot_metrics.get("robot_state_available")
            metrics.camera_available = robot_metrics.get("camera_available")
            metrics.valid_token_ratio = robot_metrics.get("valid_token_ratio")
            metrics.geometry_score_mean = robot_metrics.get("geometry_score_mean")
            metrics.geometry_score_max = robot_metrics.get("geometry_score_max")
            metrics.geometry_score_std = robot_metrics.get("geometry_score_std")
            metrics.depth_edge_score_mean = robot_metrics.get("depth_edge_score_mean")
            metrics.distance_score_mean = robot_metrics.get("mean_near_score")
            metrics.direction_score_mean = robot_metrics.get("motion_cone_score_mean") or robot_metrics.get("corridor_strength_mean")
            metrics.d_min = robot_metrics.get("d_min")
            # P0: depth conversion metadata
            metrics.depth_source_key = robot_metrics.get("depth_source_key")
            metrics.depth_conversion = robot_metrics.get("depth_conversion")
            metrics.depth_is_metric = robot_metrics.get("depth_is_metric")
            metrics.depth_unit = robot_metrics.get("depth_unit")
            metrics.depth_sim_available = robot_metrics.get("depth_sim_available")
            metrics.depth_raw_min = robot_metrics.get("depth_raw_min")
            metrics.depth_raw_max = robot_metrics.get("depth_raw_max")
            metrics.depth_raw_mean = robot_metrics.get("depth_raw_mean")
            metrics.depth_raw_std = robot_metrics.get("depth_raw_std")
            metrics.depth_metric_min = robot_metrics.get("depth_metric_min")
            metrics.depth_metric_max = robot_metrics.get("depth_metric_max")
            metrics.depth_metric_mean = robot_metrics.get("depth_metric_mean")
            metrics.depth_metric_std = robot_metrics.get("depth_metric_std")
            # P0-4: transform convention (T_robot_cam_forward)
            metrics.transform_convention = robot_metrics.get("transform_convention")
            metrics.transform_inverse_used = robot_metrics.get("transform_inverse_used")
            metrics.transform_source = robot_metrics.get("transform_source")
            metrics.transform_convention_verified = robot_metrics.get("transform_convention_verified")
            metrics.transform_convention_evidence = robot_metrics.get("transform_convention_evidence")
            metrics.mean_near_score = robot_metrics.get("mean_near_score")
            metrics.max_near_score = robot_metrics.get("max_near_score")
            metrics.motion_norm = robot_metrics.get("motion_norm")
            metrics.corridor_strength_mean = robot_metrics.get("corridor_strength_mean")
            metrics.corridor_active = robot_metrics.get("corridor_active")
            metrics.d_corridor_min = robot_metrics.get("d_corridor_min")
            metrics.edge_score_mean = robot_metrics.get("edge_score_mean")
            metrics.motion_cone_score_mean = robot_metrics.get("motion_cone_score_mean")
            metrics.motion_cone_score_max = robot_metrics.get("motion_cone_score_max")
            metrics.workspace_score_mean = robot_metrics.get("workspace_score_mean")
            metrics.workspace_score_max = robot_metrics.get("workspace_score_max")
            metrics.contact_risk_score_mean = robot_metrics.get("contact_risk_score_mean")
            metrics.contact_risk_score_max = robot_metrics.get("contact_risk_score_max")
            metrics.geo_risk_level = robot_metrics.get("geo_risk_level")
            metrics.geo_risk_score = robot_metrics.get("geo_risk_score")
            metrics.dynamic_keep_reason = robot_metrics.get("dynamic_keep_reason")
            metrics.num_high_contact_tokens = robot_metrics.get("num_high_contact_tokens")
            metrics.num_valid_3d_tokens = robot_metrics.get("num_valid_3d_tokens")
            metrics.interaction_lock = robot_metrics.get("interaction_lock")
            metrics.temporal_stability = robot_metrics.get("temporal_stability")
            metrics.history_length = robot_metrics.get("history_length")
            metrics.score_ema_enabled = robot_metrics.get("score_ema_enabled")
            metrics.near_contact_score_mean = robot_metrics.get("near_contact_score_mean")
            metrics.corridor_contact_score_mean = robot_metrics.get("corridor_contact_score_mean")
            metrics.geo_contact_score_mean = robot_metrics.get("geo_contact_score_mean")
            # New diagnostic fields: depth
            metrics.depth_min = robot_metrics.get("depth_min")
            metrics.depth_max = robot_metrics.get("depth_max")
            metrics.depth_mean = robot_metrics.get("depth_mean")
            metrics.depth_valid_ratio = robot_metrics.get("valid_token_ratio")
            # New diagnostic fields: 3D token geometry
            metrics.points_robot_min_xyz = robot_metrics.get("points_robot_min_xyz")
            metrics.points_robot_max_xyz = robot_metrics.get("points_robot_max_xyz")
            metrics.points_cam_min_xyz = robot_metrics.get("points_cam_min_xyz")
            metrics.points_cam_max_xyz = robot_metrics.get("points_cam_max_xyz")
            metrics.extrinsics_available = robot_metrics.get("extrinsics_available")
            metrics.intrinsics_available = robot_metrics.get("intrinsics_available")
            metrics.camera_frame_name = robot_metrics.get("camera_frame_name")
            metrics.geometry_frame_name = robot_metrics.get("geometry_frame_name")
            # New diagnostic fields: robot state / gripper
            metrics.ee_position = robot_metrics.get("ee_position")
            metrics.robot_state_valid = robot_metrics.get("robot_state_valid")
            metrics.motion_direction_valid = robot_metrics.get("motion_direction_valid")
            metrics.motion_direction_xyz = robot_metrics.get("motion_direction_xyz")
            metrics.distance_to_gripper_min = robot_metrics.get("distance_to_gripper_min")
            metrics.distance_to_gripper_mean = robot_metrics.get("distance_to_gripper_mean")
            metrics.distance_to_gripper_max = robot_metrics.get("distance_to_gripper_max")
            # P1-1: Score component is_none/nonzero_ratio (from hybrid_v1 branch)
            metrics.contact_risk_scores_is_none = robot_metrics.get("contact_risk_scores_is_none")
            metrics.motion_cone_scores_is_none = robot_metrics.get("motion_cone_scores_is_none")
            metrics.workspace_scores_is_none = robot_metrics.get("workspace_scores_is_none")
            metrics.contact_nonzero_ratio = robot_metrics.get("contact_nonzero_ratio")
            metrics.motion_cone_nonzero_ratio = robot_metrics.get("motion_cone_nonzero_ratio")
            metrics.workspace_nonzero_ratio = robot_metrics.get("workspace_nonzero_ratio")
            # P1-1: temporal_v1 semantics fix
            metrics.temporal_enabled = robot_metrics.get("temporal_enabled")
            metrics.interaction_lock_triggered = robot_metrics.get("interaction_lock_triggered")
            metrics.interaction_lock_reason = robot_metrics.get("interaction_lock_reason")

        # P8: Assign active_aux_metrics based on which branch was taken above.
        # This replaces all broken "robot_metrics if 'robot_metrics' in locals()" checks
        # where robot_metrics={} is always in locals() even for depth_edge_fast.
        if self.config.strategy in ("depth_edge_fast", "depth_edge_fast_diverse"):
            active_aux_metrics = depth_metrics
        else:
            active_aux_metrics = robot_metrics

        strategy = self.config.strategy
        reserve_tokens = self.config.diverse_reserve_tokens
        if self.config.strategy == "robot_geo_dynamic" and fallback_reason is None:
            try:
                score_dict = {
                    "scores": scores,
                    "edge_scores": robot_metrics.get("edge_scores"),
                    "corridor_scores": robot_metrics.get("corridor_scores"),
                    "motion_norm": robot_metrics.get("motion_norm"),
                    "valid_depth_ratio": robot_metrics.get("valid_token_ratio"),
                }
                _, dynamic_stats = compute_dynamic_keep_ratio(
                    score_dict=score_dict,
                    p_robot=robot_metrics.get("token_points_robot"),
                    gripper_pos=robot_metrics.get("gripper_pos"),
                    valid_mask=valid_mask,
                    cfg=self.config,
                )
                keep_count = int(dynamic_stats.get("dynamic_keep_k", keep_count))
                reserve_tokens = int(dynamic_stats.get("dynamic_reserve_k", reserve_tokens))
                metrics.dynamic_enabled = True
                metrics.keep_ratio_source = "dynamic_decision"
                metrics.dynamic_phase = dynamic_stats.get("dynamic_phase")
                metrics.dynamic_keep_ratio = dynamic_stats.get("dynamic_keep_ratio")
                metrics.dynamic_keep_k = keep_count
                metrics.d_min = dynamic_stats.get("d_min", metrics.d_min)
                metrics.d_topk_mean = dynamic_stats.get("d_topk_mean")
                metrics.corridor_strength = dynamic_stats.get("corridor_strength")
                metrics.edge_concentration = dynamic_stats.get("edge_concentration")
                metrics.valid_depth_ratio = dynamic_stats.get("valid_depth_ratio")
                metrics.motion_norm = dynamic_stats.get("motion_norm", metrics.motion_norm)
                metrics.phase_far_count = 1 if metrics.dynamic_phase == "far" else 0
                metrics.phase_mid_count = 1 if metrics.dynamic_phase == "mid" else 0
                metrics.phase_near_count = 1 if metrics.dynamic_phase == "near" else 0
                metrics.phase_fallback_safe_count = 1 if metrics.dynamic_phase == "fallback_safe" else 0
            except Exception as exc:
                metrics.fallback_used = True
                metrics.keep_ratio_source = "fallback"
                metrics.fallback_reason = f"dynamic_scheduler_error:{type(exc).__name__}"
                strategy = "depth_edge_fast_diverse"
                keep_count = int(round(num_tokens * self.config.keep_ratio_mid))
                reserve_tokens = self.config.diverse_reserve_tokens
                fallback_reason = None
        if self.config.strategy == "robot_geo_dynamic_v0" and fallback_reason is None:
            dynamic_decision = robot_metrics.get("dynamic_decision") or {}
            dynamic_ratio = dynamic_decision.get("keep_ratio")
            if dynamic_ratio is not None:
                keep_count = int(round(num_tokens * float(dynamic_ratio)))
                keep_count = max(1, min(num_tokens, keep_count))
                metrics.dynamic_enabled = True
                metrics.keep_ratio_source = "dynamic_decision"
                metrics.dynamic_phase = dynamic_decision.get("risk_level")
                metrics.dynamic_keep_ratio = keep_count / num_tokens if num_tokens else None
                metrics.dynamic_keep_k = keep_count
                metrics.geo_risk_level = dynamic_decision.get("risk_level")
                metrics.geo_risk_score = dynamic_decision.get("risk_score")
                metrics.dynamic_keep_reason = dynamic_decision.get("reason")
                component_summary = dynamic_decision.get("component_summary") or {}
                metrics.num_high_contact_tokens = component_summary.get("num_high_contact_tokens")
                metrics.num_valid_3d_tokens = component_summary.get("num_valid_3d_tokens")
                metrics.phase_far_count = 1 if metrics.dynamic_phase == "low" else 0
                metrics.phase_mid_count = 1 if metrics.dynamic_phase == "medium" else 0
                metrics.phase_near_count = 1 if metrics.dynamic_phase == "high" else 0
                metrics.phase_fallback_safe_count = 0
            else:
                mid_ratio = float(self.config.dynamic_keep_ratio_config.get("mid_keep_ratio", self.config.keep_ratio_mid))
                keep_count = int(round(num_tokens * mid_ratio))
                metrics.dynamic_enabled = True
                metrics.keep_ratio_source = "dynamic_mid_keep_ratio"
                metrics.dynamic_phase = "medium"
                metrics.dynamic_keep_ratio = keep_count / num_tokens if num_tokens else None
                metrics.dynamic_keep_k = keep_count
                metrics.dynamic_keep_reason = "missing_dynamic_decision_fallback"
        if self.config.strategy == "robot_geo_temporal_v0" and fallback_reason is None:
            dynamic_decision = robot_metrics.get("dynamic_decision") or {}
            dynamic_ratio = dynamic_decision.get("keep_ratio")
            if dynamic_ratio is None:
                dynamic_ratio = float(self.config.dynamic_keep_ratio_config.get("mid_keep_ratio", self.config.keep_ratio_mid))
                metrics.keep_ratio_source = "dynamic_mid_keep_ratio"
            else:
                metrics.keep_ratio_source = "dynamic_decision"
            temporal = self._temporal_history.detect_interaction_lock(
                contact_risk_score=robot_metrics.get("rule_v0_contact_risk_scores"),
                final_scores=scores,
                motion_direction=robot_metrics.get("motion_direction"),
                valid_3d_ratio=robot_metrics.get("valid_token_ratio"),
                dynamic_keep_ratio=dynamic_ratio,
                config=self.config,
                gripper_pos=robot_metrics.get("gripper_pos"),
                token_points_robot=robot_metrics.get("token_points_robot"),
            )
            lock_ratio = float(self.config.dynamic_keep_ratio_config.get("max_keep_ratio", self.config.keep_ratio_near))
            chosen_ratio = max(float(dynamic_ratio), lock_ratio) if temporal.get("interaction_lock") else float(dynamic_ratio)
            keep_count = max(1, min(num_tokens, int(round(num_tokens * chosen_ratio))))
            metrics.dynamic_enabled = True
            metrics.dynamic_phase = "interaction_lock" if temporal.get("interaction_lock") else dynamic_decision.get("risk_level", "medium")
            metrics.dynamic_keep_ratio = keep_count / num_tokens if num_tokens else None
            metrics.dynamic_keep_k = keep_count
            metrics.geo_risk_level = metrics.dynamic_phase
            metrics.geo_risk_score = dynamic_decision.get("risk_score")
            metrics.dynamic_keep_reason = "temporal_interaction_lock" if temporal.get("interaction_lock") else dynamic_decision.get("reason", "single_frame_rule")
            metrics.interaction_lock = bool(temporal.get("interaction_lock"))
            metrics.interaction_lock_reason = str(temporal.get("lock_reason", "none"))
            metrics.temporal_stability = temporal.get("temporal_stability")
            metrics.history_length = temporal.get("history_length")
            metrics.score_ema_enabled = bool(temporal.get("score_ema_enabled"))
            component_summary = dynamic_decision.get("component_summary") or {}
            metrics.num_high_contact_tokens = component_summary.get("num_high_contact_tokens")
            metrics.num_valid_3d_tokens = component_summary.get("num_valid_3d_tokens")
            metrics.phase_far_count = 1 if metrics.dynamic_phase == "low" else 0
            metrics.phase_mid_count = 1 if metrics.dynamic_phase == "medium" else 0
            metrics.phase_near_count = 1 if metrics.dynamic_phase in ("high", "interaction_lock") else 0
            metrics.phase_fallback_safe_count = 0
        if self.config.strategy in EARLY_GEOMETRY_FALLBACK_STRATEGIES and fallback_reason is None:
            strategy = "depth_edge_fast_diverse"
        if self.config.strategy == "robot_geo_contact_budget" and fallback_reason is not None:
            metrics.fallback_used = True
            metrics.keep_ratio_source = "fallback"
            metrics.fallback_reason = fallback_reason
            strategy = "depth_edge_fast_diverse"
            fallback_reason = None
        if self.config.strategy in (ROBOT_STATE_REQUIRED_LEGACY_STRATEGIES - ACGTP_STRATEGIES) and fallback_reason in (
            "missing_robot_state",
            "missing_camera",
            "missing_camera_intrinsics",
            "missing_camera_extrinsics",
        ):
            metrics.fallback_used = True
            metrics.keep_ratio_source = "fallback"
            metrics.fallback_reason = fallback_reason
            strategy = "depth_edge_fast_diverse"
            if self.config.strategy in ("robot_geo_dynamic", "robot_geo_dynamic_v0", "robot_geo_temporal_v0"):
                mid_ratio = float(self.config.dynamic_keep_ratio_config.get("mid_keep_ratio", self.config.keep_ratio_mid))
                keep_count = int(round(num_tokens * mid_ratio))
                metrics.dynamic_enabled = True
                metrics.dynamic_phase = "medium"
                metrics.dynamic_keep_ratio = keep_count / num_tokens if num_tokens else None
                metrics.dynamic_keep_k = keep_count
                metrics.geo_risk_level = "medium"
                metrics.dynamic_keep_reason = fallback_reason
            fallback_reason = None
        if fallback_reason is not None:
            metrics.fallback_used = True
            metrics.keep_ratio_source = "fallback"
            metrics.fallback_reason = fallback_reason
            if self.config.strategy == "robot_geo_dynamic":
                strategy = "uniform_grid"
                keep_count = int(round(num_tokens * self.config.keep_ratio_near))
                reserve_tokens = 0
                metrics.dynamic_enabled = True
                metrics.dynamic_phase = "fallback_safe"
                metrics.dynamic_keep_k = keep_count
                metrics.dynamic_keep_ratio = keep_count / num_tokens if num_tokens else None
                metrics.valid_depth_ratio = metrics.valid_token_ratio
                metrics.phase_far_count = 0
                metrics.phase_mid_count = 0
                metrics.phase_near_count = 0
                metrics.phase_fallback_safe_count = 1
            elif self.config.strategy == "robot_geo_dynamic_v0":
                strategy = "depth_edge_fast_diverse" if scores is not None else "uniform_grid"
                mid_ratio = float(self.config.dynamic_keep_ratio_config.get("mid_keep_ratio", self.config.keep_ratio_mid))
                keep_count = int(round(num_tokens * mid_ratio))
                metrics.dynamic_enabled = True
                metrics.dynamic_phase = "medium"
                metrics.dynamic_keep_ratio = keep_count / num_tokens if num_tokens else None
                metrics.dynamic_keep_k = keep_count
                metrics.geo_risk_level = "medium"
                dynamic_decision = robot_metrics.get("dynamic_decision") or {}
                metrics.dynamic_keep_reason = fallback_reason
                metrics.phase_far_count = 0
                metrics.phase_mid_count = 1
                metrics.phase_near_count = 0
                metrics.phase_fallback_safe_count = 0
            elif self.config.strategy == "robot_geo_temporal_v0":
                strategy = "depth_edge_fast_diverse" if scores is not None else "uniform_grid"
                mid_ratio = float(self.config.dynamic_keep_ratio_config.get("mid_keep_ratio", self.config.keep_ratio_mid))
                keep_count = int(round(num_tokens * mid_ratio))
                metrics.dynamic_enabled = True
                metrics.dynamic_phase = "medium"
                metrics.dynamic_keep_ratio = keep_count / num_tokens if num_tokens else None
                metrics.dynamic_keep_k = keep_count
                metrics.geo_risk_level = "medium"
                metrics.dynamic_keep_reason = fallback_reason
                metrics.interaction_lock = False
                metrics.interaction_lock_reason = "none"
                metrics.history_length = self._temporal_history.history_length
                metrics.phase_far_count = 0
                metrics.phase_mid_count = 1
                metrics.phase_near_count = 0
                metrics.phase_fallback_safe_count = 0
            elif self.config.strategy in ("robot_geo_hybrid_temporal_v1", "robot_geo_hybrid_temporal_edge_reserve_v1"):
                strategy = "depth_edge_fast_diverse" if scores is not None else "uniform_grid"
                mid_ratio = float(self.config.dynamic_keep_ratio_config.get("mid_keep_ratio", self.config.keep_ratio_mid))
                keep_count = int(round(num_tokens * mid_ratio))
                metrics.dynamic_enabled = True
                metrics.dynamic_phase = "medium"
                metrics.dynamic_keep_ratio = keep_count / num_tokens if num_tokens else None
                metrics.dynamic_keep_k = keep_count
                metrics.geo_risk_level = "medium"
                metrics.dynamic_keep_reason = fallback_reason
                metrics.interaction_lock = False
                metrics.interaction_lock_reason = "none"
                metrics.history_length = self._temporal_history.history_length
                metrics.score_ema_enabled = True
                metrics.score_ema_available = False
                metrics.ema_used_for_selection = False
                metrics.lock_condition_failed_reason = fallback_reason
                metrics.adaptive_threshold_mean = None
                metrics.adaptive_threshold_max = None
                metrics.interaction_lock_ratio = 0.0
                metrics.phase_far_count = 0
                metrics.phase_mid_count = 1
                metrics.phase_near_count = 0
                metrics.phase_fallback_safe_count = 0
            else:
                strategy = "none" if self.config.fallback_strategy == "no_pruning" else "uniform_grid"
                keep_count = num_tokens if strategy == "none" else keep_count

        select_start = time.perf_counter()
        if self.config.strategy in ACGTP_STRATEGIES and fallback_reason is not None:
            # ACGTP is a geometry expert. If required geometry inputs are missing,
            # do not run the branch selector on zero scores/all-token fill masks:
            # mark an explicit input fallback so diagnostics cannot be mistaken
            # for successful action-constrained token protection.
            _acgtp_input_fallback_reason = str(fallback_reason)
            _requested_keep_count = int(round(num_tokens * float(self.config.keep_ratio)))
            if scores is not None and valid_mask is not None and _acgtp_input_fallback_reason not in ("missing_depth", "missing_geometry"):
                _fallback_strategy = "depth_edge_fast_diverse"
                _fallback_keep_count = _requested_keep_count
            else:
                _fallback_strategy = "none" if self.config.fallback_strategy == "no_pruning" else "uniform_grid"
                _fallback_keep_count = num_tokens if _fallback_strategy == "none" else _requested_keep_count
            keep_indices_np, selection_meta = select_keep_indices(
                strategy=_fallback_strategy,
                num_tokens=num_tokens,
                keep_count=_fallback_keep_count,
                scores=scores,
                valid_mask=valid_mask,
                seed=self.config.seed,
                grid_size=self.config.token_grid_shape[0],
                cell_grid=self.config.cell_grid,
                reserve_tokens=reserve_tokens,
            )
            _fallback_kept = int(len(keep_indices_np))
            _is_v2 = self.config.strategy == "robot_geo_acgtp_v2"
            metrics.fallback_used = True
            metrics.keep_ratio_source = "fallback"
            metrics.fallback_reason = _acgtp_input_fallback_reason
            selection_meta.update({
                "strategy": _fallback_strategy,
                "requested_acgtp_strategy": self.config.strategy,
                "selector_name": "acgtp_input_fallback",
                "selector_function_name": "select_keep_indices",
                "selection_strategy_name": _fallback_strategy,
                "selection_stage_name": "acgtp_input_fallback",
                "fallback_used": True,
                "fallback_reason": _acgtp_input_fallback_reason,
                "input_fallback_used": True,
                "input_fallback_reason": _acgtp_input_fallback_reason,
                "acgtp_v1": not _is_v2,
                "acgtp_v2": _is_v2,
                "acgtp_fallback_used": True,
                "acgtp_fallback_reason": _acgtp_input_fallback_reason,
                "acgtp_hard_protect_count": 0,
                "acgtp_hard_protect_valid": False,
                "selected_by_scene_layout_count": 0,
                "selected_by_depth_structure_count": 0,
                "selected_by_contact_ring_count": 0,
                "selected_by_motion_corridor_count": 0,
                "selected_by_constrained_fill_count": 0,
                "selected_by_acgtp_fallback_count": _fallback_kept,
                "selected_by_semantic_count": 0,
                "selected_by_attention_count": 0,
                "selected_by_fallback": _fallback_kept,
                "selected_by_fill": 0,
                "selected_unattributed": 0,
                "acgtp_branch_accounting_valid": True,
                "acgtp_branch_sum": _fallback_kept,
                "acgtp_branch_sum_error": 0,
                "branch_accounting_valid": True,
                "branch_sum_equals_kept": True,
                "final_kept": _fallback_kept,
                "expected_kept": _fallback_keep_count,
                "K_total": _fallback_keep_count,
                "acgtp_final_kept": _fallback_kept,
                "acgtp_expected_kept": _fallback_keep_count,
                "acgtp_actual_keep_ratio": (_fallback_kept / num_tokens) if num_tokens else 0.0,
            })
            if _is_v2:
                selection_meta.update({
                    "strict_fallback_dispatch_used": False,
                    "delegated_selector_name": None,
                    "fallback_dispatch_to_v1": False,
                    "acgtp_v2_semantic_enabled": bool(self.config.acgtp_v2_semantic_enabled),
                    "acgtp_v2_semantic_backend": str(self.config.acgtp_v2_semantic_backend),
                    "acgtp_v2_semantic_confidence": 0.0,
                    "acgtp_v2_semantic_unavailable": True,
                    "acgtp_v2_semantic_fallback_reason": _acgtp_input_fallback_reason,
                    "acgtp_v2_release_quota": True,
                    "semantic_backend": str(self.config.acgtp_v2_semantic_backend),
                    "semantic_available": False,
                    "semantic_unavailable": True,
                    "semantic_confidence": 0.0,
                    "semantic_quota_released": True,
                    "attention_backend": "none",
                    "attention_source": "none",
                    "attention_available": False,
                    "attention_confidence": 0.0,
                    "attention_quota_released": True,
                    "attention_only_token_count": 0,
                    "attention_selected_by_final_count": 0,
                    "attention_top_count": 0,
                    "high_attention_low_geometry_count": 0,
                    "high_geometry_low_attention_count": 0,
                })
            fallback_reason = None
        if self.config.strategy in ("depth_edge_fast", "depth_edge_fast_diverse") and fallback_reason is None:
            keep_indices_np, selection_meta = select_keep_indices(
                strategy=self.config.strategy,
                num_tokens=num_tokens,
                keep_count=keep_count,
                scores=scores,
                valid_mask=valid_mask,
                seed=self.config.seed,
                grid_size=self.config.token_grid_shape[0],
                cell_grid=self.config.cell_grid,
                reserve_tokens=reserve_tokens,
            )
            selection_meta["fallback_used"] = False
            selection_meta["fallback_reason"] = None
            selection_meta["requested_keep_ratio"] = float(self.config.keep_ratio)
            selection_meta["keep_ratio_source"] = "cli_keep_ratio"
            selection_meta["effective_keep_count"] = int(len(keep_indices_np))
            selection_meta["original_token_count"] = int(num_tokens)
            metrics.keep_ratio_source = "cli_keep_ratio"
            self._record_selection_path_diagnostics(
                metrics=metrics,
                selector_success=True,
                exc=None,
                requested_strategy=self.config.strategy,
                effective_strategy=self.config.strategy,
                selector_name="select_keep_indices",
                fallback_selector=None,
                keep_indices_np=keep_indices_np,
                num_tokens=num_tokens,
            )
        elif self.config.strategy == "robot_geo_contact_budget" and strategy == "robot_geo_contact_budget" and fallback_reason is None:
            k_edge, k_geo, k_diverse = self._contact_budget_counts(keep_count)
            keep_indices_np, selection_meta = select_tokens_contact_budget(
                edge_score=robot_metrics.get("edge_scores"),
                geo_contact_score=robot_metrics.get("geo_contact_scores", scores),
                valid_mask=valid_mask,
                keep_k=keep_count,
                k_edge=k_edge,
                k_geo=k_geo,
                k_diverse=k_diverse,
                grid_h=self.config.token_grid_shape[0],
                grid_w=self.config.token_grid_shape[1],
                cells_h=self.config.cell_grid,
                cells_w=self.config.cell_grid,
                return_indices=bool(self.cfg.get("save_pruning_vis", False) or self.cfg.get("save_pruning_debug", False)),
                detailed_timing=self.config.detailed_pruning_timing,
            )
            self._record_selection_path_diagnostics(
                metrics=metrics,
                selector_success=True,
                exc=None,
                requested_strategy=self.config.strategy,
                effective_strategy=self.config.strategy,
                selector_name="select_tokens_contact_budget",
                fallback_selector=None,
                keep_indices_np=keep_indices_np,
                num_tokens=num_tokens,
            )
        elif self.config.strategy == "robot_geo_hybrid_v0":
            if fallback_reason is not None:
                metrics.fallback_used = True
                metrics.keep_ratio_source = "fallback"
                metrics.fallback_reason = fallback_reason
                strategy = "depth_edge_fast_diverse"
                keep_count = int(round(num_tokens * 0.75))
            else:
                try:
                    keep_indices_np, selection_meta = select_hybrid_quota_v2(
                        depth_edge_scores=robot_metrics.get("edge_scores"),
                        contact_risk_scores=robot_metrics.get("rule_v0_contact_risk_scores"),
                        distance_scores=robot_metrics.get("near_scores"),
                        motion_cone_scores=robot_metrics.get("rule_v0_motion_cone_scores"),
                        valid_mask=valid_mask,
                        keep_k=keep_count,
                        grid_h=self.config.token_grid_shape[0],
                        grid_w=self.config.token_grid_shape[1],
                        cell_grid=self.config.cell_grid,
                        seed=self.config.seed,
                    )
                    metrics.dynamic_enabled = True
                    metrics.dynamic_phase = "hybrid_v2"
                    metrics.dynamic_keep_ratio = keep_count / num_tokens if num_tokens else None
                    metrics.dynamic_keep_k = keep_count
                    metrics.dynamic_keep_reason = "hybrid_quota_v2"
                    metrics.phase_far_count = 0
                    metrics.phase_mid_count = 0
                    metrics.phase_near_count = 1
                    metrics.phase_fallback_safe_count = 0
                    self._record_selection_path_diagnostics(
                        metrics=metrics,
                        selector_success=True,
                        exc=None,
                        requested_strategy=self.config.strategy,
                        effective_strategy=self.config.strategy,
                        selector_name="select_hybrid_quota_v2",
                        fallback_selector=None,
                        keep_indices_np=keep_indices_np,
                        num_tokens=num_tokens,
                    )
                    for key in (
                        "selected_by_depth_edge_count", "selected_by_contact_count",
                        "selected_by_distance_contact_count", "selected_by_motion_count",
                        "selected_by_uniform_count", "selected_by_fill_count",
                        "K_depth_edge_quota", "K_contact_quota", "K_distance_contact_quota",
                        "K_motion_quota", "K_uniform_quota", "K_fill_quota",
                        "K_depth_edge_actual", "K_contact_actual", "K_distance_contact_actual",
                        "K_motion_actual", "K_uniform_actual", "K_fill_actual",
                        "motion_gate_tokens_total", "motion_gate_tokens_selected",
                        "motion_gate_effective", "overlap_depth_contact",
                        "overlap_depth_dist_contact", "overlap_depth_motion",
                    ):
                        if key in selection_meta:
                            setattr(metrics, key, selection_meta.get(key))
                except Exception as exc:
                    metrics.fallback_used = True
                    metrics.keep_ratio_source = "fallback"
                    metrics.fallback_reason = f"hybrid_v0_error:{type(exc).__name__}"
                    strategy = "depth_edge_fast_diverse"
                    keep_count = int(round(num_tokens * 0.75))
                    self._record_selection_path_diagnostics(
                        metrics=metrics,
                        selector_success=False,
                        exc=exc,
                        requested_strategy=self.config.strategy,
                        effective_strategy="depth_edge_fast_diverse",
                        selector_name="select_hybrid_quota_v2",
                        fallback_selector="depth_edge_fast_diverse",
                        keep_indices_np=None,
                        num_tokens=num_tokens,
                    )

        elif self.config.strategy == "robot_geo_hybrid_dynamic_v0":
            decision = robot_metrics.get("dynamic_decision") or {}
            dynamic_ratio = decision.get("keep_ratio")
            if dynamic_ratio is not None:
                keep_count = max(1, min(num_tokens, int(round(num_tokens * float(dynamic_ratio)))))
            if fallback_reason is not None:
                metrics.fallback_used = True
                metrics.keep_ratio_source = "fallback"
                metrics.fallback_reason = fallback_reason
                strategy = "depth_edge_fast_diverse"
                keep_count = int(round(num_tokens * 0.75))
                metrics.dynamic_enabled = True
                metrics.dynamic_phase = "medium"
                metrics.dynamic_keep_ratio = keep_count / num_tokens if num_tokens else None
                metrics.dynamic_keep_k = keep_count
                metrics.dynamic_keep_reason = fallback_reason
            else:
                try:
                    keep_indices_np, selection_meta = select_hybrid_quota_v2(
                        depth_edge_scores=robot_metrics.get("edge_scores"),
                        contact_risk_scores=robot_metrics.get("rule_v0_contact_risk_scores"),
                        distance_scores=robot_metrics.get("near_scores"),
                        motion_cone_scores=robot_metrics.get("rule_v0_motion_cone_scores"),
                        valid_mask=valid_mask,
                        keep_k=keep_count,
                        grid_h=self.config.token_grid_shape[0],
                        grid_w=self.config.token_grid_shape[1],
                        cell_grid=self.config.cell_grid,
                        seed=self.config.seed,
                    )
                    metrics.dynamic_enabled = True
                    metrics.dynamic_phase = f"hybrid_dynamic_{decision.get('risk_level', 'medium')}"
                    metrics.dynamic_keep_ratio = keep_count / num_tokens if num_tokens else None
                    metrics.dynamic_keep_k = keep_count
                    metrics.geo_risk_level = decision.get("risk_level")
                    metrics.geo_risk_score = decision.get("risk_score")
                    metrics.dynamic_keep_reason = f"hybrid_dynamic_{decision.get('reason', 'dynamic')}"
                    component_summary = decision.get("component_summary", {}) or {}
                    metrics.num_high_contact_tokens = component_summary.get("num_high_contact_tokens")
                    metrics.num_valid_3d_tokens = component_summary.get("num_valid_3d_tokens")
                    self._record_selection_path_diagnostics(
                        metrics=metrics,
                        selector_success=True,
                        exc=None,
                        requested_strategy=self.config.strategy,
                        effective_strategy=self.config.strategy,
                        selector_name="select_hybrid_quota_v2",
                        fallback_selector=None,
                        keep_indices_np=keep_indices_np,
                        num_tokens=num_tokens,
                    )
                    # All hybrid_dynamic fields propagated by _apply_selection_metrics() below
                except Exception as exc:
                    metrics.fallback_used = True
                    metrics.keep_ratio_source = "fallback"
                    metrics.fallback_reason = f"hybrid_dynamic_error:{type(exc).__name__}"
                    strategy = "depth_edge_fast_diverse"
                    keep_count = int(round(num_tokens * 0.75))
                    metrics.dynamic_enabled = True
                    metrics.dynamic_phase = "medium"
                    metrics.dynamic_keep_ratio = keep_count / num_tokens if num_tokens else None
                    metrics.dynamic_keep_k = keep_count
                    metrics.dynamic_keep_reason = f"hybrid_dynamic_fallback:{type(exc).__name__}"
                    self._record_selection_path_diagnostics(
                        metrics=metrics,
                        selector_success=False,
                        exc=exc,
                        requested_strategy=self.config.strategy,
                        effective_strategy="depth_edge_fast_diverse",
                        selector_name="select_hybrid_quota_v2",
                        fallback_selector="depth_edge_fast_diverse",
                        keep_indices_np=None,
                        num_tokens=num_tokens,
                    )

        if self.config.strategy == "robot_geo_hybrid_v1":
            if fallback_reason is not None:
                metrics.fallback_used = True
                metrics.keep_ratio_source = "fallback"
                metrics.fallback_reason = fallback_reason
                strategy = "depth_edge_fast_diverse"
                keep_count = int(round(num_tokens * 0.75))
            else:
                try:
                    hybrid_weights = self.config.hybrid_v1_weights
                    keep_indices_np, selection_meta = select_hybrid_v1(
                        depth_edge_scores=robot_metrics.get("edge_scores"),
                        near_scores=robot_metrics.get("near_scores"),
                        contact_risk_scores=robot_metrics.get("rule_v0_contact_risk_scores"),
                        corridor_scores=robot_metrics.get("rule_v0_motion_cone_scores"),
                        valid_mask=valid_mask,
                        keep_k=keep_count,
                        grid_h=self.config.token_grid_shape[0],
                        grid_w=self.config.token_grid_shape[1],
                        cell_grid=self.config.cell_grid,
                        seed=self.config.seed,
                        w_edge=float(hybrid_weights.get("w_edge", 0.45)),
                        w_near=float(hybrid_weights.get("w_near", 0.20)),
                        w_contact=float(hybrid_weights.get("w_contact", 0.20)),
                        w_corr=float(hybrid_weights.get("w_corr", 0.10)),
                        w_diverse=float(hybrid_weights.get("w_diverse", 0.05)),
                    )
                    metrics.dynamic_enabled = True
                    metrics.keep_ratio_source = "cli_keep_ratio"
                    metrics.dynamic_phase = "hybrid_v1"
                    metrics.dynamic_keep_ratio = keep_count / num_tokens if num_tokens else None
                    metrics.dynamic_keep_k = keep_count
                    metrics.dynamic_keep_reason = "hybrid_v1_weighted_score"
                    metrics.phase_far_count = 0
                    metrics.phase_mid_count = 0
                    metrics.phase_near_count = 1
                    metrics.phase_fallback_safe_count = 0
                    self._record_selection_path_diagnostics(
                        metrics=metrics,
                        selector_success=True,
                        exc=None,
                        requested_strategy=self.config.strategy,
                        effective_strategy=self.config.strategy,
                        selector_name="select_hybrid_v1",
                        fallback_selector=None,
                        keep_indices_np=keep_indices_np,
                        num_tokens=num_tokens,
                    )
                    # All hybrid v1 fields (score stats, weights, grid coverage, entropy)
                    # are propagated by _apply_selection_metrics() below
                except Exception as exc:
                    metrics.fallback_used = True
                    metrics.keep_ratio_source = "fallback"
                    metrics.fallback_reason = f"hybrid_v1_error:{type(exc).__name__}"
                    strategy = "depth_edge_fast_diverse"
                    keep_count = int(round(num_tokens * 0.75))
                    # robot_geo_hybrid_v1 is a SELF_HANDLED selector strategy, so the
                    # generic select_keep_indices() rescue further down is skipped.
                    # Generate the depth_edge_fast_diverse fallback selection here
                    # (mirroring the hybrid_temporal_v1 branch) so keep_indices_np is
                    # never left None on the error path.
                    keep_indices_np, selection_meta = select_keep_indices(
                        strategy=strategy,
                        num_tokens=num_tokens,
                        keep_count=keep_count,
                        scores=scores,
                        valid_mask=valid_mask,
                        seed=self.config.seed,
                        grid_size=self.config.token_grid_shape[0],
                        cell_grid=self.config.cell_grid,
                        reserve_tokens=reserve_tokens,
                    )
                    selection_meta["fallback_used"] = True
                    selection_meta["fallback_reason"] = metrics.fallback_reason
                    selection_meta["actual_keep_ratio"] = len(keep_indices_np) / num_tokens if num_tokens else 0.0
                    self._record_selection_path_diagnostics(
                        metrics=metrics,
                        selector_success=False,
                        exc=exc,
                        requested_strategy=self.config.strategy,
                        effective_strategy="depth_edge_fast_diverse",
                        selector_name="select_hybrid_v1",
                        fallback_selector="depth_edge_fast_diverse",
                        keep_indices_np=keep_indices_np,
                        num_tokens=num_tokens,
                    )

        if self.config.strategy == "robot_geo_hybrid_temporal_v1":
            if fallback_reason is not None:
                metrics.fallback_used = True
                metrics.keep_ratio_source = "fallback"
                metrics.fallback_reason = fallback_reason
                strategy = "depth_edge_fast_diverse"
                keep_count = int(round(num_tokens * 0.75))
                # ── Actually generate a valid selection ────────────────────────
                keep_indices_np, selection_meta = select_keep_indices(
                    strategy=strategy,
                    num_tokens=num_tokens,
                    keep_count=keep_count,
                    scores=scores,
                    valid_mask=valid_mask,
                    seed=self.config.seed,
                    grid_size=self.config.token_grid_shape[0],
                    cell_grid=self.config.cell_grid,
                    reserve_tokens=reserve_tokens,
                )
                selection_meta["fallback_used"] = True
                selection_meta["fallback_reason"] = fallback_reason
                selection_meta["ema_used_for_selection"] = False
                selection_meta["interaction_lock"] = False
                selection_meta["lock_reason"] = fallback_reason
                selection_meta["actual_keep_ratio"] = len(keep_indices_np) / num_tokens if num_tokens else 0.0
                # Fallback: dynamic_keep_ratio derived from actual selection, NOT from config
                dynamic_keep_ratio_fb = len(keep_indices_np) / num_tokens if num_tokens else 0.75
                metrics.dynamic_enabled = True
                metrics.dynamic_phase = "medium"
                metrics.dynamic_keep_ratio = dynamic_keep_ratio_fb
                metrics.dynamic_keep_k = len(keep_indices_np)
                metrics.geo_risk_level = "medium"
                metrics.dynamic_keep_reason = fallback_reason
                metrics.interaction_lock = False
                metrics.interaction_lock_reason = "none"
                metrics.history_length = self._temporal_history.history_length
                metrics.score_ema_enabled = True
                metrics.score_ema_available = False
                metrics.ema_used_for_selection = False
                metrics.lock_condition_failed_reason = fallback_reason
                metrics.adaptive_threshold_mean = None
                metrics.adaptive_threshold_max = None
                metrics.interaction_lock_ratio = 0.0
                metrics.phase_far_count = 0
                metrics.phase_mid_count = 1
                metrics.phase_near_count = 0
                metrics.phase_fallback_safe_count = 0
                self._record_selection_path_diagnostics(
                    metrics=metrics,
                    selector_success=True,
                    exc=None,
                    requested_strategy=self.config.strategy,
                    effective_strategy="depth_edge_fast_diverse",
                    selector_name="select_keep_indices",
                    fallback_selector="depth_edge_fast_diverse",
                    keep_indices_np=keep_indices_np,
                    num_tokens=num_tokens,
                )
            else:
                contact_risk_arr = robot_metrics.get("rule_v0_contact_risk_scores")
                contact_vals = np.asarray(contact_risk_arr, dtype=np.float32).reshape(-1) if contact_risk_arr is not None else None
                adaptive_min = float(self.config.temporal_adaptive_threshold_min)
                adaptive_pct = float(self.config.temporal_adaptive_threshold_percentile)
                adaptive_threshold = adaptive_min
                if contact_vals is not None and np.any(np.isfinite(contact_vals)):
                    pct_val = float(np.percentile(contact_vals[np.isfinite(contact_vals)], adaptive_pct))
                    adaptive_threshold = max(adaptive_min, pct_val)
                dynamic_ratio = (robot_metrics.get("dynamic_decision") or {}).get("keep_ratio")
                if dynamic_ratio is None:
                    dynamic_ratio = float(self.config.keep_ratio)
                    metrics.keep_ratio_source = "cli_keep_ratio"
                else:
                    metrics.keep_ratio_source = "dynamic_decision"
                temporal = self._temporal_history.detect_interaction_lock(
                    contact_risk_score=contact_risk_arr,
                    final_scores=scores,
                    motion_direction=robot_metrics.get("motion_direction"),
                    valid_3d_ratio=robot_metrics.get("valid_token_ratio"),
                    dynamic_keep_ratio=dynamic_ratio,
                    config=self.config,
                    gripper_pos=robot_metrics.get("gripper_pos"),
                    token_points_robot=robot_metrics.get("token_points_robot"),
                    adaptive_threshold=adaptive_threshold,
                )
                interaction_locked = bool(temporal.get("interaction_lock"))
                ema_available = bool(temporal.get("score_ema_enabled"))
                if interaction_locked:
                    lock_ratio = float(self.config.temporal_interaction_lock_conservative_ratio)
                    chosen_ratio = max(float(dynamic_ratio), lock_ratio)
                    metrics.keep_ratio_source = "dynamic_decision"
                else:
                    chosen_ratio = float(dynamic_ratio)
                keep_count = max(1, min(num_tokens, int(round(num_tokens * chosen_ratio))))
                _select_ok = False
                try:
                    hybrid_weights = self.config.hybrid_v1_weights
                    keep_indices_np, selection_meta = select_hybrid_v1(
                        depth_edge_scores=robot_metrics.get("edge_scores"),
                        near_scores=robot_metrics.get("near_scores"),
                        contact_risk_scores=contact_risk_arr,
                        corridor_scores=robot_metrics.get("rule_v0_motion_cone_scores"),
                        valid_mask=valid_mask,
                        keep_k=keep_count,
                        grid_h=self.config.token_grid_shape[0],
                        grid_w=self.config.token_grid_shape[1],
                        cell_grid=self.config.cell_grid,
                        seed=self.config.seed,
                        w_edge=float(hybrid_weights.get("w_edge", 0.45)),
                        w_near=float(hybrid_weights.get("w_near", 0.20)),
                        w_contact=float(hybrid_weights.get("w_contact", 0.20)),
                        w_corr=float(hybrid_weights.get("w_corr", 0.10)),
                        w_diverse=float(hybrid_weights.get("w_diverse", 0.05)),
                    )
                    _select_ok = True
                    self._record_selection_path_diagnostics(
                        metrics=metrics,
                        selector_success=True,
                        exc=None,
                        requested_strategy=self.config.strategy,
                        effective_strategy=self.config.strategy,
                        selector_name="select_hybrid_v1",
                        fallback_selector=None,
                        keep_indices_np=keep_indices_np,
                        num_tokens=num_tokens,
                    )
                except Exception as exc:
                    print(f"[PRUNING] select_hybrid_v1 exception (step={self._hook_step_counter}): {type(exc).__name__}: {exc}")
                    metrics.fallback_used = True
                    metrics.keep_ratio_source = "fallback"
                    metrics.fallback_reason = f"hybrid_temporal_v1_select_error:{type(exc).__name__}"
                    strategy = "depth_edge_fast_diverse"
                    keep_count = int(round(num_tokens * 0.75))
                    keep_indices_np = None
                    selection_meta = {}
                    self._record_selection_path_diagnostics(
                        metrics=metrics,
                        selector_success=False,
                        exc=exc,
                        requested_strategy=self.config.strategy,
                        effective_strategy="depth_edge_fast_diverse",
                        selector_name="select_hybrid_v1",
                        fallback_selector="depth_edge_fast_diverse",
                        keep_indices_np=None,
                        num_tokens=num_tokens,
                    )
                    metrics.dynamic_enabled = True
                    metrics.dynamic_phase = "medium"
                    metrics.dynamic_keep_ratio = keep_count / num_tokens if num_tokens else None
                    metrics.dynamic_keep_k = keep_count
                    metrics.interaction_lock = False
                    metrics.interaction_lock_reason = "select_error"
                    metrics.score_ema_enabled = True
                    metrics.score_ema_available = False
                    metrics.ema_used_for_selection = False
                    metrics.lock_condition_failed_reason = "select_error"
                    metrics.adaptive_threshold_mean = None
                    metrics.adaptive_threshold_max = None
                    metrics.interaction_lock_ratio = 0.0
                if _select_ok:
                    metrics.dynamic_enabled = True
                    metrics.dynamic_phase = "interaction_lock" if interaction_locked else "hybrid_temporal_v1"
                    metrics.dynamic_keep_ratio = keep_count / num_tokens if num_tokens else None
                    metrics.dynamic_keep_k = keep_count
                    metrics.geo_risk_level = metrics.dynamic_phase
                    metrics.dynamic_keep_reason = "temporal_interaction_lock" if interaction_locked else "hybrid_temporal_v1"
                    metrics.interaction_lock = interaction_locked
                    metrics.interaction_lock_reason = str(temporal.get("lock_reason", "none"))
                    metrics.temporal_stability = temporal.get("temporal_stability")
                    metrics.history_length = temporal.get("history_length")
                    metrics.score_ema_enabled = ema_available
                    metrics.score_ema_available = ema_available
                    # EMA is computed for diagnostics only in temporal_v1; selection still
                    # uses the current-frame hybrid score, so do not log it as used.
                    metrics.ema_used_for_selection = False
                    metrics.lock_condition_failed_reason = str(temporal.get("reason", "none"))
                    metrics.topk_contact_lock = bool(temporal.get("topk_contact_lock"))
                    metrics.elevated_current_lock = bool(temporal.get("elevated_current_lock"))
                    metrics.gripper_lock = bool(temporal.get("gripper_lock"))
                    metrics.region_lock = bool(temporal.get("region_lock"))
                    metrics.adaptive_threshold_mean = adaptive_threshold
                    metrics.adaptive_threshold_max = adaptive_threshold
                    metrics.interaction_lock_ratio = 1.0 if interaction_locked else 0.0
                    component_summary = (robot_metrics.get("dynamic_decision") or {}).get("component_summary", {}) or {}
                    metrics.num_high_contact_tokens = component_summary.get("num_high_contact_tokens")
                    metrics.num_valid_3d_tokens = component_summary.get("num_valid_3d_tokens")
                    metrics.phase_far_count = 0
                    metrics.phase_mid_count = 0
                    metrics.phase_near_count = 1 if interaction_locked else 0
                    metrics.phase_fallback_safe_count = 0
                    # All hybrid v1 fields propagated by _apply_selection_metrics() below

        # P5/P6: robot_geo_hybrid_temporal_edge_reserve variants — targeted edge-reserve ablation
        if self.config.strategy in EDGE_RESERVE_LEGACY_STRATEGIES:
            # P6: compute edge_reserve_k from edge_reserve_ratio if set
            if self.config.edge_reserve_ratio > 0:
                _computed_edge_reserve_k = int(round(self.config.edge_reserve_ratio * keep_count))
            else:
                _computed_edge_reserve_k = int(self.config.edge_reserve_k)
            edge_reserve_k = max(0, min(_computed_edge_reserve_k, keep_count))
            if fallback_reason is not None:
                metrics.fallback_used = True
                metrics.keep_ratio_source = "fallback"
                metrics.fallback_reason = fallback_reason
                strategy = "depth_edge_fast_diverse"
                keep_count = int(round(num_tokens * 0.75))
                keep_indices_np, selection_meta = select_keep_indices(
                    strategy=strategy,
                    num_tokens=num_tokens,
                    keep_count=keep_count,
                    scores=scores,
                    valid_mask=valid_mask,
                    seed=self.config.seed,
                    grid_size=self.config.token_grid_shape[0],
                    cell_grid=self.config.cell_grid,
                    reserve_tokens=reserve_tokens,
                )
                selection_meta["fallback_used"] = True
                selection_meta["fallback_reason"] = fallback_reason
                selection_meta["edge_reserve_enabled"] = False
                selection_meta["edge_reserve_ratio"] = 0.0
                selection_meta["edge_reserved_target_count"] = 0
                selection_meta["edge_reserved_actual_count"] = 0
                selection_meta["edge_reserved_survival_ratio"] = None
                selection_meta["final_selected_count"] = int(len(keep_indices_np))
                selection_meta["selected_by_edge_reserved_count"] = 0
                selection_meta["selected_by_original_hybrid_count"] = 0
                selection_meta["selected_by_fill_count"] = int(len(keep_indices_np))
                selection_meta["duplicate_edge_hybrid_count"] = 0
                selection_meta["duplicate_after_exclusion_count"] = 0
                selection_meta["duplicate_with_original_hybrid_count"] = 0
                selection_meta["edge_scores_available"] = False
                selection_meta["edge_scores_shape"] = None
                selection_meta["edge_scores_finite_ratio"] = None
                selection_meta["edge_reserve_invalid"] = True
                selection_meta["edge_reserve_invalid_reason"] = "fallback_triggered"
                selection_meta["reserved_edge_topk_count"] = None
                selection_meta["reserved_edge_kept_count"] = None
                selection_meta["reserved_edge_dropped_count"] = None
                selection_meta["reserved_edge_topk_dropped_ratio"] = None
                selection_meta["non_reserved_edge_topk_count"] = None
                selection_meta["non_reserved_edge_kept_count"] = None
                selection_meta["non_reserved_edge_dropped_count"] = None
                selection_meta["non_reserved_edge_topk_dropped_ratio"] = None
                selection_meta["overall_depth_edge_topk_count"] = None
                selection_meta["overall_depth_edge_topk_kept_count"] = None
                selection_meta["overall_depth_edge_topk_dropped_count"] = None
                selection_meta["overall_depth_edge_topk_dropped_ratio"] = None
                metrics.dynamic_enabled = True
                metrics.dynamic_phase = "medium"
                metrics.dynamic_keep_ratio = len(keep_indices_np) / num_tokens if num_tokens else 0.75
                metrics.dynamic_keep_k = len(keep_indices_np)
                metrics.geo_risk_level = "medium"
                metrics.dynamic_keep_reason = fallback_reason
                metrics.interaction_lock = False
                metrics.score_ema_enabled = True
                metrics.score_ema_available = False
                metrics.ema_used_for_selection = False
                self._record_selection_path_diagnostics(
                    metrics=metrics,
                    selector_success=True,
                    exc=None,
                    requested_strategy=self.config.strategy,
                    effective_strategy="depth_edge_fast_diverse",
                    selector_name="select_keep_indices",
                    fallback_selector="depth_edge_fast_diverse",
                    keep_indices_np=keep_indices_np,
                    num_tokens=num_tokens,
                )
            else:
                contact_risk_arr = robot_metrics.get("rule_v0_contact_risk_scores")
                contact_vals = np.asarray(contact_risk_arr, dtype=np.float32).reshape(-1) if contact_risk_arr is not None else None
                adaptive_min = float(self.config.temporal_adaptive_threshold_min)
                adaptive_pct = float(self.config.temporal_adaptive_threshold_percentile)
                adaptive_threshold = adaptive_min
                if contact_vals is not None and np.any(np.isfinite(contact_vals)):
                    pct_val = float(np.percentile(contact_vals[np.isfinite(contact_vals)], adaptive_pct))
                    adaptive_threshold = max(adaptive_min, pct_val)
                dynamic_ratio = (robot_metrics.get("dynamic_decision") or {}).get("keep_ratio")
                if dynamic_ratio is None:
                    dynamic_ratio = float(self.config.keep_ratio)
                    metrics.keep_ratio_source = "cli_keep_ratio"
                else:
                    metrics.keep_ratio_source = "dynamic_decision"
                temporal = self._temporal_history.detect_interaction_lock(
                    contact_risk_score=contact_risk_arr,
                    final_scores=scores,
                    motion_direction=robot_metrics.get("motion_direction"),
                    valid_3d_ratio=robot_metrics.get("valid_token_ratio"),
                    dynamic_keep_ratio=dynamic_ratio,
                    config=self.config,
                    gripper_pos=robot_metrics.get("gripper_pos"),
                    token_points_robot=robot_metrics.get("token_points_robot"),
                    adaptive_threshold=adaptive_threshold,
                )
                interaction_locked = bool(temporal.get("interaction_lock"))
                ema_available = bool(temporal.get("score_ema_enabled"))
                if interaction_locked:
                    lock_ratio = float(self.config.temporal_interaction_lock_conservative_ratio)
                    chosen_ratio = max(float(dynamic_ratio), lock_ratio)
                    metrics.keep_ratio_source = "dynamic_decision"
                else:
                    chosen_ratio = float(dynamic_ratio)
                keep_count = max(1, min(num_tokens, int(round(num_tokens * chosen_ratio))))
                _select_ok = False
                # P5-fix BUG A: Diagnostic + invariant check before select_hybrid_v1_edge_reserve
                _edge_scores_raw = robot_metrics.get("edge_scores")
                _edge_arr = np.asarray(_edge_scores_raw, dtype=np.float32).reshape(-1) if _edge_scores_raw is not None else None
                _edge_scores_available = _edge_arr is not None and _edge_arr.size > 0
                _edge_scores_shape = int(_edge_arr.size) if _edge_arr is not None else None
                _edge_finite_ratio = float(np.mean(np.isfinite(_edge_arr))) if _edge_arr is not None and _edge_arr.size > 0 else None
                if self.config.debug or self.config.edge_reserve_k > 0:
                    print(f"[EDGE_RESERVE] step={self._hook_step_counter}: "
                          f"edge_scores_available={_edge_scores_available}, "
                          f"edge_scores_shape={_edge_scores_shape}, "
                          f"edge_scores_finite_ratio={_edge_finite_ratio}, "
                          f"edge_reserve_k={edge_reserve_k}")
                try:
                    hybrid_weights = self.config.hybrid_v1_weights
                    # P14-B: Apply robot-self mask to near_scores before passing to selector.
                    _near_scores_for_select = robot_metrics.get("near_scores")
                    _self_mask_diag_er: Dict[str, Any] = {}
                    _self_mask_enabled_er = self.config.robot_self_mask_enabled
                    metrics.robot_self_mask_enabled = _self_mask_enabled_er
                    _gripper_pixel_er = robot_metrics.get("gripper_pixel")
                    _token_u_er = None
                    _token_v_er = None
                    if "cache" in robot_metrics and robot_metrics["cache"]:
                        _token_u_er = robot_metrics["cache"].get("u")
                        _token_v_er = robot_metrics["cache"].get("v")
                    _self_mask_arr_er = np.zeros(num_tokens, dtype=bool)
                    if _self_mask_enabled_er and _near_scores_for_select is not None and self.config.robot_self_mask_apply_to_near_score:
                        _near_arr_er = np.asarray(_near_scores_for_select, dtype=np.float32).reshape(-1)
                        _near_masked_er, _self_mask_arr_er, _self_mask_diag_er = self._apply_self_mask(
                            scores=_near_arr_er,
                            gripper_pixel=_gripper_pixel_er,
                            token_u=_token_u_er,
                            token_v=_token_v_er,
                            core_radius_px=self.config.robot_self_mask_core_radius_px,
                        )
                        _near_scores_for_select = _near_masked_er
                    keep_indices_np, selection_meta = select_hybrid_v1_edge_reserve(
                        depth_edge_scores=robot_metrics.get("edge_scores"),
                        near_scores=_near_scores_for_select,
                        contact_risk_scores=contact_risk_arr,
                        corridor_scores=robot_metrics.get("rule_v0_motion_cone_scores"),
                        valid_mask=valid_mask,
                        keep_k=keep_count,
                        edge_reserve_k=edge_reserve_k,
                        grid_h=self.config.token_grid_shape[0],
                        grid_w=self.config.token_grid_shape[1],
                        cell_grid=self.config.cell_grid,
                        seed=self.config.seed,
                        w_edge=float(hybrid_weights.get("w_edge", 0.45)),
                        w_near=float(hybrid_weights.get("w_near", 0.20)),
                        w_contact=float(hybrid_weights.get("w_contact", 0.20)),
                        w_corr=float(hybrid_weights.get("w_corr", 0.10)),
                        w_diverse=float(hybrid_weights.get("w_diverse", 0.05)),
                    )
                    _select_ok = True
                    # P14-B: Record self-mask diagnostics for edge_reserve
                    metrics.self_mask_available = _self_mask_diag_er.get("self_mask_available", False)
                    metrics.self_mask_core_radius_px = _self_mask_diag_er.get("self_mask_core_radius_px")
                    metrics.self_mask_token_count = _self_mask_diag_er.get("self_mask_token_count", 0)
                    metrics.self_mask_token_ratio = _self_mask_diag_er.get("self_mask_token_ratio", 0.0)
                    metrics.gripper_pixel_u = _self_mask_diag_er.get("gripper_pixel_u")
                    metrics.gripper_pixel_v = _self_mask_diag_er.get("gripper_pixel_v")
                    metrics.gripper_pixel_in_bounds = _self_mask_diag_er.get("gripper_pixel_in_bounds", False)
                    metrics.gripper_projection_valid = _self_mask_diag_er.get("gripper_projection_valid", False)
                    metrics.near_score_mean_before_self_mask = _self_mask_diag_er.get("near_score_mean_before_self_mask")
                    metrics.near_score_mean_after_self_mask = _self_mask_diag_er.get("near_score_mean_after_self_mask")
                    metrics.near_score_self_region_mean = _self_mask_diag_er.get("near_score_self_region_mean")
                    metrics.near_score_nonself_region_mean = _self_mask_diag_er.get("near_score_nonself_region_mean")
                    if keep_indices_np is not None and _self_mask_arr_er is not None and len(_self_mask_arr_er) > 0:
                        _sel_arr_er = np.asarray(keep_indices_np, dtype=np.int64)
                        _sel_self_er = int(np.sum(_self_mask_arr_er[_sel_arr_er])) if _sel_arr_er.size > 0 else 0
                        metrics.selected_self_mask_token_count = _sel_self_er
                        metrics.selected_self_mask_token_ratio = float(_sel_self_er / len(_sel_arr_er)) if len(_sel_arr_er) > 0 else 0.0
                    self._record_selection_path_diagnostics(
                        metrics=metrics,
                        selector_success=True,
                        exc=None,
                        requested_strategy=self.config.strategy,
                        effective_strategy=self.config.strategy,
                        selector_name="select_hybrid_v1_edge_reserve",
                        fallback_selector=None,
                        keep_indices_np=keep_indices_np,
                        num_tokens=num_tokens,
                    )
                    # P5-fix BUG A invariant: detect if edge_reserve is actually working
                    _er_actual = selection_meta.get("edge_reserved_actual_count", 0)
                    _er_target = selection_meta.get("edge_reserved_target_count", 0)
                    _er_invalid = False
                    _er_invalid_reason = None
                    if selection_meta.get("edge_reserve_enabled") is True:
                        if _er_actual is None or _er_actual == 0:
                            if not _edge_scores_available:
                                _er_invalid = True
                                _er_invalid_reason = "edge_scores_missing"
                            elif _edge_scores_available and _er_target > 0:
                                _er_invalid = True
                                _er_invalid_reason = "edge_reserve_mechanism_failed"
                        if _er_actual is not None and _er_target is not None and _er_target > 0:
                            if _er_actual < _er_target:
                                _er_invalid = True
                                _er_invalid_reason = "edge_reserve_partial_failure"
                    selection_meta["edge_reserve_invalid"] = _er_invalid
                    selection_meta["edge_reserve_invalid_reason"] = _er_invalid_reason
                    # P5-fix: always carry forward the edge_scores diagnostics
                    selection_meta["edge_scores_available"] = _edge_scores_available
                    selection_meta["edge_scores_shape"] = _edge_scores_shape
                    selection_meta["edge_scores_finite_ratio"] = _edge_finite_ratio
                except Exception as exc:
                    print(f"[PRUNING] select_hybrid_v1_edge_reserve exception (step={self._hook_step_counter}): {type(exc).__name__}: {exc}")
                    metrics.fallback_used = True
                    metrics.keep_ratio_source = "fallback"
                    metrics.fallback_reason = f"edge_reserve_select_error:{type(exc).__name__}"
                    strategy = "depth_edge_fast_diverse"
                    keep_count = int(round(num_tokens * 0.75))
                    keep_indices_np = None
                    selection_meta = {
                        "fallback_used": True,
                        "fallback_reason": f"edge_reserve_select_error:{type(exc).__name__}",
                        "edge_reserve_enabled": False,
                        "edge_reserve_ratio": 0.0,
                        "edge_reserved_target_count": 0,
                        "edge_reserved_actual_count": 0,
                        "edge_reserved_survival_ratio": None,
                        "final_selected_count": 0,
                        "selected_by_edge_reserved_count": 0,
                        "selected_by_original_hybrid_count": 0,
                        "selected_by_fill_count": 0,
                        "duplicate_edge_hybrid_count": 0,
                        "duplicate_after_exclusion_count": 0,
                        "duplicate_with_original_hybrid_count": 0,
                        "edge_scores_available": False,
                        "edge_scores_shape": None,
                        "edge_scores_finite_ratio": None,
                        "edge_reserve_invalid": True,
                        "edge_reserve_invalid_reason": "exception",
                        "reserved_edge_topk_count": None,
                        "reserved_edge_kept_count": None,
                        "reserved_edge_dropped_count": None,
                        "reserved_edge_topk_dropped_ratio": None,
                        "non_reserved_edge_topk_count": None,
                        "non_reserved_edge_kept_count": None,
                        "non_reserved_edge_dropped_count": None,
                        "non_reserved_edge_topk_dropped_ratio": None,
                        "overall_depth_edge_topk_count": None,
                        "overall_depth_edge_topk_kept_count": None,
                        "overall_depth_edge_topk_dropped_count": None,
                        "overall_depth_edge_topk_dropped_ratio": None,
                    }
                    metrics.dynamic_enabled = True
                    metrics.dynamic_phase = "medium"
                    metrics.dynamic_keep_ratio = keep_count / num_tokens if num_tokens else None
                    metrics.dynamic_keep_k = keep_count
                    metrics.interaction_lock = False
                    metrics.score_ema_enabled = True
                    metrics.score_ema_available = False
                    metrics.ema_used_for_selection = False
                    self._record_selection_path_diagnostics(
                        metrics=metrics,
                        selector_success=False,
                        exc=exc,
                        requested_strategy=self.config.strategy,
                        effective_strategy="depth_edge_fast_diverse",
                        selector_name="select_hybrid_v1_edge_reserve",
                        fallback_selector="depth_edge_fast_diverse",
                        keep_indices_np=None,
                        num_tokens=num_tokens,
                    )
                if _select_ok:
                    metrics.dynamic_enabled = True
                    metrics.dynamic_phase = "interaction_lock" if interaction_locked else "hybrid_temporal_edge_reserve_v1"
                    metrics.dynamic_keep_ratio = keep_count / num_tokens if num_tokens else None
                    metrics.dynamic_keep_k = keep_count
                    metrics.geo_risk_level = metrics.dynamic_phase
                    metrics.dynamic_keep_reason = "temporal_interaction_lock" if interaction_locked else "hybrid_temporal_edge_reserve_v1"
                    metrics.interaction_lock = interaction_locked
                    metrics.interaction_lock_reason = str(temporal.get("lock_reason", "none"))
                    metrics.temporal_stability = temporal.get("temporal_stability")
                    metrics.history_length = temporal.get("history_length")
                    metrics.score_ema_enabled = ema_available
                    metrics.score_ema_available = ema_available
                    metrics.ema_used_for_selection = False
                    metrics.lock_condition_failed_reason = str(temporal.get("reason", "none"))
                    metrics.topk_contact_lock = bool(temporal.get("topk_contact_lock"))
                    metrics.elevated_current_lock = bool(temporal.get("elevated_current_lock"))
                    metrics.gripper_lock = bool(temporal.get("gripper_lock"))
                    metrics.region_lock = bool(temporal.get("region_lock"))
                    metrics.adaptive_threshold_mean = adaptive_threshold
                    metrics.adaptive_threshold_max = adaptive_threshold
                    metrics.interaction_lock_ratio = 1.0 if interaction_locked else 0.0
                    component_summary = (robot_metrics.get("dynamic_decision") or {}).get("component_summary", {}) or {}
                    metrics.num_high_contact_tokens = component_summary.get("num_high_contact_tokens")
                    metrics.num_valid_3d_tokens = component_summary.get("num_valid_3d_tokens")
                    metrics.phase_far_count = 0
                    metrics.phase_mid_count = 0
                    metrics.phase_near_count = 1 if interaction_locked else 0
                    metrics.phase_fallback_safe_count = 0

        # P7: hybrid_budget_v2 — budget-based depth/robot protection
        if self.config.strategy == "hybrid_budget_v2":
            if fallback_reason is not None:
                metrics.fallback_used = True
                metrics.keep_ratio_source = "fallback"
                metrics.fallback_reason = fallback_reason
                strategy = "depth_edge_fast_diverse"
                keep_count = int(round(num_tokens * 0.75))
                keep_indices_np, selection_meta = select_keep_indices(
                    strategy=strategy,
                    num_tokens=num_tokens,
                    keep_count=keep_count,
                    scores=scores,
                    valid_mask=valid_mask,
                    seed=self.config.seed,
                    grid_size=self.config.token_grid_shape[0],
                    cell_grid=self.config.cell_grid,
                    reserve_tokens=reserve_tokens,
                )
                selection_meta["fallback_used"] = True
                selection_meta["fallback_reason"] = fallback_reason
                selection_meta["hybrid_budget_v2"] = False
                selection_meta["depth_edge_budget_ratio"] = float(self.config.hybrid_budget_v2_depth_edge_ratio)
                selection_meta["robot_contact_budget_ratio"] = float(self.config.hybrid_budget_v2_robot_contact_ratio)
                selection_meta["safety_budget_ratio"] = float(self.config.hybrid_budget_v2_safety_ratio)
                selection_meta["total_keep_budget"] = keep_count
                selection_meta["depth_edge_budget"] = None
                selection_meta["robot_geo_budget"] = None
                selection_meta["fill_budget"] = keep_count
                selection_meta["safety_budget"] = None
                selection_meta["K_depth_actual"] = None
                selection_meta["K_robot_actual"] = None
                selection_meta["K_fill_actual"] = None
                selection_meta["overlap_depth_robot_count"] = None
                selection_meta["overlap_depth_robot_diagnostic"] = None
                selection_meta["depth_edge_candidates_count"] = int(np.sum(valid_mask)) if valid_mask is not None else None
                selection_meta["robot_geo_candidates_count"] = int(np.sum(valid_mask)) if valid_mask is not None else None
                selection_meta["depth_edge_reserved_kept_count"] = None
                selection_meta["robot_geo_reserved_kept_count"] = None
                selection_meta["fill_from_depth_count"] = None
                selection_meta["fill_from_robot_count"] = None
                selection_meta["fill_from_other_count"] = None
                self._record_selection_path_diagnostics(
                    metrics=metrics,
                    selector_success=False,
                    exc=None,
                    requested_strategy=self.config.strategy,
                    effective_strategy=self.config.strategy,
                    selector_name="select_hybrid_budget_v2",
                    fallback_selector="depth_edge_fast_diverse",
                    keep_indices_np=keep_indices_np,
                    num_tokens=num_tokens,
                )
            else:
                _depth_scores = robot_metrics.get("edge_scores")
                _near_scores = robot_metrics.get("near_scores")
                _contact_scores = robot_metrics.get("rule_v0_contact_risk_scores", robot_metrics.get("contact_risk_scores"))
                # hybrid_budget_v2 uses depth_edge + robot_geo (use near_scores as proxy)
                _robot_geo = _near_scores
                try:
                    keep_indices_np, selection_meta = select_hybrid_budget_v2(
                        depth_edge_scores=_depth_scores,
                        robot_geo_scores=_robot_geo,
                        valid_mask=valid_mask,
                        keep_k=keep_count,
                        depth_edge_budget_ratio=float(self.config.hybrid_budget_v2_depth_edge_ratio),
                        robot_contact_budget_ratio=float(self.config.hybrid_budget_v2_robot_contact_ratio),
                        safety_budget_ratio=float(self.config.hybrid_budget_v2_safety_ratio),
                        grid_h=self.config.token_grid_shape[0],
                        grid_w=self.config.token_grid_shape[1],
                    )
                    _select_ok = True
                    # P7: Inject P7 fields into selection_meta BEFORE _record_selection_path_diagnostics
                    # so that finalize_selection_debug_info() inside it sees correct phase accounting.
                    # Include fallback_actual and selected_by_fallback_count for phase accounting.
                    _p7_fields = {
                        "hybrid_budget_v2": True,
                        "depth_edge_budget_ratio": float(self.config.hybrid_budget_v2_depth_edge_ratio),
                        "robot_contact_budget_ratio": float(self.config.hybrid_budget_v2_robot_contact_ratio),
                        "safety_budget_ratio": float(self.config.hybrid_budget_v2_safety_ratio),
                        "K_depth_actual": selection_meta.get("K_depth_actual"),
                        "K_robot_actual": selection_meta.get("K_robot_actual"),
                        "K_fill_actual": selection_meta.get("K_fill_actual"),
                        "overlap_depth_robot_count": selection_meta.get("overlap_depth_robot_count"),
                        "overlap_depth_robot_diagnostic": selection_meta.get("overlap_depth_robot_diagnostic"),
                        "depth_edge_candidates_count": selection_meta.get("depth_edge_candidates_count"),
                        "robot_geo_candidates_count": selection_meta.get("robot_geo_candidates_count"),
                        "depth_edge_reserved_kept_count": selection_meta.get("depth_edge_reserved_kept_count"),
                        "robot_geo_reserved_kept_count": selection_meta.get("robot_geo_reserved_kept_count"),
                        "fill_from_depth_count": selection_meta.get("fill_from_depth_count"),
                        "fill_from_robot_count": selection_meta.get("fill_from_robot_count"),
                        "fill_from_other_count": selection_meta.get("fill_from_other_count"),
                        # Phase accounting: include fallback count even if 0
                        "fallback_actual": selection_meta.get("fallback_actual"),
                        "selected_by_fallback_count": selection_meta.get("selected_by_fallback_count"),
                    }
                    for _k, _v in _p7_fields.items():
                        if _v is not None:
                            selection_meta[_k] = _v
                    # Also inject selected_by_fallback_count into selection_meta for finalize_debug_info
                    if "selected_by_fallback_count" not in selection_meta:
                        selection_meta["selected_by_fallback_count"] = selection_meta.get("fallback_actual")
                    self._record_selection_path_diagnostics(
                        metrics=metrics,
                        selector_success=True,
                        exc=None,
                        requested_strategy=self.config.strategy,
                        effective_strategy=self.config.strategy,
                        selector_name="select_hybrid_budget_v2",
                        fallback_selector=None,
                        keep_indices_np=keep_indices_np,
                        num_tokens=num_tokens,
                        selection_meta=selection_meta,
                    )
                    # P7: Directly compute and set correct phase accounting in metrics.
                    # finalize_selection_debug_info runs twice (inside _record_path and in post-processing),
                    # which corrupts hybrid_budget_v2 phase accounting. Override here.
                    _sm = selection_meta
                    _kept = metrics.num_visual_tokens_kept or 0
                    _p1 = int(_sm.get("selected_by_depth_edge_count") or 0)
                    _p2 = int(_sm.get("selected_by_robot_geo_count") or 0)
                    _pfill = int(_sm.get("selected_by_fill_count") or 0)
                    _pfb = int(_sm.get("fallback_actual") or _sm.get("selected_by_fallback_count") or 0)
                    _punattr = int(_sm.get("selected_unattributed") or 0)
                    _phase_sum = _p1 + _p2 + _pfill + _pfb
                    metrics.selected_by_phase1 = _p1
                    metrics.selected_by_phase2 = _p2
                    metrics.selected_by_phase3 = None
                    metrics.selected_by_fill = _pfill
                    metrics.selected_by_fallback = _pfb
                    metrics.selected_unattributed = _punattr
                    metrics.phase_accounting_sum = _phase_sum
                    if abs(_phase_sum - _kept) <= 2:
                        metrics.phase_accounting_valid = True
                        metrics.phase_accounting_error = None
                    else:
                        metrics.phase_accounting_valid = False
                        metrics.phase_accounting_error = f"phase_sum={_phase_sum} != kept={_kept}"
                    # Copy remaining P7 fields into metrics
                    metrics.hybrid_budget_v2 = True
                    metrics.depth_edge_budget_ratio = float(self.config.hybrid_budget_v2_depth_edge_ratio)
                    metrics.robot_contact_budget_ratio = float(self.config.hybrid_budget_v2_robot_contact_ratio)
                    metrics.safety_budget_ratio = float(self.config.hybrid_budget_v2_safety_ratio)
                    metrics.K_depth_actual = _sm.get("K_depth_actual")
                    metrics.K_robot_actual = _sm.get("K_robot_actual")
                    metrics.K_fill_actual = _sm.get("K_fill_actual")
                    metrics.overlap_depth_robot_count = _sm.get("overlap_depth_robot_count")
                    metrics.overlap_depth_robot_diagnostic = _sm.get("overlap_depth_robot_diagnostic")
                    metrics.depth_edge_candidates_count = _sm.get("depth_edge_candidates_count")
                    metrics.robot_geo_candidates_count = _sm.get("robot_geo_candidates_count")
                    metrics.depth_edge_reserved_kept_count = _sm.get("depth_edge_reserved_kept_count")
                    metrics.robot_geo_reserved_kept_count = _sm.get("robot_geo_reserved_kept_count")
                    metrics.fill_from_depth_count = _sm.get("fill_from_depth_count")
                    metrics.fill_from_robot_count = _sm.get("fill_from_robot_count")
                    metrics.fill_from_other_count = _sm.get("fill_from_other_count")
                    # hybrid_budget_v2 phase accounting is correct — skip post-processing
                    # _record_selection_path_diagnostics call to avoid re-corrupting it
                    _skip_postprocess_diag = True
                except Exception as exc:
                    print(f"[PRUNING] select_hybrid_budget_v2 exception (step={self._hook_step_counter}): {type(exc).__name__}: {exc}")
                    metrics.fallback_used = True
                    metrics.keep_ratio_source = "fallback"
                    metrics.fallback_reason = f"hybrid_budget_v2_error:{type(exc).__name__}"
                    keep_indices_np, selection_meta = select_keep_indices(
                        strategy="depth_edge_fast_diverse",
                        num_tokens=num_tokens,
                        keep_count=keep_count,
                        scores=_depth_scores,
                        valid_mask=valid_mask,
                        seed=self.config.seed,
                        grid_size=self.config.token_grid_shape[0],
                        cell_grid=self.config.cell_grid,
                        reserve_tokens=reserve_tokens,
                    )
                    self._record_selection_path_diagnostics(
                        metrics=metrics,
                        selector_success=False,
                        exc=exc,
                        requested_strategy=self.config.strategy,
                        effective_strategy="depth_edge_fast_diverse",
                        selector_name="select_hybrid_budget_v2",
                        fallback_selector="depth_edge_fast_diverse",
                        keep_indices_np=None,
                        num_tokens=num_tokens,
                    )
                    # hybrid_budget_v2 exception-fallback phase accounting is correct
                    _skip_postprocess_diag = True
                    selection_meta["hybrid_budget_v2"] = False
                    selection_meta["depth_edge_budget_ratio"] = float(self.config.hybrid_budget_v2_depth_edge_ratio)
                    selection_meta["robot_contact_budget_ratio"] = float(self.config.hybrid_budget_v2_robot_contact_ratio)
                    selection_meta["safety_budget_ratio"] = float(self.config.hybrid_budget_v2_safety_ratio)
                    selection_meta["total_keep_budget"] = keep_count
                    selection_meta["depth_edge_budget"] = None
                    selection_meta["robot_geo_budget"] = None
                    selection_meta["fill_budget"] = keep_count
                    selection_meta["safety_budget"] = None
                    selection_meta["K_depth_actual"] = None
                    selection_meta["K_robot_actual"] = None
                    selection_meta["K_fill_actual"] = None
                    selection_meta["overlap_depth_robot_count"] = None
                    selection_meta["overlap_depth_robot_diagnostic"] = None
                    selection_meta["depth_edge_candidates_count"] = int(np.sum(valid_mask)) if valid_mask is not None else None
                    selection_meta["robot_geo_candidates_count"] = int(np.sum(valid_mask)) if valid_mask is not None else None
                    selection_meta["depth_edge_reserved_kept_count"] = None
                    selection_meta["robot_geo_reserved_kept_count"] = None
                    selection_meta["fill_from_depth_count"] = None
                    selection_meta["fill_from_robot_count"] = None
                    selection_meta["fill_from_other_count"] = None
                    # hybrid_budget_v2 exception-fallback — skip post-processing diag
                    _skip_postprocess_diag = True

        _bbv0_strategy_check = str(self.config.strategy)
        if _bbv0_strategy_check == "robot_geo_branch_budget_v0":
            _depth_scores = robot_metrics.get("edge_scores")
            _hybrid_raw = robot_metrics.get("hybrid_final_scores")
            if _hybrid_raw is None:
                _hybrid_raw = robot_metrics.get("final_scores")
            _hybrid_scores = _hybrid_raw

            # P14-B: Apply robot-self mask to hybrid_final_scores if enabled.
            # self_mask is applied to near_score component of hybrid_final_scores only.
            _self_mask_diag: Dict[str, Any] = {}
            _self_mask_enabled = self.config.robot_self_mask_enabled
            metrics.robot_self_mask_enabled = _self_mask_enabled
            _gripper_pixel = robot_metrics.get("gripper_pixel")
            _token_u = None
            _token_v = None
            if "cache" in robot_metrics and robot_metrics["cache"]:
                _token_u = robot_metrics["cache"].get("u")
                _token_v = robot_metrics["cache"].get("v")
            if _self_mask_enabled:
                pass  # diagnostics collected below
            if _self_mask_enabled and _hybrid_scores is not None:
                _hybrid_scores, _self_mask_arr, _self_mask_diag = self._apply_self_mask(
                    scores=np.asarray(_hybrid_scores, dtype=np.float32),
                    gripper_pixel=_gripper_pixel,
                    token_u=_token_u,
                    token_v=_token_v,
                    core_radius_px=self.config.robot_self_mask_core_radius_px,
                )
                # Also apply to near_scores if available
                if robot_metrics.get("near_scores") is not None and self.config.robot_self_mask_apply_to_near_score:
                    _near_arr = np.asarray(robot_metrics["near_scores"], dtype=np.float32).reshape(-1)
                    _near_masked, _, _ = self._apply_self_mask(
                        scores=_near_arr,
                        gripper_pixel=_gripper_pixel,
                        token_u=_token_u,
                        token_v=_token_v,
                        core_radius_px=self.config.robot_self_mask_core_radius_px,
                    )
                    robot_metrics["near_scores"] = _near_masked
            else:
                _self_mask_arr = np.zeros(num_tokens, dtype=bool)

            # Track which scores were unavailable (for diagnostics)
            _de_unavailable = _depth_scores is None
            _hyb_unavailable = _hybrid_scores is None
            _fallback_reasons: List[str] = []
            if _de_unavailable:
                _fallback_reasons.append("depth_edge_scores_unavailable")
            if _hyb_unavailable:
                _fallback_reasons.append("hybrid_final_scores_unavailable")
            _fallback_reason = "|".join(_fallback_reasons) if _fallback_reasons else None

            _token_u = None
            _token_v = None
            if "cache" in robot_metrics and robot_metrics["cache"]:
                _token_u = robot_metrics["cache"].get("u")
                _token_v = robot_metrics["cache"].get("v")
            try:
                keep_indices_np, selection_meta = select_branch_budget_v0(
                    depth_edge_scores=_depth_scores,
                    hybrid_final_scores=_hybrid_scores,
                    valid_mask=valid_mask,
                    keep_k=keep_count,
                    depth_edge_budget=int(self.config.branch_budget_depth_tokens),
                    hybrid_action_budget=int(self.config.branch_budget_hybrid_tokens),
                    depth_edge_ratio_override=float(self.config.branch_budget_depth_ratio),
                    hybrid_action_ratio_override=float(self.config.branch_budget_hybrid_ratio),
                    grid_h=self.config.token_grid_shape[0],
                    grid_w=self.config.token_grid_shape[1],
                    token_u=_token_u,
                    token_v=_token_v,
                )
                _bb_fields = {
                    "branch_budget_v0": True,
                    "total_keep_budget": int(keep_count),
                    "depth_edge_budget": int(self.config.branch_budget_depth_tokens),
                    "hybrid_action_budget": int(self.config.branch_budget_hybrid_tokens),
                    "diversity_fill_budget": keep_count - int(self.config.branch_budget_depth_tokens) - int(self.config.branch_budget_hybrid_tokens),
                    "temporal_budget": None,
                    "branch_budget_depth_ratio_override": float(self.config.branch_budget_depth_ratio),
                    "branch_budget_hybrid_ratio_override": float(self.config.branch_budget_hybrid_ratio),
                    "score_path_fallback_reason": _fallback_reason,
                }
                for _k, _v in _bb_fields.items():
                    selection_meta[_k] = _v
                self._record_selection_path_diagnostics(
                    metrics=metrics,
                    selector_success=True,
                    exc=None,
                    requested_strategy=self.config.strategy,
                    effective_strategy=self.config.strategy,
                    selector_name="select_branch_budget_v0",
                    fallback_selector=None,
                    keep_indices_np=keep_indices_np,
                    num_tokens=num_tokens,
                    selection_meta=selection_meta,
                )
                # P11: Directly set phase accounting in metrics to prevent post-processing corruption
                _sm = selection_meta
                _kept = int(len(keep_indices_np)) if keep_indices_np is not None else 0
                metrics.selected_by_depth_branch_count = int(_sm.get("selected_by_depth_branch") or 0)
                metrics.selected_by_hybrid_branch_count = int(_sm.get("selected_by_hybrid_branch") or 0)
                metrics.selected_by_fill_branch_count = int(_sm.get("selected_by_fill") or 0)
                metrics.branch_budget_v0 = True
                metrics.branch_accounting_valid = bool(_sm.get("branch_accounting_valid", False))
                metrics.branch_sum_equals_kept = bool(_sm.get("branch_sum_equals_kept", False))
                metrics.overlap_depth_edge_hybrid_count = _sm.get("overlap_depth_edge_hybrid_count")
                metrics.overlap_depth_edge_hybrid_ratio = _sm.get("overlap_depth_edge_hybrid_ratio")
                metrics.non_reserved_depth_edge_dropped_ratio = _sm.get("non_reserved_depth_edge_dropped_ratio")
                # P14-B: Record self-mask diagnostics
                metrics.self_mask_available = _self_mask_diag.get("self_mask_available", False)
                metrics.self_mask_core_radius_px = _self_mask_diag.get("self_mask_core_radius_px")
                metrics.self_mask_token_count = _self_mask_diag.get("self_mask_token_count", 0)
                metrics.self_mask_token_ratio = _self_mask_diag.get("self_mask_token_ratio", 0.0)
                metrics.gripper_pixel_u = _self_mask_diag.get("gripper_pixel_u")
                metrics.gripper_pixel_v = _self_mask_diag.get("gripper_pixel_v")
                metrics.gripper_pixel_in_bounds = _self_mask_diag.get("gripper_pixel_in_bounds", False)
                metrics.gripper_projection_valid = _self_mask_diag.get("gripper_projection_valid", False)
                metrics.near_score_mean_before_self_mask = _self_mask_diag.get("near_score_mean_before_self_mask")
                metrics.near_score_mean_after_self_mask = _self_mask_diag.get("near_score_mean_after_self_mask")
                metrics.near_score_self_region_mean = _self_mask_diag.get("near_score_self_region_mean")
                metrics.near_score_nonself_region_mean = _self_mask_diag.get("near_score_nonself_region_mean")
                # Selected self-mask token counts (compute from keep_indices and self_mask_arr)
                if keep_indices_np is not None and _self_mask_arr is not None and len(_self_mask_arr) > 0:
                    _sel_arr = np.asarray(keep_indices_np, dtype=np.int64)
                    _sel_self = int(np.sum(_self_mask_arr[_sel_arr])) if _sel_arr.size > 0 else 0
                    metrics.selected_self_mask_token_count = _sel_self
                    metrics.selected_self_mask_token_ratio = float(_sel_self / len(_sel_arr)) if len(_sel_arr) > 0 else 0.0
                # Hybrid score before/after self-mask
                metrics.hybrid_score_mean_before_self_mask = _self_mask_diag.get("hybrid_score_mean_before_self_mask")
                metrics.hybrid_score_mean_after_self_mask = _self_mask_diag.get("hybrid_score_mean_after_self_mask")
                # Hybrid score stats (computed from _hybrid_scores if available)
                if _hybrid_scores is not None:
                    _hyb_a = np.asarray(_hybrid_scores, dtype=np.float32).reshape(-1)
                    _hyb_v = _hyb_a[valid_mask]
                    if _hyb_v.size > 0:
                        metrics.final_hybrid_score_mean = float(np.nanmean(_hyb_v))
                        metrics.final_hybrid_score_max = float(np.nanmax(_hyb_v))
                        metrics.final_hybrid_score_std = float(np.nanstd(_hyb_v))
                if _depth_scores is not None:
                    _de_a = np.asarray(_depth_scores, dtype=np.float32).reshape(-1)
                    _de_v = _de_a[valid_mask]
                    if _de_v.size > 0:
                        metrics.edge_score_mean = float(np.nanmean(_de_v))
                        metrics.edge_score_max = float(np.nanmax(_de_v))
                        metrics.edge_score_std = float(np.nanstd(_de_v))
                _skip_postprocess_diag = True
            except Exception as exc:
                print(f"[PRUNING] select_branch_budget_v0 exception (step={self._hook_step_counter}): {type(exc).__name__}: {exc}")
                metrics.fallback_used = True
                metrics.keep_ratio_source = "fallback"
                metrics.fallback_reason = f"branch_budget_v0_error:{type(exc).__name__}"
                keep_indices_np, selection_meta = select_keep_indices(
                    strategy="depth_edge_fast_diverse",
                    num_tokens=num_tokens,
                    keep_count=keep_count,
                    scores=_depth_scores,
                    valid_mask=valid_mask,
                    seed=self.config.seed,
                    grid_size=self.config.token_grid_shape[0],
                    cell_grid=self.config.cell_grid,
                    reserve_tokens=reserve_tokens,
                )
                self._record_selection_path_diagnostics(
                    metrics=metrics,
                    selector_success=False,
                    exc=exc,
                    requested_strategy=self.config.strategy,
                    effective_strategy="depth_edge_fast_diverse",
                    selector_name="select_branch_budget_v0",
                    fallback_selector="depth_edge_fast_diverse",
                    keep_indices_np=None,
                    num_tokens=num_tokens,
                )
                _skip_postprocess_diag = True
                selection_meta["branch_budget_v0"] = False
                selection_meta["total_keep_budget"] = keep_count
                selection_meta["depth_edge_budget"] = None
                selection_meta["hybrid_action_budget"] = None
                selection_meta["diversity_fill_budget"] = keep_count
                selection_meta["branch_accounting_valid"] = None
                selection_meta["branch_sum_equals_kept"] = None
                selection_meta["overlap_depth_edge_hybrid_count"] = None
                selection_meta["overlap_depth_edge_hybrid_ratio"] = None
                selection_meta["non_reserved_depth_edge_dropped_ratio"] = None
                selection_meta["fallback_used"] = True
                selection_meta["fallback_reason"] = f"branch_budget_v0_error:{type(exc).__name__}"
                metrics.branch_budget_v0 = False
                metrics.branch_accounting_valid = None
                metrics.branch_sum_equals_kept = None

        # ─────────────────────────────────────────────────────────────────────────
        # P15: robot_geo_acgtp_v1 — Action-Constrained Geometric Token Protection
        # ─────────────────────────────────────────────────────────────────────────
        _acgtp_strategy_check = str(self.config.strategy)
        if _acgtp_strategy_check == "robot_geo_acgtp_v1" and keep_indices_np is None:
            score_stats: Dict[str, Any] = {}  # Initialize so exception handler can reference it
            # ── Build all ACGTP-v1 branch scores ───────────────────────────────
            _scene_scores = robot_metrics.get("acgtp_scene_layout_scores")
            _de_scores = robot_metrics.get("edge_scores")
            _contact_scores = robot_metrics.get("acgtp_contact_ring_scores")
            _motion_scores = robot_metrics.get("acgtp_motion_corridor_scores")
            _action_constraint_scores = robot_metrics.get("acgtp_action_constraint_scores")
            _motion_valid = bool(robot_metrics.get("acgtp_motion_corridor_valid", False))
            _fill_mask = robot_metrics.get("acgtp_constrained_fill_mask")
            _self_core = robot_metrics.get("acgtp_self_core_mask")
            _token_u = None
            _token_v = None
            if "cache" in robot_metrics and robot_metrics["cache"]:
                _token_u = robot_metrics["cache"].get("u")
                _token_v = robot_metrics["cache"].get("v")

            # Fallback: if scene_layout not computed, derive from depth distribution
            if _scene_scores is None:
                _depth_vals = robot_metrics.get("depth_metric")
                if _depth_vals is not None:
                    _depth_a = np.asarray(_depth_vals, dtype=np.float32).reshape(-1)
                    _scene_scores = np.clip((_depth_a - 0.3) / 1.7, 0.0, 1.0)
                else:
                    _scene_scores = np.zeros(num_tokens, dtype=np.float32)

            # Fallback: if contact_ring not computed, use near_scores with self-mask
            if _contact_scores is None:
                _near_s = robot_metrics.get("near_scores")
                if _near_s is not None:
                    _near_a = np.asarray(_near_s, dtype=np.float32).reshape(-1)
                    # Simple self-filter: penalize near gripper tokens
                    _contact_scores = _near_a.copy()
                else:
                    _contact_scores = np.zeros(num_tokens, dtype=np.float32)

            # Fallback: if motion_corridor not computed, use motion_cone_scores
            if _motion_scores is None:
                _mc = robot_metrics.get("motion_cone_scores")
                if _mc is not None:
                    _motion_scores = np.asarray(_mc, dtype=np.float32).reshape(-1)
                else:
                    _motion_scores = np.zeros(num_tokens, dtype=np.float32)

            # Fallback: if depth_edge not available, use edge_scores from robot_metrics
            if _de_scores is None:
                _de_scores = robot_metrics.get("edge_scores")
            if _de_scores is None:
                _de_scores = np.zeros(num_tokens, dtype=np.float32)

            # Build constrained fill mask: tokens with valid depth + scene relevance
            if _fill_mask is None:
                _depth_metric = robot_metrics.get("depth_metric")
                if _depth_metric is not None:
                    _dm = np.asarray(_depth_metric, dtype=np.float32).reshape(-1)
                    _scene_s = np.asarray(_scene_scores, dtype=np.float32).reshape(-1)
                    _fill_mask = (np.isfinite(_dm) & (_dm > 0.1) & (_dm < 5.0) & (_scene_s > 0.0)).astype(np.float32)
                else:
                    _fill_mask = np.ones(num_tokens, dtype=np.float32)

            # Self-core mask for contact ring filtering
            if _self_core is None:
                _gripper_pixel = robot_metrics.get("gripper_pixel")
                if _gripper_pixel is not None and _token_u is not None:
                    _gx, _gy = float(_gripper_pixel[0]), float(_gripper_pixel[1])
                    _u_a = np.asarray(_token_u, dtype=np.float32).reshape(-1)
                    _v_a = np.asarray(_token_v, dtype=np.float32).reshape(-1)
                    _core_r = float(self.config.acgtp_self_core_radius_px)
                    _dist2 = (_u_a - _gx) ** 2 + (_v_a - _gy) ** 2
                    _self_core = (np.sqrt(_dist2) <= _core_r).astype(np.float32)
                else:
                    _self_core = np.zeros(num_tokens, dtype=np.float32)

            # Gripper pixel projection for contact_ring diagnostics
            _gripper_pixel_metrics = robot_metrics.get("gripper_pixel")
            # Full module results for diagnostics
            _contact_result_metrics = robot_metrics.get("acgtp_contact_ring_result")
            _motion_result_metrics = robot_metrics.get("acgtp_motion_corridor_result")
            _scene_result_metrics = robot_metrics.get("acgtp_scene_layout_result")
            _action_constraint_result_metrics = robot_metrics.get("acgtp_action_constraint_result")

            try:
                keep_indices_np, selection_meta = select_acgtp_v1(
                    scene_layout_scores=_scene_scores,
                    depth_edge_scores=_de_scores,
                    contact_ring_scores=_contact_scores,
                    motion_corridor_scores=_motion_scores,
                    valid_mask=valid_mask,
                    keep_k=keep_count,
                    constrained_fill_mask=_fill_mask,
                    token_u=_token_u,
                    token_v=_token_v,
                    grid_h=self.config.token_grid_shape[0],
                    grid_w=self.config.token_grid_shape[1],
                    w_scene_layout=float(self.config.acgtp_w_scene_layout),
                    w_depth_structure=float(self.config.acgtp_w_depth_structure),
                    w_contact_ring=float(self.config.acgtp_w_contact_ring),
                    w_motion_corridor=float(self.config.acgtp_w_motion_corridor),
                    hard_protect_ratio=float(self.config.acgtp_hard_protect_ratio),
                    motion_corridor_valid=_motion_valid,
                    self_core_mask=_self_core,
                    contact_ring_inner_px=float(self.config.acgtp_contact_ring_inner_px),
                    contact_ring_outer_px=float(self.config.acgtp_contact_ring_outer_px),
                    contact_requires_edge_or_object=bool(self.config.acgtp_contact_requires_edge_or_object),
                    depth_edge_score_for_gate=_de_scores,
                    # Pass real motion corridor config and result values for diagnostics
                    _motion_result_for_diag=_motion_result_metrics,
                    _scene_result_for_diag=_scene_result_metrics,
                    action_constraint_scores=_action_constraint_scores,
                    _action_constraint_result_for_diag=_action_constraint_result_metrics,
                )
                # ── Record ACGTP-v1 metrics ─────────────────────────────────
                _sm = selection_meta
                metrics.acgtp_v1 = True
                metrics.acgtp_w_scene_layout = float(self.config.acgtp_w_scene_layout)
                metrics.acgtp_w_depth_structure = float(self.config.acgtp_w_depth_structure)
                metrics.acgtp_w_contact_ring = float(self.config.acgtp_w_contact_ring)
                metrics.acgtp_w_motion_corridor = float(self.config.acgtp_w_motion_corridor)
                metrics.acgtp_self_core_radius_px = float(self.config.acgtp_self_core_radius_px)
                metrics.acgtp_contact_ring_inner_px = float(self.config.acgtp_contact_ring_inner_px)
                metrics.acgtp_contact_ring_outer_px = float(self.config.acgtp_contact_ring_outer_px)
                metrics.acgtp_self_core_token_count = _sm.get("acgtp_self_core_token_count")
                metrics.acgtp_self_core_token_ratio = _sm.get("acgtp_self_core_token_ratio")
                # contact_ring_token_count from selector = top-k gated candidates selected
                # The total ring/gated token counts come from the module result
                metrics.acgtp_contact_ring_token_count = _sm.get("acgtp_contact_ring_token_count")
                metrics.acgtp_contact_ring_token_ratio = _sm.get("acgtp_contact_ring_token_ratio")
                metrics.acgtp_contact_ring_gated_token_count = _sm.get("acgtp_contact_ring_gated_token_count")
                metrics.acgtp_contact_ring_valid = _sm.get("acgtp_contact_ring_valid")
                # Gripper pixel from module result
                if _contact_result_metrics is not None:
                    metrics.gripper_pixel_u = _contact_result_metrics.get("gripper_pixel_u")
                    metrics.gripper_pixel_v = _contact_result_metrics.get("gripper_pixel_v")
                    metrics.gripper_in_bounds = _contact_result_metrics.get("gripper_in_bounds")
                    # Full ring token counts from the module (not just top-k)
                    _ring_total = _contact_result_metrics.get("contact_ring_token_count")
                    if _ring_total is not None:
                        metrics.acgtp_contact_ring_token_count = _ring_total
                    _gated_total = _contact_result_metrics.get("contact_ring_gated_token_count")
                    if _gated_total is not None:
                        metrics.acgtp_contact_ring_gated_token_count = _gated_total
                # Scene layout diagnostics from module result
                if _scene_result_metrics is not None:
                    metrics.acgtp_support_plane_token_count = _scene_result_metrics.get("support_plane_token_count")
                    metrics.acgtp_object_component_token_count = _scene_result_metrics.get("object_component_token_count")
                    metrics.acgtp_boundary_token_count = _scene_result_metrics.get("boundary_token_count")
                metrics.acgtp_scene_layout_score_mean = _sm.get("acgtp_scene_layout_score_mean")
                metrics.acgtp_scene_layout_score_max = _sm.get("acgtp_scene_layout_score_max")
                if _scene_result_metrics is not None:
                    _sp_cnt = _scene_result_metrics.get("support_plane_token_count")
                    if _sp_cnt is not None:
                        metrics.acgtp_support_plane_token_count = _sp_cnt
                    _oc_cnt = _scene_result_metrics.get("object_component_token_count")
                    if _oc_cnt is not None:
                        metrics.acgtp_object_component_token_count = _oc_cnt
                    _bn_cnt = _scene_result_metrics.get("boundary_token_count")
                    if _bn_cnt is not None:
                        metrics.acgtp_boundary_token_count = _bn_cnt
                # P6: support_plane cap diagnostics
                metrics.acgtp_support_plane_candidate_count = _sm.get("acgtp_support_plane_candidate_count")
                metrics.acgtp_scene_support_plane_cap_ratio = _sm.get("acgtp_scene_support_plane_cap_ratio")
                metrics.acgtp_scene_support_plane_cap_used = _sm.get("acgtp_scene_support_plane_cap_used")
                metrics.acgtp_scene_support_plane_fallback_used = _sm.get("acgtp_scene_support_plane_fallback_used")
                metrics.acgtp_scene_support_plane_fallback_reason = _sm.get("acgtp_scene_support_plane_fallback_reason")
                # P6: object_component fallback
                metrics.acgtp_scene_object_component_fallback_used = _sm.get("acgtp_scene_object_component_fallback_used")
                metrics.acgtp_scene_object_component_fallback_reason = _sm.get("acgtp_scene_object_component_fallback_reason")
                metrics.acgtp_scene_object_component_num_components = _sm.get("acgtp_scene_object_component_num_components")
                # P6: boundary fallback + source
                metrics.acgtp_scene_boundary_fallback_used = _sm.get("acgtp_scene_boundary_fallback_used")
                metrics.acgtp_scene_boundary_fallback_reason = _sm.get("acgtp_scene_boundary_fallback_reason")
                metrics.acgtp_scene_boundary_from_object_count = _sm.get("acgtp_scene_boundary_from_object_count")
                metrics.acgtp_scene_boundary_from_depth_count = _sm.get("acgtp_scene_boundary_from_depth_count")
                # P6: per-branch timing
                metrics.acgtp_scene_layout_ms = score_stats.get("acgtp_scene_layout_ms")
                metrics.acgtp_contact_ring_ms = score_stats.get("acgtp_contact_ring_ms")
                metrics.acgtp_motion_corridor_ms = score_stats.get("acgtp_motion_corridor_ms")
                metrics.acgtp_scene_fill_candidate_count = _sm.get("acgtp_scene_fill_candidate_count")
                metrics.acgtp_scene_fill_candidate_ratio = _sm.get("acgtp_scene_fill_candidate_ratio")
                # Scene layout per-component selected attribution (post-selection, from selector metadata)
                metrics.acgtp_scene_selected_support_plane_count = _sm.get("acgtp_scene_selected_support_plane_count")
                metrics.acgtp_scene_selected_object_component_count = _sm.get("acgtp_scene_selected_object_component_count")
                metrics.acgtp_scene_selected_boundary_count = _sm.get("acgtp_scene_selected_boundary_count")
                metrics.acgtp_scene_selected_relation_count = _sm.get("acgtp_scene_selected_relation_count")
                metrics.acgtp_scene_support_plane_selected_ratio = _sm.get("acgtp_scene_support_plane_selected_ratio")
                metrics.acgtp_motion_corridor_valid = _motion_valid
                metrics.acgtp_motion_corridor_score_mean = _sm.get("acgtp_motion_corridor_score_mean")
                metrics.acgtp_motion_corridor_score_max = _sm.get("acgtp_motion_corridor_score_max")
                # Motion corridor diagnostics from module result (not fake zeros)
                if _motion_result_metrics is not None:
                    metrics.acgtp_motion_corridor_length_m = _sm.get("acgtp_motion_corridor_length_m")
                    metrics.acgtp_motion_norm_m = _motion_result_metrics.get("motion_norm_m")
                    metrics.acgtp_motion_ema_alpha = float(self.config.acgtp_motion_ema_alpha)
                else:
                    metrics.acgtp_motion_corridor_length_m = _sm.get("acgtp_motion_corridor_length_m")
                    metrics.acgtp_motion_norm_m = _sm.get("acgtp_motion_norm_m")
                    metrics.acgtp_motion_ema_alpha = _sm.get("acgtp_motion_ema_alpha")
                metrics.acgtp_depth_structure_score_mean = _sm.get("acgtp_depth_structure_score_mean")
                metrics.acgtp_depth_structure_score_max = _sm.get("acgtp_depth_structure_score_max")
                metrics.acgtp_action_constraint_score_mean = _sm.get("acgtp_action_constraint_score_mean")
                metrics.acgtp_action_constraint_score_max = _sm.get("acgtp_action_constraint_score_max")
                metrics.acgtp_action_constraint_source = _sm.get("acgtp_action_constraint_source")
                metrics.acgtp_future_action_constraint_enabled = _sm.get("acgtp_future_action_constraint_enabled")
                metrics.acgtp_future_action_constraint_valid = _sm.get("acgtp_future_action_constraint_valid")
                metrics.acgtp_future_action_constraint_disabled_reason = _sm.get("acgtp_future_action_constraint_disabled_reason")
                metrics.acgtp_future_action_constraint_score_mean = _sm.get("acgtp_future_action_constraint_score_mean")
                metrics.acgtp_future_action_constraint_score_max = _sm.get("acgtp_future_action_constraint_score_max")
                metrics.acgtp_object_side_contact_score_mean = _sm.get("acgtp_object_side_contact_score_mean")
                metrics.acgtp_object_side_contact_score_max = _sm.get("acgtp_object_side_contact_score_max")
                metrics.acgtp_swept_motion_risk_score_mean = _sm.get("acgtp_swept_motion_risk_score_mean")
                metrics.acgtp_swept_motion_risk_score_max = _sm.get("acgtp_swept_motion_risk_score_max")
                metrics.acgtp_collision_contact_risk_score_mean = _sm.get("acgtp_collision_contact_risk_score_mean")
                metrics.acgtp_collision_contact_risk_score_max = _sm.get("acgtp_collision_contact_risk_score_max")
                metrics.acgtp_contact_object_overlap_count = _sm.get("acgtp_contact_object_overlap_count")
                metrics.acgtp_robot_self_penalty_count = _sm.get("acgtp_robot_self_penalty_count")
                metrics.acgtp_action_constraint_ms = score_stats.get("acgtp_action_constraint_ms")
                metrics.acgtp_hard_protect_count = _sm.get("acgtp_hard_protect_count")
                metrics.acgtp_hard_protect_ratio = _sm.get("acgtp_hard_protect_ratio")
                metrics.acgtp_hard_protect_valid = _sm.get("acgtp_hard_protect_valid")
                metrics.selected_by_scene_layout_count = _sm.get("selected_by_scene_layout_count")
                metrics.selected_by_depth_structure_count = _sm.get("selected_by_depth_structure_count")
                metrics.selected_by_contact_ring_count = _sm.get("selected_by_contact_ring_count")
                metrics.selected_by_motion_corridor_count = _sm.get("selected_by_motion_corridor_count")
                metrics.selected_by_constrained_fill_count = _sm.get("selected_by_constrained_fill_count")
                metrics.selected_by_acgtp_fallback_count = _sm.get("selected_by_acgtp_fallback_count")
                metrics.overlap_scene_depth_count = _sm.get("overlap_scene_depth_count")
                metrics.overlap_scene_contact_count = _sm.get("overlap_scene_contact_count")
                metrics.overlap_contact_motion_count = _sm.get("overlap_contact_motion_count")
                metrics.overlap_scene_motion_count = _sm.get("overlap_scene_motion_count")
                metrics.overlap_depth_contact_count = _sm.get("overlap_depth_contact_count")
                metrics.overlap_depth_motion_count = _sm.get("overlap_depth_motion_count")
                metrics.acgtp_branch_accounting_valid = _sm.get("acgtp_branch_accounting_valid")
                metrics.acgtp_branch_sum = _sm.get("acgtp_branch_sum")
                metrics.acgtp_branch_sum_error = _sm.get("acgtp_branch_sum_error")
                metrics.branch_accounting_valid = _sm.get("branch_accounting_valid")
                metrics.branch_sum_equals_kept = _sm.get("branch_sum_equals_kept")
                metrics.acgtp_hard_protect_ratio_config = _sm.get("acgtp_hard_protect_ratio_config")
                metrics.acgtp_fallback_used = _sm.get("acgtp_fallback_used")
                metrics.acgtp_fallback_reason = _sm.get("acgtp_fallback_reason")
                metrics.acgtp_motion_disabled_reason = _sm.get("acgtp_motion_disabled_reason")
                metrics.acgtp_constrained_fill_mask = _sm.get("acgtp_constrained_fill_mask")
                metrics.acgtp_final_kept = _sm.get("final_kept")
                metrics.acgtp_expected_kept = _sm.get("expected_kept")
                metrics.acgtp_actual_keep_ratio = _sm.get("keep_ratio_actual")

                self._record_selection_path_diagnostics(
                    metrics=metrics,
                    selector_success=True,
                    exc=None,
                    requested_strategy=self.config.strategy,
                    effective_strategy=self.config.strategy,
                    selector_name="select_acgtp_v1",
                    fallback_selector=None,
                    keep_indices_np=keep_indices_np,
                    num_tokens=num_tokens,
                    selection_meta=selection_meta,
                )
                _skip_postprocess_diag = True

            except Exception as exc:
                print(f"[PRUNING] select_acgtp_v1 exception (step={self._hook_step_counter}): {type(exc).__name__}: {exc}")
                metrics.fallback_used = True
                metrics.keep_ratio_source = "fallback"
                metrics.fallback_reason = f"acgtp_v1_error:{type(exc).__name__}"
                keep_indices_np, selection_meta = select_keep_indices(
                    strategy="depth_edge_fast_diverse",
                    num_tokens=num_tokens,
                    keep_count=keep_count,
                    scores=_de_scores if _de_scores is not None else scores,
                    valid_mask=valid_mask,
                    seed=self.config.seed,
                    grid_size=self.config.token_grid_shape[0],
                    cell_grid=self.config.cell_grid,
                    reserve_tokens=reserve_tokens,
                )
                self._record_selection_path_diagnostics(
                    metrics=metrics,
                    selector_success=False,
                    exc=exc,
                    requested_strategy=self.config.strategy,
                    effective_strategy="depth_edge_fast_diverse",
                    selector_name="select_acgtp_v1",
                    fallback_selector="depth_edge_fast_diverse",
                    keep_indices_np=None,
                    num_tokens=num_tokens,
                )
                metrics.acgtp_v1 = False
                metrics.acgtp_fallback_used = True
                metrics.acgtp_fallback_reason = f"acgtp_v1_error:{type(exc).__name__}"
                _skip_postprocess_diag = True

        # ── P16: robot_geo_acgtp_v2 — Task-Semantic Anchor Branch ──────────────────────
        elif _acgtp_strategy_check == "robot_geo_acgtp_v2" and keep_indices_np is None:
            # Build geometry branch scores (same as v1)
            _scene_scores = robot_metrics.get("acgtp_scene_layout_scores")
            _de_scores = robot_metrics.get("edge_scores")
            _contact_scores = robot_metrics.get("acgtp_contact_ring_scores")
            _motion_scores = robot_metrics.get("acgtp_motion_corridor_scores")
            _action_constraint_scores = robot_metrics.get("acgtp_action_constraint_scores")
            _motion_valid = bool(robot_metrics.get("acgtp_motion_corridor_valid", False))
            _fill_mask = robot_metrics.get("acgtp_constrained_fill_mask")
            _self_core = robot_metrics.get("acgtp_self_core_mask")
            _token_u = None
            _token_v = None
            if "cache" in robot_metrics and robot_metrics["cache"]:
                _token_u = robot_metrics["cache"].get("u")
                _token_v = robot_metrics["cache"].get("v")

            # Fallback: if scene_layout not computed, derive from depth distribution
            if _scene_scores is None:
                _depth_vals = robot_metrics.get("depth_metric")
                if _depth_vals is not None:
                    _scene_scores = np.clip((np.asarray(_depth_vals, dtype=np.float32).reshape(-1) - 0.3) / 1.7, 0.0, 1.0)
                else:
                    _scene_scores = np.zeros(num_tokens, dtype=np.float32)
            # Fallback: if contact_ring not available
            if _contact_scores is None:
                _near_s = robot_metrics.get("near_scores")
                if _near_s is not None:
                    _contact_scores = np.asarray(_near_s, dtype=np.float32).reshape(-1)
                else:
                    _contact_scores = np.zeros(num_tokens, dtype=np.float32)
            # Fallback: if motion_corridor not available
            if _motion_scores is None:
                _mc = robot_metrics.get("motion_cone_scores")
                if _mc is not None:
                    _motion_scores = np.asarray(_mc, dtype=np.float32).reshape(-1)
                else:
                    _motion_scores = np.zeros(num_tokens, dtype=np.float32)
            if _de_scores is None:
                _de_scores = robot_metrics.get("edge_scores")
            if _de_scores is None:
                _de_scores = np.zeros(num_tokens, dtype=np.float32)
            # Build constrained fill mask
            if _fill_mask is None:
                _depth_metric = robot_metrics.get("depth_metric")
                if _depth_metric is not None:
                    _dm = np.asarray(_depth_metric, dtype=np.float32).reshape(-1)
                    _scene_s = np.asarray(_scene_scores, dtype=np.float32).reshape(-1)
                    _fill_mask = (np.isfinite(_dm) & (_dm > 0.1) & (_dm < 5.0) & (_scene_s > 0.0)).astype(np.float32)
                else:
                    _fill_mask = np.ones(num_tokens, dtype=np.float32)
            # Self-core mask
            if _self_core is None:
                _gripper_pixel = robot_metrics.get("gripper_pixel")
                if _gripper_pixel is not None and _token_u is not None:
                    _gx, _gy = float(_gripper_pixel[0]), float(_gripper_pixel[1])
                    _u_a = np.asarray(_token_u, dtype=np.float32).reshape(-1)
                    _v_a = np.asarray(_token_v, dtype=np.float32).reshape(-1)
                    _dist2 = (_u_a - _gx) ** 2 + (_v_a - _gy) ** 2
                    _self_core = (np.sqrt(_dist2) <= float(self.config.acgtp_self_core_radius_px)).astype(np.float32)
                else:
                    _self_core = np.zeros(num_tokens, dtype=np.float32)

            # Full module results for diagnostics
            _contact_result_metrics = robot_metrics.get("acgtp_contact_ring_result")
            _motion_result_metrics = robot_metrics.get("acgtp_motion_corridor_result")
            _scene_result_metrics = robot_metrics.get("acgtp_scene_layout_result")
            _action_constraint_result_metrics = robot_metrics.get("acgtp_action_constraint_result")

            # ── Task-Semantic Anchor Branch ────────────────────────────────────
            # Get instruction string from the robot metrics or episode state
            _instruction = None
            # Try to get instruction from robot_metrics (populated by the eval pipeline)
            if robot_metrics is not None:
                _instruction = robot_metrics.get("task_instruction") or robot_metrics.get("task_name")
            # Fall back to the latest task_name from geometry_recorder if available
            if not _instruction:
                _latest = self.geometry_recorder.get_latest() if self.geometry_recorder else None
                if _latest is not None:
                    _instruction = getattr(_latest, "task_name", None)
            if not _instruction:
                _instruction = ""

            _semantic_active = bool(self.config.acgtp_v2_semantic_enabled) and str(self.config.acgtp_v2_semantic_backend) != "none"
            if _semantic_active:
                # Parse instruction terms for diagnostics only when a semantic backend is active.
                _parsed = parse_instruction_terms(_instruction)

                _token_depth = robot_metrics.get("depth_metric")
                _token_depth_arr = np.asarray(_token_depth, dtype=np.float32).reshape(-1) if _token_depth is not None else None

                _sem_result = compute_task_semantic_anchors(
                    instruction=_instruction,
                    rgb=None,  # RGB not yet wired in this pipeline
                    token_u=_token_u,
                    token_v=_token_v,
                    token_depth=_token_depth_arr,
                    scene_result=_scene_result_metrics,
                    config=self.config,
                    semantic_enabled=True,
                    semantic_backend=str(self.config.acgtp_v2_semantic_backend),
                    w_semantic_target=float(self.config.acgtp_v2_w_semantic_target),
                    w_semantic_reference=float(self.config.acgtp_v2_w_semantic_reference),
                    w_semantic_relation=float(self.config.acgtp_v2_w_semantic_relation),
                    w_semantic_goal=float(self.config.acgtp_v2_w_semantic_goal),
                    target_cap_ratio=float(self.config.acgtp_v2_target_cap_ratio),
                    reference_cap_ratio=float(self.config.acgtp_v2_reference_cap_ratio),
                    relation_cap_ratio=float(self.config.acgtp_v2_relation_cap_ratio),
                    hard_ratio=float(self.config.acgtp_v2_semantic_hard_ratio),
                    release_quota_when_unavailable=bool(self.config.acgtp_v2_release_semantic_quota_when_unavailable),
                    grid_h=self.config.token_grid_shape[0],
                    grid_w=self.config.token_grid_shape[1],
                )
            else:
                _parsed = {
                    "parsed_target_terms": [],
                    "parsed_reference_terms": [],
                    "parsed_relation_terms": [],
                    "instruction_is_meaningful": False,
                }
                _sem_result = {
                    "semantic_anchor_scores": None,
                    "semantic_target_scores": None,
                    "semantic_reference_scores": None,
                    "semantic_relation_scores": None,
                    "semantic_goal_scores": None,
                    "semantic_confidence": 0.0,
                    "semantic_unavailable": True,
                    "semantic_fallback_reason": "semantic_disabled_fast_path",
                    "scene_layout_branch_active": False,
                    "scene_layout_available": False,
                    "scene_layout_confidence": 0.0,
                    "target_mask_count": 0,
                    "reference_mask_count": 0,
                    "relation_mask_count": 0,
                    "layout_anchor_mask_count": 0,
                    "scene_layout_indices": [],
                }

            _hist_decision: Dict[str, Any] = {"acgtp_history_enabled": False}
            if bool(getattr(self.config, "acgtp_history_enabled", False)):
                try:
                    _hist_decision = self._acgtp_history.prepare_step(
                        scene_scores=_scene_scores,
                        depth_scores=_de_scores,
                        contact_scores=_contact_scores,
                        motion_scores=_motion_scores,
                        action_scores=_action_constraint_scores,
                        valid_mask=valid_mask,
                        num_tokens=num_tokens,
                        gripper_pos=robot_metrics.get("gripper_pos"),
                        depth_valid_ratio=robot_metrics.get("depth_valid_ratio"),
                    )
                    _scene_scores = _hist_decision.get("scene_scores", _scene_scores)
                    _de_scores = _hist_decision.get("depth_scores", _de_scores)
                    _contact_scores = _hist_decision.get("contact_scores", _contact_scores)
                    _motion_scores = _hist_decision.get("motion_scores", _motion_scores)
                    _action_constraint_scores = _hist_decision.get("action_scores", _action_constraint_scores)
                except Exception as _hist_exc:
                    _hist_decision = {
                        "acgtp_history_enabled": False,
                        "acgtp_history_disabled_reason": f"history_error:{type(_hist_exc).__name__}:{str(_hist_exc)[:120]}",
                    }

            _dyn_decision: Dict[str, Any] = {"acgtp_dynamic_enabled": False}
            _dyn_w_scene = float(self.config.acgtp_w_scene_layout)
            _dyn_w_depth = float(self.config.acgtp_w_depth_structure)
            _dyn_w_contact = float(self.config.acgtp_w_contact_ring)
            _dyn_w_motion = float(self.config.acgtp_w_motion_corridor)
            _dyn_hard_ratio = float(self.config.acgtp_hard_protect_ratio)
            if bool(getattr(self.config, "acgtp_dynamic_enabled", False)):
                try:
                    _motion_norm_for_dyn = None
                    if isinstance(_motion_result_metrics, dict):
                        _motion_norm_for_dyn = _motion_result_metrics.get("motion_norm_m")
                    _dyn_decision = decide_acgtp_dynamic_budget(
                        scene_layout_scores=_scene_scores,
                        depth_structure_scores=_de_scores,
                        contact_ring_scores=_contact_scores,
                        motion_corridor_scores=_motion_scores,
                        action_constraint_scores=_action_constraint_scores,
                        valid_mask=valid_mask,
                        constrained_fill_mask=_fill_mask,
                        num_tokens=num_tokens,
                        base_keep_ratio=float(self.config.keep_ratio),
                        previous_state=self._acgtp_dynamic_state,
                        motion_corridor_valid=bool(_motion_valid),
                        motion_norm_m=_motion_norm_for_dyn,
                        depth_valid_ratio=robot_metrics.get("depth_valid_ratio"),
                        min_keep_ratio=max(
                            float(self.config.acgtp_dynamic_min_keep_ratio),
                            (
                                float(self.config.keep_ratio)
                                if not bool(getattr(self.config, "acgtp_dynamic_allow_below_base_keep_ratio", False))
                                else float(self.config.acgtp_dynamic_min_keep_ratio)
                            ) + (
                                float(self.config.acgtp_history_conservative_keep_boost)
                                if bool(_hist_decision.get("acgtp_history_conservative_mode")) else 0.0
                            ),
                        ),
                        max_keep_ratio=float(self.config.acgtp_dynamic_max_keep_ratio),
                        risk_boost_scale=float(self.config.acgtp_dynamic_risk_boost_scale),
                        confidence_prune_scale=float(self.config.acgtp_dynamic_confidence_prune_scale),
                        contact_phase_gate=str(getattr(self.config, "acgtp_dynamic_contact_phase_gate", "legacy_peak")),
                        respect_phase_min_on_candidate_gap=bool(getattr(self.config, "acgtp_dynamic_respect_phase_min_on_candidate_gap", False)),
                        shadow_contact_guard_enabled=bool(getattr(self.config, "acgtp_dynamic_shadow_contact_guard_enabled", False)),
                        shadow_contact_depth_weight_floor=float(getattr(self.config, "acgtp_dynamic_shadow_contact_depth_weight_floor", 0.30)),
                        shadow_contact_contact_weight_floor=float(getattr(self.config, "acgtp_dynamic_shadow_contact_contact_weight_floor", 0.24)),
                        shadow_contact_hard_ratio_floor=float(getattr(self.config, "acgtp_dynamic_shadow_contact_hard_ratio_floor", 0.70)),
                    )
                    _state = _dyn_decision.pop("_state", None)
                    if isinstance(_state, dict):
                        self._acgtp_dynamic_state = _state
                    keep_count = max(1, min(num_tokens, int(_dyn_decision.get("acgtp_dynamic_keep_k", keep_count))))
                    _dyn_w_scene = float(_dyn_decision.get("acgtp_dynamic_scene_weight", _dyn_w_scene))
                    _dyn_w_depth = float(_dyn_decision.get("acgtp_dynamic_depth_weight", _dyn_w_depth))
                    _dyn_w_contact = float(_dyn_decision.get("acgtp_dynamic_contact_weight", _dyn_w_contact))
                    _dyn_w_motion = float(_dyn_decision.get("acgtp_dynamic_motion_weight", _dyn_w_motion))
                    _dyn_hard_ratio = float(_dyn_decision.get("acgtp_dynamic_hard_protect_ratio", _dyn_hard_ratio))
                    if bool(_hist_decision.get("acgtp_history_conservative_mode")):
                        _dyn_hard_ratio = min(0.82, _dyn_hard_ratio + float(self.config.acgtp_history_conservative_hard_boost))
                        _dyn_w_scene += 0.03
                        _dyn_w_depth += 0.03
                        _ws = max(1e-8, _dyn_w_scene + _dyn_w_depth + _dyn_w_contact + _dyn_w_motion)
                        _dyn_w_scene, _dyn_w_depth, _dyn_w_contact, _dyn_w_motion = [
                            float(x / _ws) for x in (_dyn_w_scene, _dyn_w_depth, _dyn_w_contact, _dyn_w_motion)
                        ]
                        _dyn_decision["acgtp_history_keep_boost_applied"] = True
                    else:
                        _dyn_decision["acgtp_history_keep_boost_applied"] = False
                    metrics.dynamic_enabled = True
                    metrics.dynamic_phase = _dyn_decision.get("acgtp_dynamic_phase")
                    metrics.dynamic_keep_ratio = _dyn_decision.get("acgtp_dynamic_keep_ratio")
                    metrics.dynamic_keep_k = keep_count
                    metrics.dynamic_keep_reason = _dyn_decision.get("acgtp_dynamic_keep_reason")
                    metrics.geo_risk_score = _dyn_decision.get("acgtp_dynamic_risk")
                    metrics.geo_risk_level = _dyn_decision.get("acgtp_dynamic_phase")
                    metrics.keep_ratio_source = "acgtp_dynamic_controller"
                    metrics.effective_keep_count = keep_count
                except Exception as _dyn_exc:
                    _dyn_decision = {
                        "acgtp_dynamic_enabled": False,
                        "acgtp_dynamic_disabled_reason": f"controller_error:{type(_dyn_exc).__name__}:{str(_dyn_exc)[:120]}",
                    }
                    metrics.dynamic_enabled = False
                    metrics.dynamic_keep_reason = _dyn_decision["acgtp_dynamic_disabled_reason"]

            _use_acgtp_fast_selector = (
                bool(getattr(self.config, "acgtp_fast_selector_enabled", True))
                and not bool(getattr(self.config, "acgtp_full_diagnostics_enabled", False))
                and not _semantic_active
            )
            _acgtp_selector_fn = select_acgtp_v2_fast if _use_acgtp_fast_selector else select_acgtp_v2

            try:
                keep_indices_np, selection_meta = _acgtp_selector_fn(
                    scene_layout_scores=_scene_scores,
                    depth_edge_scores=_de_scores,
                    contact_ring_scores=_contact_scores,
                    motion_corridor_scores=_motion_scores,
                    semantic_anchor_scores=_sem_result.get("semantic_anchor_scores"),
                    semantic_target_scores=_sem_result.get("semantic_target_scores"),
                    semantic_reference_scores=_sem_result.get("semantic_reference_scores"),
                    semantic_relation_scores=_sem_result.get("semantic_relation_scores"),
                    semantic_goal_scores=_sem_result.get("semantic_goal_scores"),
                    valid_mask=valid_mask,
                    keep_k=keep_count,
                    constrained_fill_mask=_fill_mask,
                    token_u=_token_u,
                    token_v=_token_v,
                    grid_h=self.config.token_grid_shape[0],
                    grid_w=self.config.token_grid_shape[1],
                    w_scene_layout=_dyn_w_scene,
                    w_depth_structure=_dyn_w_depth,
                    w_contact_ring=_dyn_w_contact,
                    w_motion_corridor=_dyn_w_motion,
                    w_semantic=0.20,
                    hard_protect_ratio=_dyn_hard_ratio,
                    motion_corridor_valid=_motion_valid,
                    self_core_mask=_self_core,
                    contact_ring_inner_px=float(self.config.acgtp_contact_ring_inner_px),
                    contact_ring_outer_px=float(self.config.acgtp_contact_ring_outer_px),
                    contact_requires_edge_or_object=bool(self.config.acgtp_contact_requires_edge_or_object),
                    depth_edge_score_for_gate=_de_scores,
                    _motion_result_for_diag=_motion_result_metrics,
                    _scene_result_for_diag=_scene_result_metrics,
                    action_constraint_scores=_action_constraint_scores,
                    _action_constraint_result_for_diag=_action_constraint_result_metrics,
                    support_plane_cap_ratio=float(self.config.acgtp_scene_support_plane_cap_ratio),
                    # ── P16 semantic params ──
                    semantic_enabled=bool(self.config.acgtp_v2_semantic_enabled) and _semantic_active,
                    semantic_backend=str(self.config.acgtp_v2_semantic_backend),
                    semantic_confidence=float(_sem_result.get("semantic_confidence", 0.0)),
                    semantic_unavailable=bool(_sem_result.get("semantic_unavailable", True)),
                    semantic_fallback_reason=_sem_result.get("semantic_fallback_reason"),
                    release_semantic_quota_when_unavailable=bool(self.config.acgtp_v2_release_semantic_quota_when_unavailable),
                    w_semantic_target=float(self.config.acgtp_v2_w_semantic_target),
                    w_semantic_reference=float(self.config.acgtp_v2_w_semantic_reference),
                    w_semantic_relation=float(self.config.acgtp_v2_w_semantic_relation),
                    w_semantic_goal=float(self.config.acgtp_v2_w_semantic_goal),
                    target_cap_ratio=float(self.config.acgtp_v2_target_cap_ratio),
                    reference_cap_ratio=float(self.config.acgtp_v2_reference_cap_ratio),
                    relation_cap_ratio=float(self.config.acgtp_v2_relation_cap_ratio),
                    hard_semantic_ratio=float(self.config.acgtp_v2_semantic_hard_ratio),
                    parsed_target_terms=_parsed.get("parsed_target_terms"),
                    parsed_reference_terms=_parsed.get("parsed_reference_terms"),
                    parsed_relation_terms=_parsed.get("parsed_relation_terms"),
                    instruction_is_meaningful=bool(_parsed.get("instruction_is_meaningful", False)),
                    # ── P16 scene-layout branch params (from semantic backend) ──
                    scene_layout_branch_active=bool(_sem_result.get("scene_layout_branch_active", False)),
                    scene_layout_available=bool(_sem_result.get("scene_layout_available", False)),
                    scene_layout_confidence=float(_sem_result.get("scene_layout_confidence", 0.0)),
                    target_mask_count=int(_sem_result.get("target_mask_count", 0)),
                    reference_mask_count=int(_sem_result.get("reference_mask_count", 0)),
                    relation_mask_count=int(_sem_result.get("relation_mask_count", 0)),
                    layout_anchor_mask_count=int(_sem_result.get("layout_anchor_mask_count", 0)),
                    scene_layout_indices=_sem_result.get("scene_layout_indices", []),
                    # ── P16 attention branch params (disabled in strict fallback; enabled when configured) ──
                    acgtp_attention_enabled=False,  # disabled in this pipeline; set True to enable
                    acgtp_attention_backend="none",
                    acgtp_attention_min_confidence=0.0,
                    acgtp_attention_requires_geometry_alignment=True,
                    acgtp_attention_budget_ratio=0.10,
                    acgtp_attention_task_relevance_score=None,
                    acgtp_attention_task_relevance_mask=None,
                    acgtp_attention_source="none",
                    acgtp_attention_available=False,
                    acgtp_attention_confidence=0.0,
                )

                # Record ACGTP-v2 metrics
                if isinstance(_hist_decision, dict):
                    selection_meta.update({k: v for k, v in _hist_decision.items() if not k.endswith("_scores") and k not in ("scene_scores", "depth_scores", "contact_scores", "motion_scores", "action_scores")})
                if isinstance(_dyn_decision, dict):
                    selection_meta.update(_dyn_decision)
                _sm = selection_meta
                for _dyn_key in (
                    "acgtp_history_enabled",
            "acgtp_history_length",
            "acgtp_history_capacity",
            "acgtp_history_length_after_update",
            "acgtp_history_warmup",
            "acgtp_history_ema_available",
            "acgtp_history_smoothing_applied",
            "acgtp_history_conservative_mode",
            "acgtp_history_conservative_reason",
            "acgtp_history_disabled_reason",
            "acgtp_history_depth_change",
            "acgtp_history_keep_mask_iou",
            "acgtp_history_motion_stability",
            "acgtp_history_depth_valid_ratio",
            "acgtp_history_scene_ema_alpha",
            "acgtp_history_depth_ema_alpha",
            "acgtp_history_contact_ema_alpha",
            "acgtp_history_motion_ema_alpha",
            "acgtp_history_action_ema_alpha",
            "acgtp_history_phase_switch",
            "acgtp_history_force_conservative_next",
            "acgtp_history_force_conservative_reason",
            "acgtp_history_keep_boost_applied",
            "acgtp_dynamic_enabled",
                    "acgtp_dynamic_phase",
                    "acgtp_dynamic_candidate_phase",
                    "acgtp_dynamic_previous_phase",
                    "acgtp_dynamic_hysteresis_state",
                    "acgtp_dynamic_risk",
                    "acgtp_dynamic_confidence",
                    "acgtp_dynamic_keep_ratio",
                    "acgtp_dynamic_keep_k",
                    "acgtp_dynamic_base_keep_ratio",
                    "acgtp_dynamic_raw_keep_ratio",
                    "acgtp_dynamic_phase_min_keep_ratio",
                    "acgtp_dynamic_phase_max_keep_ratio",
                    "acgtp_dynamic_lock_strength",
                    "acgtp_dynamic_uncertainty_boost",
                    "acgtp_dynamic_risk_boost",
                    "acgtp_dynamic_prune_gain",
                    "acgtp_dynamic_keep_reason",
                    "acgtp_dynamic_layout_motion_alignment",
                    "acgtp_dynamic_binary_alignment",
                    "acgtp_dynamic_contact_phase_gate",
                    "acgtp_dynamic_contact_peak",
                    "acgtp_dynamic_contact_mean",
                    "acgtp_dynamic_contact_ratio",
                    "acgtp_dynamic_motion_peak",
                    "acgtp_dynamic_motion_mean",
                    "acgtp_dynamic_motion_ratio",
                    "acgtp_dynamic_physical_ratio",
                    "acgtp_dynamic_high_contact",
                    "acgtp_dynamic_high_contact_coverage",
                    "acgtp_dynamic_high_contact_legacy",
                    "acgtp_dynamic_shadow_contact_guard",
                    "acgtp_dynamic_high_motion",
                    "acgtp_dynamic_strong_layout",
                    "acgtp_dynamic_action_peak",
                    "acgtp_dynamic_action_mean",
                    "acgtp_dynamic_depth_valid_ratio",
                    "acgtp_dynamic_fill_candidate_count",
                    "acgtp_dynamic_fill_candidate_ratio",
                    "acgtp_dynamic_candidate_gap_count",
                    "acgtp_dynamic_candidate_gap_ratio",
                    "acgtp_dynamic_candidate_clamped",
                    "acgtp_dynamic_scene_weight",
                    "acgtp_dynamic_depth_weight",
                    "acgtp_dynamic_contact_weight",
                    "acgtp_dynamic_motion_weight",
                    "acgtp_dynamic_hard_protect_ratio",
                    "acgtp_dynamic_budget_vector",
                    "acgtp_dynamic_disabled_reason",
                ):
                    if hasattr(metrics, _dyn_key):
                        setattr(metrics, _dyn_key, _sm.get(_dyn_key))
                metrics.acgtp_v2 = True
                metrics.acgtp_v1 = False
                metrics.acgtp_selector_version = _sm.get("acgtp_selector_version")
                metrics.acgtp_quota_policy = _sm.get("acgtp_quota_policy")
                metrics.acgtp_fill_policy = _sm.get("acgtp_fill_policy")
                metrics.acgtp_scene_quota = _sm.get("acgtp_scene_quota")
                metrics.acgtp_depth_quota = _sm.get("acgtp_depth_quota")
                metrics.acgtp_contact_quota = _sm.get("acgtp_contact_quota")
                metrics.acgtp_motion_quota = _sm.get("acgtp_motion_quota")
                metrics.acgtp_scene_quota_weight = _sm.get("acgtp_scene_quota_weight")
                metrics.acgtp_depth_quota_weight = _sm.get("acgtp_depth_quota_weight")
                metrics.acgtp_contact_quota_weight = _sm.get("acgtp_contact_quota_weight")
                metrics.acgtp_motion_quota_weight = _sm.get("acgtp_motion_quota_weight")
                metrics.acgtp_scene_allocated = _sm.get("acgtp_scene_allocated")
                metrics.acgtp_depth_allocated = _sm.get("acgtp_depth_allocated")
                metrics.acgtp_contact_allocated = _sm.get("acgtp_contact_allocated")
                metrics.acgtp_motion_allocated = _sm.get("acgtp_motion_allocated")
                metrics.acgtp_coverage_fill_candidate_count = _sm.get("acgtp_coverage_fill_candidate_count")
                metrics.acgtp_coverage_fill_candidate_ratio = _sm.get("acgtp_coverage_fill_candidate_ratio")
                metrics.acgtp_v2_semantic_enabled = _sm.get("acgtp_v2_semantic_enabled")
                metrics.acgtp_v2_semantic_backend = _sm.get("acgtp_v2_semantic_backend")
                metrics.acgtp_v2_semantic_confidence = _sm.get("acgtp_v2_semantic_confidence")
                metrics.acgtp_v2_semantic_unavailable = _sm.get("acgtp_v2_semantic_unavailable")
                metrics.acgtp_v2_semantic_fallback_reason = _sm.get("acgtp_v2_semantic_fallback_reason")
                metrics.acgtp_v2_release_quota = _sm.get("acgtp_v2_release_quota")
                metrics.acgtp_v2_parsed_instruction_meaningful = _sm.get("acgtp_v2_parsed_instruction_meaningful")
                metrics.acgtp_v2_parsed_target_terms = str(_parsed.get("parsed_target_terms", []))
                metrics.acgtp_v2_parsed_reference_terms = str(_parsed.get("parsed_reference_terms", []))
                metrics.acgtp_v2_parsed_relation_terms = str(_parsed.get("parsed_relation_terms", []))
                metrics.acgtp_v2_w_semantic_target = _sm.get("acgtp_v2_w_semantic_target")
                metrics.acgtp_v2_w_semantic_reference = _sm.get("acgtp_v2_w_semantic_reference")
                metrics.acgtp_v2_w_semantic_relation = _sm.get("acgtp_v2_w_semantic_relation")
                metrics.acgtp_v2_w_semantic_goal = _sm.get("acgtp_v2_w_semantic_goal")
                metrics.acgtp_v2_semantic_target_token_count = _sm.get("acgtp_v2_semantic_target_token_count")
                metrics.acgtp_v2_semantic_target_kept_count = _sm.get("acgtp_v2_semantic_target_token_count")
                metrics.acgtp_v2_semantic_reference_token_count = _sm.get("acgtp_v2_semantic_reference_token_count")
                metrics.acgtp_v2_semantic_relation_token_count = _sm.get("acgtp_v2_semantic_relation_token_count")
                metrics.acgtp_v2_semantic_goal_token_count = _sm.get("acgtp_v2_semantic_goal_token_count")
                metrics.acgtp_v2_semantic_anchor_token_count = _sm.get("acgtp_v2_semantic_anchor_token_count")
                metrics.selected_by_semantic_target_count = _sm.get("selected_by_semantic_target_count")
                metrics.selected_by_semantic_reference_count = _sm.get("selected_by_semantic_reference_count")
                metrics.selected_by_semantic_relation_count = _sm.get("selected_by_semantic_relation_count")
                metrics.selected_by_semantic_goal_count = _sm.get("selected_by_semantic_goal_count")
                metrics.selected_by_scene_residual_fill_count = _sm.get("selected_by_scene_residual_fill_count")
                metrics.semantic_overlap_with_scene_count = _sm.get("semantic_overlap_with_scene_count")
                metrics.semantic_overlap_with_depth_count = _sm.get("semantic_overlap_with_depth_count")
                metrics.semantic_overlap_with_contact_count = _sm.get("semantic_overlap_with_contact_count")
                metrics.semantic_overlap_with_motion_count = _sm.get("semantic_overlap_with_motion_count")
                metrics.acgtp_v2_hard_semantic_quota = _sm.get("acgtp_v2_hard_semantic_quota")
                metrics.acgtp_v2_target_cap_k = _sm.get("acgtp_v2_target_cap_k")
                metrics.acgtp_v2_reference_cap_k = _sm.get("acgtp_v2_reference_cap_k")
                metrics.acgtp_v2_relation_cap_k = _sm.get("acgtp_v2_relation_cap_k")
                metrics.acgtp_scene_selected_residual_fill_count = _sm.get("acgtp_scene_selected_residual_fill_count")
                metrics.acgtp_scene_residual_fill_token_count = _sm.get("acgtp_scene_residual_fill_token_count")
                metrics.acgtp_scene_residual_fill_token_count_computed = _sm.get("acgtp_scene_residual_fill_token_count_computed")
                metrics.selected_by_semantic_count = _sm.get("selected_by_semantic_count")

                # P16: Strict fallback dispatch fields
                metrics.strict_fallback_dispatch_used = _sm.get("strict_fallback_dispatch_used")
                metrics.delegated_selector_name = _sm.get("delegated_selector_name")
                metrics.fallback_dispatch_to_v1 = _sm.get("fallback_dispatch_to_v1")
                # P16: Semantic state aliases
                metrics.semantic_backend = _sm.get("acgtp_v2_semantic_backend")
                metrics.semantic_unavailable = _sm.get("semantic_unavailable")
                metrics.semantic_confidence = _sm.get("semantic_confidence")
                metrics.semantic_available = _sm.get("acgtp_v2_semantic_available")
                metrics.semantic_quota_released = _sm.get("acgtp_v2_release_quota")
                # P16: Attention state
                metrics.attention_backend = _sm.get("acgtp_attention_backend")
                metrics.attention_source = _sm.get("acgtp_attention_source")
                metrics.attention_available = _sm.get("acgtp_attention_available")
                metrics.attention_confidence = _sm.get("acgtp_attention_confidence")
                metrics.attention_quota_released = _sm.get("acgtp_attention_quota_released")
                metrics.selected_by_attention_count = _sm.get("attention_selected_by_final_count")
                metrics.attention_only_token_count = _sm.get("acgtp_attention_only_token_count")
                metrics.attention_selected_by_final_count = _sm.get("attention_selected_by_final_count")
                metrics.attention_top_count = _sm.get("acgtp_attention_top_count")

                # Re-use v1 geometry branch metrics from selection_meta
                metrics.acgtp_w_scene_layout = _sm.get("acgtp_w_scene_layout")
                metrics.acgtp_w_depth_structure = _sm.get("acgtp_w_depth_structure")
                metrics.acgtp_w_contact_ring = _sm.get("acgtp_w_contact_ring")
                metrics.acgtp_w_motion_corridor = _sm.get("acgtp_w_motion_corridor")
                metrics.acgtp_self_core_radius_px = _sm.get("acgtp_self_core_radius_px")
                metrics.acgtp_contact_ring_inner_px = _sm.get("acgtp_contact_ring_inner_px")
                metrics.acgtp_contact_ring_outer_px = _sm.get("acgtp_contact_ring_outer_px")
                metrics.acgtp_self_core_token_count = _sm.get("acgtp_self_core_token_count")
                metrics.acgtp_self_core_token_ratio = _sm.get("acgtp_self_core_token_ratio")
                metrics.acgtp_contact_ring_token_count = _sm.get("acgtp_contact_ring_token_count")
                metrics.acgtp_contact_ring_token_ratio = _sm.get("acgtp_contact_ring_token_ratio")
                metrics.acgtp_contact_ring_gated_token_count = _sm.get("acgtp_contact_ring_gated_token_count")
                metrics.acgtp_contact_ring_valid = _sm.get("acgtp_contact_ring_valid")
                metrics.acgtp_support_plane_token_count = _sm.get("acgtp_support_plane_token_count")
                metrics.acgtp_support_plane_candidate_count = _sm.get("acgtp_support_plane_candidate_count")
                metrics.acgtp_object_component_token_count = _sm.get("acgtp_object_component_token_count")
                metrics.acgtp_boundary_token_count = _sm.get("acgtp_boundary_token_count")
                metrics.acgtp_scene_fill_candidate_count = _sm.get("acgtp_scene_fill_candidate_count")
                metrics.acgtp_scene_fill_candidate_ratio = _sm.get("acgtp_scene_fill_candidate_ratio")
                metrics.acgtp_scene_support_plane_cap_ratio = _sm.get("acgtp_scene_support_plane_cap_ratio")
                metrics.acgtp_scene_support_plane_cap_used = _sm.get("acgtp_scene_support_plane_cap_used")
                metrics.acgtp_scene_support_plane_fallback_used = _sm.get("acgtp_scene_support_plane_fallback_used")
                metrics.acgtp_scene_support_plane_fallback_reason = _sm.get("acgtp_scene_support_plane_fallback_reason")
                metrics.acgtp_scene_object_component_fallback_used = _sm.get("acgtp_scene_object_component_fallback_used")
                metrics.acgtp_scene_object_component_fallback_reason = _sm.get("acgtp_scene_object_component_fallback_reason")
                metrics.acgtp_scene_object_component_num_components = _sm.get("acgtp_scene_object_component_num_components")
                metrics.acgtp_scene_boundary_fallback_used = _sm.get("acgtp_scene_boundary_fallback_used")
                metrics.acgtp_scene_boundary_fallback_reason = _sm.get("acgtp_scene_boundary_fallback_reason")
                metrics.acgtp_scene_boundary_from_object_count = _sm.get("acgtp_scene_boundary_from_object_count")
                metrics.acgtp_scene_boundary_from_depth_count = _sm.get("acgtp_scene_boundary_from_depth_count")
                metrics.acgtp_scene_selected_support_plane_count = _sm.get("acgtp_scene_selected_support_plane_count")
                metrics.acgtp_scene_selected_object_component_count = _sm.get("acgtp_scene_selected_object_component_count")
                metrics.acgtp_scene_selected_boundary_count = _sm.get("acgtp_scene_selected_boundary_count")
                metrics.acgtp_scene_selected_relation_count = _sm.get("acgtp_scene_selected_relation_count")
                metrics.acgtp_scene_support_plane_selected_ratio = _sm.get("acgtp_scene_support_plane_selected_ratio")
                metrics.acgtp_motion_corridor_valid = _motion_valid
                metrics.acgtp_motion_corridor_score_mean = _sm.get("acgtp_motion_corridor_score_mean")
                metrics.acgtp_motion_corridor_score_max = _sm.get("acgtp_motion_corridor_score_max")
                metrics.acgtp_motion_norm_m = _sm.get("acgtp_motion_norm_m")
                metrics.acgtp_motion_ema_alpha = _sm.get("acgtp_motion_ema_alpha")
                metrics.acgtp_depth_structure_score_mean = _sm.get("acgtp_depth_structure_score_mean")
                metrics.acgtp_depth_structure_score_max = _sm.get("acgtp_depth_structure_score_max")
                metrics.acgtp_action_constraint_score_mean = _sm.get("acgtp_action_constraint_score_mean")
                metrics.acgtp_action_constraint_score_max = _sm.get("acgtp_action_constraint_score_max")
                metrics.acgtp_action_constraint_source = _sm.get("acgtp_action_constraint_source")
                metrics.acgtp_future_action_constraint_enabled = _sm.get("acgtp_future_action_constraint_enabled")
                metrics.acgtp_future_action_constraint_valid = _sm.get("acgtp_future_action_constraint_valid")
                metrics.acgtp_future_action_constraint_disabled_reason = _sm.get("acgtp_future_action_constraint_disabled_reason")
                metrics.acgtp_future_action_constraint_score_mean = _sm.get("acgtp_future_action_constraint_score_mean")
                metrics.acgtp_future_action_constraint_score_max = _sm.get("acgtp_future_action_constraint_score_max")
                metrics.acgtp_object_side_contact_score_mean = _sm.get("acgtp_object_side_contact_score_mean")
                metrics.acgtp_object_side_contact_score_max = _sm.get("acgtp_object_side_contact_score_max")
                metrics.acgtp_swept_motion_risk_score_mean = _sm.get("acgtp_swept_motion_risk_score_mean")
                metrics.acgtp_swept_motion_risk_score_max = _sm.get("acgtp_swept_motion_risk_score_max")
                metrics.acgtp_collision_contact_risk_score_mean = _sm.get("acgtp_collision_contact_risk_score_mean")
                metrics.acgtp_collision_contact_risk_score_max = _sm.get("acgtp_collision_contact_risk_score_max")
                metrics.acgtp_contact_object_overlap_count = _sm.get("acgtp_contact_object_overlap_count")
                metrics.acgtp_robot_self_penalty_count = _sm.get("acgtp_robot_self_penalty_count")
                metrics.acgtp_action_constraint_ms = robot_metrics.get("acgtp_action_constraint_ms")
                metrics.acgtp_hard_protect_count = _sm.get("acgtp_hard_protect_count")
                metrics.acgtp_hard_protect_ratio = _sm.get("acgtp_hard_protect_ratio")
                metrics.acgtp_hard_protect_valid = _sm.get("acgtp_hard_protect_valid")
                metrics.selected_by_scene_layout_count = _sm.get("selected_by_scene_layout_count")
                metrics.selected_by_depth_structure_count = _sm.get("selected_by_depth_structure_count")
                metrics.selected_by_contact_ring_count = _sm.get("selected_by_contact_ring_count")
                metrics.selected_by_motion_corridor_count = _sm.get("selected_by_motion_corridor_count")
                metrics.selected_by_constrained_fill_count = _sm.get("selected_by_constrained_fill_count")
                metrics.selected_by_acgtp_fallback_count = _sm.get("selected_by_acgtp_fallback_count")
                metrics.overlap_scene_depth_count = _sm.get("overlap_scene_depth_count")
                metrics.overlap_scene_contact_count = _sm.get("overlap_scene_contact_count")
                metrics.overlap_contact_motion_count = _sm.get("overlap_contact_motion_count")
                metrics.overlap_scene_motion_count = _sm.get("overlap_scene_motion_count")
                metrics.overlap_depth_contact_count = _sm.get("overlap_depth_contact_count")
                metrics.overlap_depth_motion_count = _sm.get("overlap_depth_motion_count")
                metrics.acgtp_branch_accounting_valid = _sm.get("acgtp_branch_accounting_valid")
                metrics.acgtp_branch_sum = _sm.get("acgtp_branch_sum")
                metrics.acgtp_branch_sum_error = _sm.get("acgtp_branch_sum_error")
                metrics.branch_accounting_valid = _sm.get("branch_accounting_valid")
                metrics.branch_sum_equals_kept = _sm.get("branch_sum_equals_kept")
                metrics.acgtp_hard_protect_ratio_config = _sm.get("acgtp_hard_protect_ratio")
                metrics.acgtp_fallback_used = _sm.get("acgtp_fallback_used")
                metrics.acgtp_fallback_reason = _sm.get("acgtp_fallback_reason")
                metrics.acgtp_motion_disabled_reason = _sm.get("acgtp_motion_disabled_reason")
                metrics.acgtp_constrained_fill_mask = _sm.get("acgtp_constrained_fill_mask")
                metrics.acgtp_final_kept = _sm.get("final_kept")
                metrics.acgtp_expected_kept = _sm.get("expected_kept")
                metrics.acgtp_actual_keep_ratio = _sm.get("keep_ratio_actual")

                # Gripper pixel from contact ring result
                if _contact_result_metrics is not None:
                    metrics.gripper_pixel_u = _contact_result_metrics.get("gripper_pixel_u")
                    metrics.gripper_pixel_v = _contact_result_metrics.get("gripper_pixel_v")
                    metrics.gripper_in_bounds = _contact_result_metrics.get("gripper_in_bounds")

                self._record_selection_path_diagnostics(
                    metrics=metrics,
                    selector_success=True,
                    exc=None,
                    requested_strategy=self.config.strategy,
                    effective_strategy=self.config.strategy,
                    selector_name=_sm.get("selector_function_name", "select_acgtp_v2"),
                    fallback_selector=None,
                    keep_indices_np=keep_indices_np,
                    num_tokens=num_tokens,
                    selection_meta=selection_meta,
                )
                _skip_postprocess_diag = True

            except Exception as exc:
                print(f"[PRUNING] {_acgtp_selector_fn.__name__} exception (step={self._hook_step_counter}): {type(exc).__name__}: {exc}")
                metrics.fallback_used = True
                metrics.keep_ratio_source = "fallback"
                metrics.fallback_reason = f"acgtp_v2_error:{type(exc).__name__}"
                keep_indices_np, selection_meta = select_keep_indices(
                    strategy="depth_edge_fast_diverse",
                    num_tokens=num_tokens,
                    keep_count=keep_count,
                    scores=_de_scores if _de_scores is not None else scores,
                    valid_mask=valid_mask,
                    seed=self.config.seed,
                    grid_size=self.config.token_grid_shape[0],
                    cell_grid=self.config.cell_grid,
                    reserve_tokens=reserve_tokens,
                )
                self._record_selection_path_diagnostics(
                    metrics=metrics,
                    selector_success=False,
                    exc=exc,
                    requested_strategy=self.config.strategy,
                    effective_strategy="depth_edge_fast_diverse",
                    selector_name=_acgtp_selector_fn.__name__,
                    fallback_selector="depth_edge_fast_diverse",
                    keep_indices_np=None,
                    num_tokens=num_tokens,
                )
                metrics.acgtp_v2 = False
                metrics.acgtp_fallback_used = True
        # for real exceptional paths, not for normal random/uniform/depth-edge use.
        # IMPORTANT: hybrid_budget_v2 and robot_geo_acgtp_v1 handle their own return path above (try/except block).
        if keep_indices_np is None and self.config.strategy not in SELF_HANDLED_SELECTOR_STRATEGIES:
            keep_indices_np, selection_meta = select_keep_indices(
                strategy=strategy,
                num_tokens=num_tokens,
                keep_count=keep_count,
                scores=scores,
                valid_mask=valid_mask,
                seed=self.config.seed,
                grid_size=self.config.token_grid_shape[0],
                cell_grid=self.config.cell_grid,
                reserve_tokens=reserve_tokens,
            )
            selection_meta.setdefault("fallback_used", bool(metrics.fallback_used))
            selection_meta.setdefault("fallback_reason", metrics.fallback_reason)
        # ~L871: Final validation block start
        # Check the local fallback_reason AND metrics.fallback_reason (some branches
        # set metrics.fallback_reason but may not have set the local variable).
        # NOTE: this None safety net must run BEFORE the effective_keep_count
        # setdefault below, otherwise int(len(keep_indices_np)) raises on the
        # legacy SELF_HANDLED paths whose selector threw and left it None.
        _existing_reason = fallback_reason or metrics.fallback_reason
        if keep_indices_np is None:
            _fallback_msg = _existing_reason or f"safe_fallback_uninitialized_strategy:{self.config.strategy}"
            if self.config.debug:
                print(f"[PRUNING] WARNING: keep_indices_np was None for strategy={self.config.strategy}, using safe fallback")
            if not metrics.fallback_used:
                metrics.fallback_used = True
            metrics.keep_ratio_source = "fallback"
            metrics.fallback_reason = _fallback_msg
            strategy = "depth_edge_fast" if scores is None else "depth_edge_fast_diverse"
            keep_count_safe = int(round(num_tokens * 0.75))
            keep_indices_np, selection_meta = select_keep_indices(
                strategy=strategy,
                num_tokens=num_tokens,
                keep_count=keep_count_safe,
                scores=scores,
                valid_mask=valid_mask,
                seed=self.config.seed,
                grid_size=self.config.token_grid_shape[0],
                cell_grid=self.config.cell_grid,
                reserve_tokens=reserve_tokens,
            )
            selection_meta["fallback_used"] = True
            selection_meta["fallback_reason"] = _fallback_msg
            selection_meta["actual_keep_ratio"] = len(keep_indices_np) / num_tokens if num_tokens else 0.0
            selection_meta["ema_used_for_selection"] = False
            selection_meta["interaction_lock"] = False
            selection_meta["lock_reason"] = _fallback_msg
        try:
            selection_meta.setdefault("requested_keep_ratio", float(self.config.keep_ratio))
            selection_meta.setdefault("keep_ratio_source", metrics.keep_ratio_source or self._default_keep_ratio_source())
            selection_meta.setdefault("effective_keep_count", int(len(keep_indices_np)))
            selection_meta.setdefault("original_token_count", int(num_tokens))
        except Exception as _setdefault_exc:
            print(f"[PRUNING] step={self._hook_step_counter} setdefault EXCEPTION: {_setdefault_exc}")
            raise

        # Ensure we never use stale indices from a previous call
        if len(keep_indices_np) == 0:
            _fallback_empty = _existing_reason or "safe_fallback_empty_indices"
            if not metrics.fallback_used:
                metrics.fallback_used = True
            metrics.keep_ratio_source = "fallback"
            metrics.fallback_reason = _fallback_empty
            strategy = "depth_edge_fast"
            keep_count_safe = int(round(num_tokens * 0.75))
            keep_indices_np, selection_meta = select_keep_indices(
                strategy=strategy,
                num_tokens=num_tokens,
                keep_count=keep_count_safe,
                scores=scores,
                valid_mask=valid_mask,
                seed=self.config.seed,
                grid_size=self.config.token_grid_shape[0],
                cell_grid=self.config.cell_grid,
                reserve_tokens=reserve_tokens,
            )
            selection_meta["fallback_used"] = True
            selection_meta["fallback_reason"] = _fallback_empty
            selection_meta["actual_keep_ratio"] = len(keep_indices_np) / num_tokens if num_tokens else 0.0

        # Always ensure sorted, unique indices
        keep_indices_np = np.unique(keep_indices_np).astype(np.int64)
        if keep_indices_np.ndim == 0:
            keep_indices_np = np.array([int(keep_indices_np)], dtype=np.int64)
        # Clamp to valid range
        keep_indices_np = keep_indices_np[(keep_indices_np >= 0) & (keep_indices_np < num_tokens)]
        # Ensure we have at least 1 token
        if len(keep_indices_np) == 0:
            keep_indices_np = np.array([0], dtype=np.int64)
        # Sort and deduplicate
        keep_indices_np = np.sort(np.unique(keep_indices_np))
        # ── actual_keep_ratio must come from final len / num_tokens, NOT config ─
        _final_actual_keep_ratio = len(keep_indices_np) / num_tokens if num_tokens else 1.0
        selection_meta["actual_keep_ratio"] = _final_actual_keep_ratio
        selection_meta.setdefault("requested_keep_ratio", float(self.config.keep_ratio))
        selection_meta.setdefault("keep_ratio_source", metrics.keep_ratio_source or self._default_keep_ratio_source())
        selection_meta.setdefault("effective_keep_count", int(len(keep_indices_np)))
        selection_meta.setdefault("original_token_count", int(num_tokens))
        metrics.actual_keep_ratio = _final_actual_keep_ratio
        metrics.effective_keep_count = int(len(keep_indices_np))
        metrics.original_token_count = int(num_tokens)

        # Stage X/P6: normalize selector diagnostics once on the common path.
        # Inline selector calls may have recorded older fields already; this
        # common pass fills missing fields without using one coarse guard.
        _sel_success = keep_indices_np is not None
        # ── hybrid_budget_v2: phase accounting already set in its branch; skip
        # post-processing _record_selection_path_diagnostics to avoid overwriting it
        if _skip_postprocess_diag:
            _skip_postprocess_diag = False
            # Still need to finalize scores and return
            pass
        else:
            _sel_exc = getattr(self, "_last_selector_exception", None)
            _sel_req = self.config.strategy
            _sel_eff = "depth_edge_fast_diverse" if (fallback_reason or getattr(metrics, "fallback_reason", None) or "").startswith("safe_fallback") else _sel_req
            _sel_name = (
                selection_meta.get("selector_function_name")
                or metrics.selector_function_name
                or getattr(self, "_last_selector_name", None)
                or "select_keep_indices"
            )
            _sel_fb = "depth_edge_fast_diverse" if _sel_req != _sel_eff else None
            if metrics.fallback_used and _sel_fb is None and _sel_eff != "depth_edge_fast_diverse":
                _sel_fb = selection_meta.get("fallback_selector_name")
            selection_meta = finalize_selection_debug_info(
                selection_meta,
                selector_function_name=str(_sel_name),
                strategy=str(selection_meta.get("strategy") or _sel_eff),
                keep_indices=keep_indices_np,
                num_tokens=num_tokens,
                keep_count=keep_count,
                scores=scores,
                requested_keep_ratio=float(self.config.keep_ratio),
                fallback_used=bool(metrics.fallback_used or selection_meta.get("fallback_used")),
                fallback_reason=metrics.fallback_reason or selection_meta.get("fallback_reason"),
                selection_error=selection_meta.get("selection_error"),
                selection_warning=selection_meta.get("selection_warning"),
            )
            self._record_selection_path_diagnostics(
                metrics=metrics,
                selector_success=_sel_success,
                exc=_sel_exc,
                requested_strategy=_sel_req,
                effective_strategy=str(selection_meta.get("strategy") or _sel_eff),
                selector_name=str(_sel_name),
                fallback_selector=_sel_fb or selection_meta.get("fallback_selector_name"),
                keep_indices_np=keep_indices_np,
                num_tokens=num_tokens,
                selection_meta=selection_meta,
            )
        # Reset per-step state
        if self.config.strategy == "robot_geo_acgtp_v2" and "robot_metrics" in locals() and not metrics.fallback_used:
            _hist_update = self._acgtp_history.update_after_selection(
                keep_indices=keep_indices_np,
                num_tokens=num_tokens,
                dynamic_decision=selection_meta,
                gripper_pos=robot_metrics.get("gripper_pos"),
            )
            selection_meta.update(_hist_update)
            for _hist_key in (
                "acgtp_history_enabled",
                "acgtp_history_length",
                "acgtp_history_capacity",
                "acgtp_history_length_after_update",
                "acgtp_history_warmup",
                "acgtp_history_ema_available",
                "acgtp_history_smoothing_applied",
                "acgtp_history_conservative_mode",
                "acgtp_history_conservative_reason",
                "acgtp_history_disabled_reason",
                "acgtp_history_depth_change",
                "acgtp_history_keep_mask_iou",
                "acgtp_history_motion_stability",
                "acgtp_history_depth_valid_ratio",
                "acgtp_history_scene_ema_alpha",
                "acgtp_history_depth_ema_alpha",
                "acgtp_history_contact_ema_alpha",
                "acgtp_history_motion_ema_alpha",
                "acgtp_history_action_ema_alpha",
                "acgtp_history_phase_switch",
                "acgtp_history_force_conservative_next",
                "acgtp_history_force_conservative_reason",
                "acgtp_history_keep_boost_applied",
            ):
                if hasattr(metrics, _hist_key):
                    setattr(metrics, _hist_key, selection_meta.get(_hist_key))
        self._last_selector_exception = None
        self._last_selector_name = None

        # selection_meta already set inside hybrid branches above; no redundant re-init needed
        if keep_indices_np is None and self.config.strategy not in (SELF_HANDLED_SELECTOR_STRATEGIES - {"robot_geo_branch_budget_v0", "robot_geo_acgtp_v2"}):
            keep_indices_np, selection_meta = select_keep_indices(
                strategy=strategy,
                num_tokens=num_tokens,
                keep_count=keep_count,
                scores=scores,
                valid_mask=valid_mask,
                seed=self.config.seed,
                grid_size=self.config.token_grid_shape[0],
                cell_grid=self.config.cell_grid,
                reserve_tokens=reserve_tokens,
            )
        metrics.timing.selection_ms = (time.perf_counter() - select_start) * 1000.0
        metrics.timing.pruning_time_ms = metrics.timing.selection_ms
        metrics.timing.topk_pruning_ms = metrics.timing.selection_ms
        if self.config.detailed_pruning_timing:
            metrics.timing.edge_selection_ms = selection_meta.get("edge_selection_ms")
            metrics.timing.geo_selection_ms = selection_meta.get("geo_selection_ms")
            metrics.timing.diverse_selection_ms = selection_meta.get("diverse_selection_ms")
            metrics.timing.final_merge_ms = selection_meta.get("final_merge_ms")

        validation = validate_keep_indices(keep_indices_np, len(keep_indices_np))
        metrics.keep_indices_sorted = validation["keep_indices_sorted"]
        metrics.duplicate_indices_count = validation["duplicate_indices_count"]
        _apply_aux = dict(active_aux_metrics)
        _apply_aux["num_tokens"] = num_tokens
        # P1-1: Record attribution_missing_reason for fallback path (robot_metrics not populated)
        if not _apply_aux:
            _apply_aux["attribution_missing_reason"] = (
                "hybrid_temporal_v1_fallback:robot_metrics_not_populated;"
                "edge_scores_unavailable_for_attribution"
            )
        self._apply_selection_metrics(metrics, selection_meta, keep_indices_np, _apply_aux)
        if self.config.strategy == "robot_geo_temporal_v0" and "robot_metrics" in locals() and not metrics.fallback_used:
            self._update_temporal_history_after_selection(
                keep_indices_np=keep_indices_np,
                num_tokens=num_tokens,
                scores=scores,
                robot_metrics=robot_metrics,
                dynamic_keep_ratio=metrics.dynamic_keep_ratio,
            )
        if self.config.strategy == "robot_geo_hybrid_temporal_v1" and "robot_metrics" in locals() and not metrics.fallback_used:
            self._update_temporal_history_after_selection(
                keep_indices_np=keep_indices_np,
                num_tokens=num_tokens,
                scores=scores,
                robot_metrics=robot_metrics,
                dynamic_keep_ratio=metrics.dynamic_keep_ratio,
            )
        if self.config.strategy == "robot_geo_hybrid_temporal_edge_reserve_v1" and "robot_metrics" in locals() and not metrics.fallback_used:
            self._update_temporal_history_after_selection(
                keep_indices_np=keep_indices_np,
                num_tokens=num_tokens,
                scores=scores,
                robot_metrics=robot_metrics,
                dynamic_keep_ratio=metrics.dynamic_keep_ratio,
            )

        internal_mode = self._internal_pruning_requested()
        if internal_mode:
            self._prepare_internal_pruning_plan(
                keep_indices_np=keep_indices_np,
                num_tokens=int(num_tokens),
                metrics=metrics,
                selection_meta=selection_meta,
                target_keep_ratio=float(selection_meta.get("acgtp_dynamic_keep_ratio") or selection_meta.get("requested_keep_ratio") or (len(keep_indices_np) / max(1, int(num_tokens)))),
            )
            metrics.timing.gather_ms = 0.0
            pruned = visual_tokens
            kept = int(np.asarray(keep_indices_np, dtype=np.int64).reshape(-1).size)
        else:
            gather_start = time.perf_counter()
            keep_indices = torch.as_tensor(keep_indices_np, dtype=torch.long, device=visual_tokens.device)
            pruned = visual_tokens.index_select(dim=1, index=keep_indices)
            metrics.timing.gather_ms = (time.perf_counter() - gather_start) * 1000.0
            kept = int(pruned.shape[1])
            metrics.compression_backend = "projector"
            metrics.projector_pruning_applied = bool(kept < int(num_tokens))
            metrics.internal_pruning_requested = False
            metrics.internal_pruning_plan_ready = False
            metrics.internal_pruning_applied = False

        metrics.num_visual_tokens_kept = kept
        metrics.num_visual_tokens_pruned = num_tokens - kept
        # P1: basic token count attribution fields
        metrics.selected_token_count = kept
        metrics.dropped_token_count = num_tokens - kept
        metrics.selected_token_ratio = kept / num_tokens if num_tokens else None
        metrics.dropped_token_ratio = (num_tokens - kept) / num_tokens if num_tokens else None
        metrics.retention_ratio = kept / num_tokens if num_tokens else None
        metrics.keep_ratio = kept / num_tokens if num_tokens else None
        metrics.actual_keep_ratio = metrics.keep_ratio
        metrics.keep_ratio_actual = metrics.keep_ratio
        metrics.retention_actual = metrics.keep_ratio
        metrics.effective_keep_count = kept
        metrics.original_token_count = num_tokens
        metrics.num_visual_tokens_dropped = num_tokens - kept
        metrics.num_visual_tokens_original_total = num_tokens
        metrics.num_visual_tokens_kept_total = kept
        if not internal_mode:
            self._prepare_position_preserve_info(
                keep_indices_np=keep_indices_np,
                num_tokens=int(num_tokens),
                metrics=metrics,
            )
        if metrics.requested_keep_ratio is None:
            metrics.requested_keep_ratio = float(self.config.keep_ratio)
        if metrics.keep_ratio_requested is None:
            metrics.keep_ratio_requested = metrics.requested_keep_ratio
        if metrics.keep_ratio_source is None:
            metrics.keep_ratio_source = self._default_keep_ratio_source()
        metrics.protected_token_ratio = metrics.keep_ratio
        metrics.pruned_token_ratio = 1.0 - metrics.keep_ratio if metrics.keep_ratio is not None else None
        metrics.selected_token_count_equals_kept = bool(metrics.selected_token_count == metrics.num_visual_tokens_kept)
        metrics.retention_ratio_valid = bool(metrics.retention_ratio == metrics.keep_ratio)
        metrics.pruning_result = {
            "num_tokens_before": num_tokens,
            "num_tokens_after": kept,
            "actual_keep_ratio": metrics.keep_ratio,
            "method": self.config.strategy,
            "effective_strategy": strategy,
            "selection_metadata": selection_meta,
            "compression_backend": metrics.compression_backend,
            "projector_pruning_applied": metrics.projector_pruning_applied,
            "internal_pruning_requested": metrics.internal_pruning_requested,
            "internal_pruning_plan_ready": metrics.internal_pruning_plan_ready,
            "fallback_used": metrics.fallback_used,
            "dynamic_phase": metrics.dynamic_phase,
            "dynamic_keep_k": metrics.dynamic_keep_k,
            "geo_risk_level": metrics.geo_risk_level,
            "geo_risk_score": metrics.geo_risk_score,
            "dynamic_keep_reason": metrics.dynamic_keep_reason,
            "num_high_contact_tokens": metrics.num_high_contact_tokens,
            "num_valid_3d_tokens": metrics.num_valid_3d_tokens,
            "interaction_lock": metrics.interaction_lock,
            "temporal_stability": metrics.temporal_stability,
            "history_length": metrics.history_length,
            "score_ema_enabled": metrics.score_ema_enabled,
            "score_ema_available": getattr(metrics, "score_ema_available", None),
            "ema_used_for_selection": getattr(metrics, "ema_used_for_selection", None),
            "lock_condition_failed_reason": getattr(metrics, "lock_condition_failed_reason", None),
            "keep_ratio_source": metrics.keep_ratio_source,
            "requested_keep_ratio": metrics.requested_keep_ratio,
            "effective_keep_count": metrics.effective_keep_count,
            "original_token_count": metrics.original_token_count,
            "adaptive_threshold_mean": getattr(metrics, "adaptive_threshold_mean", None),
            "adaptive_threshold_max": getattr(metrics, "adaptive_threshold_max", None),
            "interaction_lock_ratio": getattr(metrics, "interaction_lock_ratio", None),
            "interaction_lock_reason": getattr(metrics, "interaction_lock_reason", None),
            "selected_grid_coverage_ratio": getattr(metrics, "selected_grid_coverage_ratio", None),
            "grid_coverage_ratio": getattr(metrics, "grid_coverage_ratio", None),
            "final_hybrid_score_mean": getattr(metrics, "final_hybrid_score_mean", None),
            "final_hybrid_score_max": getattr(metrics, "final_hybrid_score_max", None),
        }
        if self.cfg.get("save_pruning_vis", False) or self.cfg.get("save_pruning_debug", False):
            vis_metrics = active_aux_metrics
            vis_path = self._maybe_save_visualization(
                keep_indices_np=keep_indices_np,
                scores=scores,
                aux_metrics=vis_metrics,
                selection_meta=selection_meta,
            )
            if vis_path is not None:
                metrics.pruning_result["visualization_dir"] = vis_path

        # P1: Token selection debug visualization (always for first 3 steps per method)
        _do_debug_vis = (
            self.cfg.get("save_token_selection_debug", False)
            and self._geo_debug_frames_saved < 3
            and self.config.strategy in ("robot_geo_hybrid_temporal_v1", "robot_geo_hybrid_temporal_edge_reserve_v1", "robot_geo_hybrid_v1", "depth_edge_fast")
        )
        if _do_debug_vis:
            debug_metrics = active_aux_metrics
            debug_path = self._save_token_selection_debug_visualization(
                keep_indices_np=keep_indices_np,
                scores=scores,
                aux_metrics=debug_metrics,
                selection_meta=selection_meta,
            )
            if debug_path is not None:
                self._geo_debug_frames_saved += 1
                metrics.pruning_result["token_selection_debug_dir"] = debug_path

        # P8: Enhanced dropped-token overlay (visualization-only, does not change selection)
        # Uses a SEPARATE counter so original + dropped overlays each get steps 0/1/2.
        _do_dropped_vis = (
            self.cfg.get("save_token_selection_debug", False)
            and self._dropped_overlay_frames_saved < 3
            and self.config.strategy in TOKEN_SELECTION_DEBUG_STRATEGIES
        )
        if _do_dropped_vis:
            debug_metrics2 = active_aux_metrics
            dropped_path = self._save_dropped_token_debug_visualization(
                keep_indices_np=keep_indices_np,
                aux_metrics=debug_metrics2,
                selection_meta=selection_meta,
                num_tokens=num_tokens,
            )
            if dropped_path is not None:
                self._dropped_overlay_frames_saved += 1
                metrics.pruning_result["token_selection_debug_dropped_dir"] = dropped_path

        if self.config.enable_geo_debug:
            debug_metrics = active_aux_metrics
            debug_path = self._maybe_save_geo_debug_visualization(
                keep_indices_np=keep_indices_np,
                scores=scores,
                aux_metrics=debug_metrics,
            )
            if debug_path is not None:
                metrics.pruning_result["geo_debug_dir"] = debug_path
        if metrics.d_min is not None and self.config.strategy in ("robot_geo_near", "robot_geo_corridor", "robot_geo_contact_budget", "robot_geo_rule_v0", "robot_geo_dynamic_v0", "robot_geo_temporal_v0", "robot_geo_dynamic"):
            distances = active_aux_metrics.get("distances")
            if distances is not None:
                d = np.asarray(distances, dtype=np.float32)
                valid_d = d[keep_indices_np[np.isfinite(d[keep_indices_np])]]
                metrics.d_mean_topk = float(np.mean(valid_d)) if valid_d.size else None
            corridor_distances = active_aux_metrics.get("corridor_distances")
            if corridor_distances is not None:
                cd = np.asarray(corridor_distances, dtype=np.float32)
                valid_cd = cd[keep_indices_np[np.isfinite(cd[keep_indices_np])]]
                metrics.d_corridor_mean_topk = float(np.mean(valid_cd)) if valid_cd.size else None
            corridor_scores = active_aux_metrics.get("corridor_scores")
            if corridor_scores is not None:
                cs = np.asarray(corridor_scores, dtype=np.float32)
                metrics.corridor_strength_topk_mean = float(np.mean(cs[keep_indices_np])) if keep_indices_np.size else None
            self._latest_stats = metrics.to_eval_stats()
        # --- P1-1 debug: print attribution_missing_reason if set ---
        if getattr(metrics, "attribution_missing_reason", None):
            print(f"[PRUNING] attribution_missing_reason: {metrics.attribution_missing_reason}")
        try:
            return pruned, metrics
        except Exception as exc:
            raise RuntimeError(
                f"_PruningHook.__call__ return failed: {type(exc).__name__}: {exc}"
            ) from exc

    def _mark_warmup_step(self, latest: Any) -> bool:
        episode_id = getattr(latest, "episode_id", None) if latest is not None else None
        if episode_id is not None and int(episode_id) != self._hook_episode_id:
            self._hook_episode_id = int(episode_id)
            self._hook_step_counter = 0
            self._temporal_history.reset()
            self._motion_buffer = None  # P15: reset ACGTP-v1 motion EMA on episode boundary
            self._acgtp_dynamic_state = {}  # Step 5: reset phase hysteresis per episode
            self._static_scene_cache.reset()
            self._reset_latency_plan_cache()
            self._acgtp_history.reset()  # Step 6: reset history stabilizer per episode
            self._acgtp_attention_history.clear()
        is_warmup = self._hook_step_counter == 0
        self._hook_step_counter += 1
        return bool(is_warmup)

    def _contact_budget_counts(self, keep_count: int) -> Tuple[int, int, int]:
        k_total = int(keep_count)
        edge_ratio = max(0.0, float(self.config.contact_budget_edge_ratio))
        geo_ratio = max(0.0, float(self.config.contact_budget_geo_ratio))
        diverse_ratio = max(0.0, float(self.config.contact_budget_diverse_ratio))
        ratio_sum = edge_ratio + geo_ratio + diverse_ratio
        if ratio_sum > 1e-8:
            edge_ratio /= ratio_sum
            geo_ratio /= ratio_sum
        k_edge = int(round(k_total * edge_ratio))
        k_geo = int(round(k_total * geo_ratio))
        k_edge = max(0, min(k_edge, k_total))
        k_geo = max(0, min(k_geo, k_total - k_edge))
        k_diverse = k_total - k_edge - k_geo
        return k_edge, k_geo, k_diverse

    def _update_temporal_history_after_selection(
        self,
        *,
        keep_indices_np: np.ndarray,
        num_tokens: int,
        scores: Optional[np.ndarray],
        robot_metrics: Dict[str, Any],
        dynamic_keep_ratio: Optional[float],
    ) -> None:
        keep_mask = np.zeros(int(num_tokens), dtype=np.bool_)
        idx = np.asarray(keep_indices_np, dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < int(num_tokens))]
        keep_mask[idx] = True
        latest = self.geometry_recorder.get_latest() if self.geometry_recorder is not None else None
        self._temporal_history.update(
            robot_state={"gripper_pos": robot_metrics.get("gripper_pos")},
            motion_direction=robot_metrics.get("motion_direction"),
            final_scores=scores,
            keep_mask=keep_mask,
            contact_risk_score=robot_metrics.get("rule_v0_contact_risk_scores"),
            valid_3d_ratio=robot_metrics.get("valid_token_ratio"),
            dynamic_keep_ratio=dynamic_keep_ratio,
            step_index=getattr(latest, "step_id", None) if latest is not None else None,
        )

    def _apply_selection_metrics(
        self,
        metrics: HookMetrics,
        selection_meta: Dict[str, Any],
        keep_indices_np: np.ndarray,
        aux_metrics: Dict[str, Any],
    ) -> None:
        # Preserve a fallback_used=True that an earlier missing-input branch
        # (e.g. missing_robot_state for legacy robot_geo_* strategies) already
        # set on metrics. The bulk copy loop below pulls fallback_used straight
        # from selection_meta, and the generic select_keep_indices() path returns
        # fallback_used=False, which would otherwise clobber the real signal.
        _prior_fallback_used = bool(getattr(metrics, "fallback_used", False))
        for key in (
            "K_total",
            "K_edge_target",
            "K_geo_target",
            "K_diverse_target",
            "K_edge_actual",
            "K_geo_actual",
            "K_diverse_actual",
            "selected_by_edge_count",
            "selected_by_geo_count",
            "selected_by_diverse_count",
            "overlap_edge_geo_before_dedup",
            # spatial diversity
            "selected_token_grid_entropy",
            # Hybrid v1 simple-name score stats (from select_hybrid_v1 metadata)
            "edge_score_mean", "edge_score_max", "edge_score_std",
            "near_score_mean", "near_score_max", "near_score_std",
            "contact_score_mean", "contact_score_max", "contact_score_std",
            "corridor_score_mean", "corridor_score_max", "corridor_score_std",
            "diversity_score_mean", "diversity_score_max", "diversity_score_std",
            "final_hybrid_score_mean", "final_hybrid_score_max", "final_hybrid_score_std",
            "w_edge", "w_near", "w_contact", "w_corr", "w_diverse",
            "selected_grid_coverage_ratio", "grid_coverage_ratio",
            # Hybrid dynamic (robot_geo_hybrid_dynamic_v0) selection counts
            "selected_by_depth_edge_count", "selected_by_contact_count",
            "selected_by_distance_contact_count", "selected_by_motion_count",
            "selected_by_uniform_count", "selected_by_fill_count",
            "motion_gate_effective", "overlap_depth_contact",
            # temporal / hybrid_temporal fields that come from selection_meta
            "ema_used_for_selection",
            "score_ema_available",
            "lock_condition_failed_reason",
            "topk_contact_lock",
            "elevated_current_lock",
            "gripper_lock",
            "region_lock",
            "requested_keep_ratio",
            "keep_ratio_source",
            "effective_keep_count",
            "original_token_count",
            "actual_keep_ratio",
            "attribution_missing_reason",
            "adaptive_threshold_mean",
            "adaptive_threshold_max",
            "fallback_used",
            # P1-1: token selection attribution
            "workspace_score_min",
            "workspace_score_unique_count",
            "workspace_all_one",
            "workspace_all_one_reason",
            "workspace_fallback_used",
            "workspace_bounds",
            "workspace_valid_token_ratio",
            "near_score_unique_count",
            "depth_edge_score_unique_count",
            "motion_cone_score_unique_count",
            "final_score_unique_count",
            "depth_edge_topk_count",
            "robot_geo_topk_count",
            "hybrid_final_score_topk_count",
            "depth_edge_topk_kept_in_final_count",
            "robot_geo_topk_kept_in_final_count",
            "hybrid_final_score_topk_kept_count",
            "depth_edge_topk_dropped_count",
            "robot_geo_topk_dropped_count",
            "hybrid_final_score_topk_dropped_count",
            "depth_edge_topk_dropped_ratio",
            "robot_geo_topk_dropped_ratio",
            "hybrid_final_score_topk_dropped_ratio",
            "overlap_depth_edge_robot_geo_count",
            "overlap_depth_edge_robot_geo_ratio",
            "selected_near_token_ratio",
            "selected_motion_token_ratio",
            "selected_workspace_token_ratio",
            "selected_robot_token_ratio",
            "selected_scene_token_ratio",
            "selected_background_token_ratio",
            "selected_high_depth_edge_but_low_robot_geo_count",
            "dropped_high_depth_edge_tokens_count",
            "dropped_high_robot_geo_tokens_count",
            "depth_edge_quota_count",
            "robot_geo_quota_count",
            "selected_overlap_robot_geo_depth_edge_ratio",
            "selected_overlap_robot_geo_motion_ratio",
            "selected_overlap_robot_geo_near_ratio",
            "depth_edge_topk_overlap_with_robot_geo_topk",
            "robot_geo_topk_overlap_with_depth_edge_topk",
            "selected_token_u_mean",
            "selected_token_u_std",
            "selected_token_v_mean",
            "selected_token_v_std",
            "selected_token_bbox_u_min",
            "selected_token_bbox_u_max",
            "selected_token_bbox_v_min",
            "selected_token_bbox_v_max",
            "selected_token_grid_quadrant_histogram",
            "selected_token_near_gripper_pixel_dist_mean",
            "selected_token_near_gripper_pixel_dist_median",
            # P5: edge_reserve ablation metrics
            "edge_reserve_enabled",
            "edge_reserve_ratio",
            "edge_reserved_target_count",
            "edge_reserved_actual_count",
            "edge_reserved_survival_ratio",
            "final_selected_count",
            "selected_by_edge_reserved_count",
            "selected_by_original_hybrid_count",
            "selected_by_fill_count",
            "duplicate_edge_hybrid_count",
            "K_edge_reserve_target",
            "K_edge_reserve_actual",
            # P5-fix: new renamed duplicate metrics
            "duplicate_after_exclusion_count",
            "duplicate_with_original_hybrid_count",
            # P5-fix: reserved / non-reserved split diagnostics
            "reserved_edge_topk_count",
            "reserved_edge_kept_count",
            "reserved_edge_dropped_count",
            "reserved_edge_topk_dropped_ratio",
            "non_reserved_edge_topk_count",
            "non_reserved_edge_kept_count",
            "non_reserved_edge_dropped_count",
            "non_reserved_edge_topk_dropped_ratio",
            "overall_depth_edge_topk_count",
            "overall_depth_edge_topk_kept_count",
            "overall_depth_edge_topk_dropped_count",
            "overall_depth_edge_topk_dropped_ratio",
            # P5-fix: new accounting fields
            "selected_by_phase1_hybrid_count",
            "selected_by_phase2_diversity_count",
            "selected_by_phase3_fallback_count",
            "selected_by_unattributed_count",
            "diagnostic_k_small",
            "diagnostic_k_large",
            # P5-fix: edge_reserve invalid flag and diagnostics
            "edge_reserve_invalid",
            "edge_reserve_invalid_reason",
            "edge_scores_available",
            "edge_scores_shape",
            "edge_scores_finite_ratio",
            # P6 normalized selection diagnostics
            "selector_name",
            "selector_function_name",
            "selection_strategy_name",
            "selection_stage_name",
            "keep_indices_unique",
            "keep_indices_out_of_bounds",
            "keep_ratio_requested",
            "keep_ratio_actual",
            "retention_actual",
            "selection_error",
            "selection_warning",
            "selected_by_phase1",
            "selected_by_phase2",
            "selected_by_phase3",
            "selected_by_fill",
            "selected_by_fallback",
            "selected_unattributed",
            "phase_accounting_sum",
            "phase_accounting_valid",
            "phase_accounting_error",
            "reserved_edge_dropped_ratio",
            "non_reserved_topk_count",
            "non_reserved_kept_count",
            "non_reserved_dropped_count",
            "non_reserved_dropped_ratio",
            "total_keep_budget",
            "depth_edge_budget",
            "robot_geo_budget",
            "fill_budget",
            "safety_budget",
            "score_min",
            "score_max",
            "score_mean",
            "score_std",
            "num_visual_tokens_original_total",
            "num_visual_tokens_kept_total",
            "num_visual_tokens_dropped",
            # P15: ACGTP-v1 metrics
            "acgtp_v1",
            "acgtp_w_scene_layout",
            "acgtp_w_depth_structure",
            "acgtp_w_contact_ring",
            "acgtp_w_motion_corridor",
            "acgtp_self_core_radius_px",
            "acgtp_contact_ring_inner_px",
            "acgtp_contact_ring_outer_px",
            "acgtp_self_core_token_count",
            "acgtp_self_core_token_ratio",
            "acgtp_contact_ring_token_count",
            "acgtp_contact_ring_token_ratio",
            "acgtp_contact_ring_gated_token_count",
            "acgtp_contact_ring_valid",
            "acgtp_scene_layout_score_mean",
            "acgtp_scene_layout_score_max",
            "acgtp_support_plane_token_count",
            "acgtp_object_component_token_count",
            "acgtp_boundary_token_count",
            "acgtp_scene_fill_candidate_count",
            "acgtp_scene_fill_candidate_ratio",
            # P6 new fields
            "acgtp_support_plane_candidate_count",
            "acgtp_scene_support_plane_cap_ratio",
            "acgtp_scene_support_plane_cap_used",
            "acgtp_scene_support_plane_fallback_used",
            "acgtp_scene_support_plane_fallback_reason",
            "acgtp_scene_object_component_fallback_used",
            "acgtp_scene_object_component_fallback_reason",
            "acgtp_scene_object_component_num_components",
            "acgtp_scene_boundary_fallback_used",
            "acgtp_scene_boundary_fallback_reason",
            "acgtp_scene_boundary_from_object_count",
            "acgtp_scene_boundary_from_depth_count",
            "acgtp_scene_layout_ms",
            "acgtp_contact_ring_ms",
            "acgtp_motion_corridor_ms",
            "acgtp_scene_selected_support_plane_count",
            "acgtp_scene_selected_object_component_count",
            "acgtp_scene_selected_boundary_count",
            "acgtp_scene_selected_relation_count",
            "acgtp_scene_support_plane_selected_ratio",
            "acgtp_motion_corridor_valid",
            "acgtp_motion_corridor_score_mean",
            "acgtp_motion_corridor_score_max",
            "acgtp_motion_corridor_length_m",
            "acgtp_motion_norm_m",
            "acgtp_motion_ema_alpha",
            "acgtp_depth_structure_score_mean",
            "acgtp_depth_structure_score_max",
            "acgtp_action_constraint_score_mean",
            "acgtp_action_constraint_score_max",
            "acgtp_action_constraint_source",
            "acgtp_future_action_constraint_enabled",
            "acgtp_future_action_constraint_valid",
            "acgtp_future_action_constraint_disabled_reason",
            "acgtp_future_action_constraint_score_mean",
            "acgtp_future_action_constraint_score_max",
            "acgtp_object_side_contact_score_mean",
            "acgtp_object_side_contact_score_max",
            "acgtp_swept_motion_risk_score_mean",
            "acgtp_swept_motion_risk_score_max",
            "acgtp_collision_contact_risk_score_mean",
            "acgtp_collision_contact_risk_score_max",
            "acgtp_contact_object_overlap_count",
            "acgtp_robot_self_penalty_count",
            "acgtp_action_constraint_ms",
            "acgtp_dynamic_enabled",
            "acgtp_dynamic_phase",
            "acgtp_dynamic_candidate_phase",
            "acgtp_dynamic_previous_phase",
            "acgtp_dynamic_hysteresis_state",
            "acgtp_dynamic_risk",
            "acgtp_dynamic_confidence",
            "acgtp_dynamic_keep_ratio",
            "acgtp_dynamic_keep_k",
            "acgtp_dynamic_base_keep_ratio",
            "acgtp_dynamic_raw_keep_ratio",
            "acgtp_dynamic_phase_min_keep_ratio",
            "acgtp_dynamic_phase_max_keep_ratio",
            "acgtp_dynamic_lock_strength",
            "acgtp_dynamic_uncertainty_boost",
            "acgtp_dynamic_risk_boost",
            "acgtp_dynamic_prune_gain",
            "acgtp_dynamic_keep_reason",
            "acgtp_dynamic_layout_motion_alignment",
            "acgtp_dynamic_binary_alignment",
            "acgtp_dynamic_contact_phase_gate",
            "acgtp_dynamic_contact_peak",
            "acgtp_dynamic_contact_mean",
            "acgtp_dynamic_contact_ratio",
            "acgtp_dynamic_motion_peak",
            "acgtp_dynamic_motion_mean",
            "acgtp_dynamic_motion_ratio",
            "acgtp_dynamic_physical_ratio",
            "acgtp_dynamic_high_contact",
            "acgtp_dynamic_high_contact_coverage",
            "acgtp_dynamic_high_contact_legacy",
            "acgtp_dynamic_shadow_contact_guard",
            "acgtp_dynamic_high_motion",
            "acgtp_dynamic_strong_layout",
            "acgtp_dynamic_action_peak",
            "acgtp_dynamic_action_mean",
            "acgtp_dynamic_depth_valid_ratio",
            "acgtp_dynamic_fill_candidate_count",
            "acgtp_dynamic_fill_candidate_ratio",
            "acgtp_dynamic_candidate_gap_count",
            "acgtp_dynamic_candidate_gap_ratio",
            "acgtp_dynamic_candidate_clamped",
            "acgtp_dynamic_scene_weight",
            "acgtp_dynamic_depth_weight",
            "acgtp_dynamic_contact_weight",
            "acgtp_dynamic_motion_weight",
            "acgtp_dynamic_hard_protect_ratio",
            "acgtp_dynamic_budget_vector",
            "acgtp_dynamic_disabled_reason",
            "acgtp_selector_version",
            "acgtp_quota_policy",
            "acgtp_fill_policy",
            "acgtp_hard_protect_count",
            "acgtp_hard_protect_ratio",
            "acgtp_hard_protect_valid",
            "acgtp_scene_quota",
            "acgtp_depth_quota",
            "acgtp_contact_quota",
            "acgtp_motion_quota",
            "acgtp_scene_quota_weight",
            "acgtp_depth_quota_weight",
            "acgtp_contact_quota_weight",
            "acgtp_motion_quota_weight",
            "acgtp_scene_allocated",
            "acgtp_depth_allocated",
            "acgtp_contact_allocated",
            "acgtp_motion_allocated",
            "acgtp_coverage_fill_candidate_count",
            "acgtp_coverage_fill_candidate_ratio",
            "selected_by_scene_layout_count",
            "selected_by_depth_structure_count",
            "selected_by_contact_ring_count",
            "selected_by_motion_corridor_count",
            "selected_by_constrained_fill_count",
            "selected_by_acgtp_fallback_count",
            "overlap_scene_depth_count",
            "overlap_scene_contact_count",
            "overlap_contact_motion_count",
            "overlap_scene_motion_count",
            "overlap_depth_contact_count",
            "overlap_depth_motion_count",
            "acgtp_branch_accounting_valid",
            "acgtp_branch_sum",
            "acgtp_branch_sum_error",
            "branch_accounting_valid",
            "branch_sum_equals_kept",
            "acgtp_hard_protect_ratio_config",
            "acgtp_fallback_used",
            "acgtp_fallback_reason",
            "acgtp_motion_disabled_reason",
            "acgtp_scene_layout_scores",
            "acgtp_contact_ring_scores",
            "acgtp_motion_corridor_scores",
            "acgtp_action_constraint_scores",
            "acgtp_constrained_fill_mask",
            "acgtp_robot_self_core_mask",
            "acgtp_final_kept",
            "acgtp_expected_kept",
            "acgtp_actual_keep_ratio",
            # P16: ACGTP-v2 semantic/attention/fallback fields. These are
            # copied here as well as in the native v2 branch so input-fallback
            # paths still produce a complete CSV schema.
            "acgtp_v2",
            "acgtp_v2_semantic_enabled",
            "acgtp_v2_semantic_backend",
            "acgtp_v2_semantic_confidence",
            "acgtp_v2_semantic_unavailable",
            "acgtp_v2_semantic_fallback_reason",
            "acgtp_v2_release_quota",
            "strict_fallback_dispatch_used",
            "delegated_selector_name",
            "fallback_dispatch_to_v1",
            "semantic_backend",
            "semantic_unavailable",
            "semantic_confidence",
            "semantic_available",
            "semantic_quota_released",
            "selected_by_semantic_count",
            "attention_backend",
            "attention_source",
            "attention_available",
            "attention_confidence",
            "attention_quota_released",
            "selected_by_attention_count",
            "attention_only_token_count",
            "attention_selected_by_final_count",
            "attention_top_count",
            "safe_drop_candidate_count",
            "high_attention_low_geometry_count",
            "high_geometry_low_attention_count",
        ):
            if key in selection_meta:
                setattr(metrics, key, selection_meta.get(key))
        # OR back the pre-existing fallback signal so the bulk copy never
        # downgrades a real missing-input fallback to False.
        if _prior_fallback_used:
            metrics.fallback_used = True
        if selection_meta.get("fallback_reason") and metrics.fallback_reason is None:
            metrics.fallback_reason = selection_meta.get("fallback_reason")
            metrics.fallback_used = True
            metrics.keep_ratio_source = "fallback"

        # P5-fix: edge_reserve accounting invariant checks
        _selected_by_edge_reserved = int(selection_meta.get("selected_by_edge_reserved_count") or 0)
        _selected_by_phase1 = int(selection_meta.get("selected_by_phase1_hybrid_count") or 0)
        _selected_by_phase2 = int(selection_meta.get("selected_by_phase2_diversity_count") or 0)
        _selected_by_phase3 = int(selection_meta.get("selected_by_phase3_fallback_count") or 0)
        _selected_by_fill = int(selection_meta.get("selected_by_fill_count") or 0)
        _selected_by_unattributed = int(selection_meta.get("selected_by_unattributed_count") or 0)
        _final_selected = int(selection_meta.get("final_selected_count") or 0)
        _k_diag_small = int(selection_meta.get("diagnostic_k_small") or 0)
        _k_diag_large = int(selection_meta.get("diagnostic_k_large") or 0)
        _reserved_drop_ratio = selection_meta.get("reserved_edge_topk_dropped_ratio")
        _edge_reserve_enabled = selection_meta.get("edge_reserve_enabled") is True
        _edge_reserve_seen = any(
            key in selection_meta
            for key in (
                "edge_reserve_enabled",
                "selected_by_edge_reserved_count",
                "reserved_edge_topk_count",
                "non_reserved_edge_topk_count",
                "overall_depth_edge_topk_count",
            )
        )

        # Invariant 1: all selected_by_* sum to final_selected_count
        _attributed_sum = _selected_by_edge_reserved + _selected_by_phase1 + _selected_by_phase2 + _selected_by_phase3
        _accounting_valid = bool(_attributed_sum == _final_selected and _selected_by_unattributed == 0)
        _accounting_error = None
        if _attributed_sum != _final_selected:
            _accounting_error = f"attributed_sum={_attributed_sum} != final_selected={_final_selected}"
        elif _selected_by_unattributed > 0:
            _accounting_error = f"unattributed={_selected_by_unattributed} > 0"

        # Invariant 2: if edge_reserve_enabled and survival=1.0 then reserved_drop_ratio=0
        _split_valid = True
        if _edge_reserve_enabled and _reserved_drop_ratio is not None and _reserved_drop_ratio > 0:
            _split_valid = False

        # Invariant 3: selected_token_count equals num_visual_tokens_kept
        _retention_valid = True
        if metrics.selected_token_count is not None and metrics.num_visual_tokens_kept is not None:
            _retention_valid = bool(metrics.selected_token_count == metrics.num_visual_tokens_kept)

        # Invariant 4: keep_indices have no duplicates and are sorted
        idx_raw = np.asarray(keep_indices_np, dtype=np.int64) if keep_indices_np is not None else np.array([], dtype=np.int64)
        _no_dup = bool(len(idx_raw) == len(set(idx_raw))) if idx_raw.size > 0 else True
        _sorted = bool(np.all(idx_raw[:-1] <= idx_raw[1:])) if idx_raw.size > 1 else True

        # P5-fix: record invariant check results
        metrics.edge_reserve_accounting_valid = bool(_accounting_valid) if _edge_reserve_seen else None
        metrics.edge_reserve_accounting_error = str(_accounting_error) if (_edge_reserve_seen and _accounting_error) else None
        metrics.edge_reserve_split_metrics_valid = bool(_split_valid) if _edge_reserve_seen else None
        metrics.diagnostic_k_small = _k_diag_small if _k_diag_small > 0 else None
        metrics.diagnostic_k_large = _k_diag_large if _k_diag_large > 0 else None
        metrics.no_duplicate_final_indices = _no_dup
        metrics.final_indices_sorted = _sorted
        metrics.selected_token_count_equals_kept = _retention_valid
        metrics.retention_ratio_valid = _retention_valid

        idx = np.asarray(keep_indices_np, dtype=np.int64)
        _n_from_aux = aux_metrics.get("num_tokens")
        if _n_from_aux is not None and int(_n_from_aux) > 0:
            num_tokens = int(_n_from_aux)
        else:
            num_tokens = int(idx.size) if idx.size else 256
        if idx.size == 0:
            return

        # Selected token statistics for each score component
        selected_stats = _selected_score_stats(aux_metrics, idx, num_tokens)
        for key, value in selected_stats.items():
            if value is not None and hasattr(metrics, key):
                setattr(metrics, key, value)

        # P1-1: Token selection attribution and top-k competition diagnostics
        token_grid_shape = aux_metrics.get("token_grid_shape", (16, 16))
        try:
            attribution = _compute_token_selection_attribution(
                aux_metrics, idx, num_tokens, token_grid_shape
            )
        except Exception as exc:
            attribution = {}
            metrics.attribution_missing_reason = (
                f"_compute_token_selection_attribution_error:{type(exc).__name__}:{exc}"
            )
        for key, value in attribution.items():
            if value is not None and hasattr(metrics, key):
                setattr(metrics, key, value)

    def _record_selection_path_diagnostics(
        self,
        metrics: "HookMetrics",
        selector_success: bool,
        exc: Optional[Exception],
        requested_strategy: str,
        effective_strategy: str,
        selector_name: str,
        fallback_selector: Optional[str],
        keep_indices_np: Optional[np.ndarray],
        num_tokens: int,
        selection_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record Stage X selection-path integrity diagnostics on every pruning step."""
        meta = selection_meta or {}
        metrics.requested_pruning_strategy = requested_strategy
        metrics.effective_pruning_strategy = effective_strategy
        metrics.selector_name = meta.get("selector_name", metrics.selector_name)
        metrics.selector_function_name = meta.get("selector_function_name", selector_name)
        metrics.selection_strategy_name = meta.get("selection_strategy_name", effective_strategy)
        metrics.selection_stage_name = meta.get("selection_stage_name")
        metrics.selector_success = selector_success
        # Do NOT override metrics.fallback_used here.
        # fallback_used is already used in this codebase for "fallback due to missing inputs"
        # (e.g., geometry/robot metrics unavailable). Stage X only annotates selector-* fallback.
        if exc is not None:
            metrics.selector_exception_type = type(exc).__name__
            msg = str(exc)
            metrics.selector_exception_msg = msg[:200] if msg else None
            if metrics.selection_error is None:
                metrics.selection_error = f"{type(exc).__name__}:{metrics.selector_exception_msg}"
        else:
            metrics.selector_exception_type = None
            metrics.selector_exception_msg = None
        metrics.fallback_selector_name = meta.get("fallback_selector_name", fallback_selector)
        if keep_indices_np is not None:
            metrics.keep_indices_count = int(len(keep_indices_np))
            idx = np.asarray(keep_indices_np, dtype=np.int64)
            metrics.keep_indices_sorted = bool(
                np.all(idx[:-1] <= idx[1:]) if idx.shape[0] > 1 else True
            )
            metrics.keep_indices_unique = bool(np.unique(idx).size == idx.size)
            metrics.keep_indices_out_of_bounds = bool(np.any((idx < 0) | (idx >= int(num_tokens))))
            metrics.actual_retention_ratio = float(len(keep_indices_np)) / float(num_tokens) if num_tokens else None
            metrics.keep_ratio_actual = metrics.actual_retention_ratio
            metrics.retention_actual = metrics.actual_retention_ratio
            # Infer keep_indices_source
            if selector_success:
                src = metrics.selector_function_name
            elif metrics.fallback_selector_name is not None:
                src = f"fallback_{metrics.fallback_selector_name}"
            else:
                src = None
            metrics.keep_indices_source = src
        else:
            metrics.keep_indices_count = None
            metrics.keep_indices_sorted = None
            metrics.keep_indices_unique = None
            metrics.keep_indices_out_of_bounds = None
            metrics.actual_retention_ratio = None
            metrics.keep_ratio_actual = None
            metrics.retention_actual = None
            metrics.keep_indices_source = None

        for key in (
            "keep_ratio_requested",
            "selection_error",
            "selection_warning",
            "selected_by_phase1",
            "selected_by_phase2",
            "selected_by_phase3",
            "selected_by_fill",
            "selected_by_fallback",
            "selected_unattributed",
            "phase_accounting_sum",
            "phase_accounting_valid",
            "phase_accounting_error",
            "reserved_edge_dropped_ratio",
            "non_reserved_topk_count",
            "non_reserved_kept_count",
            "non_reserved_dropped_count",
            "non_reserved_dropped_ratio",
            "total_keep_budget",
            "depth_edge_budget",
            "robot_geo_budget",
            "fill_budget",
            "safety_budget",
            "score_min",
            "score_max",
            "score_mean",
            "score_std",
            # P7: hybrid_budget_v2 fields (copied from selection_meta)
            "hybrid_budget_v2",
            "depth_edge_budget_ratio",
            "robot_contact_budget_ratio",
            "safety_budget_ratio",
            "K_depth_actual",
            "K_robot_actual",
            "K_fill_actual",
            "overlap_depth_robot_count",
            "overlap_depth_robot_diagnostic",
            "depth_edge_candidates_count",
            "robot_geo_candidates_count",
            "depth_edge_reserved_kept_count",
            "robot_geo_reserved_kept_count",
            "fill_from_depth_count",
            "fill_from_robot_count",
            "fill_from_other_count",
            "num_visual_tokens_original_total",
            "num_visual_tokens_kept_total",
            "num_visual_tokens_dropped",
            # P11.3: DE top-k attribution by branch (selector.py populates these in metadata)
            "depth_edge_topk_kept_by_depth_branch_count",
            "depth_edge_topk_kept_by_hybrid_branch_count",
            "depth_edge_topk_kept_by_fill_branch_count",
            "depth_edge_topk_kept_by_fallback_count",
            "depth_edge_topk_survival_ratio",
            # P11.3: Hybrid/final-score top-k attribution by branch
            "hybrid_final_score_topk_kept_by_depth_branch_count",
            "hybrid_final_score_topk_kept_by_hybrid_branch_count",
            "hybrid_final_score_topk_kept_by_fill_branch_count",
            "hybrid_final_score_topk_kept_by_fallback_count",
            "hybrid_final_score_topk_survival_ratio",
            # P11.3: Legacy hybrid top-k aliases
            "hybrid_topk_kept_by_depth_branch_count",
            "hybrid_topk_kept_by_hybrid_branch_count",
            "hybrid_topk_kept_by_fill_branch_count",
            "hybrid_topk_kept_by_fallback_count",
            "hybrid_topk_survival_ratio",
            # P11: branch_budget_v0 fields
            "branch_budget_v0",
            "hybrid_action_budget",
            "diversity_fill_budget",
            "temporal_budget",
            "branch_accounting_valid",
            "branch_sum_equals_kept",
            "selected_by_depth_branch",
            "selected_by_hybrid_branch",
            "selected_by_fill",
            "selected_by_fallback",
            "overlap_depth_edge_hybrid_count",
            "overlap_depth_edge_hybrid_ratio",
            "non_reserved_depth_edge_dropped_ratio",
            "non_reserved_depth_edge_count",
            "non_reserved_depth_edge_kept",
            "non_reserved_depth_edge_dropped",
            "branch_budget_depth_ratio_override",
            "branch_budget_hybrid_ratio_override",
            "depth_edge_budget_actual",
            "hybrid_action_budget_actual",
            "diversity_fill_budget_actual",
            "depth_branch_indices",
            "hybrid_branch_indices",
            "fill_branch_indices",
        ):
            if key in meta:
                setattr(metrics, key, meta.get(key))
        if metrics.keep_ratio_requested is None:
            metrics.keep_ratio_requested = metrics.requested_keep_ratio

        # If pruning is enabled, always record the requested/effective strategy.
        # For strategies that don't use selector fallback (normal case), also fill success=True.
        if metrics.requested_pruning_strategy is None:
            metrics.requested_pruning_strategy = requested_strategy
        if metrics.effective_pruning_strategy is None:
            metrics.effective_pruning_strategy = effective_strategy
        if metrics.selector_function_name is None:
            metrics.selector_function_name = selector_name
        if metrics.selector_success is None:
            metrics.selector_success = bool(selector_success)

    def _maybe_save_visualization(
        self,
        *,
        keep_indices_np: np.ndarray,
        scores: Optional[np.ndarray],
        aux_metrics: Dict[str, Any],
        selection_meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        latest = self.geometry_recorder.get_latest() if self.geometry_recorder is not None else None
        if latest is None:
            return None
        episode_id = int(getattr(latest, "episode_id", 0))
        step_id = int(getattr(latest, "step_id", 0))
        if self.cfg.get("save_pruning_debug", False):
            debug_tasks = self.cfg.get("debug_tasks")
            if debug_tasks:
                task_name = str(getattr(latest, "task_name", ""))
                task_idx = task_name.split("_")[-1] if task_name.startswith("task_") else None
                allowed = {str(x).strip() for x in str(debug_tasks).split(",") if str(x).strip()}
                if task_name not in allowed and task_idx not in allowed:
                    return None
        target_episode = self.cfg.get("pruning_vis_episode")
        target_step = self.cfg.get("pruning_vis_step")
        if target_episode is not None and int(target_episode) != episode_id:
            return None
        if target_step is not None and int(target_step) != step_id:
            return None
        try:
            base_output = str(self.cfg.get("save_dir") or "outputs")
            cache = aux_metrics.get("cache", {})
            score_maps = {
                "depth_edge_score": aux_metrics.get("edge_scores"),
                "gripper_distance_score": aux_metrics.get("near_scores"),
                "near_score": aux_metrics.get("near_scores"),
                "motion_corridor_score": aux_metrics.get("corridor_scores"),
                "corridor_score": aux_metrics.get("corridor_scores"),
                "near_contact_score": aux_metrics.get("near_contact_scores"),
                "corridor_contact_score": aux_metrics.get("corridor_contact_scores"),
                "geo_contact_score": aux_metrics.get("geo_contact_scores"),
                "final_geometry_score": scores,
            }
            selection_masks = None
            if selection_meta:
                selection_masks = {
                    "edge": selection_meta.get("selected_edge_indices", []),
                    "geo": selection_meta.get("selected_geo_indices", []),
                    "diverse": selection_meta.get("selected_diverse_indices", []),
                }
            return save_pruning_visualization(
                output_dir=base_output,
                method=self.config.strategy,
                episode_id=episode_id,
                step_id=step_id,
                rgb=getattr(latest, "rgb", None),
                depth=getattr(latest, "depth", None),
                token_u=cache.get("u") if isinstance(cache, dict) else None,
                token_v=cache.get("v") if isinstance(cache, dict) else None,
                keep_indices=keep_indices_np,
                score_maps=score_maps,
                selection_masks=selection_masks,
                token_grid_shape=self._latest_token_grid_shape or self.config.token_grid_shape,
            )
        except Exception as exc:
            if self.config.debug:
                print(f"[PRUNING VIS] warning: failed to save pruning visualization: {exc}")
            return None

    def _maybe_save_geo_debug_visualization(
        self,
        *,
        keep_indices_np: np.ndarray,
        scores: Optional[np.ndarray],
        aux_metrics: Dict[str, Any],
    ) -> Optional[str]:
        if not self.config.enable_geo_debug:
            return None
        if self.config.max_debug_frames <= 0:
            return None
        if self._geo_debug_frames_saved >= self.config.max_debug_frames:
            return None
        latest = self.geometry_recorder.get_latest() if self.geometry_recorder is not None else None
        if latest is None:
            return None
        step_id = int(getattr(latest, "step_id", 0))
        if step_id % self.config.geo_debug_interval != 0:
            return None
        try:
            cache = aux_metrics.get("cache", {})
            dynamic_decision = aux_metrics.get("dynamic_decision") or {}
            component_summary = dynamic_decision.get("component_summary", {}) if isinstance(dynamic_decision, dict) else {}
            dynamic_info = {
                "dynamic_keep_ratio": dynamic_decision.get("keep_ratio") if isinstance(dynamic_decision, dict) else None,
                "risk_level": dynamic_decision.get("risk_level") if isinstance(dynamic_decision, dict) else None,
                "risk_score": dynamic_decision.get("risk_score") if isinstance(dynamic_decision, dict) else None,
                "reason": dynamic_decision.get("reason") if isinstance(dynamic_decision, dict) else None,
                "num_high_contact_tokens": component_summary.get("num_high_contact_tokens") if isinstance(component_summary, dict) else None,
                "num_valid_3d_tokens": component_summary.get("num_valid_3d_tokens") if isinstance(component_summary, dict) else None,
            }
            score_maps = {
                "distance_to_gripper_score": _first_present(aux_metrics, "rule_v0_distance_scores", "near_scores"),
                "motion_cone_score": _first_present(aux_metrics, "rule_v0_motion_cone_scores", "corridor_scores"),
                "contact_risk_score": _first_present(aux_metrics, "rule_v0_contact_risk_scores", "geo_contact_scores"),
                "depth_edge_score": aux_metrics.get("edge_scores"),
                "final_geometry_score": scores,
            }
            path = save_geo_debug_visualization(
                enabled=True,
                output_dir=str(self.cfg.get("save_dir") or "outputs"),
                method=self.config.strategy,
                episode_id=int(getattr(latest, "episode_id", 0)),
                step_id=step_id,
                keep_indices=keep_indices_np,
                score_maps=score_maps,
                dynamic_info=dynamic_info,
                rgb=getattr(latest, "rgb", None),
                token_u=cache.get("u") if isinstance(cache, dict) else None,
                token_v=cache.get("v") if isinstance(cache, dict) else None,
                token_grid_shape=self._latest_token_grid_shape or self.config.token_grid_shape,
            )
            if path is not None:
                self._geo_debug_frames_saved += 1
            return path
        except Exception as exc:
            if self.config.debug:
                print(f"[GEO DEBUG VIS] warning: failed to save geometry debug visualization: {exc}")
            return None

    def _save_token_selection_debug_visualization(
        self,
        *,
        keep_indices_np: np.ndarray,
        scores: Optional[np.ndarray],
        aux_metrics: Dict[str, Any],
        selection_meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """P1: Save token selection debug overlay (first 3 steps of hybrid_temporal_v1 runs)."""
        try:
            from .visualization import save_token_selection_debug_visualization
        except ImportError:
            return None
        latest = self.geometry_recorder.get_latest() if self.geometry_recorder is not None else None
        if latest is None:
            return None
        episode_id = int(getattr(latest, "episode_id", 0))
        step_id = int(getattr(latest, "step_id", 0))
        grid_shape = aux_metrics.get("token_grid_shape", (16, 16))
        return save_token_selection_debug_visualization(
            output_dir=str(self.cfg.get("save_dir") or "outputs"),
            method=self.config.strategy,
            episode_id=episode_id,
            step_id=step_id,
            rgb=getattr(latest, "rgb", None),
            token_u=aux_metrics.get("token_u"),
            token_v=aux_metrics.get("token_v"),
            keep_indices=keep_indices_np,
            depth_edge_scores=aux_metrics.get("edge_scores"),
            robot_geo_scores=(
                aux_metrics.get("hybrid_final_scores")
                if aux_metrics.get("hybrid_final_scores") is not None
                else aux_metrics.get("final_scores")
            ),
            gripper_pixel=aux_metrics.get("gripper_pixel"),
            token_grid_shape=grid_shape,
        )

    def _reconstruct_dropped_token_sets(
        self,
        *,
        keep_indices_np: np.ndarray,
        aux_metrics: Dict[str, Any],
        selection_meta: Optional[Dict[str, Any]],
        num_tokens: int,
        grid_shape: Tuple[int, int],
    ) -> Dict[str, set]:
        """P8.3: Reconstruct token sets for dropped-token visualization.

        This reconstructs the sets from scores already stored in aux_metrics and
        counts stored in selection_meta. It does NOT change selection logic or scores.

        Priority order for sourcing counts (highest to lowest):
          1. selection_meta (authoritative — from the real selector accounting)
          2. aux_metrics edge_scores (for reconstruction fallback)
          3. int(round(k_final * ratio)) (last resort, with reconstruction_exact=false)

        Returns a dict with:
          - depth_edge_topk_indices
          - robot_geo_topk_indices
          - depth_edge_topk_dropped_indices
          - robot_geo_topk_dropped_indices
          - reserved_edge_indices
          - non_reserved_edge_dropped_indices
          - reconstruction_metadata: dict with count sources and exactness flags
        """
        import numpy as np

        result: Dict[str, set] = {
            "depth_edge_topk_indices": set(),
            "robot_geo_topk_indices": set(),
            "depth_edge_topk_dropped_indices": set(),
            "robot_geo_topk_dropped_indices": set(),
            "reserved_edge_indices": set(),
            "non_reserved_edge_dropped_indices": set(),
            # P11.3: Branch attribution sets
            "depth_branch_indices": set(),
            "hybrid_branch_indices": set(),
            "fill_branch_indices": set(),
            "fallback_branch_indices": set(),
        }
        reconstruction_meta: Dict[str, Any] = {
            "depth_edge_topk_count_source": None,
            "depth_edge_topk_count_exact": False,
            "reserved_edge_count_source": None,
            "reserved_edge_count_exact": False,
            "non_reserved_edge_dropped_count_source": None,
            "non_reserved_edge_dropped_count_exact": False,
        }

        n = int(num_tokens)
        if n <= 0:
            result["reconstruction_metadata"] = reconstruction_meta
            return result

        idx = np.asarray(list(keep_indices_np), dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < n)]
        keep_set = set(int(i) for i in idx.tolist())
        all_set = set(range(n))
        dropped_set = all_set - keep_set

        k_final = len(keep_set)

        # ── Depth-edge top-k ────────────────────────────────────────────────
        # Priority: 1) selection_meta (authoritative), 2) aux_metrics edge_scores
        # (score-based reconstruction is exact because scores are deterministic).
        de_topk: set = set()
        edge_arr = aux_metrics.get("edge_scores")
        if edge_arr is not None:
            edge_flat = np.asarray(edge_arr, dtype=np.float32).reshape(-1)
            if edge_flat.size >= n:
                # Try authoritative count from selection_meta first
                _sm_de_count = selection_meta.get("depth_edge_topk_count") if selection_meta else None
                if _sm_de_count is not None:
                    _de_k = int(_sm_de_count)
                    reconstruction_meta["depth_edge_topk_count_source"] = "selection_meta"
                    reconstruction_meta["depth_edge_topk_count_exact"] = True
                else:
                    # Fall back to score-based: use the same ratio the selector used
                    _de_k = max(1, int(round(k_final * 0.80)))
                    reconstruction_meta["depth_edge_topk_count_source"] = "reconstructed_from_edge_scores_and_ratio"
                    reconstruction_meta["depth_edge_topk_count_exact"] = False
                    reconstruction_meta["depth_edge_topk_ratio"] = 0.80
                edge_adj = np.where(np.isfinite(edge_flat), edge_flat, -np.inf)
                edge_order = np.argsort(-edge_adj)
                for i in range(min(_de_k, n)):
                    de_topk.add(int(edge_order[i]))
                result["depth_edge_topk_indices"] = de_topk
                result["depth_edge_topk_dropped_indices"] = de_topk - keep_set

        # ── Hybrid/final-score top-k ─────────────────────────────────────────
        # For depth_edge_fast: hybrid_final_scores == edge_scores, so this would
        # duplicate de_topk. We mark it null for depth_edge_fast in metadata later.
        # For hybrid/edge_reserve methods: this is a distinct signal (weighted combo).
        hybrid_arr = aux_metrics.get("hybrid_final_scores")
        final_arr = hybrid_arr if hybrid_arr is not None else aux_metrics.get("final_scores")
        if final_arr is not None:
            final_flat = np.asarray(final_arr, dtype=np.float32).reshape(-1)
            if final_flat.size >= n:
                # Use the same k as depth_edge top-k for consistency in cross-method comparison
                _rg_k = len(de_topk) if de_topk else max(1, int(round(k_final * 0.80)))
                final_adj = np.where(np.isfinite(final_flat), final_flat, -np.inf)
                final_order = np.argsort(-final_adj)
                rg_topk: set = set()
                for i in range(min(_rg_k, n)):
                    rg_topk.add(int(final_order[i]))
                result["robot_geo_topk_indices"] = rg_topk
                result["robot_geo_topk_dropped_indices"] = rg_topk - keep_set

        # ── Reserved edge tokens (edge_reserve methods only) ─────────────────
        # Priority: 1) selection_meta reserved_edge_topk_count (authoritative),
        # 2) aux_metrics edge_scores + k from selection_meta, 3) score fallback.
        if edge_arr is not None and "edge_reserve" in self.config.strategy:
            edge_flat2 = np.asarray(edge_arr, dtype=np.float32).reshape(-1)
            if edge_flat2.size >= n:
                edge_adj2 = np.where(np.isfinite(edge_flat2), edge_flat2, -np.inf)
                edge_order2 = np.argsort(-edge_adj2)

                # Try authoritative count from selection_meta
                _sm_res_count = selection_meta.get("reserved_edge_topk_count") if selection_meta else None
                if _sm_res_count is not None:
                    _res_k = int(_sm_res_count)
                    reconstruction_meta["reserved_edge_count_source"] = "selection_meta"
                    reconstruction_meta["reserved_edge_count_exact"] = True
                else:
                    # Try to reconstruct from edge_scores using edge_reserve_k from aux_metrics
                    _erk = aux_metrics.get("edge_reserve_k")
                    if _erk is not None:
                        _res_k = int(_erk)
                        reconstruction_meta["reserved_edge_count_source"] = "reconstructed_from_edge_scores_and_aux_metrics_edge_reserve_k"
                        reconstruction_meta["reserved_edge_count_exact"] = True
                    else:
                        # Last resort: approximate (not exact)
                        _res_k = max(0, int(round(0.40 * k_final)))
                        reconstruction_meta["reserved_edge_count_source"] = "fallback_int_0_40_k_final"
                        reconstruction_meta["reserved_edge_count_exact"] = False
                        reconstruction_meta["reserved_edge_fallback_ratio"] = 0.40

                res_set: set = set()
                for i in range(min(_res_k, n)):
                    res_set.add(int(edge_order2[i]))
                result["reserved_edge_indices"] = res_set

                # Non-reserved edge dropped = dropped tokens that were in DE top-k but NOT reserved
                non_res_in_de_topk = de_topk - res_set if de_topk else set()
                result["non_reserved_edge_dropped_indices"] = non_res_in_de_topk & dropped_set

                # Track non_reserved source
                _sm_nres_drop = selection_meta.get("non_reserved_edge_dropped_count") if selection_meta else None
                if _sm_nres_drop is not None:
                    reconstruction_meta["non_reserved_edge_dropped_count_source"] = "selection_meta"
                    reconstruction_meta["non_reserved_edge_dropped_count_exact"] = True
                else:
                    reconstruction_meta["non_reserved_edge_dropped_count_source"] = "reconstructed"
                    reconstruction_meta["non_reserved_edge_dropped_count_exact"] = True  # exact from scores

        # ── P11.3: Branch attribution sets from selection_meta ─────────────────
        if selection_meta is not None:
            _depth_branch = selection_meta.get("depth_branch_indices")
            _hybrid_branch = selection_meta.get("hybrid_branch_indices")
            _fill_branch = selection_meta.get("fill_branch_indices")
            if _depth_branch is not None and isinstance(_depth_branch, (list, set, tuple)):
                result["depth_branch_indices"] = set(int(i) for i in _depth_branch if 0 <= int(i) < n)
            if _hybrid_branch is not None and isinstance(_hybrid_branch, (list, set, tuple)):
                result["hybrid_branch_indices"] = set(int(i) for i in _hybrid_branch if 0 <= int(i) < n)
            if _fill_branch is not None and isinstance(_fill_branch, (list, set, tuple)):
                result["fill_branch_indices"] = set(int(i) for i in _fill_branch if 0 <= int(i) < n)

        result["reconstruction_metadata"] = reconstruction_meta
        return result

    def _save_dropped_token_debug_visualization(
        self,
        *,
        keep_indices_np: np.ndarray,
        aux_metrics: Dict[str, Any],
        selection_meta: Optional[Dict[str, Any]],
        num_tokens: int,
    ) -> Optional[str]:
        """P8: Save enhanced overlay with dropped token categories.

        This is visualization-only. It only calls save_token_selection_debug_with_dropped
        which renders token sets; it does NOT change any selection logic or scores.
        """
        try:
            from .visualization import save_token_selection_debug_with_dropped
        except ImportError:
            return None

        latest = self.geometry_recorder.get_latest() if self.geometry_recorder is not None else None
        if latest is None:
            return None
        if getattr(latest, "rgb", None) is None:
            return None

        episode_id = int(getattr(latest, "episode_id", 0))
        step_id = int(getattr(latest, "step_id", 0))
        grid_shape = aux_metrics.get("token_grid_shape", (16, 16))

        token_sets = self._reconstruct_dropped_token_sets(
            keep_indices_np=keep_indices_np,
            aux_metrics=aux_metrics,
            selection_meta=selection_meta,
            num_tokens=num_tokens,
            grid_shape=grid_shape,
        )

        return save_token_selection_debug_with_dropped(
            output_dir=str(self.cfg.get("save_dir") or "outputs"),
            method=self.config.strategy,
            episode_id=episode_id,
            step_id=step_id,
            rgb=getattr(latest, "rgb", None),
            token_u=aux_metrics.get("token_u"),
            token_v=aux_metrics.get("token_v"),
            keep_indices=keep_indices_np,
            depth_edge_topk_indices=list(token_sets.get("depth_edge_topk_indices", set())),
            depth_edge_dropped_indices=list(token_sets.get("depth_edge_topk_dropped_indices", set())),
            robot_geo_topk_indices=list(token_sets.get("robot_geo_topk_indices", set())),
            robot_geo_dropped_indices=list(token_sets.get("robot_geo_topk_dropped_indices", set())),
            reserved_edge_indices=list(token_sets.get("reserved_edge_indices", set())),
            non_reserved_edge_dropped_indices=list(token_sets.get("non_reserved_edge_dropped_indices", set())),
            # P11.3: Branch attribution sets from selection_meta
            depth_branch_indices=list(token_sets.get("depth_branch_indices", set())),
            hybrid_branch_indices=list(token_sets.get("hybrid_branch_indices", set())),
            fill_branch_indices=list(token_sets.get("fill_branch_indices", set())),
            gripper_pixel=aux_metrics.get("gripper_pixel"),
            token_grid_shape=grid_shape,
            selection_meta=selection_meta,
            reconstruction_metadata=token_sets.get("reconstruction_metadata", {}),
        )

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


def _selected_score_stats(aux_metrics: Dict[str, Any], idx: np.ndarray, num_tokens: int) -> Dict[str, Optional[float]]:
    """Compute score statistics over selected (kept) tokens only."""
    result: Dict[str, Optional[float]] = {}
    component_names = [
        ("edge_scores", "depth_edge_score"),
        ("near_scores", "distance_score"),
        ("rule_v0_motion_cone_scores", "motion_cone_score"),
        ("rule_v0_workspace_scores", "workspace_score"),
        ("rule_v0_contact_risk_scores", "contact_risk_score"),
        ("final_scores", "final_geometry_score"),
    ]
    for arr_key, prefix in component_names:
        arr = aux_metrics.get(arr_key)
        if arr is None:
            continue
        a = np.asarray(arr, dtype=np.float32).reshape(-1)
        if a.size == 0:
            continue
        valid_idx = idx[(idx >= 0) & (idx < num_tokens)]
        if valid_idx.size == 0:
            continue
        selected = a[valid_idx]
        selected = selected[np.isfinite(selected)]
        if selected.size == 0:
            continue
        result[f"selected_{prefix}_mean"] = float(np.mean(selected))
        sorted_sel = np.sort(selected)
        result[f"selected_{prefix}_p50"] = float(np.median(selected))
        p90_i = max(0, int(round(0.90 * selected.size)) - 1)
        result[f"selected_{prefix}_p90"] = float(sorted_sel[min(p90_i, selected.size - 1)])
        result[f"selected_{prefix}_max"] = float(np.max(selected))
    return result


def _append_score_stats(stats: Dict[str, Any], key: str, arr: np.ndarray, valid_mask: np.ndarray) -> None:
    """Compute and append distribution stats for a score array into the stats dict."""
    if arr is None:
        return
    a = np.asarray(arr, dtype=np.float32).reshape(-1)
    v = a[np.isfinite(a) & valid_mask]
    if v.size == 0:
        return
    prefix = key.replace("_scores", "_score")
    stats[f"{prefix}_mean"] = float(np.mean(v))
    stats[f"{prefix}_std"] = float(np.std(v))
    stats[f"{prefix}_min"] = float(np.min(v))
    stats[f"{prefix}_max"] = float(np.max(v))
    sorted_v = np.sort(v)
    stats[f"{prefix}_p50"] = float(np.median(v))
    p90_idx = max(0, int(round(0.90 * v.size)) - 1)
    stats[f"{prefix}_p90"] = float(sorted_v[min(p90_idx, v.size - 1)])
    if "motion_cone" in key:
        pos_ratio = float(np.mean(v > 1e-6))
        stats[f"{prefix}_positive_ratio"] = pos_ratio
        stats[f"{prefix}_zero_ratio"] = 1.0 - pos_ratio
        # motion direction norm (if stored)
        if "motion_direction" in stats and stats["motion_direction"] is not None:
            md = np.asarray(stats["motion_direction"], dtype=np.float32).reshape(-1)
            md_valid = np.isfinite(md)
            if np.any(md_valid):
                stats["motion_dir_norm_mean"] = float(np.mean(np.abs(md[md_valid])))
                stats["motion_dir_norm_min"] = float(np.min(np.abs(md[md_valid])))
                stats["motion_dir_norm_max"] = float(np.max(np.abs(md[md_valid])))


def _compute_token_selection_attribution(
    aux_metrics: Dict[str, Any],
    keep_indices: np.ndarray,
    num_tokens: int,
    token_grid_shape: Tuple[int, int],
) -> Dict[str, Any]:
    """Compute P1-1 token selection attribution and top-k competition diagnostics.

    Returns a flat dict of diagnostic fields.
    """
    result: Dict[str, Any] = {}
    n = int(num_tokens)
    if n <= 0:
        return result

    grid_h, grid_w = int(token_grid_shape[0]), int(token_grid_shape[1])
    idx = np.asarray(keep_indices, dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < n)]
    k_final = len(idx)
    keep_set = set(idx.tolist()) if k_final > 0 else set()

    # ---- Workspace score diagnostics ----
    ws_arr = aux_metrics.get("workspace_scores")
    if ws_arr is not None:
        ws = np.asarray(ws_arr, dtype=np.float32).reshape(-1)
        valid = np.isfinite(ws)
        ws_v = ws[valid] if np.any(valid) else np.array([], dtype=np.float32)
        if ws_v.size > 0:
            result["workspace_score_min"] = float(np.min(ws_v))
            result["workspace_score_unique_count"] = len(set(float(v) for v in ws_v))
            all_one = bool(np.allclose(ws_v, 1.0, atol=1e-6))
            result["workspace_all_one"] = all_one
            if all_one:
                bounds = aux_metrics.get("workspace_bounds")
                if bounds is not None:
                    result["workspace_all_one_reason"] = (
                        f"all_tokens_inside_workspace_bounds:{bounds}"
                    )
                else:
                    result["workspace_all_one_reason"] = (
                        "all_tokens_inside_default_workspace_bounds:"
                        "((-2.0,2.0),(-2.0,2.0),(-0.5,2.0))"
                    )
            valid_ws = np.sum(valid)
            result["workspace_valid_token_ratio"] = float(valid_ws) / float(n) if n > 0 else None
        else:
            result["workspace_score_min"] = None
            result["workspace_score_unique_count"] = 0
            result["workspace_all_one"] = None
            result["workspace_all_one_reason"] = "no_valid_workspace_scores"
            result["workspace_valid_token_ratio"] = 0.0

    ws_fallback = aux_metrics.get("workspace_fallback_used")
    if ws_fallback is not None:
        result["workspace_fallback_used"] = bool(ws_fallback)

    ws_bounds = aux_metrics.get("workspace_bounds")
    if ws_bounds is not None:
        result["workspace_bounds"] = str(ws_bounds)

    # ---- Score component unique counts ----
    for arr_key, field_name in [
        ("near_scores", "near_score_unique_count"),
        ("edge_scores", "depth_edge_score_unique_count"),
        ("motion_cone_scores", "motion_cone_score_unique_count"),
        ("final_scores", "final_score_unique_count"),
    ]:
        arr = aux_metrics.get(arr_key)
        if arr is not None:
            a = np.asarray(arr, dtype=np.float32).reshape(-1)
            valid = np.isfinite(a)
            v = a[valid] if np.any(valid) else np.array([], dtype=np.float32)
            result[field_name] = len(set(float(x) for x in v)) if v.size > 0 else 0

    # ---- Top-k overlap competition (depth_edge vs robot_geo vs final) ----
    # Get depth_edge scores for top-k
    edge_arr = aux_metrics.get("edge_scores")
    edge_scores_np = np.asarray(edge_arr, dtype=np.float32).reshape(-1) if edge_arr is not None else None

    # Get robot_geo final scores.
    # Priority: hybrid_final_scores (P1, precomputed) > final_scores (rule_v0 fallback)
    hybrid_arr = aux_metrics.get("hybrid_final_scores")
    final_arr = hybrid_arr if hybrid_arr is not None else aux_metrics.get("final_scores")
    final_scores_np = np.asarray(final_arr, dtype=np.float32).reshape(-1) if final_arr is not None else None

    if edge_scores_np is not None and edge_scores_np.size >= n:
        # depth_edge top-k: 80% of keep_k (matching select_hybrid_v1 phase 1)
        edge_topk_k = max(1, int(round(k_final * 0.80))) if k_final > 0 else 0
        edge_adj = np.where(np.isfinite(edge_scores_np), edge_scores_np, -np.inf)
        edge_order = np.lexsort((np.arange(n), -edge_adj))
        edge_topk = set(int(edge_order[i]) for i in range(min(edge_topk_k, n)))

        result["depth_edge_topk_count"] = len(edge_topk)
        result["depth_edge_topk_kept_in_final_count"] = len(edge_topk & keep_set)
        result["depth_edge_topk_dropped_count"] = len(edge_topk - keep_set)
        result["depth_edge_topk_dropped_ratio"] = (
            len(edge_topk - keep_set) / len(edge_topk) if edge_topk else None
        )
    else:
        result["depth_edge_topk_count"] = None
        result["depth_edge_topk_kept_in_final_count"] = None
        result["depth_edge_topk_dropped_count"] = None
        result["depth_edge_topk_dropped_ratio"] = None

    if final_scores_np is not None and final_scores_np.size >= n:
        # robot_geo top-k: same as depth_edge top-k count
        geo_topk_k = max(1, int(round(k_final * 0.80))) if k_final > 0 else 0
        geo_adj = np.where(np.isfinite(final_scores_np), final_scores_np, -np.inf)
        geo_order = np.lexsort((np.arange(n), -geo_adj))
        geo_topk = set(int(geo_order[i]) for i in range(min(geo_topk_k, n)))

        result["robot_geo_topk_count"] = len(geo_topk)
        result["robot_geo_topk_kept_in_final_count"] = len(geo_topk & keep_set)
        result["robot_geo_topk_dropped_count"] = len(geo_topk - keep_set)
        result["robot_geo_topk_dropped_ratio"] = (
            len(geo_topk - keep_set) / len(geo_topk) if geo_topk else None
        )

        if edge_scores_np is not None and edge_scores_np.size >= n:
            overlap = edge_topk & geo_topk
            result["overlap_depth_edge_robot_geo_count"] = len(overlap)
            total = len(edge_topk | geo_topk)
            result["overlap_depth_edge_robot_geo_ratio"] = len(overlap) / total if total > 0 else None
    else:
        result["robot_geo_topk_count"] = None
        result["robot_geo_topk_kept_in_final_count"] = None
        result["robot_geo_topk_dropped_count"] = None
        result["robot_geo_topk_dropped_ratio"] = None
        result["overlap_depth_edge_robot_geo_count"] = None
        result["overlap_depth_edge_robot_geo_ratio"] = None

    result["final_selected_count"] = k_final

    # ---- Selected token attribution ratios ----
    if k_final > 0 and n > 0:
        # Selected high depth_edge but low robot_geo count
        if edge_scores_np is not None and final_scores_np is not None:
            edge_topk_thresh = float(np.percentile(edge_scores_np[edge_scores_np > -np.inf], 80))
            geo_topk_thresh = float(np.percentile(final_scores_np[final_scores_np > -np.inf], 80))
            high_edge_low_geo = 0
            for i in idx:
                if i < n and edge_scores_np[i] >= edge_topk_thresh:
                    if i < n and final_scores_np[i] < geo_topk_thresh:
                        high_edge_low_geo += 1
            result["selected_high_depth_edge_but_low_robot_geo_count"] = high_edge_low_geo
        else:
            result["selected_high_depth_edge_but_low_robot_geo_count"] = None

        # Dropped high depth_edge count
        if edge_scores_np is not None:
            edge_topk_thresh = float(np.percentile(edge_scores_np[edge_scores_np > -np.inf], 80))
            dropped_high_edge = 0
            for i in range(n):
                if i not in keep_set and edge_scores_np[i] >= edge_topk_thresh:
                    dropped_high_edge += 1
            result["dropped_high_depth_edge_tokens_count"] = dropped_high_edge
        else:
            result["dropped_high_depth_edge_tokens_count"] = None

        # Dropped high robot_geo count
        if final_scores_np is not None:
            geo_topk_thresh = float(np.percentile(final_scores_np[final_scores_np > -np.inf], 80))
            dropped_high_geo = 0
            for i in range(n):
                if i not in keep_set and final_scores_np[i] >= geo_topk_thresh:
                    dropped_high_geo += 1
            result["dropped_high_robot_geo_tokens_count"] = dropped_high_geo
        else:
            result["dropped_high_robot_geo_tokens_count"] = None

    # ---- Selected token spatial distribution (UV grid coordinates) ----
    token_u = aux_metrics.get("token_u")
    token_v = aux_metrics.get("token_v")
    gripper_pixel = aux_metrics.get("gripper_pixel")

    if k_final > 0 and token_u is not None and token_v is not None:
        u_arr = np.asarray(token_u, dtype=np.float32).reshape(-1)
        v_arr = np.asarray(token_v, dtype=np.float32).reshape(-1)
        if u_arr.size >= n and v_arr.size >= n:
            u_sel = u_arr[idx[idx < u_arr.size]]
            v_sel = v_arr[idx[idx < v_arr.size]]
            if u_sel.size > 0:
                result["selected_token_u_mean"] = float(np.mean(u_sel))
                result["selected_token_u_std"] = float(np.std(u_sel)) if u_sel.size > 1 else 0.0
                result["selected_token_v_mean"] = float(np.mean(v_sel))
                result["selected_token_v_std"] = float(np.std(v_sel)) if v_sel.size > 1 else 0.0
                result["selected_token_bbox_u_min"] = int(np.min(u_sel))
                result["selected_token_bbox_u_max"] = int(np.max(u_sel))
                result["selected_token_bbox_v_min"] = int(np.min(v_sel))
                result["selected_token_bbox_v_max"] = int(np.max(v_sel))

            # Grid quadrant histogram: [TL, TR, BL, BR]
            if grid_h > 0 and grid_w > 0 and u_sel.size > 0:
                mid_u = float(np.median(u_arr[:n]))
                mid_v = float(np.median(v_arr[:n]))
                q_tl = int(np.sum((u_sel < mid_u) & (v_sel < mid_v)))
                q_tr = int(np.sum((u_sel >= mid_u) & (v_sel < mid_v)))
                q_bl = int(np.sum((u_sel < mid_u) & (v_sel >= mid_v)))
                q_br = int(np.sum((u_sel >= mid_u) & (v_sel >= mid_v)))
                result["selected_token_grid_quadrant_histogram"] = f"[{q_tl},{q_tr},{q_bl},{q_br}]"

            # Gripper pixel distance
            if gripper_pixel is not None:
                gx, gy = float(gripper_pixel[0]), float(gripper_pixel[1])
                if u_sel.size > 0 and v_sel.size > 0:
                    dists = np.sqrt((u_sel - gx) ** 2 + (v_sel - gy) ** 2)
                    result["selected_token_near_gripper_pixel_dist_mean"] = float(np.mean(dists))
                    result["selected_token_near_gripper_pixel_dist_median"] = float(np.median(dists))

    return result


def _first_present(values: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if values.get(key) is not None:
            return values.get(key)
    return None


def _compute_score_stats(arr: Optional[np.ndarray], name: str = "score") -> Dict[str, Any]:
    """Compute per-component distribution statistics from a flat numpy array."""
    if arr is None:
        return {}
    a = np.asarray(arr, dtype=np.float32).reshape(-1)
    valid = np.isfinite(a)
    if not np.any(valid):
        return {}
    v = a[valid]
    result = {}
    prefix = f"{name}_" if name else ""
    result[f"{prefix}mean"] = float(np.mean(v))
    result[f"{prefix}std"] = float(np.std(v))
    result[f"{prefix}min"] = float(np.min(v))
    result[f"{prefix}max"] = float(np.max(v))
    if v.size >= 2:
        sorted_v = np.sort(v)
        result[f"{prefix}p50"] = float(np.median(v))
        p90_idx = max(0, int(round(0.90 * v.size)) - 1)
        result[f"{prefix}p90"] = float(sorted_v[min(p90_idx, v.size - 1)])
    return result


def _compute_selected_score_stats(
    arr: Optional[np.ndarray],
    keep_indices: np.ndarray,
    num_tokens: int,
    name: str = "score",
) -> Dict[str, Any]:
    """Compute score statistics over selected (kept) tokens only."""
    if arr is None or keep_indices is None or keep_indices.size == 0:
        return {}
    a = np.asarray(arr, dtype=np.float32).reshape(-1)
    idx = np.asarray(keep_indices, dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < num_tokens)]
    if idx.size == 0:
        return {}
    valid = np.isfinite(a[idx])
    selected = a[idx[valid]]
    if selected.size == 0:
        return {}
    prefix = f"selected_{name}_"
    result: Dict[str, Any] = {}
    result[f"{prefix}mean"] = float(np.mean(selected))
    sorted_selected = np.sort(selected)
    result[f"{prefix}p50"] = float(np.median(selected))
    p90_idx = max(0, int(round(0.90 * selected.size)) - 1)
    result[f"{prefix}p90"] = float(sorted_selected[min(p90_idx, selected.size - 1)])
    result[f"{prefix}max"] = float(np.max(selected))
    return result


def _arr_stats(arr: Any, stat: str = "min") -> Optional[str]:
    """Convert an array to a 'x,y,z' string summary for min/max/mean/std."""
    if arr is None:
        return None
    try:
        a = np.asarray(arr, dtype=np.float32).reshape(-1, 3)
        valid = np.all(np.isfinite(a), axis=1)
        if not np.any(valid):
            return None
        va = a[valid]
        if stat == "min":
            vals = np.nanmin(va, axis=0)
        elif stat == "max":
            vals = np.nanmax(va, axis=0)
        elif stat == "mean":
            vals = np.nanmean(va, axis=0)
        elif stat == "std":
            vals = np.nanstd(va, axis=0)
        else:
            return None
        return f"{float(vals[0]):.4f},{float(vals[1]):.4f},{float(vals[2]):.4f}"
    except Exception:
        return None


def _arr_to_str(arr: Any) -> Optional[str]:
    """Convert a 3-element array to 'x,y,z' string."""
    if arr is None:
        return None
    try:
        a = np.asarray(arr, dtype=np.float32).reshape(-1)
        vals = a[:3]
        if not np.all(np.isfinite(vals)):
            return None
        return f"{float(vals[0]):.4f},{float(vals[1]):.4f},{float(vals[2]):.4f}"
    except Exception:
        return None


# =============================================================================
# P1: Hybrid score helpers for selection attribution
# =============================================================================


def _norm_hybrid_component(arr: Optional[np.ndarray], valid_mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Min-max normalize a score array over valid tokens.

    Returns zeros for invalid entries. Matches the normalization logic in
    ``select_hybrid_v1._norm()``.
    """
    if arr is None:
        return None
    a = np.nan_to_num(np.asarray(arr, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    else:
        valid = np.isfinite(a) & (np.abs(a) < 1e10)
    out = np.zeros_like(a)
    if not np.any(valid):
        return out
    lo, hi = float(np.min(a[valid])), float(np.max(a[valid]))
    if hi - lo > 1e-8:
        out[valid] = (a[valid] - lo) / (hi - lo)
    return out


def _build_hybrid_final_scores(
    edge: Optional[np.ndarray],
    near: Optional[np.ndarray],
    contact: Optional[np.ndarray],
    corridor: Optional[np.ndarray],
    valid_mask: np.ndarray,
    weights: Dict[str, float],
) -> Optional[np.ndarray]:
    """Build hybrid v1 final scores using the same formula as ``select_hybrid_v1``.

    If any component is None, that weight is redistributed proportionally.
    """
    w_edge = float(weights.get("w_edge", 0.45))
    w_near = float(weights.get("w_near", 0.20))
    w_contact = float(weights.get("w_contact", 0.20))
    w_corr = float(weights.get("w_corr", 0.10))
    w_diverse = float(weights.get("w_diverse", 0.05))

    n = 0
    if edge is not None:
        n = int(np.asarray(edge).reshape(-1).shape[0])
    elif near is not None:
        n = int(np.asarray(near).reshape(-1).shape[0])

    if n == 0:
        return None

    if edge is not None:
        norm_edge = _norm_hybrid_component(edge)
    else:
        norm_edge = np.zeros(n, dtype=np.float32)

    if near is not None:
        norm_near = _norm_hybrid_component(near)
    else:
        norm_near = np.zeros(n, dtype=np.float32)

    if contact is not None:
        norm_contact = _norm_hybrid_component(contact)
    else:
        norm_contact = np.zeros(n, dtype=np.float32)

    if corridor is not None:
        norm_corr = _norm_hybrid_component(corridor)
    else:
        norm_corr = np.zeros(n, dtype=np.float32)

    valid = np.asarray(valid_mask, dtype=np.bool_).reshape(-1) if valid_mask is not None else np.ones(n, dtype=np.bool_)

    scores = w_edge * norm_edge + w_near * norm_near + w_contact * norm_contact + w_corr * norm_corr
    # No w_diverse since spatial diversity is selection-based, not score-based
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    scores[~valid] = 0.0
    return scores
