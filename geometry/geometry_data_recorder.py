"""
GeometryDataRecorder: Collects geometric data during evaluation for visual token analysis.

This module provides the GeometryStepData dataclass (unified data structure) and
GeometryDataRecorder class (data collection interface) needed for geometry-guided
visual token pruning research.

Key design principles:
1. Graceful degradation: Returns None for unavailable fields without crashing
2. Baseline compatibility: Does not affect baseline mode (no-op when disabled)
3. Debug visibility: Optional detailed logging of data shapes and availability
4. Unified depth conversion: Uses geometry_depth.convert_depth_to_metric for
   robosuite z-buffer to metric depth conversion.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from geometry.geometry_depth import DepthConversionResult, convert_depth_to_metric

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class GeometryStepData:
    """
    Unified data structure for geometric information collected at each evaluation step.

    This dataclass serves as the interface between the environment and the geometry
    expert for visual token analysis. All fields are optional to ensure graceful
    degradation when certain sensor data is unavailable.

    Fields:
        rgb: RGB image (H x W x 3), uint8
        depth: Depth image (H x W), float32 in meters, or None if unavailable
        camera_intrinsics: 3x3 camera intrinsics matrix K [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        camera_extrinsics: 4x4 extrinsics matrix T_base_cam (transform from camera to base frame)
        ee_pose: 4x4 end-effector pose matrix T_base_ee (transform from ee to base frame)
        gripper_pos: 3D gripper position in base frame (x, y, z)
        prev_gripper_pos: Previous 3D gripper position in base frame
        action: Current action vector [dx, dy, dz, rx, ry, rz, gripper]
        task_name: Name of the current task
        episode_id: Episode identifier
        step_id: Step identifier within the episode
        timestamp: Wall-clock timestamp from time.perf_counter()
    """

    rgb: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    depth_metadata: Optional[Dict[str, Any]] = None  # from geometry_depth.convert_depth_to_metric
    camera_intrinsics: Optional[np.ndarray] = None
    camera_extrinsics: Optional[np.ndarray] = None
    ee_pose: Optional[np.ndarray] = None
    gripper_pos: Optional[np.ndarray] = None
    prev_gripper_pos: Optional[np.ndarray] = None
    action: Optional[np.ndarray] = None
    task_name: str = ""
    episode_id: int = 0
    step_id: int = 0
    timestamp: float = 0.0

    def get_availability_summary(self) -> Dict[str, bool]:
        """Return a dict indicating which fields are non-None."""
        return {
            "rgb": self.rgb is not None,
            "depth": self.depth is not None,
            "camera_intrinsics": self.camera_intrinsics is not None,
            "camera_extrinsics": self.camera_extrinsics is not None,
            "ee_pose": self.ee_pose is not None,
            "gripper_pos": self.gripper_pos is not None,
            "prev_gripper_pos": self.prev_gripper_pos is not None,
            "action": self.action is not None,
        }

    def get_shape_info(self) -> Dict[str, Optional[str]]:
        """Return a dict with shape strings for non-None array fields."""
        info = {}
        for field_name in ["rgb", "depth", "camera_intrinsics", "camera_extrinsics", "ee_pose",
                           "gripper_pos", "prev_gripper_pos", "action"]:
            value = getattr(self, field_name)
            if value is not None and hasattr(value, "shape"):
                info[field_name] = str(value.shape)
            else:
                info[field_name] = None
        return info


# =============================================================================
# Geometry Data Recorder
# =============================================================================


class GeometryDataRecorder:
    """
    Collects geometric data during evaluation for visual token analysis.

    This recorder acts as an interface between the environment adapter and the
    geometry expert. It handles data collection at each evaluation step and provides
    a unified API regardless of which environment is used.

    Usage:
        # In evaluation setup
        recorder = GeometryDataRecorder(enabled=True, debug=True)
        recorder.reset(episode_id=0, task_name="task_0")

        # At each evaluation step
        step_data = recorder.collect_step(
            rgb=rgb_image,
            obs=observation,
            action=predicted_action,
            step_id=current_step,
            raw_env_obs=raw_environment_obs,
            current_ee_pose=env.get_ee_pose(),
        )

        # Access collected data
        history = recorder.get_history()

    Args:
        enabled: Whether recording is enabled. If False, all methods are no-ops.
        debug: Whether to print detailed debug information about data shapes.
    """

    def __init__(self, enabled: bool = True, debug: bool = False) -> None:
        self._enabled = enabled
        self._debug = debug
        self._history: List[GeometryStepData] = []
        self._episode_id: int = 0
        self._task_name: str = ""
        self._prev_gripper_pos: Optional[np.ndarray] = None
        self._step_count: int = 0
        self._raw_obs_debug_printed = False
        self._depth_debug_printed = False
        self._camera_debug_printed = set()
        self._state_debug_printed = False

        if self._enabled:
            logger.info("[GeometryDataRecorder] Initialized (debug=%s)", self._debug)
        else:
            logger.info("[GeometryDataRecorder] Disabled (no-op mode)")

    @property
    def enabled(self) -> bool:
        """Whether recording is enabled."""
        return self._enabled

    @property
    def debug(self) -> bool:
        """Whether debug logging is enabled."""
        return self._debug

    def reset(self, episode_id: int, task_name: str) -> None:
        """
        Reset the recorder for a new episode.

        Args:
            episode_id: Unique identifier for the new episode.
            task_name: Name/description of the task for this episode.
        """
        if not self._enabled:
            return

        self._history.clear()
        self._episode_id = episode_id
        self._task_name = task_name
        self._prev_gripper_pos = None
        self._step_count = 0
        self._raw_obs_debug_printed = False
        self._depth_debug_printed = False
        self._camera_debug_printed = set()
        self._state_debug_printed = False

        if self._debug:
            logger.debug("[GeometryDataRecorder] Reset for episode=%d, task=%s", episode_id, task_name)

    def collect_step(
        self,
        rgb: np.ndarray,
        obs: Dict[str, Any],
        action: Optional[np.ndarray],
        step_id: int,
        raw_env_obs: Optional[Dict[str, Any]] = None,
        current_ee_pose: Optional[np.ndarray] = None,
    ) -> Optional[GeometryStepData]:
        """
        Collect geometric data for the current evaluation step.

        This is the main entry point for data collection. It gathers all available
        geometric information from the environment and packages it into a
        GeometryStepData instance.

        Args:
            rgb: RGB image array (H x W x 3), uint8.
            obs: Processed observation dict from env.get_observation().
            action: Predicted action vector (7,), or None when collecting
                geometry before model inference.
            step_id: Current step index within the episode.
            raw_env_obs: Raw environment observation dict (optional, for extended data).
            current_ee_pose: 4x4 end-effector pose matrix (optional).

        Returns:
            GeometryStepData instance with collected data, or None if disabled.
        """
        if not self._enabled:
            return None

        timestamp = time.perf_counter()
        if self._debug:
            self._log_raw_obs_debug(obs=obs, raw_env_obs=raw_env_obs)

        # Extract all geometric data from environment
        depth, depth_metadata = self._get_depth_from_env(raw_env_obs)
        camera_intrinsics = self._get_camera_intrinsics_from_env(raw_env_obs)
        camera_extrinsics = self._get_camera_extrinsics_from_env(raw_env_obs)

        # Get EE pose from env or use provided one
        ee_pose = current_ee_pose
        if ee_pose is None:
            ee_pose = self._get_ee_pose_from_env(raw_env_obs)
        elif self._debug:
            self._log_state_debug_once("env.get_ee_pose", np.asarray(ee_pose, dtype=np.float32))

        # Compute gripper position from EE pose
        gripper_pos = self._get_gripper_pos_from_ee_pose(ee_pose)

        # Normalize action to numpy array when available. Geometry collection is
        # intentionally allowed before model inference so current-frame geometry
        # can be used by pruning hooks.
        action_arr = None if action is None else np.asarray(action, dtype=np.float32)

        # Create step data
        step_data = GeometryStepData(
            rgb=rgb.copy() if rgb is not None else None,
            depth=depth.copy() if depth is not None else None,
            depth_metadata=depth_metadata,
            camera_intrinsics=camera_intrinsics.copy() if camera_intrinsics is not None else None,
            camera_extrinsics=camera_extrinsics.copy() if camera_extrinsics is not None else None,
            ee_pose=ee_pose.copy() if ee_pose is not None else None,
            gripper_pos=gripper_pos.copy() if gripper_pos is not None else None,
            prev_gripper_pos=self._prev_gripper_pos.copy() if self._prev_gripper_pos is not None else None,
            action=action_arr,
            task_name=self._task_name,
            episode_id=self._episode_id,
            step_id=step_id,
            timestamp=timestamp,
        )

        # Store for next step
        self._history.append(step_data)
        if gripper_pos is not None:
            self._prev_gripper_pos = gripper_pos.copy()
        self._step_count += 1

        # Debug logging
        if self._debug:
            self._log_step_debug(step_data)

        return step_data

    def get_history(self) -> List[GeometryStepData]:
        """
        Return the full history of collected step data for the current episode.

        Returns:
            List of GeometryStepData instances in chronological order.
        """
        return self._history.copy()

    def get_latest(self) -> Optional[GeometryStepData]:
        """
        Return the most recently collected step data.

        Returns:
            Latest GeometryStepData instance, or None if no data collected.
        """
        return self._history[-1] if self._history else None

    def get_prev_gripper_pos(self) -> Optional[np.ndarray]:
        """
        Get the gripper position from the previous step.

        This is useful for computing gripper velocity or detecting gripper changes.

        Returns:
            Previous gripper position (3,) or None if not available.
        """
        return self._prev_gripper_pos.copy() if self._prev_gripper_pos is not None else None

    def get_step_count(self) -> int:
        """Return the number of steps recorded in the current episode."""
        return self._step_count

    # -------------------------------------------------------------------------
    # Environment-specific data extraction methods
    # These methods handle the interface between different environment backends
    # -------------------------------------------------------------------------

    def _get_depth_from_env(
        self, obs: Optional[Dict[str, Any]]
    ) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """
        Extract and convert depth image from environment observation.

        Args:
            obs: Raw environment observation dict.

        Returns:
            Tuple of (depth_image, depth_metadata):
            - depth_image: (H x W) float32 in meters, or None
            - depth_metadata: dict from convert_depth_to_metric, or None
        """
        if obs is None:
            return None, None

        camera_name = obs.get("geometry_camera_name") or obs.get("camera_name") or "agentview"
        candidate_keys = [
            f"{camera_name}_depth_metric",
            "depth_metric",
            f"{camera_name}_depth",
            "depth",
            "depth_image",
            "depth_camera",
            "robot0_eye_depth",
            "robot0_eye_in_hand_depth",
        ]

        # Try metric depth first, then normalized robosuite depth.
        for key in candidate_keys:
            if key in obs:
                depth_raw = obs[key]
                if depth_raw is None:
                    continue
                sim = obs.get("env_sim")
                image_transform = obs.get("image_transform")
                result = convert_depth_to_metric(
                    depth_raw=depth_raw,
                    sim=sim,
                    source_key=key,
                    image_transform=image_transform,
                    _debug=self._debug,
                )
                if self._debug and not self._depth_debug_printed:
                    raw_s = result.metadata.get("depth_raw_stats", {})
                    met_s = result.metadata.get("depth_metric_stats", {})
                    print(f"[GeometryDataRecorder] depth key={key}, conversion={result.metadata['conversion']}")
                    print(f"[GeometryDataRecorder] depth_raw  min={raw_s.get('min',0):.4f} max={raw_s.get('max',0):.4f} mean={raw_s.get('mean',0):.4f} std={raw_s.get('std',0):.4f}")
                    if met_s:
                        print(f"[GeometryDataRecorder] depth_metric min={met_s.get('min',0):.4f} max={met_s.get('max',0):.4f} mean={met_s.get('mean',0):.4f} std={met_s.get('std',0):.4f}")
                    print(f"[GeometryDataRecorder] depth_is_metric={result.metadata['depth_is_metric']}, unit={result.metadata['depth_unit']}")
                    self._depth_debug_printed = True
                return result.depth, result.metadata

        # Log warning once per session for missing depth
        if not hasattr(self, "_depth_warning_issued"):
            self._depth_warning_issued = True
            logger.warning(
                "[GeometryDataRecorder] No depth data found in observation. "
                "Available keys: %s",
                list(obs.keys()) if obs else [],
            )

        return None, None

    def _get_camera_intrinsics_from_env(
        self, obs: Optional[Dict[str, Any]]
    ) -> Optional[np.ndarray]:
        """
        Extract camera intrinsics from environment observation.

        Args:
            obs: Raw environment observation dict.

        Returns:
            3x3 camera intrinsics matrix K, or None if unavailable.
        """
        if obs is None:
            return None

        # Try to find intrinsics in observation
        for key in ["camera_intrinsics", "camera_K", "K", "intrinsics"]:
            if key in obs:
                K = obs[key]
                if K is not None:
                    K_arr = np.asarray(K, dtype=np.float32)
                    if K_arr.shape == (3, 3):
                        if self._debug:
                            self._log_camera_debug_once("camera_intrinsics", key, K_arr)
                        return K_arr

        if obs.get("env_sim") is not None and obs.get("geometry_camera_name") is not None:
            try:
                from robosuite.utils import camera_utils as CU
                camera_name = obs["geometry_camera_name"]
                height = int(obs.get("camera_height", 256))
                width = int(obs.get("camera_width", 256))
                K_arr = CU.get_camera_intrinsic_matrix(obs["env_sim"], camera_name, height, width)
                K_arr = np.asarray(K_arr, dtype=np.float32)
                if obs.get("image_transform") == "rot180":
                    K_arr = K_arr.copy()
                    K_arr[0, 2] = (width - 1) - K_arr[0, 2]
                    K_arr[1, 2] = (height - 1) - K_arr[1, 2]
                if self._debug:
                    self._log_camera_debug_once("camera_intrinsics", "robosuite_camera_utils", K_arr)
                return K_arr
            except Exception as exc:
                logger.warning("[GeometryDataRecorder] Failed to read robosuite camera intrinsics: %s", exc)

        # Try to extract from observation metadata
        if "camera_param" in obs:
            params = obs["camera_param"]
            if isinstance(params, dict) and "intrinsics" in params:
                K = params["intrinsics"]
                if K is not None:
                    if self._debug:
                        logger.debug("[GeometryDataRecorder] Found intrinsics in camera_param")
                    return np.asarray(K, dtype=np.float32)

        # Log warning once per session
        if not hasattr(self, "_intrinsics_warning_issued"):
            self._intrinsics_warning_issued = True
            logger.warning("[GeometryDataRecorder] No camera intrinsics found. "
                          "You may need to implement camera calibration for this environment. "
                          "Available keys: %s", list(obs.keys()) if obs else [])

        return None

    def _get_camera_extrinsics_from_env(
        self, obs: Optional[Dict[str, Any]]
    ) -> Optional[np.ndarray]:
        """
        Extract camera extrinsics (transform from camera to base frame) from environment.

        Args:
            obs: Raw environment observation dict.

        Returns:
            4x4 extrinsics matrix T_base_cam, or None if unavailable.
        """
        if obs is None:
            return None

        # Try to find extrinsics in observation
        for key in ["camera_extrinsics", "camera_extrinsic", "T_base_cam", "camera_pose", "camera_to_base"]:
            if key in obs:
                T = obs[key]
                if T is not None:
                    T_arr = np.asarray(T, dtype=np.float32)
                    if T_arr.shape == (4, 4):
                        if self._debug:
                            self._log_camera_debug_once("camera_extrinsics", key, T_arr)
                        return T_arr

        if obs.get("env_sim") is not None and obs.get("geometry_camera_name") is not None:
            try:
                from robosuite.utils import camera_utils as CU
                camera_name = obs["geometry_camera_name"]
                T_arr = CU.get_camera_extrinsic_matrix(obs["env_sim"], camera_name)
                T_arr = np.asarray(T_arr, dtype=np.float32)
                if self._debug:
                    print("[GeometryDataRecorder] Assuming robot base frame equals world frame for LIBERO debug.")
                    self._log_camera_debug_once("camera_extrinsics", "robosuite_camera_utils", T_arr)
                return T_arr
            except Exception as exc:
                logger.warning("[GeometryDataRecorder] Failed to read robosuite camera extrinsics: %s", exc)

        # Try to extract from observation metadata
        if "camera_param" in obs:
            params = obs["camera_param"]
            if isinstance(params, dict) and "extrinsics" in params:
                T = params["extrinsics"]
                if T is not None:
                    if self._debug:
                        logger.debug("[GeometryDataRecorder] Found extrinsics in camera_param")
                    return np.asarray(T, dtype=np.float32)

        # Log warning once per session
        if not hasattr(self, "_extrinsics_warning_issued"):
            self._extrinsics_warning_issued = True
            logger.warning("[GeometryDataRecorder] No camera extrinsics found. "
                          "You may need to implement camera-to-robot calibration for this environment. "
                          "Available keys: %s", list(obs.keys()) if obs else [])

        return None

    def _get_ee_pose_from_env(self, obs: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
        """
        Extract end-effector pose from environment observation.

        Args:
            obs: Raw environment observation dict.

        Returns:
            4x4 EE pose matrix T_base_ee, or None if unavailable.
        """
        if obs is None:
            return None

        # Try to find EE pose in observation
        for key in ["ee_pose", "end_effector_pose", "eef_pose", "gripper_pose"]:
            if key in obs:
                pose = obs[key]
                if pose is not None:
                    pose_arr = np.asarray(pose, dtype=np.float32)
                    if pose_arr.shape == (4, 4):
                        if self._debug:
                            self._log_state_debug_once(key, pose_arr)
                        return pose_arr

        for pos_key, quat_key in [
            ("robot0_eef_pos", "robot0_eef_quat"),
            ("eef_pos", "eef_quat"),
            ("ee_pos", "ee_quat"),
            ("gripper_pos", "gripper_quat"),
        ]:
            if pos_key in obs and obs[pos_key] is not None:
                try:
                    T_ee = np.eye(4, dtype=np.float32)
                    T_ee[:3, 3] = np.asarray(obs[pos_key], dtype=np.float32).reshape(3)
                    if quat_key in obs and obs[quat_key] is not None:
                        from robosuite.utils import transform_utils as T
                        T_ee[:3, :3] = T.quat2mat(np.asarray(obs[quat_key], dtype=np.float32).reshape(4))
                    if self._debug:
                        self._log_state_debug_once(f"{pos_key}+{quat_key}", T_ee)
                    return T_ee
                except Exception as exc:
                    logger.warning("[GeometryDataRecorder] Failed to build ee_pose from %s/%s: %s", pos_key, quat_key, exc)

        # Try to extract from robot state
        if "robot_state" in obs:
            state = obs["robot_state"]
            if isinstance(state, dict) and "eef_pose" in state:
                pose = state["eef_pose"]
                if pose is not None:
                    if self._debug:
                        logger.debug("[GeometryDataRecorder] Found ee_pose in robot_state")
                    return np.asarray(pose, dtype=np.float32)

        # Log warning once per session
        if not hasattr(self, "_ee_pose_warning_issued"):
            self._ee_pose_warning_issued = True
            logger.warning("[GeometryDataRecorder] No ee_pose found in observation. "
                          "Available keys: %s", list(obs.keys()) if obs else [])

        return None

    def _get_gripper_pos_from_ee_pose(self, ee_pose: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Extract 3D gripper position from EE pose matrix.

        The gripper position is the translation component of the EE pose.

        Args:
            ee_pose: 4x4 EE pose matrix T_base_ee.

        Returns:
            3D gripper position (x, y, z) in base frame, or None if pose is None.
        """
        if ee_pose is None:
            return None

        pose_arr = np.asarray(ee_pose, dtype=np.float32)
        if pose_arr.shape != (4, 4):
            return None

        # Extract translation (position) from SE(3) transform
        gripper_pos = pose_arr[:3, 3]

        if self._debug:
            logger.debug("[GeometryDataRecorder] Extracted gripper_pos from ee_pose: %s", gripper_pos)

        return gripper_pos

    # -------------------------------------------------------------------------
    # Debug helpers
    # -------------------------------------------------------------------------

    def _array_stats(self, arr: np.ndarray) -> str:
        arr = np.asarray(arr)
        stats = f"shape={arr.shape}, dtype={arr.dtype}"
        if arr.size and np.issubdtype(arr.dtype, np.number):
            finite = arr[np.isfinite(arr)]
            if finite.size:
                stats += f", min={float(np.min(finite)):.6g}, max={float(np.max(finite)):.6g}"
            stats += f", contains_nan={bool(np.isnan(arr).any())}"
            stats += f", all_zero={bool(np.all(arr == 0))}"
        return stats

    def _log_raw_obs_debug(
        self,
        obs: Optional[Dict[str, Any]],
        raw_env_obs: Optional[Dict[str, Any]],
    ) -> None:
        if self._raw_obs_debug_printed:
            return
        raw_keys = sorted(list(raw_env_obs.keys())) if isinstance(raw_env_obs, dict) else []
        proc_keys = sorted(list(obs.keys())) if isinstance(obs, dict) else []
        camera_name = None
        if isinstance(raw_env_obs, dict):
            camera_name = raw_env_obs.get("geometry_camera_name")
        image_key = f"{camera_name}_image" if camera_name else None
        depth_key = f"{camera_name}_depth" if camera_name else None
        state_keys = [
            k for k in raw_keys
            if any(s in k.lower() for s in ["eef", "ee", "gripper", "robot0"])
        ]
        print(f"[GeometryDataRecorder] processed obs.keys={proc_keys}")
        print(f"[GeometryDataRecorder] raw env obs.keys={raw_keys}")
        print(f"[GeometryDataRecorder] selected camera_name={camera_name}")
        if image_key is not None and isinstance(raw_env_obs, dict):
            print(
                f"[GeometryDataRecorder] key check: "
                f"{image_key}={image_key in raw_env_obs}, {depth_key}={depth_key in raw_env_obs}"
            )
        print(f"[GeometryDataRecorder] robot-state candidate keys={state_keys}")
        self._raw_obs_debug_printed = True

    def _log_camera_debug_once(self, field_name: str, source_key: str, value: np.ndarray) -> None:
        if field_name in self._camera_debug_printed:
            return
        print(f"[GeometryDataRecorder] {field_name} source={source_key}, shape={value.shape}")
        print(value)
        self._camera_debug_printed.add(field_name)

    def _log_state_debug_once(self, source_key: str, ee_pose: np.ndarray) -> None:
        if self._state_debug_printed:
            return
        gripper_pos = ee_pose[:3, 3] if ee_pose.shape == (4, 4) else None
        print(f"[GeometryDataRecorder] ee_pose source={source_key}, shape={ee_pose.shape}")
        print(f"[GeometryDataRecorder] ee_pose is None=False")
        print(f"[GeometryDataRecorder] gripper_pos={gripper_pos.tolist() if gripper_pos is not None else None}")
        self._state_debug_printed = True

    def _log_step_debug(self, step_data: GeometryStepData) -> None:
        """Log debug information for a collected step."""
        if step_data.step_id == 0 or self._step_count == 1:
            print(f"[GeometryDataRecorder] step={step_data.step_id}, shapes={step_data.get_shape_info()}")
            print(f"[GeometryDataRecorder] availability={step_data.get_availability_summary()}")
        logger.debug("=" * 60)
        logger.debug("[GeometryDataRecorder] Step %d (ep=%d, task=%s)",
                    step_data.step_id, step_data.episode_id, step_data.task_name)
        logger.debug("-" * 40)

        # Log shapes
        shape_info = step_data.get_shape_info()
        availability = step_data.get_availability_summary()

        logger.debug("Data availability:")
        for field_name in ["rgb", "depth", "camera_intrinsics", "camera_extrinsics",
                          "ee_pose", "gripper_pos", "prev_gripper_pos", "action"]:
            avail = availability.get(field_name, False)
            shape = shape_info.get(field_name)
            status = "OK" if avail else "None"
            shape_str = f" ({shape})" if shape else ""
            logger.debug("  %-20s: %s%s", field_name, status, shape_str)

        # Log values for non-None fields
        if step_data.gripper_pos is not None:
            logger.debug("  gripper_pos values: [%.3f, %.3f, %.3f]",
                        step_data.gripper_pos[0], step_data.gripper_pos[1], step_data.gripper_pos[2])
        if step_data.prev_gripper_pos is not None:
            logger.debug("  prev_gripper_pos:    [%.3f, %.3f, %.3f]",
                        step_data.prev_gripper_pos[0], step_data.prev_gripper_pos[1], step_data.prev_gripper_pos[2])
        if step_data.action is not None:
            logger.debug("  action values:       [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
                        *step_data.action[:7])
        if step_data.camera_intrinsics is not None:
            logger.debug("  camera_intrinsics:\n%s", step_data.camera_intrinsics)
        if step_data.camera_extrinsics is not None:
            logger.debug("  camera_extrinsics:\n%s", step_data.camera_extrinsics)
        if step_data.ee_pose is not None:
            logger.debug("  ee_pose translation: [%.3f, %.3f, %.3f]",
                        step_data.ee_pose[0, 3], step_data.ee_pose[1, 3], step_data.ee_pose[2, 3])

        logger.debug("=" * 60)


# =============================================================================
# Utility Functions
# =============================================================================


def create_default_camera_intrinsics(
    focal_length: float = 200.0,
    image_width: int = 224,
    image_height: int = 224,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
) -> np.ndarray:
    """
    Create a default camera intrinsics matrix.

    This is useful for testing or when camera calibration is not available.

    Args:
        focal_length: Focal length in pixels (default: 200 for ~60deg FOV at 224x224).
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        cx: Principal point x (defaults to center of image).
        cy: Principal point y (defaults to center of image).

    Returns:
        3x3 camera intrinsics matrix K.
    """
    if cx is None:
        cx = image_width / 2.0
    if cy is None:
        cy = image_height / 2.0

    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1],
    ], dtype=np.float32)

    return K


def create_default_camera_extrinsics(
    camera_position: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    rotation_euler: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """
    Create a default camera extrinsics matrix.

    This is useful for testing or when camera calibration is not available.

    Args:
        camera_position: Camera position in base frame (x, y, z).
        rotation_euler: Camera rotation in Euler angles (rx, ry, rz) in radians.

    Returns:
        4x4 camera extrinsics matrix T_base_cam.
    """
    import math

    # Build rotation matrix from Euler angles (XYZ convention)
    rx, ry, rz = rotation_euler
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    # Rotation matrices
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)

    # Combined rotation (R = Rz * Ry * Rx)
    R = Rz @ Ry @ Rx

    # Build SE(3) transform
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = camera_position

    return T
