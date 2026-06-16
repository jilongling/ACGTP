"""Temporal history, scheduler, and dynamic-controller helpers."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Source: pruning/temporal/acgtp_history.py
# ---------------------------------------------------------------------------
from collections import deque
from typing import Any, Deque, Dict, Optional

import numpy as np


def _to_1d(arr: Optional[np.ndarray], n: Optional[int] = None) -> Optional[np.ndarray]:
    if arr is None:
        return None
    out = np.asarray(arr, dtype=np.float32).reshape(-1)
    if n is not None and out.shape[0] != n:
        return None
    return out


def _norm01(arr: Optional[np.ndarray], n: int, valid: np.ndarray) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    a = _to_1d(arr, n)
    if a is None:
        return out
    finite = valid & np.isfinite(a)
    if not np.any(finite):
        return out
    vals = a[finite]
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if hi - lo > 1e-8:
        out[finite] = (a[finite] - lo) / (hi - lo)
    else:
        out[finite] = np.clip(a[finite], 0.0, 1.0)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _ema(prev: Optional[np.ndarray], cur: Optional[np.ndarray], current_weight: float) -> Optional[np.ndarray]:
    if cur is None:
        return prev
    cur_arr = np.asarray(cur, dtype=np.float32).reshape(-1)
    if prev is None or np.asarray(prev).shape != cur_arr.shape:
        return cur_arr.copy()
    alpha = float(max(0.0, min(1.0, current_weight)))
    out = alpha * cur_arr + (1.0 - alpha) * np.asarray(prev, dtype=np.float32).reshape(-1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class ACGTPHistoryBuffer:
    """Small history buffer for branch-wise EMA and conservative gating."""

    def __init__(
        self,
        *,
        maxlen: int = 5,
        scene_current_weight: float = 0.75,
        depth_current_weight: float = 0.75,
        contact_current_weight: float = 0.45,
        motion_current_weight: float = 0.45,
        action_current_weight: float = 0.50,
        depth_change_threshold: float = 0.18,
        keep_iou_threshold: float = 0.55,
        motion_stability_threshold: float = 0.25,
    ) -> None:
        self.maxlen = max(1, int(maxlen))
        self.scene_current_weight = float(scene_current_weight)
        self.depth_current_weight = float(depth_current_weight)
        self.contact_current_weight = float(contact_current_weight)
        self.motion_current_weight = float(motion_current_weight)
        self.action_current_weight = float(action_current_weight)
        self.depth_change_threshold = float(depth_change_threshold)
        self.keep_iou_threshold = float(keep_iou_threshold)
        self.motion_stability_threshold = float(motion_stability_threshold)
        self._records: Deque[Dict[str, Any]] = deque(maxlen=self.maxlen)
        self._ema_scores: Dict[str, Optional[np.ndarray]] = {
            "scene": None,
            "depth": None,
            "contact": None,
            "motion": None,
            "action": None,
        }
        self._last_depth_norm: Optional[np.ndarray] = None
        self._last_keep_mask: Optional[np.ndarray] = None
        self._last_keep_iou: Optional[float] = None
        self._force_conservative_next: bool = False
        self._force_reason: Optional[str] = None
        self._last_prepare_meta: Dict[str, Any] = {}

    def reset(self) -> None:
        self._records.clear()
        for key in self._ema_scores:
            self._ema_scores[key] = None
        self._last_depth_norm = None
        self._last_keep_mask = None
        self._last_keep_iou = None
        self._force_conservative_next = False
        self._force_reason = None
        self._last_prepare_meta = {}

    def _motion_stability(self, gripper_pos: Optional[np.ndarray]) -> Optional[float]:
        if gripper_pos is None or len(self._records) < 2:
            return None
        cur = np.asarray(gripper_pos, dtype=np.float32).reshape(-1)
        if cur.shape[0] != 3:
            return None
        p1 = self._records[-1].get("gripper_pos")
        p0 = self._records[-2].get("gripper_pos")
        if p1 is None or p0 is None:
            return None
        p1 = np.asarray(p1, dtype=np.float32).reshape(-1)
        p0 = np.asarray(p0, dtype=np.float32).reshape(-1)
        if p1.shape[0] != 3 or p0.shape[0] != 3:
            return None
        prev = p1 - p0
        now = cur - p1
        prev_norm = float(np.linalg.norm(prev))
        now_norm = float(np.linalg.norm(now))
        if prev_norm <= 1e-8 or now_norm <= 1e-8:
            return None
        return float(np.clip(np.dot(prev, now) / (prev_norm * now_norm), -1.0, 1.0))

    def prepare_step(
        self,
        *,
        scene_scores: Optional[np.ndarray],
        depth_scores: Optional[np.ndarray],
        contact_scores: Optional[np.ndarray],
        motion_scores: Optional[np.ndarray],
        action_scores: Optional[np.ndarray],
        valid_mask: Optional[np.ndarray],
        num_tokens: int,
        gripper_pos: Optional[np.ndarray] = None,
        depth_valid_ratio: Optional[float] = None,
    ) -> Dict[str, Any]:
        n = int(max(0, num_tokens))
        if n <= 0:
            return {"acgtp_history_enabled": True, "acgtp_history_disabled_reason": "no_tokens"}

        valid = np.ones(n, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool).reshape(-1)
        if valid.shape[0] != n:
            valid = np.ones(n, dtype=bool)

        depth_norm = _norm01(depth_scores, n, valid)
        depth_change = None
        if self._last_depth_norm is not None and self._last_depth_norm.shape[0] == n and np.any(valid):
            depth_change = float(np.mean(np.abs(depth_norm[valid] - self._last_depth_norm[valid])))

        motion_stability = self._motion_stability(gripper_pos)
        history_len = len(self._records)
        warmup = history_len < 1
        ema_available = bool(history_len > 0 and all(
            self._ema_scores.get(k) is not None for k in ("scene", "depth", "contact", "motion")
        ))

        reasons = []
        if warmup:
            reasons.append("warmup")
        if self._force_conservative_next:
            reasons.append(self._force_reason or "phase_or_mask_change")
        if depth_change is not None and depth_change >= self.depth_change_threshold:
            reasons.append(f"depth_change={depth_change:.3f}")
        if self._last_keep_iou is not None and self._last_keep_iou < self.keep_iou_threshold:
            reasons.append(f"keep_iou={self._last_keep_iou:.3f}")
        if motion_stability is not None and motion_stability < self.motion_stability_threshold:
            reasons.append(f"motion_stability={motion_stability:.3f}")

        conservative = bool(reasons)
        smoothing_applied = bool(ema_available and not conservative)

        raw = {
            "scene": _to_1d(scene_scores, n),
            "depth": _to_1d(depth_scores, n),
            "contact": _to_1d(contact_scores, n),
            "motion": _to_1d(motion_scores, n),
            "action": _to_1d(action_scores, n),
        }
        weights = {
            "scene": self.scene_current_weight,
            "depth": self.depth_current_weight,
            "contact": self.contact_current_weight,
            "motion": self.motion_current_weight,
            "action": self.action_current_weight,
        }

        out_scores: Dict[str, Optional[np.ndarray]] = {}
        for key, cur in raw.items():
            if cur is None:
                out_scores[key] = None
                continue
            if smoothing_applied:
                smoothed = _ema(self._ema_scores.get(key), cur, weights[key])
                out_scores[key] = smoothed
                self._ema_scores[key] = smoothed
            else:
                cur_arr = np.asarray(cur, dtype=np.float32).reshape(-1).copy()
                out_scores[key] = cur_arr
                self._ema_scores[key] = cur_arr

        self._last_depth_norm = depth_norm
        self._force_conservative_next = False
        self._force_reason = None

        meta = {
            "acgtp_history_enabled": True,
            "acgtp_history_length": history_len,
            "acgtp_history_capacity": self.maxlen,
            "acgtp_history_warmup": warmup,
            "acgtp_history_ema_available": ema_available,
            "acgtp_history_smoothing_applied": smoothing_applied,
            "acgtp_history_conservative_mode": conservative,
            "acgtp_history_conservative_reason": ";".join(reasons) if reasons else None,
            "acgtp_history_depth_change": depth_change,
            "acgtp_history_keep_mask_iou": self._last_keep_iou,
            "acgtp_history_motion_stability": motion_stability,
            "acgtp_history_depth_valid_ratio": depth_valid_ratio,
            "acgtp_history_scene_ema_alpha": self.scene_current_weight,
            "acgtp_history_depth_ema_alpha": self.depth_current_weight,
            "acgtp_history_contact_ema_alpha": self.contact_current_weight,
            "acgtp_history_motion_ema_alpha": self.motion_current_weight,
            "acgtp_history_action_ema_alpha": self.action_current_weight,
        }
        self._last_prepare_meta = dict(meta)

        return {
            **meta,
            "scene_scores": out_scores["scene"],
            "depth_scores": out_scores["depth"],
            "contact_scores": out_scores["contact"],
            "motion_scores": out_scores["motion"],
            "action_scores": out_scores["action"],
        }

    def update_after_selection(
        self,
        *,
        keep_indices: np.ndarray,
        num_tokens: int,
        dynamic_decision: Optional[Dict[str, Any]],
        gripper_pos: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        n = int(max(0, num_tokens))
        keep_mask = np.zeros(n, dtype=bool)
        idx = np.asarray(keep_indices, dtype=np.int64).reshape(-1)
        idx = idx[(idx >= 0) & (idx < n)]
        keep_mask[idx] = True

        keep_iou = None
        if self._last_keep_mask is not None and self._last_keep_mask.shape[0] == n:
            inter = int(np.sum(keep_mask & self._last_keep_mask))
            union = int(np.sum(keep_mask | self._last_keep_mask))
            keep_iou = float(inter) / float(max(1, union))
        self._last_keep_iou = keep_iou
        self._last_keep_mask = keep_mask

        dyn = dynamic_decision or {}
        phase = dyn.get("acgtp_dynamic_phase")
        previous_phase = dyn.get("acgtp_dynamic_previous_phase")
        hysteresis = dyn.get("acgtp_dynamic_hysteresis_state")
        phase_switch = bool(previous_phase is not None and phase is not None and str(phase) != str(previous_phase))

        force_next = False
        force_reason = None
        if phase_switch or hysteresis in ("hysteresis_switch", "risk_immediate_contact"):
            force_next = True
            force_reason = "phase_switch"
        elif keep_iou is not None and keep_iou < self.keep_iou_threshold:
            force_next = True
            force_reason = f"keep_iou={keep_iou:.3f}"

        self._force_conservative_next = force_next
        self._force_reason = force_reason

        grip = None
        if gripper_pos is not None:
            g = np.asarray(gripper_pos, dtype=np.float32).reshape(-1)
            if g.shape[0] == 3:
                grip = g.copy()

        self._records.append({
            "phase": phase,
            "risk": dyn.get("acgtp_dynamic_risk"),
            "confidence": dyn.get("acgtp_dynamic_confidence"),
            "keep_ratio": float(np.sum(keep_mask)) / float(n) if n else None,
            "keep_count": int(np.sum(keep_mask)),
            "keep_iou": keep_iou,
            "gripper_pos": grip,
        })

        return {
            "acgtp_history_length_after_update": len(self._records),
            "acgtp_history_keep_mask_iou": keep_iou,
            "acgtp_history_phase_switch": phase_switch,
            "acgtp_history_force_conservative_next": force_next,
            "acgtp_history_force_conservative_reason": force_reason,
        }

# ---------------------------------------------------------------------------
# Source: pruning/temporal/acgtp_dynamic_controller.py
# ---------------------------------------------------------------------------
import json
from typing import Any, Dict, Optional

import numpy as np


def _to_1d(arr: Optional[np.ndarray], n: Optional[int] = None) -> Optional[np.ndarray]:
    if arr is None:
        return None
    out = np.asarray(arr, dtype=np.float32).reshape(-1)
    if n is not None and out.shape[0] != n:
        return None
    return out


def _norm01(arr: Optional[np.ndarray], n: int, valid: np.ndarray) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    a = _to_1d(arr, n)
    if a is None:
        return out
    finite = valid & np.isfinite(a)
    if not np.any(finite):
        return out
    vals = a[finite]
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if hi - lo > 1e-8:
        out[finite] = (a[finite] - lo) / (hi - lo)
    else:
        out[finite] = np.clip(a[finite], 0.0, 1.0)
    out[~valid] = 0.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _peak(arr: np.ndarray, valid: np.ndarray) -> float:
    vals = arr[valid] if np.any(valid) else np.array([], dtype=np.float32)
    return float(np.max(vals)) if vals.size else 0.0


def _mean(arr: np.ndarray, valid: np.ndarray) -> float:
    vals = arr[valid] if np.any(valid) else np.array([], dtype=np.float32)
    return float(np.mean(vals)) if vals.size else 0.0


def _top_mask(arr: np.ndarray, valid: np.ndarray, quantile: float = 0.75) -> np.ndarray:
    if not np.any(valid):
        return np.zeros_like(valid, dtype=bool)
    vals = arr[valid]
    if vals.size == 0 or float(np.max(vals)) <= 1e-8:
        return np.zeros_like(valid, dtype=bool)
    thr = float(np.quantile(vals, quantile))
    return valid & (arr >= max(thr, 1e-6))


def _soft_iou(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> float:
    if not np.any(valid):
        return 0.0
    aa = np.clip(a[valid], 0.0, 1.0)
    bb = np.clip(b[valid], 0.0, 1.0)
    inter = float(np.sum(np.minimum(aa, bb)))
    union = float(np.sum(np.maximum(aa, bb)))
    return inter / max(union, 1e-8)


def _top_fraction_mask(scores: np.ndarray, valid: np.ndarray, ratio: float) -> np.ndarray:
    mask = np.zeros_like(valid, dtype=bool)
    if ratio <= 0.0 or not np.any(valid):
        return mask
    arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    candidates = np.where(valid & np.isfinite(arr) & (arr > 0.0))[0]
    if candidates.size == 0:
        return mask
    k = max(1, min(candidates.size, int(np.ceil(float(valid.size) * float(ratio)))))
    order = candidates[np.lexsort((candidates, -arr[candidates]))]
    mask[order[:k]] = True
    return mask


def _clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _phase_candidate(
    *,
    prev_phase: Optional[str],
    confidence: float,
    risk: float,
    alignment: float,
    contact_peak: float,
    contact_mean: float,
    contact_ratio: float,
    motion_peak: float,
    motion_ratio: float,
    scene_peak: float,
    depth_peak: float,
    motion_corridor_valid: bool,
    contact_phase_gate: str = "legacy_peak",
) -> str:
    coverage_high_contact = (
        contact_ratio >= 0.055
        or (contact_mean >= 0.08 and contact_ratio >= 0.020)
        or (contact_peak >= 0.85 and contact_ratio >= 0.025 and alignment >= 0.05)
    )
    legacy_high_contact = contact_peak >= 0.58 or contact_ratio >= 0.055
    hybrid_high_contact = (
        coverage_high_contact
        or (contact_mean >= 0.018 and contact_ratio >= 0.015 and alignment >= 0.12)
        or (contact_peak >= 0.85 and contact_ratio >= 0.025 and alignment >= 0.25)
    )
    gate = str(contact_phase_gate or "legacy_peak").strip().lower()
    if gate == "coverage":
        high_contact = coverage_high_contact
    elif gate == "hybrid":
        high_contact = hybrid_high_contact
    else:
        high_contact = legacy_high_contact
    high_motion = motion_corridor_valid and (motion_peak >= 0.35 or motion_ratio >= 0.05)
    strong_layout = scene_peak >= 0.35 or depth_peak >= 0.55

    if high_contact and not high_motion and strong_layout:
        return "place"
    if high_contact or risk >= 0.68:
        return "contact"
    if prev_phase in ("contact", "place") and high_motion and not high_contact:
        return "retreat"
    if high_motion and (alignment >= 0.08 or confidence >= 0.38):
        return "approach"
    return "search"


def _apply_hysteresis(candidate: str, state: Dict[str, Any], risk: float) -> Dict[str, Any]:
    previous = state.get("phase")
    pending = state.get("pending_phase")
    pending_count = int(state.get("pending_count") or 0)

    if previous is None:
        phase, pending, pending_count, reason = candidate, None, 0, "init"
    elif candidate == previous:
        phase, pending, pending_count, reason = previous, None, 0, "stable"
    elif candidate == "contact" and risk >= 0.62:
        phase, pending, pending_count, reason = candidate, None, 0, "risk_immediate_contact"
    else:
        if pending == candidate:
            pending_count += 1
        else:
            pending, pending_count = candidate, 1
        if pending_count >= 2:
            phase, pending, pending_count, reason = candidate, None, 0, "hysteresis_switch"
        else:
            phase, reason = previous, "hysteresis_hold"

    return {
        "phase": phase,
        "previous_phase": previous,
        "candidate_phase": candidate,
        "pending_phase": pending,
        "pending_count": pending_count,
        "hysteresis_state": reason,
    }


def _phase_presets(phase: str, schedule: str = "legacy") -> Dict[str, float]:
    legacy_presets = {
        "search": {"scene": 0.38, "depth": 0.34, "contact": 0.14, "motion": 0.14, "hard": 0.65, "min_keep": 0.80},
        "approach": {"scene": 0.28, "depth": 0.24, "contact": 0.24, "motion": 0.24, "hard": 0.62, "min_keep": 0.70},
        "contact": {"scene": 0.22, "depth": 0.28, "contact": 0.36, "motion": 0.14, "hard": 0.72, "min_keep": 0.86},
        "place": {"scene": 0.38, "depth": 0.30, "contact": 0.22, "motion": 0.10, "hard": 0.72, "min_keep": 0.86},
        "retreat": {"scene": 0.26, "depth": 0.38, "contact": 0.10, "motion": 0.26, "hard": 0.55, "min_keep": 0.64},
    }
    aggressive_presets = {
        "search": {
            "scene": 0.34, "depth": 0.36, "contact": 0.10, "motion": 0.20,
            "hard": 0.74, "min_keep": 0.55, "max_keep": 0.70,
            "floor_scene": 0.15, "floor_depth": 0.22, "floor_contact": 0.02, "floor_motion": 0.08,
            "fill_cap": 0.35,
        },
        "approach": {
            "scene": 0.24, "depth": 0.30, "contact": 0.16, "motion": 0.30,
            "hard": 0.75, "min_keep": 0.55, "max_keep": 0.72,
            "floor_scene": 0.12, "floor_depth": 0.22, "floor_contact": 0.04, "floor_motion": 0.14,
            "fill_cap": 0.32,
        },
        "contact": {
            "scene": 0.20, "depth": 0.34, "contact": 0.32, "motion": 0.14,
            "hard": 0.78, "min_keep": 0.70, "max_keep": 0.84,
            "floor_scene": 0.12, "floor_depth": 0.25, "floor_contact": 0.06, "floor_motion": 0.08,
            "fill_cap": 0.28,
        },
        "place": {
            "scene": 0.34, "depth": 0.32, "contact": 0.22, "motion": 0.12,
            "hard": 0.78, "min_keep": 0.70, "max_keep": 0.86,
            "floor_scene": 0.18, "floor_depth": 0.24, "floor_contact": 0.05, "floor_motion": 0.06,
            "fill_cap": 0.28,
        },
        "retreat": {
            "scene": 0.20, "depth": 0.44, "contact": 0.06, "motion": 0.30,
            "hard": 0.74, "min_keep": 0.45, "max_keep": 0.65,
            "floor_scene": 0.10, "floor_depth": 0.25, "floor_contact": 0.02, "floor_motion": 0.12,
            "fill_cap": 0.35,
        },
    }
    presets = aggressive_presets if str(schedule).strip().lower() == "aggressive" else legacy_presets
    return dict(presets.get(phase, presets["search"]))


def decide_acgtp_dynamic_budget(
    *,
    scene_layout_scores: Optional[np.ndarray],
    depth_structure_scores: Optional[np.ndarray],
    contact_ring_scores: Optional[np.ndarray],
    motion_corridor_scores: Optional[np.ndarray],
    action_constraint_scores: Optional[np.ndarray],
    valid_mask: Optional[np.ndarray],
    constrained_fill_mask: Optional[np.ndarray] = None,
    num_tokens: int,
    base_keep_ratio: float,
    previous_state: Optional[Dict[str, Any]] = None,
    motion_corridor_valid: bool = False,
    motion_norm_m: Optional[float] = None,
    depth_valid_ratio: Optional[float] = None,
    min_keep_ratio: float = 0.60,
    max_keep_ratio: float = 0.95,
    risk_boost_scale: float = 0.20,
    confidence_prune_scale: float = 0.16,
    contact_phase_gate: str = "legacy_peak",
    phase_schedule: str = "legacy",
    branch_floor_enabled: bool = False,
    fill_cap_ratio: float = 1.0,
    respect_phase_min_on_candidate_gap: bool = False,
    shadow_contact_guard_enabled: bool = False,
    shadow_contact_depth_weight_floor: float = 0.30,
    shadow_contact_contact_weight_floor: float = 0.24,
    shadow_contact_hard_ratio_floor: float = 0.70,
) -> Dict[str, Any]:
    """Return a phase/risk/confidence adaptive keep-ratio and branch weights."""
    n = int(max(0, num_tokens))
    if n <= 0:
        return {
            "acgtp_dynamic_enabled": True,
            "acgtp_dynamic_phase": "fallback_safe",
            "acgtp_dynamic_keep_ratio": 1.0,
            "acgtp_dynamic_keep_k": 0,
            "acgtp_dynamic_keep_reason": "no_tokens",
            "_state": previous_state or {},
        }

    valid = np.ones(n, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool).reshape(-1)
    if valid.shape[0] != n:
        valid = np.ones(n, dtype=bool)
    fill_arr = _to_1d(constrained_fill_mask, n)
    raw_fill_available = fill_arr is not None
    if fill_arr is None:
        raw_fill_candidate_mask = valid & (scene_layout_scores is not None)
    else:
        raw_fill_candidate_mask = valid & (fill_arr > 0.5)

    scene = _norm01(scene_layout_scores, n, valid)
    depth = _norm01(depth_structure_scores, n, valid)
    contact = _norm01(contact_ring_scores, n, valid)
    motion = _norm01(motion_corridor_scores, n, valid) if motion_corridor_valid else np.zeros(n, dtype=np.float32)
    action = _norm01(action_constraint_scores, n, valid)

    # Match selector.py's coverage-aware constrained fill capacity. This keeps
    # dynamic high-risk budgets from spilling into safe fallback while avoiding
    # an over-conservative clamp to raw scene-fill tokens only.
    depth_fill_candidate_mask = _top_fraction_mask(depth, valid, 0.75)
    contact_for_fill = contact.copy()
    if np.any(valid):
        gate_threshold = float(np.percentile(depth[valid], 60))
        contact_for_fill[depth < gate_threshold] = 0.0
    contact_fill_candidate_mask = _top_fraction_mask(contact_for_fill, valid, 0.50)
    motion_fill_candidate_mask = _top_fraction_mask(motion, valid, 0.50) if motion_corridor_valid else np.zeros(n, dtype=bool)
    scene_fill_candidate_mask = raw_fill_candidate_mask if raw_fill_available else (valid & (scene > 0.0))
    fill_candidate_mask = valid & (
        scene_fill_candidate_mask
        | depth_fill_candidate_mask
        | contact_fill_candidate_mask
        | motion_fill_candidate_mask
    )
    fill_candidate_count = int(np.sum(fill_candidate_mask))
    fill_candidate_ratio = float(fill_candidate_count) / float(n) if n > 0 else 0.0

    contact_peak, motion_peak = _peak(contact, valid), _peak(motion, valid)
    scene_peak, depth_peak, action_peak = _peak(scene, valid), _peak(depth, valid), _peak(action, valid)
    contact_mean, motion_mean, action_mean = _mean(contact, valid), _mean(motion, valid), _mean(action, valid)

    contact_mask = _top_mask(contact, valid, 0.75)
    motion_mask = _top_mask(motion, valid, 0.75)
    scene_mask = _top_mask(scene, valid, 0.70)
    depth_mask = _top_mask(depth, valid, 0.70)
    physical = np.maximum(contact, motion)
    layout = np.maximum(scene, depth)
    alignment = _soft_iou(layout, physical, valid)
    phys_mask = contact_mask | motion_mask
    layout_mask = scene_mask | depth_mask
    binary_alignment = 0.0
    if np.any(phys_mask | layout_mask):
        binary_alignment = float(np.sum(phys_mask & layout_mask)) / float(max(1, np.sum(phys_mask | layout_mask)))

    valid_depth = 1.0 if depth_valid_ratio is None else _clip(float(depth_valid_ratio), 0.0, 1.0)
    depth_uncertainty = 1.0 - valid_depth
    denom = float(max(1, np.sum(valid)))
    contact_ratio = float(np.sum((contact > 0.45) & valid)) / denom
    motion_ratio = float(np.sum((motion > 0.45) & valid)) / denom
    physical_ratio = float(np.sum((physical > 0.45) & valid)) / denom
    motion_active = 1.0 if motion_corridor_valid and (motion_peak > 0.0 or (motion_norm_m or 0.0) > 1e-4) else 0.0
    phase_alignment = max(alignment, binary_alignment)
    gate_mode = str(contact_phase_gate or "legacy_peak").strip().lower()
    if gate_mode not in ("legacy_peak", "coverage", "hybrid"):
        gate_mode = "legacy_peak"
    coverage_high_contact = (
        contact_ratio >= 0.055
        or (contact_mean >= 0.08 and contact_ratio >= 0.020)
        or (contact_peak >= 0.85 and contact_ratio >= 0.025 and phase_alignment >= 0.05)
    )
    legacy_high_contact = contact_peak >= 0.58 or contact_ratio >= 0.055
    hybrid_high_contact = (
        coverage_high_contact
        or (contact_mean >= 0.018 and contact_ratio >= 0.015 and phase_alignment >= 0.12)
        or (contact_peak >= 0.85 and contact_ratio >= 0.025 and phase_alignment >= 0.25)
    )
    if gate_mode == "coverage":
        high_contact = coverage_high_contact
    elif gate_mode == "hybrid":
        high_contact = hybrid_high_contact
    else:
        high_contact = legacy_high_contact
    shadow_contact_guard = bool(
        shadow_contact_guard_enabled
        and gate_mode == "coverage"
        and legacy_high_contact
        and not coverage_high_contact
    )
    high_motion = bool(motion_corridor_valid and (motion_peak >= 0.35 or motion_ratio >= 0.05))
    strong_layout = bool(scene_peak >= 0.35 or depth_peak >= 0.55)

    risk = _clip(
        0.42 * max(contact_peak, motion_peak)
        + 0.24 * max(contact_mean, motion_mean)
        + 0.18 * physical_ratio
        + 0.10 * motion_active
        + 0.06 * depth_uncertainty,
        0.0,
        1.0,
    )
    branch_agreement = 0.5 * alignment + 0.5 * binary_alignment
    confidence = _clip(
        0.46 * branch_agreement
        + 0.24 * action_peak
        + 0.16 * action_mean
        + 0.14 * max(scene_peak, depth_peak),
        0.0,
        1.0,
    )

    state = dict(previous_state or {})
    candidate = _phase_candidate(
        prev_phase=state.get("phase"),
        confidence=confidence,
        risk=risk,
        alignment=phase_alignment,
        contact_peak=contact_peak,
        contact_mean=contact_mean,
        contact_ratio=contact_ratio,
        motion_peak=motion_peak,
        motion_ratio=motion_ratio,
        scene_peak=scene_peak,
        depth_peak=depth_peak,
        motion_corridor_valid=motion_corridor_valid,
        contact_phase_gate=gate_mode,
    )
    h = _apply_hysteresis(candidate, state, risk)
    phase = h["phase"]
    new_state = {
        "phase": phase,
        "pending_phase": h["pending_phase"],
        "pending_count": h["pending_count"],
        "last_risk": risk,
        "last_confidence": confidence,
    }

    schedule = str(phase_schedule or "legacy").strip().lower()
    if schedule not in ("legacy", "aggressive"):
        schedule = "legacy"
    preset = _phase_presets(phase, schedule=schedule)
    risk_boost = float(risk_boost_scale) * risk
    prune_gain = float(confidence_prune_scale) * confidence
    uncertainty_boost = 0.0
    lock_strength = confidence
    raw_keep = float(base_keep_ratio) + risk_boost - prune_gain

    phase_min_keep = max(float(min_keep_ratio), float(preset["min_keep"]))
    phase_max_keep = min(float(max_keep_ratio), float(preset.get("max_keep", max_keep_ratio)))
    phase_max_keep = max(phase_min_keep, phase_max_keep)
    keep_ratio = _clip(raw_keep, phase_min_keep, phase_max_keep)
    keep_k = max(1, min(n, int(round(float(n) * keep_ratio))))
    candidate_clamped = False
    candidate_gap_count = max(0, keep_k - fill_candidate_count) if fill_candidate_count > 0 else 0
    candidate_gap_ratio = float(candidate_gap_count) / float(n) if n > 0 else 0.0
    if 0 < fill_candidate_count < keep_k:
        if not (bool(respect_phase_min_on_candidate_gap) and phase in ("contact", "place")):
            keep_k = max(1, fill_candidate_count)
            candidate_clamped = True
    keep_ratio = float(keep_k) / float(n)

    low_conf = max(0.0, 0.45 - confidence)
    w_scene = preset["scene"] + 0.10 * low_conf
    w_depth = preset["depth"] + 0.08 * low_conf + 0.04 * depth_uncertainty
    w_contact = preset["contact"] + 0.12 * risk + 0.04 * contact_peak
    w_motion = preset["motion"] + 0.10 * risk * motion_active + 0.04 * motion_peak
    if shadow_contact_guard:
        w_depth = max(w_depth, float(shadow_contact_depth_weight_floor))
        w_contact = max(w_contact, float(shadow_contact_contact_weight_floor))
    if not motion_corridor_valid:
        w_scene += 0.04
        w_depth += 0.04
        w_motion *= 0.25
    weight_sum = max(1e-8, w_scene + w_depth + w_contact + w_motion)
    w_scene, w_depth, w_contact, w_motion = [float(x / weight_sum) for x in (w_scene, w_depth, w_contact, w_motion)]

    hard_ratio = _clip(float(preset["hard"]) + 0.08 * risk - 0.04 * confidence, 0.45, 0.82)
    if shadow_contact_guard:
        hard_ratio = max(hard_ratio, _clip(float(shadow_contact_hard_ratio_floor), 0.45, 0.82))
    budget_vector = {
        "scene": w_scene,
        "depth": w_depth,
        "contact": w_contact,
        "motion": w_motion,
        "hard_protect_ratio": hard_ratio,
    }
    floor_scene = floor_depth = floor_contact = floor_motion = 0
    if bool(branch_floor_enabled):
        risk_floor_boost = 1.0 + 0.25 * risk
        floor_scene = int(round(keep_k * float(preset.get("floor_scene", 0.0))))
        floor_depth = int(round(keep_k * float(preset.get("floor_depth", 0.0)) * risk_floor_boost))
        floor_contact = int(round(keep_k * float(preset.get("floor_contact", 0.0)) * (1.0 + 0.50 * max(contact_peak, contact_ratio))))
        floor_motion = int(round(keep_k * float(preset.get("floor_motion", 0.0)) * (1.0 if motion_corridor_valid else 0.25)))
        if high_contact:
            floor_depth = max(floor_depth, int(round(keep_k * 0.24)))
            floor_contact = max(floor_contact, int(round(keep_k * 0.05)))
        if phase in ("place", "contact"):
            floor_scene = max(floor_scene, int(round(keep_k * 0.12)))
        floor_scene = max(0, min(keep_k, floor_scene))
        floor_depth = max(0, min(keep_k, floor_depth))
        floor_contact = max(0, min(keep_k, floor_contact))
        floor_motion = max(0, min(keep_k, floor_motion))

    effective_fill_cap_ratio = float(fill_cap_ratio)
    if schedule == "aggressive":
        effective_fill_cap_ratio = min(effective_fill_cap_ratio, float(preset.get("fill_cap", 1.0)))
    effective_fill_cap_ratio = _clip(effective_fill_cap_ratio, 0.0, 1.0)
    fill_cap_tokens = int(round(keep_k * effective_fill_cap_ratio)) if effective_fill_cap_ratio < 1.0 else keep_k
    budget_vector.update({
        "min_scene_tokens": floor_scene,
        "min_depth_tokens": floor_depth,
        "min_contact_tokens": floor_contact,
        "min_motion_tokens": floor_motion,
        "fill_cap_tokens": fill_cap_tokens,
    })
    reason = f"phase={phase};risk={risk:.3f};confidence={confidence:.3f};hysteresis={h['hysteresis_state']}"
    if candidate_clamped:
        reason += f";candidate_clamped={fill_candidate_count}/{n}"

    return {
        "acgtp_dynamic_enabled": True,
        "acgtp_dynamic_phase": phase,
        "acgtp_dynamic_phase_schedule": schedule,
        "acgtp_dynamic_candidate_phase": candidate,
        "acgtp_dynamic_previous_phase": h["previous_phase"],
        "acgtp_dynamic_hysteresis_state": h["hysteresis_state"],
        "acgtp_dynamic_risk": risk,
        "acgtp_dynamic_confidence": confidence,
        "acgtp_dynamic_keep_ratio": keep_ratio,
        "acgtp_dynamic_keep_k": keep_k,
        "acgtp_dynamic_base_keep_ratio": float(base_keep_ratio),
        "acgtp_dynamic_raw_keep_ratio": raw_keep,
        "acgtp_dynamic_phase_min_keep_ratio": phase_min_keep,
        "acgtp_dynamic_phase_max_keep_ratio": phase_max_keep,
        "acgtp_dynamic_lock_strength": lock_strength,
        "acgtp_dynamic_uncertainty_boost": uncertainty_boost,
        "acgtp_dynamic_risk_boost": risk_boost,
        "acgtp_dynamic_prune_gain": prune_gain,
        "acgtp_dynamic_keep_reason": reason,
        "acgtp_dynamic_layout_motion_alignment": float(alignment),
        "acgtp_dynamic_binary_alignment": float(binary_alignment),
        "acgtp_dynamic_contact_phase_gate": gate_mode,
        "acgtp_dynamic_contact_peak": contact_peak,
        "acgtp_dynamic_contact_mean": contact_mean,
        "acgtp_dynamic_contact_ratio": contact_ratio,
        "acgtp_dynamic_motion_peak": motion_peak,
        "acgtp_dynamic_motion_mean": motion_mean,
        "acgtp_dynamic_motion_ratio": motion_ratio,
        "acgtp_dynamic_physical_ratio": physical_ratio,
        "acgtp_dynamic_high_contact": bool(high_contact),
        "acgtp_dynamic_high_contact_coverage": bool(coverage_high_contact),
        "acgtp_dynamic_high_contact_legacy": bool(legacy_high_contact),
        "acgtp_dynamic_shadow_contact_guard": bool(shadow_contact_guard),
        "acgtp_dynamic_high_motion": bool(high_motion),
        "acgtp_dynamic_strong_layout": bool(strong_layout),
        "acgtp_dynamic_action_peak": action_peak,
        "acgtp_dynamic_action_mean": action_mean,
        "acgtp_dynamic_depth_valid_ratio": valid_depth,
        "acgtp_dynamic_fill_candidate_count": fill_candidate_count,
        "acgtp_dynamic_fill_candidate_ratio": fill_candidate_ratio,
        "acgtp_dynamic_candidate_gap_count": candidate_gap_count,
        "acgtp_dynamic_candidate_gap_ratio": candidate_gap_ratio,
        "acgtp_dynamic_candidate_clamped": candidate_clamped,
        "acgtp_dynamic_scene_weight": w_scene,
        "acgtp_dynamic_depth_weight": w_depth,
        "acgtp_dynamic_contact_weight": w_contact,
        "acgtp_dynamic_motion_weight": w_motion,
        "acgtp_dynamic_hard_protect_ratio": hard_ratio,
        "acgtp_dynamic_branch_floor_enabled": bool(branch_floor_enabled),
        "acgtp_dynamic_min_scene_tokens": floor_scene,
        "acgtp_dynamic_min_depth_tokens": floor_depth,
        "acgtp_dynamic_min_contact_tokens": floor_contact,
        "acgtp_dynamic_min_motion_tokens": floor_motion,
        "acgtp_dynamic_fill_cap_ratio": effective_fill_cap_ratio,
        "acgtp_dynamic_fill_cap_tokens": fill_cap_tokens,
        "acgtp_dynamic_budget_vector": json.dumps(budget_vector, sort_keys=True),
        "_state": new_state,
    }

# ---------------------------------------------------------------------------
# Source: pruning/temporal/scheduler.py
# ---------------------------------------------------------------------------
from typing import Any, Dict, Tuple

import numpy as np


def compute_dynamic_keep_ratio(
    score_dict: Dict[str, Any],
    p_robot: np.ndarray,
    gripper_pos: np.ndarray,
    valid_mask: np.ndarray,
    cfg: Any,
) -> Tuple[float, Dict[str, Any]]:
    """Choose keep ratio from robot-centric geometry risk.

    This function is pure scheduling logic. It does not touch model tensors.
    """
    valid = np.asarray(valid_mask, dtype=np.bool_).reshape(-1)
    num_tokens = int(valid.shape[0])
    valid_depth_ratio = float(score_dict.get("valid_depth_ratio", np.mean(valid) if num_tokens else 0.0))

    if valid_depth_ratio < float(getattr(cfg, "min_valid_depth_ratio", getattr(cfg, "min_valid_token_ratio", 0.1))):
        return _phase_result("fallback_safe", float(cfg.keep_ratio_near), num_tokens, cfg, valid_depth_ratio)

    if p_robot is None or gripper_pos is None or not np.any(valid):
        return _phase_result("fallback_safe", float(cfg.keep_ratio_near), num_tokens, cfg, valid_depth_ratio)

    points = np.asarray(p_robot, dtype=np.float32).reshape(num_tokens, 3)
    grip = np.asarray(gripper_pos, dtype=np.float32).reshape(3)
    if not np.all(np.isfinite(grip)):
        return _phase_result("fallback_safe", float(cfg.keep_ratio_near), num_tokens, cfg, valid_depth_ratio)

    distances = np.linalg.norm(points - grip[None, :], axis=1)
    distances = np.where(np.isfinite(distances), distances, np.inf).astype(np.float32)
    valid_dist = distances[valid & np.isfinite(distances)]
    if valid_dist.size == 0:
        return _phase_result("fallback_safe", float(cfg.keep_ratio_near), num_tokens, cfg, valid_depth_ratio)

    scores = np.nan_to_num(np.asarray(score_dict.get("scores"), dtype=np.float32).reshape(num_tokens), nan=0.0)
    edge = np.nan_to_num(np.asarray(score_dict.get("edge_scores"), dtype=np.float32).reshape(num_tokens), nan=0.0)
    corridor = np.nan_to_num(np.asarray(score_dict.get("corridor_scores"), dtype=np.float32).reshape(num_tokens), nan=0.0)

    top_count = max(1, int(round(num_tokens * 0.10)))
    valid_indices = np.where(valid)[0]
    score_order = valid_indices[np.argsort(-scores[valid_indices], kind="mergesort")]
    score_top = score_order[:top_count]
    d_topk_mean = float(np.mean(distances[score_top])) if score_top.size else None
    corridor_strength = float(np.mean(corridor[score_top])) if score_top.size else 0.0

    edge_valid = np.maximum(edge[valid], 0.0)
    edge_sum = float(np.sum(edge_valid))
    if edge_sum > 1e-8:
        edge_top = np.sort(edge_valid)[-top_count:]
        edge_concentration = float(np.sum(edge_top) / edge_sum)
    else:
        edge_concentration = 0.0

    d_min = float(np.min(valid_dist))
    motion_norm = float(score_dict.get("motion_norm") or 0.0)
    if d_min < float(cfg.near_threshold) or corridor_strength > float(cfg.high_corridor_threshold):
        phase = "near"
        keep_ratio = float(cfg.keep_ratio_near)
    elif d_min < float(cfg.mid_threshold) or motion_norm > float(cfg.min_motion_norm):
        phase = "mid"
        keep_ratio = float(cfg.keep_ratio_mid)
    else:
        phase = "far"
        keep_ratio = float(cfg.keep_ratio_far)

    keep_k = _ratio_to_k(keep_ratio, num_tokens)
    reserve_k = max(16, int(round(keep_k * float(cfg.dynamic_reserve_ratio))))
    reserve_k = min(reserve_k, keep_k)
    stats = {
        "dynamic_enabled": True,
        "dynamic_phase": phase,
        "dynamic_keep_ratio": keep_k / num_tokens if num_tokens else keep_ratio,
        "dynamic_keep_k": keep_k,
        "dynamic_reserve_k": reserve_k,
        "dynamic_score_k": keep_k - reserve_k,
        "d_min": d_min,
        "d_topk_mean": d_topk_mean,
        "corridor_strength": corridor_strength,
        "edge_concentration": edge_concentration,
        "valid_depth_ratio": valid_depth_ratio,
        "motion_norm": motion_norm,
    }
    return stats["dynamic_keep_ratio"], stats


def _phase_result(phase: str, keep_ratio: float, num_tokens: int, cfg: Any, valid_depth_ratio: float) -> Tuple[float, Dict[str, Any]]:
    keep_k = _ratio_to_k(keep_ratio, num_tokens)
    reserve_k = max(16, int(round(keep_k * float(cfg.dynamic_reserve_ratio)))) if keep_k else 0
    reserve_k = min(reserve_k, keep_k)
    actual_ratio = keep_k / num_tokens if num_tokens else keep_ratio
    return actual_ratio, {
        "dynamic_enabled": True,
        "dynamic_phase": phase,
        "dynamic_keep_ratio": actual_ratio,
        "dynamic_keep_k": keep_k,
        "dynamic_reserve_k": reserve_k,
        "dynamic_score_k": keep_k - reserve_k,
        "d_min": None,
        "d_topk_mean": None,
        "corridor_strength": None,
        "edge_concentration": None,
        "valid_depth_ratio": valid_depth_ratio,
        "motion_norm": None,
    }


def _ratio_to_k(keep_ratio: float, num_tokens: int) -> int:
    if num_tokens <= 0:
        return 0
    ratio = float(keep_ratio)
    if abs(ratio - 0.50) < 1e-6 and num_tokens == 256:
        return 128
    if abs(ratio - 0.75) < 1e-6 and num_tokens == 256:
        return 192
    if abs(ratio - 0.875) < 1e-6 and num_tokens == 256:
        return 224
    return int(max(1, min(num_tokens, round(num_tokens * ratio))))

# ---------------------------------------------------------------------------
# Source: pruning/geometry/temporal_geometry.py
# ---------------------------------------------------------------------------
from collections import deque
from typing import Any, Deque, Dict, Optional, Sequence

import numpy as np


class GeometryHistoryBuffer:
    """Bounded history for temporal geometry stability and interaction lock."""

    def __init__(self, maxlen: int = 5) -> None:
        self.maxlen = max(1, int(maxlen))
        self._frames: Deque[Dict[str, Any]] = deque(maxlen=self.maxlen)

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def history_length(self) -> int:
        return len(self._frames)

    def reset(self) -> None:
        self._frames.clear()

    def update(
        self,
        *,
        robot_state: Optional[Any] = None,
        motion_direction: Optional[Any] = None,
        final_scores: Optional[Any] = None,
        keep_mask: Optional[Any] = None,
        contact_risk_score: Optional[Any] = None,
        valid_3d_ratio: Optional[float] = None,
        dynamic_keep_ratio: Optional[float] = None,
        step_index: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Append one past frame to history."""
        frame = {
            "robot_state": robot_state,
            "motion_direction": _as_vector(motion_direction, dim=3),
            "final_scores": _as_float_array(final_scores),
            "keep_mask": _as_bool_array(keep_mask),
            "contact_risk_score": _as_float_array(contact_risk_score),
            "valid_3d_ratio": _as_optional_float(valid_3d_ratio),
            "dynamic_keep_ratio": _as_optional_float(dynamic_keep_ratio),
            "step_index": step_index,
            "timestamp": timestamp,
        }
        self._frames.append(frame)

    def compute_score_ema(
        self,
        *,
        current_scores: Optional[Any] = None,
        alpha: float = 0.6,
        score_key: str = "final_scores",
    ) -> Dict[str, Any]:
        """Compute an exponential moving average over stored scores plus current."""
        alpha = float(np.clip(alpha, 0.0, 1.0))
        arrays = [_as_float_array(frame.get(score_key)) for frame in self._frames]
        if current_scores is not None:
            arrays.append(_as_float_array(current_scores))
        arrays = [arr for arr in arrays if arr is not None and arr.size > 0]
        if not arrays:
            return {"score_ema": None, "score_ema_enabled": False}

        ema = arrays[0].astype(np.float32, copy=True)
        for arr in arrays[1:]:
            if arr.shape != ema.shape:
                n = min(int(arr.size), int(ema.size))
                next_ema = ema.copy()
                next_ema[:n] = alpha * arr[:n] + (1.0 - alpha) * ema[:n]
                ema = next_ema
            else:
                ema = alpha * arr + (1.0 - alpha) * ema
        ema = np.nan_to_num(ema, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return {"score_ema": ema, "score_ema_enabled": True}

    def compute_temporal_stability(
        self,
        *,
        final_scores: Optional[Any] = None,
        keep_mask: Optional[Any] = None,
        motion_direction: Optional[Any] = None,
        contact_risk_score: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Estimate temporal stability from prior frames and the current frame."""
        if not self._frames:
            return {
                "temporal_stability": None,
                "score_stability": None,
                "keep_mask_iou": None,
                "motion_stability": None,
                "contact_risk_stability": None,
                "history_length": 0,
            }

        prev = self._frames[-1]
        score_stability = _cosine_similarity(_as_float_array(final_scores), prev.get("final_scores"))
        keep_iou = _mask_iou(_as_bool_array(keep_mask), prev.get("keep_mask"))
        motion_stability = _cosine_similarity(_as_vector(motion_direction, dim=3), prev.get("motion_direction"))
        current_contact = _risk_summary(_as_float_array(contact_risk_score))
        prev_contact = _risk_summary(prev.get("contact_risk_score"))
        contact_stability = None
        if current_contact is not None and prev_contact is not None:
            denom = max(abs(current_contact), abs(prev_contact), 1e-6)
            contact_stability = float(max(0.0, 1.0 - abs(current_contact - prev_contact) / denom))

        values = [
            value
            for value in (score_stability, keep_iou, motion_stability, contact_stability)
            if value is not None
        ]
        temporal_stability = float(np.mean(values)) if values else None
        return {
            "temporal_stability": temporal_stability,
            "score_stability": score_stability,
            "keep_mask_iou": keep_iou,
            "motion_stability": motion_stability,
            "contact_risk_stability": contact_stability,
            "history_length": len(self._frames),
        }

    def detect_interaction_lock(
        self,
        *,
        contact_risk_score: Optional[Any] = None,
        final_scores: Optional[Any] = None,
        keep_mask: Optional[Any] = None,
        motion_direction: Optional[Any] = None,
        valid_3d_ratio: Optional[float] = None,
        dynamic_keep_ratio: Optional[float] = None,
        config: Optional[Any] = None,
        gripper_pos: Optional[Any] = None,
        token_points_robot: Optional[Any] = None,
        adaptive_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Detect whether the current step should use conservative pruning.

        Conservative v2 rules:
        1. Top-k contact risk mean >= adaptive threshold (default: max(0.08, percentile(85%)))
        2. Gripper proximity to high-risk tokens
        3. Stable high-risk token region
        4. Lock triggers keep_ratio >= 0.90
        5. Episode reset clears history
        """
        # Use adaptive threshold if provided, otherwise fall back to config
        if adaptive_threshold is not None and adaptive_threshold > 0:
            threshold = float(adaptive_threshold)
        else:
            threshold = float(_cfg_value(config, "temporal_contact_risk_threshold", 0.3))

        min_frames = max(1, int(_cfg_value(config, "temporal_lock_min_frames", 2)))
        stability_threshold = float(_cfg_value(config, "temporal_stability_threshold", 0.5))
        motion_threshold = float(_cfg_value(config, "temporal_motion_cos_threshold", 0.7))
        ema_alpha = float(_cfg_value(config, "temporal_ema_alpha", 0.6))
        gripper_lock_dist = float(_cfg_value(config, "temporal_gripper_lock_dist", 0.15))

        current_contact = _risk_summary(_as_float_array(contact_risk_score))
        risk_values = [
            _risk_summary(frame.get("contact_risk_score"))
            for frame in self._frames
        ]
        if current_contact is not None:
            risk_values.append(current_contact)
        risk_values = [value for value in risk_values if value is not None]

        # === Improved adaptive threshold: fallback for low-variance contact risk ===
        # Problem: if all contact_risk values are ~0.04 and threshold=0.08, topk_contact_lock never fires
        # Fix: when p85-p50 is small (<0.02), values are consistently low - use p50-based threshold instead
        current_contact_vals = _as_float_array(contact_risk_score)
        if current_contact_vals is not None and current_contact_vals.size > 0 and risk_values:
            all_risk_vals = list(risk_values) + list(current_contact_vals.flatten())
            p85 = float(np.percentile(all_risk_vals, 85))
            p50 = float(np.percentile(all_risk_vals, 50))
            adaptive_min = max(0.08, p85)
            # If low variance (p85 close to p50), use relative threshold so lock fires when risk is elevated
            if p85 - p50 < 0.02:
                threshold = max(0.05, p50 * 1.5)
            else:
                threshold = max(adaptive_min, threshold)
            # Also check: if even the max value is below threshold, use max-based threshold
            max_risk = float(np.max(all_risk_vals))
            if max_risk < threshold and max_risk > 0:
                threshold = max(0.05, max_risk * 0.85)

        valid_ratio = _as_optional_float(valid_3d_ratio)

        # --- Rule 1: Top-k contact risk mean (with adaptive threshold fallback) ---
        topk_contact_lock = False
        if risk_values:
            recent_risk = risk_values[max(0, len(risk_values) - min_frames):]
            if len(recent_risk) >= 2 and sum(1 for v in recent_risk if v >= threshold) >= 2:
                topk_contact_lock = True

        # --- Rule 1b: Elevated current contact risk above median ---
        # This triggers when current risk is above p50, even if not above p85 threshold
        elevated_current_lock = False
        if current_contact is not None and len(risk_values) >= 2:
            p50_local = float(np.median(risk_values))
            # If current is significantly above median AND gripper is moving/stationary
            if current_contact > p50_local * 1.3 and current_contact >= 0.03:
                elevated_current_lock = True

        # --- Rule 2: Gripper proximity to high-risk tokens ---
        gripper_lock = False
        if gripper_pos is not None and token_points_robot is not None and current_contact is not None:
            gripper_lock = self._check_gripper_lock(
                gripper_pos, token_points_robot, _as_float_array(contact_risk_score),
                threshold=threshold, max_dist=gripper_lock_dist,
            )

        # --- Rule 3: Stable high-risk region ---
        region_lock = False
        if len(self._frames) >= 2:
            region_lock = self._check_region_stability(keep_mask, threshold)

        # Determine lock_reason: comma-separated list of which sub-conditions triggered
        lock_reasons = []
        if topk_contact_lock:
            lock_reasons.append("contact_risk")
        if elevated_current_lock:
            lock_reasons.append("elevated_current")
        if gripper_lock:
            lock_reasons.append("gripper_proximity")
        if region_lock:
            lock_reasons.append("region_stability")
        lock_reason = ",".join(lock_reasons) if lock_reasons else "none"

        # Conservative lock: any of the rules fires
        interaction_lock = topk_contact_lock or elevated_current_lock or gripper_lock or region_lock

        # Compute remaining conditions for logging
        stability = self.compute_temporal_stability(
            final_scores=final_scores,
            keep_mask=keep_mask,
            motion_direction=motion_direction,
            contact_risk_score=contact_risk_score,
        )
        temporal_stability = stability.get("temporal_stability")
        motion_stability = stability.get("motion_stability")
        motion_ok = motion_stability is None or motion_stability >= motion_threshold
        stable_ok = temporal_stability is None or temporal_stability >= stability_threshold
        enough_history = len(risk_values) >= 2

        if not interaction_lock:
            if not enough_history:
                reason = "insufficient_history"
            elif not _as_optional_float(valid_3d_ratio) or (_as_optional_float(valid_3d_ratio) <= 0):
                reason = "invalid_3d_ratio"
            elif not topk_contact_lock and not elevated_current_lock:
                reason = "contact_risk_below_threshold"
            elif not gripper_lock and not region_lock and not elevated_current_lock:
                reason = "no_stable_high_risk_region"
            else:
                reason = "no_lock_trigger"
        elif not enough_history:
            reason = "insufficient_history_for_lock"
        else:
            reason = "lock_triggered"

        ema = self.compute_score_ema(current_scores=final_scores, alpha=ema_alpha)
        return {
            "interaction_lock": interaction_lock,
            "temporal_stability": temporal_stability,
            "history_length": len(self._frames),
            "score_ema_enabled": bool(ema.get("score_ema_enabled")),
            "score_ema": ema.get("score_ema"),
            "topk_contact_lock": topk_contact_lock,
            "elevated_current_lock": elevated_current_lock,
            "gripper_lock": gripper_lock,
            "region_lock": region_lock,
            "lock_reason": lock_reason,
            "current_contact_risk": current_contact,
            "contact_risk_threshold": threshold,
            "adaptive_threshold_used": adaptive_threshold is not None and adaptive_threshold > 0,
            "adaptive_threshold_value": adaptive_threshold if (adaptive_threshold is not None and adaptive_threshold > 0) else None,
            "motion_stability": motion_stability,
            "keep_mask_iou": stability.get("keep_mask_iou"),
            "dynamic_keep_ratio": _as_optional_float(dynamic_keep_ratio),
            "reason": reason,
        }

    def _check_gripper_lock(
        self,
        gripper_pos: Any,
        token_points_robot: Any,
        contact_risk_score: Optional[np.ndarray],
        threshold: float = 0.3,
        max_dist: float = 0.15,
    ) -> bool:
        """Check if gripper is close to high-risk tokens."""
        try:
            grip = np.asarray(gripper_pos, dtype=np.float32).reshape(-1)
            if grip.size < 3 or not np.all(np.isfinite(grip[:3])):
                return False
            if token_points_robot is None:
                return False
            pts = np.asarray(token_points_robot, dtype=np.float32).reshape(-1, 3)
            if pts.shape[0] == 0 or not np.all(np.isfinite(pts)):
                return False
            dists = np.linalg.norm(pts - grip[:3][None, :], axis=1)
            dists[~np.isfinite(dists)] = np.inf
            if contact_risk_score is not None and contact_risk_score.size == pts.shape[0]:
                high_risk = contact_risk_score >= threshold
                if not np.any(high_risk):
                    return False
                return float(np.min(dists[high_risk])) < max_dist
            return float(np.min(dists)) < max_dist
        except Exception:
            return False

    def _check_region_stability(self, keep_mask: Optional[Any], threshold: float = 0.3) -> bool:
        """Check if high-risk token region is stable across consecutive frames."""
        if keep_mask is None or len(self._frames) < 2:
            return False
        current_mask = _as_bool_array(keep_mask)
        if current_mask is None:
            return False
        prev = self._frames[-1]
        prev_mask = _as_bool_array(prev.get("keep_mask"))
        if prev_mask is None:
            return False
        n = min(len(current_mask), len(prev_mask))
        if n == 0:
            return False
        intersection = float(np.sum(current_mask[:n] & prev_mask[:n]))
        union = float(np.sum(current_mask[:n] | prev_mask[:n]))
        if union <= 0:
            return False
        iou = intersection / union
        return iou >= 0.6


def _as_float_array(value: Optional[Any]) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _as_bool_array(value: Optional[Any]) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.bool_).reshape(-1)
    return arr if arr.size else None


def _as_vector(value: Optional[Any], dim: int) -> Optional[np.ndarray]:
    arr = _as_float_array(value)
    if arr is None or arr.size < dim:
        return None
    vec = arr[:dim].astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return None
    return vec / norm


def _as_optional_float(value: Optional[Any]) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _risk_summary(scores: Optional[np.ndarray]) -> Optional[float]:
    if scores is None or scores.size == 0:
        return None
    valid = scores[np.isfinite(scores)]
    if valid.size == 0:
        return None
    top_k = max(1, int(round(0.1 * valid.size)))
    top_vals = np.partition(valid, -top_k)[-top_k:]
    return float(np.mean(top_vals))


def _cosine_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    n = min(int(a.size), int(b.size))
    if n == 0:
        return None
    aa = np.asarray(a[:n], dtype=np.float32)
    bb = np.asarray(b[:n], dtype=np.float32)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1e-8:
        return None
    return float(np.clip(np.dot(aa, bb) / denom, -1.0, 1.0))


def _mask_iou(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    n = min(int(a.size), int(b.size))
    if n == 0:
        return None
    aa = np.asarray(a[:n], dtype=np.bool_)
    bb = np.asarray(b[:n], dtype=np.bool_)
    union = int(np.logical_or(aa, bb).sum())
    if union == 0:
        return None
    return float(np.logical_and(aa, bb).sum() / union)


def _cfg_value(config: Optional[Any], key: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)
