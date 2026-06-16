"""Internal quota configuration helpers."""

from __future__ import annotations

from typing import Any, Dict


def build_internal_quota_config(
    config: Any,
    *,
    hard_ratio: float,
    w_scene: float,
    w_depth: float,
    w_contact: float,
    w_motion: float,
) -> Dict[str, Any]:
    """Build the small config payload consumed by the internal pruning backend."""

    mode = str(getattr(config, "acgtp_internal_selection_mode", "geo_guarded") or "geo_guarded").strip().lower()
    return {
        "selection_mode": mode,
        "attention_enabled": bool(getattr(config, "acgtp_internal_attention_enabled", True)),
        "latency_mode": bool(getattr(config, "latency_mode", False)),
        "latency_fast_path": bool(getattr(config, "acgtp_internal_latency_fast_path", False))
        or bool(getattr(config, "latency_mode", False)),
        "semantic_attention_ratio": float(
            getattr(config, "acgtp_internal_attention_budget_ratio", getattr(config, "acgtp_attention_budget_ratio", 0.12))
        ),
        "historical_attention_ratio": float(getattr(config, "acgtp_internal_history_budget_ratio", 0.15)),
        "attention_requires_geometry_alignment": bool(getattr(config, "acgtp_attention_requires_geometry_alignment", True)),
        "hard_protect_ratio": float(hard_ratio),
        "functional_quota_enabled": bool(getattr(config, "acgtp_internal_functional_quota_enabled", True)),
        "layout_quota_ratio": float(getattr(config, "acgtp_internal_layout_quota_ratio", 0.30)),
        "contact_quota_ratio": float(getattr(config, "acgtp_internal_contact_quota_ratio", 0.20)),
        "motion_quota_ratio": float(getattr(config, "acgtp_internal_motion_quota_ratio", 0.15)),
        "semantic_quota_ratio": float(getattr(config, "acgtp_internal_semantic_quota_ratio", 0.12)),
        "action_quota_ratio": float(getattr(config, "acgtp_internal_action_quota_ratio", 0.08)),
        "fill_quota_ratio": float(getattr(config, "acgtp_internal_fill_quota_ratio", 0.15)),
        "w_scene": float(w_scene),
        "w_depth": float(w_depth),
        "w_contact": float(w_contact),
        "w_motion": float(w_motion),
        "history_length": int(getattr(config, "acgtp_attention_history_length", 3)),
        "risk_adaptive_enabled": bool(getattr(config, "acgtp_internal_risk_adaptive_enabled", False)),
        "high_risk_keep_ratio": float(getattr(config, "acgtp_internal_high_risk_keep_ratio", 0.85)),
        "medium_risk_keep_ratio": float(getattr(config, "acgtp_internal_medium_risk_keep_ratio", 0.55)),
        "low_risk_keep_ratio": float(getattr(config, "acgtp_internal_low_risk_keep_ratio", 0.40)),
        "risk_coverage_weight": float(getattr(config, "acgtp_internal_risk_coverage_weight", 3.0)),
        "risk_mean_weight": float(getattr(config, "acgtp_internal_risk_mean_weight", 1.5)),
        "risk_peak_weight": float(getattr(config, "acgtp_internal_risk_peak_weight", 0.15)),
        "risk_physical_weight": float(getattr(config, "acgtp_internal_risk_physical_weight", 0.85)),
        "risk_depth_weight": float(getattr(config, "acgtp_internal_risk_depth_weight", 0.15)),
        "risk_disagreement_gate": float(getattr(config, "acgtp_internal_risk_disagreement_gate", 0.45)),
        "risk_disagreement_max_bonus": float(getattr(config, "acgtp_internal_risk_disagreement_max_bonus", 0.10)),
        "risk_high_threshold": float(getattr(config, "acgtp_internal_risk_high_threshold", 0.65)),
        "risk_medium_threshold": float(getattr(config, "acgtp_internal_risk_medium_threshold", 0.35)),
        "capture_decode_attention": bool(getattr(config, "acgtp_internal_capture_decode_attention", False)),
        "trace_enabled": bool(getattr(config, "acgtp_internal_trace_enabled", True)),
    }
