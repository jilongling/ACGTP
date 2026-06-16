"""Named experiment profiles for pruning runs.

These labels are run configurations, not ``pruning_strategy`` names. Keeping
them here prevents runners and reports from quietly inventing new strategy
surfaces such as ``functional_quota_static_050``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class MethodProfile:
    label: str
    args: Tuple[str, ...]
    family: str
    description: str = ""

    def cli_args(self) -> List[str]:
        return list(self.args)


BASELINE_NONE = (
    "--pruning_strategy", "none",
    "--pruning_enabled", "false",
    "--geometry_enabled", "false",
    "--keep_ratio", "1.0",
)

def _acgtp_v2_internal_static(keep_ratio: str) -> Tuple[str, ...]:
    return (
        "--pruning_strategy", "robot_geo_acgtp_v2",
        "--pruning_enabled", "true",
        "--geometry_enabled", "true",
        "--keep_ratio", keep_ratio,
        "--acgtp_compression_backend", "internal",
        "--acgtp_internal_pruning_enabled", "true",
        "--acgtp_internal_selection_mode", "geo_guarded",
        "--acgtp_dynamic_enabled", "false",
    )


ACGTP_V2_INTERNAL_STATIC_050 = _acgtp_v2_internal_static("0.50")

ACGTP_INTERNAL_LATENCY_ACCEL = (
    "--acgtp_internal_latency_fast_path", "true",
    "--acgtp_latency_plan_cache_enabled", "true",
)


METHOD_PROFILES: Dict[str, MethodProfile] = {
    "baseline_none": MethodProfile(
        "baseline_none",
        BASELINE_NONE,
        family="baseline",
        description="No pruning; full 256 visual tokens.",
    ),
    "projector_acgtp_legacy_050": MethodProfile(
        "projector_acgtp_legacy_050",
        (
            "--pruning_strategy", "robot_geo_acgtp_v2",
            "--pruning_enabled", "true",
            "--geometry_enabled", "true",
            "--keep_ratio", "0.50",
            "--acgtp_compression_backend", "projector",
            "--acgtp_dynamic_enabled", "false",
        ),
        family="acgtp_projector",
        description="Projector-level ACGTP v2 comparison profile.",
    ),
    "internal_geo_guarded_050": MethodProfile(
        "internal_geo_guarded_050",
        ACGTP_V2_INTERNAL_STATIC_050,
        family="acgtp_internal",
        description="Internal ACGTP v2 geo-guarded profile at 50 percent visual retention.",
    ),
    "internal_dynamic_050": MethodProfile(
        "internal_dynamic_050",
        ACGTP_V2_INTERNAL_STATIC_050
        + (
            "--acgtp_internal_selection_mode", "dynamic",
            "--acgtp_internal_risk_adaptive_enabled", "true",
        ),
        family="acgtp_internal",
        description="Internal ACGTP v2 profile with risk-adaptive keep ratio.",
    ),
    "functional_quota_static_050": MethodProfile(
        "functional_quota_static_050",
        ACGTP_V2_INTERNAL_STATIC_050
        + ("--acgtp_internal_functional_quota_enabled", "true")
        + ACGTP_INTERNAL_LATENCY_ACCEL,
        family="acgtp_internal_functional_quota",
        description="Static 50 percent internal ACGTP v2 functional-quota allocation.",
    ),
    "internal_layer0_functional_quota_050": MethodProfile(
        "internal_layer0_functional_quota_050",
        ACGTP_V2_INTERNAL_STATIC_050
        + (
            "--acgtp_internal_prune_layer", "0",
            "--acgtp_internal_functional_quota_enabled", "true",
        )
        + ACGTP_INTERNAL_LATENCY_ACCEL,
        family="acgtp_internal_functional_quota_probe",
        description="Layer-0 internal functional-quota latency probe at 50 percent visual retention.",
    ),
    "functional_quota_static_033": MethodProfile(
        "functional_quota_static_033",
        _acgtp_v2_internal_static("0.33")
        + ("--acgtp_internal_functional_quota_enabled", "true")
        + ACGTP_INTERNAL_LATENCY_ACCEL,
        family="acgtp_internal_functional_quota",
        description="Static 33 percent internal ACGTP v2 functional-quota allocation.",
    ),
    "functional_quota_static_025": MethodProfile(
        "functional_quota_static_025",
        _acgtp_v2_internal_static("0.25")
        + ("--acgtp_internal_functional_quota_enabled", "true")
        + ACGTP_INTERNAL_LATENCY_ACCEL,
        family="acgtp_internal_functional_quota",
        description="Static 25 percent internal ACGTP v2 functional-quota allocation.",
    ),
    "functional_attention_off_050": MethodProfile(
        "functional_attention_off_050",
        ACGTP_V2_INTERNAL_STATIC_050
        + (
            "--acgtp_internal_functional_quota_enabled", "true",
            "--acgtp_internal_attention_enabled", "false",
        ),
        family="acgtp_internal_functional_quota",
        description="Functional quota with internal attention disabled.",
    ),
    "internal_geometry_only_050": MethodProfile(
        "internal_geometry_only_050",
        ACGTP_V2_INTERNAL_STATIC_050
        + (
            "--acgtp_internal_selection_mode", "geometry_only",
            "--acgtp_internal_attention_enabled", "false",
            "--acgtp_internal_functional_quota_enabled", "false",
        ),
        family="acgtp_internal",
        description="Internal geometry-only diagnostic profile.",
    ),
    "functional_no_layout_050": MethodProfile(
        "functional_no_layout_050",
        ACGTP_V2_INTERNAL_STATIC_050
        + (
            "--acgtp_internal_functional_quota_enabled", "true",
            "--acgtp_internal_layout_quota_ratio", "0.0",
        ),
        family="acgtp_internal_functional_quota_ablation",
        description="Functional quota with layout quota removed.",
    ),
    "functional_no_contact_050": MethodProfile(
        "functional_no_contact_050",
        ACGTP_V2_INTERNAL_STATIC_050
        + (
            "--acgtp_internal_functional_quota_enabled", "true",
            "--acgtp_internal_contact_quota_ratio", "0.0",
        ),
        family="acgtp_internal_functional_quota_ablation",
        description="Functional quota with contact quota removed.",
    ),
    "functional_no_motion_050": MethodProfile(
        "functional_no_motion_050",
        ACGTP_V2_INTERNAL_STATIC_050
        + (
            "--acgtp_internal_functional_quota_enabled", "true",
            "--acgtp_internal_motion_quota_ratio", "0.0",
        ),
        family="acgtp_internal_functional_quota_ablation",
        description="Functional quota with motion quota removed.",
    ),
    "functional_no_semantic_050": MethodProfile(
        "functional_no_semantic_050",
        ACGTP_V2_INTERNAL_STATIC_050
        + (
            "--acgtp_internal_functional_quota_enabled", "true",
            "--acgtp_internal_semantic_quota_ratio", "0.0",
        ),
        family="acgtp_internal_functional_quota_ablation",
        description="Functional quota with semantic quota removed.",
    ),
    "functional_no_fill_050": MethodProfile(
        "functional_no_fill_050",
        ACGTP_V2_INTERNAL_STATIC_050
        + (
            "--acgtp_internal_functional_quota_enabled", "true",
            "--acgtp_internal_fill_quota_ratio", "0.0",
        ),
        family="acgtp_internal_functional_quota_ablation",
        description="Functional quota with fill quota removed.",
    ),
    "legacy_geo_guarded_quota_050": MethodProfile(
        "legacy_geo_guarded_quota_050",
        ACGTP_V2_INTERNAL_STATIC_050
        + ("--acgtp_internal_functional_quota_enabled", "false"),
        family="acgtp_internal_legacy_comparison",
        description="Old geo-guarded internal quota comparison profile.",
    ),
}


CORE_SURFACE_METHOD_LABELS: Tuple[str, ...] = (
    "baseline_none",
    "projector_acgtp_legacy_050",
    "internal_geo_guarded_050",
    "internal_dynamic_050",
)

FUNCTIONAL_QUOTA_METHOD_LABELS: Tuple[str, ...] = (
    "baseline_none",
    "functional_quota_static_050",
    "internal_layer0_functional_quota_050",
    "functional_quota_static_033",
    "functional_quota_static_025",
    "functional_attention_off_050",
    "internal_geometry_only_050",
    "functional_no_layout_050",
    "functional_no_contact_050",
    "functional_no_motion_050",
    "functional_no_semantic_050",
    "functional_no_fill_050",
    "legacy_geo_guarded_quota_050",
)


def method_cli_args(label: str) -> List[str]:
    try:
        return METHOD_PROFILES[label].cli_args()
    except KeyError as exc:
        raise KeyError(f"Unknown method profile {label!r}. Available: {', '.join(method_profile_labels())}") from exc


def method_profile_labels(families: Iterable[str] | None = None) -> List[str]:
    if families is None:
        return sorted(METHOD_PROFILES)
    allowed = set(families)
    return sorted(label for label, profile in METHOD_PROFILES.items() if profile.family in allowed)
