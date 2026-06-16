#!/usr/bin/env python3
"""Audit geometry metrics semantics in summary.json / rollup_summary.json files.

Checks:
1. Baseline geometry fields should be null.
2. depth_conversion / depth_is_metric consistency.
3. transform_convention_evidence should not be null when verified=True.
4. motion_direction_valid=false steps should have motion_cone_nonzero_ratio ≈ 0.
5. Motion invalid but cone nonzero indicates a gating bug.
6. workspace_all_one counts should be consistent.
7. Success/failure split null handling.
8. Retention ratio consistency.
9. Rollup null vs 0 aggregation correctness.
10. No old-style legacy field names.

Usage:
    python scripts/audit_metrics_semantics.py --summary path/to/summary.json
    python scripts/audit_metrics_semantics.py --all path/to/experiment_dir
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        print(f"ERROR: File not found: {path}")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def load_step_records(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    records = []
    if path.suffix == ".csv":
        with open(path) as f:
            reader = csv.DictReader(f)
            records = list(reader)
    elif path.suffix == ".jsonl":
        with open(path) as f:
            for line in f:
                records.append(json.loads(line))
    return records


def check_baseline_geometry_null(summary: Dict, method: str) -> List[str]:
    """Check that baseline runs have null geometry/depth/transform fields."""
    issues = []
    is_baseline = (
        summary.get("geometry_enabled") is False
        or summary.get("pruning_enabled") is False
        or "baseline" in str(method).lower()
    )
    if not is_baseline:
        return issues

    null_checks = {
        "depth_conversion": summary.get("depth_conversion"),
        "depth_is_metric": summary.get("depth_is_metric"),
        "depth_metric_mean": summary.get("depth_metric_mean"),
        "transform_convention": summary.get("transform_convention"),
        "transform_inverse_used": summary.get("transform_inverse_used"),
        "transform_convention_verified": summary.get("transform_convention_verified"),
        "transform_convention_evidence": summary.get("transform_convention_evidence"),
    }
    for field, value in null_checks.items():
        if value is not None and value != "" and value != "None":
            issues.append(f"BASELINE_NOT_NULL: {field}={value!r} (expected null)")
    return issues


def check_depth_conversion_consistency(summary: Dict) -> List[str]:
    """Check depth conversion / depth_is_metric consistency."""
    issues = []
    dim = summary.get("depth_is_metric")
    dc = summary.get("depth_conversion")
    unit = summary.get("depth_unit")
    dmm = summary.get("depth_metric_mean")

    if dim is True:
        if dc is None or dc == "":
            issues.append("WARN: depth_is_metric=True but depth_conversion is null")
        if unit is not None and unit != "meters":
            issues.append(f"WARN: depth_is_metric=True but depth_unit={unit!r} (expected 'meters')")

    if dim is False:
        if dc is None or dc == "":
            issues.append("WARN: depth_is_metric=False but depth_conversion is null")

    if dc == "raw_no_sim_fallback":
        if dim is True:
            issues.append("ERROR: depth_conversion='raw_no_sim_fallback' but depth_is_metric=True (inconsistent)")
        issues.append("WARN: depth_conversion='raw_no_sim_fallback' — sim not available")

    if dc == "robosuite_get_real_depth_map":
        if dim is not True:
            issues.append("ERROR: depth_conversion='robosuite_get_real_depth_map' but depth_is_metric!=True")
        if dmm is not None:
            if not (0.5 <= float(dmm) <= 5.0):
                issues.append(f"WARN: depth_metric_mean={dmm:.4f}m outside typical range [0.5, 5.0]m")

    if dmm is not None and 0.95 <= float(dmm) <= 1.05 and dc != "robosuite_get_real_depth_map":
        if dim is True:
            issues.append(
                f"ERROR: depth_metric_mean={dmm:.4f}m (raw z-buffer range) "
                f"but depth_is_metric=True — likely not real metric depth"
            )

    return issues


def check_transform_metadata(summary: Dict) -> List[str]:
    """Check transform convention metadata consistency."""
    issues = []
    tc = summary.get("transform_convention")
    tcv = summary.get("transform_convention_verified")
    tiu = summary.get("transform_inverse_used")
    tce = summary.get("transform_convention_evidence")
    ts = summary.get("transform_source")

    if tcv is True:
        if tc is None or tc == "":
            issues.append("ERROR: transform_convention_verified=True but transform_convention is null")
        if tce is None or tce == "":
            issues.append("ERROR: transform_convention_verified=True but transform_convention_evidence is null/empty")
        if tiu is not False:
            issues.append(f"WARN: transform_convention_verified=True but transform_inverse_used={tiu!r} (expected False)")
    elif tcv is False:
        if tc is not None and tc != "":
            issues.append(f"WARN: transform_convention_verified=False but transform_convention={tc!r} (not verified)")

    if tc is not None and tc != "" and tc != "T_robot_cam_forward":
        issues.append(f"WARN: transform_convention={tc!r} — verify this is expected")

    return issues


def check_step_motion_semantics(step_records: List[Dict]) -> List[str]:
    """Check motion cone gating semantics in step records."""
    issues = []
    eps = 1e-6
    invalid_steps = 0
    invalid_but_cone_nonzero = 0
    cone_nonzero_when_invalid = []

    for r in step_records:
        mdv = r.get("motion_direction_valid")
        mcnr = r.get("motion_cone_nonzero_ratio")
        mcsm = r.get("motion_cone_score_mean")

        if mdv is None:
            continue

        if mdv is False:
            invalid_steps += 1
            cone_nonzero = (mcnr is not None and float(mcnr) > eps) or (mcsm is not None and float(mcsm) > eps)
            if cone_nonzero:
                invalid_but_cone_nonzero += 1
                cone_nonzero_when_invalid.append({
                    "step": r.get("step_id"),
                    "motion_cone_nonzero_ratio": mcnr,
                    "motion_cone_score_mean": mcsm,
                })

    if invalid_but_cone_nonzero > 0:
        issues.append(
            f"ERROR: {invalid_but_cone_nonzero} steps have motion_direction_valid=False "
            f"but motion_cone_nonzero_ratio > {eps} — gating may be broken"
        )
        for item in cone_nonzero_when_invalid[:5]:
            issues.append(
                f"  step={item['step']}: "
                f"motion_cone_nonzero_ratio={item['motion_cone_nonzero_ratio']}, "
                f"motion_cone_score_mean={item['motion_cone_score_mean']}"
            )

    return issues


def check_workspace_all_one(step_records: List[Dict]) -> List[str]:
    """Check workspace scoring diagnostics."""
    issues = []
    total = 0
    all_one = 0
    for r in step_records:
        ws = r.get("workspace_score_mean")
        if ws is not None and ws != "":
            total += 1
            try:
                if abs(float(ws) - 1.0) < 1e-6:
                    all_one += 1
            except (ValueError, TypeError):
                pass
    if total > 0 and all_one == total:
        issues.append(f"WARN: ALL {total} steps have workspace_score_mean=1.0 — workspace may be non-informative")
    return issues


def check_retention_ratio(summary: Dict) -> List[str]:
    """Check retention ratio consistency."""
    issues = []
    kept = summary.get("num_visual_tokens_kept")
    orig = summary.get("num_visual_tokens_original")
    rr = summary.get("token_retention_ratio")
    arr = summary.get("actual_keep_ratio")

    if kept is not None and orig is not None:
        try:
            expected = int(kept) / int(orig)
            if rr is not None:
                try:
                    if abs(float(rr) - expected) > 0.001:
                        issues.append(f"ERROR: token_retention_ratio={rr:.4f} != kept/original={expected:.4f}")
                except (ValueError, TypeError):
                    pass
            if arr is not None:
                try:
                    if abs(float(arr) - expected) > 0.001:
                        issues.append(f"ERROR: actual_keep_ratio={arr:.4f} != kept/original={expected:.4f}")
                except (ValueError, TypeError):
                    pass
        except (ValueError, TypeError, ZeroDivisionError):
            pass
    return issues


def check_success_failure_null(summary: Dict) -> List[str]:
    """Check that empty success/failure subsets are null.
    
    Logic:
    - success_only_* should be null when success_count == 0.
    - failure_only_* should be null when failure_count == 0.
    - success_only_* can have values when success_count > 0.
    - failure_only_* can have values when failure_count > 0.
    """
    issues = []
    sr = summary.get("overall_success_rate")
    success_count = summary.get("num_successes")
    failure_count = summary.get("num_failures")
    
    if sr is None and success_count is None and failure_count is None:
        return issues
    
    for field in ["mean_episode_steps_success_only", "std_episode_steps_success_only"]:
        val = summary.get(field)
        if val is None:
            continue
        # success_only fields should be null when success_count == 0
        if success_count is not None and success_count == 0:
            issues.append(f"WARN: success_count=0 but {field}={val} (should be null when no successes)")
    
    for field in ["mean_episode_steps_failure_only", "std_episode_steps_failure_only"]:
        val = summary.get(field)
        if val is None:
            continue
        # failure_only fields should be null when failure_count == 0
        if failure_count is not None and failure_count == 0:
            issues.append(f"WARN: failure_count=0 but {field}={val} (should be null when no failures)")
    
    return issues


def check_rollup_null_aggregation(rollup: Dict) -> List[str]:
    """Check rollup doesn't incorrectly aggregate null as 0."""
    issues = []
    for field in ["depth_suspicious_ratio", "T_ambiguous_ratio", "motion_invalid_but_cone_nonzero_ratio"]:
        val = rollup.get(field)
        if val is not None:
            try:
                if not (0.0 <= float(val) <= 1.0):
                    issues.append(f"ERROR: rollup.{field}={val} outside [0, 1] range")
            except (ValueError, TypeError):
                issues.append(f"ERROR: rollup.{field}={val!r} is not numeric")
    return issues


def check_legacy_field_names(summary: Dict) -> List[str]:
    """Check for old-style legacy field names."""
    issues = []
    for field in ["depth_key", "depth_scale_mode", "T_convention_used"]:
        val = summary.get(field)
        if val is not None and val != "":
            issues.append(f"LEGACY_FIELD: {field}={val!r} — use modern field names")
    return issues


def check_no_inverse_in_forward_convention(summary: Dict) -> List[str]:
    """Check that T_robot_cam_forward convention means no inverse was used."""
    issues = []
    tc = summary.get("transform_convention")
    tiu = summary.get("transform_inverse_used")
    if tc == "T_robot_cam_forward" and tiu is True:
        issues.append("ERROR: transform_convention='T_robot_cam_forward' but transform_inverse_used=True")
    return issues


def audit_summary(
    summary: Dict,
    method: str,
    step_records: Optional[List[Dict]] = None,
) -> List[str]:
    """Run all audit checks on a single summary."""
    all_issues = []
    all_issues += check_baseline_geometry_null(summary, method)
    all_issues += check_depth_conversion_consistency(summary)
    all_issues += check_transform_metadata(summary)
    all_issues += check_retention_ratio(summary)
    all_issues += check_success_failure_null(summary)
    all_issues += check_legacy_field_names(summary)
    all_issues += check_no_inverse_in_forward_convention(summary)
    if step_records:
        all_issues += check_step_motion_semantics(step_records)
        all_issues += check_workspace_all_one(step_records)
    return all_issues


def audit_rollup(rollup: Dict, method: str) -> List[str]:
    """Run audit checks on a single rollup row."""
    all_issues = []
    all_issues += check_rollup_null_aggregation(rollup)
    all_issues += check_depth_conversion_consistency(rollup)
    all_issues += check_transform_metadata(rollup)
    all_issues += check_baseline_geometry_null(rollup, method)
    return all_issues


def print_issues(issues: List[str], title: str) -> None:
    """Print issues with severity."""
    if not issues:
        print(f"  {title}: PASS")
        return
    print(f"  {title}: {len(issues)} issue(s)")
    for issue in issues:
        prefix = "ERROR" if issue.startswith("ERROR") else "WARN"
        print(f"    [{prefix}] {issue}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit geometry metrics semantics")
    parser.add_argument("--summary", type=str, help="Path to summary.json")
    parser.add_argument("--rollup", type=str, help="Path to rollup_summary.json")
    parser.add_argument("--all", type=str, help="Path to experiment directory (auto-finds summaries)")
    parser.add_argument("--steps", type=str, help="Path to step_metrics.csv (for motion checks)")
    args = parser.parse_args()

    all_passed = True

    step_records = load_step_records(args.steps) if args.steps else None
    if step_records:
        print(f"Loaded {len(step_records)} step records from {args.steps}")

    if args.summary:
        summary = load_json(args.summary)
        method = summary.get("method", Path(args.summary).parent.name)
        issues = audit_summary(summary, method, step_records)
        print(f"\n{'='*60}")
        print(f"Audit: {args.summary}")
        print(f"Method: {method}")
        print(f"{'='*60}")
        print_issues(issues, "Overall")
        if issues:
            all_passed = False

    if args.rollup:
        rollup_data = load_json(args.rollup)
        if isinstance(rollup_data, dict):
            rows = rollup_data.get("results", [rollup_data])
        elif isinstance(rollup_data, list):
            rows = rollup_data
        else:
            print(f"ERROR: Unknown rollup format")
            sys.exit(1)

        print(f"\n{'='*60}")
        print(f"Audit: {args.rollup} ({len(rows)} rows)")
        print(f"{'='*60}")
        for row in rows:
            method = row.get("method", "unknown")
            issues = audit_rollup(row, method)
            print(f"\n  Method: {method}")
            print_issues(issues, "Overall")
            if issues:
                all_passed = False

    if args.all:
        root = Path(args.all)
        print(f"\n{'='*60}")
        print(f"Audit all in: {root}")
        print(f"{'='*60}")
        for summary_path in sorted(root.rglob("summary.json")):
            try:
                summary = load_json(summary_path)
                method = summary.get("method", summary_path.parent.name)
                issues = audit_summary(summary, method)
                print(f"\n  {summary_path.parent.name}/summary.json")
                print_issues(issues, "Overall")
                if issues:
                    all_passed = False
            except Exception as e:
                print(f"\n  {summary_path}: ERROR reading: {e}")

    if all_passed:
        print("\nAll audit checks passed.")
        sys.exit(0)
    else:
        print("\nSome audit checks failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
