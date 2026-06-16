# ACGTP Optimization Fix Summary — 2026-05-29

## Overview

This document summarizes the latest ACGTP optimization/fix pass performed under `/infini-data/openvla`.

The work follows the convergence direction described in the project docs:

```text
Robot-Centric Geometry-Guarded Internal Pruning for VLA Inference
```

The main goal of this pass was to fix misleading evaluation/reporting semantics, add runtime-mode plumbing, improve internal compute probes, and start narrowing the formal strategy surface away from legacy/projector-heavy variants.

No OpenVLA model weights were modified. No training was performed. No git commit was made.

## Key Problems Addressed

### 1. Internal pruning retention was misleading

Before this pass, internal pruning reports could make `internal_acgtp_dynamic@0.50` look like it retained only `128/256 = 50%` visual tokens, because reports could use hook-level requested retention.

However, the actual internal backend widened dynamic retention to about:

```text
218 / 256 = 85.16%
```

This made speed conclusions misleading. The method was not slow because 50% internal pruning failed; it was slow because dynamic retention was actually much higher.

### 2. Projector retention and internal retention needed separation

Internal mode should keep full projector output and prune inside the LLM. Therefore:

```text
projector_retention = 1.0 in internal mode
internal_retention = actual internal hidden visual tokens kept / original visual tokens
```

Reports now distinguish:

- `projector_retention`
- `internal_retention`
- `effective_retention`
- `effective_visual_tokens_for_llm`

### 3. Runtime mode needed a formal path

The code already had pieces of fast-path logic, but this pass formalized the direction around:

```text
acgtp_runtime_mode = fast | debug | audit
```

This is intended to separate lightweight inference-time behavior from expensive diagnostic/audit behavior.

### 4. Internal compute probe needed better sweep support

The compute probe was extended so future checks can sweep internal pruning layer and target visual keep count:

```text
internal_prune_layer = 1, 2, 3
visual_keep = 64, 96, 128
```

This is important because ACGTP scoring should not be tuned further until internal pruning has a proven speed-positive operating point.

### 5. Main strategy surface needed cleanup

The codebase contains many historical strategies. This pass added report-level classification so formal comparison can focus on the intended mainline while keeping old variants available for audit/probe use.

## Files Modified

### Metrics / reporting semantics

- `pruning/metrics.py`
- `utils/metrics_logger.py`
- `scripts/eval_openvla_baseline.py`
- `scripts/build_performance_report.py`
- `scripts/probe_pruning_compute_reality.py`

### Runtime mode / fast path plumbing

- `pruning/config.py`
- `pruning/hook.py`
- `pruning/metrics.py`

### Internal compute probe support

- `scripts/probe_pruning_compute_reality.py`
- `pruning/internal_uniform_pruning.py`

### Strategy surface cleanup

- `pruning/strategy_registry.py`
- `scripts/build_performance_report.py`

## Detailed Changes

## P0 — Backend-Aware Retention Reporting

### `pruning/metrics.py`

Added/propagated:

```text
effective_visual_tokens_for_llm
```

Ensured `HookMetrics.to_eval_stats()` and `HookMetrics.to_fast_eval_stats()` derive backend-aware retention consistently:

```text
if compression_backend == internal:
    effective_retention = internal_retention
    effective_keep_count = internal_kept_visual_tokens
    original_token_count = internal_original_visual_tokens
    effective_visual_tokens_for_llm = internal_kept_visual_tokens
else:
    effective_retention = projector/generic retention
```

### `scripts/eval_openvla_baseline.py`

The final `step_stats` merge now enforces backend-aware semantics:

```text
projector_retention:
    projector output kept/original
    forced to 1.0 in internal mode

internal_retention:
    internal_kept_visual_tokens / internal_original_visual_tokens

effective_retention:
    internal_retention if backend is internal
    otherwise projector/generic retention

effective_visual_tokens_for_llm:
    internal_kept_visual_tokens if backend is internal
    otherwise projector kept tokens
```

This prevents internal pruning from being reported as projector pruning.

### `utils/metrics_logger.py`

Added step/episode summary support for:

- `effective_visual_tokens_for_llm`
- `effective_retention`
- `projector_retention`
- `internal_retention`
- `mean_internal_original_visual_tokens`
- `mean_internal_kept_visual_tokens`
- `mean_internal_pruned_visual_tokens`
- `mean_internal_original_seq_length`
- `mean_internal_kept_seq_length`
- `mean_internal_pruned_seq_length`
- `compression_backend`
- `compression_backend_distribution`

Also adjusted `token_retention_ratio` summary to prefer backend-aware `effective_retention` when available.

### `scripts/build_performance_report.py`

The report builder now includes:

- `effective_visual_tokens_for_llm`
- internal visual kept/original
- internal sequence kept/original
- projector retention
- internal retention
- retention warnings when internal backend fields are missing

A new markdown section is produced:

```text
Internal Sequence / Retention Semantics
```

This section makes it clear whether a method prunes before the LLM or inside the LLM.

### Verification result

Regenerated report:

```text
/infini-data/openvla/outputs/acgtp_small_compare_20260529_150822/performance_report_fixed_all.json
/infini-data/openvla/outputs/acgtp_small_compare_20260529_150822/performance_report_fixed_all.md
/infini-data/openvla/outputs/acgtp_small_compare_20260529_150822/performance_report_fixed_all.csv
```

Observed key fields:

```text
internal_acgtp_dynamic_050       effective_retention=0.8515625  effective_visual_tokens_for_llm=218
internal_acgtp_geo_guarded_050   effective_retention=0.5        effective_visual_tokens_for_llm=128
internal_acgtp_geometry_only_050 effective_retention=0.5        effective_visual_tokens_for_llm=128
projector_acgtp_legacy_050       effective_retention=0.5        effective_visual_tokens_for_llm=128
baseline_none                    effective_retention=1.0        effective_visual_tokens_for_llm=256
```

## P1 — Runtime Mode Plumbing

### `pruning/config.py`

Added:

```python
acgtp_runtime_mode: str = "fast"  # "fast" | "debug" | "audit"
```

Normalization behavior:

```text
fast/debug/audit accepted
invalid value falls back to fast
acgtp_full_diagnostics_enabled=True maps to audit behavior
```

### `pruning/hook.py`

Runtime helpers are present:

```python
def _runtime_mode(self) -> str
def _is_fast_runtime(self) -> bool
def _is_audit_runtime(self) -> bool
```

`_acgtp_fast_runtime_enabled()` now checks runtime mode before entering the lean hot path.

The projector hook also records:

```text
acgtp_runtime_mode
```

in latest stats.

### `pruning/metrics.py`

`to_fast_eval_stats()` now carries the same retention semantics as full stats, including:

```text
effective_visual_tokens_for_llm
```

## P2 — Internal Compute Probe Extension

### `scripts/probe_pruning_compute_reality.py`

Added CLI flag:

```bash
--internal_sweep
```

When combined with internal modes, this adds sweep cases:

```text
internal_acgtp_geometry_only_L1_K64
internal_acgtp_geometry_only_L1_K96
internal_acgtp_geometry_only_L1_K128
internal_acgtp_geometry_only_L2_K64
...
internal_acgtp_geometry_only_L3_K128

internal_uniform_L1_K64
internal_uniform_L1_K96
internal_uniform_L1_K128
...
internal_uniform_L3_K128
```

The markdown report now includes:

```text
Internal sweep speed summary
```

with:

- backend
- prune layer
- visual retention
- CUDA speedup
- wall speedup
- hook ms
- selector ms

### `pruning/internal_uniform_pruning.py`

Uniform internal pruning now records visual-token counts:

- `original_visual_tokens`
- `kept_visual_tokens`
- `pruned_visual_tokens`

This aligns internal uniform with ACGTP internal reporting.

## P5 — Strategy Surface Cleanup

### `pruning/strategy_registry.py`

Added report-level mainline method labels:

```python
MAIN_EXPERIMENT_METHOD_LABELS
```

Added:

```python
AUDIT_ONLY_STRATEGIES
is_audit_only_strategy()
is_main_experiment_method_label()
```

This does not delete legacy strategies and does not introduce new pruning strategy names. It only marks what should be considered part of the formal comparison surface.

### `scripts/build_performance_report.py`

Reports now include:

- `main_experiment_surface`
- `audit_probe_only`

If audit/probe-only methods are present, the markdown report explicitly notes them.

## Verification Performed

### Syntax checks

The following syntax check passed:

```bash
PYTHONPATH=/infini-data/openvla python -m py_compile \
  pruning/metrics.py \
  utils/metrics_logger.py \
  scripts/eval_openvla_baseline.py \
  scripts/build_performance_report.py \
  scripts/probe_pruning_compute_reality.py \
  pruning/hook.py \
  pruning/selector.py \
  pruning/internal_pruning.py \
  pruning/config.py \
  pruning/internal_uniform_pruning.py \
  pruning/strategy_registry.py
```

### Report regeneration

The existing small comparison report was regenerated successfully:

```bash
python scripts/build_performance_report.py \
  --root /infini-data/openvla/outputs/acgtp_small_compare_20260529_150822 \
  --prefix performance_report_fixed_all
```

Generated files:

```text
/infini-data/openvla/outputs/acgtp_small_compare_20260529_150822/performance_report_fixed_all.json
/infini-data/openvla/outputs/acgtp_small_compare_20260529_150822/performance_report_fixed_all.md
/infini-data/openvla/outputs/acgtp_small_compare_20260529_150822/performance_report_fixed_all.csv
```

## Current Interpretation After Fixes

The corrected report makes the central issue clearer:

```text
internal_dynamic@0.50 is not actually a 50% retention method.
It keeps about 218/256 visual tokens, or 85.16%.
```

Therefore, the current dynamic path is slow mainly because it is too conservative, not because 50% internal pruning necessarily cannot work.

The next major question is whether internal pruning itself has a speed-positive operating point. That should be answered with the new sweep probe before further ACGTP scoring changes.

## Recommended Next Probe

Run a short bounded compute probe:

```bash
/infini-data/miniconda3/envs/openvla/bin/python scripts/probe_pruning_compute_reality.py \
  --iters 3 \
  --warmup 1 \
  --task task_0 \
  --seed 7 \
  --include_internal_acgtp \
  --include_internal_uniform \
  --internal_sweep
```

This should determine whether:

1. internal uniform pruning can produce real speedup at layer 1/2/3 and 64/96/128 visual tokens;
2. ACGTP internal geometry-only has similar speed behavior;
3. the bottleneck is internal backend overhead, hook overhead, selector overhead, or dynamic retention.

## Remaining Follow-Up Work

The following work remains recommended:

1. Run the internal sweep probe and identify a speed-positive internal setting.
2. If internal uniform does not speed up, optimize internal backend overhead first.
3. If internal uniform speeds up but ACGTP does not, continue reducing hook/selector overhead.
4. Recalibrate internal dynamic risk so low geometry-attention IoU alone does not force high retention.
5. Strengthen `P_geo` hard-protection in `pruning/internal_pruning.py` so protected geometry tokens cannot be removed by final truncation or fill logic.
6. Keep projector pruning only as a legacy baseline, not the final method.
