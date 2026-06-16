# ACGTP Code Optimization Strategy

This document defines the code-level convergence path for ACGTP after the small
LIBERO-Spatial comparison in
`/infini-data/openvla/outputs/acgtp_small_compare_20260529_150822/`.

The goal is not to add more surface strategies. The goal is to converge toward
one stable method that preserves manipulation success while producing real
latency reduction.

## Current Evidence

Small comparison setup: LIBERO-Spatial first 3 tasks, 2 episodes each,
`max_steps=220`, seed 7.

| Strategy | Success | Effective Visual Keep | CUDA | Wall | LLM | Hook | Main Finding |
|---|---:|---:|---:|---:|---:|---:|---|
| `baseline_none@1.00` | 5/6 | 256/256 | 238.87 ms | 255.55 ms | 213.64 ms | N/A | Baseline. |
| `projector_acgtp_legacy@0.50` | 4/6 | 128/256 before LLM | 229.30 ms | 246.76 ms | 199.51 ms | 4.55 ms | Slight speedup, unsafe task drop. |
| `internal_acgtp_geometry_only@0.50` | 4/6 | 128/256 after layer 2 | 238.33 ms | 258.66 ms | 209.12 ms | 4.56 ms | Internal path works but speedup is consumed by overhead. |
| `internal_acgtp_geo_guarded@0.50` | 4/6 | 128/256 after layer 2 | 239.78 ms | 260.54 ms | 210.67 ms | 4.53 ms | True attention active, but no net speedup. |
| `internal_acgtp_dynamic@0.50` | 5/6 | 218/256 after layer 2 | 244.14 ms | 264.45 ms | 214.42 ms | 4.59 ms | Success preserved by widening to 0.85, therefore slow. |

The current bottleneck is twofold:

1. `K=2` internal pruning at 128 visual tokens saves only about 3-5 ms of LLM
   time, while the hook costs about 4.5 ms.
2. Dynamic risk control preserves success by keeping 218/256 visual tokens, so
   it cannot accelerate.

## Final Convergence Target

The final method should have this shape:

```text
Full visual tokens enter LLM
  -> shallow multimodal fusion for K layers
  -> geometry hard-protection prior
  -> true LLM attention verification
  -> risk-adaptive budget
  -> internal hidden-state pruning
  -> short sequence through remaining LLM layers
```

The intended division of responsibility is:

- Geometry guard preserves action-constraining visual tokens.
- True LLM attention adds semantic/action relevance after shallow fusion.
- Dynamic budget is conservative only when physical/action risk is high.
- Internal pruning is the production compute-saving backend.

Projector-level pruning remains as a legacy comparison, not the final path.

## P0. Fix Evaluation Semantics

Files:

- `scripts/build_performance_report.py`
- `utils/metrics_logger.py`
- `pruning/metrics.py`

Required changes:

- In internal backend mode, report effective retention from
  `internal_kept_visual_tokens / internal_original_visual_tokens`.
- Keep projector retention separate from internal retention.
- Add or prioritize these fields in reports:
  - `compression_backend`
  - `projector_retention`
  - `internal_retention`
  - `effective_visual_tokens_for_llm`
  - `internal_original_seq_length`
  - `internal_kept_seq_length`
  - `lm_prefill_time_ms`
  - `lm_decode_time_ms`
  - `hook_time_ms`
  - `selector_time_ms`
  - `cuda_total_ms`
  - `wall_time_ms`

Acceptance gate:

- `internal_acgtp_dynamic@0.50` must report effective retention as about
  `218/256 = 85.2%`, not `128/256 = 50%`.
- The performance report must make it obvious whether a method prunes before
  the LLM or inside the LLM.

## P1. Split Fast Runtime From Research Diagnostics

Files:

- `pruning/hook.py`
- `pruning/selector.py`
- `pruning/internal_pruning.py`
- `pruning/config.py`

Required changes:

- Add a runtime mode such as `acgtp_runtime_mode = fast | debug | audit`.
- `fast` mode should only produce data needed for inference:
  - geometry payload for internal plan handoff
  - hard-protect mask or branch scores
  - fallback status
  - minimal timing fields
- Move expensive diagnostics to `debug` or `audit`:
  - branch distribution statistics
  - score percentiles
  - high/low overlap probes
  - visualization payloads
  - verbose accounting fields that are not required for selection
- Avoid Python loops in per-token score handling where tensor operations are
  feasible.
- Avoid unnecessary CPU/GPU transfers during every action step.

Acceptance gate:

- `hook_total_ms < 2.0 ms` in fast mode.
- `selection_ms < 0.5 ms` in fast mode.
- Fast/debug modes produce identical keep indices on a fixed smoke sample.

## P2. Prove Internal Pruning Can Save Compute

Files:

- `pruning/internal_pruning.py`
- `scripts/probe_pruning_compute_reality.py`
- optional new ablation runner under `scripts/`

Before further token-selection tuning, verify the internal backend can produce
real speedup with simple selectors.

Minimal ablation matrix:

| Prune Layer | Visual Keep |
|---:|---:|
| 1 | 64, 96, 128 |
| 2 | 64, 96, 128 |
| 3 | 64, 96, 128 |

Selectors:

- `internal_uniform`
- `internal_geometry_only`

Acceptance gate:

- At least one internal setting should reach `cuda_speedup >= 1.10x` and
  `wall_speedup >= 1.05x` in compute probe.
- If uniform internal pruning cannot speed up, optimize the internal gather,
  causal mask, and cache-position path before changing ACGTP scoring.

## P3. Replace Global Competition With Protected Quota Union

Files:

- `pruning/internal_pruning.py`
- `pruning/selector.py`

The final selector must not use a single weighted global top-k over geometry,
attention, and fill scores. Use quota union instead:

```text
P_geo  = geometry hard-protected tokens
P_sem  = true text/prefill attention top-k
P_act  = historical action-attention top-k when available
P_fill = geometry-aware constrained fill

keep = P_geo union P_sem union P_act union P_fill
```

Safety rules:

- Text tokens are always preserved.
- `P_geo` cannot be removed by redundancy filtering.
- Fallback must be explicit and counted.
- If `geo_protected_count > keep_k`, raise the target budget rather than
  silently dropping protected tokens.

Acceptance gate:

- `internal_pruned_geo_critical_count = 0` for protected tokens.
- `internal_selected_by_fallback_count = 0` in normal valid-depth episodes.
- `internal_geo_attention_iou` is reported as diagnosis, not used as the only
  high-risk trigger.

## P4. Recalibrate Dynamic Risk Control

Files:

- `pruning/internal_pruning.py`
- optional `pruning/acgtp_dynamic_controller.py`

The current dynamic controller widens to 0.85 almost every step. This preserves
success but removes acceleration. The new controller should separate physical
risk from attention disagreement.

Recommended keep ratios:

| Risk Level | Effective Visual Keep Ratio |
|---|---:|
| low | 0.35-0.40 |
| medium | 0.50-0.60 |
| high | 0.75-0.85 |

High risk should require physical/action evidence such as:

- contact peak is high
- gripper is close to object/support boundary
- motion corridor is unstable
- depth validity is poor
- action delta or jerk is abnormal
- geometry-attention disagreement is high and contact/motion risk is also high

Low geometry-attention IoU alone should not force high risk, because the
current measured IoU is often around 0.09-0.11.

Acceptance gate:

- Mean effective internal retention should be `0.50-0.65`, not 0.85.
- High-risk retention should appear only on contact/place/unstable frames.
- Total success drop should stay within 5-8 points on the small validation set.

## P5. Keep the Public Strategy Surface Small

Formal experiment strategies should be limited to:

- `baseline_none`
- `projector_acgtp_legacy`
- `internal_geometry_only`
- `internal_geo_guarded`
- `internal_dynamic`

Older hybrid, proxy-attention, global-score, and branch-budget variants should
move to audit/probe-only scripts or be hidden from the main comparison tables.

Acceptance gate:

- Main reports no longer mix obsolete strategy names with the final method.
- Probe scripts can still test legacy variants when needed.

## P6. Delay the Learnable Geometry Expert

Do not train a geometry expert until the following are true:

- Internal pruning has a verified speed-positive operating point.
- Dynamic risk control does not widen to 0.85 globally.
- Geometry hard protection prevents critical-token deletion.

Once those are true, add a lightweight residual scorer only:

```text
ACR_final = ACR_rule + lambda * ACR_learned_residual
```

Recommended first network: tiny geometry CNN with robot-state FiLM, operating
on token-grid geometry features rather than RGB.

## Validation Protocol

Fast compute probe:

```text
baseline_none@1.00
internal_uniform@64/96/128, K=1/2/3
internal_geometry_only@64/96/128, K=1/2/3
```

Small success validation:

```text
LIBERO-Spatial, 3 representative tasks, 3 episodes each
baseline_none
projector_acgtp_legacy
internal_geometry_only
internal_geo_guarded
internal_dynamic
```

Promotion gate for a candidate method:

- Success drop <= 5-8 points.
- No selected task collapses to zero success.
- CUDA speedup >= 1.10x.
- Wall speedup >= 1.05x.
- Hook < 2 ms.
- Selector < 0.5 ms.
- Effective internal visual retention <= 0.65 except high-risk frames.

## Immediate Next Code Tasks

1. Fix report semantics for internal effective retention.
2. Add `acgtp_runtime_mode` and implement fast-mode metric slimming.
3. Add a compact internal ablation runner for `K x visual_keep` compute probes.
4. Recalibrate dynamic risk so low IoU alone does not force 0.85 retention.
5. Re-run small validation and compare against the promotion gate above.

