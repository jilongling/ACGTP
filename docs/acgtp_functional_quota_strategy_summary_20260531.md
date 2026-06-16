# ACGTP Functional Quota Strategy Summary, 2026-05-31

This note summarizes the current ACGTP strategy variants after the shift from
geometry-score top-k pruning to execution-function-aware structured token
allocation.

## Context

The current ACGTP direction is no longer "use geometry constraints as a single
importance score, then keep the top-k visual tokens." The updated framing is:

> VLA visual-token pruning should treat token importance as a structured budget
> allocation problem across execution functions, not as one global salience
> ranking.

The intended keep set is now an explicit union of protected and functional
branches:

```text
P_hard_geo
  union P_layout
  union P_contact
  union P_motion
  union P_sem
  union P_act
  union P_fill
```

Hard constraints still apply:

- OpenVLA weights are not modified.
- No selector training is introduced.
- Text tokens are always kept.
- Geometry hard-protect tokens are never removed by ordinary pruning.
- If geometry hard-protect exceeds the nominal budget, the effective keep ratio
  must increase rather than dropping protected tokens.
- Attention or QK-proxy disagreement is diagnostic only; it must not override
  geometry hard protection by itself.
- Fallbacks must be visible in metrics, never silent.

## Strategy Variants

### `baseline_none`

No pruning. All 256 visual tokens enter the LLM. This is the quality and timing
baseline for the current base OpenVLA tests.

Observed in the 2026-05-31 small ablation:

- Success: `8/9`
- Effective visual tokens: `256`
- Wall speed: `1.000x`
- Model speed: `1.000x`

### `legacy_geo_guarded_quota_050`

Legacy internal `geo_guarded` pruning at 50% visual-token retention.

Conceptually:

```text
P_hard_geo union P_semantic_proxy union P_fill
```

Where:

- `P_hard_geo` keeps geometry hard-protected tokens first.
- `P_semantic_proxy` is still the current QK text-to-vision proxy signal, not
  true materialized LLM attention under FlashAttention.
- `P_fill` fills the remaining budget.
- There are no explicit scene-layout, contact, or motion-corridor quota
  branches.

Interpretation:

- This variant preserves geometry hard protection, but it does not prevent
  function-level token competition.
- In practice, semantic/fill can dominate the non-hard-protected budget and
  miss scene-layout or interaction details.

Observed result:

- Success: `7/9`
- Internal retention: `50%`
- Wall speed: `0.967x`
- Model speed: `0.986x`
- `internal_pruned_geo_critical_count = 0.0`

### `functional_quota_static_050`

Current main strategy: execution-function-aware structured allocation at 50%
visual-token retention.

Conceptually:

```text
P_hard_geo
  union P_layout
  union P_contact
  union P_motion
  union P_sem
  union P_act
  union P_fill
```

Branch meanings:

- `P_hard_geo`: geometry hard-protected tokens; highest priority.
- `P_layout`: scene-layout tokens, mainly preserving depth edges, spatial
  discontinuities, object/container/cabinet boundaries, and camera-centric
  layout structure.
- `P_contact`: contact / interaction tokens, preserving gripper-proximal and
  local manipulation-critical regions.
- `P_motion`: motion-corridor tokens, preserving likely future interaction path
  regions based on end-effector motion direction.
- `P_sem`: instruction/task semantic visual tokens. In the current base setup,
  this is still a QK text-to-vision proxy unless true attention is explicitly
  probed under a non-FlashAttention path.
- `P_act`: action-related visual tokens from historical/action attention. In
  the current base test, this path is mostly not active yet.
- `P_fill`: remaining-budget fill to avoid brittle over-narrow selection.

Interpretation:

- The main change is quota-separated selection before union, not global weighted
  top-k.
- Layout, contact, motion, semantic, and fill no longer compete purely through a
  single scalar ranking.
- This directly targets the previous failure mode where local robot geometry
  tokens could crowd out scene-layout structure, or semantic tokens could miss
  low-level execution details.

Observed result:

- Success: `8/9`, matching baseline.
- Internal retention: `50%`.
- Internal sequence retention: about `55.7%`.
- Wall speed: `0.984x`.
- Model speed: `1.007x`.
- Mean LLM time moved from baseline `214.53 ms` to `209.09 ms`.
- Hook overhead was about `3.96 ms`, so there is no end-to-end speedup claim on
  base OpenVLA yet.
- `internal_pruned_geo_critical_count = 0.0`.

## Ablation Variants

### `functional_no_layout_050`

Functional quota with the layout branch removed:

```text
P_hard_geo union P_contact union P_motion union P_sem union P_act union P_fill
```

Observed result:

- Success: `7/9`.
- It drops one additional `task_2` episode compared with baseline and the full
  functional strategy.
- Interpretation: layout tokens are not just redundant with contact/semantic
  tokens. They likely protect camera-centric spatial structure needed by
  LIBERO-Spatial.

### `functional_no_contact_050`

Functional quota with the contact branch removed:

```text
P_hard_geo union P_layout union P_motion union P_sem union P_act union P_fill
```

Observed result:

- Success: `7/9`.
- It drops one additional `task_1` episode compared with baseline and the full
  functional strategy.
- Interpretation: contact / interaction tokens are not redundant with layout or
  semantic tokens. This supports keeping an explicit low-level execution branch.

### `functional_no_motion_050`

Functional quota with the motion branch removed:

```text
P_hard_geo union P_layout union P_contact union P_sem union P_act union P_fill
```

Observed result:

- Success: `8/9`, matching baseline and the full functional strategy.
- Interpretation: in this small LIBERO-Spatial sample, the motion branch has not
  shown independent quality value yet. It should not be overclaimed. It may need
  phase-adaptive scheduling to become useful, especially in approach/contact
  phases.

## Current Evidence

The small ablation was run under:

```text
outputs/function_quota_ablation_full
num_tasks = 3
num_episodes = 3
methods =
  baseline_none,
  functional_quota_static_050,
  functional_no_layout_050,
  functional_no_contact_050,
  functional_no_motion_050,
  legacy_geo_guarded_quota_050
```

The generated report is:

```text
/infini-data/openvla/outputs/function_quota_ablation_full/functional_quota_ablation_report.md
```

Summary table:

| Method | Success | Internal retention | Wall speed | Model speed | Main signal |
|---|---:|---:|---:|---:|---|
| `baseline_none` | `8/9` | N/A | `1.000x` | `1.000x` | Quality/timing baseline |
| `functional_quota_static_050` | `8/9` | `50%` | `0.984x` | `1.007x` | Main upgraded strategy; quality preserved |
| `legacy_geo_guarded_quota_050` | `7/9` | `50%` | `0.967x` | `0.986x` | Older strategy loses one extra episode |
| `functional_no_layout_050` | `7/9` | `50%` | `1.001x` | `1.022x` | Removing layout hurts quality |
| `functional_no_contact_050` | `7/9` | `50%` | `0.977x` | `0.992x` | Removing contact hurts quality |
| `functional_no_motion_050` | `8/9` | `50%` | `0.973x` | `0.986x` | Motion branch not yet proven in this sample |

## Interpretation

The best-supported current strategy is `functional_quota_static_050`.

The evidence supports three claims:

1. Structured allocation is better aligned with the current ACGTP story than a
   single global geometry/semantic score.
2. `P_layout` and `P_contact` have early ablation support: removing either one
   caused a quality drop in this small run.
3. `P_motion` remains plausible but under-proven. It should be kept as a
   candidate branch, but the next iteration should test phase-adaptive scheduling
   before treating it as a final contribution.

This run does not support an end-to-end wall-clock speedup claim on base OpenVLA.
The model is still decode-bound and the hook overhead can erase small LLM-time
savings. The current value of the result is mechanism/quality evidence: the
functional quota preserves success at 50% internal visual-token retention better
than the legacy `geo_guarded` variant.

## Next Work

Recommended next steps:

1. Expand the ablation beyond `3 x 3` episodes to confirm that layout/contact
   remain consistently useful.
2. Add phase-adaptive functional quota scheduling:
   - early phase: increase layout budget;
   - approach phase: preserve layout floor and increase motion/contact;
   - contact phase: increase contact and local depth-edge protection.
3. Re-run true-attention proxy validation for `P_sem` using the existing
   read-only probe, because the current base+FlashAttention path still uses QK
   text-to-vision proxy rather than materialized LLM attention.
4. Avoid claiming motion-corridor as proven until a larger or phase-aware test
   shows a consistent benefit.
5. Use OFT/pi0 or another lower-decode-floor target before making final
   end-to-end acceleration claims.
