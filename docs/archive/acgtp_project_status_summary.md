# ACGTP Project Status And Final Design Summary

Date: 2026-05-29

This document summarizes the current project status, the underlying design
idea, what the current pipeline implements, the main problems, and the final
target design for convergence.

## 1. Executive Summary

The project has evolved from a simple visual-token pruning trick into an
action-constrained, geometry-guarded VLA acceleration framework.

The core idea is:

```text
Do not prune visual tokens only by visual saliency.
Preserve tokens that geometrically constrain the robot's future action.
Use true LLM attention to verify semantic/action relevance.
Use dynamic risk control to decide how aggressive pruning can be.
Apply pruning inside the LLM so the saved tokens actually save compute.
```

The current implementation already contains most of the required research
components:

- RGB-D/depth-aware geometric token scoring
- scene-layout branch
- depth-structure branch
- object-side contact branch
- motion-corridor branch
- future action-constraint score
- phase/risk dynamic controller
- history buffer
- true LLM-attention-aware internal pruning backend
- projector-level legacy pruning backend
- metrics and small-scale comparison reports

However, the current system has not yet reached the desired final behavior.
The latest small comparison shows:

- projector-level pruning gives slight speedup but hurts success;
- internal pruning is structurally correct but does not yet save enough time;
- dynamic pruning preserves success by becoming too conservative;
- hook/diagnostic overhead currently consumes most of the internal pruning gain.

The next convergence target is therefore:

```text
Make internal pruning speed-positive first,
then use geometry guard and dynamic budget to preserve success.
```

## 2. Current Project Structure

The pruning project is mainly organized under `/infini-data/openvla/pruning/`.

Important modules:

| Module | Current role |
|---|---|
| `hook.py` | Main projector hook. Builds geometry features, selects/prunes tokens, or hands internal plans to the LLM backend. |
| `internal_pruning.py` | VLA-Pruner-style internal LLM pruning backend. Prunes hidden states after a chosen decoder layer. |
| `selector.py` | ACGTP selector logic and legacy selector variants. |
| `scene_layout.py` | Depth-based support/object/boundary/scene-fill scoring. |
| `depth_edge.py` | Depth edge and valid-depth scoring. |
| `contact_ring.py` | Gripper/contact-ring and self-region logic. |
| `motion_corridor.py` | Motion corridor scoring from robot/gripper motion. |
| `action_constraint.py` | Future action-constraint score from scene/depth/contact/motion branches. |
| `acgtp_dynamic_controller.py` | Phase/risk-aware dynamic keep-ratio and branch-budget controller. |
| `acgtp_history.py` | History stabilizer for recent geometry/action states. |
| `attention_relevance.py` | VLA-Pruner-inspired attention guidance/probe utilities. |
| `post_pruning.py` | Position-preserve and internal plan handoff utilities. |
| `metrics.py` | Hook and pruning metrics schema. |
| `strategy_registry.py` | Current, baseline, and legacy pruning strategy registry. |

Important scripts:

| Script | Current role |
|---|---|
| `scripts/eval_openvla_baseline.py` | Main LIBERO evaluation entry. Supports projector/internal ACGTP configs. |
| `scripts/build_performance_report.py` | Aggregates success and timing metrics into comparison reports. |
| `scripts/probe_pruning_compute_reality.py` | Compute-reality probe for pruning strategies. |
| `scripts/audit_metrics_semantics.py` | Metrics/reporting semantics audit. |
| `scripts/audit_selector_paths.py` | Selector-path audit. |
| `scripts/verify_pipeline.py` | Pipeline verification helper. |

Existing docs:

- `docs/acgtp_code_optimization_strategy.md`
- `docs/eval_protocol.md`
- `docs/pruning_compute_reality_plan.md`
- `docs/robot_geo_pruning_implementation_plan.md`

## 3. Design Philosophy

The design is based on a gap between three concepts:

```text
semantic saliency != action relevance != geometric action constraint
```

Existing VLA pruning methods usually focus on semantic saliency or
model-internal attention. That is useful, but manipulation requires preserving
regions that physically constrain the robot:

- object boundaries
- support surfaces
- container/goal regions
- obstacle/depth discontinuities
- gripper-proximal contact regions
- swept motion corridors
- regions that could change future action or collision risk

Therefore, this project should not be framed as "another token pruning
heuristic". The research story should be:

```text
Robot manipulation needs geometry-constrained token protection.
We preserve visual tokens that constrain the robot's future action,
then use true LLM attention and dynamic risk control to prune safely.
```

The final contribution should be described as:

```text
Robot-Centric Geometry-Guarded Internal Visual Token Pruning
```

or:

```text
Action-Constrained Geometry Token Protection for Efficient VLA Inference
```

The key distinction from pure VLA-Pruner-style methods is:

- VLA-Pruner mainly asks: which visual tokens does the model attend to?
- ACGTP asks: which visual tokens constrain the robot's future physical action?

The final method should combine both:

```text
geometry prior + true LLM attention verification + risk-adaptive internal pruning
```

## 4. Current Implemented Pipeline

The current pipeline can be summarized as:

```text
LIBERO observation
  -> RGB + depth + robot state
  -> OpenVLA vision encoder / projector
  -> VisualTokenPruningHook
       -> depth/scene/contact/motion/action-constraint scores
       -> optional semantic/history/dynamic modules
       -> projector pruning OR internal pruning plan
  -> LLM
       -> if internal backend enabled:
            run first K decoder layers with full visual tokens
            compute true shallow attention / QK text-to-vision relevance
            fuse geometry + attention + fill quotas
            prune hidden_states / position_ids / cache_position / causal_mask
            continue remaining LLM layers with shorter sequence
  -> action head
  -> robot action
```

### 4.1 Input And Depth Status

The LIBERO environment can provide depth when `camera_depths=True`.
The evaluation script has been patched so `robot_geo_acgtp_v2` enables depth
before environment creation. In recent checks, ACGTP no longer fell into
`missing_depth` fallback during normal `eval_openvla_baseline.py` rollout.

Expected valid diagnostics:

```text
depth_source_key = agentview_depth
depth_valid_ratio = 1.0
fallback_used = False
```

### 4.2 Geometry Branches

The current ACGTP geometry side includes:

1. `scene_layout`
   - support surface / tabletop
   - object component
   - object boundary
   - scene-aware fill candidates

2. `depth_structure`
   - depth edge
   - valid depth mask
   - structural discontinuities

3. `contact_ring`
   - gripper self-core filtering
   - object-side contact ring
   - contact region protection

4. `motion_corridor`
   - gripper/action motion corridor
   - smoothed motion direction
   - swept motion relevance

5. `future_action_constraint`
   - additive action-constraint score
   - object-side contact risk
   - swept motion risk
   - robot-self penalty

The current `action_constraint.py` intentionally uses additive/mixture scoring
instead of brittle products such as:

```text
near * motion * edge
```

This is correct for the paper story because a single noisy or weak branch
should not erase a physically meaningful contact/collision region.

### 4.3 Projector Backend

The legacy projector backend directly prunes projector visual tokens before
they enter the LLM.

Observed behavior:

- It can reduce LLM time more visibly.
- It is unsafe when aggressive, because the LLM has not yet performed shallow
  cross-modal fusion.
- In the small comparison, `projector_acgtp_legacy@0.50` improved CUDA only to
  about `1.04x`, but dropped task_1 from `2/2` to `0/2`.

Conclusion:

```text
Projector pruning is useful as a baseline/legacy comparison,
but it should not be the final method.
```

### 4.4 Internal Backend

The internal backend in `internal_pruning.py` implements the desired
VLA-Pruner-style compute-saving path.

Current behavior:

- full 256 visual tokens enter the LLM;
- first `K=2` decoder layers run on full sequence;
- backend computes true shallow text-to-vision relevance;
- if FlashAttention does not return materialized attentions, it falls back to
  Q/K text-to-vision relevance (`llm_qk_text_to_vision`);
- backend prunes hidden states by `index_select`;
- it synchronizes:
  - `hidden_states`
  - `position_ids`
  - `cache_position`
  - causal mask
- text tokens are preserved through sequence-index mapping.

Supported internal modes:

- `geometry_only`
- `attention_diagnostic`
- `geo_guarded`
- `dynamic`

Important current limitation:

```text
K=2 and visual_keep=128 currently save only about 3-5 ms LLM time.
This is not enough because the hook costs about 4.5 ms.
```

### 4.5 Dynamic Controller

The current dynamic controller exists and can preserve success by increasing
keep ratio under high risk.

However, the latest result shows that `internal_acgtp_dynamic@0.50` widened to
about `0.85` almost all the time:

```text
actual internal keep = 218/256
```

This preserved success but made the method slower than baseline.

Conclusion:

```text
Dynamic control is directionally correct,
but the risk trigger is currently too conservative.
```

## 5. Current Experiment Evidence

Latest small comparison:

```text
Output root:
/infini-data/openvla/outputs/acgtp_small_compare_20260529_150822
```

Setup:

- LIBERO-Spatial
- first 3 tasks
- 2 episodes per task
- `max_steps=220`
- seed 7

Task mapping:

| Task | Instruction |
|---|---|
| task_0 | pick up the black bowl between the plate and the ramekin and place it on the plate |
| task_1 | pick up the black bowl next to the ramekin and place it on the plate |
| task_2 | pick up the black bowl from table center and place it on the plate |

Result summary:

| Strategy | Success | Per-task success | Effective visual keep | CUDA | Wall | LLM | Hook | Selector |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| baseline_none@1.00 | 5/6 | 1/2, 2/2, 2/2 | 256/256 | 238.87 ms | 255.55 ms | 213.64 ms | N/A | N/A |
| projector_acgtp_legacy@0.50 | 4/6 | 2/2, 0/2, 2/2 | 128/256 before LLM | 229.30 ms | 246.76 ms | 199.51 ms | 4.55 ms | 1.01 ms |
| internal_acgtp_geometry_only@0.50 | 4/6 | 1/2, 1/2, 2/2 | 128/256 after layer 2 | 238.33 ms | 258.66 ms | 209.12 ms | 4.56 ms | 1.01 ms |
| internal_acgtp_geo_guarded@0.50 | 4/6 | 1/2, 1/2, 2/2 | 128/256 after layer 2 | 239.78 ms | 260.54 ms | 210.67 ms | 4.53 ms | 1.00 ms |
| internal_acgtp_dynamic@0.50 | 5/6 | 1/2, 2/2, 2/2 | 218/256 after layer 2 | 244.14 ms | 264.45 ms | 214.42 ms | 4.59 ms | 1.01 ms |

Main interpretation:

1. Projector pruning gives small speedup but is unsafe.
2. Internal pruning is functionally working but not yet speed-positive.
3. True LLM attention is active, but geometry-attention IoU is low
   (`~0.09-0.11`), so attention and geometry often focus on different tokens.
4. Dynamic pruning preserves success by being too conservative.
5. Reporting currently needs correction because internal effective retention
   can differ from hook-level retention.

## 6. Main Problems

### 6.1 Speedup Is Not Yet Competitive

The current internal backend does not yet match VLA-Pruner-like speedups.

Reasons:

- pruning happens after layer `K=2`, so early LLM layers still use full visual
  tokens;
- 128 visual tokens may still be too many for visible latency gain;
- batch size is 1 and sequence length is modest, so FLOPs reduction does not
  translate linearly into latency reduction;
- causal-mask/cache-position synchronization has fixed overhead;
- hook overhead currently consumes the saved LLM time.

### 6.2 Hook Is Still Research-Oriented

The hook currently carries too many diagnostics and research features in the
runtime path.

Measured overhead:

```text
hook ~= 4.5 ms
selector ~= 1.0 ms
```

This is too high when internal pruning only saves 3-5 ms.

Required direction:

```text
split fast / debug / audit modes
```

### 6.3 Dynamic Risk Is Too Conservative

The dynamic internal strategy is successful but slow because it widens to
`0.85` retention too often.

Root cause:

- low geometry-attention IoU strongly increases risk;
- historical action attention is often unavailable early;
- contact/motion risk terms can push the controller into high risk.

But low geometry-attention IoU alone should not imply physical high risk.
Attention and geometry naturally differ: attention may focus on the named
object, while geometry may protect support/contact/collision structure.

### 6.4 Projector Pruning Is Unsafe At Aggressive Ratios

Projector pruning cuts visual tokens before the LLM has a chance to perform
cross-modal fusion. This makes it prone to deleting tokens required for
language-conditioned action.

The task_1 collapse in the small comparison is the clearest warning.

### 6.5 Strategy Surface Is Still Too Wide

The project contains many historical strategies:

- hybrid score variants
- edge reserve variants
- branch budget variants
- proxy attention variants
- temporal/historical variants
- internal/projector variants

This is useful for ablations but makes the current method look like a mixture
of many tricks.

The formal method should be narrowed to one clean mechanism.

### 6.6 Reporting Semantics Need Correction

`build_performance_report.py` currently computes retention from
`num_visual_tokens_kept`, which can represent hook-level selected tokens.
In internal dynamic mode, the backend may widen the actual internal keep count.

Example:

```text
method label: internal_acgtp_dynamic@0.50
hook-level keep: 128/256
actual internal keep: 218/256
```

Reports must prioritize `internal_kept_visual_tokens` when
`compression_backend=internal`.

## 7. Final Scheme

The final scheme should be:

```text
Geometry-Guarded Internal Pruning with Risk-Adaptive Budget
```

Detailed pipeline:

```text
RGB-D observation + robot state + instruction
  -> OpenVLA vision encoder/projector
  -> full visual tokens enter LLM
  -> run first K decoder layers without pruning
  -> compute true LLM text/action-to-vision relevance
  -> build robot-centric geometry prior
  -> quota-union selection:
       P_geo  = geometry hard-protected tokens
       P_sem  = true semantic attention tokens
       P_act  = historical action-attention tokens
       P_fill = geometry-aware fill tokens
  -> risk-adaptive budget:
       low risk    -> aggressive keep
       medium risk -> moderate keep
       high risk   -> conservative keep
  -> internal hidden-state pruning
  -> action prediction
```

### 7.1 Geometry Prior

Geometry should output two things:

```text
geo_protect_mask
geo_soft_score
```

`geo_protect_mask` is the hard-protected set. It should include:

- contact-relevant boundary tokens
- support/container/object boundary tokens
- motion corridor tokens
- collision/contact risk tokens
- depth-discontinuity tokens

`geo_soft_score` is used only for ranking/fill, not as a single global
decider.

### 7.2 True LLM Attention Verification

The system should use true internal LLM signals after shallow fusion:

- text/prefill-to-vision attention
- QK text-to-vision relevance when materialized attention is unavailable
- historical action-to-vision attention once stable

Attention should supplement geometry, not override geometry.

### 7.3 Protected Quota Union

The final selector must avoid global weighted top-k:

```text
S_final = w_geo * S_geo + w_attention * S_attention + ...
keep = topk(S_final)
```

This is fragile because score scales differ and one branch can dominate.

Instead:

```text
keep = P_geo union P_sem union P_act union P_fill
```

Constraints:

- `P_geo` cannot be removed by redundancy filtering.
- text tokens are always preserved.
- fallback is explicit and counted.
- if protected tokens exceed the budget, the budget should rise.

### 7.4 Risk-Adaptive Budget

The dynamic controller should not be a fixed `0.50` or `0.85` policy.

Recommended behavior:

| Risk level | Effective visual keep |
|---|---:|
| low | 0.35-0.40 |
| medium | 0.50-0.60 |
| high | 0.75-0.85 |

High risk should require physical/action evidence:

- contact peak high
- gripper near object/support boundary
- unstable motion corridor
- poor depth validity
- abnormal action delta/jerk
- high geometry-attention disagreement plus high contact/motion risk

Low geometry-attention IoU alone should be treated as a diagnostic signal, not
a high-risk trigger.

## 8. What Should Not Be Done Yet

Do not train a geometry expert network yet.

The project should first prove:

1. internal pruning can produce real speedup;
2. dynamic budget can avoid global 0.85 retention;
3. geometry hard protection can prevent success collapse;
4. metrics/reporting correctly reflect effective pruning.

Only after that should the project add a lightweight learned residual scorer,
for example:

```text
ACR_final = ACR_rule + lambda * ACR_learned_residual
```

Recommended future network:

```text
Tiny geometry CNN + robot-state FiLM
```

It should operate on token-grid geometry features, not raw RGB.

## 9. Recommended Convergence Plan

### P0. Fix Report Semantics

Fix:

- `scripts/build_performance_report.py`
- `utils/metrics_logger.py`
- `pruning/metrics.py`

Goal:

```text
internal mode reports actual internal retention
```

### P1. Add Fast Runtime Mode

Fix:

- `pruning/hook.py`
- `pruning/selector.py`
- `pruning/internal_pruning.py`
- `pruning/config.py`

Goal:

```text
hook < 2 ms
selector < 0.5 ms
```

### P2. Internal Speed Ablation

Run:

```text
K = 1, 2, 3
visual_keep = 64, 96, 128
selector = internal_uniform, internal_geometry_only
```

Goal:

```text
find at least one speed-positive internal setting
CUDA speedup >= 1.10x
wall speedup >= 1.05x
```

### P3. Recalibrate Dynamic Risk

Goal:

```text
mean internal retention = 0.50-0.65
high retention only during true high-risk frames
```

### P4. Validate Final Candidate

Small validation:

```text
LIBERO-Spatial
3 representative tasks
3 episodes per task
baseline_none
projector_acgtp_legacy
internal_geometry_only
internal_geo_guarded
internal_dynamic
```

Promotion gate:

- success drop <= 5-8 points
- no selected task collapses to zero success
- CUDA speedup >= 1.10x
- wall speedup >= 1.05x
- hook < 2 ms
- selector < 0.5 ms
- effective internal retention <= 0.65 except high-risk frames

## 10. Bottom Line

The current project is already much more than a simple pruning baseline. Its
real identity is:

```text
Robot-centric geometry-constrained internal pruning for VLA inference.
```

The current pipeline has implemented most building blocks, but the final
method is not fully realized yet because:

- internal pruning does not yet save enough compute;
- hook overhead is too high;
- dynamic pruning is too conservative;
- projector pruning is unsafe when aggressive;
- reporting still needs effective-retention correction;
- too many legacy strategies obscure the main contribution.

The final convergence direction should be:

```text
First prove internal pruning saves real compute.
Then use geometry hard protection to preserve success.
Then use dynamic risk control to prune aggressively only when safe.
Finally, add a lightweight learned geometry residual only after the rule-based
pipeline is stable and speed-positive.
```

