# Robot-Centric Geometry Pruning Implementation Plan

## 0. Scope And Guardrails

This document audits the current OpenVLA external pruning pipeline and proposes a staged plan for turning it into a robot-centric 3D geometry constraint token-protection mechanism.

The intended design remains:

- OpenVLA model weights unchanged.
- Vision encoder, projector, LLM, action head, prompt, task suite, and action decoding unchanged.
- Depth, camera parameters, and robot state are used only by an external scorer.
- Pruning happens only after projector output visual tokens are produced and before those visual tokens enter the LLM/action decoding path.
- `baseline_none_keep100` must keep the original OpenVLA behavior.
- Every new mechanism must be optional and controlled by config or command-line flags.

## 1. Current Pipeline Data Flow

The current inference/evaluation flow is:

```text
RGB observation
  -> processor / image preprocessing
  -> OpenVLA vision encoder
  -> OpenVLA projector
  -> projected visual tokens
  -> external VisualTokenPruningHook
  -> pruned projected visual tokens
  -> normal OpenVLA multimodal input construction
  -> LLM / action decoding
  -> original action output logic
```

The pruning hook is implemented as a PyTorch forward hook attached to the module named `projector` in `pruning/hook.py` via `VisualTokenPruningHook.attach_to_model`. The hook receives the projector output tensor and returns either the same tensor or an `index_select`-gathered subset along the visual-token dimension.

Important current properties:

- `num_visual_tokens_original` is read from `visual_tokens.shape[1]`, not from a fixed constant.
- When pruning is disabled, `_run()` returns `visual_tokens` unchanged.
- Token gather is applied only on the projector output visual-token tensor.
- Text token construction is not directly touched by the pruning module.
- Geometry data is collected outside the OpenVLA model path and passed into the pruning hook through the geometry recorder.
- Missing geometry inputs are handled through fallback strategies instead of changing model behavior.

The relevant entry points are:

- `scripts/eval_openvla_baseline.py`: evaluation CLI, hook construction, geometry recorder setup, logging bridge.
- `pruning/hook.py`: projector-output pruning hook.
- `pruning/config.py`: strategy/config normalization.
- `pruning/selector.py`: token keep-index selection.
- `pruning/robot_geometry.py`: robot-centric score computation.
- `utils/metrics_logger.py`: per-step, per-episode, and rollup metrics.
- `scripts/run_pruning_ablation.py`: method aliases and ablation runner.
- `scripts/rollup_pruning_results.py`: summary aggregation.

## 2. Current Pruning Modes

### 2.1 `baseline_none_keep100`

Configuration:

```text
pruning_strategy = none
pruning_enabled = false
geometry_enabled = false
keep_ratio = 1.0
```

Behavior:

- No pruning hook should alter the projector output.
- Geometry is disabled.
- No depth, camera, or robot-state information is required.
- Visual tokens are kept at their original count.
- This mode is the reference path for unchanged OpenVLA behavior.

### 2.2 `random_keep075`

Behavior:

- Selects `K = int(num_visual_tokens * keep_ratio)` visual tokens using a fixed random seed.
- Sorts selected indices before gather, preserving original token order in the kept sequence.
- Does not require depth or robot geometry.
- Used mainly as a compute-saving baseline.

### 2.3 `uniform_grid_keep075`

Behavior:

- Selects a spatially spread subset of tokens.
- Does not use depth, camera parameters, or robot state.
- Useful as a non-random spatial coverage baseline.

### 2.4 `depth_edge_fast_keep075`

Behavior:

- Uses token-level metric depth sampled at the current token grid.
- Computes finite-difference depth-edge scores on the token grid.
- Selects top-K depth-edge tokens.
- Uses depth only in the external pruning hook.
- Keeps selected indices sorted before gathering visual tokens.

This is the current fastest geometry-aware baseline.

### 2.5 `depth_edge_fast_diverse_keep075`

Behavior:

- Reuses the fast token-depth path from `depth_edge_fast`.
- Selects most tokens by depth-edge score.
- Reserves a small subset for spatial diversity through a 4x4 cell layout.
- Keeps indices unique and sorted.

This is currently the strongest stable fixed-ratio baseline in the existing experiments.

### 2.6 `robot_geo_near_keep075`

Behavior:

- Computes token-level 3D points from sampled depth, cached camera rays, and camera extrinsics.
- Computes a gripper-nearness score in robot/base coordinates.
- Falls back if camera or robot-state data is unavailable.
- In the current hook flow, additive robot-geometry modes have been observed to underperform depth-edge-only methods.

### 2.7 `robot_geo_corridor_keep075`

Behavior:

- Adds previous-gripper-position state inside the hook.
- Computes a motion vector and a short forward corridor.
- Scores tokens by distance to that corridor.
- If previous gripper position is missing or motion is too small, corridor score is inactive rather than crashing.
- Falls back to depth-edge-diverse if required camera or robot-state data is missing.

### 2.8 `robot_geo_contact_budget_keep075`

Behavior:

- Splits a fixed token budget into:
  - edge-selected tokens,
  - edge-gated geometry-contact tokens,
  - spatial diversity tokens.
- Uses:
  - `S_edge`,
  - `S_near_contact = normalize(S_near) * normalize(S_edge)`,
  - `S_corridor_contact = normalize(S_corridor) * normalize(S_edge)`.
- Geometry is used as a contact-aware supplement instead of directly competing with depth-edge score for the whole top-K budget.
- Current default ratios are:

```text
K_total = 192 when keep_ratio = 0.75
K_edge = 144
K_geo_contact = 24
K_diverse = 24
```

This strategy is a bridge from pure depth-edge pruning toward robot-centric contact-aware protection while keeping the OpenVLA model unchanged.

### 2.9 `robot_geo_dynamic`

Behavior:

- Computes a dynamic keep ratio using geometric risk signals.
- Uses fixed allowed token counts corresponding to far, mid, and near/contact phases.
- Falls back safely when geometry is unavailable or depth validity is too low.
- This is implemented as an optional strategy and is not part of the baseline path.

## 3. What Already Satisfies Plug-In Token Pruning Requirements

The current code already satisfies several core requirements for a pluggable external pruning mechanism:

1. Hook location is external to model weights.
   - `VisualTokenPruningHook` attaches to `projector`.
   - The hook returns a pruned projector-output tensor.

2. Baseline can be disabled cleanly.
   - `strategy="none"` and `keep_ratio=1.0` return visual tokens unchanged.
   - `baseline_none_keep100` sets pruning and geometry disabled in the runner.

3. Visual token count is shape-driven.
   - `num_tokens = int(visual_tokens.shape[1])`.
   - Selection functions receive `num_tokens` or the actual score length.

4. Token order preservation is explicit.
   - Keep indices are sorted before `index_select`.
   - Validation records `keep_indices_sorted` and duplicate counts.

5. Fallbacks are explicit.
   - Missing depth/camera/robot-state cases can fall back to no pruning, uniform grid, or depth-edge-diverse depending on strategy.
   - `fallback_used` and `fallback_reason` are logged.

6. Metrics are separated from model logic.
   - Hook timing, selected token counts, geometry availability, valid token ratio, and detailed contact-budget stats are logged without changing action decoding.

7. Ablation entry points exist.
   - `run_pruning_ablation.py` defines method aliases.
   - `rollup_pruning_results.py` aggregates summary fields.

8. Debug visualization exists.
   - `pruning/visualization.py` can save score heatmaps and selection-source masks when visualization is enabled.
   - Visualization is off by default for formal timing.

## 4. Missing Pieces For A Full Robot-Centric 3D Geometry Constraint Mechanism

The current pipeline is close to the desired external-pruning architecture, but the robot-centric 3D constraint mechanism is still incomplete in several ways.

### 4.1 Depth / Token 3D Mapping

Current status:

- Fast token-depth sampling and sparse 3D projection exist.
- Cached token rays and token-grid assumptions exist.
- Depth edge is computed at token-grid resolution.

Remaining needs:

- More systematic validation that sampled token centers match the actual projected visual-token grid under all preprocessing modes.
- Explicit mapping metadata per step: image size, token grid, projection mode, camera signature, cache hit/miss.
- Optional consistency probe comparing projected token points with image overlays.

### 4.2 Camera Intrinsics / Extrinsics

Current status:

- Camera intrinsics/extrinsics are read by the geometry path when available.
- Fallbacks exist for missing camera data.

Remaining needs:

- Stronger validation of transform direction: `T_robot_cam` versus `T_cam_robot`.
- Explicit coordinate-frame logs: world, base, robot, camera, and whether world is being used as robot/base approximation.
- A small transform sanity test using a known camera pose and synthetic depth.

### 4.3 Gripper / End-Effector Pose

Current status:

- `extract_gripper_position` accepts several robot-state field names.
- Hook stores previous gripper position per episode.

Remaining needs:

- Better contact-point definition beyond `eef_pos` or gripper center.
- Optional adapter for simulator-specific finger body / geom / site positions.
- Logging that distinguishes:
  - eef center,
  - grip site,
  - fingertip midpoint,
  - virtual contact probe.

### 4.4 Motion Direction

Current status:

- `robot_geo_corridor` uses current and previous gripper positions.
- Small motion disables corridor score safely.

Remaining needs:

- More stable temporal filtering of gripper motion.
- Per-episode reset correctness tests.
- Motion-frame visualization and action-sensitivity checks.

### 4.5 Contact Risk Score

Current status:

- `robot_geo_contact_budget` uses edge-gated contact scores.
- Contact geometry supplements depth-edge rather than replacing it.

Remaining needs:

- Validate whether geo-contact tokens land on object contact regions rather than gripper/self/background.
- Compare cell-level contact score against action sensitivity.
- Tune budget ratios only after diagnostics, not by adding complex models.

### 4.6 Dynamic Keep Ratio

Current status:

- `robot_geo_dynamic` exists with far/mid/near phases.
- Allowed K values are constrained.

Remaining needs:

- Confirm phase transitions are not too aggressive.
- Log phase counts per task and failure cases.
- Keep dynamic logic optional and separate from fixed-ratio baselines.

### 4.7 Temporal Stability / History

Current status:

- Previous gripper position is stored.

Remaining needs:

- History of selected cells or tokens for stability analysis.
- Optional hysteresis on phase selection.
- Optional token keep-mask stability metric.

These should be added only after fixed-ratio robot geometry is stable.

### 4.8 Debug Visualization

Current status:

- Score maps and keep masks can be saved.
- Contact-budget source masks exist.

Remaining needs:

- Standard one-command visualization for a stored episode/step.
- Overlay of projected gripper/contact point and token centers.
- Cell-level action sensitivity figure aligned with the same 4x4 cell grid.

## 5. Relation To Recent Work

This project should position itself as an external, robot-centric pruning mechanism, not as a modified OpenVLA architecture.

### OpenVLA

OpenVLA provides the base model, processor, evaluation framework, and action decoding path. This project should preserve OpenVLA internals and treat pruning as a post-projector external hook.

### VLA-Pruner

VLA-Pruner motivates token pruning using semantic and action-related importance. The current project differs by avoiding decode-attention dependence and by using lightweight external geometry signals before action decoding.

### VLA-Cache

VLA-Cache motivates reducing visual-token computation by reusing relatively stable visual information. The current project can borrow the idea that many visual tokens are redundant across time, but should not introduce cache reuse until the single-step geometry scorer is stable.

### DepthCache

DepthCache motivates depth-guided spatial compression. The current `depth_edge_fast` and `depth_edge_fast_diverse` modes are aligned with this principle by operating on token-level metric depth and preserving depth discontinuities.

### VLA-IAP

VLA-IAP emphasizes interaction-first pruning and dynamic scheduling. The current contact-budget and dynamic strategies are conceptually aligned: tokens near likely interaction/contact regions receive protection, and risky phases may keep more tokens.

### ADP

ADP motivates action-aware or end-effector-aware gating. The current gripper-near and corridor scores are a lightweight external version of this idea, but they should remain outside the model and should not change action heads or action decoding.

### ThinkProprio

ThinkProprio suggests proprioception can affect visual reasoning and token selection. The current project uses proprioceptive robot state only in the pruning scorer, not as an LLM input or prompt addition.

### OC-VLA / 4D-VLA

OC-VLA and 4D-VLA motivate aligning visual observations with object-centric or 3D/4D spatial representations. The current project borrows the coordinate-alignment idea through RGB-D, intrinsics, extrinsics, and robot-frame token points, but does not inject 3D positional embeddings into OpenVLA.

## 6. Innovation Boundary

The intended contribution is:

```text
OpenVLA unchanged
+ external projector-output pruning hook
+ lightweight robot-centric 3D geometry expert
+ token score / keep mask / optional dynamic keep ratio
```

This project should not become:

- ordinary visual saliency pruning,
- decode-attention pruning,
- an MLP-trained pruning head,
- a modified OpenVLA backbone,
- a proprioception-injected LLM variant,
- a prompt-engineered robot-state method,
- a task-config-specific heuristic.

The core boundary is that depth, camera, and robot state are external signals used to decide which visual tokens to keep. They are not model inputs to OpenVLA itself.

## 7. Five Minimal Implementation Stages

Each stage below is intentionally small. Stages should be implemented and validated one at a time.

### Stage 1: Geometry Availability Contract

Goal:

Make geometry inputs explicit and auditable before adding stronger geometry constraints.

Files:

- `pruning/config.py`
- `pruning/hook.py`
- `pruning/metrics.py`
- `utils/metrics_logger.py`
- `pruning/validation_tests.py`

Functions/classes:

- Add or tighten a small geometry-availability helper in `hook.py`.
- Add a test helper that constructs missing-depth, missing-camera, and missing-robot-state cases.

Validation:

```bash
PYTHONPATH=/infini-data/openvla python -m pruning.validation_tests
python scripts/run_pruning_ablation.py --methods baseline_none_keep100,depth_edge_fast_keep075 --num_tasks 1 --num_trials_per_task 1 ...
```

Disable switch:

- `--pruning_strategy none`
- `--geometry_enabled false`

Action logic impact:

- None. This stage only clarifies availability and fallback logging.

### Stage 2: Transform Direction And Token-Point Probe

Goal:

Verify sparse token 3D projection and camera transform direction without changing selection semantics.

Files:

- `pruning/robot_geometry.py`
- `pruning/visualization.py`
- `scripts/action_sensitivity_probe.py` or a small debug-only probe script
- `pruning/validation_tests.py`

Functions/classes:

- Add a synthetic transform unit test for `project_tokens_to_robot`.
- Add optional debug metadata for transform direction and projected point ranges.

Validation:

```bash
PYTHONPATH=/infini-data/openvla python -m pruning.validation_tests
python scripts/visualize_pruning_episode.py --method depth_edge_fast_diverse --episode 0 --step 20 ...
```

Disable switch:

- Debug visualization remains off unless `--save_pruning_vis true`.

Action logic impact:

- None. This stage only validates mapping and visualization.

### Stage 3: Contact Point Adapter

Goal:

Replace ambiguous gripper-center usage with an optional contact-point adapter while preserving fallback to existing behavior.

Files:

- `pruning/robot_geometry.py`
- `pruning/config.py`
- `pruning/metrics.py`
- `pruning/validation_tests.py`

Functions/classes:

- Add `extract_contact_point(robot_state, sim_info, cfg)` or a similarly small helper.
- Return both `contact_point_available` and `contact_point_source`.

Validation:

```bash
PYTHONPATH=/infini-data/openvla python -m pruning.validation_tests
python scripts/eval_openvla_baseline.py --pruning_strategy robot_geo_contact_budget --num_tasks 1 --num_episodes 1 ...
```

Disable switch:

- Default source remains current gripper/eef position until explicitly enabled.
- `--pruning_strategy depth_edge_fast_diverse` bypasses robot-contact scoring.

Action logic impact:

- None. Only external score inputs change when enabled.

### Stage 4: Contact-Budget Diagnostics Before New Scoring

Goal:

Decide whether robot-contact tokens are useful by measuring where they land and how they correlate with action sensitivity.

Files:

- `pruning/visualization.py`
- `scripts/action_sensitivity_probe.py`
- `scripts/rollup_pruning_results.py`

Functions/classes:

- Ensure selected source masks are saved only when visualization is enabled.
- Ensure action sensitivity outputs per-cell edge/contact score and action delta.

Validation:

```bash
python scripts/action_sensitivity_probe.py --method robot_geo_contact_budget --task_id 0 --trial_idx 0 --step 20 ...
```

Disable switch:

- Probes are offline scripts only.
- No runtime effect unless explicitly invoked.

Action logic impact:

- None in formal evaluation. The probe deliberately reruns actions offline for analysis.

### Stage 5: Conservative Dynamic Keep Ratio Stabilization

Goal:

Only after fixed-ratio contact-budget is stable, make dynamic keep ratio safer and easier to interpret.

Files:

- `pruning/scheduler.py`
- `pruning/hook.py`
- `pruning/metrics.py`
- `utils/metrics_logger.py`
- `pruning/validation_tests.py`

Functions/classes:

- Add phase hysteresis or a minimum phase duration only if tests show frequent phase flicker.
- Keep K restricted to the existing allowed values.
- Add per-task phase-ratio logging.

Validation:

```bash
PYTHONPATH=/infini-data/openvla python -m pruning.validation_tests
python scripts/run_pruning_ablation.py --methods depth_edge_fast_diverse_keep075,robot_geo_dynamic --num_tasks 1 --num_trials_per_task 1 ...
```

Disable switch:

- Use any fixed-ratio method instead of `robot_geo_dynamic`.
- `baseline_none_keep100` remains unchanged.

Action logic impact:

- None. Only the number of kept projector-output visual tokens changes.

## 8. Minimal Verification Checklist For Every Future Change

For every future module, report:

1. Modified files.
2. New functions/classes.
3. Disable path and default behavior.
4. Minimum validation command.
5. Whether action output logic changed.

Required tests:

```bash
PYTHONPATH=/infini-data/openvla python -m pruning.validation_tests
python scripts/run_pruning_ablation.py \
  --checkpoint /infini-data/checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite libero_spatial \
  --num_tasks 1 \
  --num_trials_per_task 1 \
  --methods baseline_none_keep100,depth_edge_fast_keep075,depth_edge_fast_diverse_keep075
```

Expected invariants:

- `baseline_none_keep100` does not prune.
- Fixed `keep_ratio=0.75` methods keep exactly 192 of 256 tokens when N=256.
- Keep indices are unique and sorted.
- `duplicate_indices_count = 0`.
- Missing measurements are `null`, not `0.0`.
- No depth, camera, or robot-state data enters the OpenVLA model.
- Text tokens are not pruned.
- Action decoding and output format remain unchanged.
