# OpenVLA Baseline Evaluation Protocol

## 1. Overview

This document describes the evaluation protocol for the OpenVLA baseline. The goal is to establish a reliable measurement framework that captures task success rates, inference latency, GPU memory usage, and other performance metrics. This baseline will serve as the reference point for future comparisons when token pruning and geometric constraint experts are introduced.

---

## 2. Model

- **Architecture**: OpenVLA (Vision-Language-Action model)
- **Base**: LLaMA-2 7B + SigLIP vision backbone
- **Action Space**: 7-DOF end-effector delta actions (position x3, rotation x3, gripper x1)
- **Action Discretization**: 256 bins per dimension, mapped to the last 256 tokens of the LLM vocabulary

### 2.1 Checkpoint

Use a fine-tuned checkpoint appropriate for the task suite:

| Task Suite        | HuggingFace Checkpoint                        |
|-------------------|----------------------------------------------|
| libero_spatial    | `openvla/openvla-7b-finetuned-libero-spatial` |
| libero_object     | `openvla/openvla-7b-finetuned-libero-object`  |
| libero_goal       | `openvla/openvla-7b-finetuned-libero-goal`    |
| libero_10         | `openvla/openvla-7b-finetuned-libero-10`     |
| libero_90         | `openvla/openvla-7b-finetuned-libero-90`     |

---

## 3. Task Suite

### 3.1 LIBERO Simulation

- **Library**: [LIBERO](https://github.com/ARISE-Initiative/libero) (Mujoco-based)
- **Cameras**: Single agent-view camera (256x256 or 768x768)
- **Image Preprocessing**: Resize to 224x224 using Lanczos3 resampling; rotate 180 degrees (due to inverted camera in simulation)

### 3.2 Task Suites

| Suite          | # Tasks | Max Steps | Description                                   |
|----------------|---------|-----------|-----------------------------------------------|
| libero_spatial | 10      | 220       | Spatial reasoning (object positions)          |
| libero_object  | 10      | 280       | Object manipulation (pick & place)            |
| libero_goal    | 10      | 300       | Goal-conditioned (describe target state)       |
| libero_10      | 10      | 520       | Mixed: 10% of original MagicSoup tasks        |
| libero_90      | 10      | 400       | Mixed: 90% of original MagicSoup tasks       |

### 3.3 Episode Structure

1. **Reset Phase**: Initialize scene with a specific initial state (objects at designated positions)
2. **Settling Phase**: Execute 10 steps of no-op action `[0,0,0,0,0,0,-1]` to let objects settle
3. **Execution Phase**: Query model at each step until `done=True` or `max_steps` reached
4. **Success Check**: Environment's built-in success condition

---

## 4. Evaluation Procedure

### 4.1 Number of Episodes

- **Minimum**: 10 episodes per task (50 total for libero_spatial/object/goal)
- **Standard**: 50 episodes per task (recommended for statistical significance)
- **Each episode uses the same initial state** (LIBERO provides deterministic initial states per trial index)

### 4.2 Random Seed

- **Default seed**: 7
- Seed is set for Python `random`, NumPy, PyTorch CPU/GPU, and the simulation environment

### 4.3 Inference Settings

- **Precision**: BF16 (recommended), FP16, or FP32
- **Attention**: Flash Attention 2 (when supported by hardware)
- **Sampling**: Greedy decoding (`do_sample=False`)
- **Center Crop**: Enabled (`center_crop=True`) for models fine-tuned with image augmentation

---

## 5. Metrics

### 5.1 Episode-Level Metrics

| Metric                            | Description                                                       | Unit       |
|-----------------------------------|-------------------------------------------------------------------|------------|
| `task_name`                      | Identifier for the task                                           | string     |
| `episode_id`                      | Unique episode index                                              | int        |
| `success`                         | Whether the episode was successful                                | bool       |
| `num_steps`                       | Number of execution steps taken                                   | int        |
| `episode_time_sec`                | Total wall-clock time for the episode                            | seconds    |
| `avg_step_time_ms`                | Mean wall-clock time per step                                     | ms         |
| `avg_model_forward_time_ms`       | Mean model forward pass time per step                             | ms         |
| `avg_action_postprocess_time_ms`  | Mean action post-processing (unnorm, gripper) time per step        | ms         |
| `max_gpu_memory_mb`               | Peak GPU memory allocated during the episode                       | MB         |
| `seed`                            | Random seed used                                                  | int        |
| `prefill_time_ms`                 | TODO: Time for prompt encoding (prefill) phase                    | ms         |
| `decode_time_ms`                  | TODO: Time for autoregressive token generation (decode) phase      | ms         |

### 5.2 Step-Level Metrics (optional, `log_step_metrics=True`)

| Metric                            | Description                                                       | Unit       |
|-----------------------------------|-------------------------------------------------------------------|------------|
| `task_name`                       | Task identifier                                                   | string     |
| `episode_id`                      | Episode index                                                     | int        |
| `step_id`                        | Step number within the episode                                    | int        |
| `model_forward_time_ms`           | Model forward pass time                                           | ms         |
| `action_postprocess_time_ms`      | Action post-processing time                                       | ms         |
| `env_step_time_ms`                | Environment step time                                             | ms         |
| `total_step_time_ms`              | Total wall-clock time for this step                              | ms         |
| `gpu_memory_mb`                   | GPU memory in use at this step                                    | MB         |
| `num_visual_tokens_before`        | Number of visual tokens (patch embeddings) before any pruning     | int / null |

### 5.3 Summary Metrics

| Metric                              | Description                                              | Unit |
|-------------------------------------|----------------------------------------------------------|------|
| `overall_success_rate`              | Fraction of all episodes that succeeded                  | %    |
| `mean_episode_steps`                | Mean number of steps per episode                         | int  |
| `mean_episode_time_sec`             | Mean wall-clock time per episode                         | sec  |
| `mean_step_time_ms`                 | Mean step time across all episodes                       | ms   |
| `mean_model_forward_time_ms`        | Mean model forward time across all episodes               | ms   |
| `std_model_forward_time_ms`          | Std dev of model forward time                            | ms   |
| `mean_gpu_memory_mb`                | Mean peak GPU memory per episode                         | MB   |
| `per_task_success_rate`              | Success rate broken down by task name                    | dict |
| `per_task_mean_latency`             | Per-task mean model forward time                         | dict |

### 5.4 Extension Fields (for future comparisons)

| Field                    | Baseline Value | Description                                      |
|--------------------------|---------------|------------------------------------------------|
| `pruning_method`         | `"none"`      | Token pruning method (e.g., "geometric", "attention") |
| `keep_ratio`             | `1.0`         | Fraction of visual tokens kept after pruning   |
| `geometric_expert_time_ms` | `0.0`       | Time spent in geometric constraint expert       |
| `num_tokens_before`      | `null`        | Visual tokens before pruning                   |
| `num_tokens_after`       | `null`        | Visual tokens after pruning                    |

---

## 6. Timing Measurement

### 6.1 Principles

1. **GPU Synchronization**: All GPU timings are wrapped with `torch.cuda.synchronize()` before reading the clock
2. **Warm-up**: Model is loaded before timing begins; model loading time is **not** included in episode/step timings
3. **Reset Exclusion**: Environment reset time is **not** included in step-level timings
4. **Clock**: Uses `time.perf_counter()` (CPU wall-clock) for maximum precision

### 6.2 Timing Pipeline

```
total_step_time_ms
  = model_forward_time_ms
    + action_postprocess_time_ms
    + env_step_time_ms
```

### 6.3 Prefill / Decode Split (Future)

Currently the full `model.forward()` + `generate()` call is measured as a single `model_forward_time_ms`. When token pruning is integrated, the timing should be split into:

- **Prefill**: Time to encode prompt + visual tokens through the LLM
- **Decode**: Time for autoregressive generation of `action_dim` action tokens

The measurement points should be:

```
t0 = perf_counter()
  -> vision backbone + projector
t1 = perf_counter()
  -> prefill through LLM (full prompt encoding)
t2 = perf_counter()
  -> autoregressive decode (action_dim tokens)
t3 = perf_counter()
torch.cuda.synchronize()
t4 = perf_counter()

prefill_time = t2 - t1
decode_time = t3 - t2
total_forward = t4 - t0
```

---

## 7. Memory Measurement

- **API**: `torch.cuda.max_memory_allocated()` / `torch.cuda.max_memory_reserved()`
- **Unit**: Megabytes (MB)
- **Reset**: `torch.cuda.reset_peak_memory_stats()` called before each episode
- **Recording**: `max_gpu_memory_mb` is the peak allocated memory across the entire episode

---

## 8. Result Format

Results are saved to `outputs/openvla_baseline_eval/` (or as specified by `--save_dir`):

```
outputs/openvla_baseline_eval/
├── config.yaml           # Experiment configuration (reproducible)
├── episode_metrics.csv    # One row per episode
├── step_metrics.csv      # One row per step (if log_step_metrics=True)
├── summary.json           # Aggregated statistics
└── eval_protocol.md       # This document (copied for reproducibility)
```

### 8.1 `summary.json` Schema

```json
{
  "model": "OpenVLA",
  "task_suite": "libero_spatial",
  "num_episodes": 10,
  "overall_success_rate": 0.3,
  "mean_episode_steps": 120.5,
  "mean_episode_time_sec": 24.1,
  "mean_step_time_ms": 201.2,
  "mean_model_forward_time_ms": 185.3,
  "std_model_forward_time_ms": 12.7,
  "mean_action_postprocess_time_ms": 0.4,
  "mean_gpu_memory_mb": 15000.0,
  "per_task": {
    "task_0": {
      "success_rate": 0.5,
      "num_episodes": 10,
      "mean_steps": 110.0,
      "mean_model_forward_time_ms": 183.1
    }
  }
}
```

---

## 9. Baseline Reproducibility

To reproduce results exactly:

```bash
python scripts/eval_openvla_baseline.py \
  --config configs/eval_openvla_baseline.yaml \
  --seed 7
```

The `config.yaml` in the output directory captures all parameters.

---

## 10. Future Comparisons

When token pruning or geometric constraint experts are added, compare against this baseline using:

1. **Success Rate Delta**: `delta_sr = sr_pruned - sr_baseline`
2. **Latency Improvement**: `speedup = latency_baseline / latency_pruned`
3. **Memory Reduction**: `mem_reduction = 1 - (mem_pruned / mem_baseline)`
4. **Token Reduction**: `token_reduction = 1 - keep_ratio`

The comparison should be run with the **same random seed** and the **same task suite**.

---

## 11. Geometric Expert Integration Point

When adding a geometric constraint expert:

**Insert location**: Between the vision backbone output and the LLM input in `scripts/eval_openvla_baseline.py`'s `predict_action()` function.

**Procedure**:
1. Extract visual patch embeddings from `vla.vision_backbone(pixel_values)`
2. Pass through geometric expert to compute constraint masks
3. Apply mask to patch embeddings (zero out masked tokens)
4. Pass pruned embeddings through the projector
5. Continue with standard LLM forward pass

**Timing**: Measure `geometric_expert_time_ms` separately using the same profiler pattern.

---

## 12. Token Pruning Integration Point

When adding visual token pruning:

**Insert location**: In `experiments/robot/openvla_utils.py`'s `get_vla_action_with_stats()` function (or in `scripts/eval_openvla_baseline.py`'s `predict_action()`).

**Procedure**:
1. Hook the vision backbone output to count `num_visual_tokens_before`
2. Apply geometric/attention-based pruning mask
3. Count `num_visual_tokens_after`
4. Continue with pruned sequence

**Metrics to record**:
- `keep_ratio = num_visual_tokens_after / num_visual_tokens_before`
- `geometric_expert_time_ms` (if applicable)
- Update `model_forward_time_ms` to reflect pruned computation

---

## 13. Citing

If this evaluation framework is useful for your research, please cite the OpenVLA paper:

```
@article{octomatt2024openvla,
  title={OpenVLA: An Open-Source Vision-Language-Action Model},
  author={OctoMRI Team},
  year={2024}
}
```
