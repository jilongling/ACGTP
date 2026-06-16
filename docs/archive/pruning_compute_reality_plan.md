# Pruning Compute Reality Check

This note records the first controlled validation of the statement:

> ACGTP is responsible for selecting action-relevant tokens; model-internal pruning is responsible for making the model truly compute less.

## Probe

Script:

```bash
/infini-data/miniconda3/envs/openvla/bin/python \
  scripts/probe_pruning_compute_reality.py \
  --iters 8 \
  --warmup 2 \
  --task task_0 \
  --seed 7 \
  --include_acgtp
```

Main output:

```bash
/infini-data/openvla/outputs/pruning_compute_reality_20260529_121228/pruning_compute_reality_report.md
```

The probe reuses one LIBERO observation and repeatedly calls real OpenVLA inference. It intentionally tests simple `uniform_grid` retention first, so the result measures the compute benefit of projector-level token compression rather than ACGTP token quality.

## Result

Representative median result from the controlled probe:

| Mode | Prefill Seq | LLM ms | CUDA ms | Wall ms | Language Model Calls |
|---|---:|---:|---:|---:|---:|
| none@1.00 | 291 | 216.02 | 240.75 | 252.72 | 7 |
| uniform_grid@0.75 | 227 | 205.22 | 232.16 | 245.83 | 7 |
| uniform_grid@0.60 | 189 | 200.77 | 226.46 | 238.41 | 7 |
| uniform_grid@0.50 | 163 | 203.57 | 229.87 | 241.88 | 7 |
| uniform_grid@0.40 | 137 | 210.93 | 238.73 | 250.98 | 7 |
| ACGTP dynamic-fast@0.60 | 229 | 210.69 | 241.62 | 253.96 | 7 |

Interpretation:

1. The current handoff does compress the sequence.
   - Baseline prefill sequence length is about 291.
   - `uniform_grid@0.60` reduces it to about 189.
   - `uniform_grid@0.40` reduces it to about 137.

2. Projector-level pruning only gives shallow speedup.
   - LLM speedup peaks around 1.08x to 1.09x in the stable runs.
   - Wall speedup is similar or smaller once measurement noise and non-LLM overhead are included.

3. OpenVLA action prediction calls the language model 7 times per step.
   - Only the first multimodal prefill call directly benefits from a shorter visual-token prefix.
   - The later single-token decode calls still pay fixed decoder-layer MLP and generation overhead.
   - Shorter KV context helps attention, but that is not enough to dominate total time at these sequence lengths.

4. Current ACGTP dynamic-fast@0.60 is still conservative in practice.
   - It kept about 194 visual tokens in this probe, close to a 0.75 actual retention ratio.
   - Its hook cost is about 5 ms in fast mode.
   - Therefore its wall-time gain is almost fully erased in the current projector-level path.

## Conclusion

The current projector-level pruning path is correctly passing fewer visual tokens to the model, but this does not translate into strong end-to-end acceleration.

The bottleneck is no longer only token selection quality. It is the execution path:

```text
projector-level compression
    -> shorter first LLM prefill
    -> modest attention/KV benefit during decode
    -> most decoder fixed cost remains
```

Therefore, ACGTP should remain the action-relevance selector, but the next acceleration mechanism should move toward model-internal pruning and/or cache reuse.

## Next Implementation Plan

### Step A: Split prefill and decode timing

Add a timing hook that records each `language_model` call separately:

```text
call_0: multimodal prefill, inputs_embeds=[B, seq, D]
call_1..6: action-token decode, input_ids=[B, 1]
```

Report:

- prefill LLM ms
- decode LLM ms total
- decode LLM ms per token
- KV context length entering decode
- visual token retention

This will quantify exactly how much of each step can be improved by token pruning.

### Step B: Build model-internal pruning proof of concept

Start with a non-ACGTP internal pruning baseline:

```text
LLM layer k
    hidden_states: [B, seq, D]
    keep bos + text + selected visual tokens
    update position_ids
    update attention_mask
    update cache_position
    continue remaining layers with compressed hidden_states
```

First test simple selectors:

- internal uniform@0.75
- internal uniform@0.60
- internal uniform@0.50

Do not integrate ACGTP until this proves real acceleration.

Success criterion:

```text
internal uniform@0.50 gives meaningfully larger speedup than projector uniform@0.50
```

If model-internal uniform pruning still fails to improve speed, the main target should shift to VLA-Cache-style temporal reuse or action decoding optimization.

### Step C: Connect ACGTP to internal pruning

Once Step B works, use ACGTP as the keep-index provider:

```text
ACGTP geometry hard protect
    ∪ action/text attention top-k
    ∪ historical action-vision attention top-k
    -> redundancy filtering
    -> internal hidden-state pruning
```

ACGTP's role:

- protect scene/depth/contact/motion constraint tokens
- avoid robot-self over-protection
- provide safe hard-protect indices before attention-based internal pruning

VLA-Pruner-style role:

- calibrate ACGTP candidates using action/text/prefill attention
- prune hidden states inside the LLM
- update sequence bookkeeping so later layers and decode actually use the compressed context

### Step D: Compare against projector-level ACGTP

Use the same probe and a small LIBERO run:

| Strategy | Purpose |
|---|---|
| none | baseline |
| projector uniform@0.60 | current upper-bound handoff |
| projector ACGTP@0.60 | current ACGTP path |
| internal uniform@0.60 | internal pruning speed proof |
| internal ACGTP@0.60 | final target |

Report:

- success per task
- prefill LLM time
- decode LLM time
- hook/selector/internal-prune overhead
- CUDA total
- wall time
- actual retention

