# ACGTP Efficiency and Metric Audit - 2026-05-31

## Scope

This audit checks why current ACGTP visual-token pruning does not show the same clear speedup reported by VLA-Cache, VLA-Pruner, VLA-ADP, DepthCache, and VLA-IAP. It does not change pruning strategy or selector logic.

Checked local papers:

- `C:/Users/凌凌漆/Desktop/论文/VLA/VLA-Cache.pdf`
- `C:/Users/凌凌漆/Desktop/论文/VLA/VLA-Pruner.pdf`
- `C:/Users/凌凌漆/Desktop/论文/VLA/VLA-ADP.pdf`
- `C:/Users/凌凌漆/Desktop/论文/VLA/DepthCache.pdf`
- `C:/Users/凌凌漆/Desktop/论文/VLA/VLA-IAP.pdf`

Checked project files:

- `pruning/internal_pruning.py`
- `pruning/hook.py`
- `pruning/metrics.py`
- `pruning/selector.py`
- `utils/metrics_logger.py`
- `scripts/eval_openvla_baseline.py`
- `scripts/build_benchmark_metrics.py`
- `scripts/rollup_pruning_results.py`
- `scripts/run_functional_quota_ablation.py`
- `outputs/function_quota_ablation_full/**`

## Paper Metric Alignment

The five papers generally report a combination of:

- success rate or relative accuracy;
- token retention / compression ratio;
- FLOPs or FLOP ratio;
- CUDA latency or action inference latency;
- control frequency;
- max GPU memory in some cases.

Important differences from the current ACGTP run:

- VLA-Cache reports success rate, control frequency, FLOPs, and CUDA latency. Its speedup is often CUDA-latency oriented, and the method reuses static tokens across frames rather than only pruning one input sequence.
- VLA-Pruner reports success rate, FLOPs, inference latency, CUDA runtime, and memory across 50%, 25%, and 12.5% retention. It shows stronger speedup at aggressive retention and includes OpenVLA-OFT / pi0 settings where action generation cost differs from base OpenVLA.
- VLA-ADP mainly reports OpenVLA-OFT with parallel decoding and action-chunk latency. This is not the same bottleneck profile as base OpenVLA autoregressive action decoding.
- DepthCache reports end-to-end inference latency per action step and average compression ratio, but it uses token merging/cache-like temporal stabilization instead of pure pruning.
- VLA-IAP reports latency/speedup and CUDA runtime/memory at several retention rates, with dynamic retention and other models including pi0 / pi0.5.

Therefore, the current ACGTP `baseline_none` vs `functional_quota_static_050` result is not directly comparable to the strongest speedup numbers in those papers unless the report clearly separates CUDA latency, model forward, wall/action-step latency, retention ratio, and model family.

## Is Visual Token Pruning Actually Happening?

Yes. For `functional_quota_static_050` in `outputs/function_quota_ablation_full`:

- `internal_pruning_applied`: 1050 / 1050 steps.
- `internal_pruning_plan_ready`: 1050 / 1050 steps.
- `compression_backend`: `internal`.
- `projector_pruning_applied`: false, so there is no double pruning at projector output.
- `internal_original_visual_tokens`: 256.
- `internal_kept_visual_tokens`: 128.
- `internal_original_seq_length`: about 288.62.
- `internal_kept_seq_length`: about 160.62.
- `internal_pruning_layer`: 2.
- `fallback_used`: 0 / 1050.

The pruning process is therefore functionally active and reduces the sequence seen by later LLM layers.

## Why Speedup Does Not Show Through

Current main comparison:

| Strategy | Success | Tokens | LLM total | Hook | Model speed | Wall speed |
|---|---:|---:|---:|---:|---:|---:|
| baseline_none | 8/9 | 256 | 214.53 ms | N/A | 1.000x | 1.000x |
| functional_quota_static_050 | 8/9 | 128 | 209.09 ms | 3.96 ms | 1.007x | 0.984x |

Main causes:

1. Pruning happens after shallow LLM layers, not before the whole LLM. Layers before and including the prune point still process the full sequence.
2. OpenVLA base still autoregressively decodes action tokens. The logs show decode calls increase by 6 per action step; visual pruning mostly helps the multimodal prefill / post-prune layers, not the fixed decode floor.
3. Current benchmark has no true prefill/decode split. `llm_forward_time_ms` is total language-model forward hook time, so it cannot prove prefill acceleration yet.
4. Hook overhead is too large relative to the saved LLM time. For the main strategy, about 5.45 ms LLM total is saved, while hook total is about 3.96 ms. Net LLM minus hook is only about 1.48 ms.
5. Projector timing includes pruning hook overhead. Baseline projector time is about 0.36 ms; functional quota projector time is about 4.21 ms, matching the hook overhead.
6. Same 50% retention strategies have large LLM-saved spread, so trajectory mix / timing noise / success-failure length differences still affect the current 9-episode result.
7. Wall-clock latency is measured with multiple scopes. Episode-mean wall time and direct step-record wall time differ by more than 5%, so both must be reported separately.

## Metric Reliability

Reliable / usable:

- Visual token original/kept/pruned counts: real logged values.
- Internal sequence original/kept lengths: real logged values.
- Internal pruning applied/plan/fallback flags: real logged values.
- LLM total forward time: real measured by language-model forward hooks, but it is not a prefill/decode split.
- Hook total and selection time: real logged values; selection is included inside hook total.
- Model forward time: real timed around `model.predict_action(...)`.
- Success rate and episode steps: real episode logs.
- GPU memory: real runtime memory logs.

Partially reliable / diagnostic only:

- Wall speedup: real, but multiple scopes exist. Benchmark now reports both episode-mean wall and step-record wall.
- Paired comparison: task-name + trial-index paired, not strict seed/initial-state paired.
- FLOPs: token/sequence proxy only. Exact FLOPs are unavailable.
- Branch selected counts: logged, but not unique counts.

Unavailable / cannot claim:

- true LLM prefill time;
- true LLM decode time;
- prefill speedup;
- branch unique counts / overlap ratios / branch sum equals kept;
- exact FLOPs;
- cache hit rate, because ACGTP is not a cache method.

## Metric Builder Update Made During This Audit

Updated `scripts/build_benchmark_metrics.py` only.

Changes:

- Added `mean_step_wall_time_ms_step_records`.
- Added `wall_step_record_speedup_vs_baseline`.
- Added `wall_step_record_latency_reduction_ms`.
- Added `wall_timing_scope` to per-strategy benchmark summaries.
- Added `wall_timing_scope_gap_*pct` diagnostic warning when episode wall and step-record wall differ by at least 5%.
- Added both wall scopes to `benchmark_report.md` Timing Coverage.

Validation:

```bash
cd /infini-data/openvla
/infini-data/miniconda3/envs/openvla/bin/python -m py_compile \
  scripts/build_benchmark_metrics.py \
  scripts/rollup_pruning_results.py \
  scripts/run_functional_quota_ablation.py \
  scripts/eval_openvla_baseline.py \
  utils/metrics_logger.py \
  pruning/metrics.py \
  pruning/hook.py \
  pruning/selector.py

/infini-data/miniconda3/envs/openvla/bin/python scripts/build_benchmark_metrics.py \
  --input_dir /infini-data/openvla/outputs/function_quota_ablation_full \
  --baseline_strategy baseline_none
```

Both commands passed.

## Conclusion

The pruning process itself is active and correctly reduces visual tokens entering later LLM layers. The absence of stable end-to-end speedup is mainly a systems and measurement issue:

- savings are limited to post-prune LLM computation;
- hook overhead consumes most of the observed LLM-total saving;
- OpenVLA base autoregressive decode remains a fixed floor;
- current timing lacks prefill/decode attribution;
- current result size is small and timing-noisy.

Current strongest claim:

`functional_quota_static_050` preserves success rate in this 9-episode run and shows an LLM-total reduction trend at 50% visual-token retention, but it has not yet demonstrated stable wall-clock acceleration.

## Recommended Next Steps

1. Add true prefill/decode timing probes inside `model.predict_action(...)` generation without changing prompt, tokens, or action output.
2. Run a 3x3 or larger paired timing protocol with fixed task/trial/seed metadata.
3. Optimize hook runtime toward less than 1 ms before expecting wall-clock gains at 50% retention.
4. Sweep retention ratios: 75%, 50%, 37.5%, 25%, and compare quality-preserving speedup.
5. Add branch unique/overlap attribution in selector/hook logging, then rerun.
6. Keep exact FLOPs unavailable unless a real profiler is added; use token/sequence proxy only.
