# Archived ACGTP docs

This directory keeps historical ACGTP notes, superseded plans, and design
snapshots. The current operational status lives in `docs/acgtp_status_20260530.md`.
The design constitution lives in project memory: `acgtp-final-design.md`.

| File | What it is | Status |
|---|---|---|
| `acgtp_project_status_summary.md` | Big status + design dump from 2026-05-29 | Superseded. It predates the verified decode-dominance + KV-cache root cause. |
| `acgtp_code_optimization_strategy.md` | P0-P6 convergence plan | Historical. P0/P1/P3/P4 were done; the old internal speed gate is now known unreachable on base OpenVLA. |
| `acgtp_optimization_fix_summary_20260529.md` | Fix log for the 2026-05-29 metrics/runtime pass | Historical. Its "slow because retention is too conservative" interpretation is superseded. |
| `pruning_compute_reality_plan.md` | Steps A-D plan for internal pruning probes | Historical. Its prefill-only insight matured into the verified root-cause docs. |
| `robot_geo_pruning_implementation_plan.md` | Pre-internal projector-only pipeline and legacy strategies | Superseded by the internal-pruning architecture. |
| `acgtp_design_reframing_next_steps_20260531.md` | Design reframing note: move ACGTP from geometry-score pruning to execution-function-aware structured token allocation | Current planning snapshot. Keep as the next-step design reference until promoted or superseded. |

Still-current docs live one level up in `docs/`:

- `acgtp_status_20260530.md` - authoritative status and verified root cause.
- `vla_efficient_inference_survey_20260530.md` - five-paper efficient-inference survey and the three speedup-mechanism paths.
- `metric_alignment_report_20260530.md` - report-layer metric alignment to retention / success rate / speedup / mechanism evidence.
- `prefill_retention_sweep_20260530.md` - prefill-vs-retention curve; mechanism validity confirmed on base OpenVLA while wall speed is decode-bound.
- `gap1_attention_verification_wip_20260530.md` - WIP handoff for true-attention vs QK-proxy verification.
- `acgtp_layer2_semantic_attention_design_20260530.md` - layer-2 semantic/action attention design note and no-core-code plan.
- `acgtp_functional_quota_strategy_summary_20260531.md` - current strategy summary for execution-function-aware structured token allocation and the 50% retention ablation.
- `acgtp_core_code_latency_audit_20260531.md` - core-code audit, 50% visual-token pruning latency root-cause analysis, strategy explanation, and next-step system optimization plan.
- `eval_protocol.md` - baseline evaluation protocol.

## Hook Backup Note 2026-05-31

- `docs/archive/code_backups/hook_diagnostic_3p8ms_20260531.py` was created as a reference backup of the then-current `pruning/hook.py`.
- As checked on 2026-05-31, this backup and current `pruning/hook.py` have the same SHA256:
  `fd11bedf88dc1857876b47ff751dd0f97edaafeba249384bdc8793042358f0cf`.
- Therefore this backup is not a separate "old no-cache hook implementation". The same `hook.py` contains both paths.
- The observed hook timing difference comes from runtime configuration:
  `diagnostic` / clean no-cache runs use `acgtp_latency_plan_cache_enabled=false` and execute the full selector path, giving roughly 3.5-3.8 ms hook cost.
  cache latency reference runs use `acgtp_latency_plan_cache_enabled=true` and reuse pruning plans, giving roughly 0.9-1.0 ms hook cost, but this can change token-selection behavior and affected success rate in the 2026-05-31 cache run.
- Formal strategy-latency conclusions should prefer clean no-cache latency results, e.g. `/infini-data/openvla/outputs/function_quota_clean_latency_10task1ep_20260531_214000`. Cache-enabled latency results should be treated as a separate runtime-cache optimization experiment, not as the default ACGTP strategy result.
