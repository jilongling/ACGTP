# ACGTP Current Status — 2026-05-30

This is the authoritative, current status doc. It supersedes everything in
`docs/archive/`. The design "constitution" (three-layer method + hard
constraints) lives in project memory (`acgtp-final-design.md`); this doc is the
*operational* status: what's verified, what the real bottleneck is, and the
decision in front of us.

## 1. What ACGTP is

A **plug-and-play visual-token pruning module** for VLA inference. Goal: prune
redundant visual tokens after geometry-guarded protection so the model
accelerates while keeping the success-rate drop small. Target models include
OpenVLA, OpenVLA-OFT, pi0, etc. **base OpenVLA-7b is the first validation
vehicle, not the only target model** — the module is meant to be portable.

Three layers (unchanged, see memory `acgtp-final-design.md`):
1. Robot-centric geometry prior -> `geo_protect_mask` (hard) + `geo_soft_score` (soft).
2. True LLM attention verification (full 256 tokens -> first K layers -> real attention) -> P_sem, P_act.
3. Risk-adaptive internal pruning: quota union `keep = P_geo ∪ P_sem ∪ P_act ∪ P_fill`, keep ratio set by physical/action risk.

## 2. VERIFIED root cause: why pruning doesn't speed up base OpenVLA

This is the central, *current* finding (verified 2026-05-30 with
`scripts/probe_decode_cache_reality.py` and `probe_pruning_compute_reality.py`).
It **replaces** the older explanations in `docs/archive/` that blamed hook
overhead or "dynamic retention too conservative (218/256)".

base OpenVLA is **autoregressive**: `modeling_prismatic.py:518` ->
`.generate(max_new_tokens=7)` = **1 prefill + 6 serial decode** calls.
Measured: prefill ~44ms (~20%), 6x decode ~180ms (~80%). **Decode dominates.**

Three structural reasons visual pruning can't help here:

1. **Decode is memory-bandwidth bound on the 7B weights.** Each decode step
   loads the full model to push 1 token; the attention term (1 query x N keys)
   is tiny next to the per-layer MLP. Cutting KV from 291 -> 163 barely touches
   decode time.

2. **Internal pruning only fires in prefill** (`internal_pruning.py:863`,
   `is_prefill = inputs_embeds.shape[1] > 1`); decode uses the original forward.
   And in prefill, layers `0..K` (K=2) must run **full 256 tokens** to produce
   the attention used for selection; only layers `K+1..31` run the short
   sequence. So 3/32 layers never benefit.

3. **KV cache is left inconsistent across layers.** Layers 0-2 cache 291
   tokens, layers 3-31 cache 163. Proven by probe output:
   `CACHE LENGTH INCONSISTENT across layers: distinct=[163, 291]`.
   Net effect: the small prefill saving is eaten by the full first-K layers
   plus selection/mask-rebuild overhead.

**Net measured: internal_geo_guarded@0.50 LLM saved = -1.95ms (slightly
slower).** This is a structural property of decode-dominated autoregressive
VLA, **not a bug in the module and not fixable by tuning the keep ratio or the
hook.**

Contrast — VLA-Pruner (`/infini-data/VLA-Pruner-main/`, arXiv 2511.16449) gets
1.99x because it prunes **before prefill** (all 32 layers benefit) **and**
measures on **OFT (parallel decoding, prefill-dominated)**. Same pruning
action: cuts the big cost on prefill-dominated models, cuts only scraps on
decode-dominated base OpenVLA.

## 3. Verified measurements (3 task x 3 trial small validation)

| Method | Success | LLM time effect | Notes |
|---|---|---|---|
| baseline_none | 8/9 | — (cuda 251ms ref) | reference |
| projector_acgtp_legacy@0.50 | 6/9 (-22%) | **+15.75ms saved** | prunes before prefill -> all 32 layers benefit; but unsafe success drop |
| internal_geo_guarded@0.50 | 7/9 (-11%) | **-1.95ms (slower)** | the root cause above; geo_protect critical-deletion count = 0 |
| internal_dynamic@0.50 | 7/9 (-11%) | ~0 | retains 82.2% |

Key takeaways: geometry hard-protection works (critical deletions = 0, success
drop bounded). The **only** path that produced real LLM savings on base OpenVLA
is pruning **before** prefill (the projector path).

## 4. Metrics audit verdict

Trustworthy. `CUDATimer` syncs correctly; the probe LM hook syncs; reports
baseline on `none`. `cuda_speedup` is systematically > `wall_speedup` because
the CPU geometry hook isn't inside `cuda_latency` — expected, not a bug.

## 5. The decision in front of us

To show **both** speedup AND success-preservation on base OpenVLA, two routes:

- **Route A — move pruning before prefill.** Prune at the projector / pre-layer-0
  so all 32 layers + decode KV use the reduced token set. `projector_acgtp_legacy`
  already proves +15.75ms real saving; use `geo_protect_mask` to recover its
  -22% success drop. Cost: gives up the "run K layers first to read true
  attention" design. **This is the only route that yields positive speedup on
  base OpenVLA today.**
- **Route B — split the metrics by model.** Use base OpenVLA only to prove
  success-preservation (geometry guard works); measure *speedup* on a
  prefill-dominated model (OFT / pi0). Blocked today: no OFT checkpoint on disk
  (`/infini-data/checkpoints/...` and `/infini-data/model/` are both base
  autoregressive) and network is unreachable to download one; repo has only
  training-side OFT fragments, no OFT inference path.

No code change has been committed for either route yet — awaiting the route
decision.

## 6. Formal experiment surface (unchanged)

`baseline_none`, `projector_acgtp_legacy` (baseline), `internal_geometry_only`,
`internal_geo_guarded`, `internal_dynamic`. Core strategy name
`robot_geo_acgtp_v2`. Everything else is audit/probe only.

## 7. Key files

- `pruning/internal_pruning.py` — internal pruning forward (selection ~712-746, prune loop ~983-1068)
- `pruning/hook.py` — `_build_internal_geometry_payload` builds geo_protect_mask / geo_soft_score
- `pruning/config.py` — all `acgtp_*` knobs; `acgtp_internal_prune_layer=2`
- `scripts/probe_decode_cache_reality.py` — proves the KV-cache inconsistency
- `scripts/probe_pruning_compute_reality.py` — prefill/decode timing breakdown
- `scripts/run_core_surface_validation.py` — runs the 4 core methods small validation
- `scripts/eval_openvla_baseline.py` — main LIBERO eval entry
- `docs/eval_protocol.md` — baseline eval protocol (still current)
