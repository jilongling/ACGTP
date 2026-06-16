# Pruning Module Map

The pruning package is organized by responsibility. Real implementation code
lives in a small set of functional packages; most short top-level Python files
are compatibility wrappers for old experiment scripts.

## Public Entry Points

- `hook.py`: OpenVLA projector hook entry point.
- `selector.py`: public selector compatibility surface.
- `config.py`: pruning runtime configuration.
- `method_profiles.py`: experiment-profile labels such as `functional_quota_static_050`.
- `strategy_registry.py`: supported strategy sets and legacy/audit gating.
- `runtime_config.py`: runtime config normalization helpers.

## Main Path

`method profile -> robot_geo_acgtp_v2 -> internal backend -> functional quota`

- `methods/acgtp_v2.py`: current ACGTP-v2 fast/full selectors.
- `methods/functional_quota.py`: branch quota allocation, merge, fill, fallback.
- `internal/backend.py`: LLM-internal ACGTP pruning backend.
- `internal/quota_config.py`: internal quota config builder.

`functional_quota_static_050` remains a method profile, not a new
`pruning_strategy`. It expands to `robot_geo_acgtp_v2` plus the internal backend
with functional quota enabled and `keep_ratio=0.50`.

## Packages

- `methods/`: public selector implementations, ACGTP-v2 selection, functional quota policy, selector validation helpers.
- `signals/`: spatial, action, semantic, robot-geometry, cache, and temporal scoring signals.
- `runtime/`: hook runtime mixins, geometry collection, diagnostics, and post-pruning state.
- `internal/`: LLM-internal pruning backend, internal quota config, and internal uniform baseline.
- `legacy/`: ACGTP-v1, hybrid, branch-budget, legacy runtime, and audit-only strategy sets.
- `core/`: hook metrics, visualization helpers, and shared hook utilities.
- `tests/`: lightweight validation helpers kept with the pruning package.

## Compatibility

Historical imports such as `pruning.depth_edge`, `pruning.selector_core`,
`pruning.hook_fast_runtime`, and `pruning.internal_pruning` are handled by
`_compat.py` as lazy aliases. New code should prefer the functional packages
above.
