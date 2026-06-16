"""Behavior regression checks for the current pruning surface.

These tests intentionally cover a narrow, stable surface that should not change
during refactors: public imports, method-profile expansion, keep-index shape and
ordering, and internal functional-quota accounting.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Dict, Iterable, List

import numpy as np
import torch

from ..core.metrics import HookMetrics
from ..method_profiles import method_cli_args
from ..methods.acgtp_v2 import select_acgtp_v2_fast
from ..methods.baselines import select_keep_indices
from ..methods.functional_quota import select_internal_quota_tokens
from ..internal.backend import ACGTPInternalPruningBackend, InternalPruningPlan
from ..runtime.post import PostPruningStateManager
from ..strategy_registry import normalize_strategy


def _assert_keep_indices(keep: np.ndarray, expected: int, n: int = 256) -> None:
    keep = np.asarray(keep, dtype=np.int64).reshape(-1)
    assert keep.shape == (int(expected),), keep.shape
    assert np.all(keep >= 0), keep
    assert np.all(keep < int(n)), keep
    assert np.unique(keep).shape[0] == int(expected), keep
    assert np.all(keep[:-1] <= keep[1:]), keep


def _args_to_map(args: Iterable[str]) -> Dict[str, List[str]]:
    items = list(args)
    assert len(items) % 2 == 0, items
    out: Dict[str, List[str]] = {}
    for key, value in zip(items[0::2], items[1::2]):
        out.setdefault(str(key), []).append(str(value))
    return out


def test_compat_aliases() -> None:
    from pruning.depth_edge import compute_depth_edge_scores as old_depth_edge
    from pruning.hook_fast_runtime import HookFastRuntimeMixin
    from pruning.internal_pruning import ACGTPInternalPruningBackend as old_backend
    from pruning.robot_geometry import project_tokens_to_robot as old_project_tokens
    from pruning.selector_core import select_keep_indices as old_select_keep
    from pruning.signals.robot import project_tokens_to_robot
    from pruning.signals.spatial import compute_depth_edge_scores

    assert old_depth_edge is compute_depth_edge_scores
    assert old_project_tokens is project_tokens_to_robot
    assert old_backend is ACGTPInternalPruningBackend
    assert old_select_keep is select_keep_indices
    assert HookFastRuntimeMixin.__name__ == "HookFastRuntimeMixin"


def test_method_profile_surface() -> None:
    args = _args_to_map(method_cli_args("functional_quota_static_050"))
    assert args["--pruning_strategy"][-1] == "robot_geo_acgtp_v2"
    assert args["--keep_ratio"][-1] == "0.50"
    assert args["--acgtp_compression_backend"][-1] == "internal"
    assert args["--acgtp_internal_pruning_enabled"][-1] == "true"
    assert args["--acgtp_internal_selection_mode"][-1] == "geo_guarded"
    assert args["--acgtp_dynamic_enabled"][-1] == "false"
    assert args["--acgtp_internal_functional_quota_enabled"][-1] == "true"
    try:
        normalize_strategy("functional_quota_static_050")
    except ValueError:
        pass
    else:
        raise AssertionError("functional_quota_static_050 must remain a method profile, not a pruning_strategy")


def test_none_and_uniform_baselines() -> None:
    keep, meta = select_keep_indices("none", num_tokens=256, keep_count=128)
    _assert_keep_indices(keep, 256)
    assert np.array_equal(keep, np.arange(256, dtype=np.int64))
    assert meta["selection_strategy_name"] == "none"
    assert meta["keep_indices_count"] == 256
    assert meta["keep_ratio_actual"] == 1.0
    assert meta["fallback_used"] is False

    keep_128, meta_128 = select_keep_indices("uniform_grid", num_tokens=256, keep_count=128)
    _assert_keep_indices(keep_128, 128)
    assert int(np.sum(keep_128)) == 16320
    assert keep_128[:16].tolist() == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    assert keep_128[-16:].tolist() == [225, 227, 229, 231, 233, 235, 237, 239, 241, 243, 245, 247, 249, 251, 253, 255]
    assert meta_128["selection_strategy_name"] == "uniform_grid"
    assert meta_128["keep_indices_count"] == 128
    assert meta_128["keep_ratio_actual"] == 0.5

    keep_192, meta_192 = select_keep_indices("uniform_grid", num_tokens=256, keep_count=192)
    _assert_keep_indices(keep_192, 192)
    assert int(np.sum(keep_192)) == 24480
    assert keep_192[:16].tolist() == [0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20]
    assert keep_192[-16:].tolist() == [235, 236, 238, 239, 240, 242, 243, 244, 246, 247, 248, 250, 251, 252, 254, 255]
    assert meta_192["selection_strategy_name"] == "uniform_grid"
    assert meta_192["keep_indices_count"] == 192
    assert meta_192["keep_ratio_actual"] == 0.75


def test_robot_geo_acgtp_v2_fast_selector() -> None:
    n = 256
    grid = np.arange(n, dtype=np.float32)
    row = np.floor(grid / 16.0)
    col = grid % 16.0
    valid = np.ones(n, dtype=np.bool_)
    scene = row / 15.0
    depth = col / 15.0
    contact = np.zeros(n, dtype=np.float32)
    contact[80:120] = np.linspace(0.1, 1.0, 40, dtype=np.float32)
    motion = np.zeros(n, dtype=np.float32)
    motion[120:160] = np.linspace(1.0, 0.1, 40, dtype=np.float32)
    action = np.maximum(contact, motion)

    keep, meta = select_acgtp_v2_fast(
        scene_layout_scores=scene,
        depth_edge_scores=depth,
        contact_ring_scores=contact,
        motion_corridor_scores=motion,
        action_constraint_scores=action,
        valid_mask=valid,
        constrained_fill_mask=valid,
        keep_k=128,
        grid_h=16,
        grid_w=16,
        hard_protect_ratio=0.60,
        motion_corridor_valid=True,
        semantic_enabled=False,
        acgtp_attention_enabled=False,
    )
    _assert_keep_indices(keep, 128, n=n)
    assert meta["selection_strategy_name"] == "robot_geo_acgtp_v2"
    assert meta["selector_function_name"] == "select_acgtp_v2_fast"
    assert meta["acgtp_fast_selector_used"] is True
    assert meta["acgtp_branch_accounting_valid"] is True
    assert meta["branch_accounting_valid"] is True
    assert meta["expected_kept"] == 128
    assert meta["keep_indices_count"] == 128
    assert meta["keep_indices_sorted"] is True
    assert meta["keep_indices_unique"] is True
    assert meta["keep_ratio_actual"] == 0.5


def test_functional_quota_static_internal_backend() -> None:
    n = 16
    base = torch.linspace(0.05, 1.0, n).tolist()
    plan = InternalPruningPlan.from_visual_keep_indices(
        list(range(8)),
        original_visual_tokens=n,
        target_keep_ratio=0.5,
        geometry_payload={
            "valid_mask": [True] * n,
            "constrained_fill_mask": [True] * n,
            "scene_scores": base,
            "depth_scores": list(reversed(base)),
            "contact_scores": base,
            "motion_scores": base,
            "motion_corridor_valid": True,
            "layout_score": base,
            "contact_score": list(reversed(base)),
            "motion_score": base,
            "geo_protect_mask": [False] * n,
        },
        quota_config={
            "selection_mode": "geo_guarded",
            "attention_enabled": False,
            "functional_quota_enabled": True,
        },
    )
    backend = ACGTPInternalPruningBackend(None, image_token_length=n, fail_on_error=False)
    keep, info = backend.resolve_visual_keep_indices(
        plan,
        seq_len=n + 1,
        device=torch.device("cpu"),
        attention_weights=None,
    )
    assert keep.tolist() == [0, 1, 2, 11, 12, 13, 14, 15]
    assert info["internal_functional_quota_enabled"] is True
    assert info["internal_branch_accounting_valid"] is True
    assert info["internal_branch_sum_equals_kept"] is True
    assert info["internal_dynamic_keep_k"] == 8
    assert info["internal_dynamic_keep_ratio"] == 0.5
    assert info["internal_quota_layout_k"] == 3
    assert info["internal_quota_contact_k"] == 2
    assert info["internal_quota_motion_k"] == 2
    assert info["internal_quota_fill_k"] == 1
    assert info["internal_selected_by_layout_count"] >= 1
    assert info["internal_selected_by_contact_count"] >= 1
    assert info["internal_selected_by_motion_count"] >= 1


def test_functional_quota_non_fast_avoids_cpu_list_sync() -> None:
    source = inspect.getsource(select_internal_quota_tokens)
    assert ".detach().cpu().tolist()" not in source


def test_functional_quota_non_fast_preserves_candidate_order() -> None:
    n = 4
    device = torch.device("cpu")
    zero = torch.zeros(n, dtype=torch.float32, device=device)
    valid = torch.ones(n, dtype=torch.bool, device=device)
    result = select_internal_quota_tokens(
        n=n,
        device=device,
        valid=valid,
        fill_mask=valid,
        fallback=torch.as_tensor([1, 2, 1], dtype=torch.long, device=device),
        target_k=1,
        target_ratio=0.25,
        hard_k=1,
        sem_k=0,
        hist_k=0,
        sem_ratio=0.0,
        hist_ratio=0.0,
        requires_geo_alignment=True,
        quota_config={"functional_quota_enabled": True, "latency_fast_path": False},
        explicit_protect_mask=torch.zeros(n, dtype=torch.bool, device=device),
        geo_score=zero,
        scene=zero,
        depth=zero,
        contact=zero,
        motion=zero,
        layout_score=zero,
        contact_score=zero,
        motion_score=zero,
        sem_score=zero,
        hist_score=zero,
        motion_valid=False,
        sem_available=False,
        hist_available=False,
    )
    assert result.keep.tolist() == [1]


def test_internal_decode_bypass_reuses_prefill_pruned_cache() -> None:
    backend = ACGTPInternalPruningBackend(None, image_token_length=16, fail_on_error=False)
    backend.last_info = {
        "original_seq_length": 20,
        "kept_seq_length": 12,
        "internal_kv_cache_token_reduction_ratio": 0.40,
    }

    backend.record_decode_call(
        input_len=1,
        cache_before=12,
        cache_after=13,
        cache_present=True,
        cache_before_by_layer=[20, 20, 12, 12],
        cache_after_by_layer=[21, 21, 13, 13],
    )

    info = backend.stats()
    assert info["decode_calls"] == 1
    assert info["internal_decode_pruning_applied"] is False
    assert info["internal_decode_uses_pruned_prefill_cache"] is True
    assert info["internal_decode_prefill_kv_reduction_ratio"] == 0.40
    assert info["internal_decode_cache_benefit_source"] == "prefill_internal_kv_cache_token_reduction_ratio"
    assert info["internal_decode_pruning_reason"] == "decode_bypasses_internal_pruning_reuses_prefill_pruned_cache"


def test_projector_position_preserve_pre_hook() -> None:
    updates: Dict[str, object] = {}
    manager = PostPruningStateManager(
        config=SimpleNamespace(acgtp_position_preserve_enabled=True),
        update_stats=lambda **kwargs: updates.update(kwargs),
    )
    metrics = HookMetrics()
    manager.prepare_position_preserve_info(
        keep_indices_np=np.asarray([0, 2, 4, 6], dtype=np.int64),
        num_tokens=8,
        metrics=metrics,
    )
    module = SimpleNamespace()
    kwargs = {"inputs_embeds": torch.zeros(2, 8, 4)}

    _, out_kwargs = manager.language_model_pre_hook(module, (), kwargs)

    expected = torch.tensor([[0, 1, 3, 5, 7, 9, 10, 11], [0, 1, 3, 5, 7, 9, 10, 11]])
    assert torch.equal(out_kwargs["position_ids"].cpu(), expected)
    assert updates["position_preserve_reason"] == "prefill_position_ids_preserved"
    assert updates["position_preserve_applied"] is True
    assert updates["position_preserve_original_seq_len"] == 12
    assert getattr(module, "acgtp_pruning_info")["original_seq_length"] == 12


def test_projector_position_preserve_missing_pending_is_visible() -> None:
    updates: Dict[str, object] = {}
    manager = PostPruningStateManager(
        config=SimpleNamespace(acgtp_position_preserve_enabled=True),
        update_stats=lambda **kwargs: updates.update(kwargs),
    )
    metrics = HookMetrics()
    manager.prepare_position_preserve_info(
        keep_indices_np=np.asarray([0, 2, 4, 6], dtype=np.int64),
        num_tokens=8,
        metrics=metrics,
    )
    manager._pending_position_info = None
    kwargs = {"inputs_embeds": torch.zeros(1, 8, 4)}

    _, out_kwargs = manager.language_model_pre_hook(SimpleNamespace(), (), kwargs)

    assert "position_ids" not in out_kwargs
    assert updates["position_preserve_reason"] == "missing_pending_position_info"
    assert updates["position_preserve_applied"] is False
    assert updates["pruning_info_recorded"] is False
    assert updates["position_preserve_original_visual_tokens"] == 8
    assert updates["position_preserve_kept_visual_tokens"] == 4
    assert updates["position_preserve_original_seq_len"] == 12


def main() -> None:
    tests = [
        test_compat_aliases,
        test_method_profile_surface,
        test_none_and_uniform_baselines,
        test_robot_geo_acgtp_v2_fast_selector,
        test_functional_quota_static_internal_backend,
        test_functional_quota_non_fast_avoids_cpu_list_sync,
        test_functional_quota_non_fast_preserves_candidate_order,
        test_internal_decode_bypass_reuses_prefill_pruned_cache,
        test_projector_position_preserve_pre_hook,
        test_projector_position_preserve_missing_pending_is_visible,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("BEHAVIOR_REGRESSION_OK")


if __name__ == "__main__":
    main()
