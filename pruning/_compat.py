"""Lazy compatibility aliases for historical top-level pruning modules.

The implementation has moved into functional packages such as ``methods`` and
``signals``. These aliases keep imports like ``pruning.depth_edge`` working
without keeping one wrapper file per old module in the package root.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
import types
from typing import Dict, Optional


COMPAT_MODULE_ALIASES: Dict[str, str] = {
    "acgtp_dynamic_controller": "signals.temporal",
    "acgtp_history": "signals.temporal",
    "acgtp_v1_selector": "legacy.acgtp_v1",
    "acgtp_v2_fast_selector": "methods.acgtp_v2",
    "acgtp_v2_full_selector": "methods.acgtp_v2",
    "acgtp_v2_selectors": "methods.acgtp_v2",
    "action_constraint": "signals.action",
    "attention_relevance": "signals.semantic",
    "branch_budget_selector": "legacy.branch_budget",
    "contact_ring": "signals.spatial",
    "depth_edge": "signals.spatial",
    "geometry_cache": "signals.robot",
    "hook_diagnostics": "runtime.diagnostics",
    "hook_fast_runtime": "runtime.fast",
    "hook_geometry": "runtime.geometry",
    "hook_legacy_runtime": "legacy.runtime",
    "hook_utils": "core.utils",
    "hybrid_selectors": "legacy.hybrid",
    "internal_functional_quota": "methods.functional_quota",
    "internal_pruning": "internal.backend",
    "internal_quota_config": "internal.quota_config",
    "internal_uniform_pruning": "internal.uniform",
    "legacy_selectors": "legacy",
    "legacy_strategies": "legacy.strategies",
    "metrics": "core.metrics",
    "motion_corridor": "signals.action",
    "post_pruning": "runtime.post",
    "robot_geometry": "signals.robot",
    "robot_state": "signals.robot",
    "scene_layout": "signals.spatial",
    "scheduler": "signals.temporal",
    "selector_core": "methods.baselines",
    "selector_registry": "methods.registry",
    "selector_utils": "methods.utils",
    "semantic_anchors": "signals.semantic",
    "static_scene_cache": "signals.robot",
    "temporal_geometry": "signals.temporal",
    "token_geometry": "signals.robot",
    "validation_tests": "tests.validation_tests",
    "visualization": "core.visualization",
}


class _CompatAliasLoader(importlib.abc.Loader):
    def __init__(self, fullname: str, target: str) -> None:
        self.fullname = fullname
        self.target = target

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> types.ModuleType:
        return types.ModuleType(spec.name)

    def exec_module(self, module: types.ModuleType) -> None:
        spec = module.__spec__
        target_module = importlib.import_module(self.target)
        module.__dict__.update(target_module.__dict__)
        module.__name__ = self.fullname
        module.__package__ = self.fullname.rpartition(".")[0]
        module.__loader__ = self
        module.__spec__ = spec
        module.__compat_target__ = self.target
        module.__all__ = getattr(
            target_module,
            "__all__",
            [name for name in target_module.__dict__ if not name.startswith("_")],
        )


class _CompatAliasFinder(importlib.abc.MetaPathFinder):
    def __init__(self, package_name: str, aliases: Dict[str, str]) -> None:
        self.package_name = package_name
        self.aliases = {
            f"{package_name}.{name}": f"{package_name}.{target}"
            for name, target in aliases.items()
        }

    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: Optional[types.ModuleType] = None,
    ) -> Optional[importlib.machinery.ModuleSpec]:
        target_name = self.aliases.get(fullname)
        if target_name is None:
            return None
        return importlib.machinery.ModuleSpec(
            fullname,
            _CompatAliasLoader(fullname, target_name),
            origin=f"compat-alias:{target_name}",
        )


def install_compat_aliases(package_name: str) -> None:
    for finder in sys.meta_path:
        if isinstance(finder, _CompatAliasFinder) and finder.package_name == package_name:
            return
    sys.meta_path.insert(0, _CompatAliasFinder(package_name, COMPAT_MODULE_ALIASES))


def import_compat_alias(package_name: str, name: str) -> types.ModuleType:
    if name not in COMPAT_MODULE_ALIASES:
        raise KeyError(name)
    return importlib.import_module(f"{package_name}.{name}")

