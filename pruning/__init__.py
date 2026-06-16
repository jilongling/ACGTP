"""External visual-token pruning hooks for OpenVLA inference.

This package is intentionally outside the OpenVLA model implementation. It only
hooks projector output visual tokens before they are concatenated with text.
"""

from ._compat import import_compat_alias, install_compat_aliases

install_compat_aliases(__name__)

from .config import PruningHookConfig
from .hook import VisualTokenPruningHook

__all__ = ["PruningHookConfig", "VisualTokenPruningHook"]


def __getattr__(name: str):
    try:
        return import_compat_alias(__name__, name)
    except KeyError as exc:
        raise AttributeError(name) from exc
