"""
timer.py

Lightweight wall-clock timer context manager with optional CUDA synchronization.
Use this instead of ad-hoc time.time() / time.perf_counter() calls throughout
the evaluation pipeline.

Usage:
    from utils.timer import Timer

    with Timer() as t:
        result = model.forward(...)
    print(f"Elapsed: {t.elapsed_ms:.2f} ms")

    # With manual record:
    with Timer(sync_cuda=True) as t:
        ...
    profiler.record("my_stage", t.elapsed_ms)
"""

import time as _time
from typing import Optional

import torch


class Timer:
    """
    Context manager that measures wall-clock time in milliseconds.

    Args:
        sync_cuda: If True, calls torch.cuda.synchronize() before and after
                   the timed block. Required for accurate GPU operation timing.
                   Set to False for CPU-only operations (e.g. image preprocessing).
        name: Optional name for the timer (for debugging).
    """

    __slots__ = ("start", "end", "elapsed_ms", "sync_cuda", "name")

    def __init__(self, sync_cuda: bool = True, name: Optional[str] = None):
        self.sync_cuda = sync_cuda
        self.name = name
        self.start: float = 0.0
        self.end: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "Timer":
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = _time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.end = _time.perf_counter()
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed_ms = (self.end - self.start) * 1000.0

    def __repr__(self) -> str:
        suffix = f" ({self.name})" if self.name else ""
        return f"Timer{ suffix}: {self.elapsed_ms:.3f} ms"


class NullTimer:
    """
    A no-op timer for environments where CUDA is not available.
    All operations are no-ops and elapsed_ms always returns 0.0.
    """

    __slots__ = ("start", "end", "elapsed_ms", "sync_cuda", "name")

    def __init__(self, sync_cuda: bool = True, name: Optional[str] = None):
        self.name = name
        self.start: float = 0.0
        self.end: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "NullTimer":
        return self

    def __exit__(self, *args) -> None:
        pass

    def __repr__(self) -> str:
        return f"NullTimer(name={self.name})"


def get_timer(sync_cuda: bool = True, name: Optional[str] = None) -> "Timer":
    """Factory that returns a real Timer if CUDA is available, otherwise a NullTimer."""
    if sync_cuda and torch.cuda.is_available():
        return Timer(sync_cuda=True, name=name)
    return NullTimer(sync_cuda=False, name=name)
