"""
profiler.py

Inference timing profiler for OpenVLA baseline evaluation.
Provides context-manager-based timing for different stages of the inference pipeline,
with proper GPU synchronization support.
"""

import time
from contextlib import contextmanager
from typing import Dict, List, Optional

import torch


class InferenceProfiler:
    """
    Tracks timing for different stages of VLA inference.

    Usage:
        profiler = InferenceProfiler()
        with profiler.profile("model_forward"):
            action = model.predict_action(...)
        print(profiler.get_summary())
    """

    def __init__(self, use_cuda: bool = True) -> None:
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self._records: Dict[str, List[float]] = {}

    def _sync(self) -> None:
        """Synchronize CUDA operations if running on GPU."""
        if self.use_cuda:
            torch.cuda.synchronize()

    def _elapsed_ms(self, start: float, end: float) -> float:
        """Convert elapsed seconds to milliseconds."""
        return (end - start) * 1000.0

    @contextmanager
    def profile(self, name: str):
        """
        Context manager to profile a code block.

        Args:
            name: Identifier for this profiling region.

        Yields:
            The profiler instance itself so callers can annotate values.
        """
        self._sync()
        start = time.perf_counter()
        try:
            yield self
        finally:
            end = time.perf_counter()
            elapsed = self._elapsed_ms(start, end)
            if name not in self._records:
                self._records[name] = []
            self._records[name].append(elapsed)

    def record(self, name: str, milliseconds: float) -> None:
        """Manually record a timing value in milliseconds."""
        if name not in self._records:
            self._records[name] = []
        self._records[name].append(milliseconds)

    def get(self, name: str) -> List[float]:
        """Get all recorded times (in ms) for a given region."""
        return self._records.get(name, [])

    def mean(self, name: str) -> float:
        """Get mean time (in ms) for a given region."""
        times = self.get(name)
        return sum(times) / len(times) if times else 0.0

    def std(self, name: str) -> float:
        """Get standard deviation (in ms) for a given region."""
        times = self.get(name)
        if len(times) < 2:
            return 0.0
        mean_val = sum(times) / len(times)
        variance = sum((t - mean_val) ** 2 for t in times) / len(times)
        return variance ** 0.5

    def max(self, name: str) -> float:
        """Get max time (in ms) for a given region."""
        times = self.get(name)
        return max(times) if times else 0.0

    def min(self, name: str) -> float:
        """Get min time (in ms) for a given region."""
        times = self.get(name)
        return min(times) if times else 0.0

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Return a summary dict with mean/std/min/max for each region."""
        return {
            name: {
                "mean_ms": self.mean(name),
                "std_ms": self.std(name),
                "min_ms": self.min(name),
                "max_ms": self.max(name),
                "count": len(times),
            }
            for name, times in self._records.items()
        }

    def reset(self) -> None:
        """Clear all recorded timings."""
        self._records.clear()


def get_gpu_memory_mb() -> Optional[float]:
    """Get current peak GPU memory usage in MB. Returns None if CUDA is unavailable."""
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def get_gpu_memory_reserved_mb() -> Optional[float]:
    """Get current peak GPU reserved memory in MB. Returns None if CUDA is unavailable."""
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_reserved() / (1024 * 1024)


def reset_gpu_memory_stats() -> None:
    """Reset GPU memory tracking. Call before evaluation loop."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
