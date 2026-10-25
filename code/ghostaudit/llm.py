"""Shared LLM plumbing: the error type raised by model clients and in-process usage accounting.

Usage (calls, failures, input/output tokens, latency) is recorded per model from the API's own usage
fields and written into run_summary.json → usage_report.md.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


class LLMUnavailable(RuntimeError):
    pass


@dataclass
class ModelUsage:
    calls: int = 0
    failed_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0


@dataclass
class UsageTracker:
    by_model: dict = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, model, *, ok, input_tokens=0, output_tokens=0, latency=0.0):
        with self.lock:
            u = self.by_model.setdefault(model, ModelUsage())
            u.calls += 1
            u.failed_calls += 0 if ok else 1
            u.input_tokens += input_tokens
            u.output_tokens += output_tokens
            u.latency_s += latency

    def as_dict(self):
        return {m: u.__dict__.copy() for m, u in self.by_model.items()}


USAGE = UsageTracker()
