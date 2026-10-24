"""Minimal NVIDIA API (OpenAI-compatible) client with bounded retries, timeouts and usage accounting.

Credentials come only from the environment (``NVIDIA_API_KEY``); the key is never logged or printed.
Uses the standard library so no SDK is required.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
INTENT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"


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


class NvidiaClient:
    def __init__(self, api_key=None, base_url=None, timeout=30.0, max_retries=3):
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        self.base_url = (base_url or os.environ.get("NVIDIA_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _request(self, method, path, body=None):
        if not self.api_key:
            raise LLMUnavailable("NVIDIA_API_KEY is not set")
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base_url + path, data=data, method=method, headers={
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())

    def list_models(self) -> list[str]:
        return sorted(m.get("id") for m in self._request("GET", "/models").get("data", []))

    def model_available(self, model: str) -> bool:
        try:
            return model in self.list_models()
        except Exception:  # noqa: BLE001 - availability probe must never crash the pipeline
            return False

    def chat(self, model, messages, *, temperature=0.0, max_tokens=400, response_format=None, extra=None):
        """Returns (content, usage_dict). Retries transient failures with exponential backoff."""
        body = {"model": model, "messages": messages, "temperature": temperature, "top_p": 1.0,
                "max_tokens": max_tokens, "stream": False}
        if response_format:
            body["response_format"] = response_format
        if extra:
            body.update(extra)
        last = None
        for attempt in range(self.max_retries):
            t0 = time.monotonic()
            try:
                resp = self._request("POST", "/chat/completions", body)
                usage = resp.get("usage") or {}
                USAGE.record(model, ok=True, input_tokens=usage.get("prompt_tokens", 0),
                             output_tokens=usage.get("completion_tokens", 0), latency=time.monotonic() - t0)
                content = resp["choices"][0]["message"].get("content") or ""
                return content, usage
            except urllib.error.HTTPError as e:
                USAGE.record(model, ok=False, latency=time.monotonic() - t0)
                last = f"HTTP {e.code}"
                if e.code in (400, 401, 403, 404):
                    if e.code == 400 and response_format and attempt == 0:
                        body.pop("response_format", None)  # endpoint may not support structured output
                        continue
                    break
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
                USAGE.record(model, ok=False, latency=time.monotonic() - t0)
                last = type(e).__name__
            time.sleep(min(8.0, 0.5 * 2 ** attempt))
        raise LLMUnavailable(f"chat failed after retries: {last}")
