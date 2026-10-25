"""Minimal Google Gemini API client (generateContent REST) with retries, timeouts and usage accounting.

The key is read from the environment (``GEMINI_API_KEY``; ``Gemini_API_KEY`` / ``GOOGLE_API_KEY`` are
accepted too) and is sent only in the ``x-goog-api-key`` header — never in URLs or logs.
Thinking is set to ``minimal`` because classification/transcription does not need it (and it costs
latency + tokens).
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request

from .llm import USAGE, LLMUnavailable

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_MODEL = "gemini-3.5-flash-lite"


def gemini_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("Gemini_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def image_part(pil_image, max_side: int = 1600) -> dict:
    img = pil_image.convert("RGB")
    scale = min(1.0, max_side / max(img.size))
    if scale < 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(buf.getvalue()).decode()}}


class _Throttle:
    """Process-wide request pacing for the free tier (``GEMINI_RPM`` requests/minute, 0 = unlimited)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self):
        rpm = float(os.environ.get("GEMINI_RPM", "14"))
        if rpm <= 0:
            return
        with self.lock:
            now = time.monotonic()
            slot = max(now, self.next_at)
            self.next_at = slot + 60.0 / rpm
        time.sleep(max(0.0, slot - now))


THROTTLE = _Throttle()


def _retry_delay(err_body: bytes) -> float:
    m = re.search(rb'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', err_body or b"")
    return float(m.group(1)) if m else 10.0


class GeminiClient:
    def __init__(self, api_key=None, timeout=30.0, max_retries=3):
        self.api_key = api_key or gemini_key()
        self.timeout = timeout
        self.max_retries = max_retries
        self._models = None

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _request(self, method, path, body=None):
        if not self.api_key:
            raise LLMUnavailable("GEMINI_API_KEY is not set")
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(BASE_URL + path, data=data, method=method,
                                     headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())

    def list_models(self) -> list[str]:
        if self._models is None:
            data = self._request("GET", "/models?pageSize=1000")
            self._models = sorted(m["name"].split("/", 1)[-1] for m in data.get("models", []))
        return self._models

    def model_available(self, model: str) -> bool:
        try:
            return model in self.list_models()
        except Exception:  # noqa: BLE001
            return False

    def generate(self, model, parts, *, system=None, schema=None, max_tokens=400, temperature=0.0):
        """Returns (text, usage). ``parts`` is a list of Gemini content parts (text / inlineData)."""
        gc = {"temperature": temperature, "maxOutputTokens": max_tokens, "thinkingConfig": {"thinkingLevel": "minimal"}}
        if schema:
            gc["responseMimeType"] = "application/json"
            gc["responseSchema"] = schema
        body = {"contents": [{"role": "user", "parts": parts}], "generationConfig": gc}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        last = None
        rate_limited = 0
        attempt = 0
        while attempt < self.max_retries:
            THROTTLE.wait()
            t0 = time.monotonic()
            try:
                resp = self._request("POST", f"/models/{model}:generateContent", body)
                um = resp.get("usageMetadata") or {}
                out_tokens = um.get("candidatesTokenCount", 0) + um.get("thoughtsTokenCount", 0)
                USAGE.record(model, ok=True, input_tokens=um.get("promptTokenCount", 0), output_tokens=out_tokens,
                             latency=time.monotonic() - t0)
                cand = (resp.get("candidates") or [{}])[0]
                text = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts", [])
                               if not p.get("thought"))
                if not text.strip():
                    raise LLMUnavailable(f"empty response (finishReason={cand.get('finishReason')})")
                return text, um
            except urllib.error.HTTPError as e:
                USAGE.record(model, ok=False, latency=time.monotonic() - t0)
                last = f"HTTP {e.code}"
                if e.code == 429 and rate_limited < 6:
                    # quota per minute: wait as instructed by the server; does not consume an attempt
                    rate_limited += 1
                    time.sleep(min(60.0, _retry_delay(e.read()) + 1.0))
                    continue
                if e.code == 400 and attempt == 0 and "thinkingConfig" in gc:
                    gc.pop("thinkingConfig")  # model without thinking controls
                    continue
                if e.code in (400, 401, 403, 404):
                    break
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
                USAGE.record(model, ok=False, latency=time.monotonic() - t0)
                last = type(e).__name__
            time.sleep(min(8.0, 0.5 * 2 ** attempt))
            attempt += 1
        raise LLMUnavailable(f"gemini generate failed after retries: {last}")
