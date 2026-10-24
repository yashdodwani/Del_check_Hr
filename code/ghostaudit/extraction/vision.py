"""Selective vision-model fallback for layout-sensitive or low-quality OCR.

Only invoked when the quality gate fails or layout carries meaning. The model is
``nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`` on the NVIDIA API (override with ``NVIDIA_VISION_MODEL``).
It is used only when ``NVIDIA_API_KEY`` is set *and* the endpoint lists that exact model id — no silent
substitution. Otherwise the caller keeps the OCR text and records ``vision_unavailable``.
"""

from __future__ import annotations

import base64
import io
import os
import re

from ..llm import LLMUnavailable, NvidiaClient

PROMPT = (
    "Transcribe this business document/screenshot faithfully. Output one line per table row or field, "
    "keeping each row's cells together separated by ' | '. Copy identifiers (emails, resource ids such as "
    "backup_123, user ids, order numbers, timestamps, statuses) exactly. Do not summarise or infer."
)


VISION_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_served: dict = {}


def vision_model() -> str | None:
    return os.environ.get("NVIDIA_VISION_MODEL") or VISION_MODEL


def vision_enabled() -> bool:
    """Key present and the exact model id served by the configured endpoint (checked once per process)."""
    model = vision_model()
    if not (model and os.environ.get("NVIDIA_API_KEY")):
        return False
    if model not in _served:
        _served[model] = NvidiaClient(timeout=20).model_available(model)
    return _served[model]


def strip_reasoning(text: str) -> str:
    """The reasoning model may prepend a <think> block; keep only the transcription."""
    text = THINK_RE.sub("", text or "")
    return text.split("</think>", 1)[-1].strip()


def transcribe_image(pil_image, client: NvidiaClient | None = None, max_side: int = 1600) -> str | None:
    model = vision_model()
    if not model:
        return None
    client = client or NvidiaClient(timeout=90)
    img = pil_image.convert("RGB")
    scale = min(1.0, max_side / max(img.size))
    if scale < 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    messages = [{"role": "user", "content": [
        {"type": "text", "text": PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ]}]
    try:
        # reasoning models spend tokens thinking before answering, so allow a larger budget
        text, _ = client.chat(model, messages, temperature=0.0, max_tokens=4096)
        return strip_reasoning(text) or None
    except LLMUnavailable:
        return None
