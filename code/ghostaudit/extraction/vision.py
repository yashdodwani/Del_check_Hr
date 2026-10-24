"""Selective vision-model fallback for layout-sensitive or low-quality OCR.

Only invoked when the quality gate fails or layout carries meaning, and only when a vision model is
configured (``NVIDIA_VISION_MODEL`` + ``NVIDIA_API_KEY``). Otherwise the caller keeps the OCR text and
records ``vision_unavailable`` so the decision is auditable.
"""

from __future__ import annotations

import base64
import io
import os

from ..llm import LLMUnavailable, NvidiaClient

PROMPT = (
    "Transcribe this business document/screenshot faithfully. Output one line per table row or field, "
    "keeping each row's cells together separated by ' | '. Copy identifiers (emails, resource ids such as "
    "backup_123, user ids, order numbers, timestamps, statuses) exactly. Do not summarise or infer."
)


def vision_model() -> str | None:
    return os.environ.get("NVIDIA_VISION_MODEL") or None


def vision_enabled() -> bool:
    return bool(vision_model() and os.environ.get("NVIDIA_API_KEY"))


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
        text, _ = client.chat(model, messages, max_tokens=1500)
        return text
    except LLMUnavailable:
        return None
