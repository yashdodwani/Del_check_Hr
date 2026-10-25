"""Selective vision-model fallback for layout-sensitive or low-quality OCR.

Only invoked when the quality gate fails or layout carries meaning. Default model: ``gemini-3.5-flash-lite``
(Google Gemini API); ``GHOST_VISION_PROVIDER=nvidia`` selects ``nvidia/nemotron-3-nano-omni-30b-a3b-reasoning``.
A provider is used only when its key is set *and* it lists that exact model id — no silent substitution.
Otherwise the caller keeps the OCR text and records ``vision_unavailable``.
"""

from __future__ import annotations

import base64
import io
import os
import re

from ..gemini import GEMINI_MODEL, GeminiClient, gemini_key, image_part
from ..llm import LLMUnavailable, NvidiaClient

PROMPT = (
    "Transcribe this business document/screenshot faithfully. Output one line per table row or field, "
    "keeping each row's cells together separated by ' | '. Copy identifiers (emails, resource ids such as "
    "backup_123, user ids, order numbers, timestamps, statuses) exactly. Do not summarise or infer."
)


VISION_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
DEFAULT_VISION_PROVIDER = "gemini"  # chosen by the provider benchmark (code/evaluation/provider_benchmark.md)
THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_served: dict = {}


def vision_provider() -> str:
    return os.environ.get("GHOST_VISION_PROVIDER", DEFAULT_VISION_PROVIDER).lower()


def vision_model(provider: str | None = None) -> str | None:
    if (provider or vision_provider()) == "gemini":
        return os.environ.get("GEMINI_VISION_MODEL") or GEMINI_MODEL
    return os.environ.get("NVIDIA_VISION_MODEL") or VISION_MODEL


def vision_enabled(provider: str | None = None) -> bool:
    """Key present and the exact model id served by the provider (checked once per process)."""
    provider = provider or vision_provider()
    model = vision_model(provider)
    key = gemini_key() if provider == "gemini" else os.environ.get("NVIDIA_API_KEY")
    if not (model and key):
        return False
    if (provider, model) not in _served:
        client = GeminiClient(timeout=20) if provider == "gemini" else NvidiaClient(timeout=20)
        _served[(provider, model)] = client.model_available(model)
    return _served[(provider, model)]


def strip_reasoning(text: str) -> str:
    """The reasoning model may prepend a <think> block; keep only the transcription."""
    text = THINK_RE.sub("", text or "")
    return text.split("</think>", 1)[-1].strip()


def merge_with_ocr(ocr_lines: list[str], vision_text: str, cutoff: float = 0.75) -> tuple[list[str], list]:
    """Keep Tesseract's row layout, but take identifier spellings from the vision transcription.

    The vision model reads characters reliably but tends to emit one table cell per line, which loses
    the row grouping that ties a record id to its email. Tesseract keeps rows. So each OCR'd email /
    resource id that the vision text does not contain verbatim is snapped to its closest vision token.
    """
    import difflib

    from ..model import EMAIL_RE, RESOURCE_RE, USER_RE
    vocab = set(EMAIL_RE.findall(vision_text)) | {m.group(0) for m in RESOURCE_RE.finditer(vision_text)} \
        | set(USER_RE.findall(vision_text))
    pattern = re.compile(r"[\w.+\-']+@[\w.\-]+|\b[a-z_]+[._,:'][0-9OlI]{1,6}\b|\b[a-z_]+_[0-9OlI]{1,6}\b")
    corrections = []

    def fix(m):
        tok = m.group(0)
        if tok in vocab:
            return tok
        best = difflib.get_close_matches(tok, sorted(vocab), n=1, cutoff=cutoff)
        if best:
            corrections.append((tok, best[0]))
            return best[0]
        return tok
    merged = [pattern.sub(fix, line) for line in ocr_lines]
    # identifiers only the vision model saw are appended so they are not lost (without row context)
    seen = " ".join(merged)
    merged += [line.strip() for line in vision_text.splitlines()
               if line.strip() and any(v in line and v not in seen for v in vocab)]
    return merged, corrections


def transcribe_image(pil_image, client=None, max_side: int = 1600, provider: str | None = None) -> str | None:
    provider = provider or vision_provider()
    model = vision_model(provider)
    if not model:
        return None
    if provider == "gemini":
        client = client or GeminiClient(timeout=60, max_retries=2)
        try:
            text, _ = client.generate(model, [{"text": PROMPT}, image_part(pil_image, max_side)], max_tokens=2048)
            return text.strip() or None
        except LLMUnavailable:
            return None
    client = client or NvidiaClient(timeout=45, max_retries=2)  # normal latency ~10-12 s with thinking off
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
        # thinking disabled (see NvidiaClient.chat): ~10 s per screenshot instead of 1-2.5 min
        text, _ = client.chat(model, messages, temperature=0.0, max_tokens=2048)
        return strip_reasoning(text) or None
    except LLMUnavailable:
        return None
