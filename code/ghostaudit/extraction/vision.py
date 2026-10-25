"""Selective vision-model fallback for layout-sensitive or low-quality OCR.

Only invoked when the quality gate fails or layout carries meaning. Model: ``gemini-3.5-flash-lite``
(Google Gemini API; ``GEMINI_VISION_MODEL`` overrides). It is used only when the Gemini key is set *and* the
API lists that exact model id — no silent substitution. Otherwise the caller keeps the OCR text and
records ``vision_unavailable``.
"""

from __future__ import annotations

import os
import re

from ..gemini import GEMINI_MODEL, GeminiClient, gemini_key, image_part
from ..llm import LLMUnavailable

PROMPT = (
    "Transcribe this business document/screenshot faithfully. Output one line per table row or field, "
    "keeping each row's cells together separated by ' | '. Copy identifiers (emails, resource ids such as "
    "backup_123, user ids, order numbers, timestamps, statuses) exactly. Do not summarise or infer."
)

_served: dict = {}


def vision_model() -> str:
    return os.environ.get("GEMINI_VISION_MODEL") or GEMINI_MODEL


def vision_enabled() -> bool:
    """Key present and the exact model id served by the API (checked once per process)."""
    model = vision_model()
    if not gemini_key():
        return False
    if model not in _served:
        _served[model] = GeminiClient(timeout=20).model_available(model)
    return _served[model]


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


def transcribe_image(pil_image, client=None, max_side: int = 1600) -> str | None:
    client = client or GeminiClient(timeout=60, max_retries=2)
    try:
        text, _ = client.generate(vision_model(), [{"text": PROMPT}, image_part(pil_image, max_side)],
                                  max_tokens=2048)
        return text.strip() or None
    except LLMUnavailable:
        return None
