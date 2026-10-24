"""Text-quality assessment and identifier validation for extracted text.

Non-empty text is not treated as success: we look at printable/alphabetic ratios, density,
known extraction artefacts, whether identifiers look well-formed, and (when a vocabulary of
known identifiers is supplied) whether extracted identifiers match real challenge values.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from ..model import EMAIL_RE, RESOURCE_RE, USER_RE

TIMESTAMP_RE = re.compile(r"\b\d{4}-\d\d-\d\d[T ]\d\d:\d\d|\b\d{1,2} (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b")
ARTIFACT_RE = re.compile(r"\(cid:\d+\)|\ufffd|[\x00-\x08\x0b\x0c\x0e-\x1f]")
# OCR confusions seen in UI screenshots: gmall/gmai1, 0/O and 1/l swaps inside ids
SUSPICIOUS_ID_RE = re.compile(r"\b(?:acc0unt|supp0rt|anal ytics|data f1ow)\b|@gmall\.|@gmai1\.|@0utlook\.", re.I)
MALFORMED_RESOURCE_RE = re.compile(r"\b(account|search|analytics|support|export|backup|data_flow|audit_record|"
                                   r"tombstone)[ _\-]?[O0-9lI|]*[OlI|][O0-9lI|]*\b")


@dataclass
class Quality:
    score: float
    usable: bool
    chars: int
    printable_ratio: float
    alpha_ratio: float
    artifacts: int
    identifiers: dict = field(default_factory=dict)
    suspicious: list = field(default_factory=list)
    reasons: list = field(default_factory=list)

    def as_dict(self):
        return self.__dict__.copy()


def find_identifiers(text: str) -> dict:
    return {
        "emails": sorted(set(EMAIL_RE.findall(text))),
        "resource_ids": sorted({m.group(0) for m in RESOURCE_RE.finditer(text)}),
        "user_ids": sorted(set(USER_RE.findall(text))),
        "timestamps": len(TIMESTAMP_RE.findall(text)),
    }


def assess(text: str, *, page_area: float | None = None, expect_identifiers: bool = True,
           mean_confidence: float | None = None, known: "KnownVocabulary | None" = None) -> Quality:
    text = text or ""
    n = len(text)
    reasons = []
    if n == 0:
        return Quality(0.0, False, 0, 0.0, 0.0, 0, {}, [], ["empty"])
    printable = sum(1 for c in text if c.isprintable() or c in "\n\t") / n
    alpha = sum(1 for c in text if c.isalnum()) / n
    artifacts = len(ARTIFACT_RE.findall(text))
    words = re.findall(r"[A-Za-z]{3,}", text)
    ids = find_identifiers(text)
    suspicious = SUSPICIOUS_ID_RE.findall(text)
    malformed = [m.group(0) for m in MALFORMED_RESOURCE_RE.finditer(text) if not RESOURCE_RE.fullmatch(m.group(0))]
    suspicious += malformed

    score = 1.0
    if printable < 0.95:
        score -= 0.4; reasons.append(f"printable_ratio={printable:.2f}")
    if alpha < 0.45:
        score -= 0.2; reasons.append(f"alpha_ratio={alpha:.2f}")
    if artifacts:
        score -= min(0.5, 0.05 * artifacts); reasons.append(f"artifacts={artifacts}")
    if len(words) < 8:
        score -= 0.4; reasons.append(f"few_words={len(words)}")
    if page_area and n / page_area < 0.0005:
        score -= 0.2; reasons.append("low_density")
    if expect_identifiers and not (ids["emails"] or ids["resource_ids"] or ids["user_ids"]):
        score -= 0.3; reasons.append("no_identifiers")
    if suspicious:
        score -= min(0.3, 0.1 * len(suspicious)); reasons.append(f"suspicious_ids={len(suspicious)}")
    if mean_confidence is not None and mean_confidence < 75:
        score -= 0.2; reasons.append(f"low_ocr_confidence={mean_confidence:.0f}")
    if known is not None:
        unknown = [e for e in ids["emails"] if not known.is_known_email(e)]
        if unknown:
            score -= min(0.3, 0.1 * len(unknown)); reasons.append(f"unverified_emails={len(unknown)}")
    score = max(0.0, round(score, 3))
    return Quality(score, score >= 0.6, n, round(printable, 3), round(alpha, 3), artifacts, ids, suspicious, reasons)


class KnownVocabulary:
    """Known identifiers from the participant-facing CSVs, used to validate/correct OCR output."""

    def __init__(self, emails=(), user_ids=(), resource_ids=()):
        self.emails = {e.lower() for e in emails}
        self.user_ids = set(user_ids)
        self.resource_ids = set(resource_ids)
        self._email_list = sorted(self.emails)

    def is_known_email(self, e: str) -> bool:
        return e.lower() in self.emails

    def correct_email(self, e: str, cutoff: float = 0.9) -> tuple[str, bool]:
        """Return (value, corrected?) — snap an OCR'd email to the closest known one if very similar."""
        low = e.lower()
        if low in self.emails or not self._email_list:
            return e, False
        match = difflib.get_close_matches(low, self._email_list, n=1, cutoff=cutoff)
        if match:
            return match[0], True
        return e, False

    def correct_text(self, text: str) -> tuple[str, list]:
        corrections = []

        def fix(m):
            new, changed = self.correct_email(m.group(0))
            if changed:
                corrections.append((m.group(0), new))
            return new
        return EMAIL_RE.sub(fix, text), corrections
