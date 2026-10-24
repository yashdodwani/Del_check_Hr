"""Turn extracted PDF / screenshot text into Observations.

Business artifacts rarely state a deletion verdict; they show that a record exists and which
identifiers it carries. Each line mentioning a resource id (or a known display alias of one, e.g. a
statement number or attachment file name taken from the data catalog) becomes a "record is present"
observation carrying the identifiers printed on that line. Document-level identifiers are used as
links when a line carries none (detail panels usually show the account once, away from the row).
"""

from __future__ import annotations

import re

from ..model import EMAIL_RE, RESOURCE_RE, USER_RE, Observation
from .common import make_obs

NEGATIVE_RE = re.compile(r"\b(not found|no record|deleted|purged|absent|404)\b", re.I)
FLOW_OFF_RE = re.compile(r"\b(paused|disabled|stopped|inactive)\b", re.I)
RESTORE_OFF_RE = re.compile(r"\b(restore blocked|blocked|restore disabled|restore denied|deny)\b", re.I)
RETAIN_UNTIL_RE = re.compile(r"retain(?:ed)? until\s+(\d{1,2} \w{3} \d{4}|\d{4}-\d\d-\d\d)", re.I)


OCR_ID_RE = re.compile(r"\b(user|account|search|analytics|support|export|backup|data[ _]flow|audit[ _]record|"
                       r"tombstone|marketing|object|legal[ _]hold)[ _\-—]{1,2}([0-9OoIl]{1,6})\b", re.I)


def normalize_ocr_ids(text: str) -> str:
    """OCR often drops the underscore in ids ('user 17', 'data flow 1121') or swaps O/0, l/1."""
    def fix(m):
        digits = m.group(2).translate(str.maketrans("OoIl", "0011"))
        if not digits.isdigit():
            return m.group(0)
        return m.group(1).lower().replace(" ", "_") + "_" + digits
    return OCR_ID_RE.sub(fix, text)


def observations_from_text(text: str, evidence_id: str, captured_at: str, *, aliases: dict | None = None,
                           origin: str = "") -> list[Observation]:
    aliases = aliases or {}
    text = normalize_ocr_ids(text)
    doc_emails = [e.lower() for e in EMAIL_RE.findall(text)]
    doc_users = sorted(set(USER_RE.findall(text)))
    lines = [l for l in text.splitlines() if l.strip()]
    found: dict = {}

    def touch(rid, line, idx):
        o = found.get(rid)
        if o is None:
            o = make_obs(evidence_id, captured_at, rid, {}, present=True, source=f"document:{origin}")
            found[rid] = o
        window = " ".join(lines[max(0, idx - 1): idx + 2])  # cells wrapped onto neighbouring lines
        line_emails = EMAIL_RE.findall(line)
        for e in line_emails:
            o.add_link(e)
            o.identifiers.setdefault("email", e)
        for u in USER_RE.findall(line):
            o.add_link(u)
        if NEGATIVE_RE.search(line):
            o.present = False
        if o.category == "DATA_FLOW":
            o.flow_enabled = not bool(FLOW_OFF_RE.search(line))
        if o.category == "BACKUP_ARCHIVE":
            o.restore_allowed = not bool(RESTORE_OFF_RE.search(window))
        m = RETAIN_UNTIL_RE.search(window)
        if m:
            o.retention_until = m.group(1)

    for idx, line in enumerate(lines):
        for m in RESOURCE_RE.finditer(line):
            touch(m.group(0), line, idx)
        for alias, rid in aliases.items():
            if not alias or RESOURCE_RE.fullmatch(alias):
                continue
            if USER_RE.fullmatch(alias) and len(doc_users) != 1:
                continue  # a bare user id only identifies the record on a single-subject page
            if re.search(r"(?<![\w.])" + re.escape(alias.lower()) + r"(?![\w])", line.lower()):
                touch(rid, line, idx)

    # documents about one subject: attach doc-level identifiers to records that printed none
    single_subject = len(set(doc_emails)) <= 1 and len(doc_users) <= 1
    for o in found.values():
        if not o.links and single_subject:
            for e in doc_emails:
                o.add_link(e)
            for u in doc_users:
                o.add_link(u)
        if o.identifiers == {} and doc_emails and len(set(doc_emails)) == 1 and o.category not in (
                "SECURITY_AUDIT_LOG", "DATA_FLOW", "BACKUP_ARCHIVE"):
            o.identifiers["email"] = doc_emails[0]
    return list(found.values())
