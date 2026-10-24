"""Shared data model: one Observation = one fact about one resource from one evidence item."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# resource prefix -> (system_id, data_category)
RESOURCE_KINDS = {
    "account": ("account_service", "USER_PROFILE"),
    "search": ("search_service", "SEARCH_INDEX_ENTRY"),
    "analytics": ("analytics_platform", "ANALYTICS_PROFILE"),
    "support": ("support_platform", "SUPPORT_MESSAGE"),
    "export": ("export_service", "USER_EXPORT"),
    "data_flow": ("integration_service", "DATA_FLOW"),
    "backup": ("backup_service", "BACKUP_ARCHIVE"),
    "audit_record": ("security_service", "SECURITY_AUDIT_LOG"),
    "tombstone": ("identity_service", "ACCOUNT_TOMBSTONE"),
    "legal_hold": ("legal_service", "LEGAL_HOLD_RECORD"),
    "marketing": ("marketing_platform", "MARKETING_PROFILE"),
    "cache": ("cache_service", "CACHE_ENTRY"),
    "email_delivery": ("email_delivery_service", "EMAIL_DELIVERY_LOG"),
    "object": ("object_storage", "SUPPORT_ATTACHMENT"),
    "telemetry": ("telemetry_service", "PRODUCT_TELEMETRY"),
}

RESOURCE_RE = re.compile(
    r"\b(account|search|analytics|support|export|data_flow|backup|audit_record|tombstone|"
    r"legal_hold|marketing|cache|email_delivery|object|telemetry)_(\d+)\b"
)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
USER_RE = re.compile(r"\buser_\d+\b")
HASH_RE = re.compile(r"\b[0-9a-f]{64}\b")

NULLS = {None, "", "null", "none", "None", "NULL", "[REDACTED]", "redacted", "no direct identifier"}


def resource_prefix(resource_id: str) -> str:
    m = RESOURCE_RE.fullmatch(resource_id or "")
    return m.group(1) if m else ""


def default_system(resource_id: str) -> str:
    return RESOURCE_KINDS.get(resource_prefix(resource_id), ("", ""))[0]


def default_category(resource_id: str) -> str:
    return RESOURCE_KINDS.get(resource_prefix(resource_id), ("", ""))[1]


def clean(value):
    """Normalise a raw field value; returns None for null-like values."""
    if isinstance(value, (dict, list, tuple, set)):
        return value or None
    if isinstance(value, str):
        value = value.strip()
    if value in NULLS:
        return None
    return value


@dataclass
class Observation:
    evidence_id: str
    observed_at: str
    system_id: str
    resource_id: str
    category: str
    present: bool | None = None          # does the record exist in this system/copy?
    identifiers: dict = field(default_factory=dict)  # direct identifiers still stored in the record
    links: set = field(default_factory=set)          # lower-cased values tying the record to a subject
    flow_enabled: bool | None = None
    restore_allowed: bool | None = None
    status: str | None = None            # normalised label: FOUND, NOT_FOUND, IDENTIFIABLE, ...
    retention_until: str | None = None
    source: str = ""                     # parser / format that produced it
    copy: str = ""                       # e.g. primary, read_replica, failover_replica
    note: str = ""

    def add_link(self, value):
        value = clean(value)
        if isinstance(value, str):
            self.links.add(value.lower())

    def key(self):
        return (self.system_id, self.resource_id)
