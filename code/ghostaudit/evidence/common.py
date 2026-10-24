"""Helpers shared by the format-specific parsers."""

from __future__ import annotations

from ..model import Observation, clean, default_category, default_system, resource_prefix

IDENTIFIER_FIELDS = {
    # field name -> identifier kind
    "email": "email", "customer_email": "email", "owner_email": "email", "requester_email": "email",
    "uploaded_by": "email", "subject_selector": "email",
    "phone": "phone", "owner_phone": "phone", "requester_phone": "phone",
    "full_name": "name", "name": "name", "display_name": "name", "requester_name": "name",
    "email_hash": "email_hash",
    "customer_reference": "user_ref", "user_id": "user_ref", "subject_reference": "user_ref",
    "subject_ref": "user_ref",
}

LINK_FIELDS = set(IDENTIFIER_FIELDS) | {
    "lookup_key", "match_key", "scan_key", "subject_key", "correlation_key", "matched_value",
    "lookup_value", "query", "selector",
}

TRUE = {"true", "1", "yes", "y", "allowed", "allow", "enabled", "active", "on"}
FALSE = {"false", "0", "no", "n", "denied", "deny", "disabled", "stopped", "off", "blocked"}


def to_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    v = str(value).strip().lower()
    if v in TRUE:
        return True
    if v in FALSE:
        return False
    return None


def identifiers_from(fields: dict) -> dict:
    """Direct identifiers present (non-null) in a record's fields."""
    out = {}
    for k, v in fields.items():
        kind = IDENTIFIER_FIELDS.get(k)
        v = clean(v)
        if kind and isinstance(v, str):
            out.setdefault(kind, v)
    return out


def make_obs(evidence_id, observed_at, resource_id, fields=None, *, system_id=None, category=None,
             present=None, source="", copy="", status=None) -> Observation:
    fields = fields or {}
    obs = Observation(
        evidence_id=evidence_id,
        observed_at=observed_at,
        system_id=system_id or default_system(resource_id),
        resource_id=resource_id,
        category=category or default_category(resource_id),
        present=present,
        source=source,
        copy=copy,
        status=status,
    )
    obs.identifiers = identifiers_from(fields)
    for k, v in fields.items():
        if k in LINK_FIELDS:
            obs.add_link(v)
    if "retention_until" in fields or "retained_until" in fields:
        obs.retention_until = clean(fields.get("retention_until") or fields.get("retained_until"))
    return obs


def apply_result_label(obs: Observation, result: str | None, record_type: str | None = None):
    """Map the generic operational result vocabulary (FOUND, NOT_FOUND, IDENTIFIABLE, ...) onto state."""
    if record_type:
        obs.category = record_type.strip().upper()
    if not result:
        return obs
    r = result.strip().upper().replace(" ", "_")
    obs.status = r
    if r in ("NOT_FOUND", "ABSENT", "DELETED_CONFIRMED", "MISSING"):
        obs.present = False
    elif r in ("FOUND", "PRESENT", "IDENTIFIABLE", "EXPOSED"):
        obs.present = True
    elif r in ("ANONYMIZED", "REDACTED"):
        obs.present = True
        obs.identifiers = {k: v for k, v in obs.identifiers.items() if k == "user_ref" and r == "REDACTED"}
    elif r == "ACTIVE":
        obs.present = True
        if obs.category == "DATA_FLOW":
            obs.flow_enabled = True
        elif obs.category == "BACKUP_ARCHIVE":
            obs.restore_allowed = True
    elif r in ("DISABLED", "STOPPED", "PAUSED"):
        obs.present = True
        obs.flow_enabled = False
    elif r in ("BLOCKED", "RESTORE_BLOCKED"):
        obs.present = True
        obs.restore_allowed = False
    return obs


def system_for(resource_id: str, hint: str | None) -> str:
    """Prefer the canonical system for the resource prefix; fall back to a hint from the evidence."""
    return default_system(resource_id) or (hint or "")


__all__ = ["make_obs", "apply_result_label", "to_bool", "identifiers_from", "system_for",
           "resource_prefix", "clean"]
