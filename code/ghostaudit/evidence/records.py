"""Parsers for structured evidence: JSON exports, SQL snapshots and plain-text records."""

from __future__ import annotations

import json
import re
import sqlite3

from ..model import EMAIL_RE, Observation, RESOURCE_RE, clean
from .common import apply_result_label, make_obs, to_bool

PHONE_RE = re.compile(r"\+\d[\d ()\-]{6,}\d")

# collection name -> data category override (default comes from the resource prefix)
COLLECTION_CATEGORY = {
    "ticket_attachments": "SUPPORT_ATTACHMENT",
    "support_conversations": "SUPPORT_MESSAGE",
    "tickets": "SUPPORT_MESSAGE",
    "support_records": "SUPPORT_MESSAGE",
}


def _flatten(record: dict) -> dict:
    """Lift nested ``document`` / ``profile`` dicts so identifier fields are visible."""
    out = {}
    for k, v in record.items():
        if isinstance(v, dict) and k in ("document", "profile"):
            out.update(v)
        else:
            out[k] = v
    return out


def record_to_obs(evidence_id: str, observed_at: str, collection: str, record: dict,
                  *, system_hint: str = "", copy: str = "", source: str = "json") -> Observation | None:
    """Convert one stored record (bundle / replica row / export entry) into an Observation."""
    if "record_type" in record and "result" in record:  # generic operational history row
        rid = record.get("resource_id", "")
        if not RESOURCE_RE.fullmatch(rid):
            return None
        o = make_obs(evidence_id, record.get("recorded_at") or observed_at, rid, {},
                     system_id=record.get("system") or None, source=source, copy=copy)
        ref = record.get("subject_reference") or record.get("lookup_value")
        o.add_link(ref)
        apply_result_label(o, record.get("result"), record.get("record_type"))
        if o.status in ("IDENTIFIABLE", "FOUND") and ref and "@" in str(ref):
            o.identifiers = {"email": ref}
        return o

    rec = _flatten(record)
    rid = rec.get("resource_id", "")
    if not RESOURCE_RE.fullmatch(rid or ""):
        return None
    ts = rec.get("observed_at") or observed_at
    category = COLLECTION_CATEGORY.get(collection)
    present = True
    if "status_code" in rec:
        present = str(rec["status_code"]).startswith("2")
    if "download_available" in rec:
        present = bool(to_bool(rec["download_available"]))
    if "available" in rec:
        present = bool(to_bool(rec["available"]))
    if "hits" in rec:  # search index query export
        docs = (rec.get("hits") or {}).get("documents") or []
        present = bool(docs) and int((rec.get("hits") or {}).get("total", len(docs))) > 0
        if docs:
            rec = {**rec, **docs[0]}
    if str(rec.get("account_status", "")).upper() in ("DELETED", "PURGED") and rid.startswith("account_"):
        present = False

    fields = dict(rec)
    o = make_obs(evidence_id, ts, rid, fields, category=category, present=present, source=source, copy=copy,
                 system_id=None)
    if rid.startswith("tombstone_"):
        o.identifiers.pop("user_ref", None)  # the stable reference is permitted on a tombstone
        o.add_link(rec.get("subject_reference"))
    if "message_text" in rec or "message_excerpt" in rec:
        text = clean(rec.get("message_text") or rec.get("message_excerpt"))
        if text:
            for e in EMAIL_RE.findall(text):
                o.identifiers.setdefault("email", e)
                o.add_link(e)
            for p in PHONE_RE.findall(text):
                o.identifiers.setdefault("phone", p)
    if "enabled" in rec:
        o.flow_enabled = to_bool(rec["enabled"])
    if "restore_allowed" in rec:
        o.restore_allowed = to_bool(rec["restore_allowed"])
    if rid.startswith("legal_hold_"):
        o.status = "ACTIVE_HOLD"
    return o


# --------------------------------------------------------------------------- JSON

def parse_json(text: str, evidence_id: str, captured_at: str) -> tuple[list[Observation], set]:
    """Returns (observations, scope). ``scope`` lists systems fully enumerated by a snapshot."""
    data = json.loads(text)
    out: list[Observation] = []
    scope: set = set()
    if "snapshots" in data:  # records-assurance snapshot bundle
        for snap in data["snapshots"]:
            ts = snap.get("captured_at") or captured_at
            store = snap.get("data_store", "")
            coll = snap.get("collection", "")
            if coll != "operational_record_history":
                scope.add(store)
            for r in snap.get("records", []):
                o = record_to_obs(evidence_id, ts, coll, r, system_hint=store, source="snapshot_bundle")
                if o:
                    out.append(o)
    elif "service_exports" in data:  # per-service readback export
        ts0 = data.get("exported_at") or captured_at
        for svc, colls in data["service_exports"].items():
            # a readback export only covers the records it queried, so it implies no absence
            for coll, recs in colls.items():
                for r in recs:
                    o = record_to_obs(evidence_id, r.get("observed_at") or ts0, coll, r, system_hint=svc,
                                      source="service_export")
                    if o:
                        out.append(o)
    elif "records" in data:
        ts0 = data.get("captured_at") or data.get("exported_at") or captured_at
        coll = data.get("collection", "")
        for r in data["records"]:
            o = record_to_obs(evidence_id, ts0, coll, r, source="json_records")
            if o:
                out.append(o)
    return out, scope


# --------------------------------------------------------------------------- SQL

SQL_CAPTURE_RE = re.compile(r"Captured at:\s*(\S+)|snapshot_completed_at=(\S+)")

TYPED_TABLES = {
    "customer_profiles": "customer_profiles",
    "customer_search_documents": "customer_search_documents",
    "customer_analytics_profiles": "customer_analytics_reports",
    "support_records": "support_records",
    "customer_exports": "customer_export_jobs",
    "connector_subscriptions": "connector_configurations",
    "backup_restore_entries": "backup_catalog",
    "security_audit_records": "security_audit_events",
}


def parse_sql(text: str, evidence_id: str, captured_at: str) -> tuple[list[Observation], set]:
    """Load the dump into an in-memory SQLite database and read every table as a data copy."""
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript(text)
    out: list[Observation] = []
    scope: set = set()
    tables = [r[0] for r in con.execute("select name from sqlite_master where type='table'")]
    if "snapshot_scope" in tables:
        scope.update(r[0] for r in con.execute("select origin_system from snapshot_scope"))
    for t in tables:
        if t == "snapshot_scope":
            continue
        cols = [r[1] for r in con.execute(f'pragma table_info("{t}")')]
        copy = "primary"
        for suffix in ("_primary", "_read_replica", "_failover_replica", "_reporting_replica"):
            if t.endswith(suffix):
                copy = suffix[1:]
        if "_replica_" in t:
            copy = "replica_" + t.split("_replica_", 1)[1]
        for row in con.execute(f'select * from "{t}"'):
            row = dict(row)
            ts = captured_at
            if "record_json" in cols:
                rec = json.loads(row["record_json"])
                rec.setdefault("resource_id", row["resource_id"])
                o = record_to_obs(evidence_id, ts, row.get("collection_name", ""), rec,
                                  system_hint=row.get("origin_system", ""), copy=copy, source="sql_snapshot")
                if row.get("origin_system"):
                    scope.add(row["origin_system"])
            else:
                coll = TYPED_TABLES.get(t, t)
                o = record_to_obs(evidence_id, ts, coll, row, copy=copy, source="sql_snapshot")
                if o and o.system_id:
                    scope.add(o.system_id)
            if o:
                o.observed_at = captured_at  # a snapshot shows every row as of its capture time
                out.append(o)
    con.close()
    return out, scope


# --------------------------------------------------------------------------- plain-text records

TEXT_ROW_RE = re.compile(
    r"^-\s*(\S+)\s*\|\s*(\w+)\s*\|\s*(\w+)\s*\|\s*([A-Z_]+)\s*\|\s*([A-Z_]+)\s*\|\s*(.+?)\s*$")


def parse_text_records(text: str, evidence_id: str, captured_at: str) -> list[Observation]:
    """Pipe-delimited handover rows: ``- ts | system | resource | TYPE | RESULT | subject``."""
    out = []
    for line in text.splitlines():
        m = TEXT_ROW_RE.match(line.strip())
        if not m:
            continue
        ts, system, rid, rtype, result, subject = m.groups()
        if not RESOURCE_RE.fullmatch(rid):
            continue
        o = make_obs(evidence_id, ts, rid, {}, system_id=system, source="text_record")
        o.add_link(subject)
        apply_result_label(o, result, rtype)
        if o.status in ("IDENTIFIABLE", "FOUND") and "@" in subject:
            o.identifiers = {"email": subject}
        out.append(o)
    return out
