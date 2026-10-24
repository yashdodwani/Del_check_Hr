"""Log parser covering the observability formats found in the evidence:

* logfmt          ``ts level=INFO service=x trace_id=t event=e k=v``
* JSON payload    ``ts {"service": "x", "event": "e", ...}``
* bracketed       ``ts [INFO] [service] [trace] event: k="v"``
* syslog          ``ts <15>1 node-1 service trace - - k="v"``
* cloud runtime   ``ts requestId=t level=INFO function=svc message="k=\\"v\\" ..."``
* pipe-delimited  ``ts | INFO | service | trace | event | k="v"``

Lines are grouped by trace id so that the lookup key logged on a ``*_started`` line is
attached to the result logged on the matching ``*_completed`` line.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict

from ..model import Observation, RESOURCE_RE, clean
from .common import apply_result_label, make_obs, system_for, to_bool

TS_RE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)\s+(.*)$")
KV_RE = re.compile(r'(\w+)=(?:"((?:[^"\\]|\\.)*)"|([^\s|,"}]+))')
JSON_KV_RE = re.compile(r'"(\w+)"\s*:\s*"((?:[^"\\]|\\.)*)"')
BRACKET_RE = re.compile(r"^\[(\w+)\]\s+\[([\w\-]+)\]\s+\[([\w\-]+)\]\s+(\w+):")
SYSLOG_RE = re.compile(r"^<\d+>\d+\s+\S+\s+([\w\-]+)\s+([\w\-]+)\s+-\s+-")
PIPE_RE = re.compile(r"^\s*\|?\s*(\w+)\s*\|\s*([\w\-]+)\s*\|\s*([\w\-]+)\s*\|\s*(\w+)\s*\|")

LINK_KEYS = ("lookup_key", "match_key", "scan_key", "subject_key", "subject_ref", "correlation_key",
             "matched_value", "lookup_value")


def parse_line(line: str) -> dict | None:
    m = TS_RE.match(line.strip())
    if not m:
        return None
    ts, rest = m.groups()
    rec: dict = {"_ts": ts}
    if rest.lstrip().startswith("{"):
        try:
            rec.update({k: str(v) for k, v in json.loads(rest).items()})
        except json.JSONDecodeError:
            rec.update(dict(JSON_KV_RE.findall(rest)))
    else:
        b = BRACKET_RE.match(rest)
        s = SYSLOG_RE.match(rest)
        p = PIPE_RE.match(rest)
        if b:
            rec.update(level=b.group(1), service=b.group(2), trace_id=b.group(3), event=b.group(4))
        elif s:
            rec.update(service=s.group(1), trace_id=s.group(2))
        elif p:
            rec.update(level=p.group(1), service=p.group(2), trace_id=p.group(3), event=p.group(4))
        flat = rest.replace('\\"', '"')
        for k, q, u in KV_RE.findall(flat):
            if k == "message":
                continue
            rec.setdefault(k, q if q or not u else u)
        if "function" in rec and "service" not in rec:
            rec["service"] = rec["function"]
        if "requestId" in rec and "trace_id" not in rec:
            rec["trace_id"] = rec["requestId"]
    return rec


def _present_from_rows(rec):
    rows = rec.get("rows")
    status = rec.get("http_status")
    if rows is not None:
        try:
            return int(rows) > 0
        except ValueError:
            pass
    if status is not None:
        return status.startswith("2")
    return None


def _interpret(rec: dict, ctx: dict, evidence_id: str) -> Observation | None:
    """Turn one completed/result log line (+ its trace context) into an Observation."""
    rid = rec.get("resource_id")
    if not rid or not RESOURCE_RE.fullmatch(rid):
        return None
    ev = rec.get("event", "")
    svc = rec.get("service", "")
    fields = {**ctx, **rec}
    ts = rec["_ts"]

    def obs(**kw):
        o = make_obs(evidence_id, ts, rid, fields, system_id=system_for(rid, svc), source="log", **kw)
        # the lookup key that produced a hit is itself a direct identifier stored in the record
        return o

    if ev == "record_lookup":  # generic operational lookup line
        o = make_obs(evidence_id, ts, rid, {"subject_reference_link": None}, system_id=rec.get("system") or system_for(rid, svc),
                     source="log")
        o.add_link(rec.get("subject_reference"))
        apply_result_label(o, rec.get("result"), rec.get("record_type"))
        if o.status == "IDENTIFIABLE" and rec.get("subject_reference"):
            o.identifiers = {"email": rec["subject_reference"]}
        return o
    if ev.startswith("profile_lookup"):
        if not ev.endswith("completed"):
            return None
        o = obs(present=_present_from_rows(rec))
        return o
    if ev.startswith("index_query"):
        if not ev.endswith("completed"):
            return None
        res = (rec.get("result") or "").lower()
        o = obs(present=res == "hit" if res else _present_from_rows(rec))
        if o.present and not o.identifiers:
            key = clean(ctx.get("match_key") or ctx.get("lookup_key"))
            if key:
                o.identifiers = {"email": key}
        return o
    if ev in ("analytics_profile_read_completed", "analytics_profile_read"):
        o = obs(present=True)
        di = rec.get("direct_identifiers")
        if di is not None and di != "0" and not o.identifiers:
            o.identifiers = {"email": ctx.get("lookup_key") or "unknown"}
        return o
    if ev == "analytics_identity_scan":
        o = obs(present=True)
        o.identifiers = {"email": rec.get("matched_value")}
        return o
    if ev == "contact_lookup_completed":
        o = obs(present=_present_from_rows(rec), category="SECONDARY_IDENTIFIER")
        if rec.get("correlation_key"):
            o.identifiers = {"email_hash": rec["correlation_key"]}
        return o
    if ev == "support_ticket_read_completed":
        return obs(present=True)
    if ev == "message_pii_scan":
        o = obs(present=True, category="SUPPORT_MESSAGE")
        o.identifiers = {"email": rec.get("matched_value")}
        return o
    if ev == "message_read":
        o = obs(present=True, category="SUPPORT_MESSAGE")
        text = clean(rec.get("message_text"))
        o.identifiers = {k: v for k, v in o.identifiers.items() if k != "user_ref"}
        if text:
            o.identifiers["message_text"] = text
        return o
    if ev == "attachment_pii_scan":
        o = obs(present=True, category="SUPPORT_ATTACHMENT")
        o.identifiers = {"email": rec.get("matched_value")}
        return o
    if ev in ("contact_query_completed",):
        return obs(present=_present_from_rows(rec))
    if ev == "deletion_marker_check":
        found = to_bool(rec.get("marker_found"))
        return obs(present=None if found is None else not found)
    if ev in ("connector_config_read_completed", "connector_heartbeat", "connector_config_loaded"):
        o = obs(present=True)
        o.flow_enabled = to_bool(rec.get("enabled"))
        return o
    if ev == "restore_policy_read_completed":
        o = obs(present=to_bool(rec.get("archive_present")))
        o.restore_allowed = to_bool(rec.get("restore_allowed"))
        return o
    if ev == "restore_authorization_completed":
        o = obs(present=True)
        o.restore_allowed = to_bool(rec.get("decision"))
        return o
    if ev == "restore_candidate_evaluated":
        o = obs(present=True)
        eligible, blocked = to_bool(rec.get("eligible")), to_bool(rec.get("deletion_block"))
        o.restore_allowed = bool(eligible) and not bool(blocked)
        return o
    if ev in ("export_download_check_completed", "export_metadata_read_completed"):
        return obs(present=to_bool(rec.get("available")))
    if ev == "audit_record_read_completed":
        return obs(present=_present_from_rows(rec) if rec.get("rows") else True)
    if ev == "retained_order_loaded":
        return obs(present=True)
    return None


def parse_log(text: str, evidence_id: str) -> list[Observation]:
    recs = [r for r in (parse_line(l) for l in text.splitlines()) if r]
    ctx_by_trace: dict = defaultdict(dict)
    for r in recs:
        t = r.get("trace_id")
        if not t:
            continue
        for k in LINK_KEYS:
            if r.get(k):
                ctx_by_trace[t].setdefault(k, r[k])
    out = []
    for r in recs:
        ctx = ctx_by_trace.get(r.get("trace_id"), {}) if r.get("trace_id") else {}
        o = _interpret(r, ctx, evidence_id)
        if o:
            out.append(o)
    return out
