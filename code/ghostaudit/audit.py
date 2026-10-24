"""Deletion audit: identity → linked resources → state at audit_at → policy → ordered steps."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .data import Dataset
from .evidence.index import EvidenceIndex
from .identity import IdentityResolver, Subject
from .model import Observation, default_category, default_system, resource_prefix
from .attribution import WorkflowAttribution
from .policy import Policy, order_steps, step

import os
SNAPSHOT_ABSENCE = os.environ.get("GHOST_SNAPSHOT_ABSENCE", "hybrid")

PERMITTED_IDENTIFIERS = {
    # identifiers a retained/redacted record may keep
    "SUPPORT_MESSAGE": {"user_ref"},
    "ACCOUNT_TOMBSTONE": {"user_ref"},
    "SECURITY_AUDIT_LOG": {"user_ref"},
    "EMAIL_DELIVERY_LOG": {"user_ref"},
}


@dataclass
class ResourceState:
    resource_id: str
    system_id: str
    category: str
    evidence_id: str
    captured_at: str
    present: bool | None
    identifiers: dict
    flow_enabled: bool | None = None
    restore_allowed: bool | None = None
    retention_until: str | None = None
    copies: list = field(default_factory=list)
    link_reason: str = ""


@dataclass
class AuditResult:
    case_id: str
    status: str
    steps: list
    user_id: str | None = None
    trace: dict = field(default_factory=dict)


class Auditor:
    def __init__(self, ds: Dataset, index: EvidenceIndex):
        self.ds = ds
        self.index = index
        self.resolver = IdentityResolver(ds.identities, ds.audit_log)
        self.policy = Policy(ds.rules)
        self.activity_by_resource = defaultdict(list)
        for a in ds.activity:
            self.activity_by_resource[a["resource_id"]].append(a)

    # ------------------------------------------------------------------ linking
    def _obs_matches(self, o: Observation, subject: Subject) -> bool:
        if any(subject.matches(v) for v in o.links):
            return True
        return any(isinstance(v, str) and subject.matches(v) for v in o.identifiers.values())

    def linked_resources(self, subject: Subject, cutoff: str) -> tuple[dict, set]:
        """Resources tied to the subject by an identifier match.

        Identity links (which record belongs to whom) are timeless facts, so any evidence file may
        establish them; the *state* of a record is only ever read from evidence captured <= audit_at.
        ``covering_files`` (files <= cutoff that mention the subject) are used for snapshot absence.
        """
        linked: dict = {}
        covering_files = set()
        for f in self.index.files.values():
            for o in f.observations:
                if self._obs_matches(o, subject):
                    linked.setdefault(o.resource_id, f"evidence:{f.file_id}")
                    if f.captured_at <= cutoff:
                        covering_files.add(f.file_id)
        for activity_id, ts, rid, email, user, _exclusive in self.index.catalog:
            if subject.matches(email) or subject.matches(user):
                linked.setdefault(rid, f"catalog:{activity_id}")
        return linked, covering_files

    def prepare(self, requests: list[dict]) -> None:
        """Link every request in the pool up front, then attribute identifier-less workflow activity."""
        self._linked_cache = {}
        pool = []
        for r in requests:
            key = r.get("_key") or r["case_id"]
            uid = self.resolver.resolve_email(r["email"])
            if not uid:
                continue
            linked, _ = self.linked_resources(self.resolver.subject(uid), r["audit_at"])
            self._linked_cache[key] = linked
            pool.append(dict(r, _key=key))
        self.attribution = WorkflowAttribution(self.ds.activity, pool,
                                               {k: set(v) for k, v in self._linked_cache.items()})

    # ------------------------------------------------------------------ state
    def resource_state(self, rid: str, cutoff: str, covering_files: set) -> ResourceState | None:
        obs = self.index.observations(rid, cutoff)
        system = default_system(rid) or (obs[0].system_id if obs else "")
        by_file: dict = defaultdict(list)
        for o in obs:
            by_file[o.evidence_id].append(o)
        # a full snapshot that covers this subject but does not list the record shows it is gone
        for fid in covering_files:
            f = self.index.files[fid]
            if fid not in by_file and system in f.scope:
                by_file[fid].append(Observation(fid, f.captured_at, system, rid, default_category(rid),
                                                present=False, source="snapshot_absence"))
        if not by_file:
            return None
        files = sorted(by_file, key=lambda fid: (self.index.files[fid].captured_at,
                                                 max(o.observed_at for o in by_file[fid])))
        fid = files[-1]
        latest = sorted(by_file[fid], key=lambda o: o.observed_at)
        category = next((o.category for o in latest if o.category and o.category != default_category(rid)),
                        latest[-1].category or default_category(rid))
        # merge copies (primary + replicas) captured in the same file: data survives if any copy has it
        present_vals = [o.present for o in latest if o.present is not None]
        present = any(present_vals) if present_vals else None
        identifiers: dict = {}
        flow = restore = None
        retention = None
        for o in latest:
            if o.present:
                identifiers.update(o.identifiers)
            flow = o.flow_enabled if o.flow_enabled is not None else flow
            restore = o.restore_allowed if o.restore_allowed is not None else restore
            retention = o.retention_until or retention
        return ResourceState(rid, system, category, fid, self.index.files[fid].captured_at, present, identifiers,
                             flow, restore, retention, copies=sorted({o.copy for o in latest if o.copy}))

    # ------------------------------------------------------------------ decision
    def decide(self, st: ResourceState, audit_at: str):
        """Returns (findings_ops, remediation_ops, verification_ops) for one resource."""
        cat = st.category
        if cat == "DATA_FLOW":
            if st.flow_enabled:
                return ["REPORT_ACTIVE_DATA_FLOW"], ["DISABLE_DATA_FLOW"], ["VERIFY_DATA_FLOW_DISABLED"]
            if st.flow_enabled is False:
                return [], [], ["VERIFY_DATA_FLOW_DISABLED"]
            return [], [], []
        if cat == "BACKUP_ARCHIVE":
            if st.present is False:
                return [], [], []
            if st.restore_allowed:
                return ["REPORT_RESTORE_RISK"], ["BLOCK_RESTORE"], ["VERIFY_RESTORE_BLOCKED"]
            if st.restore_allowed is False:
                return [], [], ["VERIFY_RESTORE_BLOCKED"]
            return [], [], []
        action = self.policy.action(cat)
        if st.present is None:
            return [], [], []
        if action == "RETAIN":
            if st.present:
                return [], [], ["VERIFY_RECORD_PRESENT"]
            return [], [], []
        if action == "DELETE":
            if st.present:
                return ["REPORT_REMAINING_RECORD"], ["DELETE_RECORD"], ["VERIFY_RECORD_ABSENT"]
            return [], [], ["VERIFY_RECORD_ABSENT"]
        if action in ("REDACT", "ANONYMIZE"):
            if not st.present:
                return [], [], ["VERIFY_RECORD_ABSENT"]
            allowed = PERMITTED_IDENTIFIERS.get(cat, set()) if action == "REDACT" else set()
            exposed = {k: v for k, v in st.identifiers.items() if k not in allowed}
            verify = "VERIFY_RECORD_REDACTED" if action == "REDACT" else "VERIFY_RECORD_ANONYMIZED"
            if exposed:
                remed = "REDACT_RECORD" if action == "REDACT" else "ANONYMIZE_RECORD"
                return ["REPORT_EXPOSED_PII"], [remed], [verify]
            return [], [], [verify]
        return [], [], []

    # ------------------------------------------------------------------ main
    def audit(self, case: dict, *, authorized: bool) -> AuditResult:
        if not authorized:
            return AuditResult(case["case_id"], "NOT_APPLICABLE", [], trace={"reason": "deletion not authorized"})
        cutoff = case["audit_at"]
        user_id = self.resolver.resolve_email(case["email"])
        if not user_id:
            return AuditResult(case["case_id"], "INCOMPLETE", [], trace={"reason": "identity not resolved"})
        subject = self.resolver.subject(user_id, as_of=cutoff)
        linked, covering = self.linked_resources(subject, cutoff)
        key = case.get("_key") or case["case_id"]
        if getattr(self, "attribution", None):
            for rid, aid in self.attribution.resources_for(key).items():
                linked.setdefault(rid, f"workflow:{aid}")
        states, unobserved = [], []
        for rid, reason in sorted(linked.items()):
            files = covering
            if self.policy.action(default_category(rid)) == "RETAIN":
                files = set()  # a snapshot not listing a retained record does not prove it was purged
            elif SNAPSHOT_ABSENCE == "hybrid" and not self.index.observations(rid, cutoff):
                # never observed directly: fall back to the newest full snapshot of its system
                snaps = [f for f in self.index.files_before(cutoff) if default_system(rid) in f.scope]
                if snaps:
                    files = covering | {max(snaps, key=lambda f: f.captured_at).file_id}
            st = self.resource_state(rid, cutoff, files)
            if st:
                st.link_reason = reason
                states.append(st)
            else:
                unobserved.append(rid)
        findings, remediation, verification = [], [], []
        for st in states:
            f, r, v = self.decide(st, cutoff)
            findings += [(op, st) for op in f]
            remediation += [(op, st) for op in r]
            verification += [(op, st) for op in v]
        # records with no observation <= audit_at: nothing can be proven, but an incomplete case's
        # post-remediation checks must still cover the account and policy-retained records
        for rid in unobserved:
            cat = default_category(rid)
            op = {"USER_PROFILE": "VERIFY_RECORD_ABSENT"}.get(cat)
            if self.policy.action(cat) == "RETAIN" and cat != "BACKUP_ARCHIVE":
                op = "VERIFY_RECORD_PRESENT"
            if op:
                verification.append((op, ResourceState(rid, default_system(rid), cat, "", "", None, {})))
        if findings:
            status = "INCOMPLETE"
            steps = [step(op, st.system_id, st.resource_id, st.evidence_id) for op, st in findings]
            steps += [step(op, st.system_id, st.resource_id) for op, st in remediation + verification]
        else:
            status = "COMPLETE"
            steps = [step(op, st.system_id, st.resource_id, st.evidence_id) for op, st in verification
                     if st.evidence_id]
        trace = {"user_id": user_id, "linked": linked,
                 "states": {st.resource_id: {k: v for k, v in st.__dict__.items() if k != "resource_id"}
                            for st in states}}
        return AuditResult(case["case_id"], status, order_steps(steps), user_id, trace)
