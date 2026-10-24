"""Attribute deletion-worker activity to requests when evidence carries no identifier.

Some records are only ever observed without identifiers (a 404 account lookup, an anonymised
analytics row). The deletion workflow's own activity is then the only link: the worker submits the
account profile shortly before the workflow reports its status and processes the remaining records
afterwards. We attribute worker events to requests using, jointly over all requests:

1. time window around ``reported_at``;
2. agreement between the worker's reported result for the account and the request's reported_status
   (SUCCESS↔COMPLETE, PARTIAL↔PARTIALLY_COMPLETE, FAILED↔NOT_STARTED);
3. timestamp-phase consistency: events of one workflow run are scheduled at fixed second offsets
   from its account submission, so (t_event - t_account) mod 60 falls in a narrow set of phases.
   The phase set is *learned* from requests whose resources are already linked through evidence.

Activity rows remain claims about operations — they are used for linking only, never as proof of state.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta

COMPAT = {"SUCCESS": "COMPLETE", "PARTIAL": "PARTIALLY_COMPLETE", "FAILED": "NOT_STARTED"}
ACCOUNT_BEFORE = timedelta(hours=2)
ACCOUNT_AFTER = timedelta(minutes=15)
FOLLOW_BEFORE = timedelta(hours=2)
FOLLOW_AFTER = timedelta(hours=3)


def _t(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _phase(t_event: datetime, t_account: datetime) -> int:
    return int((t_event - t_account).total_seconds()) % 60


class WorkflowAttribution:
    def __init__(self, activity: list[dict], requests: list[dict], linked_by_request: dict):
        """requests: list of request dicts with a unique ``_key``; linked_by_request: key -> set(resource ids)."""
        self.worker = [a for a in activity if a["source_system"] == "deletion_worker"]
        self.by_resource = defaultdict(list)
        for a in self.worker:
            self.by_resource[a["resource_id"]].append(a)
        self.requests = {r["_key"]: r for r in requests}
        self.linked = linked_by_request
        owners = defaultdict(set)
        for k, res in linked_by_request.items():
            for rid in res:
                owners[rid].add(k)
        self.owners = owners
        self.phases = self._learn_phases()
        self.assigned: dict = defaultdict(dict)   # request key -> {resource_id: activity_id}
        self._assign_accounts()
        self._assign_followups()
        self._assign_tight_window()

    # ------------------------------------------------------------------ phase model
    def _account_event(self, rid):
        for a in self.by_resource.get(rid, []):
            if a["event_type"] == "DELETE_RECORD" and a["target_system"] == "account_service":
                return a
        return None

    def _learn_phases(self) -> set:
        counts = Counter()
        for key, res in self.linked.items():
            accts = [r for r in res if r.startswith("account_") and self._account_event(r)]
            if len(accts) != 1:
                continue
            ta = _t(self._account_event(accts[0])["event_time"])
            for rid in res:
                if rid.startswith("account_") or len(self.owners[rid]) != 1:
                    continue
                for a in self.by_resource.get(rid, []):
                    d = _t(a["event_time"]) - ta
                    if timedelta(0) < d < FOLLOW_AFTER + ACCOUNT_BEFORE:
                        counts[_phase(_t(a["event_time"]), ta)] += 1
        if not counts:
            return set()
        total = sum(counts.values())
        # keep phases that are clearly over-represented (uniform would be total/60 each)
        return {p for p, c in counts.items() if c >= max(2, 1.5 * total / 60)}

    def _phase_ok(self, t_event, t_account) -> bool:
        p = _phase(t_event, t_account)
        return any((p + d) % 60 in self.phases for d in (-1, 0, 1))  # +-1 s scheduling jitter

    # ------------------------------------------------------------------ accounts
    def _assign_accounts(self):
        account_events = [a for a in self.worker if a["event_type"] == "DELETE_RECORD"
                          and a["target_system"] == "account_service" and not self.owners.get(a["resource_id"])]
        needing = [k for k, r in self.requests.items()
                   if not any(x.startswith("account_") for x in self.linked.get(k, ()))]
        cands = []
        for k in needing:
            r = self.requests[k]
            rep = _t(r["reported_at"])
            follow = [a for rid in self.linked.get(k, ()) if len(self.owners[rid]) == 1
                      for a in self.by_resource.get(rid, [])
                      if rep - FOLLOW_BEFORE <= _t(a["event_time"]) <= rep + FOLLOW_AFTER]
            for a in account_events:
                ta = _t(a["event_time"])
                if not (rep - ACCOUNT_BEFORE <= ta <= rep + ACCOUNT_AFTER):
                    continue
                compat = COMPAT.get(a["reported_result"]) == r["reported_status"]
                hits = sum(self._phase_ok(_t(f["event_time"]), ta) for f in follow if _t(f["event_time"]) > ta)
                misses = sum(not self._phase_ok(_t(f["event_time"]), ta) for f in follow if _t(f["event_time"]) > ta)
                score = (2 if compat else 0) + hits - misses
                near = abs((rep - ta).total_seconds()) <= 900  # a submission minutes from the report is a strong cue
                cands.append((score + (3 if near else 0), compat, hits, 1 if near else 0, k, a))
        cands.sort(key=lambda c: (-c[0], -c[2], -c[3], c[4], c[5]["event_time"]))
        used_req, used_acct = set(), set()
        abstain = os.environ.get("GHOST_ATTR_ABSTAIN", "0") == "1"
        for score, compat, hits, _, k, a in cands:
            if k in used_req or a["resource_id"] in used_acct or score < 1:
                continue
            if abstain and any(c[0] == score and c[2] == hits and c[4] != k and c[5]["resource_id"] == a["resource_id"]
                               and c[4] not in used_req for c in cands):
                continue  # another open request fits this submission equally well: do not guess
            used_req.add(k)
            used_acct.add(a["resource_id"])
            self.assigned[k][a["resource_id"]] = a["activity_id"]
        # second pass: a single unclaimed submission within minutes of the report, whatever its result
        tight = timedelta(minutes=15)
        for k in needing:
            if k in used_req:
                continue
            rep = _t(self.requests[k]["reported_at"])
            near = [a for a in account_events if a["resource_id"] not in used_acct
                    and abs(_t(a["event_time"]) - rep) <= tight]
            if len(near) == 1:
                rivals = [k2 for k2 in needing if k2 != k and k2 not in used_req
                          and abs(_t(near[0]["event_time"]) - _t(self.requests[k2]["reported_at"])) <= tight]
                if not rivals:
                    used_req.add(k)
                    used_acct.add(near[0]["resource_id"])
                    self.assigned[k][near[0]["resource_id"]] = near[0]["activity_id"]

    # ------------------------------------------------------------------ follow-up records
    def _assign_followups(self):
        claims = defaultdict(list)
        for k, r in self.requests.items():
            accts = [x for x in self.linked.get(k, ()) if x.startswith("account_")] + \
                    [x for x in self.assigned.get(k, {}) if x.startswith("account_")]
            evs = [self._account_event(x) for x in accts]
            evs = [e for e in evs if e]
            if len(evs) != 1:
                continue
            ta = _t(evs[0]["event_time"])
            rep = _t(r["reported_at"])
            for a in self.worker:
                rid = a["resource_id"]
                if rid.startswith("account_") or self.owners.get(rid):
                    continue
                te = _t(a["event_time"])
                if ta < te <= min(rep + FOLLOW_AFTER, ta + FOLLOW_AFTER + ACCOUNT_BEFORE) and self._phase_ok(te, ta):
                    exact = _phase(te, ta) in self.phases
                    claims[rid].append((2 if exact else 1, -len(self.linked.get(k, ())), k, a["activity_id"]))
        for rid, ks in claims.items():
            ks.sort(reverse=True)
            best = ks[0]
            rivals = [c for c in ks[1:] if c[2] != best[2] and c[:2] == best[:2]]
            if not rivals:  # unique best claimant (exact phase first, then fewest other linked records)
                self.assigned[best[2]][rid] = best[3]

    def _assign_tight_window(self):
        """Requests with no account event at all: unique unclaimed worker events within minutes of the report."""
        tight = timedelta(minutes=15)
        taken = {rid for v in self.assigned.values() for rid in v}
        for k, r in self.requests.items():
            if any(x.startswith("account_") for x in list(self.linked.get(k, ())) + list(self.assigned.get(k, {}))):
                continue
            rep = _t(r["reported_at"])
            for a in self.worker:
                rid = a["resource_id"]
                if rid in taken or self.owners.get(rid) or abs(_t(a["event_time"]) - rep) > tight:
                    continue
                rivals = [k2 for k2, r2 in self.requests.items() if k2 != k
                          and abs(_t(a["event_time"]) - _t(r2["reported_at"])) <= tight]
                if not rivals:
                    self.assigned[k][rid] = a["activity_id"]
                    taken.add(rid)

    def resources_for(self, key: str) -> dict:
        return dict(self.assigned.get(key, {}))
