"""Identity resolution: request email (current or historical) → user_id → every known identifier."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field


def email_hash(email: str) -> str:
    return hashlib.sha256(email.strip().lower().encode()).hexdigest()


def norm_phone(p: str) -> str:
    return re.sub(r"\D", "", p or "")


@dataclass
class Subject:
    user_id: str
    name: str = ""
    emails: set = field(default_factory=set)
    phones: set = field(default_factory=set)
    hashes: set = field(default_factory=set)
    history: list = field(default_factory=list)  # audit rows used (audit_id, field, old, new, time)

    @property
    def link_values(self) -> set:
        vals = {self.user_id.lower()} | {e.lower() for e in self.emails} | set(self.hashes)
        vals |= {p.lower() for p in self.phones}
        return vals

    def matches(self, value: str) -> bool:
        if not value:
            return False
        v = value.strip().lower()
        if v in self.link_values:
            return True
        digits = norm_phone(v)
        return len(digits) >= 8 and "@" not in v and digits in {norm_phone(p) for p in self.phones}


class IdentityResolver:
    def __init__(self, identities: list[dict], audit_log: list[dict]):
        self.users = {r["user_id"]: r for r in identities}
        self.by_email: dict = {}
        self.audit_by_user: dict = {}
        for r in identities:
            self.by_email.setdefault(r["email"].lower(), r["user_id"])
        for a in sorted(audit_log, key=lambda a: a["event_time"]):
            self.audit_by_user.setdefault(a["user_id"], []).append(a)
            if a["field_name"] == "email":
                for v in (a["old_value"], a["new_value"]):
                    self.by_email.setdefault(v.lower(), a["user_id"])

    def resolve_email(self, email: str) -> str | None:
        return self.by_email.get((email or "").strip().lower())

    def subject(self, user_id: str, as_of: str | None = None) -> Subject:
        u = self.users.get(user_id, {})
        s = Subject(user_id=user_id, name=u.get("name", ""))
        if u:
            s.emails.add(u["email"])
            s.phones.add(u["phone"])
            s.hashes.add(u["email_hash"].lower())
        for a in self.audit_by_user.get(user_id, []):
            if as_of and a["event_time"] > as_of:
                # change happened after the audit cut-off: the old value was current at audit time
                pass
            for v in (a["old_value"], a["new_value"]):
                if a["field_name"] == "email":
                    s.emails.add(v)
                elif a["field_name"] == "phone":
                    s.phones.add(v)
                elif a["field_name"] == "email_hash":
                    s.hashes.add(v.lower())
            s.history.append((a["audit_id"], a["field_name"], a["old_value"], a["new_value"], a["event_time"]))
        s.hashes |= {email_hash(e) for e in s.emails}
        return s
