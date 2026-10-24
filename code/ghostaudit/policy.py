"""Deletion policy (from deletion_rules.csv) and the step grammar / ordering of the output contract."""

from __future__ import annotations

OPERATION_ORDER = [
    # FINDING
    "REPORT_REMAINING_RECORD", "REPORT_EXPOSED_PII", "REPORT_ACTIVE_DATA_FLOW", "REPORT_RESTORE_RISK",
    # REMEDIATION
    "DISABLE_DATA_FLOW", "BLOCK_RESTORE", "DELETE_RECORD", "REDACT_RECORD", "ANONYMIZE_RECORD",
    # VERIFICATION
    "VERIFY_RECORD_ABSENT", "VERIFY_RECORD_PRESENT", "VERIFY_RECORD_REDACTED", "VERIFY_RECORD_ANONYMIZED",
    "VERIFY_DATA_FLOW_DISABLED", "VERIFY_RESTORE_BLOCKED",
]
OP_RANK = {op: i for i, op in enumerate(OPERATION_ORDER)}
PHASE_RANK = {"FINDING": 0, "REMEDIATION": 1, "VERIFICATION": 2}
OP_PHASE = {op: ("FINDING" if op.startswith("REPORT_") else "VERIFICATION" if op.startswith("VERIFY_")
                 else "REMEDIATION") for op in OPERATION_ORDER}
OP_STATE = {
    "REPORT_REMAINING_RECORD": "PRESENT", "REPORT_EXPOSED_PII": "PRESENT", "REPORT_ACTIVE_DATA_FLOW": "ACTIVE",
    "REPORT_RESTORE_RISK": "ACTIVE", "DISABLE_DATA_FLOW": "DISABLED", "BLOCK_RESTORE": "BLOCKED",
    "DELETE_RECORD": "ABSENT", "REDACT_RECORD": "REDACTED", "ANONYMIZE_RECORD": "ANONYMIZED",
    "VERIFY_RECORD_ABSENT": "ABSENT", "VERIFY_RECORD_PRESENT": "PRESENT", "VERIFY_RECORD_REDACTED": "REDACTED",
    "VERIFY_RECORD_ANONYMIZED": "ANONYMIZED", "VERIFY_DATA_FLOW_DISABLED": "DISABLED",
    "VERIFY_RESTORE_BLOCKED": "BLOCKED",
}

# Categories without a rule row still need a treatment: flows must be stopped.
EXTRA_TREATMENT = {"DATA_FLOW": "DISABLE"}


class Policy:
    def __init__(self, rules: list[dict]):
        self.rules = {r["data_category"]: r for r in rules}

    def action(self, category: str) -> str | None:
        if category in self.rules:
            return self.rules[category]["required_action"]
        return EXTRA_TREATMENT.get(category)

    def rule_id(self, category: str) -> str:
        return self.rules.get(category, {}).get("rule_id", "")


def step(op, system_id, resource_id, evidence_id=""):
    return {"step_no": 0, "phase": OP_PHASE[op], "system_id": system_id, "resource_id": resource_id,
            "operation": op, "state": OP_STATE[op], "evidence_id": evidence_id}


def order_steps(steps: list[dict]) -> list[dict]:
    """Phase → operation-table order → system_id → resource_id; then renumber from 1."""
    seen, uniq = set(), []
    for s in steps:
        k = (s["operation"], s["system_id"], s["resource_id"])
        if k not in seen:
            seen.add(k)
            uniq.append(s)
    uniq.sort(key=lambda s: (PHASE_RANK[s["phase"]], OP_RANK[s["operation"]], s["system_id"], s["resource_id"]))
    for i, s in enumerate(uniq, start=1):
        s["step_no"] = i
    return uniq
