"""Identity resolution, cut-off enforcement, policy treatment, step grammar/order and output schema."""

import csv
import json

from ghostaudit.identity import IdentityResolver, email_hash
from ghostaudit.metrics import score
from ghostaudit.output import validate_output
from ghostaudit.policy import order_steps, step


def test_identity_resolution_current_and_historical():
    ids = [{"user_id": "user_1", "name": "A B", "email": "new@x.test", "phone": "+1 555 0100",
            "email_hash": email_hash("new@x.test"), "updated_at": "2026-01-01T00:00:00Z"}]
    audit = [{"audit_id": "a1", "user_id": "user_1", "event_time": "2025-12-01T00:00:00Z", "field_name": "email",
              "old_value": "Old@X.test", "new_value": "new@x.test"}]
    r = IdentityResolver(ids, audit)
    assert r.resolve_email("OLD@x.test") == "user_1" and r.resolve_email("new@x.test") == "user_1"
    s = r.subject("user_1")
    assert email_hash("old@x.test") in s.hashes and s.matches("old@x.test") and s.matches("+1 (555) 0100")
    assert not s.matches("new@x.test.evil")


def test_email_hash_is_sha256_of_lowercased_email():
    assert email_hash("A@B.C") == email_hash("a@b.c") and len(email_hash("a@b.c")) == 64


def test_step_ordering_follows_contract():
    unordered = [step("VERIFY_RECORD_ABSENT", "search_service", "search_2"),
                 step("DELETE_RECORD", "search_service", "search_2"),
                 step("DISABLE_DATA_FLOW", "integration_service", "data_flow_1"),
                 step("REPORT_REMAINING_RECORD", "search_service", "search_2", "ev"),
                 step("REPORT_ACTIVE_DATA_FLOW", "integration_service", "data_flow_1", "ev"),
                 step("VERIFY_RECORD_ABSENT", "account_service", "account_1"),
                 step("VERIFY_RECORD_PRESENT", "security_service", "audit_record_1")]
    ops = [(s["step_no"], s["operation"], s["resource_id"]) for s in order_steps(unordered)]
    assert ops == [(1, "REPORT_REMAINING_RECORD", "search_2"), (2, "REPORT_ACTIVE_DATA_FLOW", "data_flow_1"),
                   (3, "DISABLE_DATA_FLOW", "data_flow_1"), (4, "DELETE_RECORD", "search_2"),
                   (5, "VERIFY_RECORD_ABSENT", "account_1"), (6, "VERIFY_RECORD_ABSENT", "search_2"),
                   (7, "VERIFY_RECORD_PRESENT", "audit_record_1")]


def _case(run, cid):
    return run["pred"][cid]


def test_synthetic_suite_scores(synthetic_run):
    s = score(synthetic_run["pred"], synthetic_run["expected"])
    assert s["status_accuracy"] == 1.0, s["failures"]
    assert s["sequence_accuracy"] == 1.0, s["failures"]


def test_historical_cutoff_ignores_later_evidence(synthetic_run, synthetic_dataset):
    with open(synthetic_dataset / "evidence_files.csv", newline="") as f:
        by_path = {r["file_path"].rsplit("/", 1)[-1]: r["file_id"] for r in csv.DictReader(f)}
    status, steps = _case(synthetic_run, "syn_04")
    assert status == "COMPLETE"  # the post-audit "profile found again" log must not flip the answer
    assert [s["evidence_id"] for s in steps] == [by_path["syn_readback_20260714.json"]]
    assert by_path["syn_restore_side_effect_20260720.log"] not in {s["evidence_id"] for s in steps}


def test_non_deletion_is_not_applicable(synthetic_run):
    assert _case(synthetic_run, "syn_05") == ("NOT_APPLICABLE", [])


def test_retention_and_redaction(synthetic_run):
    status, steps = _case(synthetic_run, "syn_07")
    ops = {(s["resource_id"], s["operation"]) for s in steps}
    assert status == "INCOMPLETE"
    assert ("audit_record_90073", "VERIFY_RECORD_PRESENT") in ops            # retained, never deleted
    assert ("tombstone_90072", "REDACT_RECORD") in ops                       # tombstone keeps stable ref only
    assert ("support_90071", "REDACT_RECORD") in ops                         # shared conversation → redact
    assert not any(op == "DELETE_RECORD" for _, op in ops)


def test_historical_alias_resolved(synthetic_run):
    status, steps = _case(synthetic_run, "syn_08")
    assert status == "INCOMPLETE" and any(s["resource_id"] == "analytics_90082" for s in steps)


def test_replica_drift_detected(synthetic_run):
    status, steps = _case(synthetic_run, "syn_03")
    assert status == "INCOMPLETE" and steps[0]["operation"] == "REPORT_REMAINING_RECORD"
    assert all(s["resource_id"] != "account_90039" for s in steps)  # decoy row of another person


def test_evidence_references_rules(synthetic_run):
    for cid, (status, steps) in synthetic_run["pred"].items():
        for s in steps:
            if status == "COMPLETE" or s["phase"] == "FINDING":
                assert s["evidence_id"].startswith("syn_evidence_"), (cid, s)
            else:
                assert s["evidence_id"] == "", (cid, s)


def test_output_schema_and_validator(synthetic_run, synthetic_dataset):
    out = synthetic_run["out"]
    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == ["case_id", "deletion_status", "steps_json"]
    assert [r["case_id"] for r in rows] == sorted(r["case_id"] for r in rows)
    assert {r["deletion_status"] for r in rows} <= {"COMPLETE", "INCOMPLETE", "NOT_APPLICABLE"}
    for r in rows:
        assert isinstance(json.loads(r["steps_json"]), list)
    assert synthetic_run["summary"]["validation_errors"] == []


def test_validator_catches_violations(tmp_path):
    p = tmp_path / "bad.csv"
    bad = [{"step_no": 2, "phase": "FINDING", "system_id": "s", "resource_id": "r", "operation": "DELETE_RECORD",
            "state": "ABSENT", "evidence_id": "nope"}]
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "deletion_status", "steps_json"])
        w.writerow(["c2", "MAYBE", json.dumps(bad)])
        w.writerow(["c1", "NOT_APPLICABLE", json.dumps(bad)])
    errs = validate_output(p, ["c1", "c2"], {"ev"}, {}, {})
    joined = "\n".join(errs)
    for needle in ("not sorted", "bad status", "NOT_APPLICABLE must have []", "step_no", "invalid op"):
        assert needle in joined
