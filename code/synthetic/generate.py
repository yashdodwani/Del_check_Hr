"""Deterministic generator for the synthetic regression dataset.

Builds ``code/synthetic/build/dataset/`` with the same schema as the challenge dataset (identities,
audit log, activity, rules, evidence manifest, evidence files, requests) and writes the ground truth
to ``code/synthetic/expected_output.csv`` (kept separate from any pipeline output).

All people, emails (``*.example.test``) and identifiers are fictional. Nothing here is presented as
original challenge evidence; it only supplements the supplied public samples.

    python code/synthetic/generate.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from ghostaudit.paths import DATASET_DIR  # noqa: E402

BUILD = HERE / "build" / "dataset"
EXPECTED = HERE / "expected_output.csv"
SEED = 20261025


def h(email: str) -> str:
    return hashlib.sha256(email.lower().encode()).hexdigest()


def step(n, phase, system, rid, op, state, ev=""):
    return {"step_no": n, "phase": phase, "system_id": system, "resource_id": rid, "operation": op,
            "state": state, "evidence_id": ev}


def steps(*rows):
    return [step(i, *r) for i, r in enumerate(rows, start=1)]


class Builder:
    def __init__(self):
        self.rng = random.Random(SEED)
        self.identities, self.audit, self.activity, self.manifest = [], [], [], []
        self.requests, self.expected = [], []
        self._act = 0
        self._ev = 0

    # ---------------------------------------------------------------- helpers
    def user(self, uid, name, email, phone, updated="2025-06-01T10:00:00Z"):
        self.identities.append({"user_id": uid, "name": name, "email": email, "phone": phone,
                                "email_hash": h(email), "updated_at": updated})

    def act(self, t, etype, src, tgt, rid, result, details):
        self._act += 1
        self.activity.append({"activity_id": f"syn_activity_{self._act:03d}", "event_time": t, "event_type": etype,
                              "source_system": src, "target_system": tgt, "resource_id": rid,
                              "reported_result": result, "details": details})

    def evidence(self, rel, origin, captured, content=None, writer=None):
        self._ev += 1
        fid = f"syn_evidence_{self._ev:02d}"
        path = BUILD / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if writer:
            writer(path)
        else:
            path.write_text(content, encoding="utf-8")
        self.manifest.append({"file_id": fid, "file_path": rel, "origin": origin, "captured_at": captured})
        return fid

    def request(self, cid, email, req, rep_status, rep, audit, text, status, expected_steps):
        self.requests.append({"case_id": cid, "email": email, "requested_at": req, "reported_status": rep_status,
                              "reported_at": rep, "audit_at": audit, "request_text": text})
        self.expected.append((cid, status, expected_steps))

    # ---------------------------------------------------------------- scenarios
    def s01_search_index_survives(self):
        """Primary account deleted (404) but the search index still returns the customer."""
        e = "ines.valtonen@mail.example.test"
        self.user("user_9001", "Ines Valtonen", e, "+358 40 555 0101")
        log = "\n".join([
            "2026-07-01T09:00:00Z level=INFO service=readback_collector event=export_started target_count=2 mode=point_in_time",
            f"2026-07-01T09:01:00Z level=DEBUG service=account_service trace_id=syn-001 event=profile_lookup_started resource_id=account_90011 lookup_key={e}",
            "2026-07-01T09:01:00Z level=INFO service=account_service trace_id=syn-001 event=profile_lookup_completed resource_id=account_90011 rows=0 http_status=404",
            f"2026-07-01T09:02:00Z level=DEBUG service=search_service trace_id=syn-002 event=index_query_started resource_id=search_90012 lookup_key={e}",
            f"2026-07-01T09:02:00Z level=INFO service=search_service trace_id=syn-002 event=index_query_completed resource_id=search_90012 result=hit total_hits=1 index=customer_directory_v3 email={e} customer_reference=null",
            "2026-07-01T09:03:00Z level=INFO service=readback_collector event=export_completed queries=2 dropped=0", ""])
        ev = self.evidence("evidence/logs/syn_readback_20260701.log", "service_readback_logs", "2026-07-01T09:03:00Z", log)
        self.act("2026-06-25T10:00:00Z", "DELETE_RECORD", "deletion_worker", "search_service", "search_90012", "SUCCESS",
                 "Deletion worker reported the requested operation as completed. Target resource: search_90012.")
        self.request("syn_01", e, "2026-06-20T10:00:00Z", "COMPLETE", "2026-06-25T12:00:00Z", "2026-07-02T10:00:00Z",
                     "Please delete my account and every copy of my profile.", "INCOMPLETE", steps(
                         ("FINDING", "search_service", "search_90012", "REPORT_REMAINING_RECORD", "PRESENT", ev),
                         ("REMEDIATION", "search_service", "search_90012", "DELETE_RECORD", "ABSENT"),
                         ("VERIFICATION", "account_service", "account_90011", "VERIFY_RECORD_ABSENT", "ABSENT"),
                         ("VERIFICATION", "search_service", "search_90012", "VERIFY_RECORD_ABSENT", "ABSENT")))

    def s02_support_attachment(self):
        """A support ticket attachment uploaded by the customer remains (attachment → DELETE)."""
        e = "tomasz.wierzba@mail.example.test"
        self.user("user_9002", "Tomasz Wierzba", e, "+48 600 555 202")
        bundle = {"exported_at": "2026-07-05T10:00:00Z", "source": "records_assurance_service", "snapshots": [
            {"captured_at": "2026-07-05T10:00:00Z", "data_store": "account_service", "collection": "customer_profiles",
             "records": []},
            {"captured_at": "2026-07-05T10:00:00Z", "data_store": "support_platform", "collection": "ticket_attachments",
             "records": [{"resource_id": "support_90021", "ticket_id": "CS-90021", "file_name": "passport_scan.pdf",
                          "uploaded_by": e, "storage_class": "ticket_attachment"}]},
            {"captured_at": "2026-07-05T10:00:00Z", "data_store": "security_service", "collection": "security_audit_events",
             "records": [{"resource_id": "audit_record_90022", "subject_reference": "user_9002",
                          "event_name": "account_deletion_reviewed", "actor_type": "automated_privacy_workflow"}]}]}
        ev = self.evidence("evidence/records/syn_bundle_20260705.json", "records_assurance_snapshot_bundle",
                           "2026-07-05T10:00:00Z", json.dumps(bundle, indent=2))
        self.act("2026-07-01T11:00:00Z", "DELETE_RECORD", "deletion_worker", "account_service", "account_90020",
                 "SUCCESS", "Deletion worker reported the requested operation as completed. Target resource: account_90020.")
        self.request("syn_02", e, "2026-06-29T10:00:00Z", "COMPLETE", "2026-07-01T11:05:00Z", "2026-07-06T09:00:00Z",
                     "Please erase my account and the ID document I attached to my support ticket.", "INCOMPLETE", steps(
                         ("FINDING", "support_platform", "support_90021", "REPORT_REMAINING_RECORD", "PRESENT", ev),
                         ("REMEDIATION", "support_platform", "support_90021", "DELETE_RECORD", "ABSENT"),
                         ("VERIFICATION", "account_service", "account_90020", "VERIFY_RECORD_ABSENT", "ABSENT"),
                         ("VERIFICATION", "support_platform", "support_90021", "VERIFY_RECORD_ABSENT", "ABSENT"),
                         ("VERIFICATION", "security_service", "audit_record_90022", "VERIFY_RECORD_PRESENT", "PRESENT")))

    def s03_replica_drift(self):
        """Primary no longer has the profile; the read replica still does (replication drift)."""
        e = "amara.okonkwo@mail.example.test"
        self.user("user_9003", "Amara Okonkwo", e, "+234 803 555 0303")
        rec = json.dumps({"resource_id": "account_90031", "customer_reference": "user_9003", "full_name": "Amara Okonkwo",
                          "email": e, "phone": "+234 803 555 0303", "account_status": "ACTIVE"})
        other = json.dumps({"resource_id": "account_90039", "customer_reference": "user_9099", "full_name": "Decoy Person",
                            "email": "decoy.person@mail.example.test", "phone": "+1 555 0000", "account_status": "ACTIVE"})
        sql = f"""-- synthetic point-in-time database snapshot
-- Captured at: 2026-07-10T08:00:00Z
BEGIN TRANSACTION;
CREATE TABLE customer_data_primary (resource_id TEXT PRIMARY KEY, origin_system TEXT NOT NULL, collection_name TEXT NOT NULL, record_json TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE customer_data_read_replica (resource_id TEXT PRIMARY KEY, origin_system TEXT NOT NULL, collection_name TEXT NOT NULL, record_json TEXT NOT NULL, updated_at TEXT NOT NULL);
INSERT INTO customer_data_primary VALUES('account_90039','account_service','customer_profiles','{other}','2026-07-10T07:00:00Z');
INSERT INTO customer_data_read_replica VALUES('account_90031','account_service','customer_profiles','{rec}','2026-07-01T07:00:00Z');
INSERT INTO customer_data_read_replica VALUES('account_90039','account_service','customer_profiles','{other}','2026-07-10T07:00:00Z');
COMMIT;
"""
        ev = self.evidence("evidence/databases/syn_read_replica_20260710.sql", "customer_data_read_replica_capture",
                           "2026-07-10T08:00:00Z", sql)
        self.request("syn_03", e, "2026-07-01T10:00:00Z", "COMPLETE", "2026-07-02T10:00:00Z", "2026-07-11T10:00:00Z",
                     "I asked for my account to be deleted. Please make sure it is gone everywhere.", "INCOMPLETE", steps(
                         ("FINDING", "account_service", "account_90031", "REPORT_REMAINING_RECORD", "PRESENT", ev),
                         ("REMEDIATION", "account_service", "account_90031", "DELETE_RECORD", "ABSENT"),
                         ("VERIFICATION", "account_service", "account_90031", "VERIFY_RECORD_ABSENT", "ABSENT")))

    def s04_post_audit_evidence(self):
        """Evidence captured after audit_at shows a ghost record; it must not change the historical answer."""
        e = "rafael.quintero@mail.example.test"
        self.user("user_9004", "Rafael Quintero", e, "+34 611 555 404")
        early = {"exported_at": "2026-07-14T10:00:00Z", "region": "global", "service_exports": {"account_service": {
            "profile_lookups": [{"observed_at": "2026-07-14T09:00:00Z", "profile": None, "resource_id": "account_90041",
                                 "status_code": 404, "lookup": e}]}}}
        # the early export queried by email, so record the link on a profile-less lookup via the generic form
        early_rows = {"exported_at": "2026-07-14T10:00:00Z", "origin": "service_readback", "record_count": 1, "records": [
            {"recorded_at": "2026-07-14T09:00:00Z", "system": "account_service", "resource_id": "account_90041",
             "record_type": "USER_PROFILE", "result": "NOT_FOUND", "subject_reference": e}]}
        ev_early = self.evidence("evidence/records/syn_readback_20260714.json", "service_readback", "2026-07-14T10:00:00Z",
                                 json.dumps(early_rows, indent=2))
        late = "\n".join([
            f"2026-07-20T09:00:00Z level=DEBUG service=account_service trace_id=syn-041 event=profile_lookup_started resource_id=account_90041 lookup_key={e}",
            "2026-07-20T09:00:01Z level=INFO service=account_service trace_id=syn-041 event=profile_lookup_completed resource_id=account_90041 rows=1 http_status=200",
            ""])
        self.evidence("evidence/logs/syn_restore_side_effect_20260720.log", "service_readback_logs", "2026-07-20T09:00:01Z", late)
        _ = early
        self.request("syn_04", e, "2026-07-10T10:00:00Z", "COMPLETE", "2026-07-12T10:00:00Z", "2026-07-15T10:00:00Z",
                     "Please audit whether my account deletion was complete when you reviewed it.", "COMPLETE", steps(
                         ("VERIFICATION", "account_service", "account_90041", "VERIFY_RECORD_ABSENT", "ABSENT", ev_early)))

    def s05_mentions_deletion_not_authorised(self):
        e = "lena.hartmann@mail.example.test"
        self.user("user_9005", "Lena Hartmann", e, "+49 151 555 0505")
        self.request("syn_05", e, "2026-07-01T10:00:00Z", "NOT_STARTED", "2026-07-01T12:00:00Z", "2026-07-03T10:00:00Z",
                     "I read your deletion policy. I am not asking for deletion — I only want to know which records "
                     "you keep about me.", "NOT_APPLICABLE", [])

    def s06_completed_with_evidence(self):
        e = "keoni.kahale@mail.example.test"
        self.user("user_9006", "Keoni Kahale", e, "+1 808 555 0606")
        log = "\n".join([
            f"2026-07-20T08:00:00Z level=DEBUG service=account_service trace_id=syn-061 event=profile_lookup_started resource_id=account_90061 lookup_key={e}",
            "2026-07-20T08:00:00Z level=INFO service=account_service trace_id=syn-061 event=profile_lookup_completed resource_id=account_90061 rows=0 http_status=404",
            "2026-07-20T08:01:00Z level=DEBUG service=analytics_platform trace_id=syn-062 event=analytics_profile_read_started resource_id=analytics_90062 lookup_key=user_9006",
            "2026-07-20T08:01:00Z level=INFO service=analytics_platform trace_id=syn-062 event=analytics_profile_read_completed resource_id=analytics_90062 customer_reference=null email=null phone=null email_hash=null audience_cohort=general_shopper",
            "2026-07-20T08:02:00Z level=DEBUG service=security_service trace_id=syn-063 event=audit_record_read_started resource_id=audit_record_90063 lookup_key=user_9006",
            "2026-07-20T08:02:00Z level=INFO service=security_service trace_id=syn-063 event=audit_record_read_completed resource_id=audit_record_90063 rows=1 subject_reference=user_9006 event_code=ACCOUNT_DELETION_RECORDED retention_until=2032-12-31T23:59:59Z",
            "2026-07-20T08:03:00Z level=DEBUG service=integration_service trace_id=syn-064 event=connector_config_read_started resource_id=data_flow_90064 lookup_key=user_9006",
            "2026-07-20T08:03:00Z level=INFO service=integration_service trace_id=syn-064 event=connector_config_read_completed resource_id=data_flow_90064 enabled=false last_delivery_status=stopped connector_name=customer_events_connector customer_email=null customer_reference=user_9006",
            "2026-07-20T08:04:00Z level=DEBUG service=backup_service trace_id=syn-065 event=restore_policy_read_started resource_id=backup_90065 lookup_key=user_9006",
            "2026-07-20T08:04:00Z level=INFO service=backup_service trace_id=syn-065 event=restore_policy_read_completed resource_id=backup_90065 archive_present=true restore_allowed=false archive_key=archive/customer-data/backup_90065.tar.zst customer_email=null customer_reference=user_9006",
            ""])
        # the analytics row is only attributable through an earlier, still-identifiable export
        early = {"captured_at": "2026-06-01T10:00:00Z", "source": "analytics_platform", "collection": "customer_analytics_reports",
                 "records": [{"resource_id": "analytics_90062", "user_id": "user_9006", "email": e, "phone": None,
                              "email_hash": h(e), "reporting_window": "last_90_days"}]}
        self.evidence("evidence/records/syn_analytics_20260601.json", "analytics_platform", "2026-06-01T10:00:00Z",
                      json.dumps(early, indent=2))
        ev = self.evidence("evidence/logs/syn_readback_20260720.log", "service_readback_logs", "2026-07-20T08:04:00Z", log)
        self.request("syn_06", e, "2026-07-10T10:00:00Z", "COMPLETE", "2026-07-12T10:00:00Z", "2026-07-21T10:00:00Z",
                     "Please permanently delete my account and confirm that nothing identifiable remains.", "COMPLETE", steps(
                         ("VERIFICATION", "account_service", "account_90061", "VERIFY_RECORD_ABSENT", "ABSENT", ev),
                         ("VERIFICATION", "security_service", "audit_record_90063", "VERIFY_RECORD_PRESENT", "PRESENT", ev),
                         ("VERIFICATION", "analytics_platform", "analytics_90062", "VERIFY_RECORD_ANONYMIZED", "ANONYMIZED", ev),
                         ("VERIFICATION", "integration_service", "data_flow_90064", "VERIFY_DATA_FLOW_DISABLED", "DISABLED", ev),
                         ("VERIFICATION", "backup_service", "backup_90065", "VERIFY_RESTORE_BLOCKED", "BLOCKED", ev)))

    def s07_retain_and_redact(self):
        """Shared support conversation must be redacted, tombstone stripped to its stable reference, audit kept."""
        e = "priyanka.sood@mail.example.test"
        self.user("user_9007", "Priyanka Sood", e, "+91 98100 55707")
        bundle = {"exported_at": "2026-07-25T10:00:00Z", "source": "records_assurance_service", "snapshots": [
            {"captured_at": "2026-07-25T10:00:00Z", "data_store": "account_service", "collection": "customer_profiles", "records": []},
            {"captured_at": "2026-07-25T10:00:00Z", "data_store": "support_platform", "collection": "support_conversations",
             "records": [{"resource_id": "support_90071", "ticket_id": "CS-90071", "subject_reference": "user_9007",
                          "channel": "email", "message_text": f"Please send the replacement order to {e}; my neighbour "
                                                              "will collect order ORD-77001 for both of us."}]},
            {"captured_at": "2026-07-25T10:00:00Z", "data_store": "identity_service", "collection": "deleted_customer_references",
             "records": [{"resource_id": "tombstone_90072", "subject_reference": "user_9007", "name": "Priyanka Sood",
                          "email": e, "phone": None, "email_hash": h(e), "account_status": "DELETED"}]},
            {"captured_at": "2026-07-25T10:00:00Z", "data_store": "security_service", "collection": "security_audit_events",
             "records": [{"resource_id": "audit_record_90073", "subject_reference": "user_9007",
                          "event_name": "account_deletion_reviewed", "actor_type": "automated_privacy_workflow"}]}]}
        ev = self.evidence("evidence/records/syn_bundle_20260725.json", "records_assurance_snapshot_bundle",
                           "2026-07-25T10:00:00Z", json.dumps(bundle, indent=2))
        self.request("syn_07", e, "2026-07-15T10:00:00Z", "PARTIALLY_COMPLETE", "2026-07-17T10:00:00Z",
                     "2026-07-26T10:00:00Z", "Please delete my account. Keep only what you are legally required to keep.",
                     "INCOMPLETE", steps(
                         ("FINDING", "identity_service", "tombstone_90072", "REPORT_EXPOSED_PII", "PRESENT", ev),
                         ("FINDING", "support_platform", "support_90071", "REPORT_EXPOSED_PII", "PRESENT", ev),
                         ("REMEDIATION", "identity_service", "tombstone_90072", "REDACT_RECORD", "REDACTED"),
                         ("REMEDIATION", "support_platform", "support_90071", "REDACT_RECORD", "REDACTED"),
                         ("VERIFICATION", "security_service", "audit_record_90073", "VERIFY_RECORD_PRESENT", "PRESENT"),
                         ("VERIFICATION", "identity_service", "tombstone_90072", "VERIFY_RECORD_REDACTED", "REDACTED"),
                         ("VERIFICATION", "support_platform", "support_90071", "VERIFY_RECORD_REDACTED", "REDACTED")))

    def s08_historical_alias(self):
        """Request uses the old email; a contact record is keyed by the old email's hash."""
        old, new = "s.mbeki.old@mail.example.test", "sipho.mbeki@mail.example.test"
        self.user("user_9008", "Sipho Mbeki", new, "+27 82 555 0808", updated="2026-03-01T10:00:00Z")
        self.audit += [
            {"audit_id": "syn_audit_01", "user_id": "user_9008", "event_time": "2026-03-01T10:00:00Z",
             "field_name": "email", "old_value": old, "new_value": new},
            {"audit_id": "syn_audit_02", "user_id": "user_9008", "event_time": "2026-03-01T10:00:00Z",
             "field_name": "email_hash", "old_value": h(old), "new_value": h(new)}]
        log = "\n".join([
            f"2026-07-30T07:00:00Z level=DEBUG service=account_service trace_id=syn-081 event=profile_lookup_started resource_id=account_90081 lookup_key={new}",
            "2026-07-30T07:00:00Z level=INFO service=account_service trace_id=syn-081 event=profile_lookup_completed resource_id=account_90081 rows=0 http_status=404",
            f"2026-07-30T07:01:00Z level=DEBUG service=analytics_platform trace_id=syn-082 resource_id=analytics_90082 event=contact_lookup_completed rows=1 email_opt_in=true correlation_key=\"{h(old)}\"",
            ""])
        ev = self.evidence("evidence/logs/syn_service_events_20260730.log", "central_observability_export",
                           "2026-07-30T07:01:00Z", log)
        self.request("syn_08", old, "2026-07-20T10:00:00Z", "COMPLETE", "2026-07-22T10:00:00Z", "2026-07-31T10:00:00Z",
                     "I used this older address before changing it. Please delete everything tied to either address.",
                     "INCOMPLETE", steps(
                         ("FINDING", "analytics_platform", "analytics_90082", "REPORT_REMAINING_RECORD", "PRESENT", ev),
                         ("REMEDIATION", "analytics_platform", "analytics_90082", "DELETE_RECORD", "ABSENT"),
                         ("VERIFICATION", "account_service", "account_90081", "VERIFY_RECORD_ABSENT", "ABSENT"),
                         ("VERIFICATION", "analytics_platform", "analytics_90082", "VERIFY_RECORD_ABSENT", "ABSENT")))

    def s09_small_text_screenshot(self):
        """Dense, small UI text: Tesseract may misread; the pipeline must flag it for vision fallback."""
        from PIL import Image, ImageDraw, ImageFont
        e = "oyelaran.bisi@mail.example.test"
        self.user("user_9009", "Bisi Oyelaran", e, "+234 805 555 0909")

        def draw(path):
            img = Image.new("RGB", (900, 260), (247, 248, 250))
            d = ImageDraw.Draw(img)
            small = ImageFont.load_default(size=11)
            title = ImageFont.load_default(size=16)
            d.text((16, 12), "Vault Recovery Manager  /  Customer archives", fill=(20, 30, 50), font=title)
            d.text((16, 48), "Archive ID        Account                              Region        Restore", fill=(90, 90, 90), font=small)
            rows = [("backup_90091", e, "eu-west-1", "Allowed"),
                    ("backup_90098", "vendor.ops@mail.example.test", "us-east-1", "Blocked")]
            for i, (a, b, c, r) in enumerate(rows):
                y = 70 + i * 22
                d.text((16, y), f"{a:<18}{b:<37}{c:<14}{r}", fill=(30, 30, 30), font=small)
            d.text((16, 140), "Updated moments ago - 2 records", fill=(120, 120, 120), font=small)
            img.save(path)
        ev = self.evidence("evidence/images/syn_backup_console_20260805.png", "backup_service", "2026-08-05T09:00:00Z",
                           writer=draw)
        self.request("syn_09", e, "2026-07-28T10:00:00Z", "COMPLETE", "2026-07-30T10:00:00Z", "2026-08-06T10:00:00Z",
                     "Delete my account. Backups may stay, but they must not be able to restore my data.", "INCOMPLETE",
                     steps(("FINDING", "backup_service", "backup_90091", "REPORT_RESTORE_RISK", "ACTIVE", ev),
                           ("REMEDIATION", "backup_service", "backup_90091", "BLOCK_RESTORE", "BLOCKED"),
                           ("VERIFICATION", "backup_service", "backup_90091", "VERIFY_RESTORE_BLOCKED", "BLOCKED")))

    def s10_mixed_pdf(self):
        """PDF with a digital page and a scanned (image-only) page; the export id is only on the scan."""
        import io

        import pymupdf
        from PIL import Image, ImageDraw, ImageFont
        e = "marguerite.leroux@mail.example.test"
        self.user("user_9010", "Marguerite Leroux", e, "+33 6 55 50 10 10")

        def write(path):
            doc = pymupdf.open()
            p1 = doc.new_page()
            p1.insert_text((72, 90), "NORTHSTAR SHOP - DATA EXPORT HANDOVER", fontsize=14)
            p1.insert_text((72, 120), f"Customer {e}", fontsize=11)
            p1.insert_text((72, 140), "Prepared for archive review. Details of the export package follow on page 2.", fontsize=11)
            img = Image.new("L", (1240, 600), 255)
            d = ImageDraw.Draw(img)
            f = ImageFont.load_default(size=34)
            d.text((60, 80), "Export package export_90101", fill=0, font=f)
            d.text((60, 150), f"Owner {e}", fill=0, font=f)
            d.text((60, 220), "Download link active until 30 Sep 2026", fill=0, font=f)
            img = img.rotate(0.6, fillcolor=255)  # slight skew, like a scan
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            p2 = doc.new_page()
            p2.insert_image(pymupdf.Rect(36, 72, 576, 333), stream=buf.getvalue())
            doc.save(path, deflate=True, no_new_id=True)
            doc.close()
        ev = self.evidence("evidence/documents/syn_export_handover_20260812.pdf", "export_service",
                           "2026-08-12T10:00:00Z", writer=write)
        self.request("syn_10", e, "2026-08-01T10:00:00Z", "COMPLETE", "2026-08-03T10:00:00Z", "2026-08-13T10:00:00Z",
                     "Please delete my account, including any data export you prepared for me.", "INCOMPLETE", steps(
                         ("FINDING", "export_service", "export_90101", "REPORT_REMAINING_RECORD", "PRESENT", ev),
                         ("REMEDIATION", "export_service", "export_90101", "DELETE_RECORD", "ABSENT"),
                         ("VERIFICATION", "export_service", "export_90101", "VERIFY_RECORD_ABSENT", "ABSENT")))

    # ---------------------------------------------------------------- write
    def build(self):
        if BUILD.exists():
            shutil.rmtree(BUILD)
        BUILD.mkdir(parents=True)
        for name in sorted(n for n in dir(self) if n.startswith("s") and n[1:3].isdigit()):
            getattr(self, name)()
        shutil.copy(DATASET_DIR / "deletion_rules.csv", BUILD / "deletion_rules.csv")

        def write(name, rows, fields):
            with open(BUILD / name, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
                w.writeheader()
                w.writerows(rows)
        write("user_identities.csv", self.identities, ["user_id", "name", "email", "phone", "email_hash", "updated_at"])
        write("user_audit_log.csv", self.audit, ["audit_id", "user_id", "event_time", "field_name", "old_value", "new_value"])
        write("system_activity.csv", self.activity, ["activity_id", "event_time", "event_type", "source_system",
                                                     "target_system", "resource_id", "reported_result", "details"])
        write("evidence_files.csv", self.manifest, ["file_id", "file_path", "origin", "captured_at"])
        write("deletion_requests.csv", self.requests, ["case_id", "email", "requested_at", "reported_status",
                                                       "reported_at", "audit_at", "request_text"])
        with open(EXPECTED, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["case_id", "deletion_status", "steps_json"])
            for cid, status, st in sorted(self.expected):
                w.writerow([cid, status, json.dumps(st, separators=(",", ":"))])
        return BUILD


if __name__ == "__main__":
    out = Builder().build()
    print(f"synthetic dataset written to {out}; ground truth in {EXPECTED}")
