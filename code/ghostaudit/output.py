"""Output serialisation and contract validation for output.csv."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .policy import OP_PHASE, OP_STATE, PHASE_RANK, OP_RANK

STATUSES = {"COMPLETE", "INCOMPLETE", "NOT_APPLICABLE"}
STEP_KEYS = {"step_no", "phase", "system_id", "resource_id", "operation", "state", "evidence_id"}


def dumps_steps(steps: list[dict]) -> str:
    ordered = [{k: s[k] for k in ("step_no", "phase", "system_id", "resource_id", "operation", "state",
                                  "evidence_id")} for s in steps]
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def write_output(rows: list[tuple], path: Path) -> None:
    rows = sorted(rows, key=lambda r: r[0])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["case_id", "deletion_status", "steps_json"])
        for case_id, status, steps in rows:
            w.writerow([case_id, status, dumps_steps(steps)])


def validate_output(path: Path, expected_case_ids: list[str], evidence_ids: set, captured_at: dict,
                    audit_at: dict) -> list[str]:
    """Return a list of contract violations (empty list = valid)."""
    errors = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["case_id", "deletion_status", "steps_json"]:
            errors.append(f"bad header {reader.fieldnames}")
        rows = list(reader)
    ids = [r["case_id"] for r in rows]
    if ids != sorted(ids):
        errors.append("rows not sorted by case_id")
    if sorted(ids) != sorted(expected_case_ids):
        errors.append(f"case ids mismatch: {len(ids)} rows vs {len(expected_case_ids)} expected")
    if len(set(ids)) != len(ids):
        errors.append("duplicate case ids")
    for r in rows:
        cid, status = r["case_id"], r["deletion_status"]
        if status not in STATUSES:
            errors.append(f"{cid}: bad status {status}")
        try:
            steps = json.loads(r["steps_json"])
        except json.JSONDecodeError as e:
            errors.append(f"{cid}: invalid JSON ({e})")
            continue
        if not isinstance(steps, list):
            errors.append(f"{cid}: steps_json not an array")
            continue
        if status == "NOT_APPLICABLE" and steps:
            errors.append(f"{cid}: NOT_APPLICABLE must have []")
        last = (-1, -1, "", "")
        for i, s in enumerate(steps, start=1):
            if set(s) != STEP_KEYS:
                errors.append(f"{cid}: step {i} keys {sorted(s)}")
                continue
            if s["step_no"] != i:
                errors.append(f"{cid}: step_no {s['step_no']} != {i}")
            op = s["operation"]
            if op not in OP_PHASE or OP_PHASE[op] != s["phase"] or OP_STATE[op] != s["state"]:
                errors.append(f"{cid}: step {i} invalid op/phase/state {op}/{s['phase']}/{s['state']}")
                continue
            key = (PHASE_RANK[s["phase"]], OP_RANK[op], s["system_id"], s["resource_id"])
            if key < last:
                errors.append(f"{cid}: step {i} out of order")
            last = key
            ev = s["evidence_id"]
            if status == "COMPLETE" and s["phase"] != "VERIFICATION":
                errors.append(f"{cid}: COMPLETE case has non-verification step")
            if status == "COMPLETE" and not ev:
                errors.append(f"{cid}: COMPLETE verification without evidence")
            if status == "INCOMPLETE" and s["phase"] != "FINDING" and ev:
                errors.append(f"{cid}: future step {i} must have empty evidence_id")
            if status == "INCOMPLETE" and s["phase"] == "FINDING" and not ev:
                errors.append(f"{cid}: finding {i} without evidence")
            if ev:
                if ev not in evidence_ids:
                    errors.append(f"{cid}: unknown evidence id {ev}")
                elif ev in captured_at and captured_at[ev] > audit_at.get(cid, "9999"):
                    errors.append(f"{cid}: evidence {ev} captured after audit_at")
        if status == "INCOMPLETE" and not any(s.get("phase") == "FINDING" for s in steps):
            errors.append(f"{cid}: INCOMPLETE without findings")
    return errors
