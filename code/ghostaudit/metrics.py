"""Scoring of predictions against expected answers (public samples or synthetic ground truth).

Metrics
* status accuracy        — deletion_status exact match per case
* step precision/recall  — steps compared as (phase, system_id, resource_id, operation, state, evidence_id)
* field accuracy         — per field, over steps aligned by (system_id, resource_id, operation)
* sequence accuracy      — whole steps_json array identical (order-sensitive, key-order-insensitive)
"""

from __future__ import annotations

import json

FIELDS = ("step_no", "phase", "system_id", "resource_id", "operation", "state", "evidence_id")


def _tuple(s):
    return tuple(s.get(k) for k in FIELDS[1:])


def score(predicted: dict, expected: dict) -> dict:
    """predicted/expected: {case_id: (status, steps_list)}"""
    cases = sorted(expected)
    status_ok = seq_ok = 0
    tp = fp = fn = 0
    field_hits = {k: 0 for k in FIELDS}
    aligned = 0
    failures = []
    for cid in cases:
        es, esteps = expected[cid]
        ps, psteps = predicted.get(cid, ("MISSING", []))
        status_ok += es == ps
        seq_ok += json.dumps(esteps, sort_keys=True) == json.dumps(psteps, sort_keys=True)
        et = [_tuple(s) for s in esteps]
        pt = [_tuple(s) for s in psteps]
        rem = list(pt)
        hit = 0
        for t in et:
            if t in rem:
                rem.remove(t)
                hit += 1
        tp += hit
        fn += len(et) - hit
        fp += len(rem)
        pidx = {(s["system_id"], s["resource_id"], s["operation"]): s for s in psteps}
        for s in esteps:
            p = pidx.get((s["system_id"], s["resource_id"], s["operation"]))
            aligned += 1
            for k in FIELDS:
                field_hits[k] += bool(p) and p.get(k) == s.get(k)
        if es != ps or et != pt:
            failures.append({"case_id": cid, "expected_status": es, "predicted_status": ps,
                             "missing": [t for t in et if t not in pt], "extra": [t for t in pt if t not in et]})
    n = len(cases) or 1
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    return {
        "cases": len(cases),
        "status_accuracy": round(status_ok / n, 4),
        "sequence_accuracy": round(seq_ok / n, 4),
        "step_precision": round(prec, 4),
        "step_recall": round(rec, 4),
        "step_f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
        "field_accuracy": {k: round(v / aligned, 4) if aligned else 1.0 for k, v in field_hits.items()},
        "expected_steps": tp + fn, "predicted_steps": tp + fp, "matched_steps": tp,
        "failures": failures,
    }


def load_expected_csv(path) -> dict:
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        return {r["case_id"]: (r["deletion_status"], json.loads(r["steps_json"])) for r in csv.DictReader(f)}
