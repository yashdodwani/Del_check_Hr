"""End-to-end run: requests → intent → audit → output.csv (+ run summary for the reports)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from .audit import Auditor
from .data import Dataset
from .evidence.index import EvidenceIndex
from .intent import IntentClassifier, authorizes_deletion
from .extraction.vision import vision_model
from .llm import USAGE
from .output import validate_output, write_output


def run(dataset_root: Path | None = None, requests_file: str = "deletion_requests.csv", output: Path | None = None,
        *, use_llm: bool = True, allow_vision: bool = True, llm_for_all: bool = False,
        context_files: tuple = ("sample_deletion_requests.csv", "deletion_requests.csv"),
        summary_path: Path | None = None, log=print) -> dict:
    t0 = time.monotonic()
    ds = Dataset.load(dataset_root)
    log(f"loaded dataset from {ds.root}")
    index = EvidenceIndex(ds, allow_vision=allow_vision)
    errors = {f.file_id: f.error for f in index.files.values() if f.error}
    log(f"indexed {len(index.files)} evidence files, {sum(len(f.observations) for f in index.files.values())} "
        f"observations, parse errors={len(errors)}, extraction={dict(index.extraction_stats)}")

    requests = ds.requests(requests_file)
    # all participant-facing requests share one activity stream; linking them jointly disambiguates
    pool = []
    for name in dict.fromkeys(context_files + (requests_file,)):
        p = ds.root / name
        if p.exists():
            tag = "T" if name == requests_file else name
            pool += [dict(r, _key=f"{tag}:{r['case_id']}") for r in ds.requests(name)]
    auditor = Auditor(ds, index)
    auditor.prepare(pool)

    clf = IntentClassifier(use_llm=use_llm, llm_for_all=llm_for_all)
    llm_status = {"provider": clf.provider, "model": clf.model, "api_key_present": clf.client.available}
    if clf.use_llm:
        llm_status["model_served"] = clf.check_model()
    rows, traces = [], {}
    for r in requests:
        intent = clf.classify(r["request_text"])
        res = auditor.audit(dict(r, _key=f"T:{r['case_id']}"), authorized=authorizes_deletion(intent))
        rows.append((r["case_id"], res.status, res.steps))
        traces[r["case_id"]] = {"intent": asdict(intent), "status": res.status, "user_id": res.user_id,
                                "linked": res.trace.get("linked", {}), "reason": res.trace.get("reason", "")}
    out = Path(output) if output else ds.root.parent / "output.csv"
    write_output(rows, out)
    manifest = {f["file_id"]: f["captured_at"] for f in ds.manifest}
    problems = validate_output(out, [r["case_id"] for r in requests], set(manifest), manifest,
                               {r["case_id"]: r["audit_at"] for r in requests})
    status_counts: dict = {}
    for _, s, _ in rows:
        status_counts[s] = status_counts.get(s, 0) + 1
    summary = {
        "requests": len(requests), "output": str(out), "status_counts": status_counts,
        "validation_errors": problems, "evidence_parse_errors": errors,
        "extraction": dict(index.extraction_stats), "extraction_cache": {"hits": index.cache.hits,
                                                                         "misses": index.cache.misses},
        "intent_stats": clf.stats, "llm": llm_status, "usage": USAGE.as_dict(),
        "attribution_phases": sorted(auditor.attribution.phases),
        "vision_model": vision_model(), "seconds": round(time.monotonic() - t0, 2),
    }
    if summary_path:
        Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
        Path(summary_path).write_text(json.dumps({"summary": summary, "cases": traces}, indent=1, sort_keys=True,
                                                 ensure_ascii=False), encoding="utf-8")
    log(f"wrote {len(rows)} rows to {out}; statuses={status_counts}; validation errors={len(problems)}")
    return summary
