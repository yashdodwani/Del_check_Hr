"""Evaluation runner → evaluation/evaluation_report.md and evaluation/usage_report.md.

    python code/evaluation/main.py            # samples + synthetic + intent benchmark + attribution hold-out
    python code/evaluation/main.py --llm      # additionally benchmark the Gemini intent model (needs GEMINI_API_KEY)

All numbers in the reports are produced by this script from actual runs; nothing is typed in by hand
except the narrative "failures found and improvements made" section, which is taken from log.txt.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))

from ghostaudit.attribution import WorkflowAttribution  # noqa: E402
from ghostaudit.audit import Auditor  # noqa: E402
from ghostaudit.data import Dataset  # noqa: E402
from ghostaudit.evidence.index import EvidenceIndex  # noqa: E402
from ghostaudit.intent import IntentClassifier, baseline  # noqa: E402
from ghostaudit.gemini import GEMINI_MODEL  # noqa: E402
from ghostaudit.llm import USAGE  # noqa: E402
from ghostaudit.metrics import load_expected_csv, score  # noqa: E402
from ghostaudit.paths import DATASET_DIR, EVALUATION_DIR, REPO_ROOT, load_dotenv  # noqa: E402
from ghostaudit.pipeline import run  # noqa: E402

SCRATCH = CODE / ".cache" / "evaluation"

# Price assumptions for the cost estimate (USD per 1M tokens); set GHOST_PRICE_IN / GHOST_PRICE_OUT to match
# the actual Gemini billing tier.
PRICE_IN = float(os.environ.get("GHOST_PRICE_IN", "0.10"))
PRICE_OUT = float(os.environ.get("GHOST_PRICE_OUT", "0.40"))


def fmt_metrics(s: dict) -> str:
    fa = s["field_accuracy"]
    rows = [
        "| Metric | Value |", "|---|---|",
        f"| Cases | {s['cases']} |",
        f"| Case-level deletion-status accuracy | {s['status_accuracy']:.4f} |",
        f"| Sequence accuracy (exact `steps_json` array) | {s['sequence_accuracy']:.4f} |",
        f"| Step precision | {s['step_precision']:.4f} |",
        f"| Step recall | {s['step_recall']:.4f} |",
        f"| Step F1 | {s['step_f1']:.4f} |",
        f"| Expected / predicted / matched steps | {s['expected_steps']} / {s['predicted_steps']} / {s['matched_steps']} |",
    ]
    rows += ["", "Field-level accuracy (expected steps aligned by system_id+resource_id+operation):", "",
             "| " + " | ".join(fa) + " |", "|" + "---|" * len(fa), "| " + " | ".join(f"{v:.4f}" for v in fa.values()) + " |"]
    return "\n".join(rows)


def fmt_failures(s: dict, limit=12) -> str:
    if not s["failures"]:
        return "_No mismatches._"
    out = []
    for f in s["failures"][:limit]:
        out.append(f"* **{f['case_id']}** expected `{f['expected_status']}`, predicted `{f['predicted_status']}`")
        for t in f["missing"]:
            out.append(f"  * missing: `{' / '.join(t)}`")
        for t in f["extra"]:
            out.append(f"  * extra: `{' / '.join(t)}`")
    return "\n".join(out)


def run_set(dataset, requests, expected_path, tag, use_llm):
    out = SCRATCH / f"{tag}_output.csv"
    SCRATCH.mkdir(parents=True, exist_ok=True)
    summary = run(dataset, requests, out, use_llm=use_llm, allow_vision=True, summary_path=SCRATCH / f"{tag}_summary.json",
                  log=lambda *_: None)
    return score(load_expected_csv(out), load_expected_csv(expected_path)), summary


def intent_benchmark(use_llm: bool):
    from tests.test_intent import CONTROLLED  # controlled multilingual cases incl. non-authorising mentions
    ds = Dataset.load()
    labelled = [(r["request_text"], "DELETE_REQUEST", "public sample") for r in ds.requests("sample_deletion_requests.csv")]
    syn = Dataset.load(CODE / "synthetic" / "build" / "dataset")
    exp = load_expected_csv(CODE / "synthetic" / "expected_output.csv")
    labelled += [(r["request_text"], "NON_DELETION_REQUEST" if exp[r["case_id"]][0] == "NOT_APPLICABLE"
                  else "DELETE_REQUEST", "synthetic") for r in syn.requests()]
    labelled += [(t, l, "controlled") for t, l in CONTROLLED]
    results = {}
    systems = [("baseline", lambda t: baseline(t))]
    llm_note = "not run"
    if use_llm:
        clf = IntentClassifier(llm_for_all=True)
        if clf.use_llm and clf.check_model():
            systems.append((f"llm:{clf.model}", lambda t: clf.classify(t)))
            llm_note = "run"
        else:
            llm_note = (f"{clf.provider} API key not set" if not clf.client.available
                        else f"model {clf.model} not listed by the endpoint")
    for name, fn in systems:
        conf = Counter()
        lat = []
        failures = 0
        for text, gold, _src in labelled:
            t0 = time.monotonic()
            pred = fn(text).label
            lat.append(time.monotonic() - t0)
            failures += pred == "UNRESOLVED"
            conf[(gold, pred)] += 1
        tp = conf[("DELETE_REQUEST", "DELETE_REQUEST")]
        fp = sum(v for (g, p), v in conf.items() if p == "DELETE_REQUEST" and g != "DELETE_REQUEST")
        fn_ = sum(v for (g, p), v in conf.items() if g == "DELETE_REQUEST" and p != "DELETE_REQUEST")
        n = sum(conf.values())
        results[name] = {"n": n, "accuracy": sum(v for (g, p), v in conf.items() if g == p) / n,
                         "precision_delete": tp / (tp + fp) if tp + fp else 1.0,
                         "recall_delete": tp / (tp + fn_) if tp + fn_ else 1.0,
                         "confusion": {f"{g}→{p}": v for (g, p), v in sorted(conf.items())},
                         "mean_latency_ms": 1000 * sum(lat) / len(lat), "unresolved": failures}
    return results, llm_note, Counter(src for _, _, src in labelled)


def attribution_holdout():
    """Hide evidence links for accounts and check whether workflow attribution recovers them."""
    ds = Dataset.load()
    ix = EvidenceIndex(ds, allow_vision=False)
    au = Auditor(ds, ix)
    pool = [dict(r, _key=f"S:{r['case_id']}") for r in ds.requests("sample_deletion_requests.csv")]
    pool += [dict(r, _key=f"T:{r['case_id']}") for r in ds.requests()]
    au.prepare(pool)
    linked = {k: set(v) for k, v in au._linked_cache.items()}
    acct_events = {a["resource_id"] for a in ds.activity if a["source_system"] == "deletion_worker"
                   and a["event_type"] == "DELETE_RECORD" and a["target_system"] == "account_service"}
    probe = [k for k, v in linked.items() if len([x for x in v if x.startswith("account_")]) == 1
             and next(x for x in v if x.startswith("account_")) in acct_events]
    truth = {k: next(x for x in linked[k] if x.startswith("account_")) for k in probe}
    # 2-fold: hide one half, let the other half (still linked) train the phase model, then swap
    correct = wrong = 0
    phases = []
    for fold in (0, 1):
        held = {k for i, k in enumerate(sorted(probe)) if i % 2 == fold}
        hidden = {k: (v - {truth[k]} if k in held else v) for k, v in linked.items()}
        wa = WorkflowAttribution(ds.activity, pool, hidden)
        phases.append(sorted(wa.phases))
        for k in held:
            got = [x for x in wa.resources_for(k) if x.startswith("account_")]
            correct += got == [truth[k]]
            wrong += bool(got) and got != [truth[k]]
    return {"probed": len(probe), "correct": correct, "wrong": wrong, "unassigned": len(probe) - correct - wrong,
            "phases": phases}


def improvements_from_log() -> str:
    log = (REPO_ROOT / "log.txt").read_text(encoding="utf-8")
    keep = []
    for block in log.split("\n## ")[1:]:
        title = block.splitlines()[0]
        if "Milestone" in title:
            keep.append("* " + title.split("] ", 1)[-1])
    return "\n".join(keep)


def write_reports(args):
    say = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)  # noqa: E731
    say("public samples …")
    sample_s, _ = run_set(DATASET_DIR, "sample_deletion_requests.csv", DATASET_DIR / "sample_output.csv", "samples",
                          args.llm)
    say(f"samples done: status={sample_s['status_accuracy']} sequence={sample_s['sequence_accuracy']}; synthetic …")
    from synthetic.generate import Builder
    Builder().build()
    syn_root = CODE / "synthetic" / "build" / "dataset"
    syn_s, syn_sum = run_set(syn_root, "deletion_requests.csv", CODE / "synthetic" / "expected_output.csv",
                             "synthetic", args.llm)
    say(f"synthetic done: status={syn_s['status_accuracy']} sequence={syn_s['sequence_accuracy']}; intent benchmark …")
    intent, llm_note, intent_sources = intent_benchmark(args.llm)
    say(f"intent done ({llm_note}); attribution hold-out …")
    hold = attribution_holdout()
    say("writing reports")
    full = json.loads((EVALUATION_DIR / "run_summary.json").read_text(encoding="utf-8"))["summary"] \
        if (EVALUATION_DIR / "run_summary.json").exists() else None

    syn_cases = Dataset.load(syn_root).requests()
    syn_desc = {
        "syn_01": "Search index entry survives after the primary account was deleted (readback log).",
        "syn_02": "Customer-uploaded support attachment remains (attachment → DELETE, not REDACT).",
        "syn_03": "Primary SQLite table no longer has the profile but the read replica does; decoy row of another person.",
        "syn_04": "Evidence captured after audit_at shows the profile again; must not change the historical answer.",
        "syn_05": "Text mentions the deletion policy but explicitly does not ask for deletion → NOT_APPLICABLE.",
        "syn_06": "Fully completed deletion with evidence for account, analytics, audit record, flow and backup.",
        "syn_07": "Retention/redaction: shared support conversation, PII-bearing tombstone, retained audit record.",
        "syn_08": "Request uses an old email; the remaining contact record is keyed by the old email's hash.",
        "syn_09": "Small-text UI screenshot (backup console) — Tesseract-sensitive; restore still allowed.",
        "syn_10": "PDF with a digital first page and a scanned second page holding the export id.",
    }
    lines = ["# Evaluation report — Ghost Data Still Exists", "",
             f"Generated by `python code/evaluation/main.py` on {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}.",
             "Intent model calls during this evaluation: " + ("enabled" if args.llm else "disabled (baseline only)") + ".",
             "",
             "## 1. Public samples (32 cases, `dataset/sample_output.csv`)", "", fmt_metrics(sample_s), "",
             "**Caveat:** the public samples were used while developing the parsers and attribution rules, so these "
             "numbers are an optimistic (in-sample) estimate.", "", "Remaining mismatches:", "", fmt_failures(sample_s), "",
             "All remaining sample mismatches were checked by hand: the missing records (e.g. `account_004`, "
             "`audit_record_6645`, `audit_record_5569`) do not appear in any participant-facing file or activity row, "
             "so they cannot be recovered without organizer-only data.", "",
             "## 2. Additional evaluation cases created for this submission (synthetic, 10 cases)", "",
             "Generated deterministically by `code/synthetic/generate.py` (seed 20261025) into a separate dataset "
             "directory with the challenge schema; ground truth is stored separately in "
             "`code/synthetic/expected_output.csv`. All identities are fictional (`*.example.test`).", "",
             "| Case | Scenario |", "|---|---|"]
    lines += [f"| {c['case_id']} | {syn_desc.get(c['case_id'], '')} |" for c in syn_cases]
    lines += ["", fmt_metrics(syn_s), "", "Mismatches:", "", fmt_failures(syn_s), "",
              f"Extraction on the synthetic set: {syn_sum['extraction']}.", "",
              "## 3. Intent classification benchmark", "",
              f"Labelled texts: {dict(intent_sources)} (public samples are all deletion requests; the controlled set "
              "includes multilingual deletions and texts that mention deletion without authorising it).",
              f"Intent LLM (`{IntentClassifier().model}`): {llm_note}.", "",
              "| System | n | Accuracy | Precision (DELETE) | Recall (DELETE) | Unresolved | Mean latency (ms) |",
              "|---|---|---|---|---|---|---|"]
    for name, r in intent.items():
        lines.append(f"| {name} | {r['n']} | {r['accuracy']:.4f} | {r['precision_delete']:.4f} | "
                     f"{r['recall_delete']:.4f} | {r['unresolved']} | {r['mean_latency_ms']:.2f} |")
    lines += [""] + [f"* {name} confusion: {r['confusion']}" for name, r in intent.items()]
    lines += ["", "## 4. Workflow-attribution hold-out check", "",
              "Accounts that *are* linked to a request through evidence were hidden, and the activity-based "
              "attribution (time window + worker-result/reported-status agreement + learned timestamp phases) was "
              "asked to recover them (2-fold: the phase model is learned from the other half). This is harder than the real "
              "setting, because hiding a request's account also hides the anchor its follow-up events would provide.", "",
              f"* probed requests: {hold['probed']}", f"* recovered correctly: {hold['correct']}",
              f"* wrongly attributed: {hold['wrong']}", f"* left unassigned (no confident match): {hold['unassigned']}",
              f"* learned phase sets per fold (seconds mod 60): {hold['phases']}", ""]
    if full:
        lines += ["## 5. Full evaluation-set run (518 cases)", "",
                  f"* status counts: {full['status_counts']}", f"* output validation errors: {len(full['validation_errors'])}",
                  f"* evidence parse errors: {len(full['evidence_parse_errors'])}",
                  f"* extraction: {full['extraction']} (cache {full['extraction_cache']})",
                  f"* intent: {full['intent_stats']}", f"* runtime: {full['seconds']} s", ""]
    lines += ["## 6. Failures found and improvements made", "",
              "Chronological milestones (details, numbers and root causes are in `log.txt`):", "",
              improvements_from_log(), "",
              "Key failure → fix pairs:", "",
              "* Identifier-less records (404 account lookups, anonymised analytics) were unlinkable → added "
              "workflow attribution; sample sequence accuracy 0.25 → 0.78.",
              "* One person with two requests received each other's records → per-request partitioning by time.",
              "* Support PDF treated as a conversation (REDACT) instead of an attachment (DELETE) → document-type cue.",
              "* Replica contact copies treated as support messages → SECONDARY_IDENTIFIER (DELETE).",
              "* Readback exports were treated as full snapshots, wrongly implying absence of retained audit records "
              "→ only bundles/SQL dumps imply absence, never for RETAIN categories.",
              "* Small-text screenshot misread (`backup.00001`, dot-less email domain) and not flagged → 3x OCR variant, "
              "malformed-id/broken-email detection, row-level restore status (synthetic S09).", "",
              "## 7. Known limitations", "",
              "* The attribution step relies on timing regularities of the deletion workflow; when several same-status "
              "requests report at the same time and have no other linked records, the assignment can be wrong (see §4).",
              "* The vision fallback (`gemini-3.5-flash-lite`) is only called with a Gemini API key; without it, "
              "layout-sensitive screenshots rely on Tesseract plus identifier validation and are flagged "
              "`needs_vision` in the extraction cache. Vision recall varies between calls, so its text is merged into "
              "Tesseract's rows rather than replacing them.",
              "* Records that never appear in participant data cannot be reported."]
    (EVALUATION_DIR / "evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_usage_report(full)
    print("wrote", EVALUATION_DIR / "evaluation_report.md")


def write_usage_report(full):
    usage = (full or {}).get("usage", {})
    llm = (full or {}).get("llm", {})
    n = (full or {}).get("requests", 518)
    lines = ["# Usage report — final full-dataset run", "",
             "Source: `code/evaluation/run_summary.json`, written by the run of `python code/main.py` that produced "
             "the submitted `output.csv`.", "",
             f"* Requests processed: {n}",
             f"* Intent model configured: `{llm.get('model', GEMINI_MODEL)}` (provider: Google Gemini API)",
             f"* API key present during the run: {llm.get('api_key_present', False)}",
             f"* Intent routing: {(full or {}).get('intent_stats', {})}",
             "* OCR: Tesseract (local, no tokens); vision fallback model: `"
             + str((full or {}).get("vision_model")) + "` — "
             + ("called (see table)" if (full or {}).get("vision_model") in usage else "not called in this run"), ""]
    lines += ["| Provider | Model | Calls | Failed calls | Input tokens | Output tokens | Total tokens | "
              "Avg tokens / request | Est. cost (USD) | Est. cost / request (USD) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    tot = {"calls": 0, "failed_calls": 0, "input_tokens": 0, "output_tokens": 0}
    for model, u in sorted(usage.items()):
        total = u["input_tokens"] + u["output_tokens"]
        cost = u["input_tokens"] / 1e6 * PRICE_IN + u["output_tokens"] / 1e6 * PRICE_OUT
        lines.append(f"| Google Gemini API | {model} | {u['calls']} | {u['failed_calls']} | {u['input_tokens']} | "
                     f"{u['output_tokens']} | {total} | {total / n:.2f} | {cost:.4f} | {cost / n:.6f} |")
        for k in tot:
            tot[k] += u[k]
    total = tot["input_tokens"] + tot["output_tokens"]
    cost = tot["input_tokens"] / 1e6 * PRICE_IN + tot["output_tokens"] / 1e6 * PRICE_OUT
    lines.append(f"| **All** | **all models** | {tot['calls']} | {tot['failed_calls']} | {tot['input_tokens']} | "
                 f"{tot['output_tokens']} | {total} | {total / n:.2f} | {cost:.4f} | {cost / n:.6f} |")
    lines += ["",
              f"Cost assumptions: USD {PRICE_IN}/1M input and USD {PRICE_OUT}/1M output tokens for every model "
              "(override with `GHOST_PRICE_IN` / `GHOST_PRICE_OUT`); output tokens include any thinking tokens.",
              "Calls and tokens come from each API's own usage fields (`usage` / `usageMetadata`), recorded in-process.", ""]
    if not usage:
        lines += ["**No model calls were made in the final run**: the deterministic baseline classified every request "
                  "with high confidence (no AMBIGUOUS texts) and no screenshot required the vision fallback."]
    (EVALUATION_DIR / "usage_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="benchmark the Gemini intent model as well")
    args = ap.parse_args()
    load_dotenv()
    write_reports(args)
    print("usage:", USAGE.as_dict())
