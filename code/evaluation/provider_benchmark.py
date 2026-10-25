"""Head-to-head benchmark of LLM providers for the two model-backed tasks → provider_benchmark.md.

    python code/evaluation/provider_benchmark.py          # needs GEMINI_API_KEY and NVIDIA_API_KEY in .env

Intent: every text is sent to the model (``llm_for_all``) with the same system prompt.
  * labelled set: 32 public samples + 10 synthetic + 16 controlled multilingual cases (gold labels);
  * evaluation subset: all 10 non-deletion requests + 110 deletion requests (every 5th) from
    deletion_requests.csv, scored against the deterministic baseline (whose 10 non-deletion labels are
    independently confirmed by the 10 CLASSIFY_REQUEST rows in system_activity.csv). NVIDIA is called
    sequentially, Gemini is paced to its free-tier quota (``GEMINI_RPM``).
Vision: raw transcription of the 11 real screenshots plus a synthetic screenshot and a synthetic scanned PDF
page (exact ground truth). Real screenshots have no gold transcript, so identifiers are scored against the
participant vocabulary (known emails, user ids, resource ids): verified ids, recall against the union of
verified ids found by any method, near-miss misreads, and rows that keep a record id and email together.
"""

from __future__ import annotations

import collections
import difflib
import hashlib
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))

from ghostaudit.data import Dataset  # noqa: E402
from ghostaudit.evidence.documents import normalize_ocr_ids  # noqa: E402
from ghostaudit.evidence.index import EvidenceIndex  # noqa: E402
from ghostaudit.extraction.image import _ocr_lines, _preprocess  # noqa: E402
from ghostaudit.extraction.vision import transcribe_image, vision_enabled, vision_model  # noqa: E402
from ghostaudit.intent import IntentClassifier, baseline  # noqa: E402
from ghostaudit.llm import USAGE  # noqa: E402
from ghostaudit.metrics import load_expected_csv  # noqa: E402
from ghostaudit.model import EMAIL_RE, RESOURCE_RE, USER_RE  # noqa: E402
from ghostaudit.paths import CACHE_DIR, EVALUATION_DIR, load_dotenv  # noqa: E402

PROVIDERS = ("gemini", "nvidia")


def intent_sets(ds):
    from tests.test_intent import CONTROLLED
    syn = Dataset.load(CODE / "synthetic" / "build" / "dataset")
    exp = load_expected_csv(CODE / "synthetic" / "expected_output.csv")
    lab = [(r["request_text"], "DELETE_REQUEST") for r in ds.requests("sample_deletion_requests.csv")]
    lab += [(r["request_text"], "NON_DELETION_REQUEST" if exp[r["case_id"]][0] == "NOT_APPLICABLE" else "DELETE_REQUEST")
            for r in syn.requests()]
    lab += list(CONTROLLED)
    allev = [(r["request_text"], baseline(r["request_text"]).label) for r in ds.requests()]
    ev = [x for x in allev if x[1] != "DELETE_REQUEST"] + [x for x in allev if x[1] == "DELETE_REQUEST"][::5][:110]
    return lab, ev


def bench_intent(provider, sets):
    clf = IntentClassifier(provider=provider, llm_for_all=True)
    if not clf.check_model():
        return {"model": clf.model, "error": "key missing or model not served"}
    before = json.loads(json.dumps(USAGE.as_dict().get(clf.model, {})))
    out = {"model": clf.model}
    for name, items in sets.items():
        t0 = time.monotonic()

        def go(item):
            s = time.monotonic()
            r = clf.classify(item[0])
            return item[1], r.label, time.monotonic() - s, item[0]
        with ThreadPoolExecutor(1 if provider == "nvidia" else 4) as ex:
            res = list(ex.map(go, items))
        conf = collections.Counter((g, p) for g, p, _, _ in res)
        tp = conf[("DELETE_REQUEST", "DELETE_REQUEST")]
        fp = sum(v for (g, p), v in conf.items() if p == "DELETE_REQUEST" and g != "DELETE_REQUEST")
        fn = sum(v for (g, p), v in conf.items() if g == "DELETE_REQUEST" and p != "DELETE_REQUEST")
        out[name] = {"n": len(res), "accuracy": sum(v for (g, p), v in conf.items() if g == p) / len(res),
                     "precision": tp / (tp + fp) if tp + fp else 1.0, "recall": tp / (tp + fn) if tp + fn else 1.0,
                     "unresolved": sum(1 for x in res if x[1] == "UNRESOLVED"),
                     "confusion": {f"{g}→{p}": v for (g, p), v in sorted(conf.items())},
                     "errors": [t[:90] for g, p, _, t in res if g != p][:8], "wall_s": round(time.monotonic() - t0, 1)}
    after = USAGE.as_dict().get(clf.model, {})
    out["api"] = {k: after.get(k, 0) - before.get(k, 0) for k in ("calls", "failed_calls", "input_tokens",
                                                                   "output_tokens", "latency_s")}
    out["cache_hits"] = clf.stats["cache_hits"]
    return out


def _vision_cached(provider, name, img):
    """Transcripts are cached per provider+image so re-running the report does not re-spend tokens."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    key = hashlib.sha256(buf.getvalue()).hexdigest()
    p = CACHE_DIR / "provider_bench" / f"{provider}_{vision_model(provider).replace('/', '_')}_{key}.json"
    if p.exists():
        return json.loads(p.read_text())
    t0 = time.monotonic()
    text = transcribe_image(img, provider=provider) or ""
    rec = {"text": text, "latency_s": time.monotonic() - t0, "failed": not text}
    p.parent.mkdir(parents=True, exist_ok=True)
    if text:
        p.write_text(json.dumps(rec))
    return rec


def bench_vision(ds, ix):
    import pymupdf
    from PIL import Image
    vocab = ({e.lower() for e in ix.known.emails} | set(ix.by_resource) | {a["resource_id"] for a in ds.activity}
             | ix.known.user_ids)
    real = [(r["file_id"], Image.open(ds.root / r["file_path"])) for r in ds.manifest if r["file_path"].endswith(".png")]
    syn = CODE / "synthetic" / "build" / "dataset" / "evidence"
    page = pymupdf.open(syn / "documents/syn_export_handover_20260812.pdf")[1]
    truth = {"syn_screenshot": (Image.open(syn / "images/syn_backup_console_20260805.png"),
                                {"backup_90091", "backup_90098", "oyelaran.bisi@mail.example.test",
                                 "vendor.ops@mail.example.test"}),
             "syn_scanned_pdf_page": (Image.open(io.BytesIO(page.get_pixmap(dpi=300).tobytes("png"))),
                                      {"export_90101", "marguerite.leroux@mail.example.test"})}

    def toks(t):
        t = normalize_ocr_ids(t)
        return ({e.lower().rstrip(".") for e in EMAIL_RE.findall(t)} | {m.group(0) for m in RESOURCE_RE.finditer(t)}
                | set(USER_RE.findall(t)))

    def tesseract(img):
        pre = _preprocess(img, "gray_x3")
        lines, _ = _ocr_lines(pre, pre.width / img.width)
        return {"text": "\n".join(l["text"] for l in lines), "latency_s": 0.0, "failed": False}

    methods = {"tesseract (gray_x3, local)": tesseract}
    for p in PROVIDERS:
        if vision_enabled(p):
            methods[f"{p}: {vision_model(p)}"] = (lambda prov: lambda img, name=None: _vision_cached(prov, name, img))(p)
    images = real + [(k, v[0]) for k, v in truth.items()]
    outs = {m: {n: fn(img) for n, img in images} for m, fn in methods.items()}
    rows = {}
    for m in methods:
        ver = near = 0
        rec, pairs, fails, lat = [], 0, 0, []
        for n, _ in real:
            union = set().union(*[{x for x in toks(outs[k][n]["text"]) if x in vocab} for k in methods])
            tk = toks(outs[m][n]["text"])
            v = {x for x in tk if x in vocab}
            ver += len(v)
            rec.append(len(v) / len(union) if union else 1.0)
            near += sum(1 for x in tk - v if difflib.get_close_matches(x, vocab, n=1, cutoff=0.85))
            for line in outs[m][n]["text"].splitlines():
                nl = normalize_ocr_ids(line)
                pairs += bool(RESOURCE_RE.search(nl)) and bool(EMAIL_RE.search(nl))
        for n, _ in images:
            fails += outs[m][n]["failed"]
            lat.append(outs[m][n]["latency_s"])
        syn_hits = {k: f"{len(toks(outs[m][k]['text']) & g)}/{len(g)}" for k, (_, g) in truth.items()}
        rows[m] = {"verified_ids": ver, "recall_vs_union": sum(rec) / len(rec), "near_miss_misreads": near,
                   "rows_with_id_and_email": pairs, "synthetic_exact": syn_hits, "failed": fails,
                   "mean_latency_s": sum(lat) / len(lat)}
    return rows


def main():
    load_dotenv()
    ds = Dataset.load()
    ix = EvidenceIndex(ds, allow_vision=False)
    lab, ev = intent_sets(ds)
    intent = {p: bench_intent(p, {"labelled": lab, "eval_subset": ev}) for p in PROVIDERS}
    vision = bench_vision(ds, ix)
    L = ["# Provider benchmark — intent classification and vision transcription", "",
         f"Generated by `python code/evaluation/provider_benchmark.py` on {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}. "
         "Same prompts for both providers; thinking disabled/minimal for both.", "",
         "## Intent classification (every text sent to the model)", "",
         "| Provider / model | Set | n | Accuracy | Precision (DELETE) | Recall (DELETE) | Unresolved |", "|---|---|---|---|---|---|---|"]
    for p, r in intent.items():
        if "error" in r:
            L.append(f"| {p}: {r['model']} | – | – | {r['error']} | | | |")
            continue
        for s in ("labelled", "eval_subset"):
            x = r[s]
            L.append(f"| {p}: `{r['model']}` | {s} | {x['n']} | {x['accuracy']:.4f} | {x['precision']:.4f} | "
                     f"{x['recall']:.4f} | {x['unresolved']} |")
    L += ["", "API behaviour during this run (cached results from earlier runs are not re-billed):", ""]
    for p, r in intent.items():
        if "api" in r:
            a = r["api"]
            L.append(f"* {p}: new calls {a['calls']}, failed calls {a['failed_calls']}, tokens in/out "
                     f"{a['input_tokens']}/{a['output_tokens']}, cache hits {r['cache_hits']}")
            for s in ("labelled", "eval_subset"):
                if r[s]["errors"]:
                    L.append(f"  * {s} disagreements: " + "; ".join(f"`{e}`" for e in r[s]["errors"]))
    L += ["", "## Vision transcription (13 images: 11 real screenshots + 2 synthetic with exact ground truth)", "",
          "| Method | Verified ids (real) | Recall vs union | Near-miss misreads | Rows keeping id+email | "
          "Synthetic exact | Failed calls | Mean latency (s) |", "|---|---|---|---|---|---|---|---|"]
    for m, r in vision.items():
        L.append(f"| {m} | {r['verified_ids']} | {r['recall_vs_union']:.3f} | {r['near_miss_misreads']} | "
                 f"{r['rows_with_id_and_email']} | {r['synthetic_exact']} | {r['failed']} | {r['mean_latency_s']:.1f} |")
    L += ["", "Notes: a near-miss is an extracted id/email within 0.85 similarity of a known identifier but not equal "
          "to it (a likely misread). `g.silva76.jr@gmail.com` counts as a near-miss for every method although it is a "
          "real decoy address printed in evidence_09. Vision output is always merged into Tesseract's rows (ids are "
          "re-spelled from the vision text), so row preservation matters less than identifier accuracy.", "",
          "## Live (uncached) reliability observed while choosing the provider", "",
          "Recorded in `log.txt` from the first, uncached benchmark runs on 2026-10-25 (same prompt, thinking off):", "",
          "* Gemini intent: 83 calls, 0 failed; 100% on the labelled set and on the 112-text evaluation subset.",
          "* NVIDIA intent, 4 parallel requests: 377 calls, 313 failed (transient rejections) → 49/112 texts "
          "UNRESOLVED; sequential re-run of those texts: 67 calls, 21 failed, 1 text still UNRESOLVED (99.1%).",
          "* Vision, first run: Gemini recall 1.000, 1 near-miss, 0 failures, 4.4 s/image; NVIDIA recall 0.970, 2 near-misses, "
          "6/18 failed calls, 9.1 s/image. Vision recall varies between runs (Gemini 1.000 → 0.909 above), which is why "
          "vision text is merged into Tesseract output rather than replacing it.", "",
          "## Decision", "", "Gemini (`gemini-3.5-flash-lite`) is the default provider for both intent and vision "
          "(`GHOST_INTENT_PROVIDER` / `GHOST_VISION_PROVIDER` = `nvidia` switches back)."]
    (EVALUATION_DIR / "provider_benchmark.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
