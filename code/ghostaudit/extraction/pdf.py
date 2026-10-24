"""PDF evidence: native text (PyMuPDF) → per-page Tesseract OCR → selective vision fallback.

Pages are handled independently: a page with usable native text keeps it; only pages that fail the
quality gate are rendered and OCR'd, and only OCR'd pages that are still ambiguous (or whose layout
carries meaning, e.g. embedded screenshots) are sent to the vision model when one is configured.
"""

from __future__ import annotations

import io
from pathlib import Path

from . import PIPELINE_VERSION
from .cache import ExtractionCache, file_sha256
from .quality import KnownVocabulary, assess
from .vision import transcribe_image, vision_enabled, vision_model

try:
    import pymupdf
except ImportError:  # pragma: no cover
    pymupdf = None
try:
    import pytesseract
    from PIL import Image
except ImportError:  # pragma: no cover
    pytesseract = None
    Image = None

OCR_DPI = 300


def _ocr_page(page) -> tuple[str, float, object]:
    pix = page.get_pixmap(dpi=OCR_DPI)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    data = pytesseract.image_to_data(img, config="--psm 6", output_type=pytesseract.Output.DICT)
    lines: dict = {}
    confs = []
    for i, w in enumerate(data["text"]):
        w = w.strip()
        c = float(data["conf"][i])
        if not w or c < 0:
            continue
        confs.append(c)
        lines.setdefault((data["block_num"][i], data["par_num"][i], data["line_num"][i]), []).append(w)
    text = "\n".join(" ".join(v) for _, v in sorted(lines.items()))
    return text, (sum(confs) / len(confs) if confs else 0.0), img


def extract_pdf(path: Path, evidence_id: str = "", *, cache: ExtractionCache | None = None,
                known: KnownVocabulary | None = None, allow_vision: bool = True) -> dict:
    cache = cache or ExtractionCache()
    digest = file_sha256(path)
    version = f"{PIPELINE_VERSION}:pdf:vision={vision_model() if (allow_vision and vision_enabled()) else 'off'}"
    cached = cache.get(digest, version)
    if cached:
        cached["cache"] = "hit"
        return cached

    result = {"evidence_id": evidence_id, "file": path.name, "sha256": digest, "pipeline_version": version,
              "kind": "pdf", "pages": [], "failures": []}
    if pymupdf is None:
        result["failures"].append("pymupdf not installed")
        result["text"] = ""
        return result
    doc = pymupdf.open(path)
    result["page_count"] = doc.page_count
    for pno, page in enumerate(doc, start=1):
        native = page.get_text("text") or ""
        area = page.rect.width * page.rect.height
        n_images = len(page.get_images(full=True))
        q = assess(native, page_area=area, expect_identifiers=False, known=known)
        entry = {"page": pno, "method": "native", "text": native, "quality": q.as_dict(),
                 "embedded_images": n_images, "decisions": []}
        if not q.usable:
            entry["decisions"].append(f"native text unusable ({', '.join(q.reasons)}); trying OCR")
            if pytesseract is None:
                result["failures"].append(f"page {pno}: OCR unavailable (pytesseract missing)")
            else:
                try:
                    otext, conf, img = _ocr_page(page)
                    oq = assess(otext, page_area=area, expect_identifiers=False, mean_confidence=conf, known=known)
                    entry["decisions"].append(f"tesseract score={oq.score} conf={conf:.0f}")
                    if oq.score >= q.score:
                        entry.update(method="tesseract", text=otext, quality=oq.as_dict(),
                                     mean_confidence=round(conf, 1))
                        q = oq
                    layout_heavy = n_images > 0 and len(native.strip()) < 40
                    if (not q.usable or q.suspicious or layout_heavy):
                        if allow_vision and vision_enabled():
                            vtext = transcribe_image(img)
                            if vtext:
                                vq = assess(vtext, page_area=area, expect_identifiers=False, known=known)
                                entry["decisions"].append(f"vision score={vq.score}")
                                if vq.score >= q.score:
                                    entry.update(method="vision:" + str(vision_model()), text=vtext,
                                                 quality=vq.as_dict())
                            else:
                                result["failures"].append(f"page {pno}: vision call failed; kept OCR")
                        else:
                            entry["decisions"].append("vision fallback wanted but vision_unavailable")
                except Exception as e:  # noqa: BLE001
                    result["failures"].append(f"page {pno}: OCR failed: {type(e).__name__}: {e}")
        if known is not None and entry["text"]:
            entry["text"], corr = known.correct_text(entry["text"])
            if corr:
                entry["identifier_corrections"] = corr
        result["pages"].append(entry)
    doc.close()
    result["text"] = "\n".join(f"[page {p['page']}]\n{p['text']}" for p in result["pages"])
    result["methods"] = sorted({p["method"] for p in result["pages"]})
    cache.put(digest, version, result)
    result["cache"] = "miss"
    return result
