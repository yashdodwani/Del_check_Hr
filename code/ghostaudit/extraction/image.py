"""Image evidence: Tesseract first (with light preprocessing), vision model only as a fallback.

Lines are rebuilt from Tesseract word boxes so every extracted line keeps its bounding box.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageFilter, ImageOps

from . import PIPELINE_VERSION
from .cache import ExtractionCache, file_sha256
from .quality import KnownVocabulary, assess
from .vision import merge_with_ocr, transcribe_image, vision_enabled, vision_model

try:
    import pytesseract
except ImportError:  # pragma: no cover - surfaced in result metadata
    pytesseract = None

# preprocessing variants tried in order; the best-scoring transcript wins
VARIANTS = ("gray_x2", "gray_x3", "gray_x2_threshold", "original")


def _preprocess(img: Image.Image, variant: str) -> Image.Image:
    g = ImageOps.grayscale(img)
    if variant == "original":
        return img.convert("RGB")
    factor = 3 if variant.startswith("gray_x3") else 2  # small UI text needs more upscaling
    g = g.resize((g.width * factor, g.height * factor), Image.LANCZOS)
    if variant == "gray_x2_threshold":
        g = ImageOps.autocontrast(g).filter(ImageFilter.SHARPEN)
        g = g.point(lambda p: 255 if p > 165 else 0)
    return g


def _ocr_lines(img: Image.Image, scale: float) -> tuple[list[dict], float]:
    """Run Tesseract and regroup words into lines with boxes in original-image coordinates."""
    data = pytesseract.image_to_data(img, config="--psm 6", output_type=pytesseract.Output.DICT)
    lines: dict = {}
    confs = []
    for i, word in enumerate(data["text"]):
        word = word.strip()
        conf = float(data["conf"][i])
        if not word or conf < 0:
            continue
        confs.append(conf)
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        x, y, w, h = (int(data[k][i] / scale) for k in ("left", "top", "width", "height"))
        ln = lines.setdefault(key, {"words": [], "confs": [], "box": [x, y, x + w, y + h]})
        ln["words"].append(word)
        ln["confs"].append(conf)
        b = ln["box"]
        ln["box"] = [min(b[0], x), min(b[1], y), max(b[2], x + w), max(b[3], y + h)]
    out = []
    for key in sorted(lines, key=lambda k: (lines[k]["box"][1], lines[k]["box"][0])):
        ln = lines[key]
        out.append({"text": " ".join(ln["words"]), "box": ln["box"],
                    "confidence": round(sum(ln["confs"]) / len(ln["confs"]), 1)})
    mean = sum(confs) / len(confs) if confs else 0.0
    return out, mean


def extract_image(path: Path, evidence_id: str = "", *, cache: ExtractionCache | None = None,
                  known: KnownVocabulary | None = None, layout_sensitive: bool = True,
                  allow_vision: bool = True) -> dict:
    cache = cache or ExtractionCache()
    digest = file_sha256(path)
    version = f"{PIPELINE_VERSION}:image:vision={vision_model() if (allow_vision and vision_enabled()) else 'off'}"
    cached = cache.get(digest, version)
    if cached:
        cached["cache"] = "hit"
        return cached

    result = {"evidence_id": evidence_id, "file": path.name, "sha256": digest, "pipeline_version": version,
              "kind": "image", "decisions": [], "failures": []}
    if pytesseract is None:
        result["failures"].append("pytesseract not installed")
        best = {"text": "", "lines": [], "variant": None, "mean_confidence": 0.0}
        q = assess("")
    else:
        img = Image.open(path)
        candidates = []
        for variant in VARIANTS:
            try:
                pre = _preprocess(img, variant)
                scale = pre.width / img.width
                lines, mean = _ocr_lines(pre, scale)
                text = "\n".join(l["text"] for l in lines)
                q = assess(text, mean_confidence=mean, known=known)
                candidates.append((q.score, variant, text, lines, mean, q))
                result["decisions"].append({"variant": variant, "score": q.score, "mean_confidence": round(mean, 1),
                                            "reasons": q.reasons})
            except Exception as e:  # noqa: BLE001 - record and try the next variant
                result["failures"].append(f"{variant}: {type(e).__name__}: {e}")
        candidates.sort(key=lambda c: -c[0])
        if candidates:
            score, variant, text, lines, mean, q = candidates[0]
        else:
            score, variant, text, lines, mean, q = 0.0, None, "", [], 0.0, assess("")
        best = {"text": text, "lines": lines, "variant": variant, "mean_confidence": round(mean, 1)}

    corrections = []
    if known is not None and best["text"]:
        best["text"], corrections = known.correct_text(best["text"])
        for ln in best["lines"]:
            ln["text"], _ = known.correct_text(ln["text"])
    result["identifier_corrections"] = corrections

    method = "tesseract"
    needs_vision = (not q.usable) or bool(q.suspicious) or (layout_sensitive and q.score < 0.8)
    if needs_vision:
        if allow_vision and vision_enabled() and pytesseract is not None:
            vtext = transcribe_image(Image.open(path))
            if vtext:
                merged, vcorr = merge_with_ocr([l["text"] for l in best["lines"]], vtext)
                mtext = "\n".join(merged)
                vq = assess(mtext, known=known)
                result["decisions"].append({"variant": "tesseract+vision", "score": vq.score, "reasons": vq.reasons,
                                            "vision_corrections": vcorr})
                result["vision_text"] = vtext
                if vq.score >= q.score:
                    boxes = [l["box"] for l in best["lines"]]
                    best = {"text": mtext, "lines": [{"text": t, "box": boxes[i] if i < len(boxes) else None,
                                                      "confidence": None} for i, t in enumerate(merged)],
                            "variant": best["variant"], "mean_confidence": best["mean_confidence"]}
                    q, method = vq, "tesseract+vision:" + str(vision_model())
            else:
                result["failures"].append("vision call failed; kept OCR output")
        else:
            result["decisions"].append({"variant": "vision", "skipped": "vision_unavailable"})
    result.update(text=best["text"], lines=best["lines"], method=method, ocr_variant=best["variant"],
                  mean_confidence=best["mean_confidence"], quality=q.as_dict(), needs_vision=needs_vision)
    cache.put(digest, version, result)
    result["cache"] = "miss"
    return result
