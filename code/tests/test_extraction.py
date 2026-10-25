"""PDF/image extraction, fallback selection, quality gate and caching."""

import io

import pytest
from PIL import Image, ImageDraw, ImageFont

from ghostaudit.extraction import image as image_mod
from ghostaudit.extraction.cache import ExtractionCache
from ghostaudit.extraction.image import extract_image
from ghostaudit.extraction.pdf import extract_pdf
from ghostaudit.extraction.quality import KnownVocabulary, assess

pymupdf = pytest.importorskip("pymupdf")


def _mixed_pdf(path):
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((72, 90), "Support reference CS-1 / support_123 for jane.doe@mail.example.test", fontsize=11)
    p1.insert_text((72, 110), "This page has a proper digital text layer with several words in it.", fontsize=11)
    img = Image.new("L", (1200, 400), 255)
    ImageDraw.Draw(img).text((40, 100), "Export package export_456 for jane.doe@mail.example.test", fill=0,
                             font=ImageFont.load_default(size=36))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    p2 = doc.new_page()
    p2.insert_image(pymupdf.Rect(36, 72, 576, 252), stream=buf.getvalue())
    doc.save(path)
    doc.close()


def test_pdf_native_page_kept_and_scanned_page_ocrd(tmp_path):
    pdf = tmp_path / "mixed.pdf"
    _mixed_pdf(pdf)
    res = extract_pdf(pdf, "ev_x", cache=ExtractionCache(tmp_path / "c"), allow_vision=False)
    p1, p2 = res["pages"]
    assert p1["method"] == "native" and "support_123" in p1["text"]
    assert p2["method"] == "tesseract" and "export_456" in p2["text"].replace(" ", "_").replace("__", "_")
    assert any("unusable" in d for d in p2["decisions"])
    assert res["evidence_id"] == "ev_x" and [p["page"] for p in res["pages"]] == [1, 2]


def test_pdf_cache_hit_on_second_call(tmp_path):
    pdf = tmp_path / "mixed.pdf"
    _mixed_pdf(pdf)
    cache = ExtractionCache(tmp_path / "c")
    first = extract_pdf(pdf, cache=cache, allow_vision=False)
    second = extract_pdf(pdf, cache=cache, allow_vision=False)
    assert first["cache"] == "miss" and second["cache"] == "hit"
    assert first["sha256"] == second["sha256"] and second["pages"][1]["text"] == first["pages"][1]["text"]


def test_cache_key_changes_with_content(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    Image.new("RGB", (50, 50), "white").save(a)
    Image.new("RGB", (50, 50), "black").save(b)
    cache = ExtractionCache(tmp_path / "c")
    ra = extract_image(a, cache=cache, allow_vision=False)
    rb = extract_image(b, cache=cache, allow_vision=False)
    assert ra["sha256"] != rb["sha256"] and rb["cache"] == "miss"


def test_quality_gate_rejects_nonempty_garbage():
    q = assess("�� (cid:12) ~~ ;; ..", expect_identifiers=True)
    assert not q.usable and q.chars > 0


def test_quality_gate_flags_malformed_ids_and_broken_emails():
    q = assess("Archive backup.00001 owner oyelaran.bisi@maiexampletest region eu-west-1 restore allowed today")
    assert q.suspicious


def test_known_vocabulary_corrects_ocr_email():
    kv = KnownVocabulary(emails=["areed1433@gmail.com"])
    text, corr = kv.correct_text("Account areed1433@gmall.com")
    assert text == "Account areed1433@gmail.com" and corr


def test_image_vision_fallback_selected_but_unavailable(tmp_path, monkeypatch):
    """Low-quality OCR must request the vision fallback; with no model configured it is recorded, not faked."""
    p = tmp_path / "tiny.png"
    img = Image.new("RGB", (300, 40), "white")
    ImageDraw.Draw(img).text((2, 2), "bkp_9 a@b", fill=(170, 170, 170), font=ImageFont.load_default(size=6))
    img.save(p)
    for k in ("GEMINI_API_KEY", "Gemini_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(k, raising=False)  # no key → vision cannot be used, must not be faked
    res = extract_image(p, cache=ExtractionCache(tmp_path / "c"))
    assert res["needs_vision"] is True
    assert res["method"] == "tesseract"
    assert any(d.get("skipped") == "vision_unavailable" for d in res["decisions"])


def test_image_vision_fallback_used_when_configured(tmp_path, monkeypatch):
    p = tmp_path / "tiny.png"
    img = Image.new("RGB", (300, 40), "white")
    ImageDraw.Draw(img).text((2, 2), "bkp_9 a@b", fill=(170, 170, 170), font=ImageFont.load_default(size=6))
    img.save(p)
    monkeypatch.setattr(image_mod, "vision_enabled", lambda: True)
    monkeypatch.setattr(image_mod, "vision_model", lambda: "test-vision")
    monkeypatch.setattr(image_mod, "transcribe_image",
                        lambda _img: "Archive backup_90091 | owner jane.doe@mail.example.test | restore Allowed | updated today")
    res = extract_image(p, cache=ExtractionCache(tmp_path / "c"))
    assert res["method"] == "tesseract+vision:test-vision" and "backup_90091" in res["text"]


def test_synthetic_screenshot_ocr_recovers_ids(synthetic_dataset, tmp_path):
    p = synthetic_dataset / "evidence/images/syn_backup_console_20260805.png"
    res = extract_image(p, cache=ExtractionCache(tmp_path / "c"), allow_vision=False)
    assert "backup_90091" in res["text"] and "oyelaran.bisi@mail.example.test" in res["text"]
    assert all(l["box"] and len(l["box"]) == 4 for l in res["lines"])


def test_vision_model_default(monkeypatch):
    from ghostaudit.extraction import vision
    monkeypatch.delenv("GEMINI_VISION_MODEL", raising=False)
    assert vision.vision_model() == "gemini-3.5-flash-lite"


def test_vision_disabled_when_model_not_served(monkeypatch):
    from ghostaudit.extraction import vision
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setattr(vision, "_served", {})
    monkeypatch.setattr(vision.GeminiClient, "model_available", lambda self, m: False)
    assert vision.vision_enabled() is False


def test_merge_keeps_ocr_rows_and_takes_vision_spellings():
    from ghostaudit.extraction.vision import merge_with_ocr
    ocr = ["backup.00091 oyelaran.bisi@maiexample.test eu-west-1 Allowed", "backup_90098 vendor@mail.example.test Blocked"]
    vision = "backup_90091\noyelaran.bisi@mail.example.test\neu-west-1\nbackup_90098\nvendor@mail.example.test"
    merged, corr = merge_with_ocr(ocr, vision)
    assert merged[0] == "backup_90091 oyelaran.bisi@mail.example.test eu-west-1 Allowed"  # row kept, ids fixed
    assert merged[1] == ocr[1] and len(corr) == 2


def test_gemini_minimal_thinking_and_empty_response_is_error(monkeypatch):
    from ghostaudit.gemini import GeminiClient
    from ghostaudit.llm import LLMUnavailable
    sent = {}

    def fake(self, method, path, body=None):
        sent.update(body)
        return {"candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}], "usageMetadata": {}}
    monkeypatch.setattr(GeminiClient, "_request", fake)
    monkeypatch.setenv("GEMINI_RPM", "0")
    c = GeminiClient(api_key="x", max_retries=1)
    with pytest.raises(LLMUnavailable):
        c.generate("m", [{"text": "hi"}])
    assert sent["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "minimal"}
