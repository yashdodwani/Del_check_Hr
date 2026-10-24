"""Intent classification: baseline, LLM adapter validation, and fail-safe handling."""

import json

import pytest

from ghostaudit import intent as intent_mod
from ghostaudit.intent import IntentClassifier, authorizes_deletion, baseline
from ghostaudit.llm import LLMUnavailable

# controlled cases (multilingual, incl. deletion words that do NOT authorise deletion)
CONTROLLED = [
    ("Please permanently delete my account and personal data.", "DELETE_REQUEST"),
    ("Je demande la suppression définitive de mon compte.", "DELETE_REQUEST"),
    ("Solicito a exclusão permanente da conta associada a este e-mail.", "DELETE_REQUEST"),
    ("Bitte löschen Sie mein Konto vollständig.", "DELETE_REQUEST"),
    ("このメールアドレスに関連するアカウントの完全な削除を依頼します。", "DELETE_REQUEST"),
    ("이 이메일 주소에 연결된 계정의 영구 삭제를 요청합니다.", "DELETE_REQUEST"),
    ("Quiero que eliminen mi cuenta y mis datos.", "DELETE_REQUEST"),
    ("I closed my account, but an old download link still works. Please delete that export file as well.",
     "DELETE_REQUEST"),
    ("Can you verify that my account was deleted?", "DELETE_REQUEST"),
    ("Please stop the promotional emails. I still need the account, so I am not asking for deletion.",
     "NON_DELETION_REQUEST"),
    ("Could you send me a copy of my data? Please don't delete or change anything.", "NON_DELETION_REQUEST"),
    ("My surname is misspelled. Can you correct it without removing any of my records?", "NON_DELETION_REQUEST"),
    ("I can't access my profile and need help resetting the login. Do not close the account.",
     "NON_DELETION_REQUEST"),
    ("Please remove me from marketing lists only. I still want receipts.", "NON_DELETION_REQUEST"),
    ("This is an information request, not a request to erase it.", "NON_DELETION_REQUEST"),
    ("I read your deletion policy. I am not asking for deletion.", "NON_DELETION_REQUEST"),
]


@pytest.mark.parametrize("text,label", CONTROLLED)
def test_baseline_controlled_cases(text, label):
    assert baseline(text).label == label


def test_baseline_reports_ambiguous_without_cues():
    assert baseline("Hello, I have a question about my order.").label in ("AMBIGUOUS", "NON_DELETION_REQUEST")
    assert baseline("Hallo").label == "AMBIGUOUS"


class FakeClient:
    available = True

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def model_available(self, model):
        return model == intent_mod.INTENT_MODEL

    def chat(self, *a, **k):
        self.calls += 1
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r, {}


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(intent_mod, "CACHE_DIR", tmp_path)


def test_llm_used_only_for_ambiguous_and_validated():
    fake = FakeClient([json.dumps({"label": "DELETE_REQUEST", "rationale": "asks to close", "evidence_span": "Hallo",
                                   "confidence": 0.7})])
    clf = IntentClassifier(client=fake)
    clf.cache_dir = clf.cache_dir  # noqa
    assert clf.classify("Please delete my account.").method == "baseline" and fake.calls == 0
    res = clf.classify("Hallo")
    assert res.label == "DELETE_REQUEST" and res.method.startswith("llm:") and fake.calls == 1


def test_llm_invalid_output_retried_then_unresolved_never_authorises():
    fake = FakeClient(["not json", json.dumps({"label": "MAYBE"}), LLMUnavailable("timeout")])
    clf = IntentClassifier(client=fake, max_attempts=3)
    res = clf.classify("Hallo")
    assert res.label == "UNRESOLVED" and fake.calls == 3
    assert not authorizes_deletion(res)


def test_model_not_served_is_not_substituted():
    class NoModel(FakeClient):
        def model_available(self, model):
            return False
    fake = NoModel([])
    clf = IntentClassifier(client=fake)
    res = clf.classify("Hallo")
    assert res.label == "UNRESOLVED" and fake.calls == 0 and clf.use_llm is False


def test_only_explicit_delete_authorises():
    for label, ok in [("DELETE_REQUEST", True), ("NON_DELETION_REQUEST", False), ("AMBIGUOUS", False),
                      ("UNRESOLVED", False)]:
        assert authorizes_deletion(intent_mod.IntentResult(label, "x")) is ok
