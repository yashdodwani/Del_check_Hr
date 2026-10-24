"""Intent classification: does the request text actually authorise deletion?

Labels
* DELETE_REQUEST        explicit authorisation to delete the account / personal data
* NON_DELETION_REQUEST  access, export, correction, opt-out, login help, ... (incl. explicit "do not delete")
* AMBIGUOUS             insufficient or conflicting signal

Pipeline: deterministic multilingual baseline first; only AMBIGUOUS texts go to the LLM
(``nvidia/nemotron-3.5-lightning-30b-a3b`` via the NVIDIA API). If the model is unavailable or
returns invalid output after bounded retries, the label stays ``UNRESOLVED`` — it is never turned
into permission to delete. The classifier does not decide COMPLETE/INCOMPLETE/NOT_APPLICABLE; the
pipeline maps labels to authorisation separately (see ``authorizes_deletion``).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass

from .llm import INTENT_MODEL, LLMUnavailable, NvidiaClient
from .paths import CACHE_DIR

LABELS = ("DELETE_REQUEST", "NON_DELETION_REQUEST", "AMBIGUOUS")

# Explicit refusals of deletion / statements that the account must stay.
NEGATION_PATTERNS = [
    r"not (?:asking|requesting) (?:for )?(?:a |the )?(?:deletion|erasure|removal)",
    r"not a request to (?:erase|delete|remove)",
    r"\bdo(?:n't| not) (?:close|delete|erase|remove|deactivate)",
    r"\bwithout (?:removing|deleting|erasing)",
    r"\b(?:leave|keep) (?:the |my )?(?:account|profile)(?: \w+){0,3} (?:active|as it is|in place|open)",
    r"\bleave my (?:profile|account|data|records|purchase history)[\w ,]* in place",
    r"\b(?:still|continue to|want to continue) (?:need|use|using|want)\b",
    r"\bmarketing lists? only\b|\bonly (?:from|the) marketing\b",
    r"\bne (?:supprimez|fermez) pas\b|\bno (?:eliminen|borren|cierren)\b|\bnicht (?:löschen|schließen)\b",
    r"削除しないで|삭제하지 마",
]
NON_DELETION_INTENTS = [
    r"\b(?:copy|download|export|portable)\b", r"\bwhat (?:personal )?information you hold\b",
    r"\b(?:correct|misspelled|update|replace|change)\b.*\b(?:name|surname|email|address|number|phone|details)\b",
    r"\b(?:reset|recover)\b.*\blogin\b|\bcan't access\b",
    r"\b(?:promotional|marketing) (?:emails|messages|lists)\b|\bunsubscribe\b|\bopt[- ]out\b",
    r"\binvoices?\b",
]
DELETION_CUES = [
    r"\b(?:delete|deletion|deleting|deleted)\b", r"\berase|erasure|erased\b", r"\bremov(?:e|al|ed)\b",
    r"\bclose (?:my|this|the) account\b|\bclosed my account\b|\baccount (?:closed|closure)\b",
    r"\bpermanent(?:ly)? (?:account[- ])?(?:deletion|delete|close)",
    r"supprim|suppression|effac",                 # fr
    r"elimin|borra|borrado|supresión",            # es / it / pt
    r"exclus[ãa]o|excluir|apagar",                # pt
    r"lösch|entfern",                             # de
    r"cancellazion|cancellare",                   # it
    r"verwijder",                                 # nl
    r"usunię|usuń",                               # pl
    r"radera|slett|slet",                         # sv / no / da
    r"удал",                                      # ru
    r"حذف",                                       # ar / fa
    r"削除|退会",                                  # ja
    r"삭제|탈퇴",                                  # ko
    r"删除|注销|刪除",                              # zh
    r"hapus",                                     # id / ms
    r"\bsil(?:in|me)",                            # tr
    r"हटा|डिलीट",                                 # hi
]

# A direct instruction to delete outweighs topical words such as "download" or "changing".
IMPERATIVE_DELETION = [
    r"\b(?:please|kindly|can you|could you|i want you to)\b[^.?!]{0,60}\b(?:delete|remove|erase|purge|close)\b",
    r"\b(?:delete|remove|erase|purge)\s+(?:my|the|that|this|any|all|records|it)\b",
    r"\b(?:confirm|verify)\b[^.?!]{0,60}\b(?:is gone|was deleted|been deleted|fully removed|was removed|"
    r"cannot identify|no longer identif)",
    r"\b(?:finish|complete) the deletion\b",
]

SYSTEM_PROMPT = (
    "You classify customer privacy requests for a deletion-audit system. Decide whether the request "
    "text explicitly authorises deleting the customer's account or personal data. Requests for access, "
    "a copy/export, correction, marketing opt-out, login help or anything that says not to delete are "
    "NON_DELETION_REQUEST. Use AMBIGUOUS only when the text gives insufficient or conflicting signal. "
    "The text may be in any language. Reply with JSON only: "
    '{"label": "DELETE_REQUEST|NON_DELETION_REQUEST|AMBIGUOUS", "rationale": "<= 25 words", '
    '"evidence_span": "<exact quote from the request>", "confidence": 0.0-1.0}'
)

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": list(LABELS)},
        "rationale": {"type": "string"},
        "evidence_span": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["label", "rationale", "evidence_span", "confidence"],
}


@dataclass
class IntentResult:
    label: str                 # DELETE_REQUEST / NON_DELETION_REQUEST / AMBIGUOUS / UNRESOLVED
    method: str                # baseline / llm:<model> / unresolved
    rationale: str = ""
    evidence_span: str = ""
    confidence: float = 0.0
    baseline_label: str = ""
    latency_s: float = 0.0
    error: str = ""


def _find(patterns, text):
    for p in patterns:
        m = re.search(p, text, re.I)
        if m:
            return m
    return None


def baseline(text: str) -> IntentResult:
    t = " ".join((text or "").split())
    neg = _find(NEGATION_PATTERNS, t)
    other = _find(NON_DELETION_INTENTS, t)
    dele = _find(DELETION_CUES, t)
    if neg:
        return IntentResult("NON_DELETION_REQUEST", "baseline", "explicit statement that the account/data must stay",
                            neg.group(0), 0.95, "NON_DELETION_REQUEST")
    imp = _find(IMPERATIVE_DELETION, t)
    if imp:
        return IntentResult("DELETE_REQUEST", "baseline", "direct instruction to delete / verify deletion",
                            imp.group(0), 0.9, "DELETE_REQUEST")
    if dele and not other:
        return IntentResult("DELETE_REQUEST", "baseline", "explicit deletion wording", dele.group(0), 0.85,
                            "DELETE_REQUEST")
    if other and not dele:
        return IntentResult("NON_DELETION_REQUEST", "baseline", "non-deletion request type", other.group(0), 0.8,
                            "NON_DELETION_REQUEST")
    reason = "both deletion and non-deletion cues" if (dele and other) else "no recognised cue"
    return IntentResult("AMBIGUOUS", "baseline", reason, (dele or other).group(0) if (dele or other) else "", 0.3,
                        "AMBIGUOUS")


def _validate(raw: str, text: str) -> dict:
    raw = raw.strip()
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError("no JSON object in response")
    data = json.loads(m.group(0))
    if data.get("label") not in LABELS:
        raise ValueError(f"invalid label {data.get('label')!r}")
    conf = float(data.get("confidence", 0))
    if not 0 <= conf <= 1:
        raise ValueError("confidence out of range")
    span = str(data.get("evidence_span", ""))
    if span and span.lower() not in text.lower():
        data["evidence_span_verified"] = False  # keep, but flag: spans must quote the request
    return data


class IntentClassifier:
    def __init__(self, client: NvidiaClient | None = None, model: str = INTENT_MODEL, use_llm: bool = True,
                 llm_for_all: bool = False, max_attempts: int = 3):
        self.client = client or NvidiaClient(timeout=30, max_retries=2)
        self.model = model
        self.use_llm = use_llm and self.client.available
        self.llm_for_all = llm_for_all
        self.max_attempts = max_attempts
        self.cache_dir = CACHE_DIR / "intent"
        self.model_checked = None
        self.stats = {"baseline": 0, "llm": 0, "llm_failures": 0, "unresolved": 0, "cache_hits": 0}

    def check_model(self) -> bool:
        """Verify the exact model id is served by the configured endpoint (no silent substitution)."""
        if self.model_checked is None:
            self.model_checked = self.use_llm and self.client.model_available(self.model)
            if not self.model_checked:
                self.use_llm = False
        return self.model_checked

    def _cache_path(self, text):
        h = hashlib.sha256(f"{self.model}\n{SYSTEM_PROMPT}\n{text}".encode()).hexdigest()
        return self.cache_dir / f"{h}.json"

    def llm(self, text: str) -> IntentResult:
        p = self._cache_path(text)
        if p.exists():
            self.stats["cache_hits"] += 1
            return IntentResult(**json.loads(p.read_text(encoding="utf-8")))
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Request:\n{text}"}]
        last = ""
        for _ in range(self.max_attempts):
            t0 = time.monotonic()
            try:
                raw, _usage = self.client.chat(self.model, messages, temperature=0.0, max_tokens=300,
                                               response_format={"type": "json_object"})
                data = _validate(raw, text)
                res = IntentResult(data["label"], f"llm:{self.model}", data.get("rationale", "")[:300],
                                   data.get("evidence_span", "")[:300], float(data.get("confidence", 0)),
                                   latency_s=round(time.monotonic() - t0, 3))
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(asdict(res), ensure_ascii=False), encoding="utf-8")
                return res
            except (LLMUnavailable, ValueError, json.JSONDecodeError) as e:
                last = f"{type(e).__name__}: {e}"
        self.stats["llm_failures"] += 1
        return IntentResult("UNRESOLVED", "unresolved", error=last)

    def classify(self, text: str) -> IntentResult:
        base = baseline(text)
        if base.label != "AMBIGUOUS" and not self.llm_for_all:
            self.stats["baseline"] += 1
            return base
        if self.use_llm and self.check_model():
            res = self.llm(text)
            res.baseline_label = base.label
            if res.label != "UNRESOLVED":
                self.stats["llm"] += 1
                return res
        else:
            res = IntentResult("UNRESOLVED", "unresolved", error="LLM unavailable (NVIDIA_API_KEY unset or model not served)")
        self.stats["unresolved"] += 1
        res.baseline_label = base.label
        res.rationale = res.rationale or f"baseline: {base.rationale}"
        return res


def authorizes_deletion(result: IntentResult) -> bool:
    """Only an explicit DELETE_REQUEST authorises deletion; AMBIGUOUS/UNRESOLVED never do."""
    return result.label == "DELETE_REQUEST"
