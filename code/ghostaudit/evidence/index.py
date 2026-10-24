"""Evidence index: parses every manifest file once into Observations and indexes them."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from ..data import Dataset
from ..extraction.cache import ExtractionCache
from ..extraction.image import extract_image
from ..extraction.pdf import extract_pdf
from ..extraction.quality import KnownVocabulary
from ..model import Observation
from .documents import observations_from_text
from .logs import parse_log
from .records import parse_json, parse_sql, parse_text_records

LINK_DETAIL_RE = re.compile(r"linked (\S+) \(([^)]+)\) to (\S+) and (user_\d+)|confirmed (\S+) \(([^)]+)\) belongs only to (\S+) and (user_\d+)")


@dataclass
class EvidenceFile:
    file_id: str
    path: str
    origin: str
    captured_at: str
    observations: list = field(default_factory=list)
    scope: set = field(default_factory=set)       # systems this file fully enumerates (snapshots)
    extraction: dict | None = None
    error: str | None = None


def catalog_links(activity: list[dict]) -> tuple[dict, list]:
    """LINK_RESOURCE_IDENTITY rows → ({alias: resource_id}, [(activity_id, time, resource, email, user)])."""
    aliases, links = {}, []
    for a in activity:
        if a["event_type"] != "LINK_RESOURCE_IDENTITY":
            continue
        m = LINK_DETAIL_RE.search(a["details"])
        if not m:
            continue
        g = [x for x in m.groups() if x is not None]
        rid, alias, email, user = g
        aliases[alias] = rid
        links.append((a["activity_id"], a["event_time"], rid, email.rstrip(".;"), user,
                      "belongs only" in a["details"]))
    return aliases, links


class EvidenceIndex:
    def __init__(self, ds: Dataset, *, cache: ExtractionCache | None = None, allow_vision: bool = True):
        self.ds = ds
        self.cache = cache or ExtractionCache()
        self.allow_vision = allow_vision
        self.aliases, self.catalog = catalog_links(ds.activity)
        self.known = KnownVocabulary(
            emails=[r["email"] for r in ds.identities]
            + [a[v] for a in ds.audit_log if a["field_name"] == "email" for v in ("old_value", "new_value")],
            user_ids=[r["user_id"] for r in ds.identities],
            resource_ids=[a["resource_id"] for a in ds.activity],
        )
        self.files: dict[str, EvidenceFile] = {}
        self.by_resource: dict = defaultdict(list)
        self.extraction_stats: dict = defaultdict(int)
        self._load()

    def _load(self):
        for row in self.ds.manifest:
            ef = EvidenceFile(row["file_id"], row["file_path"], row["origin"], row["captured_at"])
            path = self.ds.root / row["file_path"]
            try:
                ef.observations, ef.scope = self._parse(ef, path)
            except Exception as e:  # noqa: BLE001 - one broken file must not stop the audit
                ef.error = f"{type(e).__name__}: {e}"
            self.files[ef.file_id] = ef
            for o in ef.observations:
                self.by_resource[o.resource_id].append(o)

    def _parse(self, ef: EvidenceFile, path):
        suffix = path.suffix.lower()
        text = None
        if suffix in (".log", ".txt", ".json", ".sql"):
            text = path.read_text(encoding="utf-8")
        if suffix == ".log":
            return parse_log(text, ef.file_id), set()
        if suffix == ".json":
            return parse_json(text, ef.file_id, ef.captured_at)
        if suffix == ".sql":
            return parse_sql(text, ef.file_id, ef.captured_at)
        if suffix == ".txt":
            obs = parse_text_records(text, ef.file_id, ef.captured_at)
            if not obs:
                obs = observations_from_text(text, ef.file_id, ef.captured_at, aliases=self.aliases, origin=ef.origin)
            return obs, set()
        if suffix == ".pdf":
            ex = extract_pdf(path, ef.file_id, cache=self.cache, known=self.known, allow_vision=self.allow_vision)
        elif suffix in (".png", ".jpg", ".jpeg", ".webp"):
            ex = extract_image(path, ef.file_id, cache=self.cache, known=self.known, allow_vision=self.allow_vision)
        else:
            return [], set()
        ef.extraction = {k: ex.get(k) for k in ("method", "methods", "cache", "quality", "failures",
                                                  "identifier_corrections", "needs_vision")}
        self.extraction_stats[f"{ex['kind']}:{ex.get('method') or ','.join(ex.get('methods', []))}"] += 1
        self.extraction_stats[f"cache_{ex.get('cache')}"] += 1
        return observations_from_text(ex.get("text", ""), ef.file_id, ef.captured_at, aliases=self.aliases,
                                      origin=ef.origin), set()

    def files_before(self, cutoff: str) -> list[EvidenceFile]:
        return [f for f in self.files.values() if f.captured_at <= cutoff]

    def observations(self, resource_id: str, cutoff: str) -> list[Observation]:
        return [o for o in self.by_resource.get(resource_id, []) if self.files[o.evidence_id].captured_at <= cutoff]
