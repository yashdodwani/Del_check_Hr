"""Content-addressed extraction cache.

Key = sha256(file bytes) + pipeline version + vision model id (when vision is enabled), so a file is
re-extracted only when its contents or the extraction pipeline change.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..paths import CACHE_DIR


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


class ExtractionCache:
    def __init__(self, root: Path | None = None):
        self.root = Path(root or CACHE_DIR / "extraction")
        self.hits = 0
        self.misses = 0

    def _path(self, digest: str, version: str) -> Path:
        safe = version.replace("/", "_").replace(":", "_")
        return self.root / f"{digest[:2]}" / f"{digest}.{safe}.json"

    def get(self, digest: str, version: str) -> dict | None:
        p = self._path(digest, version)
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                self.hits += 1
                return data
            except json.JSONDecodeError:
                pass
        self.misses += 1
        return None

    def put(self, digest: str, version: str, data: dict) -> None:
        p = self._path(digest, version)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)
