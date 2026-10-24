"""Build code.zip (README.md + code/) without dataset, caches, virtualenvs, secrets or nested archives.

    python code/package.py
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXCLUDE_DIRS = {"__pycache__", ".cache", ".pytest_cache", ".venv", "venv", "node_modules", ".git", "build"}
EXCLUDE_SUFFIX = {".zip", ".tar", ".gz", ".tgz", ".7z", ".rar", ".pyc"}
EXCLUDE_NAMES = {".env", ".DS_Store"}


def files():
    yield ROOT / "README.md"
    for p in sorted((ROOT / "code").rglob("*")):
        rel = p.relative_to(ROOT)
        if p.is_dir() or set(rel.parts) & EXCLUDE_DIRS or p.suffix in EXCLUDE_SUFFIX or p.name in EXCLUDE_NAMES:
            continue
        yield p


def main() -> int:
    out = ROOT / "code.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files():
            z.write(p, p.relative_to(ROOT).as_posix())
    names = zipfile.ZipFile(out).namelist()
    for required in ("README.md", "code/main.py", "code/evaluation/evaluation_report.md",
                     "code/evaluation/usage_report.md"):
        if required not in names:
            print("missing from code.zip:", required)
            return 1
    print(f"wrote {out} ({len(names)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
