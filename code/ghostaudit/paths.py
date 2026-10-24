"""Repository-relative paths. Everything resolves from this file so the tool works from any cwd."""

from pathlib import Path
import os

CODE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = CODE_DIR.parent
DATASET_DIR = Path(os.environ.get("GHOST_DATASET_DIR", REPO_ROOT / "dataset"))
CACHE_DIR = Path(os.environ.get("GHOST_CACHE_DIR", CODE_DIR / ".cache"))
OUTPUT_CSV = REPO_ROOT / "output.csv"
EVALUATION_DIR = CODE_DIR / "evaluation"


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader (no dependency). Existing environment variables win."""
    path = path or (REPO_ROOT / ".env")
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
