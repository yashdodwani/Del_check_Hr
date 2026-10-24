"""Ghost Data Still Exists — deletion-audit agent entry point.

    python code/main.py                      # full run → ./output.csv
    python code/main.py --no-llm             # deterministic baseline only (no API calls)
    python code/main.py --requests sample_deletion_requests.csv --output /tmp/sample_out.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ghostaudit.paths import EVALUATION_DIR, OUTPUT_CSV, load_dotenv  # noqa: E402
from ghostaudit.pipeline import run  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=None, help="dataset directory (default: ./dataset)")
    ap.add_argument("--requests", default="deletion_requests.csv", help="request file inside the dataset dir")
    ap.add_argument("--output", type=Path, default=OUTPUT_CSV)
    ap.add_argument("--no-llm", action="store_true", help="never call the NVIDIA intent model")
    ap.add_argument("--no-vision", action="store_true", help="never call a vision model for OCR fallback")
    ap.add_argument("--llm-all", action="store_true", help="send every request to the LLM (benchmarking)")
    ap.add_argument("--summary", type=Path, default=EVALUATION_DIR / "run_summary.json")
    args = ap.parse_args(argv)
    load_dotenv()
    summary = run(args.dataset, args.requests, args.output, use_llm=not args.no_llm,
                  allow_vision=not args.no_vision, llm_for_all=args.llm_all, summary_path=args.summary)
    if summary["validation_errors"]:
        print("OUTPUT VALIDATION FAILED:", *summary["validation_errors"][:20], sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
