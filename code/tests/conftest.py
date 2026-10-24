import sys
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))

SYN_BUILD = CODE / "synthetic" / "build" / "dataset"


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory):
    """Regenerate the synthetic dataset once per session (deterministic) and return its root."""
    from synthetic.generate import Builder  # noqa: WPS433

    Builder().build()
    return SYN_BUILD


@pytest.fixture(scope="session")
def synthetic_run(synthetic_dataset, tmp_path_factory):
    import os

    from ghostaudit.metrics import load_expected_csv
    from ghostaudit.pipeline import run

    tmp = tmp_path_factory.mktemp("synrun")
    os.environ["GHOST_CACHE_DIR"] = str(tmp / "cache")
    import ghostaudit.paths as paths
    paths.CACHE_DIR = tmp / "cache"
    out = tmp / "out.csv"
    summary = run(synthetic_dataset, output=out, use_llm=False, allow_vision=False, log=lambda *_: None)
    return {"summary": summary, "pred": load_expected_csv(out),
            "expected": load_expected_csv(CODE / "synthetic" / "expected_output.csv"), "out": out}
