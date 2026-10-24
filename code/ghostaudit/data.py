"""Loading of the participant-facing CSV files."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .paths import DATASET_DIR


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@dataclass
class Dataset:
    root: Path
    identities: list
    audit_log: list
    activity: list
    rules: list
    manifest: list

    @classmethod
    def load(cls, root: Path | None = None) -> "Dataset":
        root = Path(root or DATASET_DIR)
        return cls(
            root=root,
            identities=read_csv(root / "user_identities.csv"),
            audit_log=read_csv(root / "user_audit_log.csv"),
            activity=read_csv(root / "system_activity.csv"),
            rules=read_csv(root / "deletion_rules.csv"),
            manifest=read_csv(root / "evidence_files.csv"),
        )

    def requests(self, name: str = "deletion_requests.csv") -> list[dict]:
        return read_csv(self.root / name)
