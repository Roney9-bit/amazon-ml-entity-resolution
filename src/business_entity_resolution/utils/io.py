"""I/O helpers for challenge TSV files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def read_source(path: str | Path) -> pd.DataFrame:
    """Read a source TSV while preserving optional cleaned columns."""
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    missing = set(COLUMNS).difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    return df.copy()


def write_tsv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, lineterminator="\n")
