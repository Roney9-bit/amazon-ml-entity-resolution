#!/usr/bin/env python3
"""Run reproducible EDA on the training files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from business_entity_resolution.evaluation import load_ground_truth
from business_entity_resolution.utils.io import read_source


def summary(df: pd.DataFrame, name: str) -> None:
    print(f"\n=== {name} ===")
    print(f"Rows: {len(df):,}")
    print(f"Unique entity IDs: {df['entity_id'].nunique():,}")
    for col in ["business_name", "business_address", "country"]:
        missing = df[col].fillna("").astype(str).str.strip().eq("").sum()
        print(f"Missing {col}: {missing:,} ({missing / len(df):.2%})")
    print("Country distribution:")
    print(df["country"].value_counts(dropna=False).head(20).to_string())
    print("Name length:", df["business_name"].fillna("").astype(str).str.len().describe().round(2).to_dict())
    print("Address length:", df["business_address"].fillna("").astype(str).str.len().describe().round(2).to_dict())


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA for Amazon ML business entity resolution")
    parser.add_argument("--data-dir", default="data/train", help="Directory containing train_*.tsv files")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    s1 = read_source(data_dir / "train_source1.tsv")
    s2 = read_source(data_dir / "train_source2.tsv")
    s3 = read_source(data_dir / "train_source3.tsv")
    gt = load_ground_truth(data_dir / "train_ground_truth.tsv")

    summary(s1, "Source 1")
    summary(s2, "Source 2")
    summary(s3, "Source 3")

    match_counts = pd.Series({s1_id: len(gt.get(s1_id, set())) for s1_id in s1["entity_id"]})
    print("\n=== Ground-truth match counts per Source 1 ===")
    print(match_counts.describe().round(2).to_string())
    print("\nDistribution:")
    print(match_counts.value_counts().sort_index().to_string())

    s2_matches = sum(any(x.startswith("S2-") for x in ids) for ids in gt.values())
    s3_matches = sum(any(x.startswith("S3-") for x in ids) for ids in gt.values())
    both_matches = sum(any(x.startswith("S2-") for x in ids) and any(x.startswith("S3-") for x in ids) for ids in gt.values())
    singletons = sum(len(ids) == 0 for ids in gt.values())
    print(f"\nS1 entities with at least one S2 match: {s2_matches:,}")
    print(f"S1 entities with at least one S3 match: {s3_matches:,}")
    print(f"S1 entities matching both S2 and S3: {both_matches:,}")
    print(f"S1 singletons: {singletons:,}")


if __name__ == "__main__":
    main()
