#!/usr/bin/env python3
"""Local validator modeled on the challenge submission rules.

Uses only the Python standard library so it can also run in a minimal environment.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_tsv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty file: {path}")
    return rows[0], rows[1:]


def parse_ids(value: str) -> list[str]:
    if value == "":
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def validate(matching_path: Path, candidate_path: Path, test_dir: Path) -> list[str]:
    errors: list[str] = []

    s1_header, s1_rows = read_tsv(test_dir / "test_source1.tsv")
    s2_header, s2_rows = read_tsv(test_dir / "test_source2.tsv")
    s3_header, s3_rows = read_tsv(test_dir / "test_source3.tsv")

    expected_source_header = ["entity_id", "business_name", "business_address", "country"]
    if s1_header != expected_source_header:
        errors.append("test_source1.tsv has unexpected columns")
    if s2_header != expected_source_header:
        errors.append("test_source2.tsv has unexpected columns")
    if s3_header != expected_source_header:
        errors.append("test_source3.tsv has unexpected columns")

    test_s1 = [r[0] for r in s1_rows]
    test_s2 = {r[0] for r in s2_rows}
    test_s3 = {r[0] for r in s3_rows}
    valid_target_ids = test_s2 | test_s3

    for path, expected in [
        (matching_path, ["source1_entity_id", "matched_entity_ids"]),
        (candidate_path, ["source1_entity_id", "candidate_entity_ids"]),
    ]:
        if not path.exists():
            errors.append(f"Missing file: {path}")
            continue
        header, rows = read_tsv(path)
        if header != expected:
            errors.append(f"{path.name}: expected header {expected}, found {header}")
        for i, row in enumerate(rows, start=2):
            if len(row) != 2:
                errors.append(f"{path.name}: line {i} must have exactly 2 tab-separated columns")

    if errors:
        return errors

    matching_header, matching_rows = read_tsv(matching_path)
    candidate_header, candidate_rows = read_tsv(candidate_path)

    matching_map = {}
    candidate_map = {}

    for i, row in enumerate(matching_rows, start=2):
        s1_id, raw_ids = row
        if s1_id in matching_map:
            errors.append(f"matching_results.tsv: duplicate Source 1 ID {s1_id} (line {i})")
        matching_map[s1_id] = parse_ids(raw_ids)

    for i, row in enumerate(candidate_rows, start=2):
        s1_id, raw_ids = row
        if s1_id in candidate_map:
            errors.append(f"candidate_pairs.tsv: duplicate Source 1 ID {s1_id} (line {i})")
        candidate_map[s1_id] = parse_ids(raw_ids)

    expected_s1 = set(test_s1)
    if set(matching_map) != expected_s1:
        errors.append("matching_results.tsv must contain exactly one row for every test Source 1 entity")
    if set(candidate_map) != expected_s1:
        errors.append("candidate_pairs.tsv must contain exactly one row for every test Source 1 entity")

    for s1_id, ids in matching_map.items():
        if len(ids) != len(set(ids)):
            errors.append(f"matching_results.tsv: duplicate candidate IDs for {s1_id}")
        invalid = [x for x in ids if x not in valid_target_ids]
        if invalid:
            errors.append(f"matching_results.tsv: invalid target IDs for {s1_id}: {invalid[:10]}")
        candidates = set(candidate_map.get(s1_id, []))
        missing_from_candidates = [x for x in ids if x not in candidates]
        if missing_from_candidates:
            errors.append(
                f"matching_results.tsv: final matches not present in candidate_pairs.tsv for {s1_id}: {missing_from_candidates[:10]}"
            )

    for s1_id, ids in candidate_map.items():
        if len(ids) != len(set(ids)):
            errors.append(f"candidate_pairs.tsv: duplicate candidate IDs for {s1_id}")
        invalid = [x for x in ids if x not in valid_target_ids]
        if invalid:
            errors.append(f"candidate_pairs.tsv: invalid target IDs for {s1_id}: {invalid[:10]}")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matching", default="output/matching_results.tsv")
    parser.add_argument("--candidate", default="output/candidate_pairs.tsv")
    parser.add_argument("--test-dir", default="data/test")
    args = parser.parse_args()

    errors = validate(Path(args.matching), Path(args.candidate), Path(args.test_dir))
    if errors:
        print("FAIL")
        for i, error in enumerate(errors, 1):
            print(f"{i}. {error}")
        raise SystemExit(1)
    print("PASS")


if __name__ == "__main__":
    main()
