"""Generate test candidates and final matching_results.tsv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from business_entity_resolution.blocking import (
    BlockingConfig,
    candidates_to_submission,
    generate_candidates,
    prepare_source,
)
from business_entity_resolution.features import PairFeatureBuilder
from business_entity_resolution.models.matcher import PairMatcher
from business_entity_resolution.utils.io import read_source, write_tsv


def predictions_to_submission(
    scored_pairs: pd.DataFrame,
    source1_ids: list[str],
    threshold: float,
) -> pd.DataFrame:
    if scored_pairs.empty:
        return pd.DataFrame(
            {
                "source1_entity_id": source1_ids,
                "matched_entity_ids": [""] * len(source1_ids),
            }
        )

    accepted = scored_pairs[scored_pairs["probability"] >= threshold].copy()
    grouped = accepted.groupby("source1_entity_id")["candidate_entity_id"].apply(
        lambda s: ",".join(sorted(set(s)))
    )

    rows = []
    for s1_id in source1_ids:
        rows.append(
            {
                "source1_entity_id": s1_id,
                "matched_entity_ids": grouped.get(s1_id, ""),
            }
        )
    return pd.DataFrame(rows)


def run_prediction(
    data_dir: str,
    artifacts_dir: str,
    output_dir: str,
) -> tuple[Path, Path]:
    data_dir = Path(data_dir)
    artifacts_dir = Path(artifacts_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (artifacts_dir / "metadata.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    threshold = float(metadata["threshold"])
    blocking_config = BlockingConfig(**metadata.get("blocking_config", {}))

    s1 = prepare_source(read_source(data_dir / "test_source1.tsv"), "S1")
    s2 = prepare_source(read_source(data_dir / "test_source2.tsv"), "S2")
    s3 = prepare_source(read_source(data_dir / "test_source3.tsv"), "S3")
    targets = pd.concat([s2, s3], ignore_index=True)

    print("Generating test candidates...")
    candidates = generate_candidates(s1, s2, s3, blocking_config)
    candidate_submission = candidates_to_submission(candidates, s1["entity_id"].tolist())

    if candidates.empty:
        scored = candidates.copy()
    else:
        feature_builder = PairFeatureBuilder(s1, targets)
        X = feature_builder.transform(candidates)
        matcher = PairMatcher.load(artifacts_dir / "matcher.joblib")
        scored = candidates.copy()
        scored["probability"] = matcher.predict_proba(X)

    matching_submission = predictions_to_submission(
        scored,
        s1["entity_id"].tolist(),
        threshold,
    )

    matching_path = output_dir / "matching_results.tsv"
    candidate_path = output_dir / "candidate_pairs.tsv"
    write_tsv(matching_submission, matching_path)
    write_tsv(candidate_submission, candidate_path)

    print(f"Candidates: {len(candidates):,}")
    print(f"Threshold: {threshold:.2f}")
    print(f"Matches predicted: {(matching_submission['matched_entity_ids'] != '').sum():,} / {len(matching_submission):,}")
    print(f"Wrote {matching_path}")
    print(f"Wrote {candidate_path}")
    return matching_path, candidate_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/test")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()
    run_prediction(**vars(args))


if __name__ == "__main__":
    main()
