"""Train the pairwise matcher and tune a validation threshold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from business_entity_resolution.blocking import BlockingConfig, generate_candidates, prepare_source
from business_entity_resolution.evaluation import (
    build_pair_training_set,
    candidate_recall,
    choose_threshold,
    load_ground_truth,
    split_source1_ids,
)
from business_entity_resolution.features import PairFeatureBuilder
from business_entity_resolution.models.matcher import PairMatcher
from business_entity_resolution.utils.io import read_source


def _ensure_binary_training_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise RuntimeError("No training pairs were generated. Check your data and blocking settings.")
    if df["label"].nunique() < 2:
        raise RuntimeError(
            "Training pairs contain only one class. Increase candidate blocking coverage or adjust negatives_per_positive."
        )
    return df


def run_training(
    data_dir: str,
    artifacts_dir: str,
    validation_size: float = 0.2,
    negatives_per_positive: int = 5,
    random_state: int = 42,
) -> dict:
    data_dir = Path(data_dir)
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    s1_raw = read_source(data_dir / "train_source1.tsv")
    s2_raw = read_source(data_dir / "train_source2.tsv")
    s3_raw = read_source(data_dir / "train_source3.tsv")
    gt = load_ground_truth(data_dir / "train_ground_truth.tsv")

    s1 = prepare_source(s1_raw, "S1")
    s2 = prepare_source(s2_raw, "S2")
    s3 = prepare_source(s3_raw, "S3")
    targets = pd.concat([s2, s3], ignore_index=True)

    source1_ids = s1["entity_id"].tolist()
    train_ids, val_ids = split_source1_ids(source1_ids, validation_size, random_state)
    train_ids_set, val_ids_set = set(train_ids), set(val_ids)

    config = BlockingConfig()
    print("Generating candidates...")
    all_candidates = generate_candidates(s1, s2, s3, config)
    val_candidates = all_candidates[all_candidates["source1_entity_id"].isin(val_ids_set)].copy()
    val_gt = {k: gt.get(k, set()) for k in val_ids_set}
    val_candidate_recall = candidate_recall(val_candidates, val_gt)

    # Train on training entities only.
    train_pairs = build_pair_training_set(
        all_candidates,
        gt,
        train_ids_set,
        negatives_per_positive=negatives_per_positive,
        random_state=random_state,
    )
    train_pairs = _ensure_binary_training_data(train_pairs)

    train_s1 = s1[s1["entity_id"].isin(train_ids_set)].copy()
    train_feature_builder = PairFeatureBuilder(train_s1, targets)
    X_train = train_feature_builder.transform(train_pairs)
    y_train = train_pairs["label"].astype(int)

    matcher = PairMatcher()
    matcher.fit(X_train, y_train)

    # Score validation candidates.
    if val_candidates.empty:
        raise RuntimeError("Validation blocking produced zero candidates; adjust BlockingConfig.")
    val_feature_builder = PairFeatureBuilder(s1[s1["entity_id"].isin(val_ids_set)].copy(), targets)
    X_val = val_feature_builder.transform(val_candidates)
    val_scored = val_candidates.copy()
    val_scored["probability"] = matcher.predict_proba(X_val)

    threshold, val_f05 = choose_threshold(val_scored, val_gt)

    # Train final matcher on all labeled training examples.
    all_train_pairs = build_pair_training_set(
        all_candidates,
        gt,
        set(source1_ids),
        negatives_per_positive=negatives_per_positive,
        random_state=random_state,
    )
    all_train_pairs = _ensure_binary_training_data(all_train_pairs)
    final_feature_builder = PairFeatureBuilder(s1, targets)
    X_all = final_feature_builder.transform(all_train_pairs)
    y_all = all_train_pairs["label"].astype(int)

    final_matcher = PairMatcher()
    final_matcher.fit(X_all, y_all)
    final_matcher.save(artifacts_dir / "matcher.joblib")

    metadata = {
        "threshold": threshold,
        "validation_f0_5": val_f05,
        "validation_candidate_recall": val_candidate_recall,
        "validation_size": validation_size,
        "negatives_per_positive": negatives_per_positive,
        "random_state": random_state,
        "feature_names": final_matcher.feature_names,
        "blocking_config": config.__dict__,
        "train_rows": int(len(all_train_pairs)),
        "positive_rows": int((all_train_pairs["label"] == 1).sum()),
        "negative_rows": int((all_train_pairs["label"] == 0).sum()),
    }
    (artifacts_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Training pairs: {len(all_train_pairs):,}")
    print(f"Validation candidate recall: {val_candidate_recall:.4f}")
    print(f"Best validation F0.5: {val_f05:.4f}")
    print(f"Selected threshold: {threshold:.2f}")
    print(f"Saved model: {artifacts_dir / 'matcher.joblib'}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/train")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--negatives-per-positive", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    run_training(**vars(args))


if __name__ == "__main__":
    main()
