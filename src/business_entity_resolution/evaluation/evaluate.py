"""Training/validation dataset creation and threshold selection."""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from business_entity_resolution.evaluation.metrics import macro_entity_fbeta


def load_ground_truth(path: str) -> dict[str, set[str]]:
    gt = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    required = {"source1_entity_id", "matched_entity_ids"}
    missing = required.difference(gt.columns)
    if missing:
        raise ValueError(f"Ground truth missing columns: {sorted(missing)}")

    result: dict[str, set[str]] = {}
    for row in gt.itertuples(index=False):
        ids = {x.strip() for x in str(row.matched_entity_ids).split(",") if x.strip()}
        result[str(row.source1_entity_id)] = ids
    return result


def split_source1_ids(
    source1_ids: list[str],
    validation_size: float = 0.2,
    random_state: int = 42,
) -> tuple[list[str], list[str]]:
    train_ids, val_ids = train_test_split(
        source1_ids,
        test_size=validation_size,
        random_state=random_state,
        shuffle=True,
    )
    return list(train_ids), list(val_ids)


def candidate_pairs_for_ids(candidates: pd.DataFrame, ids: set[str]) -> pd.DataFrame:
    return candidates[candidates["source1_entity_id"].isin(ids)].copy()


def build_pair_training_set(
    candidates: pd.DataFrame,
    ground_truth: dict[str, set[str]],
    source1_ids: set[str],
    negatives_per_positive: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Construct positives plus hard negative candidates.

    All known positives for the selected Source 1 IDs are included even if a
    blocking rule missed them. This makes model training robust while candidate
    recall is evaluated separately.
    """
    rng = random.Random(random_state)
    rows = []
    candidate_lookup: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in candidates.itertuples(index=False):
        if row.source1_entity_id in source1_ids:
            candidate_lookup[row.source1_entity_id].append(
                (row.candidate_entity_id, row.candidate_source)
            )

    # Positive examples from ground truth.
    positive_pairs: dict[str, set[str]] = {}
    for s1_id in source1_ids:
        truth = ground_truth.get(s1_id, set())
        positive_pairs[s1_id] = set(truth)
        for candidate_id in sorted(truth):
            source = "S2" if candidate_id.startswith("S2-") else "S3"
            rows.append(
                {
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": candidate_id,
                    "candidate_source": source,
                    "label": 1,
                }
            )

    # Hard negatives from actual blocking candidates.
    for s1_id in source1_ids:
        positives = positive_pairs[s1_id]
        negatives = [x for x in candidate_lookup.get(s1_id, []) if x[0] not in positives]
        if len(negatives) > negatives_per_positive * max(1, len(positives)):
            negatives = rng.sample(negatives, negatives_per_positive * max(1, len(positives)))
        for candidate_id, source in negatives:
            rows.append(
                {
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": candidate_id,
                    "candidate_source": source,
                    "label": 0,
                }
            )

        # For true singletons, include a small set of hard negatives even though
        # there are zero positives, so the model learns to reject them.
        if not positives and negatives:
            keep = min(len(negatives), negatives_per_positive)
            selected = negatives if len(negatives) <= keep else rng.sample(negatives, keep)
            existing = {(r["candidate_entity_id"], r["candidate_source"]) for r in rows if r["source1_entity_id"] == s1_id}
            for candidate_id, source in selected:
                if (candidate_id, source) not in existing:
                    rows.append(
                        {
                            "source1_entity_id": s1_id,
                            "candidate_entity_id": candidate_id,
                            "candidate_source": source,
                            "label": 0,
                        }
                    )

    return pd.DataFrame(rows)


def choose_threshold(
    scored_pairs: pd.DataFrame,
    ground_truth: dict[str, set[str]],
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    if thresholds is None:
        thresholds = np.arange(0.50, 0.991, 0.01)

    best_threshold = 0.5
    best_score = -1.0
    grouped_candidates = scored_pairs.groupby("source1_entity_id")

    for threshold in thresholds:
        predictions: dict[str, set[str]] = {}
        for s1_id, group in grouped_candidates:
            predictions[s1_id] = set(
                group.loc[group["probability"] >= threshold, "candidate_entity_id"]
            )
        score = macro_entity_fbeta(predictions, ground_truth)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold, best_score
