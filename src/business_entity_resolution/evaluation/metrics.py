"""Challenge-aligned entity-level F0.5 metrics."""

from __future__ import annotations

import pandas as pd

BETA = 0.5


def fbeta(precision: float, recall: float, beta: float = BETA) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    b2 = beta * beta
    denominator = b2 * precision + recall
    return (1.0 + b2) * precision * recall / denominator if denominator else 0.0


def entity_fbeta(predicted: set[str], truth: set[str], beta: float = BETA) -> float:
    if not truth:
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0
    tp = len(predicted & truth)
    precision = tp / len(predicted)
    recall = tp / len(truth)
    return fbeta(precision, recall, beta)


def macro_entity_fbeta(
    predictions: dict[str, set[str]],
    ground_truth: dict[str, set[str]],
    beta: float = BETA,
) -> float:
    scores = []
    for s1_id, truth in ground_truth.items():
        scores.append(entity_fbeta(predictions.get(s1_id, set()), truth, beta))
    return float(sum(scores) / len(scores)) if scores else 0.0


def candidate_recall(candidates: pd.DataFrame, ground_truth: dict[str, set[str]]) -> float:
    if not ground_truth:
        return 0.0
    candidate_map = {
        s1_id: set(group["candidate_entity_id"])
        for s1_id, group in candidates.groupby("source1_entity_id")
    }
    total_true = 0
    covered = 0
    for s1_id, truth in ground_truth.items():
        total_true += len(truth)
        covered += len(truth & candidate_map.get(s1_id, set()))
    return covered / total_true if total_true else 1.0
