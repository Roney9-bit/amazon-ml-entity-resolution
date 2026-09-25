#!/usr/bin/env python3
"""Fast blocking diagnostic with the same candidate-generation channels.

The matching approach is unchanged. This script avoids constructing one giant
candidate DataFrame and repeatedly filtering it. Instead, each blocking channel
returns its ID map once, and recall is accumulated directly against ground truth.
TF-IDF queries are batched inside the blocking implementation to reduce RAM use.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

from business_entity_resolution.blocking import BlockingConfig, prepare_source
from business_entity_resolution.blocking.candidate_generation import (
    _token_block,
    _exact_block,
    _exact_pair_block,
    _number_block,
    _tfidf_block,
    _combined_tfidf_block,
)
from business_entity_resolution.evaluation import load_ground_truth
from business_entity_resolution.utils.io import read_source


REASONS = [
    "exact_name",
    "exact_address",
    "exact_name_address",
    "name_token",
    "address_token",
    "address_number",
    "name_tfidf",
    "address_tfidf",
    "combined_tfidf",
]


def update_stats(
    reason_candidates: dict[str, dict[str, set[str]]],
    union_candidates: dict[str, set[str]],
    truth: dict[str, set[str]],
    reason: str,
    blocks: dict[str, set[str]],
) -> None:
    for s1_id, ids in blocks.items():
        union_candidates[s1_id].update(ids)
        if ids:
            reason_candidates[reason][s1_id].update(ids)


def evaluate(
    union_candidates: dict[str, set[str]],
    reason_candidates: dict[str, dict[str, set[str]]],
    truth: dict[str, set[str]],
) -> None:
    total_true = sum(len(x) for x in truth.values())
    covered = sum(len(truth[s1_id] & union_candidates.get(s1_id, set())) for s1_id in truth)
    print(f"Total true matches: {total_true:,}")
    print(f"Overall candidate recall: {covered / total_true:.4f}" if total_true else "Overall candidate recall: 1.0000")

    for reason in REASONS:
        hits = sum(
            len(truth[s1_id] & reason_candidates[reason].get(s1_id, set()))
            for s1_id in truth
        )
        print(f"{reason:24s}: {hits / total_true:.4f}" if total_true else f"{reason:24s}: 1.0000")

    name_hits: dict[str, set[str]] = defaultdict(set)
    address_hits: dict[str, set[str]] = defaultdict(set)
    name_reasons = {"exact_name", "exact_name_address", "name_token", "name_tfidf", "combined_tfidf"}
    address_reasons = {"exact_address", "exact_name_address", "address_token", "address_number", "address_tfidf", "combined_tfidf"}

    for reason in name_reasons:
        for s1_id, ids in reason_candidates[reason].items():
            name_hits[s1_id].update(ids)
    for reason in address_reasons:
        for s1_id, ids in reason_candidates[reason].items():
            address_hits[s1_id].update(ids)

    address_only_true = sum(
        len((truth[s1_id] & address_hits.get(s1_id, set())) - name_hits.get(s1_id, set()))
        for s1_id in truth
    )
    name_only_true = sum(
        len((truth[s1_id] & name_hits.get(s1_id, set())) - address_hits.get(s1_id, set()))
        for s1_id in truth
    )
    print(f"\nTrue matches captured by address channels but no name channel: {address_only_true:,}")
    print(f"True matches captured by name channels but no address channel: {name_only_true:,}")

    examples = []
    for s1_id in truth:
        ids = (truth[s1_id] & address_hits.get(s1_id, set())) - name_hits.get(s1_id, set())
        for candidate_id in sorted(ids):
            examples.append((s1_id, candidate_id))
            if len(examples) >= 20:
                break
        if len(examples) >= 20:
            break

    if examples:
        print("\nFirst address-rescued true-match examples:")
        for s1_id, candidate_id in examples:
            print(f"  {s1_id} <-> {candidate_id}")

    total_candidates = sum(len(v) for v in union_candidates.values())
    covered_s1 = sum(bool(truth[s1_id] & union_candidates.get(s1_id, set())) for s1_id in truth)
    print(f"\nTotal unique candidates: {total_candidates:,}")
    print(f"S1 entities with >=1 true match captured: {covered_s1:,} / {len(truth):,}")
    print(f"Average candidates per S1: {total_candidates / len(truth):,.2f}" if truth else "Average candidates per S1: 0")


def process_pair(source1: pd.DataFrame, target: pd.DataFrame, config: BlockingConfig, truth: dict[str, set[str]], prefix: str):
    union_candidates: dict[str, set[str]] = defaultdict(set)
    reason_candidates: dict[str, dict[str, set[str]]] = {r: defaultdict(set) for r in REASONS}

    channels = []
    if config.use_exact_name:
        channels.append(("exact_name", lambda: _exact_block(source1, target, "norm_name")))
    if config.use_exact_address:
        channels.append(("exact_address", lambda: _exact_block(source1, target, "norm_address")))
    if config.use_exact_name_address:
        channels.append(("exact_name_address", lambda: _exact_pair_block(source1, target)))
    if config.use_name_token_blocking:
        channels.append(("name_token", lambda: _token_block(source1, target, "norm_name", config.name_token_max_df, config.max_candidates_per_token)))
    if config.use_address_token_blocking:
        channels.append(("address_token", lambda: _token_block(source1, target, "norm_address", config.address_token_max_df, config.max_candidates_per_token)))
    if config.use_address_number_blocking:
        channels.append(("address_number", lambda: _number_block(source1, target, config.max_candidates_per_number)))
    if config.use_name_tfidf:
        channels.append(("name_tfidf", lambda: _tfidf_block(source1, target, "norm_name", config.name_top_k, config.tfidf_batch_size, config.tfidf_n_jobs)))
    if config.use_address_tfidf:
        channels.append(("address_tfidf", lambda: _tfidf_block(source1, target, "norm_address", config.address_top_k, config.tfidf_batch_size, config.tfidf_n_jobs)))
    if config.use_combined_tfidf:
        channels.append(("combined_tfidf", lambda: _combined_tfidf_block(source1, target, config.combined_top_k, config.tfidf_batch_size, config.tfidf_n_jobs)))

    for reason, fn in channels:
        start = time.perf_counter()
        blocks = fn()
        update_stats(reason_candidates, union_candidates, truth, reason, blocks)
        elapsed = time.perf_counter() - start
        count = sum(len(v) for v in blocks.values())
        print(f"[{prefix}] {reason:22s} {elapsed:8.1f}s | raw candidates: {count:,}")

    return union_candidates, reason_candidates


def merge_stats(dst_union, dst_reasons, src_union, src_reasons):
    for s1_id, ids in src_union.items():
        dst_union[s1_id].update(ids)
    for reason in REASONS:
        for s1_id, ids in src_reasons[reason].items():
            dst_reasons[reason][s1_id].update(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast blocking diagnostic")
    parser.add_argument("--data-dir", default="data/train")
    parser.add_argument("--tfidf-batch-size", type=int, default=4096)
    parser.add_argument("--tfidf-n-jobs", type=int, default=-1)
    args = parser.parse_args()

    d = Path(args.data_dir)
    s1 = prepare_source(read_source(d / "train_source1.tsv"), "S1")
    s2 = prepare_source(read_source(d / "train_source2.tsv"), "S2")
    s3 = prepare_source(read_source(d / "train_source3.tsv"), "S3")
    gt = load_ground_truth(d / "train_ground_truth.tsv")

    config = BlockingConfig(
        tfidf_batch_size=args.tfidf_batch_size,
        tfidf_n_jobs=args.tfidf_n_jobs,
    )

    print(f"S1: {len(s1):,} | S2: {len(s2):,} | S3: {len(s3):,}")
    print(f"TF-IDF batch size: {config.tfidf_batch_size:,} | jobs: {config.tfidf_n_jobs}")
    print("Running every existing blocking channel; no channel is being removed.\n")

    total_union: dict[str, set[str]] = defaultdict(set)
    total_reasons: dict[str, dict[str, set[str]]] = {r: defaultdict(set) for r in REASONS}

    u2, r2 = process_pair(s1, s2, config, gt, "S2")
    merge_stats(total_union, total_reasons, u2, r2)
    u3, r3 = process_pair(s1, s3, config, gt, "S3")
    merge_stats(total_union, total_reasons, u3, r3)

    print("\n=== FINAL DIAGNOSTIC ===")
    evaluate(total_union, total_reasons, gt)


if __name__ == "__main__":
    main()
