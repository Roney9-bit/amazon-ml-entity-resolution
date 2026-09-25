"""Multi-channel blocking for business entity resolution.

Candidate generation is the UNION of name-based and address-based channels,
not an intersection. An exact/near address can therefore rescue a match when
business names are unrelated, while a good name can rescue a row with a
missing or damaged address.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from business_entity_resolution.preprocessing import normalize_address, normalize_name


@dataclass(frozen=True)
class BlockingConfig:
    name_top_k: int = 20
    address_top_k: int = 30
    combined_top_k: int = 10
    tfidf_batch_size: int = 4096
    tfidf_n_jobs: int = -1
    name_token_max_df: int = 100
    address_token_max_df: int = 150
    max_candidates_per_token: int = 100
    max_candidates_per_number: int = 150
    use_exact_country: bool = False
    use_exact_name: bool = True
    use_exact_address: bool = True
    use_exact_name_address: bool = True
    use_name_token_blocking: bool = True
    use_address_token_blocking: bool = True
    use_address_number_blocking: bool = True
    use_name_tfidf: bool = True
    use_address_tfidf: bool = True
    use_combined_tfidf: bool = True


def prepare_source(df: pd.DataFrame, source: str) -> pd.DataFrame:
    required = {"entity_id", "business_name", "business_address", "country"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    out = df.copy()
    out["source"] = source

    # Prefer the team's cleaned columns when supplied. Fall back to our
    # deterministic normalization for raw files / test files that lack them.
    if "clean_name" in out.columns:
        out["norm_name"] = out["clean_name"].fillna("").astype(str).str.strip()
    else:
        out["norm_name"] = out["business_name"].map(normalize_name)

    if "clean_address" in out.columns:
        out["norm_address"] = out["clean_address"].fillna("").astype(str).str.strip()
    else:
        out["norm_address"] = out["business_address"].map(normalize_address)

    if "address_numbers" in out.columns:
        out["address_number_tokens"] = out["address_numbers"].fillna("").astype(str).map(
            lambda x: {tok.strip() for tok in x.split(",") if tok.strip()}
        )
    else:
        out["address_number_tokens"] = out["norm_address"].str.findall(r"\d+").map(set)

    out["country_key"] = out["country"].fillna("").astype(str).str.strip().str.casefold()
    out["name_missing"] = out["norm_name"].eq("")
    out["address_missing"] = out["norm_address"].eq("")
    out["first_address_number"] = out["address_number_tokens"].map(
        lambda nums: sorted(nums)[0] if nums else ""
    )
    return out


def _ensure_reason_store(source1: pd.DataFrame) -> dict[str, dict[str, set[str]]]:
    return {eid: defaultdict(set) for eid in source1["entity_id"]}


def _add_candidates(
    store: dict[str, dict[str, set[str]]],
    source1_id: str,
    ids: Iterable[str],
    reason: str,
) -> None:
    bucket = store.setdefault(source1_id, {})
    for candidate_id in ids:
        if candidate_id and candidate_id != source1_id:
            bucket.setdefault(candidate_id, set()).add(reason)


def _exact_block(source1: pd.DataFrame, target: pd.DataFrame, field: str) -> dict[str, set[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    for row in target[["entity_id", field]].itertuples(index=False):
        key = row[1]
        if key:
            index[key].append(row[0])

    out: dict[str, set[str]] = {}
    for row in source1[["entity_id", field]].itertuples(index=False):
        key = row[1]
        if key:
            out[row[0]] = set(index.get(key, []))
    return out


def _exact_pair_block(source1: pd.DataFrame, target: pd.DataFrame) -> dict[str, set[str]]:
    index: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in target[["entity_id", "norm_name", "norm_address"]].itertuples(index=False):
        name, address = row[1], row[2]
        if name and address:
            index[(name, address)].append(row[0])

    out: dict[str, set[str]] = {}
    for row in source1[["entity_id", "norm_name", "norm_address"]].itertuples(index=False):
        name, address = row[1], row[2]
        if name and address:
            out[row[0]] = set(index.get((name, address), []))
    return out


def _token_block(
    source1: pd.DataFrame,
    target: pd.DataFrame,
    field: str,
    max_df: int,
    max_candidates_per_token: int,
) -> dict[str, set[str]]:
    token_index: dict[str, list[str]] = defaultdict(list)
    token_df: dict[str, int] = defaultdict(int)

    for row in target[["entity_id", field]].itertuples(index=False):
        entity_id = row[0]
        tokens = set(str(row[1]).split()) if row[1] else set()
        for token in tokens:
            if len(token) < 2:
                continue
            token_df[token] += 1
            token_index[token].append(entity_id)

    allowed = {token for token, df in token_df.items() if df <= max_df}
    out: dict[str, set[str]] = {}
    for row in source1[["entity_id", field]].itertuples(index=False):
        entity_id = row[0]
        bucket: set[str] = set()
        tokens = set(str(row[1]).split()) if row[1] else set()
        for token in sorted(tokens, key=lambda x: (-len(x), x)):
            if token not in allowed:
                continue
            ids = token_index[token]
            if len(ids) <= max_candidates_per_token:
                bucket.update(ids)
        out[entity_id] = bucket
    return out


def _number_block(
    source1: pd.DataFrame,
    target: pd.DataFrame,
    max_candidates_per_number: int,
) -> dict[str, set[str]]:
    """Block on any extracted address number, not just the first number."""
    index: dict[str, list[str]] = defaultdict(list)
    for row in target[["entity_id", "address_number_tokens"]].itertuples(index=False):
        entity_id, numbers = row
        for number in numbers:
            if number:
                index[number].append(entity_id)

    out: dict[str, set[str]] = {}
    for row in source1[["entity_id", "address_number_tokens"]].itertuples(index=False):
        entity_id, numbers = row
        bucket: set[str] = set()
        for number in numbers:
            ids = index.get(number, [])
            if ids and len(ids) <= max_candidates_per_number:
                bucket.update(ids)
        out[entity_id] = bucket
    return out


def _tfidf_block(
    source1: pd.DataFrame,
    target: pd.DataFrame,
    field: str,
    top_k: int,
    batch_size: int = 4096,
    n_jobs: int = -1,
) -> dict[str, set[str]]:
    valid_target = target[target[field].fillna("").astype(str).str.len() > 0].reset_index(drop=True)
    valid_source = source1[source1[field].fillna("").astype(str).str.len() > 0].reset_index(drop=True)
    out: dict[str, set[str]] = {eid: set() for eid in source1["entity_id"]}

    if valid_target.empty or valid_source.empty:
        return out

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=1,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
        max_features=200_000,
    )
    target_matrix = vectorizer.fit_transform(valid_target[field].astype(str))
    source_matrix = vectorizer.transform(valid_source[field].astype(str))

    k = min(top_k, len(valid_target))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=n_jobs)
    nn.fit(target_matrix)
    target_ids = valid_target["entity_id"].tolist()
    batch_size = max(1, int(batch_size))
    for start in range(0, source_matrix.shape[0], batch_size):
        stop = min(start + batch_size, source_matrix.shape[0])
        _, indices = nn.kneighbors(source_matrix[start:stop])
        for local_idx, neighbor_indices in enumerate(indices):
            source_id = valid_source.iloc[start + local_idx]["entity_id"]
            out[source_id].update(target_ids[int(i)] for i in neighbor_indices)
    return out


def _combined_text(df: pd.DataFrame) -> pd.Series:
    # Field markers keep name tokens distinct from address tokens.
    return "name_" + df["norm_name"].fillna("") + " addr_" + df["norm_address"].fillna("")


def _combined_tfidf_block(source1: pd.DataFrame, target: pd.DataFrame, top_k: int, batch_size: int = 4096, n_jobs: int = -1) -> dict[str, set[str]]:
    src = source1.copy()
    tgt = target.copy()
    src["combined_text"] = _combined_text(src)
    tgt["combined_text"] = _combined_text(tgt)
    return _tfidf_block(src, tgt, "combined_text", top_k, batch_size=batch_size, n_jobs=n_jobs)


def generate_pair_candidates(
    source1: pd.DataFrame,
    target: pd.DataFrame,
    config: BlockingConfig | None = None,
) -> pd.DataFrame:
    """Generate candidates from the UNION of name and address channels."""
    config = config or BlockingConfig()
    reasons = _ensure_reason_store(source1)

    if config.use_exact_name:
        for s1_id, ids in _exact_block(source1, target, "norm_name").items():
            _add_candidates(reasons, s1_id, ids, "exact_name")

    if config.use_exact_address:
        for s1_id, ids in _exact_block(source1, target, "norm_address").items():
            _add_candidates(reasons, s1_id, ids, "exact_address")

    if config.use_exact_name_address:
        for s1_id, ids in _exact_pair_block(source1, target).items():
            _add_candidates(reasons, s1_id, ids, "exact_name_address")

    if config.use_exact_country:
        for s1_id, ids in _exact_block(source1, target, "country_key").items():
            _add_candidates(reasons, s1_id, ids, "exact_country")

    if config.use_name_token_blocking:
        blocks = _token_block(
            source1, target, "norm_name", config.name_token_max_df, config.max_candidates_per_token
        )
        for s1_id, ids in blocks.items():
            _add_candidates(reasons, s1_id, ids, "name_token")

    if config.use_address_token_blocking:
        blocks = _token_block(
            source1, target, "norm_address", config.address_token_max_df, config.max_candidates_per_token
        )
        for s1_id, ids in blocks.items():
            _add_candidates(reasons, s1_id, ids, "address_token")

    if config.use_address_number_blocking:
        blocks = _number_block(source1, target, config.max_candidates_per_number)
        for s1_id, ids in blocks.items():
            _add_candidates(reasons, s1_id, ids, "address_number")

    if config.use_name_tfidf:
        for s1_id, ids in _tfidf_block(source1, target, "norm_name", config.name_top_k, config.tfidf_batch_size, config.tfidf_n_jobs).items():
            _add_candidates(reasons, s1_id, ids, "name_tfidf")

    if config.use_address_tfidf:
        for s1_id, ids in _tfidf_block(source1, target, "norm_address", config.address_top_k, config.tfidf_batch_size, config.tfidf_n_jobs).items():
            _add_candidates(reasons, s1_id, ids, "address_tfidf")

    if config.use_combined_tfidf:
        for s1_id, ids in _combined_tfidf_block(source1, target, config.combined_top_k, config.tfidf_batch_size, config.tfidf_n_jobs).items():
            _add_candidates(reasons, s1_id, ids, "combined_tfidf")

    rows: list[dict[str, object]] = []
    target_source = str(target["source"].iloc[0]) if not target.empty else ""
    for s1_id, candidate_map in reasons.items():
        for candidate_id in sorted(candidate_map):
            block_reasons = candidate_map[candidate_id]
            rows.append(
                {
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": candidate_id,
                    "candidate_source": target_source,
                    "block_reasons": ";".join(sorted(block_reasons)),
                    "name_block_hit": float(any(r.startswith("name_") or r == "exact_name_address" for r in block_reasons)),
                    "address_block_hit": float(any(r.startswith("address_") or r in {"exact_address", "exact_name_address"} for r in block_reasons)),
                    "exact_address_block": float("exact_address" in block_reasons),
                    "exact_name_block": float("exact_name" in block_reasons),
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "source1_entity_id", "candidate_entity_id", "candidate_source",
            "block_reasons", "name_block_hit", "address_block_hit",
            "exact_address_block", "exact_name_block",
        ],
    )


def generate_candidates(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    config: BlockingConfig | None = None,
) -> pd.DataFrame:
    """Generate the union of candidates against Source 2 and Source 3."""
    c2 = generate_pair_candidates(source1, source2, config)
    c3 = generate_pair_candidates(source1, source3, config)
    candidates = pd.concat([c2, c3], ignore_index=True)
    return candidates.drop_duplicates(["source1_entity_id", "candidate_entity_id"], keep="first")


def candidates_to_submission(candidates: pd.DataFrame, source1_ids: Iterable[str]) -> pd.DataFrame:
    grouped = candidates.groupby("source1_entity_id")["candidate_entity_id"].apply(
        lambda s: ",".join(sorted(set(s)))
    )
    rows = []
    for entity_id in source1_ids:
        rows.append({
            "source1_entity_id": entity_id,
            "candidate_entity_ids": grouped.get(entity_id, ""),
        })
    return pd.DataFrame(rows)
