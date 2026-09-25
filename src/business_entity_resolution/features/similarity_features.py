"""Missing-aware feature engineering for business entity pairs."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer

_DIGIT_RE = re.compile(r"\d+")


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    return "" if text.lower() == "nan" else text


def token_jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def digit_tokens(value: str) -> set[str]:
    return set(_DIGIT_RE.findall(value))


def digit_jaccard(a: str, b: str) -> float:
    da, db = digit_tokens(a), digit_tokens(b)
    if not da or not db:
        return 0.0
    return len(da & db) / len(da | db)


def first_number(value: str) -> str:
    match = _DIGIT_RE.search(value)
    return match.group(0) if match else ""


def length_ratio(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


@dataclass
class _TfidfIndex:
    vectorizer: TfidfVectorizer | None = None
    vectors: dict[str, object] | None = None

    def fit(self, values: list[str]) -> "_TfidfIndex":
        unique = sorted({v for v in values if v})
        self.vectors = {}
        if not unique:
            return self
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=1,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
            max_features=200_000,
        )
        matrix = self.vectorizer.fit_transform(unique)
        for idx, value in enumerate(unique):
            self.vectors[value] = matrix[idx]
        return self

    def similarity(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        if self.vectorizer is None or self.vectors is None:
            return 0.0
        va = self.vectors.get(a)
        vb = self.vectors.get(b)
        if va is None:
            va = self.vectorizer.transform([a])[0]
            self.vectors[a] = va
        if vb is None:
            vb = self.vectorizer.transform([b])[0]
            self.vectors[b] = vb
        return float(va.multiply(vb).sum())


class PairFeatureBuilder:
    FEATURE_NAMES = [
        "name_missing_any",
        "name_both_missing",
        "address_missing_any",
        "address_both_missing",
        "name_ratio",
        "name_token_sort_ratio",
        "name_token_set_ratio",
        "name_wratio",
        "name_jaccard",
        "name_tfidf_cosine",
        "address_ratio",
        "address_token_sort_ratio",
        "address_token_set_ratio",
        "address_wratio",
        "address_jaccard",
        "address_tfidf_cosine",
        "country_match",
        "country_missing_any",
        "digit_jaccard",
        "first_number_match",
        "name_length_ratio",
        "address_length_ratio",
        "name_exact",
        "address_exact",
        "both_exact",
        "address_exact_name_weak",
        "name_exact_address_weak",
        "name_and_address_both_present",
        "name_and_address_disagree",
        "source_is_s2",
        "source_is_s3",
        "name_block_hit",
        "address_block_hit",
        "exact_address_block",
        "exact_name_block",
    ]

    def __init__(self, source1: pd.DataFrame, targets: pd.DataFrame):
        self.s1 = source1.set_index("entity_id", drop=False)
        self.targets = targets.set_index("entity_id", drop=False)
        all_names = pd.concat([self.s1["norm_name"], self.targets["norm_name"]], ignore_index=True).astype(str).tolist()
        all_addresses = pd.concat([self.s1["norm_address"], self.targets["norm_address"]], ignore_index=True).astype(str).tolist()
        self.name_tfidf = _TfidfIndex().fit(all_names)
        self.address_tfidf = _TfidfIndex().fit(all_addresses)

    def _row_features(self, row) -> list[float]:
        left = self.s1.loc[row.source1_entity_id]
        right = self.targets.loc[row.candidate_entity_id]

        n1, n2 = _safe_text(left["norm_name"]), _safe_text(right["norm_name"])
        a1, a2 = _safe_text(left["norm_address"]), _safe_text(right["norm_address"])
        c1 = _safe_text(left["country"]).strip().casefold()
        c2 = _safe_text(right["country"]).strip().casefold()

        name_missing_any = float(not n1 or not n2)
        name_both_missing = float(not n1 and not n2)
        address_missing_any = float(not a1 or not a2)
        address_both_missing = float(not a1 and not a2)

        name_ratio = fuzz.ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
        name_token_sort = fuzz.token_sort_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
        name_token_set = fuzz.token_set_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
        name_wratio = fuzz.WRatio(n1, n2) / 100.0 if n1 and n2 else 0.0

        address_ratio = fuzz.ratio(a1, a2) / 100.0 if a1 and a2 else 0.0
        address_token_sort = fuzz.token_sort_ratio(a1, a2) / 100.0 if a1 and a2 else 0.0
        address_token_set = fuzz.token_set_ratio(a1, a2) / 100.0 if a1 and a2 else 0.0
        address_wratio = fuzz.WRatio(a1, a2) / 100.0 if a1 and a2 else 0.0

        name_exact = float(bool(n1 and n2 and n1 == n2))
        address_exact = float(bool(a1 and a2 and a1 == a2))
        both_exact = name_exact * address_exact
        address_exact_name_weak = address_exact * (1.0 - name_token_set)
        name_exact_address_weak = name_exact * (1.0 - address_token_set)
        both_present = float(bool(n1 and n2 and a1 and a2))
        disagreement = float(bool(n1 and n2 and a1 and a2 and name_token_set < 0.35 and address_token_set > 0.85))

        return [
            name_missing_any,
            name_both_missing,
            address_missing_any,
            address_both_missing,
            name_ratio,
            name_token_sort,
            name_token_set,
            name_wratio,
            token_jaccard(n1, n2),
            self.name_tfidf.similarity(n1, n2),
            address_ratio,
            address_token_sort,
            address_token_set,
            address_wratio,
            token_jaccard(a1, a2),
            self.address_tfidf.similarity(a1, a2),
            float(bool(c1 and c2 and c1 == c2)),
            float(not c1 or not c2),
            digit_jaccard(a1, a2),
            float(bool(first_number(a1) and first_number(a2) and first_number(a1) == first_number(a2))),
            length_ratio(n1, n2),
            length_ratio(a1, a2),
            name_exact,
            address_exact,
            both_exact,
            address_exact_name_weak,
            name_exact_address_weak,
            both_present,
            disagreement,
            float(row.candidate_source == "S2"),
            float(row.candidate_source == "S3"),
            float(getattr(row, "name_block_hit", 0.0)),
            float(getattr(row, "address_block_hit", 0.0)),
            float(getattr(row, "exact_address_block", 0.0)),
            float(getattr(row, "exact_name_block", 0.0)),
        ]

    def transform(self, pairs: pd.DataFrame) -> pd.DataFrame:
        if pairs.empty:
            return pd.DataFrame(columns=self.FEATURE_NAMES, dtype=float)
        rows = [self._row_features(row) for row in pairs.itertuples(index=False)]
        return pd.DataFrame(rows, columns=self.FEATURE_NAMES, index=pairs.index)

    def transform_with_ids(self, pairs: pd.DataFrame) -> pd.DataFrame:
        features = self.transform(pairs)
        return pd.concat([pairs.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
