import pandas as pd

from business_entity_resolution.blocking import BlockingConfig, generate_pair_candidates, prepare_source
from business_entity_resolution.features import PairFeatureBuilder


def test_address_channel_rescues_completely_different_name():
    s1 = prepare_source(pd.DataFrame([
        {"entity_id": "S1-1", "business_name": "Mirasol", "business_address": "10 Main Street, Pune", "country": "India"}
    ]), "S1")
    s2 = prepare_source(pd.DataFrame([
        {"entity_id": "S2-1", "business_name": "Team Air Pvt Ltd", "business_address": "10 Main St Pune", "country": "India"}
    ]), "S2")
    c = generate_pair_candidates(s1, s2, BlockingConfig())
    assert (c["candidate_entity_id"] == "S2-1").any()


def test_missing_address_is_not_perfect_similarity():
    s1 = prepare_source(pd.DataFrame([
        {"entity_id": "S1-1", "business_name": "Mirasol", "business_address": "", "country": "India"}
    ]), "S1")
    s2 = prepare_source(pd.DataFrame([
        {"entity_id": "S2-1", "business_name": "Mirasol", "business_address": "", "country": "India"}
    ]), "S2")
    c = generate_pair_candidates(s1, s2, BlockingConfig())
    fb = PairFeatureBuilder(s1, s2)
    X = fb.transform(c)
    assert float(X.iloc[0]["address_tfidf_cosine"]) == 0.0
    assert float(X.iloc[0]["address_jaccard"]) == 0.0
    assert float(X.iloc[0]["address_missing_any"]) == 1.0


def test_cleaned_columns_are_used():
    raw = pd.DataFrame([
        {
            "entity_id": "S1-1",
            "business_name": "Raw Name",
            "business_address": "Raw Address",
            "country": "India",
            "clean_name": "mirasol",
            "clean_address": "10 main street pune",
            "address_numbers": "10,207",
        }
    ])
    s1 = prepare_source(raw, "S1")
    assert s1.iloc[0]["norm_name"] == "mirasol"
    assert s1.iloc[0]["norm_address"] == "10 main street pune"
    assert s1.iloc[0]["address_number_tokens"] == {"10", "207"}


def test_number_block_uses_all_cleaned_address_numbers():
    s1 = prepare_source(pd.DataFrame([{
        "entity_id": "S1-1", "business_name": "A", "business_address": "x", "country": "India",
        "clean_name": "a", "clean_address": "x", "address_numbers": "10,207",
    }]), "S1")
    s2 = prepare_source(pd.DataFrame([{
        "entity_id": "S2-1", "business_name": "B", "business_address": "y", "country": "India",
        "clean_name": "b", "clean_address": "y", "address_numbers": "207,999",
    }]), "S2")
    c = generate_pair_candidates(s1, s2, BlockingConfig(
        use_exact_name=False,
        use_exact_address=False,
        use_exact_name_address=False,
        use_name_token_blocking=False,
        use_address_token_blocking=False,
        use_address_number_blocking=True,
        use_name_tfidf=False,
        use_address_tfidf=False,
        use_combined_tfidf=False,
    ))
    assert (c["candidate_entity_id"] == "S2-1").any()
