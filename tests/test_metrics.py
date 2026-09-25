from business_entity_resolution.evaluation.metrics import entity_fbeta


def test_empty_singleton_is_perfect():
    assert entity_fbeta(set(), set()) == 1.0


def test_false_positive_on_singleton_is_zero():
    assert entity_fbeta({"S2-1"}, set()) == 0.0


def test_exact_match_is_one():
    assert entity_fbeta({"S2-1", "S3-2"}, {"S2-1", "S3-2"}) == 1.0
