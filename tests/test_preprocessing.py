from business_entity_resolution.preprocessing import normalize_address, normalize_name


def test_name_normalization():
    assert normalize_name("Orelee's Barbershop") == "orelees barbershop"
    assert normalize_name("B+ Retail Inc") == "b retail inc"


def test_address_normalization():
    assert normalize_address("1795 Westchester Drive, High Point, NC") == "1795 westchester dr high point nc"
