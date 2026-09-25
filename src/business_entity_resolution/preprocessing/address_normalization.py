"""Conservative address normalization utilities."""

from __future__ import annotations

import re
import unicodedata

_NON_ALNUM = re.compile(r"[^a-z0-9#]+")
_MULTI_SPACE = re.compile(r"\s+")

# Kept intentionally conservative. These are formatting equivalents that are
# common in both US- and India-style addresses in the challenge.
_REPLACEMENTS = {
    r"\broad\b": "rd",
    r"\broadway\b": "bway",
    r"\bstreet\b": "st",
    r"\bavenue\b": "ave",
    r"\bboulevard\b": "blvd",
    r"\bdrive\b": "dr",
    r"\blane\b": "ln",
    r"\bhighway\b": "hwy",
    r"\bparkway\b": "pkwy",
    r"\bplace\b": "pl",
    r"\bcourt\b": "ct",
    r"\bapartment\b": "apt",
    r"\bsuite\b": "ste",
    r"\bunit\b": "unit",
    r"\bfloor\b": "fl",
    r"\bbuilding\b": "bldg",
    r"\bblock\b": "blk",
}


def normalize_address(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().casefold()
    if not text or text == "nan":
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("&", " and ")
    for pattern, replacement in _REPLACEMENTS.items():
        text = re.sub(pattern, replacement, text)
    text = _NON_ALNUM.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text


def address_tokens(value: object) -> list[str]:
    normalized = normalize_address(value)
    return normalized.split() if normalized else []
