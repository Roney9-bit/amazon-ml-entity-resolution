"""Name normalization utilities.

The goal is conservative normalization: remove formatting noise while keeping
business-identifying tokens such as LLC, Pvt, Ltd, etc. intact for the model.
"""

from __future__ import annotations

import re
import unicodedata

_APOSTROPHES = re.compile(r"[\u2018\u2019\u201A\u201B\u2032\u2035']")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_MULTI_SPACE = re.compile(r"\s+")


def normalize_name(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().casefold()
    if not text or text == "nan":
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = _APOSTROPHES.sub("", text)
    text = text.replace("&", " and ")
    text = _NON_ALNUM.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text


def name_tokens(value: object) -> list[str]:
    normalized = normalize_name(value)
    return normalized.split() if normalized else []
