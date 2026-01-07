from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List


NOISE_TOKENS = {
    "price",
    "prix",
    "buy",
    "acheter",
    "cost",
    "costa",
    "sale",
}


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    cleaned = strip_accents(str(text)).lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    tokens = [token for token in cleaned.split() if token and token not in NOISE_TOKENS]
    return " ".join(tokens)


def tokenize(text: str) -> List[str]:
    return normalize_text(text).split()


def join_fields(values: Iterable[object]) -> str:
    parts: List[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value)
        if not text or text == "nan":
            continue
        parts.append(text)
    return " ".join(parts)
