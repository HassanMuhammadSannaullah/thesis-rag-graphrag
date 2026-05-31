"""Normalization helpers for answer comparison."""
from __future__ import annotations

import re
import string
from datetime import datetime
from typing import Iterable


_PUNCT_TABLE = str.maketrans({ch: " " for ch in string.punctuation if ch not in {".", "%", "$", "/", "-"}})
_MONTH_FORMATS = (
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%B %d %Y",
    "%b %d %Y",
    "%d %B %Y",
    "%d %b %Y",
)
_SCALE_WORDS = {
    "k": 1_000.0,
    "thousand": 1_000.0,
    "m": 1_000_000.0,
    "million": 1_000_000.0,
    "b": 1_000_000_000.0,
    "billion": 1_000_000_000.0,
}
_UNIT_ALIASES = {
    "usd": "usd",
    "$": "usd",
    "eur": "eur",
    "euro": "eur",
    "euros": "eur",
    "%": "percent",
    "percent": "percent",
    "percentage": "percent",
    "years": "year",
    "year": "year",
    "days": "day",
    "day": "day",
    "hours": "hour",
    "hour": "hour",
}


def _clean_text(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("&", " and ")
    text = text.replace("+", " and ")
    text = re.sub(r"\bus dollars\b", " usd ", text)
    text = re.sub(r"\bu\.s\.\s*dollars\b", " usd ", text)
    text = re.sub(r"[$]", " usd ", text)
    text = re.sub(r"\bper cent\b", " percent ", text)
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_answer_text(text: str) -> str:
    text = _clean_text(text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\b(and)\b", " and ", text)
    text = re.sub(r"\b(usd|eur)\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_tokens(text: str) -> list[str]:
    normalized = normalize_answer_text(text)
    return normalized.split() if normalized else []


def try_normalize_date(text: str) -> str | None:
    cleaned = re.sub(r"[,]", " ", text.strip())
    cleaned = re.sub(r"\s+", " ", cleaned)
    for fmt in _MONTH_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def extract_numbers_with_units(text: str) -> list[dict]:
    cleaned = _clean_text(text)
    results: list[dict] = []
    pattern = re.compile(
        r"(?P<num>-?\d+(?:\.\d+)?)\s*(?P<unit>[a-z%]+)?\s*(?P<scale>thousand|million|billion|k|m|b)?"
    )
    for match in pattern.finditer(cleaned):
        raw_num = match.group("num")
        if raw_num is None:
            continue
        value = float(raw_num)
        scale_token = (match.group("scale") or "").lower()
        multiplier = _SCALE_WORDS.get(scale_token, 1.0)
        unit_token = (match.group("unit") or "").lower()
        unit = _UNIT_ALIASES.get(unit_token, unit_token or None)
        if unit == "percent":
            multiplier = 0.01
        results.append(
            {
                "value": value * multiplier,
                "raw_value": value,
                "unit": unit,
                "scale": scale_token or None,
            }
        )
    return results


def canonical_text_variants(values: Iterable[str]) -> list[str]:
    variants: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value)
        variants.append(text)
        date_norm = try_normalize_date(text)
        if date_norm:
            variants.append(date_norm)
        variants.append(normalize_answer_text(text))
    seen = set()
    unique = []
    for item in variants:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique
