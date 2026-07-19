"""Inference-time answer equivalence for Round 6 derivation audits."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, List, Optional

from eval_utils import normalize_text


_NUMERIC_RE = re.compile(
    r"^\s*[$£€]?\s*([+-]?(?:\d+(?:,\d{3})+|\d+)(?:\.\d+)?)\s*([%A-Za-z/]+)?\s*$"
)


@dataclass(frozen=True)
class InferenceAnswerKey:
    category: str
    key: str
    surface: str
    unit: str = ""

    def compact(self) -> str:
        parts = [self.category, self.key]
        if self.unit:
            parts.append(self.unit)
        return ":".join(parts)


def _decimal_key(text: str) -> Optional[tuple[str, str]]:
    match = _NUMERIC_RE.match(text)
    if not match:
        return None
    raw_number = match.group(1).replace(",", "")
    unit = (match.group(2) or "").strip().lower()
    try:
        value = Decimal(raw_number)
    except InvalidOperation:
        return None
    if value == 0:
        canonical = "0"
    else:
        canonical = format(value.normalize(), "f")
        if "." in canonical:
            canonical = canonical.rstrip("0").rstrip(".")
    return canonical, unit


def _text_key(value: Any) -> str:
    return normalize_text(value)


def inference_answer_key(value: Any) -> InferenceAnswerKey:
    surface = "" if value is None else str(value).strip()
    normalized = _text_key(surface)
    if not surface:
        return InferenceAnswerKey("UNKNOWN", "", surface)
    if normalized in {"true", "false"}:
        return InferenceAnswerKey("BOOLEAN_EXACT", normalized, surface)
    if "|" in surface:
        parts: List[str] = []
        for part in surface.split("|"):
            key = inference_answer_key(part)
            if key.category != "UNKNOWN":
                parts.append(key.compact())
        if parts:
            return InferenceAnswerKey("SET_EXACT_NORMALIZED", "|".join(sorted(parts)), surface)
    numeric = _decimal_key(surface)
    if numeric is not None:
        number, unit = numeric
        if unit:
            return InferenceAnswerKey("NUMERIC_UNIT_AWARE", number, surface, unit=unit)
        return InferenceAnswerKey("NUMERIC_EXACT_CANONICAL", number, surface)
    return InferenceAnswerKey("TEXT_EXACT_NORMALIZED", normalized, surface)


def inference_answers_equivalent(left: Any, right: Any) -> bool:
    return inference_answer_key(left).compact() == inference_answer_key(right).compact()
