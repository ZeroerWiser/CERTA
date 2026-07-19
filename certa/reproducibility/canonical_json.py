"""Canonical JSON serialization for reproducible CERTA hashes.

This module is intentionally small and dependency-free. It canonicalizes common
Python containers before JSON rendering so packet, prompt, and API request
hashes do not depend on interpreter hash seed or incidental mapping order.
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional


def _normalize_text(value: Any) -> str:
    return unicodedata.normalize("NFC", str(value))


def _canonical_sort_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonicalize(value: Any) -> Any:
    """Return a JSON-compatible object with deterministic unordered containers."""
    if is_dataclass(value) and not isinstance(value, type):
        return canonicalize(asdict(value))
    if isinstance(value, Enum):
        return canonicalize(value.value)
    if isinstance(value, Mapping):
        items = []
        for key, item_value in value.items():
            canonical_key = _normalize_text(key)
            items.append((canonical_key, canonicalize(item_value)))
        return {key: item_value for key, item_value in sorted(items, key=lambda item: item[0])}
    if isinstance(value, (set, frozenset)):
        items = [canonicalize(item) for item in value]
        return sorted(items, key=_canonical_sort_key)
    if isinstance(value, tuple):
        return [canonicalize(item) for item in value]
    if isinstance(value, list):
        return [canonicalize(item) for item in value]
    if isinstance(value, str):
        return _normalize_text(value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "NaN"}
        if math.isinf(value):
            return {"__float__": "Infinity" if value > 0 else "-Infinity"}
        if value == 0.0:
            return 0.0
        return value
    if isinstance(value, Path):
        return _normalize_text(value)
    return _normalize_text(value)


def canonical_json(value: Any, *, pretty: bool = False) -> str:
    """Render canonical JSON with stable Unicode and separator policy."""
    payload = canonicalize(value)
    if pretty:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_json_hash(value: Any, n: Optional[int] = None) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return digest[:n] if n is not None else digest


def canonical_text_hash(text: str, n: Optional[int] = None) -> str:
    digest = hashlib.sha256(_normalize_text(text or "").encode("utf-8")).hexdigest()
    return digest[:n] if n is not None else digest
