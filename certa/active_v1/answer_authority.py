"""Single answer-equivalence hash authority for CERTA Active V1."""

from __future__ import annotations

from typing import Any

from certa.derivations.answer_equivalence import inference_answer_key
from certa.reproducibility.canonical_json import canonical_json_hash


def active_answer_hash(value: Any) -> str:
    return canonical_json_hash({"equivalence_key": inference_answer_key(value).compact()})
