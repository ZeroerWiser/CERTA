"""Deterministic transport-only projection for the frozen EGRA role schema."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any, Dict, Mapping


TRANSPORT_REMOVED_PATHS = (
    "/properties/projection_candidates/uniqueItems",
    "/properties/signature_candidates/uniqueItems",
    "/allOf",
)

_UNIQUE_ITEM_PROPERTIES = (
    "projection_candidates",
    "signature_candidates",
)


def build_query_role_transport_schema(
    semantic_schema: Mapping[str, Any],
) -> Dict[str, Any]:
    """Remove only the three frozen backend-incompatible schema paths."""
    transport_schema = deepcopy(dict(semantic_schema))
    properties = transport_schema.get("properties")
    if not isinstance(properties, dict):
        raise ValueError("query_role_semantic_schema_properties_missing")
    for property_name in _UNIQUE_ITEM_PROPERTIES:
        candidate = properties.get(property_name)
        if not isinstance(candidate, dict) or candidate.get("uniqueItems") is not True:
            raise ValueError(
                f"query_role_semantic_unique_items_missing:{property_name}"
            )
        del candidate["uniqueItems"]
    if "allOf" not in transport_schema:
        raise ValueError("query_role_semantic_allof_missing")
    del transport_schema["allOf"]
    return transport_schema


def transport_adapter_sha256() -> str:
    """Bind audits to the exact deterministic adapter source."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
