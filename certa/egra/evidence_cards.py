"""Value-free cards over the existing canonical structural-group catalog."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from certa.planner.schema_view import build_canonical_structural_group_view
from certa.reproducibility.canonical_json import canonical_json_hash


CARD_SCHEMA_VERSION = "certa_egra_structural_card_v1"
ACTIVE_CARD_KINDS = ("ROW_PATH", "COLUMN_PATH", "REGION_GROUP")
EXPANSION_CARD_KINDS = ("HEADER_SUBTREE",)


def build_structural_card_schema() -> Dict[str, Any]:
    """Return the exact immutable Pack schema for one structural card."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "additionalProperties": False,
        "properties": {
            "answer_values_exposed": {"const": False},
            "axis": {"enum": ["row", "column", "mixed"]},
            "card_id": {"minLength": 1, "type": "string"},
            "catalog_sha256": {
                "pattern": "^[0-9a-f]{64}$",
                "type": "string",
            },
            "header_node_ids": {
                "items": {"type": "string"},
                "minItems": 1,
                "type": "array",
                "uniqueItems": True,
            },
            "human_readable_text": {"minLength": 1, "type": "string"},
            "member_coordinates": {
                "items": {
                    "maxItems": 2,
                    "minItems": 2,
                    "prefixItems": [
                        {"type": "integer"},
                        {"type": "integer"},
                    ],
                    "type": "array",
                },
                "type": "array",
            },
            "neighbor_card_ids": {
                "items": {"type": "string"},
                "type": "array",
                "uniqueItems": True,
            },
            "provenance_complete": {"type": "boolean"},
            "schema_version": {"const": CARD_SCHEMA_VERSION},
            "unit_kind": {
                "enum": [
                    "ROW_PATH",
                    "COLUMN_PATH",
                    "REGION_GROUP",
                    "HEADER_SUBTREE",
                ]
            },
        },
        "required": [
            "schema_version",
            "card_id",
            "catalog_sha256",
            "unit_kind",
            "axis",
            "human_readable_text",
            "header_node_ids",
            "member_coordinates",
            "neighbor_card_ids",
            "answer_values_exposed",
            "provenance_complete",
        ],
        "title": "CERTA-EGRA structural evidence card",
        "type": "object",
    }


_CATALOG_SCHEMA_VERSION = "certa_canonical_structural_group_catalog_r2_v1"
_CATALOG_BUILDER_VERSION = "certa_canonical_structural_groups_r2_v1"


def _validated_groups(catalog: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if catalog.get("schema_version") != _CATALOG_SCHEMA_VERSION:
        raise ValueError("unknown_catalog_schema_version")
    if catalog.get("builder_version") != _CATALOG_BUILDER_VERSION:
        raise ValueError("unknown_catalog_builder_version")
    catalog_sha256 = str(catalog.get("catalog_sha256") or "")
    groups = list(catalog.get("all_groups") or [])
    unhashed_groups = [
        {key: value for key, value in group.items() if key != "catalog_sha256"}
        for group in groups
    ]
    if canonical_json_hash(unhashed_groups) != catalog_sha256:
        raise ValueError("catalog_hash_mismatch")
    by_id = dict(catalog.get("group_by_id") or {})
    if set(by_id) != {str(group.get("group_id") or "") for group in groups}:
        raise ValueError("catalog_group_index_mismatch")
    provenance = dict(catalog.get("verified_header_provenance") or {})
    for group in groups:
        group_id = str(group.get("group_id") or "")
        if by_id.get(group_id) != group:
            raise ValueError(f"catalog_group_index_mismatch:{group_id}")
        if group.get("catalog_sha256") != catalog_sha256:
            raise ValueError(f"group_catalog_hash_mismatch:{group_id}")
        header_ids = [str(item) for item in group.get("ordered_header_node_ids") or []]
        if not header_ids or len(header_ids) != len(set(header_ids)):
            raise ValueError(f"invalid_header_node_ids:{group_id}")
        expected_provenance = [provenance.get(node_id) for node_id in header_ids]
        if any(item is None for item in expected_provenance):
            raise ValueError(f"missing_header_provenance:{group_id}")
        if group.get("provenance_records") != expected_provenance:
            raise ValueError(f"group_provenance_mismatch:{group_id}")
        description = " / ".join(str(item.get("display_text") or "") for item in expected_provenance)
        if group.get("display_description") != description:
            raise ValueError(f"group_description_mismatch:{group_id}")
        if not description:
            raise ValueError(f"empty_display_description:{group_id}")
        descriptor = dict(group.get("member_descriptor") or {})
        coordinates = list(descriptor.get("member_coordinates") or [])
        bindings = list(descriptor.get("coordinate_bindings") or [])
        if [item.get("coordinate") for item in bindings] != coordinates:
            raise ValueError(f"coordinate_binding_mismatch:{group_id}")
        binding_union = sorted({
            str(binding_id)
            for item in bindings
            for binding_id in (item.get("binding_ids") or [])
        })
        if binding_union != list(group.get("grounded_binding_ids") or []):
            raise ValueError(f"grounded_binding_mismatch:{group_id}")
    return groups


def _group_neighbors(
    group: Mapping[str, Any],
    region_edges: Mapping[str, tuple[str, str]],
    subtree_roots: Mapping[str, str],
    active_headers: Mapping[str, set[str]],
) -> list[str]:
    kind = str(group["group_kind"])
    group_id = str(group["group_id"])
    headers = {str(item) for item in group["ordered_header_node_ids"]}
    neighbors: set[str] = set()
    if group_id in region_edges:
        neighbors.update(region_edges[group_id])
    for region_id, path_ids in region_edges.items():
        if group_id in path_ids:
            neighbors.add(region_id)
    if kind == "HEADER_SUBTREE":
        root_id = subtree_roots[group_id]
        for active_group_id, header_ids in active_headers.items():
            if root_id in header_ids:
                neighbors.add(active_group_id)
    else:
        for subtree_id, root_id in subtree_roots.items():
            if root_id in headers:
                neighbors.add(subtree_id)
    return sorted(neighbors)


def build_structural_evidence_cards(
    catalog: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    """Wrap canonical groups without deriving a second representation."""
    catalog_sha256 = str(catalog["catalog_sha256"])
    groups = _validated_groups(catalog)
    provenance = dict(catalog.get("verified_header_provenance") or {})
    compact_view = build_canonical_structural_group_view(catalog)
    region_edges = {
        str(region_id): (str(row_id), str(column_id))
        for region_id, row_id, column_id in compact_view["groups"]["X"]
    }
    subtree_roots: dict[str, str] = {}
    active_headers: dict[str, set[str]] = {}
    for group in groups:
        if group["group_kind"] == "HEADER_SUBTREE":
            parent_scope = str(group.get("parent_scope_id") or "")
            if not parent_scope.startswith("HEADER_ROOT:"):
                raise ValueError(f"invalid_subtree_parent_scope:{group['group_id']}")
            subtree_roots[str(group["group_id"])] = parent_scope.split(":", 1)[1]
        else:
            active_headers[str(group["group_id"])] = {
                str(item) for item in group["ordered_header_node_ids"]
            }
    cards = []
    for group in groups:
        header_ids = [str(item) for item in group["ordered_header_node_ids"]]
        group_id = str(group["group_id"])
        cards.append({
            "schema_version": CARD_SCHEMA_VERSION,
            "card_id": group_id,
            "catalog_sha256": catalog_sha256,
            "unit_kind": str(group["group_kind"]),
            "axis": str(group["axis"]),
            "human_readable_text": str(group["display_description"]),
            "header_node_ids": header_ids,
            "member_coordinates": [
                [int(coordinate[0]), int(coordinate[1])]
                for coordinate in group["member_descriptor"]["member_coordinates"]
            ],
            "neighbor_card_ids": _group_neighbors(
                group,
                region_edges,
                subtree_roots,
                active_headers,
            ),
            "answer_values_exposed": False,
            "provenance_complete": bool(header_ids) and all(
                node_id in provenance for node_id in header_ids
            ),
        })
    return sorted(cards, key=lambda card: card["card_id"])
