"""Planner information-boundary views for CERTA."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence

from graph_builder import EdgeType, HCEG, NodeType

from certa.operations.contracts import (
    OPERATION_SIGNATURES,
    operation_signature_telemetry,
)


PLANNER_VIEW_VERSION = "certa_planner_boundary_view_v1"
CANONICAL_GROUP_BUILDER_VERSION = "certa_canonical_structural_groups_r2_v1"
CANONICAL_GROUP_KINDS = (
    "ROW_PATH",
    "COLUMN_PATH",
    "HEADER_SUBTREE",
    "REGION_GROUP",
)


class CERAPlannerBoundary(str, Enum):
    PROPOSAL_BLIND_SCHEMA_ONLY = "proposal_blind_schema_only"
    PROPOSAL_BLIND_VALUE_AWARE = "proposal_blind_value_aware"
    PROPOSAL_AWARE_DIAGNOSTIC = "proposal_aware_diagnostic"


_BOUNDARY_TELEMETRY = {
    CERAPlannerBoundary.PROPOSAL_BLIND_SCHEMA_ONLY: {
        "planner_boundary_condition": "proposal_blind_schema_only_v1",
        "proposal_visible_to_planner": False,
        "table_values_visible_to_planner": False,
        "boundary_ablation_arm": "A",
    },
    CERAPlannerBoundary.PROPOSAL_BLIND_VALUE_AWARE: {
        "planner_boundary_condition": "proposal_blind_value_aware_v1",
        "proposal_visible_to_planner": False,
        "table_values_visible_to_planner": True,
        "boundary_ablation_arm": "B",
    },
    CERAPlannerBoundary.PROPOSAL_AWARE_DIAGNOSTIC: {
        "planner_boundary_condition": "proposal_aware_value_aware_diagnostic_v1",
        "proposal_visible_to_planner": True,
        "table_values_visible_to_planner": True,
        "boundary_ablation_arm": "C",
    },
}


def coerce_planner_boundary(value: Any) -> CERAPlannerBoundary:
    if isinstance(value, CERAPlannerBoundary):
        return value
    try:
        return CERAPlannerBoundary(str(value or CERAPlannerBoundary.PROPOSAL_BLIND_SCHEMA_ONLY.value))
    except ValueError as exc:
        raise ValueError(f"unknown cera planner boundary: {value}") from exc


def planner_boundary_telemetry(value: Any) -> Dict[str, Any]:
    return dict(_BOUNDARY_TELEMETRY[coerce_planner_boundary(value)])


def validate_diagnostic_boundary_runtime(
    value: Any,
    *,
    cera_stage: str,
    cera_shadow_only: bool,
    cera_commit_approved_repair: bool,
) -> None:
    boundary = coerce_planner_boundary(value)
    if boundary != CERAPlannerBoundary.PROPOSAL_AWARE_DIAGNOSTIC:
        return
    if (
        str(cera_stage or "").upper() != "E71"
        or not bool(cera_shadow_only)
        or bool(cera_commit_approved_repair)
    ):
        raise ValueError(
            "proposal_aware_diagnostic requires E71, shadow-only runtime, "
            "and disabled CERA commit approval"
        )


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value or ""))


def _table_shape(table_json: Optional[Mapping[str, Any]]) -> Dict[str, int]:
    texts = (table_json or {}).get("texts") or []
    rows = len(texts) if isinstance(texts, list) else 0
    cols = max((len(row) for row in texts if isinstance(row, list)), default=0)
    top_header_rows = int((table_json or {}).get("top_header_rows_num", 1) or 1)
    left_header_cols = int((table_json or {}).get("left_header_columns_num", 1) or 1)
    return {
        "rows": rows,
        "cols": cols,
        "top_header_rows": max(0, min(top_header_rows, rows or top_header_rows)),
        "left_header_cols": max(0, min(left_header_cols, cols or left_header_cols)),
    }


def _header_axis(row: int, col: int, shape: Mapping[str, int]) -> str:
    if row >= 0 and row < int(shape.get("top_header_rows", 0)):
        return "column"
    if col >= 0 and col < int(shape.get("left_header_cols", 0)):
        return "row"
    return "header"


def _schema_nodes(graph: Optional[HCEG], shape: Mapping[str, int]) -> list[Dict[str, Any]]:
    if graph is None:
        return []
    nodes = []
    for node in sorted(graph.nodes.values(), key=lambda item: (item.row, item.col, item.node_id)):
        if node.node_type != NodeType.HEADER and not (
            node.node_type == NodeType.AGGREGATOR and node.header_level >= 0
        ):
            continue
        nodes.append(
            {
                "node_id": node.node_id,
                "node_type": node.node_type.value,
                "axis": _header_axis(node.row, node.col, shape),
                "row": node.row,
                "col": node.col,
                "header_level": node.header_level,
                "text": str(node.text or ""),
            }
        )
    return nodes


def _schema_edges(graph: Optional[HCEG]) -> list[Dict[str, str]]:
    if graph is None:
        return []
    allowed = {EdgeType.PARENT_HEADER, EdgeType.CHILD_HEADER, EdgeType.HEADER_OF}
    header_ids = {
        node_id
        for node_id, node in graph.nodes.items()
        if node.node_type == NodeType.HEADER
        or (node.node_type == NodeType.AGGREGATOR and node.header_level >= 0)
    }
    edges = []
    for edge in sorted(graph.edges, key=lambda item: (item.source, item.target, _enum_value(item.edge_type))):
        if edge.edge_type not in allowed:
            continue
        if edge.source not in header_ids or edge.target not in header_ids:
            continue
        edges.append(
            {
                "source": edge.source,
                "target": edge.target,
                "edge_type": _enum_value(edge.edge_type),
            }
        )
    return edges


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _header_sort_key(graph: HCEG, node_id: str) -> tuple[int, int, int, str]:
    node = graph.nodes[node_id]
    return (int(node.row), int(node.col), int(node.header_level), node_id)


def _tree_header_ids(
    graph: HCEG,
    table_json: Mapping[str, Any],
) -> set[str]:
    by_coordinate = {
        (int(node.row), int(node.col)): node.node_id
        for node in graph.nodes.values()
        if node.node_type in {NodeType.HEADER, NodeType.AGGREGATOR}
    }
    result: set[str] = set()

    def visit(value: Any) -> None:
        if not isinstance(value, Mapping):
            return
        coordinate = (int(value.get("row", -1)), int(value.get("column", -1)))
        if coordinate in by_coordinate:
            result.add(by_coordinate[coordinate])
        for child in value.get("children") or []:
            visit(child)

    visit(table_json.get("top_root"))
    visit(table_json.get("left_root"))
    return result


def _numeric_header_kind(text: str) -> str:
    compact = str(text or "").strip()
    if len(compact) == 4 and compact.isdigit() and 1000 <= int(compact) <= 2999:
        return "YEAR"
    date_parts = compact.replace("/", "-").split("-")
    if len(date_parts) in {2, 3} and all(part.isdigit() for part in date_parts):
        return "DATE"
    numeric = compact.replace(",", "").replace("%", "")
    try:
        float(numeric)
    except ValueError:
        return "NON_NUMERIC"
    return "NUMERIC"


def _header_provenance(
    graph: HCEG,
    table_json: Mapping[str, Any],
    shape: Mapping[str, int],
) -> dict[str, Dict[str, Any]]:
    tree_ids = _tree_header_ids(graph, table_json)
    records: dict[str, Dict[str, Any]] = {}
    for raw in _schema_nodes(graph, shape):
        node_id = str(raw["node_id"])
        text = str(raw.get("text") or "")
        numeric_kind = _numeric_header_kind(text)
        tree_verified = node_id in tree_ids
        text_verified = numeric_kind in {"NON_NUMERIC", "YEAR", "DATE"} or tree_verified
        records[node_id] = {
            "node_id": node_id,
            "axis": raw["axis"],
            "row": int(raw["row"]),
            "col": int(raw["col"]),
            "header_level": int(raw["header_level"]),
            "source": "declared_header_region",
            "tree_verified": tree_verified,
            "temporal_kind": numeric_kind if numeric_kind in {"YEAR", "DATE"} else "",
            "display_text": text if text_verified else "<masked>",
            "display_text_masked": not text_verified,
        }
    return records


def _path_maps(
    graph: HCEG,
    shape: Mapping[str, int],
    header_ids: set[str],
) -> tuple[dict[tuple[int, int], tuple[str, ...]], dict[tuple[int, int], tuple[str, ...]]]:
    rows: dict[tuple[int, int], tuple[str, ...]] = {}
    columns: dict[tuple[int, int], tuple[str, ...]] = {}
    for node in graph.nodes.values():
        if node.node_type != NodeType.CELL:
            continue
        coordinate = (int(node.row), int(node.col))
        if coordinate[0] < int(shape["top_header_rows"]) or coordinate[1] < int(shape["left_header_cols"]):
            continue
        row_ids = {
            target
            for target, edge in graph.neighbors(node.node_id)
            if target in header_ids and edge.edge_type == EdgeType.ROW_PATH
        }
        column_ids = {
            target
            for target, edge in graph.neighbors(node.node_id)
            if target in header_ids
            and edge.edge_type in {EdgeType.COL_PATH, EdgeType.VALUE_UNDER_HEADER}
        }
        if row_ids:
            rows[coordinate] = tuple(sorted(row_ids, key=lambda item: _header_sort_key(graph, item)))
        if column_ids:
            columns[coordinate] = tuple(sorted(column_ids, key=lambda item: _header_sort_key(graph, item)))
    return rows, columns


def build_canonical_structural_group_catalog(
    *,
    graph: HCEG,
    table_json: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build the question-independent canonical group catalog used by R2 audits."""
    shape = _table_shape(table_json)
    provenance = _header_provenance(graph, table_json, shape)
    row_by_coordinate, column_by_coordinate = _path_maps(
        graph,
        shape,
        set(provenance),
    )
    groups: list[Dict[str, Any]] = []

    def add_group(
        kind: str,
        axis: str,
        header_ids: tuple[str, ...],
        coordinate_bindings: Mapping[tuple[int, int], tuple[str, ...]],
        parent_scope_id: Optional[str] = None,
    ) -> None:
        coordinates = tuple(sorted(coordinate_bindings))
        if not header_ids or not coordinates:
            return
        identity = {
            "builder_version": CANONICAL_GROUP_BUILDER_VERSION,
            "group_kind": kind,
            "axis": axis,
            "ordered_header_node_ids": list(header_ids),
            "parent_scope_id": parent_scope_id,
            "member_coordinates": [list(item) for item in coordinates],
        }
        header_records = [provenance[node_id] for node_id in header_ids]
        descriptions = [record["display_text"] for record in header_records]
        row_members = sorted({row_by_coordinate[item] for item in coordinates if item in row_by_coordinate})
        column_members = sorted({column_by_coordinate[item] for item in coordinates if item in column_by_coordinate})
        groups.append({
            "_identity_sha256": _canonical_hash(identity),
            "group_kind": kind,
            "axis": axis,
            "ordered_header_node_ids": list(header_ids),
            "parent_scope_id": parent_scope_id,
            "member_descriptor": {
                "member_coordinates": [list(item) for item in coordinates],
                "coordinate_bindings": [
                    {"coordinate": list(item), "binding_ids": list(coordinate_bindings[item])}
                    for item in coordinates
                ],
                "scope_member_bindings_by_axis": {
                    "row": [list(item) for item in row_members],
                    "column": [list(item) for item in column_members],
                },
            },
            "header_levels": [int(record["header_level"]) for record in header_records],
            "provenance_records": header_records,
            "display_description": " / ".join(descriptions),
            "display_text_masked": any(record["display_text_masked"] for record in header_records),
            "grounded_binding_ids": sorted({item for value in coordinate_bindings.values() for item in value}),
        })

    for kind, axis, paths in (
        ("ROW_PATH", "row", row_by_coordinate),
        ("COLUMN_PATH", "column", column_by_coordinate),
    ):
        by_path: dict[tuple[str, ...], dict[tuple[int, int], tuple[str, ...]]] = {}
        for coordinate, path in paths.items():
            by_path.setdefault(path, {})[coordinate] = path
        for path, bindings in sorted(by_path.items()):
            add_group(kind, axis, path, bindings)

    for header_id in sorted(provenance, key=lambda item: _header_sort_key(graph, item)):
        matching_rows = {item: path for item, path in row_by_coordinate.items() if header_id in path}
        matching_columns = {item: path for item, path in column_by_coordinate.items() if header_id in path}
        selected = matching_rows or matching_columns
        axis = "row" if matching_rows else "column"
        if selected:
            ordered = tuple(sorted({value for path in selected.values() for value in path}, key=lambda item: _header_sort_key(graph, item)))
            add_group("HEADER_SUBTREE", axis, ordered, selected, f"HEADER_ROOT:{header_id}")

    region_bindings: dict[tuple[tuple[str, ...], tuple[str, ...]], dict[tuple[int, int], tuple[str, ...]]] = {}
    for coordinate in sorted(set(row_by_coordinate) & set(column_by_coordinate)):
        row_path, column_path = row_by_coordinate[coordinate], column_by_coordinate[coordinate]
        binding = tuple(dict.fromkeys((*row_path, *column_path)))
        region_bindings.setdefault((row_path, column_path), {})[coordinate] = binding
    for (row_path, column_path), bindings in sorted(region_bindings.items()):
        add_group("REGION_GROUP", "mixed", tuple(dict.fromkeys((*row_path, *column_path))), bindings)

    groups.sort(key=lambda item: (CANONICAL_GROUP_KINDS.index(item["group_kind"]), item["_identity_sha256"]))
    prefixes = {"ROW_PATH": "R", "COLUMN_PATH": "C", "HEADER_SUBTREE": "H", "REGION_GROUP": "X"}
    counters = {kind: 0 for kind in CANONICAL_GROUP_KINDS}
    for group in groups:
        kind = group["group_kind"]
        group["group_id"] = f"{prefixes[kind]}{counters[kind]}"
        counters[kind] += 1
        group.pop("_identity_sha256")
    catalog_sha256 = _canonical_hash(groups)
    for group in groups:
        group["catalog_sha256"] = catalog_sha256
    by_id = {group["group_id"]: group for group in groups}
    by_kind = {kind: [group["group_id"] for group in groups if group["group_kind"] == kind] for kind in CANONICAL_GROUP_KINDS}
    by_axis = {axis: [group["group_id"] for group in groups if group["axis"] == axis] for axis in ("row", "column", "mixed")}
    by_header = {
        node_id: [group["group_id"] for group in groups if node_id in group["ordered_header_node_ids"]]
        for node_id in sorted(provenance)
    }
    return {
        "schema_version": "certa_canonical_structural_group_catalog_r2_v1",
        "builder_version": CANONICAL_GROUP_BUILDER_VERSION,
        "catalog_sha256": catalog_sha256,
        "table_shape": shape,
        "all_groups": groups,
        "group_by_id": by_id,
        "groups_by_kind": by_kind,
        "groups_by_axis": by_axis,
        "groups_by_header_node": by_header,
        "verified_header_provenance": provenance,
        "masked_display_records": [group["group_id"] for group in groups if group["display_text_masked"]],
    }


def build_canonical_structural_group_view(catalog: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the compact value-firewalled group view exposed to the endpoint."""
    groups = list(catalog["all_groups"])
    descriptions = sorted({
        group["display_description"] for group in groups
        if group["group_kind"] != "REGION_GROUP"
    })
    description_index = {text: index for index, text in enumerate(descriptions)}
    row_ids = {
        tuple(group["ordered_header_node_ids"]): group["group_id"]
        for group in groups if group["group_kind"] == "ROW_PATH"
    }
    column_ids = {
        tuple(group["ordered_header_node_ids"]): group["group_id"]
        for group in groups if group["group_kind"] == "COLUMN_PATH"
    }

    def region_descriptor(group: Mapping[str, Any]) -> list[str]:
        members = group["member_descriptor"]["scope_member_bindings_by_axis"]
        return [row_ids[tuple(members["row"][0])], column_ids[tuple(members["column"][0])]]

    kind_codes = {"R": "ROW_PATH", "C": "COLUMN_PATH", "H": "HEADER_SUBTREE", "X": "REGION_GROUP"}
    axis_codes = {"r": "row", "c": "column", "m": "mixed"}
    code_by_axis = {value: key for key, value in axis_codes.items()}
    return {
        "schema_version": "certa_canonical_structural_group_view_r2_v1",
        "catalog_sha256": str(catalog["catalog_sha256"]),
        "table_shape": dict(catalog["table_shape"]),
        "group_encoding": {
            "R": ["group_id", "description_index"],
            "C": ["group_id", "description_index"],
            "H": ["group_id", "axis_code", "description_index"],
            "X": ["group_id", "row_path_group_id", "column_path_group_id"],
        },
        "kind_codes": kind_codes,
        "axis_codes": axis_codes,
        "descriptions": descriptions,
        "groups": {
            "R": [[group["group_id"], description_index[group["display_description"]]]
                  for group in groups if group["group_kind"] == "ROW_PATH"],
            "C": [[group["group_id"], description_index[group["display_description"]]]
                  for group in groups if group["group_kind"] == "COLUMN_PATH"],
            "H": [[group["group_id"], code_by_axis[group["axis"]],
                   description_index[group["display_description"]]]
                  for group in groups if group["group_kind"] == "HEADER_SUBTREE"],
            "X": [[group["group_id"], *region_descriptor(group)]
                  for group in groups if group["group_kind"] == "REGION_GROUP"],
        },
        "masked_group_ids": [group["group_id"] for group in groups if group["display_text_masked"]],
    }


def _query_contract_payload(query_contract: Any) -> Dict[str, Any]:
    if query_contract is None:
        return {}
    if hasattr(query_contract, "to_dict"):
        return dict(query_contract.to_dict())
    if isinstance(query_contract, Mapping):
        return dict(query_contract)
    return {}


def _query_semantics(contract: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "answer_domain": contract.get("answer_domain", "UNKNOWN"),
        "allowed_answer_domains": list(contract.get("allowed_answer_domains") or ["UNKNOWN"]),
        "allowed_projection_operators": list(contract.get("allowed_projection_operators") or ["UNKNOWN"]),
        "candidate_independent_operation_hypotheses": list(
            contract.get("candidate_independent_operation_hypotheses") or []
        ),
        "unit_or_scale_constraints": list(contract.get("unit_or_scale_constraints") or []),
    }


def _table_value_records(
    graph: Optional[HCEG],
    table_json: Optional[Mapping[str, Any]],
    shape: Mapping[str, int],
) -> list[Dict[str, Any]]:
    texts = (table_json or {}).get("texts") or []
    numeric_by_position: Dict[tuple[int, int], Any] = {}
    if graph is not None:
        for node in graph.nodes.values():
            if node.node_type == NodeType.CELL and node.numeric_value is not None:
                numeric_by_position[(int(node.row), int(node.col))] = node.numeric_value

    records: list[Dict[str, Any]] = []
    for row_index in range(int(shape.get("top_header_rows", 0)), int(shape.get("rows", 0))):
        row = texts[row_index] if row_index < len(texts) and isinstance(texts[row_index], list) else []
        for col_index in range(int(shape.get("left_header_cols", 0)), int(shape.get("cols", 0))):
            text = row[col_index] if col_index < len(row) else ""
            record: Dict[str, Any] = {
                "row": row_index,
                "col": col_index,
                "text": str(text or ""),
            }
            numeric_value = numeric_by_position.get((row_index, col_index))
            if numeric_value is not None:
                record["numeric_value"] = numeric_value
            records.append(record)
    return records


def build_proposal_blind_planner_view(
    *,
    question: str,
    graph: Optional[HCEG] = None,
    table_json: Optional[Mapping[str, Any]] = None,
    query_contract: Any = None,
    include_table_values: bool,
    legacy_query_semantics_mode: str = "active",
    allowed_signature_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Build a Planner view whose public projection cannot receive a proposal."""
    shape = _table_shape(table_json)
    contract = _query_contract_payload(query_contract)
    if legacy_query_semantics_mode not in {"active", "audit_only"}:
        raise ValueError(
            f"unknown_legacy_query_semantics_mode:{legacy_query_semantics_mode}"
        )
    signature_source = (
        OPERATION_SIGNATURES
        if allowed_signature_ids is None
        else allowed_signature_ids
    )
    signature_ids = tuple(sorted(set(str(item) for item in signature_source)))
    unknown_signatures = sorted(set(signature_ids) - set(OPERATION_SIGNATURES))
    if unknown_signatures:
        raise ValueError(f"unknown_allowed_signature_ids:{','.join(unknown_signatures)}")
    signatures = [OPERATION_SIGNATURES[item] for item in signature_ids]
    view: Dict[str, Any] = {
        "planner_view_version": PLANNER_VIEW_VERSION,
        "question": str(question or ""),
        "table_shape": shape,
        "schema_nodes": _schema_nodes(graph, shape),
        "schema_edges": _schema_edges(graph),
        "data_region": {
            "row_start": shape["top_header_rows"],
            "col_start": shape["left_header_cols"],
            "row_count": max(0, shape["rows"] - shape["top_header_rows"]),
            "col_count": max(0, shape["cols"] - shape["left_header_cols"]),
        },
        "operation_ontology": {
            "operation_families": sorted({item.operation_family for item in signatures}),
            "signature_ids": list(signature_ids),
            "signature_variants": {
                signature_id: operation_signature_telemetry(signature_id)
                for signature_id in signature_ids
            },
            "projection_operators": sorted({item.projection_operator for item in signatures}),
            "answer_domains": sorted({item.answer_domain for item in signatures}),
        },
    }
    if legacy_query_semantics_mode == "active":
        view["query_semantics"] = _query_semantics(contract)
    if include_table_values:
        view["table_values"] = _table_value_records(graph, table_json, shape)
    return view


def build_schema_only_planner_view(
    *,
    question: str,
    graph: Optional[HCEG] = None,
    table_json: Optional[Mapping[str, Any]] = None,
    query_contract: Any = None,
    legacy_query_semantics_mode: str = "active",
) -> Dict[str, Any]:
    """Build a planner input view without full data-cell values.

    The planner sees header/schema identifiers, dimensions, operation ontology,
    and query semantics. Data cell text is intentionally excluded.
    """
    return build_proposal_blind_planner_view(
        question=question,
        graph=graph,
        table_json=table_json,
        query_contract=query_contract,
        include_table_values=False,
        legacy_query_semantics_mode=legacy_query_semantics_mode,
    )


def build_proposal_aware_diagnostic_planner_view(
    *,
    question: str,
    graph: Optional[HCEG] = None,
    table_json: Optional[Mapping[str, Any]] = None,
    query_contract: Any = None,
    initial_proposal_diagnostic: Any,
) -> Dict[str, Any]:
    view = build_proposal_blind_planner_view(
        question=question,
        graph=graph,
        table_json=table_json,
        query_contract=query_contract,
        include_table_values=True,
    )
    view["initial_proposal_diagnostic"] = str(initial_proposal_diagnostic or "")
    return view
