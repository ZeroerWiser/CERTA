#!/usr/bin/env python3
"""Offline fixtures and freeze commands for the CERTA Active V1 adapter."""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

import jsonschema

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from graph_builder import EdgeType, GraphEdge, GraphNode, HCEG, NodeType

from certa.active_v1.planner_adapter import (
    active_signature_ids,
    close_compiled_payload,
    compile_active_planner_payload,
)
from certa.active_v1.role_contract import (
    ROLE_TUPLES,
    build_role_prompt,
    build_role_semantic_schema,
    build_role_wire_schema,
    validate_role_contract,
)
from certa.operations.contracts import OPERATION_SIGNATURES
from certa.planner.schema_view import build_proposal_blind_planner_view
from certa.planner.typed_planner import PLANNER_VERSION
from certa.reproducibility.canonical_json import canonical_json, canonical_json_hash


PACK = Path("/home/hsh/ME/Table/EMNLP2026/certa_goal_packs/CERTA_ACTIVE_V1_FINAL_METHOD_GOAL_REVISED_PACK")
DESIGN_SIGNATURE_IDS = tuple(ROLE_TUPLES)
FROZEN_LEGACY_HASHES = {
    "certa/egra/retrieval.py": "02d30f80ac2e3c0827c4eaf819a2aa0f66b50e42cd1b93681041fc1a25995552",
    "certa/repair/causal_epistemic_agent.py": "77619174cad1695cc11db73e2cf436c1b3d356a70e7b94b4d5870e5e0d9a9787",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_graph() -> HCEG:
    graph = HCEG()
    for node in (
        GraphNode("measure_numeric", NodeType.HEADER, row=0, col=1, text="Value"),
        GraphNode("measure_entity", NodeType.HEADER, row=0, col=2, text="Winner"),
        GraphNode("entity_a", NodeType.HEADER, row=1, col=0, text="A"),
        GraphNode("entity_b", NodeType.HEADER, row=2, col=0, text="B"),
        GraphNode("entity_c", NodeType.HEADER, row=3, col=0, text="C"),
        GraphNode("entity_d", NodeType.HEADER, row=4, col=0, text="D"),
        GraphNode("numeric_a", NodeType.CELL, row=1, col=1, text="4", numeric_value=4.0),
        GraphNode("numeric_b", NodeType.CELL, row=2, col=1, text="2", numeric_value=2.0),
        GraphNode("numeric_c", NodeType.CELL, row=3, col=1, text="4", numeric_value=4.0),
        GraphNode("numeric_d", NodeType.CELL, row=4, col=1, text="2", numeric_value=2.0),
        GraphNode("entity_value", NodeType.CELL, row=1, col=2, text="Alpha"),
    ):
        graph.add_node(node)
    for cell, entity, measure in (
        ("numeric_a", "entity_a", "measure_numeric"),
        ("numeric_b", "entity_b", "measure_numeric"),
        ("numeric_c", "entity_c", "measure_numeric"),
        ("numeric_d", "entity_d", "measure_numeric"),
        ("entity_value", "entity_a", "measure_entity"),
    ):
        graph.add_edge(GraphEdge(cell, entity, EdgeType.ROW_PATH))
        graph.add_edge(GraphEdge(cell, measure, EdgeType.COL_PATH))
    return graph


def _fixture_table() -> Dict[str, Any]:
    return {
        "texts": [
            ["Entity", "Value", "Winner"],
            ["A", "4", "Alpha"],
            ["B", "2", ""],
            ["C", "4", ""],
            ["D", "2", ""],
        ],
        "top_header_rows_num": 1,
        "left_header_columns_num": 1,
    }


def _fixture_plan(signature_id: str) -> Dict[str, Any]:
    signature = OPERATION_SIGNATURES[signature_id]
    plan: Dict[str, Any] = {
        "plan_id": "P0",
        "signature_id": signature_id,
        "operation_family": signature.operation_family,
        "semantic_result_role": signature.semantic_result_role,
        "projection_operator": signature.projection_operator,
        "answer_domain": signature.answer_domain,
        "role_bindings": {},
        "unresolved_semantics": [],
    }
    if signature_id.startswith("LOOKUP_VALUE_"):
        measure = "measure_numeric" if signature_id.endswith("SCALAR") else "measure_entity"
        plan["role_bindings"] = {"TARGET_ENTITY": ["entity_a"], "TARGET_MEASURE": [measure]}
    elif signature.operation_family in {"DIFF", "RATIO", "PAIR_COMPARE"}:
        plan["role_bindings"] = {
            "LEFT_OPERAND": ["entity_a", "measure_numeric"],
            "RIGHT_OPERAND": ["entity_b", "measure_numeric"],
        }
        if signature.operation_family == "PAIR_COMPARE":
            plan["comparison_polarity"] = "greater"
    else:
        members = [["entity_a"], ["entity_b"]]
        if signature_id == "ARGMAX_ENTITY_SET":
            members.append(["entity_c"])
        elif signature_id == "ARGMIN_ENTITY_SET":
            members.append(["entity_d"])
        plan["role_bindings"] = {
            "AGGREGATION_SCOPE": members,
            "TARGET_MEASURE": ["measure_numeric"],
        }
    return plan


def _payload(signature_id: str) -> Dict[str, Any]:
    signature = OPERATION_SIGNATURES[signature_id]
    return {
        "planner_version": PLANNER_VERSION,
        "query_semantics": {
            "operation_family": signature.operation_family,
            "answer_domain": signature.answer_domain,
            "projection_operator": signature.projection_operator,
        },
        "plans": [_fixture_plan(signature_id)],
        "unresolved_semantics": [],
    }


def _provisional_matrix(signature_id: str) -> Dict[str, Any]:
    return {
        "schema_version": "certa_active_v1_signature_capability_v1",
        "rows": [{
            "signature_id": signature_id,
            "registry_present": True,
            "active_compiler_fixture_pass": True,
            "closure_fixture_pass": True,
            "deterministic_executor_fixture_pass": True,
            "projection_fixture_pass": True,
            "serialization_roundtrip_fixture_pass": True,
            "constructor_active": True,
            "constructor_failure_reasons": [],
            "active": True,
        }],
    }


def _fixture_view(signature_id: str, graph: HCEG) -> Dict[str, Any]:
    signature = OPERATION_SIGNATURES[signature_id]
    return build_proposal_blind_planner_view(
        question="Capability fixture only.",
        graph=graph,
        table_json=_fixture_table(),
        query_contract={
            "answer_domain": signature.answer_domain,
            "allowed_answer_domains": [signature.answer_domain],
            "allowed_projection_operators": [signature.projection_operator],
            "candidate_independent_operation_hypotheses": [signature.operation_family],
            "unit_or_scale_constraints": [],
        },
        include_table_values=False,
        legacy_query_semantics_mode="active",
        allowed_signature_ids=(signature_id,),
    )


def _capability_row(signature_id: str) -> Dict[str, Any]:
    graph = _fixture_graph()
    signature = OPERATION_SIGNATURES[signature_id]
    matrix = _provisional_matrix(signature_id)
    view = _fixture_view(signature_id, graph)
    compilation = compile_active_planner_payload(_payload(signature_id), view, matrix)
    compiler_pass = compilation.ok
    closure = None
    closure_again = None
    if compiler_pass:
        try:
            closure = close_compiled_payload(compilation, graph, matrix)
            closure_again = close_compiled_payload(compilation, copy.deepcopy(graph), matrix)
        except (TypeError, ValueError):
            closure = None
    assignments = tuple(closure.assignments) if closure is not None else ()
    derivations = tuple(closure.executable_derivations) if closure is not None else ()
    closure_pass = bool(
        closure is not None and closure.resource_complete and len(assignments) == 1
        and len(derivations) == 1 and assignments[0].canonical_program_id
    )
    deterministic_pass = bool(
        closure_pass and closure_again is not None
        and canonical_json(closure.to_dict()) == canonical_json(closure_again.to_dict())
    )
    derivation = derivations[0] if derivations else None
    projection_pass = bool(
        derivation is not None
        and derivation.typed_signature == signature_id
        and derivation.projection_operator == signature.projection_operator
        and derivation.output_domain == signature.answer_domain
        and derivation.projected_answer != ""
        and derivation.provenance_complete
        and derivation.evidence_ids
    )
    serialization_pass = bool(
        compiler_pass and canonical_json(json.loads(compilation.canonical_payload)) == compilation.canonical_payload
    )
    negative = _payload(signature_id)
    negative["plans"][0]["projection_operator"] = (
        "SCALAR_RESULT_PROJECTION"
        if signature.projection_operator != "SCALAR_RESULT_PROJECTION"
        else "VALUE_PROJECTION"
    )
    negative_pass = not compile_active_planner_payload(negative, view, matrix).ok
    booleans = {
        "registry_present": signature_id in OPERATION_SIGNATURES,
        "active_compiler_fixture_pass": compiler_pass,
        "closure_fixture_pass": closure_pass,
        "deterministic_executor_fixture_pass": deterministic_pass,
        "projection_fixture_pass": projection_pass,
        "serialization_roundtrip_fixture_pass": serialization_pass,
    }
    active = all(booleans.values())
    reasons = [key for key, value in booleans.items() if not value]
    if not negative_pass:
        reasons.append("negative_fixture_failed")
    return {
        "signature_id": signature_id,
        "operation_family": signature.operation_family,
        "execution_family": signature.execution_family,
        "required_role_shapes": {role.name: role.shape for role in signature.required_roles},
        "projection_operator": signature.projection_operator,
        "answer_domain": signature.answer_domain,
        **booleans,
        "negative_fixture_pass": negative_pass,
        "canonical_program_id": assignments[0].canonical_program_id if assignments else "",
        "projected_answer_sha256": canonical_json_hash({"answer": derivation.projected_answer}) if derivation else "",
        "provenance_count": len(derivation.evidence_ids) if derivation else 0,
        "constructor_active": active,
        "constructor_failure_reasons": reasons,
        "active": active,
    }


def build_signature_capability_matrix() -> Dict[str, Any]:
    matrix = {
        "schema_version": "certa_active_v1_signature_capability_v1",
        "activation_equation": "registry_present and active_compiler_fixture_pass and closure_fixture_pass and deterministic_executor_fixture_pass and projection_fixture_pass and serialization_roundtrip_fixture_pass",
        "rows": [_capability_row(item) for item in sorted(DESIGN_SIGNATURE_IDS)],
    }
    active_signature_ids(matrix)
    if not all(row["negative_fixture_pass"] for row in matrix["rows"]):
        raise RuntimeError("capability_negative_fixture_failed")
    matrix["matrix_sha256"] = canonical_json_hash(matrix)
    return matrix


def build_signature_capability_schema() -> Dict[str, Any]:
    boolean_fields = [
        "registry_present", "active_compiler_fixture_pass", "closure_fixture_pass",
        "deterministic_executor_fixture_pass", "projection_fixture_pass",
        "serialization_roundtrip_fixture_pass", "negative_fixture_pass", "constructor_active", "active",
    ]
    properties: Dict[str, Any] = {name: {"type": "boolean"} for name in boolean_fields}
    properties.update({
        "signature_id": {"type": "string", "enum": sorted(DESIGN_SIGNATURE_IDS)},
        "operation_family": {"type": "string"}, "execution_family": {"type": "string"},
        "required_role_shapes": {"type": "object", "additionalProperties": {"type": "string"}},
        "projection_operator": {"type": "string"}, "answer_domain": {"type": "string"},
        "canonical_program_id": {"type": "string", "minLength": 1},
        "projected_answer_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "provenance_count": {"type": "integer", "minimum": 1},
        "constructor_failure_reasons": {"type": "array", "items": {"type": "string"}},
    })
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
        "properties": {
            "schema_version": {"const": "certa_active_v1_signature_capability_v1"},
            "activation_equation": {"type": "string"},
            "rows": {"type": "array", "minItems": len(DESIGN_SIGNATURE_IDS), "maxItems": len(DESIGN_SIGNATURE_IDS),
                     "items": {"type": "object", "properties": properties,
                               "required": sorted(properties), "additionalProperties": False}},
            "matrix_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
        "required": ["schema_version", "activation_equation", "rows", "matrix_sha256"],
        "additionalProperties": False,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def freeze_role_interface(matrix_path: Path, output: Path, profile_path: Path) -> Dict[str, Any]:
    if _git("status", "--porcelain"):
        raise RuntimeError("role_interface_freeze_requires_clean_worktree")
    for relative, expected in FROZEN_LEGACY_HASHES.items():
        if _sha256(REPO / relative) != expected:
            raise RuntimeError(f"default_frozen_source_changed:{relative}")
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    jsonschema.validate(matrix, build_signature_capability_schema())
    ids = active_signature_ids(matrix)
    freeze_dir = output.parent
    prompt_path = freeze_dir / "ROLE_PROMPT_TEMPLATE.txt"
    wire_path = freeze_dir / "ROLE_WIRE_SCHEMA.json"
    semantic_path = freeze_dir / "ROLE_SEMANTIC_SCHEMA.json"
    source_manifest_path = freeze_dir / "ROLE_INTERFACE_SOURCE_MANIFEST.json"
    prompt_path.write_text(build_role_prompt("<QUESTION>", ids) + "\n", encoding="utf-8")
    _write_json(wire_path, build_role_wire_schema(ids))
    _write_json(semantic_path, build_role_semantic_schema(ids))
    method_sha = _git("rev-parse", "HEAD")
    source_manifest = {
        "schema_version": "certa_active_v1_role_interface_source_manifest_v1",
        "role_source_files": {
            relative: _sha256(REPO / relative)
            for relative in ("certa/active_v1/role_contract.py", "certa/active_v1/planner_adapter.py", "tools/certa_active_v1.py")
        },
        "capability_matrix_path": str(matrix_path.resolve()),
        "capability_matrix_sha256": _sha256(matrix_path),
        "model_profile_path": str(profile_path.resolve()),
        "model_profile_sha256": _sha256(profile_path),
        "method_commit_sha": method_sha,
        "pack_manifest_sha256": _sha256(PACK / "PACK_MANIFEST.json"),
    }
    _write_json(source_manifest_path, source_manifest)
    interface = {
        "schema_version": "certa_active_role_interface_freeze_v1",
        "method_sha": method_sha,
        "role_source_sha256": canonical_json_hash(source_manifest),
        "prompt_sha256": _sha256(prompt_path),
        "wire_schema_sha256": canonical_json_hash(build_role_wire_schema(ids)),
        "semantic_schema_sha256": canonical_json_hash(build_role_semantic_schema(ids)),
        "validator_sha256": hashlib.sha256(inspect.getsource(validate_role_contract).encode("utf-8")).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    schema = json.loads((PACK / "schemas/INTERFACE_FREEZE_SCHEMA.json").read_text(encoding="utf-8"))
    jsonschema.validate(interface, schema)
    _write_json(output, interface)
    return interface


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    capability = sub.add_parser("capability-fixtures")
    capability.add_argument("--output", type=Path, required=True)
    capability.add_argument("--schema-output", type=Path)
    role = sub.add_parser("freeze-role-interface")
    role.add_argument("--capability-matrix", type=Path, required=True)
    role.add_argument("--output", type=Path, required=True)
    role.add_argument("--profile", type=Path, default=REPO / "configs/profiles/certa_active_v1.env")
    args = parser.parse_args()
    if args.command == "capability-fixtures":
        matrix = build_signature_capability_matrix()
        schema_path = args.schema_output or args.output.with_suffix(".schema.json")
        _write_json(schema_path, build_signature_capability_schema())
        _write_json(args.output, matrix)
        jsonschema.validate(matrix, build_signature_capability_schema())
    elif args.command == "freeze-role-interface":
        freeze_role_interface(args.capability_matrix, args.output, args.profile)


if __name__ == "__main__":
    main()
