#!/usr/bin/env python3
"""Offline fixtures and freeze commands for the CERTA Active V1 adapter."""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import os
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
from certa.active_v1.cohort import RUNTIME_FIELDS, select_active_cohorts
from certa.active_v1.answer_authority import active_answer_hash
from certa.active_v1.role_contract import (
    ROLE_MAX_TOKENS,
    ROLE_TUPLES,
    build_role_prompt,
    build_role_semantic_schema,
    build_role_wire_schema,
    validate_role_contract,
)
from certa.active_v1.role_contract_v3 import (
    ROLE_V3_MAX_TOKENS,
    build_role_v3_prompt,
    build_role_v3_prompt_template,
    derive_role_v3_record,
    parse_role_v3_output,
    validate_role_v3_artifacts,
)
from certa.operations.contracts import OPERATION_SIGNATURES
from certa.planner.schema_view import build_proposal_blind_planner_view
from certa.planner.typed_planner import PLANNER_VERSION
from certa.reproducibility.canonical_json import canonical_json, canonical_json_hash


PACK = Path("/home/hsh/ME/Table/EMNLP2026/certa_goal_packs/CERTA_ACTIVE_V1_FINAL_METHOD_GOAL_REVISED_PACK")
ROLE_V3_PACK = Path("/home/hsh/ME/Table/EMNLP2026/certa_goal_packs/CERTA_ACTIVE_V1_ROLE_V3_FINAL_METHOD_PACK")
DESIGN_SIGNATURE_IDS = tuple(ROLE_TUPLES)
ROLE_INTERFACE_SCHEMA_VERSION = "certa_active_role_interface_freeze_v2"
FROZEN_LEGACY_HASHES = {
    "certa/egra/retrieval.py": "02d30f80ac2e3c0827c4eaf819a2aa0f66b50e42cd1b93681041fc1a25995552",
    "certa/repair/causal_epistemic_agent.py": "77619174cad1695cc11db73e2cf436c1b3d356a70e7b94b4d5870e5e0d9a9787",
}
EXPECTED_FIXTURE_ANSWERS = {
    "ARGMAX_ENTITY": "A",
    "ARGMAX_ENTITY_SET": "A | C",
    "ARGMIN_ENTITY": "B",
    "ARGMIN_ENTITY_SET": "B | D",
    "AVERAGE_SCALAR": "3",
    "COUNT_SCALAR": "2",
    "DIFF_SCALAR": "2",
    "LOOKUP_VALUE_ENTITY": "Alpha",
    "LOOKUP_VALUE_SCALAR": "4",
    "PAIR_COMPARE_BOOLEAN": "true",
    "RATIO_SCALAR": "2",
    "SUM_SCALAR": "6",
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
        and derivations[0].projected_answer == EXPECTED_FIXTURE_ANSWERS[signature_id]
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
        "expected_projected_answer": EXPECTED_FIXTURE_ANSWERS[signature_id],
        "observed_projected_answer": derivation.projected_answer if derivation else "",
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
        "expected_projected_answer": {"type": "string", "minLength": 1},
        "observed_projected_answer": {"type": "string", "minLength": 1},
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


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    return [
        dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical_json(dict(row)) + "\n" for row in rows), encoding="utf-8")


def freeze_active_cohorts(
    dev_source: Path,
    train_source: Path,
    table_root: Path,
    historical_paths: list[Path],
    output_root: Path,
) -> Dict[str, Any]:
    historical_rows = [row for path in historical_paths for row in _read_jsonl(path)]
    result = select_active_cohorts(
        _read_jsonl(dev_source), _read_jsonl(train_source), table_root, historical_rows,
    )
    membership_fields = (
        "sample_id", "table_id", "stable_hash", "table_content_sha256",
        "source_order", "source_split",
    )
    dev_members = [{field: row[field] for field in membership_fields} for row in result["dev"]]
    holdout_members = [{field: row[field] for field in membership_fields} for row in result["holdout"]]
    integration_members = dev_members[:16]
    paths = {
        "dev_members": output_root / "freeze/DEV64_IDENTITIES.blind.jsonl",
        "holdout_members": output_root / "freeze/HOLDOUT64_IDENTITIES.blind.jsonl",
        "dev_runtime": output_root / "inputs/dev64_runtime.jsonl",
        "holdout_runtime": output_root / "inputs/holdout64_runtime.sealed.jsonl",
        "integration16": output_root / "integration/INTEGRATION16_IDENTITIES.jsonl",
    }
    _write_jsonl(paths["dev_members"], dev_members)
    _write_jsonl(paths["holdout_members"], holdout_members)
    _write_jsonl(paths["dev_runtime"], [dict(row["runtime"]) for row in result["dev"]])
    _write_jsonl(paths["holdout_runtime"], [dict(row["runtime"]) for row in result["holdout"]])
    os.chmod(paths["holdout_runtime"], 0o440)
    _write_jsonl(paths["integration16"], integration_members)
    freeze = {
        "schema_version": "certa_active_v1_cohort_selection_freeze_v1",
        "seed": result["seed"],
        "domain_separator": result["domain_separator"],
        "selection_method": "stable_sha256_one_sample_per_normalized_table_content_class",
        "selection_uses_answer_or_operation": False,
        "official_test_used": False,
        "runtime_fields": list(RUNTIME_FIELDS),
        "source_paths": {"dev": str(dev_source.resolve()), "holdout": str(train_source.resolve())},
        "source_sha256": {"dev": _sha256(dev_source), "holdout": _sha256(train_source)},
        "historical_paths": [str(path.resolve()) for path in historical_paths],
        "historical_sha256": {str(path.resolve()): _sha256(path) for path in historical_paths},
        "historical_table_count": len(result["historical_table_ids"]),
        "historical_content_class_count": result["historical_content_class_count"],
        "dev_candidate_class_count": result["dev_candidate_class_count"],
        "holdout_candidate_class_count": result["holdout_candidate_class_count"],
        "dev_count": len(dev_members),
        "holdout_count": len(holdout_members),
        "integration16_count": len(integration_members),
        "dev_holdout_table_overlap": 0,
        "dev_holdout_content_class_overlap": 0,
        "historical_selected_table_overlap": 0,
        "artifact_sha256": {key: _sha256(path) for key, path in paths.items()},
    }
    _write_json(output_root / "freeze/COHORT_SELECTION_FREEZE.json", freeze)
    return freeze


def run_active_b0(
    runtime_path: Path,
    table_root: Path,
    output_root: Path,
    cache_path: Path,
) -> Dict[str, Any]:
    from run_cscr_pipeline import (
        OpenAIChatGenerator,
        build_structure_aware_prompt,
        extract_answer,
        load_table_for_cscr,
    )

    runtime_rows = _read_jsonl(runtime_path)
    if len(runtime_rows) != 64:
        raise ValueError(f"active_b0_requires_dev64:{len(runtime_rows)}")
    output_path = output_root / "b0/DEV_B0_FREEZE.jsonl"
    ledger_path = output_root / "logs/B0_ENDPOINT_LEDGER.jsonl"
    existing = _read_jsonl(output_path) if output_path.is_file() else []
    ledger = _read_jsonl(ledger_path) if ledger_path.is_file() else []
    if len(existing) > len(runtime_rows):
        raise ValueError("active_b0_existing_rows_exceed_runtime")
    generator = OpenAIChatGenerator(
        model="Qwen3-8B",
        api_base_url="http://127.0.0.1:30338/v1",
        api_key_env="EMPTY",
        timeout=120.0,
        max_retries=0,
        rate_limit_seconds=0.0,
        max_model_len=32768,
        cache_path=str(cache_path),
        cache_mode="readwrite",
        backend_name="vllm_chat",
    )
    table_cache: Dict[str, Dict[str, Any]] = {}
    prompts: list[str] = []
    for index, runtime in enumerate(runtime_rows):
        if set(runtime) != set(RUNTIME_FIELDS):
            raise ValueError(f"active_b0_runtime_fields_mismatch:{index}")
        table = load_table_for_cscr(dict(runtime), str(table_root), table_cache, "hitab")
        prompt = build_structure_aware_prompt(table, str(runtime["question"]))
        prompts.append(prompt)
        if index < len(existing):
            row = existing[index]
            if row.get("sample_id") != runtime["id"] or row.get("table_id") != runtime["table_id"]:
                raise ValueError(f"active_b0_resume_identity_mismatch:{index}")
            continue
        request_kwargs = generator._completion_request_kwargs(
            prompt=prompt, max_new_tokens=32, temperature=0.0, top_p=1.0,
        )
        request_record = {
            "schema_version": "certa_active_v1_b0_raw_request_v1",
            "logical_call_index": index,
            "method": "POST",
            "path": "/v1/chat/completions",
            "request": request_kwargs,
            "request_sha256": canonical_json_hash(request_kwargs),
        }
        request_path = output_root / f"raw/b0/{index:03d}_request.json"
        response_path = output_root / f"raw/b0/{index:03d}_response.json"
        _write_json(request_path, request_record)
        started_at = datetime.now(timezone.utc).isoformat()
        try:
            generated = generator.generate(
                [prompt], max_new_tokens=32, temperature=0.0, top_p=1.0,
            )[0]
        except Exception as error:
            _write_json(response_path, {
                "schema_version": "certa_active_v1_b0_raw_response_v1",
                "ok": False,
                "error_type": type(error).__name__,
                "error_sha256": hashlib.sha256(str(error).encode("utf-8")).hexdigest(),
            })
            raise
        answer = extract_answer(str(generated.get("text") or ""))
        if not answer:
            raise ValueError(f"active_b0_empty_answer:{runtime['id']}")
        response_record = {
            "schema_version": "certa_active_v1_b0_raw_response_v1",
            "ok": True,
            "generation": generated,
            "generation_sha256": canonical_json_hash(generated),
        }
        _write_json(response_path, response_record)
        record = {
            "schema_version": "certa_active_v1_b0_freeze_v1",
            "sample_id": runtime["id"],
            "table_id": runtime["table_id"],
            "source_order": index,
            "question_sha256": hashlib.sha256(str(runtime["question"]).encode("utf-8")).hexdigest(),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "request_sha256": request_record["request_sha256"],
            "raw_request_path": str(request_path.resolve()),
            "raw_request_sha256": _sha256(request_path),
            "raw_response_path": str(response_path.resolve()),
            "raw_response_sha256": _sha256(response_path),
            "raw_text": generated["text"],
            "raw_text_sha256": hashlib.sha256(str(generated["text"]).encode("utf-8")).hexdigest(),
            "b0_answer": answer,
            "b0_answer_sha256": active_answer_hash(answer),
            "api_usage": generated.get("api_usage") or {},
            "generation_seconds": generated.get("generation_seconds", 0.0),
            "api_cache_hit": bool(generated.get("api_cache_hit", False)),
        }
        existing.append(record)
        ledger.append({
            "schema_version": "certa_active_v1_endpoint_ledger_v1",
            "logical_call_type": "DEV_B0",
            "logical_call_index": index,
            "sample_id": runtime["id"],
            "method": "POST",
            "path": "/v1/chat/completions",
            "started_at": started_at,
            "transport_attempts": 0 if record["api_cache_hit"] else 1,
            "cache_hit": record["api_cache_hit"],
            "request_sha256": record["request_sha256"],
            "response_sha256": record["raw_response_sha256"],
        })
        _write_jsonl(output_path, existing)
        _write_jsonl(ledger_path, ledger)

    replay = OpenAIChatGenerator(
        model="Qwen3-8B", api_base_url="http://127.0.0.1:30338/v1",
        api_key_env="EMPTY", timeout=120.0, max_retries=0,
        rate_limit_seconds=0.0, max_model_len=32768,
        cache_path=str(cache_path), cache_mode="require", backend_name="vllm_chat",
    )
    replay_rows = replay.generate(prompts, max_new_tokens=32, temperature=0.0, top_p=1.0)
    replay_match = all(
        hashlib.sha256(str(generated["text"]).encode("utf-8")).hexdigest() == frozen["raw_text_sha256"]
        and active_answer_hash(extract_answer(str(generated["text"]))) == frozen["b0_answer_sha256"]
        for generated, frozen in zip(replay_rows, existing)
    )
    if not replay_match or replay.cache_hits != 64 or replay.cache_misses != 0:
        raise RuntimeError("active_b0_cache_replay_failed")
    replay_proof = {
        "schema_version": "certa_active_v1_b0_cache_replay_v1",
        "record_count": len(existing),
        "cache_hits": replay.cache_hits,
        "cache_misses": replay.cache_misses,
        "byte_and_answer_hash_match": replay_match,
        "b0_freeze_sha256": _sha256(output_path),
        "cache_sha256": _sha256(cache_path),
    }
    _write_json(output_root / "b0/B0_CACHE_REPLAY_PROOF.json", replay_proof)
    return replay_proof


def run_sealed_role_predictions(
    questions_path: Path,
    interface_freeze_path: Path,
    matrix_path: Path,
    output_root: Path,
    cache_path: Path,
) -> Dict[str, Any]:
    from run_cscr_pipeline import OpenAIChatGenerator

    predictions_path = output_root / "role/ROLE_SEALED_PREDICTIONS.json"
    if predictions_path.exists():
        raise FileExistsError(f"refusing_to_overwrite_role_predictions:{predictions_path}")
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    items = questions.get("items")
    if not isinstance(items, list) or len(items) != 16:
        raise ValueError("sealed_role_questions_must_have_16_items")
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    active_ids = active_signature_ids(matrix)
    interface = json.loads(interface_freeze_path.read_text(encoding="utf-8"))
    if interface.get("method_sha") != _git("rev-parse", "HEAD"):
        raise ValueError("role_interface_commit_not_current_head")
    if interface.get("prompt_sha256") != _sha256(interface_freeze_path.parent / "ROLE_PROMPT_TEMPLATE.txt"):
        raise ValueError("role_interface_prompt_hash_mismatch")
    generator = OpenAIChatGenerator(
        model="Qwen3-8B", api_base_url="http://127.0.0.1:30338/v1",
        api_key_env="EMPTY", timeout=120.0, max_retries=0,
        rate_limit_seconds=0.0, max_model_len=32768,
        cache_path=str(cache_path), cache_mode="readwrite", backend_name="vllm_chat",
    )
    response_schema = build_role_wire_schema(active_ids)
    prediction_items = []
    ledger = []
    for index, item in enumerate(items):
        item_id = str(item.get("id") or "")
        question = str(item.get("question") or "")
        if not item_id or not question or set(item) != {"id", "question"}:
            raise ValueError(f"invalid_sealed_role_question:{index}")
        prompt = build_role_prompt(question, active_ids)
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "certa_active_role_v2", "schema": response_schema, "strict": True},
        }
        request_kwargs = generator._completion_request_kwargs(
            prompt=prompt, max_new_tokens=ROLE_MAX_TOKENS,
            temperature=0.0, top_p=1.0, response_format=response_format,
        )
        request_path = output_root / f"raw/role/{index:02d}_{item_id}_request.json"
        response_path = output_root / f"raw/role/{index:02d}_{item_id}_response.json"
        _write_json(request_path, {
            "schema_version": "certa_active_v1_role_raw_request_v1",
            "logical_call_index": index,
            "id": item_id,
            "method": "POST",
            "path": "/v1/chat/completions",
            "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "request": request_kwargs,
            "request_sha256": canonical_json_hash(request_kwargs),
        })
        started_at = datetime.now(timezone.utc).isoformat()
        try:
            generated = generator.generate_json_schema(
                prompt, response_schema=response_schema,
                schema_name="certa_active_role_v2", max_new_tokens=ROLE_MAX_TOKENS,
                temperature=0.0, top_p=1.0,
            )
        except Exception as error:
            _write_json(response_path, {
                "schema_version": "certa_active_v1_role_raw_response_v1",
                "ok": False, "error_type": type(error).__name__,
                "error_sha256": hashlib.sha256(str(error).encode("utf-8")).hexdigest(),
            })
            raise
        validation = validate_role_contract(str(generated.get("text") or ""), active_ids)
        _write_json(response_path, {
            "schema_version": "certa_active_v1_role_raw_response_v1",
            "ok": True,
            "generation": generated,
            "validation": {
                "parse_ok": validation.parse_ok,
                "wire_valid": validation.wire_valid,
                "semantic_schema_valid": validation.semantic_schema_valid,
                "local_validator_valid": validation.local_validator_valid,
                "parse_errors": list(validation.parse_errors),
                "wire_errors": list(validation.wire_errors),
                "semantic_errors": list(validation.semantic_errors),
                "local_errors": list(validation.local_errors),
            },
        })
        if not validation.parse_ok or not validation.wire_valid:
            raise ValueError(f"sealed_role_wire_failure:{item_id}")
        created_at = datetime.now(timezone.utc).isoformat()
        prediction_items.append({
            "id": item_id,
            "wire_valid": validation.wire_valid,
            "semantic_schema_valid": validation.semantic_schema_valid,
            "local_validator_valid": validation.local_validator_valid,
            "prediction": validation.payload,
            "raw_response_sha256": _sha256(response_path),
            "created_at": created_at,
        })
        ledger.append({
            "schema_version": "certa_active_v1_endpoint_ledger_v1",
            "logical_call_type": "SEALED_ROLE_CONFIRMATION",
            "logical_call_index": index,
            "id": item_id,
            "method": "POST", "path": "/v1/chat/completions",
            "started_at": started_at, "completed_at": created_at,
            "transport_attempts": 0 if generated.get("api_cache_hit") else 1,
            "cache_hit": bool(generated.get("api_cache_hit")),
            "request_sha256": canonical_json_hash(request_kwargs),
            "response_sha256": _sha256(response_path),
            "usage": generated.get("api_usage") or {},
            "generation_seconds": generated.get("generation_seconds", 0.0),
        })
    predictions = {
        "schema_version": "certa_active_role_predictions_v2",
        "fixture_only": False,
        "questions_sha256": _sha256(questions_path),
        "interface_freeze_sha256": _sha256(interface_freeze_path),
        "items": prediction_items,
    }
    schema_path = PACK / "schemas/ROLE_PREDICTIONS_SCHEMA.json"
    resolver = jsonschema.RefResolver(base_uri=(PACK / "schemas").resolve().as_uri() + "/", referrer=json.loads(schema_path.read_text()))
    jsonschema.validate(predictions, json.loads(schema_path.read_text()), resolver=resolver)
    _write_json(predictions_path, predictions)
    _write_jsonl(output_root / "logs/ROLE_ENDPOINT_LEDGER.jsonl", ledger)
    return predictions


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
        "schema_version": ROLE_INTERFACE_SCHEMA_VERSION,
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


def freeze_role_v3_interface(output_root: Path, profile_path: Path) -> Dict[str, Any]:
    """Freeze every Role V3 authority before the first fresh endpoint call."""
    if _git("status", "--porcelain"):
        raise RuntimeError("role_v3_interface_freeze_requires_clean_worktree")
    for relative, expected in FROZEN_LEGACY_HASHES.items():
        if _sha256(REPO / relative) != expected:
            raise RuntimeError(f"default_frozen_source_changed:{relative}")
    freeze_dir = output_root / "freeze"
    sources = {
        "ROLE_V3_PROMPT_TEMPLATE.txt": ROLE_V3_PACK / "ROLE_V3_PROMPT_TEMPLATE.txt",
        "ROLE_V3_ROLE_CARDS.json": ROLE_V3_PACK / "ROLE_V3_ROLE_CARDS.json",
        "ROLE_V3_OUTPUT_SCHEMA.json": ROLE_V3_PACK / "ROLE_V3_OUTPUT_SCHEMA.json",
        "ROLE_V3_CANONICAL_REGISTRY.json": ROLE_V3_PACK / "ROLE_V3_CANONICAL_REGISTRY.json",
        "ROLE_V3_FRESH_QUESTIONS.json": ROLE_V3_PACK / "ROLE_V3_FRESH_QUESTIONS.json",
    }
    if any((freeze_dir / name).exists() for name in sources):
        raise FileExistsError("refusing_to_overwrite_role_v3_interface")
    cards = json.loads(sources["ROLE_V3_ROLE_CARDS.json"].read_text(encoding="utf-8"))
    schema = json.loads(sources["ROLE_V3_OUTPUT_SCHEMA.json"].read_text(encoding="utf-8"))
    registry = json.loads(sources["ROLE_V3_CANONICAL_REGISTRY.json"].read_text(encoding="utf-8"))
    validate_role_v3_artifacts(cards, schema, registry)
    if build_role_v3_prompt_template(cards) != sources["ROLE_V3_PROMPT_TEMPLATE.txt"].read_text(encoding="utf-8"):
        raise ValueError("role_v3_prompt_template_mismatch")
    freeze_dir.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        (freeze_dir / name).write_bytes(source.read_bytes())
    thresholds = {
        "schema_version": "certa_active_role_v3_gate_thresholds_v1",
        "wire_required": 36,
        "supported_coverage_count_min": 18,
        "accepted_role_precision_min": 0.95,
        "false_supported_activation_max": 1,
        "critical_contrast_errors_max": 0,
    }
    _write_json(freeze_dir / "ROLE_V3_GATE_THRESHOLDS.json", thresholds)
    method_sha = _git("rev-parse", "HEAD")
    source_manifest = {
        "schema_version": "certa_active_role_v3_source_manifest_v1",
        "interface_commit": method_sha,
        "source_sha256": {
            relative: _sha256(REPO / relative) for relative in (
                "certa/active_v1/role_contract_v3.py",
                "tools/certa_active_v1.py",
                "configs/profiles/certa_active_v1.env",
            )
        },
        "default_frozen_source_sha256": dict(FROZEN_LEGACY_HASHES),
        "pack_manifest_sha256": _sha256(ROLE_V3_PACK / "PACK_MANIFEST.json"),
        "profile_sha256": _sha256(profile_path),
    }
    _write_json(freeze_dir / "ROLE_V3_SOURCE_MANIFEST.json", source_manifest)
    interface = {
        "schema_version": "certa_active_role_v3_interface_freeze_v1",
        "interface_commit": method_sha,
        "source_manifest_sha256": _sha256(freeze_dir / "ROLE_V3_SOURCE_MANIFEST.json"),
        "artifact_sha256": {name: _sha256(freeze_dir / name) for name in sources},
        "thresholds": thresholds,
        "model": {
            "model": "Qwen3-8B",
            "api_base_url": "http://127.0.0.1:30338/v1",
            "method": "POST",
            "path": "/v1/chat/completions",
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": ROLE_V3_MAX_TOKENS,
            "enable_thinking": False,
            "sdk_max_retries": 0,
            "cache_mode": "off",
            "local_http_trust_env": False,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(freeze_dir / "ROLE_V3_INTERFACE_FREEZE.json", interface)
    return interface


def run_role_v3_predictions(
    questions_path: Path,
    interface_freeze_path: Path,
    output_root: Path,
) -> Dict[str, Any]:
    """Run exactly one uncached structured-output attempt per fresh question."""
    from run_cscr_pipeline import OpenAIChatGenerator

    predictions_path = output_root / "role_v3/ROLE_V3_PREDICTIONS.json"
    close_path = output_root / "role_v3/ROLE_V3_PREDICTION_CLOSE.json"
    if predictions_path.exists() or close_path.exists():
        raise FileExistsError(f"refusing_to_overwrite_role_v3_predictions:{predictions_path}")
    if _git("status", "--porcelain"):
        raise RuntimeError("role_v3_predictions_require_clean_worktree")
    interface = json.loads(interface_freeze_path.read_text(encoding="utf-8"))
    head = _git("rev-parse", "HEAD")
    if interface.get("interface_commit") != head:
        raise ValueError("role_v3_interface_commit_not_current_head")
    freeze_dir = interface_freeze_path.parent
    for name, expected in interface.get("artifact_sha256", {}).items():
        if _sha256(freeze_dir / name) != expected:
            raise ValueError(f"role_v3_frozen_artifact_hash_mismatch:{name}")
    if _sha256(questions_path) != interface["artifact_sha256"]["ROLE_V3_FRESH_QUESTIONS.json"]:
        raise ValueError("role_v3_questions_hash_mismatch")
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    items = questions.get("items")
    if not isinstance(items, list) or len(items) != 36:
        raise ValueError("role_v3_requires_36_questions")
    cards = json.loads((freeze_dir / "ROLE_V3_ROLE_CARDS.json").read_text(encoding="utf-8"))
    schema = json.loads((freeze_dir / "ROLE_V3_OUTPUT_SCHEMA.json").read_text(encoding="utf-8"))
    registry = json.loads((freeze_dir / "ROLE_V3_CANONICAL_REGISTRY.json").read_text(encoding="utf-8"))
    validate_role_v3_artifacts(cards, schema, registry)
    generator = OpenAIChatGenerator(
        model="Qwen3-8B", api_base_url="http://127.0.0.1:30338/v1",
        api_key_env="EMPTY", timeout=120.0, max_retries=0,
        rate_limit_seconds=0.0, max_model_len=32768,
        cache_path="", cache_mode="off", backend_name="vllm_chat",
    )
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "certa_active_role_v3", "schema": schema, "strict": True},
    }
    predictions: list[Dict[str, Any]] = []
    ledger: list[Dict[str, Any]] = []
    for index, item in enumerate(items):
        if set(item) != {"id", "question"}:
            raise ValueError(f"invalid_role_v3_question:{index}")
        item_id, question = str(item["id"]), str(item["question"])
        prompt = build_role_v3_prompt(question, cards)
        request_kwargs = generator._completion_request_kwargs(
            prompt=prompt, max_new_tokens=ROLE_V3_MAX_TOKENS,
            temperature=0.0, top_p=1.0, response_format=response_format,
        )
        request_path = output_root / f"raw/role_v3/{index:02d}_{item_id}_request.json"
        response_path = output_root / f"raw/role_v3/{index:02d}_{item_id}_response.json"
        request_record = {
            "schema_version": "certa_active_role_v3_raw_request_v1",
            "logical_call_index": index, "id": item_id,
            "method": "POST", "path": "/v1/chat/completions",
            "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "request": request_kwargs,
            "request_sha256": canonical_json_hash(request_kwargs),
        }
        _write_json(request_path, request_record)
        started_at = datetime.now(timezone.utc).isoformat()
        generated: Dict[str, Any] = {}
        prediction = None
        canonical_record = None
        wire_valid = False
        error_type = None
        try:
            generated = generator.generate_json_schema(
                prompt, response_schema=schema, schema_name="certa_active_role_v3",
                max_new_tokens=ROLE_V3_MAX_TOKENS, temperature=0.0, top_p=1.0,
            )
            prediction = parse_role_v3_output(str(generated.get("text") or ""), schema)
            canonical_record = derive_role_v3_record(prediction, schema, registry)
            wire_valid = True
        except Exception as error:
            error_type = type(error).__name__
            generated = generated or {
                "text": None,
                "error_sha256": hashlib.sha256(str(error).encode("utf-8")).hexdigest(),
            }
        completed_at = datetime.now(timezone.utc).isoformat()
        _write_json(response_path, {
            "schema_version": "certa_active_role_v3_raw_response_v1",
            "ok": error_type is None,
            "generation": generated,
            "prediction": prediction,
            "canonical_record": canonical_record,
            "wire_valid": wire_valid,
            "error_type": error_type,
        })
        predictions.append({
            "id": item_id,
            "prediction": prediction,
            "canonical_record": canonical_record,
            "wire_valid": wire_valid,
            "raw_request_path": str(request_path.resolve()),
            "raw_request_sha256": _sha256(request_path),
            "raw_response_path": str(response_path.resolve()),
            "raw_response_sha256": _sha256(response_path),
            "created_at": completed_at,
        })
        ledger.append({
            "schema_version": "certa_active_role_v3_endpoint_ledger_v1",
            "logical_call_type": "ROLE_V3_FRESH_CONFIRMATION",
            "logical_call_index": index, "id": item_id,
            "method": "POST", "path": "/v1/chat/completions",
            "started_at": started_at, "completed_at": completed_at,
            "transport_attempts": 1, "cache_hit": False,
            "request_sha256": request_record["request_sha256"],
            "response_sha256": _sha256(response_path),
            "usage": generated.get("api_usage") or {},
            "generation_seconds": generated.get("generation_seconds", 0.0),
            "wire_valid": wire_valid,
        })
        _write_jsonl(output_root / "logs/ROLE_V3_ENDPOINT_LEDGER.jsonl", ledger)
    result = {
        "schema_version": "certa_active_role_v3_predictions_v1",
        "set_id": questions["set_id"],
        "interface_commit": head,
        "interface_freeze_sha256": _sha256(interface_freeze_path),
        "questions_sha256": _sha256(questions_path),
        "items": predictions,
    }
    _write_json(predictions_path, result)
    if _git("status", "--porcelain"):
        raise RuntimeError("role_v3_prediction_close_requires_clean_worktree")
    raw_requests = sorted((output_root / "raw/role_v3").glob("*_request.json"))
    raw_responses = sorted((output_root / "raw/role_v3").glob("*_response.json"))
    close = {
        "schema_version": "certa_active_role_v3_prediction_close_v1",
        "interface_commit": head,
        "interface_freeze_sha256": _sha256(interface_freeze_path),
        "questions_sha256": _sha256(questions_path),
        "predictions_sha256": _sha256(predictions_path),
        "ledger_sha256": _sha256(output_root / "logs/ROLE_V3_ENDPOINT_LEDGER.jsonl"),
        "logical_calls": len(predictions),
        "transport_attempts": sum(row["transport_attempts"] for row in ledger),
        "raw_request_count": len(raw_requests),
        "raw_response_count": len(raw_responses),
        "raw_artifact_sha256": {
            str(path.relative_to(output_root)): _sha256(path)
            for path in raw_requests + raw_responses
        },
        "worktree_clean": True,
        "closed_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(close_path, close)
    return result


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
    cohorts = sub.add_parser("freeze-cohorts")
    cohorts.add_argument("--dev-source", type=Path, required=True)
    cohorts.add_argument("--train-source", type=Path, required=True)
    cohorts.add_argument("--table-root", type=Path, required=True)
    cohorts.add_argument("--historical", type=Path, action="append", required=True)
    cohorts.add_argument("--output-root", type=Path, required=True)
    b0 = sub.add_parser("run-b0")
    b0.add_argument("--runtime", type=Path, required=True)
    b0.add_argument("--table-root", type=Path, required=True)
    b0.add_argument("--output-root", type=Path, required=True)
    b0.add_argument("--cache", type=Path, required=True)
    sealed_role = sub.add_parser("run-sealed-role")
    sealed_role.add_argument("--questions", type=Path, required=True)
    sealed_role.add_argument("--interface-freeze", type=Path, required=True)
    sealed_role.add_argument("--capability-matrix", type=Path, required=True)
    sealed_role.add_argument("--output-root", type=Path, required=True)
    sealed_role.add_argument("--cache", type=Path, required=True)
    role_v3_freeze = sub.add_parser("freeze-role-v3-interface")
    role_v3_freeze.add_argument("--output-root", type=Path, required=True)
    role_v3_freeze.add_argument("--profile", type=Path, default=REPO / "configs/profiles/certa_active_v1.env")
    role_v3_run = sub.add_parser("run-role-v3")
    role_v3_run.add_argument("--questions", type=Path, required=True)
    role_v3_run.add_argument("--interface-freeze", type=Path, required=True)
    role_v3_run.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "capability-fixtures":
        matrix = build_signature_capability_matrix()
        schema_path = args.schema_output or args.output.with_suffix(".schema.json")
        _write_json(schema_path, build_signature_capability_schema())
        _write_json(args.output, matrix)
        jsonschema.validate(matrix, build_signature_capability_schema())
    elif args.command == "freeze-role-interface":
        freeze_role_interface(args.capability_matrix, args.output, args.profile)
    elif args.command == "freeze-cohorts":
        freeze_active_cohorts(
            args.dev_source, args.train_source, args.table_root,
            args.historical, args.output_root,
        )
    elif args.command == "run-b0":
        run_active_b0(args.runtime, args.table_root, args.output_root, args.cache)
    elif args.command == "run-sealed-role":
        run_sealed_role_predictions(
            args.questions, args.interface_freeze, args.capability_matrix,
            args.output_root, args.cache,
        )
    elif args.command == "freeze-role-v3-interface":
        freeze_role_v3_interface(args.output_root, args.profile)
    elif args.command == "run-role-v3":
        run_role_v3_predictions(args.questions, args.interface_freeze, args.output_root)


if __name__ == "__main__":
    main()
