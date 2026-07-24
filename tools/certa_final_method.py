#!/usr/bin/env python3
"""Resumable development runner for the bounded CERTA final method.

The runner is intentionally thin: Role, Planner compilation, closure,
execution, raw-artifact serialization, registry reconciliation, and answer
materialization remain in their existing library authorities.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from graph_builder import build_hceg
from run_cscr_pipeline import build_structure_aware_prompt, extract_answer

from certa.active_v1.artifact_authority import (
    ArtifactContext,
    serialize_plan_closure_v3,
)
from certa.active_v1.decision_adapter import (
    assess_decision_eligibility,
    materialize_selected_final,
    reconcile_cera_decision,
)
from certa.active_v1.dataset_adapter_v1 import HiTabAdapterV1
from certa.active_v1.final_method_v1 import (
    ProgramUnionInput,
    VARIANT_IDS,
    build_complete_domain_c2_view,
    build_registry_support_state,
    materialize_registry_selection,
    select_registry_policy,
    union_exact_typed_programs,
)
from certa.active_v1.planner_bridge_v3 import (
    build_v3_arm_view,
    close_compiled_payload,
    compile_active_planner_payload,
)
from certa.active_v1.planner_transport_projection import (
    build_planner_transport_schema,
)
from certa.active_v1.role_contract_v3 import (
    build_role_v3_prompt,
    derive_role_v3_record,
    role_v3_to_planner_query_contract,
)
from certa.derivations.contrast import build_compact_behavioral_contrast_v3
from certa.derivations.iade import (
    build_basis_relative_behavior_classes,
    build_sample_fixed_role_intervention_basis,
)
from certa.egra.evidence_cards import build_structural_evidence_cards
from certa.egra.retrieval import (
    FrozenE5Encoder,
    build_card_index,
    retrieve_structural_cards,
)
from certa.planner.schema_view import build_canonical_structural_group_catalog
from certa.planner.typed_planner import (
    build_typed_derivation_planner_prompt,
    build_typed_planner_response_schema,
)
from certa.grounding.support_partition import partition_support
from certa.repair.repair_prompt import (
    CERA_V3_TEMPLATE_VERSION,
    build_cera_prompt,
)
from certa.repair.safety_validator import validate_cera_output_v3
from certa.reproducibility.canonical_json import canonical_json, canonical_json_hash
from tools.cscr_astra_eval import official_match
import tools.certa_active_v1_completion as completion_runtime


DEFAULT_OUT = Path(
    "/home/hsh/ME/Table/EMNLP2026/certa_active_v1_outputs/"
    "CERTA_FINAL_MULTI_DATASET_ADAPTER_AND_METHOD_COMPLETION"
)
DEFAULT_DATASET = Path("/home/hsh/ME/Table/EMNLP2026/CERTA/dataset")
DEFAULT_DEVELOPMENT = Path(
    "/home/hsh/ME/Table/EMNLP2026/certa_final_workspace/development"
)
ROLE_ROOT = Path(
    "/home/hsh/ME/Table/EMNLP2026/certa_active_v1_outputs/"
    "CERTA_ACTIVE_V1_ROLE_V3_FINAL/freeze"
)
MATRIX_PATH = Path(
    "/home/hsh/ME/Table/EMNLP2026/certa_active_v1_outputs/"
    "CERTA_ACTIVE_V1_FINAL_ASSIGNMENT_LEVEL_GROUNDING_AUTHORITY_REPLAY/"
    "freeze/CONSTRUCTOR_CAPABILITY_MATRIX.json"
)
DECISION_MATRIX_PATH = (
    REPO
    / "tests/active_v1/fixtures/final_completion/"
    "DECISION_CAPABILITY_MATRIX.fixture.json"
)
EMBEDDING_FREEZE = Path(
    "/home/hsh/ME/Table/EMNLP2026/certa_active_v1_outputs/"
    "CERTA_ACTIVE_V1_FINAL_RUNTIME_RECOVERY_20260722_CP2/"
    "freeze/EMBEDDING_RETRIEVAL_FREEZE.json"
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(canonical_json(dict(row)) + "\n" for row in rows),
        encoding="utf-8",
    )


def development_gold_answers(record: Mapping[str, Any]) -> list[Any]:
    labels = record.get("labels")
    answers = labels.get("answer") if isinstance(labels, Mapping) else None
    if not isinstance(answers, list) or not answers:
        raise ValueError(f"development_gold_shape_invalid:{record.get('id') or ''}")
    return list(answers)


def variant_planner_call_types(variant_id: str) -> tuple[str, ...]:
    mapping = {
        "V0_LEGACY_C2_HARD_FILTER": ("C2_LEGACY",),
        "V1_C2_COMPLETE_DOMAIN": ("C2_COMPLETE",),
        "V2_C1_C2_EXACT_PROGRAM_UNION": (
            "C1_COMPLETE",
            "C2_COMPLETE",
        ),
    }
    if variant_id not in mapping:
        raise ValueError(f"unknown_variant:{variant_id}")
    return mapping[variant_id]


def variant_artifact_arm(variant_id: str) -> str:
    mapping = {
        "V0_LEGACY_C2_HARD_FILTER": "C2_ROLE_RETRIEVAL",
        "V1_C2_COMPLETE_DOMAIN": "C2_ROLE_RETRIEVAL",
        "V2_C1_C2_EXACT_PROGRAM_UNION": "C1_C2_EXACT_PROGRAM_UNION",
    }
    if variant_id not in mapping:
        raise ValueError(f"unknown_variant:{variant_id}")
    return mapping[variant_id]


def _cached_call(
    *,
    output: Path,
    split: str,
    generator: Any,
    sample_id: str,
    call_type: str,
    prompt: str,
    max_tokens: int,
    schema: Mapping[str, Any] | None = None,
    full_schema: Mapping[str, Any] | None = None,
    planner_view: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    path = output / split / "model_outputs" / sample_id / f"{call_type}.json"
    if path.is_file():
        cached = _read_json(path)
        if (
            cached.get("prompt_sha256") != canonical_json_hash(prompt)
            or cached.get("schema_sha256")
            != (canonical_json_hash(schema) if schema is not None else "")
        ):
            raise RuntimeError(f"cached_call_identity_mismatch:{sample_id}:{call_type}")
        return cached
    text, result = completion_runtime.model_call(
        generator,
        f"{split.upper()}_{call_type}",
        sample_id,
        prompt,
        max_tokens,
        schema=schema,
        full_schema=full_schema,
        planner_view=planner_view,
    )
    record = {
        "sample_id": sample_id,
        "call_type": call_type,
        "prompt_sha256": canonical_json_hash(prompt),
        "schema_sha256": (
            canonical_json_hash(schema) if schema is not None else ""
        ),
        "text": text,
        "result": result,
    }
    _write_json(path, record)
    return record


def _planner_call(
    *,
    output: Path,
    split: str,
    generator: Any,
    sample_id: str,
    call_type: str,
    view: Mapping[str, Any],
    matrix: Mapping[str, Any],
) -> tuple[Any, Dict[str, Any]]:
    full_schema = build_typed_planner_response_schema(
        view, require_signature_id=True,
    )
    transport_schema = build_planner_transport_schema(full_schema)
    prompt = build_typed_derivation_planner_prompt(view)
    call = _cached_call(
        output=output,
        split=split,
        generator=generator,
        sample_id=sample_id,
        call_type=call_type,
        prompt=prompt,
        max_tokens=512,
        schema=transport_schema,
        full_schema=full_schema,
        planner_view=view,
    )
    compilation = compile_active_planner_payload(
        call["text"], view, matrix,
    )
    return compilation, call


def _retrieval(
    *,
    role: Mapping[str, Any],
    graph: Any,
    table: Mapping[str, Any],
    question: str,
    encoder: Any,
    parent_sha: str,
    embedding_sha: str,
) -> Dict[str, Any]:
    catalog = build_canonical_structural_group_catalog(
        graph=graph, table_json=table,
    )
    cards = build_structural_evidence_cards(catalog)
    index = build_card_index(
        cards,
        encoder,
        parent_sha=parent_sha,
        table_sha256=canonical_json_hash(table),
        embedding_file_tree_sha256=embedding_sha,
    )
    retrieval = retrieve_structural_cards(
        index,
        cards,
        question=question,
        contract=completion_runtime.v3_retrieval_contract(role),
        encoder=encoder,
    )
    retrieval["role_record_sha256"] = canonical_json_hash(role)
    return {
        "retrieval": retrieval,
        "catalog_sha256": catalog["catalog_sha256"],
        "card_count": len(cards),
    }


def _bundle_variant(
    *,
    sample_id: str,
    table_id: str,
    variant_id: str,
    role: Mapping[str, Any],
    b0_answer: Any,
    closure: Any,
    lineage: Sequence[Mapping[str, Any]],
) -> tuple[Any, list[Dict[str, Any]]]:
    arm = variant_artifact_arm(variant_id)
    bundle = serialize_plan_closure_v3(
        closure,
        context=ArtifactContext(
            sample_id=sample_id,
            table_id=table_id,
            arm=arm,
            role_id=str(role["role_id"]),
            role_record_sha256=canonical_json_hash(role),
        ),
        initial_answer=b0_answer,
    )
    vault = [
        {
            "schema_version": "certa_executed_answer_vault_v1",
            "sample_id": sample_id,
            "table_id": table_id,
            "variant_id": variant_id,
            "arm": arm,
            "canonical_program_id": str(
                derivation.operation_metadata["canonical_program_id"]
            ),
            "derivation_id": derivation.derivation_id,
            "answer_hash": completion_runtime.active_answer_hash(
                derivation.projected_answer
            ),
            "executed_answer": derivation.projected_answer,
        }
        for derivation in closure.executable_derivations
    ]
    return bundle, vault


def run_development(
    *,
    output: Path,
    dataset_root: Path,
    development_root: Path,
    limit: int | None,
    split: str = "development",
    runtime_path: Path | None = None,
) -> Dict[str, Any]:
    completion_runtime.OUT = output
    if split not in {"development", "validation", "holdout"}:
        raise ValueError(f"unknown_split:{split}")
    runtime = _read_jsonl(
        runtime_path
        if runtime_path is not None
        else development_root / "development_runtime.jsonl"
    )
    labels = (
        {
            row["id"]: row
            for row in _read_jsonl(
                development_root / "development_labels.jsonl"
            )
        }
        if split == "development"
        else {}
    )
    if limit is not None:
        runtime = runtime[:limit]
    role_schema = _read_json(ROLE_ROOT / "ROLE_V3_OUTPUT_SCHEMA.json")
    role_registry = _read_json(ROLE_ROOT / "ROLE_V3_CANONICAL_REGISTRY.json")
    role_cards = _read_json(ROLE_ROOT / "ROLE_V3_ROLE_CARDS.json")
    matrix = _read_json(MATRIX_PATH)
    decision_matrix = _read_json(DECISION_MATRIX_PATH)
    decision_active_roles = {
        row["role_id"]
        for row in decision_matrix["rows"]
        if row["decision_active"]
    }
    embedding_sha = _read_json(EMBEDDING_FREEZE)["file_tree_sha256"]
    adapter = HiTabAdapterV1(dataset_root / "hitab" / "tables" / "raw")
    generator = completion_runtime.generator()
    encoder = FrozenE5Encoder(device="cpu")
    parent_sha = completion_runtime.git("rev-parse", "HEAD")
    sample_rows = []

    for runtime_row in runtime:
        sample_id = str(runtime_row["id"])
        table_id = str(runtime_row["table_id"])
        question = str(runtime_row["question"])
        gold = (
            development_gold_answers(labels[sample_id])
            if split == "development"
            else None
        )
        if "table_artifact" in runtime_row:
            artifact_path = (
                output / "data/hitab/canonical_tables"
                / str(runtime_row["table_artifact"])
            )
            artifact = _read_json(artifact_path)
            import hashlib
            if (
                hashlib.sha256(artifact_path.read_bytes()).hexdigest()
                != runtime_row.get("table_artifact_sha256")
            ):
                raise RuntimeError(
                    f"canonical_table_artifact_hash_mismatch:{sample_id}"
                )
        else:
            native = adapter.resolve_table(table_id, runtime_record=runtime_row)
            artifact = adapter.canonicalize_table(native)
        table = artifact["table_payload"]["graph_payload"]
        graph = build_hceg(table, question)
        graph_record = graph.to_dict()
        table_sha = canonical_json_hash(artifact)

        b0_call = _cached_call(
            output=output,
            split=split,
            generator=generator,
            sample_id=sample_id,
            call_type="B0",
            prompt=build_structure_aware_prompt(table, question),
            max_tokens=32,
        )
        b0_answer = extract_answer(b0_call["text"])
        if not b0_answer:
            raise RuntimeError(f"development_b0_empty:{sample_id}")

        role_call = _cached_call(
            output=output,
            split=split,
            generator=generator,
            sample_id=sample_id,
            call_type="ROLE_V3",
            prompt=build_role_v3_prompt(question, role_cards),
            max_tokens=64,
            schema=role_schema,
        )
        role = derive_role_v3_record(
            role_call["text"], role_schema, role_registry,
        )
        variant_rows = []
        shared: Dict[str, Any] = {}
        retrieval_record: Dict[str, Any] | None = None
        if role["supported"]:
            try:
                retrieval_record = _retrieval(
                    role=role,
                    graph=graph,
                    table=table,
                    question=question,
                    encoder=encoder,
                    parent_sha=parent_sha,
                    embedding_sha=embedding_sha,
                )
            except ValueError as exc:
                retrieval_record = {"error": str(exc)}
            c1_view = build_v3_arm_view(
                "C1_ROLE_ONLY", question, graph, table, role, None, matrix,
                output_schema=role_schema, canonical_registry=role_registry,
            )
            shared["C1_COMPLETE"] = {
                "build": c1_view,
                "result": _planner_call(
                    output=output,
                    split=split,
                    generator=generator,
                    sample_id=sample_id,
                    call_type="PLANNER_C1_COMPLETE",
                    view=c1_view.view,
                    matrix=matrix,
                ),
            }
            retrieval_payload = (
                retrieval_record.get("retrieval")
                if isinstance(retrieval_record, Mapping)
                else None
            )
            if isinstance(retrieval_payload, Mapping):
                legacy_view = build_v3_arm_view(
                    "C2_ROLE_RETRIEVAL", question, graph, table, role,
                    retrieval_payload, matrix, output_schema=role_schema,
                    canonical_registry=role_registry,
                )
                complete_view = build_complete_domain_c2_view(
                    question, graph, table, role, retrieval_payload, matrix,
                    output_schema=role_schema,
                    canonical_registry=role_registry,
                )
                for call_type, built in (
                    ("C2_LEGACY", legacy_view),
                    ("C2_COMPLETE", complete_view),
                ):
                    shared[call_type] = {
                        "build": built,
                        "result": _planner_call(
                            output=output,
                            split=split,
                            generator=generator,
                            sample_id=sample_id,
                            call_type=f"PLANNER_{call_type}",
                            view=built.view,
                            matrix=matrix,
                        ),
                    }

        for variant_id in VARIANT_IDS:
            failure_reasons = []
            bundle = None
            vault: list[Dict[str, Any]] = []
            closure = None
            lineage: Sequence[Mapping[str, Any]] = ()
            compilation = None
            if not role["supported"]:
                failure_reasons.append("role_unsupported")
            elif variant_id == "V0_LEGACY_C2_HARD_FILTER":
                record = shared.get("C2_LEGACY")
                if record is None:
                    failure_reasons.append("retrieval_unavailable")
                else:
                    compilation = record["result"][0]
                    if compilation.ok:
                        closure = close_compiled_payload(
                            compilation, graph, matrix,
                        )
                    else:
                        failure_reasons.extend(compilation.errors)
            elif variant_id == "V1_C2_COMPLETE_DOMAIN":
                record = shared.get("C2_COMPLETE")
                if record is None:
                    failure_reasons.append("retrieval_unavailable")
                else:
                    compilation = record["result"][0]
                    if compilation.ok:
                        closure = close_compiled_payload(
                            compilation, graph, matrix,
                        )
                    else:
                        failure_reasons.extend(compilation.errors)
            else:
                inputs = []
                for call_type, arm in (
                    ("C1_COMPLETE", "C1_ROLE_ONLY"),
                    ("C2_COMPLETE", "C2_ROLE_RETRIEVAL"),
                ):
                    record = shared.get(call_type)
                    if record is None:
                        continue
                    item_compilation = record["result"][0]
                    if not item_compilation.ok:
                        failure_reasons.extend(
                            f"{call_type}:{error}"
                            for error in item_compilation.errors
                        )
                        continue
                    inputs.append(ProgramUnionInput(
                        arm=arm,
                        sample_id=sample_id,
                        table_id=table_id,
                        graph=graph_record,
                        role_record=role,
                        planner_view=record["build"].view,
                        compilation=item_compilation,
                    ))
                if inputs:
                    union = union_exact_typed_programs(
                        inputs,
                        full_domain_view=shared["C1_COMPLETE"]["build"].view,
                        capability_matrix=matrix,
                    )
                    compilation = union.compilation
                    lineage = union.lineage
                    closure = close_compiled_payload(
                        compilation, graph, matrix,
                    )
                else:
                    failure_reasons.append("no_valid_union_input")

            policies = {
                "B0_KEEP": {
                    "action": "KEEP_B0",
                    "selected_answer": b0_answer,
                },
                "REGISTRY_DETERMINISTIC": {
                    "action": "KEEP_B0",
                    "selected_answer": b0_answer,
                    "failure_reasons": list(failure_reasons),
                },
                "CERA_VALIDATED": {
                    "action": "KEEP_B0",
                    "selected_answer": b0_answer,
                    "status": "PENDING_PAIRED_DECISION_STAGE",
                },
            }
            support_record: Dict[str, Any] = {
                "state": "UNOBSERVED" if failure_reasons else "NoSupport",
                "valid": False if failure_reasons else True,
                "failure_reasons": list(failure_reasons),
            }
            if closure is not None:
                bundle, vault = _bundle_variant(
                    sample_id=sample_id,
                    table_id=table_id,
                    variant_id=variant_id,
                    role=role,
                    b0_answer=b0_answer,
                    closure=closure,
                    lineage=lineage,
                )
                arm = variant_artifact_arm(variant_id)
                support = build_registry_support_state(
                    sample_id=sample_id,
                    table_id=table_id,
                    variant_id=variant_id,
                    selected_arms=[arm],
                    role_record=role,
                    role_output_schema=role_schema,
                    role_registry=role_registry,
                    capability_matrix=matrix,
                    b0_answer=b0_answer,
                    executed_derivations=closure.executable_derivations,
                    raw_groundings_v3=bundle.raw_groundings,
                    raw_derivations=bundle.raw_derivations,
                    registry_entries=bundle.registry_entries,
                    answer_vault_records=vault,
                )
                selection = select_registry_policy(support)
                selected = materialize_registry_selection(
                    selection, support, vault, b0_answer,
                )
                policies["REGISTRY_DETERMINISTIC"] = {
                    **asdict(selection),
                    "selected_answer": selected.answer,
                    "materialization": asdict(selected),
                }
                support_record = asdict(support)
                partition = partition_support(
                    closure,
                    initial_proposal_answer=b0_answer,
                )
                basis = build_sample_fixed_role_intervention_basis(
                    closure.executable_derivations,
                    graph,
                )
                behavior_classes = build_basis_relative_behavior_classes(
                    closure.executable_derivations,
                    graph,
                    basis,
                )
                contrast = build_compact_behavioral_contrast_v3(
                    derivations=closure.executable_derivations,
                    behavior_classes=behavior_classes,
                    basis=basis,
                    original_answer=b0_answer,
                    query_semantics=role_v3_to_planner_query_contract(role),
                )
                eligibility = assess_decision_eligibility(
                    role_id=role["role_id"],
                    decision_active_role_ids=decision_active_roles,
                    support_partition=partition,
                    compact_contrast=contrast,
                    executed_derivations=closure.executable_derivations,
                )
                cera_policy: Dict[str, Any] = {
                    "action": "KEEP_B0",
                    "selected_answer": b0_answer,
                    "eligible": eligibility.eligible,
                    "failure_reasons": list(eligibility.failure_reasons),
                }
                if support.valid and eligibility.eligible:
                    packet = {
                        "query_contract": role_v3_to_planner_query_contract(role),
                        "compact_behavioral_contrast_v3": contrast.to_dict(),
                        "metadata": {
                            "sample_id": sample_id,
                            "table_id": table_id,
                            "variant_id": variant_id,
                            "role_record_sha256": canonical_json_hash(role),
                        },
                    }
                    prompt = build_cera_prompt(
                        packet,
                        template_version=CERA_V3_TEMPLATE_VERSION,
                    )
                    cera_call = _cached_call(
                        output=output,
                        split=split,
                        generator=generator,
                        sample_id=sample_id,
                        call_type=f"CERA_{variant_id}",
                        prompt=prompt,
                        max_tokens=512,
                    )
                    validator = validate_cera_output_v3(
                        cera_call["text"],
                        packet,
                    )
                    created_at = completion_runtime.now()
                    resolution = reconcile_cera_decision(
                        eligibility=eligibility,
                        raw_output=cera_call["text"],
                        validator=validator,
                        compact_contrast=contrast,
                        executed_derivations=closure.executable_derivations,
                        raw_derivation_records=bundle.raw_derivations,
                        registry_entries=bundle.registry_entries,
                        b0_answer=b0_answer,
                        sample_id=sample_id,
                        decision_id=f"DEV-{variant_id}-{sample_id}",
                        validator_record_id=f"DEV-VAL-{variant_id}-{sample_id}",
                        created_at=created_at,
                        artifact_arms=(arm,),
                    )
                    final = materialize_selected_final(
                        resolution,
                        b0_answer=b0_answer,
                        materialized_at=completion_runtime.now(),
                    )
                    cera_policy = {
                        "action": resolution.decision_record["action"],
                        "selected_answer": final.answer,
                        "eligible": True,
                        "failure_reasons": list(resolution.failure_reasons),
                        "decision_record": resolution.decision_record,
                        "validator_record": resolution.validator_record,
                        "reconciliation_record": resolution.reconciliation_record,
                        "selected_final_record": final.record,
                    }
                policies["CERA_VALIDATED"] = cera_policy
            for policy in policies.values():
                if gold is not None:
                    policy["correct"] = official_match(
                        "hitab", policy["selected_answer"], gold,
                    )
                policy["changed"] = (
                    completion_runtime.active_answer_hash(
                        policy["selected_answer"]
                    )
                    != completion_runtime.active_answer_hash(b0_answer)
                )
            variant_rows.append({
                "variant_id": variant_id,
                "artifact_arm": variant_artifact_arm(variant_id),
                "failure_reasons": failure_reasons,
                "planner_payload_sha256": (
                    compilation.canonical_payload_sha256
                    if compilation is not None and compilation.ok
                    else ""
                ),
                "lineage": list(lineage),
                "closure": closure.to_dict() if closure is not None else None,
                "raw_groundings": list(bundle.raw_groundings) if bundle else [],
                "raw_derivations": list(bundle.raw_derivations) if bundle else [],
                "registry_entries": list(bundle.registry_entries) if bundle else [],
                "answer_vault": vault,
                "support": support_record,
                "policies": policies,
            })
        sample = {
            "schema_version": "certa_final_development_sample_master_v1",
            "sample_id": sample_id,
            "dataset": runtime_row["dataset"],
            "table_id": table_id,
            "question_sha256": canonical_json_hash(question),
            "table_artifact_sha256": table_sha,
            "graph_sha256": canonical_json_hash(graph_record),
            "b0_answer": b0_answer,
            "role": role,
            "role_record_sha256": canonical_json_hash(role),
            "retrieval": retrieval_record,
            "variants": variant_rows,
        }
        if gold is not None:
            sample["gold_answers"] = gold
            sample["b0_correct"] = official_match(
                "hitab", b0_answer, gold,
            )
        _write_json(
            output / split / "samples" / f"{sample_id}.json",
            sample,
        )
        sample_rows.append(sample)
        _write_jsonl(
            output / split / (
                "SAMPLE_MASTER.jsonl"
                if split == "development"
                else "BLIND_SAMPLE_MASTER.jsonl"
            ),
            sample_rows,
        )
    return {
        "status": "PASS",
        "sample_count": len(sample_rows),
        "sample_master": str(
            output / split / (
                "SAMPLE_MASTER.jsonl"
                if split == "development"
                else "BLIND_SAMPLE_MASTER.jsonl"
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--development-root", type=Path, default=DEFAULT_DEVELOPMENT,
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--split",
        choices=("development", "validation", "holdout"),
        default="development",
    )
    parser.add_argument("--runtime", type=Path)
    args = parser.parse_args()
    result = run_development(
        output=args.output.resolve(),
        dataset_root=args.dataset_root.resolve(),
        development_root=args.development_root.resolve(),
        limit=args.limit,
        split=args.split,
        runtime_path=args.runtime.resolve() if args.runtime else None,
    )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
