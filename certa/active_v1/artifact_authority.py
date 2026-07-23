"""Raw constructor-artifact serialization authority for CERTA Active V1.

Provenance is authoritative only through the ``provenance_ids`` carried by the
immutable raw-derivation and registry schemas.  This module does not select an
answer or compute any scientific Gate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from certa.active_v1.answer_authority import active_answer_hash
from certa.derivations.answer_equivalence import inference_answers_equivalent
from certa.derivations.schema import ExecutableDerivation, to_jsonable
from certa.grounding.plan_closure import GroundedAssignment, PlanClosure
from certa.operations.contracts import get_operation_signature
from certa.reproducibility.canonical_json import canonical_json, canonical_json_hash


ACTIVE_ARMS = (
    "C0_SCHEMA_ONLY",
    "C1_ROLE_ONLY",
    "C2_ROLE_RETRIEVAL",
)


@dataclass(frozen=True)
class ArtifactContext:
    sample_id: str
    table_id: str
    arm: str
    role_id: str
    fixture_only: bool = False
    role_record_sha256: str = ""

    def __post_init__(self) -> None:
        for field in ("sample_id", "table_id", "role_id"):
            if not str(getattr(self, field) or "").strip():
                raise ValueError(f"artifact_context_{field}_empty")
        if self.arm not in ACTIVE_ARMS:
            raise ValueError(f"artifact_context_arm_invalid:{self.arm}")
        if not isinstance(self.fixture_only, bool):
            raise ValueError("artifact_context_fixture_only_not_boolean")

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class RawArtifactBundle:
    context: ArtifactContext
    raw_groundings: Tuple[Dict[str, Any], ...]
    raw_derivations: Tuple[Dict[str, Any], ...]
    registry_entries: Tuple[Dict[str, Any], ...]
    excluded_registry_derivation_ids: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(self)


def _binding_id(context: ArtifactContext, assignment: GroundedAssignment) -> str:
    identity = {
        "sample_id": context.sample_id,
        "table_id": context.table_id,
        "arm": context.arm,
        "role_id": context.role_id,
        "plan_id": assignment.plan_id,
        "assignment_id": assignment.assignment_id,
        "assignment_key": assignment.assignment_key,
    }
    return f"B-{canonical_json_hash(identity, 24)}"


def _valid_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _binding_id_v3(
    context: ArtifactContext,
    assignment: GroundedAssignment,
    role_bindings_sha256: str,
) -> str:
    identity = {
        "sample_id": context.sample_id,
        "table_id": context.table_id,
        "arm": context.arm,
        "role_record_sha256": context.role_record_sha256,
        "plan_id": assignment.plan_id,
        "assignment_id": assignment.assignment_id,
        "assignment_key": assignment.assignment_key,
        "role_bindings_sha256": role_bindings_sha256,
    }
    return f"B-{canonical_json_hash(identity, 24)}"


def recompute_binding_id_v3(
    grounding_record: Mapping[str, Any],
    hypothesis: Mapping[str, Any],
) -> str:
    """Recompute a V3 binding from the complete public identity preimage."""
    identity = {
        "sample_id": grounding_record.get("sample_id"),
        "table_id": grounding_record.get("table_id"),
        "arm": grounding_record.get("arm"),
        "role_record_sha256": grounding_record.get("role_record_sha256"),
        "plan_id": grounding_record.get("plan_id"),
        "assignment_id": hypothesis.get("assignment_id"),
        "assignment_key": hypothesis.get("assignment_key"),
        "role_bindings_sha256": hypothesis.get("role_bindings_sha256"),
    }
    return f"B-{canonical_json_hash(identity, 24)}"


def _provenance_ids(derivation: ExecutableDerivation) -> list[str]:
    values = {
        str(value)
        for value in (*derivation.evidence_ids, *derivation.operand_node_ids)
        if str(value)
    }
    for source, _, target in derivation.required_edge_triples:
        if str(source):
            values.add(str(source))
        if str(target):
            values.add(str(target))
    return sorted(values)


def _assert_program_identity(
    assignment: GroundedAssignment,
    derivation: ExecutableDerivation,
) -> None:
    try:
        program = json.loads(derivation.executable_program)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("executable_program_not_json") from exc
    if canonical_json(program) != derivation.executable_program:
        raise ValueError("canonical_program_round_trip_mismatch")
    expected = f"CP-{canonical_json_hash(program, 24)}"
    metadata_id = str(derivation.operation_metadata.get("canonical_program_id") or "")
    if assignment.canonical_program_id != expected or metadata_id != expected:
        raise ValueError("canonical_program_id_mismatch")


def _assert_derivation_identity(
    assignment: GroundedAssignment,
    derivation: ExecutableDerivation,
) -> None:
    checks = {
        "derivation_id": assignment.derivation_id == derivation.derivation_id,
        "operation_family": assignment.operation_family == derivation.operation_family,
        "signature_id": assignment.signature_id == derivation.typed_signature,
        "projection": assignment.projection_operator == derivation.projection_operator,
        "answer_domain": assignment.answer_domain == derivation.output_domain,
        "operand_node_ids": tuple(assignment.matched_cell_ids)
        == tuple(derivation.operand_node_ids),
        "projected_answer": assignment.projected_answer == derivation.projected_answer,
    }
    failed = sorted(field for field, matches in checks.items() if not matches)
    if failed:
        raise ValueError(f"derivation_identity_mismatch:{','.join(failed)}")
    if assignment.execution_outcome != "EXECUTED":
        raise ValueError("derivation_not_executed")
    if assignment.projection_outcome != "PROJECTED":
        raise ValueError("derivation_projection_not_valid")
    if derivation.availability != "available":
        raise ValueError("derivation_not_available")
    _assert_program_identity(assignment, derivation)


def _assert_round_trip(record: Mapping[str, Any], record_type: str) -> None:
    encoded = canonical_json(record)
    if json.loads(encoded) != dict(record):
        raise ValueError(f"{record_type}_canonical_round_trip_mismatch")


def _required_roles(assignments: Tuple[GroundedAssignment, ...]) -> list[str]:
    signature_ids = sorted({item.signature_id for item in assignments if item.signature_id})
    if len(signature_ids) > 1:
        raise ValueError("grounding_plan_signature_mismatch")
    if not signature_ids:
        return []
    signature = get_operation_signature(signature_ids[0])
    if signature is None:
        raise ValueError(f"grounding_signature_unknown:{signature_ids[0]}")
    return list(signature.required_role_names)


def _grounding_records(
    closure: PlanClosure,
    context: ArtifactContext,
) -> Tuple[Tuple[Dict[str, Any], ...], Dict[str, str]]:
    by_plan: Dict[str, list[GroundedAssignment]] = {}
    binding_by_derivation: Dict[str, str] = {}
    for assignment in closure.assignments:
        if not assignment.plan_id:
            raise ValueError("grounding_plan_id_empty")
        by_plan.setdefault(assignment.plan_id, []).append(assignment)

    records = []
    for plan_id in sorted(by_plan):
        assignments = tuple(by_plan[plan_id])
        candidates = []
        valid_ids = []
        for assignment in assignments:
            binding_id = _binding_id(context, assignment)
            valid = assignment.resolution_state == "UNIQUE" and bool(
                assignment.matched_cell_ids
            )
            candidates.append({
                "binding_id": binding_id,
                "operand_node_ids": list(assignment.matched_cell_ids),
                "valid": valid,
            })
            if valid:
                valid_ids.append(binding_id)
            if assignment.derivation_id:
                if assignment.derivation_id in binding_by_derivation:
                    raise ValueError(
                        f"duplicate_grounding_derivation_id:{assignment.derivation_id}"
                    )
                binding_by_derivation[assignment.derivation_id] = binding_id
        record = {
            "schema_version": "certa_active_grounding_record_v2",
            "fixture_only": context.fixture_only,
            "sample_id": context.sample_id,
            "arm": context.arm,
            "plan_id": plan_id,
            "required_operand_roles": _required_roles(assignments),
            "grounding_candidates": candidates,
            "selected_binding_id": valid_ids[0] if len(valid_ids) == 1 else None,
            "first_match_used": False,
        }
        _assert_round_trip(record, "raw_grounding")
        records.append(record)
    return tuple(records), binding_by_derivation


def _resolution_state_v3(assignment: GroundedAssignment) -> str:
    if assignment.resolution_state == "UNIQUE" and assignment.matched_cell_ids:
        return "EXACT"
    if assignment.resolution_state == "AMBIGUOUS":
        return "AMBIGUOUS"
    return "UNRESOLVED"


def validate_grounding_record_v3(record: Mapping[str, Any]) -> None:
    """Fail closed on noncanonical or internally inconsistent V3 authority."""
    if record.get("first_match_used") is not False:
        raise ValueError("first_match_resolution")
    if "selected_binding_id" in record:
        raise ValueError("legacy_selected_binding_forbidden")
    if not _valid_sha256(record.get("role_record_sha256")):
        raise ValueError("role_record_sha256_invalid")
    hypotheses = list(record.get("grounding_hypotheses") or ())
    authorized = list(record.get("authorized_binding_ids") or ())
    rejected = list(record.get("rejected_binding_ids") or ())
    if authorized != sorted(set(authorized)) or rejected != sorted(set(rejected)):
        raise ValueError("binding_authority_not_sorted_unique")
    if set(authorized) & set(rejected):
        raise ValueError("binding_authority_overlap")
    binding_ids = [str(item.get("binding_id") or "") for item in hypotheses]
    if len(binding_ids) != len(set(binding_ids)):
        raise ValueError("duplicate_grounding_binding_id")
    if set(binding_ids) != set(authorized) | set(rejected):
        raise ValueError("binding_authority_not_exhaustive")
    if hypotheses != sorted(
        hypotheses,
        key=lambda item: (
            str(item.get("assignment_key") or ""),
            str(item.get("assignment_id") or ""),
            str(item.get("binding_id") or ""),
        ),
    ):
        raise ValueError("grounding_hypotheses_not_canonical")
    seen_assignments = set()
    seen_derivations = set()
    for hypothesis in hypotheses:
        assignment_identity = (
            hypothesis.get("assignment_id"),
            hypothesis.get("assignment_key"),
        )
        if assignment_identity in seen_assignments:
            raise ValueError("duplicate_grounding_assignment")
        seen_assignments.add(assignment_identity)
        role_bindings_sha256 = canonical_json_hash(hypothesis.get("role_bindings"))
        if hypothesis.get("role_bindings_sha256") != role_bindings_sha256:
            raise ValueError("role_bindings_sha256_mismatch")
        if hypothesis.get("binding_id") != recompute_binding_id_v3(record, hypothesis):
            raise ValueError("grounding_binding_id_mismatch")
        state = hypothesis.get("resolution_state")
        valid = hypothesis.get("grounding_valid") is True
        failures = list(hypothesis.get("failure_reasons") or ())
        if failures != sorted(set(failures)):
            raise ValueError("grounding_failure_reasons_not_canonical")
        if state != "EXACT" and valid:
            if state == "AMBIGUOUS":
                raise ValueError("ambiguous_assignment_authorized")
            raise ValueError("unresolved_assignment_authorized")
        if valid and (
            not hypothesis.get("operand_node_ids")
            or "resource_incomplete" in failures
        ):
            raise ValueError("invalid_exact_assignment_authorized")
        if valid != (hypothesis.get("binding_id") in authorized):
            raise ValueError("grounding_valid_authority_mismatch")
        derivation_id = str(hypothesis.get("derivation_id") or "")
        if derivation_id and not valid:
            raise ValueError("rejected_assignment_has_derivation_id")
        if derivation_id and not hypothesis.get("canonical_program_id"):
            raise ValueError("grounding_derivation_program_id_empty")
        if derivation_id:
            if derivation_id in seen_derivations:
                raise ValueError("duplicate_grounding_derivation_id")
            seen_derivations.add(derivation_id)
    expected_counts = {
        "exact_hypothesis_count": sum(
            item.get("resolution_state") == "EXACT" for item in hypotheses
        ),
        "ambiguous_hypothesis_count": sum(
            item.get("resolution_state") == "AMBIGUOUS" for item in hypotheses
        ),
        "unresolved_hypothesis_count": sum(
            item.get("resolution_state") == "UNRESOLVED" for item in hypotheses
        ),
        "resource_incomplete_hypothesis_count": sum(
            "resource_incomplete" in item.get("failure_reasons", ())
            for item in hypotheses
        ),
    }
    mismatches = sorted(
        key for key, value in expected_counts.items() if record.get(key) != value
    )
    if mismatches:
        raise ValueError(f"grounding_count_mismatch:{','.join(mismatches)}")


def _grounding_records_v3(
    closure: PlanClosure,
    context: ArtifactContext,
) -> Tuple[Tuple[Dict[str, Any], ...], Dict[str, str]]:
    if not _valid_sha256(context.role_record_sha256):
        raise ValueError("artifact_context_role_record_sha256_invalid")
    by_plan: Dict[str, list[GroundedAssignment]] = {}
    for assignment in closure.assignments:
        if not assignment.plan_id:
            raise ValueError("grounding_plan_id_empty")
        by_plan.setdefault(assignment.plan_id, []).append(assignment)
    records = []
    binding_by_derivation: Dict[str, str] = {}
    for plan_id in sorted(by_plan):
        assignments = tuple(sorted(
            by_plan[plan_id],
            key=lambda item: (item.assignment_key, item.assignment_id),
        ))
        hypotheses = []
        authorized = []
        rejected = []
        for assignment in assignments:
            role_bindings = to_jsonable(assignment.role_bindings)
            role_bindings_sha256 = canonical_json_hash(role_bindings)
            binding_id = _binding_id_v3(
                context, assignment, role_bindings_sha256
            )
            state = _resolution_state_v3(assignment)
            resource_complete = closure.resource_complete and assignment.resource_complete
            valid = state == "EXACT" and resource_complete
            failure_reasons = {
                str(reason) for reason in assignment.failure_reasons if str(reason)
            }
            if not resource_complete:
                failure_reasons.add("resource_incomplete")
            if assignment.resolution_state == "UNIQUE" and not assignment.matched_cell_ids:
                failure_reasons.add("unique_resolution_without_operands")
            hypothesis = {
                "binding_id": binding_id,
                "assignment_id": assignment.assignment_id,
                "assignment_key": assignment.assignment_key,
                "role_bindings": role_bindings,
                "role_bindings_sha256": role_bindings_sha256,
                "operand_node_ids": list(assignment.matched_cell_ids),
                "resolution_state": state,
                "grounding_valid": valid,
                "derivation_id": assignment.derivation_id,
                "canonical_program_id": assignment.canonical_program_id,
                "failure_reasons": sorted(failure_reasons),
            }
            hypotheses.append(hypothesis)
            (authorized if valid else rejected).append(binding_id)
            if assignment.derivation_id:
                if not valid:
                    continue
                if assignment.derivation_id in binding_by_derivation:
                    raise ValueError(
                        f"duplicate_grounding_derivation_id:{assignment.derivation_id}"
                    )
                binding_by_derivation[assignment.derivation_id] = binding_id
        record = {
            "schema_version": "certa_active_grounding_record_v3",
            "fixture_only": context.fixture_only,
            "sample_id": context.sample_id,
            "table_id": context.table_id,
            "arm": context.arm,
            "role_record_sha256": context.role_record_sha256,
            "plan_id": plan_id,
            "required_operand_roles": _required_roles(assignments),
            "grounding_hypotheses": hypotheses,
            "authorized_binding_ids": sorted(authorized),
            "rejected_binding_ids": sorted(rejected),
            "exact_hypothesis_count": sum(
                item["resolution_state"] == "EXACT" for item in hypotheses
            ),
            "ambiguous_hypothesis_count": sum(
                item["resolution_state"] == "AMBIGUOUS" for item in hypotheses
            ),
            "unresolved_hypothesis_count": sum(
                item["resolution_state"] == "UNRESOLVED" for item in hypotheses
            ),
            "resource_incomplete_hypothesis_count": sum(
                "resource_incomplete" in item["failure_reasons"]
                for item in hypotheses
            ),
            "first_match_used": False,
        }
        validate_grounding_record_v3(record)
        _assert_round_trip(record, "raw_grounding_v3")
        records.append(record)
    return tuple(records), binding_by_derivation


def _derivation_record(
    assignment: GroundedAssignment,
    derivation: ExecutableDerivation,
    *,
    binding_id: str,
    context: ArtifactContext,
    initial_answer: Any,
) -> Dict[str, Any]:
    _assert_derivation_identity(assignment, derivation)
    answer_hash = active_answer_hash(derivation.projected_answer)
    record = {
        "schema_version": "certa_active_derivation_record_v2",
        "fixture_only": context.fixture_only,
        "sample_id": context.sample_id,
        "arm": context.arm,
        "derivation_id": derivation.derivation_id,
        "plan_id": assignment.plan_id,
        "binding_id": binding_id,
        "side": (
            "ORIGINAL"
            if inference_answers_equivalent(
                derivation.projected_answer, initial_answer
            )
            else "ALTERNATIVE"
        ),
        "signature_id": derivation.typed_signature,
        "answer_role": derivation.output_domain,
        "projection": derivation.projection_operator,
        "canonical_program_id": assignment.canonical_program_id,
        "answer_class_id": f"AC-{answer_hash[:24]}",
        "projected_answer_hash": answer_hash,
        "operand_node_ids": list(derivation.operand_node_ids),
        "provenance_ids": _provenance_ids(derivation),
        "execution_status": "EXECUTED",
        "projection_status": "VALID",
    }
    _assert_round_trip(record, "raw_derivation")
    return record


def _registry_entry(derivation: Mapping[str, Any]) -> Dict[str, Any]:
    record = {
        "schema_version": "certa_active_registry_entry_v2",
        "fixture_only": derivation["fixture_only"],
        "sample_id": derivation["sample_id"],
        "arm": derivation["arm"],
        "derivation_id": derivation["derivation_id"],
        "side": derivation["side"],
        "canonical_program_id": derivation["canonical_program_id"],
        "answer_class_id": derivation["answer_class_id"],
        "answer_hash": derivation["projected_answer_hash"],
        "provenance_ids": list(derivation["provenance_ids"]),
    }
    record["registry_entry_id"] = f"REG-{canonical_json_hash(record, 24)}"
    _assert_round_trip(record, "registry_entry")
    return record


def reconcile_registry_entry(
    registry: Mapping[str, Any],
    derivation: Mapping[str, Any],
) -> None:
    """Fail closed unless a registry entry is the exact executed raw derivation."""
    if derivation.get("execution_status") != "EXECUTED":
        raise ValueError("registry_derivation_not_executed")
    if derivation.get("projection_status") != "VALID":
        raise ValueError("registry_derivation_projection_not_valid")
    if not derivation.get("provenance_ids"):
        raise ValueError("registry_derivation_provenance_empty")
    expected_fields = {
        "schema_version": "certa_active_registry_entry_v2",
        "fixture_only": derivation.get("fixture_only"),
        "sample_id": derivation.get("sample_id"),
        "arm": derivation.get("arm"),
        "derivation_id": derivation.get("derivation_id"),
        "side": derivation.get("side"),
        "canonical_program_id": derivation.get("canonical_program_id"),
        "answer_class_id": derivation.get("answer_class_id"),
        "answer_hash": derivation.get("projected_answer_hash"),
        "provenance_ids": derivation.get("provenance_ids"),
    }
    mismatches = sorted(
        field
        for field, expected in expected_fields.items()
        if registry.get(field) != expected
    )
    if mismatches:
        raise ValueError(f"registry_derivation_mismatch:{','.join(mismatches)}")
    expected_id = f"REG-{canonical_json_hash(expected_fields, 24)}"
    if registry.get("registry_entry_id") != expected_id:
        raise ValueError("registry_entry_id_mismatch")
    _assert_round_trip(registry, "registry_entry")


def serialize_plan_closure(
    closure: PlanClosure,
    *,
    context: ArtifactContext,
    initial_answer: Any,
) -> RawArtifactBundle:
    """Project raw closure outputs into immutable Pack record schemas."""
    raw_groundings, bindings = _grounding_records(closure, context)
    assignments = {
        item.derivation_id: item
        for item in closure.assignments
        if item.derivation_id
    }
    if len(assignments) != sum(bool(item.derivation_id) for item in closure.assignments):
        raise ValueError("duplicate_assignment_derivation_id")

    derivation_records = []
    registry_entries = []
    excluded = []
    seen_derivations = set()
    for derivation in closure.executable_derivations:
        if derivation.derivation_id in seen_derivations:
            raise ValueError(f"duplicate_executable_derivation_id:{derivation.derivation_id}")
        seen_derivations.add(derivation.derivation_id)
        assignment = assignments.get(derivation.derivation_id)
        binding_id = bindings.get(derivation.derivation_id)
        if assignment is None or binding_id is None:
            raise ValueError(
                f"executable_derivation_without_grounding:{derivation.derivation_id}"
            )
        record = _derivation_record(
            assignment,
            derivation,
            binding_id=binding_id,
            context=context,
            initial_answer=initial_answer,
        )
        derivation_records.append(record)
        if (
            closure.resource_complete
            and derivation.provenance_complete
            and record["provenance_ids"]
        ):
            registry = _registry_entry(record)
            reconcile_registry_entry(registry, record)
            registry_entries.append(registry)
        else:
            excluded.append(derivation.derivation_id)

    if set(assignments) != seen_derivations:
        missing = sorted(set(assignments) - seen_derivations)
        raise ValueError(f"executed_assignment_without_derivation:{','.join(missing)}")
    return RawArtifactBundle(
        context=context,
        raw_groundings=raw_groundings,
        raw_derivations=tuple(derivation_records),
        registry_entries=tuple(registry_entries),
        excluded_registry_derivation_ids=tuple(excluded),
    )


def serialize_plan_closure_v3(
    closure: PlanClosure,
    *,
    context: ArtifactContext,
    initial_answer: Any,
) -> RawArtifactBundle:
    """Serialize assignment-level grounding authority without selecting a hypothesis."""
    raw_groundings, bindings = _grounding_records_v3(closure, context)
    assignments = {
        item.derivation_id: item
        for item in closure.assignments
        if item.derivation_id
    }
    if len(assignments) != sum(bool(item.derivation_id) for item in closure.assignments):
        raise ValueError("duplicate_assignment_derivation_id")
    derivation_records = []
    registry_entries = []
    excluded = []
    seen_derivations = set()
    for derivation in sorted(
        closure.executable_derivations,
        key=lambda item: item.derivation_id,
    ):
        if derivation.derivation_id in seen_derivations:
            raise ValueError(f"duplicate_executable_derivation_id:{derivation.derivation_id}")
        seen_derivations.add(derivation.derivation_id)
        assignment = assignments.get(derivation.derivation_id)
        binding_id = bindings.get(derivation.derivation_id)
        if assignment is None or binding_id is None:
            raise ValueError(
                "executable_derivation_without_authorized_grounding:"
                f"{derivation.derivation_id}"
            )
        record = _derivation_record(
            assignment,
            derivation,
            binding_id=binding_id,
            context=context,
            initial_answer=initial_answer,
        )
        derivation_records.append(record)
        if (
            closure.resource_complete
            and assignment.resource_complete
            and derivation.provenance_complete
            and record["provenance_ids"]
        ):
            registry = _registry_entry(record)
            reconcile_registry_entry(registry, record)
            registry_entries.append(registry)
        else:
            excluded.append(derivation.derivation_id)
    if set(assignments) != seen_derivations:
        missing = sorted(set(assignments) - seen_derivations)
        raise ValueError(f"executed_assignment_without_derivation:{','.join(missing)}")
    return RawArtifactBundle(
        context=context,
        raw_groundings=raw_groundings,
        raw_derivations=tuple(derivation_records),
        registry_entries=tuple(sorted(
            registry_entries,
            key=lambda item: (
                item["sample_id"], item["arm"], item["derivation_id"]
            ),
        )),
        excluded_registry_derivation_ids=tuple(sorted(excluded)),
    )
