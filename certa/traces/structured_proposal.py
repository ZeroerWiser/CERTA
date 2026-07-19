"""Same-call answer and public typed-state contract for CERTA finalization."""

from __future__ import annotations

from collections import namedtuple
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import time
from types import SimpleNamespace
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import jsonschema

from graph_builder import GraphNode, HCEG, NodeType

from certa.backends.openai_compatible import ResponseContractError
from certa.derivations.project import (
    ProjectionStatus,
    execute_typed_projection_from_nodes,
)
from certa.hcer.experiment import typed_answer_equivalence
from certa.operations.contracts import (
    OPERATION_SIGNATURES,
    OperationSignatureVariant,
    get_operation_signature,
)
from certa.repair.evidence_dsl import parse_number
from certa.reproducibility.canonical_json import canonical_json_hash

STRUCTURED_PROPOSAL_VERSION = "certa_operation_specific_semantic_actions_v2"
STRUCTURED_PROPOSAL_SCHEMA_NAME = "certa_operation_specific_semantic_actions"
_STEP_LIMIT = 12
_FORBIDDEN_KEYS = set(
    "claimed_value confidence correct correctness decision evidence_ids final_answer "
    "gold gold_answer input_step_ids patch patch_id private_reasoning projection "
    "reasoning role_bindings score step_id".split()
)

class FaultType(str, Enum):
    ROLE = "ROLE"
    OPERATION = "OPERATION"
    PROJECTION = "PROJECTION"
    DEPENDENCY = "DEPENDENCY"
    VALUE = "VALUE"

class AnswerTraceConsistency(str, Enum):
    CONSISTENT_EXECUTABLE = "CONSISTENT_EXECUTABLE"
    INCONSISTENT_EXECUTABLE = "INCONSISTENT_EXECUTABLE"
    NON_EXECUTABLE = "NON_EXECUTABLE"
    RESOURCE_INCOMPLETE = "RESOURCE_INCOMPLETE"

@dataclass(frozen=True)
class StructuredProposalValidation:
    ok: bool
    errors: Tuple[str, ...] = ()
    wire_schema_valid: bool = False
    local_shape_valid: bool = False
ActionShapeContract = namedtuple("ActionShapeContract",
    "signature_id minimum_arity maximum_arity relation_required allowed_relations")
@dataclass(frozen=True)
class ReferenceEntry:
    handle: str
    kind: str
    surface: str
    typed_domain: str
    graph_node_ids: Tuple[str, ...]
    row_header_path: Tuple[str, ...]
    column_header_path: Tuple[str, ...]
    provenance: Tuple[Tuple[str, str, str], ...]

@dataclass(frozen=True)
class ReferenceCatalog:
    entries: Tuple[ReferenceEntry, ...]

    @property
    def handles(self) -> Tuple[str, ...]:
        return tuple(entry.handle for entry in self.entries)

    def resolve(self, handle: str) -> ReferenceEntry:
        try:
            return next(item for item in self.entries if item.handle == handle)
        except StopIteration as error:
            raise KeyError(f"unknown_table_ref:{handle}") from error

    def handle_for_node(self, node_id: str) -> str:
        try:
            return next(
                item.handle for item in self.entries if node_id in item.graph_node_ids
            )
        except StopIteration as error:
            raise KeyError(f"uncataloged_graph_node:{node_id}") from error

@dataclass(frozen=True)
class PublicFirstVerifiableFault:
    step_id: str
    stage: str
    validation_state: str
    fault_type: FaultType
    reasons: Tuple[str, ...]
    resource_complete: bool

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(asdict(self))

@dataclass(frozen=True)
class StepResult:
    step_id: str
    value: str
    output_domain: str
    source_action_id: int
    source_table_refs: Tuple[str, ...]
    provenance: Tuple[Tuple[str, str, str], ...]
    execution_status: str

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(asdict(self))

@dataclass(frozen=True)
class CompiledPublicStep:
    step_id: str
    action_index: int
    action: Mapping[str, Any]
    op: str
    input_step_ids: Tuple[str, ...]
    evidence_ids: Tuple[str, ...]
    role_bindings: Tuple[Tuple[str, Any], ...]
    projection: str
    output_domain: str
    executed_value: str
    dependency_descendants: Tuple[str, ...]
    validation_state: str
    fault_type: Optional[FaultType]
    failure_reasons: Tuple[str, ...]
    resource_complete: bool
    step_result: StepResult
    provenance: Tuple[Tuple[str, str, str], ...]

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(asdict(self))

@dataclass(frozen=True)
class CompiledProposalTrace:
    answer: str
    actions: Tuple[Mapping[str, Any], ...]
    catalog: ReferenceCatalog
    steps: Tuple[CompiledPublicStep, ...]
    first_verifiable_fault: Optional[PublicFirstVerifiableFault]
    answer_trace_consistency: AnswerTraceConsistency
    answer_claim_consistent: bool
    validation_ok: bool
    record_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        payload = _jsonable(asdict(self))
        payload.pop("catalog")
        payload["structured_proposal_version"] = STRUCTURED_PROPOSAL_VERSION
        payload["reference_catalog_sha256"] = canonical_json_hash(
            [_jsonable(entry.__dict__) for entry in self.catalog.entries]
        )
        return payload

def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(child) for child in value]
    if isinstance(value, Enum):
        return value.value
    return value

def _entry_domain(node: Any) -> str:
    if getattr(node, "numeric_value", None) is not None:
        return "SCALAR"
    text = str(getattr(node, "text", "") or "").strip().lower()
    return "BOOLEAN" if text in {"true", "false"} else "ENTITY"

def build_reference_catalog(graph: HCEG) -> ReferenceCatalog:
    nodes = sorted(
        (node for node in graph.nodes.values()
         if node.node_type in {NodeType.CELL, NodeType.HEADER, NodeType.AGGREGATOR}
         and str(node.text or "").strip()),
        key=lambda node: (node.row, node.col, node.node_type.value, node.node_id),
    )
    entries = []
    for index, node in enumerate(nodes):
        relevant = (*graph._adj.get(node.node_id, ()), *graph._rev_adj.get(node.node_id, ()))
        path = lambda kind: tuple(sorted({
            edge.target for edge in relevant
            if edge.source == node.node_id and edge.edge_type.value == kind
        }))
        row_path, column_path = path("row_path"), path("col_path")
        domain = _entry_domain(node)
        kind = (
            "BOOLEAN" if domain == "BOOLEAN" else
            "ENTITY" if node.node_type == NodeType.HEADER else
            "ENTITY_VALUE" if row_path or column_path else "VALUE"
        )
        entries.append(ReferenceEntry(
            f"R{index}", kind, str(node.text).strip(), domain,
            (str(node.node_id),), row_path, column_path,
            provenance=tuple(sorted({
                (str(edge.source), str(edge.target), edge.edge_type.value)
                for edge in relevant
                if edge.edge_type.value in
                {"row_path", "col_path", "value_under_header", "part_of"}
            })),
        ))
    return ReferenceCatalog(tuple(entries))

def action_shape_contract(signature: OperationSignatureVariant) -> ActionShapeContract:
    family = signature.operation_family
    arity = (1, 1) if family == "LOOKUP" else (
        (2, 2) if family in {"DIFF", "RATIO", "PAIR_COMPARE"} else (1, None)
    )
    relations = tuple(signature.comparison_polarities)
    return ActionShapeContract(signature.signature_id, *arity, bool(relations), relations)
def validate_action_shape(
    action: Mapping[str, Any], signature: OperationSignatureVariant, *, step_index: int
) -> Tuple[str, ...]:
    contract = action_shape_contract(signature)
    args, relation, errors = action.get("args"), action.get("relation"), []
    if isinstance(args, list) and (len(args) < contract.minimum_arity or (
        contract.maximum_arity is not None and len(args) > contract.maximum_arity
    )):
        errors.append(f"operation_arity:S{step_index}:{contract.signature_id}")
    if contract.relation_required and relation is None:
        errors.append(f"missing_comparison_relation:S{step_index}")
    elif contract.relation_required and relation not in contract.allowed_relations:
        errors.append(f"invalid_comparison_relation:S{step_index}")
    elif not contract.relation_required and relation is not None:
        errors.append(f"unexpected_comparison_relation:S{step_index}")
    return tuple(errors)
def _operand_schema(handles: Sequence[str]) -> Dict[str, Any]:
    return {"oneOf": [
        {"type": "object", "additionalProperties": False,
         "required": ["kind", "ref"], "properties": {
             "kind": {"const": "TABLE_REF"},
             "ref": {"type": "string", "enum": list(handles)}}},
        {"type": "object", "additionalProperties": False,
         "required": ["kind", "step"], "properties": {
             "kind": {"const": "STEP_RESULT"},
             "step": {"type": "integer", "minimum": 0}}},
    ]}
def build_structured_proposal_schema(source: Any) -> Dict[str, Any]:
    catalog = source if isinstance(source, ReferenceCatalog) else build_reference_catalog(source)
    branches = []
    for signature in sorted(OPERATION_SIGNATURES.values(), key=lambda item: item.signature_id):
        contract = action_shape_contract(signature)
        args = {"type": "array", "items": {"$ref": "#/$defs/operand"},
                "minItems": contract.minimum_arity}
        if contract.maximum_arity is not None:
            args["maxItems"] = contract.maximum_arity
        properties = {"op": {"const": contract.signature_id}, "args": args}
        required = ["op", "args"]
        if contract.relation_required:
            properties["relation"] = {"type": "string",
                                      "enum": list(contract.allowed_relations)}
            required.append("relation")
        branches.append({"type": "object", "additionalProperties": False,
                         "required": required, "properties": properties})
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema",
              "$defs": {"operand": _operand_schema(catalog.handles)},
              "type": "object", "additionalProperties": False,
              "required": ["answer", "actions"], "properties": {
                  "answer": {"type": "string", "minLength": 1, "pattern": r"\S"},
                  "actions": {"type": "array", "items": {"oneOf": branches},
                              "minItems": 1, "maxItems": _STEP_LIMIT}}}
    return schema
def build_structured_proposal_prompt(
    *, question: str, graph: HCEG, table: Mapping[str, Any]
) -> str:
    del table
    catalog = build_reference_catalog(graph)
    path_surface = lambda ids: tuple(str(graph.nodes[node_id].text) for node_id in ids)
    paths = sorted({path_surface(ids) for entry in catalog.entries
                    for ids in (entry.row_header_path, entry.column_header_path)})
    path_handles = {path: f"P{index}" for index, path in enumerate(paths)}
    payload = {
        "task": (
            "Answer the table question with ordered canonical semantic actions. "
            "Use only the provided reference handles and earlier action results."
        ),
        "question": str(question),
        "reference_catalog_columns": [
            "handle", "kind", "surface", "typed_domain", "row_path", "column_path"
        ],
        "reference_path_columns": ["path", "header_surfaces"],
        "reference_paths": [[path_handles[path], list(path)] for path in paths],
        "reference_catalog": [
            [entry.handle, entry.kind, entry.surface, entry.typed_domain,
                path_handles[path_surface(entry.row_header_path)],
                path_handles[path_surface(entry.column_header_path)]]
            for entry in catalog.entries
        ],
        "operation_columns": ["op", "argument_count", "output_domain", "relation"],
        "operations": [
            [signature.signature_id, [contract.minimum_arity, contract.maximum_arity],
             signature.answer_domain, list(signature.comparison_polarities)]
            for signature in sorted(OPERATION_SIGNATURES.values(),
                                    key=lambda item: item.signature_id)
            for contract in (action_shape_contract(signature),)
        ],
        "output_rule": "Return one concise answer and only schema-defined actions.",
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def _first_forbidden_key(value: Any) -> str:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in _FORBIDDEN_KEYS:
                return str(key).lower()
            nested = _first_forbidden_key(child)
            if nested:
                return nested
    elif isinstance(value, (tuple, list)):
        for child in value:
            nested = _first_forbidden_key(child)
            if nested:
                return nested
    return ""

def validate_structured_proposal(
    payload: Any, source: Any
) -> StructuredProposalValidation:
    errors = []
    forbidden = _first_forbidden_key(payload)
    if forbidden:
        errors.append(f"forbidden_key:{forbidden}")
    validator = jsonschema.Draft202012Validator(build_structured_proposal_schema(source))
    schema_errors = list(validator.iter_errors(payload))
    if schema_errors:
        errors.append(f"schema_validation:{schema_errors[0].message}")
    wire_valid = not forbidden and not schema_errors
    actions = payload.get("actions") if isinstance(payload, Mapping) else None
    if not isinstance(actions, list):
        return StructuredProposalValidation(False, tuple(errors), wire_valid, False)
    shape_errors = []
    for index, action in enumerate(actions):
        signature = get_operation_signature(action.get("op")) if isinstance(action, Mapping) else None
        if signature is not None:
            shape_errors.extend(validate_action_shape(action, signature, step_index=index))
    errors.extend(shape_errors)
    local_shape_valid = wire_valid and not shape_errors
    return StructuredProposalValidation(local_shape_valid, tuple(errors),
                                        wire_valid, local_shape_valid)

@dataclass(frozen=True)
class _ResolvedOperand:
    node: Any
    domain: str
    source_table_refs: Tuple[str, ...]
    evidence_ids: Tuple[str, ...]
    provenance: Tuple[Tuple[str, str, str], ...]
    role_references: Tuple[str, ...]

def _dependencies(action: Mapping[str, Any]) -> Tuple[int, ...]:
    return tuple(int(arg["step"]) for arg in action.get("args") or ()
                 if arg.get("kind") == "STEP_RESULT")

def _descendants(actions: Sequence[Mapping[str, Any]]) -> Dict[str, Tuple[str, ...]]:
    children = {index: set() for index in range(len(actions))}
    for index, action in enumerate(actions):
        for parent in _dependencies(action):
            if parent < index:
                children[parent].add(index)
    result = {}
    for index in range(len(actions)):
        pending = list(children[index])
        found = set()
        while pending:
            current = pending.pop()
            if current in found:
                continue
            found.add(current)
            pending.extend(children[current])
        result[f"S{index}"] = tuple(f"S{item}" for item in sorted(found))
    return result

def _catalog_evidence(
    catalog: ReferenceCatalog, handles: Sequence[str]
) -> Tuple[str, ...]:
    return tuple(sorted({
        node_id for handle in handles for entry in (catalog.resolve(handle),)
        for node_id in (*entry.graph_node_ids, *entry.row_header_path,
                        *entry.column_header_path)
    }))

def _resolve_operand(
    operand: Mapping[str, Any],
    *,
    graph: HCEG,
    catalog: ReferenceCatalog,
    environment: Mapping[int, StepResult],
) -> _ResolvedOperand:
    if operand["kind"] == "TABLE_REF":
        entry = catalog.resolve(str(operand["ref"]))
        node = graph.nodes[entry.graph_node_ids[0]]
        return _ResolvedOperand(
            node=node,
            domain=entry.typed_domain,
            source_table_refs=(entry.handle,),
            evidence_ids=_catalog_evidence(catalog, (entry.handle,)),
            provenance=entry.provenance,
            role_references=entry.graph_node_ids,
        )
    source = environment[int(operand["step"])]
    number = parse_number(source.value) if source.output_domain == "SCALAR" else None
    return _ResolvedOperand(
        node=GraphNode("", NodeType.VALUE, text=source.value, numeric_value=number),
        domain=source.output_domain,
        source_table_refs=source.source_table_refs,
        evidence_ids=_catalog_evidence(catalog, source.source_table_refs),
        provenance=source.provenance,
        role_references=(f"STEP_RESULT:{source.step_id}",),
    )

def _role_bindings(
    signature: OperationSignatureVariant,
    operands: Sequence[_ResolvedOperand],
    catalog: ReferenceCatalog,
) -> Tuple[Tuple[str, Any], ...]:
    if signature.operation_family == "LOOKUP":
        entry = catalog.resolve(operands[0].source_table_refs[0])
        entity = entry.row_header_path or entry.graph_node_ids
        measure = entry.column_header_path or entry.graph_node_ids
        return (("TARGET_ENTITY", entity), ("TARGET_MEASURE", measure))
    references = tuple(operand.role_references for operand in operands)
    if signature.operation_family in {"DIFF", "RATIO", "PAIR_COMPARE"}:
        return (("LEFT_OPERAND", references[0]), ("RIGHT_OPERAND", references[1]))
    return (("AGGREGATION_SCOPE", references),)

def _make_step(
    index: int,
    action: Mapping[str, Any],
    signature: OperationSignatureVariant,
    descendants: Mapping[str, Tuple[str, ...]],
    *,
    dependencies: Tuple[str, ...] = (),
    evidence: Tuple[str, ...] = (),
    roles: Tuple[Tuple[str, Any], ...] = (),
    source_refs: Tuple[str, ...] = (),
    provenance: Tuple[Tuple[str, str, str], ...] = (),
    stage: str = "PROJECTED",
    state: str = "VALID",
    fault: Optional[FaultType] = None,
    reasons: Tuple[str, ...] = (),
    resource_complete: bool = True,
    value: str = "",
    output_domain: str = "",
) -> CompiledPublicStep:
    step_id = f"S{index}"
    result = StepResult(
        step_id, value, output_domain, index, source_refs, provenance, stage
    )
    return CompiledPublicStep(
        step_id, index, dict(action), signature.signature_id, dependencies,
        evidence, roles, signature.projection_operator, output_domain, value,
        descendants[step_id], state, fault, reasons, resource_complete, result,
        provenance,
    )

def _execute_action(
    index: int,
    action: Mapping[str, Any],
    *,
    graph: HCEG,
    catalog: ReferenceCatalog,
    environment: Mapping[int, StepResult],
    descendants: Mapping[str, Tuple[str, ...]],
) -> CompiledPublicStep:
    signature = get_operation_signature(action["op"])
    if signature is None:
        raise ValueError(f"unknown_signature:{action['op']}")
    dependency_ids = tuple(f"S{item}" for item in _dependencies(action))
    missing = tuple(step_id for step_id in dependency_ids
                    if int(step_id[1:]) >= index
                    or int(step_id[1:]) not in environment)
    if missing:
        return _make_step(
            index, action, signature, descendants, dependencies=dependency_ids,
            stage="DEPENDENCY", state="INVALID", fault=FaultType.DEPENDENCY,
            reasons=tuple(f"forward_or_missing_step_result:{item}" for item in missing),
            resource_complete=False,
        )
    unavailable = tuple(
        step_id for step_id in dependency_ids
        if environment[int(step_id[1:])].execution_status != "PROJECTED"
    )
    if unavailable:
        return _make_step(
            index, action, signature, descendants, dependencies=dependency_ids,
            stage="DEPENDENCY", state="INVALID",
            fault=FaultType.DEPENDENCY,
            reasons=tuple(f"unavailable_step_result:{item}" for item in unavailable),
            resource_complete=False,
        )
    operands = tuple(
        _resolve_operand(
            operand, graph=graph, catalog=catalog, environment=environment
        )
        for operand in action["args"]
    )
    source_refs = tuple(dict.fromkeys(
        handle for operand in operands for handle in operand.source_table_refs
    ))
    evidence = tuple(sorted({
        item for operand in operands for item in operand.evidence_ids
    }))
    provenance = tuple(sorted({
        item for operand in operands for item in operand.provenance
    }))
    roles = _role_bindings(signature, operands, catalog)
    numeric_family = signature.operation_family in {
        "SUM", "AVERAGE", "DIFF", "RATIO", "ARGMAX", "ARGMIN", "PAIR_COMPARE"
    }
    if numeric_family and any(operand.domain != "SCALAR" for operand in operands):
        return _make_step(
            index, action, signature, descendants, dependencies=dependency_ids,
            evidence=evidence, roles=roles, source_refs=source_refs,
            provenance=provenance, stage="OPERAND_TYPE",
            state="TYPE_MISMATCH", fault=FaultType.ROLE,
            reasons=("numeric_operation_requires_scalar_operands",),
        )
    if signature.operation_family == "RATIO":
        denominator = parse_number(str(getattr(operands[1].node, "text", "")))
        if denominator == 0:
            return _make_step(
                index, action, signature, descendants, dependencies=dependency_ids,
                evidence=evidence, roles=roles, source_refs=source_refs,
                provenance=provenance, stage="OPERATION_PRECONDITION",
                state="INVALID", fault=FaultType.OPERATION,
                reasons=("ratio_divide_by_zero",),
            )
    execution = SimpleNamespace(
        operation_family=signature.operation_family,
        projection_operator=signature.projection_operator,
        comparison_polarity=str(action.get("relation") or ""),
        operand_metadata=tuple(
            {"entity_binding_ids": list(
                catalog.resolve(operand.source_table_refs[0]).row_header_path
                if operand.source_table_refs else ()
            )}
            for operand in operands
        ),
    )
    projected = execute_typed_projection_from_nodes(
        execution, [operand.node for operand in operands], graph=graph
    )
    if projected.status != ProjectionStatus.PROJECTED:
        projection_failure = any(
            "projection" in reason or "entity_identity" in reason
            for reason in projected.failure_reasons
        )
        return _make_step(
            index, action, signature, descendants, dependencies=dependency_ids,
            evidence=evidence, roles=roles, source_refs=source_refs,
            provenance=provenance,
            stage="PROJECTION" if projection_failure else "EXECUTION",
            state="PROJECTION_FAILED" if projection_failure else "EXECUTION_FAILED",
            fault=(FaultType.PROJECTION if projection_failure else FaultType.OPERATION),
            reasons=tuple(projected.failure_reasons),
        )
    if projected.output_domain != signature.answer_domain:
        return _make_step(
            index, action, signature, descendants, dependencies=dependency_ids,
            evidence=evidence, roles=roles, source_refs=source_refs,
            provenance=provenance, stage="PROJECTION",
            state="TYPE_MISMATCH", fault=FaultType.PROJECTION,
            reasons=(
                f"output_domain_mismatch:{projected.output_domain}:{signature.answer_domain}",
            ),
        )
    return _make_step(
        index, action, signature, descendants, dependencies=dependency_ids,
        evidence=evidence, roles=roles, source_refs=source_refs,
        provenance=provenance, value=projected.value,
        output_domain=projected.output_domain,
    )

def execute_structured_action(
    action: Mapping[str, Any],
    *,
    prior_actions: Sequence[Mapping[str, Any]],
    graph: HCEG,
    catalog: ReferenceCatalog,
    environment: Mapping[int, StepResult],
) -> CompiledPublicStep:
    """Execute one appended action through the canonical typed action path."""
    actions = tuple(dict(item) for item in prior_actions) + (dict(action),)
    return _execute_action(
        len(prior_actions),
        action,
        graph=graph,
        catalog=catalog,
        environment=environment,
        descendants=_descendants(actions),
    )

def structured_action_descendants(
    actions: Sequence[Mapping[str, Any]],
) -> Dict[str, Tuple[str, ...]]:
    """Return the deterministic dependency descendants for an action sequence."""
    return _descendants(actions)

def _compiled_trace(
    *,
    answer: str,
    actions: Tuple[Mapping[str, Any], ...],
    catalog: ReferenceCatalog,
    steps: Tuple[CompiledPublicStep, ...],
    dataset: str,
) -> CompiledProposalTrace:
    failed = next(
        (step for step in steps if step.validation_state != "VALID"), None
    )
    final = steps[-1]
    answer_matches = bool(
        final.executed_value
        and typed_answer_equivalence(dataset, final.executed_value, answer)["equivalent"]
    )
    if failed is not None:
        public_fault = PublicFirstVerifiableFault(
            step_id=failed.step_id,
            stage=failed.step_result.execution_status,
            validation_state=failed.validation_state,
            fault_type=failed.fault_type or FaultType.OPERATION,
            reasons=failed.failure_reasons,
            resource_complete=failed.resource_complete,
        )
    elif not answer_matches:
        public_fault = PublicFirstVerifiableFault(
            step_id=final.step_id,
            stage="FINAL_ANSWER_RELATION",
            validation_state="INVALID",
            fault_type=FaultType.VALUE,
            reasons=("answer_not_equal_to_final_step_result",),
            resource_complete=True,
        )
    else:
        public_fault = None
    if not all(step.resource_complete for step in steps):
        consistency = AnswerTraceConsistency.RESOURCE_INCOMPLETE
    elif failed is not None:
        consistency = AnswerTraceConsistency.NON_EXECUTABLE
    elif answer_matches:
        consistency = AnswerTraceConsistency.CONSISTENT_EXECUTABLE
    else:
        consistency = AnswerTraceConsistency.INCONSISTENT_EXECUTABLE
    identity = {
        "version": STRUCTURED_PROPOSAL_VERSION,
        "answer": answer,
        "actions": _jsonable(actions),
        "compiled_steps": [step.to_dict() for step in steps],
    }
    return CompiledProposalTrace(
        answer=answer,
        actions=actions,
        catalog=catalog,
        steps=steps,
        first_verifiable_fault=public_fault,
        answer_trace_consistency=consistency,
        answer_claim_consistent=answer_matches,
        validation_ok=public_fault is None,
        record_sha256=canonical_json_hash(identity),
    )

def _compile_actions(
    *,
    answer: str,
    actions: Tuple[Mapping[str, Any], ...],
    graph: HCEG,
    catalog: ReferenceCatalog,
    dataset: str,
    reuse_steps: Sequence[CompiledPublicStep] = (),
    replay_indices: Optional[set[int]] = None,
) -> CompiledProposalTrace:
    descendants = _descendants(actions)
    environment: Dict[int, StepResult] = {}
    steps = []
    for index, action in enumerate(actions):
        if reuse_steps and replay_indices is not None and index not in replay_indices:
            step = reuse_steps[index]
        else:
            step = _execute_action(
                index, action, graph=graph, catalog=catalog,
                environment=environment, descendants=descendants,
            )
        steps.append(step)
        environment[index] = step.step_result
    return _compiled_trace(
        answer=answer, actions=actions, catalog=catalog,
        steps=tuple(steps), dataset=dataset,
    )

def compile_structured_proposal(
    payload: Mapping[str, Any], graph: HCEG, *, dataset: str
) -> CompiledProposalTrace:
    catalog = build_reference_catalog(graph)
    validation = validate_structured_proposal(payload, catalog)
    if not validation.ok:
        raise ValueError("structured_proposal:" + "|".join(validation.errors))
    actions = tuple(dict(action) for action in payload["actions"])
    return _compile_actions(
        answer=str(payload["answer"]).strip(), actions=actions,
        graph=graph, catalog=catalog, dataset=dataset,
    )

def replay_compiled_actions(
    compiled: CompiledProposalTrace,
    graph: HCEG,
    *,
    dataset: str,
    replacements: Mapping[int, Mapping[str, Any]],
) -> Tuple[CompiledProposalTrace, Tuple[str, ...], Tuple[str, ...]]:
    actions = tuple(
        dict(replacements.get(index, action))
        for index, action in enumerate(compiled.actions)
    )
    payload = {"answer": compiled.answer, "actions": list(actions)}
    validation = validate_structured_proposal(payload, compiled.catalog)
    if not validation.ok:
        raise ValueError("replayed_structured_proposal:" + "|".join(validation.errors))
    old_descendants = _descendants(compiled.actions)
    new_descendants = _descendants(actions)
    replay_indices = set(int(index) for index in replacements)
    for index in tuple(replay_indices):
        replay_indices.update(int(item[1:]) for item in old_descendants[f"S{index}"])
        replay_indices.update(int(item[1:]) for item in new_descendants[f"S{index}"])
    replayed = _compile_actions(
        answer=compiled.answer, actions=actions, graph=graph,
        catalog=compiled.catalog, dataset=dataset,
        reuse_steps=compiled.steps, replay_indices=replay_indices,
    )
    replay_ids = tuple(f"S{index}" for index in sorted(replay_indices))
    unchanged = tuple(
        f"S{index}" for index in range(len(actions)) if index not in replay_indices
    )
    return replayed, replay_ids, unchanged

def generate_structured_proposal(
    *,
    backend: Any,
    cache: Any,
    prompt: str,
    graph: HCEG,
    identity: Mapping[str, Any],
    audit_sink: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> Dict[str, Any]:
    forbidden = _first_forbidden_key(identity)
    if forbidden:
        raise ValueError(f"forbidden_generation_identity:{forbidden}")
    schema = build_structured_proposal_schema(graph)
    request_key = backend.cache_key(
        prompt,
        role="proposal",
        cache_context=identity,
        response_schema=schema,
        schema_name=STRUCTURED_PROPOSAL_SCHEMA_NAME,
    )
    cached = cache.load(request_key, identity)
    if cached is not None:
        record_key, record = cached
        return {**dict(record["payload"]), "cache_hit": True, "record_key": record_key}

    prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
    schema_sha = canonical_json_hash(schema)
    started = time.monotonic()
    try:
        response = backend.complete_json(
            prompt, role="proposal", response_schema=schema,
            schema_name=STRUCTURED_PROPOSAL_SCHEMA_NAME, cache_context=identity,
        )
    except ResponseContractError as error:
        metadata = dict(error.audit_metadata)
        raw = str(metadata.get("raw_content") or "")
        parsed, parse_error = None, ""
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as parse_exception:
            parse_error = str(parse_exception)
        validation = (
            validate_structured_proposal(parsed, graph)
            if parsed is not None else StructuredProposalValidation(False, ())
        )
        if audit_sink is not None:
            audit_sink(_response_audit_record(
                identity=identity, request_sha=str(metadata.get("request_sha256") or request_key),
                prompt_sha=prompt_sha, schema_sha=schema_sha, raw=raw, parsed=parsed,
                parse_error=parse_error or str(error), validation=validation,
                served_model=str(metadata.get("served_model") or ""),
                finish_reason=str(metadata.get("finish_reason") or ""),
                usage=metadata.get("usage") or {}, latency=time.monotonic() - started,
                sanitized=metadata.get("sanitized_raw_response_metadata") or {},
                cache_admitted=False,
            ))
        raise
    except Exception as error:
        if audit_sink is not None:
            audit_sink(_response_audit_record(
                identity=identity, request_sha=request_key, prompt_sha=prompt_sha,
                schema_sha=schema_sha, raw="", parsed=None, parse_error=str(error),
                validation=StructuredProposalValidation(False), served_model="",
                finish_reason="", usage={}, latency=time.monotonic() - started,
                sanitized={}, cache_admitted=False))
        raise
    validation = validate_structured_proposal(response.parsed_content, graph)
    raw = str(response.content)
    audit = _response_audit_record(
        identity=identity, request_sha=canonical_json_hash(response.request_payload),
        prompt_sha=prompt_sha, schema_sha=schema_sha, raw=raw,
        parsed=response.parsed_content, parse_error="", validation=validation,
        served_model=str(response.served_model), finish_reason=str(response.finish_reason),
        usage={
            "prompt_tokens": int(response.prompt_tokens),
            "completion_tokens": int(response.completion_tokens),
            "total_tokens": int(response.total_tokens),
        },
        latency=time.monotonic() - started,
        sanitized=dict(getattr(response, "audit_metadata", {})).get(
            "sanitized_raw_response_metadata", {}
        ),
        cache_admitted=validation.ok,
    )
    if audit_sink is not None:
        audit_sink(audit)
    if not validation.ok:
        raise ValueError("generated_structured_proposal:" + "|".join(validation.errors))
    generation = {
        "structured_proposal_version": STRUCTURED_PROPOSAL_VERSION,
        "content": raw,
        "parsed_content": _jsonable(response.parsed_content),
        "served_model": str(response.served_model),
        "finish_reason": str(response.finish_reason),
        "prompt_tokens": int(response.prompt_tokens),
        "completion_tokens": int(response.completion_tokens),
        "reasoning_tokens": int(response.reasoning_tokens),
        "content_tokens": int(response.content_tokens),
        "total_tokens": int(response.total_tokens),
        "request_payload": _jsonable(response.request_payload),
        "raw_response_sha256": canonical_json_hash(_jsonable(response.raw_response)),
        "latency_seconds": time.monotonic() - started,
    }
    generation["generation_record_sha256"] = canonical_json_hash(generation)
    request_contract_sha = canonical_json_hash({
        "prompt": prompt,
        "schema": schema,
        "schema_name": STRUCTURED_PROPOSAL_SCHEMA_NAME,
    })
    output_identity = {
        **dict(identity),
        "generation_record_sha256": generation["generation_record_sha256"],
    }
    record_key = cache.store(
        request_key=request_key,
        static_identity=identity,
        output_identity=output_identity,
        query_bundle_sha256=request_contract_sha,
        payload=generation,
    )
    return {**generation, "cache_hit": False, "record_key": record_key}

def _response_audit_record(
    *, identity: Mapping[str, Any], request_sha: str, prompt_sha: str,
    schema_sha: str, raw: str, parsed: Any, parse_error: str,
    validation: StructuredProposalValidation, served_model: str,
    finish_reason: str, usage: Mapping[str, Any], latency: float,
    sanitized: Mapping[str, Any], cache_admitted: bool,
) -> Dict[str, Any]:
    record = {
        "schema_version": "certa_invalid_response_audit_v2",
        "dataset": str(identity.get("dataset") or ""),
        "id": str(identity.get("sample_id") or ""),
        "request_sha256": request_sha,
        "prompt_sha256": prompt_sha,
        "response_schema_sha256": schema_sha,
        "served_model": served_model,
        "finish_reason": finish_reason,
        "usage": _jsonable(usage),
        "latency_seconds": max(0.0, latency),
        "raw_content": raw,
        "raw_content_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "sanitized_raw_response_metadata": _jsonable(sanitized),
        "json_parse_ok": parsed is not None,
        "json_parse_error": parse_error,
        "wire_schema_valid": validation.wire_schema_valid,
        "local_shape_valid": validation.local_shape_valid,
        "local_validation_ok": validation.ok,
        "local_validation_errors": list(validation.errors),
        "cache_admitted": cache_admitted,
    }
    if parsed is not None:
        record["parsed_payload"] = _jsonable(parsed)
    return record
