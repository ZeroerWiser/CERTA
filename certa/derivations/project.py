"""Projection and answer-equivalence helpers for executable derivations."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from .answer_equivalence import inference_answer_key, inference_answers_equivalent
from certa.repair.evidence_dsl import parse_number


def _node_text(node: Any) -> str:
    return str(getattr(node, "text", "") or "")


def _node_number(node: Any) -> Optional[float]:
    value = getattr(node, "numeric_value", None)
    if value is not None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return parse_number(_node_text(node))


def _format_scalar(value: float) -> str:
    if math.isfinite(value) and value == int(value):
        return str(int(value))
    if abs(value) < 100:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    if abs(value) < 1000:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.0f}"


def canonical_answer_key(value: Any) -> str:
    return inference_answer_key(value).compact()


def answers_equivalent(left: Any, right: Any) -> bool:
    return inference_answers_equivalent(left, right)


def _node_id(node: Any) -> str:
    return str(getattr(node, "node_id", "") or "")


def _selected_indices(values: List[float], operation_family: str) -> List[int]:
    if not values:
        return []
    extreme = min(values) if operation_family == "ARGMIN" else max(values)
    return [index for index, value in enumerate(values) if value == extreme]


def _pair_compare_boolean(numbers: Sequence[Optional[float]], polarity: str) -> Optional[str]:
    if len(numbers) != 2 or numbers[0] is None or numbers[1] is None:
        return None
    left = float(numbers[0])
    right = float(numbers[1])
    if polarity in {"min", "less", "less_than"}:
        ok = left < right
    elif polarity == "less_equal":
        ok = left <= right
    elif polarity == "greater":
        ok = left > right
    elif polarity == "equal":
        ok = left == right
    else:
        ok = left >= right
    return "true" if ok else "false"


class ProjectionStatus(str, Enum):
    PROJECTED = "PROJECTED"
    PROJECTION_FAILED = "PROJECTION_FAILED"


@dataclass(frozen=True)
class TypedProjectionResult:
    status: ProjectionStatus
    value: str
    output_domain: str
    projection_operator: str
    source_node_ids: Tuple[str, ...] = ()
    source_entity_binding_ids: Tuple[str, ...] = ()
    failure_reasons: Tuple[str, ...] = ()

    def to_dict(self) -> Mapping[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


def _failed_projection(projection: str, *reasons: str) -> TypedProjectionResult:
    return TypedProjectionResult(
        status=ProjectionStatus.PROJECTION_FAILED,
        value="",
        output_domain="",
        projection_operator=projection,
        failure_reasons=tuple(reason for reason in reasons if reason),
    )


def _actual_value_domain(value: Any) -> str:
    key = inference_answer_key(value)
    if key.category.startswith("NUMERIC"):
        return "SCALAR"
    if key.category == "BOOLEAN_EXACT":
        return "BOOLEAN"
    if key.category == "SET_EXACT_NORMALIZED":
        return "SET"
    return "ENTITY"


def _entity_binding_ids(derivation: Any, operand_index: int) -> Tuple[str, ...]:
    metadata = getattr(derivation, "operand_metadata", []) or []
    if operand_index < 0 or operand_index >= len(metadata):
        return ()
    values = metadata[operand_index].get("entity_binding_ids") or []
    if not isinstance(values, (list, tuple)):
        return ()
    return tuple(str(value) for value in values if str(value))


def _exact_entity_projection(
    derivation: Any,
    selected_indices: Sequence[int],
    graph: Any,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    if graph is None:
        return (), ()
    graph_nodes = getattr(graph, "nodes", {}) or {}
    labels = []
    binding_ids = []
    for index in selected_indices:
        selected_binding_ids = _entity_binding_ids(derivation, index)
        if not selected_binding_ids:
            return (), ()
        selected_labels = []
        for binding_id in selected_binding_ids:
            node = graph_nodes.get(binding_id)
            label = _node_text(node) if node is not None else ""
            if not label:
                return (), ()
            selected_labels.append(label)
            binding_ids.append(binding_id)
        labels.append(" / ".join(selected_labels))
    return tuple(sorted(set(labels))), tuple(sorted(set(binding_ids)))


def execute_typed_projection_from_nodes(
    derivation: Any,
    nodes: List[Any],
    *,
    graph: Any = None,
) -> TypedProjectionResult:
    """Execute a projection and return its actual domain and provenance."""
    family = str(getattr(derivation, "operation_family", "UNKNOWN"))
    projection = str(getattr(derivation, "projection_operator", "UNKNOWN"))
    source_ids = tuple(_node_id(node) for node in nodes if _node_id(node))

    if family in {"LOOKUP", "LOOKUP_AGGREGATE"}:
        if not nodes:
            return _failed_projection(projection, "missing_lookup_operand")
        if projection != "VALUE_PROJECTION":
            return _failed_projection(projection, "lookup_projection_not_value")
        value = _node_text(nodes[0])
        if not value:
            return _failed_projection(projection, "empty_lookup_value")
        return TypedProjectionResult(
            status=ProjectionStatus.PROJECTED,
            value=value,
            output_domain=_actual_value_domain(value),
            projection_operator=projection,
            source_node_ids=source_ids,
            source_entity_binding_ids=_entity_binding_ids(derivation, 0),
        )

    numbers = [_node_number(node) for node in nodes]
    if family in {"SUM", "AVERAGE", "DIFF", "RATIO", "COUNT", "ARGMAX", "ARGMIN", "PAIR_COMPARE"}:
        if family != "COUNT" and any(value is None for value in numbers):
            return _failed_projection(projection, "non_numeric_operand")

    if family == "SUM":
        if not numbers:
            return _failed_projection(projection, "missing_sum_operands")
        value = _format_scalar(sum(float(v) for v in numbers if v is not None))
    elif family == "AVERAGE":
        if not numbers:
            return _failed_projection(projection, "missing_average_operands")
        value = _format_scalar(sum(float(v) for v in numbers if v is not None) / len(numbers))
    elif family == "DIFF":
        if len(numbers) != 2:
            return _failed_projection(projection, "diff_requires_two_operands")
        value = _format_scalar(float(numbers[0]) - float(numbers[1]))
    elif family == "RATIO":
        if len(numbers) != 2:
            return _failed_projection(projection, "ratio_requires_two_operands")
        if float(numbers[1]) == 0.0:
            return _failed_projection(projection, "ratio_divide_by_zero")
        value = _format_scalar(float(numbers[0]) / float(numbers[1]))
    elif family == "COUNT":
        if not nodes:
            return _failed_projection(projection, "missing_count_members")
        value = str(len(nodes))
    else:
        value = ""
    if family in {"SUM", "AVERAGE", "DIFF", "RATIO", "COUNT"}:
        if projection != "SCALAR_RESULT_PROJECTION":
            return _failed_projection(projection, "scalar_operation_projection_mismatch")
        return TypedProjectionResult(
            status=ProjectionStatus.PROJECTED,
            value=value,
            output_domain="SCALAR",
            projection_operator=projection,
            source_node_ids=source_ids,
        )

    if family in {"ARGMAX", "ARGMIN"}:
        indices = _selected_indices([float(v) for v in numbers if v is not None], family)
        if not indices:
            return _failed_projection(projection, "missing_arg_extreme_operand")
        if projection == "VALUE_PROJECTION":
            extreme = (
                min(float(value) for value in numbers if value is not None)
                if family == "ARGMIN"
                else max(float(value) for value in numbers if value is not None)
            )
            return TypedProjectionResult(
                status=ProjectionStatus.PROJECTED,
                value=_format_scalar(extreme),
                output_domain="SCALAR",
                projection_operator=projection,
                source_node_ids=tuple(
                    node_id
                    for index in indices
                    for node_id in (_node_id(nodes[index]),)
                    if node_id
                ),
                source_entity_binding_ids=tuple(sorted({
                    binding_id
                    for index in indices
                    for binding_id in _entity_binding_ids(derivation, index)
                })),
            )
        if projection != "ROW_ENTITY_PROJECTION":
            return _failed_projection(projection, "unsupported_extremum_projection")
        labels, binding_ids = _exact_entity_projection(derivation, indices, graph)
        if not labels or not binding_ids:
            return _failed_projection(projection, "missing_exact_entity_identity")
        return TypedProjectionResult(
            status=ProjectionStatus.PROJECTED,
            value=" | ".join(labels) if len(labels) > 1 else labels[0],
            output_domain="SET" if len(labels) > 1 else "ENTITY",
            projection_operator=projection,
            source_node_ids=tuple(
                node_id
                for index in indices
                for node_id in (_node_id(nodes[index]),)
                if node_id
            ),
            source_entity_binding_ids=binding_ids,
        )
    if family == "PAIR_COMPARE":
        if len(numbers) != 2:
            return _failed_projection(projection, "pair_compare_requires_two_operands")
        polarity = str(getattr(derivation, "comparison_polarity", "") or "")
        if polarity == "max":
            polarity = "greater_equal"
        elif polarity == "min":
            polarity = "less_equal"
        if polarity not in {"greater", "greater_equal", "less", "less_equal", "equal"}:
            return _failed_projection(projection, "pair_compare_missing_or_invalid_polarity")
        if projection != "BOOLEAN_PROJECTION":
            return _failed_projection(projection, "pair_compare_entity_selection_not_formalized")
        value = _pair_compare_boolean(numbers, polarity)
        if value is None:
            return _failed_projection(projection, "pair_compare_requires_two_numeric_operands")
        return TypedProjectionResult(
            status=ProjectionStatus.PROJECTED,
            value=value,
            output_domain="BOOLEAN",
            projection_operator=projection,
            source_node_ids=source_ids,
        )

    return _failed_projection(projection, "unknown_operation_family")


def execute_projection_from_nodes(
    derivation: Any,
    nodes: List[Any],
    *,
    graph: Any = None,
) -> Tuple[Optional[str], List[str]]:
    """Compatibility wrapper around the typed active projection result."""
    result = execute_typed_projection_from_nodes(derivation, nodes, graph=graph)
    if result.status != ProjectionStatus.PROJECTED:
        return None, list(result.failure_reasons)
    return result.value, []
