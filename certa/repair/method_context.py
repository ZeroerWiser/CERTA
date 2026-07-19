"""Gold-free method context and posthoc evaluation records."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from types import MappingProxyType
from typing import Any, Dict, FrozenSet, Mapping, MutableSet


FORBIDDEN_METHOD_CONTEXT_KEYS: FrozenSet[str] = frozenset(
    {
        "answers",
        "correct",
        "correctness",
        "evaluation_label",
        "evaluation_labels",
        "expected_answer",
        "expected_answers",
        "gold",
        "gold_answer",
        "gold_answers",
        "gold_label",
        "gold_labels",
        "initial_correct",
        "offline_initial_correct",
        "oracle_label",
        "oracle_labels",
    }
)

METHOD_CONTEXT_FIELDS = (
    "question_frame",
    "graph_stats",
    "edge_reliability_diag",
    "layout_risk",
    "question_operation",
)


def assert_method_context_clean(value: Any) -> None:
    """Raise when a forbidden evaluation key occurs recursively."""
    _assert_clean(value, seen=set())


def _assert_clean(value: Any, *, seen: MutableSet[int]) -> None:
    if isinstance(value, Mapping):
        value_id = id(value)
        if value_id in seen:
            return
        seen.add(value_id)
        for key, nested in value.items():
            if (
                isinstance(key, str)
                and key.casefold() in FORBIDDEN_METHOD_CONTEXT_KEYS
            ):
                raise ValueError(f"forbidden_method_context_key:{key}")
            _assert_clean(nested, seen=seen)
        return

    if is_dataclass(value) and not isinstance(value, type):
        value_id = id(value)
        if value_id in seen:
            return
        seen.add(value_id)
        for item in fields(value):
            if item.name.casefold() in FORBIDDEN_METHOD_CONTEXT_KEYS:
                raise ValueError(f"forbidden_method_context_key:{item.name}")
            _assert_clean(getattr(value, item.name), seen=seen)
        return

    if isinstance(value, (list, tuple, set, frozenset)):
        value_id = id(value)
        if value_id in seen:
            return
        seen.add(value_id)
        for nested in value:
            _assert_clean(nested, seen=seen)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze(nested) for key, nested in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(nested) for nested in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(nested) for nested in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_thaw(nested) for nested in value]
    if isinstance(value, frozenset):
        return [_thaw(nested) for nested in sorted(value, key=repr)]
    return value


@dataclass(frozen=True)
class MethodInferenceContext:
    """Immutable allowlisted context available to live method inference."""

    question_frame: Any = field(default_factory=dict)
    graph_stats: Any = field(default_factory=dict)
    edge_reliability_diag: Any = field(default_factory=dict)
    layout_risk: Any = 0.0
    question_operation: Any = None

    def __post_init__(self) -> None:
        selected = {
            name: getattr(self, name) for name in METHOD_CONTEXT_FIELDS
        }
        assert_method_context_clean(selected)
        for name, value in selected.items():
            object.__setattr__(self, name, _freeze(value))

    def to_dict(self) -> Dict[str, Any]:
        """Return only clean method-facing fields as plain containers."""
        payload = {
            name: _thaw(getattr(self, name)) for name in METHOD_CONTEXT_FIELDS
        }
        assert_method_context_clean(payload)
        return payload


def build_method_inference_context(
    source: Mapping[str, Any],
) -> MethodInferenceContext:
    """Build a method context from the explicit live-inference allowlist."""
    selected = {
        "question_frame": source.get("question_frame", {}),
        "graph_stats": source.get("graph_stats", {}),
        "edge_reliability_diag": source.get("edge_reliability_diag", {}),
        "layout_risk": source.get("layout_risk", 0.0),
        "question_operation": source.get("question_operation"),
    }
    assert_method_context_clean(selected)
    return MethodInferenceContext(**selected)


@dataclass(frozen=True)
class PosthocEvaluationRecord:
    """Gold and correctness data available only after method inference."""

    gold_answer: Any = None
    correctness: Any = None
    initial_correct: Any = None
    offline_initial_correct: Any = None

    def __post_init__(self) -> None:
        for item in fields(self):
            object.__setattr__(
                self,
                item.name,
                _freeze(getattr(self, item.name)),
            )

    def to_dict(self) -> Dict[str, Any]:
        """Return posthoc evaluation fields as plain containers."""
        return {
            item.name: _thaw(getattr(self, item.name)) for item in fields(self)
        }


__all__ = [
    "FORBIDDEN_METHOD_CONTEXT_KEYS",
    "METHOD_CONTEXT_FIELDS",
    "MethodInferenceContext",
    "PosthocEvaluationRecord",
    "assert_method_context_clean",
    "build_method_inference_context",
]
