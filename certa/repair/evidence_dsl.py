"""Bounded Evidence DSL for CERA derivation checks.

Supported operations:
  SELECT, COUNT, SUM, DIFF, RATIO, COMPARE, ARGMAX, ARGMIN

The parser intentionally avoids eval, imports, attribute access, external
lookups, and nested execution. Arguments are evidence IDs such as S1 or quoted
operators for COMPARE.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .evidence_packet import CausalEvidencePacket

_CALL_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$")
_EVIDENCE_ID_RE = re.compile(r"^S\d+$")
_SAFE_EXPR_RE = re.compile(r"^[A-Za-z0-9_(),.><=!\"'\s:+\-/%]+$")
SUPPORTED_OPERATIONS = {"SELECT", "COUNT", "SUM", "DIFF", "RATIO", "COMPARE", "ARGMAX", "ARGMIN"}


@dataclass
class DSLExecutionResult:
    ok: bool
    result: str = ""
    normalized_result: str = ""
    operation: str = ""
    evidence_ids: List[str] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "result": self.result,
            "normalized_result": self.normalized_result,
            "operation": self.operation,
            "evidence_ids": self.evidence_ids,
            "trace": self.trace,
            "error": self.error,
        }


def normalize_answer_text(value: Any) -> str:
    try:
        from eval_utils import normalize_text

        return normalize_text(value)
    except Exception:
        text = str(value or "").strip().lower()
        return re.sub(r"\s+", " ", text.replace(",", ""))


def parse_number(value: Any) -> Optional[float]:
    text = str(value or "").strip()
    if not text:
        return None
    percent = text.endswith("%")
    text = text.replace(",", "").replace("$", "").replace("£", "").replace("€", "")
    text = text.rstrip("%").strip()
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        number = float(match.group(0))
    except ValueError:
        return None
    if percent:
        return number
    return number


def format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6g}"


def _packet_dict(packet: Any) -> Dict[str, Any]:
    if isinstance(packet, CausalEvidencePacket):
        return packet.to_dict()
    if isinstance(packet, Mapping):
        return dict(packet)
    return {}


def _evidence_index(packet: Any) -> Dict[str, Dict[str, Any]]:
    payload = _packet_dict(packet)
    items = payload.get("support_chain") or []
    index: Dict[str, Dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, Mapping):
            continue
        evidence_id = str(item.get("evidence_id", ""))
        if evidence_id:
            index[evidence_id] = dict(item)
    return index


def _split_args(raw_args: str) -> List[str]:
    args: List[str] = []
    current: List[str] = []
    quote: Optional[str] = None
    for ch in raw_args:
        if quote:
            current.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in {"'", '"'}:
            quote = ch
            current.append(ch)
            continue
        if ch == ",":
            arg = "".join(current).strip()
            if arg:
                args.append(arg)
            current = []
            continue
        current.append(ch)
    if quote:
        raise ValueError("unterminated_quote")
    tail = "".join(current).strip()
    if tail:
        args.append(tail)
    return args


def parse_expression(expression: str) -> Tuple[str, List[str]]:
    expr = str(expression or "").strip()
    if not expr:
        raise ValueError("empty_expression")
    if len(expr) > 500:
        raise ValueError("expression_too_long")
    if not _SAFE_EXPR_RE.match(expr):
        raise ValueError("unsafe_character")
    match = _CALL_RE.match(expr)
    if not match:
        raise ValueError("invalid_call_syntax")
    op = match.group(1).upper()
    if op not in SUPPORTED_OPERATIONS:
        raise ValueError("unsupported_operation")
    return op, _split_args(match.group(2))


def _unquote(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


def _require_evidence_ids(args: Sequence[str], index: Mapping[str, Dict[str, Any]]) -> List[str]:
    ids: List[str] = []
    for arg in args:
        evidence_id = _unquote(arg)
        if not _EVIDENCE_ID_RE.match(evidence_id):
            raise ValueError("expected_evidence_id")
        if evidence_id not in index:
            raise ValueError("unknown_evidence_id")
        ids.append(evidence_id)
    return ids


def _values(ids: Sequence[str], index: Mapping[str, Dict[str, Any]]) -> List[str]:
    return [str(index[eid].get("cell_value", "")) for eid in ids]


def _label_for_element(item: Mapping[str, Any]) -> str:
    for key in ("row_headers", "col_headers"):
        headers = item.get(key) or []
        if isinstance(headers, list):
            for header in headers:
                text = str(header).strip()
                if text:
                    return text
    return str(item.get("cell_value", ""))


def _trace(ids: Sequence[str], index: Mapping[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for eid in ids:
        item = index[eid]
        rows.append({
            "evidence_id": eid,
            "row": item.get("row"),
            "col": item.get("col"),
            "value": item.get("cell_value", ""),
            "row_headers": item.get("row_headers", []),
            "col_headers": item.get("col_headers", []),
        })
    return rows


def execute_evidence_dsl(expression: str, packet: Any) -> DSLExecutionResult:
    try:
        op, args = parse_expression(expression)
        index = _evidence_index(packet)
        if op == "COMPARE":
            if len(args) != 3:
                raise ValueError("compare_requires_three_args")
            left_id, raw_operator, right_id = args
            ids = _require_evidence_ids([left_id, right_id], index)
            operator = _unquote(raw_operator)
            if operator not in {">", "<", ">=", "<=", "==", "!=", "max", "min"}:
                raise ValueError("invalid_compare_operator")
            left_num = parse_number(index[ids[0]].get("cell_value"))
            right_num = parse_number(index[ids[1]].get("cell_value"))
            if left_num is None or right_num is None:
                raise ValueError("non_numeric_compare")
            if operator == ">":
                value = str(left_num > right_num).lower()
            elif operator == "<":
                value = str(left_num < right_num).lower()
            elif operator == ">=":
                value = str(left_num >= right_num).lower()
            elif operator == "<=":
                value = str(left_num <= right_num).lower()
            elif operator == "==":
                value = str(abs(left_num - right_num) < 1e-9).lower()
            elif operator == "!=":
                value = str(abs(left_num - right_num) >= 1e-9).lower()
            elif operator == "max":
                value = _label_for_element(index[ids[0]] if left_num >= right_num else index[ids[1]])
            else:
                value = _label_for_element(index[ids[0]] if left_num <= right_num else index[ids[1]])
            return DSLExecutionResult(True, value, normalize_answer_text(value), op, ids, _trace(ids, index))

        ids = _require_evidence_ids(args, index)
        values = _values(ids, index)
        numbers = [parse_number(v) for v in values]
        if op == "SELECT":
            if len(ids) != 1:
                raise ValueError("select_requires_one_arg")
            value = values[0]
        elif op == "COUNT":
            value = str(len(ids))
        elif op == "SUM":
            if any(v is None for v in numbers):
                raise ValueError("non_numeric_sum")
            value = format_number(sum(v for v in numbers if v is not None))
        elif op == "DIFF":
            if len(ids) != 2:
                raise ValueError("diff_requires_two_args")
            if numbers[0] is None or numbers[1] is None:
                raise ValueError("non_numeric_diff")
            value = format_number(numbers[0] - numbers[1])
        elif op == "RATIO":
            if len(ids) != 2:
                raise ValueError("ratio_requires_two_args")
            if numbers[0] is None or numbers[1] is None:
                raise ValueError("non_numeric_ratio")
            if abs(numbers[1]) < 1e-12:
                raise ValueError("division_by_zero")
            value = format_number(numbers[0] / numbers[1])
        elif op in {"ARGMAX", "ARGMIN"}:
            if not ids:
                raise ValueError("arg_requires_evidence")
            if any(v is None for v in numbers):
                raise ValueError("non_numeric_arg")
            selected_index = max(range(len(ids)), key=lambda i: numbers[i]) if op == "ARGMAX" else min(range(len(ids)), key=lambda i: numbers[i])
            value = _label_for_element(index[ids[selected_index]])
        else:
            raise ValueError("unsupported_operation")
        return DSLExecutionResult(True, value, normalize_answer_text(value), op, ids, _trace(ids, index))
    except ValueError as exc:
        return DSLExecutionResult(False, error=str(exc))


def evidence_ids_for_expression(expression: str) -> List[str]:
    try:
        _op, args = parse_expression(expression)
    except ValueError:
        return []
    ids: List[str] = []
    for arg in args:
        value = _unquote(arg)
        if _EVIDENCE_ID_RE.match(value):
            ids.append(value)
    return ids
