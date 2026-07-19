"""Minimal Safety Validator for CERA outputs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

from certa.derivations.answer_equivalence import inference_answers_equivalent

from .evidence_dsl import evidence_ids_for_expression, execute_evidence_dsl, normalize_answer_text
from .evidence_packet import CERAOutput, CausalEvidencePacket

VALID_DECISIONS = {"USE_REPAIRED", "KEEP_ORIGINAL", "INSUFFICIENT_CERTIFICATE"}
EVIDENCE_ID_RE = re.compile(r"\b(?:OS|S)\d+\b")
CF_ID_RE = re.compile(r"\bCF\d+\b")
H_ID_RE = re.compile(r"\bH\d+\b")
D_REF_RE = re.compile(r"\bD\d+\b")
E_REF_RE = re.compile(r"\bE\d+\b")
I_REF_RE = re.compile(r"\bI\d+\b")
GOLD_KEY_RE = re.compile(r"(?:^|_)(gold|expected|label)(?:_|$)", re.IGNORECASE)


@dataclass
class ValidatorResult:
    accepted: bool
    reject_reason: str = ""
    reject_reasons: List[str] = field(default_factory=list)
    decision: str = ""
    cited_evidence_ids: List[str] = field(default_factory=list)
    cited_counterfactual_ids: List[str] = field(default_factory=list)
    dsl_result: Dict[str, Any] = field(default_factory=dict)
    parsed_output: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reject_reason": self.reject_reason,
            "reject_reasons": self.reject_reasons,
            "decision": self.decision,
            "cited_evidence_ids": self.cited_evidence_ids,
            "cited_counterfactual_ids": self.cited_counterfactual_ids,
            "dsl_result": self.dsl_result,
            "parsed_output": self.parsed_output,
            "notes": self.notes,
        }


def _strip_fenced_json(text: str) -> str:
    value = str(text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
    first = value.find("{")
    last = value.rfind("}")
    if first >= 0 and last >= first:
        return value[first:last + 1]
    return value


def _parse_output(raw: Any) -> Tuple[Optional[CERAOutput], Optional[str]]:
    if isinstance(raw, CERAOutput):
        return raw, None
    if isinstance(raw, Mapping):
        return CERAOutput.from_dict(raw), None
    try:
        payload = json.loads(_strip_fenced_json(str(raw or "")))
    except Exception:
        return None, "json_parse_error"
    if not isinstance(payload, Mapping):
        return None, "json_parse_error"
    return CERAOutput.from_dict(payload), None


def _packet_dict(packet: Any) -> Dict[str, Any]:
    if isinstance(packet, CausalEvidencePacket):
        return packet.to_dict()
    if isinstance(packet, Mapping):
        return dict(packet)
    return {}


def _known_ids(packet: Any) -> Tuple[Set[str], Set[str], Set[str], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    payload = _packet_dict(packet)
    support = payload.get("support_chain") or []
    original_support = payload.get("original_support_chain") or []
    cfs = payload.get("counterfactual_chain") or []
    support_ids = {
        str(item.get("evidence_id"))
        for item in support
        if isinstance(item, Mapping) and item.get("evidence_id")
    }
    original_support_ids = {
        str(item.get("evidence_id"))
        for item in original_support
        if isinstance(item, Mapping) and item.get("evidence_id")
    }
    cf_ids = {
        str(item.get("cf_id"))
        for item in cfs
        if isinstance(item, Mapping) and item.get("cf_id")
    }
    support_index = {
        str(item.get("evidence_id")): dict(item)
        for item in list(support) + list(original_support)
        if isinstance(item, Mapping) and item.get("evidence_id")
    }
    cf_index = {
        str(item.get("cf_id")): dict(item)
        for item in cfs
        if isinstance(item, Mapping) and item.get("cf_id")
    }
    return support_ids, original_support_ids, cf_ids, support_index, cf_index


def _walk_values(value: Any) -> Iterable[Any]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield key
            yield from _walk_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_values(item)
    else:
        yield value


def _collect_citations(value: Any) -> Tuple[List[str], List[str]]:
    text_parts = [str(v) for v in _walk_values(value) if v is not None]
    text = " ".join(text_parts)
    return sorted(set(EVIDENCE_ID_RE.findall(text))), sorted(set(CF_ID_RE.findall(text)))


def _list_field_ids(value: Any, field_name: str, pattern: re.Pattern[str]) -> List[str]:
    if not isinstance(value, Mapping):
        return []
    raw = value.get(field_name)
    out: List[str] = []
    if isinstance(raw, list):
        for item in raw:
            text = str(item)
            if pattern.fullmatch(text):
                out.append(text)
    elif isinstance(raw, str):
        out.extend(pattern.findall(raw))
    return sorted(dict.fromkeys(out))


def _role_evidence_ids(value: Any) -> List[str]:
    explicit = _list_field_ids(value, "evidence_ids", EVIDENCE_ID_RE)
    if explicit:
        return explicit
    cited_s, _ = _collect_citations(value)
    return cited_s


def _role_cf_ids(value: Any) -> List[str]:
    explicit = _list_field_ids(value, "cf_ids", CF_ID_RE)
    if explicit:
        return explicit
    _cited_s, cited_cf = _collect_citations(value)
    return cited_cf


def _contains_gold_leak(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if GOLD_KEY_RE.search(str(key)):
                return True
            if _contains_gold_leak(item):
                return True
        return False
    if isinstance(value, list):
        return any(_contains_gold_leak(item) for item in value)
    return False


def _derivation_expression(output: CERAOutput) -> str:
    dp = output.derivation_program
    if isinstance(dp, str):
        return dp.strip()
    if isinstance(dp, Mapping):
        for key in ("expression", "program", "dsl", "evidence_dsl"):
            value = dp.get(key)
            if value:
                return str(value).strip()
    return ""


def _allowed_answer_values(packet: Any, dsl_result: Optional[Dict[str, Any]] = None) -> Set[str]:
    payload = _packet_dict(packet)
    values = {
        normalize_answer_text(payload.get("original_answer", "")),
        normalize_answer_text(payload.get("candidate_under_review", "")),
    }
    candidate = payload.get("candidate") if isinstance(payload.get("candidate"), Mapping) else {}
    values.add(normalize_answer_text(candidate.get("denotation", "")))
    for item in payload.get("support_chain") or []:
        if not isinstance(item, Mapping):
            continue
        values.add(normalize_answer_text(item.get("cell_value", "")))
        for key in ("row_headers", "col_headers"):
            headers = item.get(key) or []
            if isinstance(headers, list):
                for header in headers:
                    values.add(normalize_answer_text(header))
    if dsl_result and dsl_result.get("result") is not None:
        values.add(normalize_answer_text(dsl_result.get("result", "")))
    return {v for v in values if v}


def _counterfactual_observed_available(cf_id: str, cf_index: Mapping[str, Mapping[str, Any]]) -> bool:
    item = cf_index.get(cf_id) or {}
    observed = item.get("observed_effect") if isinstance(item.get("observed_effect"), Mapping) else {}
    return bool(observed.get("available")) and bool(observed.get("candidate_specific"))


def _support_outside_evidence(evidence_id: str, support_index: Mapping[str, Mapping[str, Any]]) -> bool:
    item = support_index.get(evidence_id) or {}
    return str(item.get("provenance", "")) == "executor_cell_not_in_evidence_subgraph"


def _finish(decision: str, reasons: List[str], output: Optional[CERAOutput], cited_s: List[str], cited_cf: List[str], dsl_result: Dict[str, Any], notes: Optional[List[str]] = None) -> ValidatorResult:
    reasons = [reason for reason in reasons if reason]
    return ValidatorResult(
        accepted=not reasons,
        reject_reason=reasons[0] if reasons else "",
        reject_reasons=reasons,
        decision=decision,
        cited_evidence_ids=cited_s,
        cited_counterfactual_ids=cited_cf,
        dsl_result=dsl_result,
        parsed_output=output.to_dict() if output is not None else {},
        notes=notes or [],
    )


def _compact_contrast_v3(packet: Any) -> Dict[str, Any]:
    payload = _packet_dict(packet)
    contrast = payload.get("compact_behavioral_contrast_v3") or {}
    return dict(contrast) if isinstance(contrast, Mapping) else {}


def _registry_indexes(contrast: Mapping[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    registry = contrast.get("registry") or {}
    if not isinstance(registry, Mapping):
        registry = {}
    def index(records: Any, key: str) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        if not isinstance(records, list):
            return out
        for item in records:
            if isinstance(item, Mapping) and item.get(key):
                out[str(item.get(key))] = dict(item)
        return out
    return {
        "hypotheses": index(registry.get("hypothesis_records"), "hypothesis_id"),
        "derivations": index(registry.get("derivation_records"), "derivation_ref"),
        "evidence": index(registry.get("evidence_records"), "evidence_id"),
        "interventions": index(registry.get("intervention_records"), "intervention_ref"),
    }


def _list_refs(value: Any, field_name: str, pattern: re.Pattern[str]) -> List[str]:
    if not isinstance(value, Mapping):
        return []
    raw = value.get(field_name)
    out: List[str] = []
    if isinstance(raw, list):
        for item in raw:
            text = str(item)
            if pattern.fullmatch(text):
                out.append(text)
    elif isinstance(raw, str):
        out.extend(pattern.findall(raw))
    return sorted(dict.fromkeys(out))


def _first_ref(value: Any, field_name: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, Mapping):
        return ""
    raw = str(value.get(field_name) or "")
    if pattern.fullmatch(raw):
        return raw
    refs = pattern.findall(raw)
    return refs[0] if refs else ""


def _collect_registry_refs(value: Any) -> Tuple[List[str], List[str], List[str], List[str]]:
    text_parts = [str(v) for v in _walk_values(value) if v is not None]
    text = " ".join(text_parts)
    return (
        sorted(set(H_ID_RE.findall(text))),
        sorted(set(D_REF_RE.findall(text))),
        sorted(set(E_REF_RE.findall(text))),
        sorted(set(I_REF_RE.findall(text))),
    )


def _assessment_refs(value: Any) -> Tuple[str, str, List[str], List[str]]:
    return (
        _first_ref(value, "hypothesis_id", H_ID_RE),
        _first_ref(value, "derivation_ref", D_REF_RE),
        _list_refs(value, "evidence_refs", E_REF_RE),
        _list_refs(value, "intervention_refs", I_REF_RE),
    )


def _raw_output_dict(output: CERAOutput) -> Dict[str, Any]:
    raw = output.raw if isinstance(output.raw, Mapping) else {}
    return dict(raw) if raw else output.to_dict()


def validate_cera_output_v3(raw_output: Any, packet: Any) -> ValidatorResult:
    output, parse_error = _parse_output(raw_output)
    if output is None:
        return _finish("", [parse_error or "json_parse_error"], None, [], [], {})

    reasons: List[str] = []
    notes: List[str] = []
    decision = output.decision.strip().upper()
    output.decision = decision
    if decision not in VALID_DECISIONS:
        reasons.append("invalid_decision")

    raw = _raw_output_dict(output)
    if _contains_gold_leak(raw):
        reasons.append("out_of_packet_value")

    contrast = _compact_contrast_v3(packet)
    if not contrast:
        reasons.append("missing_compact_contrast_v3")
        return _finish(decision, sorted(set(reasons), key=reasons.index), output, [], [], {}, notes)

    states = contrast.get("states") if isinstance(contrast.get("states"), Mapping) else {}
    original = contrast.get("original_hypothesis") if isinstance(contrast.get("original_hypothesis"), Mapping) else {}
    alternative = contrast.get("alternative_hypothesis") if isinstance(contrast.get("alternative_hypothesis"), Mapping) else {}
    alternatives = contrast.get("alternative_hypotheses") if isinstance(contrast.get("alternative_hypotheses"), list) else []
    indexes = _registry_indexes(contrast)

    cited_h, cited_d, cited_e, cited_i = _collect_registry_refs(raw)
    if any(ref not in indexes["hypotheses"] for ref in cited_h):
        reasons.append("missing_hypothesis_reference")
    if any(ref not in indexes["derivations"] for ref in cited_d):
        reasons.append("missing_derivation_reference")
    if any(ref not in indexes["evidence"] for ref in cited_e):
        reasons.append("missing_evidence_reference")
    if any(ref not in indexes["interventions"] for ref in cited_i):
        reasons.append("missing_intervention_reference")

    original_h, original_d, original_e, original_i = _assessment_refs(output.original_assessment)
    alternative_h, alternative_d, alternative_e, alternative_i = _assessment_refs(output.alternative_assessment)
    if original_h and original.get("hypothesis_id") and original_h != original.get("hypothesis_id"):
        reasons.append("role_citation_mismatch")
    if alternative_h and alternative.get("hypothesis_id") and alternative_h != alternative.get("hypothesis_id"):
        reasons.append("role_citation_mismatch")
    if original_d and original.get("derivation_ref") and original_d != original.get("derivation_ref"):
        reasons.append("role_citation_mismatch")
    if alternative_d and alternative.get("derivation_ref") and alternative_d != alternative.get("derivation_ref"):
        reasons.append("role_citation_mismatch")
    if original_e and not set(original_e).issubset(set(original.get("evidence_refs") or [])):
        reasons.append("role_citation_mismatch")
    if alternative_e and not set(alternative_e).issubset(set(alternative.get("evidence_refs") or [])):
        reasons.append("role_citation_mismatch")

    separating_refs = _list_refs(raw, "separating_intervention_refs", I_REF_RE)
    if not separating_refs:
        separating_refs = list(output.separating_intervention_refs) if isinstance(output.separating_intervention_refs, list) else []
        separating_refs = [str(item) for item in separating_refs if I_REF_RE.fullmatch(str(item))]
    cited_intervention_refs = sorted(set(separating_refs + original_i + alternative_i))

    if decision == "USE_REPAIRED":
        if not bool(states.get("repair_eligible", False)):
            reasons.append("repair_not_eligible")
        if len(alternatives) != 1 or not alternative:
            reasons.append("noncompact_alternative_hypotheses")
        if not original_h:
            reasons.append("missing_original_hypothesis_reference")
        if not original_d:
            reasons.append("missing_original_derivation_reference")
        if not original_e:
            reasons.append("missing_original_evidence_reference")
        if not original_i:
            reasons.append("missing_original_intervention_reference")
        if not alternative_h:
            reasons.append("missing_alternative_hypothesis_reference")
        if not alternative_d:
            reasons.append("missing_alternative_derivation_reference")
        if not alternative_e:
            reasons.append("missing_alternative_evidence_reference")
        if not alternative_i:
            reasons.append("missing_alternative_intervention_reference")
        if not output.chosen_hypothesis_id:
            reasons.append("missing_chosen_hypothesis_id")
        elif alternative.get("hypothesis_id") and output.chosen_hypothesis_id != alternative.get("hypothesis_id"):
            reasons.append("chosen_hypothesis_mismatch")
        if not str(output.final_answer).strip():
            reasons.append("empty_final_answer")
        elif alternative.get("executed_answer") and not inference_answers_equivalent(output.final_answer, alternative.get("executed_answer")):
            reasons.append("final_answer_mismatch")

        chosen_derivation = indexes["derivations"].get(str(alternative.get("derivation_ref", "")), {})
        if bool(chosen_derivation.get("fallback_only", False)):
            reasons.append("fallback_only_evidence")

        if not separating_refs:
            reasons.append("missing_separating_intervention_reference")
        else:
            usable = False
            for ref in separating_refs:
                record = indexes["interventions"].get(ref) or {}
                if bool(record.get("separating", False)) and bool(record.get("evaluable_on_both_sides", False)):
                    usable = True
            if not usable:
                reasons.append("unseparating_intervention_reference")

        final_norm_values = {
            str(original.get("executed_answer", "")),
            str(alternative.get("executed_answer", "")),
        }
        if output.final_answer and not any(inference_answers_equivalent(output.final_answer, item) for item in final_norm_values if item):
            reasons.append("out_of_packet_value")

    if decision == "KEEP_ORIGINAL" and output.chosen_hypothesis_id and original.get("hypothesis_id"):
        if output.chosen_hypothesis_id != original.get("hypothesis_id"):
            reasons.append("chosen_hypothesis_mismatch")

    validator = _finish(
        decision,
        sorted(set(reasons), key=reasons.index),
        output,
        [],
        [],
        {},
        notes,
    )
    parsed = dict(validator.parsed_output)
    parsed["cited_registry_refs"] = {
        "hypothesis_ids": cited_h,
        "derivation_refs": cited_d,
        "evidence_refs": cited_e,
        "intervention_refs": cited_i,
        "separating_intervention_refs": separating_refs,
        "all_cited_intervention_refs": cited_intervention_refs,
    }
    validator.parsed_output = parsed
    return validator


def validate_cera_output(
    raw_output: Any,
    packet: Any,
    *,
    require_derivation_program: bool = True,
    require_counterfactual_reference: bool = True,
    allow_support_only: bool = False,
    allow_outside_evidence_support: bool = False,
) -> ValidatorResult:
    output, parse_error = _parse_output(raw_output)
    if output is None:
        return _finish("", [parse_error or "json_parse_error"], None, [], [], {})

    reasons: List[str] = []
    notes: List[str] = []
    decision = output.decision.strip().upper()
    output.decision = decision
    if decision not in VALID_DECISIONS:
        reasons.append("invalid_decision")

    support_ids, original_support_ids, cf_ids, support_index, cf_index = _known_ids(packet)
    all_support_ids = support_ids | original_support_ids
    cited_s, cited_cf = _collect_citations(output.to_dict())
    if any(eid not in all_support_ids for eid in cited_s):
        reasons.append("missing_evidence_reference")
    if any(cfid not in cf_ids for cfid in cited_cf):
        reasons.append("missing_counterfactual_reference")

    if _contains_gold_leak(output.to_dict()):
        reasons.append("out_of_packet_value")

    original_role_ids = _role_evidence_ids(output.original_defense)
    candidate_role_ids = _role_evidence_ids(output.candidate_case)
    cf_role_ids = _role_cf_ids(output.counterfactual_assessment)
    if any(eid not in original_support_ids for eid in original_role_ids):
        reasons.append("role_citation_mismatch")
    if any(eid not in support_ids for eid in candidate_role_ids):
        reasons.append("role_citation_mismatch")
    if any(cfid not in cf_ids for cfid in cf_role_ids):
        reasons.append("missing_counterfactual_reference")

    if decision == "USE_REPAIRED" and not str(output.final_answer).strip():
        reasons.append("empty_final_answer")
    packet_payload = _packet_dict(packet)
    if (
        decision == "USE_REPAIRED"
        and normalize_answer_text(output.final_answer) == normalize_answer_text(packet_payload.get("original_answer", ""))
    ):
        reasons.append("no_op_repair")
    if decision == "USE_REPAIRED" and not candidate_role_ids:
        reasons.append("missing_evidence_reference")
    if (
        decision == "USE_REPAIRED"
        and require_counterfactual_reference
        and not allow_support_only
        and not cf_role_ids
    ):
        reasons.append("missing_counterfactual_reference")
    if (
        decision == "USE_REPAIRED"
        and require_counterfactual_reference
        and not allow_support_only
        and cf_role_ids
        and not any(_counterfactual_observed_available(cfid, cf_index) for cfid in cf_role_ids)
    ):
        reasons.append("unavailable_counterfactual_reference")
    if (
        decision == "USE_REPAIRED"
        and not allow_outside_evidence_support
        and any(_support_outside_evidence(eid, support_index) for eid in candidate_role_ids)
    ):
        reasons.append("outside_evidence_support")

    expression = _derivation_expression(output)
    dsl_result: Dict[str, Any] = {}
    if decision == "USE_REPAIRED":
        if require_derivation_program and not expression:
            reasons.append("invalid_dsl")
        elif expression:
            executed = execute_evidence_dsl(expression, packet)
            dsl_result = executed.to_dict()
            if not executed.ok:
                reasons.append("invalid_dsl")
            elif normalize_answer_text(output.final_answer) != executed.normalized_result:
                reasons.append("dsl_execution_mismatch")
            dsl_ids = evidence_ids_for_expression(expression)
            declared_ids = _list_field_ids(output.derivation_program, "evidence_ids", EVIDENCE_ID_RE)
            if not declared_ids or sorted(declared_ids) != sorted(dsl_ids):
                reasons.append("dsl_evidence_id_mismatch")
        elif not require_derivation_program:
            notes.append("derivation_program_not_required")

        allowed = _allowed_answer_values(packet, dsl_result)
        final_norm = normalize_answer_text(output.final_answer)
        if final_norm and allowed and final_norm not in allowed:
            reasons.append("out_of_packet_value")

        if not support_ids:
            reasons.append("unsupported_use_repaired")

    return _finish(decision, sorted(set(reasons), key=reasons.index), output, cited_s, cited_cf, dsl_result, notes)
