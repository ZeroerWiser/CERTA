"""Fail-closed Decision adapter for frozen CERTA Active V1 artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from certa.active_v1.answer_authority import active_answer_hash
from certa.active_v1.artifact_authority import reconcile_registry_entry
from certa.derivations.answer_equivalence import inference_answers_equivalent
from certa.grounding.support_partition import SupportPartition
from certa.repair.evidence_packet import CERAOutput
from certa.repair.safety_validator import ValidatorResult


DECISION_ARM = "CERA_PLUS_VALIDATOR"


@dataclass(frozen=True)
class DecisionEligibility:
    eligible: bool
    cera_call_allowed: bool
    failure_reasons: Tuple[str, ...]
    fallback: str = "B0_KEEP"


@dataclass(frozen=True)
class DecisionResolution:
    decision_record: Dict[str, Any]
    validator_record: Optional[Dict[str, Any]]
    reconciliation_record: Dict[str, Any]
    selected_answer: Any
    failure_reasons: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SelectedFinalMaterialization:
    answer: Any
    record: Dict[str, Any]


def _unique(values: Iterable[str]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _contrast_payload(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        payload = value.to_dict()
        return dict(payload) if isinstance(payload, Mapping) else {}
    return {}


def _records(value: Any) -> Tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _matches(records: Sequence[Mapping[str, Any]], field: str, value: str) -> Tuple[Mapping[str, Any], ...]:
    return tuple(item for item in records if str(item.get(field) or "") == value)


def _derivation_map(derivations: Sequence[Any]) -> Tuple[Dict[str, Any], set[str]]:
    out: Dict[str, Any] = {}
    duplicates: set[str] = set()
    for derivation in derivations:
        derivation_id = str(getattr(derivation, "derivation_id", "") or "")
        if not derivation_id:
            continue
        if derivation_id in out:
            duplicates.add(derivation_id)
        out[derivation_id] = derivation
    return out, duplicates


def _contrast_reference_failures(
    contrast: Mapping[str, Any],
    executed_derivations: Sequence[Any],
    *,
    original_ids: Optional[set[str]] = None,
    alternative_ids: Optional[set[str]] = None,
) -> Tuple[str, ...]:
    failures = []
    registry = contrast.get("registry") if isinstance(contrast.get("registry"), Mapping) else {}
    hypotheses = _records(registry.get("hypothesis_records"))
    derivation_records = _records(registry.get("derivation_records"))
    evidence = _records(registry.get("evidence_records"))
    interventions = _records(registry.get("intervention_records"))
    if not hypotheses or not derivation_records or not evidence or not interventions:
        failures.append("registry_records_incomplete")

    executed, duplicate_ids = _derivation_map(executed_derivations)
    if duplicate_ids:
        failures.append("executed_derivation_id_ambiguous")

    for side, field_name, side_ids in (
        ("original", "original_hypothesis", original_ids),
        ("alternative", "alternative_hypothesis", alternative_ids),
    ):
        hypothesis = contrast.get(field_name)
        if not isinstance(hypothesis, Mapping):
            failures.append(f"{side}_hypothesis_missing")
            continue
        hypothesis_id = str(hypothesis.get("hypothesis_id") or "")
        derivation_ref = str(hypothesis.get("derivation_ref") or "")
        derivation_id = str(hypothesis.get("derivation_id") or "")
        if str(hypothesis.get("side") or "") != side:
            failures.append(f"{side}_hypothesis_side_mismatch")
        hypothesis_rows = _matches(hypotheses, "hypothesis_id", hypothesis_id)
        if len(hypothesis_rows) != 1:
            failures.append(f"{side}_hypothesis_registry_ambiguous")
        elif (
            str(hypothesis_rows[0].get("side") or "") != side
            or str(hypothesis_rows[0].get("derivation_ref") or "") != derivation_ref
        ):
            failures.append(f"{side}_hypothesis_registry_mismatch")
        derivation_rows = _matches(derivation_records, "derivation_ref", derivation_ref)
        if len(derivation_rows) != 1:
            failures.append(f"{side}_derivation_registry_ambiguous")
        else:
            record = derivation_rows[0]
            if (
                str(record.get("hypothesis_id") or "") != hypothesis_id
                or str(record.get("derivation_id") or "") != derivation_id
            ):
                failures.append(f"{side}_derivation_registry_mismatch")
            executed_derivation = executed.get(derivation_id)
            if executed_derivation is None:
                failures.append("registry_derivation_not_executed")
            else:
                executed_answer = getattr(executed_derivation, "projected_answer", None)
                if not inference_answers_equivalent(record.get("executed_answer"), executed_answer):
                    failures.append(f"{side}_registry_answer_mismatch")
                if not inference_answers_equivalent(hypothesis.get("executed_answer"), executed_answer):
                    failures.append(f"{side}_hypothesis_answer_mismatch")
        if side_ids is not None and derivation_id not in side_ids:
            failures.append(f"{side}_partition_reference_mismatch")
    return _unique(failures)


def assess_decision_eligibility(
    *,
    role_id: str,
    decision_active_role_ids: Iterable[str],
    support_partition: SupportPartition,
    compact_contrast: Any,
    executed_derivations: Sequence[Any],
) -> DecisionEligibility:
    """Apply the frozen blind-eligibility conjunction without calling CERA."""
    failures = []
    active_ids = {str(item) for item in decision_active_role_ids}
    if role_id not in active_ids:
        failures.append("decision_inactive_role")

    original = tuple(getattr(support_partition, "original_support", ()) or ())
    alternative = tuple(getattr(support_partition, "alternative_support", ()) or ())
    if not original:
        failures.append("partition_original_missing")
    if not alternative:
        failures.append("partition_alternative_missing")
    if not bool(getattr(support_partition, "disjoint", False)):
        failures.append("partition_not_disjoint")
    if not bool(getattr(support_partition, "exhaustive", False)):
        failures.append("partition_not_exhaustive")

    original_ids = {str(getattr(item, "derivation_id", "") or "") for item in original}
    alternative_ids = {str(getattr(item, "derivation_id", "") or "") for item in alternative}
    if original_ids & alternative_ids:
        failures.append("partition_derivation_overlap")

    executed, _duplicates = _derivation_map(executed_derivations)
    for derivation_id in sorted((original_ids | alternative_ids) - set(executed)):
        if derivation_id:
            failures.append("registry_derivation_not_executed")
    for derivation_id in sorted((original_ids | alternative_ids) & set(executed)):
        if str(getattr(executed[derivation_id], "typed_signature", "") or "") != role_id:
            failures.append("role_signature_mismatch")

    contrast = _contrast_payload(compact_contrast)
    states = contrast.get("states") if isinstance(contrast.get("states"), Mapping) else {}
    for state, reason in (
        ("contrast_registry_complete", "contrast_registry_incomplete"),
        ("contrast_compact", "contrast_not_compact"),
        ("repair_eligible", "contrast_not_repair_eligible"),
    ):
        if states.get(state) is not True:
            failures.append(reason)
    alternatives = contrast.get("alternative_hypotheses")
    if not isinstance(alternatives, list) or len(alternatives) != 1:
        failures.append("paired_contrast_alternative_not_unique")
    separating = contrast.get("separating_interventions")
    if not isinstance(separating, list) or not any(
        isinstance(item, Mapping)
        and item.get("separating") is True
        and item.get("evaluable_on_both_sides") is True
        for item in separating
    ):
        failures.append("paired_contrast_separation_missing")
    if contrast.get("unknowns") not in ([], ()):
        failures.append("paired_contrast_has_unknowns")
    failures.extend(_contrast_reference_failures(
        contrast,
        executed_derivations,
        original_ids=original_ids,
        alternative_ids=alternative_ids,
    ))
    reasons = _unique(failures)
    eligible = not reasons
    return DecisionEligibility(eligible, eligible, reasons)


def _parse_cera_output(raw: Any) -> Optional[CERAOutput]:
    if isinstance(raw, CERAOutput):
        return raw
    if isinstance(raw, Mapping):
        return CERAOutput.from_dict(raw)
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            return None
        if isinstance(payload, Mapping):
            return CERAOutput.from_dict(payload)
    return None


def _validator_fields(value: Any) -> Tuple[bool, str, Mapping[str, Any]]:
    if isinstance(value, ValidatorResult):
        return bool(value.accepted), str(value.decision or ""), value.parsed_output
    if isinstance(value, Mapping):
        parsed = value.get("parsed_output")
        parsed_output = parsed if isinstance(parsed, Mapping) else {}
        return bool(value.get("accepted")), str(value.get("decision") or ""), parsed_output
    return False, "", {}


def reconcile_cera_decision(
    *,
    eligibility: DecisionEligibility,
    raw_output: Any,
    validator: Any,
    compact_contrast: Any,
    executed_derivations: Sequence[Any],
    raw_derivation_records: Sequence[Mapping[str, Any]],
    registry_entries: Sequence[Mapping[str, Any]],
    b0_answer: Any,
    sample_id: str,
    decision_id: str,
    validator_record_id: str,
    created_at: str,
    fixture_only: bool = False,
    decision_before_gold: bool = True,
) -> DecisionResolution:
    """Reconcile an existing CERA/validator result to executed registry authority."""
    failures = list(eligibility.failure_reasons if not eligibility.eligible else ())
    output = _parse_cera_output(raw_output)
    validator_accepted, validator_decision, validator_parsed = _validator_fields(validator)
    selected_answer: Any = b0_answer
    selected_hypothesis_id = None
    selected_derivation_id = None
    selected_registry_entry_id = None
    selected_answer_hash = None
    action = "KEEP_B0"

    if not eligibility.eligible:
        failures.append("decision_ineligible")
    elif output is None:
        failures.append("cera_output_unparseable")
    else:
        raw_decision = str(output.decision or "").upper()
        if not validator_accepted:
            failures.append("validator_rejected")
        if validator_decision.upper() != raw_decision:
            failures.append("validator_decision_mismatch")
        if validator_parsed:
            parsed_decision = str(validator_parsed.get("decision") or "").upper()
            parsed_hypothesis = str(validator_parsed.get("chosen_hypothesis_id") or "")
            parsed_answer = validator_parsed.get("final_answer")
            if (
                (parsed_decision and parsed_decision != raw_decision)
                or (parsed_hypothesis and parsed_hypothesis != str(output.chosen_hypothesis_id or ""))
                or (
                    parsed_answer not in (None, "")
                    and not inference_answers_equivalent(parsed_answer, output.final_answer)
                )
            ):
                failures.append("validator_output_mismatch")

        if raw_decision == "INSUFFICIENT_CERTIFICATE" and not failures:
            action = "INSUFFICIENT"
        elif raw_decision == "USE_REPAIRED":
            contrast = _contrast_payload(compact_contrast)
            alternative = contrast.get("alternative_hypothesis")
            alternatives = contrast.get("alternative_hypotheses")
            if (
                not isinstance(alternative, Mapping)
                or not isinstance(alternatives, list)
                or len(alternatives) != 1
                or str(output.chosen_hypothesis_id or "") != str(alternative.get("hypothesis_id") or "")
            ):
                failures.append("chosen_hypothesis_not_unique_alternative")
            if isinstance(alternative, Mapping):
                alternative_id = str(alternative.get("derivation_id") or "")
                executed_ids = {
                    str(getattr(item, "derivation_id", "") or "")
                    for item in executed_derivations
                }
                if alternative_id not in executed_ids:
                    failures.append("selected_derivation_not_executed")
            failures.extend(_contrast_reference_failures(contrast, executed_derivations))
            if not failures:
                derivation_id = str(alternative.get("derivation_id") or "")
                executed, duplicate_ids = _derivation_map(executed_derivations)
                derivation = executed.get(derivation_id)
                if derivation_id in duplicate_ids:
                    failures.append("selected_derivation_ambiguous")
                elif derivation is None:
                    failures.append("selected_derivation_not_executed")
                else:
                    exact_answer = getattr(derivation, "projected_answer", None)
                    if not inference_answers_equivalent(output.final_answer, exact_answer):
                        failures.append("cera_answer_not_executed_answer")
                    entries = tuple(
                        item for item in registry_entries
                        if isinstance(item, Mapping) and str(item.get("derivation_id") or "") == derivation_id
                    )
                    if not entries:
                        failures.append("registry_entry_missing")
                    elif len(entries) != 1:
                        failures.append("registry_entry_ambiguous")
                    else:
                        entry = entries[0]
                        expected_hash = active_answer_hash(exact_answer)
                        raw_records = tuple(
                            item for item in raw_derivation_records
                            if isinstance(item, Mapping)
                            and str(item.get("derivation_id") or "") == derivation_id
                        )
                        if not raw_records:
                            failures.append("raw_derivation_missing")
                        elif len(raw_records) != 1:
                            failures.append("raw_derivation_ambiguous")
                        else:
                            raw_derivation = raw_records[0]
                            if raw_derivation.get("sample_id") != sample_id:
                                failures.append("raw_derivation_sample_mismatch")
                            if raw_derivation.get("arm") != "C2_ROLE_RETRIEVAL":
                                failures.append("raw_derivation_arm_mismatch")
                            if raw_derivation.get("side") != "ALTERNATIVE":
                                failures.append("raw_derivation_not_alternative")
                            if raw_derivation.get("signature_id") != getattr(
                                derivation, "typed_signature", None
                            ):
                                failures.append("raw_derivation_signature_mismatch")
                            if raw_derivation.get("projected_answer_hash") != expected_hash:
                                failures.append("raw_derivation_answer_hash_mismatch")
                            try:
                                reconcile_registry_entry(entry, raw_derivation)
                            except ValueError as exc:
                                failures.append(str(exc))
                        if not failures:
                            action = "USE_REGISTRY"
                            selected_answer = exact_answer
                            selected_hypothesis_id = str(
                                raw_derivation.get("answer_class_id") or ""
                            )
                            selected_derivation_id = derivation_id
                            selected_registry_entry_id = str(entry.get("registry_entry_id") or "")
                            selected_answer_hash = expected_hash
        elif raw_decision not in {"KEEP_ORIGINAL", "INSUFFICIENT_CERTIFICATE", "USE_REPAIRED"}:
            failures.append("cera_decision_invalid")

    if failures:
        action = "KEEP_B0"
        selected_answer = b0_answer
        selected_hypothesis_id = None
        selected_derivation_id = None
        selected_registry_entry_id = None
        selected_answer_hash = None

    b0_hash = active_answer_hash(b0_answer)
    final_hash = selected_answer_hash or b0_hash
    selected = action == "USE_REGISTRY"
    has_pack_validator = selected and validator is not None
    validator_id = validator_record_id if has_pack_validator else None
    decision_record = {
        "schema_version": "certa_active_decision_record_v2",
        "fixture_only": bool(fixture_only),
        "sample_id": sample_id,
        "decision_id": decision_id,
        "decision_arm": DECISION_ARM,
        "action": action,
        "selected_hypothesis_id": selected_hypothesis_id,
        "selected_derivation_id": selected_derivation_id,
        "selected_registry_entry_id": selected_registry_entry_id,
        "selected_answer_hash": selected_answer_hash,
        "validator_record_id": validator_id,
        "proposed_final_answer_hash": final_hash,
        "created_at": created_at,
    }
    validator_record = None
    if has_pack_validator:
        validator_record = {
            "schema_version": "certa_active_validator_record_v2",
            "fixture_only": bool(fixture_only),
            "sample_id": sample_id,
            "validator_record_id": validator_record_id,
            "decision_id": decision_id,
            "derivation_id": selected_derivation_id,
            "registry_entry_id": selected_registry_entry_id,
            "answer_hash": final_hash,
            "accepted": bool(validator_accepted),
            "created_at": created_at,
        }
    validator_reference_match = (
        (validator_record is None and validator_id is None)
        or (validator_record is not None and validator_record["validator_record_id"] == validator_id)
    )
    selected_derivation_exists = bool(selected_derivation_id) if selected else selected_derivation_id is None
    selected_registry_exists = (
        bool(selected_registry_entry_id) if selected else selected_registry_entry_id is None
    )
    answer_hash_match = (
        final_hash == active_answer_hash(selected_answer)
        if selected else selected_answer_hash is None
    )
    validator_ok = bool(validator_accepted) if selected else True
    reconciliation_record = {
        "schema_version": "certa_active_selected_final_reconciliation_v2",
        "sample_id": sample_id,
        "decision_id": decision_id,
        "decision_arm": DECISION_ARM,
        "decision_before_gold": bool(decision_before_gold),
        "selected_derivation_exists": selected_derivation_exists,
        "selected_registry_entry_exists": selected_registry_exists,
        "alternative_side": True,
        "answer_hash_match": answer_hash_match,
        "validator_reference_match": validator_reference_match,
        "validator_accepted": validator_ok,
        "materialization_match": True,
        "valid": bool(
            decision_before_gold
            and selected_derivation_exists
            and selected_registry_exists
            and answer_hash_match
            and validator_reference_match
            and validator_ok
        ),
    }
    return DecisionResolution(
        decision_record=decision_record,
        validator_record=validator_record,
        reconciliation_record=reconciliation_record,
        selected_answer=selected_answer,
        failure_reasons=_unique(failures),
    )


def materialize_selected_final(
    resolution: DecisionResolution,
    *,
    b0_answer: Any,
    materialized_at: str,
) -> SelectedFinalMaterialization:
    """Materialize only the reconciled registry answer or the caller's frozen B0."""
    decision = resolution.decision_record
    b0_hash = active_answer_hash(b0_answer)
    use_registry = decision.get("action") == "USE_REGISTRY"
    if use_registry:
        answer = resolution.selected_answer
        answer_hash = active_answer_hash(answer)
        if (
            decision.get("selected_answer_hash") != answer_hash
            or decision.get("proposed_final_answer_hash") != answer_hash
            or not decision.get("selected_derivation_id")
            or not decision.get("selected_registry_entry_id")
        ):
            raise ValueError("selected_registry_authority_mismatch")
        source = "REGISTRY"
    else:
        answer = b0_answer
        answer_hash = b0_hash
        if decision.get("proposed_final_answer_hash") != b0_hash:
            raise ValueError("b0_authority_mismatch")
        source = "B0"
    record = {
        "schema_version": "certa_active_selected_final_record_v2",
        "fixture_only": bool(decision.get("fixture_only", False)),
        "sample_id": str(decision.get("sample_id") or ""),
        "decision_arm": DECISION_ARM,
        "decision_id": str(decision.get("decision_id") or ""),
        "b0_answer_hash": b0_hash,
        "selected_final_answer_hash": answer_hash,
        "selected_source": source,
        "materialized_at": materialized_at,
    }
    return SelectedFinalMaterialization(answer=answer, record=record)
