from copy import deepcopy

import pytest

from certa.active_v1.answer_authority import active_answer_hash
from tests.active_v1.test_assignment_level_grounding_authority import ROLE_SHA, _assignment, _bundle
from tools.compute_certa_active_constructor_gate_v3 import SAFETY_KEYS, THRESHOLDS, compute_gate, evaluate_thresholds


ARMS = ("C0_SCHEMA_ONLY", "C1_ROLE_ONLY", "C2_ROLE_RETRIEVAL")


def _identity(arm):
    return {
        "sample_id": "FX_SAMPLE",
        "table_id": "FX_TABLE",
        "arm": arm,
        "question_sha256": "1" * 64,
        "b0_answer_sha256": active_answer_hash("10"),
        "final_answer_sha256": active_answer_hash("10"),
        "method_sha": "2" * 40,
        "model_profile_sha256": "3" * 64,
        "operation_registry_sha256": "4" * 64,
        "planner_schema_sha256": "5" * 64,
        "closure_sha256": "6" * 64,
        "executor_sha256": "7" * 64,
        "artifact_schema_sha256": "8" * 64,
        "role_record_sha256": "" if arm == "C0_SCHEMA_ONLY" else ROLE_SHA,
        "gold_accessed": False,
        "runtime_leakage": False,
    }


def _paired_inputs():
    bundle = _bundle([_assignment(1, executable=True, answer="10"),
                      _assignment(2, executable=True, answer="11")])
    roles = [
        {
            "sample_id": "FX_SAMPLE",
            "record_sha256": ROLE_SHA,
            "signature": "COUNT_SCALAR",
            "answer_role": "SCALAR",
            "projection": "SCALAR_RESULT_PROJECTION",
        }
    ]
    return ([_identity(arm) for arm in ARMS], roles, list(bundle.raw_groundings),
            list(bundle.raw_derivations), list(bundle.registry_entries))


def _compute(*, ground=None, deriv=None, registry=None):
    identities, roles, base_ground, base_deriv, base_registry = _paired_inputs()
    return compute_gate(identities=identities, role_records=roles,
        groundings=base_ground if ground is None else ground,
        derivations=base_deriv if deriv is None else deriv,
        registry=base_registry if registry is None else registry,
        cost_ledger={"logical_calls": 0, "transport_attempts": 0}, allow_fixture=True)


def test_assignment_authority_yields_one_c1_paired_row_without_singleton():
    result = _compute()
    assert result["arms"]["C1_ROLE_ONLY"]["authorized_grounding_rows"] == 1
    assert result["arms"]["C1_ROLE_ONLY"]["paired_rows"] == 1
    assert result["arms"]["C1_ROLE_ONLY"]["registry_complete_paired_rows"] == 1
    assert result["safety"]["reconciliation_mismatch"] == 0
    assert result["safety"]["ambiguous_assignment_authorized"] == 0
    assert not result["pass"]


def test_derivation_binding_mutation_fails_reconciliation():
    _, _, _, derivations, _ = _paired_inputs()
    derivations[0] = dict(derivations[0], binding_id="B-" + "0" * 24)
    result = _compute(deriv=derivations)
    assert result["safety"]["reconciliation_mismatch"] == 1
    assert "safety" in result["failure_reasons"]


def test_registry_external_and_empty_provenance_fail_closed():
    _, _, _, derivations, registry = _paired_inputs()
    orphan = dict(registry[0], derivation_id="orphan", registry_entry_id="REG-" + "0" * 24)
    result = _compute(registry=registry + [orphan])
    assert result["safety"]["registry_external_contrast"] == 1
    no_provenance = [dict(row, provenance_ids=[]) for row in derivations]
    result = _compute(deriv=no_provenance)
    assert result["safety"]["registry_external_contrast"] >= 1


def test_ambiguous_assignment_can_never_be_authorized():
    _, _, groundings, _, _ = _paired_inputs()
    mutant = deepcopy(groundings)
    hypothesis = mutant[0]["grounding_hypotheses"][0]
    hypothesis["resolution_state"] = "AMBIGUOUS"
    hypothesis["grounding_valid"] = True
    mutant[0]["ambiguous_hypothesis_count"] = 1
    with pytest.raises(ValueError, match="ambiguous_assignment_authorized"):
        _compute(ground=mutant)


def test_duplicate_canonical_program_is_counted_once():
    _, _, groundings, derivations, _ = _paired_inputs()
    mutant_ground = deepcopy(groundings)
    mutant_deriv = deepcopy(derivations)
    first, second = mutant_deriv
    second.update(canonical_program_id=first["canonical_program_id"],
                  projected_answer_hash=first["projected_answer_hash"],
                  answer_class_id=first["answer_class_id"], side=first["side"])
    hypotheses = {row["binding_id"]: row for row in mutant_ground[0]["grounding_hypotheses"]}
    hypotheses[second["binding_id"]]["canonical_program_id"] = first[
        "canonical_program_id"
    ]
    result = _compute(ground=mutant_ground, deriv=mutant_deriv, registry=[])
    assert result["arms"]["C1_ROLE_ONLY"]["executable_derivation_count"] == 1


def test_threshold_contract_and_boundary_mutants():
    assert THRESHOLDS == {"c2_paired_min": 8, "paired_gain_min": 4,
        "c2_registry_complete_paired_min": 6, "registry_gain_min": 3,
        "paired_tables_min": 4, "role_compatible_precision": 1.0}
    effect = {"c2_paired": 8, "paired_gain": 4, "c2_registry_complete_paired": 6,
              "registry_gain": 3, "paired_tables": 4, "role_compatible_precision": 1.0}
    safety = {key: 0 for key in SAFETY_KEYS}
    assert evaluate_thresholds(effect, safety) == []
    reason_by_field = {"c2_paired": "paired_absolute", "paired_gain": "paired_gain",
        "c2_registry_complete_paired": "registry_absolute", "registry_gain": "registry_gain",
        "paired_tables": "table_coverage", "role_compatible_precision": "role_compatible_precision"}
    for field, reason in reason_by_field.items():
        mutant = dict(effect)
        mutant[field] -= 1 if field != "role_compatible_precision" else 0.01
        assert reason in evaluate_thresholds(mutant, safety)
    for key in SAFETY_KEYS:
        mutant = dict(safety, **{key: 1})
        assert evaluate_thresholds(effect, mutant) == ["safety"]
