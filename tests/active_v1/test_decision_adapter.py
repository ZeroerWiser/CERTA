import json
import unittest
from dataclasses import dataclass
from pathlib import Path

import jsonschema

from certa.active_v1.answer_authority import active_answer_hash
from certa.active_v1.decision_adapter import (
    DecisionEligibility, assess_decision_eligibility,
    materialize_selected_final, reconcile_cera_decision,
)
from certa.grounding.support_partition import SupportPartition
from certa.repair.safety_validator import ValidatorResult, validate_cera_output_v3
from certa.reproducibility.canonical_json import canonical_json_hash


SCHEMAS = Path("/home/hsh/ME/Table/EMNLP2026/certa_goal_packs/"
               "CERTA_ACTIVE_V1_FINAL_METHOD_GOAL_REVISED_PACK/schemas")


@dataclass(frozen=True)
class Derivation:
    derivation_id: str
    typed_signature: str
    projected_answer: str


def _hypothesis(side, number, derivation, evidence):
    return {
        "hypothesis_id": f"H{number}", "side": side,
        "derivation_ref": f"D{number}", "derivation_id": derivation.derivation_id,
        "executed_answer": derivation.projected_answer, "evidence_refs": [evidence],
    }


def fixture():
    original = Derivation("D-ORIGINAL", "COUNT_SCALAR", "2")
    alternative = Derivation("D-ALTERNATIVE", "COUNT_SCALAR", "3")
    oh = _hypothesis("original", 1, original, "E1")
    ah = _hypothesis("alternative", 2, alternative, "E2")
    intervention = {"intervention_ref": "I1", "evaluable_on_both_sides": True,
                    "separating": True}
    contrast = {
        "contrast_version": "compact_behavioral_contrast_v3",
        "states": dict.fromkeys(("contrast_constructible", "contrast_registry_complete",
                                  "contrast_compact", "repair_eligible"), True),
        "original_hypothesis": oh, "alternative_hypothesis": ah,
        "alternative_hypotheses": [ah], "separating_interventions": [intervention],
        "unknowns": [],
        "registry": {
            "hypothesis_records": [
                {k: h[k] for k in ("hypothesis_id", "side", "derivation_ref")}
                for h in (oh, ah)
            ],
            "derivation_records": [
                {k: h[k] for k in ("derivation_ref", "hypothesis_id", "derivation_id",
                                    "executed_answer")}
                for h in (oh, ah)
            ],
            "evidence_records": [{"evidence_id": x} for x in ("E1", "E2")],
            "intervention_records": [intervention],
        },
    }
    answer_hash = active_answer_hash("3")
    raw_derivation = {
        "schema_version": "certa_active_derivation_record_v2", "fixture_only": True,
        "sample_id": "S1", "arm": "C2_ROLE_RETRIEVAL",
        "derivation_id": alternative.derivation_id, "plan_id": "P-ALT",
        "binding_id": "B-ALT", "side": "ALTERNATIVE", "signature_id": "COUNT_SCALAR",
        "answer_role": "SCALAR", "projection": "SCALAR_RESULT_PROJECTION",
        "canonical_program_id": "CP-ALT", "answer_class_id": f"AC-{answer_hash[:24]}",
        "projected_answer_hash": answer_hash, "operand_node_ids": ["N1"],
        "provenance_ids": ["N1", "E1"], "execution_status": "EXECUTED",
        "projection_status": "VALID",
    }
    registry = {
        "schema_version": "certa_active_registry_entry_v2", "fixture_only": True,
        **{k: raw_derivation[k] for k in (
            "sample_id", "arm", "derivation_id", "side", "canonical_program_id",
            "answer_class_id", "provenance_ids",
        )},
        "answer_hash": answer_hash,
    }
    registry["registry_entry_id"] = f"REG-{canonical_json_hash(registry, 24)}"
    assessment = lambda h, e: {
        "hypothesis_id": h, "derivation_ref": f"D{h[1:]}",
        "evidence_refs": [e], "intervention_refs": ["I1"],
    }
    output = {
        "decision": "USE_REPAIRED", "chosen_hypothesis_id": "H2", "final_answer": "3.0",
        "original_assessment": assessment("H1", "E1"),
        "alternative_assessment": assessment("H2", "E2"),
        "separating_intervention_refs": ["I1"],
    }
    validator = validate_cera_output_v3(output, {"compact_behavioral_contrast_v3": contrast})
    assert validator.accepted, validator.reject_reasons
    partition = SupportPartition("NUMERIC_EXACT_CANONICAL:2", (original,),
                                 (alternative,), True, True)
    return locals()


def eligibility(f, active=("COUNT_SCALAR",), partition=None, derivations=None):
    return assess_decision_eligibility(
        role_id="COUNT_SCALAR", decision_active_role_ids=active,
        support_partition=partition or f["partition"], compact_contrast=f["contrast"],
        executed_derivations=derivations or (f["original"], f["alternative"]),
    )


class DecisionEligibilityTests(unittest.TestCase):
    def test_exact_conjunction_allows_only_active_registry_complete_paired_contrast(self):
        result = eligibility(fixture())
        self.assertTrue(result.eligible, result.failure_reasons)
        self.assertTrue(result.cera_call_allowed)
        self.assertEqual(result.fallback, "B0_KEEP")

    def test_inactive_missing_partition_external_and_role_mismatch_fail_closed(self):
        f = fixture()
        empty_alt = SupportPartition("k", (f["original"],), (), True, True)
        wrong = Derivation(f["alternative"].derivation_id, "SUM_SCALAR", "3")
        cases = (
            ({"active": ()}, "decision_inactive_role"),
            ({"partition": empty_alt}, "partition_alternative_missing"),
            ({"derivations": (f["original"],)}, "registry_derivation_not_executed"),
            ({"derivations": (f["original"], wrong)}, "role_signature_mismatch"),
        )
        for kwargs, reason in cases:
            with self.subTest(reason=reason):
                result = eligibility(f, **kwargs)
                self.assertFalse(result.eligible)
                self.assertFalse(result.cera_call_allowed)
                self.assertIn(reason, result.failure_reasons)


class DecisionReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.f = fixture()
        self.eligible = eligibility(self.f)

    def reconcile(self, **changes):
        f = self.f
        values = dict(
            eligibility=self.eligible, raw_output=f["output"], validator=f["validator"],
            compact_contrast=f["contrast"], executed_derivations=(f["original"], f["alternative"]),
            raw_derivation_records=[f["raw_derivation"]], registry_entries=[f["registry"]],
            b0_answer="2", sample_id="S1", decision_id="DEC-S1",
            validator_record_id="VAL-S1", created_at="2026-07-23T00:00:00+00:00",
            fixture_only=True,
        )
        return reconcile_cera_decision(**(values | changes))

    def assert_fallback(self, resolution, reason):
        decision = resolution.decision_record
        self.assertEqual(decision["action"], "KEEP_B0")
        self.assertTrue(all(decision[k] is None for k in (
            "selected_derivation_id", "selected_registry_entry_id", "selected_answer_hash")))
        self.assertEqual(resolution.selected_answer, "2")
        self.assertIn(reason, resolution.failure_reasons)

    def test_validated_registered_executed_alternative_is_only_registry_selection(self):
        f, result = self.f, self.reconcile()
        d = result.decision_record
        self.assertEqual((d["action"], d["selected_hypothesis_id"], result.selected_answer),
                         ("USE_REGISTRY", f["raw_derivation"]["answer_class_id"], "3"))
        self.assertEqual(d["selected_registry_entry_id"], f["registry"]["registry_entry_id"])
        self.assertTrue(result.validator_record["accepted"] and result.reconciliation_record["valid"])
        records = ((d, "DECISION_RECORD_SCHEMA.json"),
                   (result.validator_record, "VALIDATOR_RECORD_SCHEMA.json"),
                   (result.reconciliation_record, "REGISTRY_SELECTED_FINAL_RECONCILIATION_SCHEMA.json"),
                   (f["raw_derivation"], "RAW_DERIVATION_RECORD_SCHEMA.json"),
                   (f["registry"], "REGISTRY_ENTRY_SCHEMA.json"))
        for record, name in records:
            jsonschema.validate(record, json.loads((SCHEMAS / name).read_text()))

    def test_unregistered_unexecuted_validator_rejected_and_ineligible_cannot_select(self):
        bad_validator = ValidatorResult(False, decision="USE_REPAIRED", reject_reason="nope")
        other = Derivation("D-OTHER", "COUNT_SCALAR", "3")
        cases = (({"registry_entries": []}, "registry_entry_missing"),
                 ({"executed_derivations": (self.f["original"], other)}, "selected_derivation_not_executed"),
                 ({"validator": bad_validator}, "validator_rejected"),
                 ({"eligibility": DecisionEligibility(False, False, ("decision_inactive_role",))},
                  "decision_inactive_role"))
        for changes, reason in cases:
            with self.subTest(reason=reason):
                self.assert_fallback(self.reconcile(**changes), reason)

    def test_external_or_ambiguous_references_fail_closed(self):
        f = self.f
        bad_validator = ValidatorResult(True, decision="USE_REPAIRED",
                                        parsed_output=dict(f["output"], chosen_hypothesis_id="H999"))
        cases = (({"raw_output": dict(f["output"], chosen_hypothesis_id="H999")},
                  "chosen_hypothesis_not_unique_alternative"),
                 ({"registry_entries": [f["registry"], f["registry"]]}, "registry_entry_ambiguous"),
                 ({"validator": bad_validator}, "validator_output_mismatch"),
                 ({"registry_entries": [dict(f["registry"], registry_entry_id="")]},
                  "registry_entry_id_mismatch"))
        for changes, reason in cases:
            with self.subTest(reason=reason):
                self.assert_fallback(self.reconcile(**changes), reason)

    def test_accepted_keep_and_insufficient_have_no_pack_validator_reference(self):
        f = self.f
        for label, chosen, action in (("KEEP_ORIGINAL", "H1", "KEEP_B0"),
                                      ("INSUFFICIENT_CERTIFICATE", "", "INSUFFICIENT")):
            raw = dict(f["output"], decision=label, chosen_hypothesis_id=chosen, final_answer="")
            validator = validate_cera_output_v3(raw, {"compact_behavioral_contrast_v3": f["contrast"]})
            result = self.reconcile(raw_output=raw, validator=validator)
            self.assertEqual(result.decision_record["action"], action)
            self.assertIsNone(result.decision_record["validator_record_id"])
            self.assertIsNone(result.validator_record)
            self.assertTrue(all(result.reconciliation_record[k] for k in (
                "selected_derivation_exists", "selected_registry_entry_exists", "alternative_side",
                "answer_hash_match", "validator_reference_match", "validator_accepted",
                "materialization_match", "valid")))

    def test_outer_registry_must_exactly_match_raw_derivation_authority(self):
        for field, value in (("canonical_program_id", "CP-WRONG"),
                             ("answer_class_id", "AC-WRONG"),
                             ("provenance_ids", ["E-WRONG"])):
            result = self.reconcile(registry_entries=[dict(self.f["registry"], **{field: value})])
            self.assert_fallback(result, next(r for r in result.failure_reasons if field in r))

    def test_materializer_uses_exact_registered_answer_or_b0(self):
        when = "2026-07-23T00:01:00+00:00"
        selected = materialize_selected_final(self.reconcile(), b0_answer="2", materialized_at=when)
        self.assertEqual((selected.answer, selected.record["selected_source"]), ("3", "REGISTRY"))
        jsonschema.validate(selected.record, json.loads(
            (SCHEMAS / "SELECTED_FINAL_RECORD_SCHEMA.json").read_text()))
        rejected = self.reconcile(validator=ValidatorResult(
            False, decision="USE_REPAIRED", reject_reason="nope"))
        kept = materialize_selected_final(rejected, b0_answer="2.0", materialized_at=when)
        self.assertEqual((kept.answer, kept.record["selected_source"],
                          kept.record["selected_final_answer_hash"]),
                         ("2.0", "B0", active_answer_hash("2")))


if __name__ == "__main__":
    unittest.main()
