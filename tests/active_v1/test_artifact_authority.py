import json
import unittest
from dataclasses import replace
from pathlib import Path

import jsonschema

from certa.active_v1.answer_authority import active_answer_hash
from certa.active_v1.artifact_authority import (
    ArtifactContext, reconcile_registry_entry, serialize_plan_closure,
)
from certa.derivations.schema import ExecutableDerivation
from certa.grounding.plan_closure import ClosureOutcome, GroundedAssignment, PlanClosure
from certa.reproducibility.canonical_json import canonical_json, canonical_json_hash


PACK = Path("/home/hsh/ME/Table/EMNLP2026/certa_goal_packs/") / \
    "CERTA_ACTIVE_V1_FINAL_METHOD_GOAL_REVISED_PACK"


def schema(name):
    return json.loads((PACK / "schemas" / name).read_text(encoding="utf-8"))


def successful_pair(plan_id="P0", answer="2", provenance_complete=True):
    program = {
        "answer_domain": "SCALAR", "operation_family": "COUNT", "plan_id": plan_id,
        "projected_answer": answer, "projection_operator": "SCALAR_RESULT_PROJECTION",
        "signature_id": "COUNT_SCALAR",
    }
    executable_program = canonical_json(program)
    program_id = f"CP-{canonical_json_hash(program, 24)}"
    derivation_id = f"D-{canonical_json_hash({'program_id': program_id}, 20)}"
    assignment = GroundedAssignment(
        plan_id=plan_id, plan_ids=(plan_id,), assignment_id="A1",
        assignment_key=f"COUNT:{plan_id}",
        role_bindings={
            "AGGREGATION_SCOPE": (("n1",), ("n2",)), "TARGET_MEASURE": ("m1",),
        },
        outcome=ClosureOutcome.UNIQUE_EXECUTABLE, resolution_state="UNIQUE",
        matched_cell_ids=("n1", "n2"),
        required_edge_triples=(("n1", "MEMBER_OF", "scope"),),
        derivation_id=derivation_id, operation_family="COUNT",
        signature_id="COUNT_SCALAR", semantic_result_role="CARDINALITY",
        projection_operator="SCALAR_RESULT_PROJECTION", answer_domain="SCALAR",
        canonical_program_id=program_id, execution_outcome="EXECUTED",
        projection_outcome="PROJECTED", projected_answer=answer,
    )
    derivation = ExecutableDerivation(
        derivation_id=derivation_id, source_candidate_id=f"closure:{program_id}",
        operation_family="COUNT", operand_node_ids=["n1", "n2"],
        required_edge_triples=[("n1", "MEMBER_OF", "scope")],
        typed_signature="COUNT_SCALAR", projection_operator="SCALAR_RESULT_PROJECTION",
        projected_answer=answer, output_domain="SCALAR", evidence_ids=["edge-1"],
        executable_program=executable_program, provenance_complete=provenance_complete,
        availability="available", operation_metadata={"canonical_program_id": program_id},
    )
    return assignment, derivation


def closure(*pairs):
    return PlanClosure(
        plan_id="CLOSURE-1", operation_family="COUNT",
        assignments=tuple(pair[0] for pair in pairs),
        executable_derivations=tuple(pair[1] for pair in pairs),
        outcome_counts={"UNIQUE_EXECUTABLE": len(pairs)},
        declared_assignment_count=len(pairs), realized_assignment_count=len(pairs),
        deduplicated_program_count=len(pairs), resource_complete=True,
    )


class ArtifactAuthorityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schemas = tuple(schema(name) for name in (
            "RAW_GROUNDING_RECORD_SCHEMA.json", "RAW_DERIVATION_RECORD_SCHEMA.json",
            "REGISTRY_ENTRY_SCHEMA.json",
        ))
        cls.context = ArtifactContext(
            sample_id="S1", table_id="T1", arm="C2_ROLE_RETRIEVAL",
            role_id="COUNT_SCALAR",
        )

    def bundle(self, *pairs, initial_answer="2"):
        return serialize_plan_closure(
            closure(*pairs), context=self.context, initial_answer=initial_answer,
        )

    def test_plan_closure_objects_are_not_pack_artifact_records(self):
        assignment, derivation = successful_pair()
        for value, record_schema in zip(
            (assignment.to_dict(), derivation.to_dict(), derivation.to_dict()), self.schemas,
        ):
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate(value, record_schema)

    def test_serializes_schema_valid_authoritative_records(self):
        bundle = self.bundle(successful_pair())
        records = (
            bundle.raw_groundings[0], bundle.raw_derivations[0], bundle.registry_entries[0],
        )
        self.assertEqual(tuple(map(len, (
            bundle.raw_groundings, bundle.raw_derivations, bundle.registry_entries,
        ))), (1, 1, 1))
        for record, record_schema in zip(records, self.schemas):
            jsonschema.validate(record, record_schema)
        grounding, derivation, registry = records
        self.assertEqual(grounding["required_operand_roles"], [
            "AGGREGATION_SCOPE", "TARGET_MEASURE",
        ])
        self.assertEqual(grounding["selected_binding_id"], derivation["binding_id"])
        self.assertFalse(grounding["first_match_used"])
        self.assertEqual(derivation["answer_role"], "SCALAR")
        self.assertEqual(derivation["projected_answer_hash"], active_answer_hash("2"))
        self.assertEqual(derivation["side"], "ORIGINAL")
        self.assertEqual(registry["answer_hash"], derivation["projected_answer_hash"])
        self.assertEqual(registry["provenance_ids"], derivation["provenance_ids"])

    def test_unprovenanced_derivation_is_raw_but_never_registered(self):
        incomplete = successful_pair("P1", "3", False)
        bundle = self.bundle(successful_pair(), incomplete)
        self.assertEqual(len(bundle.raw_derivations), 2)
        self.assertEqual(len(bundle.registry_entries), 1)
        self.assertEqual(bundle.excluded_registry_derivation_ids, (incomplete[1].derivation_id,))

    def test_unexecuted_or_identity_mismatched_derivation_is_rejected(self):
        assignment, derivation = successful_pair()
        changes = (
            {"execution_outcome": "NOT_RUN"}, {"projection_outcome": "FAILED"},
            {"signature_id": "SUM_SCALAR"}, {"canonical_program_id": "CP-wrong"},
            {"projected_answer": "9"},
        )
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.bundle((replace(assignment, **change), derivation))

    def test_program_identity_and_canonical_round_trip_are_enforced(self):
        assignment, derivation = successful_pair()
        malformed = (
            replace(derivation, executable_program=json.dumps(
                json.loads(derivation.executable_program), ensure_ascii=False, indent=2,
            )),
            replace(derivation, operation_metadata={"canonical_program_id": "CP-wrong"}),
        )
        for candidate in malformed:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                self.bundle((assignment, candidate))

    def test_registry_reconciliation_rejects_every_identity_or_hash_mismatch(self):
        bundle = self.bundle(successful_pair())
        derivation, registry = bundle.raw_derivations[0], bundle.registry_entries[0]
        mutations = {
            "sample_id": "other", "arm": "C1_ROLE_ONLY", "derivation_id": "other",
            "side": "ALTERNATIVE", "canonical_program_id": "other",
            "answer_class_id": "other", "answer_hash": "0" * 64,
            "provenance_ids": ["other"], "registry_entry_id": "other",
        }
        for field, value in mutations.items():
            malformed = dict(registry); malformed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                reconcile_registry_entry(malformed, derivation)

    def test_serialization_is_deterministic_and_json_round_trippable(self):
        pair = successful_pair()
        first, second = self.bundle(pair), self.bundle(pair)
        self.assertEqual(first, second)
        encoded = canonical_json(first.to_dict())
        self.assertEqual(canonical_json(json.loads(encoded)), encoded)


if __name__ == "__main__":
    unittest.main()
