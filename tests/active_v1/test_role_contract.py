import copy
import unittest

import jsonschema

from certa.active_v1.role_contract import (
    build_role_prompt,
    build_role_semantic_schema,
    build_role_wire_schema,
    role_to_query_contract,
    to_egra_retrieval_contract,
    validate_role_contract,
)


ACTIVE = (
    "LOOKUP_VALUE_SCALAR",
    "LOOKUP_VALUE_ENTITY",
    "COUNT_SCALAR",
    "SUM_SCALAR",
    "AVERAGE_SCALAR",
    "DIFF_SCALAR",
    "RATIO_SCALAR",
    "ARGMAX_ENTITY",
    "ARGMAX_ENTITY_SET",
    "ARGMIN_ENTITY",
    "ARGMIN_ENTITY_SET",
    "PAIR_COMPARE_BOOLEAN",
)


COUNT = {
    "schema_version": "certa_active_role_contract_v2",
    "supported": True,
    "intent": "COUNT",
    "answer_role": "SCALAR",
    "projection": "SCALAR_RESULT_PROJECTION",
    "signature": "COUNT_SCALAR",
    "cardinality": "SINGLE",
    "requires_time_scope": False,
    "requires_unit_consistency": False,
}


class ActiveRoleContractTests(unittest.TestCase):
    def test_wire_schema_is_flat_but_semantic_schema_enforces_exact_tuple(self):
        wire = build_role_wire_schema(ACTIVE)
        semantic = build_role_semantic_schema(ACTIVE)
        jsonschema.validate(COUNT, wire)
        jsonschema.validate(COUNT, semantic)
        malformed = dict(COUNT, intent="SUM")
        jsonschema.validate(malformed, wire)
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(malformed, semantic)

    def test_local_validator_reports_independent_validity_layers(self):
        valid = validate_role_contract(COUNT, ACTIVE)
        self.assertTrue(valid.parse_ok)
        self.assertTrue(valid.wire_valid)
        self.assertTrue(valid.semantic_schema_valid)
        self.assertTrue(valid.local_validator_valid)
        malformed = validate_role_contract(dict(COUNT, cardinality="MULTIPLE"), ACTIVE)
        self.assertTrue(malformed.wire_valid)
        self.assertFalse(malformed.semantic_schema_valid)
        self.assertFalse(malformed.local_validator_valid)
        self.assertIn("role_tuple_not_authorized", malformed.local_errors)

    def test_unsupported_tuple_is_exact_and_inactive_signatures_are_rejected(self):
        unsupported = {
            "schema_version": "certa_active_role_contract_v2",
            "supported": False,
            "intent": "UNSUPPORTED",
            "answer_role": "UNSUPPORTED",
            "projection": "UNSUPPORTED",
            "signature": "UNSUPPORTED",
            "cardinality": "UNKNOWN",
            "requires_time_scope": False,
            "requires_unit_consistency": False,
        }
        self.assertTrue(validate_role_contract(unsupported, ACTIVE).ok)
        inactive = dict(COUNT, signature="LOOKUP_VALUE_BOOLEAN", intent="DIRECT_READ", projection="VALUE_PROJECTION")
        result = validate_role_contract(inactive, ACTIVE)
        self.assertFalse(result.wire_valid)
        self.assertFalse(result.local_validator_valid)

    def test_scheme_a_projection_only_derives_non_applicable_rank_fields(self):
        legacy = to_egra_retrieval_contract(COUNT)
        self.assertEqual(legacy["intent_family"], "COUNT")
        self.assertEqual(legacy["signature_candidates"], ["COUNT_SCALAR"])
        self.assertEqual(legacy["answer_domain"], "SCALAR")
        self.assertEqual(legacy["projection_candidates"], ["SCALAR_RESULT_PROJECTION"])
        self.assertEqual(legacy["cardinality"], "SINGLE")
        self.assertEqual(legacy["rank_direction"], "NONE")
        self.assertIsNone(legacy["rank_k"])
        self.assertEqual(role_to_query_contract(COUNT)["candidate_independent_operation_hypotheses"], ["COUNT"])
        self.assertEqual(COUNT["signature"], "COUNT_SCALAR")

    def test_prompt_is_question_only_and_exposes_only_active_signatures(self):
        prompt = build_role_prompt("How many rows qualify?", ("COUNT_SCALAR",))
        self.assertIn("How many rows qualify?", prompt)
        self.assertIn("COUNT_SCALAR", prompt)
        self.assertNotIn("ARGMAX_ENTITY", prompt)
        self.assertNotIn("rank_k", prompt)

    def test_payload_is_never_repaired(self):
        malformed = dict(COUNT, cardinality="MULTIPLE")
        before = copy.deepcopy(malformed)
        result = validate_role_contract(malformed, ACTIVE)
        self.assertEqual(malformed, before)
        self.assertEqual(result.payload, before)


if __name__ == "__main__":
    unittest.main()
