import json
import unittest
from pathlib import Path

import jsonschema

from certa.active_v1.role_contract_v3 import (
    ROLE_V3_MAX_TOKENS,
    build_role_v3_prompt,
    build_role_v3_prompt_template,
    derive_role_v3_record,
    parse_role_v3_output,
    role_v3_to_planner_query_contract,
    validate_role_v3_artifacts,
)


PACK = Path("/home/hsh/ME/Table/EMNLP2026/certa_goal_packs/CERTA_ACTIVE_V1_ROLE_V3_FINAL_METHOD_PACK")


def load(name):
    return json.loads((PACK / name).read_text(encoding="utf-8"))


class RoleContractV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cards = load("ROLE_V3_ROLE_CARDS.json")
        cls.schema = load("ROLE_V3_OUTPUT_SCHEMA.json")
        cls.registry = load("ROLE_V3_CANONICAL_REGISTRY.json")

    def test_artifacts_have_one_ordered_role_authority(self):
        role_ids = validate_role_v3_artifacts(self.cards, self.schema, self.registry)
        self.assertEqual(len(role_ids), 13)
        self.assertEqual(role_ids[-1], "UNSUPPORTED")
        self.assertEqual(set(self.schema["properties"]), {"schema_version", "role_id"})
        self.assertEqual(set(self.schema["required"]), {"schema_version", "role_id"})

    def test_generated_prompt_template_is_byte_identical_to_pack(self):
        expected = (PACK / "ROLE_V3_PROMPT_TEMPLATE.txt").read_text(encoding="utf-8")
        self.assertEqual(build_role_v3_prompt_template(self.cards), expected)

    def test_prompt_injects_only_json_encoded_question(self):
        question = 'Which row contains "Atlas"?'
        prompt = build_role_v3_prompt(question, self.cards)
        expected = (PACK / "ROLE_V3_PROMPT_TEMPLATE.txt").read_text(encoding="utf-8").replace(
            "{{QUESTION_JSON_STRING}}", json.dumps(question, ensure_ascii=False)
        )
        self.assertEqual(prompt, expected)
        self.assertEqual(prompt.count(json.dumps(question, ensure_ascii=False)), 1)
        self.assertNotIn("confidence", prompt.lower())

    def test_parser_accepts_exact_two_field_payload_only(self):
        payload = {
            "schema_version": "certa_active_role_contract_v3",
            "role_id": "COUNT_SCALAR",
        }
        self.assertEqual(parse_role_v3_output(json.dumps(payload), self.schema), payload)
        with self.assertRaises(jsonschema.ValidationError):
            parse_role_v3_output(json.dumps(dict(payload, confidence=1.0)), self.schema)
        with self.assertRaises(json.JSONDecodeError):
            parse_role_v3_output("COUNT_SCALAR", self.schema)

    def test_registry_is_the_only_authority_for_derived_semantics(self):
        output = {
            "schema_version": "certa_active_role_contract_v3",
            "role_id": "COUNT_SCALAR",
        }
        record = derive_role_v3_record(output, self.schema, self.registry)
        self.assertEqual(record, {
            "schema_version": "certa_active_role_v3_canonical_record_v1",
            "role_id": "COUNT_SCALAR",
            "supported": True,
            "intent": "COUNT",
            "answer_role": "SCALAR",
            "projection": "SCALAR_RESULT_PROJECTION",
            "cardinality": "SINGLE",
            "operation_family": "COUNT",
            "requires_time_scope": "DEFERRED_TO_GROUNDING",
            "requires_unit_consistency": "DEFERRED_TO_EXECUTION",
        })
        self.assertNotIsInstance(record["requires_time_scope"], bool)
        self.assertNotIsInstance(record["requires_unit_consistency"], bool)

    def test_unsupported_is_derived_without_planner_contract(self):
        output = {
            "schema_version": "certa_active_role_contract_v3",
            "role_id": "UNSUPPORTED",
        }
        record = derive_role_v3_record(output, self.schema, self.registry)
        self.assertFalse(record["supported"])
        self.assertEqual(record["operation_family"], "UNSUPPORTED")
        with self.assertRaisesRegex(ValueError, "unsupported_role_has_no_planner_query_contract"):
            role_v3_to_planner_query_contract(record)

    def test_planner_contract_uses_single_registry_derived_signature(self):
        record = derive_role_v3_record({
            "schema_version": "certa_active_role_contract_v3",
            "role_id": "RATIO_SCALAR",
        }, self.schema, self.registry)
        query = role_v3_to_planner_query_contract(record)
        self.assertEqual(query["allowed_signature_ids"], ["RATIO_SCALAR"])
        self.assertEqual(query["candidate_independent_operation_hypotheses"], ["RATIO"])
        self.assertEqual(query["allowed_answer_domains"], ["SCALAR"])
        self.assertEqual(query["allowed_projection_operators"], ["SCALAR_RESULT_PROJECTION"])
        self.assertEqual(query["time_scope_authority"], "DEFERRED_TO_GROUNDING")
        self.assertEqual(query["unit_consistency_authority"], "DEFERRED_TO_EXECUTION")
        self.assertNotIn("requires_time_scope", query)
        self.assertNotIn("requires_unit_consistency", query)
        self.assertEqual(ROLE_V3_MAX_TOKENS, 64)


if __name__ == "__main__":
    unittest.main()
