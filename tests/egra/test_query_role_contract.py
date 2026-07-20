import inspect
import json
import unittest

import jsonschema

from certa.reproducibility.canonical_json import canonical_json_hash
from certa.egra.query_role_contract import (
    CORE_SIGNATURE_IDS,
    QUERY_ROLE_MAX_TOKENS,
    build_query_role_prompt,
    build_query_role_response_schema,
    request_query_role_contract,
    validate_query_role_contract,
)


def scalar_lookup_payload():
    return {
        "schema_version": "certa_egra_query_contract_v1",
        "supported_by_core_signatures": True,
        "answer_domain": "SCALAR",
        "intent_family": "DIRECT_READ",
        "signature_candidates": ["LOOKUP_VALUE_SCALAR"],
        "projection_candidates": ["VALUE_PROJECTION"],
        "cardinality": "SINGLE",
        "rank_direction": "NONE",
        "rank_k": None,
        "requires_time_scope": False,
        "requires_unit_consistency": False,
        "unknowns": [],
    }


class FakeStructuredGenerator:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []
        self.model = "Qwen3-8B"
        self.api_base_url = "http://127.0.0.1:30338/v1"
        self.backend_name = "vllm_chat"
        self.chat_template_kwargs = {"enable_thinking": False}
        self.cache_mode = "readwrite"

    def generate_json_schema(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return {
            "text": json.dumps(self.payload),
            "structured_output_requested": True,
            "structured_output_mechanism": "response_format.type=json_schema",
            "structured_output_schema_hash": canonical_json_hash(kwargs["response_schema"]),
            "structured_output_fallback_used": False,
            "input_token_count": 17,
            "generated_token_count": 23,
            "generation_seconds": 0.25,
            "api_model": "Qwen3-8B",
            "api_base_url": "http://127.0.0.1:30338/v1",
            "generator_backend": "vllm_chat",
            "api_cache_hit": False,
            "api_cache_mode": "readwrite",
            "chat_template_kwargs": {"enable_thinking": False},
        }


class WrongIdentityGenerator(FakeStructuredGenerator):
    def __init__(self, payload):
        super().__init__(payload)
        self.model = "WrongModel"
        self.api_base_url = "http://wrong.invalid/v1"
        self.backend_name = "openai_chat"
        self.chat_template_kwargs = {"enable_thinking": True}


class QueryRoleContractTests(unittest.TestCase):
    def test_contract_uses_only_question_input_and_frozen_core_signatures(self):
        self.assertEqual(
            tuple(inspect.signature(build_query_role_prompt).parameters),
            ("question",),
        )
        prompt = build_query_role_prompt("Which region has the largest value?")
        self.assertIn("Which region has the largest value?", prompt)
        self.assertIn("tie_semantics", prompt)
        for forbidden in ("B0", "candidate_answer", "gold_answer", "table_values"):
            self.assertNotIn(forbidden, prompt)

        self.assertEqual(
            CORE_SIGNATURE_IDS,
            (
                "LOOKUP_VALUE_SCALAR",
                "LOOKUP_VALUE_ENTITY",
                "COUNT_SCALAR",
                "DIFF_SCALAR",
                "RATIO_SCALAR",
                "ARGMAX_ENTITY",
                "ARGMAX_ENTITY_SET",
                "ARGMIN_ENTITY",
                "ARGMIN_ENTITY_SET",
            ),
        )
        schema_text = json.dumps(build_query_role_response_schema(), sort_keys=True)
        self.assertNotIn("SUM_SCALAR", schema_text)
        self.assertNotIn("AVERAGE_SCALAR", schema_text)

    def test_schema_and_semantic_validator_accept_a_consistent_contract(self):
        payload = scalar_lookup_payload()
        schema = build_query_role_response_schema()
        self.assertEqual(
            canonical_json_hash(schema),
            "f58e8e84edb768689e406f8012c39813f79ad153e8587e8cd3341a031c1559d7",
        )
        jsonschema.validate(payload, schema)
        result = validate_query_role_contract(payload)
        self.assertTrue(result.ok, result.errors)
        self.assertEqual(result.normalized_payload, payload)

    def test_semantic_validator_rejects_family_domain_and_rank_mismatches(self):
        payload = scalar_lookup_payload()
        payload.update({
            "answer_domain": "ENTITY",
            "intent_family": "RANK_MIN",
            "rank_direction": "KTH",
            "rank_k": 3,
        })
        result = validate_query_role_contract(payload)
        self.assertFalse(result.ok)
        self.assertIn("signature_intent_mismatch:LOOKUP_VALUE_SCALAR", result.errors)
        self.assertIn("signature_answer_domain_mismatch:LOOKUP_VALUE_SCALAR", result.errors)
        self.assertIn("unsupported_rank_direction:KTH", result.errors)

    def test_cardinality_is_derived_from_the_primary_signature(self):
        payload = scalar_lookup_payload()
        payload["cardinality"] = "EXACT_K"
        result = validate_query_role_contract(payload)
        self.assertFalse(result.ok)
        self.assertIn("unsupported_cardinality:EXACT_K", result.errors)

        payload.update({
            "answer_domain": "SET",
            "intent_family": "RANK_MAX",
            "signature_candidates": ["ARGMAX_ENTITY_SET"],
            "projection_candidates": ["ROW_ENTITY_PROJECTION"],
            "cardinality": "SINGLE",
            "rank_direction": "MAX",
        })
        result = validate_query_role_contract(payload)
        self.assertFalse(result.ok)
        self.assertIn("cardinality_mismatch:SINGLE!=MULTIPLE", result.errors)

    def test_secondary_signature_uncertainty_must_be_named_explicitly(self):
        payload = scalar_lookup_payload()
        payload.update({
            "intent_family": "RATIO",
            "signature_candidates": ["RATIO_SCALAR", "LOOKUP_VALUE_SCALAR"],
            "projection_candidates": ["SCALAR_RESULT_PROJECTION", "VALUE_PROJECTION"],
        })
        result = validate_query_role_contract(payload)
        self.assertFalse(result.ok)
        self.assertIn("unnamed_candidate_uncertainty:intent_family", result.errors)

        payload["unknowns"] = ["intent_family"]
        result = validate_query_role_contract(payload)
        self.assertTrue(result.ok, result.errors)

    def test_unsupported_contract_is_fail_closed(self):
        payload = scalar_lookup_payload()
        payload.update({
            "supported_by_core_signatures": False,
            "answer_domain": "UNSUPPORTED",
            "intent_family": "UNSUPPORTED",
            "signature_candidates": [],
            "projection_candidates": [],
            "cardinality": "UNKNOWN",
            "rank_direction": "UNKNOWN",
            "rank_k": None,
            "unknowns": ["operation"],
        })
        self.assertTrue(validate_query_role_contract(payload).ok)

        payload["signature_candidates"] = ["LOOKUP_VALUE_SCALAR"]
        self.assertFalse(validate_query_role_contract(payload).ok)

    def test_request_is_single_strict_nonthinking_call_with_audited_cost(self):
        generator = FakeStructuredGenerator(scalar_lookup_payload())
        result, audit = request_query_role_contract(
            generator,
            "What is the value for A?",
        )
        self.assertTrue(result.ok, result.errors)
        self.assertEqual(len(generator.calls), 1)
        _, kwargs = generator.calls[0]
        self.assertEqual(kwargs["max_new_tokens"], QUERY_ROLE_MAX_TOKENS)
        self.assertEqual(QUERY_ROLE_MAX_TOKENS, 256)
        self.assertEqual(kwargs["temperature"], 0.0)
        self.assertEqual(kwargs["top_p"], 1.0)
        self.assertEqual(kwargs["schema_name"], "certa_egra_query_contract_v1")
        self.assertEqual(audit["calls"], 1)
        self.assertEqual(audit["prompt_tokens"], 17)
        self.assertEqual(audit["completion_tokens"], 23)
        self.assertEqual(audit["latency_seconds"], 0.25)
        self.assertFalse(audit["structured_output_fallback_used"])
        self.assertEqual(audit["sampling"], {
            "max_tokens": 256,
            "temperature": 0.0,
            "top_p": 1.0,
        })
        self.assertEqual(audit["thinking"], {"enable_thinking": False})
        self.assertEqual(audit["cache"], {"hit": False, "mode": "readwrite"})
        self.assertTrue(audit["parse_ok"])
        self.assertEqual(audit["normalized_output"], scalar_lookup_payload())
        for field in (
            "request_sha256",
            "prompt_sha256",
            "schema_sha256",
            "raw_output_sha256",
            "normalized_output_sha256",
        ):
            self.assertRegex(audit[field], r"^[0-9a-f]{64}$")

    def test_request_rejects_transport_identity_drift_before_call(self):
        generator = WrongIdentityGenerator(scalar_lookup_payload())
        with self.assertRaisesRegex(ValueError, "query_role_transport_identity_mismatch"):
            request_query_role_contract(generator, "What is A?")
        self.assertEqual(generator.calls, [])


if __name__ == "__main__":
    unittest.main()
