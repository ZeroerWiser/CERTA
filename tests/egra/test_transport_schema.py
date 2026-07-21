import ast
import copy
import hashlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import jsonschema

from certa.egra.query_role_contract import (
    build_query_role_prompt,
    build_query_role_response_schema,
    request_query_role_contract,
    validate_query_role_contract,
)
from certa.egra.transport_schema import (
    TRANSPORT_REMOVED_PATHS,
    build_query_role_transport_schema,
)
from certa.reproducibility.canonical_json import canonical_json_hash
from tests.egra.test_query_role_contract import FakeStructuredGenerator, scalar_lookup_payload
from tools.certa_egra_transport_probe import (
    build_expected_cohort_freeze,
    build_gate_role_row,
    build_probe_record,
    run_roles,
)

REMOVED_PATHS = (
    "/properties/projection_candidates/uniqueItems",
    "/properties/signature_candidates/uniqueItems",
    "/allOf",
)


def ast_body_sha256(function):
    node = ast.parse(inspect.getsource(function)).body[0]
    normalized = ast.dump(ast.Module(body=node.body, type_ignores=[]),
                          annotate_fields=True, include_attributes=False)
    return hashlib.sha256(normalized.encode()).hexdigest()


def probe(validation, audit, *, version="0.11.0", version_status=200):
    return build_probe_record(
        version_payload={"version": version},
        models_payload={"data": [{"id": "Qwen3-8B"}]},
        version_http_status=version_status, models_http_status=200,
        http_status=200, validation=validation, audit=audit,
    )


class TransportSchemaTests(unittest.TestCase):
    def test_projection_removes_exactly_the_three_frozen_wire_paths(self):
        semantic = build_query_role_response_schema()
        before = copy.deepcopy(semantic)
        transport = build_query_role_transport_schema(semantic)
        self.assertEqual(TRANSPORT_REMOVED_PATHS, REMOVED_PATHS)
        self.assertEqual(semantic, before)
        del before["properties"]["projection_candidates"]["uniqueItems"]
        del before["properties"]["signature_candidates"]["uniqueItems"]
        del before["allOf"]
        self.assertEqual(transport, before)

    def test_projection_fails_closed_when_frozen_allof_is_missing(self):
        semantic = build_query_role_response_schema()
        del semantic["allOf"]
        with self.assertRaisesRegex(ValueError, "query_role_semantic_allof_missing"):
            build_query_role_transport_schema(semantic)

    def test_duplicate_arrays_pass_transport_but_fail_full_semantics(self):
        payload = scalar_lookup_payload()
        payload["signature_candidates"] = ["LOOKUP_VALUE_SCALAR"] * 2
        payload["projection_candidates"] = ["VALUE_PROJECTION"] * 2
        jsonschema.validate(payload, build_query_role_transport_schema(
            build_query_role_response_schema()))
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(payload, build_query_role_response_schema())
        result = validate_query_role_contract(payload)
        self.assertFalse(result.ok)
        self.assertTrue(result.parse_ok)
        self.assertTrue(any(error.startswith("schema_violation:uniqueItems:")
                            for error in result.errors))

    def test_valid_payload_passes_both_schema_layers(self):
        payload = scalar_lookup_payload()
        jsonschema.validate(payload, build_query_role_response_schema())
        jsonschema.validate(payload, build_query_role_transport_schema(
            build_query_role_response_schema()))
        self.assertTrue(validate_query_role_contract(payload).ok)

    def test_frozen_semantic_prompt_and_validator_bodies_are_unchanged(self):
        expected = {
            build_query_role_response_schema: "7be4f41a410a25bcd4fc9186d52df627ae375629761989f68d621ef88bf61a37",
            build_query_role_prompt: "82ee2c5aface516f7a80145194f6611a4ec716e26e189463062428f75fedf93f",
            validate_query_role_contract: "4ae0551c5582c6136228c7114b119b572014c2e2557f7fe6a9f36aa2fb135938",
        }
        for function, digest in expected.items():
            self.assertEqual(ast_body_sha256(function), digest)
        self.assertEqual(canonical_json_hash(build_query_role_response_schema()),
                         "f58e8e84edb768689e406f8012c39813f79ad153e8587e8cd3341a031c1559d7")
        prompt_hash = canonical_json_hash({"prompt": build_query_role_prompt(
            "What is the population of North?")})
        self.assertEqual(prompt_hash,
                         "9d29682a7905fc0ad1c4bf39ebc9632f13f05659465aa55572087f117a6c87a1")

    def test_request_routes_transport_schema_and_records_both_identities(self):
        generator = FakeStructuredGenerator(scalar_lookup_payload())
        validation, audit = request_query_role_contract(
            generator, "What is the population of North?")
        self.assertTrue(validation.ok)
        sent = generator.calls[0][1]["response_schema"]
        semantic_hash = canonical_json_hash(build_query_role_response_schema())
        transport_hash = canonical_json_hash(build_query_role_transport_schema(
            build_query_role_response_schema()))
        self.assertEqual(canonical_json_hash(sent), transport_hash)
        self.assertNotEqual(transport_hash, semantic_hash)
        self.assertEqual(audit["semantic_schema_sha256"], semantic_hash)
        self.assertEqual(audit["transport_schema_sha256"], transport_hash)
        self.assertEqual(audit["structured_output_schema_sha256"], transport_hash)
        self.assertRegex(audit["adapter_sha256"], r"^[0-9a-f]{64}$")


class TransportProbeTests(unittest.TestCase):
    def test_probe_requires_exact_backend_transport_and_http_evidence(self):
        validation, audit = request_query_role_contract(
            FakeStructuredGenerator(scalar_lookup_payload()),
            "What is the population of North?")
        audit.update({"http_completed": True, "actual_attempts": 1})
        record = probe(validation, audit)
        self.assertTrue(record["pass"], record)
        self.assertEqual(record["classification"], "VALID")
        self.assertEqual((record["version_http_status"], record["models_http_status"]),
                         (200, 200))
        self.assertRegex(record["adapter_sha256"], r"^[0-9a-f]{64}$")
        tampered = dict(audit, prompt_sha256="0" * 64)
        self.assertFalse(probe(validation, tampered)["pass"])
        self.assertFalse(probe(validation, audit, version="0.10.2")["pass"])
        self.assertFalse(probe(validation, audit, version_status=201)["pass"])

        failing = FakeStructuredGenerator(scalar_lookup_payload())
        generate = failing.generate_json_schema
        failing.generate_json_schema = lambda *a, **kw: {
            **generate(*a, **kw), "error": "synthetic_backend_failure"}
        failed_validation, failed_audit = request_query_role_contract(
            failing, "What is the population of North?")
        failed_audit.update({"http_completed": True, "actual_attempts": 1})
        self.assertFalse(probe(failed_validation, failed_audit)["pass"])

        missing = FakeStructuredGenerator(scalar_lookup_payload())
        generate_missing = missing.generate_json_schema
        missing.generate_json_schema = lambda *a, **kw: {
            key: value for key, value in generate_missing(*a, **kw).items()
            if key not in {"structured_output_fallback_used", "api_cache_hit"}}
        missing_validation, missing_audit = request_query_role_contract(
            missing, "What is the population of North?")
        missing_audit.update({"http_completed": True, "actual_attempts": 1})
        self.assertFalse(probe(missing_validation, missing_audit)["pass"])

    def test_gate_projection_revalidates_status_and_fails_closed(self):
        duplicate = scalar_lookup_payload()
        duplicate["signature_candidates"] *= 2
        frozen = {
            "sample_id": "s1", "status": "VALID", "contract": duplicate,
            "audit": {"parse_ok": True, "model": "Qwen3-8B",
                      "backend": "vllm_chat",
                      "api_base_url": "http://127.0.0.1:30338/v1",
                      "thinking": {"enable_thinking": False},
                      "semantic_schema_sha256": "f" * 64,
                      "transport_schema_sha256": "e" * 64,
                      "structured_output_schema_sha256": "e" * 64,
                      "structured_output_fallback_used": False,
                      "http_completed": True}}
        row = build_gate_role_row(frozen)
        self.assertEqual(row["classification"], "INVALID")
        self.assertTrue(row["json_parse_ok"])
        self.assertTrue(row["hash_drift"])

        validation, audit = request_query_role_contract(
            FakeStructuredGenerator(scalar_lookup_payload()), "What is A?")
        audit["actual_attempts"] = 0
        del audit["structured_output_fallback_used"]
        missing = build_gate_role_row({"sample_id": "s1", "status": "VALID",
                                       "contract": scalar_lookup_payload(),
                                       "audit": audit})
        self.assertEqual((missing["http_completed"], missing["fallback_used"],
                          missing["question_only_input"]), (False, True, False))
        self.assertEqual(probe(validation, audit)["actual_attempts"], 0)
        self.assertFalse(probe(validation, audit)["pass"])

    def test_invalid_resume_never_truncates_retained_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jsonl"
            source_rows = [
                {"dataset": "hitab", "id": f"s{i}", "question": f"Q{i}?",
                 "table_id": f"t{i}", "table_source": "raw"}
                for i in range(64)]
            source.write_text("".join(json.dumps(row) + "\n" for row in source_rows))
            cohort_payload = build_expected_cohort_freeze(source)
            by_id = {row["id"]: row for row in source_rows}
            runtime_rows = [by_id[item] for item in
                            cohort_payload["matched24"]["ordered_sample_ids"]]
            tampered_runtime = root / "tampered_runtime.jsonl"
            tampered_rows = [dict(row) for row in runtime_rows]
            tampered_rows[0]["question"] = "Different question?"
            tampered_runtime.write_text("".join(
                json.dumps(row) + "\n" for row in tampered_rows))
            cohort = root / "cohort.json"
            cohort.write_text(json.dumps(cohort_payload))
            with self.assertRaisesRegex(
                    ValueError, "role_runtime_not_exact_frozen_matched24"):
                run_roles(tampered_runtime, root / "unused.jsonl",
                          root / "cache.jsonl", cohort, source,
                          limit=None, resume=False)
            frozen_rows = []
            for index, runtime_row in enumerate(runtime_rows):
                question = runtime_row["question"]
                validation, audit = request_query_role_contract(
                    FakeStructuredGenerator(scalar_lookup_payload()), question)
                audit.update({"http_completed": True, "actual_attempts": 1})
                row = {"schema_version": "certa_egra_role_freeze_row_v1",
                       "sample_id": runtime_row["id"],
                       "table_id": runtime_row["table_id"], "source_order": index,
                       "question_sha256": canonical_json_hash({"question": question}),
                       "question_only_input": True, "status": "VALID",
                       "contract": validation.normalized_payload, "errors": [],
                       "audit": audit}
                prompt_hash = canonical_json_hash({"prompt": build_query_role_prompt(question)})
                row.update(build_gate_role_row(row, expected_prompt_sha256=prompt_hash))
                frozen_rows.append(row)
            tampered_id = frozen_rows[1]["sample_id"]
            frozen_rows[1]["table_id"] = "tampered"
            runtime, output = (root / name for name in
                               ("runtime.jsonl", "roles.jsonl"))
            runtime.write_text("".join(json.dumps(row) + "\n" for row in runtime_rows))
            output.write_text("".join(json.dumps(row) + "\n" for row in frozen_rows))
            before = output.read_bytes()
            with self.assertRaisesRegex(
                    ValueError, f"existing_role_row_identity_mismatch:{tampered_id}"):
                run_roles(runtime, output, root / "cache.jsonl", cohort, source,
                          limit=None, resume=True)
            self.assertEqual(output.read_bytes(), before)

    def test_role_runner_rejects_oversized_source_before_any_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime.jsonl"
            rows = [{"dataset": "hitab", "id": f"s{i}", "question": f"Q{i}?",
                     "table_id": f"t{i}", "table_source": "raw"}
                    for i in range(25)]
            runtime.write_text("".join(json.dumps(row) + "\n" for row in rows))
            cohort = root / "cohort.json"
            cohort.write_text(json.dumps({"matched24": {"ordered_sample_ids": []}}))
            with self.assertRaisesRegex(ValueError, "source_dev64_count:25"):
                run_roles(runtime, root / "roles.jsonl", root / "cache.jsonl",
                          cohort, runtime, limit=None, resume=False)


if __name__ == "__main__":
    unittest.main()
