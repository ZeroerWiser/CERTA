import copy
import json
import tempfile
import unittest
from pathlib import Path

import jsonschema

from certa.active_v1.planner_transport_projection import (
    build_planner_transport_schema,
    planner_transport_schema_identity,
)
from certa.planner.typed_planner import validate_typed_planner_output
from tools.certa_active_v1_schema_risk_scanner import DEFAULT_LIMITS, scan_schema
from tools import certa_active_v1_completion as completion


HISTORICAL = Path(
    "/home/hsh/ME/Table/EMNLP2026/certa_active_v1_outputs/"
    "CERTA_ACTIVE_V1_FROZEN_ROLE_V3_FINAL_METHOD_COMPLETION_ARCHIVE_RESTORED/raw"
)


def _requests():
    return sorted(HISTORICAL.glob("dev_planner_*/*_request.json"))


def _request(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _schema(request):
    return request["request"]["response_format"]["json_schema"]["schema"]


def _view(request):
    prompt = request["request"]["messages"][0]["content"]
    return json.loads(prompt.split("Planner view:\n", 1)[1])


def _response_payload(request_path):
    response_path = Path(str(request_path).replace("_request.json", "_response.json"))
    response = json.loads(response_path.read_text(encoding="utf-8"))
    if response.get("ok") is not True:
        return None
    return json.loads(response["generation"]["text"])


class PlannerTransportProjectionTests(unittest.TestCase):
    def test_all_historical_schemas_project_deterministically_within_limits(self):
        paths = _requests()
        self.assertEqual(len(paths), 6)
        for path in paths:
            full = _schema(_request(path))
            first = build_planner_transport_schema(full)
            second = build_planner_transport_schema(copy.deepcopy(full))
            self.assertEqual(first, second, path)
            jsonschema.Draft202012Validator.check_schema(first)
            risk = scan_schema(first, DEFAULT_LIMITS)
            self.assertTrue(risk["risk_scan_pass"], (path, risk["limit_failures"]))

    def test_projection_preserves_transport_contract_identities(self):
        for path in _requests():
            full = _schema(_request(path))
            transport = build_planner_transport_schema(full)
            identity = planner_transport_schema_identity(full, transport)
            self.assertTrue(identity["all_preservation_checks_pass"], (path, identity))
            self.assertEqual(transport["required"], full["required"])
            self.assertIs(transport["additionalProperties"], False)
            self.assertEqual(
                transport["properties"]["plans"]["minItems"],
                full["properties"]["plans"]["minItems"],
            )

    def test_five_historical_payloads_pass_transport_and_full_local_validator(self):
        successes = 0
        for path in _requests():
            request = _request(path)
            payload = _response_payload(path)
            transport = build_planner_transport_schema(_schema(request))
            if payload is None:
                continue
            successes += 1
            jsonschema.Draft202012Validator(transport).validate(payload)
            validation = validate_typed_planner_output(
                payload, _view(request), require_signature_id=True,
            )
            self.assertTrue(validation.ok, (path, validation.errors))
        self.assertEqual(successes, 5)

    def test_malformed_categories_never_pass_full_local_acceptance(self):
        path = next(path for path in _requests() if _response_payload(path) is not None)
        request = _request(path)
        valid = _response_payload(path)
        plan = valid["plans"][0]
        mutations = {}

        value = copy.deepcopy(valid)
        value["plans"][0]["operation_family"] = "COUNT"
        mutations["signature_operation_mismatch"] = value

        value = copy.deepcopy(valid)
        value["plans"][0]["answer_domain"] = "BOOLEAN"
        mutations["answer_domain_projection_mismatch"] = value

        value = copy.deepcopy(valid)
        del value["plans"][0]["role_bindings"]["TARGET_MEASURE"]
        mutations["required_role_missing"] = value

        value = copy.deepcopy(valid)
        value["plans"][0]["role_bindings"]["GROUP_SCOPE"] = [
            next(iter(plan["role_bindings"].values()))[0]
        ]
        mutations["forbidden_role_present"] = value

        value = copy.deepcopy(valid)
        value["plans"][0].setdefault("role_domains", {})["TARGET_ENTITY"] = [
            value["plans"][0]["role_bindings"]["TARGET_MEASURE"]
        ]
        mutations["binding_domain_exclusivity"] = value

        value = copy.deepcopy(valid)
        value["plans"][0]["role_bindings"]["TARGET_ENTITY"] = [[]]
        mutations["role_shape_or_cardinality"] = value

        value = copy.deepcopy(valid)
        value["plans"][0]["role_bindings"]["TARGET_ENTITY"] = ["outside_domain"]
        mutations["reference_domain_violation"] = value

        value = copy.deepcopy(valid)
        value["plans"][0]["unknown"] = True
        mutations["additional_property"] = value

        value = copy.deepcopy(valid)
        value["plans"][0]["role_bindings"]["TARGET_MEASURE"] = []
        mutations["empty_required_array"] = value

        value = copy.deepcopy(valid)
        value["plans"] = []
        mutations["resource_limit_violation"] = value

        self.assertEqual(len(mutations), 10)
        for category, payload in mutations.items():
            validation = validate_typed_planner_output(
                payload, _view(request), require_signature_id=True,
            )
            self.assertFalse(validation.ok, (category, validation))

    def test_runner_records_both_schemas_risk_identity_and_full_local_result(self):
        path = next(path for path in _requests() if _response_payload(path) is not None)
        request = _request(path)
        full = _schema(request)
        transport = build_planner_transport_schema(full)
        payload = _response_payload(path)
        view = _view(request)

        class FakeGenerator:
            def _completion_request_kwargs(self, **kwargs):
                return {"model": "fixture", **kwargs}

            def generate_json_schema(self, _prompt, **_kwargs):
                return {"text": json.dumps(payload), "api_usage": {}, "generation_seconds": 0}

        with tempfile.TemporaryDirectory() as directory:
            previous = completion.OUT
            completion.OUT = Path(directory)
            try:
                completion.model_call(
                    FakeGenerator(), "DEV_PLANNER_C0_SCHEMA_ONLY", "fixture", "prompt", 32,
                    schema=transport, full_schema=full, planner_view=view,
                )
                endpoint = completion.jl(completion.OUT / "logs/ENDPOINT_LEDGER.jsonl")[-1]
                validation = validate_typed_planner_output(
                    payload, view, require_signature_id=True,
                )
                completion.record_planner_full_local_validation(endpoint, validation)
                stem = Path(endpoint["raw_request_path"]).name.replace("_request.json", "")
                raw_dir = Path(endpoint["raw_request_path"]).parent
                expected = {
                    "full_schema": raw_dir / f"{stem}_full_schema.json",
                    "transport_schema": raw_dir / f"{stem}_transport_schema.json",
                    "risk": raw_dir / f"{stem}_schema_risk_record.json",
                    "local": raw_dir / f"{stem}_full_local_validation.json",
                }
                self.assertTrue(all(item.is_file() for item in expected.values()))
                risk = json.loads(expected["risk"].read_text(encoding="utf-8"))
                self.assertEqual(risk["full_schema_sha256"], planner_transport_schema_identity(full, transport)["full_schema_sha256"])
                self.assertEqual(risk["transport_schema_sha256"], planner_transport_schema_identity(full, transport)["transport_schema_sha256"])
                self.assertTrue(risk["transport_risk"]["risk_scan_pass"])
                self.assertTrue(risk["full_local_validator"]["ok"])
            finally:
                completion.OUT = previous

    def test_runner_fails_closed_before_generation_on_risk_or_identity_failure(self):
        request = _request(next(iter(_requests())))
        full = _schema(request)
        transport = build_planner_transport_schema(full)
        view = _view(request)

        class NeverCalledGenerator:
            calls = 0

            def _completion_request_kwargs(self, **kwargs):
                return {"model": "fixture", **kwargs}

            def generate_json_schema(self, *_args, **_kwargs):
                self.calls += 1
                raise AssertionError("endpoint generation must not be called")

        risky = copy.deepcopy(transport)
        risky["$comment"] = "x" * DEFAULT_LIMITS["canonical_json_bytes_max"]
        altered = copy.deepcopy(transport)
        altered["properties"]["planner_version"]["const"] = "altered"
        for schema in (risky, altered):
            with tempfile.TemporaryDirectory() as directory:
                previous = completion.OUT
                completion.OUT = Path(directory)
                generator = NeverCalledGenerator()
                try:
                    with self.assertRaisesRegex(ValueError, "planner_transport_preflight_failed"):
                        completion.model_call(
                            generator, "DEV_PLANNER_C0_SCHEMA_ONLY", "fixture", "prompt", 32,
                            schema=schema, full_schema=full, planner_view=view,
                        )
                    self.assertEqual(generator.calls, 0)
                finally:
                    completion.OUT = previous

    def test_scanner_rejects_unsupported_keywords_without_exact_type_marker(self):
        schema = {
            "anyOf": [
                {"contains": {}},
                {"type": ["array", "null"], "uniqueItems": True},
                {"multipleOf": 2},
                {"format": "date"},
                {"propertyNames": {}},
            ],
            "properties": {"contains": {"type": "string"}},
        }
        risk = scan_schema(schema, DEFAULT_LIMITS)
        self.assertEqual(
            {item["keyword"] for item in risk["unsupported_xgrammar_keywords"]},
            {"contains", "uniqueItems", "multipleOf", "format", "propertyNames"},
        )
        self.assertFalse(risk["risk_scan_pass"])


if __name__ == "__main__":
    unittest.main()
