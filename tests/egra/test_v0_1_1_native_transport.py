import hashlib
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from certa.egra.query_role_contract import (
    build_query_role_prompt,
    build_query_role_response_schema,
    validate_query_role_contract,
)
from certa.egra.transport_schema import build_query_role_transport_schema
from tools.certa_egra_v0_1_1_transport import (
    CANARY_PROMPT,
    build_canary_schema,
    build_native_request,
    build_wire_fixtures,
    post_once,
    run_attestation,
    validate_attestation_inputs,
)


class NativeRequestTests(unittest.TestCase):
    def test_request_uses_only_native_vllm_schema_enforcement(self):
        schema = build_query_role_transport_schema(build_query_role_response_schema())
        prompt = build_query_role_prompt("What is the population of North?")
        request = build_native_request(prompt, schema, max_tokens=256)
        self.assertEqual(set(request), {
            "cache_mode", "chat_template_kwargs", "max_tokens", "messages",
            "model", "structured_outputs", "temperature", "top_p",
        })
        self.assertNotIn("response_format", request)
        self.assertEqual(request["structured_outputs"], {
            "json": schema, "disable_fallback": True,
        })
        self.assertEqual(request["messages"], [{"role": "user", "content": prompt}])
        self.assertEqual(request["cache_mode"], "off")
        self.assertEqual(request["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(
            (request["model"], request["temperature"], request["top_p"],
             request["max_tokens"]),
            ("Qwen3-8B", 0.0, 1.0, 256),
        )

    def test_attestation_preflight_rejects_every_auditor_counterexample(self):
        prompt = CANARY_PROMPT
        schemas = [
            build_canary_schema("A", "0123456789abcdef0123456789abcdef"),
            build_canary_schema("B", "fedcba9876543210fedcba9876543210"),
        ]
        requests = [build_native_request(prompt, schema, max_tokens=96)
                    for schema in schemas]
        self.assertEqual(validate_attestation_inputs(requests, schemas), [])
        mutations = {
            "temperature": lambda r: r.update(temperature=0.1),
            "top_p": lambda r: r.update(top_p=0.9),
            "max_tokens": lambda r: r.update(max_tokens=95),
            "thinking": lambda r: r.update(chat_template_kwargs={"enable_thinking": True}),
            "messages": lambda r: r.update(messages=[{"role": "system", "content": prompt}]),
            "cache": lambda r: r.update(cache_mode="readwrite"),
            "fallback": lambda r: r["structured_outputs"].update(disable_fallback=False),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                changed = json.loads(json.dumps(requests))
                mutate(changed[0])
                self.assertTrue(validate_attestation_inputs(changed, schemas))
        malformed = json.loads(json.dumps(schemas))
        malformed[0]["additionalProperties"] = True
        self.assertTrue(validate_attestation_inputs(requests, malformed))
        reused = [schemas[0], json.loads(json.dumps(schemas[0]))]
        self.assertTrue(validate_attestation_inputs(requests, reused))
        benign = json.loads(json.dumps(requests))
        for request in benign:
            request["messages"][0]["content"] = "Return JSON."
        self.assertIn("prompt_not_frozen_adversarial", validate_attestation_inputs(
            benign, schemas))

    def test_wire_fixture_suite_has_complete_named_coverage(self):
        pack = Path(
            "/home/hsh/ME/Table/EMNLP2026/certa_goal_packs/"
            "CERTA_EGRA_V0_1_1_FINAL_METHOD_FREEZE_AND_MATCHED24_PACK/"
            "fixtures/WIRE_SEMANTIC_NEGATIVE_FIXTURES.json"
        )
        rows = build_wire_fixtures(json.loads(pack.read_text()))
        expected = {
            "valid_scalar", "valid_entity", "valid_set", "duplicate_signature",
            "duplicate_projection", "unsupported_allof",
            "supported_true_inconsistency", "intent_mismatch", "domain_mismatch",
            "projection_mismatch", "rank_mismatch", "rank_k_forbidden",
            "cardinality_mismatch", "secondary_uncertainty",
        }
        self.assertEqual({row["id"] for row in rows}, expected)
        self.assertEqual(len(rows), 14)
        semantic = build_query_role_response_schema()
        wire = build_query_role_transport_schema(semantic)
        import jsonschema
        for row in rows:
            with self.subTest(row=row["id"]):
                try:
                    jsonschema.validate(row["payload"], wire)
                    wire_ok = True
                except jsonschema.ValidationError:
                    wire_ok = False
                try:
                    jsonschema.validate(row["payload"], semantic)
                    semantic_ok = True
                except jsonschema.ValidationError:
                    semantic_ok = False
                local_ok = validate_query_role_contract(row["payload"]).ok
                self.assertEqual(
                    (wire_ok, semantic_ok, local_ok),
                    (row["expected_wire_valid"], row["expected_semantic_valid"],
                     row["expected_validator_ok"]),
                )


class RawPostIntegrationTests(unittest.TestCase):
    def test_post_once_retains_exact_request_and_complete_raw_response_evidence(self):
        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                received.append((raw, self.headers.get("X-Request-Id")))
                body = json.dumps({
                    "id": "chatcmpl-test", "object": "chat.completion",
                    "created": 1, "model": "Qwen3-8B",
                    "choices": [{"index": 0, "message": {
                        "role": "assistant", "content": "{}"},
                        "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1,
                              "total_tokens": 4},
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("X-Request-Id", "server-test")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                return

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                request = build_native_request("Q", {"type": "object"}, max_tokens=96)
                envelope = post_once(
                    f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                    request, root / "request.json", root / "response.json",
                    root / "endpoint.jsonl", "request-test", timeout=5,
                )
                exact = json.dumps(request, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=False).encode()
                self.assertEqual(received, [(exact, "request-test")])
                self.assertEqual((root / "request.json").read_bytes(), exact)
                self.assertEqual(envelope["http_status"], 200)
                self.assertEqual(envelope["url"],
                                 f"http://127.0.0.1:{server.server_port}/v1/chat/completions")
                self.assertEqual(envelope["method"], "POST")
                self.assertEqual(envelope["request_headers"]["X-Request-Id"],
                                 "request-test")
                self.assertEqual(envelope["request_id"], "request-test")
                self.assertEqual(envelope["response_request_id"], "server-test")
                self.assertEqual(
                    envelope["raw_body_sha256"],
                    hashlib.sha256(envelope["raw_body_utf8"].encode()).hexdigest(),
                )
                ledger = [json.loads(line) for line in
                          (root / "endpoint.jsonl").read_text().splitlines()]
                self.assertEqual(len(ledger), 1)
                self.assertEqual(ledger[0]["attempt"], 1)
                self.assertEqual(ledger[0]["request_id"], "request-test")
                self.assertEqual(ledger[0]["prompt_tokens"], 3)
                self.assertEqual(ledger[0]["completion_tokens"], 1)
                self.assertEqual(ledger[0]["url"], envelope["url"])
                self.assertEqual(ledger[0]["method"], "POST")
        finally:
            server.shutdown()
            thread.join()
            server.server_close()

    def test_ledger_is_preflighted_before_request_bytes_or_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger_directory = root / "endpoint.jsonl"
            ledger_directory.mkdir()
            with self.assertRaises(IsADirectoryError):
                post_once(
                    "http://127.0.0.1:9/v1/chat/completions",
                    build_native_request("Q", {"type": "object"}, max_tokens=96),
                    root / "request.json", root / "response.json",
                    ledger_directory, "request-preflight", timeout=0.1,
                )
            self.assertFalse((root / "request.json").exists())

    def test_attestation_freezes_both_random_schemas_before_exactly_two_posts(self):
        observed = []
        artifact_root = None

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                observed.append((
                    request, self.headers.get("X-Request-Id"),
                    (artifact_root / "NONCE_PROVENANCE.json").exists(),
                ))
                properties = request["structured_outputs"]["json"]["properties"]
                content = json.dumps({key: value["const"]
                                      for key, value in properties.items()})
                body = json.dumps({
                    "id": f"chatcmpl-{len(observed)}", "object": "chat.completion",
                    "created": 1, "model": "Qwen3-8B",
                    "choices": [{"index": 0, "message": {
                        "role": "assistant", "content": content},
                        "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 9, "completion_tokens": 8,
                              "total_tokens": 17},
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                return

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                artifact_root = Path(tmp)
                result = run_attestation(
                    artifact_root, artifact_root / "endpoint.jsonl",
                    url=f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                )
                self.assertEqual(len(observed), 2)
                self.assertTrue(all(frozen for _, _, frozen in observed))
                schemas = [json.loads((artifact_root / f"schema_{label}.json").read_text())
                           for label in "AB"]
                requests = [json.loads((artifact_root / f"raw_request_{label}.json").read_text())
                            for label in "AB"]
                self.assertEqual(validate_attestation_inputs(requests, schemas), [])
                self.assertNotEqual(
                    schemas[0]["properties"]["nonce"]["const"],
                    schemas[1]["properties"]["nonce"]["const"],
                )
                request_ids = [request_id for _, request_id, _ in observed]
                for request_id in request_ids:
                    self.assertRegex(request_id, r"^egra-attestation-[0-9a-f]{32}$")
                    self.assertTrue(all(
                        schema["properties"]["nonce"]["const"] not in request_id
                        for schema in schemas
                    ))
                self.assertEqual(len(set(request_ids)), 2)
                self.assertEqual(result["post_count"], 2)
                self.assertEqual(len((artifact_root / "endpoint.jsonl").read_text().splitlines()), 2)
        finally:
            server.shutdown()
            thread.join()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
