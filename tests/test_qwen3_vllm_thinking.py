import argparse
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_cscr_pipeline import (
    OpenAIChatGenerator,
    _api_chat_template_kwargs,
    _make_llm_input_audit_record,
    extract_answer,
)


class Qwen3VllmThinkingTests(unittest.TestCase):
    def test_local_vllm_transport_ignores_ambient_proxy_settings(self):
        proxy = "http://127.0.0.1:9"
        with patch.dict(
            os.environ,
            {"HTTP_PROXY": proxy, "HTTPS_PROXY": proxy, "NO_PROXY": ""},
            clear=False,
        ):
            generator = OpenAIChatGenerator(
                model="Qwen3-8B",
                api_base_url="http://127.0.0.1:30338/v1",
                api_key_env="EMPTY",
                max_retries=0,
                cache_mode="off",
                backend_name="vllm_chat",
            )
        try:
            client = generator.client._client
            transport = client._transport_for_url(
                httpx.URL("http://127.0.0.1:30338/v1/chat/completions")
            )
            self.assertFalse(client._trust_env)
            self.assertIs(transport, client._transport)
            self.assertEqual(type(transport._pool).__name__, "ConnectionPool")
        finally:
            generator.client.close()

    def test_qwen3_vllm_chat_disables_thinking_without_affecting_other_models(self):
        self.assertEqual(_api_chat_template_kwargs("vllm_chat", "Qwen3-8B"), {"enable_thinking": False})
        self.assertEqual(_api_chat_template_kwargs("vllm_chat", "Qwen2.5-7B-Instruct"), {})
        self.assertEqual(_api_chat_template_kwargs("openai_chat", "Qwen3-8B"), {})

    def test_qwen3_chat_template_kwargs_are_part_of_cache_key(self):
        generator = OpenAIChatGenerator.__new__(OpenAIChatGenerator)
        generator.backend_name = "vllm_chat"
        generator.model = "Qwen3-8B"
        generator.api_base_url = "http://127.0.0.1:30338/v1"
        generator.chat_template_kwargs = {"enable_thinking": False}
        disabled_key = generator._cache_key("answer only", 32, 0.0, 1.0)
        generator.chat_template_kwargs = {}
        self.assertNotEqual(disabled_key, generator._cache_key("answer only", 32, 0.0, 1.0))

    def test_incomplete_thinking_output_is_not_treated_as_answer(self):
        self.assertEqual(extract_answer("<think>\ntruncated"), "")
        self.assertEqual(extract_answer("<think>reasoning</think>\n52.1"), "52.1")

    def test_qwen3_transport_kwargs_are_in_request_audit_hash(self):
        generator = OpenAIChatGenerator.__new__(OpenAIChatGenerator)
        generator.chat_template_kwargs = {"enable_thinking": False}
        args = argparse.Namespace(
            save_llm_inputs="hash",
            output_dir="/tmp/certa-test",
            llm_input_audit_file="llm_input_audit.jsonl",
            generator_backend="vllm_chat",
            model_path="Qwen3-8B",
            api_model="Qwen3-8B",
            api_base_url="http://127.0.0.1:30338/v1",
        )
        _, with_transport = _make_llm_input_audit_record(
            prepared={"item": {"id": "s-1"}},
            prompt="answer only",
            generator=generator,
            args=args,
            prompt_kind="main",
            prompt_type="default",
            max_new_tokens=32,
            temperature=0.0,
            top_p=1.0,
            logprobs=0,
        )
        generator.chat_template_kwargs = {}
        _, without_transport = _make_llm_input_audit_record(
            prepared={"item": {"id": "s-1"}},
            prompt="answer only",
            generator=generator,
            args=args,
            prompt_kind="main",
            prompt_type="default",
            max_new_tokens=32,
            temperature=0.0,
            top_p=1.0,
            logprobs=0,
        )
        self.assertNotEqual(with_transport["request_sha256"], without_transport["request_sha256"])
