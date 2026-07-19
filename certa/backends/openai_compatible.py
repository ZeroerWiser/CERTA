"""OpenAI-compatible backend with explicit reasoning/content isolation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Dict, Mapping, Optional
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

import jsonschema


_LOCAL_TOKENIZER_JSONS = {
    "DeepSeek-R1-Distill-Qwen-7B": "/home/common_data/llm/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/tokenizer.json",
    "Qwen2.5-7B-Instruct": "/home/common_data/llm/Qwen/Qwen2.5-7B-Instruct/tokenizer.json",
}


class ResponseContractError(RuntimeError):
    """Raised when an endpoint violates the declared payload contract."""

    def __init__(
        self, message: str, *, audit_metadata: Optional[Mapping[str, Any]] = None
    ) -> None:
        super().__init__(message)
        self.audit_metadata = dict(audit_metadata or {})


@dataclass(frozen=True)
class RoleSampling:
    temperature: float
    top_p: float
    max_tokens: int
    seed: int


@dataclass(frozen=True)
class BackboneProfile:
    profile_id: str
    model: str
    base_url: str
    reasoning_parser: str
    max_model_length: int
    chat_template: str
    no_system_prompt: bool
    role_sampling: Mapping[str, RoleSampling]

    def sampling_for(self, role: str) -> RoleSampling:
        if role not in self.role_sampling:
            raise ValueError(f"unsupported generation role: {role}")
        return self.role_sampling[role]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "model": self.model,
            "base_url": self.base_url.rstrip("/"),
            "reasoning_parser": self.reasoning_parser,
            "max_model_length": self.max_model_length,
            "chat_template": self.chat_template,
            "no_system_prompt": self.no_system_prompt,
            "role_sampling": {
                role: asdict(sampling) for role, sampling in sorted(self.role_sampling.items())
            },
        }

    @property
    def profile_sha256(self) -> str:
        return _canonical_hash(self.to_dict())

    @property
    def chat_template_sha256(self) -> str:
        return hashlib.sha256(self.chat_template.encode("utf-8")).hexdigest()


DEEPSEEK_R1_DISTILL_QWEN_7B_PROFILE = BackboneProfile(
    profile_id="deepseek_r1_distill_qwen_7b_round13r_v2",
    model="DeepSeek-R1-Distill-Qwen-7B",
    base_url="http://127.0.0.1:30337/v1",
    reasoning_parser="deepseek_r1",
    max_model_length=32768,
    chat_template="user_only_no_system_v1",
    no_system_prompt=True,
    role_sampling={
        "proposal": RoleSampling(temperature=0.6, top_p=0.95, max_tokens=8192, seed=0),
        "query": RoleSampling(temperature=0.0, top_p=1.0, max_tokens=2048, seed=0),
        "final": RoleSampling(temperature=0.6, top_p=0.95, max_tokens=8192, seed=0),
    },
)


QWEN2_5_7B_PROFILE = BackboneProfile(
    profile_id="qwen2_5_7b_round13_v1",
    model="Qwen2.5-7B-Instruct",
    base_url="http://127.0.0.1:30300/v1",
    reasoning_parser="none",
    max_model_length=16384,
    chat_template="user_only_server_default_qwen_v1",
    no_system_prompt=True,
    role_sampling={
        "proposal": RoleSampling(temperature=0.0, top_p=1.0, max_tokens=64, seed=0),
        "query": RoleSampling(temperature=0.0, top_p=1.0, max_tokens=512, seed=0),
        "final": RoleSampling(temperature=0.0, top_p=1.0, max_tokens=64, seed=0),
    },
)


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class HttpJsonTransport:
    """Small local-endpoint transport that bypasses ambient proxy variables."""

    def __init__(self, *, timeout: float = 120.0, api_key: str = "EMPTY"):
        self.timeout = timeout
        self.api_key = api_key
        self.opener = build_opener(ProxyHandler({}))

    def _request(self, request: Request) -> Dict[str, Any]:
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise ResponseContractError(f"endpoint HTTP {error.code}: {body}") from error
        except URLError as error:
            raise ResponseContractError(f"endpoint connection failed: {error}") from error
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as error:
            raise ResponseContractError("endpoint did not return valid JSON") from error
        if not isinstance(payload, dict):
            raise ResponseContractError("endpoint JSON root must be an object")
        return payload

    def get_json(self, url: str) -> Dict[str, Any]:
        return self._request(Request(url=url, method="GET"))

    def post_json(self, url: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return self._request(Request(url=url, data=data, headers=headers, method="POST"))


@dataclass(frozen=True)
class BackendResponse:
    content: str
    reasoning_audit: str
    reasoning_source: str
    parsed_content: Any
    served_model: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    content_tokens: int
    token_split_source: str
    total_tokens: int
    response_field_presence: Mapping[str, bool]
    request_payload: Mapping[str, Any]
    cache_key: str
    raw_response: Mapping[str, Any]
    audit_metadata: Mapping[str, Any]


class OpenAICompatibleBackend:
    def __init__(
        self,
        profile: BackboneProfile,
        *,
        transport: Optional[Any] = None,
        timeout: float = 120.0,
    ):
        if not profile.no_system_prompt:
            raise ValueError("Round 13 OpenAI-compatible profiles must prohibit system prompts")
        self.profile = profile
        self.transport = transport or HttpJsonTransport(timeout=timeout)
        self._field_tokenizer = None

    def _count_response_field_tokens(self, value: str) -> int:
        tokenizer_path = _LOCAL_TOKENIZER_JSONS.get(self.profile.model, "")
        if not tokenizer_path:
            return 0
        if self._field_tokenizer is None:
            from tokenizers import Tokenizer

            self._field_tokenizer = Tokenizer.from_file(tokenizer_path)
        return len(self._field_tokenizer.encode(value).ids)

    @property
    def models_url(self) -> str:
        return f"{self.profile.base_url.rstrip('/')}/models"

    @property
    def completions_url(self) -> str:
        return f"{self.profile.base_url.rstrip('/')}/chat/completions"

    def list_models(self) -> list[str]:
        payload = self.transport.get_json(self.models_url)
        data = payload.get("data") or []
        if not isinstance(data, list):
            raise ResponseContractError("/models data must be a list")
        return [str(item.get("id")) for item in data if isinstance(item, Mapping) and item.get("id")]

    def build_messages(self, user_prompt: str) -> list[Dict[str, str]]:
        return [{"role": "user", "content": str(user_prompt)}]

    def cache_key(
        self,
        user_prompt: str,
        *,
        role: str,
        cache_context: Optional[Mapping[str, Any]] = None,
        response_schema: Optional[Mapping[str, Any]] = None,
        schema_name: str = "",
    ) -> str:
        sampling = self.profile.sampling_for(role)
        payload = {
            "profile": self.profile.to_dict(),
            "profile_sha256": self.profile.profile_sha256,
            "chat_template_sha256": self.profile.chat_template_sha256,
            "reasoning_parser": self.profile.reasoning_parser,
            "role": role,
            "messages": self.build_messages(user_prompt),
            "sampling": asdict(sampling),
            "cache_context": dict(cache_context or {}),
            "response_schema": dict(response_schema or {}),
            "schema_name": schema_name,
        }
        return _canonical_hash(payload)

    def _request_payload(
        self,
        user_prompt: str,
        *,
        role: str,
        response_schema: Optional[Mapping[str, Any]] = None,
        schema_name: str = "",
    ) -> Dict[str, Any]:
        sampling = self.profile.sampling_for(role)
        payload: Dict[str, Any] = {
            "model": self.profile.model,
            "messages": self.build_messages(user_prompt),
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "max_tokens": sampling.max_tokens,
            "seed": sampling.seed,
            "stream": False,
        }
        if response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": dict(response_schema),
                    "strict": True,
                },
            }
        return payload

    def complete(
        self,
        user_prompt: str,
        *,
        role: str,
        cache_context: Optional[Mapping[str, Any]] = None,
    ) -> BackendResponse:
        return self._complete(
            user_prompt,
            role=role,
            cache_context=cache_context,
            response_schema=None,
            schema_name="",
        )

    def complete_json(
        self,
        user_prompt: str,
        *,
        role: str,
        response_schema: Mapping[str, Any],
        schema_name: str,
        cache_context: Optional[Mapping[str, Any]] = None,
    ) -> BackendResponse:
        return self._complete(
            user_prompt,
            role=role,
            cache_context=cache_context,
            response_schema=response_schema,
            schema_name=schema_name,
        )

    def _complete(
        self,
        user_prompt: str,
        *,
        role: str,
        cache_context: Optional[Mapping[str, Any]],
        response_schema: Optional[Mapping[str, Any]],
        schema_name: str,
    ) -> BackendResponse:
        request_payload = self._request_payload(
            user_prompt,
            role=role,
            response_schema=response_schema,
            schema_name=schema_name,
        )
        key = self.cache_key(
            user_prompt,
            role=role,
            cache_context=cache_context,
            response_schema=response_schema,
            schema_name=schema_name,
        )
        raw = self.transport.post_json(self.completions_url, request_payload)
        choices = raw.get("choices") or []
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
            raise ResponseContractError(
                "response requires choices[0]",
                audit_metadata=self._audit_metadata(request_payload, raw),
            )
        choice = choices[0]
        message = choice.get("message") or {}
        if not isinstance(message, Mapping):
            raise ResponseContractError(
                "choices[0].message must be an object",
                audit_metadata=self._audit_metadata(request_payload, raw, choice=choice),
            )
        content_value = message.get("content")
        content = content_value if isinstance(content_value, str) else ""

        if isinstance(message.get("reasoning"), str):
            reasoning = str(message.get("reasoning"))
            reasoning_source = "message.reasoning"
        elif isinstance(message.get("reasoning_content"), str):
            reasoning = str(message.get("reasoning_content"))
            reasoning_source = "message.reasoning_content"
        else:
            reasoning = ""
            reasoning_source = ""

        usage = raw.get("usage") or {}
        if not isinstance(usage, Mapping):
            usage = {}
        details = usage.get("completion_tokens_details") or {}
        if not isinstance(details, Mapping):
            details = {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        reported_reasoning_tokens = int(details.get("reasoning_tokens") or 0)
        if reported_reasoning_tokens > 0:
            reasoning_tokens = reported_reasoning_tokens
            content_tokens = (
                max(0, completion_tokens - reasoning_tokens) if content.strip() else 0
            )
            token_split_source = "endpoint_completion_tokens_details"
        else:
            reasoning_tokens = self._count_response_field_tokens(reasoning) if reasoning else 0
            content_tokens = self._count_response_field_tokens(content) if content.strip() else 0
            token_split_source = "local_tokenizer_field_encoding"
        total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
        audit_metadata = self._audit_metadata(
            request_payload,
            raw,
            choice=choice,
            message=message,
            reasoning_token_count=reasoning_tokens,
            content_token_count=content_tokens,
        )
        if not content.strip():
            raise ResponseContractError(
                "choices[0].message.content is empty or malformed",
                audit_metadata=audit_metadata,
            )

        parsed_content = None
        if response_schema is not None:
            try:
                parsed_content = json.loads(content)
            except json.JSONDecodeError as error:
                raise ResponseContractError(
                    "message.content is not valid JSON", audit_metadata=audit_metadata
                ) from error
            try:
                jsonschema.validate(parsed_content, dict(response_schema))
            except jsonschema.ValidationError as error:
                raise ResponseContractError(
                    f"message.content violates JSON schema: {error.message}",
                    audit_metadata=audit_metadata,
                ) from error
        return BackendResponse(
            content=content,
            reasoning_audit=reasoning,
            reasoning_source=reasoning_source,
            parsed_content=parsed_content,
            served_model=str(raw.get("model") or ""),
            finish_reason=str(choice.get("finish_reason") or ""),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            content_tokens=content_tokens,
            token_split_source=token_split_source,
            total_tokens=total_tokens,
            response_field_presence={
                "message.content": "content" in message,
                "message.reasoning": "reasoning" in message,
                "message.reasoning_content": "reasoning_content" in message,
            },
            request_payload=request_payload,
            cache_key=key,
            raw_response=raw,
            audit_metadata=audit_metadata,
        )

    @staticmethod
    def _audit_metadata(
        request_payload: Mapping[str, Any],
        raw: Mapping[str, Any],
        *,
        choice: Optional[Mapping[str, Any]] = None,
        message: Optional[Mapping[str, Any]] = None,
        reasoning_token_count: int = 0,
        content_token_count: int = 0,
    ) -> Dict[str, Any]:
        usage_value = raw.get("usage") if isinstance(raw, Mapping) else None
        usage = usage_value if isinstance(usage_value, Mapping) else {}
        details_value = usage.get("completion_tokens_details")
        details = details_value if isinstance(details_value, Mapping) else {}
        sanitized_usage = {
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
            "completion_tokens_details": {
                "reasoning_tokens": int(details.get("reasoning_tokens") or 0),
            },
        }
        message_mapping = message if isinstance(message, Mapping) else {}
        choice_mapping = choice if isinstance(choice, Mapping) else {}
        content = message_mapping.get("content")
        reasoning = (
            message_mapping.get("reasoning")
            if isinstance(message_mapping.get("reasoning"), str)
            else message_mapping.get("reasoning_content")
            if isinstance(message_mapping.get("reasoning_content"), str)
            else ""
        )
        choices_value = raw.get("choices") if isinstance(raw, Mapping) else None
        return {
            "raw_content": content if isinstance(content, str) else "",
            "finish_reason": str(choice_mapping.get("finish_reason") or ""),
            "served_model": str(raw.get("model") or ""),
            "field_presence": {
                "message.content": "content" in message_mapping,
                "message.reasoning": "reasoning" in message_mapping,
                "message.reasoning_content": "reasoning_content" in message_mapping,
            },
            "usage": sanitized_usage,
            "reasoning_token_count": int(reasoning_token_count),
            "content_token_count": int(content_token_count),
            "raw_response_sha256": _canonical_hash(raw),
            "request_sha256": _canonical_hash(request_payload),
            "sanitized_raw_response_metadata": {
                "root_keys": sorted(str(key) for key in raw),
                "choice_count": len(choices_value) if isinstance(choices_value, list) else 0,
                "choice_keys": sorted(str(key) for key in choice_mapping),
                "message_keys": sorted(str(key) for key in message_mapping),
                "content_type": type(content).__name__,
                "content_characters": len(content) if isinstance(content, str) else 0,
                "reasoning_characters": len(reasoning) if isinstance(reasoning, str) else 0,
            },
        }


def extract_content_answer(response: BackendResponse) -> str:
    content = response.content.strip()
    if not content:
        raise ResponseContractError("cannot extract final answer from empty message.content")
    return content
