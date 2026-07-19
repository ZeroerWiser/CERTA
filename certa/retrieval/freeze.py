"""Frozen final-request budget, template, tokenizer, and cache identity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence, Tuple

from tokenizers import Tokenizer

from .constants import RETRIEVER_VERSION

__all__ = [
    "DEEPSEEK_FINAL_CONTEXT_PROFILE",
    "QWEN_FINAL_CONTEXT_PROFILE",
    "FINAL_ANSWER_TEMPLATE_VERSION",
    "FINAL_TOKEN_CAP",
    "RETRIEVER_VERSION",
    "FinalContextContract",
    "Round13CacheIdentity",
    "SanitizedEvidenceItem",
    "serialize_sanitized_evidence",
]


FINAL_ANSWER_TEMPLATE_VERSION = "certa_round13r_structured_answer_v1"
FINAL_TOKEN_CAP = 4096


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FinalContextProfile:
    profile_id: str
    model: str
    tokenizer_json: str
    tokenizer_config_json: str
    chat_template_id: str
    chat_prefix: str
    chat_user_suffix: str
    final_token_cap: int = FINAL_TOKEN_CAP

    @property
    def chat_template_sha256(self) -> str:
        return _sha256_text(f"{self.chat_prefix}{{user_prompt}}{self.chat_user_suffix}")


QWEN_FINAL_CONTEXT_PROFILE = FinalContextProfile(
    profile_id="qwen2_5_7b_round13_final_context_v1",
    model="Qwen2.5-7B-Instruct",
    tokenizer_json="/home/common_data/llm/Qwen/Qwen2.5-7B-Instruct/tokenizer.json",
    tokenizer_config_json="/home/common_data/llm/Qwen/Qwen2.5-7B-Instruct/tokenizer_config.json",
    chat_template_id="qwen2_5_default_user_v1",
    chat_prefix=(
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. "
        "You are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
    ),
    chat_user_suffix="<|im_end|>\n<|im_start|>assistant\n",
)


DEEPSEEK_FINAL_CONTEXT_PROFILE = FinalContextProfile(
    profile_id="deepseek_r1_distill_qwen_7b_round13_final_context_v1",
    model="DeepSeek-R1-Distill-Qwen-7B",
    tokenizer_json="/home/common_data/llm/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/tokenizer.json",
    tokenizer_config_json="/home/common_data/llm/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/tokenizer_config.json",
    chat_template_id="deepseek_r1_user_only_v1",
    chat_prefix="<｜begin▁of▁sentence｜><｜User｜>",
    chat_user_suffix="<｜Assistant｜><think>\n",
)


@dataclass(frozen=True)
class FinalAnswerRequest:
    template_version: str
    user_prompt: str
    messages: Tuple[Mapping[str, str], ...]
    rendered_chat: str
    input_token_count: int
    final_token_cap: int
    tokenizer_json: str
    chat_template_sha256: str
    request_sha256: str


class FinalContextOverflow(ValueError):
    pass


@dataclass(frozen=True)
class SanitizedEvidenceItem:
    table_id: str
    sheet: str
    row: int
    column: int
    structural_path: Tuple[str, ...]
    text: str

    @property
    def identity(self) -> Tuple[str, str, int, int, Tuple[str, ...]]:
        return (self.table_id, self.sheet, self.row, self.column, self.structural_path)


def serialize_sanitized_evidence(items: Sequence[SanitizedEvidenceItem]) -> Tuple[str, ...]:
    deduplicated = {item.identity: item for item in items}
    ordered = sorted(
        deduplicated.values(),
        key=lambda item: (item.table_id, item.sheet, item.row, item.column, item.structural_path, item.text),
    )
    return tuple(
        (
            f"[table={item.table_id};sheet={item.sheet};row={item.row};"
            f"column={item.column};path={' > '.join(item.structural_path)}] {item.text}"
        )
        for item in ordered
    )


class FinalContextContract:
    def __init__(self, profile: FinalContextProfile):
        self.profile = profile
        self.tokenizer = Tokenizer.from_file(profile.tokenizer_json)

    @staticmethod
    def _user_prompt(question: str, evidence: Sequence[str]) -> str:
        evidence_text = "\n".join(str(item) for item in evidence)
        return (
            "Answer the table question using only the table evidence below.\n"
            "Question:\n"
            f"{question}\n"
            "Table evidence:\n"
            f"{evidence_text}\n"
            'Return exactly one JSON object matching {"answer": "nonempty string"}. '
            "Put only the concise answer in the answer field. Preserve entities, numeric "
            "precision, units, and list order when required."
        )

    def count_user_prompt(self, user_prompt: str) -> Tuple[str, int]:
        rendered = f"{self.profile.chat_prefix}{user_prompt}{self.profile.chat_user_suffix}"
        count = len(self.tokenizer.encode(rendered, add_special_tokens=False).ids)
        return rendered, count

    def render(self, question: str, evidence: Sequence[str]) -> FinalAnswerRequest:
        user_prompt = self._user_prompt(question, evidence)
        rendered, count = self.count_user_prompt(user_prompt)
        if count > self.profile.final_token_cap:
            raise FinalContextOverflow(
                f"final request uses {count} tokens, cap is {self.profile.final_token_cap}"
            )
        messages = ({"role": "user", "content": user_prompt},)
        request_payload = {
            "template_version": FINAL_ANSWER_TEMPLATE_VERSION,
            "model": self.profile.model,
            "messages": messages,
            "rendered_chat_sha256": _sha256_text(rendered),
            "input_token_count": count,
            "final_token_cap": self.profile.final_token_cap,
            "tokenizer_json": self.profile.tokenizer_json,
            "chat_template_sha256": self.profile.chat_template_sha256,
        }
        return FinalAnswerRequest(
            template_version=FINAL_ANSWER_TEMPLATE_VERSION,
            user_prompt=user_prompt,
            messages=messages,
            rendered_chat=rendered,
            input_token_count=count,
            final_token_cap=self.profile.final_token_cap,
            tokenizer_json=self.profile.tokenizer_json,
            chat_template_sha256=self.profile.chat_template_sha256,
            request_sha256=_sha256_json(request_payload),
        )

    def evidence_budget(self, question: str) -> int:
        empty_request = self.render(question, [])
        return self.profile.final_token_cap - empty_request.input_token_count


@dataclass(frozen=True)
class Round13CacheIdentity:
    dataset: str
    sample_id: str
    table_sha256: str
    model: str
    endpoint_contract_sha256: str
    chat_template_sha256: str
    reasoning_parser: str
    role: str
    retrieval_arm: str
    query_bundle_sha256: str
    retriever_version: str
    evidence_budget: int
    final_budget: int
    sampling_sha256: str
    seed: int

    @property
    def sha256(self) -> str:
        return _sha256_json(asdict(self))
