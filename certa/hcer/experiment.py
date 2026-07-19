"""Shared experiment utilities for the bounded Round 13 HCER matrix."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import re
from typing import Any, Mapping, Sequence, Tuple

from certa.backends.openai_compatible import ResponseContractError
from dataset_adapters import aitqa_answer_match
from eval_utils import evaluate_single, normalize_text
from structure_aware_formatter import build_structure_aware_prompt

from certa.datasets.sstqa_zh import evaluate_sstqa_answer
from certa.retrieval.freeze import FinalContextContract, SanitizedEvidenceItem


ANSWER_ENVELOPE_SCHEMA_VERSION = "certa_round13r_answer_v1"
ANSWER_ENVELOPE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer"],
    "properties": {
        "answer": {"type": "string", "minLength": 1, "pattern": r"\S"},
    },
}
ANSWER_ENVELOPE_INSTRUCTION = (
    'Return exactly one JSON object matching {"answer": "nonempty string"}. '
    "Put only the concise answer in the answer field."
)


@dataclass(frozen=True)
class ProposalPromptResult:
    prompt: str
    input_token_count: int
    input_token_cap: int
    total_rows: int
    included_rows: int
    truncated: bool
    policy_version: str = "generic_native_row_prefix_token_fit_v1"


def parse_answer_envelope(payload: Any) -> str:
    """Return the sole nonempty answer field or fail closed."""
    if not isinstance(payload, Mapping) or set(payload) != {"answer"}:
        raise ResponseContractError("answer payload must contain exactly the answer field")
    answer = payload["answer"]
    if not isinstance(answer, str) or not answer.strip():
        raise ResponseContractError("answer payload requires a nonempty string")
    return answer.strip()


def answer_surface_domain(answer: Any) -> str:
    text = str(answer or "").strip()
    has_number = bool(re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", text))
    residual = re.sub(r"[-+]?\d[\d,]*(?:\.\d+)?|[%$£€¥￥\s.,()-]", "", text)
    has_text = bool(residual)
    if has_number and has_text:
        return "mixed"
    if has_number:
        return "numeric"
    return "text" if has_text else "empty"


def evaluate_prediction(dataset: str, prediction: Any, gold: Any) -> dict[str, bool]:
    generic = evaluate_single(prediction, gold)
    result = {
        "normalized_exact_match": bool(generic["strict_em"]),
        "numeric_match": bool(generic["numeric_em"]),
        "set_match": bool(generic["set_em"]),
        "hitab_official_em": bool(generic["hitab_official_em"]),
        "numeric_unit_match": False,
        "ordered_list_match": False,
        "unordered_set_match": False,
    }
    if dataset == "hitab":
        result["official_match"] = result["hitab_official_em"]
    elif dataset == "aitqa":
        result["official_match"] = bool(aitqa_answer_match(gold, prediction))
    elif dataset == "sstqa_zh":
        sst = evaluate_sstqa_answer(prediction, gold)
        result.update({key: bool(value) for key, value in sst.items()})
        result["normalized_exact_match"] = bool(sst["text_exact"])
        result["numeric_match"] = bool(sst["numeric_match"])
        result["set_match"] = bool(sst["unordered_set_match"])
    else:
        raise ValueError(f"unsupported Round 13 dataset: {dataset}")
    return result


def typed_answer_equivalence(dataset: str, proposal: Any, final_answer: Any) -> dict[str, Any]:
    """Compare proposal and final through the dataset's gold-free answer identity."""
    identity = [proposal] if dataset in {"hitab", "aitqa"} else proposal
    evaluation = evaluate_prediction(dataset, final_answer, identity)
    if evaluation.get("numeric_unit_match"):
        identity_type = "numeric_with_unit"
    elif evaluation.get("numeric_match"):
        identity_type = "numeric"
    elif evaluation.get("ordered_list_match"):
        identity_type = "ordered_list"
    elif evaluation.get("set_match") and dataset != "sstqa_zh":
        identity_type = "dataset_authorized_set"
    elif evaluation.get("normalized_exact_match"):
        identity_type = "text_or_entity"
    else:
        identity_type = "none"
    return {
        "equivalent": bool(evaluation["official_match"]),
        "identity_type": identity_type,
        "normalized_exact_match": bool(evaluation.get("normalized_exact_match")),
        "numeric_match": bool(evaluation.get("numeric_match")),
        "numeric_unit_match": bool(evaluation.get("numeric_unit_match")),
        "ordered_list_match": bool(evaluation.get("ordered_list_match")),
        "unordered_set_match": bool(evaluation.get("unordered_set_match")),
        "dataset_authorized_set_match": bool(
            evaluation.get("set_match") and dataset != "sstqa_zh"
        ),
    }


def fit_initial_proposal_prompt(
    *,
    table: Mapping[str, Any],
    question: str,
    backend_profile: Any,
    context_contract: FinalContextContract,
) -> ProposalPromptResult:
    max_tokens = backend_profile.sampling_for("proposal").max_tokens
    input_cap = int(backend_profile.max_model_length) - int(max_tokens)
    total_rows = len(table.get("texts") or [])

    def render(row_count: int) -> Tuple[str, int]:
        view = copy.deepcopy(dict(table))
        view["texts"] = list(table.get("texts") or [])[:row_count]
        view["merged_regions"] = [
            region
            for region in (table.get("merged_regions") or [])
            if int(region.get("last_row", -1)) < row_count
        ]
        prompt = f"{build_structure_aware_prompt(view, question)}\n{ANSWER_ENVELOPE_INSTRUCTION}"
        _, count = context_contract.count_user_prompt(prompt)
        return prompt, count

    full_prompt, full_count = render(total_rows)
    if full_count <= input_cap:
        return ProposalPromptResult(
            prompt=full_prompt,
            input_token_count=full_count,
            input_token_cap=input_cap,
            total_rows=total_rows,
            included_rows=total_rows,
            truncated=False,
        )

    minimum_rows = min(total_rows, max(1, int(table.get("top_header_rows_num", 0) or 0)))
    low, high = minimum_rows, total_rows - 1
    best = None
    while low <= high:
        middle = (low + high) // 2
        prompt, count = render(middle)
        if count <= input_cap:
            best = (middle, prompt, count)
            low = middle + 1
        else:
            high = middle - 1
    if best is None:
        prompt, count = render(minimum_rows)
        raise ValueError(
            f"proposal prompt cannot fit declared context: {count} tokens > {input_cap}"
        )
    included_rows, prompt, count = best
    return ProposalPromptResult(
        prompt=prompt,
        input_token_count=count,
        input_token_cap=input_cap,
        total_rows=total_rows,
        included_rows=included_rows,
        truncated=True,
    )


def _normalized_gold_values(gold: Any) -> Tuple[str, ...]:
    values = gold if isinstance(gold, list) else [gold]
    return tuple(value for value in (normalize_text(item) for item in values) if value)


def answer_containing_evidence(gold: Any, items: Sequence[SanitizedEvidenceItem]) -> bool:
    gold_values = _normalized_gold_values(gold)
    if not gold_values:
        return False
    evidence_values = tuple(normalize_text(item.text) for item in items if item.text)
    return any(
        gold_value == evidence_value
        or (len(gold_value) >= 4 and gold_value in evidence_value)
        for gold_value in gold_values
        for evidence_value in evidence_values
    )


def parse_coordinate(value: str) -> Tuple[int, int] | None:
    match = re.fullmatch(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", str(value))
    return (int(match.group(1)), int(match.group(2))) if match else None


def linked_coordinate_sets(row: Mapping[str, Any]) -> Mapping[str, frozenset[Tuple[int, int]]]:
    quantity = set()
    entities = set()
    links = row.get("linked_cells") or {}
    for mapping in (links.get("quantity_link") or {}).values():
        if isinstance(mapping, Mapping):
            quantity.update(
                coordinate
                for coordinate in (parse_coordinate(value) for value in mapping)
                if coordinate is not None
            )
    for region in (links.get("entity_link") or {}).values():
        if not isinstance(region, Mapping):
            continue
        for mapping in region.values():
            if isinstance(mapping, Mapping):
                entities.update(
                    coordinate
                    for coordinate in (parse_coordinate(value) for value in mapping)
                    if coordinate is not None
                )
    return {"quantity": frozenset(quantity), "entity": frozenset(entities)}
