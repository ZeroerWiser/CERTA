"""Fixed multilingual-E5 retrieval over canonical structural cards."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from certa.egra.evidence_cards import ACTIVE_CARD_KINDS, build_structural_card_schema
from certa.reproducibility.canonical_json import canonical_json, canonical_json_hash


EMBEDDING_MODEL_ID = "intfloat/multilingual-e5-large"
EMBEDDING_MODEL_PATH = "/home/common_data/llm/intfloat/multilingual-e5-large"
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "
POOLING = "mean"
NORMALIZE_EMBEDDINGS = True
MAX_LENGTH = 512
ROW_TOP_K = 4
COLUMN_TOP_K = 4
REGION_TOP_K = 4
MAX_EXPANDED_CARDS = 20
MAX_REFERENCE_IDS = 64
_TOP_K = {
    "ROW_PATH": ROW_TOP_K,
    "COLUMN_PATH": COLUMN_TOP_K,
    "REGION_GROUP": REGION_TOP_K,
}
_EXPANSION_PRIORITY = {"REGION_GROUP": 0, "HEADER_SUBTREE": 1}


def build_role_conditioned_query(
    question: str,
    contract: Mapping[str, Any],
) -> str:
    """Serialize only question and frozen role fields into the E5 query."""
    payload = {
        "question": str(question or ""),
        "answer_domain": str(contract.get("answer_domain") or ""),
        "intent_family": str(contract.get("intent_family") or ""),
        "projection_candidates": list(contract.get("projection_candidates") or []),
        "cardinality": str(contract.get("cardinality") or ""),
        "rank_direction": str(contract.get("rank_direction") or ""),
        "rank_k": contract.get("rank_k"),
    }
    return f"{QUERY_PREFIX}{canonical_json(payload)}"


def build_index_cache_key(
    *,
    parent_sha: str,
    table_sha256: str,
    catalog_sha256: str,
    card_schema_sha256: str,
    embedding_file_tree_sha256: str,
) -> str:
    return canonical_json_hash({
        "parent_sha": str(parent_sha),
        "table_sha256": str(table_sha256),
        "catalog_sha256": str(catalog_sha256),
        "card_schema_sha256": str(card_schema_sha256),
        "embedding_file_tree_sha256": str(embedding_file_tree_sha256),
        "model_id": EMBEDDING_MODEL_ID,
        "pooling": POOLING,
        "normalize_embeddings": NORMALIZE_EMBEDDINGS,
        "max_length": MAX_LENGTH,
    })


def _normalized_rows(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError("embedding_matrix_must_be_nonempty_2d")
    if not np.isfinite(array).all():
        raise ValueError("embedding_matrix_nonfinite")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("embedding_zero_norm")
    return array / norms


def build_card_index(
    cards: Sequence[Mapping[str, Any]],
    encoder: Any,
    *,
    parent_sha: str,
    table_sha256: str,
    embedding_file_tree_sha256: str,
) -> Dict[str, Any]:
    """Build one question-independent active-card index for a table."""
    active = sorted(
        (dict(card) for card in cards if card.get("unit_kind") in ACTIVE_CARD_KINDS),
        key=lambda card: str(card["card_id"]),
    )
    if not active:
        raise ValueError("no_active_structural_cards")
    if any(card.get("answer_values_exposed") for card in active):
        raise ValueError("answer_value_exposed_in_card")
    if any(not card.get("provenance_complete") for card in active):
        raise ValueError("incomplete_card_provenance")
    catalog_hashes = {str(card.get("catalog_sha256") or "") for card in active}
    if len(catalog_hashes) != 1:
        raise ValueError("mixed_catalog_hashes")
    catalog_sha256 = next(iter(catalog_hashes))
    card_schema_sha256 = canonical_json_hash(build_structural_card_schema())
    passages = [
        f"{PASSAGE_PREFIX}{card['human_readable_text']}"
        for card in active
    ]
    embeddings = _normalized_rows(encoder.encode(passages))
    if embeddings.shape[0] != len(active):
        raise ValueError("embedding_card_count_mismatch")
    cache_key = build_index_cache_key(
        parent_sha=parent_sha,
        table_sha256=table_sha256,
        catalog_sha256=catalog_sha256,
        card_schema_sha256=card_schema_sha256,
        embedding_file_tree_sha256=embedding_file_tree_sha256,
    )
    payload = {
        "schema_version": "certa_egra_card_index_v1",
        "cache_key": cache_key,
        "catalog_sha256": catalog_sha256,
        "card_schema_sha256": card_schema_sha256,
        "embedding_file_tree_sha256": str(embedding_file_tree_sha256),
        "card_ids": [str(card["card_id"]) for card in active],
        "card_kinds": [str(card["unit_kind"]) for card in active],
        "passage_sha256": canonical_json_hash(passages),
        "embeddings": embeddings.tolist(),
    }
    payload["index_sha256"] = canonical_json_hash(payload)
    return payload


def retrieve_structural_cards(
    index: Mapping[str, Any],
    cards: Sequence[Mapping[str, Any]],
    *,
    question: str,
    contract: Mapping[str, Any],
    encoder: Any,
) -> Dict[str, Any]:
    """Retrieve with fixed per-kind K, deterministic ties, and catalog-only expansion."""
    card_ids = [str(item) for item in index.get("card_ids") or []]
    card_kinds = [str(item) for item in index.get("card_kinds") or []]
    embeddings = _normalized_rows(index.get("embeddings") or [])
    if embeddings.shape[0] != len(card_ids) or len(card_ids) != len(card_kinds):
        raise ValueError("invalid_card_index_shape")
    if len(card_ids) != len(set(card_ids)):
        raise ValueError("duplicate_index_card_id")
    cards_by_id = {str(card["card_id"]): dict(card) for card in cards}
    if len(cards_by_id) != len(cards):
        raise ValueError("duplicate_card_id")
    if set(card_ids) - set(cards_by_id):
        raise ValueError("index_card_missing_from_catalog")

    query_text = build_role_conditioned_query(question, contract)
    query_vector = _normalized_rows(encoder.encode([query_text]))[0]
    if query_vector.shape[0] != embeddings.shape[1]:
        raise ValueError("query_index_dimension_mismatch")
    similarities = embeddings @ query_vector
    scores = {card_id: float(score) for card_id, score in zip(card_ids, similarities)}
    ranked_by_kind: Dict[str, list[str]] = {}
    retrieved_by_kind: Dict[str, list[str]] = {}
    for kind in ACTIVE_CARD_KINDS:
        ranked = sorted(
            (card_id for card_id, card_kind in zip(card_ids, card_kinds) if card_kind == kind),
            key=lambda card_id: (-scores[card_id], card_id),
        )
        ranked_by_kind[kind] = ranked
        retrieved_by_kind[kind] = ranked[:_TOP_K[kind]]

    base_ids = [
        card_id
        for kind in ACTIVE_CARD_KINDS
        for card_id in retrieved_by_kind[kind]
    ]
    subtree_candidates = {
        neighbor_id
        for card_id in base_ids
        for neighbor_id in cards_by_id[card_id].get("neighbor_card_ids") or []
        if neighbor_id in cards_by_id
        and cards_by_id[neighbor_id].get("unit_kind") == "HEADER_SUBTREE"
        and neighbor_id not in base_ids
    }
    selected_rows = set(retrieved_by_kind["ROW_PATH"])
    selected_columns = set(retrieved_by_kind["COLUMN_PATH"])
    joined_region_candidates = {
        card_id
        for card_id, card in cards_by_id.items()
        if card.get("unit_kind") == "REGION_GROUP"
        and card_id not in base_ids
        and selected_rows.intersection(card.get("neighbor_card_ids") or [])
        and selected_columns.intersection(card.get("neighbor_card_ids") or [])
    }
    expansion_candidates = subtree_candidates | joined_region_candidates
    ordered_expansion = sorted(
        expansion_candidates,
        key=lambda card_id: (
            _EXPANSION_PRIORITY[str(cards_by_id[card_id]["unit_kind"])],
            card_id,
        ),
    )
    candidates = (base_ids + ordered_expansion)[:MAX_EXPANDED_CARDS]

    selected: list[str] = []
    reference_ids: list[str] = []
    reference_seen: set[str] = set()
    dropped_budget: list[str] = []
    for card_id in candidates:
        header_ids = [str(item) for item in cards_by_id[card_id]["header_node_ids"]]
        additions = [node_id for node_id in header_ids if node_id not in reference_seen]
        if len(reference_ids) + len(additions) > MAX_REFERENCE_IDS:
            dropped_budget.append(card_id)
            continue
        selected.append(card_id)
        reference_ids.extend(additions)
        reference_seen.update(additions)

    selected_set = set(selected)
    all_ranked = [card_id for kind in ACTIVE_CARD_KINDS for card_id in ranked_by_kind[kind]]
    return {
        "schema_version": "certa_egra_retrieval_result_v1",
        "query_sha256": canonical_json_hash({"query": query_text}),
        "index_sha256": str(index.get("index_sha256") or ""),
        "retrieved_card_ids_by_kind": retrieved_by_kind,
        "expanded_card_ids": [
            card_id for card_id in ordered_expansion if card_id in selected_set
        ],
        "selected_card_ids": selected,
        "reference_node_ids": reference_ids,
        "scores": {card_id: scores[card_id] for card_id in all_ranked},
        "dropped_ranked_card_ids": [
            card_id for card_id in all_ranked if card_id not in set(base_ids)
        ],
        "dropped_reference_budget_card_ids": dropped_budget,
        "similarity_threshold": None,
        "stable_tie_break": "card_id_ascending",
        "budgets": {
            "row_top_k": ROW_TOP_K,
            "column_top_k": COLUMN_TOP_K,
            "region_top_k": REGION_TOP_K,
            "max_expanded_cards": MAX_EXPANDED_CARDS,
            "max_reference_ids": MAX_REFERENCE_IDS,
        },
    }


class FrozenE5Encoder:
    """Lazy local encoder implementing the Pack's exact pooling contract."""

    def __init__(
        self,
        *,
        model_path: str = EMBEDDING_MODEL_PATH,
        device: str = "cpu",
    ) -> None:
        if str(Path(model_path)) != EMBEDDING_MODEL_PATH:
            raise ValueError("embedding_model_path_mismatch")
        if not Path(model_path).is_dir():
            raise FileNotFoundError("embedding_model_unavailable")
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = str(device)
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("explicit_cuda_unavailable")
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=dtype,
        ).to(self.device).eval()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        import torch
        import torch.nn.functional as functional

        inputs = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            hidden = self.model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            normalized = functional.normalize(pooled, p=2, dim=1)
        return normalized.float().cpu().numpy()
