import inspect
import unittest

import numpy as np

from certa.egra.retrieval import (
    COLUMN_TOP_K,
    MAX_EXPANDED_CARDS,
    MAX_REFERENCE_IDS,
    REGION_TOP_K,
    ROW_TOP_K,
    build_card_index,
    build_index_cache_key,
    build_role_conditioned_query,
    retrieve_structural_cards,
)


class FakeEncoder:
    def encode(self, texts):
        vectors = []
        score_by_id = {
            "R0": 1.0, "R1": 1.0, "R2": 0.9, "R3": 0.8, "R4": -1.0,
            "C0": 1.0, "C1": 0.9, "C2": 0.8, "C3": 0.7, "C4": -1.0,
            "X0": 1.0, "X1": 0.9, "X2": 0.8, "X3": 0.7, "X4": -1.0,
            "X5": -2.0,
        }
        for text in texts:
            if text.startswith("query: "):
                vectors.append([1.0, 0.0])
            else:
                card_id = text.split()[-1]
                score = score_by_id[card_id]
                vectors.append([score, 1.0 - abs(score)])
        return np.asarray(vectors, dtype=np.float32)


class RecordingEncoder:
    def __init__(self):
        self.texts = []

    def encode(self, texts):
        self.texts.extend(texts)
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


def card(card_id, kind, *, neighbors=()):
    axis = {"ROW_PATH": "row", "COLUMN_PATH": "column", "REGION_GROUP": "mixed", "HEADER_SUBTREE": "row"}[kind]
    return {
        "schema_version": "certa_egra_structural_card_v1",
        "card_id": card_id,
        "catalog_sha256": "a" * 64,
        "unit_kind": kind,
        "axis": axis,
        "human_readable_text": f"structural card {card_id}",
        "header_node_ids": [f"header_{card_id}"],
        "member_coordinates": [[1, 1]],
        "neighbor_card_ids": list(neighbors),
        "answer_values_exposed": False,
        "provenance_complete": True,
    }


def role_contract():
    return {
        "schema_version": "certa_egra_query_contract_v1",
        "supported_by_core_signatures": True,
        "answer_domain": "ENTITY",
        "intent_family": "RANK_MAX",
        "signature_candidates": ["ARGMAX_ENTITY"],
        "projection_candidates": ["ROW_ENTITY_PROJECTION"],
        "cardinality": "SINGLE",
        "rank_direction": "MAX",
        "rank_k": None,
        "requires_time_scope": False,
        "requires_unit_consistency": True,
        "unknowns": [],
    }


class RetrievalTests(unittest.TestCase):
    def test_table_local_card_ids_are_not_embedding_features(self):
        cards = [card("R0", "ROW_PATH"), card("R1", "ROW_PATH")]
        for item in cards:
            item["human_readable_text"] = "same canonical structural description"
        encoder = RecordingEncoder()
        build_card_index(
            cards,
            encoder,
            parent_sha="9" * 40,
            table_sha256="b" * 64,
            embedding_file_tree_sha256="c" * 64,
        )
        self.assertEqual(encoder.texts, [
            "passage: same canonical structural description",
            "passage: same canonical structural description",
        ])

    def test_role_query_accepts_only_question_and_frozen_contract(self):
        self.assertEqual(
            tuple(inspect.signature(build_role_conditioned_query).parameters),
            ("question", "contract"),
        )
        query = build_role_conditioned_query("Which region is largest?", role_contract())
        self.assertTrue(query.startswith("query: "))
        self.assertIn("Which region is largest?", query)
        for forbidden in ("B0", "candidate_answer", "gold_answer", "correctness"):
            self.assertNotIn(forbidden, query)
        self.assertNotIn("requires_time_scope", query)
        self.assertNotIn("requires_unit_consistency", query)

    def test_fixed_kind_budgets_ties_no_threshold_and_catalog_expansion(self):
        cards = []
        for prefix, kind in (("R", "ROW_PATH"), ("C", "COLUMN_PATH"), ("X", "REGION_GROUP")):
            for index in range(5):
                neighbors = ("H0", "X4") if prefix == "R" and index == 0 else ()
                cards.append(card(f"{prefix}{index}", kind, neighbors=neighbors))
        cards.append(card("X5", "REGION_GROUP", neighbors=("R0", "C0")))
        cards.append(card("H0", "HEADER_SUBTREE", neighbors=("R0",)))

        index = build_card_index(
            cards,
            FakeEncoder(),
            parent_sha="9" * 40,
            table_sha256="b" * 64,
            embedding_file_tree_sha256="c" * 64,
        )
        self.assertNotIn("H0", index["card_ids"])
        result = retrieve_structural_cards(
            index,
            cards,
            question="Which region is largest?",
            contract=role_contract(),
            encoder=FakeEncoder(),
        )
        self.assertEqual((ROW_TOP_K, COLUMN_TOP_K, REGION_TOP_K), (4, 4, 4))
        self.assertEqual(MAX_EXPANDED_CARDS, 20)
        self.assertEqual(MAX_REFERENCE_IDS, 64)
        self.assertEqual(result["retrieved_card_ids_by_kind"]["ROW_PATH"][:2], ["R0", "R1"])
        self.assertEqual(len(result["retrieved_card_ids_by_kind"]["ROW_PATH"]), 4)
        self.assertEqual(len(result["retrieved_card_ids_by_kind"]["COLUMN_PATH"]), 4)
        self.assertEqual(len(result["retrieved_card_ids_by_kind"]["REGION_GROUP"]), 4)
        self.assertIn("R4", result["retrieved_card_ids_by_kind"]["ROW_PATH"] + result["dropped_ranked_card_ids"])
        self.assertIn("H0", result["expanded_card_ids"])
        self.assertNotIn("X4", result["expanded_card_ids"])
        self.assertIn("X5", result["expanded_card_ids"])
        self.assertIsNone(result["similarity_threshold"])
        self.assertLessEqual(len(result["selected_card_ids"]), MAX_EXPANDED_CARDS)
        self.assertLessEqual(len(result["reference_node_ids"]), MAX_REFERENCE_IDS)

    def test_index_cache_key_binds_every_frozen_identity(self):
        values = {
            "parent_sha": "9" * 40,
            "table_sha256": "a" * 64,
            "catalog_sha256": "b" * 64,
            "card_schema_sha256": "c" * 64,
            "embedding_file_tree_sha256": "d" * 64,
        }
        baseline = build_index_cache_key(**values)
        self.assertEqual(baseline, build_index_cache_key(**values))
        for key in values:
            changed = dict(values)
            changed[key] = "e" * len(values[key])
            self.assertNotEqual(baseline, build_index_cache_key(**changed))


if __name__ == "__main__":
    unittest.main()
