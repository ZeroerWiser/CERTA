import copy
import inspect
import unittest

import jsonschema

from graph_builder import EdgeType, GraphEdge, GraphNode, HCEG, NodeType

from certa.egra.evidence_cards import (
    build_structural_card_schema,
    build_structural_evidence_cards,
)
from certa.planner.schema_view import build_canonical_structural_group_catalog
from certa.reproducibility.canonical_json import canonical_json_hash


def fixture_catalog():
    graph = HCEG()
    graph.add_node(GraphNode("measure", NodeType.HEADER, row=0, col=1, text="Population"))
    graph.add_node(GraphNode("time", NodeType.HEADER, row=0, col=2, text="2020"))
    graph.add_node(GraphNode("entity", NodeType.HEADER, row=1, col=0, text="North"))
    graph.add_node(GraphNode("value_a", NodeType.CELL, row=1, col=1, text="42", numeric_value=42.0))
    graph.add_node(GraphNode("value_b", NodeType.CELL, row=1, col=2, text="99", numeric_value=99.0))
    graph.add_edge(GraphEdge("value_a", "entity", EdgeType.ROW_PATH))
    graph.add_edge(GraphEdge("value_a", "measure", EdgeType.COL_PATH))
    graph.add_edge(GraphEdge("value_b", "entity", EdgeType.ROW_PATH))
    graph.add_edge(GraphEdge("value_b", "time", EdgeType.COL_PATH))
    table = {
        "texts": [["Region", "Population", "2020"], ["North", "42", "99"]],
        "top_header_rows_num": 1,
        "left_header_columns_num": 1,
    }
    return build_canonical_structural_group_catalog(graph=graph, table_json=table)


class StructuralEvidenceCardTests(unittest.TestCase):
    def test_card_schema_is_the_exact_pack_schema(self):
        self.assertEqual(
            canonical_json_hash(build_structural_card_schema()),
            "374e93046adbdbba39739ce3f4179e0970a61ce51030ec1ce08e44ba75f324a4",
        )

    def test_cards_are_deterministic_value_free_wrappers_of_the_canonical_catalog(self):
        catalog = fixture_catalog()
        frozen_catalog = copy.deepcopy(catalog)
        cards = build_structural_evidence_cards(catalog)
        self.assertEqual(cards, build_structural_evidence_cards(catalog))
        self.assertEqual(catalog, frozen_catalog)
        self.assertEqual(
            tuple(inspect.signature(build_structural_evidence_cards).parameters),
            ("catalog",),
        )
        self.assertEqual(len(cards), len(catalog["all_groups"]))
        self.assertEqual(
            {card["card_id"] for card in cards},
            set(catalog["group_by_id"]),
        )
        self.assertEqual(
            {card["unit_kind"] for card in cards},
            {"ROW_PATH", "COLUMN_PATH", "HEADER_SUBTREE", "REGION_GROUP"},
        )
        schema = build_structural_card_schema()
        catalog_groups = catalog["group_by_id"]
        card_ids = {card["card_id"] for card in cards}
        for card in cards:
            jsonschema.validate(card, schema)
            self.assertFalse(card["answer_values_exposed"])
            self.assertTrue(card["provenance_complete"])
            self.assertEqual(card["catalog_sha256"], catalog["catalog_sha256"])
            group = catalog_groups[card["card_id"]]
            self.assertEqual(card["header_node_ids"], group["ordered_header_node_ids"])
            self.assertEqual(
                card["member_coordinates"],
                group["member_descriptor"]["member_coordinates"],
            )
            self.assertNotIn("42", card["human_readable_text"])
            self.assertNotIn("99", card["human_readable_text"])
            self.assertNotIn(card["card_id"], card["neighbor_card_ids"])
            self.assertLessEqual(set(card["neighbor_card_ids"]), card_ids)

    def test_tampered_catalog_hash_fails_closed(self):
        catalog = fixture_catalog()
        catalog["all_groups"][0]["axis"] = "mixed"
        with self.assertRaisesRegex(ValueError, "catalog_hash_mismatch"):
            build_structural_evidence_cards(catalog)

    def test_empty_header_description_is_not_fabricated(self):
        graph = HCEG()
        graph.add_node(GraphNode("measure", NodeType.HEADER, row=0, col=1, text="Value"))
        graph.add_node(GraphNode("entity", NodeType.HEADER, row=1, col=0, text=""))
        graph.add_node(GraphNode("cell", NodeType.CELL, row=1, col=1, text="ANSWER_LEAK_SENTINEL"))
        graph.add_edge(GraphEdge("cell", "entity", EdgeType.ROW_PATH))
        graph.add_edge(GraphEdge("cell", "measure", EdgeType.COL_PATH))
        catalog = build_canonical_structural_group_catalog(
            graph=graph,
            table_json={
                "texts": [["", "Value"], ["", "ANSWER_LEAK_SENTINEL"]],
                "top_header_rows_num": 1,
                "left_header_columns_num": 1,
            },
        )
        with self.assertRaisesRegex(ValueError, "empty_display_description"):
            build_structural_evidence_cards(catalog)


if __name__ == "__main__":
    unittest.main()
