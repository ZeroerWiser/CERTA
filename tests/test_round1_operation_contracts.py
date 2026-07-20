import copy
import unittest
from types import SimpleNamespace

from graph_builder import EdgeType, GraphEdge, GraphNode, HCEG, NodeType

from certa.derivations.replay import replay_derivation_under_intervention
from certa.derivations.iade import build_sample_fixed_role_intervention_basis, evaluate_derivation_on_basis
from certa.grounding.plan_closure import ClosureOutcome, build_plan_closure
from certa.grounding.structural_resolvers import ResolutionState, resolve_atomic_operand
from certa.operations.contracts import OPERATION_SIGNATURES, validate_operation_plan


def operation_graph():
    graph = HCEG()
    for node in (
        GraphNode("measure_numeric", NodeType.HEADER, row=0, col=1, text="Value"),
        GraphNode("measure_entity", NodeType.HEADER, row=0, col=2, text="Winner"),
        GraphNode("measure_boolean", NodeType.HEADER, row=0, col=3, text="Qualified"),
        GraphNode("entity_a", NodeType.HEADER, row=1, col=0, text="A"),
        GraphNode("entity_b", NodeType.HEADER, row=2, col=0, text="B"),
        GraphNode("entity_c", NodeType.HEADER, row=3, col=0, text="C"),
        GraphNode("entity_d", NodeType.HEADER, row=4, col=0, text="D"),
        GraphNode("numeric_a", NodeType.CELL, row=1, col=1, text="4", numeric_value=4.0),
        GraphNode("numeric_b", NodeType.CELL, row=2, col=1, text="2", numeric_value=2.0),
        GraphNode("numeric_c", NodeType.CELL, row=3, col=1, text="4", numeric_value=4.0),
        GraphNode("numeric_d", NodeType.CELL, row=4, col=1, text="2", numeric_value=2.0),
        GraphNode("entity_value", NodeType.CELL, row=1, col=2, text="Alpha"),
        GraphNode("boolean_value", NodeType.CELL, row=1, col=3, text="true"),
    ):
        graph.add_node(node)
    for cell, entity, measure in (
        ("numeric_a", "entity_a", "measure_numeric"),
        ("numeric_b", "entity_b", "measure_numeric"),
        ("numeric_c", "entity_c", "measure_numeric"),
        ("numeric_d", "entity_d", "measure_numeric"),
        ("entity_value", "entity_a", "measure_entity"),
        ("boolean_value", "entity_a", "measure_boolean"),
    ):
        graph.add_edge(GraphEdge(cell, entity, EdgeType.ROW_PATH))
        graph.add_edge(GraphEdge(cell, measure, EdgeType.COL_PATH))
    return graph


def signature_plan(signature_id):
    signature = OPERATION_SIGNATURES[signature_id]
    plan = {
        "plan_id": f"P-{signature_id}",
        "signature_id": signature_id,
        "operation_family": signature.operation_family,
        "semantic_result_role": signature.semantic_result_role,
        "projection_operator": signature.projection_operator,
        "answer_domain": signature.answer_domain,
        "role_bindings": {},
        "unresolved_semantics": [],
    }
    if signature_id.startswith("LOOKUP_VALUE_"):
        measure = {
            "LOOKUP_VALUE_SCALAR": "measure_numeric",
            "LOOKUP_VALUE_ENTITY": "measure_entity",
            "LOOKUP_VALUE_BOOLEAN": "measure_boolean",
        }[signature_id]
        plan["role_bindings"] = {"TARGET_ENTITY": ["entity_a"], "TARGET_MEASURE": [measure]}
    elif signature.operation_family in {"DIFF", "RATIO", "PAIR_COMPARE"}:
        plan["role_bindings"] = {
            "LEFT_OPERAND": ["entity_a", "measure_numeric"],
            "RIGHT_OPERAND": ["entity_b", "measure_numeric"],
        }
        if signature.operation_family == "PAIR_COMPARE":
            plan["comparison_polarity"] = "greater"
    else:
        members = [["entity_a"], ["entity_b"]]
        if signature_id == "ARGMAX_ENTITY_SET":
            members.append(["entity_c"])
        elif signature_id == "ARGMIN_ENTITY_SET":
            members.append(["entity_d"])
        plan["role_bindings"] = {
            "AGGREGATION_SCOPE": members,
            "TARGET_MEASURE": ["measure_numeric"],
        }
    return plan


class Round1OperationContractTests(unittest.TestCase):
    def test_every_declared_signature_has_contract_resolution_execution_projection_provenance_and_replay(self):
        graph = operation_graph()
        reference_ids = [node_id for node_id, node in graph.nodes.items() if node.node_type == NodeType.HEADER]
        for signature_id in sorted(OPERATION_SIGNATURES):
            with self.subTest(signature_id=signature_id):
                plan = signature_plan(signature_id)
                validation = validate_operation_plan(plan, reference_ids)
                self.assertTrue(validation.ok, validation.errors)
                closure = build_plan_closure({"planner_version": "round1_fixture", "plans": [plan]}, graph)
                self.assertTrue(closure.resource_complete)
                self.assertEqual(len(closure.assignments), 1, closure.to_dict())
                self.assertEqual(closure.assignments[0].outcome, ClosureOutcome.UNIQUE_EXECUTABLE, closure.to_dict())
                self.assertEqual(len(closure.executable_derivations), 1)
                derivation = closure.executable_derivations[0]
                self.assertEqual(derivation.typed_signature, signature_id)
                self.assertTrue(derivation.provenance_complete)
                self.assertTrue(derivation.operand_node_ids)
                self.assertTrue(derivation.required_edge_triples)
                replay = replay_derivation_under_intervention(
                    intervention_id="I-BENIGN",
                    derivation=derivation,
                    intervention=SimpleNamespace(
                        intervention_type="benign_control",
                        intervened_graph=copy.deepcopy(graph),
                    ),
                )
                self.assertTrue(replay.available)
                self.assertTrue(replay.operation_executed)
                self.assertTrue(replay.projection_executed)
                self.assertFalse(replay.changed)
                self.assertEqual(replay.derivation_id, derivation.derivation_id)

    def test_sample_fixed_basis_labels_self_substitution_as_benign_control(self):
        graph = operation_graph()
        plans = [signature_plan("LOOKUP_VALUE_SCALAR")]
        closure = build_plan_closure({"planner_version": "round1_fixture", "plans": plans}, graph)
        derivation = closure.executable_derivations[0]
        basis = build_sample_fixed_role_intervention_basis([derivation], graph)
        observations = evaluate_derivation_on_basis(derivation, graph, basis)
        self.assertTrue(observations)
        self.assertTrue(all(observation.benign_control for observation in observations))
        self.assertTrue(all(observation.response_symbol == "INVARIANT" for observation in observations))

    def test_resolver_preserves_unresolved_ambiguous_unique_and_resource_incomplete(self):
        graph = operation_graph()
        unique = resolve_atomic_operand(graph, ["entity_a", "measure_numeric"])
        unresolved = resolve_atomic_operand(graph, ["missing"])
        resource = resolve_atomic_operand(graph, ["entity_a", "measure_numeric"], max_candidates=0)
        graph.add_node(GraphNode("numeric_a_duplicate", NodeType.CELL, row=4, col=1, text="4", numeric_value=4.0))
        graph.add_edge(GraphEdge("numeric_a_duplicate", "entity_a", EdgeType.ROW_PATH))
        graph.add_edge(GraphEdge("numeric_a_duplicate", "measure_numeric", EdgeType.COL_PATH))
        ambiguous = resolve_atomic_operand(graph, ["entity_a", "measure_numeric"])
        self.assertEqual(unique.state, ResolutionState.UNIQUE)
        self.assertEqual(unresolved.state, ResolutionState.UNRESOLVED)
        self.assertEqual(resource.state, ResolutionState.RESOURCE_INCOMPLETE)
        self.assertEqual(ambiguous.state, ResolutionState.AMBIGUOUS)

    def test_forbidden_role_and_unknown_reference_are_rejected(self):
        plan = signature_plan("LOOKUP_VALUE_SCALAR")
        plan["role_bindings"]["LEFT_OPERAND"] = ["entity_a"]
        bad_role = validate_operation_plan(plan, ["entity_a", "measure_numeric"])
        self.assertIn("forbidden_role:LEFT_OPERAND", bad_role.errors)
        plan = signature_plan("LOOKUP_VALUE_SCALAR")
        plan["role_bindings"]["TARGET_ENTITY"] = ["missing"]
        bad_reference = validate_operation_plan(plan, ["entity_a", "measure_numeric"])
        self.assertIn("unknown_schema_id:TARGET_ENTITY:missing", bad_reference.errors)


if __name__ == "__main__":
    unittest.main()
