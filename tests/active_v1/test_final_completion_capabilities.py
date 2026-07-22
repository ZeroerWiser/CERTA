import copy
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import jsonschema

from graph_builder import EdgeType, GraphEdge, GraphNode, HCEG, NodeType

from certa.active_v1.artifact_authority import (
    ArtifactContext,
    reconcile_registry_entry,
    serialize_plan_closure,
)
from certa.active_v1.decision_adapter import (
    assess_decision_eligibility,
    materialize_selected_final,
    reconcile_cera_decision,
)
from certa.active_v1.planner_bridge_v3 import (
    build_v3_arm_view,
    close_compiled_payload,
    compile_active_planner_payload,
)
from certa.active_v1.role_contract_v3 import derive_role_v3_record
from certa.derivations.contrast import build_compact_behavioral_contrast_v3
from certa.derivations.iade import (
    build_basis_relative_behavior_classes,
    build_sample_fixed_role_intervention_basis,
)
from certa.grounding.support_partition import partition_support
from certa.planner.typed_planner import build_typed_planner_response_schema
from certa.repair.evidence_packet import CERAOutput
from certa.repair.safety_validator import validate_cera_output_v3
from certa.reproducibility.canonical_json import canonical_json, canonical_json_hash
from tools.certa_active_v1 import _fixture_graph, _fixture_table, _payload


FIXTURES = Path(__file__).with_name("fixtures") / "final_completion"
SCHEMAS = Path(__file__).parents[2] / "schemas" / "active_v1"
COMPLETION_PACK = Path(
    "/home/hsh/ME/Table/EMNLP2026/certa_goal_packs/"
    "CERTA_ACTIVE_V1_FROZEN_ROLE_V3_FINAL_METHOD_COMPLETION_PACK"
)
ROLE_V3_PACK = Path(
    "/home/hsh/ME/Table/EMNLP2026/certa_goal_packs/"
    "CERTA_ACTIVE_V1_ROLE_V3_FINAL_METHOD_PACK"
)
BASE_PACK_SCHEMAS = Path(
    "/home/hsh/ME/Table/EMNLP2026/certa_goal_packs/"
    "CERTA_ACTIVE_V1_FINAL_METHOD_GOAL_REVISED_PACK/schemas"
)
ROLE_IDS = (
    "LOOKUP_VALUE_SCALAR",
    "LOOKUP_VALUE_ENTITY",
    "COUNT_SCALAR",
    "SUM_SCALAR",
    "AVERAGE_SCALAR",
    "DIFF_SCALAR",
    "RATIO_SCALAR",
    "ARGMAX_ENTITY",
    "ARGMAX_ENTITY_SET",
    "ARGMIN_ENTITY",
    "ARGMIN_ENTITY_SET",
    "PAIR_COMPARE_BOOLEAN",
)
CONSTRUCTOR_FIELDS = (
    "role_registry_present",
    "v3_bridge_fixture_pass",
    "planner_schema_fixture_pass",
    "active_compiler_fixture_pass",
    "grounding_fixture_pass",
    "closure_fixture_pass",
    "deterministic_executor_fixture_pass",
    "projection_fixture_pass",
    "provenance_fixture_pass",
    "registry_serialization_fixture_pass",
    "negative_fixture_pass",
)
def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _inactive_row(role_id):
    row = {field: False for field in CONSTRUCTOR_FIELDS}
    row.update({
        "role_id": role_id,
        "role_registry_present": True,
        "constructor_active": False,
        "failure_reasons": [
            field for field in CONSTRUCTOR_FIELDS if field != "role_registry_present"
        ],
        "fixture_artifact_sha256": canonical_json_hash({"role_id": role_id}),
    })
    return row


def _registry_only_matrix():
    return {
        "schema_version": "certa_active_v1_constructor_capability_v1",
        "role_registry_sha256": (
            "114065916322ce70d1ca8122e6a40f7866ee4dfaa7d9c93eba58ab741d0bf3be"
        ),
        "rows": [_inactive_row(role_id) for role_id in ROLE_IDS],
    }


def _paired_graph():
    graph = _fixture_graph()
    graph.add_node(GraphNode(
        "entity_value_b", NodeType.CELL, row=2, col=2, text="Beta",
    ))
    graph.add_edge(GraphEdge(
        "entity_value_b", "entity_b", EdgeType.ROW_PATH,
    ))
    graph.add_edge(GraphEdge(
        "entity_value_b", "measure_entity", EdgeType.COL_PATH,
    ))
    return graph


def _paired_payload(role_id):
    payload = _payload(role_id)
    alternative = copy.deepcopy(payload["plans"][0])
    alternative["plan_id"] = "P1"
    bindings = alternative["role_bindings"]
    if role_id.startswith("LOOKUP_VALUE_"):
        bindings["TARGET_ENTITY"] = ["entity_b"]
    elif role_id in {"COUNT_SCALAR", "SUM_SCALAR", "AVERAGE_SCALAR"}:
        bindings["AGGREGATION_SCOPE"] = [["entity_a"]]
    elif role_id in {"DIFF_SCALAR", "RATIO_SCALAR", "PAIR_COMPARE_BOOLEAN"}:
        bindings["LEFT_OPERAND"], bindings["RIGHT_OPERAND"] = (
            bindings["RIGHT_OPERAND"], bindings["LEFT_OPERAND"],
        )
    elif role_id == "ARGMAX_ENTITY":
        bindings["AGGREGATION_SCOPE"] = [["entity_b"]]
    elif role_id == "ARGMIN_ENTITY":
        bindings["AGGREGATION_SCOPE"] = [["entity_a"]]
    elif role_id == "ARGMAX_ENTITY_SET":
        bindings["AGGREGATION_SCOPE"] = [["entity_b"], ["entity_d"]]
    elif role_id == "ARGMIN_ENTITY_SET":
        bindings["AGGREGATION_SCOPE"] = [["entity_a"], ["entity_c"]]
    payload["plans"].append(alternative)
    return payload


def _cera_fixture_output(contrast, use_repaired):
    if not use_repaired:
        return {
            "decision": "INSUFFICIENT_CERTIFICATE",
            "chosen_hypothesis_id": "",
            "final_answer": "",
            "original_assessment": {},
            "alternative_assessment": {},
            "separating_intervention_refs": [],
        }
    original = contrast["original_hypothesis"]
    alternative = contrast["alternative_hypothesis"]
    intervention = contrast["separating_interventions"][0]["intervention_ref"]
    return {
        "decision": "USE_REPAIRED",
        "chosen_hypothesis_id": alternative["hypothesis_id"],
        "final_answer": alternative["executed_answer"],
        "original_assessment": {
            "hypothesis_id": original["hypothesis_id"],
            "derivation_ref": original["derivation_ref"],
            "evidence_refs": original["evidence_refs"],
            "intervention_refs": [intervention],
        },
        "alternative_assessment": {
            "hypothesis_id": alternative["hypothesis_id"],
            "derivation_ref": alternative["derivation_ref"],
            "evidence_refs": alternative["evidence_refs"],
            "intervention_refs": [intervention],
        },
        "separating_intervention_refs": [intervention],
    }


class FrozenCapabilityEquationRedTests(unittest.TestCase):
    def test_registry_presence_alone_cannot_activate_constructor(self):
        with self.assertRaisesRegex(
            ValueError,
            "capability_matrix_has_no_active_signature",
        ):
            build_v3_arm_view(
                "C0_SCHEMA_ONLY",
                "Fixture question",
                HCEG(),
                {"texts": []},
                None,
                None,
                _registry_only_matrix(),
            )

class FinalCompletionFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_document = _load(FIXTURES / "constructor_cases.json")
        cls.registry = _load(ROLE_V3_PACK / "ROLE_V3_CANONICAL_REGISTRY.json")
        cls.role_schema = _load(ROLE_V3_PACK / "ROLE_V3_OUTPUT_SCHEMA.json")
        cls.constructor_schema = _load(
            SCHEMAS / "CONSTRUCTOR_CAPABILITY_MATRIX.schema.json"
        )
        cls.decision_schema = _load(
            SCHEMAS / "DECISION_CAPABILITY_MATRIX.schema.json"
        )
        cls.artifact_schemas = {
            name: _load(BASE_PACK_SCHEMAS / name)
            for name in (
                "RAW_GROUNDING_RECORD_SCHEMA.json",
                "RAW_DERIVATION_RECORD_SCHEMA.json",
                "REGISTRY_ENTRY_SCHEMA.json",
            )
        }

    def _all_active_matrix(self):
        rows = []
        for case in self.fixture_document["cases"]:
            values = {field: True for field in CONSTRUCTOR_FIELDS}
            rows.append({
                "role_id": case["role_id"],
                **values,
                "constructor_active": all(values.values()),
                "failure_reasons": [],
                "fixture_artifact_sha256": canonical_json_hash(case),
            })
        return {
            "schema_version": "certa_active_v1_constructor_capability_v1",
            "role_registry_sha256": self.fixture_document["role_registry_sha256"],
            "rows": rows,
        }

    def _observe_decision_row(self, role_id, constructor_matrix):
        graph = _paired_graph()
        role = derive_role_v3_record({
            "schema_version": "certa_active_role_contract_v3",
            "role_id": role_id,
        }, self.role_schema, self.registry)
        view = build_v3_arm_view(
            "C1_ROLE_ONLY", "Decision capability fixture only.",
            graph, _fixture_table(), role, None, constructor_matrix,
            output_schema=self.role_schema,
            canonical_registry=self.registry,
        ).view
        compilation = compile_active_planner_payload(
            _paired_payload(role_id), view, constructor_matrix,
        )
        closure = close_compiled_payload(compilation, graph, constructor_matrix)
        derivations = tuple(closure.executable_derivations)
        initial_answer = derivations[0].projected_answer
        partition = partition_support(
            closure, initial_proposal_answer=initial_answer,
        )
        basis = build_sample_fixed_role_intervention_basis(derivations, graph)
        behavior_classes = build_basis_relative_behavior_classes(
            derivations, graph, basis,
        )
        contrast = build_compact_behavioral_contrast_v3(
            derivations=derivations,
            behavior_classes=behavior_classes,
            basis=basis,
            original_answer=initial_answer,
            query_semantics=view["query_semantics"],
        ).to_dict()
        states = contrast["states"]
        partition_pass = bool(
            len(partition.original_support) == 1
            and len(partition.alternative_support) == 1
            and partition.disjoint
            and partition.exhaustive
        )
        paired_pass = bool(
            states["contrast_constructible"]
            and states["contrast_compact"]
            and states["repair_eligible"]
            and len(contrast["alternative_hypotheses"]) == 1
            and not contrast["unknowns"]
            and any(
                item["evaluable_on_both_sides"] and item["separating"]
                for item in contrast["separating_interventions"]
            )
        )
        executed_ids = {item.derivation_id for item in derivations}
        registry_derivation_ids = {
            item["derivation_id"]
            for item in contrast["registry"]["derivation_records"]
        }
        registry_pass = bool(
            states["contrast_registry_complete"]
            and registry_derivation_ids == executed_ids
            and all(contrast["registry"][name] for name in (
                "hypothesis_records", "derivation_records",
                "evidence_records", "intervention_records",
            ))
        )
        candidate_eligibility = assess_decision_eligibility(
            role_id=role_id,
            decision_active_role_ids=(role_id,),
            support_partition=partition,
            compact_contrast=contrast,
            executed_derivations=derivations,
        )
        raw = _cera_fixture_output(contrast, candidate_eligibility.eligible)
        parsed = CERAOutput.from_dict(raw)
        cera_schema_pass = (
            parsed.decision == raw["decision"]
            and parsed.raw == raw
            and isinstance(parsed.separating_intervention_refs, list)
        )
        validator = validate_cera_output_v3(
            raw, {"compact_behavioral_contrast_v3": contrast},
        )
        validator_pass = validator.accepted
        bundle = serialize_plan_closure(
            closure,
            context=ArtifactContext(
                sample_id=f"fixture-{role_id}",
                table_id="fixture-table",
                arm="C2_ROLE_RETRIEVAL",
                role_id=role_id,
                fixture_only=True,
            ),
            initial_answer=initial_answer,
        )

        operational_eligibility = assess_decision_eligibility(
            role_id=role_id,
            decision_active_role_ids=(role_id,)
            if candidate_eligibility.eligible else (),
            support_partition=partition,
            compact_contrast=contrast,
            executed_derivations=derivations,
        )
        resolution = reconcile_cera_decision(
            eligibility=operational_eligibility,
            raw_output=raw if candidate_eligibility.eligible else None,
            validator=validator if candidate_eligibility.eligible else None,
            compact_contrast=contrast,
            executed_derivations=derivations,
            raw_derivation_records=bundle.raw_derivations,
            registry_entries=bundle.registry_entries,
            b0_answer=initial_answer,
            sample_id=f"fixture-{role_id}",
            decision_id=f"DEC-{role_id}",
            validator_record_id=f"VAL-{role_id}",
            created_at="2026-07-23T00:00:00+00:00",
            fixture_only=True,
        )
        materialized = materialize_selected_final(
            resolution,
            b0_answer=initial_answer,
            materialized_at="2026-07-23T00:01:00+00:00",
        )
        expected_source = "REGISTRY" if candidate_eligibility.eligible else "B0"
        materializer_pass = bool(
            materialized.record["selected_source"] == expected_source
            and (
                candidate_eligibility.eligible
                or materialized.answer == initial_answer
            )
        )
        booleans = {
            "original_alternative_partition_fixture_pass": partition_pass,
            "paired_contrast_fixture_pass": paired_pass,
            "registry_reference_fixture_pass": registry_pass,
            "cera_schema_fixture_pass": cera_schema_pass,
            "validator_fixture_pass": validator_pass,
            "materializer_fixture_pass": materializer_pass,
        }
        decision_active = all(booleans.values())
        failures = [name for name, passed in booleans.items() if not passed]
        failures.extend(
            f"eligibility:{reason}"
            for reason in candidate_eligibility.failure_reasons
        )
        failures.extend(
            f"contrast_unknown:{reason}" for reason in contrast["unknowns"]
        )

        negative_pass = not candidate_eligibility.eligible
        if candidate_eligibility.eligible:
            corrupted = [dict(item) for item in bundle.registry_entries]
            alternative_id = contrast["alternative_hypothesis"]["derivation_id"]
            target = next(
                item for item in corrupted
                if item["derivation_id"] == alternative_id
            )
            target["answer_hash"] = "0" * 64
            rejected = reconcile_cera_decision(
                eligibility=candidate_eligibility,
                raw_output=raw,
                validator=validator,
                compact_contrast=contrast,
                executed_derivations=derivations,
                raw_derivation_records=bundle.raw_derivations,
                registry_entries=corrupted,
                b0_answer=initial_answer,
                sample_id=f"fixture-{role_id}",
                decision_id=f"NEG-{role_id}",
                validator_record_id=f"NEGVAL-{role_id}",
                created_at="2026-07-23T00:00:00+00:00",
                fixture_only=True,
            )
            negative_pass = (
                rejected.decision_record["action"] == "KEEP_B0"
                and rejected.selected_answer == initial_answer
            )
        evidence = {
            "role_id": role_id,
            "answers": [item.projected_answer for item in derivations],
            "basis": [item.intervention_id for item in basis],
            "contrast_states": states,
            "contrast_unknowns": contrast["unknowns"],
            "booleans": booleans,
            "eligibility_failures": list(candidate_eligibility.failure_reasons),
            "selected_source": materialized.record["selected_source"],
            "expected_cera_calls": 1 if decision_active else 0,
            "negative_pass": negative_pass,
        }
        row = {
            "role_id": role_id,
            "constructor_active": True,
            **booleans,
            "decision_active": decision_active,
            "inactive_fallback": "B0_KEEP",
            "failure_reasons": sorted(set(failures)),
            "fixture_artifact_sha256": canonical_json_hash(evidence),
        }
        return row, {
            "candidate_eligibility": candidate_eligibility,
            "operational_eligibility": operational_eligibility,
            "materialized": materialized,
            "negative_pass": negative_pass,
            "expected_cera_calls": evidence["expected_cera_calls"],
        }

    def test_repository_capability_schemas_are_semantically_identical_to_pack(self):
        for name in (
            "CAPABILITY_GATE.schema.json",
            "CONSTRUCTOR_CAPABILITY_MATRIX.schema.json",
            "DECISION_CAPABILITY_MATRIX.schema.json",
        ):
            with self.subTest(name=name):
                self.assertEqual(
                    _load(SCHEMAS / name),
                    _load(COMPLETION_PACK / "schemas" / name),
                )

    def test_constructor_fixture_document_is_strict_and_complete(self):
        jsonschema.Draft202012Validator(
            _load(SCHEMAS / "FINAL_COMPLETION_FIXTURE_CASES.schema.json")
        ).validate(self.fixture_document)
        role_ids = [case["role_id"] for case in self.fixture_document["cases"]]
        self.assertEqual(tuple(role_ids), ROLE_IDS)
        self.assertEqual(len(role_ids), len(set(role_ids)))

    def test_all_twelve_constructor_fixtures_traverse_real_authorities(self):
        matrix = self._all_active_matrix()
        graph = _fixture_graph()
        observed_rows = []
        for case in self.fixture_document["cases"]:
            role_id = case["role_id"]
            with self.subTest(role_id=role_id):
                role = derive_role_v3_record({
                    "schema_version": "certa_active_role_contract_v3",
                    "role_id": role_id,
                }, self.role_schema, self.registry)
                built = build_v3_arm_view(
                    "C1_ROLE_ONLY",
                    "Capability fixture only.",
                    graph,
                    _fixture_table(),
                    role,
                    None,
                    matrix,
                    output_schema=self.role_schema,
                    canonical_registry=self.registry,
                )
                v3_bridge_pass = (
                    built.view["operation_ontology"]["signature_ids"] == [role_id]
                    and built.role_record_sha256 == canonical_json_hash(role)
                    and "table_values" not in built.view
                )

                payload = _payload(role_id)
                planner_schema = build_typed_planner_response_schema(
                    built.view,
                    require_signature_id=True,
                )
                jsonschema.Draft202012Validator(planner_schema).validate(payload)
                planner_schema_pass = True
                compilation = compile_active_planner_payload(payload, built.view, matrix)
                compiler_pass = compilation.ok
                closure = close_compiled_payload(compilation, graph, matrix)
                closure_again = close_compiled_payload(
                    compilation,
                    copy.deepcopy(graph),
                    matrix,
                )
                assignments = tuple(closure.assignments)
                derivations = tuple(closure.executable_derivations)
                grounding_pass = bool(
                    len(assignments) == 1
                    and assignments[0].signature_id == role_id
                    and assignments[0].role_bindings
                    and assignments[0].canonical_program_id
                )
                closure_pass = bool(
                    closure.resource_complete
                    and len(assignments) == 1
                    and len(derivations) == 1
                )
                deterministic_pass = (
                    canonical_json(closure.to_dict())
                    == canonical_json(closure_again.to_dict())
                )
                derivation = derivations[0]
                projection_pass = (
                    derivation.typed_signature == role_id
                    and derivation.projected_answer
                    == case["expected_projected_answer"]
                    and derivation.projection_operator == role["projection"]
                    and derivation.output_domain == role["answer_role"]
                )
                provenance_pass = bool(
                    derivation.provenance_complete and derivation.evidence_ids
                )

                bundle = serialize_plan_closure(
                    closure,
                    context=ArtifactContext(
                        sample_id=f"fixture-{role_id}",
                        table_id="fixture-table",
                        arm="C1_ROLE_ONLY",
                        role_id=role_id,
                        fixture_only=True,
                    ),
                    initial_answer=case["expected_projected_answer"],
                )
                jsonschema.validate(
                    bundle.raw_groundings[0],
                    self.artifact_schemas["RAW_GROUNDING_RECORD_SCHEMA.json"],
                )
                jsonschema.validate(
                    bundle.raw_derivations[0],
                    self.artifact_schemas["RAW_DERIVATION_RECORD_SCHEMA.json"],
                )
                jsonschema.validate(
                    bundle.registry_entries[0],
                    self.artifact_schemas["REGISTRY_ENTRY_SCHEMA.json"],
                )
                reconcile_registry_entry(
                    bundle.registry_entries[0],
                    bundle.raw_derivations[0],
                )
                registry_serialization_pass = True

                negative = case["negative_case"]
                malformed_role = dict(
                    role,
                    schema_version=negative["malformed_role_schema_version"],
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "role_v3_canonical_record_version_mismatch",
                ):
                    build_v3_arm_view(
                        "C1_ROLE_ONLY", "Fixture", graph, _fixture_table(),
                        malformed_role, None, matrix,
                        output_schema=self.role_schema,
                        canonical_registry=self.registry,
                    )
                malformed_authority = dict(
                    role,
                    requires_time_scope=negative["malformed_authority"],
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "role_v3_canonical_record_mismatch",
                ):
                    build_v3_arm_view(
                        "C1_ROLE_ONLY", "Fixture", graph, _fixture_table(),
                        malformed_authority, None, matrix,
                        output_schema=self.role_schema,
                        canonical_registry=self.registry,
                    )
                malformed_payload = copy.deepcopy(payload)
                malformed_payload["plans"][0]["projection_operator"] = (
                    negative["malformed_projection"]
                )
                self.assertFalse(
                    compile_active_planner_payload(
                        malformed_payload,
                        built.view,
                        matrix,
                    ).ok
                )
                malformed_registry = dict(bundle.registry_entries[0])
                malformed_registry["answer_hash"] = "0" * 64
                with self.assertRaisesRegex(
                    ValueError,
                    "registry_derivation_mismatch:answer_hash",
                ):
                    reconcile_registry_entry(
                        malformed_registry,
                        bundle.raw_derivations[0],
                    )
                negative_pass = True

                booleans = {
                    "role_registry_present": role_id in {
                        row["role_id"] for row in self.registry["roles"]
                    },
                    "v3_bridge_fixture_pass": v3_bridge_pass,
                    "planner_schema_fixture_pass": planner_schema_pass,
                    "active_compiler_fixture_pass": compiler_pass,
                    "grounding_fixture_pass": grounding_pass,
                    "closure_fixture_pass": closure_pass,
                    "deterministic_executor_fixture_pass": deterministic_pass,
                    "projection_fixture_pass": projection_pass,
                    "provenance_fixture_pass": provenance_pass,
                    "registry_serialization_fixture_pass": (
                        registry_serialization_pass
                    ),
                    "negative_fixture_pass": negative_pass,
                }
                observed_rows.append({
                    "role_id": role_id,
                    **booleans,
                    "constructor_active": all(booleans.values()),
                    "failure_reasons": [
                        name for name, passed in booleans.items() if not passed
                    ],
                    "fixture_artifact_sha256": canonical_json_hash(case),
                })

        observed = {
            "schema_version": "certa_active_v1_constructor_capability_v1",
            "role_registry_sha256": self.fixture_document["role_registry_sha256"],
            "rows": observed_rows,
        }
        jsonschema.Draft202012Validator(self.constructor_schema).validate(observed)
        self.assertTrue(all(row["constructor_active"] for row in observed_rows))
        self.assertEqual(
            observed,
            _load(FIXTURES / "CONSTRUCTOR_CAPABILITY_MATRIX.fixture.json"),
        )

    def test_actual_decision_matrix_runs_every_frozen_authority(self):
        constructor_path = FIXTURES / "CONSTRUCTOR_CAPABILITY_MATRIX.fixture.json"
        constructor = _load(constructor_path)
        expected = _load(FIXTURES / "DECISION_CAPABILITY_MATRIX.fixture.json")
        rows, details = [], {}
        for role_id in ROLE_IDS:
            row, detail = self._observe_decision_row(role_id, constructor)
            rows.append(row)
            details[role_id] = detail
        observed = {
            "schema_version": "certa_active_v1_decision_capability_v1",
            "constructor_matrix_sha256": hashlib.sha256(
                constructor_path.read_bytes()
            ).hexdigest(),
            "rows": rows,
        }
        jsonschema.Draft202012Validator(self.decision_schema).validate(observed)
        self.assertEqual(observed, expected)
        self.assertEqual(
            {row["role_id"] for row in rows if row["decision_active"]},
            {"LOOKUP_VALUE_SCALAR", "LOOKUP_VALUE_ENTITY"},
        )
        for row in rows:
            detail = details[row["role_id"]]
            self.assertTrue(detail["negative_pass"])
            if not row["decision_active"]:
                self.assertEqual(detail["expected_cera_calls"], 0)
                self.assertFalse(detail["operational_eligibility"].cera_call_allowed)
                self.assertEqual(
                    detail["materialized"].record["selected_source"], "B0",
                )

    def test_frozen_pack_capability_gate_passes_actual_matrices(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "CAPABILITY_GATE.json"
            command = [
                "/home/hsh/anaconda3/envs/table-cu128/bin/python",
                str(COMPLETION_PACK / "tools/compute_capability_gate.py"),
                "--constructor",
                str(FIXTURES / "CONSTRUCTOR_CAPABILITY_MATRIX.fixture.json"),
                "--decision",
                str(FIXTURES / "DECISION_CAPABILITY_MATRIX.fixture.json"),
                "--output",
                str(output),
            ]
            result = subprocess.run(command, check=False, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "CAPABILITY_GATE_PASS\n")
            gate = _load(output)
            jsonschema.validate(
                gate, _load(SCHEMAS / "CAPABILITY_GATE.schema.json"),
            )
            self.assertEqual(gate["constructor_active_count"], 12)
            self.assertEqual(gate["decision_active_count"], 2)


if __name__ == "__main__":
    unittest.main()
