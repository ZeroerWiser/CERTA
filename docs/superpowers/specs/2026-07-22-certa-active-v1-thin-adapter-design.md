# CERTA Active V1 Thin-Adapter Design Freeze

Status: **DESIGN APPROVED; IMPLEMENTATION AND ENDPOINT EXECUTION NOT AUTHORIZED**

Decision owner: **CERTA Research Director**

Scheme: **A — deterministic canonicalization of non-applicable legacy fields**

Namespace: `certa/active_v1/`

Parent commit: `819aa30f499f337b77b0b731b4c833a3f50726fa`

Parent tree: `9e044df0c3dabeaf988732a314910e5a980d399c`

Branch: `research/certa-active-v1`

## 1. Decision and non-goals

This design adds one versioned orchestration adapter around existing CERTA primitives. It does not modify EGRA in place and does not copy the historical pipeline. The scientific object remains:

```text
frozen B0
→ sealed-validated question role
→ provenance-aware structural retrieval
→ typed deterministic original/alternative contrast
→ registry-constrained CERA_PLUS_VALIDATOR decision
→ deterministic selected-final materialization
```

The implementation phase, when separately authorized, may add new files only in the change surface in Section 4. It may import existing public functions but may not copy their implementations.

This design does **not** authorize:

- production or test implementation;
- an endpoint request;
- opening the sealed role-label resource;
- B0 generation;
- visible, sealed, Role, Constructor, Decision, Opportunity, Holdout, or Pack-Gate execution;
- modification of any historical script, profile, test, output, terminal record, negative artifact, commit, or branch;
- a new executor, graph type, causal estimator, information score, model, reranker, regex bank, dataset lexicon, answer dictionary, debate step, or post-hoc threshold;
- changing Pack schemas, calculators, thresholds, terminal states, or access ordering.

Rejected designs are recorded explicitly:

- **in-place EGRA modification** is rejected because it would alter historical role/retrieval behavior and invalidate negative-artifact lineage;
- **full pipeline duplication** is rejected because copied closure, executor, validator, or CERA logic would create two competing authorities;
- **nullable operation-specific objects such as `rank_spec`** are rejected for Active V1 because ranking has no KTH signature and the object would add a new wire/semantic state space without executor benefit;
- **two-stage intent then operation-specific schema** is rejected because it adds one model call per role decision, a cross-stage consistency join, and new failure/retry modes while changing the preregistered single-call role interface.

The separate sealed-access manifest template is
`docs/superpowers/specs/CERTA_ACTIVE_V1_SEALED_ACCESS_MANIFEST_TEMPLATE.schema.json`.
The sealed resource location is recorded only there, never in `ROLE_INTERFACE_FREEZE.json` or this document.

## 2. Frozen inputs and source identity

The implementation must fail closed unless `HEAD`, the parent tree, and Pack identities match this table before the first source edit.

| Object | Required SHA-256 or Git identity |
|---|---|
| Git parent | `819aa30f499f337b77b0b731b4c833a3f50726fa` |
| Git parent tree | `9e044df0c3dabeaf988732a314910e5a980d399c` |
| Pack `PACK_MANIFEST.json` | `32866148bcfd4fda912210cbbc31f1c2bd17274e96e2aab31d5782a556247d89` |
| Pack `SHA256SUMS.txt` | `a76f781ae1786878516dcc36b1f1044ffc5e79465c3393882ee15e1de0d55b85` |
| Pack role-output schema | `e334c783b8df5710bc269667f2f6120c39af35ca9432ad0e1e80ad045457dbdc` |
| Pack interface-freeze schema | `79ee7881654c0ddca1b9be53ab55d7b20923958e1d453f2525c96d4feb3c3d9f` |
| Pack Constructor calculator | `8438152d6550bb2c79e2a434015e93f6e46560bebfa1104d065c56b69ae9fc86` |
| Pack Decision calculator | `caa30342fed1465e7c386b49844ed2b8668406895c4e59fa729b79f8b787adc9` |

The source functions to be reused are bound to the following design-time hashes. These are parent-source observations, not future freeze values; implementation must recompute and record them.

| Existing source | SHA-256 | Reused authority |
|---|---|---|
| `run_cscr_pipeline.py` | `409f833e478517dbb457d5dcac26f65841e78e666496bd7af378a13888ed3724` | generator and historical B0 path |
| `graph_builder.py` | `f327f13e0cf4cad58464ebced0d3d2e92b963b76bda6132333412541fb3c422e` | HCEG construction |
| `certa/operations/contracts.py` | `dc2e346420214ac9a9f44e6b9159561f75c94ea5a7d6f63e89a23b3cd0e69cfb` | signature ontology and plan validation |
| `certa/planner/schema_view.py` | `a663ac276e19239c3e0cb15edbf0f7f683af3d450d36cb1be23a474ccd06c773` | Planner view |
| `certa/planner/typed_planner.py` | `4da04fcfc85cb1f236870eb9ee886e9d6d177223d4ecf3191cf873445900fd92` | prompt, wire schema, validation |
| `certa/planner/compiler.py` | `95f25a76dfc15f50d5dcae921eae89fe57e71ba0ccc8cded960a6f49e1689259` | historical compiler comparison only |
| `certa/grounding/plan_closure.py` | `ce290776d98b4befbce582eabfeb4281053f5ba2c26c6bec0e1184de673c0239` | closure and deterministic execution |
| `certa/egra/evidence_cards.py` | `3f9fe7aff75267c10e9c50f6ae49de4fd952c81a74708a9bac9572bb2b6439de` | structural cards |
| `certa/egra/retrieval.py` | `02d30f80ac2e3c0827c4eaf819a2aa0f66b50e42cd1b93681041fc1a25995552` | frozen E5 index and retrieval |
| `certa/derivations/answer_equivalence.py` | `24c0fb93fc78102d53e8531c8b4d07fe0caa2add7123570769d0a61bd87eb0b9` | answer equivalence |
| `certa/derivations/iade.py` | implementation-time recompute | fixed-basis intervention behavior |
| `certa/derivations/contrast.py` | `96c5d9d234d90c974b346a557f74d4e6bc3163c07fcf423ae1bb71a0c0342652` | compact v3 contrast |
| `certa/repair/repair_prompt.py` | `4ed2013779f3ff6495df4c5c20576b681915e6eb213fc29d31cac77afc4d8e9d` | CERA v3 prompt |
| `certa/repair/safety_validator.py` | `277d8581e984b2b0f941c245e198f6d1cecb383612b5b0a4bc63c0d64d4e736d` | CERA v3 validator |
| `certa/repair/causal_epistemic_agent.py` | `77619174cad1695cc11db73e2cf436c1b3d356a70e7b94b4d5870e5e0d9a9787` | frozen reference call path only |

`certa/egra/retrieval.py` and `certa/repair/causal_epistemic_agent.py` are default-frozen. This design proposes **zero changes** to either. Therefore no exception red test is proposed. If implementation later discovers an inaccessible required interface, work stops: a proposed exception must identify the exact function, first add a failing red test proving that `certa/active_v1/` cannot use the existing interface, and obtain new Research Director authorization before either file is changed.

## 3. Exact call-path map

### 3.1 Orchestration boundary

The new CLI is an orchestrator, not a second pipeline. Its calls are imports into existing modules:

```text
tools/certa_active_v1.py
  ├─ role command ───────────────→ certa.active_v1.role_contract
  ├─ constructor command ────────→ certa.active_v1.planner_adapter
  │                                certa.active_v1.artifact_authority
  ├─ decision command ───────────→ certa.active_v1.decision_adapter
  └─ replay/freeze command ──────→ Pack validators/calculators as subprocesses
```

It does not import or invoke `run_cera`, `run_egra_constructor_shadow`, or any other orchestration routine in `certa/repair/causal_epistemic_agent.py`. This avoids hidden legacy gates and answer mutation while reusing the same lower-level method primitives.

### 3.2 B0 path

When later authorized, B0 is generated once per frozen split by the historical baseline path in `run_cscr_pipeline.py`; the Active adapter then treats the raw response and its equivalence hash as immutable inputs. Constructor arms never regenerate B0 and never call a B0 model.

```text
frozen runtime row
→ existing table loader/formatter
→ run_cscr_pipeline.OpenAIChatGenerator.generate
→ raw B0 response
→ active_answer_hash
→ immutable B0 record
```

The exact historical CLI arguments must be frozen in `METHOD_CONFIG_FREEZE.json` before B0. The implementation must first prove with a replay fixture that the selected profile produces the same B0 bytes and hash on repeated cache reads. B0 is never changed by constructor or decision code.

### 3.3 Role path

```text
question string only
→ certa.active_v1.role_contract.build_role_prompt
→ certa.active_v1.role_contract.build_role_wire_schema
→ run_cscr_pipeline.OpenAIChatGenerator.generate_json_schema
→ immutable raw request + raw response
→ JSON parse / wire-schema validation
→ full semantic-schema validation
→ local task-semantic validator
→ exact nine-field role record or invalid record
```

No B0, table values, table headers, operation annotation, correctness, historical error, fixture text, or label enters the prompt. Invalid output is not repaired and does not fall back to C0. The one role call returns one raw response. A transport retry is allowed only under the Pack retry rule and is logged as the same logical call.

The endpoint receives the flat transport-safe nine-field wire schema. The full semantic schema is applied locally and enumerates the exact supported tuples in Section 6. This deliberately preserves independent `wire_valid`, `semantic_schema_valid`, `local_validator_valid`, and post-label `task_semantic_valid` evidence.

### 3.4 Structural graph, card, and retrieval path

```text
runtime row + table
→ graph_builder.build_hceg
→ certa.planner.schema_view.build_canonical_structural_group_catalog
→ certa.egra.evidence_cards.build_structural_evidence_cards
→ certa.egra.retrieval.build_card_index
→ certa.active_v1.role_contract.to_egra_retrieval_contract
→ certa.egra.retrieval.retrieve_structural_cards
→ selected cards + exact reference_node_ids
```

`to_egra_retrieval_contract` is Scheme A. It is a deterministic, one-way compatibility projection used only because the frozen retrieval query serializer expects legacy field names. It cannot change the Active role record and has no selection authority. Its canonical JSON and SHA-256 are logged.

### 3.5 Planner arms

All arms call the same existing prompt/schema/validator/closure stack. Only the preregistered input information differs.

```text
graph_builder.build_hceg
→ certa.planner.schema_view.build_proposal_blind_planner_view
→ certa.planner.typed_planner.build_typed_derivation_planner_prompt
→ certa.planner.typed_planner.build_typed_planner_response_schema
→ OpenAIChatGenerator.generate_json_schema
→ certa.planner.typed_planner.validate_typed_planner_output
→ certa.active_v1.planner_adapter.compile_active_planner_payload
→ certa.grounding.plan_closure.build_plan_closure
```

| Arm | Signature allowlist | Query semantics | Schema-node domain | Retrieval |
|---|---|---|---|---|
| `C0_SCHEMA_ONLY` | every capability-`ACTIVE` signature | absent (`legacy_query_semantics_mode="audit_only"`) | complete proposal-blind schema | none |
| `C1_ROLE_ONLY` | the one frozen role signature | exact role-derived tuple | complete proposal-blind schema | none |
| `C2_ROLE_RETRIEVAL` | the same one frozen role signature | byte-identical to C1 | only frozen E5 reference IDs plus compact cards | frozen E5 |

For all three arms, `include_table_values=False`, `require_signature_id=True`, temperature/model/sampling, Planner contract, closure, executor, answer normalization, resource caps, B0, sample IDs, order, and artifact schemas are identical. C1 and C2 must carry the same `role_record_sha256`. Unsupported or invalid roles produce deterministic empty C1/C2 constructor records and no fallback.

`compile_active_planner_payload` is narrowly defined: it accepts only an already valid normalized payload, rejects signatures not `ACTIVE` in the frozen capability matrix, verifies canonical JSON round-trip identity, and emits the canonical payload consumed by `build_plan_closure`. It does not execute, infer, repair, add, rank, or select plans.

The historical `certa.planner.compiler.compile_typed_plans_to_derivations` is not on this call path. It is a lookup-only legacy compiler: it rejects every non-`LOOKUP` operation as `unsupported_operation_family`. This limitation is not hidden or modified. The Active compiler fixture is a new adapter-boundary fixture because the existing live multi-signature path already uses `validate_typed_planner_output` followed directly by `build_plan_closure`.

### 3.6 Raw constructor and registry path

```text
PlanClosure.assignments + PlanClosure.executable_derivations
→ certa.active_v1.artifact_authority.emit_raw_groundings
→ certa.active_v1.artifact_authority.emit_raw_derivations
→ exact reconciliation
→ certa.active_v1.artifact_authority.build_registry
→ Pack compute_certa_active_constructor_gate.py
```

The Pack calculator consumes the emitted JSONL files directly. No runner summary boolean is authoritative and the adapter may not copy or reimplement Gate C.

### 3.7 Decision and selected-final path

Only machine Gate O PASS makes a row eligible for decision calls.

```text
frozen C2 executable derivations
→ certa.derivations.iade.build_sample_fixed_role_intervention_basis
→ certa.derivations.iade.build_basis_relative_behavior_classes
→ certa.derivations.contrast.build_compact_behavioral_contrast_v3
→ certa.repair.evidence_packet.CausalEvidencePacket
→ certa.repair.repair_prompt.build_cera_prompt(template_version="cera_repair_v3")
→ OpenAIChatGenerator.generate_json_schema
→ certa.repair.safety_validator.validate_cera_output_v3
→ Active registry-reference reconciliation
→ deterministic materializer
→ Pack compute_certa_active_decision_gate.py
```

CERA sees only the role contract, B0 reference, compact H/D/E/I registry, executed answer values already present in that registry, and provenance summaries. It sees neither the table nor gold. Existing v3 intervention logic currently constructs its fixed basis only for LOOKUP derivations; a non-LOOKUP row therefore cannot be declared decision-eligible unless the unchanged existing functions actually produce a complete, compact, separating, registry-backed contrast. No new intervention family is authorized by this design.

## 4. Change surface and LOC budget

### 4.1 Proposed implementation files

Only these new source/config/test files may be added after separate implementation authorization:

| New file | Responsibility | Estimated LOC |
|---|---|---:|
| `certa/active_v1/__init__.py` | version exports only | 8 |
| `certa/active_v1/role_contract.py` | prompt, wire/full schemas, validator, Scheme-A retrieval projection | 195 |
| `certa/active_v1/planner_adapter.py` | three arm views, capability enforcement, canonical compile boundary | 220 |
| `certa/active_v1/artifact_authority.py` | hashes, raw grounding/derivation/registry emission and reconciliation | 225 |
| `certa/active_v1/decision_adapter.py` | compact packet, CERA response schema, registry reconciliation, materializer | 205 |
| **new production total** |  | **853** |
| **existing production modifications** | none | **0** |
| **high-risk existing production files changed** | none | **0 files** |

| New tool/config/test file | Estimated LOC |
|---|---:|
| `tools/certa_active_v1.py` | 365 |
| `scripts/07_run_certa_active_v1.sh` | 145 |
| `configs/profiles/certa_active_v1.env` | 45 |
| `tests/active_v1/test_role_contract.py` | 210 |
| `tests/active_v1/test_capability_and_planner.py` | 315 |
| `tests/active_v1/test_constructor_artifacts.py` | 255 |
| `tests/active_v1/test_decision_authority.py` | 245 |
| `tests/active_v1/test_orchestration_contract.py` | 110 |
| **tests/tools/config total** | **1,690** |

Design documents and generated run artifacts are excluded from the Pack's production/test LOC caps. Generated artifacts are data, not source. The estimate leaves 47 production LOC and 110 test/tool LOC of contingency; exceeding either estimate is allowed only while remaining within the Pack hard cap and must be reported before execution.

### 4.2 Explicitly unchanged files

No existing production file is proposed for modification. In particular:

- `certa/egra/retrieval.py` remains byte-identical;
- `certa/repair/causal_epistemic_agent.py` remains byte-identical;
- `certa/operations/contracts.py` remains the ontology authority, not an Active allowlist;
- `certa/planner/compiler.py` remains the historical lookup compiler;
- `run_cscr_pipeline.py` remains the historical generator/B0 implementation;
- all existing scripts, profiles, tests, output directories, terminal states, and negative artifacts remain unchanged;
- the main Pack and its calculators/schemas remain external immutable inputs.

## 5. Executor-backed signature capability matrix

Registry presence is necessary and insufficient. At design time all twelve Pack signatures are `CANDIDATE`; none is pre-authorized merely because it appears in `OPERATION_SIGNATURES`.

| Signature | Registry | Legacy compiler | Existing executor family | Design-time state |
|---|---:|---:|---|---|
| `LOOKUP_VALUE_SCALAR` | yes | pass | `lookup_value` | `CANDIDATE` |
| `LOOKUP_VALUE_ENTITY` | yes | pass | `lookup_value` | `CANDIDATE` |
| `COUNT_SCALAR` | yes | fail: `unsupported_operation_family` | `count_scope` | `CANDIDATE` |
| `SUM_SCALAR` | yes | fail: `unsupported_operation_family` | `sum_scope` | `CANDIDATE` |
| `AVERAGE_SCALAR` | yes | fail: `unsupported_operation_family` | `average_scope` | `CANDIDATE` |
| `DIFF_SCALAR` | yes | fail: `unsupported_operation_family` | `difference` | `CANDIDATE` |
| `RATIO_SCALAR` | yes | fail: `unsupported_operation_family` | `ratio` | `CANDIDATE` |
| `ARGMAX_ENTITY` | yes | fail: `unsupported_operation_family` | `argmax_relation` | `CANDIDATE` |
| `ARGMAX_ENTITY_SET` | yes | fail: `unsupported_operation_family` | `argmax_relation` | `CANDIDATE` |
| `ARGMIN_ENTITY` | yes | fail: `unsupported_operation_family` | `argmin_relation` | `CANDIDATE` |
| `ARGMIN_ENTITY_SET` | yes | fail: `unsupported_operation_family` | `argmin_relation` | `CANDIDATE` |
| `PAIR_COMPARE_BOOLEAN` | yes | fail: `unsupported_operation_family` | `pair_boolean_compare` | `CANDIDATE` |

The existing parent fixture
`tests/test_round1_operation_contracts.py::Round1OperationContractTests::test_every_declared_signature_has_contract_resolution_execution_projection_provenance_and_replay`
exercises contract resolution, closure, deterministic execution, projection, provenance, and benign replay for every registered signature. It does not exercise the new Active compiler boundary or the required canonical serialization round trip. It is therefore supporting evidence, not activation evidence.

The implementation must produce one machine-readable row per signature with these exact booleans:

```json
{
  "signature_id": "COUNT_SCALAR",
  "registry_present": true,
  "active_compiler_fixture_pass": false,
  "closure_fixture_pass": false,
  "deterministic_executor_fixture_pass": false,
  "projection_fixture_pass": false,
  "serialization_roundtrip_fixture_pass": false,
  "active": false,
  "failure_reasons": []
}
```

The activation equation is fixed:

```text
active = registry_present
         and active_compiler_fixture_pass
         and closure_fixture_pass
         and deterministic_executor_fixture_pass
         and projection_fixture_pass
         and serialization_roundtrip_fixture_pass
```

Each fixture must assert the signature ID, operation family, required role shapes, canonical program ID, deterministic executed value, projection operator, answer domain, nonempty provenance, and byte-identical canonical JSON after deserialize/serialize. At least one negative fixture per signature must prove malformed role shape or projection is rejected. The matrix, its test report, and its SHA-256 are frozen **before** the role prompt, schema, predictions, or labels. The role prompt exposes only rows where `active=true`; every other signature is classified as `UNSUPPORTED` for this Goal.

## 6. Nine-field role contract

The endpoint payload has exactly nine fields, including `schema_version`:

```json
{
  "schema_version": "certa_active_role_contract_v2",
  "supported": true,
  "intent": "COUNT",
  "answer_role": "SCALAR",
  "projection": "SCALAR_RESULT_PROJECTION",
  "signature": "COUNT_SCALAR",
  "cardinality": "SINGLE",
  "requires_time_scope": false,
  "requires_unit_consistency": false
}
```

The full semantic schema and local validator enforce these exact tuples:

| `signature` | `intent` | `answer_role` | `projection` | `cardinality` | operation family |
|---|---|---|---|---|---|
| `LOOKUP_VALUE_SCALAR` | `DIRECT_READ` | `SCALAR` | `VALUE_PROJECTION` | `SINGLE` | `LOOKUP` |
| `LOOKUP_VALUE_ENTITY` | `DIRECT_READ` | `ENTITY` | `VALUE_PROJECTION` | `SINGLE` | `LOOKUP` |
| `COUNT_SCALAR` | `COUNT` | `SCALAR` | `SCALAR_RESULT_PROJECTION` | `SINGLE` | `COUNT` |
| `SUM_SCALAR` | `SUM` | `SCALAR` | `SCALAR_RESULT_PROJECTION` | `SINGLE` | `SUM` |
| `AVERAGE_SCALAR` | `AVERAGE` | `SCALAR` | `SCALAR_RESULT_PROJECTION` | `SINGLE` | `AVERAGE` |
| `DIFF_SCALAR` | `DIFFERENCE` | `SCALAR` | `SCALAR_RESULT_PROJECTION` | `SINGLE` | `DIFF` |
| `RATIO_SCALAR` | `RATIO` | `SCALAR` | `SCALAR_RESULT_PROJECTION` | `SINGLE` | `RATIO` |
| `ARGMAX_ENTITY` | `ARGMAX` | `ENTITY` | `ROW_ENTITY_PROJECTION` | `SINGLE` | `ARGMAX` |
| `ARGMAX_ENTITY_SET` | `ARGMAX` | `SET` | `ROW_ENTITY_PROJECTION` | `MULTIPLE` | `ARGMAX` |
| `ARGMIN_ENTITY` | `ARGMIN` | `ENTITY` | `ROW_ENTITY_PROJECTION` | `SINGLE` | `ARGMIN` |
| `ARGMIN_ENTITY_SET` | `ARGMIN` | `SET` | `ROW_ENTITY_PROJECTION` | `MULTIPLE` | `ARGMIN` |
| `PAIR_COMPARE_BOOLEAN` | `PAIR_COMPARE` | `BOOLEAN` | `BOOLEAN_PROJECTION` | `SINGLE` | `PAIR_COMPARE` |

Unsupported is one exact canonical tuple:

```json
{
  "schema_version": "certa_active_role_contract_v2",
  "supported": false,
  "intent": "UNSUPPORTED",
  "answer_role": "UNSUPPORTED",
  "projection": "UNSUPPORTED",
  "signature": "UNSUPPORTED",
  "cardinality": "UNKNOWN",
  "requires_time_scope": false,
  "requires_unit_consistency": false
}
```

The model's core semantic fields are never silently rewritten. Any mismatch among `supported`, `intent`, `answer_role`, `projection`, `signature`, or `cardinality` is a semantic-schema and local-validator failure. The two requirement flags are model predictions and remain separately scored or audited; they may enrich query constraints but cannot activate a signature, create an executor, or select an answer.

### 6.1 Scheme-A legacy compatibility projection

The Active record is authoritative. For frozen EGRA retrieval only, the adapter deterministically derives this non-authoritative view:

| Legacy retrieval key | Exact derivation |
|---|---|
| `supported_by_core_signatures` | Active `supported` |
| `answer_domain` | Active `answer_role` |
| `intent_family` | `DIRECT_READ`, `COUNT`, `SUM`, `AVERAGE`, `DIFFERENCE`, `RATIO`, or `PAIR_COMPARE` unchanged; `ARGMAX→RANK_MAX`; `ARGMIN→RANK_MIN`; unsupported unchanged |
| `signature_candidates` | `[signature]` when supported, otherwise `[]` |
| `projection_candidates` | `[projection]` when supported, otherwise `[]` |
| `cardinality` | Active `cardinality` |
| `rank_direction` | `MAX` for `ARGMAX`, `MIN` for `ARGMIN`, otherwise `NONE` |
| `rank_k` | always `null` because no Active signature implements KTH |
| `requires_time_scope` | unchanged |
| `requires_unit_consistency` | unchanged |
| `unknowns` | `[]` |

This is the only deterministic canonicalization. In particular, a non-ranking operation can never carry `rank_direction=UNKNOWN` or `rank_k=1`, which removes the Gate-W non-applicable-field failure without changing its operation, role, projection, candidate space, executor, or answer authority. The legacy view is not written as `ROLE_RECORDS.jsonl`, not scored as a role output, and not accepted by the legacy role validator; it is serialized solely for `retrieve_structural_cards`, hashed, and discarded after Planner-view construction.

## 7. Planner adapter contract

The adapter has these pure interfaces:

```text
build_arm_view(arm, question, graph, table_json, role, retrieval, active_signatures)
  -> PlannerViewBuild

compile_active_planner_payload(raw, view, capability_matrix)
  -> ActiveCompilationResult(normalized_payload, errors)

close_compiled_payload(compilation, graph, capability_matrix)
  -> PlanClosure
```

Preconditions and failure behavior are fixed:

1. `build_arm_view` rejects an unknown arm, invalid role, inactive signature, missing retrieval result for C2, retrieval reference outside the complete schema domain, or any C1/C2 role hash mismatch.
2. C0 receives all and only active signatures. C1/C2 receive exactly one active signature.
3. The view is proposal-blind and value-firewalled. B0 and final answers are forbidden recursively from the view and prompt.
4. `validate_typed_planner_output(..., require_signature_id=True)` is mandatory.
5. `compile_active_planner_payload` accepts only `validation.ok=true`, performs no repair, and rejects any plan signature absent or inactive in the matrix.
6. Canonical serialization must be byte-identical across repeat calls.
7. `build_plan_closure(..., allowed_signature_ids=...)` receives the same allowlist used to build the view.
8. Resource-incomplete closure cannot be executable or silently truncated.
9. No invalid/unsupported role falls back to a broader signature space.
10. Planner output never becomes a final answer; only deterministic closure execution may create a candidate answer.

## 8. Raw-constructor artifact authority

### 8.1 Groundings

`PlanClosure.assignments` is the sole grounding source. Because closure may deduplicate one assignment across multiple `plan_ids`, the emitter expands it back to one Pack grounding record per `(sample_id, arm, plan_id)`.

For each plan:

- `required_operand_roles` comes from the frozen signature contract;
- `grounding_candidates` contains every assignment that names the plan;
- `binding_id` is the closure `assignment_id`;
- `operand_node_ids` is the exact resolved/matched operand-node sequence;
- candidate `valid=true` iff `resolution_state == "UNIQUE"` and `resource_complete == true`; execution success is not part of grounding validity;
- `selected_binding_id` is set iff exactly one candidate is valid; otherwise it is `null`;
- `first_match_used` is always `false`; any true value is a permanent safety failure;
- closure outcome and failure reasons are preserved in an additional checksummed `CLOSURE_OUTCOMES.jsonl` sidecar, never inferred from a summary.

### 8.2 Derivations

A Pack raw derivation is emitted only when the selected unique binding has a deterministic executable derivation and a valid projection. Failures remain visible in raw grounding and closure-outcome artifacts; they are not given fabricated answer hashes.

If closure merged plan IDs, one record is emitted per plan ID with a deterministic derivation ID over `(sample_id, arm, plan_id, binding_id, canonical_program_id)`. Every record takes these values directly from the selected closure assignment/derivation:

- signature, role, projection and canonical program;
- operand node IDs and provenance IDs;
- execution and projection status;
- projected answer equivalence hash;
- answer-class ID;
- original/alternative side relative to frozen B0.

No plan score, array order, first match, registry presence, or gold field can cause emission.

### 8.3 Answer equivalence and hashes

One function is authoritative everywhere:

```text
active_answer_hash(value)
  = SHA256(UTF8(canonical_json({
      "equivalence_key": inference_answer_key(value).compact()
    })))
```

The same function hashes B0, executed projections, registry answers, proposed finals, selected finals, and post-freeze gold. Thus the Pack calculators' exact hash comparison implements the existing inference-answer equivalence contract instead of surface-string equality.

`answer_class_id` is `AC-` followed by the full `active_answer_hash`. Raw answer text remains in the blind checksummed answer vault described below; public Pack records retain hashes.

### 8.4 Registry

`FROZEN_REGISTRY.jsonl` is built only from raw derivations satisfying all of:

```text
selected unique binding
and execution_status == EXECUTED
and projection_status == VALID
and nonempty provenance_ids
and exact signature/role/projection contract
```

Each admitted derivation has exactly one registry entry and each entry points to exactly one admitted derivation. Side, program, answer class, answer hash, and provenance must match byte-for-byte or Gate reconciliation fails. The registry cannot add a candidate, change an answer, choose an alternative, or make a signature active.

The Pack registry schema intentionally stores hashes. CERA v3 requires the already-executed answer value and H/D/E/I records, so two additional blind artifacts are frozen with the constructor:

- `constructor/FROZEN_ANSWER_VAULT.jsonl`: content-addressed `answer_hash → executed_answer` records, with exactly one canonical answer per hash;
- `constructor/FROZEN_CONTRAST_PROVENANCE.jsonl`: the exact fixed-basis H/D/E/I registry and intervention responses derived from existing deterministic functions.

Neither sidecar contains gold or an answer absent from raw deterministic execution. Both are included in `CONSTRUCTOR_FREEZE.json`, the required-artifact ledger, and terminal checksums. Any one-to-many hash mapping, missing derivation, provenance mismatch, or post-freeze change is a permanent failure and forces B0.

## 9. Registry, validator, and materializer decision authority

| Component | May do | May not do |
|---|---|---|
| operation registry | describe signatures and executor contracts | activate a signature or select a candidate |
| capability matrix | admit signatures whose five fixture gates pass | infer task semantics or select an answer |
| role validator | accept/reject the exact nine-field contract | repair fields or broaden to C0 |
| Planner validator/Active compiler | accept and canonicalize typed plan structure | execute, rank, or generate final answers |
| closure/executor | ground and deterministically execute typed plans | use gold or choose a final answer |
| raw artifact emitter | serialize exact closure/executor facts | infer facts from runner summaries |
| frozen registry | admit reconciled executable candidates | create external candidates or choose one |
| CERA | propose KEEP, USE one cited registry hypothesis, or INSUFFICIENT | generate an unregistered answer or see gold/table |
| CERA v3 validator | accept/reject cited H/D/E/I and executed-answer consistency | invent or substitute a candidate |
| Active reconciliation | map accepted H/D/E/I to one Pack registry/derivation/hash | accept ambiguous or external mappings |
| deterministic materializer | emit registry answer iff all authorities agree; otherwise B0 | call a model, normalize to a new value, or override validator |
| Pack calculators | compute Gates and terminal decisions from frozen raw artifacts | mutate method records or thresholds |

The materializer rule is exact:

```text
if action == USE_REGISTRY
   and CERA-v3 validator accepted
   and exactly one selected H maps to exactly one D
   and D maps to exactly one Pack registry entry
   and H/D/registry/answer-vault hashes all agree
   and the derivation side is ALTERNATIVE
then selected_final = the answer-vault value for that registry answer hash
else selected_final = frozen B0
```

`KEEP_B0` and `INSUFFICIENT` always materialize B0. `DETERMINISTIC_SELECTOR` is a frozen control arm and never informs the primary `CERA_PLUS_VALIDATOR` decision. The primary decision cannot run before Gate O PASS.

## 10. Freeze and sealed-access manifests

The Pack `INTERFACE_FREEZE_SCHEMA.json` has `additionalProperties=false`, so `ROLE_INTERFACE_FREEZE.json` remains exactly:

```text
schema_version
method_sha
role_source_sha256
prompt_sha256
wire_schema_sha256
semantic_schema_sha256
validator_sha256
created_at
```

To bind the additional required identities without changing that schema, `role_source_sha256` is the SHA-256 of canonical JSON in `freeze/ROLE_INTERFACE_SOURCE_MANIFEST.json`. That source manifest contains only:

```text
role source file hashes
capability-matrix path and SHA-256
model-profile path and SHA-256
method commit SHA
Pack manifest SHA-256
```

It contains no question, prediction, label, label hash, label path, access time, or score. `ROLE_INTERFACE_FREEZE.json` therefore remains restricted to source, prompt, schema, validator, capability-matrix, model, and commit identities, with the latter identities transitively bound through `role_source_sha256`.

The separate sealed-access manifest follows its committed JSON Schema template. Before access it records the required resource location, immutable question/prediction/interface identities, required access ordering, authorized accessor class, and `label_opened=false`. The independent analyzer fills the observed label hash and access timestamp only after verifying the interface commit and prediction close. The runtime process never receives this manifest or the label path.

## 11. Replay and test plan

Implementation must be test-driven and proceed in this order. These commands are specifications only and are **not authorized for execution by this design commit**.

### 11.1 Preflight and parent proof

```bash
export CERTA_ACTIVE_PYTHON="${CERTA_ACTIVE_PYTHON:-$(realpath "$(command -v python)")}"
export CERTA_ACTIVE_OUTPUT=/home/hsh/ME/Table/EMNLP2026/certa_active_v1_outputs/CERTA_ACTIVE_V1_FINAL
test -x "$CERTA_ACTIVE_PYTHON"
test "$(git rev-parse origin/master)" = "819aa30f499f337b77b0b731b4c833a3f50726fa"
test "$(git rev-parse origin/master^{tree})" = "9e044df0c3dabeaf988732a314910e5a980d399c"
test "$(git merge-base HEAD origin/master)" = "819aa30f499f337b77b0b731b4c833a3f50726fa"
test -z "$(git status --porcelain)"
"$CERTA_ACTIVE_PYTHON" /home/hsh/ME/Table/EMNLP2026/certa_goal_packs/CERTA_ACTIVE_V1_FINAL_METHOD_GOAL_REVISED_PACK/runtime_preflight.py \
  --output /home/hsh/ME/Table/EMNLP2026/certa_active_v1_outputs/CERTA_ACTIVE_V1_FINAL/intake/PYTHON_RUNTIME_PREFLIGHT.json
```

### 11.2 Red/green unit order

```bash
"$CERTA_ACTIVE_PYTHON" -m pytest -q tests/active_v1/test_role_contract.py
"$CERTA_ACTIVE_PYTHON" -m pytest -q tests/active_v1/test_capability_and_planner.py
"$CERTA_ACTIVE_PYTHON" -m pytest -q tests/active_v1/test_constructor_artifacts.py
"$CERTA_ACTIVE_PYTHON" -m pytest -q tests/active_v1/test_decision_authority.py
"$CERTA_ACTIVE_PYTHON" -m pytest -q tests/active_v1/test_orchestration_contract.py
```

Required hostile fixtures include malformed JSON; extra/missing role fields; every inconsistent role tuple; inactive signature; duplicate/unknown plan IDs; wrong role shape; unknown schema ID; projection/domain mismatch; ambiguous grounding; resource-incomplete closure; first-match attempt; nondeterministic replay; empty provenance; registry-external answer; duplicate registry mapping; answer-vault tamper; CERA out-of-registry text; invalid H/D/E/I citation; unseparating intervention; validator rejection; post-freeze timestamp; and any attempted non-B0 fallback.

### 11.3 Capability and serialization freeze

```bash
"$CERTA_ACTIVE_PYTHON" tools/certa_active_v1.py capability-fixtures \
  --output "$CERTA_ACTIVE_OUTPUT/freeze/SIGNATURE_CAPABILITY_MATRIX.json"
"$CERTA_ACTIVE_PYTHON" tools/certa_active_v1.py freeze-role-interface \
  --capability-matrix "$CERTA_ACTIVE_OUTPUT/freeze/SIGNATURE_CAPABILITY_MATRIX.json" \
  --output "$CERTA_ACTIVE_OUTPUT/freeze/ROLE_INTERFACE_FREEZE.json"
```

The second command must refuse to run if any active row lacks a passing fixture dimension or if the default-frozen source hashes changed.

### 11.4 Offline fixture replay

```bash
PACK=/home/hsh/ME/Table/EMNLP2026/certa_goal_packs/CERTA_ACTIVE_V1_FINAL_METHOD_GOAL_REVISED_PACK
"$CERTA_ACTIVE_PYTHON" "$PACK/validate_pack.py"
"$CERTA_ACTIVE_PYTHON" "$PACK/semantic_validate_pack.py"
"$CERTA_ACTIVE_PYTHON" "$PACK/tools/compute_certa_active_constructor_gate.py" \
  --allow-fixture \
  --identities "$PACK/fixtures/CONSTRUCTOR_IDENTITIES.jsonl" \
  --role-records "$PACK/fixtures/ROLE_RECORDS.jsonl" \
  --groundings "$PACK/fixtures/RAW_GROUNDINGS.jsonl" \
  --derivations "$PACK/fixtures/RAW_DERIVATIONS.jsonl" \
  --registry "$PACK/fixtures/REGISTRY.jsonl" \
  --cost-ledger "$PACK/fixtures/COST_LEDGER.json" \
  --output "$CERTA_ACTIVE_OUTPUT/replay/FIXTURE_CONSTRUCTOR_GATE.json"
```

The Decision fixture replay uses the Pack's exact `compute_certa_active_decision_gate.py` argument contract and writes only under `$CERTA_ACTIVE_OUTPUT/replay/`. It must never point a real run at fixture IDs or use fixture output as scientific evidence.

### 11.5 Existing regression and immutability checks

```bash
"$CERTA_ACTIVE_PYTHON" -m pytest -q tests
git diff --exit-code 819aa30f499f337b77b0b731b4c833a3f50726fa -- \
  certa/egra/retrieval.py certa/repair/causal_epistemic_agent.py
sha256sum certa/egra/retrieval.py certa/repair/causal_epistemic_agent.py
git diff --check
```

The expected default-frozen hashes are the values in Section 2. Any mismatch stops implementation before endpoint execution.

### 11.6 Later scientific execution order

After separate endpoint authorization, the only valid order is:

```text
preflight → cohorts/B0 freeze → capability matrix → role interface commit/hash
→ sealed role predictions close → independent label access → Role Gate
→ integration16 → at most one non-semantic wiring repair
→ matched dev64 C0/C1/C2 → constructor freeze → Gate C
→ independent gold access → Gate O
→ primary decision freeze → eligible-only dev CERA → materialize → dev Gate
→ if positive, one frozen table-disjoint holdout → final freeze
```

The four Pack read-only checkpoint auditors run only at their specified checkpoints. They may not edit or call endpoints.

### 11.7 Endpoint call ceiling

The exact Pack maximum remains 668 logical calls:

| Component | Maximum logical calls |
|---|---:|
| visible role development | 12 |
| sealed role confirmation | 16 |
| dev B0 | 64 |
| dev role | 64 |
| dev Planner C0/C1/C2 | 192 |
| dev CERA, eligible only | 64 |
| holdout B0 | 64 |
| holdout role | 64 |
| holdout frozen C2 Planner | 64 |
| holdout CERA, eligible only | 64 |
| **total** | **668** |

Integration16 is a cache-reused prefix of dev and adds zero logical calls. A transport-failed logical call may have at most two attempts; a semantically valid but wrong output is never retried.

## 12. Scientific-method preservation

Scheme A changes representation at an adapter boundary, not the estimand or method object:

- role remains a question-only LLM classification;
- retrieval remains the same frozen E5 model, cards, K, expansion, and deterministic tie-breaking;
- Planner remains the same typed schema-only LLM call;
- closure and all executors remain deterministic existing implementations;
- original/alternative is still answer equivalence to frozen B0;
- compact contrast remains existing fixed-basis behavioral equivalence and H/D/E/I provenance;
- CERA remains the only repair decision model and cannot generate outside the registry;
- the safety validator and deterministic materializer retain final-answer authority;
- Pack gates, cohorts, thresholds, access order, and terminal states do not change.

The interface recovery does change which malformed legacy representation is possible: non-ranking Active roles no longer expose model-controlled `rank_direction` or `rank_k`. Because those fields have no operation, candidate-space, executor, projection, or final-answer authority for non-ranking signatures, deriving `NONE/null` in the retrieval-only compatibility view removes a non-applicable representation degree of freedom without changing core role semantics.

## 13. Hostile self-audit

| Attack | Falsifiable defense | Failure action |
|---|---|---|
| “This is a copied pipeline.” | No existing implementation is copied; imports and exact public calls are listed in Section 3; new production budget is 853 LOC. | Reject any duplicated executor/retrieval/CERA block. |
| “The adapter silently edits model semantics.” | Authoritative nine-field outputs are never repaired; inconsistent core tuples fail. Only a non-authoritative legacy retrieval view derives rank fields. | Role invalid; no fallback. |
| “Registry membership activates signatures.” | Every row begins `CANDIDATE`; `active` is the conjunction of five committed fixture booleans. | Exclude signature from prompt/allowlist. |
| “The legacy compiler gap is hidden.” | Section 5 reports the two passes and ten exact `unsupported_operation_family` failures; the legacy compiler is not modified. | Fail any claim that legacy compiler supports all signatures. |
| “Active compiler is just a renamed pass-through.” | It accepts only validated normalized plans, enforces the frozen capability allowlist, and proves canonical round-trip identity; closure remains the separate executor. | Activation fixture fails. |
| “C2 changes more than retrieval.” | Cross-arm identities bind B0, model, schema, registry, closure, executor, budgets, samples/order; C1/C2 bind the same role hash. | Gate C safety failure. |
| “Grounding success is inferred from execution.” | Grounding validity uses only `resolution_state==UNIQUE` and `resource_complete`; execution is recorded separately. | Raw reconciliation failure. |
| “Failed plans disappear.” | Every closure assignment and failure is in raw grounding plus `CLOSURE_OUTCOMES.jsonl`; only answer-bearing derivations require valid execution/projection. | Required-artifact failure. |
| “Registry or CERA can invent an answer.” | Registry is one-to-one with raw executed derivations; CERA must cite H/D/E/I; materializer requires registry and answer-vault equality. | B0 fallback plus safety count. |
| “The answer vault is a hidden candidate source.” | Every vault row reconciles to an executed derivation/hash; no unmatched value is accepted. | Permanent reconciliation failure. |
| “Non-LOOKUP CERA evidence is invented.” | Existing fixed-basis implementation is reused unchanged and currently builds basis items only for LOOKUP. Non-LOOKUP is ineligible unless existing functions produce complete separating evidence. | B0 fallback; no new intervention code. |
| “Gold can influence construction or decision.” | Runtime rows exclude gold; constructor and primary decisions freeze before independent access; timestamps/hashes are checked by Pack tools. | Permanent terminal failure. |
| “The sealed path leaked into the interface.” | It appears only in the separate manifest-template file; `ROLE_INTERFACE_FREEZE.json` schema cannot contain it. | Freeze refused. |
| “Pack identity fields were extended ad hoc.” | `ROLE_INTERFACE_FREEZE.json` stays Pack-valid; `role_source_sha256` binds a separate source-only identity manifest. | Interface-freeze validation failure. |
| “A frozen legacy file had to change.” | Proposed modifications are zero and the exact pre/post hashes are asserted. | Stop and request exception with failing red test. |
| “Fixture success is reported as science.” | Fixture records remain `fixture_only=true`; real Pack tools reject them without `--allow-fixture`; fixture outputs live under replay only. | No scientific claim; real run blocked. |
| “The adapter changes thresholds after observing outputs.” | All Gate calculators and thresholds are immutable Pack inputs and are hashed before execution. | Permanent freeze. |

### Self-audit disposition

The design is scientifically defensible **only if** implementation preserves all fail-closed rules above. The largest residual risk is not the rank canonicalization; it is over-claiming operation support from registry/closure presence before committed end-to-end capability fixtures, and over-claiming decision eligibility for non-LOOKUP signatures when the existing v3 fixed-basis code may not create separating interventions. Both risks are explicitly converted into pre-role activation gates or B0 fallback.

## 14. Design submission boundary

This commit may contain only this specification and the separate sealed-access manifest JSON Schema template. It intentionally contains no implementation plan, source, test, profile, runtime artifact, result, label, score, terminal-state change, or endpoint evidence. The next valid action is Research Director review. Implementation remains unauthorized.
