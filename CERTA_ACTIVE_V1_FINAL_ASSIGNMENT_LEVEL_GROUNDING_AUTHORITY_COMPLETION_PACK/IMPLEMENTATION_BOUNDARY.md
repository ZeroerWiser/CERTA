# Implementation Boundary

## Authorized production paths

```text
certa/active_v1/artifact_authority.py
tools/certa_active_v1_completion.py
tools/compute_certa_active_constructor_gate_v3.py
schemas/active_v1/RAW_GROUNDING_RECORD_V3.schema.json
```

A new small helper under `certa/active_v1/` is allowed only if it keeps `artifact_authority.py` focused and contains no scoring or model call.

## Authorized tests

```text
tests/active_v1/test_assignment_level_grounding_authority.py
tests/active_v1/test_constructor_gate_v3.py
tests/active_v1/fixtures/grounding_authority/*
```

## Explicitly frozen paths

```text
certa/active_v1/role_contract_v3.py
certa/active_v1/planner_bridge_v3.py
certa/active_v1/planner_transport_projection.py
certa/planner/typed_planner.py
certa/planner/schema_view.py
certa/egra/retrieval.py
certa/grounding/structural_resolvers.py
certa/grounding/plan_closure.py
certa/operations/contracts.py
certa/derivations/project.py
certa/derivations/answer_equivalence.py
certa/derivations/iade.py
certa/derivations/contrast.py
certa/active_v1/decision_adapter.py
certa/repair/*
graph_builder.py
run_cscr_pipeline.py
```

## Size limit

Production additions/modifications: at most 500 logical lines. Tests/fixtures/tools: at most 900 logical lines. One implementation commit before replay; no source change after replay begins.

## Forbidden approaches

No first-match selection, answer agreement selection, gold use, fuzzy matching, lexical tie-break, learned ranker, confidence score, additional LLM, voting, new retrieval, new operation or relaxed Gate threshold.
