# CERTA Active V1 Frozen Role V3 Final Method Completion Plan

Status: implementation authorized by the immutable completion Pack after restored-archive validation.

Execution root: `/home/hsh/ME/Table/EMNLP2026/CERTA`

Output root: `/home/hsh/ME/Table/EMNLP2026/certa_active_v1_outputs/CERTA_ACTIVE_V1_FROZEN_ROLE_V3_FINAL_METHOD_COMPLETION_ARCHIVE_RESTORED`

## Boundaries

- Preserve Role V3, Planner core, closure, executor, operation registry, retrieval, answer equivalence, and legacy CERA source byte-for-byte.
- Do not access gold or sealed Role V3 labels while implementing or testing.
- Do not call an endpoint before source, prompt, schema, capability, validator, materializer, eligibility, calculator, and commit identities are frozen.
- Do not create a second constructor, Decision policy, retriever, executor, or intervention family.
- Stop at the first failed frozen Gate.

## Implementation wave

1. `PlannerBridgeImplementer`
   - Owns only `certa/active_v1/planner_bridge_v3.py` and `tests/active_v1/test_planner_bridge_v3.py`.
   - RED: prove a frozen Role V3 canonical record cannot pass the V2 `RoleValidation` boundary.
   - GREEN: map the V3 record directly to the existing Planner query contract; preserve one role signature for C1/C2, all constructor-active signatures for C0, byte-identical C1/C2 role hashes, and fail-closed C2 reference validation.
   - Reuse `build_proposal_blind_planner_view`, `compile_active_planner_payload`, and `close_compiled_payload`; do not copy closure or execution logic.

2. `ArtifactAuthorityImplementer`
   - Owns only `certa/active_v1/artifact_authority.py` and `tests/active_v1/test_artifact_authority.py`.
   - RED: prove raw `PlanClosure` objects do not by themselves satisfy Pack raw-grounding, raw-derivation, and registry schemas.
   - GREEN: serialize closure assignments and executable derivations into Pack-bound records; reconcile sample/arm/signature/program/answer/provenance hashes; reject registry entries without executed derivation authority.
   - This module has no answer-selection, correctness, or Gate authority.

3. `DecisionAdapterImplementer`
   - Owns only `certa/active_v1/decision_adapter.py` and `tests/active_v1/test_decision_adapter.py`.
   - RED: prove an unregistered, unexecuted, validator-rejected, decision-inactive, or ineligible alternative cannot become selected-final.
   - GREEN: compute blind eligibility from paired registry-complete contrast plus decision capability, reconcile CERA V3 validator references to the frozen registry, and materialize either an exact registered executed answer or B0.
   - Reuse existing contrast, CERA schema/prompt, safety validator, and answer-equivalence primitives; do not modify or reimplement them.

4. `SchemaAndFixtureImplementer`
   - Owns only new files under `schemas/active_v1/`, `tests/active_v1/fixtures/final_completion/`, and `tests/active_v1/test_final_completion_capabilities.py`.
   - RED: show a registry-only signature cannot become constructor-active and an incomplete decision path cannot become decision-active.
   - GREEN: provide one positive and one negative actual-call-chain fixture for each of the twelve Role V3 supported IDs; bind constructor and Decision matrix rows to the immutable Pack schemas and exact conjunction equations.

Each implementer runs only focused offline tests and returns changed paths, line counts, hashes, RED/GREEN evidence, invariants, and stop conditions. No implementer edits shared integration files, commits, calls endpoints, or reads gold/sealed resources.

## Director integration

1. Review every implementation patch against ownership, no-duplication, and frozen-source constraints.
2. Add only the minimal shared runner/profile/tool wiring needed for:
   - capability matrices and capability Gate;
   - stable first16 Integration C0/C1/C2 records;
   - matched dev64 C0/C1/C2 records and raw Gate C inputs;
   - blind Decision request construction, registry reconciliation, selected-final close, and one dev unblind;
   - conditional frozen holdout.
3. Add/adjust shared tests before implementation code for each Director-owned behavior.
4. Enforce POST for all `/v1/chat/completions` probes and model requests; GET is permitted only for `/v1/models` identity inspection.
5. Run the complete offline suite, schema validation, capability fixtures, LOC audit, forbidden-path diff audit, and source/config hash freeze.
6. Commit once the implementation is locally green; record the exact commit and clean status before the first scientific endpoint call.

## Verification and stop sequence

1. Constructor capability: exactly twelve rows and all `constructor_active=true`; otherwise `FREEZE_CERTA_ACTIVE_CAPABILITY_FAILED`.
2. Decision capability: exactly twelve rows; inactive rows force zero CERA calls and B0 keep.
3. Read-only capability audit.
4. Integration16: 16 samples x 3 arms; one wiring-only repair of at most 100 production LOC is permitted only after a red test; a second or semantic failure stops.
5. Matched dev64: 64 samples x 3 arms, reusing Integration16 cache; raw immutable Gate C calculator controls continuation.
6. Freeze constructor and blind Decision identities/eligible IDs; close all selected finals before one dev-gold access.
7. In the single dev unblind, compute frozen Gate O, Decision Gate, CC/CW/WC/WW, costs, bootstrap intervals, clustered bootstrap, and McNemar exact test; no later endpoint call or method/config edit.
8. Run one table-disjoint holdout only if both dev Gates pass; freeze predictions before its single unblind.
9. Produce the first-failure terminal or the authorized positive terminal, complete checksums, call/token/latency ledger, clean Git status, and verified Git bundle.

Maximum new logical model calls after completed Role V3 qualification: 576.
