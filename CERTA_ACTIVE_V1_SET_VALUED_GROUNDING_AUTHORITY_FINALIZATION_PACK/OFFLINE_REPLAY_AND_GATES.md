# Offline Replay and Gates

## Phase A — evidence-only forensic replay

Before source edits or endpoint calls, read the immutable Constructor64 artifacts and reconstruct the authority relation from existing grounding candidates and derivation binding IDs.

Required outputs:

```text
AUTHORITY_FORENSIC_ROWS.jsonl
AUTHORITY_MISMATCH_CLASSIFICATION.json
SET_VALUED_SHADOW_GROUNDINGS.jsonl
SET_VALUED_SHADOW_RECONCILIATION.jsonl
SET_VALUED_SHADOW_GATE_C.json
```

The replay must classify every previous mismatch and prove whether it is explained solely by the singleton authorization rule.

## Gate A — authority validity

All conditions are required:

- 192/192 identities preserved;
- all old valid candidates are individually resolver-UNIQUE with nonempty operands;
- authorized IDs equal exactly the valid candidate IDs;
- no invalid or ambiguous candidate is authorized;
- every singleton record preserves its old selected binding;
- no first-match, score, threshold, answer correctness or gold access;
- raw derivation, program, answer, side and provenance hashes are unchanged;
- every newly reconciled derivation is admitted only by binding-set membership;
- zero unexplained reconciliation mismatches.

Failure terminal:

```text
FREEZE_CERTA_ACTIVE_SET_VALUED_AUTHORITY_INVALID
```

## Gate Q — scientific opportunity

Recompute the original Gate C with the original cohort and unchanged thresholds. Gate Q passes only when the revised authority makes the original Gate C PASS without changing Planner outputs, raw executions, cohorts or thresholds.

The report must include paired executable rows/tables, registry-complete paired rows, C2-versus-controls gains, safety counters, signature/operation/table breakdown and the exact rows responsible for every change.

Failure terminal:

```text
FREEZE_CERTA_ACTIVE_SET_VALUED_AUTHORITY_INSUFFICIENT
```

If Gate Q fails, make no production commit, do not run another model call, permanently stop method engineering and narrow the paper claim.

## Phase B — implementation

Only Gate A and Gate Q PASS authorize implementation.

Allowed production scope:

```text
certa/active_v1/artifact_authority.py
one versioned grounding schema
one versioned Constructor Gate calculator
tools/certa_active_v1_completion.py minimal wiring
```

Tests may be added under `tests/active_v1/`.

Ceilings:

```text
production changed/new LOC <= 450
tests <= 900 LOC
high-risk existing production files modified <= 2
default-frozen Planner/closure/operation/retrieval/Decision files modified = 0
```

Required tests:

- singleton compatibility;
- multiple individually UNIQUE bindings authorized;
- internal resolver ambiguity rejected;
- mixed valid/invalid candidates;
- deterministic ordering and hashes;
- derivation membership reconciliation;
- no answer-equivalence collapse;
- malformed and resource-incomplete cases fail closed;
- old 161+168 test suites remain passing.

## Phase C — final scientific execution

After one ordinary commit and push, use a new output root and the verified user-managed Qwen3-8B service. Run Integration16, Constructor64 and the original Gate C. At any failure stop permanently.

Only live Gate C PASS continues to actual selected-final Decision, prediction close, one dev unblind, Gate O/CC/CW/WC/WW and conditional table-disjoint holdout.
