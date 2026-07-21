# CERTA Active V1 Thin-Adapter Implementation Amendment

Status:

```text
DESIGN_AMENDMENT APPROVED
IMPLEMENTATION AND ENDPOINT EXECUTION AUTHORIZED AFTER THIS DOCS-ONLY COMMIT
```

This amendment supplements, but does not replace or rewrite:

```text
docs/superpowers/specs/2026-07-22-certa-active-v1-thin-adapter-design.md
docs/superpowers/specs/CERTA_ACTIVE_V1_SEALED_ACCESS_MANIFEST_TEMPLATE.schema.json
```

It changes no method object, Pack Gate, threshold, cohort, model, endpoint, embedding, K, prompt
taxonomy, sealed question, or sealed label.

## 1. Exact resume identity

Implementation resumes from the existing design branch and commit:

```text
branch = research/certa-active-v1
required local HEAD = 02371bf8acba96bb6ad34ec74be384f3daa3708b
required remote ref = origin/research/certa-active-v1
required remote ref SHA = 02371bf8acba96bb6ad34ec74be384f3daa3708b

lineage parent = 819aa30f499f337b77b0b731b4c833a3f50726fa
parent tree = 9e044df0c3dabeaf988732a314910e5a980d399c
```

The branch must not be reset, rebased, amended, deleted, or recreated from master.

This amendment is the first new commit after `02371bf8...`. That commit contains documentation only.
After it is committed and the worktree is clean, implementation continues on the same branch.

## 2. Canonical sealed-label path

The authoritative label path is the const value in the committed sealed-access manifest schema:

```text
/home/hsh/ME/Table/EMNLP2026/certa_active_v1_sealed/
CERTA_ACTIVE_V1_ROLE_SEALED_LABELS/
ROLE_SEALED_CONFIRMATION_LABELS.json
```

Stale shorter paths in external Pack prose or fixtures are non-authoritative.

The implementation must:

- use this nested path exactly;
- fail closed if it is absent, not a regular file, has an unexpected hash, or violates the expected ACL;
- never copy, move, rename, flatten, or symlink the label file;
- record the path only in the separate sealed-access manifest;
- keep `ROLE_INTERFACE_FREEZE.json` free of label paths, labels, label hashes, access times, and scores.

Required ordering remains:

```text
role-interface commit
→ clean worktree and source-hash verification
→ ROLE_INTERFACE_FREEZE.json
→ prediction close
→ sealed-access authorization
→ first label open and access log
→ independent Role Gate
→ permanent prohibition on role edits
```

## 3. Constructor capability and decision capability are distinct

Registry presence is insufficient for either capability.

### 3.1 Constructor capability

The existing canonical artifact:

```text
freeze/SIGNATURE_CAPABILITY_MATRIX.json
```

remains the constructor capability matrix for Pack and design compatibility.

For every signature it must contain or bind:

```text
registry_present
active_compiler_fixture_pass
closure_fixture_pass
deterministic_executor_fixture_pass
projection_fixture_pass
serialization_roundtrip_fixture_pass
constructor_active
constructor_failure_reasons
active
```

Compatibility invariant:

```text
active == constructor_active
```

The role prompt and C0/C1/C2 Planner allowlists use only `constructor_active=true`.

Constructor activation is:

```text
constructor_active =
    registry_present
    and active_compiler_fixture_pass
    and closure_fixture_pass
    and deterministic_executor_fixture_pass
    and projection_fixture_pass
    and serialization_roundtrip_fixture_pass
```

The Active boundary is named a **validated closure-payload adapter**, not a general compiler, unless it
performs additional operation-specific compilation.

### 3.2 Decision capability

Create the companion artifact:

```text
freeze/DECISION_SIGNATURE_CAPABILITY_MATRIX.json
```

For every constructor-active signature record:

```text
signature_id
constructor_capability_matrix_sha256
constructor_active
contrast_fixture_pass
registry_fixture_pass
validator_materializer_fixture_pass
decision_active
decision_failure_reasons
```

Decision activation is:

```text
decision_active =
    constructor_active
    and contrast_fixture_pass
    and registry_fixture_pass
    and validator_materializer_fixture_pass
```

`PRIMARY_DECISION_FREEZE.json` must bind the decision matrix path and SHA-256 through its existing
source/config identity mechanism.

Only `decision_active=true` rows may invoke `CERA_PLUS_VALIDATOR`.

A signature may be:

```text
constructor_active=true
decision_active=false
```

Such rows remain valid Constructor evidence, but their selected-final action is deterministically
`B0_KEEP`. They cannot support a paper claim of active decision coverage.

The existing Pack Gate O remains an overall constructor-opportunity analysis. The decision stage must
add a descriptive decision-eligible opportunity census, but no Pack threshold is changed. The
preregistered Decision Gate remains the only authority for positive selected-final evidence.

## 4. Row-local retrieval failure is fail-closed and sample-preserving

Expected sample-scoped graph/card/index/retrieval failures must not terminate the split and must not
fall back to C1 or C0.

For every affected C2 row, emit a checksummed record to:

```text
constructor/C2_ROW_FAILURES.jsonl
```

Required fields:

```text
schema_version
sample_id
table_id
arm = C2_ROLE_RETRIEVAL
failure_stage
error_code
exception_class
message_sha256
graph_sha256
group_catalog_sha256
card_catalog_sha256
retrieval_config_sha256
role_record_sha256
row_preserved = true
fallback_arm = NONE
created_at
```

The corresponding C2 identity row remains present. Its raw grounding/derivation/registry contribution
is empty, and the earliest failure stage is included in `CONSTRUCTOR_FAILURE_TAXONOMY.csv`.

Expected row-local errors include missing active cards, incomplete card provenance, answer-value
exposure, empty retrieved references, or a sample-scoped catalog inconsistency.

Global invariant failures still stop the run, including:

```text
source/config hash drift
embedding model unavailable
shared index configuration mismatch
operation/closure/executor identity mismatch
artifact schema mismatch
corrupt split-level manifest
```

Do not catch `BaseException`, suppress unknown errors, or translate global identity failure into a row
failure.

## 5. Integration16 must inspect real Active structural objects

Before invoking the unchanged Pack operational-sentinel calculator, emit:

```text
integration/INTEGRATION16_ACTIVE_STRUCTURE_PREFLIGHT.jsonl
```

For every sentinel sample record:

```text
sample_id
table_id
role_supported
constructor_active
hceg_node_count
canonical_group_count
active_card_count
retrieved_reference_node_ids
complete_schema_node_ids_sha256
reference_subset_valid
preflight_status
failure_reasons
```

Mechanical requirements:

- every sentinel row has a nonempty HCEG;
- every sentinel row has a nonempty canonical structural-group catalog;
- every C2-eligible row has a nonempty active evidence-card catalog;
- every C2-eligible row has nonempty retrieved reference node IDs;
- every retrieved reference node ID is a member of the complete proposal-blind schema domain;
- unsupported or constructor-inactive roles produce deterministic empty C1/C2 records and are not
  silently broadened;
- legacy `schema_edges > 0` is not a universal requirement.

For the unchanged Pack identity schema:

```text
graph_node_count
```

means the actual HCEG node count, and:

```text
card_count
```

means the pre-retrieval active structural-card catalog count, not the number of selected top-K cards.

The Active structure preflight must pass before the Pack Integration16 result can be treated as valid.
Initial failure permits the already authorized single non-semantic wiring repair. A second failure
returns `FREEZE_CERTA_ACTIVE_INTEGRATION_FAILED`.

## 6. Non-changes

This amendment does not authorize:

- changing Pack Gates, calculators, schemas, thresholds, cohorts, terminal states, retry rules, model,
  endpoint, sampling, embedding, K, sealed questions, or sealed labels;
- editing `certa/egra/retrieval.py`;
- editing `certa/repair/causal_epistemic_agent.py`;
- modifying old EGRA source, profiles, scripts, tests, outputs, or terminal artifacts;
- a new graph type, intervention family, model, reranker, lexical rules, score, threshold, or method
  object;
- CERA before Gate O;
- post-gold or post-holdout method changes.

## 7. Authorization transition

After this document is committed as a docs-only commit and the branch is clean:

```text
AUTHORIZE_CERTA_ACTIVE_V1_IMPLEMENTATION_AND_ENDPOINT_EXECUTION
```

becomes effective under the existing
`CERTA_ACTIVE_V1_FINAL_METHOD_GOAL_REVISED_PACK`.

No further design phase follows.
---

## 8. Micro-revision: exact sealed-file enforcement and path override

This section is binding and overrides any less-specific wording above.

```text
canonical_label_path = /home/hsh/ME/Table/EMNLP2026/certa_active_v1_sealed/CERTA_ACTIVE_V1_ROLE_SEALED_LABELS/ROLE_SEALED_CONFIRMATION_LABELS.json
labels_sha256 = 99f2dd85a7c8635bc0ea7061b868a0c8bd31d6ab2197f2510ec34a6f7f51b522
required_file_type = regular_file
required_mode = 0440
symlink_allowed = false
```

At authorized label access, the independent analyzer must mechanically require:

```text
lstat(path) is not a symbolic link
stat.S_ISREG(lstat(path).st_mode) is true
(stat(path).st_mode & 0o777) == 0o440
SHA256(file bytes) == labels_sha256
```

The access manifest records the actual numeric `uid`, `gid`, device, inode, byte size, `mtime_ns`,
mode, file type, symlink status, first-open time, analyzer PID, analyzer source/command hashes,
prediction-close hash, and Role Gate output hash. User and group names are not hard-coded.

The committed sealed-access manifest schema and canonical nested path override every stale short path
in immutable Pack prose, fixtures, or historical manifests. In particular,
`ROLE_VALIDATION_FREEZE_MANIFEST.json` is authoritative only for question identity, question hashes,
count, order, and prediction-set identity. It is not authoritative for label-path equality. A stale
path mismatch alone is not a blocker when the canonical path, SHA, ACL, question identity, and access
ordering pass.

The sealed file must never be copied, moved, flattened, renamed, or symlinked.

## 9. Micro-revision: machine Decision capability contract

Implementation must materialize and checksum:

```text
freeze/DECISION_SIGNATURE_CAPABILITY_MATRIX.schema.json
freeze/DECISION_SIGNATURE_CAPABILITY_MATRIX.json
```

Required matrix identity:

```text
schema_version = certa_active_v1_decision_capability_v1
```

Required invariants:

```text
exactly one row for every constructor-active signature
no row for a constructor-inactive signature
unique signature_id
canonical signature_id ordering
constructor_capability_matrix_sha256 required and exact
fixture booleans mechanically recomputed
matrix schema and matrix included in REQUIRED_ARTIFACTS and final SHA256SUMS
```

Each row must contain:

```text
signature_id
constructor_active
contrast_fixture_pass
registry_fixture_pass
validator_materializer_fixture_pass
decision_active
decision_failure_reasons
```

```text
decision_active =
    constructor_active
    and contrast_fixture_pass
    and registry_fixture_pass
    and validator_materializer_fixture_pass
```

`PRIMARY_DECISION_FREEZE.json` binds the constructor matrix SHA, decision schema SHA, and decision
matrix SHA.

After Gate O, compute a frozen decision-eligible opportunity census. If:

```text
decision_eligible_opportunity_count == 0
```

then:

```text
CERA logical calls = 0
CERA attempts = 0
all dev rows deterministically materialize B0_KEEP
terminal = FREEZE_CERTA_ACTIVE_DECISION_FAILED
failure_reason = ZERO_DECISION_ELIGIBLE_OPPORTUNITY
holdout = NOT RUN
```

This is a fail-closed decision stop, not a new Gate or threshold change.

## 10. Micro-revision: fresh blind Role analyzer

The final sealed Role Gate must be run by a fresh process/read-only subagent that did not participate
in role prompt, schema, validator, or production development. Its source and command hashes are frozen
before first label access. It receives labels only after prediction close.

Its output is terminal-only. It may compute the Role Gate, confusion matrices, counterexamples,
checksums, and stop recommendation. It cannot edit code, call the role endpoint, retry predictions,
or return label-derived development advice. No role-interface change follows sealed-label access.

The current Research Director review environment is not the independent Role analyzer.

## 11. Micro-revision: checkpoint subagent outputs

Exactly four one-time read-only audits are authorized:

```text
Checkpoint 1:
reviews/CHECKPOINT1_CAPABILITY_IMPLEMENTATION_AUDIT.json
reviews/CHECKPOINT1_CAPABILITY_IMPLEMENTATION_AUDIT.md

Checkpoint 2:
reviews/CHECKPOINT2_SEALED_ROLE_AUDIT.json
reviews/CHECKPOINT2_SEALED_ROLE_AUDIT.md

Checkpoint 3:
reviews/CHECKPOINT3_CONSTRUCTOR_RETRIEVAL_AUDIT.json
reviews/CHECKPOINT3_CONSTRUCTOR_RETRIEVAL_AUDIT.md

Checkpoint 4:
reviews/CHECKPOINT4_DECISION_SAFETY_AAAI_AUDIT.json
reviews/CHECKPOINT4_DECISION_SAFETY_AAAI_AUDIT.md
```

Each subagent reads only already frozen checkpoint artifacts. It cannot edit code, call endpoints,
lower Gates, suggest post-hoc thresholds, or run continuously. Checkpoint 2 is the fresh independent
Role analyzer and cannot feed sealed results back into role development.

## 12. Maximum remaining phases

```text
Phase M — current and final method-engineering Goal
Phase E — one conditional EXPERIMENT_ONLY stage after a positive Phase M; method changes = 0
Phase P — Paper/Release Freeze; method changes = 0 and method selection = 0
```

If Phase M fails any scientific Gate, Phase E is skipped. No second Active-V1 method-engineering Round
is authorized. Ordinary checkpoints continue without a new design approval; only a proposal to modify
a default-frozen legacy file pauses for an exception.
