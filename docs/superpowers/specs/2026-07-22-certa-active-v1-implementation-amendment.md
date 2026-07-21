/goal CERTA_ACTIVE_V1_IMPLEMENTATION_AND_ENDPOINT_EXECUTION

Operate only in `/home/hsh/ME/Table/EMNLP2026/CERTA` and continue using the immutable
`/home/hsh/ME/Table/EMNLP2026/certa_goal_packs/CERTA_ACTIVE_V1_FINAL_METHOD_GOAL_REVISED_PACK`
plus the already deployed sealed-label resource. Fetch remote refs and require:

`origin/master == 819aa30f499f337b77b0b731b4c833a3f50726fa`

`origin/research/certa-active-v1 == 02371bf8acba96bb6ad34ec74be384f3daa3708b`

Resume on clean `research/certa-active-v1@02371bf8acba96bb6ad34ec74be384f3daa3708b`.
Do not reset, rebase, amend, delete or recreate the branch, and do not restart from master.

First add only
`docs/superpowers/specs/2026-07-22-certa-active-v1-implementation-amendment.md`
using the exact Research Director amendment, commit it as a docs-only commit, verify the worktree is
clean, record the amendment commit SHA and continue in this same Goal. Do not pause for another design
round.

The amendment has five binding rules:

1. The canonical sealed-label file is
   `/home/hsh/ME/Table/EMNLP2026/certa_active_v1_sealed/CERTA_ACTIVE_V1_ROLE_SEALED_LABELS/ROLE_SEALED_CONFIRMATION_LABELS.json`.
   Never copy, move, flatten, rename or symlink it. Keep the path only in the separate access manifest.

2. Preserve `SIGNATURE_CAPABILITY_MATRIX.json` as the constructor matrix with
   `active == constructor_active`. Add `DECISION_SIGNATURE_CAPABILITY_MATRIX.json`; Role and Planner
   allowlists use constructor capability, while CERA eligibility uses decision capability. A
   constructor-active but decision-inactive row must deterministically keep B0.

3. Convert expected sample-scoped C2 graph/card/index/retrieval failures into explicit checksummed
   `C2_ROW_FAILURES.jsonl` records while preserving all rows and using no fallback arm. Global identity,
   environment, shared-index or artifact-schema failures still stop the run.

4. Before the unchanged Pack Integration16 calculator, mechanically verify actual HCEG nodes,
   canonical structural groups, active evidence cards, nonempty retrieved reference IDs for eligible
   rows, and reference-ID membership in the complete proposal-blind schema domain. Do not use legacy
   `schema_edges > 0` as a universal condition.

5. Keep `certa/egra/retrieval.py`, `certa/repair/causal_epistemic_agent.py`, old EGRA source,
   profiles, scripts, tests, terminal states and historical artifacts byte-identical.

After the docs-only amendment commit, implementation and endpoint execution are authorized. Execute
the existing revised Pack sequence without changing its Gates, thresholds, cohorts, endpoint,
embedding, K, sealed questions or labels:

Checkpoint 1: implement the versioned `certa/active_v1/` thin adapter, tests, profile and CLI; produce
constructor/decision capability matrices, red/green fixtures, row-local failure artifacts and Active
Integration16 preflight; run runtime preflight and regression tests before any endpoint call.

Checkpoint 2: commit and freeze the role interface; close sealed predictions before using the
canonical nested label path; run the independent sealed Role Gate and confusion matrices; make no role
edit after label access.

Checkpoint 3: run cache-reused Integration16 and matched dev64 C0/C1/C2; preserve all identities,
groundings, derivations, row failures and registry; compute Gate C from raw artifacts; freeze the
constructor before independent gold Gate O.

Checkpoint 4: only after Gate O PASS, freeze and run the primary `CERA_PLUS_VALIDATOR` decision with
`DETERMINISTIC_SELECTOR` control and `B0_KEEP` fallback, only on decision-active registry-complete
paired rows; deterministically materialize selected-final answers; compute dev CC/CW/WC/WW; run one
table-disjoint frozen holdout only if dev passes; then permanently freeze the method.

At the first failed scientific Gate, stop all method work and generate only the required terminal,
audits, costs, checksums and verified Git bundle. No new method Round, new module object, post-hoc
threshold, second model, reranker, lexical rule bank, registry-external answer or post-gold method edit
is authorized. Return all four checkpoint records, exact commits/diff, LOC, endpoint/token/latency
ledgers, raw artifacts, machine Gates, clean status and final bundle.

MICRO-REVISION OVERRIDES — these clauses are binding and override less-specific Pack or amendment text:

A. Exact sealed file and path:
`/home/hsh/ME/Table/EMNLP2026/certa_active_v1_sealed/CERTA_ACTIVE_V1_ROLE_SEALED_LABELS/ROLE_SEALED_CONFIRMATION_LABELS.json`
must be a non-symlink regular file with permission bits exactly `0440` and SHA256
`99f2dd85a7c8635bc0ea7061b868a0c8bd31d6ab2197f2510ec34a6f7f51b522`.
Record numeric UID/GID, device, inode, size, mtime, mode and access times. The committed sealed-access
schema and this nested path override all stale short paths. Do not compare label-path equality against
`ROLE_VALIDATION_FREEZE_MANIFEST.json`; use it only for question/hash/count/order and prediction-set
identities. Never copy, flatten, rename, move or symlink the label file.

B. Decision capability:
materialize and checksum `freeze/DECISION_SIGNATURE_CAPABILITY_MATRIX.schema.json` and
`freeze/DECISION_SIGNATURE_CAPABILITY_MATRIX.json` with
`schema_version=certa_active_v1_decision_capability_v1`; require exactly one unique canonical row per
constructor-active signature, no constructor-inactive rows, an exact constructor-matrix SHA, and
mechanically recomputed contrast/registry/validator-materializer fixtures. Bind schema/matrix hashes in
`PRIMARY_DECISION_FREEZE.json` and final required-artifact/checksum ledgers.

C. Zero decision opportunity:
after Gate O and decision-capability freeze, compute the decision-eligible opportunity census. If the
count is zero, make zero CERA calls and attempts, deterministically materialize `B0_KEEP` for all dev
rows, return `FREEZE_CERTA_ACTIVE_DECISION_FAILED` with
`ZERO_DECISION_ELIGIBLE_OPPORTUNITY`, and do not run holdout.

D. Fresh blind Role analyzer:
the final sealed Role Gate must be run by a fresh process/read-only subagent that did not participate
in prompt/schema/validator/production development. Freeze its source and command hashes before label
access. It receives labels only after prediction close. Its terminal-only output cannot trigger role
edits, retries, prompt/schema changes, or new role calls. The current Research Director review
environment is not that analyzer.

E. Checkpoint subagents:
run exactly one read-only audit after each checkpoint and return:
`CHECKPOINT1_CAPABILITY_IMPLEMENTATION_AUDIT.json/.md`,
`CHECKPOINT2_SEALED_ROLE_AUDIT.json/.md`,
`CHECKPOINT3_CONSTRUCTOR_RETRIEVAL_AUDIT.json/.md`, and
`CHECKPOINT4_DECISION_SAFETY_AAAI_AUDIT.json/.md`.
They read frozen artifacts only, never edit code or call endpoints, and do not run continuously.

F. Remaining phases:
M is this final method-engineering Goal. E is one conditional experiment-only stage after positive M
with zero method changes. P is Paper/Release Freeze with zero method changes and zero method
selection. If M fails, skip E. No second Active-V1 method-engineering Round is authorized.

After the docs-only amendment commit, continue directly through the four checkpoints without waiting
for ordinary approval. Pause only if a default-frozen legacy file must be changed.
::: 
