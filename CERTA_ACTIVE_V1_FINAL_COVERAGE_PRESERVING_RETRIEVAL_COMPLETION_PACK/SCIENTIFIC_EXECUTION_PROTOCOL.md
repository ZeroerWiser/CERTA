# Scientific Execution Protocol

## Inputs to preserve

- accepted method commit: `a6818af3c157f3416bdff84925e003e36b3c4583`;
- prior Gate C V3 recovery root:
  `/home/hsh/ME/Table/EMNLP2026/certa_active_v1_outputs/CERTA_ACTIVE_V1_GATE_C_V3_RECOVERY`;
- prior assignment-authority replay root:
  `/home/hsh/ME/Table/EMNLP2026/certa_active_v1_outputs/CERTA_ACTIVE_V1_FINAL_ASSIGNMENT_LEVEL_GROUNDING_AUTHORITY_REPLAY`;
- prior V2R2 constructor root:
  `/home/hsh/ME/Table/EMNLP2026/certa_active_v1_outputs/CERTA_ACTIVE_V1_PREWARMED_IDENTITY_V2R2_CONTROLLER_RECOVERY_FINAL_SCIENTIFIC_DAG`.

All previous terminals remain immutable.

## New output root

```text
/home/hsh/ME/Table/EMNLP2026/certa_active_v1_outputs/CERTA_ACTIVE_V1_FINAL_COVERAGE_PRESERVING_RETRIEVAL
```

## Phase 0: audit and tests

Before editing, four read-only auditors close the prior C1/C2 lineage and identify which C1 paired row and derivations disappeared under C2 hard projection. No gold may be read.

After implementation, require:

- complete offline suite PASS;
- direct-script and `python -m` smoke tests PASS for every new tool;
- C1/C2 complete reference domains identical on all 64 samples;
- C1/C2 typed response schemas identical on all 64 samples;
- C0/C1 prompt hashes byte-identical to prior frozen prompts;
- C2 focus IDs are a subset of the complete domain;
- no retrieved focus field enters response output or grounding authority;
- all previously frozen source blobs unchanged except explicitly allowed files.

Commit and push once. Source becomes immutable before scientific calls.

## Phase 1: replay C0/C1

Reconstruct C0 and C1 views and verify their prompt, transport-schema, full-schema, normalized payload, graph and closure hashes against frozen artifacts. Reuse their validated outputs only after every identity check passes.

Endpoint calls for C0 and C1 must be zero.

## Phase 2: C2 Integration16

For the first 16 dev samples:

- rebuild Role-conditioned retrieval using the frozen E5 index and budgets;
- build the complete-domain C2 view plus advisory focus;
- record C1/C2 domain hashes and focus coverage;
- call the frozen Qwen3-8B service once per eligible C2 sample;
- use temperature 0, top_p 1, max tokens 512 and zero retries;
- perform full-local validation, compilation, closure, V3 serialization and registry reconciliation;
- compute the unchanged operational sentinel;
- run the read-only RetrievalDomainMonotonicityAuditor and IntegrationFailureTaxonomist.

Any domain deletion, safety failure, schema drift, parse failure, grounding failure or endpoint retry stops the Goal.

## Phase 3: remaining C2 dev64

Only Integration16 PASS authorizes the remaining 48 C2 samples. C0/C1 remain replay-only. Close all 192 matched identities and compute unchanged Gate C V3.

## Required mechanism metrics

Report for each arm and by signature/table:

- authorized rows;
- executable rows and derivations;
- original-only, alternative-only and paired rows;
- registry-complete paired rows;
- paired tables;
- role-compatible precision;
- C2 paired gain and registry gain;
- C2 focus-hit derivations;
- C2 non-focus derivations admitted by the complete domain;
- C1 derivations retained or recovered in C2;
- focus-domain coverage ratio;
- prompt and completion tokens and latency.

## Gate and terminal policy

Do not change Gate C thresholds.

```text
Gate C V3 PASS
→ FREEZE_CERTA_ACTIVE_METHOD_READY_FOR_FINAL_DECISION_EXECUTION

Safety-clean but Gate C V3 failure
→ FREEZE_CERTA_ACTIVE_COVERAGE_PRESERVING_RETRIEVAL_VALID_NO_PAIRED_CONTRAST
```

A valid failure permanently stops Active V1 method development. Do not invent another component.

Decision, gold access, CERA, unblinding and holdout are forbidden in this Goal.
