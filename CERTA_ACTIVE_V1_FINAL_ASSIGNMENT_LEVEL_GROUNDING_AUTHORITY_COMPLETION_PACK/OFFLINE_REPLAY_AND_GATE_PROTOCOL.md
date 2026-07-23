# Offline Replay and Gate Protocol

The previous dev64 Planner calls and normalized payloads are frozen evidence. The final authority correction must be evaluated without new endpoint calls.

## Replay inputs

Use the prior output root's runtime identities, B0, Role V3 records, `constructor/STATE/*.json`, Planner raw requests/responses, full-local validation, graphs reconstructed from the unchanged tables and normalized Planner payloads.

For every replayed state:

1. verify source manifest and file hashes;
2. verify normalized payload hash against the frozen state;
3. rebuild graph and closure under unchanged frozen code;
4. require rebuilt closure SHA to equal the prior closure SHA;
5. serialize only through grounding authority V3;
6. compute the repo-native Constructor Gate V3.

No endpoint, gold, sealed label or answer-based inspection is allowed.

## Gate thresholds — unchanged

```text
C2 paired executable rows >= 8
C2 paired gain over max(C0,C1) >= 4
C2 registry-complete paired rows >= 6
C2 registry gain over max(C0,C1) >= 3
paired evidence spans >= 4 tables
executable/registry reconciliation precision = 1.0
all safety counters = 0
```

## Required analysis

Account mechanically for all 27 prior grounding-authority mismatches. Each must become either an exact authorized binding with reconciled derivation or a precise fail-closed reason. No unexplained mismatch is permitted.

Report coverage by sample, arm, Role signature, operation family, table and failure stage. Report intra-assignment ambiguity separately from inter-assignment multiplicity.

## Positive terminal

`FREEZE_CERTA_ACTIVE_METHOD_READY_FOR_FINAL_DECISION_EXECUTION`

This unlocks a later execution-only Goal for actual Decision, dev unblind and holdout.
