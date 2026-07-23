# Terminal States

## Positive

```text
FREEZE_CERTA_ACTIVE_METHOD_READY_FOR_FINAL_DECISION_EXECUTION
```

Requirements: all tests pass, source limits pass, zero endpoint/gold/sealed access, all 27 mismatches accounted for, Gate C V3 PASS with unchanged thresholds, clean pushed branch and verified bundle.

## Negative — authority implementation invalid

```text
FREEZE_CERTA_ACTIVE_GROUNDING_AUTHORITY_REPLAY_FAILED
```

Use for safety mismatch, closure drift, unexplained reconciliation mismatch, ambiguous assignment authorization, registry-external derivation, threshold modification or test failure.

## Negative — valid correction but central contrast absent

```text
FREEZE_CERTA_ACTIVE_GROUNDING_AUTHORITY_VALID_NO_PAIRED_CONTRAST
```

Use when the authority implementation is mechanically valid and safe but unchanged Gate C still lacks sufficient paired executable contrast. Stop the Active V1 method line and narrow the paper claim; do not add another Planner, retriever or selector.
