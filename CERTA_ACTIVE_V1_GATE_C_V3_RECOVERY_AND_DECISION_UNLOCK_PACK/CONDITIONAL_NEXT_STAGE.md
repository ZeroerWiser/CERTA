# Conditional Next Stage

## Gate PASS

The only positive terminal is:

```text
FREEZE_CERTA_ACTIVE_METHOD_READY_FOR_FINAL_DECISION_EXECUTION
```

It unlocks one execution-only Pack bound to commit `a6818af3c157f3416bdff84925e003e36b3c4583` and the exact recovered Gate hash. That later Pack will:

1. construct a new final-execution root from the V3 replay artifacts and checksum-valid prior Planner states;
2. adopt the user-managed Qwen3-8B service without changing its lifecycle;
3. run actual blind Decision and selected-final prediction close;
4. perform one dev unblind for Gate O, CC/CW/WC/WW, statistics, and costs;
5. proceed to table-disjoint holdout only after positive dev Gates;
6. freeze the method permanently before the full experiment matrix.

No method source change is permitted after Gate PASS.

## Gate valid failure

The terminal is:

```text
FREEZE_CERTA_ACTIVE_GROUNDING_AUTHORITY_VALID_NO_PAIRED_CONTRAST
```

Decision, gold access, and holdout remain forbidden. The next Research Director action is claim narrowing or an independently justified new method decision—not a retry, threshold change, cohort selection, or incremental engineering loop.

## Full experiment matrix

The full HiTab/AIT-QA × Qwen3/Qwen2.5/Llama matrix is authorized only after actual Decision and holdout pass. Software existence, fixture coverage, oracle opportunity, or Gate recovery alone cannot unlock it.