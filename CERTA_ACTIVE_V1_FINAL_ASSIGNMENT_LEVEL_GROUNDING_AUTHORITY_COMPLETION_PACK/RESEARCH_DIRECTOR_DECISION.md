# CERTA Research Director Decision

```text
AUTHORIZE_CERTA_ACTIVE_V1_FINAL_ASSIGNMENT_LEVEL_GROUNDING_AUTHORITY_COMPLETION
SUPERSEDE_FREEZE_CERTA_ACTIVE_CONSTRUCTOR_FAILED_FOR_ASSIGNMENT_LEVEL_AUTHORITY_ONLY
```

This is the final bounded method-completion authorization before full Decision and experiment execution. It corrects one semantic inconsistency: a Planner role-domain contains multiple complete hypotheses, while the current artifact authority treats multiple uniquely grounded assignments as absence of one plan-level selected binding.

The correction must authorize each exact assignment-level grounding independently. It must never choose by first match, lexical score, confidence, gold, answer correctness or an arbitrary threshold.

Role V3, retrieval, Planner, schemas presented to the Planner, transport projection, operation contracts, structural resolvers, closure enumeration, executor, answer equivalence and Decision remain unchanged.
