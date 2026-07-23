# Audit and Diagnosis

## Verified result

The V2R2 run reached the scientific method plane. Integration16 passed. Constructor64 failed Gate C with 192 identities, 127 valid Planner rows, 127 grounded rows, 13 unique bindings, 12 raw executed derivations, 10 provenance/registry-complete derivations and zero paired executable contrast.

All 27 reported reconciliation mismatches were grounding-authority mismatches. Two samples contained many exact valid assignments but no plan-level selected binding:

- `24fad1e0d3751475d5673e86e2c3a6ea`: 1,188 candidates, 38 valid, 22 derivation exclusions.
- `84eab744af0915c0e4deb855efe0321d`: 539 candidates, 21 valid, 5 derivation exclusions.

No side, answer, provenance or registry-content mismatch was reported.

## Root cause

The closure correctly enumerates role-domain assignments and requires every assignment to resolve exactly. The serializer then groups assignments by plan and writes `selected_binding_id` only when the plan has exactly one valid assignment. This collapses two distinct concepts:

1. intra-assignment ambiguity: one declared structural conjunction resolves to multiple cells — invalid and fail-closed;
2. inter-assignment multiplicity: multiple different complete assignments each resolve uniquely — these are the finite executable hypotheses CERTA needs for original/alternative contrast.

The current Gate reconciliation incorrectly rejects case 2.

## Scientific implication

The central CERTA contrast claim is not yet established. The current engineering stack is substantially complete, but actual selected-final accuracy, WC/CW, harmful-revision reduction and holdout evidence remain unmeasured. This authorization repairs only the missing hypothesis authority needed to make those measurements possible.
