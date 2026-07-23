# Assignment-Level Grounding Authority Contract

## Invariant A — exact assignment grounding

For one complete role assignment, zero matching operand structures is `UNRESOLVED`; more than one matching structure for a required atomic conjunction is `AMBIGUOUS`; exactly one is `EXACT`.

## Invariant B — hypothesis multiplicity is not ambiguity

Different role-domain assignments may each be `EXACT`. They remain separate grounded hypotheses and may produce different executable answers. No pre-Decision component selects one of them.

## Invariant C — binding identity

Every exact hypothesis has an immutable `binding_id` derived from sample, table, arm, role record, plan, assignment identity and assignment key. Every raw derivation references exactly one authorized binding.

## Invariant D — registry closure

A registry entry is valid only if its raw derivation is executed, projected, provenance-complete and references an authorized exact assignment-level binding. A plan-level singleton is not required.

## Invariant E — no heuristic selection

`first_match_used=false`. No lexical similarity, embedding score, model confidence, answer agreement, gold label or arbitrary ranking may select a binding.

## Invariant F — boundedness

Existing finite role domains, exact reference IDs, operation contracts, resource limits and canonical-program deduplication remain authoritative.

## Required grounding record V3

A plan-level record may contain multiple `grounding_hypotheses`. Each hypothesis records at minimum:

```text
binding_id
assignment_id
assignment_key
role_bindings_sha256
operand_node_ids
resolution_state
grounding_valid
derivation_id
canonical_program_id
failure_reasons
```

The plan record contains sorted `authorized_binding_ids`, rejected binding IDs, ambiguity counts and `first_match_used=false`. `selected_binding_id` is not an authority field.
