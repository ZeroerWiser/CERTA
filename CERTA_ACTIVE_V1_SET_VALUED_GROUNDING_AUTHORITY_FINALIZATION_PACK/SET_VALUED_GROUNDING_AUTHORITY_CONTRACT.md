# Set-Valued Grounding Authority Contract

## Diagnosis

`PlanClosure` constructs a finite product of declared role domains, executes each individually resolved assignment, and deduplicates by canonical program. Multiple executable assignments are therefore intended hypotheses, not evidence that each assignment is internally ambiguous.

The current artifact layer marks each assignment valid when its resolver state is `UNIQUE`, but records a selected binding only when the complete plan has exactly one valid binding. This singleton rule conflicts with executable hypothesis contrast and suppresses multi-hypothesis registry authority.

## Correct authority

For each raw grounding record:

```text
authorized_binding_ids
= sorted(binding_id for candidate in grounding_candidates
         if candidate.valid is true)
```

A binding is authorized only when its own structural resolver was UNIQUE and it has nonempty operands. The authority is a finite set, not a ranked choice.

Compatibility rule:

```text
selected_binding_id = the sole authorized ID when cardinality == 1
selected_binding_id = null otherwise
```

A derivation is grounding-authorized iff:

```text
derivation.binding_id in authorized_binding_ids
```

## Invariants

1. No first-match or lexical ranking.
2. No confidence score, distance threshold, majority vote or gold signal.
3. Resolver-level AMBIGUOUS, UNRESOLVED, INVALID and RESOURCE_INCOMPLETE assignments remain unauthorized.
4. Existing singleton rows remain byte-equivalent except for additive schema fields.
5. Every authorized derivation retains its exact canonical program, operands, projected answer and provenance.
6. Multiple derivations with different programs, answers or provenance remain separate registry hypotheses.
7. No answer-equivalence collapse is used to manufacture uniqueness.
8. Resource ceilings and complete enumeration remain unchanged.

## Schema evolution

Use a new versioned grounding record schema. Required additive fields:

```text
authorization_mode = FINITE_SET_OF_INDIVIDUALLY_UNIQUE_BINDINGS
authorized_binding_ids = sorted unique array
```

Do not reinterpret old v2 records in place. The old terminal and old records remain immutable.
