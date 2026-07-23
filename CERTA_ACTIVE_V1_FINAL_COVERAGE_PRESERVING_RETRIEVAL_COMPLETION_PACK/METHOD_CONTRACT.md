# Coverage-Preserving Retrieval Guidance Contract

## Scientific diagnosis

For the same table, question and Role V3 record, C1 exposes the complete proposal-blind header reference domain. Current C2 destructively projects this domain onto `retrieval.reference_node_ids` and removes non-induced edges. The resulting C2 domain is therefore a strict and potentially disconnected subset of C1.

The accepted Gate evidence is:

- C0: 2 derivations, 0 paired rows;
- C1: 29 derivations, 1 paired row;
- C2: 6 derivations, 0 paired rows;
- all safety counters: zero;
- C2 paired gain and registry gain: -1.

This is a retrieval-recall failure, not a grounding-authority or registry-safety failure.

## Final method property

Define the complete Role-aligned reference domain as `D` and the retrieved focus subset as `F`.

```text
F ⊆ D
D_C2 = D_C1 = D
```

Retrieval is guidance, not authority:

```text
retrieval focus F
→ Planner search priority
→ full-domain typed validation over D
→ exact assignment-level grounding
→ deterministic execution and projection
→ registry authority
```

Retrieval must never:

- delete a schema node or structural edge from `D`;
- change the allowed Role V3 signature;
- authorize or reject a binding;
- use B0, gold, correctness or answer agreement;
- add an execution-time score or reranker;
- bypass full-local validation, closure, provenance or registry reconciliation.

## Required C2 Planner view

For a fixed table/question/Role record:

1. C1 and C2 `schema_nodes` must be byte-identical.
2. C1 and C2 `schema_edges` must be byte-identical.
3. C1 and C2 typed response schemas must be byte-identical.
4. C2 adds a closed `retrieval_focus` record containing only:
   - sorted unique `reference_node_ids`;
   - sorted unique `selected_card_ids`;
   - `mode = advisory_complete_domain`;
   - `reference_domain_complete = true`.
5. Every schema node may carry a deterministic boolean `retrieval_focused`; this annotation must not change its node identity or the response-schema enum.
6. The Planner instruction must state that focus references are preferred when sufficient, while non-focus references remain legal when needed to complete a role binding or enumerate finite structurally distinct alternatives.
7. Planner output must still be validated against the complete domain `D`.

## Contrast-complete enumeration rule

The Planner must not collapse multiple structurally plausible Role-compatible bindings merely because one lies in the retrieved focus. When finite alternatives exist, it may use `role_domains` to enumerate them. It must not invent alternatives, exceed the existing plan limit, or create answer-conditioned hypotheses.

## Scientific invariants

- proposal blindness;
- zero gold access before prediction close;
- exact Role V3 signature restriction;
- full-local typed validation;
- assignment-level exact grounding authority V3;
- first-match forbidden;
- provenance-complete registry;
- unchanged Gate C V3 thresholds;
- unchanged Decision policy and eligibility.
