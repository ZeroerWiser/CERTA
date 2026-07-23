# Implementation Boundary

Start from:

```text
research/certa-active-v1-grounding-authority-final
a6818af3c157f3416bdff84925e003e36b3c4583
```

Create:

```text
research/certa-active-v1-coverage-preserving-retrieval-final
```

## Production files permitted to change

1. `certa/active_v1/planner_bridge_v3.py`
2. `certa/planner/typed_planner.py`
3. `tools/certa_active_v1_completion.py`

A fourth production file is permitted only if a small shared canonical validator is necessary. It must be justified by two read-only auditors before editing.

## Test files permitted

- new or existing tests under `tests/active_v1/`;
- one real-artifact fixture containing hashes and structural counts, but no gold labels or answer text.

## Maximum scope

- at most 320 changed production lines;
- at most 600 changed test lines;
- one normal commit;
- no merge commits;
- no post-result source edit;
- no changes after the first scientific C2 request.

## Required implementation

### `planner_bridge_v3.py`

Replace destructive C2 filtering with a canonical advisory focus record. Preserve complete `schema_nodes` and `schema_edges`. Validate that every focus ID belongs to the complete domain. Add deterministic focus annotations without changing node identities.

### `typed_planner.py`

Conditionally add a concise retrieval-guidance instruction only when the Planner view contains `retrieval_focus.mode = advisory_complete_domain`. The instruction must preserve full-domain legality and finite alternative enumeration. C0 and C1 prompt bytes must remain unchanged for all frozen dev64 samples.

### runner

Create a new output root and execution surface that:

- verifies prior Gate and replay artifacts;
- replays C0 and C1 without endpoint calls;
- proves reconstructed C0/C1 Planner prompts and response schemas match their frozen records;
- runs only C2 Planner calls under the new view;
- performs Integration16 before the remaining dev64 C2 calls;
- serializes with accepted grounding authority V3;
- computes unchanged Gate C V3;
- records all calls, tokens, latency, prompt hashes and domain hashes.

## Frozen production files

Do not change:

- Role V3 and canonical registry;
- `certa/egra/retrieval.py`, its model, top-K and budgets;
- schema-view construction of the complete domain;
- Planner output schema grammar and plan limits;
- transport projection;
- operation contracts;
- compiler;
- structural resolvers;
- plan closure;
- executor and projection;
- answer equivalence;
- assignment-level grounding authority V3;
- registry reconciliation;
- Decision, CERA, validator and materializer;
- cohorts, sample IDs, Gates and thresholds.

## Forbidden substitutes

Do not add:

- fuzzy matching;
- lexical or embedding reranking after retrieval;
- score thresholds;
- answer-conditioned search;
- B0-aware planning;
- self-consistency sampling;
- a second Planner;
- a second Constructor;
- manual sample patches;
- operation-specific hard-coded IDs;
- new intervention families.
