# Read-only Sub-agent Protocol

The main Codex agent is the sole editor, endpoint caller, committer and terminal writer. Sub-agents are read-only and may write only their named reports under the new output root.

## 1. RetrievalProjectionFailureAnalyst

Output: `audits/01_RETRIEVAL_PROJECTION_FAILURE_ANALYSIS.md`

Reconstruct prior C1 and C2 artifacts for all 64 samples. Identify:

- the C1 paired sample;
- every C1 executable derivation absent from C2;
- whether each loss follows from deleted schema nodes, deleted structural edges, Planner-output narrowing, or later closure;
- the six surviving C2 derivations and their focus membership;
- whether the evidence supports the destructive-projection diagnosis.

No gold, answer correctness or new endpoint calls.

## 2. RetrievalDomainMonotonicityAuditor

Output: `audits/02_RETRIEVAL_DOMAIN_MONOTONICITY_AUDIT.md`

Before calls, verify on all dev64 samples:

- C1 and corrected C2 node domains are identical;
- C1 and corrected C2 edge domains are identical;
- typed response schemas are identical;
- focus IDs are sorted, unique and within the domain;
- C0/C1 prompts remain byte-identical to prior prompts;
- C2 adds only advisory focus metadata;
- no proposal, B0, gold, score or answer field enters the focus record.

Return PASS/FAIL with exact hashes and first mismatch.

## 3. ContrastCompletenessAuditor

Output: `audits/03_CONTRAST_COMPLETENESS_AUDIT.md`

After execution, audit each C2 sample for:

- number of Planner plans and finite role alternatives;
- exact authorized assignments;
- executed original and alternative hypotheses;
- focus and non-focus provenance;
- registry completeness;
- the first stage preventing pairing.

Distinguish retrieval focus failure, Planner enumeration failure, grounding failure, execution equivalence and absence of a valid original/alternative opportunity.

## 4. HostileAAAIMethodAuditor

Output: `audits/04_HOSTILE_AAAI_METHOD_AUDIT.md`

Evaluate whether the final method is a coherent research contribution rather than accumulated engineering. Check:

- explicit problem formulation;
- reference-domain monotonicity;
- separation of retrieval guidance from grounding authority;
- no answer-conditioned heuristic;
- falsifiable mechanism Gate;
- adequate evidence for moving to Decision;
- whether any claim must be narrowed.

The auditor may recommend STOP but may not propose or implement another module.
