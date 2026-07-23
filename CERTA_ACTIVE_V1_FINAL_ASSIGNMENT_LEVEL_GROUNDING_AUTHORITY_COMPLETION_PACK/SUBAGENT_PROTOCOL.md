# Sub-agent Protocol

Use four bounded read-only sub-agents. The main Codex agent alone edits source, commits, runs replay and writes the terminal.

1. **GroundingAuthorityFormalAuditor**
   - prove the distinction between intra-assignment ambiguity and inter-assignment hypothesis multiplicity;
   - check that the proposed record and Gate preserve proposal blindness and no-gold inference.

2. **RealTableBindingFailureAnalyst**
   - inspect the two identified samples and every prior reconciliation mismatch;
   - produce assignment → operands → derivation → registry lineage maps;
   - do not propose lexical or score-based fixes.

3. **ArtifactGateReconciliationAuditor**
   - audit schema V3, serializer and Gate V3 field by field;
   - verify that every registry entry traces to an exact authorized binding and that no ambiguous assignment is authorized.

4. **HostileAAAIReviewer**
   - test whether the correction is a principled completion of the declared finite hypothesis space or a post-hoc relaxation;
   - identify claim boundaries if Gate C still fails.

Sub-agents do not call endpoints, write source, read gold/sealed labels or run in the background.
