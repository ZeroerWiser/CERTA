# Read-only Sub-agent Protocol

The main Codex agent is the sole executor and writer outside `reviews/`. Sub-agents make no endpoint calls, do not edit the repository, and do not modify Gate inputs.

## 1. GateEntryPointRootCauseAuditor

Verify that the previous failure was caused by direct execution of `tools/compute_certa_active_constructor_gate_v3.py` before the repository root was available to Python imports. Compare the direct-script import order with the import-safe recovery. Confirm that no Gate code, input, threshold, or method object changed. Write `reviews/GATE_ENTRYPOINT_ROOT_CAUSE_AUDIT.md` with `PASS` or `FAIL`.

## 2. GateArtifactLineageAuditor

Trace every Gate input to the failed replay manifest. Verify 127 state rebuild records, 27 reconciled prior mismatches, the zero-access ledger, 128 V3 grounding records, 37 raw derivations, and 37 registry records, or report the exact observed counts if different. Confirm that the computed Gate is reproducible solely from frozen files. Write `reviews/GATE_C_V3_ARTIFACT_AUDIT.md`.

## 3. HostileAAAIConstructorReviewer

Review the Gate output as a skeptical AAAI reviewer. Distinguish software correctness from scientific sufficiency. Evaluate safety counts, C0/C1/C2 paired rows, registry-complete paired rows, paired tables, role-compatible precision, and unchanged thresholds. Do not propose threshold tuning or unrestricted engineering. Write `reviews/GATE_C_V3_HOSTILE_AAAI_AUDIT.md` with one verdict:

- `METHOD_READY_FOR_DECISION`, or
- `VALID_METHOD_NO_SUFFICIENT_PAIRED_CONTRAST`.

The verdict must match the machine Gate.