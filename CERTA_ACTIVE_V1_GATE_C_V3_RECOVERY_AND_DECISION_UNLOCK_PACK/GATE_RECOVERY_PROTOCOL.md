# Gate C V3 Recovery Protocol

## Scope

This protocol computes the unchanged repo-native Constructor Gate C V3 from checksum-valid artifacts already materialized by the zero-call assignment-authority replay. It does not rerun Role, Planner, grounding, closure, execution, registry construction, or any endpoint request.

## Mechanical sequence

1. Verify clean `research/certa-active-v1-grounding-authority-final@a6818af3c157f3416bdff84925e003e36b3c4583` and its remote ref.
2. Verify the seven changed source/test Git blobs.
3. Verify the immutable failed replay terminal, failure record, 538-entry checksum manifest, Git bundle, and all selected Gate inputs.
4. Require the offline access ledger to report zero endpoint, gold, sealed-label, Decision, CERA, and holdout access.
5. Insert the repository root into `sys.path` before importing `tools.compute_certa_active_constructor_gate_v3`.
6. Require the in-code Gate thresholds to equal the frozen binding.
7. Compute Gate C once and write it into a new recovery root.
8. Run three independent read-only audits.
9. Finalize the recovery root atomically.

## Interpretation

A PASS means the assignment-level authority method is ready for actual selected-final Decision execution. A valid failure means that the implemented method does not produce sufficient real-table paired executable contrast under the unchanged Gate. It is not authorization to tune thresholds, select a more favorable cohort, add a heuristic, or modify the method after seeing Gate metrics.

## Retry policy

There is no scientific retry. A purely mechanical failure before Gate creation may be corrected only when all source and replay hashes remain unchanged and the correction does not alter the Gate inputs or implementation. Once `CONSTRUCTOR_GATE_C_V3.json` exists, it is final.