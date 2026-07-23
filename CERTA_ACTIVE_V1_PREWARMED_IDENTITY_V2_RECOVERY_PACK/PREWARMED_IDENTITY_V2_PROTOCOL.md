# Prewarmed Identity V2 Recovery Protocol

## Diagnosis

The current service owns port 30338 and its process tree is valid. The blocked identity mixed the live PID/GPU fields with a cmdline SHA from an older service lineage and omitted all log identity fields.

## Recovery

1. Preserve the old blocked output root, old identity archive, diagnostic output and logs as read-only evidence.
2. Keep PID 1349993 running; do not signal or restart it.
3. Run `PREWARMED_IDENTITY_REFREEZE.py` once.
4. The script must verify the diagnostic facts, live PID/start ticks, semantic vLLM arguments, GPU, process-tree port ownership, live stdout/stderr log inode and three fresh readiness rounds.
5. It creates a new immutable V2 evidence directory, deterministic archive and external binding manifest.
6. Run `RUNTIME_CONTROLLER_ADOPT_V2.py bootstrap-adopt-existing-service-v2` under a new output root.
7. The V2 controller re-verifies the binding manifest, archive, identity, process, live log and readiness; writes controller state before terminal; runs frozen `freeze()` and `preflight()`; then advances the unchanged scientific DAG.

## Strict boundary

No service lifecycle action, scientific retry, CERTA source edit, Role V3 rerun, prompt/schema/threshold change, or Gate change is authorized.

The old `BLOCKED_PREWARMED_SERVICE_IDENTITY_FAILED` remains immutable and correct for the mixed-lineage identity attempt. The V2 execution uses a new lineage and output root.
