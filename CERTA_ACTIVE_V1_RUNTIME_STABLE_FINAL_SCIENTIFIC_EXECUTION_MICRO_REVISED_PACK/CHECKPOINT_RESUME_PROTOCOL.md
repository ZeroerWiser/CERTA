# Checkpoint Resume Protocol

A completed logical call has checksum-valid request, usable response, endpoint-ledger entry, schema identities where applicable, and full-local validation where applicable.

Completed logical calls are never regenerated. The frozen runner resumes from missing sample/arm artifacts. The controller snapshots the checkpoint before and after every stage and rejects duplicate successful logical calls.

An authorized transport replay keeps the same controller-level `logical_call_id` and records `attempt_index=1` and `attempt_index=2`. The controller verifies canonical request-body equality after replay. If a usable response exists but the frozen runner cannot consume it without regeneration, stop as `BLOCKED_RUNTIME_CHECKPOINT_NOT_RESUMABLE`.
