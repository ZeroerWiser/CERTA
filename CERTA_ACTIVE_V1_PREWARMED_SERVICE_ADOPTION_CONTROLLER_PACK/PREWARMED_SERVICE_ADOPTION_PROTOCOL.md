# Prewarmed Service Adoption Protocol

The controller adopts an already-running user-managed vLLM process. It has no authority to start, stop, restart, signal, replace, or reconfigure that process.

## Mechanical adoption

Before any scientific action, the controller:

1. verifies the frozen Git branch, remote and method blobs;
2. verifies the evidence archive byte hash and reads the exact `PREWARMED_VLLM_SERVICE_IDENTITY.json`;
3. verifies PID, `/proc` start ticks, executable, normalized command line and command hash;
4. verifies `CUDA_VISIBLE_DEVICES=3` from the process environment;
5. verifies that the PID owns `127.0.0.1:30338`;
6. verifies model root, served model, `max_model_len=32768`, and the identity-bound regular readable log file, device and inode;
7. executes three fresh `/health` and `/v1/models` rounds with a 10-second per-request timeout;
8. accepts HTTP 200 with an empty `/health` body;
9. requires `/v1/models` to contain exactly one model, `Qwen3-8B`;
10. creates a new output root, writes `RUNTIME_CONTROLLER_STATE.json`, imports the frozen runner, overrides only its external `OUT`, invokes `freeze()`, and runs frozen `preflight()` once.

No cold-start timer or startup terminal applies. Identity/readiness failure stops as an adoption failure and never triggers service lifecycle operations.

## Scientific execution

After adoption PASS, execution proceeds only through the frozen runner stages:

```text
Integration16
→ sample/arm failure taxonomy
→ matched dev64 C0/C1/C2
→ Gate C
→ actual selected-final Decision
→ prediction close
→ single dev unblind: Gate O + CC/CW/WC/WW
→ permanent method freeze
→ blind table-disjoint holdout
→ holdout unblind
```

Scientific POST calls retain zero retries. The adoption controller performs no exact transport replay because Codex has no service-lifecycle authority in this Pack. On process loss, connection loss, identity drift, or unavailable service, it preserves the checkpoint and stops.
