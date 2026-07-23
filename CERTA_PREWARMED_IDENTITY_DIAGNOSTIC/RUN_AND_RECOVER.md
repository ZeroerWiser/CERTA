# CERTA Prewarmed Identity Diagnosis and Recovery

This procedure is read-only with respect to the vLLM service and CERTA method.

## 1. Preserve the blocked output

```bash
export OLD_ROOT=/home/hsh/ME/Table/EMNLP2026/certa_active_v1_outputs/CERTA_ACTIVE_V1_PREWARMED_ADOPTION_FINAL_SCIENTIFIC_DAG
export DIAG_ROOT=/home/hsh/ME/Table/EMNLP2026/certa_runtime_evidence/PREWARMED_IDENTITY_DIAGNOSTIC_20260723
mkdir -p "$DIAG_ROOT"
cp -a "$OLD_ROOT/runtime/RUNTIME_CONTROLLER_STATE.json" "$DIAG_ROOT/" 2>/dev/null || true
cp -a "$OLD_ROOT/terminal/FINAL_TERMINAL_STATE.json" "$DIAG_ROOT/" 2>/dev/null || true
cp -a "$OLD_ROOT/terminal/TERMINAL_REPORT.md" "$DIAG_ROOT/" 2>/dev/null || true
```

## 2. Run the read-only diagnostic

```bash
export CERTA_ACTIVE_PYTHON=/home/hsh/anaconda3/envs/table-cu128/bin/python
export ARCHIVE=/home/hsh/ME/Table/EMNLP2026/certa_runtime_evidence/QWEN3_8B_PREWARMED_SERVICE_20260723.tar.gz
export IDENTITY=/home/hsh/ME/Table/EMNLP2026/certa_runtime_evidence/QWEN3_8B_PREWARMED_SERVICE_20260723/PREWARMED_VLLM_SERVICE_IDENTITY.json

"$CERTA_ACTIVE_PYTHON" PREWARMED_IDENTITY_DIAGNOSTIC.py \
  --archive "$ARCHIVE" \
  --identity "$IDENTITY" \
  --output "$DIAG_ROOT/PREWARMED_IDENTITY_DIAGNOSTIC.json"
```

Do not restart, stop, signal, or modify PID 1349993 while diagnosing.

## 3. Interpret the first critical failure

- `archive_sha256` or `identity_byte_identical_to_archive`: redeploy the exact original archive and extract it again; do not edit the JSON manually.
- `pid_exists` or `process_start_ticks`: the frozen process no longer exists or PID was reused. Freeze a new identity from the currently running verified process and use a new evidence archive and output root.
- `cmdline_exact` only, while all required cmdline tokens pass: controller comparison is overly lexical. Authorize a controller-only semantic cmdline comparison; do not modify the service.
- `root_pid_directly_owns_listening_port` fails but `root_or_descendant_owns_listening_port` passes: the listener belongs to a verified descendant. Amend the controller to accept the frozen process tree, not only the parent PID.
- `root_or_descendant_owns_listening_port` fails: the frozen PID/process tree does not own port 30338. Do not adopt it.
- `CUDA_VISIBLE_DEVICES`, model tokens, model endpoint, or max-model-length fails: wrong service identity. Do not continue.
- `log_device` or `log_inode` fails while the same live log path remains readable: the identity bundle bound a copied/rotated log inode. Re-freeze the current live log identity without restarting the service, using a new evidence archive and output root.
- readiness fails while identity passes: preserve service evidence and stop; this is a runtime availability failure, not a scientific result.

## 4. Required return to Research Director

Return only:

```text
PREWARMED_IDENTITY_DIAGNOSTIC.json
current PREWARMED_VLLM_SERVICE_IDENTITY.json
sha256sum of the evidence archive and identity JSON
ls -li of the live log path
ss -ltnp output for port 30338
ps -o pid,ppid,lstart,cmd for PID 1349993 and its children
```

The original blocked terminal remains immutable. Any reexecution uses a new output root.
