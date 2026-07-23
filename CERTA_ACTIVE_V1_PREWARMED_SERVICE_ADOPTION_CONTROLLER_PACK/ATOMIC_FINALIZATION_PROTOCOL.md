# Atomic Finalization Protocol

The adoption controller writes controller state before any terminal. Finalization is external to the CERTA method and changes no scientific decision.

For any infrastructure or scientific terminal, the controller creates a same-filesystem staging directory and writes:

```text
FINAL_TERMINAL_STATE.json
FINAL_METHOD_FREEZE_MANIFEST.json
REQUIRED_ARTIFACTS.json
COST_LEDGER.json
TERMINAL_REPORT.md
CERTA_ACTIVE_V1_FINAL_COMPLETION.bundle
SHA256SUMS.txt
```

It verifies clean Git state, creates and verifies the Git bundle, records every required artifact as `PRESENT`, `NOT_REACHED`, or `MISSING_REQUIRED`, hashes all output files except the checksum manifest itself, fsyncs files and directories, and atomically renames the staging directory to `terminal/`.

An existing terminal directory is never overwritten. A finalization error preserves the staging directory and raises `BLOCKED_ATOMIC_FINALIZATION_FAILED`; it does not delete prior artifacts.
