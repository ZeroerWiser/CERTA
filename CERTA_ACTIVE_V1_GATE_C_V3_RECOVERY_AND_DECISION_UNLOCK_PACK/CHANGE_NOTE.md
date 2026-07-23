# Change Note

This Pack changes no CERTA method source. It corrects only the execution mode used to reach the already implemented Gate C V3:

```text
invalid: python tools/compute_certa_active_constructor_gate_v3.py ...
valid:   insert repository root, import compute_gate, evaluate frozen files
```

The Gate implementation, thresholds, replay inputs, method commit, cohort, and zero-access boundary remain unchanged. The failed replay root and terminal remain immutable.