# Runtime Lifecycle Policy

The external controller owns runtime only. It does not modify the CERTA repository.

```text
startup window = 900 seconds
poll interval = 10 seconds
per-GET timeout = 10 seconds
required readiness = 3 consecutive /health and /v1/models passes
expected model = Qwen3-8B
startup restart = at most one, only if the verified process exits before readiness
```

Readiness GETs are not scientific calls or retries. A live process that merely misses the 15-minute window is not restarted.

Scientific POSTs have zero automatic retries. Across M3, one exact replay is permitted only after mechanically proven EngineDead, service-process death, or connection loss; no usable assistant response; identical logical ID, request body, prompt, model, full/transport schema, sampling and token budget; and clean service restart. Parse, local validation, grounding, closure, execution, projection, provenance, length, answer, validator and Gate failures are never replayed.
