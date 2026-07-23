# Scientific Failure Taxonomy

Infrastructure terminals:

```text
BLOCKED_CERTA_ACTIVE_RUNTIME_STABLE_INVALID_START
BLOCKED_RUNTIME_READINESS_FAILED
BLOCKED_RUNTIME_PROCESS_DIED
BLOCKED_RUNTIME_CHECKPOINT_NOT_RESUMABLE
```

Scientific terminals:

```text
FREEZE_CERTA_ACTIVE_INTEGRATION_FAILED
FREEZE_CERTA_ACTIVE_CONSTRUCTOR_FAILED
FREEZE_CERTA_ACTIVE_CONSTRUCTOR_VALID_NO_OPPORTUNITY
FREEZE_CERTA_ACTIVE_DECISION_FAILED
FREEZE_CERTA_ACTIVE_HOLDOUT_FAILED
FREEZE_CERTA_ACTIVE_METHOD_EXPERIMENT_ONLY
```

Sample/arm failure stages:

```text
transport
parse
full-local validation
Planner semantics
grounding
closure
execution
projection
provenance
registry
```

Infrastructure recovery never changes a scientific classification. At the first scientific failure, stop permanently and preserve raw artifacts.
