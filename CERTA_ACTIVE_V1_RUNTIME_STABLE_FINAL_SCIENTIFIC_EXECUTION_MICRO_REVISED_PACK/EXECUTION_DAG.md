# Execution DAG

```text
validate_pack + controller intake
  → bounded readiness
  → frozen preflight
  → Integration16 close
  → IntegrationFailureTaxonomist
  → matched dev64 C0/C1/C2
  → raw-artifact Gate C
  → PerSignatureConstructorAuditor
  → blind actual selected-final Decision
  → selected-final prediction close
  → DecisionSafetyAuditor
  → single dev unblind: Gate O + CC/CW/WC/WW
  → AbstractClaimAuditor
  → permanent method freeze
  → blind table-disjoint holdout
  → holdout unblind
```

No backward method-edit edge exists. Intake/readiness should consume at most 5% of execution attention; Integration16/taxonomy 35%; matched constructor/Gate C 30%; Decision/selected-final 20%; unblind/holdout/statistics/audit 10%.
