# Blind Decision Protocol

After Gate C PASS, freeze Constructor, eligibility, exact request set, CERA schema/model/sampling, validator and deterministic materializer before dev gold access.

Execute B0_KEEP, DETERMINISTIC_SELECTOR and CERA_PLUS_VALIDATOR. Only decision-active, role-compatible, registry-complete paired rows may call CERA. Decision-inactive and ineligible rows make zero CERA calls and keep B0.

Every changed answer must reference an executed registry entry, pass the frozen validator, reconcile all IDs and hashes, and be emitted by the deterministic materializer. Freeze selected-final before one dev unblind.

`DecisionSafetyAuditor` verifies eligibility, CERA calls, registry references, validator, materializer, actual selected-final and CC/CW/WC/WW.
