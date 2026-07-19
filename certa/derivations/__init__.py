"""Typed executable derivation helpers for CERTA."""

from .answer_equivalence import inference_answer_key, inference_answers_equivalent
from .admissibility import build_admissible_candidate_set, check_candidate_contract
from .contrast import (
    CompactBehavioralContrastV3,
    build_compact_behavioral_contrast_v2,
    build_compact_behavioral_contrast_v3,
    build_minimal_contrast_set,
)
from .iade import (
    BasisRelativeBehaviorClass,
    HypothesisBehaviorVector,
    PairedInterventionObservation,
    RoleInterventionBasisItem,
    RoleInterventionObservation,
    build_basis_relative_behavior_classes,
    build_role_binding_substitution_pairs,
    build_sample_fixed_role_intervention_basis,
    evaluate_derivation_on_basis,
    iade_behavior_signatures,
)
from .lattice import build_derivation_lattice
from .materialize import materialize_derivations
from .pools import (
    AuditDerivationPool,
    DecisionDerivationPool,
    build_audit_derivation_pool,
    build_decision_derivation_pool,
)
from .frontier import build_symmetric_derivation_frontier
from .project import answers_equivalent, canonical_answer_key
from .replay import replay_derivation_under_intervention
from .support import reconstruct_original_support_hypotheses
from .support_symmetry import build_original_support_symmetry_v3
from .schema import (
    AdmissibleCandidateSet,
    CandidateAdmissibilityResult,
    CandidateContractCheck,
    ExecutableDerivation,
    OriginalSupportHypothesis,
    OriginalSupportHypothesisSet,
    PreEvidenceQueryContract,
    ReplayResult,
)

__all__ = [
    "AdmissibleCandidateSet",
    "AuditDerivationPool",
    "CandidateAdmissibilityResult",
    "CandidateContractCheck",
    "CompactBehavioralContrastV3",
    "DecisionDerivationPool",
    "ExecutableDerivation",
    "OriginalSupportHypothesis",
    "OriginalSupportHypothesisSet",
    "BasisRelativeBehaviorClass",
    "HypothesisBehaviorVector",
    "PairedInterventionObservation",
    "PreEvidenceQueryContract",
    "ReplayResult",
    "RoleInterventionBasisItem",
    "RoleInterventionObservation",
    "answers_equivalent",
    "build_admissible_candidate_set",
    "build_audit_derivation_pool",
    "build_basis_relative_behavior_classes",
    "build_compact_behavioral_contrast_v2",
    "build_compact_behavioral_contrast_v3",
    "build_decision_derivation_pool",
    "build_derivation_lattice",
    "build_minimal_contrast_set",
    "build_original_support_symmetry_v3",
    "build_role_binding_substitution_pairs",
    "build_sample_fixed_role_intervention_basis",
    "build_symmetric_derivation_frontier",
    "canonical_answer_key",
    "check_candidate_contract",
    "evaluate_derivation_on_basis",
    "iade_behavior_signatures",
    "inference_answer_key",
    "inference_answers_equivalent",
    "materialize_derivations",
    "reconstruct_original_support_hypotheses",
    "replay_derivation_under_intervention",
]
