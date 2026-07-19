"""Plan-conditioned grounding helpers for CERTA Round 9."""

from .plan_closure import ClosureOutcome, GroundedAssignment, PlanClosure, build_plan_closure
from .support_partition import SupportPartition, partition_support
from .structural_resolvers import (
    AtomicResolution,
    EntityMeasureRelationResolution,
    EntityValueMember,
    ResolutionState,
    ScopeMemberProvenance,
    ScopeResolution,
    resolve_atomic_operand,
    resolve_entity_measure_relation,
    resolve_finite_scope,
)

__all__ = [
    "ClosureOutcome",
    "GroundedAssignment",
    "PlanClosure",
    "SupportPartition",
    "AtomicResolution",
    "EntityMeasureRelationResolution",
    "EntityValueMember",
    "ResolutionState",
    "ScopeMemberProvenance",
    "ScopeResolution",
    "build_plan_closure",
    "partition_support",
    "resolve_atomic_operand",
    "resolve_entity_measure_relation",
    "resolve_finite_scope",
]
