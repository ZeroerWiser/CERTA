"""Public typed reasoning-trace contracts for CERTA."""

from .typed_trace import (
    TRACE_VERSION,
    FirstVerifiableFailure,
    IntentHypothesis,
    RoleBindingStep,
    TraceVerificationStep,
    TypedExecutableReasoningTrace,
    VerificationStage,
    VerificationStatus,
    build_intent_prompt,
    build_intent_response_schema,
    build_role_binding_prompt,
    build_role_binding_response_schema,
    build_typed_executable_traces,
    build_validation_failure_records,
    first_verifiable_failure,
    validate_intent_output,
    validate_role_binding_output,
)
from .minimal_patch import PATCH_VERSION, build_minimal_structural_patch_registry

__all__ = [
    "TRACE_VERSION",
    "FirstVerifiableFailure",
    "IntentHypothesis",
    "RoleBindingStep",
    "TraceVerificationStep",
    "TypedExecutableReasoningTrace",
    "VerificationStage",
    "VerificationStatus",
    "build_intent_prompt",
    "build_intent_response_schema",
    "build_role_binding_prompt",
    "build_role_binding_response_schema",
    "build_typed_executable_traces",
    "build_validation_failure_records",
    "first_verifiable_failure",
    "validate_intent_output",
    "validate_role_binding_output",
    "PATCH_VERSION",
    "build_minimal_structural_patch_registry",
]
