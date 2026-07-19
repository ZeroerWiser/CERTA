"""Canonical semantic operation signatures for CERTA."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


OPERATION_CONTRACT_VERSION = "certa_operation_signature_registry_v2"
COMPARISON_POLARITIES = ("greater", "greater_equal", "less", "less_equal", "equal")
FINAL_SUPPORTED_OPERATIONS = (
    "LOOKUP",
    "SUM",
    "AVERAGE",
    "COUNT",
    "DIFF",
    "RATIO",
    "ARGMAX",
    "ARGMIN",
    "PAIR_COMPARE",
)


@dataclass(frozen=True)
class RoleContract:
    name: str
    shape: str
    min_items: int = 1
    max_items: Optional[int] = None
    ordered: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "shape": self.shape,
            "min_items": self.min_items,
            "max_items": self.max_items,
            "ordered": self.ordered,
        }


@dataclass(frozen=True)
class OperationSignatureVariant:
    signature_id: str
    operation_family: str
    semantic_result_role: str
    required_roles: Tuple[RoleContract, ...]
    optional_roles: Tuple[RoleContract, ...]
    projection_operator: str
    answer_domain: str
    resolution_mode: str
    execution_family: str
    ordered_semantics: str = "order_invariant"
    comparison_polarities: Tuple[str, ...] = ()
    tie_semantics: str = "not_applicable"
    projection_source: str = "executed_value"

    @property
    def allowed_roles(self) -> Tuple[RoleContract, ...]:
        return self.required_roles + self.optional_roles

    @property
    def role_names(self) -> Tuple[str, ...]:
        return tuple(role.name for role in self.allowed_roles)

    @property
    def required_role_names(self) -> Tuple[str, ...]:
        return tuple(role.name for role in self.required_roles)

    @property
    def projection_domain_pairs(self) -> Tuple[Tuple[str, str], ...]:
        return ((self.projection_operator, self.answer_domain),)

    @property
    def ordered(self) -> bool:
        return self.ordered_semantics == "ordered_operands"

    def role(self, name: str) -> Optional[RoleContract]:
        return next((role for role in self.allowed_roles if role.name == name), None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_version": OPERATION_CONTRACT_VERSION,
            "signature_id": self.signature_id,
            "operation_family": self.operation_family,
            "semantic_result_role": self.semantic_result_role,
            "required_roles": [role.to_dict() for role in self.required_roles],
            "optional_roles": [role.to_dict() for role in self.optional_roles],
            "forbidden_roles": sorted(set(ROLE_VOCABULARY) - set(self.role_names)),
            "projection_operator": self.projection_operator,
            "answer_domain": self.answer_domain,
            "resolution_mode": self.resolution_mode,
            "execution_family": self.execution_family,
            "ordered_semantics": self.ordered_semantics,
            "comparison_polarities": list(self.comparison_polarities),
            "tie_semantics": self.tie_semantics,
            "projection_source": self.projection_source,
        }


# Compatibility type name. Active Round 12 authority is signature-level.
OperationContract = OperationSignatureVariant


@dataclass(frozen=True)
class ExcludedOperationSignature:
    signature_id: str
    operation_family: str
    semantic_result_role: str
    reason: str


@dataclass(frozen=True)
class OperationContractValidation:
    ok: bool
    operation_family: str
    signature_id: str = ""
    errors: Tuple[str, ...] = ()
    normalized_role_bindings: Tuple[Tuple[str, Any], ...] = ()
    normalized_role_domains: Tuple[Tuple[str, Tuple[Any, ...]], ...] = ()

    def role_bindings_dict(self) -> Dict[str, Any]:
        return dict(self.normalized_role_bindings)

    def role_domains_dict(self) -> Dict[str, Tuple[Any, ...]]:
        return dict(self.normalized_role_domains)


def _role(
    name: str,
    shape: str = "structural_conjunction",
    *,
    min_items: int = 1,
    max_items: Optional[int] = None,
    ordered: bool = False,
) -> RoleContract:
    return RoleContract(
        name=name,
        shape=shape,
        min_items=min_items,
        max_items=max_items,
        ordered=ordered,
    )


_ENTITY_SCOPE = _role("AGGREGATION_SCOPE", "finite_scope_members")
_MEASURE = _role("TARGET_MEASURE")
_GROUP = _role("GROUP_SCOPE")
_TIME = _role("TIME_SCOPE")
_LEFT = _role("LEFT_OPERAND")
_RIGHT = _role("RIGHT_OPERAND")


def _signature(
    signature_id: str,
    operation_family: str,
    semantic_result_role: str,
    required_roles: Tuple[RoleContract, ...],
    optional_roles: Tuple[RoleContract, ...],
    projection_operator: str,
    answer_domain: str,
    resolution_mode: str,
    execution_family: str,
    **kwargs: Any,
) -> OperationSignatureVariant:
    return OperationSignatureVariant(
        signature_id=signature_id,
        operation_family=operation_family,
        semantic_result_role=semantic_result_role,
        required_roles=required_roles,
        optional_roles=optional_roles,
        projection_operator=projection_operator,
        answer_domain=answer_domain,
        resolution_mode=resolution_mode,
        execution_family=execution_family,
        **kwargs,
    )


OPERATION_SIGNATURES: Dict[str, OperationSignatureVariant] = {
    item.signature_id: item
    for item in (
        _signature("LOOKUP_VALUE_SCALAR", "LOOKUP", "VALUE", (_role("TARGET_ENTITY"), _MEASURE), (_TIME,), "VALUE_PROJECTION", "SCALAR", "atomic_lookup", "lookup_value"),
        _signature("LOOKUP_VALUE_ENTITY", "LOOKUP", "VALUE", (_role("TARGET_ENTITY"), _MEASURE), (_TIME,), "VALUE_PROJECTION", "ENTITY", "atomic_lookup", "lookup_value"),
        _signature("LOOKUP_VALUE_BOOLEAN", "LOOKUP", "VALUE", (_role("TARGET_ENTITY"), _MEASURE), (_TIME,), "VALUE_PROJECTION", "BOOLEAN", "atomic_lookup", "lookup_value"),
        _signature("SUM_SCALAR", "SUM", "VALUE", (_ENTITY_SCOPE, _MEASURE), (_GROUP, _TIME), "SCALAR_RESULT_PROJECTION", "SCALAR", "finite_numeric_scope", "sum_scope"),
        _signature("AVERAGE_SCALAR", "AVERAGE", "VALUE", (_ENTITY_SCOPE, _MEASURE), (_GROUP, _TIME), "SCALAR_RESULT_PROJECTION", "SCALAR", "finite_numeric_scope", "average_scope"),
        _signature("COUNT_SCALAR", "COUNT", "CARDINALITY", (_ENTITY_SCOPE, _MEASURE), (_GROUP, _TIME), "SCALAR_RESULT_PROJECTION", "SCALAR", "finite_scope", "count_scope"),
        _signature("DIFF_SCALAR", "DIFF", "VALUE", (_LEFT, _RIGHT), (), "SCALAR_RESULT_PROJECTION", "SCALAR", "ordered_atomic_operands", "difference", ordered_semantics="ordered_operands"),
        _signature("RATIO_SCALAR", "RATIO", "VALUE", (_LEFT, _RIGHT), (), "SCALAR_RESULT_PROJECTION", "SCALAR", "ordered_atomic_operands", "ratio", ordered_semantics="ordered_operands"),
        _signature("ARGMAX_VALUE", "ARGMAX", "VALUE", (_ENTITY_SCOPE, _MEASURE), (_GROUP, _TIME), "VALUE_PROJECTION", "SCALAR", "entity_value_relation", "argmax_relation", tie_semantics="equal_values_collapse", projection_source="selected_value_nodes"),
        _signature("ARGMAX_ENTITY", "ARGMAX", "ENTITY", (_ENTITY_SCOPE, _MEASURE), (_GROUP, _TIME), "ROW_ENTITY_PROJECTION", "ENTITY", "entity_value_relation", "argmax_relation", tie_semantics="unique_only", projection_source="relation_entity_binding_ids"),
        _signature("ARGMAX_ENTITY_SET", "ARGMAX", "ENTITY_SET", (_ENTITY_SCOPE, _MEASURE), (_GROUP, _TIME), "ROW_ENTITY_PROJECTION", "SET", "entity_value_relation", "argmax_relation", tie_semantics="multiple_only_canonical_set", projection_source="relation_entity_binding_ids"),
        _signature("ARGMIN_VALUE", "ARGMIN", "VALUE", (_ENTITY_SCOPE, _MEASURE), (_GROUP, _TIME), "VALUE_PROJECTION", "SCALAR", "entity_value_relation", "argmin_relation", tie_semantics="equal_values_collapse", projection_source="selected_value_nodes"),
        _signature("ARGMIN_ENTITY", "ARGMIN", "ENTITY", (_ENTITY_SCOPE, _MEASURE), (_GROUP, _TIME), "ROW_ENTITY_PROJECTION", "ENTITY", "entity_value_relation", "argmin_relation", tie_semantics="unique_only", projection_source="relation_entity_binding_ids"),
        _signature("ARGMIN_ENTITY_SET", "ARGMIN", "ENTITY_SET", (_ENTITY_SCOPE, _MEASURE), (_GROUP, _TIME), "ROW_ENTITY_PROJECTION", "SET", "entity_value_relation", "argmin_relation", tie_semantics="multiple_only_canonical_set", projection_source="relation_entity_binding_ids"),
        _signature("PAIR_COMPARE_BOOLEAN", "PAIR_COMPARE", "BOOLEAN_RELATION", (_LEFT, _RIGHT), (), "BOOLEAN_PROJECTION", "BOOLEAN", "ordered_atomic_operands", "pair_boolean_compare", ordered_semantics="ordered_operands", comparison_polarities=COMPARISON_POLARITIES, projection_source="ordered_numeric_operands"),
    )
}


EXCLUDED_OPERATION_SIGNATURES = {
    "PAIR_COMPARE_ENTITY": ExcludedOperationSignature(
        signature_id="PAIR_COMPARE_ENTITY",
        operation_family="PAIR_COMPARE",
        semantic_result_role="ENTITY_SELECTION",
        reason=(
            "Current LEFT_OPERAND and RIGHT_OPERAND bindings identify numeric cells "
            "without mandatory exact entity provenance. Entity selection is excluded "
            "until an entity-bearing operand form is formalized."
        ),
    )
}


ROLE_VOCABULARY = {
    role.name
    for signature in OPERATION_SIGNATURES.values()
    for role in signature.allowed_roles
}


OPERATION_SIGNATURES_BY_OPERATION: Dict[str, Tuple[OperationSignatureVariant, ...]] = {
    operation: tuple(
        signature
        for signature in OPERATION_SIGNATURES.values()
        if signature.operation_family == operation
    )
    for operation in FINAL_SUPPORTED_OPERATIONS
}


_DEFAULT_SIGNATURE_BY_OPERATION = {
    "LOOKUP": "LOOKUP_VALUE_SCALAR",
    "SUM": "SUM_SCALAR",
    "AVERAGE": "AVERAGE_SCALAR",
    "COUNT": "COUNT_SCALAR",
    "DIFF": "DIFF_SCALAR",
    "RATIO": "RATIO_SCALAR",
    "ARGMAX": "ARGMAX_ENTITY",
    "ARGMIN": "ARGMIN_ENTITY",
    "PAIR_COMPARE": "PAIR_COMPARE_BOOLEAN",
}

# Backward-compatible family view. New generation and closure select signature_id.
OPERATION_CONTRACTS = {
    operation: OPERATION_SIGNATURES[signature_id]
    for operation, signature_id in _DEFAULT_SIGNATURE_BY_OPERATION.items()
}


def operation_signature_variants(operation_family: Any) -> Tuple[OperationSignatureVariant, ...]:
    return OPERATION_SIGNATURES_BY_OPERATION.get(str(operation_family or ""), ())


def get_operation_signature(signature_id: Any) -> Optional[OperationSignatureVariant]:
    return OPERATION_SIGNATURES.get(str(signature_id or ""))


def get_operation_contract(identifier: Any) -> Optional[OperationSignatureVariant]:
    text = str(identifier or "")
    return OPERATION_SIGNATURES.get(text) or OPERATION_CONTRACTS.get(text)


def resolve_operation_signature(plan: Mapping[str, Any]) -> Optional[OperationSignatureVariant]:
    signature_id = str(plan.get("signature_id") or "")
    if signature_id:
        return get_operation_signature(signature_id)
    operation = str(plan.get("operation_family") or "")
    projection = str(plan.get("projection_operator") or "")
    domain = str(plan.get("answer_domain") or "")
    matches = tuple(
        signature
        for signature in operation_signature_variants(operation)
        if signature.projection_operator == projection and signature.answer_domain == domain
    )
    return matches[0] if len(matches) == 1 else None


def signature_domains(operation_families: Iterable[Any]) -> Tuple[str, ...]:
    operations = {str(item) for item in operation_families}
    return tuple(sorted({
        signature.answer_domain
        for signature in OPERATION_SIGNATURES.values()
        if signature.operation_family in operations
    }))


def signature_projections(operation_families: Iterable[Any]) -> Tuple[str, ...]:
    operations = {str(item) for item in operation_families}
    return tuple(sorted({
        signature.projection_operator
        for signature in OPERATION_SIGNATURES.values()
        if signature.operation_family in operations
    }))


def _terminal_schema(reference_ids: Sequence[str]) -> Dict[str, Any]:
    return {"type": "string", "enum": sorted({str(item) for item in reference_ids})}


def _array_schema(items: Dict[str, Any], role: Optional[RoleContract] = None) -> Dict[str, Any]:
    schema: Dict[str, Any] = {"type": "array", "items": items}
    if role is not None:
        schema["minItems"] = role.min_items
        if role.max_items is not None:
            schema["maxItems"] = role.max_items
    else:
        schema["minItems"] = 1
    return schema


def build_operation_role_schema_defs(reference_ids: Sequence[str]) -> Dict[str, Any]:
    terminal = _terminal_schema(reference_ids)
    return {
        "terminal_role_reference": terminal,
        "structural_conjunction": _array_schema({"$ref": "#/$defs/terminal_role_reference"}),
        "structural_conjunction_domain": _array_schema({"$ref": "#/$defs/structural_conjunction"}),
        "finite_scope_members": _array_schema({"$ref": "#/$defs/structural_conjunction"}),
        "finite_scope_members_domain": _array_schema({"$ref": "#/$defs/finite_scope_members"}),
        # Legacy names remain schema aliases only for historical readers.
        "role_binding": _array_schema({"$ref": "#/$defs/terminal_role_reference"}),
        "role_domain": _array_schema({"$ref": "#/$defs/structural_conjunction"}),
    }


def _role_schema_ref(role: RoleContract, *, domain: bool, legacy_shape: bool) -> Dict[str, Any]:
    if legacy_shape:
        name = "role_domain" if domain else "role_binding"
        return {"$ref": f"#/$defs/{name}"}

    if role.shape == "structural_conjunction":
        binding_schema = _array_schema(
            {"$ref": "#/$defs/terminal_role_reference"},
            role,
        )
    elif role.shape == "finite_scope_members":
        binding_schema = _array_schema(
            {"$ref": "#/$defs/structural_conjunction"},
            role,
        )
    else:
        raise ValueError(f"unsupported_role_shape:{role.name}:{role.shape}")
    return _array_schema(binding_schema) if domain else binding_schema


def _role_source_variants(signature: OperationSignatureVariant) -> Iterable[Dict[str, str]]:
    choices = [
        ("binding", "domain")
        if role.name in signature.required_role_names
        else ("absent", "binding", "domain")
        for role in signature.allowed_roles
    ]
    for selected in product(*choices):
        yield {role.name: source for role, source in zip(signature.allowed_roles, selected)}


def _signature_plan_variants(
    signature: OperationSignatureVariant,
    *,
    require_signature_id: bool,
    legacy_shape: bool,
) -> list[Dict[str, Any]]:
    common_properties: Dict[str, Any] = {
        "plan_id": {"type": "string", "pattern": r"^P[0-9]+$"},
        "operation_family": {"type": "string", "const": signature.operation_family},
        "answer_domain": {"type": "string", "const": signature.answer_domain},
        "projection_operator": {"type": "string", "const": signature.projection_operator},
        "unresolved_semantics": {"type": "array", "items": {"type": "string"}},
    }
    common_required = [
        "operation_family",
        "answer_domain",
        "projection_operator",
        "unresolved_semantics",
    ]
    if require_signature_id:
        common_properties.update({
            "signature_id": {"type": "string", "const": signature.signature_id},
            "semantic_result_role": {"type": "string", "const": signature.semantic_result_role},
        })
        common_required.extend(("signature_id", "semantic_result_role"))
    if signature.comparison_polarities:
        common_properties["comparison_polarity"] = {
            "type": "string",
            "enum": list(signature.comparison_polarities),
        }
        common_required.append("comparison_polarity")

    variants = []
    for sources in _role_source_variants(signature):
        properties = dict(common_properties)
        required = list(common_required)
        binding_roles = sorted(role for role, source in sources.items() if source == "binding")
        domain_roles = sorted(role for role, source in sources.items() if source == "domain")
        if binding_roles:
            properties["role_bindings"] = {
                "type": "object",
                "properties": {
                    role_name: _role_schema_ref(
                        signature.role(role_name),
                        domain=False,
                        legacy_shape=legacy_shape,
                    )
                    for role_name in binding_roles
                },
                "required": binding_roles,
                "additionalProperties": False,
            }
            required.append("role_bindings")
        if domain_roles:
            properties["role_domains"] = {
                "type": "object",
                "properties": {
                    role_name: _role_schema_ref(
                        signature.role(role_name),
                        domain=True,
                        legacy_shape=legacy_shape,
                    )
                    for role_name in domain_roles
                },
                "required": domain_roles,
                "additionalProperties": False,
            }
            required.append("role_domains")
        variants.append({
            "type": "object",
            "properties": properties,
            "required": sorted(required),
            "additionalProperties": False,
        })
    return variants


def build_operation_plan_schema(
    identifier: str,
    reference_ids: Sequence[str],
    *,
    include_shared_defs: bool = True,
    require_signature_id: Optional[bool] = None,
) -> Dict[str, Any]:
    explicit = get_operation_signature(identifier)
    signatures = (explicit,) if explicit is not None else operation_signature_variants(identifier)
    if not signatures:
        raise ValueError(f"unsupported_operation_contract:{identifier}")
    strict = bool(explicit) if require_signature_id is None else require_signature_id
    variants = [
        variant
        for signature in signatures
        for variant in _signature_plan_variants(
            signature,
            require_signature_id=strict,
            legacy_shape=not strict,
        )
    ]
    schema: Dict[str, Any] = {"anyOf": variants}
    if include_shared_defs:
        schema["$defs"] = build_operation_role_schema_defs(reference_ids)
    return schema


def _cardinality_errors(role: RoleContract, count: int, errors: list[str]) -> None:
    if count == 0:
        errors.append(f"empty_role:{role.name}")
    if count < role.min_items:
        errors.append(f"role_cardinality:{role.name}:{count}<{role.min_items}")
    if role.max_items is not None and count > role.max_items:
        errors.append(f"role_cardinality:{role.name}:{count}>{role.max_items}")


def _normalize_conjunction(
    role: RoleContract,
    value: Any,
    schema_ids: set[str],
    errors: list[str],
) -> Tuple[str, ...]:
    if not isinstance(value, list) or any(isinstance(item, (list, tuple, dict)) for item in value):
        errors.append(f"invalid_role_shape:{role.name}")
        return ()
    _cardinality_errors(role, len(value), errors)
    normalized = tuple(str(item) for item in value if str(item))
    if len(normalized) != len(value):
        errors.append(f"empty_role_reference:{role.name}")
    for schema_id in normalized:
        if schema_id not in schema_ids:
            errors.append(f"unknown_schema_id:{role.name}:{schema_id}")
    if role.ordered:
        return normalized
    return tuple(sorted(set(normalized)))


def _normalize_binding(
    role: RoleContract,
    value: Any,
    schema_ids: set[str],
    errors: list[str],
    *,
    legacy_shape: bool,
) -> Any:
    if role.shape == "structural_conjunction":
        return _normalize_conjunction(role, value, schema_ids, errors)
    if role.shape != "finite_scope_members":
        errors.append(f"unsupported_role_shape:{role.name}:{role.shape}")
        return ()
    if legacy_shape and isinstance(value, list) and all(not isinstance(item, (list, tuple, dict)) for item in value):
        return _normalize_conjunction(role, value, schema_ids, errors)
    if not isinstance(value, list) or any(not isinstance(member, list) for member in value):
        errors.append(f"invalid_role_shape:{role.name}")
        return ()
    _cardinality_errors(role, len(value), errors)
    member_role = RoleContract(role.name, "structural_conjunction", 1, None, False)
    members = tuple(
        conjunction
        for member in value
        for conjunction in (_normalize_conjunction(member_role, member, schema_ids, errors),)
        if conjunction
    )
    return tuple(sorted(set(members)))


def _normalize_domain(
    role: RoleContract,
    value: Any,
    schema_ids: set[str],
    errors: list[str],
    *,
    legacy_shape: bool,
) -> Tuple[Any, ...]:
    if not isinstance(value, list):
        errors.append(f"role_domain_not_list:{role.name}")
        return ()
    if not value:
        errors.append(f"empty_role_domain:{role.name}")
        return ()
    options = []
    for option in value:
        normalized_option = option
        if legacy_shape and role.shape == "structural_conjunction" and isinstance(option, str):
            normalized_option = [option]
        normalized = _normalize_binding(
            role,
            normalized_option,
            schema_ids,
            errors,
            legacy_shape=legacy_shape,
        )
        if normalized:
            options.append(normalized)
    return tuple(sorted(set(options)))


def validate_operation_plan(
    plan: Mapping[str, Any],
    reference_ids: Iterable[str],
) -> OperationContractValidation:
    operation_family = str(plan.get("operation_family") or "")
    strict = bool(plan.get("signature_id"))
    signature = resolve_operation_signature(plan)
    if signature is None and not strict:
        signature = get_operation_contract(operation_family)
    if signature is None:
        reason = (
            f"unsupported_operation_signature:{plan.get('signature_id')}"
            if strict
            else f"unsupported_operation_contract:{operation_family}"
        )
        return OperationContractValidation(
            ok=False,
            operation_family=operation_family,
            signature_id=str(plan.get("signature_id") or ""),
            errors=(reason,),
        )

    errors: list[str] = []
    allowed_fields = {
        "plan_id",
        "signature_id",
        "operation_family",
        "semantic_result_role",
        "answer_domain",
        "projection_operator",
        "role_bindings",
        "role_domains",
        "comparison_polarity",
        "unresolved_semantics",
    }
    errors.extend(
        f"unknown_operation_plan_field:{field}"
        for field in sorted(set(plan) - allowed_fields)
    )
    expected_fields = {
        "operation_family": signature.operation_family,
        "projection_operator": signature.projection_operator,
        "answer_domain": signature.answer_domain,
    }
    if strict:
        expected_fields["semantic_result_role"] = signature.semantic_result_role
    for field, expected in expected_fields.items():
        actual = str(plan.get(field) or "")
        if actual != expected:
            errors.append(f"signature_field_mismatch:{field}:{actual}!={expected}")
    if "unresolved_semantics" not in plan:
        errors.append("missing_unresolved_semantics")
    elif not isinstance(plan["unresolved_semantics"], list):
        errors.append("unresolved_semantics_not_list")
    elif any(not isinstance(item, str) for item in plan["unresolved_semantics"]):
        errors.append("unresolved_semantics_item_not_string")

    bindings_value = plan["role_bindings"] if "role_bindings" in plan else {}
    if not isinstance(bindings_value, Mapping):
        errors.append("role_bindings_not_object")
        bindings_value = {}
    domains_value = plan["role_domains"] if "role_domains" in plan else {}
    if not isinstance(domains_value, Mapping):
        errors.append("role_domains_not_object")
        domains_value = {}
    present_roles = set(str(role) for role in bindings_value) | set(str(role) for role in domains_value)
    for role_name in sorted(present_roles - set(signature.role_names)):
        errors.append(f"forbidden_role:{role_name}")
    for role_name in signature.required_role_names:
        if role_name not in bindings_value and role_name not in domains_value:
            errors.append(f"missing_required_role:{role_name}")

    schema_ids = {str(item) for item in reference_ids}
    normalized_bindings: Dict[str, Any] = {}
    normalized_domains: Dict[str, Tuple[Any, ...]] = {}
    for role in signature.allowed_roles:
        if role.name in bindings_value:
            normalized_bindings[role.name] = _normalize_binding(
                role,
                bindings_value[role.name],
                schema_ids,
                errors,
                legacy_shape=not strict,
            )
        if role.name in domains_value:
            normalized_domains[role.name] = _normalize_domain(
                role,
                domains_value[role.name],
                schema_ids,
                errors,
                legacy_shape=not strict,
            )
        if role.name in normalized_bindings and role.name in normalized_domains:
            if normalized_bindings[role.name] not in normalized_domains[role.name]:
                errors.append(f"binding_not_in_domain:{role.name}")

    polarity = str(plan.get("comparison_polarity") or "")
    if signature.comparison_polarities:
        if not polarity:
            errors.append("missing_comparison_polarity")
        elif polarity not in signature.comparison_polarities:
            errors.append(f"invalid_comparison_polarity:{polarity}")
    elif "comparison_polarity" in plan:
        errors.append("forbidden_comparison_polarity")

    return OperationContractValidation(
        ok=not errors,
        operation_family=operation_family,
        signature_id=signature.signature_id,
        errors=tuple(errors),
        normalized_role_bindings=tuple(sorted(normalized_bindings.items())),
        normalized_role_domains=tuple(sorted(normalized_domains.items())),
    )


def operation_signature_telemetry(signature_id: str) -> Dict[str, Any]:
    signature = get_operation_signature(signature_id)
    if signature is None:
        return {
            "contract_version": OPERATION_CONTRACT_VERSION,
            "signature_id": str(signature_id or ""),
            "supported": False,
        }
    payload = signature.to_dict()
    payload["supported"] = True
    return payload


def operation_contract_telemetry(operation_family: str) -> Dict[str, Any]:
    variants = operation_signature_variants(operation_family)
    if not variants:
        return {
            "contract_version": OPERATION_CONTRACT_VERSION,
            "operation_family": str(operation_family or ""),
            "supported": False,
        }
    default = OPERATION_CONTRACTS[str(operation_family)]
    return {
        "contract_version": OPERATION_CONTRACT_VERSION,
        "operation_family": str(operation_family),
        "supported": True,
        "signature_ids": [signature.signature_id for signature in variants],
        "signature_variants": [signature.to_dict() for signature in variants],
        # Compatibility summary fields are derived from the same registry.
        "required_roles": list(default.required_role_names),
        "optional_roles": [role.name for role in default.optional_roles],
        "role_shapes": {role.name: role.shape for role in default.allowed_roles},
        "projection_domain_pairs": [
            [signature.projection_operator, signature.answer_domain]
            for signature in variants
        ],
        "resolution_mode": default.resolution_mode,
        "execution_family": default.execution_family,
        "ordered": default.ordered,
        "comparison_polarities": list(default.comparison_polarities),
        "tie_semantics": default.tie_semantics,
    }
