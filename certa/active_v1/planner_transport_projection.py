"""Compact transport-only projection of the full typed Planner schema."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping

from certa.reproducibility.canonical_json import canonical_json, canonical_json_hash


_SCALAR_FIELDS = ("signature_id", "operation_family", "semantic_result_role", "answer_domain", "projection_operator", "comparison_polarity")
_ROLE_CONTAINERS = ("role_bindings", "role_domains")


def _object_leaves(schema: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    choices = schema.get("anyOf")
    if isinstance(choices, list):
        for choice in choices:
            if isinstance(choice, Mapping):
                yield from _object_leaves(choice)
        return
    if isinstance(schema.get("properties"), Mapping):
        yield schema


def _values(leaves: Iterable[Mapping[str, Any]], field: str) -> list[str]:
    values = set()
    for leaf in leaves:
        prop = leaf.get("properties", {}).get(field, {})
        if "const" in prop:
            values.add(str(prop["const"]))
        values.update(str(item) for item in prop.get("enum", []))
    return sorted(values)


def _types(schema: Mapping[str, Any]) -> list[str]:
    return sorted({str(item) for item in ([schema.get("type")] + [leaf.get("type") for leaf in _object_leaves(schema)]) if item})


def _string_enum(leaves: list[Mapping[str, Any]], field: str) -> dict[str, Any]:
    values = _values(leaves, field)
    if not values:
        raise ValueError(f"planner_transport_field_domain_empty:{field}")
    return {"type": "string", "enum": values}


def _shared_schema(leaves: list[Mapping[str, Any]], field: str) -> dict[str, Any]:
    schemas = {
        canonical_json(leaf["properties"][field]): leaf["properties"][field]
        for leaf in leaves
        if field in leaf.get("properties", {})
    }
    if not schemas:
        raise ValueError(f"planner_transport_field_schema_missing:{field}")
    if len(schemas) != 1:
        raise ValueError(f"planner_transport_field_schema_not_shared:{field}")
    return deepcopy(next(iter(schemas.values())))


def _role_container(
    leaves: list[Mapping[str, Any]], container_name: str,
) -> dict[str, Any]:
    role_schemas: dict[str, dict[str, Mapping[str, Any]]] = {}
    for leaf in leaves:
        container = leaf.get("properties", {}).get(container_name, {})
        for role, schema in container.get("properties", {}).items():
            role_schemas.setdefault(str(role), {})[canonical_json(schema)] = schema
    properties = {}
    for role, schemas in sorted(role_schemas.items()):
        variants = [deepcopy(schemas[key]) for key in sorted(schemas)]
        properties[role] = variants[0] if len(variants) == 1 else {"anyOf": variants}
    if not properties:
        raise ValueError(f"planner_transport_role_container_empty:{container_name}")
    return {"type": "object", "properties": properties, "additionalProperties": False}


def build_planner_transport_schema(full_schema: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only Cartesian semantic/binding variants from a full Planner schema."""
    properties = full_schema.get("properties")
    if not isinstance(properties, Mapping) or set(properties) != {
        "planner_version", "query_semantics", "plans", "unresolved_semantics",
    }:
        raise ValueError("planner_transport_top_level_contract_mismatch")
    plan_schema = properties.get("plans", {}).get("items")
    query_schema = properties.get("query_semantics")
    if not isinstance(plan_schema, Mapping) or not isinstance(query_schema, Mapping):
        raise ValueError("planner_transport_variant_schema_missing")
    plan_leaves = list(_object_leaves(plan_schema))
    query_leaves = list(_object_leaves(query_schema))
    if not plan_leaves or not query_leaves:
        raise ValueError("planner_transport_variant_schema_empty")

    plan_properties = {
        "plan_id": _shared_schema(plan_leaves, "plan_id"),
        "unresolved_semantics": _shared_schema(plan_leaves, "unresolved_semantics"),
    }
    plan_properties.update({
        field: _string_enum(plan_leaves, field)
        for field in _SCALAR_FIELDS if _values(plan_leaves, field)
    })
    plan_properties.update({name: _role_container(plan_leaves, name) for name in _ROLE_CONTAINERS})
    plan_required = sorted(set.intersection(*(set(leaf.get("required", [])) for leaf in plan_leaves)))
    plan = {
        "type": "object",
        "properties": plan_properties,
        "required": plan_required,
        "additionalProperties": False,
    }
    query = {
        "type": "object",
        "properties": {
            field: _string_enum(query_leaves, field)
            for field in ("operation_family", "answer_domain", "projection_operator")
        },
        "required": sorted(set.intersection(*(set(leaf.get("required", [])) for leaf in query_leaves))),
        "additionalProperties": False,
    }
    plans = deepcopy(properties["plans"])
    plans["items"] = plan
    return {
        "$schema": full_schema.get("$schema", "https://json-schema.org/draft/2020-12/schema"),
        "type": "object",
        "properties": {
            "planner_version": deepcopy(properties["planner_version"]),
            "query_semantics": query,
            "plans": plans,
            "unresolved_semantics": deepcopy(properties["unresolved_semantics"]),
        },
        "required": deepcopy(full_schema.get("required", [])),
        "additionalProperties": False,
        "$defs": deepcopy(full_schema.get("$defs", {})),
    }


def planner_transport_schema_identity(
    full_schema: Mapping[str, Any], transport_schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Return falsifiable preservation checks and enum/reference hashes."""
    full_plans = list(_object_leaves(full_schema["properties"]["plans"]["items"]))
    transport_plan = transport_schema["properties"]["plans"]["items"]
    checks = {
        "top_level_field_set": set(full_schema["properties"]) == set(transport_schema["properties"]),
        "top_level_json_types": {_key: _types(_value) for _key, _value in full_schema["properties"].items()} == {_key: _types(_value) for _key, _value in transport_schema["properties"].items()},
        "planner_version": full_schema["properties"]["planner_version"] == transport_schema["properties"]["planner_version"],
        "reference_id_enum": full_schema.get("$defs", {}).get("terminal_role_reference") == transport_schema.get("$defs", {}).get("terminal_role_reference"),
        "signature_enum": _values(full_plans, "signature_id") == transport_plan["properties"]["signature_id"]["enum"],
        "operation_family_enum": _values(full_plans, "operation_family") == transport_plan["properties"]["operation_family"]["enum"],
        "answer_domain_enum": _values(full_plans, "answer_domain") == transport_plan["properties"]["answer_domain"]["enum"],
        "projection_operator_enum": _values(full_plans, "projection_operator") == transport_plan["properties"]["projection_operator"]["enum"],
        "role_container_keys": all(
            {role for leaf in full_plans for role in leaf.get("properties", {}).get(name, {}).get("properties", {})}
            == set(transport_plan["properties"][name]["properties"])
            for name in _ROLE_CONTAINERS
        ),
        "additional_properties_false": all(
            item.get("additionalProperties") is False
            for item in (transport_schema, transport_schema["properties"]["query_semantics"], transport_plan,
                         *(transport_plan["properties"][name] for name in _ROLE_CONTAINERS))
        ),
        "required_nonempty_arrays": transport_schema["properties"]["plans"].get("minItems") == full_schema["properties"]["plans"].get("minItems") == 1 and all(_role_container(full_plans, name)["properties"] == transport_plan["properties"][name]["properties"] for name in _ROLE_CONTAINERS),
    }
    domains = {
        field: transport_plan["properties"][field]["enum"]
        for field in ("signature_id", "operation_family", "answer_domain", "projection_operator")
    }
    return {
        "full_schema_sha256": canonical_json_hash(full_schema),
        "transport_schema_sha256": canonical_json_hash(transport_schema),
        "reference_domain_sha256": canonical_json_hash(transport_schema["$defs"]["terminal_role_reference"]["enum"]),
        "enum_sha256": {field: canonical_json_hash(values) for field, values in domains.items()},
        "preservation_checks": checks,
        "all_preservation_checks_pass": all(checks.values()),
    }
