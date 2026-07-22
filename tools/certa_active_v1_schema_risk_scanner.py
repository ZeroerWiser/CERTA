#!/usr/bin/env python3
"""Deterministic offline complexity scanner for Planner transport schemas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from certa.reproducibility.canonical_json import canonical_json


DEFAULT_LIMITS = {
    "any_of_items_max": 8,
    "canonical_json_bytes_max": 32768,
    "maximum_depth_max": 10,
    "property_declarations_max": 128,
    "reference_occurrences_max": 64,
    "unsupported_xgrammar_keywords_max": 0,
}


def _walk(value: Any, path: str = "$", depth: int = 0):
    yield path, value, depth
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk(child, f"{path}/{key}", depth + 1)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, f"{path}/{index}", depth + 1)


def scan_schema(schema: Mapping[str, Any], limits: Mapping[str, int]) -> dict[str, Any]:
    nodes = list(_walk(schema))
    unsupported = []
    for path, value, _ in nodes:
        if not isinstance(value, Mapping):
            continue
        keys = set(value)
        bad = keys & {"multipleOf", "uniqueItems", "contains", "minContains", "maxContains", "format", "minProperties", "maxProperties", "propertyNames", "patternProperties"}
        if path.endswith(("/properties", "/$defs")):
            bad = set()
        unsupported.extend({"path": path, "keyword": key} for key in sorted(bad))
    result = {
        "canonical_json_bytes": len(canonical_json(schema).encode()),
        "maximum_depth": max(depth for _, _, depth in nodes),
        "property_declarations": sum(
            len(value["properties"]) for _, value, _ in nodes
            if isinstance(value, Mapping) and isinstance(value.get("properties"), Mapping)
        ),
        "reference_occurrences": sum(
            1 for _, value, _ in nodes if isinstance(value, Mapping) and "$ref" in value
        ),
        "any_of_items_max": max([
            len(value["anyOf"]) for _, value, _ in nodes
            if isinstance(value, Mapping) and isinstance(value.get("anyOf"), list)
        ] or [0]),
        "unsupported_xgrammar_keywords": unsupported,
        "unsupported_xgrammar_keywords_count": len(unsupported),
    }
    fields = {
        "canonical_json_bytes": "canonical_json_bytes_max",
        "maximum_depth": "maximum_depth_max",
        "property_declarations": "property_declarations_max",
        "reference_occurrences": "reference_occurrences_max",
        "any_of_items_max": "any_of_items_max",
        "unsupported_xgrammar_keywords_count": "unsupported_xgrammar_keywords_max",
    }
    result["limit_failures"] = [
        f"{field}:{result[field]}>{limits[limit]}"
        for field, limit in fields.items() if result[field] > limits[limit]
    ]
    result["risk_scan_pass"] = not result["limit_failures"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = scan_schema(json.loads(args.schema.read_text(encoding="utf-8")), DEFAULT_LIMITS)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0 if result["risk_scan_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
