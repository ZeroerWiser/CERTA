"""Utilities for auditing legacy heuristic usage."""

from __future__ import annotations

from typing import Any, Dict, Mapping


LEGACY_HEURISTIC_FLAGS = (
    "heuristic_surface_used_for_commit",
    "operation_support_commit_role_source_is_surface_heuristic",
    "operation_support_commit_surface_named_entity_anchor_used",
    "hceg_fallback_applied",
    "certificate_commit_applied",
    "api_format_normalizer_applied",
    "normalizer_applied",
    "oracle_normalizer_applied",
    "self_consistency_changed",
)


def summarize_legacy_heuristics(row: Mapping[str, Any]) -> Dict[str, Any]:
    active = [name for name in LEGACY_HEURISTIC_FLAGS if bool(row.get(name))]
    return {
        "legacy_heuristic_usage_count": len(active),
        "legacy_heuristic_flags": active,
    }
