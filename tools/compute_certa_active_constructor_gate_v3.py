#!/usr/bin/env python3
"""Repo-native Constructor Gate C for assignment-level grounding authority V3."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import jsonschema

from certa.active_v1.artifact_authority import (
    reconcile_registry_entry,
    validate_grounding_record_v3,
)
from certa.reproducibility.canonical_json import canonical_json_hash


ARMS = ("C0_SCHEMA_ONLY", "C1_ROLE_ONLY", "C2_ROLE_RETRIEVAL")
COMMON = (
    "table_id",
    "question_sha256",
    "b0_answer_sha256",
    "method_sha",
    "model_profile_sha256",
    "operation_registry_sha256",
    "planner_schema_sha256",
    "closure_sha256",
    "executor_sha256",
    "artifact_schema_sha256",
)
SAFETY_KEYS = (
    "task_role_incompatible_executable",
    "registry_external_contrast",
    "b0_mutation",
    "gold_leakage",
    "runtime_leakage",
    "first_match_resolution",
    "identity_mismatch",
    "reconciliation_mismatch",
    "fixture_id_count",
    "ambiguous_assignment_authorized",
    "grounding_authority_mismatch",
)
THRESHOLDS = {
    "c2_paired_min": 8,
    "paired_gain_min": 4,
    "c2_registry_complete_paired_min": 6,
    "registry_gain_min": 3,
    "paired_tables_min": 4,
    "role_compatible_precision": 1.0,
}
REPO = Path(__file__).resolve().parents[1]
GROUNDING_SCHEMA = json.loads(
    (REPO / "schemas/active_v1/RAW_GROUNDING_RECORD_V3.schema.json").read_text()
)


def _jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate_thresholds(
    effect: Mapping[str, Any],
    safety: Mapping[str, int],
) -> list[str]:
    checks = (
        (all(value == 0 for value in safety.values()), "safety"),
        (effect["c2_paired"] >= THRESHOLDS["c2_paired_min"], "paired_absolute"),
        (effect["paired_gain"] >= THRESHOLDS["paired_gain_min"], "paired_gain"),
        (
            effect["c2_registry_complete_paired"]
            >= THRESHOLDS["c2_registry_complete_paired_min"],
            "registry_absolute",
        ),
        (effect["registry_gain"] >= THRESHOLDS["registry_gain_min"], "registry_gain"),
        (effect["paired_tables"] >= THRESHOLDS["paired_tables_min"], "table_coverage"),
        (
            effect["role_compatible_precision"]
            == THRESHOLDS["role_compatible_precision"],
            "role_compatible_precision",
        ),
    )
    return [name for passed, name in checks if not passed]


def _canonical_registry_match(
    registry: Mapping[str, Any],
    derivation: Mapping[str, Any],
) -> bool:
    if derivation.get("provenance_ids") != sorted(set(derivation.get("provenance_ids", ()))):
        return False
    try:
        reconcile_registry_entry(registry, derivation)
    except ValueError:
        return False
    return True


def compute_gate(
    *,
    identities: Sequence[Mapping[str, Any]],
    role_records: Sequence[Mapping[str, Any]],
    groundings: Sequence[Mapping[str, Any]],
    derivations: Sequence[Mapping[str, Any]],
    registry: Sequence[Mapping[str, Any]],
    cost_ledger: Mapping[str, Any],
    allow_fixture: bool = False,
) -> dict[str, Any]:
    all_records = [*groundings, *derivations, *registry]
    fixture_count = sum(
        int(row.get("fixture_only") or str(row.get("sample_id", "")).startswith("FX_"))
        for row in all_records
    )
    if fixture_count and not allow_fixture:
        raise ValueError("fixture_forbidden")
    by_id = {(row["sample_id"], row["arm"]): row for row in identities}
    if len(by_id) != len(identities):
        raise ValueError("duplicate_sample_arm_identity")
    sample_sets = {
        arm: {row["sample_id"] for row in identities if row["arm"] == arm}
        for arm in ARMS
    }
    sample_equal = len({frozenset(values) for values in sample_sets.values()}) == 1
    expected_count = len(sample_sets[ARMS[0]]) if allow_fixture else 64
    if (
        not sample_equal
        or expected_count < 1
        or any(len(sample_sets[arm]) != expected_count for arm in ARMS)
    ):
        raise ValueError("matched_sample_set_failure")
    samples = sorted(sample_sets[ARMS[0]])
    role_by = {row["sample_id"]: row for row in role_records}
    if len(role_by) != len(role_records):
        raise ValueError("duplicate_role_record")
    identity_mismatch = 0
    role_hash_mismatch = 0
    for sample_id in samples:
        rows = [by_id[(sample_id, arm)] for arm in ARMS]
        identity_mismatch += sum(
            int(row.get(key) != rows[0].get(key))
            for row in rows[1:]
            for key in COMMON
        )
        role_hash_mismatch += int(
            rows[1].get("role_record_sha256") != rows[2].get("role_record_sha256")
        )
        role = role_by.get(sample_id)
        if role:
            role_hash_mismatch += int(
                rows[1].get("role_record_sha256") != role.get("record_sha256")
            )
    ground_by: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for record in groundings:
        jsonschema.validate(record, GROUNDING_SCHEMA)
        validate_grounding_record_v3(record)
        key = (record["sample_id"], record["arm"], record["plan_id"])
        if key in ground_by:
            raise ValueError("duplicate_grounding_plan_record")
        ground_by[key] = record
    derivation_by = defaultdict(list)
    for row in derivations:
        derivation_by[(row["sample_id"], row["arm"])].append(row)
    registry_by = defaultdict(list)
    for index, row in enumerate(registry):
        registry_by[(row["sample_id"], row["arm"], row["derivation_id"])].append(
            (index, row)
        )
    safety = {key: 0 for key in SAFETY_KEYS}
    safety["identity_mismatch"] = identity_mismatch + role_hash_mismatch
    safety["fixture_id_count"] = 0 if allow_fixture else fixture_count
    summaries: dict[str, dict[str, Any]] = {}
    consumed_registry: set[int] = set()
    for arm in ARMS:
        authorized_rows = executable_rows = paired_rows = registry_paired_rows = 0
        role_compatible_count = executable_count = 0
        paired_tables = set()
        for sample_id in samples:
            identity = by_id[(sample_id, arm)]
            safety["b0_mutation"] += int(
                identity["final_answer_sha256"] != identity["b0_answer_sha256"]
            )
            safety["gold_leakage"] += int(identity["gold_accessed"])
            safety["runtime_leakage"] += int(identity["runtime_leakage"])
            sample_groundings = {
                plan_id: record
                for (sid, record_arm, plan_id), record in ground_by.items()
                if sid == sample_id and record_arm == arm
            }
            authorized: dict[tuple[str, str], Mapping[str, Any]] = {}
            for plan_id, record in sample_groundings.items():
                safety["first_match_resolution"] += int(record["first_match_used"])
                safety["grounding_authority_mismatch"] += int(
                    record["table_id"] != identity["table_id"]
                )
                if arm != "C0_SCHEMA_ONLY":
                    safety["grounding_authority_mismatch"] += int(
                        record["role_record_sha256"]
                        != identity["role_record_sha256"]
                    )
                hypotheses = {
                    row["binding_id"]: row for row in record["grounding_hypotheses"]
                }
                for binding_id in record["authorized_binding_ids"]:
                    hypothesis = hypotheses[binding_id]
                    if hypothesis["resolution_state"] != "EXACT":
                        safety["ambiguous_assignment_authorized"] += 1
                    authorized[(plan_id, binding_id)] = hypothesis
            authorized_rows += int(bool(authorized))
            executable = []
            program_answers: dict[str, str] = {}
            seen_derivations = set()
            for derivation in sorted(
                derivation_by[(sample_id, arm)],
                key=lambda row: row["derivation_id"],
            ):
                if (
                    derivation.get("execution_status") != "EXECUTED"
                    or derivation.get("projection_status") != "VALID"
                ):
                    continue
                derivation_id = derivation["derivation_id"]
                if derivation_id in seen_derivations:
                    safety["reconciliation_mismatch"] += 1
                    continue
                seen_derivations.add(derivation_id)
                hypothesis = authorized.get(
                    (derivation.get("plan_id"), derivation.get("binding_id"))
                )
                fields_match = bool(
                    hypothesis
                    and hypothesis["derivation_id"] == derivation_id
                    and hypothesis["canonical_program_id"]
                    == derivation.get("canonical_program_id")
                    and hypothesis["operand_node_ids"]
                    == derivation.get("operand_node_ids")
                )
                if not fields_match:
                    safety["reconciliation_mismatch"] += 1
                    continue
                computed_side = (
                    "ORIGINAL"
                    if derivation["projected_answer_hash"]
                    == identity["b0_answer_sha256"]
                    else "ALTERNATIVE"
                )
                if derivation.get("side") != computed_side:
                    safety["reconciliation_mismatch"] += 1
                program_id = derivation["canonical_program_id"]
                answer_hash = derivation["projected_answer_hash"]
                if program_id in program_answers:
                    safety["reconciliation_mismatch"] += int(
                        program_answers[program_id] != answer_hash
                    )
                    continue
                program_answers[program_id] = answer_hash
                executable.append((derivation, computed_side))
            executable_rows += int(bool(executable))
            executable_count += len(executable)
            role = role_by.get(sample_id)
            for derivation, _ in executable:
                compatible = True
                if arm in ("C1_ROLE_ONLY", "C2_ROLE_RETRIEVAL"):
                    compatible = bool(
                        role
                        and derivation["signature_id"] == role["signature"]
                        and derivation["answer_role"] == role["answer_role"]
                        and derivation["projection"] == role["projection"]
                    )
                role_compatible_count += int(compatible)
                safety["task_role_incompatible_executable"] += int(not compatible)
            original = [row for row in executable if row[1] == "ORIGINAL"]
            alternative = [row for row in executable if row[1] == "ALTERNATIVE"]
            paired = bool(original and alternative)
            paired_rows += int(paired)
            if paired:
                paired_tables.add(identity["table_id"])
            registry_complete = True
            for derivation, _ in executable:
                matches = registry_by[
                    (sample_id, arm, derivation["derivation_id"])
                ]
                exact = [
                    (index, row)
                    for index, row in matches
                    if _canonical_registry_match(row, derivation)
                ]
                if len(exact) != 1:
                    registry_complete = False
                    safety["registry_external_contrast"] += 1
                else:
                    consumed_registry.add(exact[0][0])
            registry_paired_rows += int(paired and registry_complete)
        summaries[arm] = {
            "authorized_grounding_rows": authorized_rows,
            "unique_grounding_rows": authorized_rows,
            "executable_rows": executable_rows,
            "paired_rows": paired_rows,
            "registry_complete_paired_rows": registry_paired_rows,
            "paired_tables": len(paired_tables),
            "role_compatible_executable_count": role_compatible_count,
            "executable_derivation_count": executable_count,
        }
    safety["registry_external_contrast"] += len(registry) - len(consumed_registry)
    c0, c1, c2 = (summaries[arm] for arm in ARMS)
    precision = (
        c2["role_compatible_executable_count"] / c2["executable_derivation_count"]
        if c2["executable_derivation_count"]
        else 0.0
    )
    effect = {
        "c2_paired": c2["paired_rows"],
        "paired_gain": c2["paired_rows"] - max(c0["paired_rows"], c1["paired_rows"]),
        "c2_registry_complete_paired": c2["registry_complete_paired_rows"],
        "registry_gain": c2["registry_complete_paired_rows"]
        - max(
            c0["registry_complete_paired_rows"],
            c1["registry_complete_paired_rows"],
        ),
        "paired_tables": c2["paired_tables"],
        "role_compatible_precision": precision,
    }
    failures = evaluate_thresholds(effect, safety)
    return {
        "schema_version": "certa_active_constructor_gate_v3",
        "computed_by": "tools/compute_certa_active_constructor_gate_v3.py",
        "sample_count": len(samples),
        "cross_arm_identity": {
            "sample_sets_equal": sample_equal,
            "identity_mismatch_count": identity_mismatch,
            "c1_c2_role_record_mismatch_count": role_hash_mismatch,
        },
        "arms": summaries,
        "safety": safety,
        "primary_effect": effect,
        "thresholds": dict(THRESHOLDS),
        "thresholds_sha256": canonical_json_hash(THRESHOLDS),
        "mechanism_metrics": {"cost_ledger": dict(cost_ledger)},
        "pass": not failures,
        "failure_reasons": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "identities",
        "role_records",
        "groundings",
        "derivations",
        "registry",
        "cost_ledger",
        "output",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    parser.add_argument("--allow-fixture", action="store_true")
    args = parser.parse_args()
    result = compute_gate(
        identities=_jsonl(args.identities),
        role_records=_jsonl(args.role_records),
        groundings=_jsonl(args.groundings),
        derivations=_jsonl(args.derivations),
        registry=_jsonl(args.registry),
        cost_ledger=json.loads(Path(args.cost_ledger).read_text(encoding="utf-8")),
        allow_fixture=args.allow_fixture,
    )
    result["source_artifacts"] = {
        "identities_sha256": _sha256(args.identities),
        "role_records_sha256": _sha256(args.role_records),
        "groundings_sha256": _sha256(args.groundings),
        "derivations_sha256": _sha256(args.derivations),
        "registry_sha256": _sha256(args.registry),
    }
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PASS" if result["pass"] else "FREEZE_CERTA_ACTIVE_CONSTRUCTOR_FAILED")
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
