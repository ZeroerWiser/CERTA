#!/usr/bin/env python3
"""Atomically materialize CERTA final adapter and canonical-runtime artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from certa.active_v1.dataset_adapter_v1 import (  # noqa: E402
    AITQAAdapterV1,
    DatasetAdapterError,
    HiTabAdapterV1,
    SSTQAZhAdapterV1,
    TableResolutionError,
    canonical_json_sha256,
    roundtrip_adapter_artifact,
)
from certa.reproducibility.canonical_json import canonical_json  # noqa: E402


BASELINE_COMMIT = "a6818af3c157f3416bdff84925e003e36b3c4583"
FINAL_BRANCH = "research/certa-final-multidataset-method"
DEFAULT_SENTINELS = {
    "hitab": [
        "1057",
        "124_totto14665-1",
        "23_totto2677-1",
        "2421",
        "306_totto35002-0",
        "1319",
        "50_164_tab3",
        "74_3_nsf21317-tab003",
    ],
    "aitqa": [
        "tab-3",
        "tab-34",
        "tab-85",
        "tab-10",
        "tab-109",
        "tab-35",
        "tab-49",
        "tab-5",
    ],
    "sstqa_zh": [
        "49",
        "61",
        "4",
        "95",
        "15",
        "56",
        "71",
        "81",
    ],
}


class MaterializationError(RuntimeError):
    """Base class for one-shot materialization failures."""


class OutputRootExistsError(MaterializationError):
    """Raised rather than overwriting any prior run root."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return _sha256(path)


def _write_canonical_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")
    return _sha256(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(canonical_json(dict(row)) + "\n" for row in rows),
        encoding="utf-8",
    )
    return _sha256(path)


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MaterializationError(
                    f"invalid_jsonl:{path}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise MaterializationError(
                    f"jsonl_row_not_object:{path}:{line_number}"
                )
            rows.append(row)
    return rows


def _git(repo: Path, *arguments: str, allow_failure: bool = False) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode and not allow_failure:
        raise MaterializationError(
            f"git_command_failed:{' '.join(arguments)}:"
            f"{result.stderr.strip()}"
        )
    return result.stdout.strip()


def _question_index(
    path: Path,
    *,
    question_fields: Sequence[str],
) -> Dict[str, Dict[str, str]]:
    rows = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.suffix == ".json"
        else _read_jsonl(path)
    )
    if not isinstance(rows, list):
        raise MaterializationError(f"question_source_not_list:{path}")
    questions: Dict[str, Dict[str, str]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise MaterializationError(f"question_row_not_object:{path}")
        table_id = str(row.get("table_id") or "")
        question = next(
            (
                str(row.get(field) or "")
                for field in question_fields
                if str(row.get(field) or "")
            ),
            "",
        )
        if table_id and question:
            metadata = {
                "question": question,
                "table_source": str(row.get("table_source") or ""),
            }
            existing = questions.setdefault(table_id, metadata)
            if (
                existing["table_source"]
                and metadata["table_source"]
                and existing["table_source"] != metadata["table_source"]
            ):
                raise MaterializationError(
                    f"question_source_alias_conflict:{path}:{table_id}"
                )
            if not existing["table_source"] and metadata["table_source"]:
                existing["table_source"] = metadata["table_source"]
    return questions


def _cohort_resolution(
    adapter: HiTabAdapterV1,
    path: Path,
) -> Dict[str, Any]:
    rows = _read_jsonl(path)
    unresolved = []
    table_ids = []
    for row in rows:
        table_id = str(row.get("table_id") or "")
        table_ids.append(table_id)
        try:
            adapter.resolve_table(table_id)
        except DatasetAdapterError as exc:
            unresolved.append(
                {"table_id": table_id, "error": str(exc)}
            )
    return {
        "source_path": str(path.resolve()),
        "source_sha256": _sha256(path),
        "row_count": len(rows),
        "unique_id_count": len(
            {str(row.get("id") or "") for row in rows}
        ),
        "unique_table_count": len(set(table_ids)),
        "resolved_table_count": len(rows) - len(unresolved),
        "all_resolved": not unresolved,
        "unresolved": unresolved,
    }


def _project_blind_runtime(
    path: Path,
) -> list[Dict[str, str]]:
    projected = []
    for line_number, source in enumerate(_read_jsonl(path), start=1):
        row = {
            key: str(source.get(key) or "")
            for key in ("dataset", "id", "question", "table_id")
        }
        if any(not value for value in row.values()):
            raise MaterializationError(
                f"blind_runtime_field_empty:{path}:{line_number}"
            )
        projected.append(row)
    ids = [row["id"] for row in projected]
    tables = [row["table_id"] for row in projected]
    if len(ids) != len(set(ids)):
        raise MaterializationError(f"blind_runtime_duplicate_id:{path}")
    if len(tables) != len(set(tables)):
        raise MaterializationError(
            f"blind_runtime_duplicate_table_id:{path}"
        )
    return projected


def _safe_artifact_name(table_id: str) -> str:
    digest = hashlib.sha256(table_id.encode("utf-8")).hexdigest()
    return f"{digest[:24]}.json"


def _dataset_index_records(
    adapter: Any,
) -> list[Dict[str, Any]]:
    return [
        entry.to_record()
        for _, entry in sorted(adapter.index_tables().items())
    ]


def _negative_isolation(
    *,
    dataset_name: str,
    adapter: Any,
    valid_table_id: str,
    valid_question: str,
) -> Dict[str, Any]:
    invalid_id = "__CERTA_INVALID_TABLE_ID__"
    if dataset_name == "sstqa_zh":
        empty = next(
            (
                table_id
                for table_id, entry in adapter.index_tables().items()
                if str(entry.source_identity.get("workbook_id")) == "49"
                and int(entry.source_identity.get("sheet_index", 0)) > 0
            ),
            "",
        )
        if empty:
            invalid_id = empty
    invalid_error = ""
    try:
        invalid = adapter.resolve_table(invalid_id)
        adapter.canonicalize_table(invalid)
    except DatasetAdapterError as exc:
        invalid_error = f"{type(exc).__name__}:{exc}"
    if not invalid_error:
        raise MaterializationError(
            f"negative_table_did_not_fail:{dataset_name}:{invalid_id}"
        )
    valid_artifact = adapter.canonicalize_table(
        adapter.resolve_table(valid_table_id)
    )
    valid_smoke = roundtrip_adapter_artifact(
        valid_artifact,
        question=valid_question,
    )
    return {
        "invalid_table_id": invalid_id,
        "invalid_failure": invalid_error,
        "valid_table_id_after_failure": valid_table_id,
        "valid_table_pass_after_failure": valid_smoke["pass"],
        "pass": bool(invalid_error and valid_smoke["pass"]),
    }


def _materialize_sentinels(
    *,
    output: Path,
    dataset_name: str,
    adapter: Any,
    sentinel_ids: Sequence[str],
    questions: Mapping[str, Mapping[str, str]],
    minimum_sentinels: int,
) -> Dict[str, Any]:
    if len(sentinel_ids) < minimum_sentinels:
        raise MaterializationError(
            f"sentinel_count_below_minimum:{dataset_name}:"
            f"{len(sentinel_ids)}<{minimum_sentinels}"
        )
    if len(set(sentinel_ids)) != len(sentinel_ids):
        raise MaterializationError(
            f"duplicate_sentinel_id:{dataset_name}"
        )
    rows = []
    sentinel_root = output / "data" / dataset_name / "sentinels"
    for requested_table_id in sentinel_ids:
        metadata = questions.get(str(requested_table_id)) or {}
        question = str(metadata.get("question") or "")
        if not question:
            raise MaterializationError(
                f"sentinel_question_missing:{dataset_name}:"
                f"{requested_table_id}"
            )
        native = adapter.resolve_table(
            requested_table_id,
            runtime_record=metadata,
        )
        artifact = adapter.canonicalize_table(native)
        artifact_name = _safe_artifact_name(native.table_id)
        artifact_path = sentinel_root / artifact_name
        artifact_sha256 = _write_canonical_json(
            artifact_path,
            artifact,
        )
        smoke = roundtrip_adapter_artifact(
            artifact,
            question=question,
        )
        rows.append(
            {
                "requested_table_id": str(requested_table_id),
                "canonical_table_id": native.table_id,
                "question": question,
                "source_path": str(native.source_path),
                "source_sha256": native.source_sha256,
                "canonical_artifact": str(
                    artifact_path.relative_to(output)
                ),
                "canonical_artifact_sha256": artifact_sha256,
                "canonical_content_sha256": canonical_json_sha256(
                    artifact
                ),
                "structure_summary": artifact["structure_summary"],
                "roundtrip_smoke": smoke,
                "grounding_smoke_scope": (
                    "STRUCTURAL_LOOKUP_FIXTURE_NOT_QUESTION_ANSWER"
                ),
            }
        )
    negative = _negative_isolation(
        dataset_name=dataset_name,
        adapter=adapter,
        valid_table_id=str(sentinel_ids[0]),
        valid_question=str(
            questions[str(sentinel_ids[0])]["question"]
        ),
    )
    failures = [
        row["canonical_table_id"]
        for row in rows
        if not row["roundtrip_smoke"]["pass"]
    ]
    report = {
        "schema_version": "certa_adapter_sentinel_report_v1",
        "dataset": adapter.dataset_id,
        "adapter_id": adapter.adapter_id,
        "sentinel_count": len(rows),
        "minimum_required": minimum_sentinels,
        "grounding_smoke_scope": (
            "STRUCTURAL_LOOKUP_FIXTURE_NOT_QUESTION_ANSWER"
        ),
        "sentinels": rows,
        "negative_isolation": negative,
        "failure_table_ids": failures,
        "pass": not failures and negative["pass"],
    }
    if not report["pass"]:
        raise MaterializationError(
            f"adapter_sentinel_failed:{dataset_name}"
        )
    _write_json(
        output
        / "data"
        / dataset_name
        / "ADAPTER_SENTINEL_REPORT.json",
        report,
    )
    return report


def _materialize_hitab_runtimes(
    *,
    output: Path,
    adapter: HiTabAdapterV1,
    validation_source: Path,
    holdout_source: Path,
) -> Dict[str, Any]:
    validation = _project_blind_runtime(validation_source)
    holdout = _project_blind_runtime(holdout_source)
    validation_ids = {row["id"] for row in validation}
    holdout_ids = {row["id"] for row in holdout}
    validation_tables = {row["table_id"] for row in validation}
    holdout_tables = {row["table_id"] for row in holdout}
    id_overlap = sorted(validation_ids & holdout_ids)
    table_overlap = sorted(validation_tables & holdout_tables)
    if id_overlap:
        raise MaterializationError(
            "validation_holdout_id_overlap:"
            + ",".join(id_overlap)
        )
    if table_overlap:
        raise MaterializationError(
            "validation_holdout_table_overlap:"
            + ",".join(table_overlap)
        )
    cache_root = output / "data" / "hitab" / "canonical_tables"
    cache_records: Dict[str, Dict[str, Any]] = {}
    for table_id in sorted(validation_tables | holdout_tables):
        native = adapter.resolve_table(table_id)
        artifact = adapter.canonicalize_table(native)
        if artifact["dataset"] != "HiTab":
            raise MaterializationError(
                f"canonical_hitab_dataset_mismatch:{table_id}"
            )
        filename = _safe_artifact_name(table_id)
        path = cache_root / filename
        file_sha256 = _write_canonical_json(path, artifact)
        cache_records[table_id] = {
            "table_id": table_id,
            "table_artifact": filename,
            "table_artifact_sha256": file_sha256,
            "canonical_content_sha256": canonical_json_sha256(artifact),
            "source_path": str(native.source_path),
            "source_sha256": native.source_sha256,
            "structure_summary": artifact["structure_summary"],
        }

    def runtime_rows(rows: Sequence[Mapping[str, str]]) -> list[Dict[str, str]]:
        result = []
        for source in rows:
            record = cache_records[source["table_id"]]
            result.append(
                {
                    "dataset": source["dataset"],
                    "id": source["id"],
                    "question": source["question"],
                    "table_id": source["table_id"],
                    "table_artifact": record["table_artifact"],
                    "table_artifact_sha256": record[
                        "table_artifact_sha256"
                    ],
                }
            )
        return result

    validation_path = (
        output / "data" / "hitab" / "validation_runtime_v3.jsonl"
    )
    holdout_path = (
        output / "data" / "hitab" / "holdout_runtime_v3.jsonl"
    )
    validation_sha256 = _write_jsonl(
        validation_path,
        runtime_rows(validation),
    )
    holdout_sha256 = _write_jsonl(
        holdout_path,
        runtime_rows(holdout),
    )
    manifest = {
        "schema_version": "certa_hitab_canonical_table_manifest_v1",
        "adapter_id": adapter.adapter_id,
        "cache_root": str(cache_root.relative_to(output)),
        "canonical_table_count": len(cache_records),
        "tables": [
            cache_records[table_id]
            for table_id in sorted(cache_records)
        ],
        "validation": {
            "source_path": str(validation_source.resolve()),
            "source_sha256": _sha256(validation_source),
            "runtime_path": str(validation_path.relative_to(output)),
            "runtime_sha256": validation_sha256,
            "row_count": len(validation),
            "unique_table_count": len(validation_tables),
        },
        "holdout": {
            "source_path": str(holdout_source.resolve()),
            "source_sha256": _sha256(holdout_source),
            "runtime_path": str(holdout_path.relative_to(output)),
            "runtime_sha256": holdout_sha256,
            "row_count": len(holdout),
            "unique_table_count": len(holdout_tables),
        },
        "validation_holdout_id_overlap": id_overlap,
        "validation_holdout_table_overlap": table_overlap,
        "all_tables_resolved": len(cache_records)
        == len(validation_tables | holdout_tables),
    }
    _write_json(
        output
        / "data"
        / "hitab"
        / "CANONICAL_TABLE_MANIFEST.json",
        manifest,
    )
    return manifest


def _discovery_audit(bindings: Mapping[str, Any]) -> str:
    datasets = bindings["datasets"]
    cohort = bindings["cohort_resolution"]
    return f"""# CERTA dataset discovery audit

## Authoritative root

The operator-authoritative native dataset root is
`{bindings["operator_authoritative_dataset_root"]}`.

## HiTab

- Root: `{datasets["hitab"]["root"]}`
- Native table files: {datasets["hitab"]["table_count"]}
- Identity rule: filename stem; `<root>/<table_id>.json`
- `table_source` is provenance only and is never interpreted as a path.
- Native index SHA256: `{datasets["hitab"]["table_index_sha256"]}`
- Existing graph compatibility is obtained by retaining native
  `row_index`/`column_index` and adding deterministic `row`/`column`
  aliases. Declared tree/span coordinates extend the empty graph grid when
  necessary; the untouched native grid remains in `native_payload`.

## AIT-QA

- Root: `{datasets["aitqa"]["root"]}`
- Raw structural source: `{datasets["aitqa"]["raw_source_path"]}`
- Raw source SHA256: `{datasets["aitqa"]["raw_source_sha256"]}`
- Raw table identities: {datasets["aitqa"]["table_count"]}
- Clean evaluator identities: {datasets["aitqa"]["clean_table_count"]}
- The raw object is structural authority. The versioned projection must
  exactly reproduce the clean flattened table for every clean table ID.
  Extra orphan row headers remain in `native_payload` but do not fabricate
  data rows.

## SSTQA_zh

- Root: `{datasets["sstqa_zh"]["root"]}`
- Workbook root: `{datasets["sstqa_zh"]["workbook_root"]}`
- Workbooks: {datasets["sstqa_zh"]["workbook_count"]}
- Deterministic sheet identities: {datasets["sstqa_zh"]["table_count"]}
- Workbook question IDs resolve to the workbook active sheet. Explicit
  sheet IDs bind workbook ID, zero-based sheet index, and the first 16 hex
  characters of SHA256(NFC(sheet title)).
- Native coordinates, Unicode, formulas, cached values, formats, styles,
  merges, row heights, column widths, workbook identity, and sheet identity
  are preserved. The graph view declares a geometry-only
  `minimal_exact_groundable_header_bands_v1` projection and makes no native
  semantic-header claim.

## Cohort resolution

- Development: {cohort["development"]["resolved_table_count"]}/{
    cohort["development"]["row_count"]
  } resolved.
- Validation: {cohort["validation"]["resolved_table_count"]}/{
    cohort["validation"]["row_count"]
  } resolved.
- Holdout: {cohort["holdout"]["resolved_table_count"]}/{
    cohort["holdout"]["row_count"]
  } resolved.
- Validation–holdout ID overlap: {
    len(cohort["validation_holdout_id_overlap"])
  }.
- Validation–holdout table overlap: {
    len(cohort["validation_holdout_table_overlap"])
  }.

## Label boundary

Discovery parses the native public question-container files enumerated in
`native_question_container_access`. Those containers also carry the recorded
label keys; only the listed question/identity/provenance fields are projected,
and no label value is copied into an adapter, prompt, Planner view, sentinel,
or inference artifact. These native public containers are not the sealed final
validation or holdout label sources. The strict-v2 runtimes are read only for
the four authorized fields `dataset`, `id`, `question`, and `table_id`; no
validation or holdout label source is read.

## Sentinel claim boundary

The sentinel round trip is a deterministic structural LOOKUP fixture selected
from the graph. It proves serialization, graph construction, Planner-domain
exposure, exact local grounding, closure, projection, and provenance plumbing.
It does not claim that the recorded natural-language sentinel question is
answered or semantically groundable, and it is not QA-performance evidence.
"""


def _materialize_into(
    *,
    repo: Path,
    pack: Path,
    dataset_root: Path,
    development: Path,
    strict_v2: Path,
    output: Path,
    sentinel_ids: Mapping[str, Sequence[str]],
    minimum_sentinels: int,
) -> Dict[str, Any]:
    hitab_root = dataset_root / "hitab" / "tables" / "raw"
    aitqa_root = dataset_root / "AIT-QA"
    sstqa_root = dataset_root / "SSTQA-zh"
    hitab = HiTabAdapterV1(hitab_root)
    aitqa = AITQAAdapterV1(aitqa_root)
    sstqa = SSTQAZhAdapterV1(sstqa_root)
    discoveries = {
        "hitab": hitab.discover(),
        "aitqa": aitqa.discover(),
        "sstqa_zh": sstqa.discover(),
    }

    index_paths = {}
    index_hashes = {}
    for name, adapter in (
        ("hitab", hitab),
        ("aitqa", aitqa),
        ("sstqa_zh", sstqa),
    ):
        path = output / "data" / name / "TABLE_INDEX.jsonl"
        index_hashes[name] = _write_jsonl(
            path,
            _dataset_index_records(adapter),
        )
        index_paths[name] = str(path.relative_to(output))

    development_runtime = development / "development_runtime.jsonl"
    validation_source = strict_v2 / "fresh_validation_runtime_v2.jsonl"
    holdout_source = strict_v2 / "fresh_holdout_runtime_v2.jsonl"
    cohort_resolution = {
        "development": _cohort_resolution(hitab, development_runtime),
        "validation": _cohort_resolution(hitab, validation_source),
        "holdout": _cohort_resolution(hitab, holdout_source),
    }
    validation_rows = _project_blind_runtime(validation_source)
    holdout_rows = _project_blind_runtime(holdout_source)
    cohort_resolution["validation_holdout_id_overlap"] = sorted(
        {row["id"] for row in validation_rows}
        & {row["id"] for row in holdout_rows}
    )
    cohort_resolution["validation_holdout_table_overlap"] = sorted(
        {row["table_id"] for row in validation_rows}
        & {row["table_id"] for row in holdout_rows}
    )
    if not all(
        cohort_resolution[name]["all_resolved"]
        for name in ("development", "validation", "holdout")
    ):
        raise MaterializationError("hitab_cohort_table_resolution_failed")
    if (
        cohort_resolution["validation_holdout_id_overlap"]
        or cohort_resolution["validation_holdout_table_overlap"]
    ):
        raise MaterializationError(
            "validation_holdout_disjointness_failed"
        )

    hitab_questions = _question_index(
        dataset_root / "hitab" / "test_samples_clean.jsonl",
        question_fields=("question",),
    )
    aitqa_questions = _question_index(
        aitqa_root / "test_samples.jsonl",
        question_fields=("question",),
    )
    sstqa_questions = _question_index(
        sstqa_root / "test.jsonl",
        question_fields=("query", "question"),
    )
    native_question_containers = [
        {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "container_read": True,
            "projected_fields": list(projected_fields),
            "known_label_keys_excluded": list(label_keys),
            "label_values_used_for_inference": False,
            "scientific_split_status": split_status,
        }
        for path, projected_fields, label_keys, split_status in (
            (
                dataset_root / "hitab" / "test_samples_clean.jsonl",
                ("table_id", "question", "table_source"),
                ("answer", "answer_formulas"),
                "NATIVE_PUBLIC_QUESTION_CONTAINER_NOT_FINAL_VALIDATION_OR_HOLDOUT",
            ),
            (
                aitqa_root / "test_samples.jsonl",
                ("table_id", "question"),
                ("answers",),
                "NATIVE_PUBLIC_QUESTION_CONTAINER_ADAPTER_SMOKE_ONLY",
            ),
            (
                sstqa_root / "test.jsonl",
                ("table_id", "query", "question"),
                ("label",),
                "NATIVE_PUBLIC_QUESTION_CONTAINER_ADAPTER_SMOKE_ONLY",
            ),
        )
    ]
    sentinel_reports = {
        "hitab": _materialize_sentinels(
            output=output,
            dataset_name="hitab",
            adapter=hitab,
            sentinel_ids=sentinel_ids["hitab"],
            questions=hitab_questions,
            minimum_sentinels=minimum_sentinels,
        ),
        "aitqa": _materialize_sentinels(
            output=output,
            dataset_name="aitqa",
            adapter=aitqa,
            sentinel_ids=sentinel_ids["aitqa"],
            questions=aitqa_questions,
            minimum_sentinels=minimum_sentinels,
        ),
        "sstqa_zh": _materialize_sentinels(
            output=output,
            dataset_name="sstqa_zh",
            adapter=sstqa,
            sentinel_ids=sentinel_ids["sstqa_zh"],
            questions=sstqa_questions,
            minimum_sentinels=minimum_sentinels,
        ),
    }
    canonical_manifest = _materialize_hitab_runtimes(
        output=output,
        adapter=hitab,
        validation_source=validation_source,
        holdout_source=holdout_source,
    )

    loader_paths = {
        name: repo / relative
        for name, relative in {
            "versioned_adapter": (
                "certa/active_v1/dataset_adapter_v1.py"
            ),
            "legacy_dataset_adapter": "dataset_adapters.py",
            "sstqa_native_converter": "certa/datasets/sstqa_zh.py",
            "graph_builder": "graph_builder.py",
            "planner_view": "certa/planner/schema_view.py",
        }.items()
    }
    bindings = {
        "schema_version": "certa_dataset_root_bindings_v1",
        "operator_authoritative_dataset_root": str(
            dataset_root.resolve()
        ),
        "adapter_contract_path": str(
            (pack / "DATASET_ADAPTER_CONTRACT.md").resolve()
        ),
        "adapter_contract_sha256": _sha256(
            pack / "DATASET_ADAPTER_CONTRACT.md"
        ),
        "loaders": {
            name: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
            for name, path in loader_paths.items()
        },
        "datasets": {
            "hitab": {
                "root": str(hitab_root.resolve()),
                "format": "JSON",
                "table_id_rule": "filename_stem",
                "source_alias_rule": "provenance_only_not_path",
                "table_count": len(hitab.index_tables()),
                "table_index_path": index_paths["hitab"],
                "table_index_sha256": index_hashes["hitab"],
                "question_source_path": str(
                    (
                        dataset_root
                        / "hitab"
                        / "test_samples_clean.jsonl"
                    ).resolve()
                ),
                "question_source_sha256": _sha256(
                    dataset_root
                    / "hitab"
                    / "test_samples_clean.jsonl"
                ),
                "adapter_id": hitab.adapter_id,
            },
            "aitqa": {
                "root": str(aitqa_root.resolve()),
                "format": "JSONL raw structural tables",
                "table_id_rule": (
                    "raw_row.table_id_equals_raw_table.id"
                ),
                "table_count": len(aitqa.index_tables()),
                "clean_table_count": discoveries["aitqa"][
                    "clean_table_count"
                ],
                "table_index_path": index_paths["aitqa"],
                "table_index_sha256": index_hashes["aitqa"],
                "raw_source_path": str(aitqa.raw_path.resolve()),
                "raw_source_sha256": _sha256(aitqa.raw_path),
                "clean_source_path": str(aitqa.clean_path.resolve()),
                "clean_source_sha256": _sha256(aitqa.clean_path),
                "adapter_id": aitqa.adapter_id,
            },
            "sstqa_zh": {
                "root": str(sstqa_root.resolve()),
                "workbook_root": str(sstqa.workbook_root.resolve()),
                "format": "XLSX",
                "table_id_rule": discoveries["sstqa_zh"][
                    "table_id_rule"
                ],
                "question_alias_rule": discoveries["sstqa_zh"][
                    "question_alias_rule"
                ],
                "workbook_count": discoveries["sstqa_zh"][
                    "workbook_count"
                ],
                "table_count": len(sstqa.index_tables()),
                "table_index_path": index_paths["sstqa_zh"],
                "table_index_sha256": index_hashes["sstqa_zh"],
                "question_source_path": str(
                    (sstqa_root / "test.jsonl").resolve()
                ),
                "question_source_sha256": _sha256(
                    sstqa_root / "test.jsonl"
                ),
                "adapter_id": sstqa.adapter_id,
            },
        },
        "cohort_resolution": cohort_resolution,
        "sentinel_reports": {
            name: {
                "path": (
                    f"data/{name}/ADAPTER_SENTINEL_REPORT.json"
                ),
                "pass": report["pass"],
                "sentinel_count": report["sentinel_count"],
            }
            for name, report in sentinel_reports.items()
        },
        "canonical_runtime_manifest": (
            "data/hitab/CANONICAL_TABLE_MANIFEST.json"
        ),
        "canonical_runtime_table_count": canonical_manifest[
            "canonical_table_count"
        ],
        "native_question_container_access": native_question_containers,
        "adapter_materializer": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
    }
    _write_json(
        output / "data" / "DATASET_ROOT_BINDINGS.json",
        bindings,
    )
    audit_path = output / "data" / "DATASET_DISCOVERY_AUDIT.md"
    audit_path.write_text(
        _discovery_audit(bindings),
        encoding="utf-8",
    )

    identity = {
        "schema_version": "certa_repository_runtime_identity_v1",
        "required_baseline_commit": BASELINE_COMMIT,
        "head_commit": _git(repo, "rev-parse", "HEAD"),
        "head_tree": _git(repo, "rev-parse", "HEAD^{tree}"),
        "branch": _git(repo, "branch", "--show-current"),
        "upstream": _git(
            repo,
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
            allow_failure=True,
        ),
        "required_branch": FINAL_BRANCH,
        "worktree_status_porcelain": _git(
            repo,
            "status",
            "--porcelain",
        ).splitlines(),
        "pack_root": str(pack.resolve()),
        "pack_sha256s_path": str((pack / "SHA256SUMS.txt").resolve()),
        "pack_sha256s_sha256": _sha256(pack / "SHA256SUMS.txt"),
        "dataset_root": str(dataset_root.resolve()),
        "output_root_was_absent": True,
        "endpoint_calls": 0,
        "label_sources_read": [],
        "native_label_bearing_containers_read": [
            item["path"] for item in native_question_containers
        ],
        "native_container_label_values_used_for_inference": False,
    }
    _write_json(
        output / "intake" / "repository_runtime_identity.json",
        identity,
    )
    return {
        "status": "PASS",
        "output": str(output),
        "dataset_root_bindings": (
            "data/DATASET_ROOT_BINDINGS.json"
        ),
        "canonical_runtime_manifest": (
            "data/hitab/CANONICAL_TABLE_MANIFEST.json"
        ),
        "sentinel_reports": {
            name: report["pass"]
            for name, report in sentinel_reports.items()
        },
    }


def materialize_adapter_stage(
    *,
    repo: Path,
    pack: Path,
    dataset_root: Path,
    development: Path,
    strict_v2: Path,
    output: Path,
    sentinel_ids: Mapping[str, Sequence[str]] = DEFAULT_SENTINELS,
    minimum_sentinels: int = 8,
) -> Dict[str, Any]:
    """Create the complete adapter-stage output only when the root is absent."""
    repo = Path(repo).resolve()
    pack = Path(pack).resolve()
    dataset_root = Path(dataset_root).resolve()
    development = Path(development).resolve()
    strict_v2 = Path(strict_v2).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise OutputRootExistsError(f"output_root_exists:{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.staging.",
            dir=output.parent,
        )
    )
    try:
        result = _materialize_into(
            repo=repo,
            pack=pack,
            dataset_root=dataset_root,
            development=development,
            strict_v2=strict_v2,
            output=staging,
            sentinel_ids=sentinel_ids,
            minimum_sentinels=minimum_sentinels,
        )
        staging.rename(output)
        result["output"] = str(output)
        return result
    except Exception:
        shutil.rmtree(staging)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--pack", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--development", required=True)
    parser.add_argument("--strict-v2", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = materialize_adapter_stage(
        repo=Path(args.repo),
        pack=Path(args.pack),
        dataset_root=Path(args.dataset_root),
        development=Path(args.development),
        strict_v2=Path(args.strict_v2),
        output=Path(args.output),
    )
    print(canonical_json(result))


if __name__ == "__main__":
    main()
