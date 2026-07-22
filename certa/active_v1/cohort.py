"""Gold-blind, alias-safe cohort selection for CERTA Active V1."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from certa.reproducibility.canonical_json import canonical_json_hash


ACTIVE_COHORT_SEED = 20260721
ACTIVE_COHORT_DOMAIN = b"certa-active-v1-cohort-v1\0"
RUNTIME_FIELDS = ("dataset", "id", "question", "table_id", "table_source")


def _normalize(value: Any, *, top_level: bool = False) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if not (top_level and str(key) == "title")
        }
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, str):
        return " ".join(unicodedata.normalize("NFKC", value).split())
    return value


def _table_content_hash(path: Path) -> str:
    return canonical_json_hash(_normalize(json.loads(path.read_text(encoding="utf-8")), top_level=True))


def _stable_hash(kind: str, *values: str) -> str:
    payload = b"\0".join(
        unicodedata.normalize("NFC", value).encode("utf-8") for value in values
    )
    return hashlib.sha256(
        ACTIVE_COHORT_DOMAIN
        + kind.encode("ascii")
        + b"\0"
        + str(ACTIVE_COHORT_SEED).encode("ascii")
        + b"\0"
        + payload
    ).hexdigest()


def _project(
    rows: Sequence[Mapping[str, Any]],
    table_root: Path,
    split: str,
    table_hashes: Dict[str, str],
) -> list[Dict[str, Any]]:
    projected = []
    for source in rows:
        if set(source) != set(RUNTIME_FIELDS):
            raise ValueError(f"identity_fields_mismatch:{sorted(source)}")
        runtime = {field: source.get(field) for field in RUNTIME_FIELDS}
        sample_id = str(runtime["id"] or "")
        table_id = str(runtime["table_id"] or "")
        question = str(runtime["question"] or "")
        if not sample_id or not table_id or not question or runtime["dataset"] != "hitab":
            raise ValueError(f"invalid_identity:{split}:{sample_id}:{table_id}")
        table_path = table_root / f"{table_id}.json"
        if not table_path.is_file():
            raise FileNotFoundError(f"missing_raw_table:{table_id}")
        content_hash = table_hashes.setdefault(table_id, _table_content_hash(table_path))
        projected.append({
            "sample_id": sample_id,
            "table_id": table_id,
            "table_content_sha256": content_hash,
            "table_stable_hash": _stable_hash("table", table_id),
            "stable_hash": _stable_hash("sample", table_id, sample_id),
            "source_split": split,
            "runtime": runtime,
        })
    return projected


def _representatives(
    rows: Sequence[Mapping[str, Any]],
    excluded_table_ids: set[str],
    excluded_content_hashes: set[str],
) -> list[Dict[str, Any]]:
    by_content: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["table_id"] not in excluded_table_ids and row["table_content_sha256"] not in excluded_content_hashes:
            by_content[str(row["table_content_sha256"])].append(row)
    selected = []
    for content_hash, members in by_content.items():
        table_id = min(
            {str(item["table_id"]) for item in members},
            key=lambda value: (_stable_hash("table", value), value),
        )
        representative = min(
            (item for item in members if item["table_id"] == table_id),
            key=lambda item: (str(item["stable_hash"]), str(item["sample_id"])),
        )
        selected.append(dict(representative, table_content_sha256=content_hash))
    return sorted(
        selected,
        key=lambda item: (
            str(item["table_stable_hash"]), str(item["stable_hash"]),
            str(item["table_id"]), str(item["sample_id"]),
        ),
    )


def select_active_cohorts(
    dev_identity_rows: Sequence[Mapping[str, Any]],
    train_identity_rows: Sequence[Mapping[str, Any]],
    table_root: Path,
    historical_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Select dev64/holdout64 using identity-only inputs and fixed seed 20260721."""
    table_root = Path(table_root)
    historical_table_ids = {str(row.get("table_id") or "") for row in historical_rows}
    if "" in historical_table_ids:
        raise ValueError("historical_row_missing_table_id")
    table_hashes: Dict[str, str] = {}
    historical_content_hashes = {
        table_hashes.setdefault(table_id, _table_content_hash(table_root / f"{table_id}.json"))
        for table_id in historical_table_ids
    }
    dev_rows = _project(dev_identity_rows, table_root, "dev", table_hashes)
    train_rows = _project(train_identity_rows, table_root, "train", table_hashes)
    sample_ids = [str(row["sample_id"]) for row in dev_rows + train_rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("duplicate_sample_id")
    dev_candidates = _representatives(dev_rows, historical_table_ids, historical_content_hashes)
    if len(dev_candidates) < 64:
        raise ValueError(f"insufficient_dev_content_classes:{len(dev_candidates)}")
    dev = dev_candidates[:64]
    dev_content_hashes = {str(row["table_content_sha256"]) for row in dev}
    holdout_candidates = _representatives(
        train_rows,
        historical_table_ids | {str(row["table_id"]) for row in dev},
        historical_content_hashes | dev_content_hashes,
    )
    if len(holdout_candidates) < 64:
        raise ValueError(f"insufficient_holdout_content_classes:{len(holdout_candidates)}")
    holdout = holdout_candidates[:64]
    for rows in (dev, holdout):
        for source_order, row in enumerate(rows):
            row["source_order"] = source_order
    return {
        "schema_version": "certa_active_v1_cohort_selection_v1",
        "seed": ACTIVE_COHORT_SEED,
        "domain_separator": ACTIVE_COHORT_DOMAIN.decode("utf-8"),
        "historical_table_ids": sorted(historical_table_ids),
        "historical_content_class_count": len(historical_content_hashes),
        "dev_candidate_class_count": len(dev_candidates),
        "holdout_candidate_class_count": len(holdout_candidates),
        "dev": dev,
        "holdout": holdout,
        "integration16": dev[:16],
    }
