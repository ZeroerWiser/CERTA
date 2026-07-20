#!/usr/bin/env python3
"""Blind cohort preparation and artifact helpers for CERTA-EGRA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from certa.egra.artifacts import (
    build_constructor_sample_rows,
    freeze_b0_rows,
    freeze_role_contract_rows,
    unblind_constructor_sample_rows,
)
from certa.egra.query_role_contract import (
    QUERY_ROLE_CONTRACT_VERSION,
    QUERY_ROLE_MAX_TOKENS,
    build_query_role_response_schema,
)
from certa.reproducibility.canonical_json import canonical_json, canonical_json_hash


SEED = 20260720
DOMAIN = b"certa-egra-cohort-v1\0"
RUNTIME_FIELDS = ("dataset", "id", "question", "table_id", "table_source")
FORBIDDEN_SELECTION_FIELDS = {
    "answer",
    "answers",
    "gold",
    "gold_answer",
    "aggregation",
    "answer_formulas",
    "linked_cells",
    "reference_cells_map",
    "operation",
    "question_operation",
    "correctness",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical_json(dict(row)) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    return [
        dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sanitize_egra_source(source_path: Path, output_path: Path) -> Dict[str, Any]:
    """Project a raw source into a label-free identity file in a separate step."""
    rows = []
    seen_ids = set()
    for source in _read_jsonl(Path(source_path)):
        row = {field: source.get(field) for field in RUNTIME_FIELDS}
        row["dataset"] = str(row.get("dataset") or "hitab")
        for field in ("id", "question", "table_id"):
            if not str(row.get(field) or ""):
                raise ValueError(f"invalid_source_identity:{field}")
        sample_id = str(row["id"])
        if sample_id in seen_ids:
            raise ValueError(f"duplicate_sample_id:{sample_id}")
        seen_ids.add(sample_id)
        rows.append(row)
    _write_jsonl(Path(output_path), rows)
    return {
        "row_count": len(rows),
        "output_sha256": _file_sha256(Path(output_path)),
        "runtime_fields": list(RUNTIME_FIELDS),
    }


def _normalize_table_value(value: Any, *, top_level: bool = False) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_table_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
            if not (top_level and str(key) == "title")
        }
    if isinstance(value, list):
        return [_normalize_table_value(item) for item in value]
    if isinstance(value, str):
        return " ".join(unicodedata.normalize("NFKC", value).split())
    return value


def _table_content_sha256(path: Path) -> str:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return canonical_json_hash(_normalize_table_value(raw, top_level=True))


def _stable_table_hash(table_id: str, seed: int) -> str:
    value = unicodedata.normalize("NFC", str(table_id)).encode("utf-8")
    return hashlib.sha256(
        DOMAIN + b"table\0" + str(seed).encode("ascii") + b"\0" + value
    ).hexdigest()


def _stable_sample_hash(sample_id: str, table_id: str, seed: int) -> str:
    table_value = unicodedata.normalize("NFC", str(table_id)).encode("utf-8")
    sample_value = unicodedata.normalize("NFC", str(sample_id)).encode("utf-8")
    return hashlib.sha256(
        DOMAIN
        + b"sample\0"
        + str(seed).encode("ascii")
        + b"\0"
        + table_value
        + b"\0"
        + sample_value
    ).hexdigest()


def _project_source(
    path: Path,
    table_root: Path,
    *,
    split: str,
    seed: int,
    table_hash_cache: Dict[str, str],
) -> list[Dict[str, Any]]:
    projected = []
    for source in _read_jsonl(path):
        forbidden = sorted(set(source) & FORBIDDEN_SELECTION_FIELDS)
        if forbidden:
            raise ValueError(f"selection_source_contains_forbidden_fields:{forbidden}")
        if set(source) != set(RUNTIME_FIELDS):
            raise ValueError(
                f"selection_source_fields_mismatch:{sorted(source)}"
            )
        sample_id = str(source.get("id") or "")
        table_id = str(source.get("table_id") or "")
        question = str(source.get("question") or "")
        if not sample_id or not table_id or not question:
            raise ValueError(f"invalid_source_identity:{split}:{sample_id}:{table_id}")
        table_path = table_root / f"{table_id}.json"
        if not table_path.is_file():
            raise FileNotFoundError(f"missing_raw_table:{table_id}")
        table_content_sha256 = table_hash_cache.setdefault(
            table_id,
            _table_content_sha256(table_path),
        )
        projected.append({
            "sample_id": sample_id,
            "table_id": table_id,
            "table_content_sha256": table_content_sha256,
            "table_stable_hash": _stable_table_hash(table_id, seed),
            "stable_hash": _stable_sample_hash(sample_id, table_id, seed),
            "runtime": {
                "dataset": "hitab",
                "id": sample_id,
                "question": question,
                "table_id": table_id,
                "table_source": str(source.get("table_source") or ""),
            },
        })
    return projected


def _class_representatives(
    rows: Sequence[Mapping[str, Any]],
    *,
    excluded_table_ids: set[str],
    excluded_content_hashes: set[str],
) -> list[Dict[str, Any]]:
    by_content: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["table_id"] in excluded_table_ids:
            continue
        if row["table_content_sha256"] in excluded_content_hashes:
            continue
        by_content[str(row["table_content_sha256"])].append(row)
    representatives = []
    for content_hash, members in by_content.items():
        table_ids = {str(item["table_id"]) for item in members}
        representative_table_id = min(
            table_ids,
            key=lambda table_id: (_stable_table_hash(table_id, SEED), table_id),
        )
        table_rows = [item for item in members if item["table_id"] == representative_table_id]
        representative = min(
            table_rows,
            key=lambda item: (str(item["stable_hash"]), str(item["sample_id"])),
        )
        row = dict(representative)
        row["class_min_table_hash"] = min(
            str(item["table_stable_hash"]) for item in members
        )
        row["table_content_sha256"] = content_hash
        representatives.append(row)
    return sorted(
        representatives,
        key=lambda item: (
            str(item["class_min_table_hash"]),
            str(item["table_stable_hash"]),
            str(item["stable_hash"]),
            str(item["table_id"]),
            str(item["sample_id"]),
        ),
    )


def _fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    text = "".join(
        f"{index}\t{row['sample_id']}\t{row['table_id']}\t{row['stable_hash']}\n"
        for index, row in enumerate(rows)
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepare_egra_cohorts(
    *,
    dev_source: Path,
    train_source: Path,
    table_root: Path,
    historical_cohort_paths: Sequence[Path],
    output_root: Path,
    sealed_gold_root: Path,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Freeze 64 dev and 64 alias-safe holdout rows without using labels."""
    if seed != SEED:
        raise ValueError(f"seed_mismatch:{seed}")
    dev_source = Path(dev_source)
    train_source = Path(train_source)
    table_root = Path(table_root)
    output_root = Path(output_root)
    sealed_gold_root = Path(sealed_gold_root)

    historical_table_ids: set[str] = set()
    for path in historical_cohort_paths:
        for row in _read_jsonl(Path(path)):
            table_id = str(row.get("table_id") or "")
            if not table_id:
                raise ValueError(f"historical_row_missing_table_id:{path}")
            historical_table_ids.add(table_id)
    table_hash_cache: Dict[str, str] = {}
    historical_content_hashes = {
        table_hash_cache.setdefault(
            table_id,
            _table_content_sha256(table_root / f"{table_id}.json"),
        )
        for table_id in historical_table_ids
    }
    dev_rows = _project_source(
        dev_source,
        table_root,
        split="dev",
        seed=seed,
        table_hash_cache=table_hash_cache,
    )
    train_rows = _project_source(
        train_source,
        table_root,
        split="train",
        seed=seed,
        table_hash_cache=table_hash_cache,
    )
    sample_ids = [str(row["sample_id"]) for row in (*dev_rows, *train_rows)]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("duplicate_sample_id")

    dev_candidates = _class_representatives(
        dev_rows,
        excluded_table_ids=historical_table_ids,
        excluded_content_hashes=historical_content_hashes,
    )
    if len(dev_candidates) < 64:
        raise ValueError(f"insufficient_dev_content_classes:{len(dev_candidates)}")
    selected_dev = dev_candidates[:64]
    selected_dev_content = {
        str(row["table_content_sha256"]) for row in selected_dev
    }
    holdout_candidates = _class_representatives(
        train_rows,
        excluded_table_ids=historical_table_ids,
        excluded_content_hashes=historical_content_hashes | selected_dev_content,
    )
    if len(holdout_candidates) < 64:
        raise ValueError(
            f"insufficient_holdout_content_classes:{len(holdout_candidates)}"
        )
    selected_holdout = holdout_candidates[:64]

    dev_members = [{
        "sample_id": row["sample_id"],
        "table_id": row["table_id"],
        "stable_hash": row["stable_hash"],
        "source_order": index,
        "source_split": "dev",
        "table_content_sha256": row["table_content_sha256"],
    } for index, row in enumerate(selected_dev)]
    holdout_members = [{
        "sample_id": row["sample_id"],
        "table_id": row["table_id"],
        "stable_hash": row["stable_hash"],
        "source_order": index,
        "source_split": "train",
        "table_content_sha256": row["table_content_sha256"],
    } for index, row in enumerate(selected_holdout)]
    manifest = {
        "schema_version": "certa_egra_cohort_v1",
        "seed": seed,
        "dev_ids": [str(row["sample_id"]) for row in selected_dev],
        "holdout_ids": [str(row["sample_id"]) for row in selected_holdout],
        "dev_table_ids": [str(row["table_id"]) for row in selected_dev],
        "holdout_table_ids": [str(row["table_id"]) for row in selected_holdout],
        "r1_r2_excluded_table_ids": sorted(historical_table_ids),
        "table_disjoint": True,
        "selection_fields": ["sample_id", "table_id", "stable_hash"],
        "gold_blind": True,
    }
    dev_fingerprint = _fingerprint(selected_dev)
    holdout_fingerprint = _fingerprint(selected_holdout)
    _write_json(output_root / "freeze/COHORT_MANIFEST.json", manifest)
    _write_jsonl(output_root / "freeze/DEV_COHORT.jsonl", dev_members)
    _write_jsonl(
        output_root / "freeze/HOLDOUT_COHORT_SEALED.jsonl",
        holdout_members,
    )
    _write_jsonl(
        output_root / "inputs/dev_runtime.jsonl",
        (row["runtime"] for row in selected_dev),
    )
    _write_jsonl(
        sealed_gold_root / "holdout_runtime.jsonl",
        (row["runtime"] for row in selected_holdout),
    )
    raw_records = [
        f"{_file_sha256(path)}  {path.name}\n"
        for path in sorted(table_root.glob("*.json"), key=lambda item: item.name)
    ]
    audit = {
        "schema_version": "certa_egra_cohort_selection_audit_v1",
        "seed": seed,
        "domain_separator": DOMAIN.decode("utf-8"),
        "source_mapping": {"dev": str(dev_source), "holdout": str(train_source)},
        "source_sha256": {
            "dev": _file_sha256(dev_source),
            "train": _file_sha256(train_source),
        },
        "raw_table_tree_sha256": hashlib.sha256(
            "".join(raw_records).encode("utf-8")
        ).hexdigest(),
        "historical_paths": [str(Path(path)) for path in historical_cohort_paths],
        "historical_table_count": len(historical_table_ids),
        "historical_content_class_count": len(historical_content_hashes),
        "normalization": "recursive_NFKC_whitespace_collapse_drop_top_level_title",
        "selection_uses_answer_or_operation": False,
        "runtime_fields": list(RUNTIME_FIELDS),
        "dev_candidate_class_count": len(dev_candidates),
        "holdout_candidate_class_count_after_dev_block": len(holdout_candidates),
        "dev_fingerprint": dev_fingerprint,
        "first_24_fingerprint": _fingerprint(selected_dev[:24]),
        "holdout_fingerprint": holdout_fingerprint,
        "dev_holdout_table_id_overlap": 0,
        "dev_holdout_content_class_overlap": 0,
        "manifest_sha256": canonical_json_hash(manifest),
    }
    _write_json(output_root / "audit/COHORT_SELECTION_AUDIT.json", audit)
    return {
        "dev_count": len(selected_dev),
        "holdout_count": len(selected_holdout),
        "dev_fingerprint": dev_fingerprint,
        "holdout_fingerprint": holdout_fingerprint,
        "manifest_sha256": audit["manifest_sha256"],
    }


def seal_egra_gold(
    *,
    dev_source: Path,
    train_source: Path,
    output_root: Path,
    sealed_gold_root: Path,
) -> Dict[str, Any]:
    """Exact-join labels only after blind cohort selection has completed."""
    output_root = Path(output_root)
    sealed_gold_root = Path(sealed_gold_root)
    selected = {
        "dev": _read_jsonl(output_root / "freeze/DEV_COHORT.jsonl"),
        "holdout": _read_jsonl(output_root / "freeze/HOLDOUT_COHORT_SEALED.jsonl"),
    }
    sources = {
        "dev": _read_jsonl(Path(dev_source)),
        "holdout": _read_jsonl(Path(train_source)),
    }
    result = {}
    for split in ("dev", "holdout"):
        by_id: Dict[str, Dict[str, Any]] = {}
        for row in sources[split]:
            sample_id = str(row.get("id") or "")
            if not sample_id or sample_id in by_id:
                raise ValueError(f"invalid_gold_source_id:{split}:{sample_id}")
            by_id[sample_id] = row
        gold_rows = []
        for member in selected[split]:
            sample_id = str(member["sample_id"])
            source = by_id.get(sample_id)
            if source is None:
                raise ValueError(f"gold_join_missing:{split}:{sample_id}")
            if str(source.get("table_id") or "") != str(member["table_id"]):
                raise ValueError(f"gold_join_table_mismatch:{split}:{sample_id}")
            gold_rows.append({
                "sample_id": sample_id,
                "table_id": str(member["table_id"]),
                "gold_answer": source.get("answer"),
            })
        path = sealed_gold_root / f"{split}_gold.jsonl"
        temp_path = path.with_name(f".{path.name}.tmp")
        _write_jsonl(temp_path, gold_rows)
        os.chmod(temp_path, 0o440)
        os.replace(temp_path, path)
        result[f"{split}_count"] = len(gold_rows)
        result[f"{split}_sha256"] = _file_sha256(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    sanitize = subparsers.add_parser("sanitize-source")
    sanitize.add_argument("--source", type=Path, required=True)
    sanitize.add_argument("--output", type=Path, required=True)
    prepare = subparsers.add_parser("prepare-cohorts")
    prepare.add_argument("--dev-source", type=Path, required=True)
    prepare.add_argument("--train-source", type=Path, required=True)
    prepare.add_argument("--table-root", type=Path, required=True)
    prepare.add_argument("--historical", type=Path, action="append", required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--sealed-gold-root", type=Path, required=True)
    seal = subparsers.add_parser("seal-gold")
    seal.add_argument("--dev-source", type=Path, required=True)
    seal.add_argument("--train-source", type=Path, required=True)
    seal.add_argument("--output-root", type=Path, required=True)
    seal.add_argument("--sealed-gold-root", type=Path, required=True)
    role = subparsers.add_parser("freeze-role")
    role.add_argument("--runtime", type=Path, required=True)
    role.add_argument("--output", type=Path, required=True)
    role.add_argument("--manifest", type=Path, required=True)
    role.add_argument("--cache", type=Path, required=True)
    role.add_argument("--limit", type=int)
    role.add_argument("--resume", action="store_true")
    b0 = subparsers.add_parser("freeze-b0")
    b0.add_argument("--runtime", type=Path, required=True)
    b0.add_argument("--predictions", type=Path, required=True)
    b0.add_argument("--output", type=Path, required=True)
    b0.add_argument("--limit", type=int)
    b0.add_argument("--replace", action="store_true")
    constructor = subparsers.add_parser("constructor-master")
    constructor.add_argument("--runtime", type=Path, required=True)
    constructor.add_argument("--predictions", type=Path, action="append", required=True)
    constructor.add_argument(
        "--split",
        choices=["dev", "holdout", "r2_failure_replay"],
        required=True,
    )
    constructor.add_argument("--output", type=Path, required=True)
    unblind = subparsers.add_parser("unblind-constructor")
    unblind.add_argument("--blind", type=Path, required=True)
    unblind.add_argument("--gold", type=Path, required=True)
    unblind.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "sanitize-source":
        result = sanitize_egra_source(args.source, args.output)
    elif args.command == "prepare-cohorts":
        result = prepare_egra_cohorts(
            dev_source=args.dev_source,
            train_source=args.train_source,
            table_root=args.table_root,
            historical_cohort_paths=args.historical,
            output_root=args.output_root,
            sealed_gold_root=args.sealed_gold_root,
        )
    elif args.command == "seal-gold":
        result = seal_egra_gold(
            dev_source=args.dev_source,
            train_source=args.train_source,
            output_root=args.output_root,
            sealed_gold_root=args.sealed_gold_root,
        )
    elif args.command == "freeze-role":
        runtime_rows = _read_jsonl(args.runtime)
        if args.limit is not None:
            runtime_rows = runtime_rows[:args.limit]
        if args.output.exists() and not args.resume:
            raise FileExistsError(f"refusing_to_overwrite_role_freeze:{args.output}")
        existing = _read_jsonl(args.output) if args.output.exists() else []
        from run_cscr_pipeline import OpenAIChatGenerator

        generator = OpenAIChatGenerator(
            model="Qwen3-8B",
            api_base_url="http://127.0.0.1:30338/v1",
            api_key_env="EMPTY",
            timeout=120.0,
            max_retries=0,
            rate_limit_seconds=0.0,
            max_model_len=32768,
            cache_path=str(args.cache),
            cache_mode="readwrite",
            backend_name="vllm_chat",
        )
        rows = freeze_role_contract_rows(runtime_rows, generator, existing)
        _write_jsonl(args.output, rows)
        manifest = {
            "schema_version": "certa_egra_role_contract_freeze_v1",
            "contract_version": QUERY_ROLE_CONTRACT_VERSION,
            "schema_sha256": canonical_json_hash(build_query_role_response_schema()),
            "model": "Qwen3-8B",
            "backend": "vllm_chat",
            "api_base_url": "http://127.0.0.1:30338/v1",
            "thinking": {"enable_thinking": False},
            "sampling": {
                "max_tokens": QUERY_ROLE_MAX_TOKENS,
                "temperature": 0.0,
                "top_p": 1.0,
            },
            "sample_count": len(rows),
            "logical_calls": sum(
                int((row.get("audit") or {}).get("calls", 0) or 0)
                for row in rows
            ),
            "valid_count": sum(row.get("status") == "VALID" for row in rows),
            "unsupported_count": sum(
                row.get("status") == "UNSUPPORTED" for row in rows
            ),
            "invalid_count": sum(row.get("status") == "INVALID" for row in rows),
            "records_sha256": _file_sha256(args.output),
        }
        _write_json(args.manifest, manifest)
        result = manifest
    elif args.command == "freeze-b0":
        runtime_rows = _read_jsonl(args.runtime)
        if args.limit is not None:
            runtime_rows = runtime_rows[:args.limit]
        prediction_rows = _read_jsonl(args.predictions)
        rows = freeze_b0_rows(runtime_rows, prediction_rows)
        if args.output.exists() and not args.replace:
            raise FileExistsError(f"refusing_to_overwrite_b0_freeze:{args.output}")
        _write_jsonl(args.output, rows)
        result = {
            "schema_version": "certa_egra_b0_freeze_summary_v1",
            "sample_count": len(rows),
            "records_sha256": _file_sha256(args.output),
        }
    elif args.command == "constructor-master":
        runtime_rows = _read_jsonl(args.runtime)
        rows = []
        for path in args.predictions:
            prediction_rows = _read_jsonl(path)
            rows.extend(build_constructor_sample_rows(
                runtime_rows[:len(prediction_rows)],
                prediction_rows,
                split=args.split,
            ))
        _write_jsonl(args.output, rows)
        result = {
            "schema_version": "certa_egra_constructor_master_summary_v1",
            "row_count": len(rows),
            "records_sha256": _file_sha256(args.output),
        }
    else:
        rows = unblind_constructor_sample_rows(
            _read_jsonl(args.blind),
            _read_jsonl(args.gold),
        )
        _write_jsonl(args.output, rows)
        result = {
            "schema_version": "certa_egra_constructor_unblind_summary_v1",
            "row_count": len(rows),
            "records_sha256": _file_sha256(args.output),
        }
    print(canonical_json(result))


if __name__ == "__main__":
    main()
