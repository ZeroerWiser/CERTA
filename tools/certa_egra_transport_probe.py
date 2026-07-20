#!/usr/bin/env python3
"""Run the frozen CERTA-EGRA transport probe and role-record requests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping
from urllib.request import ProxyHandler, Request, build_opener

from certa.egra.query_role_contract import (
    FROZEN_API_BASE_URL,
    FROZEN_BACKEND,
    FROZEN_MODEL,
    FROZEN_THINKING,
    QueryRoleValidation,
    build_query_role_prompt,
    build_query_role_response_schema,
    request_query_role_contract,
    validate_query_role_contract,
)
from certa.egra.transport_schema import build_query_role_transport_schema
from certa.egra.transport_schema import transport_adapter_sha256
from certa.reproducibility.canonical_json import canonical_json, canonical_json_hash


SYNTHETIC_QUESTION = "What is the population of North?"
SOURCE_SHA = "1cba72ebf1316bdde3f7ceeba70b6934202ca864"
COHORT_SALT = "CERTA-EGRA-v0.1|20260720|dev64-prefix"


def _schema_hashes() -> tuple[str, str]:
    semantic = build_query_role_response_schema()
    return (
        canonical_json_hash(semantic),
        canonical_json_hash(build_query_role_transport_schema(semantic)),
    )


def _classification(validation: QueryRoleValidation) -> str:
    if not validation.ok:
        return "INVALID"
    return (
        "VALID"
        if validation.normalized_payload.get("supported_by_core_signatures")
        else "UNSUPPORTED"
    )


def _identity_drift(audit: Mapping[str, Any]) -> bool:
    return {
        "model": audit.get("model"),
        "backend": audit.get("backend"),
        "api_base_url": audit.get("api_base_url"),
        "thinking": audit.get("thinking"),
    } != {
        "model": FROZEN_MODEL,
        "backend": FROZEN_BACKEND,
        "api_base_url": FROZEN_API_BASE_URL,
        "thinking": FROZEN_THINKING,
    }


def _hash_drift(
    audit: Mapping[str, Any], expected_prompt_sha256: str | None
) -> bool:
    semantic_hash, transport_hash = _schema_hashes()
    return (
        audit.get("semantic_schema_sha256") != semantic_hash
        or audit.get("transport_schema_sha256") != transport_hash
        or audit.get("structured_output_schema_sha256") != transport_hash
        or audit.get("adapter_sha256") != transport_adapter_sha256()
        or not expected_prompt_sha256
        or audit.get("prompt_sha256") != expected_prompt_sha256
        or audit.get("structured_output_mechanism")
        != "response_format.type=json_schema"
        or bool(audit.get("transport_errors"))
    )


def build_gate_role_row(
    frozen: Mapping[str, Any], *, expected_prompt_sha256: str | None = None
) -> dict[str, Any]:
    """Project a retained frozen role record into the Pack gate vocabulary."""
    audit = dict(frozen.get("audit") or {})
    contract = dict(frozen.get("contract") or {})
    validation = validate_query_role_contract(contract)
    classification = _classification(validation)
    reported_classification = str(frozen.get("status") or "")
    return {
        "sample_id": str(frozen.get("sample_id") or ""),
        "classification": classification,
        "http_completed": audit.get("http_completed") is True,
        "json_parse_ok": audit.get("parse_ok") is True and validation.parse_ok,
        "fallback_used": audit.get("structured_output_fallback_used") is not False,
        "identity_drift": _identity_drift(audit),
        "hash_drift": (
            _hash_drift(audit, expected_prompt_sha256)
            or reported_classification != classification
        ),
        "question_only_input": frozen.get("question_only_input") is True,
        "answer_domain": contract.get("answer_domain"),
        "intent_family": contract.get("intent_family"),
        "signature_candidates": list(contract.get("signature_candidates") or []),
        "projection_candidates": list(contract.get("projection_candidates") or []),
    }


def build_probe_record(
    *,
    version_payload: Mapping[str, Any],
    models_payload: Mapping[str, Any],
    version_http_status: int,
    models_http_status: int,
    http_status: int,
    validation: QueryRoleValidation,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute the no-data probe result without granting semantic authority."""
    frozen = {
        "sample_id": "SYNTHETIC_NO_DATA_PROBE",
        "status": _classification(validation),
        "contract": dict(validation.normalized_payload),
        "audit": dict(audit),
        "question_only_input": True,
    }
    expected_prompt_sha256 = canonical_json_hash({
        "prompt": build_query_role_prompt(SYNTHETIC_QUESTION)
    })
    gate = build_gate_role_row(
        frozen, expected_prompt_sha256=expected_prompt_sha256
    )
    models = sorted(
        str(item.get("id"))
        for item in models_payload.get("data", [])
        if isinstance(item, Mapping) and item.get("id")
    )
    version = str(version_payload.get("version") or "")
    actual_attempts_value = audit.get("actual_attempts")
    actual_attempts = (
        int(actual_attempts_value)
        if isinstance(actual_attempts_value, (int, float))
        else -1
    )
    passed = all((
        version == "0.11.0",
        models == [FROZEN_MODEL],
        version_http_status == 200,
        models_http_status == 200,
        http_status == 200,
        gate["http_completed"],
        gate["json_parse_ok"],
        not gate["fallback_used"],
        not gate["identity_drift"],
        not gate["hash_drift"],
        actual_attempts == 1,
        (audit.get("cache") or {}).get("hit") is False,
    ))
    return {
        "schema_version": "certa_egra_capability_probe_v1",
        "question_sha256": canonical_json_hash({"question": SYNTHETIC_QUESTION}),
        "version": version,
        "models": models,
        "model": str(audit.get("model") or ""),
        "version_http_status": version_http_status,
        "models_http_status": models_http_status,
        "http_status": http_status,
        "classification": gate["classification"],
        "json_parse_ok": gate["json_parse_ok"],
        "fallback_used": gate["fallback_used"],
        "identity_drift": gate["identity_drift"],
        "hash_drift": gate["hash_drift"],
        "adapter_sha256": audit.get("adapter_sha256"),
        "semantic_schema_sha256": audit.get("semantic_schema_sha256"),
        "transport_schema_sha256": audit.get("transport_schema_sha256"),
        "prompt_sha256": audit.get("prompt_sha256"),
        "endpoint": audit.get("api_base_url"),
        "thinking": audit.get("thinking"),
        "actual_attempts": actual_attempts,
        "audit": dict(audit),
        "pass": passed,
    }


def _generator(cache: Path, *, cache_mode: str = "readwrite"):
    from run_cscr_pipeline import OpenAIChatGenerator

    return OpenAIChatGenerator(
        model=FROZEN_MODEL,
        api_base_url=FROZEN_API_BASE_URL,
        api_key_env="EMPTY",
        timeout=120.0,
        max_retries=0,
        rate_limit_seconds=0.0,
        max_model_len=32768,
        cache_path=str(cache) if cache_mode != "off" else "",
        cache_mode=cache_mode,
        backend_name=FROZEN_BACKEND,
    )


class _ObservedProbeGenerator:
    """One uncached raw HTTP POST whose status is retained as evidence."""

    model = FROZEN_MODEL
    api_base_url = FROZEN_API_BASE_URL
    backend_name = FROZEN_BACKEND
    chat_template_kwargs = FROZEN_THINKING
    cache_mode = "off"

    def __init__(self, opener):
        self.opener = opener
        self.http_status = 0

    def generate_json_schema(
        self, prompt, *, response_schema, schema_name, max_new_tokens,
        temperature, top_p,
    ):
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": response_schema,
                            "strict": True},
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature, "top_p": top_p,
            "max_tokens": max_new_tokens, "stream": False,
            "response_format": response_format,
            "chat_template_kwargs": self.chat_template_kwargs,
        }
        request = Request(
            f"{self.api_base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"), method="POST",
            headers={"Authorization": "Bearer EMPTY",
                     "Content-Type": "application/json"},
        )
        started = time.perf_counter()
        with self.opener.open(request, timeout=120) as response:
            self.http_status = int(response.status)
            raw = json.load(response)
        elapsed = time.perf_counter() - started
        message = ((raw.get("choices") or [{}])[0].get("message") or {})
        usage = raw.get("usage") or {}
        text = str(message.get("content") or "")
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        return {
            "text": text, "input_token_count": prompt_tokens,
            "generated_token_count": completion_tokens,
            "generation_seconds": elapsed, "api_model": self.model,
            "api_base_url": self.api_base_url,
            "generator_backend": self.backend_name,
            "api_cache_hit": False, "api_cache_mode": self.cache_mode,
            "structured_output_requested": True,
            "structured_output_mechanism": "response_format.type=json_schema",
            "structured_output_schema_hash": canonical_json_hash(response_schema),
            "structured_output_fallback_used": False,
            "chat_template_kwargs": self.chat_template_kwargs,
        }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_expected_cohort_freeze(source_cohort: Path) -> dict[str, Any]:
    """Reconstruct the exact 64→24→8 freeze from the sealed dev source."""
    raw = source_cohort.read_bytes()
    rows = [
        json.loads(line) for line in raw.decode("utf-8").splitlines()
        if line.strip()
    ]
    runtime_fields = {"dataset", "id", "question", "table_id", "table_source"}
    if len(rows) != 64:
        raise ValueError(f"source_dev64_count:{len(rows)}")
    if any(set(row) != runtime_fields for row in rows):
        raise ValueError("source_dev64_runtime_fields")
    selected = []
    for row in rows:
        sample_id = str(row.get("id") or "")
        table_id = str(row.get("table_id") or "")
        if not sample_id or not table_id:
            raise ValueError("source_dev64_empty_identity")
        key = hashlib.sha256(
            f"{COHORT_SALT}|{table_id}|{sample_id}".encode("utf-8")
        ).hexdigest()
        selected.append((key, table_id, sample_id))
    if len({item[2] for item in selected}) != 64:
        raise ValueError("source_dev64_duplicate_sample_id")
    selected.sort()

    def subset(items):
        ids = [item[2] for item in items]
        return {
            "count": len(items), "ordered_sample_ids": ids,
            "ordered_selection_keys": [item[0] for item in items],
            "manifest_sha256": hashlib.sha256(
                json.dumps(ids).encode("utf-8")
            ).hexdigest(),
        }

    return {
        "schema_version": "certa_egra_cohort_selection_freeze_v1",
        "source_sha": SOURCE_SHA,
        "source_cohort_sha256": hashlib.sha256(raw).hexdigest(),
        "seed": 20260720, "salt": COHORT_SALT,
        "algorithm": "sort_sha256_salt_table_id_sample_id_then_table_id_sample_id",
        "created_before_role_outputs": True,
        "created_before_planner_outputs": True,
        "created_before_closure_outputs": True,
        "created_before_gold_access": True,
        "dev64": subset(selected),
        "matched24": subset(selected[:24]),
        "transport8": subset(selected[:8]),
    }


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(canonical_json(dict(row)) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def run_probe(output: Path, cache: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing_to_rerun_capability_probe:{output}")
    opener = build_opener(ProxyHandler({}))
    headers = {"Authorization": "Bearer EMPTY"}
    with opener.open(Request("http://127.0.0.1:30338/version", headers=headers), timeout=10) as response:
        version_http_status = int(response.status)
        version_payload = json.load(response)
    with opener.open(Request(f"{FROZEN_API_BASE_URL}/models", headers=headers), timeout=10) as response:
        models_http_status = int(response.status)
        models_payload = json.load(response)
    generator = _ObservedProbeGenerator(opener)
    validation, audit = request_query_role_contract(
        generator, SYNTHETIC_QUESTION
    )
    audit.update({"http_completed": True, "http_status": generator.http_status,
                  "actual_attempts": 1})
    record = build_probe_record(
        version_payload=version_payload,
        models_payload=models_payload,
        version_http_status=version_http_status,
        models_http_status=models_http_status,
        http_status=generator.http_status,
        validation=validation,
        audit=audit,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def run_roles(
    runtime: Path,
    output: Path,
    cache: Path,
    cohort_freeze: Path,
    source_cohort: Path,
    *,
    limit: int | None,
    resume: bool,
) -> list[dict[str, Any]]:
    all_runtime_rows = _read_jsonl(runtime)
    cohort = json.loads(cohort_freeze.read_text(encoding="utf-8"))
    expected_cohort = build_expected_cohort_freeze(source_cohort)
    if cohort != expected_cohort:
        raise ValueError("cohort_freeze_not_exact_source_projection")
    expected_ids = list(cohort["matched24"]["ordered_sample_ids"])
    source_by_id = {
        str(row.get("id") or ""): row for row in _read_jsonl(source_cohort)
    }
    expected_runtime_rows = [source_by_id[sample_id] for sample_id in expected_ids]
    if (
        len(all_runtime_rows) != 24
        or [str(row.get("id") or "") for row in all_runtime_rows] != expected_ids
        or all_runtime_rows != expected_runtime_rows
        or any(
            set(row) != {"dataset", "id", "question", "table_id", "table_source"}
            for row in all_runtime_rows
        )
    ):
        raise ValueError("role_runtime_not_exact_frozen_matched24")
    if limit not in {None, 8, 24}:
        raise ValueError(f"invalid_role_limit:{limit}")
    runtime_rows = all_runtime_rows[:8] if limit == 8 else all_runtime_rows
    existing = _read_jsonl(output) if output.exists() and resume else []
    if output.exists() and not resume:
        raise FileExistsError(f"refusing_to_overwrite_role_rows:{output}")
    existing_by_id = {str(row.get("sample_id") or ""): row for row in existing}
    if len(existing_by_id) != len(existing):
        raise ValueError("duplicate_existing_role_row")
    if len(existing) > len(runtime_rows):
        raise ValueError("existing_role_rows_not_prefix")
    for source_order, frozen in enumerate(existing):
        runtime_row = runtime_rows[source_order]
        sample_id = str(runtime_row.get("id") or "")
        table_id = str(runtime_row.get("table_id") or "")
        question = str(runtime_row.get("question") or "")
        question_hash = canonical_json_hash({"question": question})
        prompt_hash = canonical_json_hash({"prompt": build_query_role_prompt(question)})
        expected_gate = build_gate_role_row(
            frozen, expected_prompt_sha256=prompt_hash
        )
        if (
            str(frozen.get("sample_id") or "") != sample_id
            or str(frozen.get("table_id") or "") != table_id
            or str(frozen.get("question_sha256") or "") != question_hash
            or int(frozen.get("source_order", -1)) != source_order
            or any(frozen.get(key) != value for key, value in expected_gate.items())
        ):
            raise ValueError(f"existing_role_row_identity_mismatch:{sample_id}")
    rows = [dict(row) for row in existing]
    generator = _generator(cache) if len(rows) < len(runtime_rows) else None
    for source_order in range(len(rows), len(runtime_rows)):
        runtime_row = runtime_rows[source_order]
        sample_id = str(runtime_row["id"])
        table_id = str(runtime_row["table_id"])
        question = str(runtime_row["question"])
        question_hash = canonical_json_hash({"question": question})
        prompt_hash = canonical_json_hash({"prompt": build_query_role_prompt(question)})
        try:
            if generator is None:
                raise RuntimeError("role_generator_missing")
            try:
                validation, audit = request_query_role_contract(generator, question)
                cache_hit = bool((audit.get("cache") or {}).get("hit"))
                audit.update({"http_completed": True, "http_status": 200,
                              "actual_attempts": 0 if cache_hit else 1})
            except Exception:
                raise
        except Exception as error:
            validation = QueryRoleValidation(
                False, False, (f"generation_exception:{error}",), {}
            )
            semantic_hash, transport_hash = _schema_hashes()
            audit = {
                "calls": 1, "actual_attempts": 1, "http_completed": False,
                "http_status": 0, "parse_ok": False, "model": FROZEN_MODEL,
                "backend": FROZEN_BACKEND, "api_base_url": FROZEN_API_BASE_URL,
                "thinking": FROZEN_THINKING,
                "semantic_schema_sha256": semantic_hash,
                "transport_schema_sha256": transport_hash,
                "structured_output_schema_sha256": transport_hash,
                "adapter_sha256": transport_adapter_sha256(),
                "prompt_sha256": prompt_hash,
                "structured_output_mechanism": "response_format.type=json_schema",
                "structured_output_fallback_used": False,
                "transport_errors": list(validation.errors),
                "semantic_errors": [], "errors": list(validation.errors),
                "cache": {"hit": False, "mode": "readwrite"},
            }
        frozen = {
            "schema_version": "certa_egra_role_freeze_row_v1",
            "sample_id": sample_id, "table_id": table_id,
            "source_order": source_order, "question_sha256": question_hash,
            "question_only_input": True, "status": _classification(validation),
            "contract": dict(validation.normalized_payload),
            "errors": list(validation.errors), "audit": audit,
        }
        frozen.update(build_gate_role_row(
            frozen, expected_prompt_sha256=prompt_hash
        ))
        rows.append(frozen)
        _write_jsonl(output, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    probe = commands.add_parser("probe")
    probe.add_argument("--output", type=Path, required=True)
    probe.add_argument("--cache", type=Path, required=True)
    roles = commands.add_parser("roles")
    roles.add_argument("--runtime", type=Path, required=True)
    roles.add_argument("--output", type=Path, required=True)
    roles.add_argument("--cache", type=Path, required=True)
    roles.add_argument("--cohort-freeze", type=Path, required=True)
    roles.add_argument("--source-cohort", type=Path, required=True)
    roles.add_argument("--limit", type=int)
    roles.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = (
        run_probe(args.output, args.cache)
        if args.command == "probe"
        else {"row_count": len(run_roles(
            args.runtime, args.output, args.cache, args.cohort_freeze,
            args.source_cohort,
            limit=args.limit, resume=args.resume
        ))}
    )
    print(canonical_json(result))


if __name__ == "__main__":
    main()
