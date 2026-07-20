"""Blind row projections required by the CERTA-EGRA constructor experiment."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from eval_utils import hitab_official_em

from certa.egra.query_role_contract import request_query_role_contract
from certa.reproducibility.canonical_json import canonical_json_hash


RUNTIME_FIELDS = {"dataset", "id", "question", "table_id", "table_source"}


def freeze_role_contract_rows(
    runtime_rows: Sequence[Mapping[str, Any]],
    generator: Any,
    existing_rows: Sequence[Mapping[str, Any]] = (),
) -> list[Dict[str, Any]]:
    existing_by_id = {
        str(row.get("sample_id") or ""): dict(row)
        for row in existing_rows
    }
    if len(existing_by_id) != len(existing_rows):
        raise ValueError("duplicate_existing_role_freeze_id")
    frozen = []
    for source_order, runtime in enumerate(runtime_rows):
        if set(runtime) != RUNTIME_FIELDS:
            raise ValueError(f"role_runtime_fields_mismatch:{sorted(runtime)}")
        sample_id = str(runtime.get("id") or "")
        table_id = str(runtime.get("table_id") or "")
        question = str(runtime.get("question") or "")
        if not sample_id or not table_id or not question:
            raise ValueError(f"invalid_role_runtime_identity:{sample_id}")
        question_sha256 = canonical_json_hash({"question": question})
        row = existing_by_id.get(sample_id)
        if row is None:
            validation, audit = request_query_role_contract(generator, question)
            row = {
                "schema_version": "certa_egra_role_freeze_row_v1",
                "sample_id": sample_id,
                "table_id": table_id,
                "source_order": source_order,
                "question_sha256": question_sha256,
                "status": (
                    "VALID"
                    if validation.ok and validation.normalized_payload.get(
                        "supported_by_core_signatures"
                    )
                    else "UNSUPPORTED"
                    if validation.ok
                    else "INVALID"
                ),
                "contract": dict(validation.normalized_payload),
                "errors": list(validation.errors),
                "audit": audit,
            }
        if (
            row.get("schema_version") != "certa_egra_role_freeze_row_v1"
            or str(row.get("table_id") or "") != table_id
            or str(row.get("question_sha256") or "") != question_sha256
        ):
            raise ValueError(f"existing_role_freeze_mismatch:{sample_id}")
        frozen.append(dict(row))
    return frozen


def freeze_b0_rows(
    runtime_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    predictions = {
        str(row.get("id") or ""): row
        for row in prediction_rows
    }
    if len(predictions) != len(prediction_rows):
        raise ValueError("duplicate_b0_prediction_id")
    expected_ids = {str(row.get("id") or "") for row in runtime_rows}
    if set(predictions) != expected_ids:
        raise ValueError("b0_prediction_sample_set_mismatch")
    frozen = []
    for source_order, runtime in enumerate(runtime_rows):
        if set(runtime) != RUNTIME_FIELDS:
            raise ValueError(f"b0_runtime_fields_mismatch:{sorted(runtime)}")
        sample_id = str(runtime["id"])
        table_id = str(runtime["table_id"])
        question = str(runtime["question"])
        prediction = predictions[sample_id]
        if (
            str(prediction.get("table_id") or "") != table_id
            or str(prediction.get("question") or "") != question
        ):
            raise ValueError(f"b0_prediction_identity_mismatch:{sample_id}")
        llm_answer = str(prediction.get("llm_answer") or "")
        if str(prediction.get("final_answer") or "") != llm_answer:
            raise ValueError(f"b0_prediction_mutated:{sample_id}")
        raw_output = str(prediction.get("llm_raw_output") or "")
        if not raw_output.strip() or not llm_answer.strip():
            raise ValueError(f"b0_prediction_empty:{sample_id}")
        frozen.append({
            "schema_version": "certa_egra_b0_freeze_v1",
            "sample_id": sample_id,
            "table_id": table_id,
            "source_order": source_order,
            "question_sha256": canonical_json_hash({"question": question}),
            "b0_answer_sha256": canonical_json_hash({"answer": llm_answer}),
            "generation": {
                "text": raw_output,
                "logprobs": None,
                "generation_seconds": float(
                    prediction.get("llm_generation_seconds", 0.0) or 0.0
                ),
                "generated_token_count": int(
                    prediction.get("generated_token_count", 0) or 0
                ),
                "black_box_api": bool(
                    prediction.get("black_box_api_generator", False)
                ),
                "api_model": str(prediction.get("api_model") or ""),
                "api_base_url": str(prediction.get("api_base_url") or ""),
                "api_key_env": str(prediction.get("api_key_env") or ""),
                "generator_backend": str(prediction.get("generator_backend") or ""),
                "api_usage": dict(prediction.get("api_usage") or {}),
                "api_cache_hit": bool(prediction.get("api_cache_hit", False)),
                "api_cache_mode": str(prediction.get("api_cache_mode") or ""),
                "chat_template_kwargs": dict(
                    prediction.get("chat_template_kwargs") or {}
                ),
            },
        })
    return frozen


def build_constructor_sample_rows(
    runtime_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
) -> list[Dict[str, Any]]:
    if split not in {"dev", "holdout", "r2_failure_replay"}:
        raise ValueError(f"invalid_constructor_split:{split}")
    predictions = {
        str(row.get("id") or ""): row
        for row in prediction_rows
    }
    if len(predictions) != len(prediction_rows):
        raise ValueError("duplicate_constructor_prediction_id")
    expected_ids = {str(row.get("id") or "") for row in runtime_rows}
    if set(predictions) != expected_ids:
        raise ValueError("constructor_prediction_sample_set_mismatch")

    rows = []
    for source_order, runtime in enumerate(runtime_rows):
        if set(runtime) != RUNTIME_FIELDS:
            raise ValueError(f"constructor_runtime_fields_mismatch:{sorted(runtime)}")
        sample_id = str(runtime["id"])
        prediction = predictions[sample_id]
        table_id = str(runtime["table_id"])
        question = str(runtime["question"])
        if (
            str(prediction.get("table_id") or "") != table_id
            or str(prediction.get("question") or "") != question
        ):
            raise ValueError(f"constructor_prediction_identity_mismatch:{sample_id}")

        arm = str(prediction.get("certa_egra_arm") or "")
        contract = dict(prediction.get("certa_egra_role_contract") or {})
        role_valid = bool(prediction.get("certa_egra_role_contract_valid"))
        supported = (
            True
            if arm == "C0_FLAT_SCHEMA_CURRENT"
            else bool(contract.get("supported_by_core_signatures"))
        )
        role_status = (
            "NOT_RUN"
            if arm == "C0_FLAT_SCHEMA_CURRENT"
            else "VALID"
            if role_valid and supported
            else "UNSUPPORTED"
            if role_valid
            else "INVALID"
        )
        retrieval = dict(prediction.get("certa_egra_retrieval") or {})
        retrieval_config = {
            key: retrieval.get(key)
            for key in ("budgets", "similarity_threshold", "stable_tie_break")
            if key in retrieval
        }
        derivation_count = int(
            prediction.get("cera_planner_derivation_count", 0) or 0
        )
        outcome_counts = dict(
            prediction.get("cera_round9_closure_outcome_counts") or {}
        )
        unique_resolution = int(
            outcome_counts.get("UNIQUE_EXECUTABLE", 0) or 0
        ) > 0
        original_count = int(
            prediction.get("cera_round9_partition_original_count", 0) or 0
        )
        alternative_count = int(
            prediction.get("cera_round9_partition_alternative_count", 0) or 0
        )
        closure_records = list(
            prediction.get("cera_round10_closure_audit_records") or []
        )
        complete_records = [
            record
            for record in closure_records
            if record.get("resource_complete")
            and str(record.get("canonical_program_id") or "")
            and list(record.get("provenance_ids") or [])
        ]
        registry_outside_count = max(0, derivation_count - len(complete_records))
        signature_candidates = set(contract.get("signature_candidates") or [])
        role_incompatible = 0
        if arm != "C0_FLAT_SCHEMA_CURRENT" and supported:
            role_incompatible = sum(
                1
                for record in (
                    prediction.get("cera_round12_semantic_type_audit_records") or []
                )
                if record.get("closure_outcome") == "UNIQUE_EXECUTABLE"
                and str(record.get("signature_id") or "") not in signature_candidates
            )
        paired = original_count > 0 and alternative_count > 0
        registry_ready = (
            paired
            and derivation_count > 0
            and len(complete_records) == derivation_count
            and registry_outside_count == 0
        )
        runtime_leakage = bool(
            set(prediction)
            & {
                "answer",
                "answers",
                "gold",
                "gold_answer",
                "expected_answer",
                "aggregation",
                "correctness",
            }
        ) or bool(prediction.get("cera_planner_proposal_visible_to_planner")) or bool(
            prediction.get("cera_planner_table_values_visible_to_planner")
        )

        failure_reasons = []
        failure_stage = ""
        if role_status == "INVALID":
            failure_stage = "QUERY_ROLE_CONTRACT"
            failure_reasons.extend(
                str(reason)
                for reason in prediction.get("certa_egra_role_contract_errors", [])
            )
        elif role_status == "UNSUPPORTED":
            failure_stage = "UNSUPPORTED_BY_CORE_SIGNATURES"
        elif prediction.get("certa_egra_construction_error"):
            failure_stage = "STRUCTURAL_CARD_PROJECTION"
            failure_reasons.append(str(prediction["certa_egra_construction_error"]))
        elif prediction.get("cera_planner_generation_error"):
            failure_stage = "PLANNER_GENERATION"
            failure_reasons.append(str(prediction["cera_planner_generation_error"]))
        elif not prediction.get("cera_planner_validation_ok"):
            failure_stage = "PLANNER_VALIDATION"
            failure_reasons.extend(
                str(reason)
                for reason in prediction.get("cera_planner_validation_errors", [])
            )
        elif not unique_resolution:
            failure_stage = "OPERAND_RESOLUTION"
            failure_reasons.extend(
                str(reason)
                for record in prediction.get("cera_planner_compile_failures", [])
                for reason in record.get("failure_reasons", [])
            )

        rows.append({
            "sample_id": sample_id,
            "table_id": table_id,
            "split": split,
            "arm": arm,
            "source_order": source_order,
            "role_contract_status": role_status,
            "role_contract_hash": canonical_json_hash(contract) if contract else "",
            "supported_by_core_signatures": supported,
            "answer_domain": str(contract.get("answer_domain") or "UNSUPPORTED"),
            "intent_family": str(contract.get("intent_family") or "UNSUPPORTED"),
            "retrieval_config_hash": (
                canonical_json_hash(retrieval_config) if retrieval_config else ""
            ),
            "retrieved_card_ids": list(retrieval.get("selected_card_ids") or []),
            "reference_node_ids": list(retrieval.get("reference_node_ids") or []),
            "planner_request_hash": str(
                prediction.get("cera_planner_request_hash") or ""
            ),
            "planner_valid_plan_count": int(
                prediction.get("cera_planner_valid_plan_count", 0) or 0
            ),
            "unique_operand_resolution": unique_resolution,
            "executable_derivation_count": derivation_count,
            "original_support": original_count > 0,
            "alternative_support": alternative_count > 0,
            "paired_executable": paired,
            "constructor_registry_ready": registry_ready,
            "role_incompatible_executable_count": role_incompatible,
            "registry_outside_answer_count": registry_outside_count,
            "b0_mutation": str(prediction.get("final_answer") or "") != str(
                prediction.get("llm_answer") or ""
            ),
            "runtime_leakage": runtime_leakage,
            "planner_prompt_tokens": int(
                prediction.get("cera_planner_input_tokens", 0) or 0
            ),
            "planner_latency_seconds": float(
                prediction.get("cera_planner_latency_seconds", 0.0) or 0.0
            ),
            "gold_join_status": "NOT_ACCESSED",
            "gold_answer_in_executable_space_postfreeze": None,
            "oracle_repairable_postfreeze": None,
            "failure_stage": failure_stage,
            "failure_reasons": failure_reasons,
            "b0_answer": str(prediction.get("llm_answer") or ""),
            "executable_answers": [
                str(record.get("projected_answer") or "")
                for record in complete_records
                if str(record.get("projected_answer") or "").strip()
            ],
        })
    return rows


def unblind_constructor_sample_rows(
    blind_rows: Sequence[Mapping[str, Any]],
    gold_rows: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    gold_by_id = {
        str(row.get("sample_id") or ""): row
        for row in gold_rows
    }
    if len(gold_by_id) != len(gold_rows):
        raise ValueError("duplicate_constructor_gold_id")
    expected_ids = {str(row.get("sample_id") or "") for row in blind_rows}
    if set(gold_by_id) != expected_ids:
        raise ValueError("constructor_gold_sample_set_mismatch")
    result = []
    for blind in blind_rows:
        row = dict(blind)
        sample_id = str(row.get("sample_id") or "")
        gold = gold_by_id[sample_id]
        if str(gold.get("table_id") or "") != str(row.get("table_id") or ""):
            raise ValueError(f"constructor_gold_table_mismatch:{sample_id}")
        gold_answer = gold.get("gold_answer")
        executable_answers = list(row.get("executable_answers") or [])
        gold_in_space = any(
            hitab_official_em(answer, gold_answer)
            for answer in executable_answers
        )
        b0_correct = hitab_official_em(row.get("b0_answer", ""), gold_answer)
        row.update({
            "gold_join_status": "JOINED_POSTFREEZE",
            "gold_answer_sha256": canonical_json_hash({"gold_answer": gold_answer}),
            "gold_answer_in_executable_space_postfreeze": gold_in_space,
            "oracle_repairable_postfreeze": gold_in_space and not b0_correct,
            "b0_correct_postfreeze": b0_correct,
        })
        result.append(row)
    return result
