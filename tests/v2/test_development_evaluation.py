from certa.v2.evaluation import analyze_development


def exact_match(prediction, gold) -> bool:
    return prediction == gold


def sample(
    sample_id: str,
    table_id: str,
    b0: str,
    alternative: str | None,
    selected: dict[str, str],
) -> dict:
    candidates = [
        {
            "candidate_id": f"{sample_id}-b0",
            "candidate_source": "B0",
            "candidate_answer": b0,
        }
    ]
    proofs = [
        {
            "candidate_id": f"{sample_id}-b0",
            "candidate_source": "B0",
            "overall_state": "UNKNOWN",
        }
    ]
    registry = []
    if alternative is not None:
        candidates.append(
            {
                "candidate_id": f"{sample_id}-alt",
                "candidate_source": "EXECUTED_REGISTRY",
                "candidate_answer": alternative,
            }
        )
        proofs.append(
            {
                "candidate_id": f"{sample_id}-alt",
                "candidate_source": "EXECUTED_REGISTRY",
                "overall_state": "PASS",
            }
        )
        registry.append({"executed_answer": alternative})
    return {
        "sample_id": sample_id,
        "table_id": table_id,
        "b0_answer": b0,
        "role": {"role_id": "COUNT_SCALAR"},
        "candidates": candidates,
        "registry": registry,
        "proofs": proofs,
        "proof_failures": [],
        "selected_finals": selected,
        "decisions": {
            variant: {
                "action": "REPAIR" if answer != b0 else "KEEP_B0",
                "validator_approved": True,
            }
            for variant, answer in selected.items()
        },
    }


def test_development_analysis_separates_oracle_recall_from_selection() -> None:
    variants = (
        "V2-A_EXECUTABLE_SEARCH_ONLY",
        "V2-B_PROOF_DOMINANCE",
        "V2-C_PROOF_VERIFIER",
    )
    rows = [
        sample(
            "s1",
            "t1",
            "wrong",
            "gold",
            {
                variants[0]: "gold",
                variants[1]: "wrong",
                variants[2]: "wrong",
            },
        ),
        sample(
            "s2",
            "t2",
            "right",
            None,
            {variant: "right" for variant in variants},
        ),
    ]
    oracle, metrics = analyze_development(
        rows,
        {"s1": "gold", "s2": "right"},
        match=exact_match,
        call_metrics={"calls": 10, "tokens": 100, "latency_seconds": 20.0},
    )
    assert oracle["b0_wrong_rows"] == 1
    assert oracle["correct_alternative_rows"] == 1
    assert oracle["correct_alternative_oracle_recall"] == 1.0
    assert oracle["answer_class_recall"]["registry_alternative_rows"] == 1
    assert metrics[variants[0]]["WC"] == 1
    assert metrics[variants[0]]["selected_correct"] == 2
    assert metrics[variants[1]]["WC"] == 0
    assert metrics[variants[1]]["selected_correct"] == 1
    assert metrics[variants[0]]["alternative_proof_states"]["PASS"] == 1
    assert metrics[variants[0]]["proof_coverage"]["observed"] == 3
    assert metrics[variants[0]]["calls_per_WC"] == 10.0
