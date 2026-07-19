#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List
from _common_io import pct, read_json_dict as load_json


def model_from_dir(name: str, dataset: str, judge_model: str) -> str:
    prefix = f"{dataset}_"
    suffix = f"_{judge_model}"
    value = name
    if value.startswith(prefix):
        value = value[len(prefix):]
    if value.endswith(suffix):
        value = value[:-len(suffix)]
    return value or name


def row_from_metrics(path: Path) -> Dict[str, Any]:
    m = load_json(path)
    dataset = str(m.get("dataset") or path.parent.name.split("_", 1)[0])
    judge_model = str(m.get("judge_model") or "")
    official_vs = (
        m.get("judge_textual_vs_official")
        or m.get("judge_textual_vs_hitab_official")
        or m.get("judge_textual_vs_aitqa_official")
        or {}
    )
    max_vs = (
        m.get("judge_max_vs_official_max")
        or m.get("judge_max_vs_hitab_official_max")
        or m.get("judge_max_vs_aitqa_official_max")
        or {}
    )
    return {
        "metrics_path": str(path),
        "run_dir": str(path.parent),
        "dataset": dataset,
        "model": model_from_dir(path.parent.name, dataset, judge_model),
        "judge_model": judge_model,
        "total_questions": m.get("total_questions", 0),
        "coverage": m.get("coverage", 0.0),
        "official_textual_accuracy": m.get("Official_textual_accuracy", 0.0),
        "official_max_accuracy": m.get("Official_max_accuracy", 0.0),
        "em_textual_accuracy": m.get("EM_textual_accuracy", 0.0),
        "em_max_accuracy": m.get("EM_max_accuracy", 0.0),
        "llm_textual_accuracy": m.get("LLM_textual_accuracy", 0.0),
        "llm_max_accuracy": m.get("LLM_max_accuracy", 0.0),
        "judge_textual_false_positive": official_vs.get("false_positive_official_wrong_judge_correct", 0),
        "judge_textual_false_negative": official_vs.get("false_negative_official_correct_judge_wrong", 0),
        "judge_max_false_positive": max_vs.get("false_positive_official_wrong_judge_correct", 0),
        "judge_max_false_negative": max_vs.get("false_negative_official_correct_judge_wrong", 0),
        "judge_textual_unparsed_count": m.get("judge_textual_unparsed_count", 0),
        "judge_textual_error_count": m.get("judge_textual_error_count", 0),
        "judge_symbolic_unparsed_count": m.get("judge_symbolic_unparsed_count", 0),
        "judge_symbolic_error_count": m.get("judge_symbolic_error_count", 0),
        "judge_cache_hits": m.get("judge_cache_hits", 0),
        "judge_cache_misses": m.get("judge_cache_misses", 0),
        "judge_prompt_version": m.get("judge_prompt_version", ""),
    }


def collect(paths: List[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        if path.is_file() and path.name == "evaluation_metrics.json":
            rows.append(row_from_metrics(path))
        elif path.is_dir():
            for metrics_path in sorted(path.glob("*/evaluation_metrics.json")):
                rows.append(row_from_metrics(metrics_path))
    rows.sort(key=lambda r: (str(r["dataset"]), str(r["model"])))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def render_markdown(rows: List[Dict[str, Any]]) -> str:
    headers = [
        "Dataset", "Model", "N", "Official", "EM", "LLM Judge", "LLM Max",
        "FP/FN", "Coverage", "Cache", "Parse Err", "Run Dir",
    ]
    lines = ["# ASTRA-Compatible Judge Summary", "", f"Total runs: {len(rows)}", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        parse_errors = (
            int(row.get("judge_textual_unparsed_count", 0) or 0)
            + int(row.get("judge_textual_error_count", 0) or 0)
            + int(row.get("judge_symbolic_unparsed_count", 0) or 0)
            + int(row.get("judge_symbolic_error_count", 0) or 0)
        )
        values = [
            str(row["dataset"]),
            str(row["model"]),
            str(row["total_questions"]),
            pct(row["official_textual_accuracy"]),
            pct(row["em_textual_accuracy"]),
            pct(row["llm_textual_accuracy"]),
            pct(row["llm_max_accuracy"]),
            f"{row['judge_textual_false_positive']}/{row['judge_textual_false_negative']}",
            pct(row["coverage"]),
            f"{row['judge_cache_hits']}/{row['judge_cache_misses']}",
            str(parse_errors),
            f"`{row['run_dir']}`",
        ]
        lines.append("| " + " | ".join(v.replace("|", "/") for v in values) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ASTRA-compatible CSCR evaluation metric files.")
    parser.add_argument("paths", nargs="+", help="Evaluation root directories or evaluation_metrics.json files")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    rows = collect([Path(p) for p in args.paths])
    payload = {"inputs": args.paths, "runs": rows}
    md = render_markdown(rows)
    if args.output_json:
        write_json(Path(args.output_json), payload)
    if args.output_md:
        out = Path(args.output_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
