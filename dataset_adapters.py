"""
Dataset adapters for the CSCR pipeline.

The core pipeline consumes a HiTab-like table object.  AIT-QA is kept as a
first-class source schema, then compiled into that internal contract at load
time so existing graph, executor, and certificate code can be reused.
"""

import json
import os
import re
import string
from typing import Any, Dict, Iterable, List, Optional

from certa.datasets.sstqa_zh import canonical_table_to_hitab_like, evaluate_sstqa_answer


def normalize_dataset_name(name: str) -> str:
    key = (name or "hitab").strip().lower().replace("_", "-")
    aliases = {
        "hi-tab": "hitab",
        "hitab": "hitab",
        "ait": "aitqa",
        "ait-qa": "aitqa",
        "aitqa": "aitqa",
        "sstqa-zh": "sstqa_zh",
        "sstqazh": "sstqa_zh",
        "table-bench": "tablebench",
        "tablebench": "tablebench",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported dataset: {name}")
    return aliases[key]


def normalize_item_for_cscr(item: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    dataset = normalize_dataset_name(dataset)
    out = dict(item)
    out["dataset"] = dataset
    if dataset == "aitqa":
        out["answer"] = item.get("answers", item.get("answer", []))
        out["dataset_question_type"] = item.get("type", "")
        out["row_hierarchy_needed"] = item.get("row_hierarchy_needed", "")
        out["paraphrase_group"] = item.get("paraphrase_group", "")
    elif dataset == "sstqa_zh":
        out["question"] = item.get("query", item.get("question", ""))
        out["answer"] = item.get("label", item.get("answer", ""))
        out["dataset_question_type"] = item.get("type", "")
        out["dataset_difficulty"] = item.get("difficulty", "")
    elif dataset == "tablebench":
        out["answer"] = item.get("answer", item.get("gold_answer", []))
        out["dataset_question_type"] = item.get("qtype", "")
        out["dataset_question_subtype"] = item.get("qsubtype", "")
    else:
        out.setdefault("answer", item.get("answer", item.get("answers", [])))
    return out


def load_table_for_cscr(
    item: Dict[str, Any],
    table_dir: str,
    cache: Dict[str, Dict[str, Any]],
    dataset: str,
) -> Dict[str, Any]:
    dataset = normalize_dataset_name(dataset)
    table_id = item.get("table_id") or item.get("context", "")
    cache_key = f"{dataset}:{table_id or item.get('id', '')}"
    if cache_key in cache:
        return cache[cache_key]

    if dataset == "aitqa":
        raw_table = item.get("table") or _load_aitqa_table_from_dir(table_id, table_dir)
        table = aitqa_table_to_hitab_like(raw_table, table_id=table_id)
        cache[cache_key] = table
        return table
    if dataset == "sstqa_zh":
        table = _load_sstqa_table(table_id, table_dir)
        cache[cache_key] = table
        return table
    if dataset == "tablebench":
        table = tablebench_table_to_hitab_like(item.get("table") or {}, table_id=table_id or item.get("id", ""))
        cache[cache_key] = table
        return table

    table = _load_hitab_table(table_id, table_dir)
    cache[cache_key] = table
    return table


def _load_hitab_table(table_id: str, table_dir: str) -> Dict[str, Any]:
    candidates = [
        os.path.join(table_dir, f"{table_id}.json"),
        os.path.join(table_dir, table_id),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return empty_hitab_like_table(table_id=table_id)


def _load_aitqa_table_from_dir(table_id: str, table_dir: str) -> Dict[str, Any]:
    candidates = [
        os.path.join(table_dir, f"{table_id}.json"),
        os.path.join(table_dir, table_id),
        os.path.join(table_dir, "aitqa_tables.jsonl"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("id") == table_id:
                        return row
        else:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return {}


def _load_sstqa_table(table_id: str, table_dir: str) -> Dict[str, Any]:
    path = os.path.join(table_dir, f"{table_id}.json")
    if not os.path.exists(path):
        return empty_hitab_like_table(table_id=table_id)
    with open(path, "r", encoding="utf-8") as handle:
        return canonical_table_to_hitab_like(json.load(handle))


def empty_hitab_like_table(table_id: str = "") -> Dict[str, Any]:
    return {
        "id": table_id,
        "texts": [],
        "top_root": _root_node(),
        "left_root": _root_node(),
        "merged_regions": [],
        "top_header_rows_num": 1,
        "left_header_columns_num": 1,
    }


def aitqa_table_to_hitab_like(raw_table: Dict[str, Any], table_id: str = "") -> Dict[str, Any]:
    if not raw_table:
        return empty_hitab_like_table(table_id=table_id)
    if isinstance(raw_table, list):
        rows = _as_rows(raw_table)
        if not rows:
            return empty_hitab_like_table(table_id=table_id)
        raw_table = {
            "id": table_id,
            "column_header": [[cell] for cell in rows[0]],
            "row_header": [],
            "data": rows[1:],
        }
    if not isinstance(raw_table, dict):
        return empty_hitab_like_table(table_id=table_id)

    column_header = _as_rows(raw_table.get("column_header", []))
    row_header = _as_rows(raw_table.get("row_header", []))
    data = _as_rows(raw_table.get("data", []))

    has_row_header = bool(row_header)
    left_cols = max((len(r) for r in row_header), default=0) if has_row_header else 1
    left_cols = max(left_cols, 1)

    data_col_start = 0 if has_row_header else 1
    data_col_count = _infer_data_column_count(data, column_header, data_col_start)
    header_cols = column_header[data_col_start:data_col_start + data_col_count]
    top_rows = max((len(h) for h in header_cols), default=1)
    top_rows = max(top_rows, 1)

    texts: List[List[str]] = []
    for level in range(top_rows):
        row = ["" for _ in range(left_cols)]
        for header in header_cols:
            row.append(header[level] if level < len(header) else "")
        texts.append(row)

    for ridx, data_row in enumerate(data):
        if has_row_header:
            left = _pad(row_header[ridx] if ridx < len(row_header) else [], left_cols)
            values = data_row[:data_col_count]
        else:
            left = [data_row[0] if data_row else ""]
            values = data_row[data_col_start:data_col_start + data_col_count]
        texts.append(_pad(left, left_cols) + _pad(values, data_col_count))

    width = max((len(r) for r in texts), default=left_cols + data_col_count)
    texts = [_pad(r, width) for r in texts]

    table = {
        "id": raw_table.get("id", table_id),
        "table_id": raw_table.get("id", table_id),
        "title": raw_table.get("title", raw_table.get("caption", "")),
        "source_format": "aitqa",
        "texts": texts,
        "top_root": _build_top_root(top_rows, left_cols, data_col_count),
        "left_root": _build_left_root(top_rows, left_cols, len(data), row_header if has_row_header else None),
        "merged_regions": [],
        "top_header_rows_num": top_rows,
        "left_header_columns_num": left_cols,
        "aitqa_meta": {
            "original_column_count": len(column_header),
            "original_row_count": len(data),
            "has_explicit_row_header": has_row_header,
        },
    }
    return table


def tablebench_table_to_hitab_like(raw_table: Dict[str, Any], table_id: str = "") -> Dict[str, Any]:
    if not raw_table:
        return empty_hitab_like_table(table_id=table_id)
    columns = [_cell_text(c) for c in raw_table.get("columns", [])]
    data = _as_rows(raw_table.get("data", []))
    if not columns and data:
        columns = [f"col_{i}" for i in range(len(data[0]))]

    left_cols = 1 if columns else 0
    data_col_start = 1 if left_cols else 0
    value_columns = columns[data_col_start:]
    if not value_columns and columns:
        value_columns = columns
        data_col_start = 0
        left_cols = 0

    texts: List[List[str]] = []
    texts.append(([""] if left_cols else []) + value_columns)
    for row in data:
        left = [row[0] if row and left_cols else ""] if left_cols else []
        values = row[data_col_start:data_col_start + len(value_columns)]
        texts.append(left + _pad(values, len(value_columns)))

    width = max((len(r) for r in texts), default=len(value_columns) + left_cols)
    texts = [_pad(r, width) for r in texts]
    return {
        "id": table_id,
        "table_id": table_id,
        "title": raw_table.get("title", raw_table.get("caption", "")) if isinstance(raw_table, dict) else "",
        "source_format": "tablebench",
        "texts": texts,
        "top_root": _build_top_root(1, left_cols, len(value_columns)),
        "left_root": _build_left_root(1, left_cols, len(data), None) if left_cols else _root_node(),
        "merged_regions": [],
        "top_header_rows_num": 1,
        "left_header_columns_num": left_cols,
        "tablebench_meta": {
            "original_column_count": len(columns),
            "original_row_count": len(data),
            "first_column_as_row_header": bool(left_cols),
        },
    }


def aitqa_answer_match(ground_truth: Any, prediction: Any) -> bool:
    if isinstance(ground_truth, list):
        return any(aitqa_answer_match(gt, prediction) for gt in ground_truth)
    norm_gt = normalize_aitqa_answer(ground_truth)
    norm_pred = normalize_aitqa_answer(prediction)
    try:
        return abs(float(norm_gt) - float(norm_pred)) < 1e-6
    except (TypeError, ValueError, OverflowError):
        return norm_gt == norm_pred


def normalize_aitqa_answer(answer: Any) -> str:
    if isinstance(answer, list):
        return "|".join(normalize_aitqa_answer(a) for a in answer)
    raw = str(answer).lower().strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1].strip()
    raw = raw.strip("\"'")
    text = raw
    text = re.sub(r"\b(million|millions|billion|billions|thousand|thousands|gallon|gallons)\b", "", text)
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]
    text = text.replace("$", "").replace(",", "").replace("%", "").strip()
    try:
        value = float(text)
        if value == int(value):
            return str(int(value))
        return str(value)
    except (TypeError, ValueError, OverflowError):
        text = raw.replace("$", "").replace("%", "")
    text = re.sub(r"[\[\]\(\){}\"'`]", " ", text)
    text = re.sub(r"[^\w\s.-]", " ", text)
    return " ".join(text.split())


def tablebench_answer_match(ground_truth: Any, prediction: Any) -> bool:
    if isinstance(ground_truth, list):
        return any(tablebench_answer_match(gt, prediction) for gt in ground_truth)
    norm_gt = normalize_tablebench_answer(ground_truth)
    norm_pred = normalize_tablebench_answer(prediction)
    try:
        return abs(float(norm_gt) - float(norm_pred)) < 1e-6
    except (TypeError, ValueError, OverflowError):
        return norm_gt == norm_pred


def normalize_tablebench_answer(answer: Any) -> str:
    if not answer:
        return ""
    text = str(answer).lower().strip()
    number_match = re.search(r"([-+]?[\d,]+\.?\d*)", text)
    if number_match:
        num_str = number_match.group(1).replace(",", "")
        try:
            num_val = float(num_str)
            if num_val == int(num_val):
                return str(int(num_val))
            return str(num_val)
        except (TypeError, ValueError, OverflowError):
            return num_str

    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    exclude = set(string.punctuation)
    return white_space_fix(remove_articles("".join(ch for ch in text if ch not in exclude)))


def dataset_answer_match(dataset: str, ground_truth: Any, prediction: Any) -> bool:
    dataset = normalize_dataset_name(dataset)
    if dataset == "aitqa":
        return aitqa_answer_match(ground_truth, prediction)
    if dataset == "tablebench":
        return tablebench_answer_match(ground_truth, prediction)
    if dataset == "sstqa_zh":
        return evaluate_sstqa_answer(prediction, ground_truth)["official_match"]
    return False


def _as_rows(value: Any) -> List[List[str]]:
    if not isinstance(value, list):
        return []
    rows: List[List[str]] = []
    for row in value:
        if isinstance(row, list):
            rows.append([_cell_text(x) for x in row])
        else:
            rows.append([_cell_text(row)])
    return rows


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _pad(values: Iterable[str], size: int) -> List[str]:
    row = list(values)
    if len(row) < size:
        row.extend([""] * (size - len(row)))
    return row[:size]


def _infer_data_column_count(data: List[List[str]], column_header: List[List[str]], data_col_start: int) -> int:
    max_data = max((max(0, len(row) - data_col_start) for row in data), default=0)
    max_header = max(0, len(column_header) - data_col_start)
    return max(1, min(max_data or max_header, max_header or max_data or 1))


def _root_node() -> Dict[str, Any]:
    return {"row": -1, "column": -1, "row_index": -1, "column_index": -1, "children": []}


def _node(row: int, col: int, children: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    return {
        "row": row,
        "column": col,
        "row_index": row,
        "column_index": col,
        "children": children or [],
    }


def _build_top_root(top_rows: int, left_cols: int, data_col_count: int) -> Dict[str, Any]:
    root = _root_node()
    for offset in range(data_col_count):
        col = left_cols + offset
        parent = None
        for level in range(top_rows - 1, -1, -1):
            parent = _node(level, col, [parent] if parent else [])
        if parent:
            root["children"].append(parent)
    return root


def _build_left_root(
    top_rows: int,
    left_cols: int,
    data_rows: int,
    row_header: Optional[List[List[str]]],
) -> Dict[str, Any]:
    root = _root_node()
    for ridx in range(data_rows):
        row = top_rows + ridx
        parent = None
        depth = left_cols if row_header is not None else 1
        for col in range(depth - 1, -1, -1):
            parent = _node(row, col, [parent] if parent else [])
        if parent:
            root["children"].append(parent)
    return root
