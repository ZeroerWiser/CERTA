"""
structure_aware_formatter.py — CSCR Phase 1A
结构感知表格格式化 + 问题类型分析 + Logit 熵校准

核心思想：
- 利用 HiTab 表格 JSON 中的 top_root / left_root 层级结构显式标注表头层级
- 利用 merged_regions 标注合并区域
- 根据问题关键词推断操作类型，在 prompt 中给出操作提示
- 提供基于 vLLM logprobs 的归一化熵校准信号
"""

import math
import re
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# 聚合关键词（多语言）
# ---------------------------------------------------------------------------

AGGREGATION_KEYWORDS = (
    "total", "sum", "average", "mean", "overall", "all", "subtotal",
    "grand total", "net", "gross", "aggregate", "合计", "平均", "总计",
)

PERCENTAGE_KEYWORDS = {"percent", "%", "ratio", "proportion", "share", "rate"}


# ---------------------------------------------------------------------------
# Internal table-structure prompt formatter
# ---------------------------------------------------------------------------

class _TableStructurePromptFormatter:
    """将 HiTab 表格 JSON 转化为带结构标注的文本表示"""

    def __init__(self, table_json: dict):
        self.table = table_json
        self.title = table_json.get("title", "")
        self.texts = table_json.get("texts", [])
        self.top_root = table_json.get("top_root", {})
        self.left_root = table_json.get("left_root", {})
        self.merged_regions = table_json.get("merged_regions", [])
        self.top_header_rows = table_json.get("top_header_rows_num", 1)
        self.left_header_cols = table_json.get("left_header_columns_num", 1)
        self.n_rows = len(self.texts)
        self.n_cols = max((len(row) for row in self.texts), default=0) if self.texts else 0

    # ---- 列表头层级 ----

    def get_header_hierarchy(self) -> Dict[str, Any]:
        """提取列表头层级为嵌套 dict"""
        return self._traverse_tree(self.top_root)

    def _traverse_tree(self, node: dict) -> Dict[str, Any]:
        """递归遍历 top_root/left_root 树"""
        if not node:
            return {}
        r, c = node.get("row_index", -1), node.get("column_index", -1)
        children = node.get("children", [])
        if r == -1 and c == -1:
            # 根节点
            result = {}
            for child in children:
                child_result = self._traverse_tree(child)
                result.update(child_result)
            return result
        text = self._get_cell_text(r, c)
        if children:
            sub = {}
            for child in children:
                sub.update(self._traverse_tree(child))
            return {text: sub}
        else:
            return {text: (r, c)}

    def get_row_hierarchy(self) -> Dict[str, Any]:
        """提取行表头层级"""
        return self._traverse_tree(self.left_root)

    # ---- 合并区域 ----

    def get_merged_descriptions(self) -> List[str]:
        """生成合并区域的文本描述"""
        descs = []
        for mr in self.merged_regions:
            r1, r2 = mr["first_row"], mr["last_row"]
            c1, c2 = mr["first_column"], mr["last_column"]
            text = self._get_cell_text(r1, c1)
            if r1 == r2 and c1 != c2:
                descs.append(f'"{text}" spans columns {c1}-{c2}')
            elif r1 != r2 and c1 == c2:
                descs.append(f'"{text}" spans rows {r1}-{r2}')
            elif r1 != r2 and c1 != c2:
                descs.append(f'"{text}" spans rows {r1}-{r2}, columns {c1}-{c2}')
        return descs

    # ---- 聚合单元格识别 ----

    def identify_aggregation_cells(self) -> List[Tuple[int, int, str]]:
        """扫描包含聚合关键词的单元格"""
        results = []
        for r in range(self.n_rows):
            for c in range(len(self.texts[r]) if r < len(self.texts) else 0):
                text = str(self.texts[r][c]).lower().strip()
                for kw in AGGREGATION_KEYWORDS:
                    if kw in text:
                        results.append((r, c, kw))
                        break
        return results

    # ---- 数据区域 ----

    def get_data_region(self) -> Tuple[int, int]:
        """返回数据区域的起始 (row, col)"""
        return (self.top_header_rows, self.left_header_cols)

    # ---- 格式化 ----

    def format_structured(self) -> str:
        """生成带结构标注的表格文本"""
        lines = []
        data_start_row, data_start_col = self.get_data_region()
        agg_cells = set((r, c) for r, c, _ in self.identify_aggregation_cells())

        for r, row in enumerate(self.texts):
            prefix = ""
            if r < data_start_row:
                prefix = f"[H{r}] "  # 标注为表头行
            elif any((r, c) in agg_cells for c in range(len(row))):
                prefix = "[AGG] "  # 标注为聚合行

            cells = []
            for c, cell in enumerate(row):
                cell_text = str(cell) if cell else ""
                if c < data_start_col and r >= data_start_row:
                    cell_text = f"«{cell_text}»"  # 标注行表头
                cells.append(cell_text)
            lines.append(prefix + " | ".join(cells))

        return "\n".join(lines)

    def format_col_hierarchy_desc(self) -> str:
        """生成列表头层级的文本描述"""
        hierarchy = self.get_header_hierarchy()
        return self._format_hierarchy_dict(hierarchy, indent=0)

    def format_row_hierarchy_desc(self) -> str:
        """生成行表头层级的文本描述"""
        hierarchy = self.get_row_hierarchy()
        return self._format_hierarchy_dict(hierarchy, indent=0)

    def _format_hierarchy_dict(self, d: dict, indent: int) -> str:
        """递归格式化层级 dict"""
        lines = []
        prefix = "  " * indent
        for key, val in d.items():
            if isinstance(val, dict):
                lines.append(f"{prefix}- {key}")
                lines.append(self._format_hierarchy_dict(val, indent + 1))
            else:
                lines.append(f"{prefix}- {key} → cell{val}")
        return "\n".join(lines)

    # ---- 辅助 ----

    def _get_cell_text(self, r: int, c: int) -> str:
        if 0 <= r < len(self.texts) and 0 <= c < len(self.texts[r]):
            return _cell_text_value(self.texts[r][c])
        return ""


# ---------------------------------------------------------------------------
# QuestionAnalyzer
# ---------------------------------------------------------------------------

OPERATION_PATTERNS = {
    "diff": [
        r"\bdifference\b", r"\bhow many more\b", r"\bhow many less\b",
        r"\bincrease\b", r"\bdecrease\b", r"\bchange\b", r"\bhigher than\b.*\bhow\b",
        r"\blower than\b.*\bhow\b", r"\bmore than\b.*\bhow\b",
    ],
    "compare": [
        r"\bwhich\b.*\bhigher\b", r"\bwhich\b.*\blower\b",
        r"\bwhich\b.*\bmore\b", r"\bwhich\b.*\bless\b",
        r"\bwhich\b.*\bgreater\b", r"\bwhich\b.*\bsmaller\b",
        r"\bwho\b.*\bmore\b", r"\bwho\b.*\bless\b",
        r"\bmore likely\b", r"\bless likely\b",
    ],
    "argmax": [
        r"\bwhich\b.*\bhighest\b", r"\bwhich\b.*\blargest\b",
        r"\bwhich\b.*\bmost\b", r"\bwhich\b.*\bmaximum\b",
        r"\bwhich\b.*\bbiggest\b",
    ],
    "argmin": [
        r"\bwhich\b.*\blowest\b", r"\bwhich\b.*\bsmallest\b",
        r"\bwhich\b.*\bleast\b", r"\bwhich\b.*\bminimum\b",
        r"\bwhich\b.*\bfewest\b",
    ],
    "average": [
        r"\baverage\b", r"\bmean\b", r"\bavg\b",
    ],
    "sum": [
        r"\btotal\b", r"\bsum\b", r"\bcombined\b", r"\baltogether\b",
    ],
    "ratio": [
        r"\bratio\b", r"\bproportion\b", r"\btimes\b",
        r"\bpercentage of\b",
    ],
    "count": [
        r"\bhow many\b(?!.*\bpercent\b)(?!.*\bdollars\b)(?!.*\b%\b)",
    ],
}

OPERATION_GUIDANCE = {
    "lookup": "Look up the value directly from the table cell that matches the question entities.",
    "diff": "Find TWO values and compute their difference. Identify which values to subtract.",
    "compare": "Compare two values and determine which entity has the higher/lower value.",
    "argmax": "Find the entity with the MAXIMUM value in the relevant row or column.",
    "argmin": "Find the entity with the MINIMUM value in the relevant row or column.",
    "sum": "Sum all relevant values in the identified row or column.",
    "average": "Average all relevant numeric values after applying the question condition.",
    "ratio": "Compute the ratio between two identified values.",
    "count": "Count the number of entities or values matching the criterion.",
}


class QuestionAnalyzer:
    """分析问题以推断操作类型和实体"""

    def __init__(self, question: str):
        self.question = question.lower().strip()
        self.operation_type = self.classify_operation()
        self.entities = self.extract_entities()

    def classify_operation(self) -> str:
        """基于关键词模式匹配推断操作类型"""
        q = self.question
        for op_type, patterns in OPERATION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, q, re.IGNORECASE):
                    return op_type
        return "lookup"

    def extract_entities(self) -> List[str]:
        """提取可能与表头匹配的实体 mention"""
        q = self.question
        # 移除常见问句词
        stopwords = {
            "what", "which", "how", "many", "much", "is", "are", "was", "were",
            "the", "a", "an", "of", "in", "for", "to", "and", "or", "by",
            "did", "do", "does", "has", "have", "had", "been", "percent",
            "percentage", "that", "this", "those", "than", "from", "with",
            "between", "among", "more", "less", "higher", "lower",
        }
        # 按标点和空白分词
        tokens = re.findall(r"[a-z0-9]+(?:[\'-][a-z0-9]+)*", q)
        # 构建 n-gram entities (1 to 4 grams)
        entities = []
        for n in range(4, 0, -1):
            for i in range(len(tokens) - n + 1):
                ngram = " ".join(tokens[i:i + n])
                # 跳过纯停用词 n-gram
                if all(t in stopwords for t in tokens[i:i + n]):
                    continue
                if len(ngram) > 2:
                    entities.append(ngram)
        return entities


# ---------------------------------------------------------------------------
# 构建结构感知 Prompt
# ---------------------------------------------------------------------------

CSCR_STRUCTURE_AWARE_PROMPT = """You are an expert at understanding hierarchical tables with complex header structures.

## Table Structure
Title: {title}

### Column Header Hierarchy
{col_hierarchy_desc}

### Row Header Hierarchy
{row_hierarchy_desc}

### Merged Regions
{merged_desc}

### Data Table
{formatted_table}

{aggregation_note}

## Question
{question}

## Operation Hint
This appears to be a {operation_type} question. {operation_guidance}

Return exactly one short answer. Do not include reasoning.
Answer: """


def build_structure_aware_prompt(table_json: dict, question: str) -> str:
    """构建结构感知 prompt"""
    formatter = _TableStructurePromptFormatter(table_json)
    analyzer = QuestionAnalyzer(question)

    col_hier = formatter.format_col_hierarchy_desc()
    row_hier = formatter.format_row_hierarchy_desc()
    merged = formatter.get_merged_descriptions()
    merged_desc = "\n".join(merged) if merged else "No merged regions."
    formatted_table = formatter.format_structured()

    agg_cells = formatter.identify_aggregation_cells()
    if agg_cells:
        agg_note = "Note: This table contains aggregation cells: " + ", ".join(
            f'"{formatter._get_cell_text(r, c)}" (row {r}, col {c}, type: {t})'
            for r, c, t in agg_cells
        )
    else:
        agg_note = ""

    op_type = analyzer.operation_type
    op_guidance = OPERATION_GUIDANCE.get(op_type, OPERATION_GUIDANCE["lookup"])

    return CSCR_STRUCTURE_AWARE_PROMPT.format(
        title=formatter.title or "(untitled)",
        col_hierarchy_desc=col_hier or "Flat headers (no hierarchy).",
        row_hierarchy_desc=row_hier or "Flat row labels.",
        merged_desc=merged_desc,
        formatted_table=formatted_table,
        aggregation_note=agg_note,
        question=question,
        operation_type=op_type,
        operation_guidance=op_guidance,
    )


# ---------------------------------------------------------------------------
# v8.0a: SCM-CoT Prompt (Structural Causal Model Chain-of-Thought)
# ---------------------------------------------------------------------------

SCM_COT_PROMPT = """You are an expert at understanding hierarchical tables with complex header structures.

## Table Structure
Title: {title}

### Column Header Hierarchy
{col_hierarchy_desc}

### Row Header Hierarchy
{row_hierarchy_desc}

### Merged Regions
{merged_desc}

### Data Table
{formatted_table}

{aggregation_note}

## Relevant Evidence
{causal_evidence_path}

{structural_candidates}

## Question
{question}

## Operation Hint
{operation_hint}

Return exactly one short answer. Do not include reasoning.
Answer: """


def build_scm_cot_prompt(
    table_json: dict,
    question: str,
    evidence=None,
    graph=None,
    exec_candidates=None,
    graph_stats: dict = None,
) -> str:
    """构建 SCM-CoT 增强 prompt (v8.0a)

    理论框架 (蓝图 v2 §3.1-3.3):
      将 HCEG 的因果 DAG 结构形式化为 SCM, 在 SCM 约束下引导 LLM 推理路径。
      通过注入三个结构化信号到 prompt:
      1. Causal Evidence Path: 从 evidence_retriever 产出的最小充分子图
         (信息瓶颈理论下的最小充分统计量)
      2. Structural Candidates: executor 产出的候选作为反事实参考
      3. Structural Complexity: 图统计量的自然语言描述

    不是启发式的 operation hint 或 weighted score,
    而是从图拓扑中提取的因果约束, 引导 LLM 沿正确的结构路径推理。
    """
    formatter = _TableStructurePromptFormatter(table_json)

    col_hier = formatter.format_col_hierarchy_desc()
    row_hier = formatter.format_row_hierarchy_desc()
    merged = formatter.get_merged_descriptions()
    merged_desc = "\n".join(merged) if merged else "No merged regions."
    formatted_table = formatter.format_structured()

    agg_cells = formatter.identify_aggregation_cells()
    if agg_cells:
        agg_note = "Note: This table contains aggregation cells: " + ", ".join(
            f'"{formatter._get_cell_text(r, c)}" (row {r}, col {c}, type: {t})'
            for r, c, t in agg_cells
        )
    else:
        agg_note = ""

    # --- 1. Causal Evidence Path (精简版) ---
    causal_path_text = ""
    if evidence is not None:
        path_text = evidence.to_causal_path_text()
        if path_text and path_text.strip() and path_text != "No causal evidence path found.":
            # 限制长度，避免 prompt 膨胀
            lines = path_text.strip().splitlines()[:10]
            causal_path_text = "Key evidence cells from structural analysis:\n" + "\n".join(lines)

    # --- 2. Structural Candidates (仅显示有效候选的答案) ---
    candidates_text = ""
    if exec_candidates:
        valid_cands = [c for c in exec_candidates if c.executor_valid]
        if valid_cands:
            cand_items = []
            seen = set()
            for cand in valid_cands[:3]:  # 最多 3 个
                den = str(cand.denotation)[:50]
                if den not in seen:
                    seen.add(den)
                    op = cand.operation.value if hasattr(cand.operation, 'value') else str(cand.operation)
                    cand_items.append(f'"{den}" ({op})')
            if cand_items:
                candidates_text = "Computed candidates: " + ", ".join(cand_items)

    # --- Operation hint (保持原有风格) ---
    analyzer = QuestionAnalyzer(question)
    op_type = analyzer.operation_type
    op_guidance = OPERATION_GUIDANCE.get(op_type, OPERATION_GUIDANCE["lookup"])
    operation_hint = f"This appears to be a {op_type} question. {op_guidance}"

    return SCM_COT_PROMPT.format(
        title=formatter.title or "(untitled)",
        col_hierarchy_desc=col_hier or "Flat headers (no hierarchy).",
        row_hierarchy_desc=row_hier or "Flat row labels.",
        merged_desc=merged_desc,
        formatted_table=formatted_table,
        aggregation_note=agg_note,
        causal_evidence_path=causal_path_text,
        structural_candidates=candidates_text,
        operation_hint=operation_hint,
        question=question,
    )


# ---------------------------------------------------------------------------
# Logit 熵校准
# ---------------------------------------------------------------------------

def compute_first_token_entropy(logprobs_dict: dict, top_k: int = 10) -> float:
    """计算第一个内容承载 answer token 的归一化熵

    参数:
        logprobs_dict: vLLM 输出的 logprobs dict {token: logprob}
        top_k: 使用 top-K 个 logits

    返回:
        归一化熵 ∈ [0, 1]，0 = 完全确定，1 = 均匀分布
    """
    if not logprobs_dict:
        return 1.0

    # 获取 top-k logprobs
    items = sorted(logprobs_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
    if not items:
        return 1.0

    # 转换为概率
    log_probs = [lp for _, lp in items]
    max_lp = max(log_probs)
    # 数值稳定的 softmax
    probs = [math.exp(lp - max_lp) for lp in log_probs]
    total = sum(probs)
    probs = [p / total for p in probs]

    # 计算熵
    entropy = 0.0
    for p in probs:
        if p > 1e-10:
            entropy -= p * math.log(p)

    # 归一化
    max_entropy = math.log(min(top_k, len(probs)))
    if max_entropy < 1e-10:
        return 0.0

    return entropy / max_entropy


def greedy_confidence_from_logprobs(logprobs_list: list, max_tokens: int = 5) -> float:
    """从前 N 个 answer token 的 logprobs 计算贪婪置信度

    置信度 = exp(mean_logprob) = 几何平均 token 概率

    参数:
        logprobs_list: vLLM 输出的 logprobs 列表，每个元素是 {token: logprob} dict
        max_tokens: 使用前 N 个 token

    返回:
        置信度 ∈ (0, 1]
    """
    if not logprobs_list:
        return 0.5

    selected = logprobs_list[:max_tokens]
    top_logprobs = []
    for token_dict in selected:
        if isinstance(token_dict, dict) and token_dict:
            # 取最高概率的 token 的 logprob
            max_lp = max(token_dict.values())
            top_logprobs.append(max_lp)
        elif isinstance(token_dict, (int, float)):
            top_logprobs.append(float(token_dict))

    if not top_logprobs:
        return 0.5

    mean_lp = sum(top_logprobs) / len(top_logprobs)
    confidence = math.exp(mean_lp)
    return max(0.0, min(1.0, confidence))


# ---------------------------------------------------------------------------
# v8.0b: Baseline E Prompt (简洁 prompt，已验证 64.65% EM)
# ---------------------------------------------------------------------------
#You are capable of effectively identifying the hierarchical structure of the table. Based on the provided table and textual description, please provide the answer to the question.
BASELINE_E_PROMPT = """You are answering a question using one hierarchical table.
Use only the table evidence. Pay attention to row headers, column headers, merged or blank cells, units, percentages, totals, and the table title.

## Table
{table}

## Question
{question}

Return exactly one short answer. Do not include reasoning.
Answer: """

BENCHMARK_QA_BASELINE_PROMPT = """Based on the table below, please answer the question, the answer should be short and simple. It can be a number, a word, or a phrase in the table, but not a full sentence.
Use only the table evidence. If the table contains row headers or column headers, bind each value to its row and column scope before answering.

## Table
{table}

## Question
{question}

Return exactly one short answer after "Answer:". Do not include reasoning, tags, or extra text.
Answer: """

OPERATION_AWARE_BASELINE_PROMPT = """Based on the table below, please answer the question, the answer should be short and simple. It can be a number, a word, or a phrase in the table, but not a full sentence.
Use only the table evidence. For lookup questions, copy the relevant cell value. For filtering, sum, average, count, ranking, comparison, arithmetic, or time questions, first identify the relevant rows or columns, then apply the requested operation to all and only those values.
Keep units, signs, percentages, and numeric precision when they are part of the table value or the computed answer.

## Table
{table}

## Question
{question}

Return exactly one short answer after "Answer:". Do not include reasoning, tags, or extra text.
Answer: """


def _prompt_source_format(table_json: dict, dataset_prompt_policy: str = "auto") -> str:
    if dataset_prompt_policy == "legacy":
        return "hitab"
    source = str(table_json.get("source_format", "hitab") or "hitab").lower()
    if dataset_prompt_policy == "benchmark":
        if source in ("aitqa", "tablebench"):
            return "benchmark_tableqa"
        return "hitab"
    if dataset_prompt_policy == "operation":
        if source == "tablebench":
            return "tablebench_operation"
        if source == "aitqa":
            return "benchmark_tableqa"
        return "hitab"
    if dataset_prompt_policy == "auto":
        if source == "aitqa":
            return "benchmark_tableqa"
        return "hitab"
    return "hitab"


def _format_table_plain(table_json: dict) -> str:
    """将表格格式化为纯文本 pipe-separated 格式（与 Baseline E 一致）"""
    parts = []
    title = table_json.get("title")
    if title:
        parts.append(f"Title: {title}")

    texts = table_json.get("texts", [])
    for row in texts:
        cells = []
        for cell in row:
            if isinstance(cell, dict):
                cells.append(str(cell.get("value", "")))
            else:
                cells.append(str(cell) if cell else "")
        parts.append(" | ".join(cells))

    return "\n".join(parts)


def build_baseline_e_prompt(
    table_json: dict,
    question: str,
    dataset_prompt_policy: str = "auto",
) -> str:
    """构建 Baseline E 风格的简洁 prompt (v8.0b P0)

    理论依据:
      Baseline E 使用简洁 prompt 在 HiTab 上达到 64.65% EM。
      当前 structure_aware prompt 添加了层级描述/操作提示等信息，
      但实验表明这些附加信息反而干扰了 LLM 的推理 (-4.11pp)。
      回到简洁 prompt 是恢复性能的第一步。
    """
    table_text = _format_table_plain(table_json)
    source = _prompt_source_format(table_json, dataset_prompt_policy)
    if source == "benchmark_tableqa":
        template = BENCHMARK_QA_BASELINE_PROMPT
    elif source == "tablebench_operation":
        template = OPERATION_AWARE_BASELINE_PROMPT
    else:
        template = BASELINE_E_PROMPT
    return template.format(table=table_text, question=question)


# ---------------------------------------------------------------------------
# v8.4: Question-Aware Table Pruning Prompt
# ---------------------------------------------------------------------------

_TABLE_PRUNED_PROMPT = """You are answering a question using one hierarchical table.
Use only the table evidence. The table below is question-focused: irrelevant rows/columns may be omitted, but headers and neighboring context are preserved.
Pay attention to row headers, column headers, merged or blank cells, units, percentages, totals, and the table title.

## Pruned Table
{table}

## Pruning Summary
{summary}

## Question
{question}

Return exactly one short answer. Do not include reasoning.
Answer: """


def _tokenize_for_pruning(text: str) -> List[str]:
    """轻量 tokenization，用于问题-表格重叠打分。"""
    if not text:
        return []
    tokens = re.findall(r"[A-Za-z0-9]+(?:\.[0-9]+)?|[\u4e00-\u9fff]+", str(text).lower())
    stop = {
        "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "by", "with",
        "what", "which", "who", "when", "where", "how", "is", "are", "was", "were",
        "as", "from", "than", "total", "sum", "average", "mean", "value", "number",
    }
    return [t for t in tokens if len(t) > 1 and t not in stop]


def _cell_text_value(cell: Any) -> str:
    if isinstance(cell, dict):
        return str(cell.get("value", ""))
    return str(cell) if cell is not None else ""


def _format_pruned_table(
    table_json: dict,
    question: str,
    evidence=None,
    graph=None,
    max_rows: int = 18,
    max_cols: int = 10,
    neighbor: int = 1,
) -> Tuple[str, str]:
    """基于行列内聚团的表格剪枝。

    设计原则：
      1. 表头行/列是结构因果锚点，必须保留；
      2. 问题词与行/列文本重叠越高，越可能属于目标行/列团；
      3. evidence anchor 所在行/列提供 HCEG 结构先验；
      4. 保留目标行/列的邻域，避免误删 sibling/total 上下文。

    返回：剪枝后的 pipe table 文本 + 剪枝摘要。
    """
    texts = table_json.get("texts", []) or []
    n_rows = len(texts)
    n_cols = max((len(r) for r in texts), default=0)
    if not texts or n_rows == 0 or n_cols == 0:
        return _format_table_plain(table_json), "empty-table-fallback"

    top_header_rows = int(table_json.get("top_header_rows_num", 1) or 1)
    left_header_cols = int(table_json.get("left_header_columns_num", 1) or 1)
    top_header_rows = max(1, min(top_header_rows, n_rows))
    left_header_cols = max(1, min(left_header_cols, n_cols))

    q_tokens = set(_tokenize_for_pruning(question))

    def row_text(r: int) -> str:
        return " ".join(_cell_text_value(c) for c in texts[r])

    def col_text(c: int) -> str:
        vals = []
        for r in range(n_rows):
            if c < len(texts[r]):
                vals.append(_cell_text_value(texts[r][c]))
        return " ".join(vals)

    row_scores: Dict[int, float] = {r: 0.0 for r in range(n_rows)}
    col_scores: Dict[int, float] = {c: 0.0 for c in range(n_cols)}

    for r in range(n_rows):
        toks = set(_tokenize_for_pruning(row_text(r)))
        overlap = len(q_tokens & toks)
        if overlap:
            row_scores[r] += overlap * 1.0
        # 行表头区域匹配更重要
        header_toks = set(_tokenize_for_pruning(" ".join(_cell_text_value(texts[r][c]) for c in range(min(left_header_cols, len(texts[r]))))))
        row_scores[r] += len(q_tokens & header_toks) * 1.5

    for c in range(n_cols):
        toks = set(_tokenize_for_pruning(col_text(c)))
        overlap = len(q_tokens & toks)
        if overlap:
            col_scores[c] += overlap * 1.0
        header_toks = set(_tokenize_for_pruning(" ".join(_cell_text_value(texts[r][c]) for r in range(top_header_rows) if c < len(texts[r]))))
        col_scores[c] += len(q_tokens & header_toks) * 1.5

    # HCEG/evidence anchor 行列先验
    anchor_positions: List[Tuple[int, int]] = []
    if evidence is not None and graph is not None:
        for nid in getattr(evidence, "anchor_nodes", []) or []:
            node = getattr(graph, "nodes", {}).get(nid)
            pos = getattr(node, "position", None) if node is not None else None
            if pos and len(pos) == 2:
                r, c = pos
                if 0 <= r < n_rows and 0 <= c < n_cols:
                    anchor_positions.append((r, c))
                    row_scores[r] += 3.0
                    col_scores[c] += 3.0

    keep_rows = set(range(top_header_rows))
    keep_cols = set(range(left_header_cols))

    # 根据分数选核心行列；至少保留若干数据行/列，避免过度剪枝
    data_rows = [r for r in range(top_header_rows, n_rows)]
    data_cols = [c for c in range(left_header_cols, n_cols)]
    ranked_rows = sorted(data_rows, key=lambda r: (row_scores[r], -r), reverse=True)
    ranked_cols = sorted(data_cols, key=lambda c: (col_scores[c], -c), reverse=True)

    row_budget = max(1, max_rows - top_header_rows)
    col_budget = max(1, max_cols - left_header_cols)
    seed_rows = [r for r in ranked_rows if row_scores[r] > 0][:row_budget]
    seed_cols = [c for c in ranked_cols if col_scores[c] > 0][:col_budget]

    # 若没有匹配，退化为保留前若干行列，保持安全
    if not seed_rows:
        seed_rows = data_rows[:row_budget]
    if not seed_cols:
        seed_cols = data_cols[:col_budget]

    for r in seed_rows:
        for rr in range(max(top_header_rows, r - neighbor), min(n_rows, r + neighbor + 1)):
            keep_rows.add(rr)
    for c in seed_cols:
        for cc in range(max(left_header_cols, c - neighbor), min(n_cols, c + neighbor + 1)):
            keep_cols.add(cc)

    # 二次裁剪预算：邻域扩张后可能超预算
    if len(keep_rows) > max_rows:
        header = set(range(top_header_rows))
        others = [r for r in keep_rows if r not in header]
        others = sorted(others, key=lambda r: (row_scores[r], -abs(r - (sum(seed_rows) / max(1, len(seed_rows))))), reverse=True)
        keep_rows = header | set(others[: max_rows - len(header)])
    if len(keep_cols) > max_cols:
        header = set(range(left_header_cols))
        others = [c for c in keep_cols if c not in header]
        others = sorted(others, key=lambda c: (col_scores[c], -abs(c - (sum(seed_cols) / max(1, len(seed_cols))))), reverse=True)
        keep_cols = header | set(others[: max_cols - len(header)])

    keep_rows_list = sorted(keep_rows)
    keep_cols_list = sorted(keep_cols)

    parts = []
    title = table_json.get("title")
    if title:
        parts.append(f"Title: {title}")

    last_r = -1
    omitted_row_blocks = 0
    for r in keep_rows_list:
        if last_r >= 0 and r > last_r + 1:
            parts.append(f"... [{r - last_r - 1} rows omitted] ...")
            omitted_row_blocks += 1
        cells = []
        last_c = -1
        for c in keep_cols_list:
            if last_c >= 0 and c > last_c + 1:
                cells.append(f"... {c - last_c - 1} cols omitted ...")
            cell = texts[r][c] if c < len(texts[r]) else ""
            cells.append(_cell_text_value(cell))
            last_c = c
        parts.append(" | ".join(cells))
        last_r = r

    summary = (
        f"kept_rows={len(keep_rows_list)}/{n_rows}, kept_cols={len(keep_cols_list)}/{n_cols}, "
        f"anchor_positions={anchor_positions[:6]}, omitted_row_blocks={omitted_row_blocks}"
    )
    return "\n".join(parts), summary


def build_table_pruned_prompt(
    table_json: dict,
    question: str,
    evidence=None,
    graph=None,
    max_rows: int = 18,
    max_cols: int = 10,
) -> str:
    """构建 v8.4 Question-Aware Table Pruning prompt。

    该 prompt 保持 Baseline E 的短答风格，但把完整表格替换为基于 HCEG/问题
    相关性的行列内聚团子表，目标是降低 binding_error。
    """
    table_text, summary = _format_pruned_table(
        table_json=table_json,
        question=question,
        evidence=evidence,
        graph=graph,
        max_rows=max_rows,
        max_cols=max_cols,
    )
    return _TABLE_PRUNED_PROMPT.format(table=table_text, summary=summary, question=question)


# ---------------------------------------------------------------------------
# v8.5: Safe Table Focus Prompt (完整表格 + 结构焦点提示)
# ---------------------------------------------------------------------------

_TABLE_FOCUS_PROMPT = """You are answering a question using one hierarchical table.
Use only the table evidence. Pay attention to row headers, column headers, merged or blank cells, units, percentages, totals, and the table title.
The focus hints below are soft guidance only: do not ignore the full table, and verify the final answer against the complete table.

## Table
{table}

## Structural Focus Hints
{focus_hints}

## Question
{question}

Return exactly one short answer. Do not include reasoning.
Answer: """

_BENCHMARK_QA_TABLE_FOCUS_PROMPT = """Based on the table below, please answer the question, the answer should be short and simple. It can be a number, a word, or a phrase in the table, but not a full sentence.
Use only the table evidence. Treat the focus hints as soft guidance; verify the final answer against the complete table and preserve the row/column header scope.

## Table
{table}

## Focus Hints
{focus_hints}

## Question
{question}

Return exactly one short answer after "Answer:". Do not include reasoning, tags, or extra text.
Answer: """

_OPERATION_AWARE_TABLE_FOCUS_PROMPT = """Based on the table below, please answer the question, the answer should be short and simple. It can be a number, a word, or a phrase in the table, but not a full sentence.
Use only the table evidence. Treat the focus hints as soft guidance. For filtering, sum, average, count, ranking, comparison, arithmetic, or time questions, verify the relevant rows or columns against the complete table before producing the final answer.

## Table
{table}

## Focus Hints
{focus_hints}

## Question
{question}

Return exactly one short answer after "Answer:". Do not include reasoning, tags, or extra text.
Answer: """


def _rank_table_focus_groups(
    table_json: dict,
    question: str,
    evidence=None,
    graph=None,
    max_rows: int = 6,
    max_cols: int = 5,
) -> Tuple[List[Tuple[int, float, str]], List[Tuple[int, float, str]], str]:
    """为 v8.5 table_focus 生成软性的行/列焦点摘要。

    与 v8.4 table_pruned 的关键区别：这里**只排序和摘要，不删除表格内容**。
    实验18/19/20证明硬剪枝会造成大量 correct→binding_error，因此 v8.5 将
    行/列内聚团作为 test-time probe/hint，而不是作为输入表格替代物。
    """
    texts = table_json.get("texts", []) or []
    n_rows = len(texts)
    n_cols = max((len(r) for r in texts), default=0)
    if not texts or n_rows == 0 or n_cols == 0:
        return [], [], "empty-table"

    top_header_rows = int(table_json.get("top_header_rows_num", 1) or 1)
    left_header_cols = int(table_json.get("left_header_columns_num", 1) or 1)
    top_header_rows = max(1, min(top_header_rows, n_rows))
    left_header_cols = max(1, min(left_header_cols, n_cols))
    q_tokens = set(_tokenize_for_pruning(question))

    def row_values(r: int) -> List[str]:
        return [_cell_text_value(c) for c in texts[r]]

    def col_values(c: int) -> List[str]:
        vals = []
        for r in range(n_rows):
            if c < len(texts[r]):
                vals.append(_cell_text_value(texts[r][c]))
        return vals

    row_scores: Dict[int, float] = {r: 0.0 for r in range(n_rows)}
    col_scores: Dict[int, float] = {c: 0.0 for c in range(n_cols)}

    for r in range(n_rows):
        vals = row_values(r)
        all_toks = set(_tokenize_for_pruning(" ".join(vals)))
        header_toks = set(_tokenize_for_pruning(" ".join(vals[:left_header_cols])))
        row_scores[r] += len(q_tokens & all_toks) * 1.0
        row_scores[r] += len(q_tokens & header_toks) * 1.8

    for c in range(n_cols):
        vals = col_values(c)
        all_toks = set(_tokenize_for_pruning(" ".join(vals)))
        header_toks = set(_tokenize_for_pruning(" ".join(vals[:top_header_rows])))
        col_scores[c] += len(q_tokens & all_toks) * 1.0
        col_scores[c] += len(q_tokens & header_toks) * 1.8

    anchor_positions: List[Tuple[int, int]] = []
    if evidence is not None and graph is not None:
        for nid in getattr(evidence, "anchor_nodes", []) or []:
            node = getattr(graph, "nodes", {}).get(nid)
            pos = getattr(node, "position", None) if node is not None else None
            if pos and len(pos) == 2:
                r, c = pos
                if 0 <= r < n_rows and 0 <= c < n_cols:
                    anchor_positions.append((r, c))
                    row_scores[r] += 3.0
                    col_scores[c] += 3.0
                    # 邻近 sibling/total 上下文只加软分，不剪枝
                    for rr in (r - 1, r + 1):
                        if top_header_rows <= rr < n_rows:
                            row_scores[rr] += 0.5
                    for cc in (c - 1, c + 1):
                        if left_header_cols <= cc < n_cols:
                            col_scores[cc] += 0.5

    ranked_rows = sorted(
        [r for r in range(top_header_rows, n_rows) if row_scores[r] > 0],
        key=lambda r: (row_scores[r], -r),
        reverse=True,
    )[:max_rows]
    ranked_cols = sorted(
        [c for c in range(left_header_cols, n_cols) if col_scores[c] > 0],
        key=lambda c: (col_scores[c], -c),
        reverse=True,
    )[:max_cols]

    row_summaries = []
    for r in ranked_rows:
        vals = row_values(r)
        header = " | ".join(vals[:left_header_cols]).strip() or f"row {r}"
        sample = " | ".join(v for v in vals[left_header_cols:left_header_cols + 4] if v).strip()
        row_summaries.append((r, row_scores[r], f"row {r}: {header}" + (f" -> {sample}" if sample else "")))

    col_summaries = []
    for c in ranked_cols:
        vals = col_values(c)
        header = " | ".join(vals[:top_header_rows]).strip() or f"col {c}"
        sample = " | ".join(v for v in vals[top_header_rows:top_header_rows + 4] if v).strip()
        col_summaries.append((c, col_scores[c], f"col {c}: {header}" + (f" -> {sample}" if sample else "")))

    summary = (
        f"table_shape={n_rows}x{n_cols}, top_header_rows={top_header_rows}, "
        f"left_header_cols={left_header_cols}, evidence_anchor_positions={anchor_positions[:8]}"
    )
    return row_summaries, col_summaries, summary


def build_table_focus_prompt(
    table_json: dict,
    question: str,
    evidence=None,
    graph=None,
    max_rows: int = 6,
    max_cols: int = 5,
    dataset_prompt_policy: str = "auto",
) -> str:
    """构建 v8.5 Safe Table Focus prompt。

    设计目标：保留 Baseline E 的完整表格可见性，同时用 HCEG/问题匹配生成软
    focus hints。该版本用于替代 v8.4 的硬剪枝主线，因为实验18/19/20显示
    硬剪枝显著增加 binding_error。
    """
    table_text = _format_table_plain(table_json)
    rows, cols, summary = _rank_table_focus_groups(
        table_json=table_json,
        question=question,
        evidence=evidence,
        graph=graph,
        max_rows=max_rows,
        max_cols=max_cols,
    )
    hint_lines = [summary]
    if rows:
        hint_lines.append("Candidate rows (soft, verify against full table):")
        hint_lines.extend(f"- {txt} [score={score:.2f}]" for _, score, txt in rows)
    if cols:
        hint_lines.append("Candidate columns (soft, verify against full table):")
        hint_lines.extend(f"- {txt} [score={score:.2f}]" for _, score, txt in cols)
    if not rows and not cols:
        hint_lines.append("No reliable lexical/HCEG focus group found; use the full table directly.")
    source = _prompt_source_format(table_json, dataset_prompt_policy)
    if source == "benchmark_tableqa":
        template = _BENCHMARK_QA_TABLE_FOCUS_PROMPT
    elif source == "tablebench_operation":
        template = _OPERATION_AWARE_TABLE_FOCUS_PROMPT
    else:
        template = _TABLE_FOCUS_PROMPT
    return template.format(
        table=table_text,
        focus_hints="\n".join(hint_lines),
        question=question,
    )


# ---------------------------------------------------------------------------
# v8.0b: Selective Evidence Prompt (选择性证据注入)
# v9.0 优化: 深层语义感知锚点排序 + 数值上下文提取
# ---------------------------------------------------------------------------

SELECTIVE_EVIDENCE_PROMPT = """You are answering a question using one hierarchical table.
Use only the table evidence. Pay attention to row headers, column headers, merged or blank cells, units, percentages, totals, and the table title.

## Table
{table}

{evidence_section}

## Question
{question}

Return exactly one short answer. Do not include reasoning.
Answer: """


def build_selective_evidence_prompt(
    table_json: dict,
    question: str,
    evidence=None,
    graph=None,
) -> str:
    """构建选择性证据注入 prompt (v8.0b P2)

    理论依据 (信息瓶颈原理, 蓝图 v2 §3.3):
      只有当证据检索质量高（锚点数 ≥ 2 且检索评分 > 0）时，
      才注入最小化的证据提示。否则回退到简洁 prompt。

      关键创新：不注入图拓扑细节（edge type, node type），
      而是将图推理结果转化为 LLM 可理解的自然语言指引。
      这避免了 v8.0a SCM-CoT 中的"错误锚点传播"问题。

    v8.0b 实证 (已验证 64.96% EM — 历史最高):
      简洁锚点 hint（仅锚点文本名称 + "Focus on..." 指引）是最优策略。
      v9.0 的数值上下文注入 (+268 chars) 导致 binding_error +30，
      non-abstain EM 从 73.3%→72.1% (-1.2pp)。

      核心教训：prompt 中的额外数值信息对 LLM 是噪声而非信号，
      因为 LLM 已经在表格中看到了这些数值。重复呈现反而增加歧义。
    """
    table_text = _format_table_plain(table_json)

    evidence_section = ""

    if evidence is not None:
        n_anchors = len(evidence.anchor_nodes)
        n_cells = evidence.num_cells
        retrieval_score = evidence.retrieval_score

        # 质量门控：只有高质量证据才注入
        if n_anchors >= 2 and retrieval_score > 0 and n_cells >= 2:
            # 提取锚点文本（自然语言形式，不暴露图内部细节）
            anchor_texts = []
            if hasattr(evidence, 'graph') and evidence.graph:
                for aid in evidence.anchor_nodes[:5]:  # 最多 5 个锚点
                    anode = evidence.graph.nodes.get(aid)
                    if anode and anode.text:
                        text = anode.text.strip()
                        if text and text not in anchor_texts:
                            anchor_texts.append(text)

            if anchor_texts:
                # 以自然语言形式提示 LLM 关注的区域
                anchor_hint = ", ".join(f'"{t}"' for t in anchor_texts[:4])
                evidence_section = f"## Hint\nThe question likely relates to these table elements: {anchor_hint}. Focus on the rows and columns containing these terms."

    return SELECTIVE_EVIDENCE_PROMPT.format(
        table=table_text,
        question=question,
        evidence_section=evidence_section,
    )


# ---------------------------------------------------------------------------
# v8.1: Intersection Hint Prompt (APR 高熵路由目标)
# ---------------------------------------------------------------------------

INTERSECTION_HINT_PROMPT = """You are answering a question using one hierarchical table.
Use only the table evidence. Pay attention to row headers, column headers, merged or blank cells, units, percentages, totals, and the table title.

## Table
{table}

## Hint
For this question, the answer is likely found at the intersection of:
{intersection_hints}
Check the cell(s) at this intersection.

## Question
{question}

Return exactly one short answer. Do not include reasoning.
Answer: """


def _find_intersection_hints(
    table_json: dict,
    question: str,
    evidence=None,
    graph=None,
) -> str:
    """从 HCEG 图的因果路径中提取行列交叉点提示（v8.1）

    理论依据：
      binding_error 的根因是 LLM 未能正确定位问题实体在表格中的行列坐标。
      诊断数据显示 88% 的 binding_error 发生在 ≤3000 字符的短表中，
      因此不需要 Table Pruning，而需要精确的坐标指引。

      通过 HCEG 图的因果路径（ENTITY_MENTION → 表头 → ROW_PATH/COL_PATH → CELL），
      可以追溯问题实体对应的行表头和列表头，生成：
        - Row containing: "Alberta"
        - Column containing: "2009"
      这种行列交叉点提示直接修复 LLM 的空间绑定错误。

    策略：
      1. 从问题节点的 ENTITY_MENTION 边出发找到锚点
      2. 从锚点沿 ROW_PATH / COL_PATH 找到对应的行/列表头
      3. 区分行表头和列表头（通过 header_level 和节点位置）
      4. 生成 "Row containing: X" / "Column containing: Y" 格式的提示
    """
    from graph_builder import NodeType, EdgeType

    row_headers = []
    col_headers = []

    if graph is not None and evidence is not None:
        # 获取锚点节点
        anchor_ids = evidence.anchor_nodes[:6]  # 最多考虑 6 个锚点

        # 收集锚点的行/列表头信息
        left_cols = table_json.get("left_header_columns_num", 1)
        top_rows = table_json.get("top_header_rows_num", 1)
        texts_table = table_json.get("texts", [])

        for aid in anchor_ids:
            anode = graph.nodes.get(aid)
            if anode is None:
                continue

            # 情况1: 锚点本身就是表头节点
            if anode.node_type == NodeType.HEADER:
                text = anode.text.strip()
                if not text:
                    continue
                # 判断是行表头还是列表头
                if anode.row >= top_rows and anode.col < left_cols:
                    # 行表头区域
                    if text not in row_headers:
                        row_headers.append(text)
                elif anode.row < top_rows and anode.col >= left_cols:
                    # 列表头区域
                    if text not in col_headers:
                        col_headers.append(text)
                elif anode.col < left_cols:
                    if text not in row_headers:
                        row_headers.append(text)
                else:
                    if text not in col_headers:
                        col_headers.append(text)
                continue

            # 情况2: 锚点是 CELL 节点 → 追溯其绑定的表头
            if anode.node_type in (NodeType.CELL, NodeType.AGGREGATOR):
                # 沿 ROW_PATH 找行表头
                for nid, edge in graph.neighbors(aid, {EdgeType.ROW_PATH}):
                    rnode = graph.nodes.get(nid)
                    if rnode and rnode.text.strip():
                        text = rnode.text.strip()
                        if text not in row_headers:
                            row_headers.append(text)
                # 沿 COL_PATH 找列表头
                for nid, edge in graph.neighbors(aid, {EdgeType.COL_PATH}):
                    cnode = graph.nodes.get(nid)
                    if cnode and cnode.text.strip():
                        text = cnode.text.strip()
                        if text not in col_headers:
                            col_headers.append(text)
                # 沿 VALUE_UNDER_HEADER 找对应表头
                for nid, edge in graph.neighbors(aid, {EdgeType.VALUE_UNDER_HEADER}):
                    hnode = graph.nodes.get(nid)
                    if hnode and hnode.text.strip():
                        text = hnode.text.strip()
                        if hnode.col < left_cols:
                            if text not in row_headers:
                                row_headers.append(text)
                        else:
                            if text not in col_headers:
                                col_headers.append(text)

            # 情况3: 锚点匹配到问题实体 → 看其在表格中的位置
            if not row_headers and not col_headers:
                # 尝试在表格文本中直接查找匹配
                q_text = anode.text.strip().lower()
                if q_text and len(q_text) >= 2:
                    for r, row in enumerate(texts_table):
                        for c, cell_val in enumerate(row):
                            cell_text = str(cell_val).strip().lower() if cell_val else ""
                            if q_text in cell_text or cell_text in q_text:
                                if r < top_rows:
                                    orig = str(cell_val).strip()
                                    if orig and orig not in col_headers:
                                        col_headers.append(orig)
                                elif c < left_cols:
                                    orig = str(cell_val).strip()
                                    if orig and orig not in row_headers:
                                        row_headers.append(orig)

    # 如果图路径没有找到足够的提示，尝试纯文本匹配
    if not row_headers and not col_headers:
        row_headers, col_headers = _fallback_text_match_headers(
            table_json, question
        )

    # 格式化提示
    hints = []
    for rh in row_headers[:3]:  # 最多3个行提示
        hints.append(f'- Row containing: "{rh}"')
    for ch in col_headers[:3]:  # 最多3个列提示
        hints.append(f'- Column containing: "{ch}"')

    return "\n".join(hints) if hints else ""


def _fallback_text_match_headers(
    table_json: dict,
    question: str,
) -> tuple:
    """当图路径为空时，使用纯文本匹配从问题中提取行列表头（v8.1 fallback）

    策略：将问题中的实体词（去除停用词）与表头区域的单元格文本匹配。
    """
    import re

    row_headers = []
    col_headers = []

    texts = table_json.get("texts", [])
    left_cols = table_json.get("left_header_columns_num", 1)
    top_rows = table_json.get("top_header_rows_num", 1)

    if not texts:
        return row_headers, col_headers

    # 提取问题中可能是实体的词组（2+ 字符, 非纯数字但可含数字如年份）
    q_lower = question.lower()
    # 提取带引号的实体
    quoted = re.findall(r'"([^"]+)"', question) + re.findall(r"'([^']+)'", question)

    # 提取年份和数字
    years = re.findall(r'\b((?:19|20)\d{2})\b', question)

    # 在行表头区域（左侧列，数据行）中匹配
    for r in range(top_rows, len(texts)):
        for c in range(min(left_cols, len(texts[r]))):
            cell_val = texts[r][c]
            cell_text = str(cell_val).strip() if cell_val else ""
            if not cell_text or len(cell_text) < 2:
                continue
            cell_lower = cell_text.lower()
            # 匹配引号实体
            for q_ent in quoted:
                if q_ent.lower() in cell_lower or cell_lower in q_ent.lower():
                    if cell_text not in row_headers:
                        row_headers.append(cell_text)
            # 匹配问题子串
            if len(cell_lower) >= 3 and cell_lower in q_lower:
                if cell_text not in row_headers:
                    row_headers.append(cell_text)

    # 在列表头区域（顶部行，数据列）中匹配
    for r in range(min(top_rows, len(texts))):
        for c in range(left_cols, len(texts[r]) if r < len(texts) else 0):
            cell_val = texts[r][c]
            cell_text = str(cell_val).strip() if cell_val else ""
            if not cell_text or len(cell_text) < 2:
                continue
            cell_lower = cell_text.lower()
            # 匹配引号实体
            for q_ent in quoted:
                if q_ent.lower() in cell_lower or cell_lower in q_ent.lower():
                    if cell_text not in col_headers:
                        col_headers.append(cell_text)
            # 匹配年份
            for yr in years:
                if yr in cell_text:
                    if cell_text not in col_headers:
                        col_headers.append(cell_text)
            # 匹配问题子串
            if len(cell_lower) >= 3 and cell_lower in q_lower:
                if cell_text not in col_headers:
                    col_headers.append(cell_text)

    return row_headers[:3], col_headers[:3]


def build_intersection_hint_prompt(
    table_json: dict,
    question: str,
    evidence=None,
    graph=None,
) -> str:
    """构建行列交叉点提示 prompt（v8.1 APR 高熵路由目标）

    理论依据：
      APR 路由将高熵样本（entropy >= 0.20, 占 23% 样本, 38.7% EM）路由到此 prompt。
      这些样本的 LLM 不确定性高，主要原因是 binding_error（LLM 无法正确定位
      问题实体在表格中的行列位置）。

      通过 HCEG 图的因果路径追溯问题实体 → 行/列表头，生成精确的空间坐标提示，
      直接引导 LLM 关注正确的单元格，从而修复 binding_error。

    预期效果：
      高熵带 38.7% -> 45%+ EM（+6pp），整体 +1.5pp
    """
    table_text = _format_table_plain(table_json)

    intersection_hints = _find_intersection_hints(
        table_json, question, evidence, graph
    )

    if not intersection_hints:
        # 无法提取交叉点提示时，回退到 selective_evidence prompt
        return build_selective_evidence_prompt(
            table_json, question, evidence, graph
        )

    return INTERSECTION_HINT_PROMPT.format(
        table=table_text,
        question=question,
        intersection_hints=intersection_hints,
    )


# ---------------------------------------------------------------------------
# v8.7 build_complementary_prompt 已移除 (诊断显示 v8.7 second-round 未带来收益)
# v8.8 改为升级 v8.3 APR 的路由信号 (credal_gate + non_degradation_guard)
# ---------------------------------------------------------------------------
