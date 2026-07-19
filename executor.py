"""
executor.py — CSCR Phase 4: 类型化执行器 + Lookup-Before-Compute

蓝图原则：LLM 不做任何数值计算，所有 +/-/×/÷/ratio/percentage 由执行器完成。
优先级链：
  1. lookup_aggregate  — 聚合单元格直接查找
  2. lookup_cell       — 普通单元格直接查找
  3. compare           — 两值比较，返回标签
  4. arithmetic        — 数值运算（sum/average/diff/ratio/count/argmax/argmin）

v5.0 升级 (方向 A):
  - Character n-gram binding: 提升实体-表头匹配精度
  - 数值/年份专用匹配: 避免 "2010" vs "2010-11" 的误匹配
  - 真实 binding_confidence: 基于匹配质量而非 score + 0.5

v6.0 升级:
  - GraphAwareExecutor: 基于 HCEG 图拓扑执行的执行器
  - 修复 SCCI 退化 (v5.1 所有样本 SCCI=0)
  - 干预改变图结构 → 路径断开 → executor 输出改变 → flip=1
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 类型定义
# ---------------------------------------------------------------------------

class OperationType(Enum):
    LOOKUP_AGGREGATE = "lookup_aggregate"
    LOOKUP_CELL = "lookup_cell"
    COMPARE = "compare"
    ARITHMETIC = "arithmetic"


@dataclass
class CellRef:
    row: int
    col: int
    value: str
    row_headers: List[str] = field(default_factory=list)
    col_headers: List[str] = field(default_factory=list)


@dataclass
class ExecutorResult:
    denotation: str
    operation: OperationType
    priority: int  # 1-4
    cells_used: List[CellRef] = field(default_factory=list)
    computation_trace: str = ""
    executor_valid: bool = True
    confidence: float = 1.0
    operation_metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

AGGREGATION_KEYWORDS = {
    "total", "sum", "average", "mean", "overall", "all", "subtotal",
    "grand total", "net", "gross", "aggregate",
}

# 问题中常见的停用词，用于实体提取
_QUESTION_STOPWORDS = {
    "what", "which", "how", "many", "much", "is", "are", "was", "were",
    "the", "a", "an", "of", "in", "for", "to", "and", "or", "by",
    "did", "do", "does", "has", "have", "had", "been", "percent",
    "percentage", "that", "this", "those", "than", "from", "with",
    "between", "among", "more", "less", "higher", "lower", "total",
    "number", "value", "amount", "rate", "count", "year", "years",
    "during", "at", "on", "about", "per", "each", "every",
    "it", "its", "their", "not", "no", "yes", "there",
}


def _extract_entities_from_question(question: str) -> List[str]:
    """从问题中提取可能对应表头的实体 mention（n-gram）"""
    q = question.lower().strip()
    tokens = re.findall(r"[a-z0-9]+(?:['\-][a-z0-9]+)*", q)
    entities = []
    for n in range(4, 0, -1):
        for i in range(len(tokens) - n + 1):
            ngram_tokens = tokens[i:i + n]
            if all(t in _QUESTION_STOPWORDS for t in ngram_tokens):
                continue
            ngram = " ".join(ngram_tokens)
            if len(ngram) > 2:
                entities.append(ngram)
    return entities


def _entity_match_score(entities: List[str], header: str) -> float:
    """计算实体列表与表头的最佳匹配分数"""
    h_lower = header.lower().strip()
    if not h_lower:
        return 0.0
    best = 0.0
    for ent in entities:
        score = _fuzzy_match(ent, h_lower)
        if score > best:
            best = score
    return best

def _parse_number(text: str) -> Optional[float]:
    """从文本中提取数值"""
    if not text:
        return None
    text = text.strip().replace(",", "").replace("$", "").replace("£", "").replace("€", "")
    text = text.strip()
    match = re.match(r"^[-+]?\d*\.?\d+\s*%?$", text)
    if not match:
        # 尝试宽松匹配
        match = re.search(r"[-+]?\d[\d,]*\.?\d*", text)
        if not match:
            return None
        text = match.group(0).replace(",", "")
    else:
        text = text.rstrip("% ")
    try:
        return float(text)
    except ValueError:
        return None


def _operand_records(cells: List[CellRef], roles: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for idx, ref in enumerate(cells or []):
        role = roles[idx] if roles and idx < len(roles) else "operand"
        records.append({
            "role": role,
            "node_id": f"cell_{ref.row}_{ref.col}",
            "row": ref.row,
            "col": ref.col,
            "value": ref.value,
        })
    return records


def _operation_metadata(
    family: str,
    *,
    projection_operator: str = "",
    answer_domain: str = "",
    comparison_polarity: str = "unknown",
    operator: str = "",
    operands: Optional[List[CellRef]] = None,
    operand_roles: Optional[List[str]] = None,
    selected_operand_index: Optional[int] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "operation_family": family,
        "semantic_source": "executor_structured_metadata_v1",
    }
    if projection_operator:
        payload["projection_operator"] = projection_operator
    if answer_domain:
        payload["answer_domain"] = answer_domain
    if comparison_polarity and comparison_polarity != "unknown":
        payload["comparison_polarity"] = comparison_polarity
    if operator:
        payload["operator"] = operator
    if selected_operand_index is not None:
        payload["selected_operand_index"] = selected_operand_index
    if operands:
        payload["operands"] = _operand_records(operands, operand_roles)
    return payload


def _normalize(text: str) -> str:
    """基础文本规范化"""
    return text.lower().strip().replace(",", "").replace(".", "").replace("'", "").replace('"', '')


def _char_ngrams(text: str, n: int = 3) -> Counter:
    """生成字符 n-gram 的频率计数器"""
    text = _normalize(text)
    if len(text) < n:
        return Counter([text]) if text else Counter()
    return Counter(text[i:i + n] for i in range(len(text) - n + 1))


def _char_ngram_similarity(s1: str, s2: str, n: int = 3) -> float:
    """基于字符 n-gram 的 Jaccard 相似度 (v5.0 方向 A)

    比 token-level Jaccard 更精细，能区分:
    - "2010" vs "2010-11" → 高 token overlap 但低 char-ngram overlap
    - "revenue" vs "revenues" → 高 char-ngram 相似度
    """
    ng1 = _char_ngrams(s1, n)
    ng2 = _char_ngrams(s2, n)
    if not ng1 or not ng2:
        return 0.0
    # Jaccard on multisets: |intersection| / |union|
    intersection = sum((ng1 & ng2).values())
    union = sum((ng1 | ng2).values())
    return intersection / union if union > 0 else 0.0


def _is_numeric_string(text: str) -> bool:
    """判断文本是否主要是数值/年份"""
    cleaned = _normalize(text).replace("-", "").replace("/", "").replace(" ", "")
    return bool(re.match(r'^\d+$', cleaned))


def _numeric_match(query: str, candidate: str) -> float:
    """数值/年份专用匹配 (v5.0 方向 A)

    对数字精确匹配给高分，部分匹配降分。
    例: "2010" vs "2010" → 1.0
        "2010" vs "2010-11" → 0.4 (部分包含但不精确)
        "2010" vs "2011" → 0.0
    """
    q = _normalize(query).strip()
    c = _normalize(candidate).strip()

    # 提取所有数字序列
    q_nums = re.findall(r'\d+', q)
    c_nums = re.findall(r'\d+', c)

    if not q_nums or not c_nums:
        return 0.0

    # 精确匹配: 所有数字序列完全一致
    if set(q_nums) == set(c_nums) and len(q_nums) == len(c_nums):
        return 1.0

    # 所有 query 数字都在 candidate 中出现 (子集匹配)
    if set(q_nums).issubset(set(c_nums)):
        # 但 candidate 有更多数字 → 降分 (e.g. "2010" in "2010-11")
        ratio = len(q_nums) / len(c_nums)
        return 0.3 + 0.3 * ratio

    # 部分数字匹配
    common = set(q_nums) & set(c_nums)
    if common:
        return 0.2 * len(common) / max(len(q_nums), len(c_nums))

    return 0.0


def _token_overlap(text1: str, text2: str) -> float:
    """两段文本的 token Jaccard 相似度"""
    tokens1 = set(_normalize(text1).split())
    tokens2 = set(_normalize(text2).split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    return len(intersection) / len(union)


def _fuzzy_match(query: str, candidate: str) -> float:
    """增强模糊匹配分数 (v5.0 升级)

    融合 token Jaccard + char n-gram + 数值专用匹配:
    - 精确匹配 → 1.0
    - 包含关系 → 基于长度比的分数 (避免 "a" in "table" 得高分)
    - 数值匹配 → 专用路径
    - 其他 → token overlap 与 char n-gram 的加权平均
    """
    q, c = _normalize(query), _normalize(candidate)
    if q == c:
        return 1.0

    # 检查是否为数值/年份匹配
    if _is_numeric_string(query) or _is_numeric_string(candidate):
        num_score = _numeric_match(query, candidate)
        if num_score > 0:
            return num_score

    # 包含关系: 根据长度比计算分数 (短串在长串中)
    if q in c:
        length_ratio = len(q) / len(c)
        return 0.5 + 0.4 * length_ratio  # range: [0.5, 0.9]
    if c in q:
        length_ratio = len(c) / len(q)
        return 0.5 + 0.4 * length_ratio

    # 混合评分: token overlap + char n-gram
    token_score = _token_overlap(query, candidate)
    char_score = _char_ngram_similarity(query, candidate)
    # 加权: char n-gram 权重更高 (对部分匹配更敏感)
    return 0.4 * token_score + 0.6 * char_score


# ---------------------------------------------------------------------------
# TypedExecutor
# ---------------------------------------------------------------------------

class TypedExecutor:
    """类型化执行器：基于表格结构执行精确计算"""

    def __init__(self, table_json: dict):
        self.texts = table_json.get("texts", [])
        self.title = table_json.get("title", "")
        self.top_header_rows = table_json.get("top_header_rows_num", 1)
        self.left_header_cols = table_json.get("left_header_columns_num", 1)
        self.merged_regions = table_json.get("merged_regions", [])
        self.n_rows = len(self.texts)
        self.n_cols = max((len(r) for r in self.texts), default=0) if self.texts else 0
        # 预建索引
        self._agg_cells = self._identify_aggregation_cells()
        self._header_index = self._build_header_index()

    def execute(self, question: str, operation_type: str = "auto",
                target_cells: List[Tuple[int, int]] = None) -> ExecutorResult:
        """主执行入口，按优先级链尝试"""
        q_lower = question.lower()

        # 优先级 1：聚合节点查找
        result = self._try_lookup_aggregate(question)
        if result:
            return result

        # 优先级 2：单元格直接查找
        if operation_type in ("auto", "lookup", "none"):
            result = self._try_lookup_cell(question, target_cells)
            if result:
                return result

        # 优先级 3：比较操作
        if operation_type in ("auto", "compare", "pair-argmax", "pair-argmin",
                              "greater_than", "less_than", "argmax", "argmin"):
            result = self._try_compare(question, target_cells)
            if result:
                return result

        # 优先级 4：数值计算
        result = self._execute_arithmetic(question, operation_type, target_cells)
        return result

    # ---- 优先级 1：聚合查找 ----

    def _try_lookup_aggregate(self, question: str) -> Optional[ExecutorResult]:
        """尝试从聚合单元格直接查找答案。

        修复 v2: 不再跳过行表头中的聚合标签。
        当 "total" 等关键词出现在行表头（left_header_cols）时，
        根据问题中的列实体匹配来定位同行的数据列。
        """
        if not self._agg_cells:
            return None

        entities = _extract_entities_from_question(question)
        best_match = None
        best_score = 0.0

        for r, c, agg_type in self._agg_cells:
            cell_ref = self._get_cell_ref(r, c)

            # 用实体匹配替代整个问题匹配
            row_score = max((_entity_match_score(entities, h) for h in cell_ref.row_headers), default=0)
            col_score = max((_entity_match_score(entities, h) for h in cell_ref.col_headers), default=0)

            if c < self.left_header_cols:
                # 聚合标签在行表头中 (e.g., texts[r][0]="total")
                # 需要根据问题中的列实体来定位同行的数据列
                # 首先检查问题是否真的在问这个聚合行
                agg_mentioned = agg_type in question.lower()
                row_label_score = row_score
                if not agg_mentioned and row_label_score < 0.3:
                    continue

                # 在同行的数据列中，找与问题列实体最匹配的列
                best_col_score = 0.0
                best_data_col = -1
                for dc in range(self.left_header_cols, self.n_cols):
                    col_headers = []
                    for hr in range(self.top_header_rows):
                        h = self._get_cell_text(hr, dc)
                        if h:
                            col_headers.append(h)
                    if col_headers:
                        cs = max(_entity_match_score(entities, h) for h in col_headers)
                        if cs > best_col_score:
                            best_col_score = cs
                            best_data_col = dc

                if best_data_col >= 0:
                    combined = (max(row_label_score, 0.5 if agg_mentioned else 0.0) + best_col_score) / 2
                    if combined > best_score and combined >= 0.25:
                        value = self._get_cell_text(r, best_data_col)
                        if value and _parse_number(value) is not None:
                            data_ref = self._get_cell_ref(r, best_data_col)
                            best_score = combined
                            best_match = ExecutorResult(
                                denotation=value.strip(),
                                operation=OperationType.LOOKUP_AGGREGATE,
                                priority=1,
                                cells_used=[data_ref],
                                computation_trace=f"Lookup aggregate: row {r} ({agg_type}), col {best_data_col} = {value}",
                                executor_valid=True,
                                confidence=min(1.0, combined + 0.3),
                                operation_metadata=_operation_metadata(
                                    "LOOKUP_AGGREGATE",
                                    projection_operator="VALUE_PROJECTION",
                                    answer_domain="SCALAR",
                                    operator="lookup_aggregate",
                                    operands=[data_ref],
                                    operand_roles=["lookup_value"],
                                ),
                            )
                # 如果没有列匹配，尝试该行唯一数据列的情况
                elif self.n_cols - self.left_header_cols == 1:
                    dc = self.left_header_cols
                    value = self._get_cell_text(r, dc)
                    if value and _parse_number(value) is not None:
                        score = 0.5 if agg_mentioned else row_label_score
                        if score > best_score and score >= 0.25:
                            best_score = score
                            data_ref = self._get_cell_ref(r, dc)
                            best_match = ExecutorResult(
                                denotation=value.strip(),
                                operation=OperationType.LOOKUP_AGGREGATE,
                                priority=1,
                                cells_used=[data_ref],
                                computation_trace=f"Lookup aggregate: row {r} ({agg_type}), single data col {dc} = {value}",
                                executor_valid=True,
                                confidence=min(1.0, score + 0.3),
                                operation_metadata=_operation_metadata(
                                    "LOOKUP_AGGREGATE",
                                    projection_operator="VALUE_PROJECTION",
                                    answer_domain="SCALAR",
                                    operator="lookup_aggregate",
                                    operands=[data_ref],
                                    operand_roles=["lookup_value"],
                                ),
                            )
            else:
                # 聚合标签在数据区域（直接包含数值）
                score = max(row_score, col_score)
                if score > best_score and score >= 0.3:
                    value = self._get_cell_text(r, c)
                    if _parse_number(value) is not None:
                        best_score = score
                        best_match = ExecutorResult(
                            denotation=value.strip(),
                            operation=OperationType.LOOKUP_AGGREGATE,
                            priority=1,
                            cells_used=[cell_ref],
                            computation_trace=f"Lookup aggregate cell ({r},{c}) = {value}",
                            executor_valid=True,
                            confidence=min(1.0, score + 0.3),
                            operation_metadata=_operation_metadata(
                                "LOOKUP_AGGREGATE",
                                projection_operator="VALUE_PROJECTION",
                                answer_domain="SCALAR",
                                operator="lookup_aggregate",
                                operands=[cell_ref],
                                operand_roles=["lookup_value"],
                            ),
                        )

        return best_match

    # ---- 优先级 2：单元格查找 ----

    def _try_lookup_cell(self, question: str, target_cells: List[Tuple[int, int]] = None) -> Optional[ExecutorResult]:
        """直接查找匹配问题的单元格值"""
        if target_cells:
            # 使用指定的目标单元格
            for r, c in target_cells:
                value = self._get_cell_text(r, c)
                if value:
                    return ExecutorResult(
                        denotation=value.strip(),
                        operation=OperationType.LOOKUP_CELL,
                        priority=2,
                        cells_used=[self._get_cell_ref(r, c)],
                        computation_trace=f"Lookup cell ({r},{c}) = {value}",
                        executor_valid=True,
                        confidence=1.0,
                        operation_metadata=_operation_metadata(
                            "LOOKUP",
                            projection_operator="VALUE_PROJECTION",
                            answer_domain="SCALAR" if _parse_number(value) is not None else "ENTITY",
                            operator="lookup_cell",
                            operands=[self._get_cell_ref(r, c)],
                            operand_roles=["lookup_value"],
                        ),
                    )
            return None

        # 自动定位：通过问题实体匹配表头，找交叉单元格
        matching_cells = self._find_matching_cells(question)
        if matching_cells:
            r, c, score = matching_cells[0]
            value = self._get_cell_text(r, c)
            if value and score >= 0.25:
                # v5.0: 真实 binding_confidence — 不再盲目 +0.5
                # score 本身反映了匹配质量 (0.25~1.0)
                # 精确匹配 (score≥0.9): confidence=score
                # 部分匹配 (score<0.9): 按比例降低
                binding_conf = score if score >= 0.9 else score * 0.85
                return ExecutorResult(
                    denotation=value.strip(),
                    operation=OperationType.LOOKUP_CELL,
                    priority=2,
                    cells_used=[self._get_cell_ref(r, c)],
                    computation_trace=f"Lookup cell ({r},{c}) = {value}, match_score={score:.3f}, binding_conf={binding_conf:.3f}",
                    executor_valid=True,
                    confidence=binding_conf,
                    operation_metadata=_operation_metadata(
                        "LOOKUP",
                        projection_operator="VALUE_PROJECTION",
                        answer_domain="SCALAR" if _parse_number(value) is not None else "ENTITY",
                        operator="lookup_cell",
                        operands=[self._get_cell_ref(r, c)],
                        operand_roles=["lookup_value"],
                    ),
                )
        return None

    # ---- 优先级 3：比较 ----

    def _try_compare(self, question: str, target_cells: List[Tuple[int, int]] = None) -> Optional[ExecutorResult]:
        """比较两个值，返回对应标签"""
        q_lower = question.lower()
        is_max = any(kw in q_lower for kw in ["higher", "more", "greater", "largest", "highest", "most", "maximum", "more likely"])
        is_min = any(kw in q_lower for kw in ["lower", "less", "smaller", "lowest", "least", "minimum", "fewer", "less likely"])

        if target_cells and len(target_cells) >= 2:
            cells = target_cells[:2]
        else:
            # 找两个匹配的数据单元格
            matches = self._find_matching_cells(question)
            if len(matches) < 2:
                return None
            cells = [(r, c) for r, c, _ in matches[:2]]

        r1, c1 = cells[0]
        r2, c2 = cells[1]
        v1 = _parse_number(self._get_cell_text(r1, c1))
        v2 = _parse_number(self._get_cell_text(r2, c2))

        if v1 is None or v2 is None:
            return None

        ref1 = self._get_cell_ref(r1, c1)
        ref2 = self._get_cell_ref(r2, c2)

        # 确定返回哪个标签
        if is_max:
            winner = ref1 if v1 >= v2 else ref2
            polarity = "max"
        elif is_min:
            winner = ref1 if v1 <= v2 else ref2
            polarity = "min"
        else:
            winner = ref1 if v1 >= v2 else ref2
            polarity = "greater_equal"

        # 返回标签（行表头或列表头中最具辨别力的）
        label = self._get_discriminative_label(ref1, ref2, winner)

        return ExecutorResult(
            denotation=label,
            operation=OperationType.COMPARE,
            priority=3,
            cells_used=[ref1, ref2],
            computation_trace=f"Compare ({r1},{c1})={v1} vs ({r2},{c2})={v2}, winner={label}",
            executor_valid=True,
            confidence=0.9,
            operation_metadata=_operation_metadata(
                "PAIR_COMPARE",
                projection_operator="ROW_ENTITY_PROJECTION",
                answer_domain="ENTITY",
                comparison_polarity=polarity,
                operator="pair_compare",
                operands=[ref1, ref2],
                operand_roles=["left", "right"],
                selected_operand_index=0 if winner is ref1 else 1,
            ),
        )

    # ---- 优先级 4：数值计算 ----

    def _execute_arithmetic(self, question: str, operation_type: str,
                            target_cells: List[Tuple[int, int]] = None) -> ExecutorResult:
        """执行数值运算。

        修复 v2:
        - 增加合理性检查：匹配分数过低时标记 executor_valid=False
        - 降低 arithmetic 的默认 confidence（因为 cell binding 不确定）
        - 移除 default 分支的盲目返回，改为标记无效
        - ratio 结果增加合理性范围检查
        """
        if target_cells:
            matches = target_cells
            match_confidence = 1.0  # 外部指定的单元格，完全信任
        else:
            raw_matches = self._find_matching_cells(question)
            if not raw_matches:
                return ExecutorResult(
                    denotation="",
                    operation=OperationType.ARITHMETIC,
                    priority=4,
                    cells_used=[],
                    computation_trace="No matching cells found",
                    executor_valid=False,
                    confidence=0.0,
                )
            # 取匹配分数作为 cell binding 的置信度
            match_confidence = raw_matches[0][2] if raw_matches else 0.0
            matches = [(r, c) for r, c, _ in raw_matches]

        values = []
        refs = []
        for r, c in matches:
            v = _parse_number(self._get_cell_text(r, c))
            if v is not None:
                values.append(v)
                refs.append(self._get_cell_ref(r, c))

        if not values:
            return ExecutorResult(
                denotation="",
                operation=OperationType.ARITHMETIC,
                priority=4,
                cells_used=[self._get_cell_ref(r, c) for r, c in matches],
                computation_trace="No numeric values found in matched cells",
                executor_valid=False,
                confidence=0.0,
            )

        q_lower = question.lower()
        result_val = None
        trace = ""
        # 基础置信度：arithmetic 本身就不太可靠，再乘以 cell binding 质量
        base_confidence = min(0.6, match_confidence * 0.7)

        if operation_type in ("diff", "difference") or "difference" in q_lower or "change" in q_lower:
            family = "DIFF"
            if len(values) >= 2:
                result_val = values[0] - values[1]
                trace = f"{values[0]} - {values[1]} = {result_val}"
            elif len(values) == 1:
                result_val = values[0]
                trace = f"Only one value found: {values[0]}"

        elif operation_type == "average" or "average" in q_lower or "mean" in q_lower or "avg" in q_lower:
            family = "AVERAGE"
            result_val = sum(values) / len(values)
            trace = f"average({', '.join(str(v) for v in values)}) = {result_val}"

        elif operation_type == "sum" or "total" in q_lower or "sum" in q_lower:
            family = "SUM"
            result_val = sum(values)
            trace = " + ".join(str(v) for v in values) + f" = {result_val}"

        elif operation_type in ("ratio", "percentage"):
            family = "RATIO"
            if len(values) >= 2 and values[1] != 0:
                result_val = values[0] / values[1]
                trace = f"{values[0]} / {values[1]} = {result_val}"
                # 合理性检查：ratio 通常在 0.001 ~ 1000 之间
                if result_val != 0 and (abs(result_val) > 10000 or abs(result_val) < 1e-6):
                    trace += " [SUSPICIOUS: ratio out of expected range]"
                    base_confidence *= 0.3

        elif operation_type in ("argmax",) or any(kw in q_lower for kw in ["highest", "largest", "maximum", "most"]):
            max_idx = values.index(max(values))
            label = self._get_label_for_cell(refs[max_idx])
            return ExecutorResult(
                denotation=label,
                operation=OperationType.ARITHMETIC,
                priority=4,
                cells_used=refs,
                computation_trace=f"argmax over {values} -> index {max_idx} -> {label}",
                executor_valid=True,
                confidence=base_confidence,
                operation_metadata=_operation_metadata(
                    "ARGMAX",
                    projection_operator="ROW_ENTITY_PROJECTION",
                    answer_domain="ENTITY",
                    comparison_polarity="max",
                    operator="argmax",
                    operands=refs,
                    operand_roles=["arg_value"] * len(refs),
                    selected_operand_index=max_idx,
                ),
            )

        elif operation_type in ("argmin",) or any(kw in q_lower for kw in ["lowest", "smallest", "minimum", "least"]):
            min_idx = values.index(min(values))
            label = self._get_label_for_cell(refs[min_idx])
            return ExecutorResult(
                denotation=label,
                operation=OperationType.ARITHMETIC,
                priority=4,
                cells_used=refs,
                computation_trace=f"argmin over {values} -> index {min_idx} -> {label}",
                executor_valid=True,
                confidence=base_confidence,
                operation_metadata=_operation_metadata(
                    "ARGMIN",
                    projection_operator="ROW_ENTITY_PROJECTION",
                    answer_domain="ENTITY",
                    comparison_polarity="min",
                    operator="argmin",
                    operands=refs,
                    operand_roles=["arg_value"] * len(refs),
                    selected_operand_index=min_idx,
                ),
            )

        elif operation_type == "count":
            result_val = len(values)
            trace = f"count = {len(values)}"
            family = "COUNT"

        else:
            # 默认分支：operation_type 不明确时，不盲目返回值
            # 标记为无效，让仲裁器不会使用
            return ExecutorResult(
                denotation="",
                operation=OperationType.ARITHMETIC,
                priority=4,
                cells_used=refs,
                computation_trace=f"Unknown operation_type='{operation_type}', {len(values)} values found but not computed",
                executor_valid=False,
                confidence=0.0,
            )

        if result_val is not None:
            # 格式化输出
            if isinstance(result_val, float) and result_val == int(result_val):
                denotation = str(int(result_val))
            elif isinstance(result_val, float):
                denotation = f"{result_val:.2f}" if abs(result_val) < 100 else (
                    f"{result_val:.1f}" if abs(result_val) < 1000 else f"{result_val:.0f}"
                )
            else:
                denotation = str(result_val)
        else:
            denotation = ""

        return ExecutorResult(
            denotation=denotation,
            operation=OperationType.ARITHMETIC,
            priority=4,
            cells_used=refs,
            computation_trace=trace,
            executor_valid=bool(denotation),
            confidence=base_confidence if denotation else 0.0,
            operation_metadata=_operation_metadata(
                locals().get("family", {
                    "average": "AVERAGE",
                    "sum": "SUM",
                    "difference": "DIFF",
                    "ratio": "RATIO",
                    "percentage": "RATIO",
                }.get(operation_type, "UNKNOWN")),
                projection_operator="SCALAR_RESULT_PROJECTION",
                answer_domain="SCALAR",
                operator=operation_type,
                operands=refs,
                operand_roles=["numeric_operand"] * len(refs),
            ),
        )

    # ---- 内部辅助方法 ----

    def _get_cell_text(self, r: int, c: int) -> str:
        if 0 <= r < len(self.texts) and 0 <= c < len(self.texts[r]):
            return str(self.texts[r][c]) if self.texts[r][c] else ""
        return ""

    def _get_cell_ref(self, r: int, c: int) -> CellRef:
        value = self._get_cell_text(r, c)
        row_headers = []
        for hc in range(self.left_header_cols):
            h = self._get_cell_text(r, hc)
            if h:
                row_headers.append(h)
        col_headers = []
        for hr in range(self.top_header_rows):
            h = self._get_cell_text(hr, c)
            if h:
                col_headers.append(h)
        return CellRef(row=r, col=c, value=value, row_headers=row_headers, col_headers=col_headers)

    def _identify_aggregation_cells(self) -> List[Tuple[int, int, str]]:
        results = []
        for r in range(self.n_rows):
            for c in range(len(self.texts[r]) if r < len(self.texts) else 0):
                text = str(self.texts[r][c]).lower().strip()
                for kw in AGGREGATION_KEYWORDS:
                    if kw in text:
                        results.append((r, c, kw))
                        break
        return results

    def _build_header_index(self) -> Dict[str, List[Tuple[int, int]]]:
        """构建表头文本到位置的索引"""
        index = {}
        # 列表头
        for r in range(self.top_header_rows):
            for c in range(self.n_cols):
                text = self._get_cell_text(r, c).lower().strip()
                if text:
                    index.setdefault(text, []).append((r, c))
        # 行表头
        for r in range(self.top_header_rows, self.n_rows):
            for c in range(self.left_header_cols):
                text = self._get_cell_text(r, c).lower().strip()
                if text:
                    index.setdefault(text, []).append((r, c))
        return index

    def _find_matching_cells(self, question: str) -> List[Tuple[int, int, float]]:
        """找到与问题匹配的数据单元格。

        修复 v2: 使用实体 n-gram 匹配替代整个问题匹配。
        - 从问题中提取实体 mention（去除停用词的 1-4 gram）
        - 用实体与表头做模糊匹配，而非用整个 question
        - 提高匹配阈值从 0.15 到 0.3，减少误匹配
        """
        entities = _extract_entities_from_question(question)
        if not entities:
            return []

        # 匹配行表头
        row_scores = {}
        for r in range(self.top_header_rows, self.n_rows):
            headers = []
            for c in range(self.left_header_cols):
                h = self._get_cell_text(r, c)
                if h:
                    headers.append(h)
            if headers:
                score = max(_entity_match_score(entities, h) for h in headers)
                if score > 0.3:
                    row_scores[r] = score

        # 匹配列表头
        col_scores = {}
        for c in range(self.left_header_cols, self.n_cols):
            headers = []
            for r in range(self.top_header_rows):
                h = self._get_cell_text(r, c)
                if h:
                    headers.append(h)
            if headers:
                score = max(_entity_match_score(entities, h) for h in headers)
                if score > 0.3:
                    col_scores[c] = score

        # 交叉得到候选数据单元格
        candidates = []
        for r, r_score in row_scores.items():
            for c, c_score in col_scores.items():
                # 使用几何平均提高区分度（两端都需要高分）
                combined = (r_score * c_score) ** 0.5
                value = self._get_cell_text(r, c)
                if value:
                    candidates.append((r, c, combined))

        # 如果没有交叉结果，退化为仅行或仅列匹配
        if not candidates:
            if row_scores and not col_scores:
                # 仅行匹配：取最佳行的所有数值列
                best_r = max(row_scores, key=row_scores.get)
                for c in range(self.left_header_cols, self.n_cols):
                    value = self._get_cell_text(best_r, c)
                    if value and _parse_number(value) is not None:
                        candidates.append((best_r, c, row_scores[best_r] * 0.4))
            elif col_scores and not row_scores:
                # 仅列匹配：取最佳列的所有数值行
                best_c = max(col_scores, key=col_scores.get)
                for r in range(self.top_header_rows, self.n_rows):
                    value = self._get_cell_text(r, best_c)
                    if value and _parse_number(value) is not None:
                        candidates.append((r, best_c, col_scores[best_c] * 0.4))

        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates

    def _get_discriminative_label(self, ref1: CellRef, ref2: CellRef, winner: CellRef) -> str:
        """获取最具辨别力的标签"""
        # 比较两个 ref 的表头，找到不同的部分
        all_headers_1 = ref1.row_headers + ref1.col_headers
        all_headers_2 = ref2.row_headers + ref2.col_headers
        winner_headers = winner.row_headers + winner.col_headers

        for h in winner_headers:
            h_lower = h.lower()
            # 检查这个表头是否具有辨别力（只在 winner 中）
            if h_lower and h_lower not in [x.lower() for x in (all_headers_1 if winner is ref2 else all_headers_2)]:
                return h
        # 退化为返回第一个非空表头
        for h in winner_headers:
            if h.strip():
                return h
        return winner.value

    def _get_label_for_cell(self, ref: CellRef) -> str:
        """获取单元格的标签（用于 argmax/argmin 返回值）"""
        # 优先返回行表头（通常是实体名）
        for h in ref.row_headers:
            if h.strip():
                return h
        for h in ref.col_headers:
            if h.strip():
                return h
        return ref.value


# ---------------------------------------------------------------------------
# 候选生成
# ---------------------------------------------------------------------------

def generate_candidates(table_json: dict, question: str, operation_type: str = "auto") -> List[ExecutorResult]:
    """为单个问题生成多个候选答案 (v5.0 升级: 完整候选闭包)

    返回所有可能的候选答案，不做去重。
    每个候选都带有真实 binding_confidence，供 Certificate Dominance 使用。
    """
    executor = TypedExecutor(table_json)
    candidates = []

    # 优先级 1：聚合查找
    r1 = executor._try_lookup_aggregate(question)
    if r1:
        candidates.append(r1)

    # 优先级 2：直接查找
    r2 = executor._try_lookup_cell(question)
    if r2:
        candidates.append(r2)

    # 优先级 3：比较
    r3 = executor._try_compare(question)
    if r3:
        candidates.append(r3)

    # 优先级 4：算术
    r4 = executor._execute_arithmetic(question, operation_type)
    if r4 and r4.executor_valid:
        candidates.append(r4)

    # 按优先级排序
    candidates.sort(key=lambda x: x.priority)

    # 去重: 相同 denotation 的候选只保留优先级最高的
    seen_denotations = set()
    deduped = []
    for c in candidates:
        norm_den = _normalize(c.denotation) if c.denotation else ""
        if norm_den and norm_den in seen_denotations:
            continue
        if norm_den:
            seen_denotations.add(norm_den)
        deduped.append(c)

    return deduped


def candidates_summary(candidates: List[ExecutorResult]) -> List[Dict[str, Any]]:
    """生成候选摘要用于诊断输出"""
    return [
        {
            "denotation": c.denotation,
            "operation": c.operation.value,
            "priority": c.priority,
            "confidence": round(c.confidence, 4),
            "valid": c.executor_valid,
            "support_cell_count": len(c.cells_used),
            "support_cells": [
                {
                    "row": ref.row,
                    "col": ref.col,
                    "value": ref.value,
                    "row_headers": ref.row_headers[:4],
                    "col_headers": ref.col_headers[:4],
                }
                for ref in c.cells_used[:12]
            ],
            "trace": c.computation_trace[:120],
            "operation_metadata": dict(c.operation_metadata or {}),
        }
        for c in candidates
    ]


def executor_result_summary(result: Optional[ExecutorResult], max_cells: int = 16) -> Dict[str, Any]:
    """Serialize executor support without changing arbitration behavior."""
    if result is None:
        return {
            "denotation": "",
            "operation": "none",
            "priority": None,
            "confidence": 0.0,
            "valid": False,
            "support_cell_count": 0,
            "support_cells": [],
            "trace": "",
        }
    support_cells = []
    numeric_values = []
    for ref in result.cells_used[:max_cells]:
        cell = {
            "row": ref.row,
            "col": ref.col,
            "value": ref.value,
            "row_headers": ref.row_headers[:6],
            "col_headers": ref.col_headers[:6],
        }
        support_cells.append(cell)
        parsed = _parse_number(ref.value)
        if parsed is not None:
            numeric_values.append(parsed)
    return {
        "denotation": result.denotation,
        "operation": result.operation.value,
        "priority": result.priority,
        "confidence": round(result.confidence, 4),
        "valid": result.executor_valid,
        "support_cell_count": len(result.cells_used),
        "support_cells": support_cells,
        "numeric_value_count": len(numeric_values),
        "numeric_min": min(numeric_values) if numeric_values else None,
        "numeric_max": max(numeric_values) if numeric_values else None,
        "trace": result.computation_trace[:240],
        "operation_metadata": dict(result.operation_metadata or {}),
    }


# ---------------------------------------------------------------------------
# Graph-Aware Executor (v6.0: 修复 SCCI 退化)
# ---------------------------------------------------------------------------

class GraphAwareExecutor:
    """基于 HCEG 图拓扑的执行器 (v6.0)

    核心差异 vs TypedExecutor:
    - TypedExecutor 从 table_json 文本匹配定位单元格 → 干预不影响输出
    - GraphAwareExecutor 从 HCEG 图的边遍历定位单元格 → 干预删边后路径断开

    蓝图 v2 §3.2:
      flip(c, I_j) = 1 ←→ executor(G with do(S = I_j(G))) ≠ executor(G)

    SCCI 修复原理:
      InterventionEngine 生成干预后的 HCEG 图 (删除边/交换绑定)
      GraphAwareExecutor 在干预图上执行 → 路径断开 → 不同输出 → flip=1
      BIR: benign 干预 (删无关区域) 不影响路径 → flip=0 → BIR ≈ 1.0
      ASR: adversarial 干预 (删关键边) 断开路径 → flip=1 → ASR > 0
    """

    def __init__(self, graph, table_json: dict = None):
        """
        Args:
            graph: HCEG 图实例 (可能是干预后的图)
            table_json: 原始表格 JSON (用于读取单元格文本值, 仅当图节点缺少 text 时)
        """
        self.graph = graph
        self.table_json = table_json
        self._texts = table_json.get("texts", []) if table_json else []

    def execute(self, question: str) -> Optional[str]:
        """在图上执行问题, 返回 denotation 或 None

        执行路径:
        1. 找到问题节点 (QUESTION 类型)
        2. 通过 ENTITY_MENTION 边找到锚点 (header/cell 节点)
        3. 从锚点出发, 通过 ROW_PATH/COL_PATH/VALUE_UNDER_HEADER 边追踪到数据单元格
        4. 返回最佳数据单元格的文本值

        当干预删除了关键边时, 步骤 3 中路径断开, 返回不同结果或 None → flip=1
        """
        # --- Step 1: 找问题节点 ---
        question_nodes = [
            n for n in self.graph.nodes.values()
            if n.node_type.value == "question"
        ]
        if not question_nodes:
            return None

        q_node = question_nodes[0]

        # --- Step 2: 通过 ENTITY_MENTION 边找锚点 ---
        anchor_ids = []
        for e in self.graph._adj.get(q_node.node_id, []):
            if e.edge_type.value == "entity_mention":
                anchor_ids.append(e.target)

        if not anchor_ids:
            return None

        # --- Step 3: 从锚点追踪到数据单元格 ---
        data_cells = []

        for anchor_id in anchor_ids:
            anchor_node = self.graph.nodes.get(anchor_id)
            if not anchor_node:
                continue

            # 从锚点出发, 通过结构边找到相关的数据单元格
            reachable = self._traverse_to_data_cells(anchor_id, max_depth=4)
            for cell_id, path_edges in reachable:
                cell_node = self.graph.nodes.get(cell_id)
                if cell_node and cell_node.is_numeric:
                    data_cells.append((cell_id, cell_node, path_edges))

        if not data_cells:
            # 路径断开 — 干预成功导致无法定位数据单元格
            return None

        # --- Step 4: 选择最佳数据单元格 ---
        # 优先选择通过最多结构边可达的单元格 (更强的图支持)
        best_cell = max(data_cells, key=lambda x: len(x[2]))
        cell_node = best_cell[1]

        # 获取单元格文本值
        value = self._get_cell_value(cell_node)
        return value

    def execute_with_path(self, question: str):
        """执行并返回 (denotation, graph_path)

        graph_path: 从问题节点到答案单元格经过的所有边的列表
        用于 path_verified consensus 验证
        """
        question_nodes = [
            n for n in self.graph.nodes.values()
            if n.node_type.value == "question"
        ]
        if not question_nodes:
            return None, []

        q_node = question_nodes[0]

        # 找锚点
        anchor_ids = []
        q_to_anchor_edges = []
        for e in self.graph._adj.get(q_node.node_id, []):
            if e.edge_type.value == "entity_mention":
                anchor_ids.append(e.target)
                q_to_anchor_edges.append(e)

        if not anchor_ids:
            return None, []

        # 追踪到数据单元格
        all_paths = []
        for i, anchor_id in enumerate(anchor_ids):
            reachable = self._traverse_to_data_cells(anchor_id, max_depth=4)
            for cell_id, path_edges in reachable:
                cell_node = self.graph.nodes.get(cell_id)
                if cell_node and cell_node.is_numeric:
                    edge_idx = min(i, len(q_to_anchor_edges) - 1)
                    full_path = [q_to_anchor_edges[edge_idx]] + path_edges
                    all_paths.append((cell_id, cell_node, full_path))

        if not all_paths:
            return None, []

        best = max(all_paths, key=lambda x: len(x[2]))
        value = self._get_cell_value(best[1])
        graph_path = [
            {
                "source": e.source,
                "target": e.target,
                "edge_type": e.edge_type.value,
            }
            for e in best[2]
        ]
        return value, graph_path

    def _traverse_to_data_cells(self, start_id: str, max_depth: int = 4):
        """BFS 遍历图, 从锚点沿因果语义边找到数据单元格

        仅允许因果语义边类型 (不含空间边):
        - ROW_PATH, COL_PATH: 表头到数据单元格的绑定 (因果必要)
        - VALUE_UNDER_HEADER: 值到表头的关联 (因果必要)
        - AGGREGATE_DEPENDS: 聚合依赖 (因果必要)

        不使用空间边 (up/down/left/right):
        - 空间边提供冗余路径, 绕过绑定边
        - 当干预删除 ROW_PATH/COL_PATH 时, 空间边仍能到达目标 → SCCI 退化
        - 蓝图 v2 §3.2: flip 必须反映因果依赖的破坏, 空间邻接不是因果依赖

        Returns: [(cell_id, [path_edges])]
        """
        traversal_edge_types = {
            "row_path", "col_path", "value_under_header",
            "aggregate_depends",
        }

        visited = {start_id}
        queue = [(start_id, [])]  # (node_id, path_edges)
        results = []

        while queue:
            current_id, path = queue.pop(0)
            if len(path) >= max_depth:
                continue

            # 检查出边
            for e in self.graph._adj.get(current_id, []):
                if e.edge_type.value in traversal_edge_types and e.target not in visited:
                    visited.add(e.target)
                    new_path = path + [e]
                    target_node = self.graph.nodes.get(e.target)
                    if target_node:
                        if target_node.node_type.value == "cell" and target_node.is_numeric:
                            results.append((e.target, new_path))
                        elif target_node.node_type.value in ("header", "value", "aggregator"):
                            queue.append((e.target, new_path))

            # 也检查入边 (反向遍历 — ROW_PATH/COL_PATH 可以双向)
            for e in self.graph._rev_adj.get(current_id, []):
                if e.edge_type.value in traversal_edge_types and e.source not in visited:
                    visited.add(e.source)
                    new_path = path + [e]
                    source_node = self.graph.nodes.get(e.source)
                    if source_node:
                        if source_node.node_type.value == "cell" and source_node.is_numeric:
                            results.append((e.source, new_path))
                        elif source_node.node_type.value in ("header", "value", "aggregator"):
                            queue.append((e.source, new_path))

        return results

    def _get_cell_value(self, node) -> Optional[str]:
        """获取节点的文本值"""
        # 优先从节点自身获取
        if node.text:
            return node.text.strip()
        # 从 table_json 中读取
        if self._texts and 0 <= node.row < len(self._texts):
            row = self._texts[node.row]
            if 0 <= node.col < len(row):
                return str(row[node.col]).strip()
        # 从数值获取
        if node.numeric_value is not None:
            return str(node.numeric_value)
        return None
