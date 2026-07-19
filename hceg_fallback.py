"""
hceg_fallback.py — v9.1 HCEG-Fallback 直检兜底模块

触发条件（在 finalize_after_llm 中调用）：
  - credal_width >= cw_threshold（默认 0.30）
  - 或 coarse_question_type == "compare" 且 cw >= 0.15
  - 或 question_operation in ("compare", "diff") 且 cw >= 0.10

算法：
  1. anchor 重定位：用 question 关键词在 HCEG 上做 token overlap 检索
  2. 根据 coarse_type 选择 retrieval 策略：
     - compare     → 找两个 anchors，对比数值大小，返回较大/较小的 cell 文本
     - lookup_cell → anchor 直接定位 cell，返回 cell 文本
     - proportion / arithmetic → 找数值列 + 聚合（sum/ratio）
     - lookup_aggregate → 取 top-3 cell 文本
     - count       → 计数 anchor 邻居
     - superlative → argmax/argmin
  3. 输出 denotation（字符串），与 executor 兼容

设计约束：
  - graph_builder / evidence_retriever 是主框架依赖，缺失时应直接失败
  - 不修改 graph / evidence 对象
  - 写入 result["hceg_fallback_*"] 诊断字段，不影响主流程
"""
import re
from typing import Any, List, Optional, Tuple

from graph_builder import EdgeType


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _token_overlap(a: str, b: str) -> float:
    """计算两个字符串的 token 重叠率（Jaccard）。"""
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _parse_number(text: str) -> Optional[float]:
    """从文本中提取数值。"""
    if text is None:
        return None
    text = str(text).strip()
    # 去掉百分号、逗号、货币符号
    text = re.sub(r"[%,$£€]", "", text).replace(",", "").strip()
    try:
        return float(text)
    except (ValueError, TypeError):
        return None


def _node_type_value(node: Any) -> str:
    ntype = getattr(node, "node_type", "")
    return getattr(ntype, "value", str(ntype))


def _cell_text(graph: "HCEG", node_id: str) -> str:
    """获取节点的文本内容。"""
    node = graph.nodes.get(node_id)
    if node is None:
        return ""
    text = getattr(node, "text", "") or ""
    if text:
        return str(text)
    num = getattr(node, "numeric_value", None)
    if num is not None:
        if abs(float(num) - round(float(num))) < 1e-9:
            return str(int(round(float(num))))
        return str(num)
    for attr in ("value", "label"):
        val = getattr(node, attr, None)
        if val:
            return str(val)
    return ""


def _connected_nodes(graph: "HCEG", node_id: str):
    """Yield one-hop outgoing and incoming neighbors."""
    seen = set()
    for neighbor_id, edge in graph.neighbors(node_id):
        if neighbor_id not in seen:
            seen.add(neighbor_id)
            yield neighbor_id, edge
    for neighbor_id, edge in graph.predecessors(node_id):
        if neighbor_id not in seen:
            seen.add(neighbor_id)
            yield neighbor_id, edge


def infer_expected_answer_role(
    question: str,
    coarse_type: str = "",
    question_operation: str = "",
) -> str:
    """Infer the answer role requested by the question surface."""
    q = (question or "").lower()
    numeric_cues = (
        "how many", "how much", "what percentage", "what percent",
        "what is the percentage", "what was the percentage",
        "what proportion", "what is the proportion", "what was the proportion",
        "what rate", "what is the rate", "what was the rate",
        "what ratio", "what is the ratio", "what was the ratio",
        "what number", "what amount", "what value", "total", "average",
        "sum", "difference", "increase", "decrease", "change",
        "margin point", "percentage point", "million dollars",
        "per 100,000",
    )
    entity_cues = (
        "which", "who", "whom", "whose", "where", "which type",
        "which kind", "which group", "which region", "which visible",
    )
    if any(cue in q for cue in numeric_cues):
        return "numeric"
    if re.search(
        r"\b(?:what|which)\s+(?:is|was|were|are)\s+(?:the\s+)?"
        r"(?:percentage|percent|proportion|rate|rates|ratio|number|amount|value|score|total|average)\b",
        q,
    ):
        return "numeric"
    if re.search(
        r"\b(?:highest|lowest|largest|smallest|maximum|minimum)\s+"
        r"(?:percentage|percent|proportion|rate|rates|ratio|number|amount|value|score)\b",
        q,
    ):
        return "numeric"
    if coarse_type in {"count", "arithmetic", "times", "trend"}:
        return "numeric"
    if question_operation in {"count", "sum", "average", "diff", "difference", "ratio", "proportion"}:
        return "numeric"
    if any(cue in q for cue in entity_cues):
        return "entity"
    if coarse_type in {"compare", "superlative"} or question_operation in {"compare", "argmax", "argmin"}:
        return "entity"
    return "unknown"


def _is_entity_label(text: str) -> bool:
    text = str(text or "").strip()
    if not text or len(text) > 96:
        return False
    if _parse_number(text) is not None:
        return False
    if re.fullmatch(r"[-+.,%$£€\\d\\s/]+", text):
        return False
    return True


def _add_label_candidate(
    candidates: List[Tuple[str, float, str]],
    text: str,
    score: float,
    source: str,
    question: str,
) -> None:
    label = str(text or "").strip()
    if not _is_entity_label(label):
        return
    q = (question or "").lower()
    low = label.lower()
    if low in q:
        score += 3.0
    score += min(_token_overlap(question, label), 1.0)
    candidates.append((label, score, source))


def _best_entity_label_for_value(
    graph: "HCEG",
    value_node_id: str,
    question: str,
) -> Optional[str]:
    """Map a numeric evidence cell back to its row/entity label."""
    if graph is None:
        raise ValueError("_best_entity_label_for_value requires a constructed HCEG graph")
    if value_node_id not in graph.nodes:
        return None
    node = graph.nodes[value_node_id]
    candidates: List[Tuple[str, float, str]] = []

    for hid, _edge in graph.neighbors(value_node_id, {EdgeType.ROW_PATH}):
        _add_label_candidate(candidates, _cell_text(graph, hid), 4.0, "row_path", question)
    for hid, _edge in graph.neighbors(value_node_id, {EdgeType.COL_PATH}):
        _add_label_candidate(candidates, _cell_text(graph, hid), 4.0, "col_path", question)
    for hid, _edge in graph.neighbors(value_node_id, {EdgeType.VALUE_UNDER_HEADER}):
        _add_label_candidate(candidates, _cell_text(graph, hid), 3.5, "value_under_header", question)

    if getattr(node, "row", -1) >= 0:
        for nid, n in graph.nodes.items():
            if nid == value_node_id or getattr(n, "row", -2) != node.row:
                continue
            if getattr(n, "col", -1) >= 0 and node.col >= 0 and n.col < node.col:
                score = 3.0 + max(0.0, 1.0 - 0.15 * abs(node.col - n.col))
            else:
                score = 1.0
            ntype = _node_type_value(n)
            if ntype == "header":
                score += 1.0
            _add_label_candidate(candidates, _cell_text(graph, nid), score, "same_row", question)
    if getattr(node, "col", -1) >= 0:
        for nid, n in graph.nodes.items():
            if nid == value_node_id or getattr(n, "col", -2) != node.col:
                continue
            if getattr(n, "row", -1) >= 0 and node.row >= 0 and n.row < node.row:
                score = 3.0 + max(0.0, 1.0 - 0.15 * abs(node.row - n.row))
            else:
                score = 1.0
            ntype = _node_type_value(n)
            if ntype == "header":
                score += 1.0
            _add_label_candidate(candidates, _cell_text(graph, nid), score, "same_col", question)

    for nid, edge in _connected_nodes(graph, value_node_id):
        etype = getattr(getattr(edge, "edge_type", ""), "value", str(getattr(edge, "edge_type", "")))
        if etype in {
            "row_path", "col_path", "value_under_header",
            "parent_header", "child_header", "left", "right", "up", "down",
            "same_row", "same_col",
        }:
            _add_label_candidate(candidates, _cell_text(graph, nid), 1.5, f"neighbor:{etype}", question)

    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[1], len(x[0])))
    return candidates[0][0]


def _answer_for_expected_role(
    graph: "HCEG",
    node_id: str,
    numeric_text: str,
    question: str,
    expected_role: str,
) -> str:
    if expected_role == "entity":
        label = _best_entity_label_for_value(graph, node_id, question)
        if label:
            return label
    return numeric_text


# ---------------------------------------------------------------------------
# Anchor 重定位
# ---------------------------------------------------------------------------

def _find_best_anchors(
    graph: "HCEG",
    question: str,
    top_k: int = 5,
    min_overlap: float = 0.1,
) -> List[Tuple[str, float]]:
    """
    在 HCEG 中找与 question 最相关的 anchor 节点（按 token overlap 排序）。
    返回 [(node_id, score), ...] 降序。
    """
    if graph is None:
        raise ValueError("_find_best_anchors requires a constructed HCEG graph")
    scored = []
    for nid, node in graph.nodes.items():
        if _node_type_value(node) == "question":
            continue
        label = _cell_text(graph, nid)
        score = _token_overlap(question, label)
        if score >= min_overlap:
            scored.append((nid, score))
    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


def _find_numeric_neighbors(
    graph: "HCEG",
    anchor_id: str,
    max_hops: int = 2,
) -> List[Tuple[str, float]]:
    """
    从 anchor 出发，BFS 找数值型邻居节点。
    返回 [(node_id, numeric_value), ...] 按数值降序。
    """
    if graph is None:
        raise ValueError("_find_numeric_neighbors requires a constructed HCEG graph")
    visited = {anchor_id}
    queue = [(anchor_id, 0)]
    numeric_nodes = []
    while queue:
        nid, depth = queue.pop(0)
        if depth > max_hops:
            continue
        for neighbor_id, _ in _connected_nodes(graph, nid):
            if neighbor_id in visited:
                continue
            visited.add(neighbor_id)
            text = _cell_text(graph, neighbor_id)
            val = _parse_number(text)
            if val is not None:
                numeric_nodes.append((neighbor_id, val))
            if depth + 1 <= max_hops:
                queue.append((neighbor_id, depth + 1))
    numeric_nodes.sort(key=lambda x: -x[1])
    return numeric_nodes


# ---------------------------------------------------------------------------
# 策略函数
# ---------------------------------------------------------------------------

def _fallback_compare(
    graph: "HCEG",
    evidence: Optional["EvidenceSubgraph"],
    question: str,
    role_aware: bool = False,
) -> Optional[str]:
    """
    compare 策略：找两个 anchors，对比数值大小，返回较大/较小的 cell 文本。
    """
    anchors = _find_best_anchors(graph, question, top_k=6)
    if len(anchors) < 2:
        return None
    # 找两个有数值的 anchor
    numeric_anchors = []
    for nid, _ in anchors:
        text = _cell_text(graph, nid)
        val = _parse_number(text)
        if val is not None:
            numeric_anchors.append((nid, val, text))
        if len(numeric_anchors) >= 2:
            break
    if len(numeric_anchors) < 2:
        # 退而找 anchor 的数值邻居
        for nid, _ in anchors[:2]:
            neighbors = _find_numeric_neighbors(graph, nid, max_hops=1)
            if neighbors:
                nb_id, nb_val = neighbors[0]
                numeric_anchors.append((nb_id, nb_val, _cell_text(graph, nb_id)))
            if len(numeric_anchors) >= 2:
                break
    if len(numeric_anchors) < 2:
        return None
    # 判断 question 是 "greater/more/higher" 还是 "less/fewer/lower"
    q_lower = question.lower()
    if any(w in q_lower for w in ("greater", "more", "higher", "larger", "bigger", "most")):
        winner = max(numeric_anchors, key=lambda x: x[1])
    elif any(w in q_lower for w in ("less", "fewer", "lower", "smaller", "least")):
        winner = min(numeric_anchors, key=lambda x: x[1])
    else:
        # 默认返回较大值
        winner = max(numeric_anchors, key=lambda x: x[1])
    if role_aware:
        return _answer_for_expected_role(
            graph,
            winner[0],
            winner[2],
            question,
            infer_expected_answer_role(question, "compare", "compare"),
        )
    return winner[2]


def _fallback_lookup_cell(
    graph: "HCEG",
    evidence: Optional["EvidenceSubgraph"],
    question: str,
    role_aware: bool = False,
) -> Optional[str]:
    """
    lookup_cell 策略：anchor 直接定位 cell，返回 cell 文本。
    """
    anchors = _find_best_anchors(graph, question, top_k=3)
    if not anchors:
        return None
    # 优先返回有数值的 anchor
    for nid, _ in anchors:
        text = _cell_text(graph, nid)
        if _parse_number(text) is not None:
            if role_aware:
                return _answer_for_expected_role(
                    graph,
                    nid,
                    text,
                    question,
                    infer_expected_answer_role(question, "lookup", "lookup"),
                )
            return text
    # 否则返回最高分 anchor 的邻居数值
    top_nid = anchors[0][0]
    neighbors = _find_numeric_neighbors(graph, top_nid, max_hops=1)
    if neighbors:
        nb_id = neighbors[0][0]
        text = _cell_text(graph, nb_id)
        if role_aware:
            return _answer_for_expected_role(
                graph,
                nb_id,
                text,
                question,
                infer_expected_answer_role(question, "lookup", "lookup"),
            )
        return text
    return _cell_text(graph, top_nid)


def _fallback_proportion(
    graph: "HCEG",
    evidence: Optional["EvidenceSubgraph"],
    question: str,
) -> Optional[str]:
    """
    proportion 策略：找数值列，返回百分比形式。
    """
    anchors = _find_best_anchors(graph, question, top_k=3)
    if not anchors:
        return None
    for nid, _ in anchors:
        text = _cell_text(graph, nid)
        val = _parse_number(text)
        if val is not None:
            # 如果值在 0-1 之间，转为百分比
            if 0 < val <= 1.0:
                return f"{val * 100:.1f}%"
            elif 0 < val <= 100:
                return f"{val:.1f}%"
    # 找邻居
    for nid, _ in anchors[:2]:
        neighbors = _find_numeric_neighbors(graph, nid, max_hops=1)
        for nb_id, nb_val in neighbors:
            if 0 < nb_val <= 100:
                return f"{nb_val:.1f}%"
    return None


def _fallback_lookup_aggregate(
    graph: "HCEG",
    evidence: Optional["EvidenceSubgraph"],
    question: str,
) -> Optional[str]:
    """
    lookup_aggregate 策略：取 top-3 cell 文本，用逗号连接。
    """
    anchors = _find_best_anchors(graph, question, top_k=5)
    if not anchors:
        return None
    texts = []
    for nid, _ in anchors:
        text = _cell_text(graph, nid)
        if text and text not in texts:
            texts.append(text)
        if len(texts) >= 3:
            break
    if texts:
        return ", ".join(texts)
    return None


def _fallback_count(
    graph: "HCEG",
    evidence: Optional["EvidenceSubgraph"],
    question: str,
) -> Optional[str]:
    """
    count 策略：计数 anchor 邻居数量。
    """
    anchors = _find_best_anchors(graph, question, top_k=2)
    if not anchors:
        return None
    top_nid = anchors[0][0]
    neighbors = list(graph.neighbors(top_nid))
    count = len(neighbors)
    if count > 0:
        return str(count)
    return None


def _fallback_superlative(
    graph: "HCEG",
    evidence: Optional["EvidenceSubgraph"],
    question: str,
    role_aware: bool = False,
) -> Optional[str]:
    """
    superlative 策略：argmax/argmin。
    """
    anchors = _find_best_anchors(graph, question, top_k=5)
    if not anchors:
        return None
    # 收集所有 anchor 的数值
    numeric_anchors = []
    for nid, _ in anchors:
        text = _cell_text(graph, nid)
        val = _parse_number(text)
        if val is not None:
            numeric_anchors.append((nid, val, text))
    if not numeric_anchors:
        # 找邻居
        for nid, _ in anchors[:2]:
            for nb_id, nb_val in _find_numeric_neighbors(graph, nid, max_hops=1):
                numeric_anchors.append((nb_id, nb_val, _cell_text(graph, nb_id)))
    if not numeric_anchors:
        return None
    q_lower = question.lower()
    if any(w in q_lower for w in ("largest", "highest", "most", "greatest", "maximum", "max")):
        winner = max(numeric_anchors, key=lambda x: x[1])
    else:
        winner = min(numeric_anchors, key=lambda x: x[1])
    if role_aware:
        return _answer_for_expected_role(
            graph,
            winner[0],
            winner[2],
            question,
            infer_expected_answer_role(question, "superlative", "argmax"),
        )
    return winner[2]


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def hceg_direct_retrieve(
    graph: Optional["HCEG"],
    evidence: Optional["EvidenceSubgraph"],
    question: str,
    coarse_type: str,
    question_operation: str = "",
    role_aware: bool = False,
) -> Optional[str]:
    """
    HCEG 直检兜底主函数。

    参数：
      graph: HCEG 图对象（可为 None，此时返回 None）
      evidence: EvidenceSubgraph（可为 None）
      question: 原始问题文本
      coarse_type: coarse_question_type（proportion/lookup/count/compare/superlative/arithmetic/times/trend）
      question_operation: question_operation（lookup/ratio/count/compare/diff/sum/argmax/argmin）

    返回：
      denotation 字符串，或 None（无法检索时）
    """
    if graph is None:
        raise ValueError("hceg_direct_retrieve requires a constructed HCEG graph")
    # 根据 coarse_type 选策略
    if coarse_type == "compare" or question_operation in ("compare",):
        return _fallback_compare(graph, evidence, question, role_aware=role_aware)
    if coarse_type == "superlative" or question_operation in ("argmax", "argmin"):
        return _fallback_superlative(graph, evidence, question, role_aware=role_aware)
    if coarse_type == "count" or question_operation == "count":
        return _fallback_count(graph, evidence, question)
    if coarse_type == "proportion" or question_operation in ("ratio",):
        return _fallback_proportion(graph, evidence, question)
    if coarse_type == "lookup" or question_operation in ("lookup", "lookup_cell"):
        return _fallback_lookup_cell(graph, evidence, question, role_aware=role_aware)
    if question_operation in ("lookup_aggregate",):
        return _fallback_lookup_aggregate(graph, evidence, question)
    if coarse_type == "arithmetic" or question_operation in ("diff", "sum"):
        # arithmetic 退化到 lookup_cell（找最相关数值）
        return _fallback_lookup_cell(graph, evidence, question, role_aware=role_aware)
    # 默认 lookup_cell
    return _fallback_lookup_cell(graph, evidence, question, role_aware=role_aware)


def should_trigger_fallback(
    credal_width: float,
    coarse_type: str,
    question_operation: str,
    answer_source: str,
    cw_threshold: float = 0.30,
    compare_cw_threshold: float = 0.15,
    diff_cw_threshold: float = 0.10,
) -> Tuple[bool, str]:
    """
    判断是否触发 HCEG-Fallback。

    返回 (should_trigger: bool, reason: str)
    """
    # 已经是 path_verified_consensus 或 consensus_cert，不触发
    if answer_source in ("path_verified_consensus", "consensus_cert"):
        return False, "path_consensus_trusted"
    # 高 cw 触发
    if credal_width >= cw_threshold:
        return True, f"cw={credal_width:.3f}>={cw_threshold}"
    # compare 类型更早触发
    if coarse_type == "compare" and credal_width >= compare_cw_threshold:
        return True, f"compare+cw={credal_width:.3f}>={compare_cw_threshold}"
    # diff/compare 操作更早触发
    if question_operation in ("compare", "diff") and credal_width >= diff_cw_threshold:
        return True, f"op={question_operation}+cw={credal_width:.3f}>={diff_cw_threshold}"
    return False, "below_threshold"
