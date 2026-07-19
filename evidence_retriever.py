"""
evidence_retriever.py — CSCR Phase 3: 两阶段证据检索 + 结构对比

蓝图规范 (v1 §3.3, §4.1-4.2 + v2 §3.2):

阶段 1 — 语义锚点定位 (Semantic Anchor):
  从问题出发, 在 HCEG 中找到与问题实体最匹配的表头/单元格节点集合作为锚点。

阶段 2 — 结构扩展 (Structural Expansion):
  从锚点出发, 沿结构边/语义绑定边扩展, 收集最小充分证据子图。

信息瓶颈原则 (v1 §3.3):
  最优证据子图 G_sub 满足:
    I(a, T | G_sub, q) = 0  (充分性)
    min I(T, G_sub | q)      (最小性)

反事实干预接口 (v2 §3.2):
  提供 benign / adversarial 干预操作, 用于 SCCI 计算。
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from graph_builder import (
    HCEG,
    EdgeType,
    GraphEdge,
    GraphNode,
    NodeType,
    _normalize_text,
    _token_overlap,
    build_hceg,
)
from certa.retrieval.constants import RETRIEVER_VERSION


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

# 扩展时使用的边类型集合
STRUCTURAL_EDGES = {
    EdgeType.CHILD_HEADER, EdgeType.PARENT_HEADER,
    EdgeType.HEADER_OF, EdgeType.SPAN_OF, EdgeType.MERGED_INTO,
}

SEMANTIC_EDGES = {
    EdgeType.VALUE_UNDER_HEADER, EdgeType.ROW_PATH, EdgeType.COL_PATH,
}

EXECUTION_EDGES = {
    EdgeType.AGGREGATE_DEPENDS, EdgeType.COMPARISON_BETWEEN,
    EdgeType.PART_OF,
}

ALL_EXPANSION_EDGES = STRUCTURAL_EDGES | SEMANTIC_EDGES | EXECUTION_EDGES


# ---------------------------------------------------------------------------
# 检索结果
# ---------------------------------------------------------------------------

@dataclass
class EvidenceSubgraph:
    """检索到的证据子图"""
    graph: HCEG
    anchor_nodes: List[str]         # 语义锚点节点 ID
    evidence_nodes: Set[str]        # 所有证据节点 ID
    evidence_edges: List[GraphEdge] # 证据边
    retrieval_score: float = 0.0    # 检索质量评分
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_cells(self) -> int:
        return sum(1 for nid in self.evidence_nodes
                   if nid in self.graph.nodes
                   and self.graph.nodes[nid].node_type in (NodeType.CELL, NodeType.AGGREGATOR))

    @property
    def has_aggregator(self) -> bool:
        return any(
            nid in self.graph.nodes and self.graph.nodes[nid].node_type == NodeType.AGGREGATOR
            for nid in self.evidence_nodes
        )

    def to_text_evidence(self) -> str:
        """将证据子图转为文本描述 (用于 prompt 注入)"""
        lines = []
        # 收集证据单元格, 按 (row, col) 排序
        cells = []
        for nid in sorted(self.evidence_nodes):
            node = self.graph.nodes.get(nid)
            if not node:
                continue
            if node.node_type in (NodeType.CELL, NodeType.AGGREGATOR, NodeType.HEADER):
                cells.append(node)

        cells.sort(key=lambda n: (n.row, n.col))

        # 按行分组
        current_row = -1
        row_items: List[str] = []
        for cell in cells:
            if cell.row != current_row:
                if row_items:
                    lines.append(" | ".join(row_items))
                current_row = cell.row
                row_items = []
            prefix = ""
            if cell.node_type == NodeType.AGGREGATOR:
                prefix = f"[AGG:{cell.aggregation_type}] "
            elif cell.node_type == NodeType.HEADER:
                prefix = f"[H{cell.header_level}] "
            row_items.append(f"{prefix}{cell.text}")

        if row_items:
            lines.append(" | ".join(row_items))

        return "\n".join(lines)

    def to_causal_path_text(self) -> str:
        """将证据子图转为因果推理路径文本 (v8.0a SCM-CoT)

        理论基础 (蓝图 v2 §3.1):
          HCEG 图中 question → entity_mention → anchor → (row_path/col_path/
          value_under_header) → data_cell 定义了一条因果推理路径。
          将此路径显式呈现给 LLM，约束其推理空间到因果可达的单元格集合。

        输出格式:
          Anchor: "header_text" (row R, col C)
            → Data cell: "value" at (R', C') via row_path
            → Data cell: "value" at (R', C') via col_path
        """
        lines = []

        # 收集锚点节点信息
        anchors_info = []
        for aid in self.anchor_nodes:
            anode = self.graph.nodes.get(aid)
            if not anode:
                continue
            anchors_info.append((aid, anode))

        if not anchors_info:
            return "No causal evidence path found."

        # 因果语义边类型 (用于路径追踪)
        CAUSAL_EDGE_TYPES = {
            EdgeType.ROW_PATH, EdgeType.COL_PATH,
            EdgeType.VALUE_UNDER_HEADER, EdgeType.AGGREGATE_DEPENDS,
        }

        for aid, anode in anchors_info:
            atype = anode.node_type.value
            aloc = f"row {anode.row}, col {anode.col}" if anode.row is not None else ""
            lines.append(f'Anchor: "{anode.text}" ({atype}, {aloc})')

            # 从锚点沿因果边找到数据单元格
            reachable_cells = []
            # BFS 1-hop: anchor → data cells via causal edges
            for nid, edge in self.graph.neighbors(aid, CAUSAL_EDGE_TYPES):
                node = self.graph.nodes.get(nid)
                if not node:
                    continue
                if node.node_type in (NodeType.CELL, NodeType.AGGREGATOR, NodeType.VALUE):
                    reachable_cells.append((node, edge.edge_type.value))

            # 反向: data cells → anchor via causal edges
            for nid, edge in self.graph.predecessors(aid, CAUSAL_EDGE_TYPES):
                node = self.graph.nodes.get(nid)
                if not node:
                    continue
                if node.node_type in (NodeType.CELL, NodeType.AGGREGATOR, NodeType.VALUE):
                    reachable_cells.append((node, edge.edge_type.value))

            # 去重
            seen_ids = set()
            unique_cells = []
            for node, etype in reachable_cells:
                if node.node_id not in seen_ids:
                    seen_ids.add(node.node_id)
                    unique_cells.append((node, etype))

            # 只保留证据子图内的节点
            for node, etype in unique_cells:
                if node.node_id in self.evidence_nodes:
                    loc = f"row {node.row}, col {node.col}" if node.row is not None else ""
                    val = node.text[:50] if node.text else ""
                    lines.append(f'  → Cell: "{val}" ({loc}) via {etype}')

            # 如果锚点没有直接因果邻居，列出证据子图内同行/同列的单元格
            if not unique_cells:
                same_row_col = []
                for nid in self.evidence_nodes:
                    node = self.graph.nodes.get(nid)
                    if not node or node.node_id == aid:
                        continue
                    if node.node_type not in (NodeType.CELL, NodeType.AGGREGATOR):
                        continue
                    if (anode.row is not None and node.row == anode.row) or \
                       (anode.col is not None and node.col == anode.col):
                        same_row_col.append(node)
                for node in same_row_col[:5]:
                    loc = f"row {node.row}, col {node.col}" if node.row is not None else ""
                    lines.append(f'  → Related: "{node.text[:50]}" ({loc})')

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 干预类型 (用于 SCCI 计算)
# ---------------------------------------------------------------------------

class InterventionType(Enum):
    """蓝图 v1 §4.3 的 8 类干预"""
    # Benign (对正确候选应无影响)
    BENIGN_IRRELEVANT = "benign_irrelevant"       # 删除无关结构区域
    # Adversarial (对正确候选应产生影响)
    SUPPORT_DELETE = "support_delete"              # 删除支持边
    REQUIRED_EDGE_DELETE = "required_edge_delete"  # 删除可执行派生依赖边
    BINDING_SWAP = "binding_swap"                  # 替换 row/col header 绑定
    OPERATOR_REPLACE = "operator_replace"          # 替换操作符
    ANCHOR_SHIFT = "anchor_shift"                  # 锚点偏移到相邻行/列
    SIBLING_SUBSTITUTE = "sibling_substitute"      # 替换为同层级 sibling


BENIGN_INTERVENTIONS = {InterventionType.BENIGN_IRRELEVANT}
ADVERSARIAL_INTERVENTIONS = {
    InterventionType.SUPPORT_DELETE,
    InterventionType.REQUIRED_EDGE_DELETE,
    InterventionType.BINDING_SWAP,
    InterventionType.OPERATOR_REPLACE,
    InterventionType.ANCHOR_SHIFT,
    InterventionType.SIBLING_SUBSTITUTE,
}


@dataclass
class InterventionResult:
    """单次干预的结果"""
    intervention_type: InterventionType
    intervened_graph: HCEG
    removed_nodes: List[str] = field(default_factory=list)
    removed_edges: List[GraphEdge] = field(default_factory=list)
    modified_nodes: List[str] = field(default_factory=list)
    description: str = ""


# ---------------------------------------------------------------------------
# EvidenceRetriever — 核心检索器
# ---------------------------------------------------------------------------

class EvidenceRetriever:
    """
    两阶段证据检索器:

    Stage 1 — 语义锚点定位: 从问题节点出发, 找到 entity_mention 边指向的锚点集
    Stage 2 — 结构扩展: 从锚点 BFS 扩展, 沿结构/语义/执行边收集最小充分子图
    """

    def __init__(
        self,
        graph: HCEG,
        max_expansion_hops: int = 3,
        max_evidence_cells: int = 30,
        include_headers: bool = True,
        include_aggregators: bool = True,
    ):
        self.graph = graph
        self.max_expansion_hops = max_expansion_hops
        self.max_evidence_cells = max_evidence_cells
        self.include_headers = include_headers
        self.include_aggregators = include_aggregators
        self._last_native_node_order: List[str] = []

    def retrieve(self, question: Optional[str] = None) -> EvidenceSubgraph:
        """执行完整的两阶段检索"""
        # Stage 1: 语义锚点定位
        anchors = self._find_semantic_anchors(question)

        if not anchors:
            # 回退: 如果没有锚点, 返回全图
            native_order = list(self.graph.nodes.keys())
            return EvidenceSubgraph(
                graph=self.graph,
                anchor_nodes=[],
                evidence_nodes=set(self.graph.nodes.keys()),
                evidence_edges=list(self.graph.edges),
                retrieval_score=0.0,
                metadata={
                    "fallback": True,
                    "native_node_order": native_order,
                    "retriever_version": RETRIEVER_VERSION,
                },
            )

        # Stage 2: 结构扩展
        evidence_nodes = self._structural_expansion(anchors)

        # 构建证据子图
        subgraph = self.graph.subgraph(evidence_nodes)

        return EvidenceSubgraph(
            graph=subgraph,
            anchor_nodes=anchors,
            evidence_nodes=evidence_nodes,
            evidence_edges=list(subgraph.edges),
            retrieval_score=self._compute_retrieval_score(anchors, evidence_nodes),
            metadata={
                "num_anchors": len(anchors),
                "expansion_hops": self.max_expansion_hops,
                "native_node_order": list(self._last_native_node_order),
                "retriever_version": RETRIEVER_VERSION,
            },
        )

    def _find_semantic_anchors(self, question: Optional[str] = None) -> List[str]:
        """
        Stage 1: 从问题节点的 entity_mention 边找到语义锚点。
        如果问题节点不存在, 则用文本匹配动态计算。
        """
        anchors_with_score: List[Tuple[str, float]] = []

        # 方法1: 从图中的 question 节点出发
        q_nodes = self.graph.get_nodes_by_type(NodeType.QUESTION)
        if q_nodes:
            for qn in q_nodes:
                for nid, edge in self.graph.neighbors(qn.node_id, {EdgeType.ENTITY_MENTION}):
                    anchors_with_score.append((nid, edge.weight))

        # 方法2: 如果没有 question 节点但提供了 question 文本, 动态匹配
        if not anchors_with_score and question:
            q_lower = _normalize_text(question)
            for nid, node in self.graph.nodes.items():
                if node.node_type in (NodeType.QUESTION, NodeType.VALUE, NodeType.SPAN, NodeType.CANDIDATE):
                    continue
                if not node.text.strip():
                    continue
                overlap = _token_overlap(q_lower, node.text)
                node_lower = _normalize_text(node.text)
                is_sub = len(node_lower) >= 2 and node_lower in q_lower
                if overlap >= 0.3 or is_sub:
                    score = max(overlap, 0.8 if is_sub else 0.0)
                    anchors_with_score.append((nid, score))

        # 去重并排序
        seen = set()
        unique = []
        for nid, score in sorted(anchors_with_score, key=lambda x: -x[1]):
            if nid not in seen:
                seen.add(nid)
                unique.append(nid)

        return unique

    def _structural_expansion(self, anchor_ids: List[str]) -> Set[str]:
        """
        Stage 2: 从锚点 BFS 扩展, 沿结构/语义/执行边收集证据节点。

        扩展策略:
        1. 从锚点出发 BFS, 优先沿语义绑定边
        2. 遇到聚合节点时, 沿 aggregate_depends 边扩展到其覆盖的单元格
        3. 遇到表头节点时, 沿 child_header/parent_header 扩展到层级结构
        4. 收集锚点同行/同列的数据单元格
        """
        evidence: Set[str] = set()
        visited: Set[str] = set()
        native_order: List[str] = []
        queue: deque[Tuple[str, int]] = deque()  # (node_id, hop_count)

        # 初始化锚点
        for aid in anchor_ids:
            if aid in self.graph.nodes:
                queue.append((aid, 0))
                evidence.add(aid)
                visited.add(aid)
                native_order.append(aid)

        cell_count = 0

        while queue:
            nid, hop = queue.popleft()

            if hop >= self.max_expansion_hops:
                continue

            node = self.graph.nodes.get(nid)
            if not node:
                continue

            # 获取邻居 (正向 + 反向)
            forward_neighbors = self.graph.neighbors(nid, ALL_EXPANSION_EDGES)
            backward_neighbors = self.graph.predecessors(nid, ALL_EXPANSION_EDGES)

            all_neighbors = [(tid, e) for tid, e in forward_neighbors] + \
                            [(sid, e) for sid, e in backward_neighbors]

            for neighbor_id, edge in all_neighbors:
                if neighbor_id in visited:
                    continue

                neighbor = self.graph.nodes.get(neighbor_id)
                if not neighbor:
                    continue

                # 过滤策略
                should_add = False

                if neighbor.node_type == NodeType.HEADER:
                    should_add = self.include_headers
                elif neighbor.node_type == NodeType.AGGREGATOR:
                    should_add = self.include_aggregators
                elif neighbor.node_type == NodeType.CELL:
                    if cell_count < self.max_evidence_cells:
                        should_add = True
                elif neighbor.node_type == NodeType.VALUE:
                    should_add = True  # VALUE 节点始终包含
                elif neighbor.node_type == NodeType.SPAN:
                    should_add = True
                elif neighbor.node_type == NodeType.CANDIDATE:
                    should_add = False  # 候选节点不通过检索添加

                if should_add:
                    evidence.add(neighbor_id)
                    visited.add(neighbor_id)
                    native_order.append(neighbor_id)
                    if neighbor.node_type in (NodeType.CELL, NodeType.AGGREGATOR):
                        cell_count += 1
                    queue.append((neighbor_id, hop + 1))

        # 额外: 确保锚点的同行/同列表头被包含
        for aid in anchor_ids:
            node = self.graph.nodes.get(aid)
            if not node or node.row < 0 or node.col < 0:
                continue
            # 补充同行/同列的表头
            for nid, n in self.graph.nodes.items():
                if nid in evidence:
                    continue
                if n.node_type == NodeType.HEADER:
                    if n.row == node.row or n.col == node.col:
                        evidence.add(nid)
                        native_order.append(nid)

        self._last_native_node_order = native_order
        return evidence

    def _compute_retrieval_score(self, anchors: List[str], evidence: Set[str]) -> float:
        """计算检索质量评分 (用于诊断)"""
        if not anchors:
            return 0.0
        # 基于锚点数量和证据覆盖率
        anchor_score = min(len(anchors) / 3.0, 1.0)
        # 证据中聚合节点的覆盖
        agg_in_evidence = sum(
            1 for nid in evidence
            if nid in self.graph.nodes
            and self.graph.nodes[nid].node_type == NodeType.AGGREGATOR
        )
        agg_total = len(self.graph.get_nodes_by_type(NodeType.AGGREGATOR))
        agg_coverage = agg_in_evidence / max(agg_total, 1)
        return 0.6 * anchor_score + 0.4 * agg_coverage


# ---------------------------------------------------------------------------
# InterventionEngine — 反事实干预引擎
# ---------------------------------------------------------------------------

class InterventionEngine:
    """
    在 HCEG 上执行结构化反事实干预, 用于 SCCI 计算。

    蓝图 v2 §3.2:
    - I_benign: 删除无关区域 → 正确候选应不变
    - I_adversarial: 删除关键边、替换绑定等 → 正确候选应翻转

    SCCI(c) = BIR(c) × ASR(c)
    BIR = benign invariance rate (对 benign 干预不变的比率)
    ASR = adversarial sensitivity rate (对 adversarial 干预翻转的比率)
    """

    def __init__(self, full_graph: HCEG, evidence: EvidenceSubgraph):
        self.full_graph = full_graph
        self.evidence = evidence

    def generate_interventions(self) -> List[InterventionResult]:
        """生成所有可用的干预"""
        results = []

        # Benign: 删除无关区域
        benign = self._intervene_benign_irrelevant()
        if benign:
            results.append(benign)

        # Adversarial: 删除支持边
        for res in self._intervene_support_delete():
            results.append(res)

        # Adversarial: 绑定交换
        swap = self._intervene_binding_swap()
        if swap:
            results.append(swap)

        # Adversarial: 锚点偏移
        shift = self._intervene_anchor_shift()
        if shift:
            results.append(shift)

        # Adversarial: 同层替换
        sibling = self._intervene_sibling_substitute()
        if sibling:
            results.append(sibling)

        return results

    def _intervene_benign_irrelevant(self) -> Optional[InterventionResult]:
        """
        v6.1: 删除证据子图之外的无关数据单元格 (带路径保护)。

        预期: 正确候选的答案不变 (因为无关区域对答案没有因果影响)。

        v6.0 问题: 删除节点会同时删除其所有边，可能意外切断
        GraphAwareExecutor 的 BFS 遍历路径 → BIR=0。

        v6.1 修复: 只删除纯叶子数值节点 (没有被其他节点通过因果边引用的)，
        且不删除任何在 question→anchor→data_cell 路径上的节点。
        """
        evidence_ids = self.evidence.evidence_nodes

        # 因果语义边类型
        causal_edge_types = {
            EdgeType.VALUE_UNDER_HEADER, EdgeType.ROW_PATH, EdgeType.COL_PATH,
            EdgeType.AGGREGATE_DEPENDS, EdgeType.ENTITY_MENTION,
        }

        # 找到所有在因果路径上的节点 (通过反向追踪)
        # 从 question 节点出发, BFS 找到所有可达节点
        causal_reachable = set()
        question_ids = [
            nid for nid, node in self.full_graph.nodes.items()
            if node.node_type == NodeType.QUESTION
        ]
        queue = list(question_ids)
        causal_reachable.update(queue)
        while queue:
            nid = queue.pop(0)
            # 正向边
            for e in self.full_graph._adj.get(nid, []):
                if e.edge_type in causal_edge_types and e.target not in causal_reachable:
                    causal_reachable.add(e.target)
                    queue.append(e.target)
            # 反向边 (因果路径可双向)
            for e in self.full_graph._rev_adj.get(nid, []):
                if e.edge_type in causal_edge_types and e.source not in causal_reachable:
                    causal_reachable.add(e.source)
                    queue.append(e.source)

        irrelevant = [
            nid for nid, node in self.full_graph.nodes.items()
            if nid not in evidence_ids
            and nid not in causal_reachable  # v6.1: 路径保护
            and node.node_type == NodeType.CELL
            and node.is_numeric
        ]

        if not irrelevant:
            return None

        intervened = self._deep_copy_graph(self.full_graph)
        removed_nodes = []
        for nid in irrelevant[:10]:  # 最多删 10 个, 避免过度干预
            removed = intervened.remove_node(nid)
            if removed:
                removed_nodes.append(nid)

        return InterventionResult(
            intervention_type=InterventionType.BENIGN_IRRELEVANT,
            intervened_graph=intervened,
            removed_nodes=removed_nodes,
            description=f"Removed {len(removed_nodes)} irrelevant cells (path-protected)",
        )

    def _intervene_support_delete(self) -> List[InterventionResult]:
        """
        v6.0: 按 (source, target) 对删除所有因果语义边。

        修复: 之前逐条删除边, 但同一对节点有多条冗余边 (value_under_header + col_path),
        删一条时另一条仍提供路径 → SCCI 无法翻转。
        现在按 (source, target) 分组, 一次删除同一对的所有因果边。
        """
        results = []
        critical_edge_types = {
            EdgeType.VALUE_UNDER_HEADER, EdgeType.ROW_PATH, EdgeType.COL_PATH,
            EdgeType.AGGREGATE_DEPENDS,
        }

        # 收集所有关键边, 按 (source, target) 对分组
        pair_edges = {}
        for e in self.evidence.evidence_edges:
            if e.edge_type in critical_edge_types:
                pair = (e.source, e.target)
                pair_edges.setdefault(pair, []).append(e)

        # 对每个 (source, target) 对生成一个干预, 删除该对的所有因果边
        for (src, tgt), edges in list(pair_edges.items())[:5]:
            intervened = self._deep_copy_graph(self.full_graph)
            all_removed = []
            for edge in edges:
                removed = intervened.remove_edge(edge.source, edge.target, edge.edge_type)
                all_removed.extend(removed)
            if all_removed:
                edge_types_str = "+".join(e.edge_type.value for e in edges)
                results.append(InterventionResult(
                    intervention_type=InterventionType.SUPPORT_DELETE,
                    intervened_graph=intervened,
                    removed_edges=all_removed,
                    description=f"Deleted all causal edges ({edge_types_str}): {src} -> {tgt}",
                ))

        return results

    def _intervene_binding_swap(self) -> Optional[InterventionResult]:
        """
        交换证据子图中某个数据单元格的行/列表头绑定。
        预期: 正确候选的答案翻转 (因为绑定到了错误的表头)。
        """
        intervened = self._deep_copy_graph(self.full_graph)

        # 找到有 ROW_PATH 或 COL_PATH 的数据单元格
        data_cells = [
            nid for nid, node in intervened.nodes.items()
            if node.node_type == NodeType.CELL and node.is_numeric
        ]

        if not data_cells:
            return None

        # 选择第一个数据单元格, 删除其绑定边
        target = data_cells[0]
        removed_edges = []

        # 收集现有绑定
        bindings = [
            e for e in intervened.edges
            if e.source == target and e.edge_type in {EdgeType.ROW_PATH, EdgeType.COL_PATH}
        ]
        for e in bindings:
            removed = intervened.remove_edge(e.source, e.target, e.edge_type)
            removed_edges.extend(removed)

        if not removed_edges:
            return None

        # 添加到相邻行/列的错误绑定 (模拟 binding swap)
        node = intervened.nodes[target]
        # 找同列但不同行的表头
        for nid, n in intervened.nodes.items():
            if n.node_type == NodeType.HEADER and n.col == node.col and n.row != node.row:
                intervened.add_edge(GraphEdge(
                    source=target, target=nid,
                    edge_type=EdgeType.COL_PATH,
                    metadata={"swapped": True},
                ))
                break

        return InterventionResult(
            intervention_type=InterventionType.BINDING_SWAP,
            intervened_graph=intervened,
            removed_edges=removed_edges,
            modified_nodes=[target],
            description=f"Swapped header binding for cell {target}",
        )

    def _intervene_anchor_shift(self) -> Optional[InterventionResult]:
        """
        将语义锚点偏移到相邻的行/列。
        预期: 正确候选的答案翻转。
        """
        if not self.evidence.anchor_nodes:
            return None

        intervened = self._deep_copy_graph(self.full_graph)

        anchor = self.evidence.anchor_nodes[0]
        anchor_node = intervened.nodes.get(anchor)
        if not anchor_node or anchor_node.row < 0:
            return None

        # 找到锚点的 entity_mention 边并重新指向相邻行
        modified = False
        for e in list(intervened.edges):
            if e.target == anchor and e.edge_type == EdgeType.ENTITY_MENTION:
                # 找相邻行的同类型节点
                shifted_row = anchor_node.row + 1
                shifted_id = f"cell_{shifted_row}_{anchor_node.col}"
                if shifted_id in intervened.nodes:
                    intervened.remove_edge(e.source, e.target, e.edge_type)
                    intervened.add_edge(GraphEdge(
                        source=e.source, target=shifted_id,
                        edge_type=EdgeType.ENTITY_MENTION,
                        metadata={"shifted": True},
                    ))
                    modified = True
                    break

        if not modified:
            return None

        return InterventionResult(
            intervention_type=InterventionType.ANCHOR_SHIFT,
            intervened_graph=intervened,
            modified_nodes=[anchor],
            description=f"Shifted anchor {anchor} to adjacent row",
        )

    def _intervene_sibling_substitute(self) -> Optional[InterventionResult]:
        """
        将锚点替换为同层级的 sibling 表头。
        预期: 正确候选的答案翻转。
        """
        if not self.evidence.anchor_nodes:
            return None

        anchor = self.evidence.anchor_nodes[0]
        anchor_node = self.full_graph.nodes.get(anchor)
        if not anchor_node:
            return None

        # 找到锚点的 parent
        parents = self.full_graph.predecessors(anchor, {EdgeType.CHILD_HEADER})
        if not parents:
            return None

        parent_id = parents[0][0]
        # 找 parent 的其他 children (siblings)
        siblings = self.full_graph.neighbors(parent_id, {EdgeType.CHILD_HEADER})
        siblings = [sid for sid, _ in siblings if sid != anchor]

        if not siblings:
            return None

        sibling_id = siblings[0]
        sibling_node = self.full_graph.nodes.get(sibling_id)
        if not sibling_node:
            return None

        # 在证据子图中将锚点的文本替换为 sibling 的文本
        intervened = self._deep_copy_graph(self.full_graph)
        if anchor in intervened.nodes:
            intervened.nodes[anchor].text = sibling_node.text
            intervened.nodes[anchor].metadata["substituted_from"] = sibling_id

        return InterventionResult(
            intervention_type=InterventionType.SIBLING_SUBSTITUTE,
            intervened_graph=intervened,
            modified_nodes=[anchor],
            description=f"Substituted anchor {anchor} text with sibling {sibling_id}",
        )

    def _deep_copy_graph(self, graph: HCEG) -> HCEG:
        """深拷贝图 (避免修改原图)"""
        new_graph = HCEG()
        for nid, node in graph.nodes.items():
            new_node = GraphNode(
                node_id=node.node_id,
                node_type=node.node_type,
                row=node.row, col=node.col,
                text=node.text,
                numeric_value=node.numeric_value,
                header_level=node.header_level,
                aggregation_type=node.aggregation_type,
                metadata=dict(node.metadata),
            )
            new_graph.add_node(new_node)
        for e in graph.edges:
            new_graph.add_edge(GraphEdge(
                source=e.source, target=e.target,
                edge_type=e.edge_type,
                weight=e.weight,
                metadata=dict(e.metadata),
            ))
        return new_graph


# ---------------------------------------------------------------------------
# SCCI 计算器
# ---------------------------------------------------------------------------

@dataclass
class SCCIResult:
    """SCCI 计算结果"""
    candidate_id: str
    bir: float                           # Benign Invariance Rate
    asr: float                           # Adversarial Sensitivity Rate
    scci: float                          # BIR × ASR
    benign_details: List[Dict[str, Any]] = field(default_factory=list)
    adversarial_details: List[Dict[str, Any]] = field(default_factory=list)


def compute_scci(
    original_denotation: str,
    interventions: List[InterventionResult],
    executor_fn,
    question: str,
) -> SCCIResult:
    """
    计算 SCCI (Structured Causal Confidence Indicator)。

    蓝图 v2 §3.2:
    BIR(c) = (1/|I_benign|) · Σ (1 - flip(c, j))
    ASR(c) = (1/|I_adversarial|) · Σ flip(c, j)
    SCCI(c) = BIR(c) × ASR(c)

    参数:
        original_denotation: 原始图上的执行器输出
        interventions: 干预结果列表
        executor_fn: 在干预图上执行的函数 (graph, question) -> denotation
        question: 问题文本
    """
    benign_results = []
    adversarial_results = []

    for intv in interventions:
        intervened_denotation = executor_fn(intv.intervened_graph, question)

        # v6.0 修复: None 也视为翻转 (路径断开 = 因果依赖被破坏)
        # 旧逻辑: flipped = (intv is not None and intv != orig) → None 不算翻转
        # 新逻辑: flipped = (intv 与 orig 不同, 包括 intv=None 的情况)
        if intervened_denotation is None:
            flipped = (original_denotation is not None and original_denotation != "")
        else:
            flipped = (_normalize_text(str(intervened_denotation)) != _normalize_text(str(original_denotation)))

        detail = {
            "type": intv.intervention_type.value,
            "description": intv.description,
            "flipped": flipped,
            "original": original_denotation,
            "intervened": intervened_denotation,
        }

        if intv.intervention_type in BENIGN_INTERVENTIONS:
            benign_results.append(detail)
        else:
            adversarial_results.append(detail)

    # 计算 BIR
    if benign_results:
        bir = sum(1 - int(d["flipped"]) for d in benign_results) / len(benign_results)
    else:
        bir = 1.0  # 无 benign 干预时默认不变

    # v6.1: 计算 ASR (加权翻转 — 区分路径断开 vs 路径偏移)
    if adversarial_results:
        weighted_flips = 0.0
        for d in adversarial_results:
            if d["flipped"]:
                if d["intervened"] is None:
                    # null_flip: 路径完全断开 → 高权重 (因果依赖被破坏)
                    weighted_flips += 1.0
                else:
                    # shift_flip: 路径偏移到不同值 → 中权重 (部分因果保留)
                    weighted_flips += 0.7
        asr = weighted_flips / len(adversarial_results)
    else:
        asr = 0.0  # 无 adversarial 干预时默认无敏感性

    scci = bir * asr

    return SCCIResult(
        candidate_id="",
        bir=bir,
        asr=asr,
        scci=scci,
        benign_details=benign_results,
        adversarial_details=adversarial_results,
    )


# ---------------------------------------------------------------------------
# 便捷接口
# ---------------------------------------------------------------------------

def retrieve_evidence(
    table_json: Dict[str, Any],
    question: str,
    max_hops: int = 3,
    max_cells: int = 30,
) -> EvidenceSubgraph:
    """一步完成: 构建 HCEG + 检索证据子图"""
    graph = build_hceg(table_json=table_json, question=question)
    retriever = EvidenceRetriever(
        graph=graph,
        max_expansion_hops=max_hops,
        max_evidence_cells=max_cells,
    )
    return retriever.retrieve(question)


def retrieve_and_intervene(
    table_json: Dict[str, Any],
    question: str,
    max_hops: int = 3,
    max_cells: int = 30,
) -> Tuple[EvidenceSubgraph, List[InterventionResult]]:
    """一步完成: 构建 HCEG + 检索 + 生成干预"""
    graph = build_hceg(table_json=table_json, question=question)
    retriever = EvidenceRetriever(
        graph=graph,
        max_expansion_hops=max_hops,
        max_evidence_cells=max_cells,
    )
    evidence = retriever.retrieve(question)
    engine = InterventionEngine(full_graph=graph, evidence=evidence)
    interventions = engine.generate_interventions()
    return evidence, interventions


# ---------------------------------------------------------------------------
# CLI: 测试检索
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Evidence Retriever - 测试检索")
    parser.add_argument("--table", required=True, help="HiTab 表格 JSON 路径")
    parser.add_argument("--question", required=True, help="问题文本")
    parser.add_argument("--max-hops", type=int, default=3, help="最大扩展跳数")
    parser.add_argument("--max-cells", type=int, default=30, help="最大证据单元格数")
    parser.add_argument("--interventions", action="store_true", help="生成并显示干预")
    args = parser.parse_args()

    with open(args.table, "r", encoding="utf-8") as f:
        table_json = json.load(f)

    if args.interventions:
        evidence, interventions = retrieve_and_intervene(
            table_json=table_json,
            question=args.question,
            max_hops=args.max_hops,
            max_cells=args.max_cells,
        )
    else:
        evidence = retrieve_evidence(
            table_json=table_json,
            question=args.question,
            max_hops=args.max_hops,
            max_cells=args.max_cells,
        )
        interventions = []

    print("=== 证据检索结果 ===")
    print(f"  锚点数量: {len(evidence.anchor_nodes)}")
    print(f"  证据节点数: {len(evidence.evidence_nodes)}")
    print(f"  证据边数: {len(evidence.evidence_edges)}")
    print(f"  数据单元格数: {evidence.num_cells}")
    print(f"  包含聚合节点: {evidence.has_aggregator}")
    print(f"  检索分数: {evidence.retrieval_score:.3f}")

    print("\n  锚点:")
    for aid in evidence.anchor_nodes[:5]:
        node = evidence.graph.nodes.get(aid)
        if node:
            print(f"    {aid}: [{node.row},{node.col}] \"{node.text}\"")

    print("\n  证据文本:")
    print(evidence.to_text_evidence())

    if interventions:
        print(f"\n=== 干预列表 ({len(interventions)}) ===")
        for intv in interventions:
            print(f"  [{intv.intervention_type.value}] {intv.description}")
