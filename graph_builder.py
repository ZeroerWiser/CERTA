"""
graph_builder.py — CSCR Phase 1B: 异构因果证据图 (HCEG) 构建器

蓝图规范 (v1 §4.1 + v2 §4.2-4.3):

节点类型:
  V = {V_cell, V_header, V_span, V_value, V_aggregator, V_question, V_candidate}

边类型 (6 族):
  空间边:      up, down, left, right, same_row, same_col
  结构边:      part_of, span_of, header_of, parent_header, child_header, merged_into
  语义绑定边:  value_under_header, row_path, col_path
  执行依赖边:  aggregate_depends, comparison_between
  问题证据边:  entity_mention, constraint_target, op_demand
  因果支持边:  necessary_for, sufficient_with, cf_sensitive_to  (Phase 5+ 填充)

设计原则:
  - 纯 Python dict/list 构建, 不依赖 networkx (减少依赖)
  - 可选导出为 networkx DiGraph 用于下游图算法
  - 聚合节点识别支持多语言关键词
  - 支持 HiTab 的 top_root / left_root 层级和 merged_regions
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# 枚举定义
# ---------------------------------------------------------------------------

class NodeType(Enum):
    CELL = "cell"
    HEADER = "header"
    SPAN = "span"            # 合并区域虚拟节点
    VALUE = "value"          # 数值节点 (从 cell 中提取的纯数值)
    AGGREGATOR = "aggregator"  # 聚合节点 (Total / Average / ...)
    QUESTION = "question"
    CANDIDATE = "candidate"


class EdgeType(Enum):
    # 空间边
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    SAME_ROW = "same_row"
    SAME_COL = "same_col"
    # 结构边
    PART_OF = "part_of"
    SPAN_OF = "span_of"
    HEADER_OF = "header_of"
    PARENT_HEADER = "parent_header"
    CHILD_HEADER = "child_header"
    MERGED_INTO = "merged_into"
    # 语义绑定边
    VALUE_UNDER_HEADER = "value_under_header"
    ROW_PATH = "row_path"
    COL_PATH = "col_path"
    # 执行依赖边
    AGGREGATE_DEPENDS = "aggregate_depends"
    COMPARISON_BETWEEN = "comparison_between"
    # 问题证据边
    ENTITY_MENTION = "entity_mention"
    CONSTRAINT_TARGET = "constraint_target"
    OP_DEMAND = "op_demand"
    # 因果支持边 (Phase 5+ 填充, 先定义枚举)
    NECESSARY_FOR = "necessary_for"
    SUFFICIENT_WITH = "sufficient_with"
    CF_SENSITIVE_TO = "cf_sensitive_to"


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class GraphNode:
    node_id: str
    node_type: NodeType
    row: int = -1
    col: int = -1
    text: str = ""
    numeric_value: Optional[float] = None
    header_level: int = -1       # 表头层级 (0=最外层)
    aggregation_type: str = ""   # sum / average / ratio / count / ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_numeric(self) -> bool:
        return self.numeric_value is not None


@dataclass
class GraphEdge:
    source: str
    target: str
    edge_type: EdgeType
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HCEG:
    """异构因果证据图"""
    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    edges: List[GraphEdge] = field(default_factory=list)
    # 索引结构 (构建后填充)
    _adj: Dict[str, List[GraphEdge]] = field(default_factory=dict)
    _rev_adj: Dict[str, List[GraphEdge]] = field(default_factory=dict)

    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.node_id] = node
        if node.node_id not in self._adj:
            self._adj[node.node_id] = []
        if node.node_id not in self._rev_adj:
            self._rev_adj[node.node_id] = []

    def add_edge(self, edge: GraphEdge) -> None:
        self.edges.append(edge)
        if edge.source not in self._adj:
            self._adj[edge.source] = []
        self._adj[edge.source].append(edge)
        if edge.target not in self._rev_adj:
            self._rev_adj[edge.target] = []
        self._rev_adj[edge.target].append(edge)

    def neighbors(self, node_id: str, edge_types: Optional[Set[EdgeType]] = None) -> List[Tuple[str, GraphEdge]]:
        """返回 (neighbor_id, edge) 列表"""
        results = []
        for e in self._adj.get(node_id, []):
            if edge_types is None or e.edge_type in edge_types:
                results.append((e.target, e))
        return results

    def predecessors(self, node_id: str, edge_types: Optional[Set[EdgeType]] = None) -> List[Tuple[str, GraphEdge]]:
        results = []
        for e in self._rev_adj.get(node_id, []):
            if edge_types is None or e.edge_type in edge_types:
                results.append((e.source, e))
        return results

    def get_nodes_by_type(self, ntype: NodeType) -> List[GraphNode]:
        return [n for n in self.nodes.values() if n.node_type == ntype]

    def subgraph(self, node_ids: Set[str]) -> "HCEG":
        """提取子图"""
        sub = HCEG()
        for nid in node_ids:
            if nid in self.nodes:
                sub.add_node(self.nodes[nid])
        for e in self.edges:
            if e.source in node_ids and e.target in node_ids:
                sub.add_edge(e)
        return sub

    def remove_edge(self, source: str, target: str, edge_type: Optional[EdgeType] = None) -> List[GraphEdge]:
        """移除边并返回被移除的边 (用于反事实干预)"""
        removed = []
        keep = []
        for e in self.edges:
            if e.source == source and e.target == target:
                if edge_type is None or e.edge_type == edge_type:
                    removed.append(e)
                    continue
            keep.append(e)
        self.edges = keep
        # 重建索引
        self._rebuild_adj()
        return removed

    def remove_node(self, node_id: str) -> Optional[GraphNode]:
        """移除节点及其关联的所有边 (用于反事实干预)"""
        node = self.nodes.pop(node_id, None)
        if node is not None:
            self.edges = [e for e in self.edges if e.source != node_id and e.target != node_id]
            self._rebuild_adj()
        return node

    def _rebuild_adj(self) -> None:
        self._adj = {nid: [] for nid in self.nodes}
        self._rev_adj = {nid: [] for nid in self.nodes}
        for e in self.edges:
            if e.source in self._adj:
                self._adj[e.source].append(e)
            if e.target in self._rev_adj:
                self._rev_adj[e.target].append(e)

    def stats(self) -> Dict[str, Any]:
        node_counts = {}
        for n in self.nodes.values():
            key = n.node_type.value
            node_counts[key] = node_counts.get(key, 0) + 1
        edge_counts = {}
        for e in self.edges:
            key = e.edge_type.value
            edge_counts[key] = edge_counts.get(key, 0) + 1
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "node_types": node_counts,
            "edge_types": edge_counts,
        }

    def to_dict(self) -> Dict[str, Any]:
        """序列化为 JSON-friendly dict"""
        return {
            "nodes": {
                nid: {
                    "node_type": n.node_type.value,
                    "row": n.row, "col": n.col,
                    "text": n.text,
                    "numeric_value": n.numeric_value,
                    "header_level": n.header_level,
                    "aggregation_type": n.aggregation_type,
                    "metadata": n.metadata,
                }
                for nid, n in self.nodes.items()
            },
            "edges": [
                {
                    "source": e.source, "target": e.target,
                    "edge_type": e.edge_type.value,
                    "weight": e.weight,
                    "metadata": e.metadata,
                }
                for e in self.edges
            ],
        }


# ---------------------------------------------------------------------------
# 聚合关键词 (多语言)
# ---------------------------------------------------------------------------

AGGREGATION_KEYWORDS: Dict[str, str] = {
    # English
    "total": "sum", "sum": "sum", "subtotal": "sum", "grand total": "sum",
    "net": "sum", "gross": "sum", "aggregate": "sum",
    "average": "average", "mean": "average", "avg": "average",
    "percent": "ratio", "percentage": "ratio", "%": "ratio",
    "ratio": "ratio", "proportion": "ratio", "share": "ratio", "rate": "ratio",
    "count": "count", "number of": "count",
    # Chinese
    "合计": "sum", "总计": "sum", "小计": "sum", "总额": "sum",
    "平均": "average", "均值": "average",
    "占比": "ratio", "比例": "ratio", "百分比": "ratio",
    "数量": "count", "个数": "count",
}


# ---------------------------------------------------------------------------
# 数值解析
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(
    r"^[\\$€£¥]?\s*[-+]?\s*[\d,]+(?:\.\d+)?\s*%?\s*$"
)


def _parse_number(text: str) -> Optional[float]:
    """解析单元格文本中的数值"""
    if not text or not text.strip():
        return None
    t = text.strip()
    # 去除货币符号
    t = re.sub(r"[\$€£¥]", "", t).strip()
    # 去除千分位逗号
    t = t.replace(",", "")
    # 处理百分号
    is_pct = t.endswith("%")
    if is_pct:
        t = t[:-1].strip()
    # 处理括号表示负数 (12.5) -> -12.5
    if t.startswith("(") and t.endswith(")"):
        t = "-" + t[1:-1]
    try:
        v = float(t)
        if is_pct:
            v /= 100.0
        return v
    except (ValueError, OverflowError):
        return None


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _token_overlap(a: str, b: str) -> float:
    """两个字符串的 token 重叠 Jaccard 系数"""
    ta = set(_normalize_text(a).split())
    tb = set(_normalize_text(b).split())
    if not ta or not tb:
        return 0.0
    inter = ta & tb
    union = ta | tb
    return len(inter) / len(union)


# ---------------------------------------------------------------------------
# _HCEGBuilder — 核心构建器
# ---------------------------------------------------------------------------

class _HCEGBuilder:
    """
    从 HiTab 表格 JSON 构建 HCEG。

    参数:
        table_json: HiTab 格式的表格 JSON
        question: 问题文本 (可选, 用于添加问题节点和证据边)
        add_spatial: 是否添加空间邻接边
        add_value_nodes: 是否为数值单元格添加独立 value 节点
        max_spatial_dist: 空间边的最大距离 (1=仅相邻)
    """

    def __init__(
        self,
        table_json: Dict[str, Any],
        question: Optional[str] = None,
        add_spatial: bool = True,
        add_value_nodes: bool = True,
        max_spatial_dist: int = 1,
    ):
        self.table = table_json
        self.question = question
        self.texts = table_json.get("texts", [])
        self.top_root = table_json.get("top_root", {})
        self.left_root = table_json.get("left_root", {})
        self.merged_regions = table_json.get("merged_regions", [])
        self.top_header_rows = table_json.get("top_header_rows_num", 1)
        self.left_header_cols = table_json.get("left_header_columns_num", 1)
        self.n_rows = len(self.texts)
        self.n_cols = max((len(row) for row in self.texts), default=0) if self.texts else 0
        self.add_spatial = add_spatial
        self.add_value_nodes = add_value_nodes
        self.max_spatial_dist = max_spatial_dist

        self.graph = HCEG()
        # 内部索引
        self._cell_id_map: Dict[Tuple[int, int], str] = {}  # (r, c) -> node_id
        self._header_nodes: List[str] = []
        self._agg_nodes: List[str] = []
        self._merged_map: Dict[Tuple[int, int], Tuple[int, int]] = {}  # (r,c) -> anchor(r,c)

    def build(self) -> HCEG:
        """构建完整的 HCEG"""
        self._build_merged_map()
        self._add_cell_nodes()
        self._add_header_hierarchy_edges()
        self._identify_aggregation_nodes()
        if self.add_spatial:
            self._add_spatial_edges()
        self._add_semantic_binding_edges()
        self._add_aggregation_dependency_edges()
        if self.add_value_nodes:
            self._add_value_nodes()
        if self.merged_regions:
            self._add_merged_region_edges()
        if self.question:
            self._add_question_node()
        return self.graph

    # ---- 1. 构建合并区域映射 ----

    def _build_merged_map(self) -> None:
        """将合并区域中的每个 (r, c) 映射到锚点 (first_row, first_col)"""
        for mr in self.merged_regions:
            r1 = mr.get("first_row", 0)
            r2 = mr.get("last_row", r1)
            c1 = mr.get("first_column", 0)
            c2 = mr.get("last_column", c1)
            anchor = (r1, c1)
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    if (r, c) != anchor:
                        self._merged_map[(r, c)] = anchor

    # ---- 2. 添加单元格节点 ----

    def _add_cell_nodes(self) -> None:
        """为表格中每个有效单元格创建节点"""
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                # 被合并到其他位置的单元格跳过 (仅保留锚点)
                if (r, c) in self._merged_map:
                    anchor = self._merged_map[(r, c)]
                    self._cell_id_map[(r, c)] = self._cell_id_map.get(anchor, f"cell_{anchor[0]}_{anchor[1]}")
                    continue

                text = self._get_cell_text(r, c)
                node_id = f"cell_{r}_{c}"
                numeric_val = _parse_number(text)

                # 判断是表头还是数据单元格
                is_header = r < self.top_header_rows or c < self.left_header_cols
                ntype = NodeType.HEADER if is_header else NodeType.CELL

                # 确定表头层级
                header_level = -1
                if r < self.top_header_rows:
                    header_level = r
                elif c < self.left_header_cols:
                    header_level = c

                node = GraphNode(
                    node_id=node_id,
                    node_type=ntype,
                    row=r, col=c,
                    text=text,
                    numeric_value=numeric_val,
                    header_level=header_level,
                )
                self.graph.add_node(node)
                self._cell_id_map[(r, c)] = node_id

                if is_header:
                    self._header_nodes.append(node_id)

    # ---- 3. 添加表头层级结构边 ----

    def _add_header_hierarchy_edges(self) -> None:
        """从 top_root / left_root 树结构中添加 parent_header / child_header 边"""
        # 处理列表头树
        if self.top_root and "children" in self.top_root:
            self._traverse_header_tree(self.top_root, is_column_header=True)
        # 处理行表头树
        if self.left_root and "children" in self.left_root:
            self._traverse_header_tree(self.left_root, is_column_header=False)

    def _traverse_header_tree(
        self, node: Dict[str, Any], is_column_header: bool, parent_id: Optional[str] = None
    ) -> None:
        """递归遍历 HiTab 的表头树, 建立层级边"""
        r = node.get("row", -1)
        c = node.get("column", -1)

        current_id = None
        if r >= 0 and c >= 0:
            # 解引用合并映射
            if (r, c) in self._merged_map:
                anchor = self._merged_map[(r, c)]
                current_id = self._cell_id_map.get(anchor)
            else:
                current_id = self._cell_id_map.get((r, c))

            # 建立 parent → child 边
            if parent_id is not None and current_id is not None and parent_id != current_id:
                self.graph.add_edge(GraphEdge(
                    source=parent_id, target=current_id,
                    edge_type=EdgeType.CHILD_HEADER,
                ))
                self.graph.add_edge(GraphEdge(
                    source=current_id, target=parent_id,
                    edge_type=EdgeType.PARENT_HEADER,
                ))

        # 递归子节点
        children = node.get("children", [])
        next_parent = current_id if current_id else parent_id
        for child in children:
            self._traverse_header_tree(child, is_column_header, next_parent)

    # ---- 4. 识别聚合节点 ----

    def _identify_aggregation_nodes(self) -> None:
        """扫描所有单元格, 识别包含聚合关键词的节点并标记为 AGGREGATOR"""
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if (r, c) in self._merged_map:
                    continue
                text_lower = _normalize_text(self._get_cell_text(r, c))
                if not text_lower:
                    continue
                for kw, agg_type in AGGREGATION_KEYWORDS.items():
                    if kw in text_lower:
                        nid = self._cell_id_map.get((r, c))
                        if nid and nid in self.graph.nodes:
                            node = self.graph.nodes[nid]
                            node.node_type = NodeType.AGGREGATOR
                            node.aggregation_type = agg_type
                            self._agg_nodes.append(nid)
                        break

    # ---- 5. 空间邻接边 ----

    def _add_spatial_edges(self) -> None:
        """添加相邻单元格间的空间边 (up/down/left/right)"""
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                src = self._cell_id_map.get((r, c))
                if not src:
                    continue
                for dr, dc, etype in [
                    (-1, 0, EdgeType.UP), (1, 0, EdgeType.DOWN),
                    (0, -1, EdgeType.LEFT), (0, 1, EdgeType.RIGHT),
                ]:
                    for dist in range(1, self.max_spatial_dist + 1):
                        nr, nc = r + dr * dist, c + dc * dist
                        if 0 <= nr < self.n_rows and 0 <= nc < self.n_cols:
                            tgt = self._cell_id_map.get((nr, nc))
                            if tgt and tgt != src:
                                self.graph.add_edge(GraphEdge(
                                    source=src, target=tgt,
                                    edge_type=etype,
                                    weight=1.0 / dist,
                                ))

    # ---- 6. 语义绑定边 ----

    def _add_semantic_binding_edges(self) -> None:
        """
        为每个数据单元格添加:
        - value_under_header: 数据格 -> 列表头
        - row_path: 数据格 -> 行表头
        - col_path: 数据格 -> 列表头
        """
        data_start_row = self.top_header_rows
        data_start_col = self.left_header_cols

        for r in range(data_start_row, self.n_rows):
            for c in range(data_start_col, self.n_cols):
                cell_id = self._cell_id_map.get((r, c))
                if not cell_id:
                    continue

                # 列表头绑定 (同列的所有表头行)
                for hr in range(self.top_header_rows):
                    hid = self._cell_id_map.get((hr, c))
                    if hid and hid != cell_id:
                        self.graph.add_edge(GraphEdge(
                            source=cell_id, target=hid,
                            edge_type=EdgeType.VALUE_UNDER_HEADER,
                        ))
                        self.graph.add_edge(GraphEdge(
                            source=cell_id, target=hid,
                            edge_type=EdgeType.COL_PATH,
                        ))

                # 行表头绑定 (同行的所有左侧表头列)
                for hc in range(self.left_header_cols):
                    hid = self._cell_id_map.get((r, hc))
                    if hid and hid != cell_id:
                        self.graph.add_edge(GraphEdge(
                            source=cell_id, target=hid,
                            edge_type=EdgeType.ROW_PATH,
                        ))

    # ---- 7. 聚合依赖边 ----

    def _add_aggregation_dependency_edges(self) -> None:
        """
        为聚合节点建立 aggregate_depends 边, 连接到其覆盖的数值单元格。

        策略:
        - 聚合在行表头中 → 聚合同列的数据单元格
        - 聚合在列表头中 → 聚合同行的数据单元格
        - 聚合在数据区域 → 同行 or 同列的数据 (由聚合方向判断)
        """
        data_start_row = self.top_header_rows
        data_start_col = self.left_header_cols

        for agg_id in self._agg_nodes:
            agg_node = self.graph.nodes[agg_id]
            r, c = agg_node.row, agg_node.col

            dependent_cells: List[str] = []

            # 策略1: 聚合节点在行表头区域 (c < left_header_cols)
            # 聚合同行的所有数据单元格
            if c < data_start_col:
                for dc in range(data_start_col, self.n_cols):
                    dep_id = self._cell_id_map.get((r, dc))
                    if dep_id and dep_id != agg_id:
                        dependent_cells.append(dep_id)

            # 策略2: 聚合节点在列表头区域 (r < top_header_rows)
            # 聚合同列的所有数据单元格
            elif r < data_start_row:
                for dr in range(data_start_row, self.n_rows):
                    dep_id = self._cell_id_map.get((dr, c))
                    if dep_id and dep_id != agg_id:
                        dependent_cells.append(dep_id)

            # 策略3: 聚合节点在数据区域
            else:
                # 判断聚合方向: 检查同行/同列中哪个方向有更多数据
                same_row_count = 0
                same_col_count = 0
                for dc in range(data_start_col, self.n_cols):
                    if dc != c and self._cell_id_map.get((r, dc)):
                        cid = self._cell_id_map[(r, dc)]
                        nd = self.graph.nodes.get(cid)
                        if nd and nd.is_numeric:
                            same_row_count += 1
                for dr in range(data_start_row, self.n_rows):
                    if dr != r and self._cell_id_map.get((dr, c)):
                        cid = self._cell_id_map[(dr, c)]
                        nd = self.graph.nodes.get(cid)
                        if nd and nd.is_numeric:
                            same_col_count += 1

                if same_col_count >= same_row_count:
                    # 列方向聚合 (更常见: Total 在最后一行)
                    for dr in range(data_start_row, self.n_rows):
                        if dr != r:
                            dep_id = self._cell_id_map.get((dr, c))
                            if dep_id and dep_id != agg_id:
                                nd = self.graph.nodes.get(dep_id)
                                if nd and nd.node_type != NodeType.AGGREGATOR:
                                    dependent_cells.append(dep_id)
                else:
                    # 行方向聚合
                    for dc in range(data_start_col, self.n_cols):
                        if dc != c:
                            dep_id = self._cell_id_map.get((r, dc))
                            if dep_id and dep_id != agg_id:
                                nd = self.graph.nodes.get(dep_id)
                                if nd and nd.node_type != NodeType.AGGREGATOR:
                                    dependent_cells.append(dep_id)

            # 添加聚合依赖边
            for dep_id in dependent_cells:
                self.graph.add_edge(GraphEdge(
                    source=agg_id, target=dep_id,
                    edge_type=EdgeType.AGGREGATE_DEPENDS,
                ))

    # ---- 8. 数值 Value 节点 ----

    def _add_value_nodes(self) -> None:
        """为数值单元格添加独立的 VALUE 节点 (便于执行器引用)"""
        data_start_row = self.top_header_rows
        data_start_col = self.left_header_cols

        for r in range(data_start_row, self.n_rows):
            for c in range(data_start_col, self.n_cols):
                cell_id = self._cell_id_map.get((r, c))
                if not cell_id:
                    continue
                node = self.graph.nodes.get(cell_id)
                if not node or not node.is_numeric:
                    continue

                val_id = f"val_{r}_{c}"
                val_node = GraphNode(
                    node_id=val_id,
                    node_type=NodeType.VALUE,
                    row=r, col=c,
                    text=str(node.numeric_value),
                    numeric_value=node.numeric_value,
                )
                self.graph.add_node(val_node)
                self.graph.add_edge(GraphEdge(
                    source=cell_id, target=val_id,
                    edge_type=EdgeType.PART_OF,
                    metadata={"relation": "cell_has_value"},
                ))

    # ---- 9. 合并区域边 ----

    def _add_merged_region_edges(self) -> None:
        """为合并区域添加 SPAN 节点和 merged_into / span_of 边"""
        for idx, mr in enumerate(self.merged_regions):
            r1 = mr.get("first_row", 0)
            r2 = mr.get("last_row", r1)
            c1 = mr.get("first_column", 0)
            c2 = mr.get("last_column", c1)

            # 仅对跨越多个单元格的区域创建 span 节点
            if r1 == r2 and c1 == c2:
                continue

            anchor_text = self._get_cell_text(r1, c1)
            span_id = f"span_{idx}_{r1}_{c1}"
            span_node = GraphNode(
                node_id=span_id,
                node_type=NodeType.SPAN,
                row=r1, col=c1,
                text=anchor_text,
                metadata={"rows": [r1, r2], "cols": [c1, c2]},
            )
            self.graph.add_node(span_node)

            # 锚点单元格 -> span
            anchor_id = self._cell_id_map.get((r1, c1))
            if anchor_id:
                self.graph.add_edge(GraphEdge(
                    source=anchor_id, target=span_id,
                    edge_type=EdgeType.SPAN_OF,
                ))

            # 区域内所有被合并的位置 -> span
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    if (r, c) == (r1, c1):
                        continue
                    cid = self._cell_id_map.get((r, c))
                    if cid:
                        self.graph.add_edge(GraphEdge(
                            source=cid, target=span_id,
                            edge_type=EdgeType.MERGED_INTO,
                        ))

    # ---- 10. 问题节点和证据边 ----

    def _add_question_node(self) -> None:
        """添加问题节点, 并将问题中提及的实体与表格节点用 entity_mention 边连接"""
        q_id = "question_0"
        q_node = GraphNode(
            node_id=q_id,
            node_type=NodeType.QUESTION,
            text=self.question or "",
        )
        self.graph.add_node(q_node)

        if not self.question:
            return

        q_lower = _normalize_text(self.question)

        # 对每个表头/单元格计算与问题的文本重叠
        for nid, node in list(self.graph.nodes.items()):
            if node.node_type in (NodeType.QUESTION, NodeType.VALUE, NodeType.SPAN):
                continue
            if not node.text.strip():
                continue

            overlap = _token_overlap(q_lower, node.text)
            # 也检查子串匹配
            node_lower = _normalize_text(node.text)
            is_substring = len(node_lower) >= 2 and node_lower in q_lower

            if overlap >= 0.3 or is_substring:
                weight = max(overlap, 0.8 if is_substring else 0.0)
                self.graph.add_edge(GraphEdge(
                    source=q_id, target=nid,
                    edge_type=EdgeType.ENTITY_MENTION,
                    weight=weight,
                    metadata={"overlap": overlap, "substring": is_substring},
                ))

        # 检测操作类型需求
        op_keywords = {
            "sum": ["sum", "total", "add", "combined", "合计", "总"],
            "diff": ["difference", "subtract", "decrease", "increase", "change", "变化", "差"],
            "ratio": ["ratio", "percent", "proportion", "share", "%", "占比", "比例"],
            "compare": ["more", "less", "greater", "higher", "lower", "most", "least",
                        "largest", "smallest", "biggest", "最大", "最小", "最多", "最少"],
            "count": ["how many", "count", "number of", "多少", "几个"],
            "average": ["average", "mean", "平均"],
        }
        for op, keywords in op_keywords.items():
            if any(kw in q_lower for kw in keywords):
                self.graph.nodes[q_id].metadata["detected_op"] = op
                # 找到聚合节点中匹配的
                for agg_id in self._agg_nodes:
                    agg = self.graph.nodes[agg_id]
                    if agg.aggregation_type == op or (op == "compare" and agg.is_numeric):
                        self.graph.add_edge(GraphEdge(
                            source=q_id, target=agg_id,
                            edge_type=EdgeType.OP_DEMAND,
                            metadata={"op_type": op},
                        ))
                break

    # ---- 工具方法 ----

    def _get_cell_text(self, r: int, c: int) -> str:
        if 0 <= r < len(self.texts) and 0 <= c < len(self.texts[r]):
            cell = self.texts[r][c]
            if isinstance(cell, list):
                return " ".join(str(x) for x in cell)
            return str(cell) if cell is not None else ""
        return ""




# ---------------------------------------------------------------------------
# 便捷接口
# ---------------------------------------------------------------------------

def build_hceg(
    table_json: Dict[str, Any],
    question: Optional[str] = None,
    add_spatial: bool = True,
    add_value_nodes: bool = True,
    max_spatial_dist: int = 1,
) -> HCEG:
    """一步构建 HCEG 的便捷函数"""
    builder = _HCEGBuilder(
        table_json=table_json,
        question=question,
        add_spatial=add_spatial,
        add_value_nodes=add_value_nodes,
        max_spatial_dist=max_spatial_dist,
    )
    return builder.build()



# ---------------------------------------------------------------------------
# CLI: 测试构建并打印统计
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="HCEG Builder - 测试构建")
    parser.add_argument("--table", required=True, help="HiTab 表格 JSON 文件路径")
    parser.add_argument("--question", default=None, help="问题文本")
    parser.add_argument("--output", default=None, help="输出 HCEG JSON 路径")
    parser.add_argument("--no-spatial", action="store_true", help="不添加空间边")
    args = parser.parse_args()

    with open(args.table, "r", encoding="utf-8") as f:
        table_json = json.load(f)

    graph = build_hceg(
        table_json=table_json,
        question=args.question,
        add_spatial=not args.no_spatial,
    )

    stats = graph.stats()
    print("=== HCEG 构建统计 ===")
    print(f"  节点总数: {stats['total_nodes']}")
    print(f"  边总数:   {stats['total_edges']}")
    print(f"  节点类型分布:")
    for k, v in sorted(stats["node_types"].items()):
        print(f"    {k}: {v}")
    print(f"  边类型分布:")
    for k, v in sorted(stats["edge_types"].items()):
        print(f"    {k}: {v}")

    # 打印聚合节点
    agg_nodes = graph.get_nodes_by_type(NodeType.AGGREGATOR)
    if agg_nodes:
        print(f"\n  聚合节点 ({len(agg_nodes)}):")
        for n in agg_nodes[:10]:
            deps = graph.neighbors(n.node_id, {EdgeType.AGGREGATE_DEPENDS})
            print(f"    [{n.row},{n.col}] \"{n.text}\" ({n.aggregation_type}) -> {len(deps)} 依赖")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(graph.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"\n  已保存到: {args.output}")
