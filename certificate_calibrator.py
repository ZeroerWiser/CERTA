"""
certificate_calibrator.py — CSCR Phase 6: Certificate Matrix + Dominance 决策

蓝图规范 (v2 §3.2, §6.2-6.3):

核心思想：
  1. 为每个候选答案构建 Certificate (多维信任凭证)
  2. 使用 SCCI (图干预) + LLM confidence + executor validity 构建 Certificate Matrix
  3. Certificate Dominance 规则：无权重的候选选择，优先 lookup 路径
  4. Conformal Abstention: 基于校准集学习 SCCI 阈值

与 Ca2KG CCI 的核心区别：
  - CCI: prompt-level 干预 → LLM概率一致性 → 退化为频率投票
  - SCCI: graph-evidence-level 干预 → 执行器确定性输出变化 → 无退化

v5.0 升级:
  方向 B: Conformal Abstention — 基于综合置信度的校准弃答
  方向 C: 多候选闭包 — 完整候选生成 + Certificate Dominance 决策

v6.0 升级:
  - Graph-Aware SCCI: 使用 GraphAwareExecutor 在干预图上执行 → SCCI 不再退化
  - Path-Verified Consensus: 通过图路径验证的共识候选优先级更高
  - Conformal Abstention 激活: SCCI 作为主校准信号
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from evidence_retriever import (
    EvidenceSubgraph,
    InterventionResult,
    SCCIResult,
    compute_scci,
)
from executor import (
    ExecutorResult,
    GraphAwareExecutor,
    OperationType,
)

from causal_predictor import predict_success
from graph_builder import HCEG, EdgeType, NodeType

from structural_cert_utils import (
    candidate_evidence_alignment,
    generate_candidate_targeted_interventions,
    evidence_is_fallback,
)


# ---------------------------------------------------------------------------
# Certificate 数据结构
# ---------------------------------------------------------------------------

@dataclass
class Certificate:
    """候选答案的多维信任凭证 (蓝图 v2 §6.2, v6.0 升级)"""
    executor_valid: bool = False
    path_complete: bool = False
    is_lookup: bool = False
    is_lookup_aggregate: bool = False
    aggregate_node_verified: bool = False
    binding_confidence: float = 0.0
    llm_confidence: float = 0.0
    scci: float = 0.0
    bir: float = 0.0
    asr: float = 0.0
    consensus_with_llm: bool = False
    evidence_retained: bool = True
    operator_compatible: bool = True
    # v6.0: Graph path verification
    path_verified: bool = False        # 图路径中每条边都在原图中存在
    graph_path: list = field(default_factory=list)  # 从问题到答案的图路径
    evidence_alignment: Dict[str, Any] = field(default_factory=dict)
    candidate_evidence_coverage: float = 0.0
    candidate_effective_evidence_coverage: float = 0.0
    ib_mdl_score: float = 0.0
    evidence_fallback: bool = False

    def dominance_score(self) -> float:
        """层次化排序评分（非权重融合）"""
        score = 0.0
        if self.is_lookup_aggregate and self.aggregate_node_verified:
            score += 100.0
        elif self.is_lookup:
            score += 50.0
        if self.consensus_with_llm:
            score += 30.0
        # v6.0: path_verified 候选获得额外排序加分
        if self.path_verified:
            score += 15.0
        score += self.scci * 20.0
        if self.executor_valid:
            score += 10.0
        score += self.llm_confidence * 5.0
        return score

    def composite_confidence(self) -> float:
        """综合置信度 (v6.0: SCCI 作为主校准信号)

        融合 LLM confidence + binding confidence + SCCI:
        - 这不是决策用的分数，而是用于校准/弃答判断
        - v6.0: SCCI 现在有真实值 (不再全为0), 给予更大权重
        """
        base = self.llm_confidence

        # 如果有共识，提升置信度
        if self.consensus_with_llm:
            base = min(1.0, base * 1.15)

        # v6.0: SCCI 有真实信号后, 用它调整置信度
        if self.scci > 0.3 and self.executor_valid:
            base = min(1.0, base + self.scci * 0.10)
        elif self.scci == 0.0 and self.executor_valid and self.asr == 0.0:
            # SCCI=0 且 ASR=0 意味着 adversarial 干预未导致翻转
            # 这可能说明执行器结果不依赖关键证据 → 不可靠
            base *= 0.97

        if self.executor_valid and self.binding_confidence < 0.3:
            base *= 0.95
        if self.executor_valid and self.evidence_fallback:
            base *= 0.85
        if self.executor_valid and self.candidate_effective_evidence_coverage and self.candidate_effective_evidence_coverage < 0.8:
            base *= 0.97

        # v6.0: path_verified 提升置信度
        if self.path_verified:
            base = min(1.0, base * 1.05)

        return base

    def to_dict(self) -> Dict[str, Any]:
        return {
            "executor_valid": self.executor_valid,
            "path_complete": self.path_complete,
            "is_lookup": self.is_lookup,
            "is_lookup_aggregate": self.is_lookup_aggregate,
            "aggregate_node_verified": self.aggregate_node_verified,
            "consensus_with_llm": self.consensus_with_llm,
            "binding_confidence": round(self.binding_confidence, 4),
            "scci": round(self.scci, 4),
            "bir": round(self.bir, 4),
            "asr": round(self.asr, 4),
            "path_verified": self.path_verified,
            "candidate_evidence_coverage": round(self.candidate_evidence_coverage, 4),
            "candidate_effective_evidence_coverage": round(self.candidate_effective_evidence_coverage, 4),
            "ib_mdl_score": round(self.ib_mdl_score, 4),
            "evidence_fallback": self.evidence_fallback,
            "evidence_alignment": self.evidence_alignment,
            "dominance_score": round(self.dominance_score(), 4),
            "composite_confidence": round(self.composite_confidence(), 4),
        }


@dataclass
class CertifiedCandidate:
    """带证书的候选答案"""
    denotation: str
    operation: OperationType
    priority: int
    certificate: Certificate
    cells_used: list = field(default_factory=list)
    computation_trace: str = ""
    source: str = ""
    operation_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.certificate.executor_valid and bool(self.denotation)


def _normalize_candidate_denotation(value: Any) -> str:
    try:
        from eval_utils import normalize_text

        return normalize_text(value)
    except Exception:
        return str(value or "").strip().lower()


def _serialize_cell_ref(ref: Any) -> Dict[str, Any]:
    return {
        "row": getattr(ref, "row", None),
        "col": getattr(ref, "col", None),
        "value": str(getattr(ref, "value", "")),
        "row_headers": [str(x) for x in (getattr(ref, "row_headers", []) or [])],
        "col_headers": [str(x) for x in (getattr(ref, "col_headers", []) or [])],
    }


def _unknown_certificate_field(field_name: str) -> Dict[str, Any]:
    return {
        "value": "unknown",
        "availability": "uncomputed",
        "provenance": "not_computed_in_certificate_calibrator",
        "field": field_name,
    }


def _serialize_certified_candidate_full(candidate: CertifiedCandidate, index: int) -> Dict[str, Any]:
    cert = candidate.certificate
    certificate_payload = cert.to_dict()
    certificate_payload.update({
        "binding_confidence": round(cert.binding_confidence, 4),
        "llm_confidence": round(cert.llm_confidence, 4),
        "path_complete": cert.path_complete,
        "evidence_retained": cert.evidence_retained,
        "operation_compatible": bool(getattr(cert, "operator_compatible", True)),
        "unit_compatible": _unknown_certificate_field("unit_compatible"),
        "scale_compatible": _unknown_certificate_field("scale_compatible"),
        "answer_role_compatible": _unknown_certificate_field("answer_role_compatible"),
        "graph_path": cert.graph_path[:16],
        "dominance_score": round(cert.dominance_score(), 4),
        "legacy_heuristic_used": _unknown_certificate_field("legacy_heuristic_used"),
        "legacy_heuristic_source": "uncomputed",
    })
    return {
        "candidate_id": f"cand_{index}",
        "denotation": candidate.denotation,
        "normalized_denotation": _normalize_candidate_denotation(candidate.denotation),
        "operation": candidate.operation.value,
        "priority": candidate.priority,
        "cells_used": [_serialize_cell_ref(ref) for ref in (candidate.cells_used or [])],
        "computation_trace": candidate.computation_trace,
        "operation_metadata": dict(candidate.operation_metadata or {}),
        "source": candidate.source,
        "certificate": certificate_payload,
    }


# ---------------------------------------------------------------------------
# Certificate Builder
# ---------------------------------------------------------------------------

class CertificateBuilder:
    """为候选答案构建 Certificate"""

    def __init__(
        self,
        table_json: dict,
        question: str,
        evidence: Optional[EvidenceSubgraph] = None,
        graph: Optional[HCEG] = None,
    ):
        self.table_json = table_json
        self.question = question
        self.evidence = evidence
        self.graph = graph

    def build_certificate(
        self,
        executor_result: Optional[ExecutorResult],
        llm_answer: str = "",
        llm_confidence: float = 0.5,
        scci_result: Optional[SCCIResult] = None,
    ) -> Certificate:
        """为单个候选构建 Certificate"""
        cert = Certificate()

        # --- 执行器维度 ---
        if executor_result:
            cert.executor_valid = executor_result.executor_valid
            cert.binding_confidence = executor_result.confidence
            cert.is_lookup = executor_result.priority <= 2
            cert.is_lookup_aggregate = (
                executor_result.operation == OperationType.LOOKUP_AGGREGATE
            )
            cert.operator_compatible = True
            if cert.is_lookup_aggregate and self.evidence:
                cert.aggregate_node_verified = self._verify_aggregate_coverage(
                    executor_result
                )

        # --- LLM 维度 ---
        cert.llm_confidence = llm_confidence

        # --- 共识维度 ---
        if executor_result and executor_result.denotation:
            from eval_utils import normalize_text
            exec_norm = normalize_text(executor_result.denotation)
            llm_norm = normalize_text(llm_answer)
            cert.consensus_with_llm = (exec_norm == llm_norm)

        # --- SCCI 维度 ---
        if scci_result:
            cert.scci = scci_result.scci
            cert.bir = scci_result.bir
            cert.asr = scci_result.asr

        # --- 证据路径完整性 + 候选证据对齐 ---
        if self.evidence:
            cert.path_complete = self._check_path_completeness(executor_result)
            cert.evidence_retained = len(self.evidence.anchor_nodes) > 0
            cert.evidence_fallback = evidence_is_fallback(self.evidence)
            if executor_result:
                align = candidate_evidence_alignment(executor_result, self.evidence)
                cert.evidence_alignment = align
                cert.candidate_evidence_coverage = float(align.get("coverage", 0.0) or 0.0)
                cert.candidate_effective_evidence_coverage = float(align.get("effective_coverage", cert.candidate_evidence_coverage) or 0.0)
                if align.get("candidate_cells"):
                    cert.path_complete = cert.path_complete and bool(align.get("path_complete_by_cells"))
                    cert.evidence_retained = cert.evidence_retained and cert.candidate_effective_evidence_coverage >= 0.8
                    cert.binding_confidence *= max(0.20, cert.candidate_effective_evidence_coverage)
            cert.ib_mdl_score = float(self.evidence.metadata.get("ib_mdl_score", 0.0) or 0.0)

        # --- v6.0: 图路径验证 ---
        # 使用 GraphAwareExecutor 在原始图上执行, 获取 graph_path
        # 然后验证路径中每条边是否存在于原图中
        if self.graph and executor_result and executor_result.executor_valid:
            graph_exec = GraphAwareExecutor(self.graph, self.table_json)
            _, graph_path = graph_exec.execute_with_path(self.question)
            if graph_path:
                cert.graph_path = graph_path
                # 验证: 路径中每条边是否在原图中真实存在
                cert.path_verified = self._verify_graph_path(graph_path)

        return cert

    def _verify_aggregate_coverage(self, executor_result: ExecutorResult) -> bool:
        """验证聚合节点是否覆盖了问题要求的范围"""
        if not self.evidence:
            return False
        agg_nodes = self.evidence.graph.get_nodes_by_type(NodeType.AGGREGATOR)
        if not agg_nodes:
            return False
        for node in agg_nodes:
            deps = self.evidence.graph.neighbors(
                node.node_id, {EdgeType.AGGREGATE_DEPENDS}
            )
            if deps:
                return True
        return False

    def _check_path_completeness(
        self, executor_result: Optional[ExecutorResult]
    ) -> bool:
        """检查从问题到答案的证据路径是否完整"""
        if not self.evidence or not executor_result:
            return False
        if not self.evidence.anchor_nodes:
            return False
        if not executor_result.cells_used:
            return True
        for cell in executor_result.cells_used:
            cell_id = f"cell_{cell.row}_{cell.col}"
            if cell_id not in self.evidence.evidence_nodes:
                return False
        return True

    def _verify_graph_path(self, graph_path: List[Dict[str, str]]) -> bool:
        """验证图路径中的每条边是否在原始 HCEG 图中真实存在 (v6.0)

        这提供了语义验证: 不仅检查文本匹配 (binding_confidence),
        还验证从问题到答案的图拓扑路径完整性。

        蓝图 v2 §6.3: path_complete 验证的升级版本。
        """
        if not self.graph or not graph_path:
            return False
        for edge_info in graph_path:
            src = edge_info.get("source", "")
            tgt = edge_info.get("target", "")
            etype = edge_info.get("edge_type", "")
            # 检查边是否存在于原图中
            found = False
            for e in self.graph._adj.get(src, []):
                if e.target == tgt and e.edge_type.value == etype:
                    found = True
                    break
            if not found:
                # 也检查反向 (某些边类型可双向遍历)
                for e in self.graph._rev_adj.get(tgt, []):
                    if e.source == src and e.edge_type.value == etype:
                        found = True
                        break
            if not found:
                return False
        return True


# ---------------------------------------------------------------------------
# Certificate Dominance 决策器
# ---------------------------------------------------------------------------

class CertificateDominance:
    """
    Certificate Dominance 决策 (蓝图 v2 §6.3, v6.0 升级):

    v4.0-v4.2 实验教训:
    - cert_lookup_aggregate: 255 次触发, EM=4.7%, 灾难性回归
    - cert_lookup_cell: 24 次触发, EM=12.5%, 仍有回归
    - 根因: executor binding 精度不够，不能信任 executor 单独决策

    v6.0 策略:
    1. 共识候选 (executor + LLM 一致) → 最优先 (已验证 83.9% EM)
    1b. 路径验证共识 (v6.0): 共识 + path_verified → 更高可信度
    2. 多候选竞争: 当存在多个 executor 候选时，选 binding 最高的
    3. [已禁用] lookup_cell 覆写
    4. [保留但不启用] lookup_aggregate
    5. Conformal Abstention: 基于校准阈值的弃答
    6. Fallback → LLM (SCCI 辅助微调)
    """

    def __init__(self, scci_threshold: float = 0.3, fallback_to_llm: bool = True,
                 conformal_abstainer: Optional['ConformalAbstainer'] = None,
                 success_predictor: Optional[Any] = None,
                 pipeline_result: Optional[Dict[str, Any]] = None):
        self.scci_threshold = scci_threshold
        self.fallback_to_llm = fallback_to_llm
        self.conformal_abstainer = conformal_abstainer
        self.success_predictor = success_predictor
        self.pipeline_result = pipeline_result

    def select(
        self,
        candidates: List[CertifiedCandidate],
        llm_answer: str = "",
        llm_confidence: float = 0.5,
    ) -> Tuple[str, str, float, Optional[Certificate]]:
        """
        从候选中选择最终答案 (v6.0: 多候选 + path_verified + conformal)

        返回: (final_answer, source, confidence, certificate)
        """
        if not candidates:
            return llm_answer, "llm_only", llm_confidence, None

        valid = [c for c in candidates if c.is_valid]

        # --- 规则 1: 共识候选最优先 ---
        # v3/v4 实证: consensus EM=83.9% (118 样本), 是最可靠的信号
        consensus = [c for c in valid if c.certificate.consensus_with_llm]
        if consensus:
            # v6.0: 路径验证的共识候选获得更高可信度提升
            path_verified_consensus = [
                c for c in consensus if c.certificate.path_verified
            ]
            if path_verified_consensus:
                best = max(path_verified_consensus,
                           key=lambda c: c.certificate.dominance_score())
                boosted = min(
                    1.0,
                    max(llm_confidence, best.certificate.binding_confidence) * 1.20,
                )
                return best.denotation, "path_verified_consensus", boosted, best.certificate

            # 普通共识 (无路径验证)
            best = max(consensus, key=lambda c: c.certificate.dominance_score())
            boosted = min(
                1.0,
                max(llm_confidence, best.certificate.binding_confidence) * 1.15,
            )
            return best.denotation, "consensus_cert", boosted, best.certificate

        # --- 规则 2: 多候选共识 (v5.0 方向 C) ---
        # 如果多个 executor 候选给出相同答案，提升信任度
        if len(valid) >= 2:
            from eval_utils import normalize_text
            answer_counts: Dict[str, List[CertifiedCandidate]] = {}
            for c in valid:
                norm = normalize_text(c.denotation)
                answer_counts.setdefault(norm, []).append(c)
            # 找到获得最多操作类型支持的答案
            for norm_ans, group in sorted(answer_counts.items(),
                                           key=lambda x: len(x[1]), reverse=True):
                if len(group) >= 2:
                    best = max(group, key=lambda c: c.certificate.dominance_score())
                    conf = min(1.0, max(c.certificate.binding_confidence for c in group) * 1.1)
                    return best.denotation, "multi_executor_consensus", conf, best.certificate

        # --- 规则 3: SCCI + Success Predictor 驱动覆写 (v8.5 收紧) ---
        # v8.5 诊断结论：success_predictor_v2.pt 来自 7B predictions，跨模型/API 黑盒
        # 泛化不能假设成立。因此 SP 不再允许单独驱动覆写，只能作为 path-verified
        # executor 候选的弱门控信号，避免跨模型分布偏移造成错误覆盖。
        # 五条件门控:
        #   1. executor 候选的 SCCI > τ_scci (因果稳定性)
        #   2. LLM confidence < τ_llm (LLM 不确定)
        #   3. Success Predictor 对该操作的预测 > τ_sp (学习到的成功概率)
        #   4. binding_confidence > τ_bind (绑定质量)
        #   5. path_verified=True (结构路径验证；跨模型安全阀)
        if valid and self.success_predictor is not None:
            # 构建 pred_like 用于 SP 推理
            if self.pipeline_result is not None:
                _pred_like = dict(self.pipeline_result)
                # cert_info 尚未完成, 但已有 scci/bir/asr
                _ci = _pred_like.get("certificate_info", {})
                if valid:
                    best_valid = max(valid, key=lambda c: c.certificate.dominance_score())
                    _ci["scci"] = best_valid.certificate.scci
                    _ci["bir"] = best_valid.certificate.bir
                    _ci["asr"] = best_valid.certificate.asr
                    _ci["num_candidates"] = len(valid)
                    _ci["calibration_data"] = {
                        "best_binding_conf": best_valid.certificate.binding_confidence,
                        "any_path_verified": best_valid.certificate.path_verified,
                    }
                    _ci["candidate_details"] = [{
                        "dominance_score": best_valid.certificate.dominance_score(),
                    }]
                _pred_like["certificate_info"] = _ci
                sp_probs = predict_success(self.success_predictor, _pred_like)
            else:
                sp_probs = {}

            if sp_probs:
                # 门控阈值（v8.5: 跨模型安全优先，阈值比 v7.0b 更保守）
                TAU_SCCI = 0.5       # 因果稳定性阈值
                TAU_LLM = 0.55       # LLM 不确定性阈值 (confidence < 此值)
                TAU_SP = 0.70        # SP 成功概率阈值
                TAU_BIND = 0.7       # 绑定质量阈值

                for c in sorted(valid, key=lambda c: c.certificate.dominance_score(), reverse=True):
                    op_name = c.operation.value if hasattr(c.operation, 'value') else str(c.operation)
                    sp_prob = sp_probs.get(op_name, 0.0)

                    gate_scci = c.certificate.scci > TAU_SCCI
                    gate_llm = llm_confidence < TAU_LLM
                    gate_sp = sp_prob > TAU_SP
                    gate_bind = c.certificate.binding_confidence > TAU_BIND
                    gate_path = c.certificate.path_verified  # 额外: 路径验证

                    if gate_scci and gate_llm and gate_sp and gate_bind and gate_path:
                        boosted = min(1.0, max(sp_prob, c.certificate.binding_confidence) * 1.05)
                        return c.denotation, "scci_sp_path_verified_overwrite", boosted, c.certificate

        # --- 规则 4: [已禁用] lookup_cell 覆写 ---
        # v5.0 实证: 30 次触发, EM=10.0% (3/30), 导致 15 个回归
        # 根因: binding_confidence=1.0 不代表绑定的是正确的目标单元格

        # --- 规则 5: Conformal Abstention (v6.0: SCCI 主导) ---
        if self.conformal_abstainer and self.conformal_abstainer.calibrated:
            composite_conf = llm_confidence
            if valid:
                # v6.0: 使用最佳候选的 composite confidence (含 SCCI 信号)
                best_composite = max(c.certificate.composite_confidence() for c in valid)
                composite_conf = max(composite_conf, best_composite)
            if self.conformal_abstainer.should_abstain(composite_conf):
                return "", "conformal_abstain", 0.0, None

        # --- Fallback: 信任 LLM (SCCI 辅助置信度微调) ---
        if self.fallback_to_llm and llm_answer:
            adjusted_conf = llm_confidence
            if valid:
                best_scci = max(c.certificate.scci for c in valid)
                # v6.0: SCCI 现在有真实信号, 可以做更有意义的调整
                if best_scci > 0.5 and llm_confidence < 0.7:
                    # 高 SCCI 意味着执行器答案有结构因果支持
                    # 但未通过共识 → LLM 答案不同 → 降低 LLM 置信度
                    adjusted_conf = llm_confidence * 0.90
                elif best_scci > 0.3 and llm_confidence < 0.5:
                    adjusted_conf = llm_confidence * 0.95
            return llm_answer, "llm_cert_adjusted", adjusted_conf, None

        return "", "abstain", 0.0, None


# ---------------------------------------------------------------------------
# Conformal Abstention (v6.0: SCCI 主导校准)
# ---------------------------------------------------------------------------

class ConformalAbstainer:
    """
    Conformal Risk Control (蓝图 v2 §3.3, v6.0 升级)

    在校准集上学习综合置信度阈值 τ*:
    τ* = argmin τ s.t. P(a_gold ∈ {c : conf(c) ≥ τ}) ≥ 1 - α

    v6.0 升级: SCCI 有真实信号后, 用 composite_confidence (含 SCCI) 作为校准维度
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.threshold: float = 0.0
        self.calibrated: bool = False
        self._calibration_stats: Dict[str, Any] = {}

    def calibrate(
        self,
        calibration_data: List[Tuple[float, bool]],
    ) -> float:
        """从校准数据学习置信度阈值

        Args:
            calibration_data: List of (composite_confidence, is_correct) tuples
        Returns:
            learned threshold τ*
        """
        if not calibration_data:
            self.threshold = 0.0
            self.calibrated = True
            return self.threshold

        sorted_data = sorted(calibration_data, key=lambda x: x[0], reverse=True)
        n = len(sorted_data)
        target_coverage = 1.0 - self.alpha
        best_tau = 0.0
        best_coverage = 0.0
        best_precision = 0.0

        # v6.1: 两层校准策略
        # 层1: 严格 — 找 precision >= 1-α 的最高阈值 (conformal 保证)
        # 层2: 宽松 — 如果层1 失败, 找 precision 最高的合理阈值 (选择性预测)
        best_relaxed_tau = 0.0
        best_relaxed_precision = 0.0
        best_relaxed_coverage = 0.0

        for i in range(n):
            tau = sorted_data[i][0]
            above = [(s, c) for s, c in sorted_data if s >= tau]
            if not above:
                continue
            correct_above = sum(1 for _, c in above if c)
            coverage = len(above) / n  # 非弃答率
            precision = correct_above / len(above)  # 非弃答准确率

            # 层1: 严格 conformal
            if precision >= target_coverage:
                if tau > best_tau:
                    best_tau = tau
                    best_coverage = coverage
                    best_precision = precision

            # 层2: 选择性预测 — 在合理覆盖率下 (>=30%) 找最高精度
            if coverage >= 0.3 and precision > best_relaxed_precision:
                best_relaxed_tau = tau
                best_relaxed_precision = precision
                best_relaxed_coverage = coverage

        # v6.1: 如果严格阈值为 0 (未达标), 使用宽松阈值
        if best_tau == 0.0 and best_relaxed_tau > 0.0:
            best_tau = best_relaxed_tau
            best_coverage = best_relaxed_coverage
            best_precision = best_relaxed_precision

        self.threshold = best_tau
        self.calibrated = True
        self._calibration_stats = {
            "alpha": self.alpha,
            "target_coverage": target_coverage,
            "learned_threshold": round(best_tau, 4),
            "non_abstention_rate": round(best_coverage, 4),
            "precision_at_threshold": round(best_precision, 4),
            "calibration_size": n,
            "relaxed_fallback": best_tau == best_relaxed_tau and best_tau > 0,
        }
        return self.threshold

    def calibrate_from_predictions(self, predictions_path: str) -> float:
        """从已有的 predictions.jsonl 文件学习阈值 (v6.0: SCCI 主导)

        v6.0 升级: 优先使用 composite_confidence (包含 SCCI 信号),
        如果 predictions 中有 certificate_info.calibration_data.composite_confidence
        则使用它, 否则 fallback 到 final_confidence。
        """
        calibration_data = []
        with open(predictions_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    pred = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in calibration predictions at line {line_no}: {predictions_path}"
                    ) from exc
                # v6.0: 优先使用 composite_confidence
                cert_cal = pred.get("certificate_info", {}).get("calibration_data", {})
                conf = cert_cal.get(
                    "composite_confidence",
                    pred.get("final_confidence", pred.get("llm_confidence", 0.5))
                )
                correct = pred.get("hitab_official_em", False)
                calibration_data.append((float(conf), bool(correct)))

        if not calibration_data:
            raise ValueError(f"No calibration rows found in {predictions_path}")

        return self.calibrate(calibration_data)

    def should_abstain(self, composite_confidence: float) -> bool:
        """判断是否应该弃答"""
        if not self.calibrated:
            return False
        return composite_confidence < self.threshold

    def get_stats(self) -> Dict[str, Any]:
        """返回校准统计信息"""
        return self._calibration_stats.copy()


# ---------------------------------------------------------------------------
# 统一的 Certificate-aware 仲裁器 (v6.0: Graph-Aware SCCI)
# ---------------------------------------------------------------------------

def certificate_aware_arbitrate(
    llm_answer: str,
    llm_confidence: float,
    executor_result: Optional[ExecutorResult],
    table_json: dict,
    question: str,
    evidence: Optional[EvidenceSubgraph] = None,
    graph: Optional[HCEG] = None,
    interventions: Optional[List[InterventionResult]] = None,
    scci_threshold: float = 0.1,
    all_candidates: Optional[List[ExecutorResult]] = None,
    conformal_abstainer: Optional[ConformalAbstainer] = None,
    success_predictor: Optional[Any] = None,
    pipeline_result: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str, float, Dict[str, Any]]:
    """
    Certificate-aware 仲裁器 (v6.0: Graph-Aware SCCI + Path Consensus)

    v6.0 升级:
    - executor_fn 使用 GraphAwareExecutor (在干预图上执行)
      → 干预改变图拓扑 → executor 输出改变 → SCCI 不再退化
    - Certificate 包含 graph_path 和 path_verified 字段
    - Conformal Abstention 可选激活

    返回: (final_answer, source, confidence, certificate_info)
    """
    cert_info: Dict[str, Any] = {}

    # --- 确定候选列表 ---
    if all_candidates is None:
        # 向后兼容: 只有单一 executor 结果
        candidate_results = [executor_result] if (executor_result and executor_result.executor_valid) else []
    else:
        candidate_results = [c for c in all_candidates if c.executor_valid]

    # --- 快速路径：无执行器候选 ---
    if not candidate_results:
        return llm_answer, "llm_only", llm_confidence, cert_info

    # --- SCCI executor hook ---
    def executor_fn(intervened_graph, q):
        graph_exec = GraphAwareExecutor(intervened_graph, table_json)
        return graph_exec.execute(q)

    legacy_scci_result = None
    use_candidate_scci = not bool(
        pipeline_result.get("disable_candidate_scci", False) if isinstance(pipeline_result, dict) else False
    )
    primary_denotation = candidate_results[0].denotation if candidate_results else None
    if (not use_candidate_scci) and interventions and primary_denotation:
        legacy_scci_result = compute_scci(
            original_denotation=primary_denotation,
            interventions=interventions,
            executor_fn=executor_fn,
            question=question,
        )
        cert_info["scci"] = round(legacy_scci_result.scci, 4)
        cert_info["bir"] = round(legacy_scci_result.bir, 4)
        cert_info["asr"] = round(legacy_scci_result.asr, 4)
        cert_info["scci_mode"] = "legacy_sample_level"
    else:
        cert_info["scci_mode"] = "candidate_specific"

    # --- 构建 Certificate Builder ---
    builder = CertificateBuilder(
        table_json=table_json,
        question=question,
        evidence=evidence,
        graph=graph,
    )

    # --- 为每个 executor 候选构建 Certificate ---
    certified_candidates: List[CertifiedCandidate] = []
    candidate_scci_values: List[float] = []
    candidate_intervention_counts: List[int] = []
    for exec_result in candidate_results:
        candidate_scci_result = legacy_scci_result
        if use_candidate_scci and graph is not None:
            targeted = generate_candidate_targeted_interventions(
                graph=graph,
                evidence=evidence,
                candidate=exec_result,
                max_benign=5,
            )
            candidate_intervention_counts.append(len(targeted))
            if targeted and exec_result.denotation:
                candidate_scci_result = compute_scci(
                    original_denotation=exec_result.denotation,
                    interventions=targeted,
                    executor_fn=executor_fn,
                    question=question,
                )
        certificate = builder.build_certificate(
            executor_result=exec_result,
            llm_answer=llm_answer,
            llm_confidence=llm_confidence,
            scci_result=candidate_scci_result,
        )
        candidate_scci_values.append(certificate.scci)
        certified_candidates.append(CertifiedCandidate(
            denotation=exec_result.denotation,
            operation=exec_result.operation,
            priority=exec_result.priority,
            certificate=certificate,
            cells_used=exec_result.cells_used,
            computation_trace=exec_result.computation_trace,
            source=f"executor_{exec_result.operation.value}",
            operation_metadata=dict(getattr(exec_result, "operation_metadata", {}) or {}),
        ))
    if candidate_scci_values:
        cert_info["scci"] = round(max(candidate_scci_values), 4)
        cert_info["bir"] = round(max(c.certificate.bir for c in certified_candidates), 4)
        cert_info["asr"] = round(max(c.certificate.asr for c in certified_candidates), 4)
        cert_info["candidate_scci_values"] = [round(v, 4) for v in candidate_scci_values]
        cert_info["candidate_intervention_counts"] = candidate_intervention_counts

    # --- 诊断: 记录所有候选的 Certificate ---
    cert_info["num_candidates"] = len(certified_candidates)
    cert_info["certified_candidates_full"] = [
        _serialize_certified_candidate_full(c, i)
        for i, c in enumerate(certified_candidates)
    ]
    cert_info["candidate_details"] = [
        {
            "denotation": c.denotation[:50],
            "operation": c.operation.value,
            "operation_family": str((c.operation_metadata or {}).get("operation_family", "")),
            "comparison_polarity": str((c.operation_metadata or {}).get("comparison_polarity", "")),
            "priority": c.priority,
            "binding_conf": round(c.certificate.binding_confidence, 4),
            "consensus": c.certificate.consensus_with_llm,
            "path_verified": c.certificate.path_verified,
            "candidate_evidence_coverage": round(c.certificate.candidate_evidence_coverage, 4),
            "candidate_effective_evidence_coverage": round(c.certificate.candidate_effective_evidence_coverage, 4),
            "evidence_fallback": c.certificate.evidence_fallback,
            "ib_mdl_score": round(c.certificate.ib_mdl_score, 4),
            "scci": round(c.certificate.scci, 4),
            "bir": round(c.certificate.bir, 4),
            "asr": round(c.certificate.asr, 4),
            "dominance_score": round(c.certificate.dominance_score(), 4),
        }
        for c in certified_candidates
    ]
    cert_info["rescue_candidate_diagnostics"] = [
        {
            "denotation": c.denotation[:50],
            "operation": c.operation.value,
            "scci": round(c.certificate.scci, 4),
            "candidate_effective_evidence_coverage": round(c.certificate.candidate_effective_evidence_coverage, 4),
            "path_verified": c.certificate.path_verified,
            "dominance_score": round(c.certificate.dominance_score(), 4),
        }
        for c in certified_candidates
        if (not c.certificate.consensus_with_llm)
        and c.certificate.path_verified
        and (not c.certificate.evidence_fallback)
        and c.certificate.candidate_effective_evidence_coverage >= 1.0
        and c.certificate.scci >= 0.7
    ]

    # --- Certificate Dominance 决策 ---
    dominance = CertificateDominance(
        scci_threshold=scci_threshold,
        fallback_to_llm=True,
        conformal_abstainer=conformal_abstainer,
        success_predictor=success_predictor,    # v7.0b: SP 覆写门控
        pipeline_result=pipeline_result,        # v7.0b: 完整特征
    )
    final_answer, source, confidence, cert = dominance.select(
        candidates=certified_candidates,
        llm_answer=llm_answer,
        llm_confidence=llm_confidence,
    )
    cert_info["dominance_source"] = source
    if cert:
        cert_info["certificate"] = cert.to_dict()

    # --- v7.0b: Success Predictor 预测 (修复特征传递) ---
    if success_predictor is not None:
        # v7.0b 修复: 使用 pipeline_result (包含完整 graph_stats, prompt_length 等)
        # v7.0a 的 bug: 这里构建的 pred_like 中 graph_stats={}, prompt_length=0
        # 导致所有样本特征几乎相同 → 预测值趋同 → 无区分力
        if pipeline_result is not None:
            # 直接使用 pipeline_result, 并注入最新的 cert_info
            pred_like = dict(pipeline_result)
            pred_like["certificate_info"] = cert_info
        else:
            # 向后兼容: 如果未传入 pipeline_result, 回退到旧方式
            pred_like = {
                "graph_stats": {},
                "certificate_info": cert_info,
                "first_token_entropy": 0.0,
                "llm_confidence": llm_confidence,
                "prompt_length": 0,
                "evidence_score": 0.0,
                "evidence_num_anchors": 0,
            }
        success_probs = predict_success(success_predictor, pred_like)
        cert_info["success_prediction"] = success_probs

    # --- 校准数据收集 (v6.0: 含 SCCI 信号) ---
    cert_info["calibration_data"] = {
        "llm_confidence": round(llm_confidence, 4),
        "composite_confidence": round(
            max(c.certificate.composite_confidence() for c in certified_candidates)
            if certified_candidates else llm_confidence,
            4,
        ),
        "has_consensus": any(c.certificate.consensus_with_llm for c in certified_candidates),
        "best_binding_conf": round(
            max(c.certificate.binding_confidence for c in certified_candidates)
            if certified_candidates else 0.0,
            4,
        ),
        "best_scci": round(
            max(c.certificate.scci for c in certified_candidates)
            if certified_candidates else 0.0,
            4,
        ),
        "any_path_verified": any(c.certificate.path_verified for c in certified_candidates),
    }

    return final_answer, source, confidence, cert_info
