"""
credal_probe.py — CSCR v8.6 Credal Probe 诊断层

纯诊断模块：计算层次化因子评分和 Credal 不确定性区间。
**不改变任何答案**（只读旁路），所有输出写入 result["probe_diagnostics"]。

理论基础：
  - 层次化因子评分：基于 HCEG 图拓扑、证据覆盖、锚点深度、因果路径强度、
    执行器一致性、绑定质量的多因子复合评分。
  - Credal 不确定性区间：用纯 Python 实现轻量版 Bernoulli credal set
    不确定性分解（AU/EU 代理）。
  - 风险分层：按 factor score 和 credal width 将样本分为 low/medium/high risk。

为 v9.x Credal Meta-Controller 提供实证数据：
  - 哪些因子组合下 EM 最高/最低？
  - credal width 与 EM 的相关性如何？
  - risk_tier 分桶是否有预测力？

依赖：仅标准库 (math, typing)，不依赖 torch/numpy/scipy。
"""

import math
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# 1) 层次化因子评分
# ---------------------------------------------------------------------------

# 因子权重（预设，v9.x 可用数据驱动学习）
_FACTOR_WEIGHTS = {
    "graph_density": 0.10,
    "evidence_coverage": 0.20,
    "anchor_depth_mean": 0.15,
    "causal_path_strength": 0.20,
    "executor_agreement": 0.20,
    "binding_quality": 0.15,
}


def compute_hierarchical_factor_score(
    graph: Optional[Any] = None,
    evidence: Optional[Any] = None,
    executor_result: Optional[Any] = None,
    cert_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """计算层次化因子评分。

    所有输入均可为 None（退化为全零）。
    返回包含各因子值和 composite_factor_score 的 dict。
    """
    ci = cert_info or {}
    factors: Dict[str, float] = {}

    # --- graph_density: 边密度 ---
    if graph is not None:
        n_nodes = len(getattr(graph, "nodes", {}))
        n_edges = len(getattr(graph, "edges", []))
        denom = max(n_nodes * (n_nodes - 1), 1)
        factors["graph_density"] = min(n_edges / denom, 1.0)
    else:
        factors["graph_density"] = 0.0

    # --- evidence_coverage: 证据节点占图节点的比例 ---
    if evidence is not None and graph is not None:
        ev_cells = getattr(evidence, "num_cells", 0)
        n_nodes = max(len(getattr(graph, "nodes", {})), 1)
        factors["evidence_coverage"] = min(ev_cells / n_nodes, 1.0)
    else:
        factors["evidence_coverage"] = 0.0

    # --- anchor_depth_mean: 锚点平均 header_level ---
    if evidence is not None and graph is not None:
        anchor_ids = getattr(evidence, "anchor_nodes", [])
        depths = []
        nodes_dict = getattr(graph, "nodes", {})
        for aid in anchor_ids:
            node = nodes_dict.get(aid)
            if node is not None:
                hl = getattr(node, "header_level", -1)
                if hl >= 0:
                    depths.append(hl)
        if depths:
            # 归一化到 [0, 1]，假设最大深度 5
            factors["anchor_depth_mean"] = min(sum(depths) / len(depths) / 5.0, 1.0)
        else:
            factors["anchor_depth_mean"] = 0.0
    else:
        factors["anchor_depth_mean"] = 0.0

    # --- causal_path_strength: SCCI 或 evidence retrieval_score ---
    scci = ci.get("scci", None)
    if scci is not None and isinstance(scci, (int, float)):
        factors["causal_path_strength"] = min(max(float(scci), 0.0), 1.0)
    elif evidence is not None:
        rs = getattr(evidence, "retrieval_score", 0.0)
        factors["causal_path_strength"] = min(max(float(rs), 0.0), 1.0)
    else:
        factors["causal_path_strength"] = 0.0

    # --- executor_agreement: LLM-executor 共识 ---
    has_consensus = ci.get("has_consensus", False)
    factors["executor_agreement"] = 1.0 if has_consensus else 0.0

    # --- binding_quality: 最佳绑定置信度 ---
    bbc = ci.get("best_binding_conf", 0.0)
    factors["binding_quality"] = min(max(float(bbc) if bbc else 0.0, 0.0), 1.0)

    # --- composite_factor_score: 加权平均 ---
    weighted_sum = sum(
        factors.get(k, 0.0) * w for k, w in _FACTOR_WEIGHTS.items()
    )
    total_weight = sum(_FACTOR_WEIGHTS.values())
    composite = weighted_sum / total_weight if total_weight > 0 else 0.0

    return {
        **factors,
        "composite_factor_score": round(composite, 6),
    }


# ---------------------------------------------------------------------------
# 2) Credal 不确定性探针
# ---------------------------------------------------------------------------

def _h_bern(p: float) -> float:
    """Bernoulli 二元熵 H(p) = -p*log(p) - (1-p)*log(1-p)，以 nats 计。"""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


def _renyi_entropy_alpha2(probs: Sequence[float]) -> float:
    """Rényi entropy of order 2: H_2 = -log(sum(p_i^2))."""
    if not probs:
        return 0.0
    sum_sq = sum(p * p for p in probs if p > 0)
    if sum_sq <= 0:
        return 0.0
    return -math.log(sum_sq)


def compute_credal_uncertainty_probe(
    first_token_entropy: float = 0.0,
    llm_confidence: float = 0.5,
    logprobs_list: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """计算轻量版 Credal 不确定性区间（纯 Python，无 torch 依赖）。

    使用标量 Python 计算以避免推理环境额外依赖。

    参数:
        first_token_entropy: 首 token 的 top-K 熵
        llm_confidence: LLM 输出的 greedy confidence
        logprobs_list: 可选的 logprobs 列表（第一个元素为 top-K dict）

    返回:
        dict 包含 p_L, p_U, credal_width, aleatoric_proxy, epistemic_proxy, renyi_entropy
    """
    ent = max(float(first_token_entropy), 0.0)
    conf = max(0.0, min(1.0, float(llm_confidence)))

    # 从 entropy 推算 credal 区间宽度
    # 熵越大 → 区间越宽 → 不确定性越高
    # scale 因子：log(K) ≈ 1.6 (K=5)，归一化到 [0, 0.5]
    scale = 0.5 / max(math.log(5), 0.01)  # ≈ 0.31
    half_width = min(ent * scale, 0.5)

    p_L = max(0.0, conf - half_width)
    p_U = min(1.0, conf + half_width)
    credal_width = p_U - p_L

    # Aleatoric proxy: Bernoulli entropy at point estimate
    aleatoric_proxy = _h_bern(conf)
    # Epistemic proxy: credal width 超出 aleatoric 部分
    # EU = TU - AU 的轻量代理
    epistemic_proxy = max(0.0, credal_width - aleatoric_proxy)

    # Rényi entropy from logprobs (if available)
    renyi = 0.0
    if logprobs_list and len(logprobs_list) > 0:
        first_logprobs = logprobs_list[0]
        if isinstance(first_logprobs, dict):
            # logprobs dict: token -> log_prob
            raw_probs = []
            for _token, lp in first_logprobs.items():
                try:
                    raw_probs.append(math.exp(float(lp)))
                except (ValueError, TypeError, OverflowError):
                    continue
            if raw_probs:
                # 归一化
                total = sum(raw_probs)
                if total > 0:
                    norm_probs = [p / total for p in raw_probs]
                    renyi = _renyi_entropy_alpha2(norm_probs)

    return {
        "p_L": round(p_L, 6),
        "p_U": round(p_U, 6),
        "credal_width": round(credal_width, 6),
        "aleatoric_proxy": round(aleatoric_proxy, 6),
        "epistemic_proxy": round(epistemic_proxy, 6),
        "renyi_entropy": round(renyi, 6),
    }


# ---------------------------------------------------------------------------
# 3) 组合诊断入口
# ---------------------------------------------------------------------------

def compute_probe_diagnostics(
    graph: Optional[Any] = None,
    evidence: Optional[Any] = None,
    executor_result: Optional[Any] = None,
    cert_info: Optional[Dict[str, Any]] = None,
    first_token_entropy: float = 0.0,
    llm_confidence: float = 0.5,
    logprobs_list: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """v8.6 Credal Probe 诊断入口。

    组合调用层次化因子评分和 credal 不确定性探针，
    返回完整诊断结果和风险分层。

    **不改变任何答案**。
    """
    hf = compute_hierarchical_factor_score(
        graph=graph,
        evidence=evidence,
        executor_result=executor_result,
        cert_info=cert_info,
    )
    cp = compute_credal_uncertainty_probe(
        first_token_entropy=first_token_entropy,
        llm_confidence=llm_confidence,
        logprobs_list=logprobs_list,
    )

    # 风险分层 (v8.7: 基于 E25-27 实证数据校准)
    # E25-27 诊断发现: factor_score max ≈ 0.38, 旧阈值 ≥0.6 永远不触发 low_risk
    # 新阈值: cw<0.1 → 76% EM (低风险), cw∈[0.1,0.3) → 45% EM (中风险), cw≥0.3 → 15% EM (高风险)
    factor_score = hf.get("composite_factor_score", 0.0)
    cw = cp.get("credal_width", 1.0)

    if factor_score >= 0.25 and cw < 0.1:
        tier = "low_risk"
    elif cw < 0.3:
        tier = "medium_risk"
    else:
        tier = "high_risk"

    return {
        "hierarchical_factors": hf,
        "credal_probe": cp,
        "probe_risk_tier": tier,
    }


# ---------------------------------------------------------------------------
# 4) 分桶汇总统计
# ---------------------------------------------------------------------------

def aggregate_probe_metrics(
    predictions: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """按 probe_diagnostics 分桶统计 hitab_official_em。

    供 compute_full_metrics 调用，写入 metrics.json["probe_metrics"]。
    """
    if not predictions:
        return {}

    # 辅助：分桶统计
    def _bucket_em(
        items: Sequence[Dict[str, Any]],
        key_fn,
        buckets: Sequence[tuple],
    ) -> List[Dict[str, Any]]:
        results = []
        for lo, hi in buckets:
            bucket_items = [
                p for p in items
                if lo <= key_fn(p) < hi
            ]
            if bucket_items:
                em = sum(
                    1 for p in bucket_items if p.get("hitab_official_em", False)
                ) / len(bucket_items)
                results.append({
                    "range": [lo, hi],
                    "count": len(bucket_items),
                    "hitab_em": round(em, 6),
                })
        return results

    # 过滤有 probe_diagnostics 的样本
    probed = [p for p in predictions if "probe_diagnostics" in p]
    if not probed:
        return {"probed_count": 0}

    # --- 按 risk_tier 分桶 ---
    tier_stats: Dict[str, Dict[str, Any]] = {}
    for p in probed:
        tier = p["probe_diagnostics"].get("probe_risk_tier", "unknown")
        if tier not in tier_stats:
            tier_stats[tier] = {"count": 0, "correct": 0}
        tier_stats[tier]["count"] += 1
        if p.get("hitab_official_em", False):
            tier_stats[tier]["correct"] += 1
    risk_tier_em = {
        tier: {
            "count": s["count"],
            "hitab_em": round(s["correct"] / s["count"], 6) if s["count"] else 0.0,
        }
        for tier, s in sorted(tier_stats.items())
    }

    # --- 按 credal_width 分桶 ---
    def _cw(p: Dict) -> float:
        return p.get("probe_diagnostics", {}).get("credal_probe", {}).get("credal_width", 0.0)

    credal_width_buckets = _bucket_em(
        probed, _cw,
        [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)],
    )

    # --- 按 composite_factor_score 分桶 ---
    def _fs(p: Dict) -> float:
        return p.get("probe_diagnostics", {}).get("hierarchical_factors", {}).get("composite_factor_score", 0.0)

    factor_score_buckets = _bucket_em(
        probed, _fs,
        [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)],
    )

    return {
        "probed_count": len(probed),
        "risk_tier_em": risk_tier_em,
        "credal_width_buckets": credal_width_buckets,
        "factor_score_buckets": factor_score_buckets,
    }
