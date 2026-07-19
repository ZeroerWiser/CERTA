"""
causal_predictor.py — CSCR v7.0a: Success Predictor + Causal Feature Extraction

理论框架 (蓝图 v2 §5.2 命题 3):
  Certificate Dominance 的候选选择需要比 binding_confidence 更深层的信号。
  Success Predictor 从图统计量 + SCCI 指标 + LLM 信号中学习每个操作动作
  (lookup_aggregate, lookup_cell, compare, arithmetic) 的成功概率。

设计原则:
  - 不是启发式 (weighted score / margin threshold)
  - 而是从数据中学习的二分类器, 预测 "该操作在当前样本上能否得到正确答案"
  - 多头设计: 同一样本可能有多个动作都能成功
  - 训练数据来自 v6.1 的 predictions.jsonl

架构:
  Success Predictor: 4-head MLP (input → hidden=64 → hidden=32 → 4 独立 sigmoid)
  输入特征 (20 维):
    - 图统计量 (8d): nodes, edges, depth, width, cell_count, header_count, agg_count, edge_density
    - LLM 信号 (3d): logit_entropy, llm_confidence, prompt_length_normalized
    - SCCI 指标 (3d): scci, bir, asr
    - Executor 信号 (4d): binding_confidence, path_verified, num_candidates, best_dominance_score
    - 证据信号 (2d): evidence_score, num_anchors_normalized
"""

import argparse
import json
import math
import os
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# 特征提取
# ---------------------------------------------------------------------------

FEATURE_DIM = 20
ACTION_NAMES = ["lookup_aggregate", "lookup_cell", "compare", "arithmetic"]
NUM_ACTIONS = len(ACTION_NAMES)


def extract_features(pred: Dict[str, Any]) -> List[float]:
    """从单个 prediction 记录中提取 20 维特征向量

    特征组:
    [0-7]  图统计量: nodes, edges, depth, width, cell_count, header_count, agg_count, edge_density
    [8-10] LLM 信号: logit_entropy, llm_confidence, prompt_length_normalized
    [11-13] SCCI 指标: scci, bir, asr
    [14-17] Executor 信号: binding_conf, path_verified, num_candidates, best_dominance
    [18-19] 证据信号: evidence_score, num_anchors_normalized
    """
    gs = pred.get("graph_stats", {})
    ci = pred.get("certificate_info", {})
    cal_data = ci.get("calibration_data", {})

    # 图统计量 (8d)
    total_nodes = gs.get("total_nodes", 0)
    total_edges = gs.get("total_edges", 0)
    nt = gs.get("node_types", {})
    cell_count = nt.get("cell", 0)
    header_count = nt.get("header", 0)
    agg_count = nt.get("aggregator", 0)
    # 估算深度和宽度
    depth = math.log1p(total_nodes) if total_nodes > 0 else 0
    width = cell_count / max(header_count, 1)
    edge_density = total_edges / max(total_nodes * (total_nodes - 1), 1)

    # LLM 信号 (3d)
    logit_entropy = pred.get("first_token_entropy", 0.0)
    llm_confidence = pred.get("llm_confidence", 0.5)
    prompt_length = pred.get("prompt_length", 1000) / 5000.0  # 归一化

    # SCCI 指标 (3d)
    scci = ci.get("scci", 0.0)
    bir = ci.get("bir", 0.0)
    asr = ci.get("asr", 0.0)

    # Executor 信号 (4d)
    binding_conf = cal_data.get("best_binding_conf", 0.0)
    path_verified = 1.0 if cal_data.get("any_path_verified", False) else 0.0
    num_candidates = ci.get("num_candidates", 0) / 5.0  # 归一化
    # 最佳 dominance score
    best_dom = 0.0
    for cd in ci.get("candidate_details", []):
        dom = cd.get("dominance_score", 0)
        if dom > best_dom:
            best_dom = dom
    best_dom = best_dom / 150.0  # 归一化 (max ~150)

    # 证据信号 (2d)
    evidence_score = pred.get("evidence_score", 0.0)
    num_anchors = pred.get("evidence_num_anchors", 0) / 10.0  # 归一化

    return [
        # 图统计量 (8d)
        total_nodes / 200.0, total_edges / 1000.0, depth / 6.0, width / 10.0,
        cell_count / 100.0, header_count / 30.0, agg_count / 5.0, edge_density,
        # LLM 信号 (3d)
        logit_entropy, llm_confidence, prompt_length,
        # SCCI (3d)
        scci, bir, asr,
        # Executor (4d)
        binding_conf, path_verified, num_candidates, best_dom,
        # 证据 (2d)
        evidence_score, num_anchors,
    ]


def extract_action_labels(pred: Dict[str, Any]) -> List[float]:
    """提取 4 个动作的成功标签 (多标签)

    对每个操作类型, 检查该操作的候选是否与 gold_answer 匹配。
    注意: 这不只是看最终答案是否正确, 而是看该操作路径是否能产出正确答案。
    """
    gold = pred.get("gold_answer", "")
    is_correct = pred.get("hitab_official_em", False)

    labels = [0.0] * NUM_ACTIONS

    # 从 exec_candidates_summary 中检查每个操作的候选
    candidates = pred.get("exec_candidates_summary", [])
    if not candidates:
        # 如果没有候选信息, 用 executor_operation + is_correct 推断
        if is_correct:
            op = pred.get("executor_operation", "")
            source = pred.get("answer_source", "")
            if "consensus" in source or op:
                for i, name in enumerate(ACTION_NAMES):
                    if name == op:
                        labels[i] = 1.0
        return labels

    # 逐候选检查
    from eval_utils import hitab_official_em

    for cand in candidates:
        den = cand.get("denotation", "")
        op = cand.get("operation", "")
        if den and gold and hitab_official_em(den, gold):
            for i, name in enumerate(ACTION_NAMES):
                if name == op:
                    labels[i] = 1.0

    return labels


# ---------------------------------------------------------------------------
# Success Predictor MLP
# ---------------------------------------------------------------------------

class SuccessPredictor(nn.Module):
    """多头 Success Predictor (v7.0a)

    4 个独立的 sigmoid 输出头, 预测每个操作动作的成功概率。
    同一样本可能有多个动作都能成功 → 多标签分类。
    """

    def __init__(self, input_dim: int = FEATURE_DIM, hidden_dim: int = 64,
                 num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        # 4 个独立输出头
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim // 2, 1) for _ in range(num_actions)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回 (batch, num_actions) 的 sigmoid 概率"""
        shared_out = self.shared(x)
        logits = torch.cat([head(shared_out) for head in self.heads], dim=-1)
        return torch.sigmoid(logits)

    def predict(self, features: List[float]) -> Dict[str, float]:
        """单样本推理, 返回 {action_name: success_probability}"""
        self.eval()
        with torch.no_grad():
            x = torch.tensor([features], dtype=torch.float32)
            probs = self(x)[0]
        return {name: round(probs[i].item(), 4) for i, name in enumerate(ACTION_NAMES)}


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

def train_from_predictions(predictions_path: str, save_path: str,
                           epochs: int = 5, lr: float = 1e-3,
                           batch_size: int = 64) -> Dict[str, Any]:
    """从 predictions.jsonl 训练 Success Predictor

    Returns: 训练统计信息
    """
    # 加载数据
    features_list = []
    labels_list = []

    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            pred = json.loads(line.strip())
            features = extract_features(pred)
            labels = extract_action_labels(pred)
            features_list.append(features)
            labels_list.append(labels)

    n = len(features_list)
    print(f"Loaded {n} samples from {predictions_path}")

    # 统计标签分布
    label_sums = [sum(l[i] for l in labels_list) for i in range(NUM_ACTIONS)]
    for i, name in enumerate(ACTION_NAMES):
        print(f"  {name}: {int(label_sums[i])}/{n} positive ({label_sums[i]/n*100:.1f}%)")

    # 转换为 tensor
    X = torch.tensor(features_list, dtype=torch.float32)
    Y = torch.tensor(labels_list, dtype=torch.float32)

    # 80/20 分割
    split_idx = int(n * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    Y_train, Y_val = Y[:split_idx], Y[split_idx:]

    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 模型
    model = SuccessPredictor(input_dim=FEATURE_DIM)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练
    stats = {"epochs": [], "train_loss": [], "val_loss": [], "val_auc": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_Y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_X)

        train_loss = total_loss / len(X_train)

        # 验证
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, Y_val).item()

            # 计算各动作 AUC
            action_aucs = []
            for i, name in enumerate(ACTION_NAMES):
                y_true = Y_val[:, i].numpy()
                y_pred = val_pred[:, i].numpy()
                if y_true.sum() > 0 and y_true.sum() < len(y_true):
                    # 简单 AUC: 排序后计算
                    pairs = sorted(zip(y_pred, y_true), reverse=True)
                    tp = 0
                    fp = 0
                    auc_sum = 0
                    pos = int(y_true.sum())
                    neg = len(y_true) - pos
                    for score, label in pairs:
                        if label > 0.5:
                            tp += 1
                        else:
                            fp += 1
                            auc_sum += tp
                    auc = auc_sum / max(pos * neg, 1)
                    action_aucs.append((name, round(auc, 4)))
                else:
                    action_aucs.append((name, 0.0))

        stats["epochs"].append(epoch + 1)
        stats["train_loss"].append(round(train_loss, 6))
        stats["val_loss"].append(round(val_loss, 6))
        stats["val_auc"].append(dict(action_aucs))

        print(f"  Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        for name, auc in action_aucs:
            print(f"    {name} AUC={auc:.4f}")

    # 保存
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_dim": FEATURE_DIM,
        "num_actions": NUM_ACTIONS,
        "action_names": ACTION_NAMES,
        "stats": stats,
    }, save_path)
    print(f"Model saved to {save_path}")

    return stats


# ---------------------------------------------------------------------------
# 推理接口
# ---------------------------------------------------------------------------

def load_predictor(model_path: str) -> SuccessPredictor:
    """加载训练好的 Success Predictor"""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = SuccessPredictor(
        input_dim=checkpoint.get("feature_dim", FEATURE_DIM),
        num_actions=checkpoint.get("num_actions", NUM_ACTIONS),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_success(model: SuccessPredictor, pred: Dict[str, Any]) -> Dict[str, float]:
    """对单个样本预测各操作的成功概率"""
    features = extract_features(pred)
    return model.predict(features)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSCR v7.0a: Success Predictor")
    parser.add_argument("--train-from", required=True,
                        help="predictions.jsonl 路径 (训练数据)")
    parser.add_argument("--save-model", default="outputs/cscr/success_predictor.pt",
                        help="模型保存路径")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)

    args = parser.parse_args()

    stats = train_from_predictions(
        predictions_path=args.train_from,
        save_path=args.save_model,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )

    print("\nTraining complete!")
    print(f"Final stats: {json.dumps(stats, indent=2)}")
