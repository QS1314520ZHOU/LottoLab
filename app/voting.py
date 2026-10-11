# -*- coding: utf-8 -*-
"""三级多算法投票（蓝球市场 v1）。

第一级：同类算法内部等权合并（避免相似算法重复加票）。
第二级：跨类别等权（频率/结构/跨期/ML/人工/玄学——玄学无特权）。
第三级：稳定性门槛 → 票份额不足或分差过小则弃权（记录原因）。

票数 ≠ 中奖概率：输出的 final 只是“研究用投票分布”，
仅当向用户展示校准过的模型概率时才可标注为概率。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

CLASSES = ("frequency", "structure", "cross", "ml", "manual",
           "metaphysics")

# 弃权门槛（回测前冻结）：top1 份额或与 top2 分差不足 → abstain
TOP1_MIN = 1.0 / 16 * 1.8      # ≈0.1125
MARGIN_MIN = 0.06


@dataclass
class Voter:
    name: str
    cls: str                       # CLASSES 之一
    vector: np.ndarray             # (16,) 非负，内部会归一化
    note: str = ""

    def __post_init__(self) -> None:
        v = np.asarray(self.vector, dtype=float)
        s = v.sum()
        self.vector = v / s if s > 0 else np.full(16, 1 / 16)


@dataclass
class VoteOutcome:
    final: np.ndarray
    class_vectors: dict[str, np.ndarray]
    top1: int                      # 1..16
    top1_share: float
    margin: float
    abstain: bool
    abstain_reason: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


def pick_vector(pick: Optional[int]) -> np.ndarray:
    """把“单号候选”转为票向量：命中格 0.5，其余均分 0.5。"""
    v = np.full(16, 0.5 / 16)
    if pick is not None and 1 <= pick <= 16:
        v[pick - 1] += 0.5
    return v


def aggregate(voters: list[Voter]) -> VoteOutcome:
    # 第一级：类内等权
    class_vecs: dict[str, np.ndarray] = {}
    for cls in CLASSES:
        members = [v for v in voters if v.cls == cls]
        if not members:
            continue
        class_vecs[cls] = np.mean([v.vector for v in members], axis=0)
    if not class_vecs:
        raise ValueError("没有任何投票者")
    # 第二级：跨类等权
    final = np.mean(list(class_vecs.values()), axis=0)
    final = final / final.sum()
    order = np.argsort(final)[::-1]
    top1 = int(order[0]) + 1
    top1_share = float(final[order[0]])
    margin = float(final[order[0]] - final[order[1]])
    # 第三级：稳定性门槛
    abstain = False
    reason = ""
    if top1_share < TOP1_MIN:
        abstain, reason = True, f"top1份额{top1_share:.3f}<{TOP1_MIN:.3f}"
    elif margin < MARGIN_MIN:
        abstain, reason = True, f"分差{margin:.3f}<{MARGIN_MIN:.3f}"
    return VoteOutcome(final=final, class_vectors=class_vecs, top1=top1,
                       top1_share=top1_share, margin=margin,
                       abstain=abstain, abstain_reason=reason,
                       extras={"n_voters": len(voters),
                               "n_classes": len(class_vecs)})
