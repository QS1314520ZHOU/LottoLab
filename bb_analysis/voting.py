# -*- coding: utf-8 -*-
"""T8 · 三级投票（全开一肖市场）与预测审计。

第一级  类内等权：统计类(B/C)、ML类(D)、规律类(仅 validated 规律可入选)
第二级  跨类等权
第三级  稳定门槛：top1−top2 < MARGIN → 弃权（记原因）

权重依据（预先定义）：
  - 默认所有入选者等权；
  - 规律投票权 = status=='validated' 门禁（当前注册表无 validated → 规律不投票）；
  - 模型"更高级别"仅当验证段 beat_baseline 评级（见 models.paired_brier_test）。
票数不是中奖概率；p_est 必须来自模型输出并注明口径。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .combinations import ZODIAC_ORDER

MARGIN = 0.02          # 全开概率向量的弃权分差（冻结值）
CLASSES = ("stats", "ml", "patterns")


@dataclass
class Voter:
    name: str
    cls: str
    probs: np.ndarray            # (12,)
    note: str = ""

    def __post_init__(self) -> None:
        v = np.asarray(self.probs, dtype=float)
        self.probs = np.clip(v, 0.0, 1.0)


@dataclass
class VoteOutcome:
    final: np.ndarray
    pick: Optional[int]           # 下标 or None(弃权)
    pick_name: Optional[str]
    top1_p: float
    margin: float
    abstain: bool
    abstain_reason: str
    class_picks: dict[str, str]
    votes: dict[str, Any] = field(default_factory=dict)


def aggregate(voters: list[Voter]) -> VoteOutcome:
    if not voters:
        return VoteOutcome(final=np.zeros(12), pick=None, pick_name=None,
                           top1_p=0.0, margin=0.0, abstain=True,
                           abstain_reason="no_voters", class_picks={})
    class_vecs: dict[str, np.ndarray] = {}
    class_picks: dict[str, str] = {}
    for cls in CLASSES:
        members = [v for v in voters if v.cls == cls]
        if not members:
            continue
        vec = np.mean([v.probs for v in members], axis=0)
        class_vecs[cls] = vec
        class_picks[cls] = ZODIAC_ORDER[int(np.argmax(vec))]
    final = np.mean(list(class_vecs.values()), axis=0)
    order = np.argsort(final)[::-1]
    top1, second = int(order[0]), int(order[1])
    margin = float(final[top1] - final[second])
    abstain = margin < MARGIN
    return VoteOutcome(
        final=final,
        pick=None if abstain else top1,
        pick_name=None if abstain else ZODIAC_ORDER[top1],
        top1_p=float(final[top1]), margin=margin, abstain=abstain,
        abstain_reason=("margin_below_%.2f" % MARGIN) if abstain else "",
        class_picks=class_picks,
        votes={v.name: {"cls": v.cls,
                        "pick": ZODIAC_ORDER[int(np.argmax(v.probs))],
                        "top_p": float(np.max(v.probs)),
                        "note": v.note} for v in voters},
    )


def build_voters(b_vec: Any, c_vec: Any, d_vec: Any,
                 validated_patterns: list[tuple[str, np.ndarray]]
                 ) -> list[Voter]:
    """组装投票人。规律仅在 validated 时获得投票权。"""
    voters = [
        Voter("B频率", "stats", b_vec.probs, f"w={b_vec.n}"),
        Voter("C贝叶斯", "stats", c_vec.probs, f"w={c_vec.n}"),
    ]
    if d_vec is not None:
        voters.append(Voter("D逻辑回归", "ml", d_vec.probs, "stride重训"))
    for name, vec in validated_patterns:
        voters.append(Voter(name, "patterns", vec, "validated门禁通过"))
    return voters
