# -*- coding: utf-8 -*-
"""生肖关系特征：转移概率、共现对、长龙/断龙后验统计。

全部基于“每期生肖是否出现”的0/1矩阵；只在给定历史（目标期之前）上统计。
输出用于特征与规律挖掘，是否可预测由回测判定，这里不作因果断言。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np


def transition_stats(presence: np.ndarray,
                     zodiacs: Sequence[str]) -> dict[tuple[str, str], dict]:
    """P(z' 出现 | z 出现) 及 (次数, 条件频次)。

    presence: (T, Z) bool；只统计成对相邻期 (t, t+1)，两端都已发生。
    """
    out: dict[tuple[str, str], dict] = {}
    t_size = presence.shape[0]
    for i, z in enumerate(zodiacs):
        cond = presence[:-1, i]
        n_cond = int(cond.sum())
        if n_cond == 0:
            continue
        for j, z2 in enumerate(zodiacs):
            nxt = presence[1:, j]
            hit = int((cond & nxt).sum())
            out[(z, z2)] = {
                "n_cond": n_cond,
                "n_both": hit,
                "p_next": hit / n_cond,
                "p_base": float(presence[:, j].mean()) if t_size else float("nan"),
            }
    return out


def cooccurrence(presence: np.ndarray,
                 zodiacs: Sequence[str]) -> dict[tuple[str, str], int]:
    """同期共现次数矩阵（无序对）。"""
    out: dict[tuple[str, str], int] = {}
    z_size = len(zodiacs)
    for i in range(z_size):
        for j in range(i + 1, z_size):
            out[(zodiacs[i], zodiacs[j])] = int(
                (presence[:, i] & presence[:, j]).sum())
    return out


def run_outcomes(hits: np.ndarray) -> dict[int, dict[int, int]]:
    """长龙统计：命中连续长度 L 结束后，下一期 hit->1 / miss->0 的计数。

    返回 {L: {1: 次数, 0: 次数}}。用于“连L期后下一期”频率与断龙研究；
    不预设“必断”或“必续”。
    """
    out: dict[int, dict[int, int]] = defaultdict(lambda: {0: 0, 1: 0})
    n = len(hits)
    i = 0
    while i < n:
        if hits[i]:
            j = i
            while j < n and hits[j]:
                j += 1
            L = j - i
            if j < n:              # 连开结束后还有下一期可评分
                out[L][int(bool(hits[j]))] += 1
            i = j
        else:
            i += 1
    return dict(out)
