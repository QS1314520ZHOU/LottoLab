# -*- coding: utf-8 -*-
"""特征层统一入口：出现矩阵构建 + 单期特征快照。

口径声明（禁止混用）：
  * 生肖/组合特征一律基于“该生肖在该期7个号码中是否出现”（0/1）。
  * 号码级原始出现次数保留在 draw_zodiacs，按需另行统计。
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .cross_draw import lagged_pairs, prev_draw_features
from .frequency import scope_stats
from .structure import structure_features
from .zodiac_relations import cooccurrence, run_outcomes, transition_stats

__all__ = [
    "presence_matrix", "combo_presence", "single_feature_snapshot",
    "prev_draw_features", "structure_features", "transition_stats",
    "cooccurrence", "run_outcomes", "lagged_pairs", "scope_stats",
]


def presence_matrix(present_sets: Sequence[set[str]],
                    zodiacs: Sequence[str]) -> np.ndarray:
    """(T, Z) bool：第 t 期是否出现生肖 z。present_sets 按期升序。"""
    idx = {z: i for i, z in enumerate(zodiacs)}
    mat = np.zeros((len(present_sets), len(zodiacs)), dtype=bool)
    for t, s in enumerate(present_sets):
        for z in s:
            if z in idx:
                mat[t, idx[z]] = True
    return mat


def combo_presence(presence: np.ndarray,
                   member_idx: Sequence[int]) -> np.ndarray:
    """组合命中序列：全部成员生肖都出现。"""
    if not member_idx:
        raise ValueError("组合不能为空")
    out = np.ones(presence.shape[0], dtype=bool)
    for i in member_idx:
        out &= presence[:, i]
    return out


def single_feature_snapshot(hits: np.ndarray, windows: tuple[int, ...],
                            p0: float,
                            history: list[dict[str, Any]] | None = None
                            ) -> dict[str, Any]:
    """某 as_of 时刻（hits 只含目标期之前）的完整特征快照。"""
    snap = scope_stats(hits, windows, p0)
    snap["prev"] = prev_draw_features(history) if history is not None else {}
    return snap
