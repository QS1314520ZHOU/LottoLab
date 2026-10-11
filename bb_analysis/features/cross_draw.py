# -*- coding: utf-8 -*-
"""跨期特征：只能用“上一期已完整公布的结果”预测下一期。

泄漏红线：预测第 t 期时，禁止使用第 t 期的任何字段（含当期正码）。
本模块 API 一律以“历史序列”为输入，天然只见过去。
"""

from __future__ import annotations

from typing import Any, Sequence

from .structure import structure_features


def prev_draw_features(history: list[dict[str, Any]]) -> dict[str, Any]:
    """历史最后一期（= 上一期）的特征，用于预测其下一期。"""
    if not history:
        return {"has_prev": 0}
    prev = history[-1]
    nums = [int(prev[f"p{i}"]) for i in range(1, 7)]
    sp = int(prev["sp"])
    if len(history) >= 2:
        p2 = history[-2]
        prev_prev = ([int(p2[f"p{i}"]) for i in range(1, 7)]
                     + [int(p2["sp"])])
    else:
        prev_prev = None
    st = structure_features(nums + [sp], prev_prev)
    return {
        "has_prev": 1,
        "prev_sp": sp,
        "prev_sp_tail": sp % 10,
        "prev_sp_odd": int(sp % 2 == 1),
        "prev_sp_big": int(sp >= 25),
        "prev_sp_plus1_tail": (sp + 1) % 10,
        "prev_sp_minus1_tail": (sp - 1) % 10,
        "prev_sum": st["sum"],
        "prev_span": st["span"],
        "prev_big_count": st["big_count"],
        "prev_odd_count": st["odd_count"],
        "prev_repeat": st["repeat_from_prev"],
    }


def lagged_pairs(history: list[dict[str, Any]],
                 key_a, key_b) -> list[tuple[Any, Any]]:
    """构造 (第t期A, 第t+1期B) 的跨期配对（全部端点都来自已公布期）。"""
    pairs = []
    for i in range(len(history) - 1):
        pairs.append((key_a(history[i]), key_b(history[i + 1])))
    return pairs
