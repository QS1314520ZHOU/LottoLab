# -*- coding: utf-8 -*-
"""单期数字结构特征：和值、跨度、邻号、重号、尾数、奇偶比、大小比。

全部由6正码+1特码直接计算；只对已公布期计算，不涉及未来信息。
大小分界默认 24（1–24小 / 25–49大），可由调用方覆盖。
"""

from __future__ import annotations

from typing import Sequence

DEFAULT_BIG_FROM = 25  # 大号起点（含）


def tails(numbers: Sequence[int]) -> list[int]:
    return [n % 10 for n in numbers]


def structure_features(numbers: Sequence[int],
                       prev_numbers: Sequence[int] | None = None,
                       big_from: int = DEFAULT_BIG_FROM) -> dict:
    """一期开奖的结构特征；prev_numbers 用于“重号”计算。"""
    if len(numbers) < 2:
        raise ValueError("至少需要2个号码")
    odd = sum(1 for n in numbers if n % 2 == 1)
    big = sum(1 for n in numbers if n >= big_from)
    adjacent = sum(
        1 for i, a in enumerate(sorted(numbers))
        for b in sorted(numbers)[i + 1:] if b == a + 1
    )
    repeats = (
        len(set(numbers) & set(prev_numbers)) if prev_numbers else 0
    )
    return {
        "sum": int(sum(numbers)),
        "span": int(max(numbers) - min(numbers)),
        "adjacent_pairs": adjacent,
        "repeat_from_prev": repeats,
        "odd_count": odd,
        "even_count": len(numbers) - odd,
        "big_count": big,
        "small_count": len(numbers) - big,
        "tails": tails(numbers),
        "tail_set": sorted(set(tails(numbers))),
    }
