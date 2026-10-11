# -*- coding: utf-8 -*-
"""理论概率引擎：不放回抽样下“k个生肖全部出现于D个开奖号”的精确概率。

采用包含-排斥（inclusion–exclusion）对可枚举的组合空间精确计算：
    P(全部出现) = Σ_{S ⊆ 选中生肖} (-1)^|S| · C(N - Σ_{i∈S} size_i, D) / C(N, D)
单肖是 k=1 的特例（超几何）。禁止用单肖概率相乘近似组合概率。
提供蒙特卡洛交叉验证；无法从规则确定时抛 TheoryPending（不得冒充精确值）。
"""

from __future__ import annotations

import math
import random
from itertools import combinations
from typing import Iterable, Sequence


class TheoryPending(Exception):
    """规则不足以给出精确概率（应标记为待配置）。"""


def _comb(n: int, k: int) -> int:
    if k < 0 or n < 0 or n < k:
        return 0
    return math.comb(n, k)


def all_present_prob(sizes: Sequence[int], draw_size: int = 7,
                     total: int = 49) -> float:
    """k个互斥生肖（号码数 sizes）全部出现在 D 个不放回抽样号码中的概率。"""
    k = len(sizes)
    if k == 0:
        return 1.0
    if sum(sizes) > total:
        raise TheoryPending(f"生肖号码总数 {sum(sizes)} 超过号码空间 {total}")
    denom = _comb(total, draw_size)
    if denom == 0:
        raise TheoryPending(f"非法抽样参数 total={total} draw_size={draw_size}")
    acc = 0
    for j in range(k + 1):
        sign = (-1) ** j
        for subset in combinations(range(k), j):
            excluded = sum(sizes[i] for i in subset)
            acc += sign * _comb(total - excluded, draw_size)
    return acc / denom


def single_prob(zodiac_size: int, draw_size: int = 7, total: int = 49) -> float:
    """单肖命中概率（超几何补）：1 - C(N-s, D)/C(N, D)。"""
    if not (0 < zodiac_size <= total):
        raise TheoryPending(f"生肖号码数非法: {zodiac_size}")
    return 1.0 - _comb(total - zodiac_size, draw_size) / _comb(total, draw_size)


def monte_carlo_all_present(
    sizes: Sequence[int],
    draw_size: int = 7,
    total: int = 49,
    n_sims: int = 200_000,
    seed: int = 42,
) -> float:
    """蒙特卡洛估计（交叉验证用），固定随机种子。"""
    rng = random.Random(seed)
    # 每个生肖的号码集合（按连续块划分即可，概率只依赖大小）
    blocks = []
    cursor = 0
    for s in sizes:
        blocks.append(set(range(cursor, cursor + s)))
        cursor += s
    hit = 0
    universe = range(total)
    for _ in range(n_sims):
        drawn = set(rng.sample(universe, draw_size))
        if all(blk & drawn for blk in blocks):
            hit += 1
    return hit / n_sims
