# -*- coding: utf-8 -*-
"""组合枚举与赔率期望：12生肖的单肖~五连肖全量组合。

全部 C(12,k) 组合逐一枚举（不只统计历史热门组合）；每条候选记录：
成员、理论概率（theory 精确包含-排斥）、赔率、期望净收益、盈亏平衡率、
历史命中数与有效样本量（由调用方回填）、不确定性（二项/正态近似区间）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable, Optional, Sequence

from .config import AppConfig
from .rules import breakeven_rate, expected_net
from .theory import all_present_prob

ZODIAC_ORDER = ("鼠", "牛", "虎", "兔", "龙", "蛇",
                "马", "羊", "猴", "鸡", "狗", "猪")


@dataclass(frozen=True)
class ComboSpec:
    """一个候选组合（单肖或连肖）。"""

    k: int
    members: tuple[str, ...]
    zodiac_sizes: tuple[int, ...]
    p_theory: float
    odds: float
    status: str = "ok"          # ok / current_year_unconfigured / pending
    # 由回测/统计回填：
    hist_hits: int = 0
    hist_n: int = 0
    p_est: Optional[float] = None
    p_lo: Optional[float] = None
    p_hi: Optional[float] = None

    @property
    def combo_id(self) -> str:
        return "+".join(self.members)

    @property
    def ev_per_unit(self) -> float:
        """每 1 元投注的期望净收益。"""
        return expected_net(1.0, self.p_theory, self.odds)

    @property
    def breakeven(self) -> float:
        return breakeven_rate(self.odds)

    def wilson_interval(self, z: float = 1.96) -> tuple[float, float]:
        """历史命中率的 Wilson 置信区间（无数据返回 (nan,nan)。"""
        n, ph = self.hist_n, (self.hist_hits / self.hist_n if self.hist_n else 0.0)
        if n == 0:
            return (float("nan"), float("nan"))
        denom = 1 + z * z / n
        center = (ph + z * z / (2 * n)) / denom
        half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / denom
        return (max(0.0, center - half), min(1.0, center + half))


def zodiac_sizes(cfg: AppConfig) -> dict[str, int]:
    """按映射配置统计每个生肖的号码数（不硬编码）。"""
    table = cfg.mapping.versions[0].table
    sizes: dict[str, int] = {}
    for z in table.values():
        sizes[z] = sizes.get(z, 0) + 1
    return sizes


def enumerate_combos(
    cfg: AppConfig,
    ks: Iterable[int] = (1, 2, 3, 4, 5),
    max_per_k: int = 0,
) -> list[ComboSpec]:
    """枚举 k=1..5 的全部组合；max_per_k>0 时截断（默认全量）。"""
    sizes = zodiac_sizes(cfg)
    order = [z for z in ZODIAC_ORDER if z in sizes]
    out: list[ComboSpec] = []
    for k in ks:
        if k == 1:
            for z in order:
                odds, status = cfg.odds.single_odds(z)
                out.append(ComboSpec(
                    k=1, members=(z,), zodiac_sizes=(sizes[z],),
                    p_theory=all_present_prob([sizes[z]],
                                              cfg.rules.draw_size,
                                              sum(sizes.values())),
                    odds=odds, status=status,
                ))
        else:
            odds = cfg.odds.combo_odds_for(k)
            total = sum(sizes.values())
            for members in combinations(order, k):
                ss = tuple(sizes[m] for m in members)
                out.append(ComboSpec(
                    k=k, members=members, zodiac_sizes=ss,
                    p_theory=all_present_prob(ss, cfg.rules.draw_size, total),
                    odds=odds, status="ok",
                ))
        if max_per_k and k > 1:
            # 截断仅用于控制回测规模（报告需注明）；默认 0 = 全量
            group = [c for c in out if c.k == k]
            keep = group[:max_per_k]
            drop = set(id(c) for c in group[max_per_k:])
            out = [c for c in out if id(c) not in drop]
    return out


def combo_counts(ks: Iterable[int] = (1, 2, 3, 4, 5)) -> dict[int, int]:
    """各 k 的理论组合数 C(12,k)。"""
    return {k: math.comb(12, k) for k in ks}
