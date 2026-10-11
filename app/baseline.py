# -*- coding: utf-8 -*-
"""公平随机基线：解析概率 + 固定种子蒙特卡洛交叉验证。

标准双色球：红球 C(33,6) 不放回，蓝球 1..16 独立。
单注一等奖概率 = 1 / (C(33,6) * 16)。
“单个红球入选”概率 = 6/33 —— 与“整组命中”严格区分，禁止混用。
"""

from __future__ import annotations

import random
from math import comb
from typing import Sequence

from .config import SSQConfig


def prize1_probability(cfg: SSQConfig) -> float:
    return 1.0 / (comb(cfg.red_pool, cfg.red_count)
                  * (cfg.blue_pool ** cfg.blue_count))


def single_red_inclusion(cfg: SSQConfig) -> float:
    """一个指定红球出现在开奖6红中的概率。"""
    return cfg.red_count / cfg.red_pool


def blue_uniform(cfg: SSQConfig) -> float:
    return 1.0 / cfg.blue_pool


def overlap_expect(m: int, cfg: SSQConfig) -> float:
    """随机 m 红组合与开奖 6 红的期望重叠数 = 6m/33。"""
    return cfg.red_count * m / cfg.red_pool


def red_hits_pmf(j: int, cfg: SSQConfig) -> dict[int, float]:
    """投注含 j 个红球时，开奖命中 K 个红球的精确分布（超几何）。"""
    n, k = cfg.red_pool, cfg.red_count
    pmf = {}
    for K in range(0, min(j, k) + 1):
        pmf[K] = (comb(j, K) * comb(n - j, k - K)) / comb(n, k)
    return pmf


def monte_carlo_prize1(cfg: SSQConfig, n_sims: int = 200_000,
                       seed: int | None = None) -> float:
    """MC 验证一等奖概率（固定种子，报告误差范围）。"""
    rng = random.Random(seed if seed is not None else cfg.seed)
    pool = range(1, cfg.red_pool + 1)
    hit = 0
    for _ in range(n_sims):
        reds = set(rng.sample(pool, cfg.red_count))
        blue = rng.randint(1, cfg.blue_pool)
        # 与“固定一注”比较：等价于该注红球全中+蓝中
        ticket = set(rng.sample(pool, cfg.red_count))
        tblue = rng.randint(1, cfg.blue_pool)
        if reds == ticket and blue == tblue:
            hit += 1
    return hit / n_sims


def mc_stderr(p: float, n: int) -> float:
    return (p * (1 - p) / n) ** 0.5


def uniform_blue_scores(cfg: SSQConfig) -> list[float]:
    return [blue_uniform(cfg)] * cfg.blue_pool


def uniform_red_scores(cfg: SSQConfig) -> list[float]:
    return [single_red_inclusion(cfg)] * cfg.red_pool
