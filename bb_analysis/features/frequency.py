# -*- coding: utf-8 -*-
"""冷热/遗漏/连开等窗口统计（统一按“每期生肖是否出现”口径）。

约定：输入 hit_series 是长度为 H 的0/1序列，只包含目标期之前的开奖；
as_of 语义由调用方通过切片保证，本模块绝不接触未来数据。
所有窗口长度来自配置（cfg.windows）。
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np


def window_hits(hits: np.ndarray, w: int) -> int:
    """最近 w 期命中次数（不足 w 期取全部）。"""
    if w <= 0:
        raise ValueError("window 必须为正")
    return int(hits[-w:].sum())


def window_freq(hits: np.ndarray, w: int) -> float:
    seg = hits[-w:]
    if seg.size == 0:
        return float("nan")
    return float(seg.mean())


def omission(hits: np.ndarray) -> int:
    """当前遗漏：距最近一次命中的期数；从未命中记为 len(hits)。

    开出当期后归零：序列末位为1 → 遗漏=0；随后每未中一期 +1。
    """
    if hits.size == 0:
        return 0
    idx = np.nonzero(hits)[0]
    if idx.size == 0:
        return int(hits.size)
    return int(hits.size - 1 - idx[-1])


def current_streak(hits: np.ndarray) -> int:
    """当前连开长度：末尾连续命中的个数（末位未中则为0）。"""
    n = 0
    for v in hits[::-1]:
        if v:
            n += 1
        else:
            break
    return n


def max_streak(hits: np.ndarray) -> int:
    """历史最长连开。"""
    best = cur = 0
    for v in hits:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def longest_miss(hits: np.ndarray) -> int:
    """历史最长连续未中。"""
    best = cur = 0
    for v in hits:
        cur = 0 if v else cur + 1
        best = max(best, cur)
    return best


def surprise_z(hits: np.ndarray, w: int, p0: float) -> float:
    """最近 w 期相对理论基线 p0 的标准化偏差（爆冷指标，事前定义）。"""
    k = window_hits(hits, w)
    n = min(w, hits.size)
    if n == 0 or p0 <= 0 or p0 >= 1:
        return float("nan")
    return (k - n * p0) / math.sqrt(n * p0 * (1 - p0))


def freq_delta(hits: np.ndarray, w_short: int, w_long: int) -> float:
    """短期频率 - 长期频率（热冷漂移）。"""
    return window_freq(hits, w_short) - window_freq(hits, w_long)


def scope_stats(hits: np.ndarray, windows: tuple[int, ...],
                p0: float) -> dict[str, Any]:
    """一个作用域（生肖/组合）在某 as_of 的全量窗口特征。"""
    per_window = {}
    for w in windows:
        k = window_hits(hits, w)
        per_window[w] = {
            "hits": k,
            "freq": k / min(w, hits.size) if hits.size else float("nan"),
            "surprise_z": surprise_z(hits, w, p0),
        }
    short = windows[0]
    long_w = windows[-1]
    return {
        "omission": omission(hits),
        "streak": current_streak(hits),
        "max_streak": max_streak(hits),
        "longest_miss": longest_miss(hits),
        "freq_delta": freq_delta(hits, short, long_w),
        "n_eff": int(hits.size),
        "windows": per_window,
    }
