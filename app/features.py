# -*- coding: utf-8 -*-
"""双色球特征工程（fv1）：只允许使用目标期之前的开奖。

第 t 期 payload 仅由 draws[:t] 计算，包含：
  - 红球/蓝球窗口频率（窗口可配置，默认 10/20/50/100）
  - 红球/蓝球当前遗漏（最近一次出现距 t-1 的期数；t-1 出现则 0）
  - 上期结构（蓝球、蓝奇偶、红和值/跨度/奇偶、上期红球重复数）
时间泄漏由 test_no_future_leakage 单测保证。
"""

from __future__ import annotations

import json
from typing import Any

from .config import SSQConfig

FEATURE_VERSION = "fv1"


def omission_at(occurrences: list[int], t: int) -> int:
    """occurrences 为已出现期下标（升序）；t 为目标期下标。

    返回截至 t-1 的遗漏：t-1 出现过 → 0；从未出现 → t。
    """
    if not occurrences:
        return t
    last = occurrences[-1]
    if last >= t:
        raise ValueError("occurrences 含未来期")
    return (t - 1) - last


def build_features(draws: list[dict[str, Any]],
                   cfg: SSQConfig) -> dict[str, dict[str, Any]]:
    """draws 升序；返回 issue -> payload。"""
    n = len(draws)
    red_occ: dict[int, list[int]] = {r: [] for r in range(1, 34)}
    blue_occ: dict[int, list[int]] = {b: [] for b in range(1, 17)}
    out: dict[str, dict[str, Any]] = {}

    for t in range(n):
        d = draws[t]
        payload: dict[str, Any] = {"t": t}
        for w in cfg.windows:
            lo = max(0, t - w)
            red_cnt = [0] * 34
            blue_cnt = [0] * 17
            for i in range(lo, t):
                row = draws[i]
                for j in range(1, 7):
                    red_cnt[row[f"r{j}"]] += 1
                blue_cnt[row["b1"]] += 1
            payload[f"red_freq_{w}"] = [red_cnt[r] / w for r in range(1, 34)]
            payload[f"blue_freq_{w}"] = [blue_cnt[b] / w for b in range(1, 17)]
        for b in range(1, 17):
            payload[f"blue_om_{b}"] = omission_at(blue_occ[b], t)
        for r in range(1, 34):
            payload[f"red_om_{r}"] = omission_at(red_occ[r], t)
        if t > 0:
            prev = draws[t - 1]
            prev_red = [prev[f"r{i}"] for i in range(1, 7)]
            payload["prev"] = {
                "blue": prev["b1"],
                "blue_odd": int(prev["b1"] % 2 == 1),
                "blue_half": int(prev["b1"] >= 9),   # 1-8 / 9-16
                "red_sum": sum(prev_red),
                "red_span": max(prev_red) - min(prev_red),
                "red_odd": sum(1 for x in prev_red if x % 2),
                "red_repeat_prev": _repeat(draws, t - 1),
            }
        out[d["issue"]] = payload
        # 本期结果仅供 t+1 及以后
        for j in range(1, 7):
            red_occ[d[f"r{j}"]].append(t)
        blue_occ[d["b1"]].append(t)
    return out


def _repeat(draws: list[dict[str, Any]], idx: int) -> int:
    """idx 期与 idx-1 期红球重复数（在 idx+1 期可用）。"""
    if idx <= 0:
        return 0
    a = {draws[idx][f"r{i}"] for i in range(1, 7)}
    b = {draws[idx - 1][f"r{i}"] for i in range(1, 7)}
    return len(a & b)


def save_features(cfg: SSQConfig, feats: dict[str, dict[str, Any]]) -> int:
    from . import db
    db.migrate(cfg.db_path)
    rows = [(iss, FEATURE_VERSION, json.dumps(p, ensure_ascii=False),
             db.utcnow())
            for iss, p in feats.items()]
    with db.connect(cfg.db_path) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO features(issue,feature_version,payload,"
            "computed_at) VALUES(?,?,?,?)", rows)
    return len(rows)
