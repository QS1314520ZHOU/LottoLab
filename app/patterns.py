# -*- coding: utf-8 -*-
"""规律库：统一评分卡 + 人工规则 + 玄学实验 + 自动发现（含多重检验）。

所有规则输出同一套评分卡字段（样本量/基线/观测/区间/四分块稳定性/
p 值/BH校正/状态），禁止用“研究分数”冒充命中概率。
状态机：exploratory（探索）→ validated（留出段确认）/ rejected。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

import numpy as np
from scipy import stats

from .config import SSQConfig

# 规则签名：给定 draws[:t]（升序），返回 1..16 的蓝球预测或 None（弃权）
BlueRule = Callable[[list[dict[str, Any]]], Optional[int]]


# ---------- 统计工具 ----------

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    denom = 1 + z * z / n
    center = (ph + z * z / (2 * n)) / denom
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def binom_p(k: int, n: int, p0: float) -> float:
    """单尾：观测 ≥ k 的概率（检验是否优于基线）。"""
    if n == 0:
        return 1.0
    return float(stats.binom.sf(k - 1, n, p0))


def bh_adjust(pvals: list[float]) -> list[float]:
    """Benjamini–Hochberg FDR 校正。"""
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(pvals)
    out = np.empty(m, dtype=float)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        val = pvals[idx] * m / (rank + 1)
        prev = min(prev, val)
        out[idx] = prev
    return out.tolist()


def block_stability(hits: list[int], n_blocks: int = 4) -> float:
    """四分块命中率的 max-min（跨期稳定性；越大越可疑）。"""
    arr = np.asarray(hits, dtype=float)
    if arr.size < n_blocks:
        return float("nan")
    blocks = np.array_split(arr, n_blocks)
    rates = [b.mean() for b in blocks if b.size]
    return float(max(rates) - min(rates))


# ---------- 评分卡 ----------

@dataclass
class RuleScore:
    name: str
    category: str
    version: str
    formula: str
    source: str                     # human|metaphysics|auto
    n_trigger: int
    baseline_p: float
    observed_p: float
    diff: float
    ci_lo: float
    ci_hi: float
    stability: float
    p_value: float
    p_adjusted: float = float("nan")
    holdout_p: Optional[float] = None
    holdout_n: Optional[int] = None
    status: str = "exploratory"     # exploratory|validated|rejected
    research_score: float = field(default=0.0)  # 仅排序用，不是概率
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------- 规则集 ----------

def _hot_blue(draws: list[dict[str, Any]], w: int = 10) -> int:
    cnt = [0] * 17
    for d in draws[-w:]:
        cnt[d["b1"]] += 1
    best = max(cnt[1:])
    return next(b for b in range(1, 17) if cnt[b] == best)


def _cold_blue(draws: list[dict[str, Any]], w: int = 200) -> int:
    occ = {b: -1 for b in range(1, 17)}
    for i, d in enumerate(draws[-w:]):
        occ[d["b1"]] = i
    oldest = min(occ.values())
    return next(b for b in range(1, 17) if occ[b] == oldest)


def _prev_same(draws: list[dict[str, Any]]) -> int | None:
    return draws[-1]["b1"] if draws else None


def _parity_flip_minom(draws: list[dict[str, Any]]) -> int | None:
    """上期蓝奇偶翻转，取该奇偶中遗漏最大的号（结构类规则）。"""
    if not draws:
        return None
    prev = draws[-1]["b1"]
    target_parity = 1 - (prev % 2)
    occ = {b: -1 for b in range(1, 17) if b % 2 == target_parity}
    for i, d in enumerate(draws):
        if d["b1"] in occ:
            occ[d["b1"]] = i
    oldest = min(occ.values())
    return next(b for b, v in occ.items() if v == oldest)


def _markov_blue(draws: list[dict[str, Any]], min_support: int = 20) -> int | None:
    """上期蓝 → 下期蓝 最大频转移（跨期类）。"""
    if len(draws) < 30:
        return None
    prev = draws[-1]["b1"]
    cnt: dict[int, int] = {}
    for a, b in zip(draws, draws[1:]):
        if a["b1"] == prev:
            cnt[b["b1"]] = cnt.get(b["b1"], 0) + 1
    if sum(cnt.values()) < min_support:
        return None
    return max(cnt, key=lambda k: (cnt[k], -k))


def _day_mod(draws: list[dict[str, Any]]) -> int | None:
    """玄学实验A：开奖日的日号 mod 16 + 1（公式预先固定）。"""
    if not draws:
        return None
    d = draws[-1].get("draw_date")
    if not d:
        return None
    try:
        day = int(d[8:10])
    except (TypeError, ValueError):
        return None
    return day % 16 + 1


def _issue_tail(draws: list[dict[str, Any]]) -> int | None:
    """玄学实验B：上期期号尾两位 mod 16 + 1（公式预先固定）。"""
    if not draws:
        return None
    tail = draws[-1]["issue"][-2:]
    if not tail.isdigit():
        return None
    return int(tail) % 16 + 1


@dataclass(frozen=True)
class Rule:
    name: str
    category: str       # frequency|cross|structure|metaphysics
    version: str
    formula: str
    source: str
    fn: BlueRule


def default_rules(cfg: SSQConfig) -> list[Rule]:
    return [
        Rule("上期蓝同号", "cross", "r1", "pick = prev_blue",
             "human", _prev_same),
        Rule("热10蓝", "frequency", "r1", "argmax count(blue, last10)",
             "human", lambda d: _hot_blue(d, 10)),
        Rule("冷200蓝", "frequency", "r1",
             "argmax omission(blue, last200)", "human",
             lambda d: _cold_blue(d, 200)),
        Rule("奇偶翻转+最大遗漏", "structure", "r1",
             "parity=1-prev_parity; pick max-omission", "human",
             _parity_flip_minom),
        Rule("Markov蓝转移", "cross", "r1",
             "argmax next|prev_blue (support>=20)", "auto", _markov_blue),
        Rule("日号mod16", "metaphysics", "r1",
             "day_of_month % 16 + 1", "metaphysics", _day_mod),
        Rule("期号尾mod16", "metaphysics", "r1",
             "issue_tail % 16 + 1", "metaphysics", _issue_tail),
    ]


# ---------- 评估 ----------

def evaluate_rule(rule: Rule, draws: list[dict[str, Any]],
                  start: int, end: int, cfg: SSQConfig,
                  holdout_start: Optional[int] = None) -> RuleScore:
    """在 [start, end) 逐期触发评估；[holdout_start, end) 单独复评。"""
    base = 1.0 / cfg.blue_pool
    hits: list[int] = []
    picks: list[int] = []
    for t in range(start, min(end, len(draws))):
        pick = rule.fn(draws[:t])
        if pick is None:
            continue
        ok = int(pick == draws[t]["b1"])
        hits.append(ok)
        picks.append(pick)
    n = len(hits)
    k = int(sum(hits))
    lo, hi = wilson(k, n)
    p = binom_p(k, n, base)
    hold_p = hold_n = None
    status = "exploratory"
    if holdout_start is not None and holdout_start < end:
        h_hits = hits[max(0, holdout_start - start):]
        hold_n = len(h_hits)
        hold_k = int(sum(h_hits))
        hold_p = binom_p(hold_k, hold_n, base)
        if hold_n >= 50 and hold_p < 0.05 and (k / n if n else 0) > base:
            status = "validated"
    obs = (k / n) if n else float("nan")
    return RuleScore(
        name=rule.name, category=rule.category, version=rule.version,
        formula=rule.formula, source=rule.source,
        n_trigger=n, baseline_p=base, observed_p=obs,
        diff=(obs - base) if n else float("nan"),
        ci_lo=lo, ci_hi=hi,
        stability=block_stability(hits) if n >= 16 else float("nan"),
        p_value=p, holdout_p=hold_p, holdout_n=hold_n, status=status,
        research_score=(obs - base) * np.sqrt(n) if n else 0.0,
        detail={"first_picks": picks[:10]},
    )


def score_all(rules: list[Rule], draws: list[dict[str, Any]],
              start: int, end: int, cfg: SSQConfig,
              holdout_start: Optional[int] = None) -> list[RuleScore]:
    scores = [evaluate_rule(r, draws, start, end, cfg, holdout_start)
              for r in rules]
    adj = bh_adjust([s.p_value for s in scores])
    for s, a in zip(scores, adj):
        s.p_adjusted = a
        if s.status == "validated" and a >= 0.05:
            s.status = "exploratory"   # 多重校正后未过 → 不得称 validated
    return sorted(scores, key=lambda s: s.research_score, reverse=True)


# ---------- 自动发现（探索性，必须带多重检验） ----------

def discover_markov_cells(draws: list[dict[str, Any]], cfg: SSQConfig,
                          min_support: int = 30) -> dict[str, Any]:
    """蓝球 16×16 转移：全局独立性卡方 + 高提升单元格（BH 校正）。"""
    n = len(draws) - 1
    table = np.zeros((16, 16), dtype=int)
    for a, b in zip(draws, draws[1:]):
        table[a["b1"] - 1, b["b1"] - 1] += 1
    chi2, p, dof, _ = stats.chi2_contingency(table)
    cells = []
    row_sum = table.sum(axis=1, keepdims=True).clip(min=1)
    total = table.sum()
    col_sum = table.sum(axis=0)
    for i in range(16):
        for j in range(16):
            obs = table[i, j]
            if obs < min_support // 10:
                continue
            exp = row_sum[i, 0] * col_sum[j] / total
            lift = (obs / exp) if exp else 0.0
            cells.append({"from": i + 1, "to": j + 1, "n": int(obs),
                          "lift": lift,
                          "p": float(stats.binom.sf(obs - 1, int(row_sum[i, 0]),
                                                    col_sum[j] / total))})
    adj = bh_adjust([c["p"] for c in cells])
    for c, a in zip(cells, adj):
        c["p_adjusted"] = a
    cells.sort(key=lambda c: c["lift"], reverse=True)
    return {"n_transitions": n, "chi2": float(chi2), "dof": int(dof),
            "p_independence": float(p), "top_cells": cells[:20],
            "n_cells_tested": len(cells)}


def discover_red_pairs(draws: list[dict[str, Any]], cfg: SSQConfig,
                       min_support: int = 30, top: int = 15) -> list[dict]:
    """红球对共现：支持度/提升度 vs 基线 C(6,2)/C(33,2) 比值 + BH。"""
    from itertools import combinations
    base_pair = (cfg.red_count * (cfg.red_count - 1) / 2) / \
                (cfg.red_pool * (cfg.red_pool - 1) / 2)
    cnt: dict[tuple[int, int], int] = {}
    for d in draws:
        reds = sorted(d[f"r{i}"] for i in range(1, 7))
        for pr in combinations(reds, 2):
            cnt[pr] = cnt.get(pr, 0) + 1
    rows = []
    n = len(draws)
    for pr, c in cnt.items():
        if c < min_support:
            continue
        p = c / n
        rows.append({"pair": pr, "n": c, "freq": p, "base": base_pair,
                     "lift": p / base_pair,
                     "p": float(stats.binom.sf(c - 1, n, base_pair))})
    adj = bh_adjust([r["p"] for r in rows])
    for r, a in zip(rows, adj):
        r["p_adjusted"] = a
    rows.sort(key=lambda r: r["lift"], reverse=True)
    return rows[:top], len(rows)


def record_rules_to_db(cfg: SSQConfig, rules: list[Rule],
                       scores: list[RuleScore]) -> None:
    """pattern_definitions + experiment_results 落库（含尝试总数）。"""
    from . import db
    db.migrate(cfg.db_path)
    with db.connect(cfg.db_path) as conn:
        for r, s in zip(rules, scores):
            pid = db.record_pattern(conn, r.name, r.category, r.version,
                                    {"formula": r.formula}, r.source)
            db.record_experiment(
                conn, pattern_id=pid, test_name="blue_rule_binom",
                n_tests=len(scores), p_value=s.p_value,
                p_adjusted=s.p_adjusted,
                effect={"diff": s.diff, "observed": s.observed_p,
                        "n": s.n_trigger},
                conclusion=s.status,
                detail={"holdout_p": s.holdout_p, "stability": s.stability})
