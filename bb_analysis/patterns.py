# -*- coding: utf-8 -*-
"""T7 · BB 规律库：统一注册表 + 评分卡 + 多重检验 + 描述/探索/验证三分。

覆盖：热冷、遗漏、连断（长龙/断龙）、生肖转移、平码↔特码关系，
以及既有“除12余数/位移”类规律。

统一评分卡字段（每条规律必填）：
  pid / 定义 / 输入 / 方法 / 目标事件 / 基线 / 样本量 n / 效果量(diff) /
  置信区间 / p 值 / BH q 值 / 状态 / 适用范围 / 理论EV

状态机：
  descriptive   描述性统计（不作预测断言）
  exploratory   探索性（未过 holdout 与 BH 双门槛）
  validated     独立验证（dev 显著 + BH q<0.05 + holdout 同向复现）
  rejected      检验过但方向为负/无法复现（保留失败记录）

口径统一：所有预测类规律一律以【全开一肖】（生肖出现于7个开奖号）评分，
基线 = theory.single_prob(该肖号码数)（47.17%/55.39%），禁止用1/12冒充。
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from scipy import stats

from .combinations import ZODIAC_ORDER
from .config import AppConfig, load_config
from .theory import single_prob

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WARMUP = 50
HOLDOUT_FRAC = 0.2

# 上期特肖对冲（对冲生肖关系，来自项目既有口径）
CHONG = {"鼠": "马", "马": "鼠", "牛": "羊", "羊": "牛", "虎": "猴",
         "猴": "虎", "兔": "鸡", "鸡": "兔", "龙": "狗", "狗": "龙",
         "蛇": "猪", "猪": "蛇"}

ZIDX = {z: i for i, z in enumerate(ZODIAC_ORDER)}


# ---------- 数据轴 ----------

@dataclass
class BBPanel:
    """分析面板：全部轴只含已开奖历史。"""

    issues: list[str]
    dates: list[Optional[date]]
    presence: np.ndarray          # (T,12) 7号全开出现
    reg_presence: np.ndarray      # (T,12) 仅6正码出现（平特关系口径）
    special_idx: np.ndarray       # (T,) 特码生肖下标
    p1_idx: np.ndarray            # (T,) 第一平码生肖下标
    sp_num: np.ndarray            # (T,) 特码号码
    p1_num: np.ndarray            # (T,) 第一平码号码
    zsize: np.ndarray             # (12,)


def load_panel(cfg: AppConfig | None = None) -> BBPanel:
    from . import database as db
    cfg = cfg or load_config()
    db.migrate(cfg.db_path)
    with db.connect(cfg.db_path) as conn:
        rows = db.load_draws(conn)
    rows.sort(key=lambda r: r["issue"])
    issues = [r["issue"] for r in rows]
    dates: list[Optional[date]] = []
    for r in rows:
        iss = r["issue"]
        dates.append(date(int(iss[:4]), int(iss[4:6]), int(iss[6:8])))
    T, Z = len(issues), len(ZODIAC_ORDER)
    presence = np.zeros((T, Z), dtype=bool)
    reg_presence = np.zeros((T, Z), dtype=bool)
    special = np.zeros(T, dtype=int)
    p1 = np.zeros(T, dtype=int)
    sp_num = np.zeros(T, dtype=int)
    p1_num = np.zeros(T, dtype=int)
    for i, r in enumerate(rows):
        for k in range(1, 7):
            presence[i, ZIDX[_z(r[f"p{k}"])]] = True
        reg_presence[i] = presence[i]
        presence[i, ZIDX[_z(r["sp"])]] = True
        special[i] = ZIDX[_z(r["sp"])]
        p1[i] = ZIDX[_z(r["p1"])]
        sp_num[i] = r["sp"]
        p1_num[i] = r["p1"]
    zsize = np.array([sum(1 for n in range(1, 50) if _z(n) == z)
                      for z in ZODIAC_ORDER], dtype=int)
    return BBPanel(issues, dates, presence, reg_presence, special, p1,
                   sp_num, p1_num, zsize)


def _z(n: int) -> str:
    from .combinations import zodiac_sizes  # 局部避免环
    mod = {1: "马", 2: "蛇", 3: "龙", 4: "兔", 5: "虎", 6: "牛",
           7: "鼠", 8: "猪", 9: "狗", 10: "鸡", 11: "猴", 0: "羊"}
    return mod[n % 12]


def p_theory_of(idx: int, panel: BBPanel) -> float:
    return single_prob(int(panel.zsize[idx]))


def odds_on(cfg: AppConfig, zodiac_name: str, d: Optional[date]) -> float:
    return cfg.odds.single_odds(zodiac_name, d)[0]


# ---------- 评分卡 ----------

@dataclass(frozen=True)
class PatternDef:
    pid: str
    name: str
    category: str            # 热冷|遗漏|连断|转移|平特|位移|自动发现|关系
    definition: str
    inputs: str
    method: str
    target: str              # '全开一肖' | '描述' | '同期关系'
    baseline_desc: str
    scope: str
    source: str              # human|auto|prior_work


@dataclass
class PatternResult:
    pid: str
    name: str
    category: str
    definition: str
    inputs: str
    method: str
    target: str
    baseline_desc: str
    scope: str
    source: str
    segment: str             # dev|holdout|full|descriptive
    n: int
    hits: Optional[int]
    baseline: Optional[float]   # 平均理论基线
    observed: Optional[float]
    diff: Optional[float]
    ci_lo: Optional[float]
    ci_hi: Optional[float]
    z: Optional[float]
    p_value: Optional[float]
    p_adjusted: Optional[float]
    ev_theory: Optional[float]   # 每1元理论期望净收益
    holdout_obs: Optional[float]
    holdout_p: Optional[float]
    status: str
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------- 预测评测器 ----------

def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def eval_pick_rule(
    panel: BBPanel,
    cfg: AppConfig,
    fn: Callable[[BBPanel, int], Optional[int]],
    lo: int,
    hi: int,
) -> dict[str, Any]:
    """在 panel[lo,hi) 逐期触发评估（fn 只能看 [:t] 的 panel 数据由调用约定：本函数仅传 t，fn 内部切片）。"""
    picks: list[int] = []
    ys: list[int] = []
    ps: list[float] = []
    evs: list[float] = []
    for t in range(lo, hi):
        z = fn(panel, t)
        if z is None:
            continue
        picks.append(z)
        ys.append(int(panel.presence[t, z]))
        p = p_theory_of(z, panel)
        ps.append(p)
        o = odds_on(cfg, ZODIAC_ORDER[z], panel.dates[t])
        evs.append(p * o - 1.0)
    n = len(picks)
    k = int(sum(ys))
    obs = k / n if n else float("nan")
    base = float(np.mean(ps)) if n else float("nan")
    var = float(sum(p * (1 - p) for p in ps)) or 1e-9
    zscore = (k - sum(ps)) / math.sqrt(var) if n else float("nan")
    pval = float(1 - stats.norm.cdf(zscore)) if n else 1.0
    lo_ci, hi_ci = _wilson(k, n)
    return {"n": n, "hits": k, "baseline": base, "observed": obs,
            "diff": (obs - base) if n else float("nan"),
            "ci": (lo_ci, hi_ci), "z": zscore, "p": pval,
            "ev": float(np.mean(evs)) if n else float("nan"),
            "first_picks": [ZODIAC_ORDER[i] for i in picks[:8]]}


# ---------- 规则库（全部只读 panel[:t]） ----------

def r_hot10(p: BBPanel, t: int) -> Optional[int]:
    if t < 10:
        return None
    cnt = p.presence[t - 10:t].sum(axis=0)
    return int(np.argmax(cnt))


def r_cold30(p: BBPanel, t: int) -> Optional[int]:
    if t < 30:
        return None
    cnt = p.presence[t - 30:t].sum(axis=0)
    best = cnt.min()
    cands = [i for i in range(12) if cnt[i] == best]
    if len(cands) == 1:
        return cands[0]
    # 平手取缺席最久（最近一次出现最早），再取小序
    def last_occ(z: int) -> int:
        idx = np.nonzero(p.presence[max(0, t - 60):t, z])[0]
        return int(idx[-1]) if idx.size else -10**9
    return min(cands, key=lambda z: (last_occ(z), z))


def r_streak3(p: BBPanel, t: int) -> Optional[int]:
    """长龙跟随：上期特肖在最近3期特肖中出现≥2次才追，否则弃权。"""
    if t < 4:
        return None
    prev = p.special_idx[t - 1]
    recent = p.special_idx[max(0, t - 3):t]
    return int(prev) if int((recent == prev).sum()) >= 2 else None


def r_chong(p: BBPanel, t: int) -> Optional[int]:
    if t < 1:
        return None
    return ZIDX[CHONG[ZODIAC_ORDER[p.special_idx[t - 1]]]]


def r_p1(p: BBPanel, t: int) -> Optional[int]:
    return int(p.p1_idx[t - 1]) if t >= 1 else None


def r_p1_plus(p: BBPanel, t: int) -> Optional[int]:
    z = r_p1(p, t)
    return None if z is None else (z + 1) % 12


def r_p1_minus(p: BBPanel, t: int) -> Optional[int]:
    z = r_p1(p, t)
    return None if z is None else (z - 1) % 12


def _zn(n: int) -> int:
    return ZIDX[_z(n)]


def r_sp_plus(p: BBPanel, t: int) -> Optional[int]:
    if t < 1:
        return None
    return _zn_wrap(p.sp_num[t - 1] + 1)


def r_sp_minus(p: BBPanel, t: int) -> Optional[int]:
    if t < 1:
        return None
    return _zn_wrap(p.sp_num[t - 1] - 1)


def _zn_wrap(n: int) -> int:
    """号码±位移回绕1..49后取生肖（项目既有的 wrap49 口径）。"""
    n = ((n - 1) % 49) + 1
    return ZIDX[_z(n)]


def r_markov_sp(p: BBPanel, t: int, min_support: int = 15) -> Optional[int]:
    """上期特肖 → 下期特肖最大频转移；样本不足弃权。"""
    if t < 30:
        return None
    prev = p.special_idx[t - 1]
    cur = p.special_idx[:t - 1]      # 第 i 期
    nxt = p.special_idx[1:t]         # 第 i+1 期（i+1 <= t-1，均属历史）
    if cur.size == 0:
        return None
    mask = cur == prev
    if int(mask.sum()) < min_support:
        return None
    targets = nxt[mask]
    cnt = np.bincount(targets, minlength=12)
    return int(np.argmax(cnt))


PRED_RULES: list[tuple[PatternDef, Callable]] = [
    (PatternDef("hot10", "热10肖", "热冷",
                "最近10期全开出现次数最多的肖（平手取小序）",
                "presence[:t] 窗口10", "argmax 计数；每期必选",
                "全开一肖", "theory.single_prob(肖号码数) 平均", "BB一肖@真实赔率",
                "human"), r_hot10),
    (PatternDef("cold30", "冷30回补", "遗漏",
                "最近30期出现最少且缺席最久的肖（回补假设，待检验）",
                "presence[:t] 窗口30", "argmin 计数+最久缺席；每期必选",
                "全开一肖", "同上", "BB一肖@真实赔率", "human"), r_cold30),
    (PatternDef("streak3", "长龙跟随", "连断",
                "上期特肖在近3期特肖中出现≥2次则追该肖，否则弃权",
                "special_idx[:t]", "条件触发；否则不押",
                "全开一肖", "同上（触发子集）", "BB一肖@真实赔率", "human"),
     r_streak3),
    (PatternDef("chong", "特肖对冲", "转移",
                "上期特肖的地支对冲肖", "special_idx[t-1]",
                "查表对冲；每期必选", "全开一肖", "同上", "BB一肖@真实赔率",
                "human"), r_chong),
    (PatternDef("p1_z", "平1生肖", "平特",
                "上期第一平码所在生肖", "p1_idx[t-1]", "直接取肖；每期必选",
                "全开一肖", "同上", "BB一肖@真实赔率", "prior_work"),
     r_p1),
    (PatternDef("p1_plus1", "平1生肖+1", "平特",
                "上期第一平码生肖顺移1位", "p1_idx[t-1]",
                "(idx+1)%12；每期必选", "全开一肖", "同上",
                "BB一肖@真实赔率", "prior_work"), r_p1_plus),
    (PatternDef("p1_minus1", "平1生肖-1", "平特",
                "上期第一平码生肖逆移1位", "p1_idx[t-1]",
                "(idx-1)%12；每期必选", "全开一肖", "同上",
                "BB一肖@真实赔率", "prior_work"), r_p1_minus),
    (PatternDef("sp_plus1", "特码+1", "位移",
                "上期特码号码+1（49回绕）所在生肖", "sp_num[t-1]",
                "wrap49(n+1)→肖；每期必选", "全开一肖", "同上",
                "BB一肖@真实赔率", "prior_work"), r_sp_plus),
    (PatternDef("sp_minus1", "特码-1", "位移",
                "上期特码号码-1（回绕）所在生肖", "sp_num[t-1]",
                "wrap49(n-1)→肖；每期必选", "全开一肖", "同上",
                "BB一肖@真实赔率", "prior_work"), r_sp_minus),
    (PatternDef("markov_sp", "特肖Markov", "自动发现",
                "上期特肖条件下下期特肖最大频转移（支持度≥15）",
                "special_idx[:t]", "条件计数 argmax；不足弃权",
                "全开一肖", "同上（触发子集）", "BB一肖@真实赔率", "auto"),
     r_markov_sp),
]


# ---------- 描述性统计 ----------

def table_hotcold(panel: BBPanel, windows=(10, 20, 50)) -> dict[str, Any]:
    out = {}
    for w in windows:
        cnt = panel.presence[-w:].sum(axis=0)
        out[f"w{w}"] = {ZODIAC_ORDER[i]: int(cnt[i]) for i in range(12)}
    out["theory_per_window"] = {
        ZODIAC_ORDER[i]: round(p_theory_of(i, panel) * w, 2)
        for i in range(12) for w in [windows[-1]]}
    return out


def table_omission(panel: BBPanel) -> dict[str, int]:
    out = {}
    for z in range(12):
        idx = np.nonzero(panel.presence[:, z])[0]
        out[ZODIAC_ORDER[z]] = int(panel.presence.shape[0] - 1 - idx[-1]) \
            if idx.size else int(panel.presence.shape[0])
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def table_streaks(panel: BBPanel) -> dict[str, Any]:
    """长龙/断龙：每个肖连开长度 L 结束后，下一期全开命中率与样本量。"""
    from .features.zodiac_relations import run_outcomes
    out: dict[str, Any] = {}
    for z in range(12):
        ro = run_outcomes(panel.presence[:, z].astype(int))
        rows = []
        for L in sorted(ro):
            hit = ro[L][1]
            tot = ro[L][0] + ro[L][1]
            if tot >= 5:
                lo, hi = _wilson(hit, tot)
                rows.append({"streak": L, "n": tot, "next_hit": hit / tot,
                             "ci": [lo, hi]})
        out[ZODIAC_ORDER[z]] = rows
    return out


def test_same_period_pingte(panel: BBPanel) -> dict[str, Any]:
    """同期关系（描述+检验）：特码生肖也出现在6正码中的频率。

    红线：同期结构只能用于描述；预测当期特码时不得使用当期正码。
    """
    return _same_period_exact(panel)


def _same_period_exact(panel: BBPanel) -> dict[str, Any]:
    """特码生肖同时出现在6正码：观测 vs 条件超几何基线。"""
    hits = 0
    base_sum = 0.0
    n = panel.presence.shape[0]
    for t in range(n):
        z = int(panel.special_idx[t])
        hits += int(panel.reg_presence[t, z])
        kk = int(panel.zsize[z]) - 1       # 特码占1号，剩 size-1 在48池
        p_miss = 1.0
        for i in range(6):
            p_miss *= (48 - kk - i) / (48 - i)
        base_sum += 1.0 - p_miss
    obs = hits / n if n else float("nan")
    base = base_sum / n if n else float("nan")
    var = max(1e-9, n * base * (1 - base))
    z = (hits - base_sum) / math.sqrt(var)
    return {"n": n, "hits": hits, "observed": obs, "baseline": base,
            "diff": obs - base, "z": z,
            "p_two_sided": float(2 * (1 - stats.norm.cdf(abs(z))))}


def test_special_transition_chi2(panel: BBPanel) -> dict[str, Any]:
    """特肖 12×12 转移独立性（卡方），自动发现类。"""
    table = np.zeros((12, 12), dtype=int)
    sp = panel.special_idx
    for a, b in zip(sp, sp[1:]):
        table[a, b] += 1
    chi2, p, dof, _ = stats.chi2_contingency(table)
    return {"chi2": float(chi2), "dof": int(dof), "p": float(p),
            "n": int(len(sp) - 1)}


# ---------- 注册表执行 ----------

def _bh(pvals: list[float]) -> list[float]:
    m = len(pvals)
    if not m:
        return []
    order = np.argsort(pvals)
    out = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        prev = min(prev, pvals[i] * m / (rank + 1))
        out[i] = prev
    return out.tolist()


def run_registry(cfg: AppConfig | None = None,
                 export: bool = True) -> tuple[list[PatternResult], str]:
    cfg = cfg or load_config()
    panel = load_panel(cfg)
    T = len(panel.issues)
    dev_end = WARMUP + int((T - WARMUP) * (1 - HOLDOUT_FRAC))
    hold_lo = dev_end

    results: list[PatternResult] = []
    pred_pvals: list[float] = []
    pred_idx: list[int] = []

    # 1) 预测类规律：dev 段 + holdout 段
    for entry, fn in PRED_RULES:
        dev = eval_pick_rule(panel, cfg, fn, WARMUP, dev_end)
        hold = eval_pick_rule(panel, cfg, fn, hold_lo, T)
        res = PatternResult(
            **{k: getattr(entry, k) for k in
               ("pid", "name", "category", "definition", "inputs",
                "method", "target", "baseline_desc", "scope", "source")},
            segment="dev", n=dev["n"], hits=dev["hits"],
            baseline=dev["baseline"], observed=dev["observed"],
            diff=dev["diff"], ci_lo=dev["ci"][0], ci_hi=dev["ci"][1],
            z=dev["z"], p_value=dev["p"], p_adjusted=None,
            ev_theory=dev["ev"], holdout_obs=hold["observed"],
            holdout_p=hold["p"], status="exploratory",
            notes=f"holdout n={hold['n']} hits={hold['hits']}",
            extra={"first_picks": dev["first_picks"],
                   "holdout_n": hold["n"], "holdout_hits": hold["hits"]})
        pred_pvals.append(dev["p"])
        pred_idx.append(len(results))
        results.append(res)

    # 2) 同期平特关系（检验）
    same = _same_period_exact(panel)
    rel_entry = PatternDef(
        "same_pingte", "平特同期关系", "平特",
        "特码生肖同时出现在6正码中的频率 vs 超几何条件基线",
        "draws 全期（描述性，不可用于当期预测）",
        "条件超几何基线 + 正态近似检验", "同期关系",
        "1 - C(48-(size-1),6)/C(48,6)", "开奖结构描述（含泄漏红线说明）",
        "auto")
    results.append(PatternResult(
        **{k: getattr(rel_entry, k) for k in
           ("pid", "name", "category", "definition", "inputs", "method",
            "target", "baseline_desc", "scope", "source")},
        segment="full", n=same["n"], hits=same["hits"],
        baseline=same["baseline"], observed=same["observed"],
        diff=same["diff"], ci_lo=math.nan, ci_hi=math.nan,
        z=same["z"], p_value=same["p_two_sided"], p_adjusted=None,
        ev_theory=None, holdout_obs=None, holdout_p=None,
        status="exploratory",
        notes="同期结构描述；预测特码时不得使用当期正码"))
    pred_pvals.append(same["p_two_sided"])
    pred_idx.append(len(results) - 1)

    # 3) 特肖转移卡方（自动发现）
    tr = test_special_transition_chi2(panel)
    tr_entry = PatternDef(
        "sp_transition", "特肖转移独立性", "自动发现",
        "特码生肖12×12转移矩阵卡方独立性检验",
        "special_idx 序列", "chi2_contingency", "同期关系→跨期检验",
        "转移独立（uniform）", "全期", "auto")
    results.append(PatternResult(
        **{k: getattr(tr_entry, k) for k in
           ("pid", "name", "category", "definition", "inputs", "method",
            "target", "baseline_desc", "scope", "source")},
        segment="full", n=tr["n"], hits=None, baseline=None,
        observed=None, diff=None, ci_lo=None, ci_hi=None,
        z=None, p_value=tr["p"], p_adjusted=None, ev_theory=None,
        holdout_obs=None, holdout_p=None, status="exploratory",
        notes=f"chi2={tr['chi2']:.1f} dof={tr['dof']}",
        extra=tr))
    pred_pvals.append(tr["p"])
    pred_idx.append(len(results) - 1)

    # 4) BH 多重校正（覆盖全部检验）
    adj = _bh(pred_pvals)
    for i, a in zip(pred_idx, adj):
        results[i].p_adjusted = a

    # 5) 状态机
    for res in results:
        if res.target in ("描述",):
            res.status = "descriptive"
            continue
        q = res.p_adjusted
        if res.pid in ("same_pingte", "sp_transition"):
            res.status = "validated" if (q is not None and q < 0.05) \
                else "rejected"
            if res.status == "rejected":
                res.notes += "；未过BH→保留失败记录"
            continue
        # 预测类：BH + holdout 同向同显著
        hold_ok = (res.holdout_obs is not None
                   and res.diff is not None
                   and res.holdout_obs > (res.baseline or 0)
                   and (res.holdout_p or 1) < 0.05)
        if q is not None and q < 0.05 and hold_ok:
            res.status = "validated"
        elif res.diff is not None and res.diff < 0 and (q or 1) > 0.05:
            res.status = "rejected"
        else:
            res.status = "exploratory"

    # 6) 描述性表（不参与检验）
    desc_defs = [
        ("hotcold_table", "热冷多窗口计数表", "热冷",
         "窗口10/20/50的全开出现次数 vs 理论期望",
         table_hotcold(panel)),
        ("omission_rank", "当前遗漏排名", "遗漏",
         "距上次全开出现的期数", table_omission(panel)),
        ("streak_break_table", "长龙/断龙后续命中率", "连断",
         "连开L期结束后下一期全开命中率（Wilson CI，n≥5）",
         table_streaks(panel)),
    ]
    for pid, name, cat, desc, extra in desc_defs:
        results.append(PatternResult(
            pid=pid, name=name, category=cat, definition=desc,
            inputs="presence 全期", method="描述统计", target="描述",
            baseline_desc="不适用（描述性）", scope="历史描述",
            source="auto", segment="descriptive", n=0, hits=None,
            baseline=None, observed=None, diff=None, ci_lo=None,
            ci_hi=None, z=None, p_value=None, p_adjusted=None,
            ev_theory=None, holdout_obs=None, holdout_p=None,
            status="descriptive", extra=extra))

    # 7) 导出
    text = format_registry(results)
    if export:
        out = PROJECT_ROOT / "reports" / "bb_pattern_registry.json"
        out.parent.mkdir(exist_ok=True)
        payload = {
            "generated": "bb_analysis.patterns.run_registry",
            "n_patterns": len(results),
            "bh_denominator": len(pred_pvals),
            "segments": {"warmup": WARMUP, "dev_end": dev_end,
                         "holdout": [hold_lo, T]},
            "results": [r.to_dict() for r in results],
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                  default=str), encoding="utf-8")
        text += f"\n\n已导出: {out}"
    return results, text


def format_registry(results: list[PatternResult]) -> str:
    lines = ["=" * 100,
             "BB 规律注册表（评分卡口径：全开一肖；基线=理论命中率；BH 校正分母=全部检验数）",
             "=" * 100,
             f"{'pid':<14} {'名称':<12} {'类':<6} {'n':>5} {'观测':>7} "
             f"{'基线':>7} {'差':>7} {'95%CI':>15} {'p':>7} {'q':>7} "
             f"{'EV':>7} {'状态':>12}"]
    for r in results:
        if r.status == "descriptive":
            lines.append(f"{r.pid:<14} {r.name:<12} {r.category:<6} "
                         f"{'-':>5} {'-':>7} {'-':>7} {'-':>7} {'-':>15} "
                         f"{'-':>7} {'-':>7} {'-':>7} descriptive")
            continue
        ci = (f"[{r.ci_lo:.1%},{r.ci_hi:.1%}]"
              if (r.ci_lo is not None and r.ci_hi is not None
                  and not math.isnan(r.ci_lo)) else "-")
        q = f"{r.p_adjusted:.3f}" if r.p_adjusted is not None else "-"
        ev = f"{r.ev_theory:+.3f}" if r.ev_theory is not None else "-"
        obs = f"{r.observed:.1%}" if r.observed is not None else "-"
        base = f"{r.baseline:.1%}" if r.baseline is not None else "-"
        diff = f"{r.diff:+.1%}" if r.diff is not None else "-"
        pv = f"{r.p_value:.3f}" if r.p_value is not None else "-"
        lines.append(f"{r.pid:<14} {r.name:<12} {r.category:<6} "
                     f"{r.n:>5} {obs:>7} {base:>7} {diff:>7} {ci:>15} "
                     f"{pv:>7} {q:>7} {ev:>7} {r.status:>12}")
    lines += ["",
              "状态说明：descriptive=描述统计；exploratory=探索性（未过双门槛）；",
              "          validated=BH q<0.05 且 holdout 同向复现；rejected=已检验为负/未复现（失败记录保留）。",
              "EV = 理论 p×赔率−1（每1元），仅对预测类；负值=即使命中率达基线也亏。"]
    return "\n".join(lines)
