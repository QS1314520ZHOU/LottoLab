# -*- coding: utf-8 -*-
"""T10 · 资金策略模拟：固定注额 vs 三期下注策略（同一信号、同一路径）。

三期策略规则（任务书原文，逐期严格执行）：
  第 1 期（周期位0）：初始注额 s
  第 2 期（周期位1）：上一期未中 → 2s；上一期中 → s
  第 3 期（周期位2）：s（不论第2期输赢）
  之后按 3 期为周期重复。

结算口径（真实赔率配置，总返还倍数 O 含本金）：
  中奖净收益 = stake × (O − 1)；未中净收益 = −stake。

硬约束：
  - 策略只改变注额节奏，不改变命中概率（测试断言两策略命中序列一致）；
  - 报告总投入/总返还/净盈亏/最大回撤/最大连亏/最大资金需求/破产风险；
  - 绝不将加倍描述为提高中奖概率的方法。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .config import AppConfig, load_config

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent


class Strategy:
    """注额策略接口。t 为 1 起期号（会话内），prev_lost 为上一期是否未中。"""

    name = "base"

    def stake(self, t: int, prev_lost: Optional[bool]) -> float:
        raise NotImplementedError


@dataclass
class FixedStake(Strategy):
    base: float
    name: str = "fixed"

    def stake(self, t: int, prev_lost: Optional[bool]) -> float:
        return self.base


@dataclass
class ThreePeriod(Strategy):
    """三期周期策略（见模块 docstring）。"""

    base: float
    name: str = "three_period"

    def stake(self, t: int, prev_lost: Optional[bool]) -> float:
        pos = (t - 1) % 3
        if pos == 0:
            return self.base
        if pos == 1:
            return self.base * 2.0 if prev_lost else self.base
        return self.base


@dataclass
class Ledger:
    strategy: str
    initial: float
    stakes: list[float] = field(default_factory=list)
    nets: list[float] = field(default_factory=list)
    balances: list[float] = field(default_factory=list)
    ruin_at: Optional[int] = None      # 第几期（会话内）无法继续
    hit_flags: list[bool] = field(default_factory=list)

    @property
    def total_staked(self) -> float:
        return float(sum(self.stakes))

    @property
    def total_return(self) -> float:
        # 每期返还 = stake + net（未中时 net=−stake → 返还0）
        return self.total_staked + self.net

    @property
    def net(self) -> float:
        return float(sum(self.nets))

    @property
    def max_drawdown(self) -> float:
        peak, mdd = 0.0, 0.0
        for b in self.balances:
            peak = max(peak, b)
            mdd = max(mdd, peak - b)
        return mdd

    @property
    def max_consec_loss(self) -> int:
        best = cur = 0
        for h in self.hit_flags:
            cur = cur + 1 if not h else 0
            best = max(best, cur)
        return best

    @property
    def max_capital_need(self) -> float:
        """最大垫付需求：累计投入 − 累计返还 的峰值（资金占用）。"""
        staked = ret = peak = 0.0
        for s, n in zip(self.stakes, self.nets):
            staked += s
            ret += s + n if n > 0 else 0.0   # 中奖时返还=stake+net
            peak = max(peak, staked - ret)
        return peak


def simulate(outcomes: list[dict[str, Any]], strategy: Strategy,
             initial: float) -> Ledger:
    """outcomes: [{hit:bool, odds:float}, ...] 与策略无关的共同信号路径。"""
    led = Ledger(strategy.name, initial)
    balance = initial
    prev_lost: Optional[bool] = None
    for i, o in enumerate(outcomes):
        t = i + 1
        s = strategy.stake(t, prev_lost)
        if balance < s:
            led.ruin_at = t
            break
        if o["hit"]:
            net = s * (o["odds"] - 1.0)
        else:
            net = -s
        balance += net
        led.stakes.append(s)
        led.nets.append(net)
        led.balances.append(balance)
        led.hit_flags.append(bool(o["hit"]))
        prev_lost = not o["hit"]
    return led


# ---------- 信号路径：从回测预测+开奖重建 ----------

def load_signal_path(cfg: AppConfig | None = None) -> list[dict[str, Any]]:
    """最近一次回测 run 的 selected 预测 → (hit, odds) 序列。"""
    from . import database as db
    cfg = cfg or load_config()
    db.migrate(cfg.db_path)
    with db.connect(cfg.db_path) as conn:
        run = conn.execute(
            "SELECT id FROM backtest_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if run is None:
            return []
        rows = [dict(r) for r in conn.execute(
            "SELECT p.target_issue, p.combo, p.odds, p.selected FROM "
            "predictions p WHERE p.run_id=? AND p.selected=1 AND p.combo!=''"
            " ORDER BY p.target_issue", (run["id"],))]
        draws = {d["issue"]: d for d in db.load_draws(conn)}
    out = []
    for r in rows:
        d = draws.get(r["target_issue"])
        if d is None:
            continue
        z = r["combo"]
        hit = any(_znum(d, i) == z for i in range(1, 7)) or \
            _znum(d, 7) == z
        out.append({"hit": hit, "odds": float(r["odds"]), "zodiac": z})
    return out


def _znum(d: dict, k: int) -> str:
    key = "sp" if k == 7 else f"p{k}"
    mod = {1: "马", 2: "蛇", 3: "龙", 4: "兔", 5: "虎", 6: "牛",
           7: "鼠", 8: "猪", 9: "狗", 10: "鸡", 11: "猴", 0: "羊"}
    return mod[d[key] % 12]


# ---------- MC 压力测试 ----------

def monte_carlo(outcomes: list[dict[str, Any]], strategy_cls, base: float,
                initial: float, n_sims: int = 2000,
                seed: int = 42) -> dict[str, Any]:
    """破产风险：对历史 (hit, odds) 序列做 i.i.d. 重采样。

    假设：各期结果独立同分布（经验分布重采样）——真实序列可能有依赖，
    结果仅作压力参考，不是破产概率的真实估计。
    """
    rng = np.random.default_rng(seed)
    n = len(outcomes)
    if n == 0:
        return {"ruin_rate": float("nan"), "note": "no_signal"}
    ruins = 0
    nets = []
    mdds = []
    for _ in range(n_sims):
        idx = rng.integers(0, n, size=n)
        path = [outcomes[i] for i in idx]
        led = simulate(path, strategy_cls(base), initial)
        if led.ruin_at is not None:
            ruins += 1
        nets.append(led.net)
        mdds.append(led.max_drawdown)
    return {"n_sims": n_sims, "seed": seed,
            "ruin_rate": ruins / n_sims,
            "net_mean": float(np.mean(nets)),
            "net_p5": float(np.percentile(nets, 5)),
            "net_p95": float(np.percentile(nets, 95)),
            "mdd_mean": float(np.mean(mdds)),
            "mdd_p95": float(np.percentile(mdds, 95)),
            "assumption": "经验分布 i.i.d. 重采样；非真实依赖结构"}


def run_bankroll(cfg: AppConfig | None = None,
                 initials: tuple[float, ...] = (100.0, 200.0, 500.0),
                 base: float = 5.0) -> str:
    cfg = cfg or load_config()
    path = load_signal_path(cfg)
    if not path:
        return "无可用信号路径：请先运行回测（backtest）。"
    hits = [o["hit"] for o in path]
    lines = ["=" * 84,
             "BB 资金策略模拟（同一信号路径、同一开奖；策略不改变命中率）",
             f"信号路径：回测 selected 预测 {len(path)} 期 ｜ "
             f"命中率 {np.mean(hits):.1%} ｜ 基础注额 {base} ｜ "
             f"赔率=总返还倍数（含本金），净收益=stake×(O−1)",
             "-" * 84,
             f"{'初始资金':>8} {'策略':<14} {'总投入':>9} {'总返还':>9} "
             f"{'净盈亏':>9} {'最大回撤':>9} {'最大连亏':>8} "
             f"{'最大垫付':>9} {'破产期':>7}"]
    rows = []
    for init in initials:
        for cls in (FixedStake, ThreePeriod):
            led = simulate(path, cls(base), init)
            ruin = str(led.ruin_at) if led.ruin_at else "-"
            lines.append(f"{init:>8.0f} {cls.__name__:<14} "
                         f"{led.total_staked:>9.2f} "
                         f"{led.total_staked + led.net:>9.2f} "
                         f"{led.net:>+9.2f} {led.max_drawdown:>9.2f} "
                         f"{led.max_consec_loss:>8d} "
                         f"{led.max_capital_need:>9.2f} {ruin:>7}")
            rows.append({"initial": init, "strategy": cls.__name__,
                         "net": led.net, "mdd": led.max_drawdown,
                         "ruin_at": led.ruin_at})
    lines.append("-" * 84)
    lines.append("[MC 破产压力 · 200期窗口 · 初始200 · 注额5]")
    for cls in (FixedStake, ThreePeriod):
        mc = monte_carlo(path, cls, base, 200.0)
        lines.append(f"  {cls.__name__:<14} 破产率={mc['ruin_rate']:.2%} "
                     f"净利P5..P95=[{mc['net_p5']:+.1f},{mc['net_p95']:+.1f}] "
                     f"回撤P95={mc['mdd_p95']:.1f} ｜ 假设：{mc['assumption']}")
    lines += [
        "结论与红线：",
        "* 三期策略与固定注额命中序列完全相同——加倍只改变注额节奏，"
        "不提高任何中奖概率；期望净收益方向由赔率与命中率决定，加倍放大波动。",
        "* 破产率/回撤均为重采样压力值，依赖“各期独立同分布”假设，"
        "真实路径可能更差；小样本信号（251期）下所有数字仅供参考。",
        "* 本表不构成投注建议。"]
    out = PROJECT_ROOT / "reports" / "bb_bankroll.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "path_len": len(path),
                               "hit_rate": float(np.mean(hits))},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    return "\n".join(lines) + f"\n\n已导出: {out}"
