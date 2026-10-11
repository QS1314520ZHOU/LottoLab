# -*- coding: utf-8 -*-
"""票据层（P1）：候选排序 / 合法性校验 / 固定预算组合 / 同口径 ROI 风险。

口径红线：
  * 蓝球 top1 命中 ≠ 整注中奖率；红球指标一律叫“单球命中/覆盖”。
  * 所有策略使用**相同预算、相同测试期**对比；预算 = k 注 × 单注2元。
  * 一/二等浮动奖按 0 保守计入（见 config/app.json prize_fixed.note）。
  * 随机对照固定种子，可复现。
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .config import PROJECT_ROOT, SSQConfig
from .features import FEATURE_VERSION
from .models import model_c_blue, model_c_red

PRICE_DEFAULT = 2.0


def validate_reds(reds: list[int], cfg: SSQConfig) -> None:
    if (len(reds) != cfg.red_count or len(set(reds)) != cfg.red_count
            or not all(1 <= x <= cfg.red_pool for x in reds)):
        raise ValueError(f"红球非法: {reds}（须{cfg.red_count}个互异1-"
                         f"{cfg.red_pool}）")


def validate_blue(blue: int, cfg: SSQConfig) -> None:
    if not (1 <= blue <= cfg.blue_pool):
        raise ValueError(f"蓝球非法: {blue}（须1-{cfg.blue_pool}）")


@dataclass(frozen=True)
class Ticket:
    reds: tuple[int, ...]
    blue: int

    def key(self) -> tuple:
        return (tuple(sorted(self.reds)), self.blue)


def rank_candidates(payload: dict[str, Any],
                    cfg: SSQConfig) -> dict[str, list[tuple[int, float]]]:
    """按 fv1 特征输出候选排序（概率降序）：
    blue：蓝球16分类后验概率；red：单球入选概率（Beta后验）。"""
    blue = model_c_blue(payload, cfg)
    red, _lo, _hi = model_c_red(payload, cfg)
    blue_rank = sorted(
        ((b, float(blue.probs[b - 1])) for b in range(1, cfg.blue_pool + 1)),
        key=lambda x: -x[1])
    red_rank = sorted(
        ((r, float(red[r - 1])) for r in range(1, cfg.red_pool + 1)),
        key=lambda x: -x[1])
    return {"blue": blue_rank, "red": red_rank}


def make_tickets(cfg: SSQConfig, budget_yuan: float, strategy: str,
                 ranks: Optional[dict[str, list[tuple[int, float]]]] = None,
                 seed: int = 42) -> dict[str, Any]:
    """固定预算生成互异合法票据。

    strategy: 'model'（需 ranks，红球按排名轮转覆盖+蓝球首选）
              'random'（固定种子随机）
    返回 {tickets, k, cost, budget_left, strategy, budget}。
    """
    price = float(cfg.ticket_price)
    k = int(budget_yuan // price)
    if k <= 0:
        raise ValueError(f"预算 {budget_yuan} 元不足 1 注（单注 {price} 元）")
    rng = random.Random(seed)
    tickets: set[tuple] = set()
    attempt = 0
    guard = 0
    # model 策略：蓝球取 top-k 轮换（避免10注同蓝全军覆没），
    # 红球按入选概率加权无放回抽样（保证组合分散）
    blue_cycle: list[int] = []
    red_weights: list[float] = []
    if strategy == "model":
        assert ranks is not None, "model 策略需要 ranks"
        blue_cycle = [b for b, _ in ranks["blue"][:max(2, k)]]
        total_w = sum(w for _, w in ranks["red"]) or 1.0
        red_weights = [w / total_w for _, w in ranks["red"]]
        red_vals = [v for v, _ in ranks["red"]]
    while len(tickets) < k and guard < k * 500:
        guard += 1
        attempt += 1
        if strategy == "model":
            blue = blue_cycle[(attempt - 1) % len(blue_cycle)]
            # 按概率无放回抽6红（大权重号更常出现但组合互异）
            picked: list[int] = []
            pool = list(red_vals)
            weights = list(red_weights)
            for _ in range(cfg.red_count):
                tot = sum(weights)
                if tot <= 0:
                    break
                r = rng.random() * tot
                acc = 0.0
                idx = 0
                for i, w in enumerate(weights):
                    acc += w
                    if r <= acc:
                        idx = i
                        break
                picked.append(pool.pop(idx))
                weights.pop(idx)
            if len(picked) < cfg.red_count:
                continue
            reds = sorted(picked)
        elif strategy == "random":
            reds = sorted(rng.sample(range(1, cfg.red_pool + 1),
                                     cfg.red_count))
            blue = rng.randint(1, cfg.blue_pool)
        else:
            raise ValueError(f"未知策略 {strategy}")
        validate_reds(reds, cfg)
        validate_blue(blue, cfg)
        tickets.add((tuple(reds), blue))
    if len(tickets) < k:
        raise RuntimeError("组合去重后数量不足（预算过大或随机空间冲突）")
    out = [Ticket(tuple(sorted(t[0])), t[1]) for t in sorted(tickets)]
    cost = len(out) * price
    return {"tickets": out, "k": len(out), "cost": cost,
            "budget_left": round(budget_yuan - cost, 2),
            "strategy": strategy, "budget": budget_yuan}


def prize_for(red_hit: int, blue_hit: bool,
              cfg: SSQConfig) -> float:
    """固定奖映射（一/二等浮动=0）。"""
    p = cfg.prize_fixed
    if red_hit == 6 and blue_hit:
        return 0.0            # 一等奖浮动，保守计0（假设已标注）
    if red_hit == 5 and blue_hit:
        return 0.0            # 二等奖浮动，保守计0
    if red_hit == 4 and blue_hit:
        return float(p["4plus1"])
    if red_hit == 3 and blue_hit:
        return float(p["3plus1"])
    if red_hit == 2 and blue_hit:
        return float(p["2plus1"])
    if blue_hit and red_hit in (0, 1):
        return float(p["0plus1"] if red_hit == 0 else p["1plus1"])
    return 0.0


@dataclass
class BudgetResult:
    strategy: str
    budget: float
    price: float
    periods: int = 0
    total_cost: float = 0.0
    total_return: float = 0.0
    net: float = 0.0
    roi: float = float("nan")
    prize_periods: int = 0          # 有任何返奖的期数
    blue_hits: int = 0              # 蓝球命中注数（单指标，非整注中奖率）
    big_prizes: int = 0             # ≥1000元返奖次数（三等3000主导方差）
    red_ball_hits: int = 0          # 红球单球命中累计
    red_balls: int = 0              # 红球参与总数（分母）
    max_drawdown: float = 0.0
    longest_lose_streak: int = 0
    curve: list[float] = field(default_factory=list)


def evaluate_budget(cfg: SSQConfig,
                    draws: list[dict[str, Any]],
                    ticket_sets: list[list[Ticket]],
                    label: str) -> BudgetResult:
    """逐期结算；draws 与 ticket_sets 等长，同周期同预算。"""
    price = float(cfg.ticket_price)
    res = BudgetResult(strategy=label, budget=price * len(ticket_sets[0]),
                       price=price)
    equity = 0.0
    peak = 0.0
    lose = 0
    for d, tickets in zip(draws, ticket_sets):
        draw_reds = {d[f"r{i}"] for i in range(1, 7)}
        draw_blue = d["b1"]
        res.periods += 1
        cost = len(tickets) * price
        ret = 0.0
        for t in tickets:
            res.red_balls += len(t.reds)
            res.red_ball_hits += len(set(t.reds) & draw_reds)
            rh = len(set(t.reds) & draw_reds)
            bh = t.blue == draw_blue
            pr = prize_for(rh, bh, cfg)
            ret += pr
            if pr >= 1000:
                res.big_prizes += 1
            if bh:
                res.blue_hits += 1
        res.total_cost += cost
        res.total_return += ret
        equity += ret - cost
        res.curve.append(equity)
        peak = max(peak, equity)
        res.max_drawdown = max(res.max_drawdown, peak - equity)
        if ret > 0:
            res.prize_periods += 1
            lose = 0
        else:
            lose += 1
            res.longest_lose_streak = max(res.longest_lose_streak, lose)
    res.net = res.total_return - res.total_cost
    if res.total_cost > 0:
        res.roi = res.net / res.total_cost
    return res


def run_budget_study(cfg: SSQConfig | None = None,
                     budget_yuan: float = 20.0,
                     split_frac: float = 0.3,
                     seed: int = 42) -> str:
    """测试段（后 split_frac）同预算对比 model vs random，并导出 JSON。"""
    from . import db
    from .config import load_config
    cfg = cfg or load_config()
    db.migrate(cfg.db_path)
    with db.connect(cfg.db_path) as conn:
        draws = db.load_draws(conn, status="ok")
    if len(draws) < 200:
        return "数据不足"
    from .features import build_features
    feats = build_features(draws, cfg)
    start = int(len(draws) * (1 - split_frac))
    test_draws = draws[start:]
    # 逐期排名（只用 [:t] 特征）
    model_tickets: list[list[Ticket]] = []
    rand_tickets: list[list[Ticket]] = []
    for t in range(start, len(draws)):
        ranks = rank_candidates(feats[draws[t]["issue"]], cfg)
        model_tickets.append(
            make_tickets(cfg, budget_yuan, "model", ranks, seed)["tickets"])
        rand_tickets.append(
            make_tickets(cfg, budget_yuan, "random", None, seed + t)["tickets"])
    res_m = evaluate_budget(cfg, test_draws, model_tickets, "model")
    res_r = evaluate_budget(cfg, test_draws, rand_tickets,
                            f"random(seed={seed})")

    def row(r: BudgetResult) -> dict[str, Any]:
        return {"strategy": r.strategy, "budget_per_period": r.budget,
                "periods": r.periods, "total_cost": r.total_cost,
                "total_return": r.total_return, "net": r.net,
                "roi": r.roi,
                "prize_period_rate": r.prize_periods / max(1, r.periods),
                "blue_hit_per_ticket":
                    r.blue_hits / max(1, r.periods * int(
                        r.budget // r.price)),
                "red_ball_hit_rate": r.red_ball_hits / max(1, r.red_balls),
                "big_prizes_ge_1000": r.big_prizes,
                "max_drawdown": r.max_drawdown,
                "longest_lose_streak": r.longest_lose_streak}

    payload = {
        "test_period": [test_draws[0]["issue"], test_draws[-1]["issue"]],
        "budget_per_period": budget_yuan, "price": cfg.ticket_price,
        "prizes": cfg.prize_fixed,
        "assumptions": [
            "一/二等浮动奖按0保守计入",
            "两策略同预算同测试期；随机策略固定种子",
            "红球指标=单球命中率，蓝球指标=每注命中率，均非整注中奖率",
            f"模型策略只用 {FEATURE_VERSION}（[:t]）特征",
            "三等3000元主导方差：千期级ROI噪声可达±15%，策略差异主要来自大额奖时机",
        ],
        "model": row(res_m), "random": row(res_r),
    }
    out = PROJECT_ROOT / "reports" / "ssq_budget_report.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    lines = ["=" * 78,
             "双色球固定预算对照（同预算同测试期 · 单注2元）",
             f"测试期: {payload['test_period']} ｜ 预算 {budget_yuan} 元/期",
             f"{'策略':<16} {'总成本':>9} {'总返奖':>9} {'净盈亏':>9} "
             f"{'ROI':>8} {'有奖期占比':>9} {'≥1000奖':>7} "
             f"{'最大回撤':>9} {'最长连亏':>8}"]
    for r in (res_m, res_r):
        lines.append(f"{r.strategy:<16} {r.total_cost:>9.0f} "
                     f"{r.total_return:>9.0f} {r.net:>+9.0f} "
                     f"{r.roi:>8.1%} "
                     f"{r.prize_periods / max(1, r.periods):>9.1%} "
                     f"{r.big_prizes:>7d} "
                     f"{r.max_drawdown:>9.1f} {r.longest_lose_streak:>8d}")
    lines += [
        "口径：蓝球=每注命中率（非整注中奖率）；红球=单球命中率；",
        "      一/二等浮动奖按0保守计入，ROI 被系统性低估（保守口径）。",
        "方差警示：三等3000元主导方差（见≥1000奖列），千期级ROI噪声可达"
        "±15%——两策略差异主要来自大额奖出现时机，不代表模型优劣。",
        "结论仅描述该测试段；小样本下差异不代表未来优势。"]
    txt = "\n".join(lines)
    (PROJECT_ROOT / "reports" / "ssq_budget_report.txt").write_text(
        txt, encoding="utf-8")
    return txt + f"\n\n已导出: {out}"
