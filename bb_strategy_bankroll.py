# -*- coding: utf-8 -*-
"""
bb_strategy_bankroll.py

按【算法选肖 + 真实赔率】的会话资金模拟：
  赔率：4号肖 2.06 ｜ 马 1.76（算法选到马就按1.76结算）
  本金200 ｜ 每注5平注

对8个短窗策略 + 随机对照，回答：
  1) 枚举历史全部10期/20期窗口：P(赢) 与 每期期望
  2) 蒙特卡洛（按各策略经验结局分布重采样）：多少期爆仓
真实BB数据与纯随机数据各跑一遍——算法是否真的改变了赔率命运，对照见分晓。

用法：python bb_strategy_bankroll.py
"""

import random

import numpy as np

import bb_zodiac_v7 as bb
from bb_zodiac_shortwin import STRATEGIES, WARMUP

BANKROLL = 200.0
STAKE = 5.0
ODDS4, ODDS_MA = 2.06, 1.76
SESSIONS = (10, 20)
N_PATHS = 60_000
CAP = 5_000          # 模拟上限5000期（约250~500天）
SEED = 42


def build_rows(draws):
    """策略名 -> [(赔率, 是否命中), ...]，逐期跟随算法选肖。"""
    out = {}
    for name, picker in STRATEGIES:
        rows = []
        for t in range(WARMUP, len(draws)):
            z = picker(draws, t)
            odds = ODDS_MA if z == "马" else ODDS4
            hit = bool(set(bb.zodiac_numbers(z)) & set(draws[t].numbers))
            rows.append((odds, hit))
        out[name] = rows

    rng = random.Random(7)
    rows = []
    for t in range(WARMUP, len(draws)):
        z = bb.ZODIACS[rng.randrange(len(bb.ZODIACS))]
        odds = ODDS_MA if z == "马" else ODDS4
        hit = bool(set(bb.zodiac_numbers(z)) & set(draws[t].numbers))
        rows.append((odds, hit))
    out["随机对照"] = rows
    return out


def net_array(rows):
    return np.array(
        [STAKE * (odds - 1.0) if hit else -STAKE for odds, hit in rows],
        dtype=np.float32,
    )


def session_stats(nets, L):
    pref = np.concatenate([[0.0], np.cumsum(nets, dtype=np.float64)])
    total = pref[L:] - pref[:-L]
    return float(np.mean(total > 0)), float(np.mean(total))


def bust_mc(nets, seed):
    rng = np.random.default_rng(seed)
    n = len(nets)
    bal = np.full(N_PATHS, BANKROLL, dtype=np.float32)
    alive = np.ones(N_PATHS, dtype=bool)
    ruin_t = np.full(N_PATHS, -1, dtype=np.int32)
    period = 0
    while alive.any() and period < CAP:
        period += 1
        idx = np.flatnonzero(alive)
        bal[idx] += nets[rng.integers(0, n, size=idx.size)]
        broke = bal[idx] < STAKE
        if broke.any():
            ruin_t[idx[broke]] = period
            alive[idx[broke]] = False
    return ruin_t


def run(csv_path, label):
    draws = bb.load_csv(csv_path)
    draws.sort(key=bb.issue_sort_key)
    rows_map = build_rows(draws)

    print("=" * 96)
    print(f"【{label}】共{len(draws)}期 ｜ 评估 {WARMUP}..{len(draws)-1} "
          f"共 {len(draws) - WARMUP} 期 ｜ 本金200 每注5 ｜ 4号肖@2.06 马@1.76")

    print(f"\n{'策略':<12} {'命中率':>7} {'选马占比':>7} {'EV/期':>8} "
          f"{'P赢10期':>8} {'P赢20期':>8} {'期望/10期':>9} {'期望/20期':>9}")
    nets_map = {}
    for name, rows in rows_map.items():
        nets = net_array(rows)
        nets_map[name] = nets
        hits = [h for _, h in rows]
        ma_frac = sum(1 for o, _ in rows if o == ODDS_MA) / len(rows)
        ev = float(np.mean(nets))
        p10, e10 = session_stats(nets, 10)
        p20, e20 = session_stats(nets, 20)
        print(f"{name:<12} {np.mean(hits):>7.2%} {ma_frac:>7.1%} "
              f"{ev:>+8.3f} {p10:>8.2%} {p20:>8.2%} "
              f"{e10:>+9.2f} {e20:>+9.2f}")

    print(f"\n{'策略':<12} {'中位爆仓期数':>10} {'P(≤200期爆仓)':>12} "
          f"P(≤600期)  P(≤1000期)  P(≤5000期)")
    for i, (name, nets) in enumerate(nets_map.items()):
        ruin_t = bust_mc(nets, SEED + i)
        busted = ruin_t[ruin_t != -1]
        med = float(np.median(busted)) if busted.size else float("nan")
        p200 = np.mean((ruin_t != -1) & (ruin_t <= 200))
        p600 = np.mean((ruin_t != -1) & (ruin_t <= 600))
        p1000 = np.mean((ruin_t != -1) & (ruin_t <= 1000))
        p5000 = np.mean(ruin_t != -1)
        print(f"{name:<12} {med:>10.0f} {p200:>12.2%} "
              f"{p600:>9.2%} {p1000:>10.2%} {p5000:>11.2%}")


if __name__ == "__main__":
    import sys
    run(sys.argv[1], sys.argv[2])
    if len(sys.argv) >= 4:
        run(sys.argv[3], sys.argv[4])
