# -*- coding: utf-8 -*-
"""
bb_session_sim.py

10期/20期短会话赢面模拟（对应"每天只玩10~20期"的真实场景）。

对每个策略生成逐期选肖序列，再用滑动窗口枚举所有可能的10期/20期会话：
- 单肖全开：每期1注，net = 赔率*命中数 - 期数
- 单肖特肖：同上，口径换成特码
- 双肖全开：每期2注各1单位（策略肖 + 30期最冷肖），net = 赔率*总命中 - 2*期数
- 马丁格尔：同一批命中序列上模拟（赢回1单位、输翻倍、赢后重置）

输出：P(会话盈利)、每期期望、马丁格尔尾部风险。
真实数据与纯随机数据各跑一遍对照。

用法：python bb_session_sim.py data/bb_live_history.csv "真实BB" [data/bb_null_history.csv "随机对照"]
"""

import math
import random
import sys

from scipy.stats import binom

import bb_zodiac_v7 as bb
from bb_zodiac_shortwin import (
    STRATEGIES, P_OPEN, P_SPECIAL, WARMUP,
    pick_cold, appearances, absence,
)

OPEN_ODDS = 2.1   # 全开一肖（用户口径1.8~2.1取上限）
SP_ODDS = 11.0    # 特肖（用户口径10~12取中值）
SESSIONS = (10, 20)


def second_pick(draws, t, z1):
    """双肖的第二注：30期最冷肖（若与第一注重复则取次冷）。"""
    z2 = pick_cold(draws, t, 30)
    if z2 != z1:
        return z2
    ranked = sorted(
        bb.ZODIACS,
        key=lambda z: (appearances(draws, z, t - 30, t),
                       -absence(draws, z, t), bb.ZODIACS.index(z)),
    )
    for z in ranked:
        if z != z1:
            return z
    return z1


def build_sequences(draws):
    """每个策略 -> (全开命中列表, 特肖命中列表, 双肖全开命中列表[每期0/1/2])。"""
    ts = range(WARMUP, len(draws))
    eval_draws = draws[WARMUP:]
    seqs = {}

    for name, picker in STRATEGIES:
        ho, hs, h2 = [], [], []
        for t, d in zip(ts, eval_draws):
            z1 = picker(draws, t)
            nums1 = set(bb.zodiac_numbers(z1))
            dn = set(d.numbers)
            hit1 = bool(nums1 & dn)
            sp1 = d.special in nums1
            z2 = second_pick(draws, t, z1)
            hit2 = bool(set(bb.zodiac_numbers(z2)) & dn)
            ho.append(hit1)
            hs.append(sp1)
            h2.append(int(hit1) + int(hit2))
        seqs[name] = (ho, hs, h2)

    # 随机单肖（固定种子可复现）
    rng = random.Random(7)
    rh = []
    rhs = []
    for d in eval_draws:
        z = bb.ZODIACS[rng.randrange(len(bb.ZODIACS))]
        nums = set(bb.zodiac_numbers(z))
        rh.append(bool(nums & set(d.numbers)))
        rhs.append(d.special in nums)
    seqs["随机单肖"] = (rh, rhs, None)

    # 随机双肖
    rng2 = random.Random(8)
    r2 = []
    for d in eval_draws:
        a, b = rng2.sample(range(len(bb.ZODIACS)), 2)
        na = set(bb.zodiac_numbers(bb.ZODIACS[a]))
        nb = set(bb.zodiac_numbers(bb.ZODIACS[b]))
        dn = set(d.numbers)
        r2.append(int(bool(na & dn)) + int(bool(nb & dn)))
    seqs["随机双肖"] = (None, None, r2)

    return seqs


def window_stats(hits, odds, stake_per_period, L):
    """滑动窗口：返回 (窗口数, 盈利窗口占比, 平均每期净利)。"""
    n = len(hits)
    m = n - L + 1
    if m <= 0:
        return 0, float("nan"), float("nan")
    win = 0
    total_net = 0.0
    prefix = [0] * (n + 1)
    for i, h in enumerate(hits):
        prefix[i + 1] = prefix[i] + h
    for s in range(m):
        h_sum = prefix[s + L] - prefix[s]
        net = odds * h_sum - stake_per_period * L
        total_net += net
        win += int(net > 0)
    return m, win / m, total_net / m / L


def martingale_net(hits, odds, base=1.0):
    """赢回一单位、输翻倍、赢后重置；返回会话净利与最低点。"""
    stake = base
    net = 0.0
    trough = 0.0
    for h in hits:
        if h:
            net += (odds - 1.0) * stake
            stake = base
        else:
            net -= stake
            stake *= 2.0
        trough = min(trough, net)
    return net, trough


def martingale_stats(hits, odds):
    n = len(hits)
    out = {}
    for L in SESSIONS:
        m = n - L + 1
        if m <= 0:
            continue
        wins = 0
        totals = []
        big_loss = 0
        worst = 0.0
        for s in range(m):
            net, _ = martingale_net(hits[s:s + L], odds)
            totals.append(net)
            wins += int(net > 0)
            big_loss += int(net <= -100)
            worst = min(worst, net)
        out[L] = (wins / m, sum(totals) / m, big_loss / m, worst)
    return out


def analytic_random_single(L, odds, kind):
    """随机单肖的解析解：全开 p=(ΣP_OPEN)/12，特肖 p=1/12。"""
    if kind == "open":
        p = sum(P_OPEN[z] for z in bb.ZODIACS) / len(bb.ZODIACS)
    else:
        p = sum(P_SPECIAL[z] for z in bb.ZODIACS) / len(bb.ZODIACS)
    # net = odds*h - L > 0  =>  h >= floor(L/odds)+1
    k = math.floor(L / odds) + 1
    p_win = float(binom.sf(k - 1, L, p))
    ev = p * odds - 1.0
    return p_win, ev


def run(csv_path, label):
    draws = bb.load_csv(csv_path)
    draws.sort(key=bb.issue_sort_key)
    n = len(draws)
    seqs = build_sequences(draws)

    print("=" * 88)
    print(f"【{label}】{n}期 ｜ 评估 {WARMUP}..{n-1} 共 {n - WARMUP} 期 ｜ "
          f"全开赔率{OPEN_ODDS} 特肖赔率{SP_ODDS} ｜ "
          f"10期窗 {n - WARMUP - 9} 个 / 20期窗 {n - WARMUP - 19} 个")

    # --- 表A：单肖全开 ---
    print("\n[表A] 单肖全开 @赔率%.1f —— P(会话盈利) / 每期期望" % OPEN_ODDS)
    print(f"{'策略':<12} {'P赢10期':>8} {'P赢20期':>8} {'期望/期':>9}")
    rows_a = []
    for name, (ho, hs, _) in seqs.items():
        if ho is None:
            continue
        r = {}
        for L in SESSIONS:
            _, pw, ev = window_stats(ho, OPEN_ODDS, 1.0, L)
            r[L] = pw
        _, _, ev = window_stats(ho, OPEN_ODDS, 1.0, 10)
        rows_a.append((name, r[10], r[20], ev))
        print(f"{name:<12} {r[10]:>8.1%} {r[20]:>8.1%} {ev:>+9.2%}")
    p10a, ev_a = analytic_random_single(10, OPEN_ODDS, "open")
    p20a, _ = analytic_random_single(20, OPEN_ODDS, "open")
    print(f"{'随机(解析)':<12} {p10a:>8.1%} {p20a:>8.1%} {ev_a:>+9.2%}"
          f"   p={sum(P_OPEN[z] for z in bb.ZODIACS) / 12:.4f}")

    # --- 表B：单肖特肖 ---
    print("\n[表B] 单肖特肖 @赔率%.0f" % SP_ODDS)
    print(f"{'策略':<12} {'P赢10期':>8} {'P赢20期':>8} {'期望/期':>9}")
    for name, (_, hs, _) in seqs.items():
        if hs is None:
            continue
        r = {}
        for L in SESSIONS:
            _, pw, _ = window_stats(hs, SP_ODDS, 1.0, L)
            r[L] = pw
        _, _, ev = window_stats(hs, SP_ODDS, 1.0, 10)
        print(f"{name:<12} {r[10]:>8.1%} {r[20]:>8.1%} {ev:>+9.2%}")
    p10, ev = analytic_random_single(10, SP_ODDS, "sp")
    p20, _ = analytic_random_single(20, SP_ODDS, "sp")
    print(f"{'随机(解析)':<12} {p10:>8.1%} {p20:>8.1%} {ev:>+9.2%}")

    # --- 表C：双肖全开 ---
    print(f"\n[表C] 双肖全开（每期2注×1单位）@赔率{OPEN_ODDS}")
    print(f"{'策略':<12} {'P赢10期':>8} {'P赢20期':>8} {'期望/期':>9}")
    for name, (_, _, h2) in seqs.items():
        if h2 is None:
            continue
        r = {}
        for L in SESSIONS:
            _, pw, _ = window_stats(h2, OPEN_ODDS, 2.0, L)
            r[L] = pw
        _, _, ev = window_stats(h2, OPEN_ODDS, 2.0, 10)
        print(f"{name:<12} {r[10]:>8.1%} {r[20]:>8.1%} {ev:>+9.2%}")

    # --- 表D：马丁格尔（单肖全开） ---
    print(f"\n[表D] 马丁格尔（赢回1单位/输翻倍/赢后重置，全开@{OPEN_ODDS}）")
    print(f"{'策略':<12} {'P赢10期':>8} {'P赢20期':>8} "
          f"{'期望10期':>9} {'P(20期亏≥100)':>13} {'最差20期':>9}")
    for name, (ho, _, _) in seqs.items():
        if ho is None:
            continue
        st = martingale_stats(ho, OPEN_ODDS)
        if 10 not in st or 20 not in st:
            continue
        w10, e10, _, _ = st[10]
        w20, _, bl20, worst20 = st[20]
        print(f"{name:<12} {w10:>8.1%} {w20:>8.1%} {e10/10:>+9.2%} "
              f"{bl20:>13.1%} {worst20:>9.0f}")

    print()


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) >= 2:
        run(args[0], args[1])
    if len(args) >= 4:
        run(args[2], args[3])
