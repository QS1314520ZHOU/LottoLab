# -*- coding: utf-8 -*-
"""
bb_bankroll_sim.py

真实赔率下的资金会话模拟：
  全开一肖 4号肖 赔率2.06（保本命中率48.54%）
  马(5号肖) 赔率1.76（保本命中率56.82%）
  本金200元，每注5元，平注，随机选肖

回答三个问题：
  1. 随机打10期/20期，赢面（会话结束时本金>200）是多少
  2. 多少期会把200元输完（爆仓时间分布）
  3. 打多少期赢得最多（期望余额 / 期望历史最高余额 / 爆仓率 随期数变化）

附：马丁格尔（5/10/20/40/80/160）同赔率下的赢面与爆仓率。

精确部分用多项式/二项分布；资金曲线用蒙特卡洛（含爆仓截断）。
用法：python bb_bankroll_sim.py
"""

import math

import numpy as np
from scipy.stats import binom

# ---------- 参数 ----------
BANKROLL = 200.0
STAKE = 5.0
ODDS4, ODDS_MA = 2.06, 1.76

C49_7 = math.comb(49, 7)
P4 = 1.0 - math.comb(45, 7) / C49_7    # 4号肖全开命中率
P_MA = 1.0 - math.comb(44, 7) / C49_7   # 马全开命中率
PROFIT4 = STAKE * (ODDS4 - 1.0)         # +5.30
PROFIT_MA = STAKE * (ODDS_MA - 1.0)     # +3.80
LOSS = STAKE                            # -5.00

P_PICK_MA = 1.0 / 12.0
# 随机选肖时每期三种结果的概率
A_WIN4 = (1.0 - P_PICK_MA) * P4         # 中4号肖
B_WINMA = P_PICK_MA * P_MA              # 中马
C_LOSS = 1.0 - A_WIN4 - B_WINMA         # 输

EV_PERIOD = A_WIN4 * PROFIT4 + B_WINMA * PROFIT_MA - C_LOSS * LOSS

MILESTONES = [10, 20, 30, 50, 100, 200, 500]
SEED = 42


# ---------- 精确计算：会话赢面（不考虑爆仓，仅期数<=30时安全） ----------
def exact_random_session(L):
    """随机选肖，L期平注，净利分布的 P(赢)/P(平)/E[净利]。"""
    p_win = p_be = ev = 0.0
    for w4 in range(L + 1):
        rest = L - w4
        for wma in range(rest + 1):
            wl = rest - wma
            prob = (math.comb(L, w4) * math.comb(rest, wma)
                    * (A_WIN4 ** w4) * (B_WINMA ** wma) * (C_LOSS ** wl))
            net = PROFIT4 * w4 + PROFIT_MA * wma - LOSS * wl
            ev += prob * net
            if net > 0:
                p_win += prob
            elif abs(net) < 1e-9:
                p_be += prob
    return p_win, p_be, ev


def exact_fixed_session(L, win_prob, profit):
    """固定押某一类肖：net = profit*k - LOSS*(L-k) > 0 => k >= floor+1。"""
    k = math.floor(STAKE * L / (profit + STAKE)) + 1
    if k > L:
        return 0.0
    return float(binom.sf(k - 1, L, win_prob))


# ---------- 蒙特卡洛：平注资金曲线（含爆仓） ----------
def simulate_flat(n_paths, seed):
    rng = np.random.default_rng(seed)
    bal = np.full(n_paths, BANKROLL)
    peak = bal.copy()
    ruin_t = np.full(n_paths, -1, dtype=np.int64)
    alive = np.arange(n_paths)
    mil_set = set(MILESTONES)
    snap = {}
    period = 0

    while alive.size and period < 30000:
        period += 1
        x = alive
        is_ma = rng.random(x.size) < P_PICK_MA
        win = rng.random(x.size) < np.where(is_ma, P_MA, P4)
        profit = np.where(win,
                          np.where(is_ma, PROFIT_MA, PROFIT4),
                          -LOSS)
        bal[x] += profit
        np.maximum(peak, bal, out=peak)
        broke = bal[x] < STAKE
        if broke.any():
            ruin_t[x[broke]] = period
        alive = x[~broke]
        if period in mil_set:
            snap[period] = (bal.copy(), peak.copy(), ruin_t.copy())

    return bal, peak, ruin_t, snap, alive.size, period


# ---------- 蒙特卡洛：马丁格尔 ----------
def simulate_martingale(n_paths, L, seed):
    """赢/输后重置为5，输后翻倍（5,10,20,40,80,160）；
    下一手超过余额即爆仓（该场结束，余额保留）。"""
    rng = np.random.default_rng(seed)
    bal = np.full(n_paths, BANKROLL)
    stake = np.full(n_paths, STAKE)
    busted = np.zeros(n_paths, dtype=bool)
    bust_t = np.full(n_paths, -1, dtype=np.int64)

    for t in range(1, L + 1):
        need = ~busted
        can_bet = need & (bal >= stake)
        busted[need & ~can_bet] = True
        bust_t[need & ~can_bet] = t
        x = np.where(can_bet)[0]
        if x.size == 0:
            break
        is_ma = rng.random(x.size) < P_PICK_MA
        win = rng.random(x.size) < np.where(is_ma, P_MA, P4)
        odds = np.where(is_ma, ODDS_MA, ODDS4)
        bal[x] += np.where(win, stake[x] * (odds - 1.0), -stake[x])
        stake[x] = np.where(win, STAKE, stake[x] * 2.0)

    net = bal - BANKROLL
    return net, busted, bust_t


def main():
    print("=" * 84)
    print(f"底账：本金{BANKROLL:.0f} ｜ 每注{STAKE:.0f} ｜ 4号肖@{ODDS4} "
          f"命中{P4:.4f}（保本命中率{1/ODDS4:.4f}）")
    print(f"      马@{ODDS_MA} 命中{P_MA:.4f}（保本命中率{1/ODDS_MA:.4f}）")
    print(f"      随机选肖每期期望 = {EV_PERIOD:+.4f} 元"
          f"（10期{10*EV_PERIOD:+.2f}，20期{20*EV_PERIOD:+.2f}）")

    # ---- 表1：精确会话赢面 ----
    print("\n[表1] 精确会话赢面（P=会话结束本金>200，不含爆仓风险）")
    print(f"{'口径':<14} {'P赢10期':>9} {'P赢20期':>9} "
          f"{'期望10期':>9} {'期望20期':>9}")
    p10, _, e10 = exact_random_session(10)
    p20, _, e20 = exact_random_session(20)
    print(f"{'随机肖':<14} {p10:>9.2%} {p20:>9.2%} "
          f"{e10:>+9.2f} {e20:>+9.2f}")
    for name, p, odds in (("只押马@1.76", P_MA, ODDS_MA),
                          ("只押4号肖@2.06", P4, ODDS4)):
        pp10 = exact_fixed_session(10, p, p and (odds - 1) * STAKE)
        pp20 = exact_fixed_session(20, p, (odds - 1) * STAKE)
        ev10 = STAKE * 10 * (p * odds - 1)
        ev20 = STAKE * 20 * (p * odds - 1)
        print(f"{name:<14} {pp10:>9.2%} {pp20:>9.2%} "
              f"{ev10:>+9.2f} {ev20:>+9.2f}")

    # ---- 表2：平注资金曲线（MC，含爆仓） ----
    print("\n[表2] 平注随机肖资金曲线（蒙特卡洛40万条，含爆仓截断）")
    N = 400_000
    bal, peak, ruin_t, snap, n_capped, last_period = simulate_flat(N, SEED)

    print(f"{'期数':>6} {'赢面':>8} {'期望余额':>10} {'期望最高曾到':>12} "
          f"{'P(曾+50以上)':>12} {'爆仓率':>8}")
    for L in MILESTONES:
        if L not in snap:
            continue
        b, pk, rt = snap[L]
        win_rate = np.mean(b > BANKROLL)
        e_bal = np.mean(b)
        e_peak = np.mean(pk)
        p_peak50 = np.mean(pk >= BANKROLL + 50)
        p_ruin = np.mean(rt != -1)
        print(f"{L:>6} {win_rate:>8.2%} {e_bal - BANKROLL:>+10.2f} "
              f"{e_peak - BANKROLL:>+12.2f} {p_peak50:>12.2%} {p_ruin:>8.2%}")

    # ---- 表3：爆仓时间 ----
    print("\n[表3] 多少期输完200元（爆仓=余额<5元无法继续）")
    ruined = ruin_t[ruin_t != -1]
    if n_capped:
        print(f"  （{n_capped}条到{last_period}期上限仍未爆仓，按右删失）")
    qs = np.percentile(ruined, [10, 25, 50, 75, 90])
    print(f"  平均 {ruined.mean():.0f} 期 ｜ 中位数 {qs[2]:.0f} 期 "
          f"｜ P10 {qs[0]:.0f} / P25 {qs[1]:.0f} / P75 {qs[3]:.0f} / P90 {qs[4]:.0f}")
    for cap, label in ((20, "1天(20期)"), (100, "5天(100期)"),
                       (200, "10天(200期)"), (600, "30天(600期)"),
                       (1000, "50天(1000期)")):
        print(f"  P(≤{label}) = {np.mean(ruined <= cap):.2%}")

    # ---- 表4：马丁格尔 ----
    print("\n[表4] 马丁格尔 5→10→20→40→80→160（赢重置/输翻倍/余额不足即爆仓）")
    print(f"{'场次':>6} {'赢面':>8} {'爆仓率':>8} {'期望净利':>10} {'最差一场':>10}")
    for L in (10, 20):
        net, busted, bust_t = simulate_martingale(400_000, L, SEED + L)
        print(f"{L:>6} {np.mean(net > 0):>8.2%} {np.mean(busted):>8.2%} "
              f"{net.mean():>+10.2f} {net.min():>10.0f}")


if __name__ == "__main__":
    main()
