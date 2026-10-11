# -*- coding: utf-8 -*-
"""
bb_engine_bankroll.py

用 bb_complete_v6 主引擎逐期回放，每期产出一个"一肖"推荐，
外加用户自述的"除12余数"规律家族，按真实赔率做资金模拟：
  4号肖 2.06 ｜ 马 1.76 ｜ 本金200 ｜ 每注5 平注

回答三个问题：
  1) 随机打10期/20期，赢面（结束本金>200）是多少
  2) 多少期会把200元输完
  3) 打多少期赢得最多（EV符号 + 期望余额随期数）

策略：
  E1 引擎全开选肖   argmax P(该肖出现在7个开奖号中)（由 position_probabilities 精确聚合）
  E2 引擎策略榜     argmax strategy_zodiac_probability（3.3σ门槛激活的候选家族投票）
  U1 平1余数        上期第1个平码所在肖
  U2 平1余数+1      该肖在12肖循环中的下一个
  U3 平1余数-1      该肖的前一个
  U4 特+1           上期特码+1所在肖（49→1回绕）
  U5 特-1扩大12     上期特码-1，不足1时+12（用户口径"不够12就扩大12"）
  R  随机对照

用法：python bb_engine_bankroll.py [data/bb_live_history.csv]
"""

import sys

import numpy as np

import bb_complete_v6 as eng

CSV_DEFAULT = "data/bb_live_history.csv"
BANKROLL = 200.0
STAKE = 5.0
ODDS4, ODDS_MA = 2.06, 1.76
WARMUP = 50
N_PATHS = 60_000
CAP = 5_000
SEED = 42
SESSIONS = (10, 20)
MILESTONES = (10, 20, 50, 100, 200, 500)


# ---------- 引擎选肖 ----------
def pick_by_inclusion(pred):
    """E1：最可能出现于7个开奖号的肖。"""
    probs = pred["position_probabilities"]
    best_z, best_p = None, -1.0
    for z in eng.ZODIACS:
        p_miss = 1.0
        for n in eng.zodiac_numbers(z):
            for pos in range(7):
                p_miss *= 1.0 - float(probs[pos][n - 1])
        p_hit = 1.0 - p_miss
        if p_hit > best_p:
            best_z, best_p = z, p_hit
    return best_z


def pick_by_strategy_board(pred):
    """E2：策略榜融合分布的最高肖（无激活候选时为号码数基线）。"""
    zp = pred["strategy_zodiac_probability"]
    return eng.ZODIACS[int(np.argmax(zp))]


# ---------- 用户"除12余数"规律 ----------
def zodiac_cycle_step(z, delta):
    i = eng.ZODIACS.index(z)
    return eng.ZODIACS[(i + delta) % len(eng.ZODIACS)]


def user_rules(prev):
    """上期开奖 → (U1..U5) 推荐肖。"""
    p1 = prev.regulars[0]
    sp = prev.special
    u1 = eng.get_zodiac(p1)
    u2 = zodiac_cycle_step(u1, +1)
    u3 = zodiac_cycle_step(u1, -1)
    u4 = eng.get_zodiac(eng.wrap49(sp + 1))
    sp_m = sp - 1
    if sp_m < 1:
        sp_m += 12          # "不够12的就扩大12"
    u5 = eng.get_zodiac(sp_m)
    return u1, u2, u3, u4, u5


# ---------- 回放 ----------
def replay_engine(draws):
    """逐期回放主引擎，返回 E1/E2 在每期(0..n-1)的推荐肖。"""
    engine = eng.BBEngine()
    e1 = [None] * len(draws)
    e2 = [None] * len(draws)
    engine.process_draw(draws[0])
    for t in range(1, len(draws)):
        pred = engine.predict()
        e1[t] = pick_by_inclusion(pred)
        e2[t] = pick_by_strategy_board(pred)
        engine.process_draw(draws[t])
    return e1, e2


def build_nets(draws, picks):
    """推荐序列 → 每期净利数组（-5 或 +5.3/+3.8）。"""
    nets = np.zeros(len(draws), dtype=np.float64)
    for t, z in enumerate(picks):
        if z is None:
            nets[t] = 0.0
            continue
        if t == 0:
            nets[t] = 0.0
            continue
        hit = bool(set(eng.zodiac_numbers(z)) & set(draws[t].numbers))
        if hit:
            nets[t] = STAKE * ((ODDS_MA if z == "马" else ODDS4) - 1.0)
        else:
            nets[t] = -STAKE
    return nets


def session_stats(nets, L, start):
    """滑动窗口 P(净利>0) 与期望净利，窗口只取 start 之后。"""
    s = nets[start:]
    if len(s) < L:
        return float("nan"), float("nan")
    pref = np.concatenate([[0.0], np.cumsum(s)])
    total = pref[L:] - pref[:-L]
    return float(np.mean(total > 0)), float(np.mean(total))


def bust_mc(nets, start, seed, n_paths=N_PATHS, cap=CAP):
    """按经验结局重采样的爆仓模拟（从 start 期起打）。"""
    rng = np.random.default_rng(seed)
    pool = nets[start:]
    pool = pool[pool != 0.0]
    if pool.size == 0:
        return np.full(n_paths, -1, dtype=np.int32)
    bal = np.full(n_paths, BANKROLL, dtype=np.float64)
    alive = np.ones(n_paths, dtype=bool)
    ruin_t = np.full(n_paths, -1, dtype=np.int32)
    period = 0
    while alive.any() and period < cap:
        period += 1
        idx = np.flatnonzero(alive)
        bal[idx] += pool[rng.integers(0, pool.size, size=idx.size)]
        broke = bal[idx] < STAKE
        if broke.any():
            ruin_t[idx[broke]] = period
            alive[idx[broke]] = False
    return ruin_t


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else CSV_DEFAULT
    draws = eng.load_csv(path)
    draws.sort(key=eng.issue_sort_key)
    n = len(draws)

    print("=" * 96)
    print(f"回放【{path}】共{n}期 ｜ 评估 {WARMUP}..{n-1} 共 {n - WARMUP} 期 "
          f"｜ 本金200 每注5 ｜ 4号肖@2.06 马@1.76")

    # ---- 推荐序列 ----
    e1, e2 = replay_engine(draws)
    picks_map = {"E1 引擎全开": e1, "E2 引擎策略榜": e2}
    rng_pick = np.random.default_rng(SEED)
    picks_map["R 随机对照"] = [
        None if t == 0 else eng.ZODIACS[i]
        for t, i in enumerate(rng_pick.integers(0, len(eng.ZODIACS), n))
    ]
    for t in range(1, n):
        u1, u2, u3, u4, u5 = user_rules(draws[t - 1])
        for key, z in zip(("U1 平1余数", "U2 平1+1", "U3 平1-1", "U4 特+1", "U5 特-1扩12"),
                          (u1, u2, u3, u4, u5)):
            picks_map.setdefault(key, [None] * n)[t] = z

    # ---- 表1：命中率 / EV / 10期20期赢面 ----
    print(f"\n{'策略':<14} {'命中率':>7} {'选马':>6} {'EV/期':>8} "
          f"{'赢10期':>8} {'赢20期':>8} {'期望10期':>9} {'期望20期':>9}")
    nets_map = {}
    for name, picks in picks_map.items():
        nets = build_nets(draws, picks)
        nets_map[name] = nets
        s = slice(WARMUP, n)
        seg = nets[s]
        seg_picks = picks[s]
        played = seg != 0.0
        if played.sum() == 0:
            continue
        hit_net = seg[played]
        hits = hit_net > 0
        ma = sum(1 for z in seg_picks if z == "马")
        ev = float(np.mean(hit_net))
        p10, e10 = session_stats(nets, 10, WARMUP)
        p20, e20 = session_stats(nets, 20, WARMUP)
        hit_rate = float(np.mean(hits))
        print(f"{name:<14} {hit_rate:>7.2%} {ma:>6} {ev:>+8.3f} "
              f"{p10:>8.2%} {p20:>8.2%} {e10:>+9.2f} {e20:>+9.2f}")

    # ---- 表2：爆仓 ----
    print(f"\n{'策略':<14} {'中位爆仓期':>10} {'P(≤200期)':>10} "
          f"{'P(≤600期)':>10} {'P(≤1000期)':>11} {'P(≤5000期)':>11}")
    for i, (name, nets) in enumerate(nets_map.items()):
        ruin_t = bust_mc(nets, WARMUP, SEED + i)
        busted = ruin_t[ruin_t != -1]
        med = float(np.median(busted)) if busted.size else float("nan")
        print(f"{name:<14} {med:>10.0f} "
              f"{np.mean((ruin_t != -1) & (ruin_t <= 200)):>10.2%} "
              f"{np.mean((ruin_t != -1) & (ruin_t <= 600)):>10.2%} "
              f"{np.mean((ruin_t != -1) & (ruin_t <= 1000)):>11.2%} "
              f"{np.mean(ruin_t != -1):>11.2%}")

    # ---- 表3：期望余额随期数（赢得最多） ----
    print("\n[期望余额随期数] EV>0 打越久期望越高；EV<0 第0期不打赢得最多")
    print(f"{'期数':>6} " + " ".join(f"{name.split()[0]:>8}" for name in nets_map))
    for L in MILESTONES:
        row = []
        for nets in nets_map.values():
            seg = nets[WARMUP:WARMUP + L]
            played = seg[seg != 0.0]
            row.append(played.sum())
        print(f"{L:>6} " + " ".join(f"{v:>+8.2f}" for v in row))


if __name__ == "__main__":
    main()
