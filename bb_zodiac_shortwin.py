# -*- coding: utf-8 -*-
"""
bb_zodiac_shortwin.py

BB短窗生肖策略检验（全开一肖 + 特肖 双口径）。

玩法口径（按BB规则）：
- 全开一肖：本期7个号码含所选生肖任一号码即中
  4号肖 p = 1 - C(45,7)/C(49,7) = 47.17%；5号肖(马) p = 55.39%
- 特肖：特码落所选生肖即中
  4号肖 p = 4/49 = 8.16%；5号肖 p = 5/49 = 10.20%

待测策略（每期必选1肖，信号只用t期之前的数据）：
  S1 热10    最近10期出现次数最多
  S2 热15    最近15期出现次数最多
  S3 冷30    最近30期出现最少(平手取缺席最久)
  S4 冷50    最近50期出现最少(平手取缺席最久)
  S5 惯性    最近3期出现>=2的热肖(无则回退S3)
  S6 六冲    上期特肖的地支对冲肖
  S7 三合    上期特肖的三合局成员中最冷者(30期)
  S8 家野    最近15期占优类(家禽/野兽)中最冷30期的肖

统计：命中数/基线(按所选肖号码数调整)/正态z/单尾p/EV(多档赔率)
Bonferroni: 检验总数 = 策略数 × 2口径

用法：python bb_zodiac_shortwin.py data/bb_live_history.csv "标签"
"""

import sys
import math
from collections import Counter

from scipy.stats import binom

import bb_zodiac_v7 as bb

WARMUP = 50

CHONG = {
    "鼠": "马", "马": "鼠", "牛": "羊", "羊": "牛",
    "虎": "猴", "猴": "虎", "兔": "鸡", "鸡": "兔",
    "龙": "狗", "狗": "龙", "蛇": "猪", "猪": "蛇",
}
SANHE = {
    "猴": "鼠龙", "鼠": "猴龙", "龙": "猴鼠",
    "虎": "马狗", "马": "虎狗", "狗": "虎马",
    "蛇": "鸡牛", "鸡": "蛇牛", "牛": "蛇鸡",
    "猪": "兔羊", "兔": "猪羊", "羊": "猪兔",
}
JIA = {"牛", "马", "羊", "鸡", "狗", "猪"}
YE = {"鼠", "虎", "兔", "龙", "蛇", "猴"}

# BB全开一肖/特肖理论概率
P_OPEN = {}
P_SPECIAL = {}
for z in bb.ZODIACS:
    k = len(bb.zodiac_numbers(z))
    # 1 - C(49-k,7)/C(49,7)
    p = 1.0
    for i in range(7):
        p *= (49 - k - i) / (49 - i)
    P_OPEN[z] = 1.0 - p
    P_SPECIAL[z] = k / 49.0


def appearances(draws, zodiac, start, end):
    """draws[start:end] 中含该生肖的期数。"""
    nums = set(bb.zodiac_numbers(zodiac))
    return sum(1 for d in draws[start:end] if nums & set(d.numbers))


def absence(draws, zodiac, t):
    """截至t期(不含t)，该生肖已连续缺席期数。"""
    nums = set(bb.zodiac_numbers(zodiac))
    for i in range(t - 1, -1, -1):
        if nums & set(draws[i].numbers):
            return t - 1 - i
    return t


def pick_hot(draws, t, window):
    scores = {z: appearances(draws, z, max(0, t - window), t) for z in bb.ZODIACS}
    best = max(scores.values())
    cands = [z for z in bb.ZODIACS if scores[z] == best]
    return cands[0]


def pick_cold(draws, t, window):
    scores = {z: appearances(draws, z, max(0, t - window), t) for z in bb.ZODIACS}
    best = min(scores.values())
    cands = [z for z in bb.ZODIACS if scores[z] == best]
    # 平手：缺席最久优先，再按固定生肖序
    cands.sort(key=lambda z: (-absence(draws, z, t), bb.ZODIACS.index(z)))
    return cands[0]


def pick_momentum(draws, t):
    hot = [z for z in bb.ZODIACS if appearances(draws, z, t - 3, t) >= 2]
    if hot:
        hot.sort(key=lambda z: (-absence(draws, z, t), bb.ZODIACS.index(z)))
        return hot[0]
    return pick_cold(draws, t, 30)


def pick_chong(draws, t):
    prev_zodiac = bb.get_zodiac(draws[t - 1].special)
    return CHONG[prev_zodiac]


def pick_sanhe(draws, t):
    prev_zodiac = bb.get_zodiac(draws[t - 1].special)
    cands = list(SANHE[prev_zodiac])
    cands.sort(key=lambda z: (appearances(draws, z, t - 30, t), absence(draws, z, t)))
    return cands[0]


def pick_jia_ye(draws, t):
    start = max(0, t - 15)
    jia_set, ye_set = set(), set()
    for d in draws[start:t]:
        for n in d.numbers:
            z = bb.get_zodiac(n)
            (jia_set if z in JIA else ye_set).add(z)
    dominant = JIA if len(jia_set) >= len(ye_set) else YE
    cands = sorted(
        dominant,
        key=lambda z: (appearances(draws, z, t - 30, t), absence(draws, z, t)),
    )
    return cands[0]


STRATEGIES = [
    ("S1 热10",        lambda d, t: pick_hot(d, t, 10)),
    ("S2 热15",        lambda d, t: pick_hot(d, t, 15)),
    ("S3 冷30回补",    lambda d, t: pick_cold(d, t, 30)),
    ("S4 冷50回补",    lambda d, t: pick_cold(d, t, 50)),
    ("S5 热惯性3期",   pick_momentum),
    ("S6 六冲",        pick_chong),
    ("S7 三合",        pick_sanhe),
    ("S8 家野对冲",    pick_jia_ye),
]


def evaluate(csv_path, label):
    draws = bb.load_csv(csv_path)
    draws.sort(key=bb.issue_sort_key)
    n = len(draws)
    eval_count = n - WARMUP
    n_tests = len(STRATEGIES) * 2

    print("=" * 78)
    print(f"【{label}】{n} 期 ｜ 评估 {WARMUP}..{n-1} 共 {eval_count} 期 ｜ "
          f"全开基线 47.17%/55.39% ｜ 特肖基线 8.16%/10.20%")
    print(f"Bonferroni显著阈值 α = 0.05/{n_tests} = {0.05 / n_tests:.4f}")

    results = []
    for name, picker in STRATEGIES:
        open_hits = sp_hits = 0
        open_base = sp_base = 0.0
        picks = Counter()
        for t in range(WARMUP, n):
            z = picker(draws, t)
            picks[z] += 1
            nums = set(bb.zodiac_numbers(z))
            draw_nums = set(draws[t].numbers)
            hit_open = int(bool(nums & draw_nums))
            hit_sp = int(draws[t].special in nums)
            open_hits += hit_open
            sp_hits += hit_sp
            open_base += P_OPEN[z]
            sp_base += P_SPECIAL[z]

        open_rate = open_hits / eval_count
        sp_rate = sp_hits / eval_count
        open_p_base = open_base / eval_count
        sp_p_base = sp_base / eval_count

        def z_and_p(hits, p):
            se = math.sqrt(max(1e-9, eval_count * p * (1 - p)))
            z_score = (hits - eval_count * p) / se
            p_val = float(binom.sf(hits - 1, eval_count, p))  # 单尾：优于基线
            return z_score, p_val

        oz, op = z_and_p(open_hits, open_p_base)
        sz, sp = z_and_p(sp_hits, sp_p_base)
        results.append((name, open_hits, open_rate, open_p_base, oz, op,
                        sp_hits, sp_rate, sp_p_base, sz, sp, picks))

        flag_o = " ★" if op < 0.05 / n_tests else (" *" if op < 0.05 else "")
        flag_s = " ★" if sp < 0.05 / n_tests else (" *" if sp < 0.05 else "")
        print(f"{name:<12} 全开 {open_hits:>3}/{eval_count} = {open_rate:6.2%} "
              f"(基线{open_p_base:6.2%}, z={oz:+.2f}, p={op:.3f}){flag_o}")
        print(f"{'':<12} 特肖 {sp_hits:>3}/{eval_count} = {sp_rate:6.2%} "
              f"(基线{sp_p_base:6.2%}, z={sz:+.2f}, p={sp:.3f}){flag_s}")

    # 最优策略的EV表
    best = max(results, key=lambda r: r[4])
    name, oh, orate, obase, _, _, sh, srate, sbase, _, _, _ = best
    print(f"\n[EV参考] 最优全开策略: {name} 估计命中 {orate:.2%} (基线{obase:.2%})")
    for odds in (1.8, 1.9, 2.0, 2.1, 2.2):
        ev_base = obase * odds - 1
        ev_est = orate * odds - 1
        print(f"  赔率{odds:.1f}: 基线EV {ev_base:+.1%} ｜ 该策略估计EV {ev_est:+.1%}")
    best_sp = max(results, key=lambda r: r[9])
    print(f"[EV参考] 最优特肖策略: {best_sp[0]} 估计命中 {best_sp[7]:.2%} "
          f"(基线{best_sp[8]:.2%})")
    for odds in (10, 11, 12):
        ev_base = best_sp[8] * odds - 1
        ev_est = best_sp[7] * odds - 1
        print(f"  赔率{odds}: 基线EV {ev_base:+.1%} ｜ 该策略估计EV {ev_est:+.1%}")

    top_pick = results[0][11].most_common(3)
    print(f"\nS1热10选肖分布 top3: {top_pick}")


if __name__ == "__main__":
    evaluate(sys.argv[1], sys.argv[2])
