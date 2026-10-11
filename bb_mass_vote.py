# -*- coding: utf-8 -*-
"""
bb_mass_vote.py

大规模公式变体投票筛选实验（约1.13亿种变体）。

变体族：f(x, y) = (a*x + b*y + c) mod 49 + 1
  x = 位置s1在滞后l1期的开奖号码，y = 位置s2在滞后l2期的开奖号码
  s1, s2 ∈ {平1..平6, 特码}（7个），l1, l2 ∈ 1..7
  a ∈ 1..20，b ∈ ±1..±24，c ∈ 0..48
  总数 = 7^4 * 20 * 48 * 49 = 112,943,040

流程（严格时间顺序，无未来信息泄漏）：
  1. 前30期预热剔除
  2. 训练段120期：每个变体统计特码命中数，低于阈值淘汰（去掉不利因素）
  3. 测试段49期：幸存变体一人一票，多数票产生组合，与实际开奖对比
  4. 同一流程在纯随机数据上对照运行，检验筛选出的"超级算法"是否只是选择偏差

用法：
    python bb_mass_vote.py data/bb_live_history.csv "真实数据"
"""

import sys
import math

import numpy as np
from scipy.stats import binom

import bb_zodiac_v7 as bb

WARMUP = 30
TRAIN_START, TRAIN_END = 30, 150   # [30, 150) = 120期
TEST_START, TEST_END = 150, 199    # [150, 199) = 49期

A_VALS = np.arange(1, 21)              # 20
B_VALS = np.concatenate([np.arange(1, 25), np.arange(-24, 0)])  # 48
C_VALS = np.arange(49)                 # 49
LAGS = range(1, 8)                     # 7
N_POSITIONS = 7

N_VARIANTS = (7 ** 4) * len(A_VALS) * len(B_VALS) * len(C_VALS)
HIT_SHAPE = (7, 7, N_POSITIONS, N_POSITIONS,
             len(A_VALS), len(B_VALS), len(C_VALS))

VOTE_THRESHOLD = 8  # 训练段120期命中>=8才幸存（随机基线期望约2.45）


def shifted(nums: np.ndarray, lag: int) -> np.ndarray:
    """X[t] = nums[t-lag]，前lag期填0（不参与评估，评估段t>=30）。"""
    out = np.zeros_like(nums)
    out[lag:] = nums[:-lag]
    return out


def build_ab(nums: np.ndarray) -> tuple:
    """预计算 (系数*号码)%49，返回 A[lag][a] -> [T,7] uint8，B[lag][b] -> [T,7]。"""
    a_tables = []
    for lag in LAGS:
        x = shifted(nums, lag).astype(np.uint16)
        a_tables.append(
            np.stack([(int(a) * x % 49).astype(np.uint8) for a in A_VALS])
        )
    b_tables = []
    for lag in LAGS:
        y = shifted(nums, lag).astype(np.int16)
        b_tables.append(
            np.stack([(int(b) * y % 49).astype(np.uint8) for b in B_VALS])
        )
    return a_tables, b_tables


def accumulate_hits(nums: np.ndarray, special: np.ndarray):
    a_tables, b_tables = build_ab(nums)
    hits_train = np.zeros(HIT_SHAPE, dtype=np.uint8)
    hits_test = np.zeros(HIT_SHAPE, dtype=np.uint8)
    train_idx = np.arange(TRAIN_START, TRAIN_END)
    test_idx = np.arange(TEST_START, TEST_END)
    train_target = special[train_idx].astype(np.int16) - 1
    test_target = special[test_idx].astype(np.int16) - 1

    for li, l1 in enumerate(LAGS):
        for lj, l2 in enumerate(LAGS):
            b_tab = b_tables[lj]
            for s1 in range(N_POSITIONS):
                a_slice = a_tables[li][:, :, s1]          # [a, T]
                for s2 in range(N_POSITIONS):
                    b_slice = b_tab[:, :, s2]             # [b, T]
                    # full[a, t, b, c] = (a*x + b*y + c) % 49
                    full = (
                        a_slice[:, :, None, None]
                        + b_slice.T[None, :, :, None]
                        + C_VALS[None, None, None, :]
                    )
                    full %= 49
                    mask = full[:, TRAIN_START:TRAIN_END] == train_target[None, :, None, None]
                    hits_train[li, lj, s1, s2] = mask.sum(axis=1, dtype=np.uint8)
                    mask = full[:, TEST_START:TEST_END] == test_target[None, :, None, None]
                    hits_test[li, lj, s1, s2] = mask.sum(axis=1, dtype=np.uint8)

    return hits_train, hits_test, a_tables, b_tables


def decode(flat_index: int) -> str:
    li, lj, s1, s2, ai, bi, ci = np.unravel_index(flat_index, HIT_SHAPE)
    return (
        f"平{s1 + 1 if s1 < 6 else '特'}(t-{li + 1}) * {A_VALS[ai]}"
        f" {'+' if B_VALS[bi] > 0 else '-'} {abs(B_VALS[bi])} * "
        f"平{s2 + 1 if s2 < 6 else '特'}(t-{lj + 1}) + {ci} (mod 49)"
    )


def run_experiment(csv_path: str, label: str) -> dict:
    draws = bb.load_csv(csv_path)
    draws.sort(key=bb.issue_sort_key)
    nums = np.array([d.numbers for d in draws], dtype=np.uint8)
    special = nums[:, 6]
    n_train = TRAIN_END - TRAIN_START
    n_test = TEST_END - TEST_START
    baseline = 1.0 / 49.0

    print("=" * 76)
    print(f"【{label}】共 {len(draws)} 期 ｜ 变体总数 {N_VARIANTS:,} ({N_VARIANTS / 1e8:.2f}亿)")
    print(f"预热 {WARMUP} 期 ｜ 训练段 {n_train} 期 [{TRAIN_START},{TRAIN_END}) "
          f"｜ 测试段 {n_test} 期 [{TEST_START},{TEST_END}) ｜ 单期随机基线 {baseline:.2%}")

    hits_train, hits_test, a_tables, b_tables = accumulate_hits(nums, special)

    # 自检：全体变体的平均命中率必须≈1/49，否则统计管道有bug
    overall_rate = float(hits_train.sum()) / (N_VARIANTS * n_train)
    if abs(overall_rate - baseline) > 0.002:
        raise AssertionError(
            f"命中率自检失败: {overall_rate:.5f} vs 理论 {baseline:.5f}"
        )

    # --- 训练段筛选：各阈值下的幸存者数量 ---
    print("\n[训练段筛选] 随机数据下的期望幸存数 = N * P(Binom(120, 1/49) >= k)")
    survivors = hits_train >= VOTE_THRESHOLD
    for k in (6, 7, 8, 10, 12):
        count = int(np.count_nonzero(hits_train >= k))
        expected = N_VARIANTS * float(binom.sf(k - 1, n_train, baseline))
        print(f"  命中>={k:>2}: 真实/本数据幸存 {count:>10,} ｜ 纯随机期望 {expected:>10,.0f}")

    # --- 训练段Top10（选择偏差展示）---
    flat_train = hits_train.reshape(-1)
    flat_test = hits_test.reshape(-1)
    top10 = np.argpartition(flat_train, -10)[-10:]
    top10 = top10[np.argsort(-flat_train[top10])]
    print(f"\n[训练段Top10变体] 最高命中 {int(flat_train[top10[0]])}/{n_train} "
          f"= {flat_train[top10[0]] / n_train:.1%}（纯随机下1.13亿个变体的最高分期望约 "
          f"{1.0 + math.sqrt(2 * math.log(N_VARIANTS)) * math.sqrt(n_train * baseline * (1 - baseline)) + n_train * baseline:.0f}/{n_train}）")
    test_range = np.arange(TEST_START, TEST_END)
    actual_test = special[test_range]
    for idx in top10[:5]:
        print(f"  训练 {int(flat_train[idx]):>3}/{n_train} | 测试 {int(flat_test[idx]):>2}/{n_test} "
              f"| {decode(int(idx))}")

    # --- Bonferroni多重检验校正 ---
    alpha = 0.05 / N_VARIANTS
    k_star = int(binom.isf(alpha, n_train, baseline)) + 1
    print(f"\n[多重检验校正] Bonferroni阈值：训练命中>={k_star} 才能在5%水平"
          f"排除纯随机（校正后）")

    # --- 投票：幸存者对测试段一票一票 ---
    print(f"\n[投票] 阈值 train>={VOTE_THRESHOLD}，幸存 {int(np.count_nonzero(survivors)):,} 个变体")
    votes = np.zeros((n_test, 49), dtype=np.int32)
    for li in range(len(LAGS)):
        for lj in range(len(LAGS)):
            block_sel = survivors[li, lj]          # [s1, s2, a, b, c]
            if not block_sel.any():
                continue
            s1s, s2s, ais, bis, cis = np.nonzero(block_sel)
            for s1 in np.unique(s1s):
                m1 = s1s == s1
                for s2 in np.unique(s2s[m1]):
                    m2 = m1 & (s2s == s2)
                    if not m2.any():
                        continue
                    # 幸存变体在测试段的预测 [K, n_test]
                    av = a_tables[li][ais[m2]][:, test_range][:, :, s1]
                    bv = b_tables[lj][bis[m2]][:, test_range][:, :, s2]
                    cv = cis[m2][:, None]
                    preds = (av + bv + cv) % 49 + 1
                    for j in range(n_test):
                        votes[j] += np.bincount(preds[:, j], minlength=50)[1:50]

    ensemble = np.argmax(votes, axis=1) + 1
    ens_hits = int(np.sum(ensemble == actual_test))
    p = float(binom.sf(ens_hits - 1, n_test, baseline))
    print(f"  测试段投票命中：{ens_hits}/{n_test} = {ens_hits / n_test:.2%} "
          f"（基线 {baseline:.2%}，二项检验 p={p:.3f}）")

    # 生肖口径
    ens_zodiac = [bb.get_zodiac(int(v)) for v in ensemble]
    act_zodiac = [bb.get_zodiac(int(v)) for v in actual_test]
    z_hits = sum(1 for a, b_ in zip(ens_zodiac, act_zodiac) if a == b_)
    z_base = sum(len(bb.zodiac_numbers(z)) / 49.0 for z in ens_zodiac) / n_test
    print(f"  测试段一肖口径：{z_hits}/{n_test} = {z_hits / n_test:.2%}（覆盖基线 {z_base:.2%}）")

    # 平均票数（投票集中度）
    margin = np.sort(votes, axis=1)
    avg_top = margin[:, -1].mean()
    print(f"  平均最高票数 {avg_top:.0f} 票 / 幸存者 {int(np.count_nonzero(survivors)):,}")

    return {
        "ensemble": ensemble,
        "actual": actual_test,
        "votes": votes,
        "n_survivors": int(np.count_nonzero(survivors)),
    }


if __name__ == "__main__":
    run_experiment(sys.argv[1], sys.argv[2])
