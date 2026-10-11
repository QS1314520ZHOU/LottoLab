# -*- coding: utf-8 -*-
"""
bb_ml.py — 按"概率建模+历史数据分析"框架做的 BB六合彩 机器学习实验

对照三个模型（教程第七节口径）：
  模型A 公平基线      每个号码 P=7/49=1/7（常数）
  模型B 频率+贝叶斯平滑  Beta(1,6) 后验，近30期频率
  模型C 逻辑回归       历史频率/遗漏/上期特征 -> 号码出现概率
  模型D 随机森林       同特征的非线性对照

流程严格按教程：
  1) 按【完整期次】先切训练/测试，再展开每期49个号码（防泄漏）
  2) 预测目标：号码 i 是否出现在第 t 期的7个开奖号里
  3) 指标：Brier 分数 / 对数损失（与基线同测试期对比）
  4) 落到玩法：号码概率聚合为一肖概率 -> 选肖 -> 命中率/EV@2.06/1.76

用法：python bb_ml.py
"""

import math

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

CSV = "data/bb_live_history.csv"
WINDOW = 30
TRAIN_END = 175      # 训练期次 [WINDOW, TRAIN_END)，测试 [TRAIN_END, n)——按期切分

MOD2Z = {1: "马", 2: "蛇", 3: "龙", 4: "兔", 5: "虎", 6: "牛",
         7: "鼠", 8: "猪", 9: "狗", 10: "鸡", 11: "猴", 0: "羊"}
ZODIACS = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]
ODDS4, ODDS_MA = 2.06, 1.76


def zodiac(n):
    return MOD2Z[n % 12]


def zsize(z):
    return 5 if z == "马" else 4


def load_matrix(path=CSV):
    df = pd.read_csv(path).sort_values("issue").reset_index(drop=True)
    cols = [f"p{i}" for i in range(1, 7)] + ["sp"]
    assert ((df[cols] >= 1) & (df[cols] <= 49)).all().all(), "号码越界"
    n = len(df)
    mat = np.zeros((n, 49), dtype=np.int8)
    for t in range(n):
        mat[t, df.loc[t, cols].astype(int).to_numpy() - 1] = 1
    return df, mat


def build_features(mat, t, window=WINDOW):
    """第t期开赛前可知的特征（只用 [t-window, t)）。返回 (49,4)。"""
    hist = mat[t - window:t]
    freq = hist.sum(axis=0) / window
    freq10 = mat[t - 10:t].sum(axis=0) / 10.0
    last = mat[t - 1].astype(float)
    # 遗漏：距上次出现的期数（封顶30）
    omit = np.full(49, window, dtype=float)
    for i in range(49):
        idx = np.nonzero(hist[:, i])[0]
        if idx.size:
            omit[i] = window - 1 - idx[-1]
    omit = omit / window
    return np.stack([freq, freq10, last, omit], axis=1)


def bayesian_probs(mat, t, window=WINDOW, a=1.0, b=6.0):
    """模型B：Beta(a,b) 后验均值，先验均值=1/7。"""
    s = mat[t - window:t].sum(axis=0)
    return (s + a) / (window + a + b)


def main():
    df, mat = load_matrix()
    n = len(df)
    print("=" * 78)
    print(f"数据 {n}期 ｜ 训练期次 [{WINDOW},{TRAIN_END}) 测试 [{TRAIN_END},{n}) "
          f"（先按期切分再展开号码）")

    # ---- 构造样本（按期展开） ----
    tr_t = range(WINDOW, TRAIN_END)
    te_t = range(TRAIN_END, n)

    def dataset(ts):
        X, Y, T = [], [], []
        for t in ts:
            f = build_features(mat, t)
            X.append(f)
            Y.append(mat[t])
            T.append(np.full(49, t))
        return np.concatenate(X), np.concatenate(Y), np.concatenate(T)

    Xtr, Ytr, _ = dataset(tr_t)
    Xte, Yte, Tte = dataset(te_t)
    print(f"训练样本 {len(Ytr)}（{len(list(tr_t))}期×49）  "
          f"测试样本 {len(Yte)}（{len(list(te_t))}期×49）")

    # ---- 四个模型 ----
    pA = np.full(len(Yte), 7 / 49)                                  # 公平基线
    pB = np.concatenate([bayesian_probs(mat, t) for t in te_t])     # 贝叶斯平滑

    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, Ytr)
    pC = clf.predict_proba(Xte)[:, 1]

    rf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
    rf.fit(Xtr, Ytr)
    pD = rf.predict_proba(Xte)[:, 1]

    print(f"\n{'模型':<16} {'Brier(↓)':>10} {'对数损失(↓)':>12}")
    for name, p in (("A 公平基线 1/7", pA), ("B 频率+贝叶斯", pB),
                    ("C 逻辑回归", pC), ("D 随机森林", pD)):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        print(f"{name:<16} {brier_score_loss(Yte, p):>10.5f} "
              f"{log_loss(Yte, p):>12.5f}")

    # ---- 聚合到一肖：玩法层评估 ----
    test_ts = list(te_t)
    idx = 0
    picks = {k: [] for k in ("A", "B", "C", "D", "R")}
    for t in test_ts:
        probs = {}
        for key, p49 in (("A", np.full(49, 7 / 49)),
                         ("B", bayesian_probs(mat, t)),
                         ("C", pC[idx:idx + 49]),
                         ("D", pD[idx:idx + 49])):
            # P(肖出现在7号) ≈ 1 - Π(1-p_i)
            ph = {}
            for z in ZODIACS:
                pm = 1.0
                for i in range(49):
                    if zodiac(i + 1) == z:
                        pm *= (1 - p49[i])
                ph[z] = 1 - pm
            probs[key] = max(ph, key=ph.get)
        idx += 49
        rng = np.random.default_rng(t)
        for k in picks:
            picks[k].append(probs.get(k, ZODIACS[int(rng.integers(12))]))

    print(f"\n{'选肖来源':<16} {'测试命中率':>9} {'EV/注':>8} {'注解':>14}")
    rows = []
    for k, label in (("A", "A 基线(最大后验)"), ("B", "B 贝叶斯平滑"),
                     ("C", "C 逻辑回归"), ("D", "D 随机森林"),
                     ("R", "R 随机选肖")):
        hits = wins = 0
        ev = 0.0
        for z, t in zip(picks[k], test_ts):
            h = any(zodiac(x) == z for x in
                    [df.loc[t, f"p{i}"] for i in range(1, 7)] + [df.loc[t, "sp"]])
            o = ODDS_MA if z == "马" else ODDS4
            wins += int(h)
            ev += (o - 1) if h else -1
            hits += 1
        print(f"{label:<16} {wins/hits:>9.1%} {ev/hits:>+8.3f}")
        rows.append((k, wins / hits, ev / hits))

    # 理论对照
    print(f"{'理论随机基线':<16} {'≈47.4%':>9} {'≈-0.14':>8} {'恒定负EV':>14}")

    best = max(rows, key=lambda r: r[2])
    print(f"\n[结论] 测试期上表现最好的是 {best[0]}：{best[1]:.1%} 命中 / "
          f"EV {best[2]:+.3f}。若与基线差距小且无统计显著性，"
          f"按教程口径判定：模型未学到可利用信号。")


if __name__ == "__main__":
    main()
