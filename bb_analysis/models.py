# -*- coding: utf-8 -*-
"""T8 · BB 模型层：公平基线 / 频率 / 贝叶斯 / 逻辑回归 + 概率校准。

所有模型输出 12 维【全开出现概率】向量（多标签），特征只许用 [:t]。

评级规则（预先定义，禁止事后挪动）：
  在验证段上，模型相对公平基线 A 的配对 ΔBrier<0 且配对 t 检验 p<0.05
  → "beat_baseline"；否则 "unproven"。评级只升不降，且换数据须重验。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from scipy import stats

from .combinations import ZODIAC_ORDER
from .patterns import BBPanel, p_theory_of
from .theory import single_prob

BAYES_PRIOR = 20.0      # Beta 先验强度（均值=理论 p0）


# ---------- 输出 ----------

@dataclass
class ZVector:
    probs: np.ndarray            # (12,)
    lo: Optional[np.ndarray] = None
    hi: Optional[np.ndarray] = None
    n: int = 0
    model: str = "A"


def model_a(panel: BBPanel) -> ZVector:
    """公平随机基线：理论命中率（4号肖47.17%/5号肖55.39%）。"""
    p = np.array([p_theory_of(z, panel) for z in range(12)])
    return ZVector(probs=p, model="A")


def model_b(panel: BBPanel, t: int, w: int = 30) -> ZVector:
    """滚动窗口频率（presence 口径）。"""
    lo = max(0, t - w)
    n = t - lo
    if n == 0:
        return ZVector(probs=np.array([p_theory_of(z, panel)
                                       for z in range(12)]), model="B")
    freq = panel.presence[lo:t].sum(axis=0) / n
    return ZVector(probs=freq, n=n, model="B")


def model_c(panel: BBPanel, t: int, w: int = 30,
            s: float = BAYES_PRIOR) -> ZVector:
    """Beta-Binomial 后验均值 + 90% 可信区间（先验均值=理论 p0）。"""
    lo = max(0, t - w)
    hits = panel.presence[lo:t].sum(axis=0).astype(float)
    n = t - lo
    if n == 0:
        mean = np.array([p_theory_of(z, panel) for z in range(12)])
        return ZVector(probs=mean, lo=mean, hi=mean, model="C")
    p0 = np.array([p_theory_of(z, panel) for z in range(12)])
    a = hits + p0 * s
    b = (n - hits) + (1 - p0) * s
    mean = a / (a + b)
    lo90 = stats.beta.ppf(0.05, a, b)
    hi90 = stats.beta.ppf(0.95, a, b)
    return ZVector(probs=mean, lo=lo90, hi=hi90, n=n, model="C")


# ---------- D：逐肖二分类逻辑回归（stride 重训） ----------

def _d_features(panel: BBPanel, t: int, z: int) -> np.ndarray:
    w10 = panel.presence[max(0, t - 10):t, z].mean() if t else 0.0
    w30 = panel.presence[max(0, t - 30):t, z].mean() if t else 0.0
    idx = np.nonzero(panel.presence[:t, z])[0]
    om = (t - 1 - idx[-1]) if idx.size else t
    streak = 0
    for v in panel.presence[:t, z][::-1]:
        if v:
            streak += 1
        else:
            break
    prev_sp = int(t > 0 and panel.special_idx[t - 1] == z)
    prev_p1 = int(t > 0 and panel.p1_idx[t - 1] == z)
    return np.array([w10, w30, min(om, 60) / 60.0, min(streak, 10) / 10.0,
                     prev_sp, prev_p1], dtype=float)


class ModelD:
    name = "D_logreg"

    def __init__(self, stride: int = 25, seed: int = 42):
        self.stride = stride
        self.seed = seed
        self.clfs: list[Any] = [None] * 12
        self.fitted_at = -10**9
        self.failed = False

    def maybe_fit(self, panel: BBPanel, t: int, min_train: int = 40) -> bool:
        if self.failed or t < min_train:
            return False
        if t - self.fitted_at < self.stride and self.clfs[0] is not None:
            return True
        try:
            from sklearn.linear_model import LogisticRegression
            for z in range(12):
                X = np.array([_d_features(panel, i, z) for i in range(t)])
                y = panel.presence[:t, z].astype(int)
                if len(np.unique(y)) < 2:
                    continue
                clf = LogisticRegression(max_iter=200, C=1.0,
                                         random_state=self.seed)
                clf.fit(X, y)
                self.clfs[z] = clf
            self.fitted_at = t
            return self.clfs[0] is not None
        except Exception:
            self.failed = True
            return False

    def predict(self, panel: BBPanel, t: int) -> Optional[ZVector]:
        if self.clfs[0] is None:
            return None
        probs = np.empty(12)
        try:
            for z in range(12):
                clf = self.clfs[z]
                if clf is None:
                    probs[z] = p_theory_of(z, panel)
                    continue
                probs[z] = float(clf.predict_proba(
                    _d_features(panel, t, z)[None, :])[0, 1])
        except Exception:
            return None
        return ZVector(probs=probs, n=t, model="D")


# ---------- 评级与校准 ----------

def paired_brier_test(p_model: np.ndarray, p_base: np.ndarray,
                      y: np.ndarray) -> dict[str, Any]:
    """逐样本多标签 Brier 配对差检验（模型 vs 基线）。"""
    br_m = ((p_model - y) ** 2).sum(axis=1)
    br_a = ((p_base - y) ** 2).sum(axis=1)
    diff = br_m - br_a
    t, p = stats.ttest_rel(br_m, br_a)
    return {"brier_model": float(br_m.mean()),
            "brier_base": float(br_a.mean()),
            "delta": float(diff.mean()), "t": float(t), "p": float(p),
            "grade": "beat_baseline" if (diff.mean() < 0 and p < 0.05)
            else "unproven"}


def calibration(probs: np.ndarray, y: np.ndarray,
                n_bins: int = 8) -> dict[str, Any]:
    """可靠性分箱 + ECE（多标签展平）。"""
    p = probs.ravel()
    yy = y.ravel().astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    ece = 0.0
    for b in range(n_bins):
        if b < n_bins - 1:
            m = (p >= edges[b]) & (p < edges[b + 1])
        else:
            m = (p >= edges[b]) & (p <= edges[b + 1])
        cnt = int(m.sum())
        if cnt == 0:
            continue
        conf = float(p[m].mean())
        acc = float(yy[m].mean())
        ece += cnt / len(p) * abs(conf - acc)
        bins.append({"bin": f"[{edges[b]:.2f},{edges[b + 1]:.2f})",
                     "n": cnt, "conf": conf, "freq": acc})
    return {"ece": float(ece), "bins": bins}
