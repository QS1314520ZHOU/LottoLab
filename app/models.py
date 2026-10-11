# -*- coding: utf-8 -*-
"""统计与机器学习模型（输出概率分布，不只给单号）。

A. 公平基线：蓝均匀 1/16，红入选 6/33
B. 滚动窗口频率
C. 贝叶斯平滑（蓝：Dirichlet；红：Beta），输出后验区间
D. 正则化逻辑回归（蓝球多分类；按 stride 重训以控成本）

全部输入 = 目标期之前生成的 features payload。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .baseline import single_red_inclusion, uniform_blue_scores, uniform_red_scores
from .config import SSQConfig

PRIOR_STRENGTH = 1.0   # 蓝球 Dirichlet 每类先验伪计数
RED_PRIOR = 8.0        # 红球 Beta 先验强度（均值 6/33）


@dataclass
class BlueDist:
    probs: np.ndarray            # (16,)
    lo: Optional[np.ndarray] = None   # 可信区间下界（模型C）
    hi: Optional[np.ndarray] = None
    n: int = 0


def model_a_blue(cfg: SSQConfig) -> BlueDist:
    return BlueDist(probs=np.asarray(uniform_blue_scores(cfg)), n=0)


def model_a_red(cfg: SSQConfig) -> np.ndarray:
    return np.asarray(uniform_red_scores(cfg))


def model_b_blue(payload: dict[str, Any], cfg: SSQConfig,
                 w: int = 100) -> BlueDist:
    freq = np.asarray(payload[f"blue_freq_{min(w, cfg.windows[-1])}"],
                      dtype=float)
    s = freq.sum()
    probs = freq / s if s > 0 else np.full(16, 1 / 16)
    return BlueDist(probs=probs, n=w)


def model_b_red(payload: dict[str, Any], cfg: SSQConfig,
                w: int = 100) -> np.ndarray:
    return np.asarray(payload[f"red_freq_{min(w, cfg.windows[-1])}"],
                      dtype=float)


def model_c_blue(payload: dict[str, Any], cfg: SSQConfig,
                 w: int = 100) -> BlueDist:
    """Dirichlet 平滑后验均值 + 正态近似 90% 可信区间。"""
    from scipy.stats import norm
    freq = np.asarray(payload[f"blue_freq_{min(w, cfg.windows[-1])}"],
                      dtype=float)
    counts = freq * w
    alpha = counts + PRIOR_STRENGTH
    probs = alpha / alpha.sum()
    z = norm.ppf(0.95)
    se = np.sqrt(probs * (1 - probs) / (w + 16 * PRIOR_STRENGTH))
    return BlueDist(probs=probs, lo=np.clip(probs - z * se, 0, 1),
                    hi=np.clip(probs + z * se, 0, 1), n=w)


def model_c_red(payload: dict[str, Any], cfg: SSQConfig,
                w: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Beta 后验均值与 90% 可信区间（红球入选概率）。"""
    from scipy.stats import beta as beta_dist
    freq = np.asarray(payload[f"red_freq_{min(w, cfg.windows[-1])}"],
                      dtype=float)
    hits = freq * w
    p0 = single_red_inclusion(cfg)
    a = hits + p0 * RED_PRIOR
    b = (w - hits) + (1 - p0) * RED_PRIOR
    mean = a / (a + b)
    lo = beta_dist.ppf(0.05, a, b)
    hi = beta_dist.ppf(0.95, a, b)
    return mean, lo, hi


def _d_row(payload: dict[str, Any], cfg: SSQConfig) -> np.ndarray:
    w = cfg.windows[0]
    row = list(payload[f"blue_freq_{w}"]) + \
          [payload[f"blue_om_{b}"] / 100.0 for b in range(1, 17)]
    prev = payload.get("prev") or {}
    row += [prev.get("blue", 8) / 16.0, prev.get("blue_odd", 0.0),
            prev.get("blue_half", 0.0), prev.get("red_repeat_prev", 0) / 6.0]
    return np.asarray(row, dtype=float)


class ModelD:
    """蓝球多分类逻辑回归；stride 重训控制成本。"""

    name = "D_logreg"

    def __init__(self, cfg: SSQConfig, stride: int = 50):
        self.cfg = cfg
        self.stride = stride
        self._clf = None
        self._fitted_at = -10**9
        self._failed = False

    def maybe_fit(self, feats: dict[str, dict[str, Any]],
                  draws: list[dict[str, Any]], t: int) -> bool:
        if self._failed:
            return False
        if t - self._fitted_at < self.stride and self._clf is not None:
            return True
        try:
            from sklearn.linear_model import LogisticRegression
            X, y = [], []
            start = self.cfg.min_train // 2
            for i in range(start, t):
                payload = feats[draws[i]["issue"]]
                X.append(_d_row(payload, self.cfg))
                y.append(draws[i]["b1"] - 1)
            if len(set(y)) < 4 or len(y) < 80:
                return False
            clf = LogisticRegression(max_iter=300, C=1.0,
                                     random_state=self.cfg.seed)
            clf.fit(np.asarray(X), np.asarray(y))
            self._clf = clf
            self._fitted_at = t
            return True
        except Exception:
            self._failed = True
            return False

    def predict(self, payload: dict[str, Any]) -> BlueDist | None:
        if self._clf is None:
            return None
        try:
            p = self._clf.predict_proba(_d_row(payload, self.cfg)[None, :])[0]
        except Exception:
            return None
        return BlueDist(probs=p, n=int(self._fitted_at))
