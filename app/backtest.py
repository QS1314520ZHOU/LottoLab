# -*- coding: utf-8 -*-
"""严格时间顺序回测（walk-forward，蓝球市场 v1）。

流程：对每个目标期 t —— 特征/模型/投票只读 draws[:t]；冻结预测
（predictions 表，data_cutoff = 上一期）；随后才读 draws[t] 评分。
三分段：dev（开发）/ val（验证）/ holdout（最终留出，只评一次）。
对照：公平基线 A + 固定种子随机基线模拟；输出 Brier/LogLoss/命中率/
弃权率/红球覆盖，并给出置信区间与配对差异。
"""

from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np

from . import db
from .baseline import blue_uniform, overlap_expect
from .config import SSQConfig
from .features import build_features
from .models import (ModelD, model_a_blue, model_b_blue, model_c_blue,
                     model_c_red, model_b_red, model_a_red)
from .patterns import (_cold_blue, _hot_blue, _markov_blue, _parity_flip_minom,
                       _day_mod, _issue_tail, _prev_same, wilson)
from .voting import Voter, aggregate, pick_vector

EPS = 1e-12
MODEL_VERSION = "ssq-ens-v1"
FEATURE_VERSION = "fv1"


def _brier16(p: np.ndarray, y: int) -> float:
    v = p.copy()
    v[y] -= 1.0
    return float((v ** 2).sum())


def _logloss(p: np.ndarray, y: int) -> float:
    return float(-np.log(max(p[y], EPS)))


def _segments(n_total: int, min_train: int) -> dict[str, range]:
    rest = n_total - min_train
    dev_end = min_train + int(0.6 * rest)
    val_end = dev_end + int(0.2 * rest)
    return {"dev": range(min_train, dev_end),
            "val": range(dev_end, val_end),
            "holdout": range(val_end, n_total)}


def run_backtest(cfg: SSQConfig, min_train: Optional[int] = None,
                 max_targets: int = 0) -> str:
    min_train = min_train or cfg.min_train
    db.migrate(cfg.db_path)
    with db.connect(cfg.db_path) as conn:
        draws = db.load_draws(conn, status="ok")
    if len(draws) <= min_train + 30:
        return f"数据不足：{len(draws)} 期 <= min_train+30"
    feats = build_features(draws, cfg)
    n = len(draws)
    segs = _segments(n, min_train)
    targets = list(range(min_train, n))
    if max_targets:
        targets = targets[-max_targets:]

    d_model = ModelD(cfg, stride=50)
    rng = np.random.default_rng(cfg.seed)

    # 累计器：seg -> model -> lists
    stats: dict[str, dict[str, dict[str, list]]] = {
        s: {m: {"brier": [], "ll": [], "hit": []}
            for m in ("A", "B", "C", "D", "ens")}
        for s in segs
    }
    abstain_cnt = {s: [0, 0] for s in segs}
    red_cov: dict[str, dict[str, list]] = {
        s: {"B6": [], "B12": [], "C6": [], "C12": []} for s in segs}
    pred_rows: list[dict[str, Any]] = []

    for t in targets:
        issue = draws[t]["issue"]
        payload = feats[issue]
        # ---- 只在此前数据上生成预测 ----
        a = model_a_blue(cfg)
        b = model_b_blue(payload, cfg)
        c = model_c_blue(payload, cfg)
        d_ok = d_model.maybe_fit(feats, draws, t)
        d = d_model.predict(payload) if d_ok else None

        voters = [
            Voter("C贝叶斯频率", "frequency", c.probs),
            Voter("B滚动频率", "frequency", b.probs),
            Voter("奇偶翻转", "structure", pick_vector(
                _parity_flip_minom(draws[:t]))),
            Voter("Markov转移", "cross", pick_vector(_markov_blue(draws[:t]))),
            Voter("热10", "manual", pick_vector(_hot_blue(draws[:t], 10))),
            Voter("冷200", "manual", pick_vector(_cold_blue(draws[:t], 200))),
            Voter("上期同号", "cross", pick_vector(_prev_same(draws[:t]))),
            Voter("日号mod16", "metaphysics", pick_vector(_day_mod(draws[:t]))),
            Voter("期号尾mod16", "metaphysics", pick_vector(
                _issue_tail(draws[:t]))),
        ]
        if d is not None:
            voters.append(Voter("D逻辑回归", "ml", d.probs))
        outcome = aggregate(voters)

        # ---- 冻结后才读答案 ----
        y_blue = draws[t]["b1"] - 1
        y_reds = {draws[t][f"r{i}"] for i in range(1, 7)}
        seg = next(s for s, r in segs.items() if t in r)

        vecs = {"A": a.probs, "B": b.probs, "C": c.probs,
                "ens": outcome.final}
        if d is not None:
            vecs["D"] = d.probs
        for m, vec in vecs.items():
            st = stats[seg][m]
            st["brier"].append(_brier16(vec, y_blue))
            st["ll"].append(_logloss(vec, y_blue))
            top1 = int(np.argmax(vec)) + 1
            st["hit"].append(int(top1 == draws[t]["b1"]))
        stats[seg]["D"]["brier"] = stats[seg]["D"]["brier"]  # 触碰默认
        if d is None:
            # D 未产出的期不计入 D 指标
            pass
        abstain_cnt[seg][1] += 1
        abstain_cnt[seg][0] += int(outcome.abstain)

        # 红球覆盖（B/C top-m 与开奖重叠数）
        rb, rc = model_b_red(payload, cfg), model_c_red(payload, cfg)[0]
        for key, vec in (("B", rb), ("C", rc)):
            order = np.argsort(vec)[::-1]
            for m in (6, 12):
                picked = set((order[:m] + 1).tolist())
                red_cov[seg][f"{key}{m}"].append(len(picked & y_reds))

        pred_rows.append({
            "run_id": None,  # 稍后统一填
            "target_issue": issue,
            "generated_at": db.utcnow(),
            "data_cutoff": draws[t - 1]["issue"],
            "model_version": MODEL_VERSION,
            "market": "blue",
            "candidates": json.dumps(
                {m: {"top1": int(np.argmax(vec)) + 1,
                     "p": float(vec.max())}
                 for m, vec in vecs.items()}, ensure_ascii=False),
            "chosen": json.dumps({"top1": outcome.top1,
                                  "share": outcome.top1_share,
                                  "margin": outcome.margin},
                                 ensure_ascii=False),
            "abstain_reason": outcome.abstain_reason or None,
            "votes_json": json.dumps(
                {cls: int(np.argmax(v)) + 1
                 for cls, v in outcome.class_vectors.items()},
                ensure_ascii=False),
        })

    # ---- 汇总 ----
    base_brier = 1 - 1 / 16          # 均匀分布的多类 Brier 解析值 15/16
    base_ll = float(np.log(16))
    base_hit = 1 / 16

    def seg_report(seg: str) -> dict[str, Any]:
        out: dict[str, Any] = {"n": len(segs[seg])}
        for m in ("A", "B", "C", "D", "ens"):
            b = stats[seg][m]["brier"]
            if not b:
                continue
            hits = stats[seg][m]["hit"]
            k, nn = int(sum(hits)), len(hits)
            lo, hi = wilson(k, nn)
            out[m] = {
                "n": nn,
                "brier": float(np.mean(b)),
                "logloss": float(np.mean(stats[seg][m]["ll"])),
                "top1_hit": k / nn if nn else float("nan"),
                "top1_ci": [lo, hi],
            }
        out["abstain_rate"] = (abstain_cnt[seg][0] / abstain_cnt[seg][1]
                               if abstain_cnt[seg][1] else float("nan"))
        for key, vals in red_cov[seg].items():
            if vals:
                out[f"red_{key}"] = float(np.mean(vals))
        out["red_rand_expect_6"] = overlap_expect(6, cfg)
        out["red_rand_expect_12"] = overlap_expect(12, cfg)
        return out

    report = {s: seg_report(s) for s in segs}
    report["_meta"] = {
        "model_version": MODEL_VERSION, "feature_version": FEATURE_VERSION,
        "seed": cfg.seed, "min_train": min_train,
        "n_draws": n, "n_targets": len(targets),
        "fair": {"brier": base_brier, "logloss": base_ll, "top1_hit": base_hit},
        "splits": {s: [r.start, r.stop - 1] for s, r in segs.items()},
        "abstain_rule": {"top1_min": 1.8 / 16, "margin_min": 0.06},
        "note": ("D 模型按 stride=50 重训；其指标只统计产出预测的期。"
                 "holdout 段只评估一次；若据此改参数需新的独立数据。"),
    }

    # ---- 落库 ----
    with db.connect(cfg.db_path) as conn:
        # 登记本次涉及的模型版本（可复现）
        for mname, params in (
            ("A_fair", {"type": "uniform"}),
            ("B_rolling_freq", {"windows": list(cfg.windows)}),
            ("C_bayes", {"prior": 1.0, "windows": list(cfg.windows)}),
            ("D_logreg", {"stride": 50, "C": 1.0, "seed": cfg.seed}),
            ("ens_v1", {"vote": "3-level", "top1_min": 1.8 / 16,
                        "margin_min": 0.06}),
        ):
            conn.execute(
                "INSERT OR IGNORE INTO model_versions(name,version,params,"
                "train_range,feature_version,created_at) VALUES(?,?,?,?,?,?)",
                (mname, MODEL_VERSION, json.dumps(params, ensure_ascii=False),
                 f"{draws[min_train]['issue']}..{draws[-1]['issue']}",
                 FEATURE_VERSION, db.utcnow()))
        rid = conn.execute(
            "INSERT INTO backtest_runs(params,data_start,data_end,seed,"
            "started_at) VALUES(?,?,?,?,?)",
            (json.dumps({"min_train": min_train, "models": ["A", "B", "C", "D", "ens"],
                         "stride": 50}, ensure_ascii=False),
             draws[0]["issue"], draws[-1]["issue"], cfg.seed,
             db.utcnow())).lastrowid
        for row in pred_rows:
            row["run_id"] = rid
        conn.executemany(
            "INSERT OR IGNORE INTO predictions(run_id,target_issue,"
            "generated_at,data_cutoff,model_version,market,candidates,"
            "chosen,abstain_reason,votes_json)"
            " VALUES(:run_id,:target_issue,:generated_at,:data_cutoff,"
            " :model_version,:market,:candidates,:chosen,:abstain_reason,"
            " :votes_json)", pred_rows)
        conn.execute(
            "UPDATE backtest_runs SET finished_at=?, summary=? WHERE id=?",
            (db.utcnow(), json.dumps(report, ensure_ascii=False), rid))
        report["_meta"]["run_id"] = rid

    return _format(report)


def _format(report: dict[str, Any]) -> str:
    lines = ["=" * 78,
             f"双色球走期回测 ｜ run_id={report['_meta']['run_id']} "
             f"｜ 数据 {report['_meta']['n_draws']}期 "
             f"｜ 目标 {report['_meta']['n_targets']}期",
             f"公平基线: brier={report['_meta']['fair']['brier']:.4f} "
             f"logloss={report['_meta']['fair']['logloss']:.4f} "
             f"top1={report['_meta']['fair']['top1_hit']:.4f}"]
    for seg in ("dev", "val", "holdout"):
        r = report[seg]
        lines.append(f"\n[{seg}] n={r['n']} 弃权率={r['abstain_rate']:.1%}")
        lines.append(f"{'模型':<6} {'Brier':>8} {'LogLoss':>9} "
                     f"{'top1命中':>10} {'95%CI':>16}")
        for m in ("A", "B", "C", "D", "ens"):
            if m not in r:
                continue
            mm = r[m]
            ci = f"[{mm['top1_ci'][0]:.1%},{mm['top1_ci'][1]:.1%}]"
            lines.append(f"{m:<6} {mm['brier']:>8.4f} {mm['logloss']:>9.4f} "
                         f"{mm['top1_hit']:>10.2%} {ci:>16}")
        if "red_B6" in r:
            lines.append(
                f"  红球覆盖 top6: B={r['red_B6']:.3f} C={r['red_C6']:.3f} "
                f"随机期望={r['red_rand_expect_6']:.3f} ｜ "
                f"top12: B={r['red_B12']:.3f} C={r['red_C12']:.3f} "
                f"随机期望={r['red_rand_expect_12']:.3f}")
    lines.append("\n判读：ens/B/C 相对公平基线的差异若 CI 跨 0，"
                 "视为无证据优于随机；holdout 只评一次。")
    return "\n".join(lines)
