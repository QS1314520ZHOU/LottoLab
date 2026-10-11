# -*- coding: utf-8 -*-
"""T9 · BB 严格走期回测（全开一肖市场）。

流程（对每个目标期 t）：
  1) 只读 panel[:t]：模型 A/B/C/D 输出12维出现概率；
  2) 投票（validated 规律才有投票权；当前注册表无 → 规律不入局）；
  3) 冻结预测写入 predictions（data_cutoff = 上一期期号）；
  4) 之后才读 panel[t] 评分。

三分段：dev / val / holdout（60/20/20，自 WARMUP 起）。
评级：val 段配对 ΔBrier vs 公平基线 A（beat_baseline / unproven）。
基线口径：BB 全开一肖理论命中率（theory.single_prob），非双色球口径。
多重检验：holdout 段对 4 个被检对象（B/C/D/ens）的命中-基线检验做 BH。
小样本/区间跨基线 → 结论一律“证据不足”。
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
from scipy import stats

from . import database as db
from .combinations import ZODIAC_ORDER
from .config import AppConfig, load_config
from .models import (ModelD, calibration, model_a, model_b, model_c,
                     paired_brier_test)
from .patterns import PRED_RULES, WARMUP, load_panel, p_theory_of, run_registry
from .voting import build_voters, aggregate

MODEL_VERSION = "bb-ens-v1"


def _segments(T: int) -> dict[str, range]:
    rest = T - WARMUP
    dev_end = WARMUP + int(0.6 * rest)
    val_end = dev_end + int(0.2 * rest)
    return {"dev": range(WARMUP, dev_end),
            "val": range(dev_end, val_end),
            "holdout": range(val_end, T)}


def _brier12(p: np.ndarray, y: np.ndarray) -> float:
    return float(((p - y) ** 2).sum())


def _logloss12(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def run_backtest(cfg: AppConfig | None = None) -> str:
    cfg = cfg or load_config()
    panel = load_panel(cfg)
    T = len(panel.issues)
    segs = _segments(T)

    # 规律投票权：仅 validated（预期为空，如实处理）
    reg_results, _ = run_registry(cfg, export=False)
    validated = []
    pid_to_fn = dict((e.pid, fn) for e, fn in PRED_RULES)
    for r in reg_results:
        if r.status == "validated" and r.pid in pid_to_fn:
            validated.append((r.pid, pid_to_fn[r.pid]))

    d_model = ModelD(stride=25, seed=cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    accum: dict[str, dict[str, dict[str, list]]] = {
        s: {m: {"br": [], "ll": [], "hit": [], "p": [], "y": [],
                "base_top": []}
            for m in ("A", "B", "C", "D", "ens")} for s in segs}
    abstains = {s: [0, 0] for s in segs}
    ens_probs: list[np.ndarray] = []
    ens_y: list[np.ndarray] = []
    ens_seg: list[str] = []
    pred_rows: list[dict[str, Any]] = []

    for t in range(WARMUP, T):
        # ---- 预测（仅 [:t]）----
        a = model_a(panel)
        b = model_b(panel, t)
        c = model_c(panel, t)
        d_ok = d_model.maybe_fit(panel, t)
        d = d_model.predict(panel, t) if d_ok else None
        pat_vecs = []
        for pid, fn in validated:
            z = fn(panel, t)
            if z is not None:
                vec = np.full(12, 0.05)
                vec[z] = 0.5
                pat_vecs.append((pid, vec))
        outcome = aggregate(build_voters(b, c, d, pat_vecs))

        # ---- 冻结后读答案 ----
        y = panel.presence[t].astype(float)
        seg = next(s for s, r in segs.items() if t in r)
        vecs = {"A": a.probs, "B": b.probs, "C": c.probs, "ens": outcome.final}
        if d is not None:
            vecs["D"] = d.probs
        for m, vec in vecs.items():
            st = accum[seg][m]
            st["br"].append(_brier12(vec, y))
            st["ll"].append(_logloss12(vec, y))
            top = int(np.argmax(vec))
            if m == "ens" and outcome.pick is not None:
                st["hit"].append(int(y[outcome.pick]))
                st["p"].append(float(outcome.final[outcome.pick]))
                st["base_top"].append(p_theory_of(outcome.pick, panel))
            elif m != "ens":
                st["hit"].append(int(y[top]))
                st["p"].append(float(vec[top]))
                st["base_top"].append(p_theory_of(top, panel))
            st["y"].append(y)
        abstains[seg][1] += 1
        abstains[seg][0] += int(outcome.abstain)
        # ens 逐期对齐记录（弃权=None），供分块稳定性用
        accum[seg]["ens"].setdefault("period_hit", []).append(
            None if outcome.pick is None else int(y[outcome.pick]))
        if seg == "val" or seg == "holdout":
            ens_probs.append(outcome.final)
            ens_y.append(y)
            ens_seg.append(seg)

        # ---- 审计落库 ----
        if outcome.pick is not None:
            zname = outcome.pick_name
            odds, _st = cfg.odds.single_odds(zname, panel.dates[t])
            p_base = p_theory_of(outcome.pick, panel)
            p_est = float(outcome.final[outcome.pick])
        else:
            # 弃权行也必须留审计：用 0 占位，靠 combo='' + abstain_reason 标识
            zname, odds, p_base, p_est = "", 0.0, 0.0, 0.0
        pred_rows.append({
            "run_id": None,
            "target_issue": panel.issues[t],
            "generated_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds"),
            "model_version": MODEL_VERSION,
            "market": "single_zodiac",
            "k": 1,
            "combo": zname,
            "p_est": p_est,
            "p_base": p_base,
            "p_lo": None, "p_hi": None,
            "sample_n": int(t),
            "odds": odds,
            "data_cutoff": panel.issues[t - 1],
            "selected": int(outcome.pick is not None),
            "votes_json": json.dumps(
                {"classes": outcome.class_picks, "votes": outcome.votes,
                 "top1_p": outcome.top1_p, "margin": outcome.margin},
                ensure_ascii=False),
            "abstain_reason": outcome.abstain_reason or None,
        })

    # ---- 评级（val 段配对检验，逐期与 A 对齐）----
    ratings = {}

    def pair_test(seg: str, m: str) -> Optional[dict]:
        bm = accum[seg][m]["br"]
        ba = accum[seg]["A"]["br"]
        if not bm or len(bm) != len(ba):
            return None
        diff = np.asarray(bm) - np.asarray(ba)
        t_, p_ = stats.ttest_rel(bm, ba)
        return {"brier_model": float(np.mean(bm)),
                "brier_base": float(np.mean(ba)),
                "delta": float(diff.mean()), "t": float(t_),
                "p": float(p_),
                "grade": "beat_baseline"
                if diff.mean() < 0 and p_ < 0.05 else "unproven"}

    for m in ("B", "C", "D", "ens"):
        r = pair_test("val", m)
        ratings[m] = r or {"grade": "unproven", "note": "val 段无数据"}

    # ---- 校准（ens @ val）----
    val_idx = [i for i, s in enumerate(ens_seg) if s == "val"]
    calib = {}
    if val_idx:
        P = np.array([ens_probs[i] for i in val_idx])
        Y = np.array([ens_y[i] for i in val_idx])
        calib = calibration(P, Y)

    # ---- holdout 命中 vs 各自公平基线（top1 位置的理论命中率）+ BH ----
    hold_tests = []
    for m in ("B", "C", "D", "ens"):
        hits = accum["holdout"][m]["hit"]
        base_tops = accum["holdout"][m]["base_top"]
        if not hits:
            hold_tests.append({"model": m, "p": 1.0, "note": "no_data"})
            continue
        k = int(sum(hits))
        n = len(hits)
        base_sum = float(sum(base_tops))
        var = float(sum(p * (1 - p) for p in base_tops)) or 1e-9
        z = (k - base_sum) / math.sqrt(var)
        hold_tests.append({"model": m, "n": n, "hits": k,
                           "rate": k / n,
                           "base": base_sum / n,
                           "z": z, "p": float(1 - stats.norm.cdf(z))})
    order = np.argsort([x["p"] for x in hold_tests])
    adj = [1.0] * len(hold_tests)
    prev = 1.0
    m_cnt = len(hold_tests)
    for rank in range(m_cnt - 1, -1, -1):
        i = int(order[rank])
        prev = min(prev, hold_tests[i]["p"] * m_cnt / (rank + 1))
        adj[i] = prev
    for row, q in zip(hold_tests, adj):
        row["q"] = q

    # ---- 随机基线 MC ----
    base_avg = float(np.mean([p_theory_of(z, panel) for z in range(12)]))
    mc_hits = []
    for _ in range(2000):
        cnt = 0
        for t in segs["holdout"]:
            z = int(rng.integers(12))
            cnt += int(panel.presence[t, z])
        mc_hits.append(cnt / max(1, len(segs["holdout"])))
    mc = {"mean": float(np.mean(mc_hits)),
          "ci95": [float(np.percentile(mc_hits, 2.5)),
                   float(np.percentile(mc_hits, 97.5))],
          "seed": cfg.seed, "sims": 2000}

    # ---- 跨时间窗口：holdout 三分块稳定性（前瞻性逐块评估）----
    hold_list = list(segs["holdout"])
    n_blk = 3
    blocks = np.array_split(np.array(hold_list), n_blk)
    holdout_blocks: dict[str, list[dict[str, Any]]] = {}
    for m in ("A", "B", "C", "D", "ens"):
        if m == "ens":
            seq = accum["holdout"]["ens"].get("period_hit", [])
        else:
            seq = accum["holdout"][m]["hit"]
        rows = []
        offset = 0
        for bi, blk in enumerate(blocks):
            seg_hits = [h for h in seq[offset:offset + len(blk)]
                        if h is not None]
            offset += len(blk)
            if not seg_hits:
                continue
            k, nn = int(sum(seg_hits)), len(seg_hits)
            ph = k / nn
            se = math.sqrt(max(1e-9, ph * (1 - ph) / nn))
            rows.append({"block": bi + 1, "n": nn, "hit": ph,
                         "abstain": len(blk) - nn if m == "ens" else 0,
                         "ci": [max(0.0, ph - 1.96 * se),
                                min(1.0, ph + 1.96 * se)]})
        holdout_blocks[m] = rows

    # ---- 汇总 ----
    def seg_row(seg: str, m: str) -> Optional[dict]:
        st = accum[seg][m]
        if not st["br"]:
            return None
        hits = st["hit"]
        n = len(hits)
        k = int(sum(hits)) if hits else 0
        ph = k / n if n else float("nan")
        se = math.sqrt(max(1e-9, ph * (1 - ph) / n)) if n else float("nan")
        return {"n": n, "brier": float(np.mean(st["br"])),
                "logloss": float(np.mean(st["ll"])),
                "top1_hit": ph, "ci": [ph - 1.96 * se, ph + 1.96 * se],
                "abstain_rate": (abstains[seg][0] / abstains[seg][1]
                                 if abstains[seg][1] else float("nan"))}

    # ---- 统一结论句（预先定义：评级达标 且 holdout 校正显著 才算有证据）----
    beat = [m for m, r in (ratings or {}).items()
            if isinstance(r, dict) and r.get("grade") == "beat_baseline"]
    sig = [r["model"] for r in hold_tests
           if r.get("q") is not None and r["q"] < 0.05]
    if beat and sig:
        conclusion = (f"发现统计证据：评级达标 {beat} 且 holdout 显著 {sig}"
                      f"（仍需前瞻性数据复验，不构成未来优势承诺）")
    else:
        conclusion = ("未发现可靠优势：无模型达到 beat_baseline 评级"
                      "或 holdout 多重校正后不显著；报告不将任何差异表述为优势。")

    summary: dict[str, Any] = {
        "meta": {"model_version": MODEL_VERSION, "seed": cfg.seed,
                 "warmup": WARMUP,
                 "segments": {s: [r.start, r.stop - 1]
                              for s, r in segs.items()},
                 "n_draws": T,
                 "baseline_hit": base_avg,
                 "baseline_note": "BB全开一肖理论命中率均值（4号肖47.17%/"
                                  "5号肖55.39%），非双色球口径"},
        "ratings": ratings,
        "calibration_ens_val": calib,
        "holdout_tests": hold_tests,
        "holdout_blocks": holdout_blocks,
        "random_baseline_mc": mc,
        "conclusion": conclusion,
        "patterns_validated_for_voting": [p for p, _ in validated],
    }
    for s in segs:
        summary[s] = {m: seg_row(s, m)
                      for m in ("A", "B", "C", "D", "ens")
                      if seg_row(s, m)}

    # ---- 落库 ----
    db.migrate(cfg.db_path)
    with db.connect(cfg.db_path) as conn:
        run_id = db.create_run(
            conn,
            {"min_train": WARMUP, "models": ["A", "B", "C", "D", "ens"],
             "vote": "3level", "margin": 0.02},
            panel.issues[WARMUP], panel.issues[-1], MODEL_VERSION)
        for row in pred_rows:
            row["run_id"] = run_id
        db.insert_predictions(conn, pred_rows)
        db.finish_run(conn, run_id, summary)

    text = _fmt(summary, run_id)
    return text


def _fmt(s: dict[str, Any], run_id: int) -> str:
    lines = ["=" * 88,
             f"BB 走期回测 ｜ run_id={run_id} ｜ {s['meta']['model_version']} "
             f"｜ 数据 {s['meta']['n_draws']}期 seed={s['meta']['seed']}",
             f"基线：{s['meta']['baseline_note']}（均值 {s['meta']['baseline_hit']:.4f}）",
             f"分段：{s['meta']['segments']}"]
    for seg in ("dev", "val", "holdout"):
        block = s.get(seg) or {}
        if not block:
            continue
        lines.append(f"\n[{seg}]")
        lines.append(f"{'模型':<6} {'Brier':>8} {'LogLoss':>9} "
                     f"{'top1命中':>9} {'95%CI':>16} {'弃权率':>8}")
        for m, r in block.items():
            if not r:
                continue
            ci = f"[{r['ci'][0]:.1%},{r['ci'][1]:.1%}]"
            ab = f"{r['abstain_rate']:.1%}" if m == "ens" else "-"
            lines.append(f"{m:<6} {r['brier']:>8.4f} {r['logloss']:>9.4f} "
                         f"{r['top1_hit']:>9.2%} {ci:>16} {ab:>8}")
    lines.append("\n[评级 · val 段 · 预先定义门槛 beat_baseline]")
    for m, r in s["ratings"].items():
        if r and "delta" in r:
            lines.append(f"  {m:<5} ΔBrier={r['delta']:+.4f} "
                         f"p={r['p']:.3f} → {r['grade']}")
        else:
            lines.append(f"  {m:<5} {r.get('grade', 'unproven')}")
    lines.append("\n[holdout · 命中 vs 公平基线 · BH 校正（分母=4）]")
    for r in s["holdout_tests"]:
        if "rate" in r:
            lines.append(f"  {r['model']:<5} {r['hits']}/{r['n']}="
                         f"{r['rate']:.1%} vs {r['base']:.1%} "
                         f"z={r['z']:+.2f} p={r['p']:.3f} q={r['q']:.3f}"
                         f" → {'显著' if r['q'] < 0.05 else '不显著/证据不足'}")
        else:
            lines.append(f"  {r['model']:<5} {r.get('note')}")
    mc = s["random_baseline_mc"]
    lines.append(f"\n[随机基线 MC] {mc['sims']}次 seed={mc['seed']} "
                 f"holdout 命中均值 {mc['mean']:.1%} "
                 f"CI95=[{mc['ci95'][0]:.1%},{mc['ci95'][1]:.1%}]")
    blocks = s.get("holdout_blocks") or {}
    if blocks:
        lines.append("[holdout 跨时间窗口·三分块 top1命中]")
        for m, rows in blocks.items():
            if not rows:
                continue
            cells = " | ".join(
                f"块{r['block']}:{r['hit']:.1%}[{r['ci'][0]:.0%},"
                f"{r['ci'][1]:.0%}]" for r in rows)
            lines.append(f"  {m:<5} {cells}")
    cal = s.get("calibration_ens_val") or {}
    if cal:
        lines.append(f"[校准 · ens@val] ECE={cal['ece']:.4f} "
                     f"分箱数={len(cal['bins'])}")
    lines.append(f"\n[投票权] validated 规律: "
                 f"{s['patterns_validated_for_voting'] or '无（全部未过双门槛，规律不投票）'}")
    lines.append(f"\n[结论] {s.get('conclusion', '未发现可靠优势')}")
    return "\n".join(lines)
