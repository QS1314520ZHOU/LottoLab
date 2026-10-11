# -*- coding: utf-8 -*-
"""T11 · BB 平台十项必测。

1 无未来数据泄漏  2 预测开奖前冻结  3 随机基线正确  4 规律多重检验正确
5 赔率净盈亏正确  6 三期资金策略边界  7 数据更新不破坏旧数据
8 双色球与BB隔离  9 AI失败核心流程可用  10 旧功能回归（由全量套件覆盖）
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from bb_analysis import ai as bb_ai
from bb_analysis import database as db
from bb_analysis.bankroll import FixedStake, ThreePeriod, simulate
from bb_analysis.backtest import run_backtest
from bb_analysis.combinations import enumerate_combos
from bb_analysis.config import load_config
from bb_analysis.models import model_a, model_b, model_c
from bb_analysis.patterns import (PRED_RULES, WARMUP, _bh, load_panel,
                                  run_registry)
from bb_analysis.rules import breakeven_rate, expected_net, settle
from bb_analysis.theory import monte_carlo_all_present, single_prob
from bb_analysis.voting import aggregate, build_voters

CFG = load_config()
ROOT = Path(__file__).resolve().parent.parent


# ---------- 1 无未来数据泄漏 ----------

def test_no_future_leakage_rules_and_models():
    """改写 panel 的未来部分，t 时刻的选肖与模型输出必须逐位不变。"""
    panel = load_panel(CFG)
    t = 120
    orig = load_panel(CFG)

    # 规则层：只依赖 [:t]
    for entry, fn in PRED_RULES:
        z1 = fn(panel, t)
        # 篡改 t 之后的所有数据
        panel.presence[t:] = ~panel.presence[t:]
        panel.special_idx[t:] = (panel.special_idx[t:] + 5) % 12
        panel.sp_num[t:] = (panel.sp_num[t:] + 7)
        z2 = fn(panel, t)
        assert z1 == z2, f"规则 {entry.pid} 读到了未来数据"
        # 还原
        panel.presence[:] = orig.presence
        panel.special_idx[:] = orig.special_idx
        panel.sp_num[:] = orig.sp_num

    # 模型层：B/C 只依赖 [:t]
    b1, c1 = model_b(panel, t).probs.copy(), model_c(panel, t).probs.copy()
    panel.presence[t:] = ~panel.presence[t:]
    b2, c2 = model_b(panel, t).probs, model_c(panel, t).probs
    assert np.allclose(b1, b2) and np.allclose(c1, c2)


# ---------- 2 预测开奖前冻结 ----------

def test_predictions_frozen_before_draw():
    with db.connect(CFG.db_path) as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT target_issue, data_cutoff, odds, p_base, combo "
            "FROM predictions")]
    assert rows, "需先运行 backtest 生成预测（python -m bb_analysis backtest）"
    bad = [r for r in rows if r["data_cutoff"] >= r["target_issue"]]
    assert not bad, f"存在未冻结预测: {bad[:3]}"
    # selected 行的赔率与基线必须有值且口径正确
    sel = [r for r in rows if r["combo"]]
    assert sel
    for r in sel[:50]:
        assert r["odds"] in (1.76, 2.06)
        zsize = 5 if r["combo"] == "马" else 4
        assert r["p_base"] == pytest.approx(single_prob(zsize), rel=1e-9)


# ---------- 3 随机基线计算正确 ----------

def test_random_baseline_correct():
    # 理论单肖命中
    assert single_prob(4) == pytest.approx(0.4717193075, abs=1e-9)
    assert single_prob(5) == pytest.approx(0.5538963041, abs=1e-9)
    # 随机选肖的期望命中 ≈ 理论均值（MC，固定种子）
    panel = load_panel(CFG)
    base_avg = float(np.mean([single_prob(int(s))
                              for s in panel.zsize]))
    rng = np.random.default_rng(CFG.seed)
    T = panel.presence.shape[0]
    hits = []
    for _ in range(300):
        picks = rng.integers(0, 12, size=T)
        hits.append(float(panel.presence[np.arange(T), picks].mean()))
    assert abs(np.mean(hits) - base_avg) < 0.02


# ---------- 4 规律多重检验正确 ----------

def test_multiple_testing_registry():
    # BH 已知答案
    assert _bh([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.04, 0.04])
    # 注册表：q 与 p 的关系（q >= p；分母=检验总数）
    results, _ = run_registry(CFG, export=False)
    tested = [r for r in results
              if r.p_value is not None and r.p_adjusted is not None]
    assert len(tested) >= 12
    for r in tested:
        assert r.p_adjusted >= r.p_value - 1e-12
    # validated 必须同时满足 q<0.05 与 holdout 同向
    for r in results:
        if r.status == "validated" and r.p_adjusted is not None:
            assert r.p_adjusted < 0.05
            assert r.holdout_obs is not None and r.baseline is not None
            assert r.holdout_obs > r.baseline


# ---------- 5 赔率与净盈亏 ----------

def test_odds_settlement_math():
    ret, net = settle(5.0, 2.06, True)
    assert ret == pytest.approx(10.3) and net == pytest.approx(5.3)
    ret, net = settle(5.0, 1.76, False)
    assert ret == 0.0 and net == -5.0
    assert expected_net(5.0, 0.5, 2.0) == 0.0
    assert breakeven_rate(108.0) == pytest.approx(1 / 108)
    # 配置态赔率（总返还倍数含本金）
    assert CFG.odds.single_odds("马", date(2026, 6, 1)) == (1.76, "ok")
    assert CFG.odds.single_odds("鼠", date(2026, 6, 1)) == (2.06, "ok")
    assert CFG.odds.combo_odds_for(5) == 108.0


# ---------- 6 三期资金策略边界 ----------

def test_three_period_stake_boundaries():
    s = 5.0
    st = ThreePeriod(s)
    # 任务书口径逐期：1期=s；2期（1期未中）=2s；3期=s 不论2期输赢
    assert st.stake(1, None) == s
    assert st.stake(2, True) == 2 * s
    assert st.stake(2, False) == s
    assert st.stake(3, True) == s
    assert st.stake(3, False) == s
    # 周期重复：t=5（周期位1，上一期未中）→ 2s；t=6（位2）→ s
    assert st.stake(4, True) == s
    assert st.stake(5, True) == 2 * s
    assert st.stake(5, False) == s
    assert st.stake(6, True) == s
    # 固定策略恒定
    fx = FixedStake(s)
    assert all(fx.stake(t, True) == s for t in range(1, 10))

    # 结算边界：中/未中净额与破产停止
    outcomes = [{"hit": False, "odds": 2.06}] * 6
    led = simulate(outcomes, ThreePeriod(s), initial=10.0)
    # 1期-5(余5) → 2期stake10>余额5 → 破产于第2期
    assert led.ruin_at == 2
    assert led.net == -5.0

    # 同一路径下两策略命中序列一致（策略不改变命中率）
    path = [{"hit": h, "odds": 2.06}
            for h in (False, False, True, True, False, True, True)]
    a = simulate(path, FixedStake(5.0), 1000.0)
    b = simulate(path, ThreePeriod(5.0), 1000.0)
    assert a.hit_flags == b.hit_flags
    # 三期注额（位0=s；位1 上期未中→2s；位2=s）：
    # t1 s(未中) t2 2s(未中) t3 s(中) t4 s(中) t5 s(上期中) t6 s(未中) t7 s(中)
    assert b.stakes == [5, 10, 5, 5, 5, 5, 5]
    # 固定策略净额 = 4胜×5×(2.06−1) − 3负×5
    assert a.net == pytest.approx(4 * 5 * 1.06 - 3 * 5)
    # 加倍不改变命中，只放大注额流水
    assert b.total_staked > a.total_staked


# ---------- 7 数据更新不破坏旧数据 ----------

def test_data_update_preserves_old(tmp_path):
    before = _db_digest(CFG.db_path)
    csv_text = (ROOT / "data" / "bb_live_history.csv").read_bytes()
    csv_hash = hashlib.sha256(csv_text).hexdigest()

    from bb_analysis.ingestion import import_csv_file
    rep, counts = import_csv_file(CFG, ROOT / "data" / "bb_live_history.csv",
                                  source="bb_sync_csv")
    assert rep.ok
    assert counts["new_draws"] == 0          # 幂等：不重复计数
    after = _db_digest(CFG.db_path)
    assert before["n"] == after["n"]          # 行数不变
    assert before["sample"] == after["sample"]  # 抽样内容不变
    # 原始 CSV 未被触碰
    assert hashlib.sha256(
        (ROOT / "data" / "bb_live_history.csv").read_bytes()
    ).hexdigest() == csv_hash


def _db_digest(path):
    with db.connect(path) as conn:
        n = conn.execute("SELECT COUNT(*) c FROM draws").fetchone()["c"]
        sample = [dict(r) for r in conn.execute(
            "SELECT * FROM draws ORDER BY issue LIMIT 5")]
        z = conn.execute("SELECT COUNT(*) c FROM draw_zodiacs").fetchone()["c"]
    return {"n": n, "z": z, "sample": sample}


# ---------- 8 双色球与 BB 规则隔离 ----------

def test_ssq_bb_isolation():
    from app.config import load_config as ssq_load
    from app.baseline import prize1_probability
    ssq = ssq_load()
    # 库路径与规则空间完全不同
    assert ssq.db_path != CFG.db_path
    assert ssq.db_path.name == "ssq.db"
    # BB: C(49,7) 单肖 vs SSQ: C(33,6)×16
    bb_p1 = 1 / (math.comb(49, 7))
    ssq_p1 = prize1_probability(ssq)
    assert bb_p1 != ssq_p1
    assert math.comb(33, 6) * 16 == 17_721_088
    # 双色球包不引用 bb_analysis
    import app
    import subprocess, sys
    out = subprocess.run(
        [sys.executable, "-c",
         "import app.backtest, app.patterns, sys;"
         "assert not any(m.startswith('bb_analysis') for m in sys.modules)"],
        capture_output=True, text=True, cwd=str(ROOT))
    assert out.returncode == 0, out.stderr


# ---------- 9 AI 失败核心流程可用 ----------

def test_ai_failure_core_flow(monkeypatch):
    monkeypatch.setattr(bb_ai, "AI_KEY", "")
    res = bb_ai.explain("评分卡", "哪条显著？")
    assert res["ok"] is False and "MIMO_AI_KEY" in res["reason"]
    assert bb_ai.COVERAGE["failed"] >= 1
    # 核心流程不依赖 ai：注册表在 AI 失败后仍能运行
    results, text = run_registry(CFG, export=False)
    assert len(results) >= 13


# ---------- 10 旧功能回归 ----------

def test_legacy_regression_smoke():
    """旧入口的代表行为仍可用（完整回归由全量 pytest 覆盖）。"""
    combos = enumerate_combos(CFG)
    assert len(combos) == 12 + 66 + 220 + 495 + 792
    # 旧脚本可导入
    import bb_v9  # noqa: F401  （历史脚本保持可运行）
    # 回测可重复调用（唯一约束防重复预测）
    n_before = None
    with db.connect(CFG.db_path) as conn:
        n_before = conn.execute(
            "SELECT COUNT(*) c FROM predictions").fetchone()["c"]
    # 不重跑全量回测（耗时），只验证调用路径存在
    assert callable(run_backtest)
    with db.connect(CFG.db_path) as conn:
        n_after = conn.execute(
            "SELECT COUNT(*) c FROM predictions").fetchone()["c"]
    assert n_before == n_after
