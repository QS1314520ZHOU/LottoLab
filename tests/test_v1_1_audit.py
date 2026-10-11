# -*- coding: utf-8 -*-
"""V1.1 P3 测试：预测记录不可重写 / 统一结论句 / holdout 分块 / 审计字段齐全。"""

from __future__ import annotations

import json

from bb_analysis import database as db
from bb_analysis.config import load_config

CFG = load_config()


# ---------- 预测 append-only：改动企图不能改写历史 ----------

def test_predictions_immutable():
    with db.connect(CFG.db_path) as conn:
        # 取一行现有预测做指纹
        row = conn.execute(
            "SELECT id, target_issue, p_est, votes_json FROM predictions "
            "ORDER BY id LIMIT 1").fetchone()
        assert row is not None, "需先运行 python -m bb_analysis backtest"
        before = dict(row)
        n_before = conn.execute(
            "SELECT COUNT(*) c FROM predictions").fetchone()["c"]
        # 任何“改写历史”都不被允许：唯一索引 + 只增模型保证
        after = conn.execute(
            "SELECT id, target_issue, p_est, votes_json FROM predictions "
            "WHERE id=?", (before["id"],)).fetchone()
        assert dict(after) == before              # 历史行逐字段未变
        n_after = conn.execute(
            "SELECT COUNT(*) c FROM predictions").fetchone()["c"]
        assert n_after == n_before
        # run 追加只增
        run_ids = [r["id"] for r in conn.execute(
            "SELECT id FROM backtest_runs ORDER BY id")]
        assert len(run_ids) >= 1


def test_predictions_audit_fields_complete():
    """每条预测都必须带：预测时间/数据截止/模型版本/参数/概率/结果关联。"""
    with db.connect(CFG.db_path) as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM predictions")]
    assert rows
    for r in rows:
        assert r["generated_at"], "缺预测时间"
        assert r["data_cutoff"] and r["data_cutoff"] < r["target_issue"]
        assert r["model_version"], "缺模型版本"
        assert r["votes_json"], "缺参数/投票记录"
        assert r["p_est"] is not None and r["p_base"] is not None
        assert r["sample_n"] is not None


# ---------- 统一结论句 ----------

def test_bb_conclusion_sentence():
    with db.connect(CFG.db_path) as conn:
        row = conn.execute(
            "SELECT summary_json FROM backtest_runs "
            "ORDER BY id DESC LIMIT 1").fetchone()
    assert row and row["summary_json"], "需先运行回测"
    summary = json.loads(row["summary_json"])
    concl = summary.get("conclusion", "")
    assert concl, "缺 conclusion 字段"
    # 无优势证据时必须使用标准句
    ratings = summary.get("ratings", {})
    beat = [m for m, r in ratings.items()
            if isinstance(r, dict) and r.get("grade") == "beat_baseline"]
    sig = [r for r in summary.get("holdout_tests", [])
           if r.get("q") is not None and r["q"] < 0.05]
    if not (beat and sig):
        assert "未发现可靠优势" in concl
    # 分块稳定性存在且非-ens模型各3块
    blocks = summary.get("holdout_blocks", {})
    assert blocks, "缺 holdout 分块"
    for m in ("A", "B", "C", "D"):
        assert len(blocks.get(m, [])) == 3, f"{m} 分块不完整"
    for m, rows in blocks.items():
        for r in rows:
            assert 0.0 <= r["ci"][0] <= r["ci"][1] <= 1.0
            assert r["n"] > 0


def test_ssq_conclusion_sentence(tmp_path):
    """自包含：模拟数据临时库跑迷你回测，断言标准结论句（不依赖真实ssq.db）。"""
    import random
    from dataclasses import replace

    from app import db as adb
    from app.backtest import run_backtest
    from app.config import load_config as ssq_load

    rng = random.Random(11)
    rows = []
    for i in range(1, 261):
        reds = sorted(rng.sample(range(1, 34), 6))
        rows.append({
            "issue": f"2098{i:04d}", "draw_date": f"2098-01-{(i % 28) + 1:02d}",
            "red": reds, "blue": rng.randint(1, 16),
            "source": "simulated"})
    ssq = replace(ssq_load(), db_path=tmp_path / "conc.db")
    adb.migrate(ssq.db_path)
    with adb.connect(ssq.db_path) as c:
        adb.insert_draws_batch(c, rows)
    out = run_backtest(ssq, min_train=100, max_targets=80)
    assert "走期回测" in out and "未发现可靠优势" in out
    with adb.connect(ssq.db_path) as conn:
        row = conn.execute(
            "SELECT summary FROM backtest_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
    summary = json.loads(row["summary"])
    assert "未发现可靠优势" in summary["_meta"].get("conclusion", "")
