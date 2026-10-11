# -*- coding: utf-8 -*-
"""端到端测试：模拟数据（明确标注）→ 导入 → 校验 → 特征 → 回测 → 报告。

全程使用临时数据库，不联网、不触碰真实 ssq.db 与 BB 数据。
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import replace
from pathlib import Path

import pytest

from app import db
from app.config import load_config
from app.validate import validate_rows


def _make_simulated_draws(n: int = 360, seed: int = 7):
    """随机模拟数据 —— 仅测试用，source=simulated 与真实数据严格区分。"""
    rng = random.Random(seed)
    rows = []
    for i in range(1, n + 1):
        reds = sorted(rng.sample(range(1, 34), 6))
        rows.append({
            "issue": f"2099{i:04d}",
            "draw_date": f"2099-01-{(i % 28) + 1:02d}",
            "red": reds,
            "blue": rng.randint(1, 16),
            "source": "simulated",
        })
    return rows


@pytest.fixture()
def sim_cfg(tmp_path):
    cfg = load_config()
    return replace(cfg, db_path=tmp_path / "ssq_e2e.db")


def test_e2e_sample_to_report(sim_cfg, tmp_path, monkeypatch):
    rows = _make_simulated_draws()

    # 0) 明确标注模拟数据
    assert all(r["source"] == "simulated" for r in rows)

    # 1) 校验
    rep = validate_rows(rows, sim_cfg)
    assert rep.ok, rep.errors

    # 2) 导入（幂等）
    db.migrate(sim_cfg.db_path)
    with db.connect(sim_cfg.db_path) as c:
        n1 = db.insert_draws_batch(c, rows)["added"]
        n2 = db.insert_draws_batch(c, rows)["added"]
    assert n1 == 360 and n2 == 0

    # 3) 校验库内数据
    from app.validate import report_database
    info = report_database(sim_cfg)
    assert info["valid_rows"] == 360

    # 4) 分析（特征+评分卡+发现）
    from app.analyze import run_analysis
    out = run_analysis(sim_cfg, pattern_limit=3)
    assert "规律尝试总数" in out and "exploratory" in out

    # 5) 回测（限制目标数控制时长）
    from app.backtest import run_backtest
    bt = run_backtest(sim_cfg, min_train=120, max_targets=120)
    assert "走期回测" in bt and "holdout" in bt
    with db.connect(sim_cfg.db_path) as c:
        preds = c.execute(
            "SELECT COUNT(*) c FROM predictions").fetchone()["c"]
        assert preds > 0
        # 泄漏：每条预测的 data_cutoff < target_issue
        leak = c.execute(
            "SELECT COUNT(*) c FROM predictions WHERE data_cutoff >= target_issue"
        ).fetchone()["c"]
        assert leak == 0
        run = c.execute("SELECT summary FROM backtest_runs "
                        "ORDER BY id DESC LIMIT 1").fetchone()
        summary = json.loads(run["summary"])
        assert "holdout" in summary

    # 6) 报告导出
    import app.reports as reports_mod
    monkeypatch.setattr(reports_mod, "PROJECT_ROOT", tmp_path)
    from app.reports import generate_report
    text = generate_report(sim_cfg)
    assert (tmp_path / "reports" / "ssq_report.json").exists()
    assert (tmp_path / "reports" / "ssq_report.txt").exists()
    payload = json.loads(
        (tmp_path / "reports" / "ssq_report.json").read_text("utf-8"))
    assert payload["data"]["valid_rows"] == 360
    assert any("不构成预测" in w for w in payload["warnings"])


def test_e2e_csv_roundtrip(tmp_path):
    """样例 CSV（标注 simulated）读写往返。"""
    rows = _make_simulated_draws(10)
    p = tmp_path / "sample.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["issue", "draw_date", "r1", "r2", "r3", "r4", "r5",
                    "r6", "b1", "source"])
        for r in rows:
            w.writerow([r["issue"], r["draw_date"], *r["red"], r["blue"],
                        r["source"]])
    # 读回
    read_back = []
    with open(p, encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            read_back.append({
                "issue": rec["issue"], "draw_date": rec["draw_date"],
                "red": [int(rec[f"r{i}"]) for i in range(1, 7)],
                "blue": int(rec["b1"]), "source": rec["source"]})
    assert read_back == rows
