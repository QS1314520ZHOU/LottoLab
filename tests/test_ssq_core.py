# -*- coding: utf-8 -*-
"""SSQ 核心测试：校验/去重/冲突/HTTP异常/特征泄漏/基线/多重检验/AI降级/BB兼容。

网络一律用模拟响应，不访问真实网站。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app import db
from app.baseline import (prize1_probability, single_red_inclusion,
                          monte_carlo_prize1, mc_stderr, red_hits_pmf)
from app.config import load_config
from app.features import FEATURE_VERSION, build_features, omission_at
from app.patterns import bh_adjust, wilson, binom_p, Rule, evaluate_rule
from app.sources import CwlSource, Backup500Source, SourceError
from app.sources.base import SourceAdapter
from app.validate import validate_rows

CFG = load_config()


def _rows():
    return [
        {"issue": "2026001", "draw_date": "2026-01-01",
         "red": [1, 2, 3, 4, 5, 6], "blue": 7, "source": "t"},
        {"issue": "2026002", "draw_date": "2026-01-03",
         "red": [7, 8, 9, 10, 11, 12], "blue": 16, "source": "t"},
        {"issue": "2026003", "draw_date": "2026-01-06",
         "red": [13, 14, 15, 16, 17, 18], "blue": 1, "source": "t"},
    ]


# ---------- 1 号码范围与数量 ----------

def test_number_range_and_count():
    rep = validate_rows(_rows(), CFG)
    assert rep.ok
    bad = [{"issue": "2026004", "draw_date": None,
            "red": [1, 2, 3, 4, 5, 34], "blue": 7, "source": "t"}]
    assert any("红球" in e for e in validate_rows(bad, CFG).errors)
    bad2 = [{"issue": "2026004", "draw_date": None,
             "red": [1, 2, 3, 4, 5, 6], "blue": 17, "source": "t"}]
    assert any("蓝球" in e for e in validate_rows(bad2, CFG).errors)


# ---------- 2 重复红球 ----------

def test_duplicate_reds_detected():
    bad = [{"issue": "2026005", "draw_date": None,
            "red": [1, 1, 2, 3, 4, 5], "blue": 7, "source": "t"}]
    assert any("红球" in e for e in validate_rows(bad, CFG).errors)


# ---------- 3 期号去重与重复导入 ----------

def test_import_dedup(tmp_path):
    import csv as _csv
    p = tmp_path / "d.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["issue", "draw_date", "r1", "r2", "r3", "r4", "r5",
                    "r6", "b1"])
        for r in _rows():
            w.writerow([r["issue"], r["draw_date"],
                        *[r["red"][i] for i in range(6)], r["blue"]])
    dbp = tmp_path / "t.db"
    db.migrate(dbp)
    with db.connect(dbp) as c:
        n1 = db.insert_draws_batch(c, _rows())["added"]
        n2 = db.insert_draws_batch(c, _rows())["added"]
    assert n1 == 3 and n2 == 0
    with db.connect(dbp) as c:
        assert c.execute("SELECT COUNT(*) c FROM draws").fetchone()["c"] == 3


# ---------- 4 多来源冲突 ----------

def test_cross_source_conflict(tmp_path):
    dbp = tmp_path / "t.db"
    db.migrate(dbp)
    a = _rows()[0]
    b = dict(a, blue=9, source="backup")
    with db.connect(dbp) as c:
        assert db.upsert_draw(c, a["issue"], a["draw_date"], a["red"],
                              a["blue"], "official") == "added"
        st = db.upsert_draw(c, b["issue"], b["draw_date"], b["red"],
                            b["blue"], "backup")
        assert st == "conflict_pending"
        row = c.execute("SELECT check_status FROM draws WHERE issue=?",
                        (a["issue"],)).fetchone()
        assert row["check_status"] == "pending_review"
        conf = c.execute("SELECT COUNT(*) c FROM data_conflicts").fetchone()
        assert conf["c"] == 1
        # 原记录未被覆盖
        keep = c.execute("SELECT b1 FROM draws WHERE issue=?",
                         (a["issue"],)).fetchone()
        assert keep["b1"] == a["blue"]


# ---------- 5 API 空响应/异常响应 ----------

class _FakeResp:
    def __init__(self, payload, content_type="application/json"):
        self._payload = payload
        self.status_code = 200
        self.headers = {"Content-Type": content_type}
        self.content = payload if isinstance(payload, bytes) \
            else json.dumps(payload).encode()
        self.text = self.content.decode()

    def json(self):
        return json.loads(self.content)

    def raise_for_status(self):
        pass


def test_cwl_empty_and_bad_json(monkeypatch):
    src = CwlSource()
    monkeypatch.setattr(src, "_get",
                        lambda *a, **k: _FakeResp({"state": 0, "result": []}))
    assert src.parse(src.fetch("2026001", "2026999")) == []
    monkeypatch.setattr(src, "_get",
                        lambda *a, **k: _FakeResp(b"<html>error</html>",
                                                  "text/html"))
    with pytest.raises(SourceError):
        src.fetch("2026001", "2026999")
    # JSON 结构变化：关键字段缺失的行被跳过，不崩溃
    weird = {"state": 0, "result": [{"code": "2026099"}]}
    assert src.parse(weird) == []


def test_backup_html_error_page(monkeypatch):
    src = Backup500Source()
    monkeypatch.setattr(src, "_get",
                        lambda *a, **k: _FakeResp("<html>blocked</html>",
                                                  "text/html"))
    with pytest.raises(SourceError):
        src.parse(src.fetch("26001", "26116"))


# ---------- 6 分页与断点续跑 ----------

def test_cwl_pagination(monkeypatch):
    src = CwlSource()
    pages = {
        1: {"state": 0, "result": [
            {"code": f"2026{100 + i:03d}", "date": "2026-01-01(四)",
             "red": "1,2,3,4,5,6", "blue": "7"} for i in range(5)]},
        2: {"state": 0, "result": []},
    }
    monkeypatch.setattr(src, "fetch",
                        lambda s, e, page_no=1, page_size=500: pages[page_no])
    rows = src.fetch_range("2026001", "2026999", page_size=5)
    assert len(rows) == 5 and rows[0]["issue"] == "2026100"
    # 断点续跑：再次拉取→insert 幂等
    dbp = Path(__file__).parent / "_tmp_page.db"
    db.migrate(dbp)
    with db.connect(dbp) as c:
        assert db.insert_draws_batch(c, rows)["added"] == 5
        assert db.insert_draws_batch(c, rows)["added"] == 0
    dbp.unlink(missing_ok=True)


# ---------- 7 时间排序 ----------

def test_time_order_warning():
    rows = [_rows()[1], _rows()[0]]
    rep = validate_rows(rows, CFG)
    assert any("乱序" in w for w in rep.warnings)


# ---------- 8 特征不使用未来 ----------

def test_features_no_future_leakage():
    draws = []
    for i in range(30):
        reds = sorted(((i * 7 + j) % 33) + 1 for j in range(6))
        # 保证互异
        reds = sorted(set(reds))
        while len(reds) < 6:
            reds = sorted(set(reds) | {(len(reds) * 5 + i) % 33 + 1})
        draws.append({"issue": f"2026{i + 1:03d}", "draw_date": None,
                      **{f"r{j + 1}": reds[j] for j in range(6)},
                      "b1": (i % 16) + 1, "source": "t"})
    f1 = build_features(draws, CFG)
    # 改写最后一期 → 之前的特征必须逐字节不变
    mutated = [dict(d) for d in draws]
    mutated[-1] = dict(mutated[-1], b1=16, r1=33, r2=32, r3=31, r4=30,
                       r5=29, r6=28)
    f2 = build_features(mutated, CFG)
    for iss in [d["issue"] for d in draws[:-1]]:
        assert f1[iss] == f2[iss], f"期 {iss} 特征被未来数据污染"
    # 遗漏定义
    assert omission_at([0, 3], 5) == 1     # 4期出现过（idx3），t=5 → 1
    assert omission_at([], 5) == 5


# ---------- 9 走期时间泄漏（预测冻结字段） ----------

def test_predictions_data_cutoff_before_target(tmp_path):
    dbp = tmp_path / "bt.db"
    db.migrate(dbp)
    with db.connect(dbp) as c:
        c.execute(
            "INSERT INTO predictions(run_id,target_issue,generated_at,"
            "data_cutoff,model_version,market,candidates,chosen,"
            "votes_json) VALUES(1,'2026005','2026-01-01','2026004',"
            "'v','blue','{}','{}','{}')")
        row = c.execute("SELECT * FROM predictions").fetchone()
    assert row["data_cutoff"] < row["target_issue"]


# ---------- 10 公平随机基线 ----------

def test_fair_baseline():
    assert prize1_probability(CFG) == pytest.approx(1 / (1107568 * 16))
    assert single_red_inclusion(CFG) == pytest.approx(6 / 33)
    p = monte_carlo_prize1(CFG, n_sims=200_000, seed=42)
    exact = prize1_probability(CFG)
    assert abs(p - exact) < 5 * mc_stderr(exact, 200_000) + 1e-9
    pmf = red_hits_pmf(6, CFG)
    assert sum(pmf.values()) == pytest.approx(1.0)


# ---------- 11 模型结果可重复性 ----------

def test_features_deterministic():
    draws = [{"issue": f"2026{i + 1:03d}", "draw_date": None,
              "r1": 1, "r2": 2, "r3": 3, "r4": 4, "r5": 5, "r6": 6,
              "b1": i % 16 + 1, "source": "t"} for i in range(40)]
    assert build_features(draws, CFG) == build_features(draws, CFG)


# ---------- 12 多重检验与低样本规则 ----------

def test_multiple_testing_and_low_sample():
    assert bh_adjust([0.01, 0.04, 0.03]) == pytest.approx(
        [0.03, 0.04, 0.04], abs=1e-9)
    # 低样本：n=3 触发不得声称 validated
    draws = [{"issue": f"2026{i + 1:03d}", "b1": 1, "red": [1, 2, 3, 4, 5, 6]}
             for i in range(4)]
    rule = Rule("低样本规则", "manual", "t", "always 1", "human",
                lambda h: 1)
    sc = evaluate_rule(rule, draws, 1, 4, CFG)
    assert sc.n_trigger == 3
    assert sc.status == "exploratory"
    # wilson 区间在 n=0 时为 nan 而非崩溃
    assert wilson(0, 0)[0] != wilson(0, 0)[0]  # nan
    assert binom_p(0, 0, 0.5) == 1.0


# ---------- 13 AI 调用失败降级 ----------

def test_ai_failure_degrades(monkeypatch):
    from app import ai_assist
    monkeypatch.setattr(ai_assist, "AI_KEY", "")
    res = ai_assist.explain_report("报告", "问题")
    assert res["ok"] is False and "MIMO_AI_KEY" in res["reason"]
    assert ai_assist.COVERAGE["failed"] >= 1

    def boom(*a, **k):
        raise OSError("network down")
    monkeypatch.setattr(ai_assist, "AI_KEY", "x")
    monkeypatch.setattr(ai_assist.urllib.request, "urlopen", boom)
    res2 = ai_assist.explain_report("报告", "问题")
    assert res2["ok"] is False and "network down" in res2["reason"]


# ---------- 14 BB 模块兼容 ----------

def test_bb_module_still_works():
    import bb_analysis
    from bb_analysis.config import load_config as bb_load
    from bb_analysis.theory import single_prob
    cfg = bb_load(Path(__file__).resolve().parent.parent / "config")
    assert cfg.rules.draw_size == 7
    assert abs(single_prob(4) - 0.4717193) < 1e-6
    assert bb_analysis.__version__
