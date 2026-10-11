# -*- coding: utf-8 -*-
"""V1.1 P1 测试：票据合法性/组合去重/预算同口径/黄金奖金表/
历史修正场景/报告来源分标一致性。"""

from __future__ import annotations

import pytest

from app import db
from app.config import load_config
from app.tickets import (Ticket, evaluate_budget, make_tickets,
                         prize_for, validate_blue, validate_reds)
from app.validate import report_database

CFG = load_config()


# ---------- 票据合法性 ----------

def test_ticket_validation():
    validate_reds([1, 2, 3, 4, 5, 6], CFG)
    validate_blue(16, CFG)
    with pytest.raises(ValueError):
        validate_reds([1, 1, 2, 3, 4, 5], CFG)      # 重复红
    with pytest.raises(ValueError):
        validate_reds([1, 2, 3, 4, 5, 34], CFG)     # 越界红
    with pytest.raises(ValueError):
        validate_reds([1, 2, 3, 4, 5], CFG)         # 数量不足
    with pytest.raises(ValueError):
        validate_blue(17, CFG)                       # 越界蓝


# ---------- 组合生成：预算/去重/成本 ----------

def test_make_tickets_budget_dedup_cost():
    out = make_tickets(CFG, 20.0, "random", None, seed=7)
    assert out["k"] == 10                      # 20 // 2
    assert out["cost"] == 20.0 and out["budget_left"] == 0.0
    keys = [t.key() for t in out["tickets"]]
    assert len(set(keys)) == 10                # 互异
    for t in out["tickets"]:
        validate_reds(list(t.reds), CFG)
        validate_blue(t.blue, CFG)
    # 同种子可复现
    out2 = make_tickets(CFG, 20.0, "random", None, seed=7)
    assert [t.key() for t in out2["tickets"]] == keys
    # 预算不足1注 → 明确报错
    with pytest.raises(ValueError):
        make_tickets(CFG, 1.0, "random", None)


def test_make_tickets_model_strategy():
    ranks = {"red": [(i, 0.1) for i in range(1, 34)],
             "blue": [(i, 0.06) for i in range(1, 17)]}
    out = make_tickets(CFG, 10.0, "model", ranks, seed=3)
    assert out["k"] == 5
    assert len({t.key() for t in out["tickets"]}) == 5
    blues = {t.blue for t in out["tickets"]}
    assert len(blues) >= 2                     # 蓝球轮换而非全同


# ---------- 黄金奖金表 ----------

def test_prize_golden_cases():
    assert prize_for(6, True, CFG) == 0.0      # 一等浮动→保守0
    assert prize_for(5, True, CFG) == 0.0      # 二等浮动→保守0
    assert prize_for(4, True, CFG) == 3000.0   # 三等
    assert prize_for(3, True, CFG) == 200.0    # 四等
    assert prize_for(2, True, CFG) == 10.0     # 五等
    assert prize_for(1, True, CFG) == 5.0      # 六等
    assert prize_for(0, True, CFG) == 5.0      # 六等（0+1）
    assert prize_for(4, False, CFG) == 0.0     # 蓝不中 → 无奖
    assert prize_for(2, False, CFG) == 0.0
    # 配置读取自 app.json
    assert CFG.ticket_price == 2.0
    assert CFG.prize_fixed["4plus1"] == 3000


# ---------- 同预算同周期对照 ----------

def test_same_budget_same_periods():
    draws = [{"issue": f"2099{i:04d}", "draw_date": None,
              **{f"r{j}": j for j in range(1, 7)}, "b1": 1}
             for i in range(30)]
    # 构造两组各30期、每期1注的票据
    t_model = [make_tickets(CFG, 2.0, "random", None, seed=s)["tickets"]
               for s in range(30)]
    t_rand = [make_tickets(CFG, 2.0, "random", None, seed=100 + s)["tickets"]
              for s in range(30)]
    rm = evaluate_budget(CFG, draws, t_model, "model")
    rr = evaluate_budget(CFG, draws, t_rand, "random")
    assert rm.total_cost == rr.total_cost == 60.0
    assert rm.periods == rr.periods == 30
    # 指标命名空间存在且红球分母一致
    assert rm.red_balls == rr.red_balls == 180


# ---------- 历史修正场景：不覆盖、记冲突、revision+1 ----------

def test_revision_conflict_not_overwrite(tmp_path):
    dbp = tmp_path / "rev.db"
    db.migrate(dbp)
    reds, blue = [1, 2, 3, 4, 5, 6], 7
    with db.connect(dbp) as c:
        assert db.upsert_draw(c, "2099001", None, reds, blue,
                              "cwl_official") == "added"
        # 官方“修正”：同期不同号码 → 冲突待核对，不覆盖原记录
        st = db.upsert_draw(c, "2099001", None, [7, 8, 9, 10, 11, 12], 9,
                            "cwl_official")
        assert st == "conflict_pending"
        row = c.execute("SELECT * FROM draws WHERE issue='2099001'").fetchone()
        assert row["b1"] == 7 and row["r1"] == 1      # 原数据保留
        assert row["check_status"] == "pending_review"
        assert row["revision"] == 1                    # 修订计数
        conf = c.execute("SELECT * FROM data_conflicts").fetchone()
        assert conf["b_blue"] == "9" and conf["a_blue"] == "7"


# ---------- 报告来源分标与一致性 ----------

def test_report_source_split_consistency():
    rep = report_database(CFG)
    with db.connect(CFG.db_path) as c:
        n = c.execute("SELECT COUNT(*) c FROM draws").fetchone()["c"]
    assert rep["total_rows"] == n == (
        rep["official_rows"] + rep["nonofficial_rows"])
    assert sum(rep["by_source"].values()) == n
    # 本工作区数据：官方段与非官方补齐段必须分别可计数
    assert rep["official_rows"] > 0 and rep["nonofficial_rows"] > 0
