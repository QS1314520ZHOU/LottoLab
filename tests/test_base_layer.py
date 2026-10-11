# -*- coding: utf-8 -*-
"""基础层测试：配置/映射/校验/规则结算/理论概率/组合枚举/幂等导入。"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from bb_analysis import database as db
from bb_analysis.combinations import combo_counts, enumerate_combos, zodiac_sizes
from bb_analysis.config import MappingConfig, MappingVersion, load_config
from bb_analysis.ingestion import import_csv_file, read_csv
from bb_analysis.mapping import map_draw, zodiac_present_set
from bb_analysis.rules import (breakeven_rate, combo_hit, expected_net,
                               settle, single_hit)
from bb_analysis.theory import (all_present_prob, monte_carlo_all_present,
                                single_prob)
from bb_analysis.validation import validate_draws

ROOT = Path(__file__).resolve().parent.parent
CFG = load_config(ROOT / "config")


def _draw(issue: str = "202601010001", **over) -> dict:
    row = {"issue": issue, "draw_time": None,
           "p1": 7, "p2": 19, "p3": 31, "p4": 43, "p5": 1, "p6": 12,
           "sp": 25, "source": "test", "is_simulated": 1}
    row.update(over)
    return row


# ---------- 配置 ----------

def test_config_current_year_odds_configured_from_evidence():
    """当前配置：马=当年生肖@1.76（截图证据），到期回落 default。"""
    assert CFG.odds.current_year_zodiac == "马"
    assert CFG.odds.single_odds("马", date(2026, 10, 11)) == (1.76, "ok")
    assert CFG.odds.single_odds("鼠", date(2026, 10, 11)) == (2.06, "ok")
    assert CFG.odds.single_odds("马", date(2027, 3, 1)) == (2.06, "ok")
    assert not CFG.warnings


def test_config_unconfigured_current_year_warns():
    """未配置当年生肖时：全部按 default 并显式告警（不得猜测）。"""
    cy = dict(current_year_zodiac=None)
    unconf = replace(CFG.odds, **cy)
    odds, status = unconf.single_odds("马")
    assert odds == 2.06 and status == "current_year_unconfigured"


def test_config_current_year_odds_when_set():
    odds_cfg = replace(
        CFG.odds,
        current_year_zodiac="马",
        current_year_odds=1.76,
        current_year_from=date(2026, 1, 1),
        current_year_to=date(2026, 12, 31),
    )
    assert odds_cfg.single_odds("马", date(2026, 6, 1)) == (1.76, "ok")
    assert odds_cfg.single_odds("鼠", date(2026, 6, 1)) == (2.06, "ok")
    assert odds_cfg.single_odds("马", date(2027, 1, 1)) == (2.06, "ok")


# ---------- 映射 ----------

def test_map_draw_matches_bb_platform_table():
    rows = map_draw(_draw(), CFG.mapping)
    assert len(rows) == 7
    by_num = {r[2]: r[3] for r in rows}
    assert by_num[7] == "鼠" and by_num[1] == "马" and by_num[25] == "马"
    assert by_num[12] == "羊"
    # 位置 1..6 正码，7 特别号
    assert [r[1] for r in rows] == [1, 2, 3, 4, 5, 6, 7]
    assert zodiac_present_set(rows) == {"鼠", "马", "羊"}


def test_mapping_version_by_date():
    v1 = MappingVersion("v1", "s", date(2026, 1, 1), date(2026, 6, 30),
                        {1: "马"})
    v2 = MappingVersion("v2", "s", date(2026, 7, 1), None, {1: "羊"})
    mc = MappingConfig((v1, v2))
    assert mc.for_date(date(2026, 3, 1)).version == "v1"
    assert mc.for_date(date(2026, 8, 1)).version == "v2"
    with pytest.raises(Exception):
        mc.for_date(date(2025, 1, 1))


def test_map_draw_missing_number_raises():
    bad = MappingConfig((MappingVersion("v", "s", None, None, {1: "马"}),))
    with pytest.raises(Exception):
        map_draw(_draw(), bad)


# ---------- 校验 ----------

def test_validation_ok_and_error_paths():
    rep = validate_draws([_draw()], CFG)
    assert rep.ok, rep.errors

    rep = validate_draws([_draw(), _draw()], CFG)
    assert any("重复期号" in e for e in rep.errors)

    rep = validate_draws([_draw(p1=99)], CFG)
    assert any("越界" in e for e in rep.errors)

    rep = validate_draws([_draw(p2=7)], CFG)  # 与 p1 重复
    assert any("号码重复" in e for e in rep.errors)

    rep = validate_draws([], CFG)
    assert not rep.ok and any("数据为空" in e for e in rep.errors)

    rep = validate_draws([_draw(issue="bad")], CFG)
    assert any("期号" in e for e in rep.errors)


# ---------- 规则结算 ----------

def test_settlement_and_ev_formulas():
    ret, net = settle(5.0, 2.06, hit=True)
    assert ret == pytest.approx(10.30) and net == pytest.approx(5.30)
    ret, net = settle(5.0, 1.76, hit=False)
    assert ret == 0.0 and net == -5.0
    assert expected_net(5.0, 0.5, 2.0) == pytest.approx(0.0)
    assert breakeven_rate(2.06) == pytest.approx(1 / 2.06)
    with pytest.raises(ValueError):
        settle(-1.0, 2.0, True)


def test_combo_rule_all_present_and_multiplicity():
    present = {"鼠", "马", "羊"}   # 马在该期出现多次也只算“出现”
    assert single_hit("鼠", present)
    assert combo_hit(["鼠", "马"], present)
    assert not combo_hit(["鼠", "虎"], present)
    assert combo_hit(["鼠"], present, "all_zodiacs_present")


# ---------- 理论概率 ----------

def test_single_prob_known_values():
    assert single_prob(4) == pytest.approx(0.4717193075, abs=1e-9)
    assert single_prob(5) == pytest.approx(0.5538963041, abs=1e-9)


def test_combo_prob_not_product_of_singles():
    p2 = all_present_prob([4, 4])
    singles = single_prob(4)
    assert p2 > 0
    # 相乘会丢掉“同时出现”的排斥项，必须明显不同
    assert abs(p2 - singles * singles) > 1e-3
    # 独立手算核对：P = 1 - 2P(A缺) + P(A、B都缺)
    c = math.comb
    hand = (1 - 2 * c(45, 7) / c(49, 7) + c(41, 7) / c(49, 7))
    assert p2 == pytest.approx(hand, rel=1e-12)


def test_theory_monte_carlo_cross_validation():
    exact = all_present_prob([4, 4])
    mc = monte_carlo_all_present([4, 4], n_sims=60_000, seed=42)
    assert abs(mc - exact) < 0.01
    exact5 = all_present_prob([5, 4, 4, 4, 4])
    mc5 = monte_carlo_all_present([5, 4, 4, 4, 4], n_sims=60_000, seed=7)
    assert abs(mc5 - exact5) < 0.01


# ---------- 组合枚举 ----------

def test_combo_enumeration_counts_and_odds():
    counts = combo_counts()
    assert counts == {1: 12, 2: 66, 3: 220, 4: 495, 5: 792}
    specs = enumerate_combos(CFG)
    assert len(specs) == sum(counts.values())
    singles = [s for s in specs if s.k == 1]
    # 2026 配置态：马@1.76（当年生肖），其余@2.06
    assert all(s.odds == (1.76 if s.members == ("马",) else 2.06)
               for s in singles)
    assert all(s.status == "ok" for s in singles)
    five = [s for s in specs if s.k == 5]
    assert len(five) == 792 and all(s.odds == 108.0 for s in five)
    # 单肖理论概率与独立计算一致
    m = next(s for s in singles if s.members == ("马",))
    assert m.p_theory == pytest.approx(single_prob(5), rel=1e-12)
    assert five[0].breakeven == pytest.approx(1 / 108.0)
    assert five[0].ev_per_unit == pytest.approx(
        expected_net(1.0, five[0].p_theory, 108.0))


def test_zodiac_sizes_from_mapping():
    sizes = zodiac_sizes(CFG)
    assert sum(sizes.values()) == 49
    assert sizes["马"] == 5 and sizes["鼠"] == 4


# ---------- 幂等导入 ----------

def test_idempotent_import(tmp_path):
    csv_path = tmp_path / "sim.csv"
    csv_path.write_text(
        "issue,p1,p2,p3,p4,p5,p6,sp\n"
        "202601010001,7,19,31,43,1,13,25\n"
        "202601010002,2,14,26,38,6,18,30\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "t.db"
    rep, counts = import_csv_file(CFG, csv_path, simulated=True,
                                  db_path=db_path)
    assert rep.ok and counts["new_draws"] == 2
    # 重复导入不重复计数
    rep2, counts2 = import_csv_file(CFG, csv_path, simulated=True,
                                    db_path=db_path)
    assert rep2.ok and counts2["new_draws"] == 0
    with db.connect(db_path) as conn:
        n = conn.execute("SELECT COUNT(*) c FROM draws").fetchone()["c"]
        assert n == 2
        nz = conn.execute("SELECT COUNT(*) c FROM draw_zodiacs").fetchone()["c"]
        assert nz == 14


def test_read_csv_requires_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("issue,p1\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_csv(bad)
