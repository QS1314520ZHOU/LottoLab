# -*- coding: utf-8 -*-
"""V1.1 P2 测试：BB 赔率证据链 / 映射与赔率日期边界 /
黄金结算用例 / 资金策略极值边界。"""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest

from bb_analysis.bankroll import FixedStake, ThreePeriod, simulate
from bb_analysis.combinations import enumerate_combos
from bb_analysis.config import MappingConfig, MappingVersion, load_config
from bb_analysis.rules import breakeven_rate, combo_hit, expected_net, settle

CFG = load_config()


# ---------- 赔率证据链 ----------

def test_odds_evidence_chain():
    ev = CFG.odds.evidence
    assert "screenshot" in ev["single_default"]      # 截图证据
    assert "screenshot" in ev["current_year"]
    assert "user_spec" in ev["combo"] and "待确认" in ev["combo"]
    # 连肖待确认必须冒泡成告警（报告可见）
    assert any("待确认" in w for w in CFG.warnings)
    # 未取得证据的赔率不会被“自动修正”：combo 值保持任务书原值
    assert CFG.odds.combo_odds == {2: 4.2, 3: 10.6, 4: 30.7, 5: 108.0}


# ---------- 日期边界：映射生效日 / 赔率年切换 ----------

def test_mapping_effective_boundary():
    v1 = MappingVersion("v1", "s", date(2026, 1, 1), date(2026, 6, 30),
                        {1: "马"})
    v2 = MappingVersion("v2", "s", date(2026, 7, 1), None, {1: "羊"})
    mc = MappingConfig((v1, v2))
    assert mc.for_date(date(2026, 1, 1)).version == "v1"       # 生效日当天
    assert mc.for_date(date(2026, 6, 30)).version == "v1"   # 失效日当天 = v1
    assert mc.for_date(date(2026, 7, 1)).version == "v2"    # 新版首日 = v2
    with pytest.raises(Exception):
        mc.for_date(date(2025, 12, 31))                     # 生效前无版本


def test_odds_year_switch_boundary():
    assert CFG.odds.single_odds("马", date(2026, 12, 31)) == (1.76, "ok")
    assert CFG.odds.single_odds("马", date(2027, 1, 1)) == (2.06, "ok")
    assert CFG.odds.single_odds("马", date(2026, 1, 1)) == (1.76, "ok")


# ---------- 黄金结算用例（已知输入→已知输出，独立手算） ----------

def test_golden_settlement_cases():
    cases = [
        # (stake, odds, hit, 期望总返还, 期望净盈亏)
        (5.0, 2.06, True, 10.30, 5.30),     # 4号肖中
        (5.0, 1.76, True, 8.80, 3.80),      # 马中
        (5.0, 2.06, False, 0.0, -5.0),
        (5.0, 1.76, False, 0.0, -5.0),
        (5.0, 108.0, True, 540.0, 535.0),   # 五连肖中
        (5.0, 4.2, True, 21.0, 16.0),       # 二连肖中
        (0.0, 2.06, True, 0.0, 0.0),        # 零注额边界
    ]
    for stake, odds, hit, exp_ret, exp_net in cases:
        ret, net = settle(stake, odds, hit)
        assert ret == pytest.approx(exp_ret), (stake, odds, hit)
        assert net == pytest.approx(exp_net), (stake, odds, hit)
    # EV 与盈亏平衡（独立手算）
    assert expected_net(10.0, 0.5, 2.0) == pytest.approx(0.0)
    assert expected_net(5.0, 0.4717193075, 2.06) == pytest.approx(
        5.0 * (0.4717193075 * 2.06 - 1.0))
    assert breakeven_rate(4.2) == pytest.approx(1 / 4.2)
    # 中奖判定黄金用例：全部出现才算中（可配置规则）
    present = {"鼠", "马", "羊"}
    assert combo_hit(["鼠", "马"], present) is True
    assert combo_hit(["鼠", "龙"], present) is False
    # 连肖赔率对应的理论盈亏平衡（读配置）
    specs = enumerate_combos(CFG, ks=(5,))
    five = next(s for s in specs if s.k == 5)
    assert five.odds == 108.0
    assert five.breakeven == pytest.approx(1 / 108.0)


# ---------- 资金策略极值边界 ----------

def test_fund_zero_and_exact_stake_boundary():
    # 边界1：余额恰好等于注额 → 可继续；打完归零 → 下一期破产
    path = [{"hit": False, "odds": 2.06}] * 3
    led = simulate(path, FixedStake(5.0), initial=10.0)
    assert led.balances == [5.0, 0.0]      # 两期后归零
    assert led.ruin_at == 3                # 第3期余额0<5 → 无法继续

    # 边界2：初始资金=注额，首期未中即归零
    led0 = simulate(path, FixedStake(5.0), initial=5.0)
    assert led0.balances == [0.0] and led0.ruin_at == 2

    # 边界3：资金<注额起步 → 立即破产
    ledx = simulate(path, FixedStake(5.0), initial=4.99)
    assert ledx.ruin_at == 1 and ledx.net == 0.0


def test_extreme_drawdown_and_consec_loss():
    # 已知路径：胜-负-负-负-胜 → MDD 与连亏手算
    path = [{"hit": True, "odds": 2.06},
            {"hit": False, "odds": 2.06},
            {"hit": False, "odds": 2.06},
            {"hit": False, "odds": 2.06},
            {"hit": True, "odds": 2.06}]
    led = simulate(path, FixedStake(5.0), initial=100.0)
    # 净额: +5.3, -5, -5, -5, +5.3 → 余额 105.3,100.3,95.3,90.3,95.6
    assert [round(b, 2) for b in led.balances] == \
        [105.3, 100.3, 95.3, 90.3, 95.6]
    assert led.max_consec_loss == 3
    # 峰谷: peak105.3 → trough90.3 → MDD=15.0
    assert led.max_drawdown == pytest.approx(15.0)
    assert led.net == pytest.approx(5.3 * 2 - 15)

    # 三期策略极端：连续未中触发翻倍后的注额上限与垫付
    p2 = [{"hit": False, "odds": 2.06}] * 3
    led2 = simulate(p2, ThreePeriod(5.0), initial=100.0)
    assert led2.stakes == [5.0, 10.0, 5.0]       # 周期内注额序列
    assert led2.net == -20.0                      # -5-10-5
    assert led2.max_consec_loss == 3


def test_fund_strategies_hit_flags_unchanged_by_stakes():
    """资金策略不改变命中序列（同信号公平对比）。"""
    path = [{"hit": bool(i % 3), "odds": 2.06} for i in range(12)]
    a = simulate(path, FixedStake(5.0), 1000.0)
    b = simulate(path, ThreePeriod(5.0), 1000.0)
    assert a.hit_flags == b.hit_flags
    assert a.total_staked != b.total_staked      # 但流水不同
