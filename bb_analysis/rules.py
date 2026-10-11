# -*- coding: utf-8 -*-
"""中奖规则与赔率结算（规则独立封装，便于按平台真实规则修订）。

赔率 O 为总返还倍数（含本金）：
    中奖净收益 = stake * (O - 1)
    未中净收益 = -stake
    期望净收益 = stake * (p * O - 1)
    盈亏平衡命中率 = 1 / O
"""

from __future__ import annotations

from typing import Callable, Iterable

from .config import OddsConfig

# 组合中奖规则注册表：名字 -> 判定函数
# 判定函数签名: (所选生肖集合, 该期出现生肖集合) -> bool
ComboRule = Callable[[frozenset[str], frozenset[str]], bool]


def rule_all_zodiacs_present(selected: frozenset[str],
                             present: frozenset[str]) -> bool:
    """当前默认（待平台确认）：所选全部生肖都至少出现于7个号码中。"""
    return selected <= present


COMBO_RULES: dict[str, ComboRule] = {
    "all_zodiacs_present": rule_all_zodiacs_present,
}


def get_combo_rule(name: str) -> ComboRule:
    if name not in COMBO_RULES:
        raise KeyError(f"未注册的中奖规则: {name}（可选 {list(COMBO_RULES)}）")
    return COMBO_RULES[name]


def settle(stake: float, odds: float, hit: bool) -> tuple[float, float]:
    """返回 (总返还, 净盈亏)。"""
    if stake < 0:
        raise ValueError("stake 不能为负")
    if odds <= 0:
        raise ValueError("odds 必须为正")
    ret = stake * odds if hit else 0.0
    return ret, ret - stake


def expected_net(stake: float, p: float, odds: float) -> float:
    """期望净收益 s*(p*O-1)。"""
    return stake * (p * odds - 1.0)


def breakeven_rate(odds: float) -> float:
    """盈亏平衡命中率 1/O。"""
    return 1.0 / odds


def single_hit(zodiac: str, present: Iterable[str]) -> bool:
    return zodiac in set(present)


def combo_hit(selected: Iterable[str], present: Iterable[str],
              rule_name: str = "all_zodiacs_present") -> bool:
    return get_combo_rule(rule_name)(frozenset(selected), frozenset(present))


def odds_for_single(odds_cfg: OddsConfig, zodiac: str, on=None) -> float:
    return odds_cfg.single_odds(zodiac, on)[0]
