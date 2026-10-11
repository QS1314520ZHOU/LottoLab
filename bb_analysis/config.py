# -*- coding: utf-8 -*-
"""集中配置加载：app.json / odds.json / mapping.json + 环境变量覆盖。

所有与平台规则相关的参数（赔率、当年生肖、映射版本、是否允许重号）均来自
配置文件，代码中不硬编码平台规则；缺失时显式标记“待配置”并产生告警。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ConfigError(Exception):
    """配置缺失或非法。"""


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"配置文件不存在: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    return date.fromisoformat(value)


@dataclass(frozen=True)
class RuleConfig:
    """开奖与中奖基础规则。"""

    regular_count: int
    special_count: int
    numbers_can_repeat: bool
    repeat_rule_source: str
    combo_win_rule: str

    @property
    def draw_size(self) -> int:
        return self.regular_count + self.special_count


@dataclass(frozen=True)
class OddsConfig:
    """赔率配置（总返还倍数，含本金），带版本与生效期。"""

    version: str
    effective_from: Optional[date]
    default_single: float
    current_year_zodiac: Optional[str]
    current_year_odds: float
    current_year_from: Optional[date]
    current_year_to: Optional[date]
    combo_odds: dict[int, float]
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def single_odds(self, zodiac: str, on: Optional[date] = None) -> tuple[float, str]:
        """返回 (赔率, 状态)。状态 ok / current_year_unconfigured。"""
        if (
            self.current_year_zodiac
            and zodiac == self.current_year_zodiac
            and (on is None or (self.current_year_from or date.min) <= on)
            and (on is None or (self.current_year_to or date.max) >= on)
        ):
            return self.current_year_odds, "ok"
        return self.default_single, (
            "ok" if self.current_year_zodiac else "current_year_unconfigured"
        )

    def combo_odds_for(self, k: int) -> float:
        if k < 2 or k not in self.combo_odds:
            raise ConfigError(f"k={k} 的组合赔率未配置")
        return self.combo_odds[k]


@dataclass(frozen=True)
class MappingVersion:
    version: str
    source: str
    effective_from: Optional[date]
    effective_to: Optional[date]
    table: dict[int, str]

    def active_on(self, d: Optional[date]) -> bool:
        if d is None:
            return self.effective_from is None
        if self.effective_from and d < self.effective_from:
            return False
        if self.effective_to and d > self.effective_to:
            return False
        return True


@dataclass(frozen=True)
class MappingConfig:
    versions: tuple[MappingVersion, ...]

    def for_date(self, d: Optional[date]) -> MappingVersion:
        for v in self.versions:
            if v.active_on(d):
                return v
        raise ConfigError(f"日期 {d} 没有生效的生肖映射版本")

    def all_numbers(self) -> set[int]:
        nums: set[int] = set()
        for v in self.versions:
            nums |= set(v.table)
        return nums


@dataclass(frozen=True)
class AppConfig:
    windows: tuple[int, ...]
    seed: int
    db_path: Path
    config_dir: Path
    rules: RuleConfig
    odds: OddsConfig
    mapping: MappingConfig
    backtest: dict[str, Any]
    prior_strength: float
    bankroll: dict[str, Any]
    warnings: tuple[str, ...]

    @property
    def number_range(self) -> tuple[int, int]:
        nums = self.mapping.all_numbers()
        return min(nums), max(nums)


def load_config(config_dir: Optional[str | Path] = None) -> AppConfig:
    """从配置目录加载全部配置；支持 BB_DB_PATH / BB_SEED 环境变量覆盖。"""
    base = Path(
        config_dir
        or os.getenv("BB_CONFIG_DIR")
        or (Path(__file__).resolve().parent.parent / "config")
    )
    app = _load_json(base / "app.json")
    odds_raw = _load_json(base / "odds.json")
    map_raw = _load_json(base / "mapping.json")

    rules = RuleConfig(
        regular_count=int(app["rules"]["regular_count"]),
        special_count=int(app["rules"]["special_count"]),
        numbers_can_repeat=bool(app["rules"]["numbers_can_repeat_within_draw"]),
        repeat_rule_source=str(app["rules"].get("repeat_rule_source", "")),
        combo_win_rule=str(app["rules"]["combo_win_rule"]),
    )

    cy = odds_raw["single"]["current_year"]
    warnings: list[str] = []
    if not cy.get("zodiac"):
        warnings.append(
            "OddsConfig: 当年生肖未配置（odds.json single.current_year.zodiac=null），"
            "全部单肖暂按 default 赔率结算，1.76 档未启用"
        )
    odds = OddsConfig(
        version=odds_raw["version"],
        effective_from=_parse_date(odds_raw.get("effective_from")),
        default_single=float(odds_raw["single"]["default"]),
        current_year_zodiac=cy.get("zodiac"),
        current_year_odds=float(cy.get("odds", 1.76)),
        current_year_from=_parse_date(cy.get("effective_from")),
        current_year_to=_parse_date(cy.get("effective_to")),
        combo_odds={int(k): float(v) for k, v in odds_raw["combo"].items()},
        warnings=tuple(warnings),
    )

    versions = []
    for v in map_raw["versions"]:
        versions.append(
            MappingVersion(
                version=v["version"],
                source=v["source"],
                effective_from=_parse_date(v.get("effective_from")),
                effective_to=_parse_date(v.get("effective_to")),
                table={int(k): str(val) for k, val in v["map"].items()},
            )
        )
    if not versions:
        raise ConfigError("mapping.json 未定义任何映射版本")
    mapping = MappingConfig(versions=tuple(versions))

    db_rel = os.getenv("BB_DB_PATH") or app.get("db_path", "data/bb_analysis.db")
    project_root = Path(__file__).resolve().parent.parent
    db_path = Path(db_rel)
    if not db_path.is_absolute():
        db_path = project_root / db_path

    seed = int(os.getenv("BB_SEED") or app.get("seed", 42))

    return AppConfig(
        windows=tuple(int(w) for w in app["windows"]),
        seed=seed,
        db_path=db_path,
        config_dir=base,
        rules=rules,
        odds=odds,
        mapping=mapping,
        backtest=dict(app.get("backtest", {})),
        prior_strength=float(app.get("prior_strength", 20.0)),
        bankroll=dict(app.get("bankroll", {})),
        warnings=tuple(warnings),
    )
