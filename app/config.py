# -*- coding: utf-8 -*-
"""双色球系统配置：路径、窗口、种子、限速与环境变量。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_WINDOWS = (10, 20, 50, 100)


@dataclass(frozen=True)
class SSQConfig:
    db_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("SSQ_DB_PATH", str(PROJECT_ROOT / "data" / "ssq.db")))
    )
    seed: int = field(default_factory=lambda: int(os.getenv("SSQ_SEED", "42")))
    windows: tuple[int, ...] = DEFAULT_WINDOWS
    request_timeout: float = 20.0
    min_request_interval: float = 1.0   # 秒，页间最小间隔
    max_retries: int = 2
    # 号码空间（标准双色球规则）
    red_count: int = 6
    red_pool: int = 33
    blue_count: int = 1
    blue_pool: int = 16
    # 回测
    min_train: int = 100
    test_ratio: float = 0.3
    # 规律挖掘
    min_support: int = 30
    alpha: float = 0.05
    # 票据/奖金（来自 app.json ticket 节，官方公开规则）
    ticket_price: float = field(
        default_factory=lambda: float(
            (_load_ticket()) .get("price_yuan", 2.0)))
    prize_fixed: dict = field(
        default_factory=lambda: dict(
            (_load_ticket()).get("prize_fixed", {})))

    @property
    def first_prize_denominator(self) -> int:
        from math import comb
        return comb(self.red_pool, self.red_count) * (self.blue_pool ** self.blue_count)


def _load_ticket() -> dict:
    import json as _json
    path = PROJECT_ROOT / "config" / "app.json"
    try:
        with open(path, encoding="utf-8") as f:
            return _json.load(f).get("ticket", {})
    except (OSError, ValueError):
        return {}


def load_config() -> SSQConfig:
    return SSQConfig()
