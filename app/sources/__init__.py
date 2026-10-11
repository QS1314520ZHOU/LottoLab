# -*- coding: utf-8 -*-
"""数据源注册与自动选择：优先官方，故障切换备用；冲突只记录不覆盖。"""

from __future__ import annotations

from .backup500 import Backup500Source
from .base import SourceAdapter, SourceError
from .cwl import CwlSource

SOURCES: dict[str, type[SourceAdapter]] = {
    CwlSource.name: CwlSource,
    Backup500Source.name: Backup500Source,
}

# 自动选择优先级（官方优先）
PRIORITY = [CwlSource.name, Backup500Source.name]


def get_source(name: str) -> SourceAdapter:
    if name not in SOURCES:
        raise KeyError(f"未知数据源 {name}，可选 {list(SOURCES)}")
    return SOURCES[name]()


def ordered_sources() -> list[SourceAdapter]:
    return [get_source(n) for n in PRIORITY]
