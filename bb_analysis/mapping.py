# -*- coding: utf-8 -*-
"""生肖映射：按开奖日期选择生效版本，展开 draw_zodiacs。

同一生肖在一期出现多次时，draw_zodiacs 按号码/位置逐条保留原始次数，
供“按生肖是否出现”与“按号码次数”两种统计口径分别使用（禁止混用）。
"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional

from .config import MappingConfig, MappingVersion


class MappingError(Exception):
    """映射缺失或冲突。"""


def resolve_version(mapping: MappingConfig, on: Optional[date]) -> MappingVersion:
    return mapping.for_date(on)


def map_draw(
    draw: dict[str, Any],
    mapping: MappingConfig,
) -> list[tuple[str, int, int, str, str]]:
    """把一期开奖展开为 draw_zodiacs 行。

    返回 (issue, position, number, zodiac, mapping_version) 列表；
    position 1..6 为正码，7 为特别号。
    """
    draw_date: Optional[date] = None
    if draw.get("draw_time"):
        draw_date = date.fromisoformat(str(draw["draw_time"])[:10])
    elif draw.get("issue"):
        iss = str(draw["issue"])
        if len(iss) >= 8 and iss[:8].isdigit():
            draw_date = date(int(iss[:4]), int(iss[4:6]), int(iss[6:8]))
    ver = resolve_version(mapping, draw_date)

    rows: list[tuple[str, int, int, str, str]] = []
    numbers = [int(draw[f"p{i}"]) for i in range(1, 7)] + [int(draw["sp"])]
    for pos, num in enumerate(numbers, start=1):
        z = ver.table.get(num)
        if z is None:
            raise MappingError(
                f"期号 {draw['issue']} 号码 {num} 不在映射版本 {ver.version} 中"
            )
        rows.append((str(draw["issue"]), pos, num, z, ver.version))
    return rows


def zodiac_present_set(
    rows: list[tuple[str, int, int, str, str]],
) -> set[str]:
    """该期出现过的生肖集合（组合命中判断用，同一生肖多次只算出现）。"""
    return {r[3] for r in rows}
