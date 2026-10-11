# -*- coding: utf-8 -*-
"""开奖数据校验：期号、号码范围、格式、映射完整性、缺期/重复期。

结果汇总为 ValidationReport（可 JSON 序列化），禁止用虚构数据修补错误。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

from .config import AppConfig
from .mapping import map_draw


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats,
        }


def _issue_day(issue: str) -> Optional[date]:
    if len(issue) >= 8 and issue[:8].isdigit():
        try:
            return date(int(issue[:4]), int(issue[4:6]), int(issue[6:8]))
        except ValueError:
            return None
    return None


def validate_draws(draws: list[dict[str, Any]], cfg: AppConfig) -> ValidationReport:
    """校验导入前的原始开奖行。draws 需含 issue/p1..p6/sp 字段。"""
    rep = ValidationReport()
    lo, hi = cfg.number_range
    seen: set[str] = set()
    dup_numbers_period = 0
    prev_day: Optional[date] = None
    prev_seq: Optional[int] = None

    if not draws:
        rep.errors.append("数据为空：没有任何开奖记录")
        return rep

    for i, d in enumerate(draws):
        issue = str(d.get("issue", ""))
        if not issue:
            rep.errors.append(f"第{i + 1}行缺少期号")
            continue
        if issue in seen:
            rep.errors.append(f"重复期号: {issue}")
        seen.add(issue)

        day = _issue_day(issue)
        if day is None:
            rep.errors.append(f"期号无法解析出日期: {issue}")
        else:
            if prev_day == day and prev_seq is not None:
                seq = int(issue[8:]) if issue[8:].isdigit() else None
                if seq is not None and seq != prev_seq + 1:
                    rep.warnings.append(
                        f"{day} 期号不连续: {prev_seq} -> {seq}（可能缺期）"
                    )
                if seq is not None:
                    prev_seq = seq
            elif prev_day is not None and day != prev_day:
                prev_seq = None
            if prev_day is not None and day < prev_day:
                rep.errors.append(f"期号时间倒序: {issue}")
            prev_day = day
            if issue[8:].isdigit() and prev_seq is None:
                prev_seq = int(issue[8:])

        try:
            nums = [int(d[f"p{j}"]) for j in range(1, 7)]
            sp = int(d["sp"])
        except (KeyError, TypeError, ValueError) as exc:
            rep.errors.append(f"{issue}: 正码/特别号格式错误 ({exc})")
            continue

        all_nums = nums + [sp]
        if len(all_nums) != cfg.rules.draw_size:
            rep.errors.append(
                f"{issue}: 号码数量 {len(all_nums)} != {cfg.rules.draw_size}"
            )
        bad = [n for n in all_nums if not (lo <= n <= hi)]
        if bad:
            rep.errors.append(f"{issue}: 号码越界 {bad}（允许 {lo}..{hi}）")
        if len(set(all_nums)) != len(all_nums):
            dup_numbers_period += 1
            if not cfg.rules.numbers_can_repeat:
                rep.errors.append(
                    f"{issue}: 单期内号码重复 {sorted(set(all_nums))} "
                    f"（规则 numbers_can_repeat=false；如平台允许请改配置）"
                )

        # 映射完整性（用期号日期选版本）
        try:
            map_draw(d, cfg.mapping)
        except Exception as exc:  # MappingError / ConfigError
            rep.errors.append(f"{issue}: 映射失败 {exc}")

    rep.stats = {
        "n_draws": len(draws),
        "n_unique": len(seen),
        "dup_number_periods": dup_numbers_period,
        "issue_first": draws[0].get("issue"),
        "issue_last": draws[-1].get("issue"),
    }
    if cfg.odds.warnings:
        rep.warnings.extend(cfg.odds.warnings)
    return rep
