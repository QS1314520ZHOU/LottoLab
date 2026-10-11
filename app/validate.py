# -*- coding: utf-8 -*-
"""双色球数据校验：范围、重复、格式、缺期提示、冲突汇总。

原则：期号不连续 ≠ 删除理由；错误进报告/冲突队列；禁止虚构数据补齐。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from .config import SSQConfig


@dataclass
class SSQReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "errors": self.errors,
                "warnings": self.warnings, "stats": self.stats}


def validate_rows(rows: list[dict[str, Any]], cfg: SSQConfig) -> SSQReport:
    rep = SSQReport()
    if not rows:
        rep.errors.append("无数据")
        return rep
    seen: set[str] = set()
    prev_issue: str | None = None
    bad_date = dup = 0
    for r in rows:
        issue = str(r.get("issue", ""))
        if issue in seen:
            dup += 1
            rep.errors.append(f"重复期号 {issue}")
        seen.add(issue)
        red = r.get("red") or []
        blue = r.get("blue")
        if (len(red) != cfg.red_count or len(set(red)) != cfg.red_count
                or not all(1 <= x <= cfg.red_pool for x in red)):
            rep.errors.append(f"{issue}: 红球非法 {red}")
        if not isinstance(blue, int) or not (1 <= blue <= cfg.blue_pool):
            rep.errors.append(f"{issue}: 蓝球非法 {blue}")
        d = r.get("draw_date")
        if d:
            try:
                date.fromisoformat(d)
            except ValueError:
                bad_date += 1
                rep.warnings.append(f"{issue}: 日期不可解析 {d}")
        if prev_issue is not None and issue < prev_issue:
            rep.warnings.append(f"{issue}: 期号乱序（前一期 {prev_issue}）")
        prev_issue = issue
    # 缺期提示（只提示不删除）
    ordered = sorted(seen)
    gaps = []
    for a, b in zip(ordered, ordered[1:]):
        if a[:4] == b[:4] and b.isdigit() and a.isdigit():
            if int(b[4:]) - int(a[4:]) > 1:
                gaps.append(f"{a}->{b}")
    if gaps:
        rep.warnings.append(
            f"同年度期号缺口 {len(gaps)} 处（可能缺期或历史规则，不删除）: "
            + ", ".join(gaps[:5]))
    rep.stats = {"n": len(rows), "unique": len(seen), "duplicates": dup,
                 "bad_date": bad_date, "range": [ordered[0], ordered[-1]]}
    return rep


def report_database(cfg: SSQConfig) -> dict[str, Any]:
    """report-data / validate-data 的库内统计。"""
    from . import db
    db.migrate(cfg.db_path)
    with db.connect(cfg.db_path) as conn:
        draws = db.load_draws(conn, status=None)
        ok = [d for d in draws if d["check_status"] == "ok"]
        pending = [d for d in draws if d["check_status"] == "pending_review"]
        conflicts = conn.execute(
            "SELECT COUNT(*) c FROM data_conflicts WHERE resolved='pending'"
        ).fetchone()["c"]
        runs = [dict(r) for r in conn.execute(
            "SELECT * FROM ingestion_runs ORDER BY id DESC LIMIT 5")]
        srcs = [dict(r) for r in conn.execute("SELECT * FROM data_sources")]
    years: dict[str, int] = {}
    for d in ok:
        years[d["issue"][:4]] = years.get(d["issue"][:4], 0) + 1
    by_source: dict[str, int] = {}
    with db.connect(cfg.db_path) as conn:
        for r in conn.execute(
                "SELECT source, COUNT(*) c FROM draws GROUP BY source"):
            by_source[r["source"]] = r["c"]
    return {
        "total_rows": len(draws),
        "valid_rows": len(ok),
        "pending_review": len(pending),
        "pending_conflicts": conflicts,
        "by_source": by_source,
        "official_rows": sum(c for s, c in by_source.items()
                             if "official" in s),
        "nonofficial_rows": sum(c for s, c in by_source.items()
                                if "official" not in s),
        "issue_range": [ok[0]["issue"], ok[-1]["issue"]] if ok else None,
        "per_year": dict(sorted(years.items())),
        "recent_runs": runs,
        "sources": srcs,
    }


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)
