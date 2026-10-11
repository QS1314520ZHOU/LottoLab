# -*- coding: utf-8 -*-
"""采集编排：discover / validate-sources / import-history / update。

- 初始化：官方年份分块 2013→今；再用备用源补 2003–2012。
- 增量：从库内最新期向后拉取，并回查最近 10 期发现修正。
- 断点续跑 = 全流程幂等（upsert + 期号去重），重复执行不重复计数。
- 冲突：只记 data_conflicts 并标记 pending_review，绝不覆盖可信记录。
"""

from __future__ import annotations

from datetime import date
from typing import Any

from . import db
from .config import SSQConfig
from .sources import (Backup500Source, CwlSource, SourceError,
                      ordered_sources)


def discover_sources(cfg: SSQConfig) -> list[dict[str, Any]]:
    """探测每个来源的可用性并登记 data_sources。"""
    db.migrate(cfg.db_path)
    results: list[dict[str, Any]] = []
    with db.connect(cfg.db_path) as conn:
        for src in ordered_sources():
            item = {"name": src.name, "url": src.url,
                    "official": src.is_official}
            try:
                probe = src.probe()
                item.update(status="verified", n=probe["n"],
                            sample_issue=probe["sample"]["issue"],
                            coverage=src.coverage_note)
                db.record_source(conn, {**src.meta(), "last_verified": db.utcnow()})
            except SourceError as exc:
                item.update(status="failed", error=str(exc))
                db.record_source(conn, {**src.meta(), "status": "failed",
                                        "last_verified": db.utcnow()})
            results.append(item)
    return results


def _cwl() -> CwlSource:
    return CwlSource()


def _b500() -> Backup500Source:
    return Backup500Source()


def import_history(cfg: SSQConfig, full: bool = True) -> dict[str, Any]:
    """初始化/全量导入：官方 2013+ 年份分块；备用源补 2003–2012。"""
    db.migrate(cfg.db_path)
    year = date.today().year
    summary: dict[str, Any] = {"mode": "full" if full else "fast",
                               "fetched": {}, "counts": {}}
    with db.connect(cfg.db_path) as conn:
        run_id = db.start_run(conn, "cwl_official+chart500_backup", "full")
        try:
            rows = _cwl().fetch_year_chunks(2013, year)
        except SourceError as exc:
            summary["cwl_error"] = str(exc)
            rows = []
        summary["fetched"]["cwl_official"] = len(rows)
        # 备用源补 2003–2012（失败不阻断官方数据）
        try:
            b_rows = _b500().parse(_b500().fetch("03001", f"{str(year)[2:]}999"))
            # 只保留官方覆盖不到的早期
            early = [r for r in b_rows if r["issue"] < "2013001"]
            summary["fetched"]["chart500_backup"] = len(early)
            rows = rows + early
        except SourceError as exc:
            summary["backup_error"] = str(exc)
        counts = db.insert_draws_batch(conn, rows)
        summary["counts"] = counts
        db.finish_run(conn, run_id,
                      {"fetched": len(rows), "added": counts["added"],
                       "updated": counts["conflict_pending"],
                       "skipped": counts["unchanged"],
                       "detail": summary})
    return summary


def update(cfg: SSQConfig, lookback: int = 10) -> dict[str, Any]:
    """增量：库内最新期→今（官方）；回查最近 lookback 期找修正。"""
    db.migrate(cfg.db_path)
    with db.connect(cfg.db_path) as conn:
        run_id = db.start_run(conn, "cwl_official", "incremental")
        rows_all = db.load_draws(conn, status=None)
        latest = max((r["issue"] for r in rows_all), default="2013001")
        year = date.today().year
        fetched: list[dict[str, Any]] = []
        note: dict[str, Any] = {"latest_before": latest}
        try:
            # 1) 最新期之后
            fetched += _cwl().fetch_range(latest, f"{year}999")
            # 2) 回查最近 lookback 期（发现修正）
            lo = int(latest[4:])
            start = f"{latest[:4]}{max(1, lo - lookback):03d}"
            fetched += _cwl().fetch_range(start, f"{year}999")
        except SourceError as exc:
            note["error"] = str(exc)
        # 去重（同 issue 可能被两次范围拉取重复）
        dedup = {r["issue"]: r for r in fetched}
        counts = db.insert_draws_batch(conn, list(dedup.values()))
        summary = {"mode": "incremental", "fetched": len(dedup),
                   "counts": counts, "note": note}
        db.finish_run(conn, run_id,
                      {"fetched": len(dedup), "added": counts["added"],
                       "updated": counts["conflict_pending"],
                       "skipped": counts["unchanged"],
                       "detail": summary})
        # 未开奖/未发布：官方无新期时 fetched 仅含回查 → 正常情况，不报错
        summary["new_issues"] = [i for i in dedup if i > latest]
        return summary


def cross_check(cfg: SSQConfig, sample_years: tuple[int, ...] = (2024, 2025)
                ) -> dict[str, Any]:
    """validate-sources：抽若干年度用两源比对，只记冲突不覆盖。"""
    db.migrate(cfg.db_path)
    result: dict[str, Any] = {"compared": 0, "mismatches": [],
                              "errors": []}
    try:
        off_rows = []
        for y in sample_years:
            off_rows += _cwl().fetch_range(f"{y}001", f"{y}999")
        off = {r["issue"]: r for r in off_rows}
        bak_rows = _b500().parse(
            _b500().fetch(f"{str(min(sample_years))[2:]}001",
                          f"{str(max(sample_years))[2:]}999"))
        bak = {r["issue"]: r for r in bak_rows}
    except SourceError as exc:
        result["errors"].append(str(exc))
        return result
    for issue in sorted(set(off) & set(bak)):
        result["compared"] += 1
        a, b = off[issue], bak[issue]
        if a["red"] != b["red"] or a["blue"] != b["blue"]:
            result["mismatches"].append(issue)
    # 顺手把官方数据落库（幂等）
    with db.connect(cfg.db_path) as conn:
        db.insert_draws_batch(conn, list(off.values()))
    return result
