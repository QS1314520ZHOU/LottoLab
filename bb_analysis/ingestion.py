# -*- coding: utf-8 -*-
"""数据导入：统一 CSV 接口（含现有 bb_sync.py 采集产物）+ 幂等入库流程。

现有采集器 bb_sync.py 写出 data/bb_live_history.csv（issue,p1..p6,sp），
本模块直接以该格式为统一数据接口；可通过 source 参数标注来源，
模拟数据必须显式传 simulated=True，绝不与真实数据混淆。
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from . import database as db
from .config import AppConfig
from .mapping import map_draw, zodiac_present_set
from .validation import ValidationReport, validate_draws

REQUIRED = ["issue", "p1", "p2", "p3", "p4", "p5", "p6", "sp"]


def read_csv(path: str | Path, default_source: str = "csv",
             simulated: bool = False) -> list[dict[str, Any]]:
    """读取开奖 CSV。可选 source 列覆盖 default_source。"""
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return []
        missing = [c for c in REQUIRED if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV 缺少列: {missing}")
        for raw in reader:
            row: dict[str, Any] = {
                "issue": (raw.get("issue") or "").strip(),
                "draw_time": (raw.get("draw_time") or "").strip() or None,
            }
            for c in REQUIRED[1:]:
                row[c] = int(str(raw[c]).strip())
            row["source"] = (raw.get("source") or default_source).strip()
            row["is_simulated"] = int(
                simulated or "simulat" in row["source"].lower()
            )
            rows.append(row)
    return rows


def validate_and_import(
    cfg: AppConfig,
    rows: list[dict[str, Any]],
    db_path: Optional[str | Path] = None,
) -> tuple[ValidationReport, dict[str, int]]:
    """校验 → 迁移 → 幂等导入 draws + draw_zodiacs。

    返回 (校验报告, 计数{new_draws,new_zodiac_rows})。
    校验有错误时不写库。
    """
    rep = validate_draws(rows, cfg)
    counts = {"new_draws": 0, "new_zodiac_rows": 0}
    if not rep.ok:
        return rep, counts

    path = db_path or cfg.db_path
    db.migrate(path)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with db.connect(path) as conn:
        # 先落映射版本（draw_zodiacs 有外键依赖）
        for ver in cfg.mapping.versions:
            db.insert_mapping_version(
                conn, ver.version, ver.table,
                ver.effective_from.isoformat() if ver.effective_from else None,
                ver.effective_to.isoformat() if ver.effective_to else None,
                ver.source,
            )
        db_rows = []
        z_rows: list[tuple[str, int, int, str, str]] = []
        for r in rows:
            db_rows.append({
                "issue": r["issue"],
                "draw_time": r.get("draw_time"),
                "p1": r["p1"], "p2": r["p2"], "p3": r["p3"],
                "p4": r["p4"], "p5": r["p5"], "p6": r["p6"],
                "sp": r["sp"],
                "source": r.get("source", "csv"),
                "fetched_at": now,
                "check_status": "ok",
                "is_simulated": r.get("is_simulated", 0),
            })
            z_rows.extend(map_draw(r, cfg.mapping))
        counts["new_draws"] = db.insert_draws(conn, db_rows)
        # 只为已入库（含历史重复）期号写映射，INSERT OR IGNORE 保证幂等
        counts["new_zodiac_rows"] = db.insert_draw_zodiacs(conn, z_rows)
    return rep, counts


def import_csv_file(
    cfg: AppConfig,
    path: str | Path,
    simulated: bool = False,
    source: str = "csv",
    db_path: Optional[str | Path] = None,
) -> tuple[ValidationReport, dict[str, int]]:
    rows = read_csv(path, default_source=source, simulated=simulated)
    return validate_and_import(cfg, rows, db_path=db_path)


def present_sets(conn) -> dict[str, set[str]]:
    """issue -> 该期出现的生肖集合（组合命中/特征计算共用）。"""
    out: dict[str, set[str]] = {}
    for r in conn.execute(
        "SELECT issue, zodiac FROM draw_zodiacs ORDER BY issue, position"
    ):
        out.setdefault(r["issue"], set()).add(r["zodiac"])
    return out


__all__ = [
    "read_csv", "validate_and_import", "import_csv_file",
    "present_sets", "zodiac_present_set",
]
