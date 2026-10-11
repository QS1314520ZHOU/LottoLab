# -*- coding: utf-8 -*-
"""SQLite 数据层（双色球）：10 张表、迁移、事务写入。

表清单：draws / data_sources / ingestion_runs / data_conflicts /
features / pattern_definitions / predictions / backtest_runs /
model_versions / experiment_results。
原始开奖数据只增不改；派生数据独立存放。
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

SCHEMA_VERSION = 1

_MIGRATION_1 = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS draws (
    issue        TEXT PRIMARY KEY,
    draw_date    TEXT,                 -- YYYY-MM-DD
    r1 INTEGER NOT NULL, r2 INTEGER NOT NULL, r3 INTEGER NOT NULL,
    r4 INTEGER NOT NULL, r5 INTEGER NOT NULL, r6 INTEGER NOT NULL,
    b1           INTEGER NOT NULL,
    source       TEXT NOT NULL,
    first_seen   TEXT NOT NULL,
    last_checked TEXT NOT NULL,
    check_status TEXT NOT NULL DEFAULT 'ok',   -- ok|pending_review|rejected
    revision     INTEGER NOT NULL DEFAULT 0,   -- 同期被修正则 +1
    notes        TEXT
);

CREATE TABLE IF NOT EXISTS data_sources (
    name          TEXT PRIMARY KEY,
    url           TEXT NOT NULL,
    is_official   INTEGER NOT NULL,
    needs_api_key INTEGER NOT NULL DEFAULT 0,
    paid          INTEGER NOT NULL DEFAULT 0,
    terms_notes   TEXT,
    last_verified TEXT,
    coverage_note TEXT,
    status        TEXT     -- verified|failed|unverified
);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    mode   TEXT NOT NULL,          -- full|incremental|validate
    started_at TEXT NOT NULL,
    finished_at TEXT,
    fetched INTEGER DEFAULT 0,
    added    INTEGER DEFAULT 0,
    updated  INTEGER DEFAULT 0,
    skipped  INTEGER DEFAULT 0,
    failed   INTEGER DEFAULT 0,
    detail   TEXT
);

CREATE TABLE IF NOT EXISTS data_conflicts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    issue TEXT NOT NULL,
    detected_at TEXT NOT NULL,
    a_source TEXT, a_red TEXT, a_blue TEXT,
    b_source TEXT, b_red TEXT, b_blue TEXT,
    resolved TEXT NOT NULL DEFAULT 'pending',   -- pending|kept_a|kept_b|manual
    resolver_note TEXT
);

CREATE TABLE IF NOT EXISTS features (
    issue TEXT NOT NULL,
    feature_version TEXT NOT NULL,
    payload TEXT NOT NULL,          -- JSON：该期截止开奖前可算的全部特征
    computed_at TEXT NOT NULL,
    PRIMARY KEY (issue, feature_version)
);

CREATE TABLE IF NOT EXISTS pattern_definitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    category TEXT NOT NULL,         -- frequency|cross|structure|manual|metaphysics|auto
    version TEXT NOT NULL,
    formula TEXT NOT NULL,          -- 可执行公式/参数 JSON
    source TEXT NOT NULL,           -- human|auto|ai
    created_at TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    UNIQUE(name, version)
);

CREATE TABLE IF NOT EXISTS model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    params TEXT NOT NULL,
    train_range TEXT,
    feature_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(name, version)
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    target_issue TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    data_cutoff TEXT NOT NULL,      -- 必须 < target_issue 对应期
    model_version TEXT NOT NULL,
    market TEXT NOT NULL,           -- red|blue
    candidates TEXT NOT NULL,       -- JSON：各算法原始分数/候选/权重
    chosen TEXT NOT NULL,           -- JSON：最终候选
    abstain_reason TEXT,
    votes_json TEXT NOT NULL,
    UNIQUE(run_id, target_issue, market, model_version)
);

CREATE TABLE IF NOT EXISTS backtest_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    params TEXT NOT NULL,
    data_start TEXT NOT NULL,
    data_end TEXT NOT NULL,
    seed INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    summary TEXT
);

CREATE TABLE IF NOT EXISTS experiment_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    pattern_id INTEGER,
    test_name TEXT NOT NULL,
    n_tests INTEGER NOT NULL,       -- 多重检验总数
    p_value REAL,
    p_adjusted REAL,
    effect TEXT,
    conclusion TEXT NOT NULL,       -- exploratory|validated|rejected
    created_at TEXT NOT NULL,
    detail TEXT
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def migrate(db_path: str | Path) -> int:
    with connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)")
        applied = {r["version"] for r in
                   conn.execute("SELECT version FROM schema_migrations")}
        if 1 not in applied:
            conn.executescript(_MIGRATION_1)
            conn.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES(1,?)",
                (utcnow(),))
        return SCHEMA_VERSION


# ---------- draws ----------

def _draw_row(issue: str, draw_date: Optional[str], reds: list[int],
              blue: int, source: str, status: str = "ok",
              notes: Optional[str] = None) -> dict[str, Any]:
    now = utcnow()
    r = sorted(int(x) for x in reds)
    return dict(issue=issue, draw_date=draw_date,
                r1=r[0], r2=r[1], r3=r[2], r4=r[3], r5=r[4], r6=r[5],
                b1=int(blue), source=source, first_seen=now,
                last_checked=now, check_status=status, revision=0,
                notes=notes)


def upsert_draw(conn: sqlite3.Connection, issue: str,
                draw_date: Optional[str], reds: list[int], blue: int,
                source: str) -> str:
    """插入或核对一期。返回 'added'|'unchanged'|'revised'|'conflict_pending'。"""
    row = _draw_row(issue, draw_date, reds, blue, source)
    existing = conn.execute("SELECT * FROM draws WHERE issue=?",
                            (issue,)).fetchone()
    if existing is None:
        conn.execute(
            "INSERT INTO draws(issue,draw_date,r1,r2,r3,r4,r5,r6,b1,"
            "source,first_seen,last_checked,check_status,revision,notes)"
            " VALUES(:issue,:draw_date,:r1,:r2,:r3,:r4,:r5,:r6,:b1,"
            " :source,:first_seen,:last_checked,:check_status,:revision,:notes)",
            row)
        return "added"
    same = all(existing[f"r{i}"] == row[f"r{i}"] for i in range(1, 7)) \
        and existing["b1"] == row["b1"]
    if same:
        conn.execute("UPDATE draws SET last_checked=? WHERE issue=?",
                     (utcnow(), issue))
        return "unchanged"
    # 数据不一致：绝不静默覆盖 → 记冲突 + 标记待核对，保留原记录
    conn.execute(
        "INSERT INTO data_conflicts(issue,detected_at,a_source,a_red,a_blue,"
        "b_source,b_red,b_blue) VALUES(?,?,?,?,?,?,?,?)",
        (issue, utcnow(), existing["source"],
         ",".join(str(existing[f"r{i}"]) for i in range(1, 7)),
         str(existing["b1"]), source,
         ",".join(str(x) for x in sorted(reds)), str(blue)))
    conn.execute("UPDATE draws SET last_checked=?, check_status='pending_review',"
                 " revision=revision+1, notes=? WHERE issue=?",
                 (utcnow(), f"conflict with {source}", issue))
    return "conflict_pending"


def insert_draws_batch(conn: sqlite3.Connection,
                       rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"added": 0, "unchanged": 0, "revised": 0, "conflict_pending": 0}
    for r in rows:
        st = upsert_draw(conn, r["issue"], r.get("draw_date"),
                         r["red"], r["blue"], r["source"])
        counts[st] = counts.get(st, 0) + 1
    return counts


def load_draws(conn: sqlite3.Connection,
               status: Optional[str] = "ok") -> list[dict[str, Any]]:
    sql = "SELECT * FROM draws"
    args: tuple = ()
    if status:
        sql += " WHERE check_status=?"
        args = (status,)
    sql += " ORDER BY issue"
    return [dict(r) for r in conn.execute(sql, args)]


# ---------- 其余表的简单写入 ----------

def record_source(conn: sqlite3.Connection, meta: dict[str, Any]) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO data_sources"
        "(name,url,is_official,needs_api_key,paid,terms_notes,"
        " last_verified,coverage_note,status)"
        " VALUES(:name,:url,:is_official,:needs_api_key,:paid,:terms_notes,"
        " :last_verified,:coverage_note,:status)",
        {**meta, "is_official": int(meta.get("is_official", 0)),
         "needs_api_key": int(meta.get("needs_api_key", 0)),
         "paid": int(meta.get("paid", 0))})


def start_run(conn: sqlite3.Connection, source: str, mode: str) -> int:
    cur = conn.execute(
        "INSERT INTO ingestion_runs(source,mode,started_at) VALUES(?,?,?)",
        (source, mode, utcnow()))
    return int(cur.lastrowid)


def finish_run(conn: sqlite3.Connection, run_id: int,
               counts: dict[str, Any]) -> None:
    conn.execute(
        "UPDATE ingestion_runs SET finished_at=?, fetched=?, added=?,"
        " updated=?, skipped=?, failed=?, detail=? WHERE id=?",
        (utcnow(), counts.get("fetched", 0), counts.get("added", 0),
         counts.get("updated", 0), counts.get("skipped", 0),
         counts.get("failed", 0),
         json.dumps(counts.get("detail", {}), ensure_ascii=False), run_id))


def record_pattern(conn: sqlite3.Connection, name: str, category: str,
                   version: str, formula: Any, source: str) -> int:
    cur = conn.execute(
        "INSERT OR IGNORE INTO pattern_definitions"
        "(name,category,version,formula,source,created_at)"
        " VALUES(?,?,?,?,?,?)",
        (name, category, version,
         json.dumps(formula, ensure_ascii=False), source, utcnow()))
    if cur.rowcount == 0:
        row = conn.execute(
            "SELECT id FROM pattern_definitions WHERE name=? AND version=?",
            (name, version)).fetchone()
        return int(row["id"])
    return int(cur.lastrowid)


def record_experiment(conn: sqlite3.Connection, **kw: Any) -> int:
    kw.setdefault("run_id", None)
    kw.setdefault("pattern_id", None)
    kw.setdefault("p_value", None)
    kw.setdefault("p_adjusted", None)
    if isinstance(kw.get("effect"), (dict, list)):
        kw["effect"] = json.dumps(kw["effect"], ensure_ascii=False)
    kw.setdefault("effect", None)
    cur = conn.execute(
        "INSERT INTO experiment_results"
        "(run_id,pattern_id,test_name,n_tests,p_value,p_adjusted,effect,"
        " conclusion,detail,created_at)"
        " VALUES(:run_id,:pattern_id,:test_name,:n_tests,:p_value,"
        " :p_adjusted,:effect,:conclusion,:detail,:created_at)",
        {**kw, "detail": json.dumps(kw.get("detail", {}), ensure_ascii=False),
         "created_at": utcnow()})
    return int(cur.lastrowid)
