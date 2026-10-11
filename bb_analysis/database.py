# -*- coding: utf-8 -*-
"""SQLite 数据层：模式、迁移、幂等写入与查询接口。

实体：draws / zodiac_mapping / draw_zodiacs / features / predictions /
settlements / backtest_runs。主键、唯一约束、索引齐全；重复导入用
INSERT OR IGNORE + 行数差计量，绝不重复计数；原始开奖数据只增不改。
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

SCHEMA_VERSION = 2

_MIGRATIONS: dict[int, str] = {
    1: """
    CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS draws (
        issue      TEXT PRIMARY KEY,
        draw_time  TEXT,
        p1 INTEGER NOT NULL, p2 INTEGER NOT NULL, p3 INTEGER NOT NULL,
        p4 INTEGER NOT NULL, p5 INTEGER NOT NULL, p6 INTEGER NOT NULL,
        sp         INTEGER NOT NULL,
        source     TEXT NOT NULL,
        fetched_at TEXT NOT NULL,
        check_status TEXT NOT NULL DEFAULT 'pending',
        is_simulated INTEGER NOT NULL DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS zodiac_mapping (
        version       TEXT NOT NULL,
        number        INTEGER NOT NULL,
        zodiac        TEXT NOT NULL,
        effective_from TEXT,
        effective_to   TEXT,
        source         TEXT NOT NULL,
        PRIMARY KEY (version, number)
    );

    CREATE TABLE IF NOT EXISTS draw_zodiacs (
        issue     TEXT NOT NULL,
        position  INTEGER NOT NULL,   -- 1..6 正码, 7 特别号
        number    INTEGER NOT NULL,
        zodiac    TEXT NOT NULL,
        mapping_version TEXT NOT NULL,
        PRIMARY KEY (issue, position),
        FOREIGN KEY (issue) REFERENCES draws(issue),
        FOREIGN KEY (mapping_version, number)
            REFERENCES zodiac_mapping(version, number)
    );

    CREATE TABLE IF NOT EXISTS features (
        scope_type  TEXT NOT NULL,    -- 'single' | 'combo'
        scope_id    TEXT NOT NULL,    -- '鼠' 或 '鼠+牛'
        as_of_issue TEXT NOT NULL,    -- 预测截止点（只含该期之前的数据）
        window      INTEGER NOT NULL,
        hits        INTEGER NOT NULL,
        freq        REAL NOT NULL,
        omission    INTEGER NOT NULL,
        streak      INTEGER NOT NULL,
        max_streak  INTEGER NOT NULL,
        longest_miss INTEGER NOT NULL,
        freq_delta  REAL NOT NULL,
        surprise_z  REAL NOT NULL,
        n_eff       INTEGER NOT NULL,
        payload     TEXT,             -- 转移频率等附加特征 JSON
        computed_at TEXT NOT NULL,
        PRIMARY KEY (scope_type, scope_id, as_of_issue, window)
    );

    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER NOT NULL,
        target_issue TEXT NOT NULL,
        generated_at TEXT NOT NULL,
        model_version TEXT NOT NULL,
        market TEXT NOT NULL,          -- 'single' | 'combo'
        k INTEGER NOT NULL,
        combo TEXT NOT NULL,           -- '鼠' 或 '鼠+牛'（成员排序）
        p_est REAL NOT NULL,
        p_base REAL NOT NULL,
        p_lo REAL, p_hi REAL,
        sample_n INTEGER NOT NULL,
        odds REAL NOT NULL,
        data_cutoff TEXT NOT NULL,     -- 必须 < target_issue
        selected INTEGER NOT NULL DEFAULT 0,
        FOREIGN KEY (run_id) REFERENCES backtest_runs(id)
    );
    CREATE UNIQUE INDEX IF NOT EXISTS ux_pred
        ON predictions(run_id, target_issue, market, combo, model_version);
    CREATE INDEX IF NOT EXISTS ix_pred_target
        ON predictions(target_issue);

    CREATE TABLE IF NOT EXISTS settlements (
        prediction_id INTEGER PRIMARY KEY,
        run_id INTEGER NOT NULL,
        strategy TEXT NOT NULL,
        stake REAL NOT NULL,
        hit INTEGER NOT NULL,
        return_total REAL NOT NULL,
        net REAL NOT NULL,
        cum_net REAL NOT NULL,
        balance REAL NOT NULL,
        FOREIGN KEY (prediction_id) REFERENCES predictions(id)
    );
    CREATE INDEX IF NOT EXISTS ix_settle_run ON settlements(run_id);

    CREATE TABLE IF NOT EXISTS backtest_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        params_json TEXT NOT NULL,
        data_start TEXT NOT NULL,
        data_end TEXT NOT NULL,
        model_version TEXT NOT NULL,
        started_at TEXT NOT NULL,
        finished_at TEXT,
        summary_json TEXT
    );
    """,
    2: """
    ALTER TABLE predictions ADD COLUMN votes_json TEXT;
    ALTER TABLE predictions ADD COLUMN abstain_reason TEXT;
    """,
}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    """打开连接（外键开启、行转 dict）。"""
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
    """按版本号顺序应用迁移，返回最终 schema 版本。"""
    with connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        applied = {
            r["version"]
            for r in conn.execute("SELECT version FROM schema_migrations")
        }
        for ver in sorted(_MIGRATIONS):
            if ver in applied:
                continue
            conn.executescript(_MIGRATIONS[ver])
            conn.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES(?,?)",
                (ver, utcnow()),
            )
            applied.add(ver)
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        return max(applied) if applied else 0


def insert_draws(
    conn: sqlite3.Connection,
    rows: Iterable[dict[str, Any]],
) -> int:
    """幂等导入 draws，返回真正新增行数。"""
    cur = conn.executemany(
        "INSERT OR IGNORE INTO draws"
        "(issue, draw_time, p1,p2,p3,p4,p5,p6, sp,"
        " source, fetched_at, check_status, is_simulated)"
        " VALUES(:issue,:draw_time,:p1,:p2,:p3,:p4,:p5,:p6,:sp,"
        " :source,:fetched_at,:check_status,:is_simulated)",
        list(rows),
    )
    return cur.rowcount


def insert_mapping_version(
    conn: sqlite3.Connection,
    version: str,
    table: dict[int, str],
    effective_from: Optional[str],
    effective_to: Optional[str],
    source: str,
) -> int:
    """幂等导入一个映射版本，返回新增行数。"""
    rows = [
        (version, num, z, effective_from, effective_to, source)
        for num, z in sorted(table.items())
    ]
    cur = conn.executemany(
        "INSERT OR IGNORE INTO zodiac_mapping"
        "(version, number, zodiac, effective_from, effective_to, source)"
        " VALUES(?,?,?,?,?,?)",
        rows,
    )
    return cur.rowcount


def insert_draw_zodiacs(
    conn: sqlite3.Connection,
    rows: Iterable[tuple[str, int, int, str, str]],
) -> int:
    """(issue, position, number, zodiac, mapping_version) 幂等导入。"""
    cur = conn.executemany(
        "INSERT OR IGNORE INTO draw_zodiacs"
        "(issue, position, number, zodiac, mapping_version) VALUES(?,?,?,?,?)",
        list(rows),
    )
    return cur.rowcount


def insert_features(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    cur = conn.executemany(
        "INSERT OR REPLACE INTO features"
        "(scope_type, scope_id, as_of_issue, window, hits, freq, omission,"
        " streak, max_streak, longest_miss, freq_delta, surprise_z, n_eff,"
        " payload, computed_at)"
        " VALUES(:scope_type,:scope_id,:as_of_issue,:window,:hits,:freq,"
        " :omission,:streak,:max_streak,:longest_miss,:freq_delta,"
        " :surprise_z,:n_eff,:payload,:computed_at)",
        rows,
    )
    return cur.rowcount


def create_run(conn: sqlite3.Connection, params: dict[str, Any],
               data_start: str, data_end: str,
               model_version: str) -> int:
    cur = conn.execute(
        "INSERT INTO backtest_runs"
        "(params_json, data_start, data_end, model_version, started_at)"
        " VALUES(?,?,?,?,?)",
        (json.dumps(params, ensure_ascii=False), data_start, data_end,
         model_version, utcnow()),
    )
    return int(cur.lastrowid)


def finish_run(conn: sqlite3.Connection, run_id: int,
               summary: dict[str, Any]) -> None:
    conn.execute(
        "UPDATE backtest_runs SET finished_at=?, summary_json=? WHERE id=?",
        (utcnow(), json.dumps(summary, ensure_ascii=False), run_id),
    )


def insert_predictions(
    conn: sqlite3.Connection,
    rows: list[dict[str, Any]],
) -> int:
    for r in rows:
        r.setdefault("votes_json", None)
        r.setdefault("abstain_reason", None)
    cur = conn.executemany(
        "INSERT OR IGNORE INTO predictions"
        "(run_id, target_issue, generated_at, model_version, market, k, combo,"
        " p_est, p_base, p_lo, p_hi, sample_n, odds, data_cutoff, selected,"
        " votes_json, abstain_reason)"
        " VALUES(:run_id,:target_issue,:generated_at,:model_version,:market,"
        " :k,:combo,:p_est,:p_base,:p_lo,:p_hi,:sample_n,:odds,:data_cutoff,"
        " :selected, :votes_json, :abstain_reason)",
        rows,
    )
    return cur.rowcount


def insert_settlements(
    conn: sqlite3.Connection,
    rows: list[dict[str, Any]],
) -> int:
    cur = conn.executemany(
        "INSERT OR REPLACE INTO settlements"
        "(prediction_id, run_id, strategy, stake, hit, return_total, net,"
        " cum_net, balance)"
        " VALUES(:prediction_id,:run_id,:strategy,:stake,:hit,:return_total,"
        " :net,:cum_net,:balance)",
        rows,
    )
    return cur.rowcount


def load_draws(conn: sqlite3.Connection,
               include_simulated: bool = False) -> list[dict[str, Any]]:
    sql = "SELECT * FROM draws"
    if not include_simulated:
        sql += " WHERE is_simulated=0"
    sql += " ORDER BY issue"
    return [dict(r) for r in conn.execute(sql)]


def load_presence(conn: sqlite3.Connection,
                  include_simulated: bool = False) -> tuple[list[str], list[str]]:
    """返回 (issues, zodiacs) —— 后续特征/回测的基础轴。"""
    where = "" if include_simulated else " WHERE is_simulated=0"
    issues = [r["issue"] for r in conn.execute(
        "SELECT issue FROM draws" + where + " ORDER BY issue")]
    zodiacs = sorted({r["zodiac"] for r in conn.execute(
        "SELECT DISTINCT zodiac FROM draw_zodiacs")})
    return issues, zodiacs

