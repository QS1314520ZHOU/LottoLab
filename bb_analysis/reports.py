# -*- coding: utf-8 -*-
"""T10 · BB 分析报告生成：数据时间 / 规律证据等级 / 回测 / 风险 / 免责。

导出 reports/bb_report.txt 与 bb_report.json（含参数与版本，可复现）。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from .config import PROJECT_ROOT, AppConfig, load_config


def build_report(cfg: AppConfig | None = None) -> dict[str, Any]:
    from . import database as db
    from .bankroll import run_bankroll
    from .backtest import run_backtest  # 仅用于文档引用，不重跑
    from .patterns import run_registry

    cfg = cfg or load_config()
    db.migrate(cfg.db_path)

    # 数据更新时间与覆盖
    with db.connect(cfg.db_path) as conn:
        meta = conn.execute(
            "SELECT COUNT(*) n, MIN(issue) lo, MAX(issue) hi, "
            "MAX(fetched_at) upd FROM draws").fetchone()
        runs = [dict(r) for r in conn.execute(
            "SELECT id, model_version, data_start, data_end, started_at,"
            " finished_at FROM backtest_runs ORDER BY id DESC LIMIT 3")]
        n_preds = conn.execute(
            "SELECT COUNT(*) c, SUM(selected) s FROM predictions").fetchone()
        schema = conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()

    # 规律证据等级
    results, _ = run_registry(cfg, export=False)
    levels: dict[str, list[str]] = {"validated": [], "exploratory": [],
                                    "rejected": [], "descriptive": []}
    for r in results:
        levels.setdefault(r.status, []).append(r.pid)

    # 回测摘要（读最近一次 run）
    bt_summary: dict[str, Any] = {}
    with db.connect(cfg.db_path) as conn:
        row = conn.execute(
            "SELECT summary_json AS summary FROM backtest_runs "
            "ORDER BY id DESC LIMIT 1").fetchone()
        if row and row["summary"]:
            bt_summary = json.loads(row["summary"])

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "generator": "bb_analysis.reports",
        "data": {"n": meta["n"], "range": [meta["lo"], meta["hi"]],
                 "last_checked": meta["upd"], "schema_version":
                 schema["value"] if schema else None},
        "predictions": {"n": n_preds["c"], "selected": n_preds["s"] or 0},
        "runs": runs,
        "evidence_levels": levels,
        "backtest": bt_summary,
        "bankroll_text": run_bankroll(cfg),
        "odds_config": {
            "version": cfg.odds.version,
            "default_single": cfg.odds.default_single,
            "current_year": cfg.odds.current_year_zodiac,
            "current_year_odds": cfg.odds.current_year_odds,
            "combo": {str(k): v for k, v in cfg.odds.combo_odds.items()},
            "status_note": "当年生肖=马依截图证据配置，待复核",
        },
        "warnings": [
            "全部规律为探索性或描述性，无一通过 BH+holdout 双门槛——"
            "不存在已验证的预测优势。",
            "回测评级全部 unproven；holdout 结论=证据不足。",
            "资金策略不改变中奖概率；任何注额方案不保证盈利。",
            "本报告为研究记录，不构成投注建议。",
            *cfg.warnings,
        ],
    }
    return report


def write_report(cfg: AppConfig | None = None) -> str:
    cfg = cfg or load_config()
    rep = build_report(cfg)
    out_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "bb_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")

    lines = ["=" * 76,
             "BB六合彩分析报告（研究用途 · 无保证中奖声明）",
             "=" * 76,
             f"生成时间: {rep['generated_at']}",
             f"数据: {rep['data']['n']}期 {rep['data']['range']} "
             f"｜ 最后校验 {rep['data']['last_checked']}",
             f"赔率配置: {rep['odds_config']['version']} "
             f"默认{rep['odds_config']['default_single']} "
             f"当年生肖{rep['odds_config']['current_year']}"
             f"@{rep['odds_config']['current_year_odds']}"
             f"（{rep['odds_config']['status_note']}）",
             f"预测落库: {rep['predictions']['n']} 条 "
             f"（selected={rep['predictions']['selected']}）",
             "", "[证据等级]"]
    for lvl, pids in rep["evidence_levels"].items():
        lines.append(f"  {lvl:<14} {len(pids)} 条: {', '.join(pids) or '-'}")
    bt = rep["backtest"]
    if bt:
        lines += ["", "[回测（最近一次 run）]"]
        meta = bt.get("meta", {})
        lines.append(f"  分段 {meta.get('segments')} ｜ 基线 "
                     f"{meta.get('baseline_hit')}")
        for m, r in (bt.get("ratings") or {}).items():
            if isinstance(r, dict) and "grade" in r:
                lines.append(f"  评级 {m}: {r['grade']}"
                             + (f" (ΔBrier={r.get('delta'):+.4f}, "
                                f"p={r.get('p'):.3f})"
                                if "delta" in r else ""))
        for r in bt.get("holdout_tests", []):
            if "rate" in r:
                lines.append(
                    f"  holdout {r['model']}: {r['rate']:.1%} vs "
                    f"{r['base']:.1%} q={r['q']:.3f} → "
                    f"{'显著' if r['q'] < 0.05 else '证据不足'}")
    lines += ["", "[资金风险]", rep["bankroll_text"], "",
              "[限制与红线]"]
    lines += [f"  * {w}" for w in rep["warnings"]]
    text = "\n".join(lines)
    (out_dir / "bb_report.txt").write_text(text, encoding="utf-8")
    return text + f"\n\n已导出: {out_dir / 'bb_report.txt'}\n{out_dir / 'bb_report.json'}"
