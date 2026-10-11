# -*- coding: utf-8 -*-
"""报告生成：数据覆盖 + 规律评分卡 + 回测摘要 + 免责与限制。

导出 reports/ssq_report.txt 与 .json（含参数与版本，可复现）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import db
from .config import PROJECT_ROOT, SSQConfig
from .features import FEATURE_VERSION
from .validate import report_database


def generate_report(cfg: SSQConfig) -> str:
    db.migrate(cfg.db_path)
    with db.connect(cfg.db_path) as conn:
        patterns = [dict(r) for r in conn.execute(
            "SELECT * FROM pattern_definitions ORDER BY id")]
        experiments = [dict(r) for r in conn.execute(
            "SELECT * FROM experiment_results ORDER BY id DESC LIMIT 50")]
        runs = [dict(r) for r in conn.execute(
            "SELECT id,params,data_start,data_end,seed,summary "
            "FROM backtest_runs ORDER BY id DESC LIMIT 3")]
        preds = conn.execute(
            "SELECT COUNT(*) c, MIN(target_issue) lo, MAX(target_issue) hi "
            "FROM predictions").fetchone()
    data = report_database(cfg)

    payload: dict[str, Any] = {
        "generated_by": "app.reports",
        "feature_version": FEATURE_VERSION,
        "data": {k: data[k] for k in
                 ("total_rows", "valid_rows", "pending_conflicts",
                  "issue_range", "per_year")},
        "patterns": patterns,
        "experiments": experiments,
        "backtests": [{"id": r["id"], "params": r["params"],
                       "range": [r["data_start"], r["data_end"]],
                       "seed": r["seed"],
                       "summary": json.loads(r["summary"]) if r["summary"]
                       else None}
                      for r in runs],
        "predictions": {"n": preds["c"], "range": [preds["lo"], preds["hi"]]},
        "warnings": [
            "本报告全部内容为历史统计描述与研究性回测，不构成预测或投注建议。",
            "规律状态非 validated = 研究假设，未证明可预测未来。",
            "holdout 段只评估一次；根据其结果改参数后必须用新数据重验。",
            *list(cfg.__dict__.get("warnings", [])),
        ],
    }

    out_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    json_path = out_dir / "ssq_report.json"
    txt_path = out_dir / "ssq_report.txt"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    lines = ["=" * 72, "双色球历史数据分析报告（研究用途）", "=" * 72,
             f"数据覆盖: {data['issue_range']} ｜ 有效 {data['valid_rows']} 期"
             f" ｜ 冲突待核对 {data['pending_conflicts']}"]
    bt = payload["backtests"][0] if payload["backtests"] else None
    if bt and bt["summary"]:
        m = bt["summary"]
        lines.append(f"最近回测 run#{bt['id']} ｜ {bt['range']} ｜ seed={bt['seed']}")
        for seg in ("dev", "val", "holdout"):
            if seg in m:
                r = m[seg]
                ens = r.get("ens")
                if ens:
                    lines.append(
                        f"  [{seg}] ens Brier={ens['brier']:.4f} "
                        f"LogLoss={ens['logloss']:.4f} "
                        f"top1={ens['top1_hit']:.2%} "
                        f"CI=[{ens['top1_ci'][0]:.1%},{ens['top1_ci'][1]:.1%}]"
                        f" ｜ 公平基线 top1={m['_meta']['fair']['top1_hit']:.2%}")
    lines.append(f"已登记规律 {len(patterns)} 条，实验记录 {len(experiments)} 条")
    lines += ["", "--- 限制与已知问题 ---",
              "* 官方数据源仅覆盖2013+，2003-2012 来自非官方备用源（已标注）。",
              "* robots.txt 对带查询串 URL 有 Disallow 规则（见 DATA_SOURCES.md）。",
              "* 蓝球投票为 v1：红球侧只做概率模型与覆盖度评估，不投票。",
              "* AI 助手为可选组件，失败会记录，不影响核心流程。"]
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    return "\n".join(lines) + f"\n\n已导出: {txt_path}\n{json_path}"
