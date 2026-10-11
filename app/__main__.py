# -*- coding: utf-8 -*-
"""命令行入口：python -m app <command>。

数据命令：discover-sources | validate-sources | import-history |
          update | validate-data | report-data
分析命令：analyze | backtest | report   （见 docs/README）
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app",
                                     description="双色球历史数据与规律分析系统")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("discover-sources", help="探测并登记数据源")
    sub.add_parser("validate-sources", help="双源抽样交叉核对")
    p = sub.add_parser("import-history", help="初始化/全量导入")
    p.add_argument("--fast", action="store_true", help="跳过备用源早期补齐")
    sub.add_parser("update", help="增量更新（含近10期回查）")
    p = sub.add_parser("validate-data", help="库内数据质量校验")
    p.add_argument("--json", action="store_true")
    sub.add_parser("report-data", help="数据覆盖与采集报告")
    p = sub.add_parser("analyze", help="特征+规律挖掘（详见 analyze 模块）")
    p.add_argument("--pattern-limit", type=int, default=0)
    p = sub.add_parser("backtest", help="严格走期回测")
    p.add_argument("--min-train", type=int, default=None)
    p.add_argument("--max-targets", type=int, default=0)
    p = sub.add_parser("report", help="生成分析报告（txt/json）")

    args = parser.parse_args(argv)
    cfg = load_config()

    if args.cmd == "discover-sources":
        from . import ingest
        print(json.dumps(ingest.discover_sources(cfg), ensure_ascii=False,
                         indent=2))
    elif args.cmd == "validate-sources":
        from . import ingest
        print(json.dumps(ingest.cross_check(cfg), ensure_ascii=False,
                         indent=2))
    elif args.cmd == "import-history":
        from . import ingest
        print(json.dumps(ingest.import_history(cfg, full=not args.fast),
                         ensure_ascii=False, indent=2, default=str))
    elif args.cmd == "update":
        from . import ingest
        print(json.dumps(ingest.update(cfg), ensure_ascii=False, indent=2,
                         default=str))
    elif args.cmd == "validate-data":
        from .validate import report_database, validate_rows
        from . import db as _db
        _db.migrate(cfg.db_path)
        with _db.connect(cfg.db_path) as conn:
            rows = _db.load_draws(conn, status=None)
        norm = [{"issue": r["issue"], "draw_date": r["draw_date"],
                 "red": [r[f"r{i}"] for i in range(1, 7)], "blue": r["b1"]}
                for r in rows]
        rep = validate_rows(norm, cfg)
        out = rep.to_dict()
        if args.json:
            print(json.dumps(out, ensure_ascii=False, indent=2))
        else:
            print(f"校验 {'通过' if rep.ok else '失败'} ｜ {out['stats']}")
            for e in out["errors"][:20]:
                print("  ERROR", e)
            for w in out["warnings"][:20]:
                print("  WARN ", w)
        return 0 if rep.ok else 1
    elif args.cmd == "report-data":
        from .validate import report_database, dumps
        print(dumps(report_database(cfg)))
    elif args.cmd == "analyze":
        from .analyze import run_analysis
        print(run_analysis(cfg, pattern_limit=args.pattern_limit))
    elif args.cmd == "backtest":
        from .backtest import run_backtest
        print(run_backtest(cfg, min_train=args.min_train,
                           max_targets=args.max_targets))
    elif args.cmd == "report":
        from .reports import generate_report
        print(generate_report(cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
