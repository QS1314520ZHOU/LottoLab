# -*- coding: utf-8 -*-
"""bb_analysis 命令行：python -m bb_analysis <command>。

commands:
  import      导入 data/bb_live_history.csv（幂等）
  patterns    运行规律注册表 → reports/bb_pattern_registry.json
  backtest    严格走期回测（冻结预测落库）
  bankroll    固定 vs 三期资金模拟 → reports/bb_bankroll.json
  report      汇总报告 → reports/bb_report.txt/.json
  ui          启动 Streamlit 面板（或手动 streamlit run ...）
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m bb_analysis")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("import")
    sub.add_parser("patterns")
    sub.add_parser("backtest")
    sub.add_parser("bankroll")
    sub.add_parser("report")
    sub.add_parser("ui")
    args = p.parse_args(argv)

    if args.cmd == "import":
        from .config import load_config
        from .ingestion import import_csv_file
        cfg = load_config()
        rep, counts = import_csv_file(cfg, "data/bb_live_history.csv",
                                      source="bb_sync_csv")
        print("校验:", rep.ok, rep.stats)
        print("入库:", counts)
        if not rep.ok:
            print("错误:", rep.errors[:10])
            return 1
    elif args.cmd == "patterns":
        from .patterns import run_registry
        _, text = run_registry()
        print(text)
    elif args.cmd == "backtest":
        from .backtest import run_backtest
        print(run_backtest())
    elif args.cmd == "bankroll":
        from .bankroll import run_bankroll
        print(run_bankroll())
    elif args.cmd == "report":
        from .reports import write_report
        print(write_report())
    elif args.cmd == "ui":
        import subprocess
        from .config import PROJECT_ROOT
        script = PROJECT_ROOT / "bb_analysis" / "ui_app.py"
        print(f"启动: streamlit run {script}")
        return subprocess.call(
            [sys.executable, "-m", "streamlit", "run", str(script)],
            cwd=str(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
