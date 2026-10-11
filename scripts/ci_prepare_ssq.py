# -*- coding: utf-8 -*-
"""CI 夹具准备：在无网络、无真实库的干净克隆中生成 data/ssq.db。

- 用固定种子生成 350 期【模拟数据】（source=simulated，绝不冒充真实开奖）
- 跑一次迷你走期回测，产出 backtest_runs.summary（供指标一致性测试）
- BB 侧不在此处理（CI 已有 python -m bb_analysis import/backtest 步骤）
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db
from app.backtest import run_backtest
from app.config import load_config


def main() -> None:
    cfg = load_config()
    rng = random.Random(20261012)
    rows = []
    for i in range(1, 351):
        reds = sorted(rng.sample(range(1, 34), 6))
        rows.append({
            "issue": f"2099{i:04d}",
            "draw_date": f"2099-01-{(i % 28) + 1:02d}",
            "red": reds, "blue": rng.randint(1, 16),
            "source": "simulated",
        })
    db.migrate(cfg.db_path)
    with db.connect(cfg.db_path) as conn:
        counts = db.insert_draws_batch(conn, rows)
    print("fixture draws:", counts)
    text = run_backtest(cfg, min_train=100, max_targets=120)
    print(text.splitlines()[0])
    print("fixture ready:", cfg.db_path)


if __name__ == "__main__":
    main()
