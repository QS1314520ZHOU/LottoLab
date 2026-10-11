# 旧版 BB 脚本用途标注（V1.1）

正式入口（新）：

| 入口 | 用途 |
| --- | --- |
| `python -m bb_analysis …` | BB 正式 CLI（import/patterns/backtest/bankroll/report/ui） |
| `python -m app …` | 双色球正式 CLI |
| `bb_sync.py` | BB 开奖采集（唯一保留的原采集器，行为未改动） |

历史/研究脚本（**保留但非正式入口**；均已通过 compile 检查，不代表仍在维护）：

| 脚本 | 用途（历史） | 风险/处置 |
| --- | --- | --- |
| `bb_v9.py` | 前代“检测+候选+验证门”研究脚本 | AI key 已改为环境变量；保留作研究存档 |
| `bb_ml.py` | 双色球 ML 对照实验（教程复现） | 结论已并入 `docs/`；保留 |
| `bb_engine_bankroll.py` | 旧引擎回放+资金模拟 | 结果被 `bb_analysis/backtest+bankroll` 取代 |
| `bb_bankroll_sim.py` `bb_strategy_bankroll.py` `bb_session_sim.py` | 早期资金/会话模拟 | 同上，口径较旧（无验证门），勿引用其数字 |
| `bb_complete*.py` `bb_core.py` `bb_mass_vote.py` | 旧融合引擎/投票实验 | 依赖 `models/*.joblib`（版本7 vs 代码6，**加载会被拒**）；不迁移、不删除，作考古材料 |
| `bb_zodiac_v7.py` `bb_zodiac_shortwin.py` | 生肖映射与短窗策略检验 | 映射已被 `config/mapping.json` 取代；保留其数据读取函数供旧脚本运行 |
| `bb_diag_backtest.py` `bb_cookie.py` `bb_live.py` | 诊断/凭据加载/在线预测实验 | `bb_cookie` 仍被 `bb_sync` 使用，**勿删** |
| `新建 文本文档.txt` | 含凭据片段的本地草稿 | 已 gitignore；建议人工自行删除 |

处置原则（V1.1）：**先记录用途，再决定弃用**；本版本不删除任何历史文件，
不修改 `bb_sync.py` 采集行为；被 `models/*.joblib` 依赖的旧引擎只读封存。
