# 架构说明（双色球模块 `app/`）

定位：独立于 BB 分析系统（`bb_analysis/`、既有抓取器）的双色球历史数据与规律研究系统。
核心原则：**先找规律 → 再验证规律 → 最后才给投票权**；任何历史拟合都不代表未来中奖。

## 分层

```
数据层   app/sources/*   适配器（fetch/parse/validate，限速+重试+退避）
        app/ingest.py   discover / 交叉核对 / 全量导入 / 增量更新
        app/db.py       SQLite 10 表 + 迁移（draws 只增不改，冲突进 data_conflicts）
        app/validate.py 质量校验与覆盖报告
规则层   app/features.py fv1 特征（严格只用目标期之前；单测保证）
概率层   app/baseline.py C(33,6)×16 解析基线 + 固定种子 MC 交叉验证
规律层   app/patterns.py 7 条规则（频率/结构/跨期/玄学）统一评分卡 +
                        Markov/红球对自动发现 + BH 多重检验
模型层   app/models.py   A 公平 / B 滚动频率 / C 贝叶斯(区间) / D 逻辑回归(stride重训)
投票层   app/voting.py   三级投票：类内等权 → 跨类等权 → 稳定性门槛(可弃权)
回测层   app/backtest.py walk-forward + dev/val/holdout 三分段 + 冻结预测落库
辅助     app/ai_assist.py 可选 LLM 解释（失败记录覆盖率，不参与核心链路）
        app/reports.py  TXT/JSON 报告（含参数与版本，可复现）
入口     app/__main__.py python -m app <command>
```

## 关键不变式

1. `features[t]` 只依赖 `draws[:t]`（测试 `test_features_no_future_leakage`）。
2. `predictions.data_cutoff < target_issue`（测试断言 + e2e 全表检查）。
3. 同期数据跨来源不一致 → 记 `data_conflicts` + `pending_review`，绝不覆盖。
4. 模拟数据 `source=simulated`，与真实数据物理可区分（不同库/不同标记）。
5. 玄学规则公式预先固定、无特权权重，与统计规则同一评分卡。
6. 票数 ≠ 中奖概率；只有校准过的概率模型才以概率呈现。

## 与 BB 系统的边界

- BB：`bb_analysis/` + `bb_sync.py` + `data/bb_live_history.csv` + `data/bb_analysis.db`。
- SSQ：`app/` + `data/ssq.db`。两套配置、库、测试完全隔离；
  `tests/test_base_layer.py` 持续守护 BB 行为不被改坏。
