# BB六合彩分析平台 · 文档（T11）

> 定位：历史数据分析与概率研究。**无已验证的预测优势，不保证中奖，不构成投注建议。**

## 1. 架构

```
bb_analysis/
  config.py       集中配置（app.json / odds.json / mapping.json + 环境变量）
  database.py     SQLite 迁移 v1→v2（draws/mapping/draw_zodiacs/features/
                  predictions/settlements/backtest_runs；v2 增 votes/abstain 列）
  mapping.py      生肖映射按开奖日期选版本；draw_zodiacs 逐号保留原始次数
  validation.py   校验报告（期号/范围/重复/映射完整性）
  ingestion.py    CSV 统一接口（bb_sync 采集产物）→ 幂等入库
  rules.py        中奖判定（all_zodiacs_present 可配置）与赔率结算公式
  theory.py       精确概率：包含-排斥（不放回）+ 蒙特卡洛交叉验证
  combinations.py 单肖~五连肖全量枚举与 EV
  features/       frequency / structure / cross_draw / zodiac_relations
  patterns.py     ★T7 规律注册表（评分卡+BH+holdout 状态机）
  models.py       ★T8 A公平/B频率/C贝叶斯(+区间)/D逻辑回归 + 校准 + 评级
  voting.py       ★T8 三级投票（类内→跨类→稳定门槛；validated 才有规律票权）
  ai.py           ★T8 只读 AI 解释（失败记录 coverage，不改任何状态）
  backtest.py     ★T9 走期回测（三分段/冻结落库/配对评级/holdout BH/随机MC）
  bankroll.py     ★T10 固定 vs 三期策略 + MC 压力（同信号同路径）
  reports.py      ★T10 汇总报告 txt/json
  ui_app.py       ★T10 Streamlit 面板（模型表现/风险/数据时间/证据等级）
  __main__.py     CLI：python -m bb_analysis <import|patterns|backtest|
                  bankroll|report|ui>
```

边界：与双色球系统 `app/` 零依赖互通（测试 `test_ssq_bb_isolation` 断言）；
原抓取器 `bb_sync.py` 与历史 CSV 未修改。

## 2. 数据

- 来源：既有 `bb_sync.py` 写出的 `data/bb_live_history.csv`（251期，
  202610100685~202610110215）→ `data/bb_analysis.db`。
- 幂等导入：重复执行 `python -m bb_analysis import` 不重复计数，
  原 CSV 校验和不变（测试 `test_data_update_preserves_old`）。
- 平台接口实测记录见 `docs/DATA_SOURCES.md`（该文档主体为双色球来源；
  BB 接口由 bb_sync.py 使用，未在本阶段重新探测）。

## 3. 生肖映射版本

- `config/mapping.json`：版本 `bb-2026.1`，来源=「BB平台一肖页截图
  逐号核对（2026-10-11）」，12生肖49号全覆盖；按开奖日期选生效版本。
- 号码口径：`生肖 = 号码 mod 12`（01=马 … 12=羊），测试
  `test_map_draw_matches_bb_platform_table` 锁定。

## 4. 赔率（总返还倍数，含本金）

`config/odds.json`（版本 bb-odds-2026.1）：

| 项 | 值 | 依据 |
| --- | --- | --- |
| 单肖 default | 2.06 | 平台截图（11个4号肖） |
| 单肖 当年生肖 | 1.76 | 截图马=1.76 + 2026丙午马年；**待用户复核**；有效期至2026-12-31，到期自动回落2.06 |
| 二连肖/三/四/五 | 4.2 / 10.6 / 30.7 / 108 | 任务书（首版规格） |

结算：中奖净收益 = stake×(O−1)，未中 = −stake，盈亏平衡率 = 1/O。

## 5. 规律定义（T7 注册表）

- 位置：代码 `patterns.py`；数据导出 `reports/bb_pattern_registry.json`。
- 15 条：10 条预测类（热冷/遗漏/连断/平特/位移/自动发现）+ 2 条检验
  （平特同期、特肖转移卡方）+ 3 条描述表（热冷多窗/遗漏排名/长龙断龙）。
- 统一评分卡：n / 基线（theory 全开口径）/ 观测 / 差 / Wilson CI /
  p / BH q / 理论EV / 状态 / 适用范围。
- 状态机：descriptive → exploratory → validated（**q<0.05 且 holdout
  同向复现**）/ rejected（失败记录保留）。
- **当前状态：validated=0**（例：特码+1 观测56.2%、p=0.018、q=0.218
  → exploratory）。任何“高频=信号”的说法不被本库接受。

## 6. 回测方法（T9）

- 走期：目标期 t 只读 [:t]；预测落库 `predictions`，
  `data_cutoff = 上一期期号`（测试断言全表 data_cutoff < target）。
- 三分段（自 warmup=50）：dev 60% / val 20% / holdout 20%；
  holdout 只评估一次，之后改参数必须用新数据。
- 指标：多标签 Brier、二元 LogLoss、top1 命中+95%CI、弃权率、
  校准 ECE、与公平基线的配对 ΔBrier（t 检验）、holdout 命中 vs
  各自 top1 理论基线（正态近似 + BH 分母4）、随机选肖 MC（seed=42，
  2000次）。
- 基线口径 = BB 全开一肖理论命中率（47.17%/55.39%），**不套用双色球**。
- 评级门槛（预先冻结）：val 段 ΔBrier<0 且 p<0.05 → beat_baseline。
- **当前评级：B/C/D/ens 全部 unproven；holdout 全部 q≥0.05 → 证据不足。**

## 7. 资金策略（T10）

- 三期规则逐期：位0=s；位1=上期未中?2s:s；位2=s（周期重复）——
  边界测试见 `test_three_period_stake_boundaries`。
- 固定 vs 三期在同一信号路径上对比；断言命中序列完全一致
  （策略不改变概率）；MC 破产压力注明 i.i.d. 重采样假设。
- 结果：`reports/bb_bankroll.json`（初始100/200/500 × 两策略）。

## 8. 风险与红线

1. 251 期小样本：多数区间宽、检出力低——**证据不足是常态结论**。
2. 规律全部 exploratory/descriptive，无 validated；回测全部 unproven。
3. 加倍不提高命中率，只放大波动与资金需求（MC 显示回撤P95更高）。
4. 赔率1.76档配置待复核（见 §4）。
5. 本仓库当前**无 git 版本控制**（建议 `git init` 后再大改）。

## 9. 可复现步骤

```bash
pip install -r requirements.txt
python -m bb_analysis import        # 导入（幂等）
python -m bb_analysis patterns      # → reports/bb_pattern_registry.json
python -m bb_analysis backtest      # → predictions + backtest_runs
python -m bb_analysis bankroll      # → reports/bb_bankroll.json
python -m bb_analysis report        # → reports/bb_report.txt/.json
python -m bb_analysis ui            # Streamlit 面板（或 streamlit run bb_analysis/ui_app.py）
python -m pytest tests/ -q          # 43 项测试
# 双色球侧（互不影响）：python -m app backtest / report …
```

随机种子固定 42；版本与参数随 backtest_runs.params / 报告 JSON 落库。
