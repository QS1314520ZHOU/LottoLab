# LottoLab 前端实施计划（V1.2 + V1.3 合并）

- 依据：对**真实代码**的核对（非 README 推断），核对日期 2026-10-11
- 结论先行：**没有现成 Web API**；唯一 UI 是 `bb_analysis/ui_app.py`（BB 单系统、
  无导航、非中文工作台风格）。技术选型 = **Streamlit 单入口 + service 服务层**
  （已装 1.65，零新增大型框架；matplotlib 3.9 画图）。

## 1. 现有功能在哪（代码事实）

| 能力 | 位置 | 可否直接复用 |
| --- | --- | --- |
| SSQ 数据导入/校验/更新/冲突 | `app/{ingest,validate,db}.py` | ✅ 全部经 service 调用 |
| SSQ 特征/模型A-D/排名 | `app/{features,models}.py`、`tickets.rank_candidates` | ✅ 推荐页数据源 |
| SSQ 候选号生成（合法校验/去重/预算） | `app/tickets.make_tickets/validate_*` | ✅ “生成候选”按钮 |
| SSQ 对奖（官方固定奖级） | `app/tickets.prize_for`（config/app.json 证据标注） | ✅ 对奖页 |
| SSQ 走期回测/结论句 | `app/backtest.py`（backtest_runs 4条） | ✅ 结果读取+任务重跑 |
| SSQ 预算ROI/回撤/连亏 | `app/tickets.evaluate_budget` | ✅ 资金页 |
| SSQ 规律评分卡 | `app/patterns`（experiment_results） | ✅ 规律页 |
| BB 数据/映射/赔率(带证据) | `bb_analysis/{ingestion,config,mapping}.py` | ✅ |
| BB 规律注册表15条+证据等级 | `bb_analysis/patterns`（registry JSON 导出） | ✅ |
| BB 模型/投票/走期回测 | `bb_analysis/{models,voting,backtest}.py`（runs 5条） | ✅ |
| BB 资金：固定/三期策略+MC | `bb_analysis/bankroll.py` | ✅（三期规则=已确认代码） |
| BB 对奖（单肖/连肖判定+结算） | `bb_analysis/{rules,combinations}.py` | ✅ 单肖已证实；连肖规则**待确认**要标注 |
| BB 采集 | `bb_sync.py`（cookie） | ✅ 仅作任务触发；**凭据不入页/不入日志** |
| AI 解释 | `bb_analysis/ai.py` / `app/ai_assist.py` | ✅ 可选、失败降级（已有测试） |

## 2. 已有后端但缺前端（本次做）

总览/最新开奖、历史开奖(搜索分页导出)、红蓝球分析网格、候选生成与保存、走势图表、
规律卡片、模型评估(三段切换+白话解释)、双系统对奖、双系统资金模拟、数据管理任务按钮、
赔率规则页(证据/待核实)、策略回测页(结果展示+受限设置)、设置页。

## 3. 确实未实现（页面显式“尚未实现”，不造假）

- SSQ 官方逐期奖金列表（数据源不含）→ 对奖仅用配置固定奖并标注来源
- BB 连肖正式中奖口径（rules=待确认）→ 只显示命中情况+“无法按正式规则结算”
- 回测的自定义期号/玩法选择（后端固定参数）→ 下拉禁用+原因
- 进度百分比（后端无进度流）→ 只显示阶段状态，不造进度条

## 4. 前后端连接

```
web/app.py（入口：系统切换+左导航+顶栏+页脚）
   └─ 页面模块只调用 web/service.py
        ├─ 结构化返回 {ok, data, error(中文原因+建议)}
        ├─ 任务表 TaskStore：线程执行、状态/日志/防重复提交
        └─ SSQ 与 BB 两套独立函数（不同 db/规则，禁止混用）
CLI 完全保留；service 复用 CLI 同一函数，不复制算法。
```

不用 HTTP 也可满足“明确接口”要求（本地单进程 Streamlit）；
service 即 API 层，未来可平移到 FastAPI。

## 5. 页面与数据接口映射

| 页面 | 主要数据/任务（service 函数） |
| --- | --- |
| 总览(两系统) | `overview_ssq/overview_bb`：最新开奖、期数、更新时间、回测结论、冲突数 |
| SSQ 号码推荐 | `ssq_scores`(红33/蓝16网格)、`ssq_generate`(N组候选,保存)、`ssq_saved_picks` |
| SSQ 历史开奖 | `ssq_history(期号/日期/分页)`、`export_csv` |
| SSQ 走势 | `ssq_trends(window)` → matplotlib 图组(频率/遗漏/奇偶/大小) |
| SSQ 规律 | `ssq_patterns`（experiment_results+白话解释+证据等级） |
| SSQ 模型评估 | `ssq_backtest_latest`（分段表+指标白话+结论句） |
| SSQ 对奖 | `ssq_check(号码, 期号|全史)` → 命中/奖级/金额(prize_for) |
| SSQ 资金 | `ssq_budget(budget,期数)` → evaluate_budget 对照 |
| BB 总览/开奖 | `bb_overview/bb_history/bb_draw_detail(带生肖+映射版本)` |
| BB 号码与生肖 | `bb_stats(window, scope=号码|正码|特码|生肖)` |
| BB 生肖组合 | `bb_combo(玩法k, 组合)` → 理论概率+历史命中+规则说明(待确认标注) |
| BB 赔率规则 | `bb_odds_config`（odds.json+evidence+核实状态） |
| BB 推荐 | `bb_research(玩法/历史范围)` → 历史统计参考(带注册表证据等级,可保存) |
| BB 对奖 | `bb_check(玩法,生肖,期号)` → 命中+结算(单肖)/待确认(连肖) |
| BB 策略回测 | `bb_backtest_latest` + `run_task('bb_backtest')`（设置项禁用+原因） |
| BB 资金 | `bb_bankroll(initial,base)` → 固定vs三期+MC |
| 数据管理(两系统) | `run_task('ssq_update'|'ssq_validate'|'bb_import'|'bb_backtest'|...)` |
| 设置 | `settings_info`（配置只读+证据+版本） |

## 6. 流程

阶段1 本文件 ✅ → 阶段2 service+壳+总览/开奖/历史+测试 → 阶段3 推荐/对奖/规律/模型
→ 阶段4 回测/资金/数据管理/错误处理 → 阶段5 全量测试+截图+README。每阶段跑
`pytest tests/` 保持全绿。
