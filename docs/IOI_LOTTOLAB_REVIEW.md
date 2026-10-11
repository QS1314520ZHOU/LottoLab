# IOI-LottoLab 参考仓库审查报告

- 审查日期：2026-10-11
- 参考仓库：https://github.com/ioi-studio/IOI-LottoLab （MIT License © 2026 IOI Studio）
- 审查方式：`github.com` 直连克隆失败（Connection was reset），改经 `raw.githubusercontent.com` 只读拉取以下文件逐行/整篇审阅：
  - `README.md`（整篇）
  - `LICENSE`（整篇，MIT）
  - `history_sources.py`（整篇）
  - `ssq_db.py`（整篇）
- 其余清单文件（`ssq_analyzer.py`、`quant_models.py`、`rigorous_backtest_v8.py`、`*_prediction_audit.py`、`auto_optimizer.py`、`single_ticket_ensemble.py` 等）：**仅依据 README 的架构描述审阅，未逐行读码**，结论一律按“未验证”处理，不复用其任何代码或成绩。
- 未运行仓库附带的任何 EXE 或预编译程序；未导入其数据库/模型状态文件。

## 1. 总体结论

该项目最有价值的部分是**研究流程设计**（走期回测、开奖前锁定审计、稳定性闸门、公开失败结果），而不是任何“预测能力”。其 README 自述：近500期隔离验证中算法任意奖命中率 32.80% vs 公平基线 32.93%（p=0.6203），红蓝候选多重校正后 p=1.0，已回退等概率生产逻辑。**本项目不引用其任何历史成绩作为己方结论。**

## 2. 模块逐项决定

| 参考模块 | 审查深度 | 决定 | 证据与原因 |
| --- | --- | --- | --- |
| 数据采集 `history_sources.py` | 整篇读码 | **ADAPT** | 证实：官方 CWL 接口支持 `issueStart/issueEnd`、仅覆盖2013+；备用源 datachart.500.com HTML 表格、5位期号规范化；双源合并官方优先+冲突清单。本项目自写适配器（独立 fetch/parse/validate、限速与退避、冲突入表），采用同一来源策略但不复制其代码。 |
| 数据库 `ssq_db.py` | 整篇读码 | **REIMPLEMENT** | 其 schema 仅一张 `draws` 表（把和值/跨度等派生特征内联进开奖表），无来源台账、冲突记录、采集运行、规律定义、审计字段，不满足本项目“原始数据不可被特征覆盖+可审计”要求。借鉴“派生列+校验后写入”思想，按本项目 10 表 schema 重写。 |
| 特征工程 `ssq_analyzer.py` | README 级 | **REIMPLEMENT** | 未读源码，不复用；本项目按自定特征规范实现并配泄漏测试。 |
| 规律挖掘/量化模型 `quant_models.py` | README 级 | **REIMPLEMENT** | 其候选模型清单（EWMA、Markov、贝叶斯收缩等）作为方法参考；实现与统计门槛由本项目自写。 |
| 走期回测 `rigorous_backtest_v8.py` | README 级 | **REIMPLEMENT** | 采纳其公开原则（目标期不得进入当期训练、与公平基线同口径、报告失败），代码自写，含三分段与多重检验。 |
| 预测审计 `*_prediction_audit.py` | README 级 | **ADAPT** | “开奖前冻结、开奖后按原记录结算”的台账思想直接采纳，落到本项目 `predictions` 表（generated_at < target_date、data_cutoff 字段）与结算流程。 |
| 参数优化 `auto_optimizer.py` | README 级 | **REJECT（暂不引入）** | README 自述存在“近100期优势不保持”的调参压力；在本项目数据量下引入自动调参极易把验证集变成训练集。先做固定参数基线，后续若引入必须配独立留出段。 |
| 单注聚合 `single_ticket_ensemble.py` | README 级 | **REJECT（本期不做）** | 依赖大量历史回测数据与在线权重，超出本期范围；其“同成本比较”思想记入 BACKTEST_METHODOLOGY 待办。 |
| 界面/仪表盘（Flask web_app） | README 级 | **ADAPT** | 借鉴“数据-候选-回测-审计”四区布局；本项目用可选 Streamlit 实现，不引入 Flask 依赖。 |
| 附带 SQLite / 模型状态 / EXE | 未使用 | **REJECT** | 计划明确禁止迁移其数据与状态文件；EXE 不运行；数据库与代码版本一致性无法验证。 |
| README 成绩表 | 已读 | **REJECT（不作为结论）** | 17.4 要求独立复核；本项目不复现、不引用其数值成绩。 |

## 3. 许可证与安全审查

- LICENSE = MIT：允许复用但需保留版权声明。**本项目实际未复制其代码**，仅在本文档与 README 致谢段保留署名与许可证声明；若后续复制任何片段，须在文件头保留 `Copyright (c) 2026 IOI Studio` 与 MIT 全文链接。
- 依赖许可证：其使用 Flask、requests、bs4 等（均 MIT/BSD 系）；本项目不因此引入 Flask。
- 安全检查（基于已读源码与 README）：`history_sources.py` / `ssq_db.py` 无硬编码密钥、无危险命令、网络请求只读且带超时；README 提及仓库含预编译 EXE——**未运行、不建议运行**，已按计划隔离。
- 数据一致性：仓库自带历史库/审计库与代码版本关系无法验证 → 不迁移。

## 4. 独立复核项对照（计划 17.4）

| 检查点 | 本项目对策 |
| --- | --- |
| 预测只用目标期之前数据 | `data_cutoff < target` 由回测器强制 + `test_no_future_leakage` |
| 数据完整/去重/规则一致 | `validate-data` 命令 + 冲突表 + 单测 |
| 开奖前冻结 | `predictions.generated_at` 不可变、结算只读原记录 |
| 同成本比较 | 回测固定注数/预算口径，写入 BACKTEST_METHODOLOGY |
| 测试集反复调参 | 开发/验证/留出三分段；本期参数冻结后只跑一次留出 |
| 随机基线组合空间 | `baseline.py` 用 C(33,6)×16 解析概率+固定种子 MC 双轨 |
| 多重比较 | Bonferroni + BH-FDR，记录尝试总数 |
| 结果可复现 | 固定种子、记录依赖版本、报告带参数快照 |

## 5. 归属声明

- 参考项目原有功能、其公开成绩与本项目无关；本项目任何“已验证”结论均以本地代码+本地数据重跑为准。
- 实际复用：**零行代码**（来源策略与审计思想为 ADAPT 级借鉴，见上表）。
- 重新实现：数据库、特征、模型、回测、投票、报告全部为本项目新写。
