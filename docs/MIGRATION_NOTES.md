# 迁移与数据说明

## 数据库

- SSQ：`data/ssq.db`（`python -m app` 首次任意命令自动 `migrate()`；
  schema 版本记录在 `schema_migrations`，当前 v1）。
- BB：`data/bb_analysis.db`（`bb_analysis.database.migrate()`）。
- 备份：直接复制 db 文件即可（SQLite 单文件）；建议备份前执行
  `python -m app validate-data` 确认无 pending 冲突。

## 数据来源与覆盖

| 段 | 来源 | 标注 |
| --- | --- | --- |
| 2013001–2026116（2074 期） | 中国福彩官网接口 | official |
| 2003001–2012xxx（1439 期） | 500 彩票网图表页 | 非官方备用，交叉核对后入库 |

- 官方接口实测仅覆盖 2013 起（见 `docs/DATA_SOURCES.md`）。
- 每行 `draws.source` 记录来源；`first_seen/last_checked` 记录时间。
- 冲突不覆盖：进 `data_conflicts` 并把该期标 `pending_review`。
- 更新：`python -m app update`（含近 10 期回查修正）。

## IOI-LottoLab 数据/模型

**不迁移**。其仓库附带 SQLite、模型状态、EXE 一律不导入、不运行，
理由见 `docs/IOI_LOTTOLAB_REVIEW.md`。

## 模拟数据

- 测试内生成，`source=simulated`；
- e2e 使用独立临时库，绝不写入真实 `ssq.db`；
- `samples/sample_draws_simulated.csv` 为样例，文件内自带 source 列。
