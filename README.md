# BB六合彩分析系统 + 双色球历史数据与规律分析系统

本工作区包含两套**相互独立**的研究系统（数据、库、配置、测试完全隔离）：

| 系统 | 代码 | 数据 | 定位 |
| --- | --- | --- | --- |
| BB六合彩 | `bb_analysis/`、`bb_sync.py` 等既有脚本 | `data/bb_live_history.csv`、`data/bb_analysis.db` | 高频一肖玩法的规则/概率/回测研究 |
| 双色球 SSQ | `app/` | `data/ssq.db` | 官方数据采集 + 规律挖掘 + 严格走期回测 |

> ⚠️ 两套系统都是**历史数据分析与概率研究工具**：不承诺预测未来、不构成投注建议。
> 任何“回测成绩”都不代表下一期优势；规律未过独立留出验证前一律标记为探索性。

## 一、环境与安装

Python 3.11+（实测 3.12）：

```bash
pip install -r requirements.txt
cp .env.example .env        # 按需修改路径/种子；不含任何密钥
```

## 二、双色球系统（app/）

### 1. 数据命令

```bash
python -m app discover-sources   # 探测并登记数据源（官网+500备用）
python -m app validate-sources   # 双源抽样交叉核对（只记冲突不覆盖）
python -m app import-history     # 初始化全量导入（2003→今，幂等可断点重跑）
python -m app update             # 增量更新（含近10期回查修正）
python -m app validate-data      # 数据质量校验（红蓝范围/重复/缺期提示）
python -m app report-data        # 覆盖与采集审计报告
```

数据源调研实测记录见 `docs/DATA_SOURCES.md`（官网接口仅覆盖2013+；
robots.txt 对查询串有 Disallow 规则，系统以≥1s间隔低频访问并如实披露）。

### 2. 分析与回测

```bash
python -m app analyze            # 特征落库 + 7条规律评分卡 + Markov/红球对自动发现
python -m app backtest           # 严格走期回测（dev/val/holdout 三分段）
python -m app report             # 导出 reports/ssq_report.txt / .json
```

可选 AI 助手（失败自动降级、不阻断核心流程）：

```bash
$env:MIMO_AI_KEY="tp-..."   # 或写入 .env 后手动导出
```

### 3. 输出解读

- 评分卡列：样本量 / 观测 / 基线 / 差 / Wilson CI / 四分块稳定 / p / BH q / 状态；
- `research_score` **仅供排序，不是命中概率**；
- 回测段 `holdout` 只评估一次；`ens/B/C` 相对公平基线差异若 CI 跨 0 → 无证据优于随机；
- 红球覆盖对照随机期望 `6m/33`（top6 → 1.091）。

## 三、BB六合彩系统（bb_analysis/）

完整文档：`docs/BB_PLATFORM.md`（架构/映射/赔率/规律/回测/风险/复现）。

```bash
python -m bb_analysis import        # 导入 bb_live_history.csv（幂等，不动原CSV）
python -m bb_analysis patterns      # 规律注册表 → reports/bb_pattern_registry.json
python -m bb_analysis backtest      # 严格走期回测（冻结预测落库）
python -m bb_analysis bankroll      # 固定 vs 三期资金模拟 → bb_bankroll.json
python -m bb_analysis report        # 汇总报告 → bb_report.txt/.json
python -m bb_analysis ui            # Streamlit 面板（模型/风险/数据时间/证据等级）
```

生肖映射：`config/mapping.json`（来源=BB平台一肖页截图核对，多版本按日期生效）；
赔率：`config/odds.json`（2026 当年生肖=马@1.76，依截图证据配置，**待复核**）。
当前证据状态：**无 validated 规律、回测评级全部 unproven——不存在已验证优势**。

## 四、测试

```bash
python -m pytest tests/ -q
```

32 项全部通过，覆盖：号码校验、重复检测、去重导入、多源冲突、API 异常响应、
分页与断点续跑、时间排序、特征无未来泄漏、回测冻结字段、公平基线（解析=MC）、
可重复性、多重检验/低样本、AI 降级、BB 兼容，以及“样例数据→报告”端到端。

## 五、文档索引

- `docs/DATA_SOURCES.md` — 数据源实测与限制（含失败记录）
- `docs/IOI_LOTTOLAB_REVIEW.md` — 参考仓库审查（REUSE/ADAPT/REIMPLEMENT/REJECT）
- `docs/ARCHITECTURE.md` — 分层架构与不变式
- `docs/BACKTEST_METHODOLOGY.md` — 回测方法学与判定口径
- `docs/MIGRATION_NOTES.md` — 数据库/迁移/数据覆盖说明
- `reports/ssq_report.txt|json` — 示例分析报告（可复现，含参数与版本）
- `samples/sample_draws_simulated.csv` — 标注为模拟数据的样例

## 六、已知限制

1. 官方数据 2013 前缺口由非官方源补齐，已按来源标注。
2. SSQ 投注 ROI/资金模拟未实现（BB 侧另有实现）；蓝球投票为 v1，红球不参与投票。
3. Streamlit 界面为可选依赖（未安装不影响 CLI）；AI 助手需自备 API key。
4. IOI-LottoLab 仅只读审查，未复用其代码、数据与成绩。
