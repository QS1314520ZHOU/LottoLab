# V1.1 测试报告（真实运行结果）

- 环境：Windows / Python 3.12.7 / 本工作区 `D:\BB六合彩`
- 原则：以下每个数字都来自本版本验收期间的**实际命令输出**；历史文档中的旧数字不采信。

## 1. 本工作区测试套件

| 时间点 | 命令 | 结果 |
| --- | --- | --- |
| P0 开始基线 | `python -m pytest tests/ -q` | **43 passed**（4.85s） |
| P1+P2 完成后 | 同上 | **57 passed**（6.82s） |
| P3 完成后 | 同上 | **61 passed**（6.66s） |

覆盖构成：BB 基础层 16 + 双色球核心/e2e 17 + BB平台十项 10 + P1票据 7 +
P2边界/黄金/资金 7 + P3审计 4 = **61**。

## 2. 静态与编译

| 命令 | 结果 |
| --- | --- |
| `python -m compileall app bb_analysis bb_v9.py bb_sync.py tests` | exit 0 |
| `ruff check --select E9,F63,F7,F82,F821 app bb_analysis tests` | `All checks passed!` |

## 3. 公开命令逐条实测（本机）

| 命令 | 实际结果 |
| --- | --- |
| `python -m app discover-sources` | cwl_official=verified；chart500_backup=verified |
| `python -m app validate-sources` | compared=302，mismatches=[] |
| `python -m app import-history` | 幂等：added=0 / unchanged=3513 |
| `python -m app update` | fetched=11，unchanged=11，new_issues=[] |
| `python -m app validate-data` | 校验通过（n=3513，0重复0越界） |
| `python -m app report-data` | 3513期，2003001–2026116 |
| `python -m app analyze` | 特征3513期；7条规律（分母=7）；BH后无显著 |
| `python -m app backtest` | run 执行成功；结论句=未发现可靠优势 |
| `python -m app tickets --budget 20` | model ROI −11.9% / random +19.4%（≥1000奖3 vs5，方差警示已打印） |
| `python -m app report` | 导出 ssq_report.txt/.json（含来源分标 官方2074+非官方1439） |
| `python -m bb_analysis import` | 校验True；new_draws=0（幂等） |
| `python -m bb_analysis patterns` | 15条评分卡；validated=0；导出 registry.json |
| `python -m bb_analysis backtest` | run 执行成功；三分块+结论句=未发现可靠优势 |
| `python -m bb_analysis bankroll` | 固定 vs 三期同信号表+MC压力；导出 bb_bankroll.json |
| `python -m bb_analysis report` | 导出 bb_report.txt/.json（含结论句） |
| `python -m bb_analysis ui` | streamlit 脚本 exit=0；headless HTTP 200（此前实测） |

## 4. 干净环境全流程（本版本真实执行）

- 方式：`git clone .` 到 `%TEMP%\lottolab_clean`（不含 db）→ 新建 `.venv`
  → `pip install -r requirements.txt` → 编译/ruff/双系统导入/回测/测试。
- 结果（脚本日志 `clean_env.log`）：

| 步骤 | 结果 |
| --- | --- |
| pip install | exit 0（**numpy 2.5.4**、streamlit 1.65.0 —— 与本地 numpy1.26.4 双兼容） |
| compileall / ruff | exit 0 / All checks passed |
| bb import + backtest | exit 0（251期入库） |
| app import-history（网络） | exit 0（3513期全量入库） |
| app backtest | exit 0 |
| pytest（**已提交状态**） | **43 passed**（10.03s）——该克隆基于提交 `bc657fb`，不含尚未提交的 V1.1 测试；V1.1 合入后需复跑（见 §7） |

## 5. CI（GitHub Actions）——已回填

- 工作流：`.github/workflows/ci.yml`（3.11+3.12 矩阵：安装→编译→ruff→密钥扫描→
  bb 数据准备→pytest）。
- 首次运行 `38151141221`：**failure**（pytest 阶段——两个新测试依赖真实 ssq.db，
  CI 无该库）；修复为自包含夹具（commit `8a6c4f1`）。
- 修复后运行 `38151303387`：**completed / success**（3.11 与 3.12 两 job 全绿）。
  URL: https://github.com/QS1314520ZHOU/LottoLab/actions/runs/38151303387
- 结论：61 项测试在干净 Ubuntu、Python 3.11 与 3.12 下全部通过；
  ruff/compile/密钥扫描通过。

## 6. 无法本地验证/阻塞项

- 仓库 Description / Topics：需登录 github.com 网页或带凭据 API——
  **无凭据，阻塞**；建议文案见 §8。
- 3.11 矩阵：本地只有 3.12，3.11 结论依赖 CI（推送后回填）。

## 7. 复现步骤（干净环境）

```bash
git clone https://github.com/QS1314520ZHOU/LottoLab.git && cd LottoLab
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m bb_analysis import && python -m bb_analysis backtest
python -m app import-history && python -m app backtest
python -m pytest tests/ -q     # 期望 61 passed（V1.1 起）
```

## 8. 建议的仓库 Description / Topics（需人工设置）

- Description: `可安装、可测试、可复现、可审计的彩票历史数据分析与研究平台（双色球+BB）— 不承诺预测优势`
- Topics: `lottery-analysis` `walk-forward-backtest` `statistical-audit` `python` `sqlite`
  `reproducible-research` `ssq`
