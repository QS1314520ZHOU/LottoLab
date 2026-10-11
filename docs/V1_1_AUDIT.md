# LottoLab V1.1 验收审计（真实状态）

- 审计时间：2026-10-11（本机 Windows，Python 3.12.7）
- Git：单提交 `bc657fb`（main = origin/main，工作区干净）
- **本轮实测**：`python -m pytest tests/ -q` → **43 passed（4.85s，刚运行）**；
  `python -m compileall app bb_analysis …` → 退出码 0。
- 本文档只记录**实际运行过**的结果；文档中的历史数字不作为本轮证据。

## 状态图例

| 标签 | 含义 |
| --- | --- |
| ✅ 已验证 | 本轮或可复现命令有实际运行证据（附命令） |
| 🟡 已实现未验证 | 代码存在，但本轮未运行/依赖外部条件 |
| ⭕ 未实现 | 未开始或已明确不做 |
| ⚠️ 有风险 | 存在缺陷、依赖未证实假设或潜在破坏性 |

## 1. 总体功能状态

### 数据层

| 功能 | 状态 | 证据/说明 |
| --- | --- | --- |
| SSQ 官方源发现与实测 | ✅ | `python -m app discover-sources`（本轮历史实测，见 DATA_SOURCES.md）；**本轮需重跑确认** |
| SSQ 全量导入/增量更新/校验/报告四命令 | ✅ | 历史实测 3513 期（官方2074+非官方1439）；本轮需重跑确认 |
| SSQ 双源冲突只记录不覆盖 | ✅ | `tests/test_ssq_core.py::test_cross_source_conflict`（43内） |
| SSQ 分页/断点/空响应/HTML异常 | ✅ | `test_cwl_pagination` / `test_cwl_empty_and_bad_json` 等 |
| BB CSV 幂等导入+原文件不破坏 | ✅ | `test_data_update_preserves_old`（43内） |
| 历史开奖“修正”写入 revision 流程 | 🟡 | upsert 冲突路径已实现（SSQ），**未构造真实“官方修正”场景测试**（P1 补） |
| 数据源凭据脱敏/日志无密钥 | ✅ | Git 历史扫描零命中；cookie.txt 在 .gitignore |

### BB 规则/赔率/结算

| 功能 | 状态 | 证据 |
| --- | --- | --- |
| 生肖映射版本与日期选择 | ✅ | `test_mapping_version_by_date` |
| 赔率配置（马@1.76 至2026-12-31 到期回落） | ✅ | `test_config_current_year_odds_configured_from_evidence` |
| 连肖赔率 4.2/10.6/30.7/108 | ⚠️ | 来自任务书，**无平台截图二次证据 → 待确认** |
| 结算公式（stake×(O−1)、1/O 盈亏线） | ✅ | `test_settlement_and_ev_formulas` |
| 已知输入→已知结果黄金结算表 | 🟡 | 现有断言零散，P2 补独立黄金用例 |
| 映射年份切换/边界日测试 | 🟡 | 有 v1/v2 日期选择测试；**缺“生效日当天/前一天”边界与赔率年切换联动**（P2 补） |

### 规律/模型/回测/审计

| 功能 | 状态 | 证据 |
| --- | --- | --- |
| BB 规律注册表15条+BH+holdout状态机 | ✅ | `test_multiple_testing_registry`；`reports/bb_pattern_registry.json` |
| 规律状态：validated=0（无一过双门槛） | ✅ | 注册表导出（0 validated） |
| SSQ 规律7条+experiment_results落库 | ✅ | analyze 实测（q 全部≥0.31，均 exploratory） |
| 模型A/B/C/D+三级投票+弃权 | ✅ | 回测实测；评级全部 unproven |
| 走期冻结（data_cutoff<target 全表） | ✅ | `test_predictions_frozen_before_target`（0泄漏） |
| 基线=玩法口径（BB 47.17/55.39；SSQ C(33,6)×16） | ✅ | `test_random_baseline_correct` / `test_fair_baseline` |
| 特征无未来泄漏 | ✅ | `test_no_future_leakage_rules_and_models` + SSQ 对应测试 |
| 预测历史不可被改写（append-only） | 🟡 | 唯一索引防重，**缺“改动后历史行校验和不变”测试**（P3 补） |
| 跨时间窗口（分块命中稳定性） | ⭕ | **未实现**（P3 补 holdout 分块指标） |
| 报告统一输出“未发现可靠优势” | 🟡 | 现有措辞“证据不足”，P3 统一标准句 |
| AI 只读/失败降级 | ✅ | `test_ai_failure_degrades` + `test_ai_failure_core_flow` |

### 资金策略

| 功能 | 状态 | 证据 |
| --- | --- | --- |
| 固定 vs 三期（同信号同命中断言） | ✅ | `test_three_period_stake_boundaries` |
| 资金不足/归零边界 | 🟡 | 有 ruin_at=2 用例；**缺“余额恰好等于注额/归零当期”金边界**（P2 补） |
| MC 破产压力（i.i.d. 假设已注明） | ✅ | `reports/bb_bankroll.json` 实测 |
| SSQ 固定预算 ROI/返奖/回撤/最长连亏 | ⭕ | **未实现**（P1 补） |

### 双色球候选/组合（P1 范围）

| 功能 | 状态 | 说明 |
| --- | --- | --- |
| 红球/蓝球候选排序+概率+合法性校验 | ⭕ | 未实现（模型已有概率向量，缺票据层） |
| 固定预算组合生成/去重/成本核算 | ⭕ | 未实现 |
| 红蓝分开评估（不把单球指标当中奖率） | 🟡 | 回测有红覆盖/蓝top1；票据层需保持口径 |
| 固定预算随机对照+同预算同周期 | ⭕ | 未实现（P1） |

### 工程/发布

| 功能 | 状态 | 证据 |
| --- | --- | --- |
| .gitignore/凭据排除 | ✅ | 扫描零命中 |
| GitHub Actions CI | ⭕ | 未实现（P0） |
| LICENSE/许可声明 | ⭕ | 未实现（P0） |
| README 数字与实测一致 | ⚠️ | README 写“32 项”，**实测43 → 待统一**（P0） |
| 干净虚拟环境安装全流程 | ⭕ | 未验证（P4） |
| 旧脚本用途标注 | ⭕ | 未实现（P4） |
| 仓库 Description/Topics | ⭕ | 需 GitHub 账号授权，**可能阻塞**（P4） |

## 2. 风险清单（开始前）

1. 无 git 历史可回滚以外的备份（单提交）；大改前建议先打 tag。
2. 连肖赔率、连肖中奖口径缺平台证据（已在文档标待确认）。
3. ~~`reference/IOI-LottoLab`（若本地存在）已忽略不上传；其内容未逐文件审计~~
   → **已关闭（2026-10-11）**：代理可用后浅克隆成功，完成安全扫描（2 EXE 未运行、
   顶层无硬编码凭据、5 db 未迁移）与两处关键读码（审计冻结/走期写法），
   见 `docs/IOI_LOTTOLAB_REVIEW.md` §1.1。
4. GitHub Actions/网页设置需推送后远端验证；直连 github 被墙，需 `http.proxy=socks5://127.0.0.1:10808`。
5. SSQ 官方数据 2003–2012 缺口为非官方补齐（报告必须分别标记——当前部分报告已标，P1 复核）。

## 3. P0–P4 完成后状态回填（V1.1 验收轮）

| 项 | 开始前 | 完成后 | 证据 |
| --- | --- | --- | --- |
| Git 历史密钥 | 未审 | ✅ 零命中 | `git log -S`+模式扫描（1提交） |
| CI | ⭕ | ✅ 已远端验证 | run 38151303387 success（3.11/3.12 双绿，见测试报告§5） |
| LICENSE | ⭕ | ✅ | MIT + 第三方声明 |
| 公开命令实测 | 🟡 | ✅ 全部16条实测 | 测试报告§3 |
| README 数字统一 | ⚠️(32) | ✅ 改为61并注明报告位置 | README |
| SSQ 票据/预算ROI/来源分标/修正场景 | ⭕ | ✅ | `tests/test_v1_1_ssq.py` 7项 + `python -m app tickets` 实测 |
| BB 赔率证据链/边界/黄金结算/资金极值 | 🟡 | ✅ | `tests/test_v1_1_bb.py` 7项 |
| 预测不可重写/分块/结论句 | ⭕/🟡 | ✅ | `tests/test_v1_1_audit.py` 4项；两侧报告含“未发现可靠优势” |
| 干净环境全流程 | ⭕ | ✅（基于旧提交43项） | 测试报告§4；v1.1 合入后复跑 |
| 旧脚本标注 | ⭕ | ✅ | docs/LEGACY_SCRIPTS.md |
| Description/Topics | ⭕ | ⚠️ 阻塞（无凭据） | 文案见测试报告§8 |
| 风险3（IOI未读码） | ⚠️ | ✅ 已关闭 | IOI_LOTTOLAB_REVIEW §1.1 |
