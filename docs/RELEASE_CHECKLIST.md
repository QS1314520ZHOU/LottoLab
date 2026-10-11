# V1.1 发布检查清单（RELEASE CHECKLIST）

每次发布前逐项打勾；未完成项必须在发布说明中列为已知问题。

## 安全

- [x] Git 历史无真实 Cookie / API Key / 密码（`git log -S` + 模式扫描零命中）
- [x] `cookie.txt`、`新建 文本文档.txt`、`.env`、`*.db`、`models/*.joblib` 在 .gitignore
- [x] 代码中 AI 凭据只走环境变量（`bb_v9.py` 硬编码已清除）
- [x] CI 含 secret 扫描步骤
- [ ] 若发现凭据外泄：立即在平台侧轮换/吊销，再用 `git filter-repo` 清史并强推（本仓库未触发）

## 质量门（本地实测）

- [x] `python -m pytest tests/ -q` 全绿（结果见 docs/V1_1_TEST_REPORT.md）
- [x] `python -m compileall app bb_analysis tests` 通过
- [x] `ruff check --select E9,F63,F7,F82,F821` 通过
- [x] 干净克隆 + 全新 venv 走完 双系统核心流程（见测试报告 §4）
- [x] 所有公开 CLI 命令真实运行验证（见测试报告 §3）

## 文档一致性

- [x] README 测试数量 = 实测数量
- [x] 数据范围 2003001–2026116 / BB 251 期与报告一致
- [x] 结论句“未发现可靠优势”已写入两侧回测与报告
- [x] 旧脚本用途表 docs/LEGACY_SCRIPTS.md
- [x] LICENSE 存在且 README 声明

## CI / 远端

- [x] `.github/workflows/ci.yml`（3.11/3.12 矩阵、安装、编译、ruff、密钥扫描、pytest）
- [x] GitHub Actions 实际跑绿：run `38151303387` = success（3.11+3.12 双绿；
      首次失败已修复为自包含测试）
- [ ] 仓库 Description / Topics（需账号网页操作，文案见测试报告 §8）

## 发布物

- [x] reports/ 下四份示例报告（ssq_report / bb_report / bb_pattern_registry / bb_bankroll / ssq_budget）
- [x] docs/V1_1_AUDIT.md 状态表更新
- [x] docs/V1_1_TEST_REPORT.md 附真实命令与输出摘要
- [ ] 打 tag `v1.1`（待远端 CI 绿后执行）
