# 数据源调研与验证记录（双色球 SSQ）

- 调研日期：2026-10-11（本机，中国大陆网络环境）
- 调研方式：实际 HTTP 请求（Python urllib/requests，超时20s，带浏览器 UA 与 Referer）
- 结论摘要：官方来源可用但仅覆盖 2013 年起；备用来源 500 彩票网可达（HTML 表格）；robots.txt 对带查询串 URL 有 Disallow 规则，系统采取低频限速访问。

## 1. 官方来源：中国福利彩票官网

| 项目 | 内容 |
| --- | --- |
| 名称 | 中国福利彩票官方网站开奖公告查询接口 |
| URL | `https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice?name=ssq&issueCount=30&pageNo=1&pageSize=30&systemType=PC` |
| 是否官方 | ✅ 是（www.cwl.gov.cn） |
| API Key | 不需要 |
| 收费 | 免费 |
| 最近验证 | 2026-10-11 |

### 实测结果

1. **HTTP 200**，`Content-Type: text/json;charset=UTF-8`。
2. 响应结构：`{"state":0,"message":"查询成功","total":…,"pageNum":…,"pageNo":…,"pageSize":…,"result":[…]}`。
3. 记录字段（实测样例 2026116 期）：
   - `code`：期号（如 `"2026116"`，7位）
   - `date`：开奖日期（如 `"2026-10-11(日)"`，含星期后缀，需清洗）
   - `red`：红球，逗号分隔升序（`"10,19,24,27,28,33"`）
   - `blue`：蓝球（`"12"`）
   - 其余：`detailsLink`、`week`、`sales`、`prizegrades` 等（本项目不依赖）
4. **排序**：最新期在前（降序）。
5. **分页与窗口**：
   - `issueCount` 定义“最近 N 期”窗口，`total` 回显该值；
   - `pageNo`/`pageSize` 在窗口内翻页（实测 `issueCount=1000,pageSize=50`：pageNo=1 → 2026116..2026067，pageNo=2 → 2026066..2026017，pageNo=3 → 2026016..2025118）；
   - 超窗的 `pageNo` 返回空 `result`（无报错）。
6. **范围查询**：支持 `issueStart` / `issueEnd`（实测 `2013001..2013010` 精确返回10期）。
7. **覆盖范围**：`issueStart=2003001` 返回 `message="查询失败，没有查到数据"` —— **官方接口仅覆盖 2013 年起**（双色球 2003 年首发，2003–2012 需备用源）。
8. **限制**：未发现明确频率文档；实测连续请求正常。系统内置 ≥1 秒间隔、指数退避、结果缓存。
9. **robots.txt**（`https://www.cwl.gov.cn/robots.txt`，HTTP 200）：
   ```
   User-agent:  *
   Disallow: /*?*
   Disallow: /*#*
   Disallow: /.js$ /.php$ /.css$
   ```
   `Disallow: /*?*` 按字面禁止所有带查询串的 URL。本接口即查询串形式。**已知冲突**：本项目按“小频率、可审计、只读、不绕过任何控制”的原则访问，每次拉取间隔 ≥1s、仅初始化与增量更新时调用；如管理方提出要求应立即停用。此风险已在此如实记录，不作掩饰。

## 2. 备用来源：500 彩票网历史数据

| 项目 | 内容 |
| --- | --- |
| 名称 | 500 彩票网双色球历史开奖图表 |
| URL | `https://datachart.500.com/ssq/history/newinc/history.php?start=03001&end=03010` |
| 是否官方 | ❌ 否（仅作备用/交叉核对） |
| API Key | 不需要 |
| 收费 | 免费 |
| 最近验证 | 2026-10-11 |

### 实测结果

- HTTP 200，HTML 文档（约 6.5KB 响应体），页面含 `id="tdata"` 表格容器。
- 需要 `Referer: https://datachart.500.com/ssq/history/history.shtml`。
- 期号格式为 5 位（`03001`），需规范化为 7 位（`2003001`）。
- 文本编码：实测响应字节按 UTF-8 解码正常（历史资料常标注 GB 系，解析时以响应头/探测为准，失败则回退 gb18030）。
- 解析容错：无 `tdata` 表格、0 行可解析数据时适配器报错，不写库。
- **合并策略**：官方优先；备用源数据只有在通过校验且与官方不冲突时补齐 2003–2012；冲突写入 `data_conflicts` 待人工核对，禁止静默覆盖。

## 3. 其他考虑过的来源

| 来源 | 状态 | 说明 |
| --- | --- | --- |
| 各省福彩中心公告页 | 未逐一接入 | 可作为后续交叉核对来源，本期未验证 |
| 第三方聚合 API | 不采用 | 非官方、条款不明，仅允许人工核对 |
| IOI-LottoLab 内置 SQLite | 不迁移 | 参考仓库附带数据库不直接导入，理由见 `IOI_LOTTOLAB_REVIEW.md` |

## 4. 失败/受限记录

- 官方接口 2003–2012 期段查询：`查询失败，没有查到数据`（覆盖缺口，需备用源补）。
- `github.com` 直接克隆参考仓库：`Connection was reset`（网络不可达）；改用 `raw.githubusercontent.com` 只读拉取 README / LICENSE / history_sources.py / ssq_db.py 完成审查。
- robots.txt 的 `/*?*` 规则与查询串接口存在字面冲突（见上，已披露）。

## 5. 访问规范（本项目实现）

- 所有请求：20s 超时、浏览器 UA、官方源带 Referer、重试 2 次指数退避、页间 ≥1s 间隔。
- 不伪造身份、不轮换 IP、不绕过验证码/登录。
- 不在日志与仓库中记录任何凭据（本项目所有来源均无需凭据）。
- `ingestion_runs` 记录每次采集的时间、来源、条数与错误，供审计。
