# -*- coding: utf-8 -*-
"""web.ssq_pages — 双色球 10 个页面（全部经 web.service 取真实数据）。"""

from __future__ import annotations

import json
from datetime import date

import streamlit as st

from . import service as S
from .ui_common import (bar_chart, balls_row, card, evidence_tag, fig_to_streamlit,
                        hbar_chart, line_chart, note_box, page_head, show,
                        warn_box)

EVIDENCE_LINE = ("本系统全部模型尚未证实优于随机选择：以下所有百分比为"
                 "历史统计或模型估计，不代表下一期中奖概率。")


def page_overview() -> None:
    page_head("总览", "一屏看懂：最新开奖、本期候选与数据状态")

    def _latest(d):
        st.markdown("#### 最新开奖")
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown(f"**期号 {d['issue']}** ｜ {d.get('date') or '日期缺失'}")
            balls_row(d["reds"], "red")
            balls_row([d["blue"]], "blue")
        with c2:
            st.markdown(f"来源：{d['source_zh']}（{d['source']}）")
            st.markdown(f"数据更新时间：{d.get('updated') or '未知'}")
            st.markdown(f"校验状态：{d['check']}")
    show(S.ssq_latest(), _latest)
    st.divider()

    st.markdown("#### 本期候选号码")
    saved = S.ssq_saved_picks(1)
    if saved["ok"] and saved["data"]:
        g = saved["data"][0]
        st.markdown(f"预测目标（数据截止）期号：**{g['data_cutoff']}** ｜ "
                    f"生成时间：{g['generated_at']}")
        for grp in g["groups"][:3]:
            balls_row(grp["reds"], "red")
            balls_row([grp["blue"]], "blue")
            st.caption(f"编号 {grp['gid']} ｜ 模型 {grp['model_version']} "
                       f"｜ {grp['reason']}")
        warn_box("风险提示：" + EVIDENCE_LINE)
    else:
        st.info("当前暂无可用预测。请到“号码推荐”页点击生成并保存候选号码"
                "（不会自动随机填充）。")
    st.divider()

    st.markdown("#### 研究状态（普通中文）")
    h = S.ssq_health()
    if not h["ok"]:
        show(h, lambda d: None)
        return
    d = h["data"]
    bt = S.ssq_backtest_latest()
    concl = "尚未运行回测"
    if bt["ok"] and bt["data"]["latest"].get("summary"):
        concl = bt["data"]["latest"]["summary"]["_meta"].get(
            "conclusion", "（缺结论字段）")
    c1, c2, c3 = st.columns(3)
    c1.metric("历史数据期数", f"{d['n']:,}")
    c2.metric("最近更新", str(d["updated"])[:16])
    c3.metric("待处理冲突", d["pending"])
    c4, c5 = st.columns(2)
    c4.metric("数据范围", f"{d['first']} ~ {d['last']}")
    c5.metric("最近回测", "已完成" if bt["ok"] else "未运行")
    note_box(f"研究结论：{concl}")
    srcs = "、".join(f"{k}:{v}期" for k, v in d["by_source"].items())
    st.caption(f"数据来源构成：{srcs}（官方与非官方补齐分标存放）")


def page_recommend() -> None:
    page_head("号码推荐", "红球01–33 / 蓝球01–16 独立分析；生成的候选可保存")
    warn_box(EVIDENCE_LINE)
    res = S.ssq_scores()
    if not res["ok"]:
        show(res, lambda d: None)
        return
    d = res["data"]
    lab = d["labels"]

    st.markdown("### 红球分析（01–33）")
    col1, col2, col3 = st.columns([2, 2, 2])
    order = col1.radio("排序方式", ["全部", "按历史频率", "按遗漏期数",
                                    "按模型评分"], horizontal=False)
    win = col2.selectbox("历史窗口", ["全部"] + [str(w) for w in d["windows"]])
    view = d["reds"]
    if win != "全部":
        # 窗口频率需重算 —— 用 trends 接口
        tr = S.ssq_trends(int(win))
        if tr["ok"]:
            for item, f in zip(view, tr["data"]["red_freq"]):
                item["wfreq"] = f
    if order == "按历史频率":
        view = sorted(view, key=lambda x: -x["hist"])
    elif order == "按遗漏期数":
        view = sorted(view, key=lambda x: -x["omission"])
    elif order == "按模型评分":
        view = sorted(view, key=lambda x: -x["score"])
    with col3:
        st.markdown("**字段口径**")
        st.caption(lab["hist"])
        st.caption(lab["omission"])
        st.caption(lab["red_score"])
    rows = []
    for x in view:
        row = {"号码": f"{x['n']:02d}", "历史次数": x["hist"],
               "遗漏": x["omission"], "模型评分": x["score"]}
        if "wfreq" in x:
            row["窗口次数"] = x["wfreq"]
        rows.append(row)
    st.dataframe(rows, use_container_width=True, hide_index=True, height=320)
    note_box("历史出现次数高，不代表下一期一定更容易出现；遗漏较久，"
             "也不代表下一期一定会补出。")

    st.markdown("### 蓝球分析（01–16，独立模型）")
    b_rows = [{"蓝球": f"{x['n']:02d}", "历史次数": x["hist"],
               "遗漏": x["omission"], "模型评分": x["score"],
               "候选排名": x["rank"]} for x in
              sorted(d["blues"], key=lambda x: x["n"])]
    st.dataframe(b_rows, use_container_width=True, hide_index=True)
    st.caption(f"公平随机基线：蓝球 1/16 = {d['baseline']['blue_uniform']:.2%}；"
               f"单红球入选基线 6/33 ≈ {d['baseline']['red_uniform']:.1%}。"
               "红球“入选概率”与整注中奖概率是不同概念。")

    st.divider()
    st.markdown("### 生成候选号码")
    n_g = st.slider("候选组数", 1, 10, 3)
    c1, c2 = st.columns(2)
    if c1.button("生成一组候选号码", type="primary"):
        with st.spinner("正在按模型评分生成…"):
            g = S.ssq_generate(n_g)
        if g["ok"]:
            st.session_state["ssq_gen"] = g["data"]
        else:
            st.session_state.pop("ssq_gen", None)
            show(g, lambda d_: None)
    if c2.button("重新生成（换随机种子）"):
        with st.spinner("生成中…"):
            g = S.ssq_generate(n_g, seed=int(date.today().strftime("%j")))
        show(g, lambda d_: st.session_state.update(ssq_gen=d_))

    gen = st.session_state.get("ssq_gen")
    if gen:
        for grp in gen["groups"]:
            st.markdown(f"**组 {grp['gid']}** ｜ 目标期 {grp['data_cutoff']} "
                        f"｜ 生成 {grp['generated_at']} ｜ "
                        f"模型 {grp['model_version']}")
            balls_row(grp["reds"], "red")
            balls_row([grp["blue"]], "blue")
            st.code(" ".join(f"{n:02d}" for n in grp["reds"])
                    + f" + 蓝 {grp['blue']:02d}", language="text")
            st.caption(grp["reason"])
        if st.button("保存本次预测记录"):
            r = S.ssq_save_picks(gen["groups"], note="网页生成")
            if r["ok"]:
                st.success(f"已保存 {r['data']['saved']} 组（证据等级：未证实）")
            else:
                show(r, lambda d_: None)
    st.divider()
    saved = S.ssq_saved_picks(10)
    if saved["ok"] and saved["data"]:
        st.markdown("### 历史保存记录")
        for item in saved["data"]:
            st.caption(f"记录#{item['id']} ｜ 截止 {item['data_cutoff']} "
                       f"｜ {item['generated_at']} ｜ {len(item['groups'])} 组")


def page_history() -> None:
    page_head("历史开奖", "查询、分页、查看单期与导出 CSV")
    c1, c2, c3 = st.columns([2, 2, 1])
    q = c1.text_input("按期号搜索", "")
    d1 = c2.date_input("起始日期", value=None)
    d2 = c3.date_input("结束日期", value=None)
    page = st.number_input("页码", 1, 10_000, 1)
    res = S.ssq_history(q, str(d1) if d1 else "", str(d2) if d2 else "",
                        int(page), 20)
    if not res["ok"]:
        show(res, lambda d: None)
        return
    d = res["data"]
    st.caption(f"共 {d['total']} 条，第 {d['page']}/{d['pages']} 页")
    for it in d["items"]:
        mark = "🔴" if it["source_zh"] == "非官方备用" else "✅"
        with st.expander(f"{it['issue']} ｜ {it['date'] or ''} ｜ "
                         f"来源{mark}{it['source_zh']} ｜ {it['check']}"):
            balls_row(it["reds"], "red")
            balls_row([it["blue"]], "blue")
    if st.button("导出 CSV（当前筛选最多5000条）"):
        r = S.ssq_export_history()
        if r["ok"]:
            st.success(f"已导出 {r['data']['n']} 条 → {r['data']['path']}")
        else:
            show(r, lambda d_: None)


def page_trends() -> None:
    page_head("号码走势", "图表均标注数据范围、样本量与统计口径")
    win = st.radio("历史范围", [30, 50, 100, 200, 0], horizontal=True,
                   format_func=lambda x: "全部历史" if x == 0 else f"最近{x}期")
    res = S.ssq_trends(win)
    if not res["ok"]:
        show(res, lambda d: None)
        return
    d = res["data"]
    st.caption(f"实际范围：{d['range'][0]} ~ {d['range'][1]}（{d['n']}期，"
               "受可用数据限制）")
    fig = bar_chart(list(range(1, 34)), d["red_freq"],
                    "红球出现频率（次数）", d["notes"]["freq"], "#ef4444")
    fig_to_streamlit(fig)
    fig = bar_chart(list(range(1, 17)), d["blue_freq"],
                    "蓝球出现频率（次数）", d["notes"]["freq"], "#3b82f6")
    fig_to_streamlit(fig)
    fig = bar_chart(list(range(1, 34)), d["red_omission"],
                    "红球遗漏（距上次出现期数）", d["notes"]["omission"],
                    "#22d3ee")
    fig_to_streamlit(fig)
    c1, c2 = st.columns(2)
    with c1:
        fig = bar_chart(["奇数红球", "偶数红球"], [d["odd"], d["even"]],
                        "红球奇偶分布（号码个数）", d["notes"]["oddeven"],
                        "#f59e0b")
        fig_to_streamlit(fig)
    with c2:
        fig = bar_chart([f"小号(01-16)", f"大号(17-33)"],
                        [d["small"], d["big"]],
                        "红球大小分布（号码个数）",
                        f"分界为 {d['big_from']}；{d['notes']['freq']}",
                        "#4ade80")
        fig_to_streamlit(fig)


PATTERN_EXPLAIN = {
    "上期蓝同号": "押“上期蓝球号码”本身，看它下期再次出现的频率。",
    "热10蓝": "押最近10期出现最多的蓝球。",
    "冷200蓝": "押缺席最久的蓝球（回补假设，待验证）。",
    "奇偶翻转+最大遗漏": "上期蓝球奇偶取反，在该奇偶里选遗漏最大的号。",
    "Markov蓝转移": "按“上期蓝球→下期蓝球”最大频转移表选号。",
    "日号mod16": "玄学实验：开奖日的日号除16取余+1（公式预先固定）。",
    "期号尾mod16": "玄学实验：上期期号尾两位除16取余+1（公式预先固定）。",
}


def page_patterns() -> None:
    page_head("规律分析实验室", "统一评分卡：样本量/基线/差异/校正/证据等级")
    res = S.ssq_patterns()
    if not res["ok"]:
        show(res, lambda d: None)
        return
    d = res["data"]
    note_box(f"多重检验分母见“校正”列来源；{d['baseline_note']}。"
             "以下规律均未达到“已验证”标准前，只作研究假设。")
    rows = []
    for it in d["items"]:
        rows.append({
            "规律": it["name"], "一句话解释":
                PATTERN_EXPLAIN.get(it["name"], "见公式列"),
            "样本/尝试": it["n_tests"],
            "p值": round(it["p"], 4) if isinstance(it["p"], float) else it["p"],
            "校正q": (round(it["q"], 4)
                      if isinstance(it["q"], float) else it["q"]),
            "证据等级": it["evidence_zh"],
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)
    for it in d["items"][:7]:
        st.markdown(f"{evidence_tag(it['evidence_zh'])} **{it['name']}** — "
                    f"{PATTERN_EXPLAIN.get(it['name'], '')}")
    warn_box("全部规律的当前状态如实展示；未通过独立验证的不会标为“已验证”。")


MODEL_EXPLAIN = {
    "命中率": "预测目标（如蓝球号码）被猜中的比例。",
    "随机基线": "公平随机选号时的预期表现（蓝球=6.25%）。",
    "Brier 分数": "概率预测质量，越低越好；仅在相同目标与定义下可比。",
    "LogLoss": "概率预测质量，越低越好；对自信的错误预测惩罚更重。",
    "95%置信区间": "真实表现可能落入的不确定范围；跨基线则证据不足。",
    "弃权率": "模型选择不作预测的比例（弃权不计入命中率分母时会标注）。",
}


def page_models() -> None:
    page_head("模型评估", "专业指标集中在这里；首页不放技术术语")
    with st.expander("指标白话解释（点开）"):
        for k, v in MODEL_EXPLAIN.items():
            st.markdown(f"- **{k}**：{v}")
    res = S.ssq_backtest_latest()
    if not res["ok"]:
        show(res, lambda d: None)
        return
    latest = res["data"]["latest"]
    s = latest["summary"]
    seg = st.radio("数据分段", ["dev（开发）", "val（验证）",
                                "holdout（最终留出）"], horizontal=True)
    key = seg.split("（")[0]
    block = s.get(key) or {}
    fair = s["_meta"]["fair"]
    rows = [{"模型": "随机公平基线", "Brier": fair["brier"],
             "LogLoss": fair["logloss"],
             "命中率": f"{fair['top1_hit']:.1%}", "95%CI": "—"}]
    for m in ("A", "B", "C", "D", "ens"):
        r = block.get(m)
        if not isinstance(r, dict):
            continue
        ci = r.get("top1_ci") or r.get("ci") or [0.0, 0.0]
        rows.append({
            "模型": {"A": "A 公平基线", "B": "B 频率模型",
                     "C": "C 贝叶斯", "D": "D 逻辑回归",
                     "ens": "Ensemble 投票"}[m],
            "Brier": round(r["brier"], 4),
            "LogLoss": round(r["logloss"], 4),
            "命中率": f"{r['top1_hit']:.1%}",
            "95%CI": f"[{ci[0]:.1%}, {ci[1]:.1%}]",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)
    if block.get("ens"):
        st.caption(f"Ensemble 弃权率：{block['ens'].get('abstain_rate')}")
    note_box(s["_meta"].get("conclusion", "未发现可靠优势"))
    st.caption(f"run#{latest['id']} ｜ {latest['data_start']}~"
               f"{latest['data_end']} ｜ 完成于 {latest['finished_at']}")


def page_checkprize() -> None:
    page_head("历史对奖", "输入号码与期号，按官方固定奖级核对（研究用途）")
    warn_box("“命中红球数量”“是否中奖”“奖金金额”是三个不同概念："
             "双色球中1个红球并不等于中奖，须红蓝组合满足奖级条件；"
             "一、二等奖为浮动奖，本项目按 0 元保守计入。")
    txt = st.text_input("输入 6 个红球（01-33，空格或逗号分隔）",
                        "01 02 03 04 05 06")
    blue = st.number_input("蓝球 (01-16)", 1, 16, 7)
    mode = st.radio("查询方式", ["指定一期", "扫描全部历史"], horizontal=True)
    issue = ""
    if mode == "指定一期":
        issue = st.text_input("历史期号（如 2026001）", "2026001")
    nums = []
    for tok in txt.replace(",", " ").split():
        if tok.strip():
            try:
                nums.append(int(tok))
            except ValueError:
                st.error(f"无法解析的号码：{tok}")
                return
    if st.button("开始对奖", type="primary"):
        with st.spinner("核对中…"):
            r = S.ssq_check(nums, int(blue), issue if mode == "指定一期"
                            else "")
        if not r["ok"]:
            show(r, lambda d: None)
            return
        d = r["data"]
        st.caption(d["rule_note"])
        if d["scope"] == "single":
            it = d["results"][0]
            st.markdown(f"### 期号 {it['issue']}")
            st.markdown(f"红球命中 **{it['red_hit']}/6** ｜ 蓝球 "
                        f"**{'✓ 中' if it['blue_hit'] else '✗ 未中'}**")
            st.markdown(f"奖级：**{it['tier']}** ｜ 按配置奖金："
                        f"**{it['prize']} 元**")
            if it["tier"].startswith(("一等奖", "二等奖")):
                st.info("浮动奖：本项目不估算具体金额（数据源未提供官方"
                        "逐期奖金）。")
        else:
            st.markdown(f"扫描 {d['n_checked']} 期，中奖期数 "
                        f"**{len(d['results'])}**，配置口径合计 "
                        f"**{d['total_prize']} 元**（浮动奖=0）")
            rows = [{"期号": x["issue"], "红命中": x["red_hit"],
                     "蓝": "✓" if x["blue_hit"] else "",
                     "奖级": x["tier"], "奖金": x["prize"]}
                    for x in d["results"]]
            st.dataframe(rows, use_container_width=True, hide_index=True)
            if st.button("导出对奖结果 CSV"):
                import pandas as pd
                from pathlib import Path
                p = (Path(S.EXPORT_DIR) / "ssq_check.csv")
                p.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(p, index=False,
                                          encoding="utf-8-sig")
                st.success(f"已导出 → {p}")


def page_budget() -> None:
    page_head("资金模拟", "固定预算下的历史投入/返还/回撤（双色球口径）")
    budget = st.number_input("每期预算（元，单注2元）", 2, 200, 20, step=2)
    if st.button("运行对照模拟", type="primary"):
        with st.spinner("按历史测试段计算（同预算、同周期）…"):
            r = S.ssq_budget(float(budget))
        if r["ok"]:
            st.code(r["data"], language="text")
        else:
            show(r, lambda d: None)
    warn_box("一/二等浮动奖按0保守计入；三等3000元主导方差，"
             "短测试段 ROI 噪声可达 ±15%。任何预算策略都不改变中奖概率，"
             "不保证盈利、不承诺回本。")


def page_dataops() -> None:
    page_head("数据管理", "网页触发真实任务；状态、日志与结果均为实际输出")
    h = S.ssq_health()
    if h["ok"]:
        d = h["data"]
        st.markdown(f"当前数据：**{d['n']}期** ｜ {d['first']}~{d['last']} "
                    f"｜ 更新 {d['updated']} ｜ 待核 {d['pending']}")
    else:
        show(h, lambda d: None)
    st.markdown("#### 可用操作")
    cols = st.columns(3)
    actions = [("update", "同步最新数据"), ("validate", "检查数据完整性"),
               ("analyze", "运行分析"), ("backtest", "执行回测"),
               ("report", "生成分析报告")]
    for i, (act, label) in enumerate(actions):
        danger = act == "update"
        with cols[i % 3]:
            if danger:
                ok_go = st.checkbox(f"我确认要联网更新 {label}", key=f"c_{act}")
            else:
                ok_go = True
            if st.button(label, key=f"b_{act}", disabled=not ok_go):
                r = S.ssq_task(act)
                if r["ok"]:
                    st.toast(f"已提交：{label}")
                else:
                    st.session_state[f"err_{act}"] = r
            e = st.session_state.pop(f"err_{act}", None)
            if e:
                st.warning(e["error"] + ("｜" + e["hint"] if e["hint"] else ""))
    st.markdown("#### 任务状态")
    tasks = [t for t in S.TASKS.status() if t["name"].startswith("ssq_")]
    if not tasks:
        st.caption("暂无任务记录。")
    for t in reversed(tasks):
        icon = {"done": "✅", "failed": "❌", "running": "⏳"}.get(
            t["status"], "⌛")
        with st.expander(f"{icon} {t['label']} ｜ {t['status']}"):
            st.caption(f"开始 {t['started']} ｜ 结束 {t['finished']}")
            if t["error"]:
                st.error(t["error"])
            if isinstance(t.get("result"), (dict, list)):
                st.json(t["result"])
            elif isinstance(t.get("result"), str):
                st.code(t["result"][-2500:], language="text")
            st.text("\n".join(t["log"][-20:]))
    if st.button("刷新任务状态"):
        st.rerun()


def page_settings() -> None:
    page_head("设置", "只读展示当前配置、版本与运行环境")
    res = S.settings_info()
    if not res["ok"]:
        show(res, lambda d: None)
        return
    d = res["data"]
    st.markdown("### 双色球")
    st.json(d["ssq"])
    st.markdown("### BB六合彩（仅信息展示）")
    st.json({k: d["bb"][k] for k in d["bb"]})
    st.markdown("### AI 服务")
    st.write(("已配置环境变量" if d["ai"]["env_key_set"]
              else "未配置（推荐）") + " — " + d["ai"]["note"])
    note_box("修改配置需编辑 config/*.json 并重启；页面不提供写入，"
             "防止误改规则。")


PAGES = {
    "总览": page_overview,
    "号码推荐": page_recommend,
    "历史开奖": page_history,
    "号码走势": page_trends,
    "规律分析": page_patterns,
    "模型评估": page_models,
    "历史对奖": page_checkprize,
    "资金模拟": page_budget,
    "数据管理": page_dataops,
    "设置": page_settings,
}
