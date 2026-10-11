# -*- coding: utf-8 -*-
"""web.bb_pages — BB六合彩 12 个页面（独立数据/规则，不与双色球混用）。"""

from __future__ import annotations

import streamlit as st

from . import service as S
from .ui_common import (bar_chart, card, evidence_tag, fig_to_streamlit,
                        note_box, page_head, show, warn_box, zball_row,
                        balls_row)

ORDER = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡",
         "狗", "猪"]


def page_overview() -> None:
    page_head("BB六合彩总览", "最新开奖 · 生肖 · 数据概况 · 研究结论")
    res = S.bb_overview()
    if not res["ok"]:
        show(res, lambda d: None)
        return
    d = res["data"]
    lat = d["latest"]
    st.markdown("#### 最新开奖")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(f"**期号 {lat['issue']}** ｜ {lat['date']}")
        st.markdown("正码（6个）")
        zball_row(list(zip(lat["regulars"], lat["zodiacs"]["regulars"])))
        st.markdown("特别号码")
        balls_row([lat["special"]], "blue")
        st.caption(f"特别号生肖：**{lat['zodiacs']['special']}** ｜ "
                   f"映射版本 {lat['mapping_version']}")
    with c2:
        st.markdown(f"数据来源：{lat['source']}")
        st.markdown(f"更新时间：{lat.get('updated') or '未知'}")
        st.markdown(f"校验状态：{lat['check']}")
    st.divider()
    st.markdown("#### 生肖信息")
    st.caption("号码→生肖使用按开奖日期生效的映射版本（见“赔率与规则”页）；"
               "并非所有年份共用一张固定表。")
    order = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡",
             "狗", "猪"]
    stats = S.bb_stats("number", window=0)
    if stats["ok"]:
        cnt = {x["n"]: x for x in stats["data"]["items"]}
        rows = []
        for n in lat["regulars"] + [lat["special"]]:
            info = cnt.get(n, {})
            rows.append({"号码": f"{n:02d}",
                         "生肖": S._bb_zodiac(n),
                         "历史出现": info.get("count"),
                         "最近出现距今": info.get("omission")})
        st.dataframe(rows, use_container_width=True, hide_index=True)
    st.divider()
    st.markdown("#### 历史数据概况")
    h = d["health"]
    c = st.columns(4)
    c[0].metric("已收集期数", f"{h['n']:,}")
    c[1].metric("范围", f"{h['first']}~{h['last']}")
    c[2].metric("待核对", h["pending"])
    c[3].metric("最近更新", str(h["updated"])[:16])
    st.caption("重复记录：0（期号主键+幂等导入保证）｜ 缺失记录：期号"
               "连续性仅提示不删除（详见校验报告）")
    st.divider()
    st.markdown("#### 研究结果摘要")
    bt = d["backtest"]
    st.markdown(f"- 最近回测时间：{bt['finished'] or '暂无'}")
    st.markdown(f"- 测试范围：{bt['tested'] or '—'}")
    concl = bt["conclusion"]
    if "未发现可靠优势" in concl:
        note_box(f"与随机基线比较：**{concl}**（这是正常研究结论）")
    else:
        st.markdown(f"- 结论：{concl}")
    st.caption("模型验证状态：validated 规律 = 0（详见“规律/组合”与"
               "“策略回测”页）；不因某生肖近期出现多就称其为高胜率。")


def page_latest() -> None:
    page_head("最新开奖", "正码与特别号码视觉区分；号码可点击查看生肖统计")
    res = S.bb_latest()
    if not res["ok"]:
        show(res, lambda d: None)
        return
    d = res["data"]
    st.markdown(f"### 期号 {d['issue']}（{d['date']}）")
    st.markdown("**正码**")
    zball_row(list(zip(d["regulars"], d["zodiacs"]["regulars"])))
    st.markdown("**特别号码**")
    balls_row([d["special"]], "blue")
    st.caption(f"特别号生肖：{d['zodiacs']['special']} ｜ 映射版本 "
               f"{d['mapping_version']} ｜ 来源 {d['source']} ｜ 更新 "
               f"{d.get('updated')}")
    st.divider()
    st.markdown("### 号码详情（点击查看）")
    st.caption("选择号码查看生肖、所属期与历史统计口径")
    num = st.selectbox("号码", sorted(set(d["regulars"] + [d["special"]])))
    stats = S.bb_stats("number", 0)
    if stats["ok"] and num:
        info = next((x for x in stats["data"]["items"] if x["n"] == num),
                    None)
        if info:
            card(f"号码 {num:02d}",
                 f"生肖：<b>{S._bb_zodiac(num)}</b> ｜ 历史出现 "
                 f"<b>{info['count']}</b> 次 ｜ 距上次 <b>"
                 f"{info['omission']}</b> 期",
                 note=stats["data"]["note"] + "（历史统计，不代表下期）")


def page_numbers() -> None:
    page_head("号码与生肖", "正码与特别号码分开统计；生肖映射按期号版本")
    scope = st.radio("统计对象", ["生肖分布", "全部7号分布", "仅正码分布",
                                  "仅特别号分布"], horizontal=True)
    win = st.select_slider("时间范围", options=[30, 100, 300, 0],
                           value=100, format_func=lambda x:
                           "全部历史" if x == 0 else f"最近{x}期")
    scope_map = {"全部7号分布": "number", "仅正码分布": "regular",
                 "仅特别号分布": "special"}
    if scope == "生肖分布":
        res = S.bb_stats("zodiac_full", win)
        if not res["ok"]:
            show(res, lambda d: None)
            return
        d = res["data"]
        st.caption(d["note"] + " ｜ " + d["sub_note"])
        fig = bar_chart(ORDER, [x["full"] for x in d["items"]],
                        "生肖全开出现次数（出现于7号）", d["note"], "#22d3ee")
        fig_to_streamlit(fig)
        fig = bar_chart(ORDER, [x["special"] for x in d["items"]],
                        "生肖作为特别号出现次数（与上图口径不同）",
                        d["note"], "#f59e0b")
        fig_to_streamlit(fig)
        rows = [{"生肖": x["zodiac"], "全开次数": x["full"],
                 "全开频率": x["full_freq"], "特码次数": x["special"]}
                for x in d["items"]]
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.markdown("**单个生肖历史查询**")
        z = st.selectbox("选择生肖", ORDER)
        item = next(x for x in d["items"] if x["zodiac"] == z)
        note_box(f"在{d['note']}内，生肖【{z}】全开出现 {item['full']} 次、"
                 f"作为特码 {item['special']} 次。这是历史统计，"
                 f"不代表下一期更容易出现。")
    else:
        res = S.bb_stats(scope_map[scope], win)
        if not res["ok"]:
            show(res, lambda d: None)
            return
        d = res["data"]
        st.caption(d["note"] + " ｜ 口径：" + d["sub_note"])
        fig = bar_chart([x["n"] for x in d["items"]],
                        [x["count"] for x in d["items"]],
                        f"{scope}出现次数（01-49）", d["note"], "#ef4444")
        fig_to_streamlit(fig)
        st.dataframe([{"号码": f"{x['n']:02d}", "次数": x["count"],
                       "频率": x["freq"], "遗漏": x["omission"]}
                      for x in d["items"]],
                     use_container_width=True, hide_index=True)
        note_box("正码与特别号码统计口径不同，不可混合解读；"
                 "历史次数高≠下期更容易出现。")


def page_combo() -> None:
    page_head("生肖组合分析", "单肖~五肖；规则、赔率与证据状态逐项展示")
    k = st.radio("玩法类型", [1, 2, 3, 4, 5], horizontal=True,
                 format_func=lambda x: f"{x}生肖（{x}连肖）" if x > 1
                 else "单生肖")
    members = st.multiselect(f"选择 {k} 个生肖", ORDER,
                             default=ORDER[:k] if k <= 2 else [])
    if len(members) != k:
        st.warning(f"请恰好选择 {k} 个生肖（当前 {len(members)} 个）。")
        return
    if st.button("查看规则与历史", type="primary"):
        r = S.bb_combo(k, members)
        if not r["ok"]:
            show(r, lambda d: None)
            return
        d = r["data"]
        st.markdown("#### 规则说明")
        st.write(d["rule_note"])
        st.markdown("#### 赔率与状态")
        st.markdown(f"- 参考赔率：**{d['odds']} 倍**（总返还，含本金）")
        st.markdown(f"- 核实状态：{d['odds_status']}")
        st.markdown(f"- 盈亏平衡命中率：{d['breakeven']:.4f}"
                    f"（1/{d['odds']:.4g}）")
        st.markdown("#### 概率与历史")
        st.markdown(f"- 理论命中概率（不放回精确计算）：**{d['p_theory']:.6f}**")
        hh = d["hist"]
        st.markdown(f"- 历史命中：{hh['hits']}/{hh['n']} = "
                    f"**{(hh['rate'] or 0):.2%}**")
        if k >= 2:
            warn_box("该玩法的中奖口径与赔率均为项目资料（待确认）→ "
                     "历史命中仅供研究，**无法按正式规则结算收益**。")
        else:
            st.caption("单肖口径已由平台截图证实（出现于7号即中）。")
        note_box(hh["vs_theory"] + "；理论概率为数学基准，历史频率围绕"
                 "它波动不构成可利用规律。")


def page_research() -> None:
    page_head("号码推荐（BB）", "本后端提供的是“历史统计参考”，不是AI预测")
    win = st.select_slider("分析历史范围", [30, 50, 100, 251], value=100,
                           format_func=lambda x: f"最近{x}期")
    if st.button("生成研究结果", type="primary"):
        r = S.bb_research(win)
        if not r["ok"]:
            show(r, lambda d: None)
            return
        d = r["data"]
        st.session_state["bb_res"] = d
    d = st.session_state.get("bb_res")
    if not d:
        st.info("尚未生成。选择历史范围后点击“生成研究结果”。")
        return
    st.markdown(f"### {d['name']}")
    st.markdown(f"分析范围：最近 {d['history']} 期")
    c1, c2 = st.columns(2)
    c1.metric("区间最热生肖（出现最多）", f"{d['hot']['zodiac']} "
              f"（{d['hot']['full']} 次）")
    c2.metric("区间最冷生肖（出现最少）", f"{d['cold']['zodiac']} "
              f"（{d['cold']['full']} 次）")
    st.markdown("出现最多的前3： " + "、".join(
        f"{x['zodiac']}({x['full']})" for x in d["top_full"]))
    st.markdown("#### 关联规律的证据等级（来自规律注册表）")
    reg = d.get("registry") or []
    rows = [{"规律": r.get("name"), "类别": r.get("category"),
             "证据": r.get("evidence_zh") or r.get("status"),
             "n": r.get("n"), "q": r.get("p_adjusted")}
            for r in reg[:12]]
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    warn_box(d["disclaimer"])
    if st.button("保存本结果（研究快照）"):
        rr = S.bb_save_research(d)
        if rr["ok"]:
            st.success(f"已保存，生成时间 {rr['data']['generated_at']}；"
                       "证据等级：未证实优于随机")
        else:
            show(rr, lambda x: None)


def page_history() -> None:
    page_head("历史开奖查询", "期号搜索、生肖展示、来源与映射版本")
    q = st.text_input("按期号搜索", "")
    page = st.number_input("页码", 1, 10_000, 1)
    res = S.bb_history(q, int(page), 20)
    if not res["ok"]:
        show(res, lambda d: None)
        return
    d = res["data"]
    st.caption(f"共 {d['total']} 条 ｜ 第 {d['page']}/{d['pages']} 页")
    for it in d["items"]:
        with st.expander(f"{it['issue']} ｜ {it['date']} ｜ 特码生肖 "
                         f"{it['sp_zodiac']} ｜ {it['source']} ｜ "
                         f"映射 {it['mapping_version']}"):
            st.markdown("正码")
            zball_row(list(zip(it["regulars"], it["zodiacs"])))
            st.markdown("特别号码")
            balls_row([it["special"]], "blue")
    if st.button("导出查询结果 CSV"):
        r = S.bb_export_history()
        if r["ok"]:
            st.success(f"已导出 {r['data']['n']} 条 → {r['data']['path']}")
        else:
            show(r, lambda d_: None)


def page_check() -> None:
    page_head("历史对奖（BB）", "按项目规则判定命中；连肖规则未确认不结算")
    kind = st.radio("投注玩法", ["单生肖", "生肖组合"], horizontal=True)
    k = 1 if kind == "单生肖" else st.selectbox("组合大小", [2, 3, 4, 5])
    members = st.multiselect(f"选择 {k} 个生肖", ORDER,
                             default=["鼠"] if k == 1 else [])
    issue = st.text_input("开奖期号（7位）", "202610110215")
    stake = st.number_input("投注额（元）", 1.0, 1000.0, 5.0, step=5.0)
    if st.button("对奖", type="primary"):
        if len(members) != k:
            st.warning(f"请恰好选择 {k} 个生肖。")
            return
        r = S.bb_check("single" if k == 1 else "combo", members, issue,
                       float(stake))
        if not r["ok"]:
            show(r, lambda d: None)
            return
        d = r["data"]
        st.markdown(f"### 期号 {d['issue']}")
        st.markdown("开奖正码：")
        zball_row(list(zip(d["draw"]["regulars"],
                           [S._bb_zodiac(x) for x in d["draw"]["regulars"]])))
        st.markdown(f"特别号码：{d['draw']['special']:02d}（生肖 "
                    f"{S._bb_zodiac(d['draw']['special'])}）")
        st.markdown(f"你的投注：{'+'.join(members)}（{d['bet']['type']}）"
                    f"，注额 {d['bet']['stake']} 元")
        if d["hit"]:
            st.success(f"命中判定：**中**（命中成员："
                       f"{'、'.join(d['hit_members'])}）")
        else:
            st.error(f"命中判定：**未中**（仅 {'、'.join(d['hit_members'])}"
                     f" 出现）")
        if d["settled"]:
            st.markdown(f"赔率 {d['odds']}（{d['odds_status']}）｜ "
                        f"总返还 **{d['return_total']:.2f}** ｜ "
                        f"净盈亏 **{d['net']:+.2f}**")
        else:
            warn_box(d["rule_note"])
        st.caption("提示：命中某个生肖不等于该玩法按正式规则中奖——"
                   "结算口径见上方说明。")


def page_odds() -> None:
    page_head("赔率与规则", "赔率来源、核实状态与生效期逐项可查")
    res = S.bb_odds()
    if not res["ok"]:
        show(res, lambda d: None)
        return
    d = res["data"]
    note_box(d["unit_note"])
    rows = [{"玩法": r["play"], "赔率": r["odds"], "适用": r["zodiac"],
             "生效": r["effective"], "核实状态": r["status"],
             "证据来源": (r["evidence"][:60] + "…")
             if len(r["evidence"]) > 60 else r["evidence"]}
            for r in d["rows"]]
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.markdown("#### 中奖规则")
    st.markdown("- **单肖（已证实）**：所选生肖出现于当期 7 个开奖号即中。")
    st.markdown("- **二~五连肖（待确认）**：暂按“所选生肖全部出现”判定；"
                 "平台真实口径未取得证据前不用于收益结算。")
    st.markdown("#### 生肖映射版本")
    for v in d.get("_", []) or []:
        st.write(v)
    info = S.settings_info()
    if info["ok"]:
        st.caption("映射版本：" + "、".join(info["data"]["bb"][
            "mapping_versions"]) + "（按开奖日期选择生效版本）")
    for w in d.get("warnings", []):
        st.warning(w)


def page_backtest() -> None:
    page_head("策略回测（BB）", "走期回测结果展示；设置项以后端实际支持为准")
    st.markdown("#### 回测设置")
    st.selectbox("玩法", ["全开一肖（当前后端唯一支持）"], disabled=True)
    st.selectbox("赔率规则版本", [S.bb_odds()["data"]["version"]
                                  if S.bb_odds()["ok"] else "—"],
                 disabled=True)
    st.number_input("投注金额", value=5.0, disabled=True,
                    help="后端回测固定平注5元，暂不支持修改")
    st.selectbox("研究策略", ["bb-ens-v1（三级投票）"], disabled=True)
    st.selectbox("开始/结束期号", ["后端固定 warmup=50 起，暂不支持自选"],
                 disabled=True,
                 help="自定义期号范围尚未实现——为防“挑时间段”人为选择偏差")
    st.caption("禁用原因：后端回测接口当前仅支持固定参数；未实现的选项"
               "不开放，避免前端假装可配置。")
    c1, c2 = st.columns(2)
    if c1.button("运行回测（后台任务）", type="primary"):
        r = S.bb_task("backtest")
        st.toast(r["data"]["label"]) if r["ok"] else st.warning(r["error"])
    if c2.button("刷新结果"):
        st.rerun()
    tasks = [t for t in S.TASKS.status() if t["name"] == "bb_backtest"]
    if tasks and tasks[0]["status"] == "running":
        st.info("回测运行中…（真实状态，无百分比进度）")

    res = S.bb_backtest_latest()
    if not res["ok"]:
        show(res, lambda d: None)
        return
    latest = res["data"]["latest"]
    s = latest["summary"]
    st.markdown(f"### 结果（run#{latest['id']}，完成于 "
                f"{latest['finished_at']}）")
    seg = st.radio("分段", ["dev", "val", "holdout"], horizontal=True)
    block = s.get(seg) or {}
    rows = []
    for m, r in block.items():
        if not r:
            continue
        rows.append({"模型": m, "样本": r["n"],
                     "Brier": round(r["brier"], 4),
                     "命中率": f"{r['top1_hit']:.1%}",
                     "95%CI": f"[{r['ci'][0]:.1%},{r['ci'][1]:.1%}]"
                     + ("｜弃权" if m == "ens" else "")})
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.markdown("#### 与随机选择的对比")
    mc = s.get("random_baseline_mc") or {}
    st.markdown(f"- 随机选肖（{mc.get('sims')}次模拟，seed="
                f"{mc.get('seed')}）holdout 命中均值 "
                f"**{(mc.get('mean') or 0):.1%}**，95%区间 "
                f"[{mc['ci95'][0]:.1%}, {mc['ci95'][1]:.1%}]" if mc else "—")
    note_box(s.get("conclusion", "未发现可靠优势"))
    with st.expander("跨时间窗口（holdout 三分块命中率）"):
        st.json(s.get("holdout_blocks", {}))


def page_bankroll() -> None:
    page_head("资金与风险模拟", "固定注额 vs 三阶段策略（同一信号同一开奖）")
    c1, c2 = st.columns(2)
    init = c1.number_input("初始资金（元）", 50.0, 10_000.0, 200.0,
                           step=50.0)
    base = c2.number_input("初始投注额（元）", 1.0, 100.0, 5.0, step=1.0)
    st.markdown("#### 三阶段规则（与代码一致，未改原策略）")
    st.markdown("1) 第1期投初始额；2) 若第1期未中，第2期投 2×初始额"
                "（第1期命中则仍投初始额）；3) 第3期恢复初始额（不论第2期）；"
                "之后按 3 期为周期重复。")
    if st.button("运行模拟", type="primary"):
        with st.spinner("同一信号路径回放 + MC 压力（约数秒）…"):
            r = S.bb_bankroll(init, base)
        if r["ok"]:
            st.code(r["data"], language="text")
        else:
            show(r, lambda d: None)
    warn_box("加倍不改变任何命中概率，只改变注额节奏与资金波动；"
             "连续未命中可能导致资金快速减少。任何策略都不能保证盈利或回本。"
             "赔率来自配置，若“待核实”档被使用，收益结论同样待核实。")


def page_dataops() -> None:
    page_head("数据管理（BB）", "导入、规律、回测、采集等真实任务")
    h = S.bb_health()
    if h["ok"]:
        d = h["data"]
        st.markdown(f"数据：**{d['n']}期** ｜ {d['first']}~{d['last']} "
                    f"｜ 更新 {d['updated']} ｜ 待核 {d['pending']}")
    else:
        show(h, lambda d: None)
    cols = st.columns(3)
    actions = [("import", "导入历史数据", False), ("patterns", "执行规律分析",
                                                    False),
               ("backtest", "执行回测", False), ("report", "生成分析报告",
                                                 False),
               ("sync", "同步最新开奖", True)]
    for i, (act, label, danger) in enumerate(actions):
        with cols[i % 3]:
            if danger:
                go = st.checkbox("我确认使用本地 Cookie 联网同步", key=f"c_{act}")
            else:
                go = True
            if st.button(label, key=f"b_{act}", disabled=not go):
                r = S.bb_task(act)
                st.toast("已提交：" + label) if r["ok"] else \
                    st.warning(r["error"])
    st.caption("同步任务仅调用项目自带 bb_sync.py；Cookie 不会出现在"
               "页面、日志或接口返回中。")
    st.markdown("#### 任务状态")
    tasks = [t for t in S.TASKS.status() if t["name"].startswith("bb_")]
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
    page_head("系统设置（BB）", "只读：赔率版本、映射版本、口径说明")
    res = S.settings_info()
    if not res["ok"]:
        show(res, lambda d: None)
        return
    d = res["data"]
    st.json(d["bb"])
    st.markdown("### 当前中奖规则状态")
    st.markdown("- 单肖全开：**已证实**（平台截图）")
    st.markdown("- 二~五连肖：**待确认**（配置函数可替换，暂按全部出现）")
    st.markdown("### 赔率")
    st.markdown("- 单肖 2.06 / 当年生肖 1.76（截图证据，2026=马年）")
    st.markdown("- 连肖 4.2/10.6/30.7/108：**待核实**（任务书资料）")
    note_box("修改赔率或映射请编辑 config/odds.json、config/mapping.json "
             "并重启；页面不提供写入。")


PAGES = {
    "BB六合彩总览": page_overview,
    "最新开奖": page_latest,
    "号码与生肖": page_numbers,
    "生肖组合分析": page_combo,
    "号码推荐": page_research,
    "历史开奖查询": page_history,
    "历史对奖": page_check,
    "赔率与规则": page_odds,
    "策略回测": page_backtest,
    "资金与风险模拟": page_bankroll,
    "数据管理": page_dataops,
    "系统设置": page_settings,
}
