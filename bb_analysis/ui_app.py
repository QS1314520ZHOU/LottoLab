# -*- coding: utf-8 -*-
"""T10 · Streamlit 本地界面：模型表现 / 风险指标 / 数据时间 / 证据等级。

启动：streamlit run bb_analysis/ui_app.py
红线：页面固定展示“无保证中奖”与证据等级，不提供任何必中宣传。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from bb_analysis.config import PROJECT_ROOT, load_config  # noqa: E402

st.set_page_config(page_title="BB六合彩研究面板", layout="wide")

cfg = load_config()
from bb_analysis import database as db  # noqa: E402

db.migrate(cfg.db_path)

st.title("BB六合彩历史分析 · 研究面板")
st.warning("⚠️ 本面板仅为历史统计与回测研究：无任何已验证的预测优势，"
           "不构成投注建议，不保证中奖。")

c1, c2, c3 = st.columns(3)
with db.connect(cfg.db_path) as conn:
    meta = conn.execute(
        "SELECT COUNT(*) n, MIN(issue) lo, MAX(issue) hi, "
        "MAX(fetched_at) upd FROM draws").fetchone()
    runs = [dict(r) for r in conn.execute(
        "SELECT id, model_version, data_start, data_end, summary_json AS summary FROM backtest_runs ORDER BY id DESC LIMIT 1")]
    n_pred = conn.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"]

c1.metric("数据期数", f"{meta['n']}")
c2.metric("数据范围", f"{meta['lo']} ~ {meta['hi']}")
c3.metric("数据最后更新", (meta["upd"] or "-")[:19])

tab1, tab2, tab3 = st.tabs(["模型表现", "风险指标", "证据等级"])

with tab1:
    if runs and runs[0].get("summary"):
        s = json.loads(runs[0]["summary"])
        st.caption(f"run#{runs[0]['id']} · {runs[0]['model_version']} · "
                   f"{runs[0]['data_start']}~{runs[0]['data_end']} · "
                   f"预测 {n_pred} 条")
        for seg in ("dev", "val", "holdout"):
            block = s.get(seg) or {}
            if not block:
                continue
            rows = [{"模型": m, "Brier": round(r["brier"], 4),
                     "LogLoss": round(r["logloss"], 4),
                     "top1命中": f"{r['top1_hit']:.1%}",
                     "95%CI": f"[{r['ci'][0]:.1%},{r['ci'][1]:.1%}]",
                     "弃权率": (f"{r['abstain_rate']:.1%}"
                                if r.get("abstain_rate") == r.get(
                                    "abstain_rate") else "-")}
                    for m, r in block.items() if r]
            st.subheader(f"{seg} 段")
            st.dataframe(rows, use_container_width=True, hide_index=True)
        st.subheader("评级（val 段 · 预定义门槛）")
        st.json(s.get("ratings"))
        st.subheader("holdout 检验（BH 校正）")
        st.dataframe(s.get("holdout_tests", []), use_container_width=True,
                     hide_index=True)
        cal = s.get("calibration_ens_val") or {}
        st.caption(f"ens@val 校准 ECE = {cal.get('ece', float('nan')):.4f}")
    else:
        st.info("尚无回测记录：先运行 `python -m bb_analysis backtest`")

with tab2:
    br = PROJECT_ROOT / "reports" / "bb_bankroll.json"
    if br.exists():
        data = json.loads(br.read_text("utf-8"))
        st.caption(f"信号路径 {data.get('path_len')} 期 · "
                   f"命中率 {data.get('hit_rate'):.1%}")
        st.dataframe(data.get("rows", []), use_container_width=True,
                     hide_index=True)
        st.subheader("策略对照文本")
        st.caption("固定注额 vs 三期策略：同一信号同一开奖；"
                   "加倍不改变中奖概率，只放大波动。")
    else:
        st.info("尚无资金模拟：先运行 `python -m bb_analysis bankroll`")
    st.subheader("固定 vs 三期（说明）")
    st.markdown(
        "- 三期规则：第1期 s；第2期 若第1期未中则 2s 否则 s；第3期 s（周期重复）。\n"
        "- 净收益 = stake×(O−1)（O=总返还倍数，含本金）；未中 = −stake。\n"
        "- **任何注额策略都不提高命中率。**")

with tab3:
    reg_path = PROJECT_ROOT / "reports" / "bb_pattern_registry.json"
    if reg_path.exists():
        reg = json.loads(reg_path.read_text("utf-8"))
        by_status: dict[str, list] = {}
        for r in reg["results"]:
            by_status.setdefault(r["status"], []).append(r)
        for status in ("validated", "exploratory", "rejected",
                       "descriptive"):
            items = by_status.get(status, [])
            color = {"validated": "🟢", "exploratory": "🟡",
                     "rejected": "🔴", "descriptive": "⚪"}[status]
            st.subheader(f"{color} {status}（{len(items)}）")
            if items:
                st.dataframe(
                    [{"pid": r["pid"], "名称": r["name"],
                      "n": r["n"],
                      "观测": (f"{r['observed']:.1%}"
                               if r["observed"] is not None else "-"),
                      "基线": (f"{r['baseline']:.1%}"
                               if r["baseline"] is not None else "-"),
                      "q": (round(r["p_adjusted"], 3)
                            if r["p_adjusted"] is not None else "-")}
                     for r in items],
                    use_container_width=True, hide_index=True)
        st.caption(f"多重检验分母（BH）= {reg['bh_denominator']}；"
                   "validated 需同时满足 q<0.05 与 holdout 同向复现。")
    else:
        st.info("尚无注册表：先运行 `python -m bb_analysis patterns`")

st.divider()
st.caption("复现步骤见 docs/BB_PLATFORM.md ｜ 数据来源见该文档“数据”节 "
           "｜ 生成时间以数据最后更新为准")
