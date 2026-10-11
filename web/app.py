# -*- coding: utf-8 -*-
"""LottoLab 网页入口。

启动（Windows PowerShell，项目根目录）：
    streamlit run web\app.py
或：
    python -m web.app
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from web import service as S
from web.bb_pages import PAGES as BB_PAGES
from web.ssq_pages import PAGES as SSQ_PAGES
from web.ui_common import evidence_tag, inject, note_box, warn_box

st.set_page_config(page_title="LottoLab", page_icon="🎯",
                   layout="wide")

SYSTEMS = {
    "双色球": {"pages": SSQ_PAGES, "health": S.ssq_health,
               "footer": "双色球数据源：中国福彩官网(2013+)＋500彩票网备用(2003–2012)"},
    "BB六合彩": {"pages": BB_PAGES, "health": S.bb_health,
                 "footer": "BB数据源：bb_sync.py 采集（Cookie 本地保存，"
                           "不经过本页面）"},
}


@st.cache_data(ttl=60)
def _health_cached(sys_key: str) -> dict:
    fn = SYSTEMS[sys_key]["health"]
    return fn()


def main() -> None:
    inject()

    # ---------- 侧边栏：系统切换 + 导航 ----------
    with st.sidebar:
        st.markdown("## 🎯 LottoLab")
        st.caption("彩票历史数据分析 · 研究平台")
        system = st.radio("系统切换", list(SYSTEMS.keys()),
                          key="system",
                          horizontal=True,
                          captions=["红球/蓝球玩法", "正码/生肖玩法"])
        cfg = SYSTEMS[system]
        st.markdown("### 导航")
        page_name = st.radio("页面", list(cfg["pages"].keys()),
                             key=f"nav_{system}",
                             label_visibility="collapsed")
        st.divider()
        st.caption("证据等级图例")
        for lv in ("描述性", "探索性", "已验证", "未通过"):
            st.markdown(evidence_tag(lv), unsafe_allow_html=True)
        st.divider()
        st.caption("本站仅为历史统计与回测研究，"
                   "未证实可预测未来，不构成投注建议。")

    # ---------- 顶栏 ----------
    st.markdown(
        f"<div style='display:flex;justify-content:space-between;"
        f"align-items:center;border-bottom:1px solid #1d2a4a;"
        f"padding-bottom:8px;margin-bottom:14px'>"
        f"<div><span class='ll-title'>{page_name}</span>"
        f"<span class='ll-sub' style='margin-left:14px'>"
        f"当前系统：{system}</span></div></div>",
        unsafe_allow_html=True)

    h = _health_cached(system)
    top = st.columns([3, 2, 1, 1])
    if h["ok"]:
        d = h["data"]
        top[0].metric("数据期数", f"{d['n']:,}")
        top[1].metric("数据更新时间", str(d.get("updated"))[:16])
        top[2].metric("待核对", d.get("pending", 0))
    else:
        top[0].error(h["error"])
        top[1].caption(h.get("hint", ""))
    if top[3].button("刷新", use_container_width=True):
        _health_cached.clear()
        st.rerun()

    # ---------- 内容 ----------
    page_fn = cfg["pages"].get(page_name)
    if page_fn is None:
        st.info(f"页面“{page_name}”尚未实现。")
    else:
        try:
            page_fn()
        except Exception as e:  # 页面级兜底：中文提示，不甩堆栈
            st.error(f"页面渲染出错：{type(e).__name__}")
            st.info("建议：点击右上角“刷新”；若持续出现请到“数据管理”"
                    "检查数据状态，或在终端查看服务日志。")
            with st.expander("技术细节（供排查）"):
                st.code(str(e))

    # ---------- 页脚 ----------
    st.divider()
    footer_txt = cfg["footer"]
    model_info = "模型批次：bb-ens-v1" if system == "BB六合彩" \
        else "模型批次：ssq-A/B/C/D+ens-v1"
    st.markdown(
        f"<div style='color:#6b82a8;font-size:12px'>{footer_txt} ｜ "
        f"最后数据更新：{str(h['data'].get('updated')) if h['ok'] else '—'}"
        f" ｜ {model_info} ｜ 证据等级：未证实优于随机"
        f"{' ｜ 异常：' + h['error'] if not h['ok'] else ''}</div>",
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
