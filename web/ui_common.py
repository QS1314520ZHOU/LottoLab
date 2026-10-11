# -*- coding: utf-8 -*-
"""web.ui_common — 深色量化工作台主题、导航壳、号码球、卡片、图表。"""

from __future__ import annotations

import io
from typing import Any, Callable, Optional

import streamlit as st

CSS = """
<style>
  .stApp { background-color: #0b1220; color: #dbe7ff; }
  header[data-testid="stHeader"] { background: #0b1220; }
  [data-testid="stSidebar"] {
      background: #0e1730; border-right: 1px solid #1d2a4a;
  }
  .blockcontainer { max-width: 1400px; }
  h1,h2,h3 { color: #e8f1ff !important; letter-spacing: .02em; }
  .ll-title { font-size: 22px; font-weight: 700; color: #e8f1ff; }
  .ll-sub { color: #8aa3c8; font-size: 13px; margin-bottom: 10px; }
  .ll-card {
      background: #131c31; border: 1px solid #1d2a4a; border-radius: 12px;
      padding: 14px 16px; margin-bottom: 10px;
  }
  .ll-metric { font-size: 30px; font-weight: 700; color: #22d3ee; }
  .ll-metric-label { font-size: 13px; color: #8aa3c8; }
  .ball { display:inline-flex; align-items:center; justify-content:center;
      width:44px; height:44px; border-radius:50%; margin:4px;
      font-weight:700; font-size:17px; color:#fff;
      box-shadow: inset -3px -4px 8px rgba(0,0,0,.35); }
  .ball-red { background: linear-gradient(145deg,#f05252,#b91c1c); }
  .ball-blue { background: linear-gradient(145deg,#4f8df7,#1d4ed8); }
  .ball-gray { background: linear-gradient(145deg,#64748b,#334155); }
  .ll-tag { display:inline-block; padding:2px 10px; border-radius:999px;
      font-size:12px; margin-right:6px; border:1px solid transparent; }
  .tag-desc { background:#1f2937; color:#9ca3af; border-color:#374151; }
  .tag-exp  { background:#3b2f10; color:#f59e0b; border-color:#78350f; }
  .tag-ing  { background:#0e3a4a; color:#22d3ee; border-color:#155e75; }
  .tag-val  { background:#10341f; color:#4ade80; border-color:#166534; }
  .tag-fail { background:#3b1212; color:#f87171; border-color:#7f1d1d; }
  .ll-note { background:#0f1b33; border-left:3px solid #22d3ee;
      padding:8px 12px; font-size:13px; color:#a9c0e4; border-radius:6px; }
  .ll-warn { background:#2a1f0c; border-left:3px solid #f59e0b;
      padding:8px 12px; font-size:13px; color:#fcd9a0; border-radius:6px; }
  div[data-testid="stMetric"] {
      background:#131c31; border:1px solid #1d2a4a; padding:10px 14px;
      border-radius:10px; }
  .stButton>button { background:#152238; color:#cfe0ff;
      border:1px solid #27406e; border-radius:8px; }
  .stButton>button:hover { border-color:#22d3ee; color:#22d3ee; }
  footer { color:#6b82a8 !important; }
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def ball(n: int, kind: str = "red", pad: int = 2) -> str:
    cls = {"red": "ball-red", "blue": "ball-blue"}.get(kind, "ball-gray")
    return (f"<span class='ball {cls}' style='margin:{pad}px'>"
            f"{n:02d}</span>")


def balls_row(nums: list[int], kind: str = "red") -> None:
    st.markdown("".join(ball(n, kind) for n in nums),
                unsafe_allow_html=True)


def zball_row(pairs: list[tuple[int, str]]) -> None:
    """(号码, 生肖) 展示：正码灰底+生肖小字。"""
    html = "".join(
        f"<span class='ball ball-gray' title='{z}' style='margin:4px'>"
        f"{n:02d}<small style='font-size:10px;display:block;"
        f"line-height:8px'>{z}</small></span>" for n, z in pairs)
    st.markdown(html, unsafe_allow_html=True)


EVIDENCE_CLS = {
    "描述性": "tag-desc", "探索性": "tag-exp", "验证中": "tag-ing",
    "已验证": "tag-val", "未通过": "tag-fail",
    "未证实优于随机": "tag-exp", "待核实": "tag-ing",
}


def evidence_tag(level: str) -> str:
    cls = EVIDENCE_CLS.get(level, "tag-desc")
    return f"<span class='ll-tag {cls}'>{level}</span>"


def card(title: str = "", body: str = "", note: str = "") -> None:
    inner = (f"<div class='ll-card'><b>{title}</b><br>{body}"
             + (f"<div class='ll-note' style='margin-top:8px'>{note}</div>"
                if note else "") + "</div>")
    st.markdown(inner, unsafe_allow_html=True)


def warn_box(text: str) -> None:
    st.markdown(f"<div class='ll-warn'>{text}</div>",
                unsafe_allow_html=True)


def note_box(text: str) -> None:
    st.markdown(f"<div class='ll-note'>{text}</div>",
                unsafe_allow_html=True)


def page_head(title: str, sub: str = "") -> None:
    st.markdown(f"<div class='ll-title'>{title}</div>", unsafe_allow_html=True)
    if sub:
        st.markdown(f"<div class='ll-sub'>{sub}</div>",
                    unsafe_allow_html=True)


def show(result: dict, ok_renderer: Callable[[Any], None]) -> None:
    """统一渲染 service 返回：失败→中文错误+建议；成功→回调。"""
    if not result.get("ok"):
        st.error(result.get("error") or "未知错误")
        hint = result.get("hint")
        if hint:
            st.info(f"建议：{hint}")
        return
    ok_renderer(result["data"])


def safe(fn: Callable[[], Any], what: str) -> Any:
    try:
        return fn()
    except Exception as e:
        st.error(f"{what}失败：{type(e).__name__}")
        st.info("建议：刷新页面重试；若持续失败请到“数据管理”检查数据状态。")
        return None


# ---------- matplotlib 深色图表 ----------

def _style(ax, title: str, note: str) -> None:
    ax.set_facecolor("#131c31")
    ax.figure.set_facecolor("#0b1220")
    ax.title.set_color("#e8f1ff")
    ax.title.set_text(title)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color("#8aa3c8")
    ax.tick_params(colors="#8aa3af")
    for spine in ax.spines.values():
        spine.set_color("#1d2a4a")
    if note:
        ax.figure.text(0.01, 0.005, note, color="#6b82a8", fontsize=8)


def bar_chart(labels: list, values: list, title: str, note: str = "",
              color: str = "#22d3ee", highlight: Optional[set] = None
              ) -> "Any":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                       "Noto Sans CJK SC", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(12, 3.6), dpi=120)
    colors = [color] * len(labels)
    if highlight:
        colors = ["#f59e0b" if str(l) in highlight or l in highlight
                  else color for l in labels]
    ax.bar([str(x) for x in labels], values, color=colors, width=0.72)
    _style(ax, title, note)
    fig.tight_layout()
    return fig


def hbar_chart(labels: list, values: list, title: str, note: str = ""
               ) -> "Any":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                       "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(7, 5.2), dpi=120)
    ax.barh([str(x) for x in labels][::-1], values[::-1], color="#22d3ee")
    _style(ax, title, note)
    fig.tight_layout()
    return fig


def line_chart(labels: list, values: list, title: str, note: str = "",
               color: str = "#22d3ee", hline: Optional[float] = None
               ) -> "Any":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                       "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(11, 3.6), dpi=120)
    ax.plot(labels, values, color=color, linewidth=1.6)
    if hline is not None:
        ax.axhline(hline, color="#f59e0b", linestyle="--", linewidth=1,
                   label="基线")
        ax.legend(facecolor="#131c31", labelcolor="#dbe7ff")
    _style(ax, title, note)
    fig.tight_layout()
    return fig


def fig_to_streamlit(fig) -> None:
    st.pyplot(fig)
    import matplotlib.pyplot as plt
    plt.close(fig)
