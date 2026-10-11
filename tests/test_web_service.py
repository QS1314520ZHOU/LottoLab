# -*- coding: utf-8 -*-
"""V1.2/1.3 前端服务层测试（验收清单 1–15 的可自动化部分）。

约定：只读真实库 + 临时任务；不联网、不污染历史数据。
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from web import service as S

ROOT = Path(__file__).resolve().parent.parent


# 1-2 首页真实数据 / 最新开奖正确
def test_home_reads_real_latest():
    for fn, n_red in ((S.ssq_latest, 6), (S.bb_latest, None)):
        r = fn()
        assert r["ok"], r["error"]
        d = r["data"]
        if n_red:
            assert len(d["reds"]) == n_red
            assert 1 <= d["blue"] <= 16
            assert d["issue"].isdigit()
        else:
            assert len(d["regulars"]) == 6
            assert 1 <= d["special"] <= 49
            assert d["mapping_version"].startswith("bb-")


def test_latest_matches_database():
    # 与库内最后一期逐字段一致（防“编造开奖”）
    from app import db
    from app.config import load_config
    cfg = load_config()
    with db.connect(cfg.db_path) as conn:
        rows = db.load_draws(conn)
    rows.sort(key=lambda r: r["issue"])
    last = rows[-1]
    d = S.ssq_latest()["data"]
    assert d["issue"] == last["issue"]
    assert d["reds"] == [last[f"r{i}"] for i in range(1, 7)]
    assert d["blue"] == last["b1"]


# 3 红蓝数量与范围
def test_scores_ranges():
    d = S.ssq_scores()["data"]
    assert len(d["reds"]) == 33 and len(d["blues"]) == 16
    assert all(1 <= x["n"] <= 33 for x in d["reds"])
    assert all(1 <= x["n"] <= 16 for x in d["blues"])
    assert d["baseline"]["blue_uniform"] == pytest.approx(1 / 16)


# 4 推荐号码合法且无重复红
def test_generate_valid_unique():
    d = S.ssq_generate(3)["data"]
    assert len(d["groups"]) == 3
    for g in d["groups"]:
        assert len(g["reds"]) == 6 and len(set(g["reds"])) == 6
        assert all(1 <= x <= 33 for x in g["reds"])
        assert g["reds"] == sorted(g["reds"])
        assert 1 <= g["blue"] <= 16
        assert g["gid"] and g["data_cutoff"] and g["model_version"]


# 5 无预测时不造假（service 层如实返回空列表；页面渲染空态）
def test_no_fake_predictions():
    r = S.ssq_saved_picks(1000)
    assert r["ok"]
    assert isinstance(r["data"], list)
    for item in r["data"]:
        # 每条保存记录都必须来自真实生成结构
        for g in item["groups"]:
            assert set(g) >= {"gid", "reds", "blue", "model_version",
                              "data_cutoff", "generated_at"}


# 6 历史筛选
def test_history_filter():
    latest = S.ssq_latest()["data"]["issue"]
    r = S.ssq_history(query=latest, page=1, per_page=5)
    assert r["ok"] and r["data"]["total"] == 1
    assert r["data"]["items"][0]["issue"] == latest
    empty = S.ssq_history(query="不存在的期号zzz")
    assert empty["ok"] and empty["data"]["total"] == 0


# 7 对奖规则（黄金用例：命中真实开奖 → 一等奖浮动=0；4+1=3000）
def test_check_prize_golden():
    d = S.ssq_latest()["data"]
    r = S.ssq_check(d["reds"], d["blue"], d["issue"])
    assert r["ok"]
    it = r["data"]["results"][0]
    assert it["red_hit"] == 6 and it["blue_hit"] == 1
    assert "一等奖" in it["tier"] and it["prize"] == 0.0
    # 4红+对蓝 → 三等3000（用真实开奖的4红 + 2个未中红）
    reds4 = d["reds"][:4] + [
        x for x in range(1, 34) if x not in d["reds"]][:2]
    r2 = S.ssq_check(reds4, d["blue"], d["issue"])
    assert r2["ok"]
    it2 = r2["data"]["results"][0]
    assert it2["red_hit"] == 4 and it2["tier"].startswith("三等")
    assert it2["prize"] == 3000.0
    # 非法输入 → 中文错误
    bad = S.ssq_check([1, 1, 2, 3, 4, 5], 7, d["issue"])
    assert not bad["ok"] and "红球" in bad["error"]


# 8 双色球与 BB 规则隔离
def test_system_isolation():
    info = S.settings_info()["data"]
    assert Path(info["ssq"]["db"]).name == "ssq.db"
    assert Path(info["bb"]["db"]).name == "bb_analysis.db"
    assert info["ssq"]["db"] != info["bb"]["db"]
    # BB 对奖不接受双色球期号
    r = S.bb_check("single", ["鼠"], "2026001")
    assert not r["ok"] and "未找到期号" in r["error"]
    # 双色球对奖不接受 BB 期号
    r2 = S.ssq_check([1, 2, 3, 4, 5, 6], 7, "202610110215")
    assert not r2["ok"]
    # 连肖规则待确认 → 不结算
    ok = S.bb_check("combo", ["鼠", "马"], "202610110215")
    assert ok["ok"] and ok["data"]["settled"] is False
    assert "无法按正式规则结算" in ok["data"]["rule_note"]


# 9 指标与公平基线定义一致
def test_metrics_match_baseline_defs():
    from app.baseline import prize1_probability
    from app.config import load_config
    ssq = load_config()
    bt = S.ssq_backtest_latest()["data"]["latest"]["summary"]
    fair = bt["_meta"]["fair"]
    assert fair["brier"] == pytest.approx(1 - 1 / 16)      # 多类Brier解析值
    assert fair["logloss"] == pytest.approx(math_log(16))
    assert fair["top1_hit"] == pytest.approx(1 / 16)
    assert prize1_probability(ssq) == pytest.approx(1 / 17_721_088)
    bbt = S.bb_backtest_latest()["data"]["latest"]["summary"]
    base = bbt["meta"]["baseline_hit"]
    assert 0.47 < base < 0.49          # BB 全开均值口径（非1/12、非6.25%）
    assert "非双色球口径" in bbt["meta"]["baseline_note"]


def math_log(x: float) -> float:
    import math
    return math.log(x)


# 10 数据导入不破坏旧数据（BB 幂等导入接口）
def test_bb_import_idempotent():
    r = S.bb_task("import")
    # 提交后轮询至完成
    for _ in range(60):
        st = S.TASKS.status("bb_import")
        if st["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert st["status"] == "done", st.get("error")
    assert st["result"]["counts"]["new_draws"] == 0
    assert st["result"]["valid"] is True


# 11 任务防重复提交
def test_task_duplicate_guard():
    name = "test_dup_guard"
    gate = threading.Event()

    def slow():
        gate.wait(timeout=10)
        return "ok"

    r1 = S.TASKS.submit(name, "测试任务", slow)
    assert r1["ok"]
    r2 = S.TASKS.submit(name, "测试任务", slow)
    assert not r2["ok"] and "重复提交" in r2["error"]
    gate.set()
    for _ in range(40):
        if S.TASKS.status(name)["status"] == "done":
            break
        time.sleep(0.1)
    assert S.TASKS.status(name)["status"] == "done"


# 12 AI 缺失时核心可用
def test_ai_absent_core_available(monkeypatch):
    monkeypatch.delenv("MIMO_AI_KEY", raising=False)
    import importlib
    import bb_analysis.ai as bai
    importlib.reload(bai)
    assert not bai.explain("x", "y")["ok"]
    # 核心接口不受影响
    assert S.ssq_latest()["ok"] and S.bb_overview()["ok"]


# 13 API 异常中文提示
def test_errors_are_chinese():
    r = S.ssq_check([1, 2, 3, 4, 5, 6], 99, "x")
    assert not r["ok"]
    assert any("一" <= ch <= "鿿" for ch in r["error"])
    assert r["hint"] != ""


# 14 CLI 原功能仍可用
def test_cli_still_works():
    kw = dict(capture_output=True, text=True, cwd=str(ROOT), timeout=60,
              encoding="utf-8", errors="replace")
    p = subprocess.run([sys.executable, "-m", "app", "--help"], **kw)
    assert p.returncode == 0 and "import-history" in p.stdout
    p2 = subprocess.run([sys.executable, "-m", "bb_analysis", "--help"], **kw)
    assert p2.returncode == 0 and "backtest" in p2.stdout


# 15 页面函数可渲染（bare 模式全量烟测，按钮不会被点击）
def test_all_pages_render_bare():
    from web import bb_pages, ssq_pages
    errors = []
    for name, fn in {**ssq_pages.PAGES, **bb_pages.PAGES}.items():
        try:
            fn()
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")
    assert not errors, "\n".join(errors)


# 附：赔率与组合接口
def test_odds_and_combo_endpoints():
    odds = S.bb_odds()["data"]
    assert odds["rows"][0]["odds"] == 1.76
    assert any("总返还" in r for r in [odds["unit_note"]])
    combo = S.bb_combo(5, ["鼠", "牛", "虎", "兔", "龙"])["data"]
    assert combo["rule_confirmed"] is False
    assert "待确认" in combo["odds_status"] or \
        "待确认" in combo["rule_note"]
