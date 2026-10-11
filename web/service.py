# -*- coding: utf-8 -*-
"""web.service — 前端服务层（唯一数据/任务入口）。

设计约定：
  * 所有函数返回 {"ok": bool, "data": ..., "error": str, "hint": str}；
    错误必须是普通用户可读的中文原因+建议，禁止抛 traceback 给页面。
  * SSQ 与 BB 各自独立函数（不同数据库/规则），禁止交叉取数。
  * 不在本层复述算法：直接调用 app/ 与 bb_analysis/ 的既有函数。
  * 长任务经 TaskStore 线程执行：状态可查、日志真实、防重复提交。
  * 不打印/返回任何 Cookie、密钥。
"""

from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPORT_DIR = PROJECT_ROOT / "reports" / "exports"

# 证据等级中文（两系统统一口径）——映射仅做展示，不改变后端判定
EVIDENCE_ZH = {
    "descriptive": "描述性", "exploratory": "探索性", "validating": "验证中",
    "validated": "已验证", "rejected": "未通过",
    "unproven": "未证实优于随机", "pending": "待核实",
}


def ok(data: Any = None) -> dict:
    return {"ok": True, "data": data, "error": "", "hint": ""}


def err(error: str, hint: str = "") -> dict:
    return {"ok": False, "data": None, "error": error, "hint": hint}


def _zh_exc(e: Exception, what: str) -> dict:
    return err(f"{what}失败：{type(e).__name__}",
               "请查看服务端日志；若数据库缺失，先在数据管理页执行导入。")


# ---------------- 任务机制 ----------------

@dataclass
class Task:
    name: str
    label: str
    status: str = "queued"        # queued/running/done/failed
    started: Optional[str] = None
    finished: Optional[str] = None
    log: list[str] = field(default_factory=list)
    result: Any = None
    error: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "label": self.label, "status": self.status,
                "started": self.started, "finished": self.finished,
                "log": self.log[-50:], "error": self.error,
                "result": self.result}


class TaskStore:
    """进程内任务表：同名任务运行中时拒绝重复提交。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[str, Task] = {}
        self._pool = ThreadPoolExecutor(max_workers=2,
                                        thread_name_prefix="lottolab")

    def status(self, name: Optional[str] = None) -> list[dict] | dict:
        with self._lock:
            if name is None:
                return [t.to_dict() for t in self._tasks.values()]
            t = self._tasks.get(name)
            return t.to_dict() if t else err(f"任务不存在：{name}")

    def submit(self, name: str, label: str, fn: Callable[[], Any]) -> dict:
        with self._lock:
            cur = self._tasks.get(name)
            if cur and cur.status in ("queued", "running"):
                return err(f"任务“{label}”正在运行，请勿重复提交",
                           "等待其结束后再试；状态可在下方任务列表查看。")
            task = Task(name=name, label=label, status="queued")
            self._tasks[name] = task

        def _run() -> None:
            task.status = "running"
            task.started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            task.log.append(f"[{task.started}] 开始：{label}")
            try:
                task.result = fn()
                task.status = "done"
                task.log.append("完成")
            except Exception as exc:  # 任何异常转中文状态，不外泄堆栈
                task.status = "failed"
                task.error = f"{type(exc).__name__}: {exc}"
                task.log.append(f"失败：{task.error}")
                task.log.append(traceback.format_exc(limit=3))
            finally:
                task.finished = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self._pool.submit(_run)
        return ok(task.to_dict())


TASKS = TaskStore()


# ================= 双色球 SSQ =================

def _ssq_cfg():
    from app.config import load_config
    return load_config()


def _ssq_draws(status: Optional[str] = "ok") -> list[dict]:
    from app import db
    cfg = _ssq_cfg()
    db.migrate(cfg.db_path)
    with db.connect(cfg.db_path) as conn:
        return db.load_draws(conn, status=status)


def ssq_health() -> dict:
    try:
        cfg = _ssq_cfg()
        if not Path(cfg.db_path).exists():
            return err("双色球数据库不存在",
                       "请先到“数据管理”页点击“同步/导入历史数据”。")
        rows = _ssq_draws(None)
        if not rows:
            return err("数据库为空，暂无开奖可展示",
                       "到“数据管理”页执行“同步最新数据”。")
        rows.sort(key=lambda r: r["issue"])
        last = rows[-1]
        return ok({
            "n": len(rows), "first": rows[0]["issue"], "last": last["issue"],
            "updated": last.get("last_checked") or last.get("first_seen"),
            "pending": sum(1 for r in rows
                           if r.get("check_status") != "ok"),
            "by_source": _count_by(rows, "source"),
        })
    except Exception as e:
        return _zh_exc(e, "读取双色球数据状态")


def _count_by(rows: list[dict], key: str) -> dict:
    out: dict[str, int] = {}
    for r in rows:
        out[str(r.get(key))] = out.get(str(r.get(key)), 0) + 1
    return out


def _source_zh(src: str) -> str:
    return "官方" if "official" in src else "非官方备用"


def ssq_latest() -> dict:
    h = ssq_health()
    if not h["ok"]:
        return h
    rows = _ssq_draws()
    rows.sort(key=lambda r: r["issue"])
    d = rows[-1]
    return ok({
        "issue": d["issue"], "date": d.get("draw_date"),
        "reds": [d[f"r{i}"] for i in range(1, 7)], "blue": d["b1"],
        "source": d["source"], "source_zh": _source_zh(d["source"]),
        "updated": d.get("last_checked"), "check": d["check_status"],
    })


def ssq_history(query: str = "", date_from: str = "", date_to: str = "",
                page: int = 1, per_page: int = 20) -> dict:
    try:
        rows = _ssq_draws(None)
        rows.sort(key=lambda r: r["issue"], reverse=True)
        if query:
            rows = [r for r in rows if query.strip() in r["issue"]]
        if date_from:
            rows = [r for r in rows if (r.get("draw_date") or "") >= date_from]
        if date_to:
            rows = [r for r in rows if (r.get("draw_date") or "") <= date_to]
        total = len(rows)
        pages = max(1, (total + per_page - 1) // per_page)
        page = min(max(1, page), pages)
        slice_ = rows[(page - 1) * per_page: page * per_page]
        items = [{
            "issue": r["issue"], "date": r.get("draw_date"),
            "reds": [r[f"r{i}"] for i in range(1, 7)], "blue": r["b1"],
            "source_zh": _source_zh(r["source"]),
            "check": r["check_status"],
        } for r in slice_]
        return ok({"items": items, "total": total, "page": page,
                   "pages": pages})
    except Exception as e:
        return _zh_exc(e, "查询历史开奖")


def ssq_export_history(limit: int = 5000) -> dict:
    try:
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        rows = _ssq_draws(None)
        rows.sort(key=lambda r: r["issue"], reverse=True)
        rows = rows[:limit]
        path = EXPORT_DIR / "ssq_history.csv"
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            f.write("issue,draw_date,r1,r2,r3,r4,r5,r6,b1,source,check\n")
            for r in rows:
                f.write(",".join([
                    r["issue"], r.get("draw_date") or "",
                    *[str(r[f"r{i}"]) for i in range(1, 7)],
                    str(r["b1"]), r["source"], r["check_status"]]) + "\n")
        return ok({"path": str(path), "n": len(rows)})
    except Exception as e:
        return _zh_exc(e, "导出CSV")


# ---- 候选/推荐 ----

_feat_cache: dict[str, Any] = {"key": None, "feats": None}


def _ssq_features() -> dict:
    """全量 fv1 特征（按 数据长度+末期 缓存）。"""
    from app.features import build_features
    rows = _ssq_draws()
    rows.sort(key=lambda r: r["issue"])
    key = (len(rows), rows[-1]["issue"] if rows else "")
    if _feat_cache["key"] == key:
        return _feat_cache["feats"]
    feats = build_features(rows, _ssq_cfg())
    _feat_cache.update(key=key, feats=feats)
    return feats


def ssq_scores() -> dict:
    """红33/蓝16 网格：历史计数+遗漏+模型估计（口径分离标注）。"""
    try:
        import numpy as np
        from app.models import model_c_blue, model_c_red
        cfg = _ssq_cfg()
        rows = _ssq_draws()
        rows.sort(key=lambda r: r["issue"])
        feats = _ssq_features()
        last = feats[rows[-1]["issue"]]
        red_hist = [0] * 34
        blue_hist = [0] * 17
        red_om = [len(rows)] * 34
        blue_om = [len(rows)] * 17
        for i, r in enumerate(rows):
            for k in range(1, 7):
                v = r[f"r{k}"]
                red_hist[v] += 1
                red_om[v] = len(rows) - 1 - i
            b = r["b1"]
            blue_hist[b] += 1
            blue_om[b] = len(rows) - 1 - i
        red_p, _lo, _hi = model_c_red(last, cfg)
        blue_dist = model_c_blue(last, cfg)
        reds = [{"n": i, "hist": red_hist[i], "omission": red_om[i],
                 "score": round(float(red_p[i - 1]), 4)}
                for i in range(1, 34)]
        blues = [{"n": i, "hist": blue_hist[i], "omission": blue_om[i],
                  "score": round(float(blue_dist.probs[i - 1]), 4),
                  "rank": 0} for i in range(1, 17)]
        for rank, item in enumerate(
                sorted(blues, key=lambda x: -x["score"]), start=1):
            item["rank"] = rank
        return ok({
            "reds": reds, "blues": blues,
            "windows": list(cfg.windows),
            "evidence": "unproven",
            "labels": {
                "red_score": "入选概率估计（贝叶斯平滑·未外部校准）",
                "blue_score": "蓝球模型评分（后验质量，非已证实中奖率）",
                "hist": "历史出现次数（历史统计，非预测）",
                "omission": "距上次出现期数（遗漏，非补出承诺）",
            },
            "baseline": {"blue_uniform": 1 / 16,
                         "red_uniform": 6 / 33},
            "data_cutoff": rows[-1]["issue"],
        })
    except Exception as e:
        return _zh_exc(e, "计算号码分析")


def ssq_generate(n_groups: int = 1, seed: int = 42) -> dict:
    """真实生成候选（调用 tickets.make_tickets 模型策略），不伪造。"""
    try:
        from app.tickets import make_tickets, rank_candidates
        cfg = _ssq_cfg()
        rows = _ssq_draws()
        rows.sort(key=lambda r: r["issue"])
        feats = _ssq_features()
        ranks = rank_candidates(feats[rows[-1]["issue"]], cfg)
        budget = 2.0 * n_groups
        out = make_tickets(cfg, budget, "model", ranks, seed=seed)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        groups = []
        for i, t in enumerate(out["tickets"], start=1):
            groups.append({
                "gid": uuid.uuid4().hex[:8],
                "reds": sorted(t.reds), "blue": t.blue,
                "generated_at": now,
                "data_cutoff": rows[-1]["issue"],
                "model_version": "ssq-C_bayes-fv1",
                "reason": "红球按贝叶斯入选估计加权抽样；蓝球取模型评分前列轮换"
                          "（历史统计参考，尚未证实优于随机）",
            })
        return ok({"groups": groups, "n": len(groups),
                   "evidence": "unproven"})
    except Exception as e:
        return _zh_exc(e, "生成候选号码")


def ssq_save_picks(groups: list[dict], note: str = "") -> dict:
    try:
        from app import db
        cfg = _ssq_cfg()
        if not groups:
            return err("没有可保存的候选", "请先生成候选号码。")
        g0 = groups[0]
        with db.connect(cfg.db_path) as conn:
            db.insert_predictions(conn, [{
                "run_id": None,
                "target_issue": g0.get("data_cutoff", "saved"),
                "generated_at": g0.get("generated_at") or datetime.now().isoformat(
                    timespec="seconds"),
                "model_version": g0.get("model_version", "ssq-saved"),
                "market": "ssq_ticket",
                "candidates": json.dumps(groups, ensure_ascii=False),
                "chosen": json.dumps(groups[0], ensure_ascii=False),
                "votes_json": json.dumps(
                    {"note": note, "evidence": "unproven",
                     "purpose": "网页保存的研究候选"}, ensure_ascii=False),
            }])
        return ok({"saved": len(groups)})
    except Exception as e:
        return _zh_exc(e, "保存预测记录")


def ssq_saved_picks(limit: int = 20) -> dict:
    try:
        from app import db
        cfg = _ssq_cfg()
        with db.connect(cfg.db_path) as conn:
            rows = [dict(r) for r in conn.execute(
                "SELECT * FROM predictions WHERE market='ssq_ticket' "
                "ORDER BY id DESC LIMIT ?", (limit,))]
        items = []
        for r in rows:
            groups = json.loads(r["candidates"] or "[]")
            items.append({"id": r["id"], "generated_at": r["generated_at"],
                          "data_cutoff": r["target_issue"],
                          "groups": groups,
                          "note": json.loads(r["votes_json"] or "{}").get(
                              "note", "")})
        return ok(items)
    except Exception as e:
        return _zh_exc(e, "读取已保存预测")


# ---- 对奖 ----

def ssq_check(reds: list[int], blue: int, issue: str = "") -> dict:
    """单期对奖（issue 必填）或全史扫描（issue=''）。规则=config 固定奖。"""
    try:
        from app import db
        from app.tickets import prize_for, validate_blue, validate_reds
        cfg = _ssq_cfg()
        reds = [int(x) for x in reds]
        blue = int(blue)
        validate_reds(reds, cfg)
        validate_blue(blue, cfg)
        with db.connect(cfg.db_path) as conn:
            rows = db.load_draws(conn, status=None)
        rows.sort(key=lambda r: r["issue"])

        def hit_of(d: dict) -> dict:
            dr = {d[f"r{i}"] for i in range(1, 7)}
            rh = len(set(reds) & dr)
            bh = int(blue == d["b1"])
            prize = prize_for(rh, bool(bh), cfg)
            tier = ("一等奖" if (rh == 6 and bh) else
                    "二等奖" if (rh == 5 and bh) else
                    "三等奖" if (rh == 4 and bh) else
                    "四等奖" if (rh == 3 and bh) else
                    "五等奖" if (rh == 2 and bh) else
                    "六等奖" if (bh and rh <= 1) else "未中奖")
            if tier in ("一等奖", "二等奖"):
                tier += "（浮动奖·本项目按0计）"
            return {"issue": d["issue"], "red_hit": rh, "blue_hit": bh,
                    "tier": tier, "prize": prize}

        if issue:
            d = next((x for x in rows if x["issue"] == issue.strip()), None)
            if d is None:
                return err(f"未找到期号 {issue}",
                           "请检查期号是否存在于历史数据中。")
            res = hit_of(d)
            return ok({"scope": "single", "results": [res],
                       "rule_note": "固定奖级来自 config/app.json（官方公开规则）；"
                                    "一/二等浮动奖按0保守计入。"})
        wins = [hit_of(d) for d in rows]
        wins = [w for w in wins if w["tier"] != "未中奖"]
        return ok({"scope": "all", "results": wins, "n_checked": len(rows),
                   "total_prize": sum(w["prize"] for w in wins),
                   "rule_note": "扫描全部历史；一/二等浮动奖按0保守计入。"})
    except ValueError as e:
        return err(f"号码格式不正确：{e}",
                   "红球须为01–33的6个不重复号码；蓝球01–16。")
    except Exception as e:
        return _zh_exc(e, "对奖")


# ---- 回测/规律/资金（读结果 + 任务） ----

def ssq_backtest_latest() -> dict:
    try:
        from app import db
        cfg = _ssq_cfg()
        with db.connect(cfg.db_path) as conn:
            rows = [dict(r) for r in conn.execute(
                "SELECT id,params,data_start,data_end,started_at,finished_at,"
                "summary FROM backtest_runs ORDER BY id DESC LIMIT 5")]
        if not rows or not rows[0]["summary"]:
            return err("暂无回测记录",
                       "请先到“数据管理”页运行“执行回测”。")
        for r in rows:
            r["summary"] = json.loads(r["summary"]) if r["summary"] else None
        return ok({"latest": rows[0], "runs": rows})
    except Exception as e:
        return _zh_exc(e, "读取回测结果")


def ssq_trends(window: int = 100) -> dict:
    """走势图数据：窗口内红/蓝频率、奇偶、大小、遗漏（口径标注齐全）。"""
    try:
        rows = _ssq_draws()
        rows.sort(key=lambda r: r["issue"])
        if window and len(rows) > window:
            rows = rows[-window:]
        n = len(rows)
        if n == 0:
            return err("没有可用数据", "请先同步历史数据。")
        red_cnt = [0] * 34
        blue_cnt = [0] * 17
        red_om = [-1] * 34
        blue_om = [-1] * 17
        odd = even = big = small = 0
        for i, r in enumerate(rows):
            for k in range(1, 7):
                v = r[f"r{k}"]
                red_cnt[v] += 1
                red_om[v] = i
                if v % 2:
                    odd += 1
                else:
                    even += 1
                if v >= 17:
                    big += 1
                else:
                    small += 1
            b = r["b1"]
            blue_cnt[b] += 1
            blue_om[b] = i
        return ok({
            "n": n, "range": [rows[0]["issue"], rows[-1]["issue"]],
            "red_freq": red_cnt[1:], "blue_freq": blue_cnt[1:],
            "red_omission": [(n - 1 - x) if x >= 0 else n
                             for x in red_om[1:]],
            "odd": odd, "even": even, "big": big, "small": small,
            "big_from": 17,
            "notes": {
                "freq": f"窗口最近{n}期（{rows[0]['issue']}~{rows[-1]['issue']}）"
                        "的出现次数；历史频率≠下期概率",
                "omission": f"截至{rows[-1]['issue']}的遗漏期数；"
                            "遗漏久≠下期补出",
                "oddeven": f"窗口内6红奇偶合计数（样本 {n} 期 × 6 = {n*6}）",
            },
        })
    except Exception as e:
        return _zh_exc(e, "计算走势图")


def ssq_patterns() -> dict:
    try:
        from app import db
        cfg = _ssq_cfg()
        with db.connect(cfg.db_path) as conn:
            rows = [dict(r) for r in conn.execute(
                "SELECT e.*, p.name, p.category, p.formula "
                "FROM experiment_results e "
                "LEFT JOIN pattern_definitions p ON e.pattern_id=p.id "
                "ORDER BY e.id DESC LIMIT 50")]
        items = []
        for r in rows:
            eff = json.loads(r["effect"]) if isinstance(r.get("effect"), str) \
                and r["effect"].startswith("{") else r.get("effect")
            items.append({
                "name": r.get("name") or r.get("test_name"),
                "category": r.get("category") or "检验",
                "conclusion": r.get("conclusion"),
                "evidence_zh": EVIDENCE_ZH.get(r.get("conclusion", ""),
                                               "探索性"),
                "p": r.get("p_value"), "q": r.get("p_adjusted"),
                "n_tests": r.get("n_tests"),
                "effect": eff, "created": r.get("created_at"),
            })
        return ok({"items": items,
                   "baseline_note": "基线=各自玩法公平随机（非统一1/12）"})
    except Exception as e:
        return _zh_exc(e, "读取规律分析")


def ssq_budget(budget: float = 20.0) -> dict:
    from app.tickets import run_budget_study
    return ok(run_budget_study(budget_yuan=budget))


# ---- 任务注册 ----

def ssq_task(action: str) -> dict:
    """action: update | validate | analyze | backtest | report | tickets"""
    labels = {
        "update": "双色球·同步最新数据", "validate": "双色球·校验数据",
        "analyze": "双色球·运行分析", "backtest": "双色球·执行回测",
        "report": "双色球·生成报告",
    }
    if action not in labels:
        return err(f"未知操作：{action}")

    def _fn():
        from app import ingest, reports as rp
        from app.validate import report_database, validate_rows
        cfg = _ssq_cfg()
        if action == "update":
            return ingest.update(cfg)
        if action == "validate":
            rows = _ssq_draws(None)
            norm = [{"issue": r["issue"], "draw_date": r.get("draw_date"),
                     "red": [r[f"r{i}"] for i in range(1, 7)],
                     "blue": r["b1"]} for r in rows]
            rep = validate_rows(norm, cfg)
            return rep.to_dict()
        if action == "analyze":
            from app.analyze import run_analysis
            return run_analysis(cfg)[-3000:]
        if action == "backtest":
            from app.backtest import run_backtest
            return run_backtest(cfg)[-3000:]
        if action == "report":
            return rp.generate_report(cfg)[-1500:]
        return "ok"

    return TASKS.submit(f"ssq_{action}", labels[action], _fn)


# ================= BB 六合彩 =================

def _bb_cfg():
    from bb_analysis.config import load_config
    return load_config()


def _bb_draws() -> list[dict]:
    from bb_analysis import database as db
    cfg = _bb_cfg()
    db.migrate(cfg.db_path)
    with db.connect(cfg.db_path) as conn:
        rows = db.load_draws(conn)
    rows.sort(key=lambda r: r["issue"])
    return rows


def _bb_zodiac(n: int) -> str:
    mod = {1: "马", 2: "蛇", 3: "龙", 4: "兔", 5: "虎", 6: "牛",
           7: "鼠", 8: "猪", 9: "狗", 10: "鸡", 11: "猴", 0: "羊"}
    return mod[n % 12]


def _bb_zodiac_version(issue: str) -> str:
    from bb_analysis.config import load_config
    from datetime import date
    cfg = _bb_cfg()
    d = date(int(issue[:4]), int(issue[4:6]), int(issue[6:8]))
    return cfg.mapping.for_date(d).version


def bb_health() -> dict:
    try:
        rows = _bb_draws()
        if not rows:
            return err("BB 数据库为空",
                       "先到“数据管理”页执行“导入历史数据”。")
        last = rows[-1]
        return ok({
            "n": len(rows), "first": rows[0]["issue"], "last": last["issue"],
            "updated": last.get("fetched_at"),
            "pending": sum(1 for r in rows
                           if r.get("check_status") != "ok"),
            "by_source": {s: sum(1 for r in rows if r["source"] == s)
                          for s in {r["source"] for r in rows}},
        })
    except Exception as e:
        return _zh_exc(e, "读取BB数据状态")


def bb_latest() -> dict:
    h = bb_health()
    if not h["ok"]:
        return h
    rows = _bb_draws()
    d = rows[-1]
    regulars = [d[f"p{i}"] for i in range(1, 7)]
    sp = d["sp"]
    return ok({
        "issue": d["issue"], "date": d["issue"][:4] + "-" + d["issue"][4:6]
        + "-" + d["issue"][6:8],
        "regulars": regulars, "special": sp,
        "zodiacs": {"regulars": [_bb_zodiac(n) for n in regulars],
                    "special": _bb_zodiac(sp)},
        "mapping_version": _bb_zodiac_version(d["issue"]),
        "source": d["source"], "updated": d.get("fetched_at"),
        "check": d["check_status"],
    })


def bb_history(query: str = "", page: int = 1, per_page: int = 20) -> dict:
    try:
        rows = _bb_draws()
        rows.reverse()                              # 最新在前
        if query:
            rows = [r for r in rows if query.strip() in r["issue"]]
        total = len(rows)
        pages = max(1, (total + per_page - 1) // per_page)
        page = min(max(1, page), pages)
        items = []
        for r in rows[(page - 1) * per_page: page * per_page]:
            nums = [r[f"p{i}"] for i in range(1, 7)]
            items.append({
                "issue": r["issue"],
                "date": r["issue"][:4] + "-" + r["issue"][4:6] + "-"
                + r["issue"][6:8],
                "regulars": nums, "special": r["sp"],
                "zodiacs": [_bb_zodiac(n) for n in nums],
                "sp_zodiac": _bb_zodiac(r["sp"]),
                "mapping_version": _bb_zodiac_version(r["issue"]),
                "source": r["source"], "check": r["check_status"],
            })
        return ok({"items": items, "total": total, "page": page,
                   "pages": pages})
    except Exception as e:
        return _zh_exc(e, "查询BB历史开奖")


def bb_export_history(limit: int = 5000) -> dict:
    try:
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        rows = _bb_draws()[:limit]
        path = EXPORT_DIR / "bb_history.csv"
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            f.write("issue,p1,p2,p3,p4,p5,p6,sp,source,check,mapping_version\n")
            for r in rows:
                f.write(",".join([
                    r["issue"], *[str(r[f"p{i}"]) for i in range(1, 7)],
                    str(r["sp"]), r["source"], r["check_status"],
                    _bb_zodiac_version(r["issue"])]) + "\n")
        return ok({"path": str(path), "n": len(rows)})
    except Exception as e:
        return _zh_exc(e, "导出CSV")


def bb_overview() -> dict:
    try:
        h = bb_health()
        if not h["ok"]:
            return h
        # 研究结果摘要（读最近一次回测结论，不重算）
        from bb_analysis import database as db
        cfg = _bb_cfg()
        with db.connect(cfg.db_path) as conn:
            run = conn.execute(
                "SELECT id,finished_at,summary_json FROM backtest_runs "
                "ORDER BY id DESC LIMIT 1").fetchone()
        concl = ""
        tested = ""
        if run and run["summary_json"]:
            s = json.loads(run["summary_json"])
            concl = s.get("conclusion", "")
            tested = f"{s.get('meta', {}).get('n_draws', '')}期数据 / " \
                     f"分段{ s.get('meta', {}).get('segments', {}) }"
        return ok({
            "health": h["data"], "latest": bb_latest()["data"],
            "backtest": {"finished": run["finished_at"] if run else None,
                         "tested": tested, "conclusion": concl or "暂无回测"},
        })
    except Exception as e:
        return _zh_exc(e, "生成BB总览")


def bb_stats(scope: str = "zodiac_full", window: int = 0,
             zodiac: str = "") -> dict:
    """scope: number|regular|special|zodiac_full|zodiac_special；window=0 全部。"""
    try:
        rows = _bb_draws()
        if window:
            rows = rows[-window:]
        n = len(rows)
        if n == 0:
            return err("没有可用历史数据", "请先导入数据。")
        note = f"统计范围：最近{n}期（{rows[0]['issue']}~{rows[-1]['issue']}）"

        if scope in ("number", "regular", "special"):
            counts = [0] * 50
            last_seen = [-1] * 50
            for i, r in enumerate(rows):
                nums = ([r[f"p{k}"] for k in range(1, 7)]
                        if scope == "regular" else
                        [r["sp"]] if scope == "special" else
                        [r[f"p{k}"] for k in range(1, 7)] + [r["sp"]])
                for v in nums:
                    counts[v] += 1
                    last_seen[v] = i
            items = [{"n": i, "count": counts[i],
                      "omission": (n - 1 - last_seen[i])
                      if last_seen[i] >= 0 else n,
                      "freq": round(counts[i] / n, 4)}
                     for i in range(1, 50)]
            return ok({"kind": "number", "items": items, "note": note,
                       "sub_note": {"number": "全部7号",
                                    "regular": "仅6正码",
                                    "special": "仅特别号"}[scope]})
        # 生肖
        full = [0] * 12
        spc = [0] * 12
        order = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴",
                 "鸡", "狗", "猪"]
        for r in rows:
            seen = set()
            for k in range(1, 7):
                seen.add(_bb_zodiac(r[f"p{k}"]))
            for z in seen:
                full[order.index(z)] += 1
            spc[order.index(_bb_zodiac(r["sp"]))] += 1
        items = [{"zodiac": z,
                  "full": full[i], "special": spc[i],
                  "full_freq": round(full[i] / n, 4)}
                 for i, z in enumerate(order)]
        return ok({"kind": "zodiac", "items": items, "note": note,
                   "sub_note": "“全开”=出现于7号（一肖口径）；“特码”单独统计"})
    except Exception as e:
        return _zh_exc(e, "统计分析")


def bb_combo(k: int, members: list[str] | None = None) -> dict:
    try:
        from bb_analysis.combinations import (enumerate_combos,
                                              zodiac_sizes)
        from bb_analysis.theory import all_present_prob
        cfg = _bb_cfg()
        rows = _bb_draws()
        order = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴",
                 "鸡", "狗", "猪"]
        if k == 1:
            members = members or ["鼠"]
        if members is None:
            members = order[:k]
        if len(members) != k:
            return err(f"{k}生肖玩法需要选择{k}个生肖",
                       f"当前选择 {len(members)} 个，请调整。")
        sizes = zodiac_sizes(cfg)
        p = all_present_prob([sizes[z] for z in members],
                             cfg.rules.draw_size, sum(sizes.values()))
        # 历史命中（全开口径：所有生肖都出现于该期7号）
        n = len(rows)
        present_sets = []
        for r in rows:
            s = set(_bb_zodiac(r[f"p{i}"]) for i in range(1, 7))
            s.add(_bb_zodiac(r["sp"]))
            present_sets.append(s)
        hits = sum(1 for s in present_sets if set(members) <= s)
        if k == 1:
            odds, status = cfg.odds.single_odds(members[0])
            rule_confirmed = True
            rule_note = "单肖=该生肖出现于7个开奖号即中（平台截图已证实）"
        else:
            odds = cfg.odds.combo_odds_for(k)
            status = "待确认（连肖规则与赔率均为项目资料，未取得平台二次证实）"
            rule_confirmed = False
            rule_note = "暂按“所选生肖全部出现于7号”判定（可配置规则），" \
                        "平台真实口径待确认 → 无法按正式规则结算"
        return ok({
            "k": k, "members": members, "p_theory": p, "odds": odds,
            "odds_status": status, "rule_confirmed": rule_confirmed,
            "rule_note": rule_note,
            "hist": {"n": n, "hits": hits,
                     "rate": hits / n if n else None,
                     "vs_theory": "历史频率 vs 理论概率（历史≠未来）"},
            "breakeven": 1.0 / odds,
        })
    except Exception as e:
        return _zh_exc(e, "生肖组合分析")


def bb_odds() -> dict:
    try:
        cfg = _bb_cfg()
        o = cfg.odds
        rows = [
            {"play": "当年生肖单选", "odds": o.current_year_odds,
             "zodiac": o.current_year_zodiac or "（未配置）",
             "evidence": o.evidence.get("current_year", ""),
             "status": "已按截图配置·待复核"
             if o.current_year_zodiac else "待核实",
             "effective": f"{o.current_year_from} ~ {o.current_year_to}"},
            {"play": "非当年生肖单选", "odds": o.default_single,
             "zodiac": "全部非当年生肖",
             "evidence": o.evidence.get("single_default", ""),
             "status": "已核实（平台截图）", "effective": "长期"},
        ]
        for k in sorted(o.combo_odds):
            rows.append({"play": f"{k}生肖组合", "odds": o.combo_odds[k],
                         "zodiac": "任意组合",
                         "evidence": o.evidence.get("combo", ""),
                         "status": "待核实（user_spec 无截图证据）",
                         "effective": "项目资料版本 " + o.version})
        return ok({"version": o.version, "rows": rows,
                   "unit_note": "赔率=总返还倍数（含本金）；净收益=注额×(O−1)"
                                "；盈亏平衡命中率=1/O",
                   "warnings": list(cfg.warnings)})
    except Exception as e:
        return _zh_exc(e, "读取赔率配置")


def bb_research(history: int = 100) -> dict:
    """历史统计参考（非预测）：热/冷/遗漏 + 注册表证据等级。"""
    try:
        stats = bb_stats("zodiac_full", window=history)
        if not stats["ok"]:
            return stats
        items = stats["data"]["items"]
        hot = max(items, key=lambda x: x["full"])
        cold = min(items, key=lambda x: x["full"])
        reg = bb_registry()
        return ok({
            "history": history,
            "hot": hot, "cold": cold,
            "top_full": sorted(items, key=lambda x: -x["full"])[:3],
            "registry": reg["data"] if reg["ok"] else [],
            "evidence": "unproven",
            "disclaimer": "历史频率和模型评分不代表下一期中奖概率。"
                          "当前结果仅用于研究，不保证中奖。",
            "name": "历史统计参考（本后端无已验证预测模型）",
        })
    except Exception as e:
        return _zh_exc(e, "生成研究参考")


def bb_save_research(snapshot: dict) -> dict:
    """保存“历史统计参考”快照（明确标注非预测）。"""
    try:
        from bb_analysis import database as db
        cfg = _bb_cfg()
        now = datetime.now().isoformat(timespec="seconds")
        rows = _bb_draws()
        with db.connect(cfg.db_path) as conn:
            db.insert_predictions(conn, [{
                "run_id": None,
                "target_issue": rows[-1]["issue"],
                "generated_at": now,
                "model_version": "bb-stats-reference-v1",
                "market": "bb_research",
                "k": int(snapshot.get("history", 0)),
                "combo": json.dumps(snapshot, ensure_ascii=False)[:1800],
                "p_est": 0.0, "p_base": 0.0, "p_lo": None, "p_hi": None,
                "sample_n": int(snapshot.get("history", 0)),
                "odds": 0.0,
                "data_cutoff": rows[-1]["issue"],
                "selected": 0,
                "votes_json": json.dumps(
                    {"kind": "history_stats_reference",
                     "evidence": "unproven"}, ensure_ascii=False),
            }])
        return ok({"saved": True, "generated_at": now})
    except Exception as e:
        return _zh_exc(e, "保存研究快照")


def bb_registry() -> dict:
    path = PROJECT_ROOT / "reports" / "bb_pattern_registry.json"
    if not path.exists():
        return err("规律注册表尚未生成",
                   "到“数据管理”页运行“执行规律分析”。")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for r in payload.get("results", []):
            r["evidence_zh"] = EVIDENCE_ZH.get(r.get("status", ""),
                                               r.get("status", ""))
        return ok(payload)
    except Exception as e:
        return _zh_exc(e, "读取规律注册表")


def bb_check(bet_type: str, members: list[str], issue: str,
             stake: float = 5.0) -> dict:
    """bet_type: 'single' | 'combo'。单肖可结算；连肖规则待确认只显示命中。"""
    try:
        from bb_analysis.rules import combo_hit, settle
        cfg = _bb_cfg()
        rows = _bb_draws()
        d = next((r for r in rows if r["issue"] == issue.strip()), None)
        if d is None:
            return err(f"未找到期号 {issue}",
                       "请输入存在于历史库中的期号（7位）。")
        present = set(_bb_zodiac(r) for r in
                      [d[f"p{i}"] for i in range(1, 7)] + [d["sp"]])
        hit = combo_hit(members, present, cfg.rules.combo_win_rule)
        result = {
            "issue": d["issue"],
            "draw": {"regulars": [d[f"p{i}"] for i in range(1, 7)],
                     "special": d["sp"],
                     "zodiacs": sorted(present)},
            "bet": {"type": bet_type, "members": members,
                    "stake": stake},
            "hit": hit,
            "hit_members": [z for z in members if z in present],
        }
        if bet_type == "single" and len(members) == 1:
            odds, st = cfg.odds.single_odds(
                members[0], datetime.strptime(d["issue"][:8],
                                              "%Y%m%d").date())
            ret, net = settle(stake, odds, hit)
            result.update({"odds": odds, "odds_status": st,
                           "return_total": ret, "net": net,
                           "settled": True,
                           "rule_note": "单肖：出现于7号即中；结算已确认口径"})
        else:
            result.update({"settled": False,
                           "rule_note": "连肖玩法：中奖口径与赔率均为项目资料"
                                        "（待确认）→ 仅展示命中情况，"
                                        "无法按正式规则结算"})
        return ok(result)
    except Exception as e:
        return _zh_exc(e, "BB对奖")


def bb_backtest_latest() -> dict:
    try:
        from bb_analysis import database as db
        cfg = _bb_cfg()
        with db.connect(cfg.db_path) as conn:
            rows = [dict(r) for r in conn.execute(
                "SELECT id,model_version,data_start,data_end,started_at,"
                "finished_at,summary_json,params_json FROM backtest_runs "
                "ORDER BY id DESC LIMIT 5")]
        if not rows:
            return err("暂无回测记录",
                       "到“数据管理”页运行“执行回测”。")
        for r in rows:
            r["summary"] = json.loads(r.pop("summary_json") or "null")
            r["params"] = json.loads(r.pop("params_json") or "{}")
        return ok({"latest": rows[0], "runs": rows})
    except Exception as e:
        return _zh_exc(e, "读取BB回测")


def bb_bankroll(initial: float = 200.0, base: float = 5.0) -> dict:
    from bb_analysis.bankroll import run_bankroll
    return ok(run_bankroll(initials=(100.0, initial, 500.0), base=base))


def bb_task(action: str) -> dict:
    """action: import | patterns | backtest | bankroll | sync | report"""
    labels = {
        "import": "BB·导入历史数据", "patterns": "BB·执行规律分析",
        "backtest": "BB·执行回测", "bankroll": "BB·资金模拟",
        "sync": "BB·同步最新开奖（使用本地Cookie）",
        "report": "BB·生成分析报告",
    }
    if action not in labels:
        return err(f"未知操作：{action}")

    def _fn():
        cfg = _bb_cfg()
        if action == "import":
            from bb_analysis.ingestion import import_csv_file
            rep, counts = import_csv_file(
                cfg, PROJECT_ROOT / "data" / "bb_live_history.csv",
                source="bb_sync_csv")
            return {"valid": rep.ok, "stats": rep.stats, "counts": counts,
                    "errors": rep.errors[:10]}
        if action == "patterns":
            from bb_analysis.patterns import run_registry
            _, text = run_registry(cfg)
            return text[-3000:]
        if action == "backtest":
            from bb_analysis.backtest import run_backtest
            return run_backtest(cfg)[-3000:]
        if action == "bankroll":
            from bb_analysis.bankroll import run_bankroll
            return run_bankroll()[-3000:]
        if action == "report":
            from bb_analysis.reports import write_report
            return write_report(cfg)[-1500:]
        if action == "sync":
            # 固定调用本项目采集脚本（非任意shell）；脚本不打印任何凭据
            import subprocess
            import sys as _sys
            p = subprocess.run(
                [_sys.executable, str(PROJECT_ROOT / "bb_sync.py")],
                capture_output=True, text=True, timeout=180,
                cwd=str(PROJECT_ROOT), encoding="utf-8", errors="replace")
            out = (p.stdout or "") + (p.stderr or "")
            if p.returncode != 0:
                raise RuntimeError(
                    f"采集失败(code={p.returncode})：{out.strip()[:300]}")
            return out[-3000:]
        return "ok"

    return TASKS.submit(f"bb_{action}", labels[action], _fn)


# ---- 设置 ----

def settings_info() -> dict:
    try:
        ssq = _ssq_cfg()
        bb = _bb_cfg()
        return ok({
            "ssq": {"db": str(ssq.db_path), "windows": list(ssq.windows),
                    "seed": ssq.seed,
                    "ticket_price": ssq.ticket_price,
                    "prizes": ssq.prize_fixed},
            "bb": {"db": str(bb.db_path), "odds_version": bb.odds.version,
                   "mapping_versions": [v.version for v in
                                        bb.mapping.versions],
                   "seed": bb.seed, "windows": list(bb.windows)},
            "ai": {"env_key_set": bool(__import__("os").getenv(
                "MIMO_AI_KEY")),
                   "note": "AI 仅用于可选解释，缺失时核心功能不受影响"},
        })
    except Exception as e:
        return _zh_exc(e, "读取设置")
