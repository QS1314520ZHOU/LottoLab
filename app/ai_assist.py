# -*- coding: utf-8 -*-
"""可选 AI 研究助手：阅读报告、解释统计关系、提假设（失败必记录）。

核心采集/计算/回测不依赖本模块。API 未配置或调用失败 → 返回
{'ok': False, 'reason': ...}，并在 coverage 中累计失败数，绝不静默。
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

AI_BASE = os.getenv("MIMO_AI_BASE",
                    "https://token-plan-cn.xiaomimimo.com/v1")
AI_MODEL = os.getenv("MIMO_AI_MODEL", "mimo-v2.6-flash")
AI_KEY = os.getenv("MIMO_AI_KEY", "")

COVERAGE = {"calls": 0, "ok": 0, "failed": 0, "unparseable": 0}


def explain_report(report_text: str, question: str) -> dict[str, Any]:
    """让 AI 阅读报告片段并回答；输出仅作解释，不作统计证据。"""
    COVERAGE["calls"] += 1
    if not AI_KEY:
        COVERAGE["failed"] += 1
        return {"ok": False, "reason": "MIMO_AI_KEY 未配置（可选功能）"}
    prompt = (
        "你是统计研究助手。下面是双色球历史分析报告节选。回答问题，"
        "只依据文本中的数据；不得编造数据、不得把相关当因果、"
        "不得声称某号码更容易中奖。若证据不足请明说。\n\n"
        f"报告：\n{report_text[:4000]}\n\n问题：{question}"
    )
    body = json.dumps({"model": AI_MODEL,
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 4000}).encode()
    req = urllib.request.Request(
        AI_BASE + "/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + AI_KEY})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        text = data["choices"][0]["message"].get("content") or ""
        if not text.strip():
            COVERAGE["unparseable"] += 1
            return {"ok": False, "reason": "返回空文本", "coverage": dict(COVERAGE)}
        COVERAGE["ok"] += 1
        return {"ok": True, "text": text, "coverage": dict(COVERAGE)}
    except Exception as exc:  # 网络/解析失败全部记录
        COVERAGE["failed"] += 1
        return {"ok": False, "reason": str(exc), "coverage": dict(COVERAGE)}
