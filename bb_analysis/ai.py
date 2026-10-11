# -*- coding: utf-8 -*-
"""T8 · AI 辅助解释（只读）：解读评分卡、提可检验假设。

权限边界（硬约束）：
  - AI 只输出解释与假设文本；
  - 不提供评分、不写库、不改变任何规律状态/评级；
  - 调用失败/空返回记录 coverage，核心流程不依赖它。
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

COVERAGE = {"calls": 0, "ok": 0, "failed": 0, "empty": 0}


def explain(text: str, question: str) -> dict[str, Any]:
    """返回 {ok, text|reason, coverage}；绝不抛异常。"""
    COVERAGE["calls"] += 1
    if not AI_KEY:
        COVERAGE["failed"] += 1
        return {"ok": False, "reason": "MIMO_AI_KEY 未配置（可选功能）",
                "coverage": dict(COVERAGE)}
    prompt = (
        "你是统计研究助手。以下 BB六合彩规律评分卡/回测文本只读。"
        "回答要求：只引用文本数据；不得编造；不得把相关当因果；"
        "不得声称某生肖更容易中；不得替任何规律改状态或打分。\n\n"
        f"{text[:4000]}\n\n问题：{question}"
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
        out = (data["choices"][0]["message"].get("content") or "").strip()
        if not out:
            COVERAGE["empty"] += 1
            return {"ok": False, "reason": "返回空文本",
                    "coverage": dict(COVERAGE)}
        COVERAGE["ok"] += 1
        return {"ok": True, "text": out, "coverage": dict(COVERAGE)}
    except Exception as exc:
        COVERAGE["failed"] += 1
        return {"ok": False, "reason": str(exc),
                "coverage": dict(COVERAGE)}
