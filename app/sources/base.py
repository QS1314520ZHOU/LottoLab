# -*- coding: utf-8 -*-
"""数据源适配器基类：fetch / parse / validate + 限速重试。"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Optional

import requests

LAST_CALL_AT: dict[str, float] = {}


class SourceError(Exception):
    """数据源访问或解析失败。"""


class SourceAdapter(ABC):
    """每个来源独立实现 fetch/parse/validate。"""

    name: str = "base"
    url: str = ""
    is_official: bool = False
    needs_api_key: bool = False
    paid: bool = False
    terms_notes: str = ""
    coverage_note: str = ""

    def __init__(self, timeout: float = 20.0, min_interval: float = 1.0,
                 max_retries: int = 2):
        self.timeout = timeout
        self.min_interval = min_interval
        self.max_retries = max_retries

    # ---- 网络层 ----
    def _throttle(self) -> None:
        last = LAST_CALL_AT.get(self.name, 0.0)
        wait = self.min_interval - (time.monotonic() - last)
        if wait > 0:
            time.sleep(wait)
        LAST_CALL_AT[self.name] = time.monotonic()

    def _get(self, url: str, params: Optional[dict] = None,
             headers: Optional[dict] = None) -> requests.Response:
        headers = {"User-Agent":
                   "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 Chrome/125.0 Safari/537.36",
                   **(headers or {})}
        backoff = 1.0
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                resp = requests.get(url, params=params, headers=headers,
                                    timeout=self.timeout)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise SourceError(f"HTTP {resp.status_code}")
                resp.raise_for_status()
                if not resp.content:
                    raise SourceError("空响应")
                return resp
            except (requests.RequestException, SourceError) as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(backoff)
                    backoff *= 2
        raise SourceError(f"{self.name} 请求失败: {last_exc}")

    # ---- 业务层 ----
    @abstractmethod
    def fetch(self, **kwargs: Any) -> Any:
        """拉取原始响应（已含重试/限速）。"""

    @abstractmethod
    def parse(self, raw: Any) -> list[dict[str, Any]]:
        """解析为标准行：{issue, draw_date, red:[6], blue:int, source}。"""

    def validate(self, rows: list[dict[str, Any]]) -> list[str]:
        """默认校验：6红1蓝、红1-33互异、蓝1-16、期号7位。"""
        errors: list[str] = []
        for r in rows:
            issue = str(r.get("issue", ""))
            if len(issue) != 7 or not issue.isdigit():
                errors.append(f"{issue}: 期号非法")
                continue
            red = r.get("red") or []
            blue = r.get("blue")
            if (len(red) != 6 or len(set(red)) != 6
                    or not all(1 <= x <= 33 for x in red)):
                errors.append(f"{issue}: 红球非法 {red}")
            if not isinstance(blue, int) or not (1 <= blue <= 16):
                errors.append(f"{issue}: 蓝球非法 {blue}")
        return errors

    def meta(self) -> dict[str, Any]:
        return {"name": self.name, "url": self.url,
                "is_official": int(self.is_official),
                "needs_api_key": int(self.needs_api_key),
                "paid": int(self.paid), "terms_notes": self.terms_notes,
                "coverage_note": self.coverage_note, "status": "verified"}
