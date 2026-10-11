# -*- coding: utf-8 -*-
"""官方来源：中国福利彩票官网 findDrawNotice 接口（已实测，见 DATA_SOURCES.md）。"""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Any

from .base import SourceAdapter, SourceError

API = ("https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/"
       "kjxx/findDrawNotice")

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


class CwlSource(SourceAdapter):
    name = "cwl_official"
    url = API
    is_official = True
    terms_notes = ("robots.txt Disallow: /*?*（带查询串）；低频访问、"
                   "不绕过任何控制，见 docs/DATA_SOURCES.md")
    coverage_note = "issueStart/issueEnd 支持，实测仅覆盖 2013 年起"

    def __init__(self, **kw: Any):
        kw.setdefault("min_interval", 1.0)
        super().__init__(**kw)
        self.headers = {"Referer": "https://www.cwl.gov.cn/ygkj/wqkjgg/ssq/",
                        "Accept": "application/json"}

    def fetch(self, issue_start: str, issue_end: str,
              page_no: int = 1, page_size: int = 500) -> Any:
        params = {"name": "ssq", "issueStart": issue_start,
                  "issueEnd": issue_end, "pageNo": page_no,
                  "pageSize": page_size, "systemType": "PC"}
        resp = self._get(API, params=params, headers=self.headers)
        ctype = resp.headers.get("Content-Type", "")
        if "json" not in ctype and not resp.text.lstrip().startswith("{"):
            raise SourceError(f"非 JSON 响应: {ctype}")
        return resp.json()

    def parse(self, raw: Any) -> list[dict[str, Any]]:
        if not isinstance(raw, dict) or raw.get("state") not in (0, None):
            raise SourceError(f"接口返回异常: {raw.get('message') if isinstance(raw, dict) else raw}")
        out: list[dict[str, Any]] = []
        for item in raw.get("result") or []:
            try:
                m = _DATE_RE.search(str(item.get("date", "")))
                out.append({
                    "issue": str(item["code"]),
                    "draw_date": m.group(1) if m else None,
                    "red": sorted(int(x) for x in str(item["red"]).split(",")),
                    "blue": int(item["blue"]),
                    "source": self.name,
                })
            except (KeyError, TypeError, ValueError):
                continue
        return out

    def fetch_range(self, issue_start: str, issue_end: str,
                    page_size: int = 500) -> list[dict[str, Any]]:
        """按期号范围拉取（分页直到空页；页间限速由基类负责）。"""
        rows: list[dict[str, Any]] = []
        page = 1
        while True:
            raw = self.fetch(issue_start, issue_end, page_no=page,
                             page_size=page_size)
            batch = self.parse(raw)
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < page_size:
                break
            page += 1
        return rows

    def fetch_year_chunks(self, start_year: int = 2013,
                          end_year: int | None = None) -> list[dict[str, Any]]:
        """按年份分块拉全量（每块约153期，避免单请求过大）。"""
        end_year = end_year or date.today().year
        rows: list[dict[str, Any]] = []
        for y in range(start_year, end_year + 1):
            rows.extend(self.fetch_range(f"{y}001", f"{y}999"))
        return rows

    def probe(self) -> dict[str, Any]:
        """discover-sources 用：取最新1期验证可用性。"""
        raw = self.fetch(str(date.today().year) + "001",
                         str(date.today().year) + "999", page_size=1)
        rows = self.parse(raw)
        if not rows:
            raise SourceError("探针未取到数据")
        return {"n": len(rows), "sample": rows[0]}
