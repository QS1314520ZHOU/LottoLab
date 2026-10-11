# -*- coding: utf-8 -*-
"""备用来源：500 彩票网历史开奖 HTML 表格（非官方，仅补齐/交叉核对）。

页面结构：#tdata 内 <tr> 行，单元格依次为期号、红1..6、蓝1。
期号 5 位（如 03001）规范化为 7 位（2003001）。
"""

from __future__ import annotations

import re
from typing import Any

from .base import SourceAdapter, SourceError

URL = "https://datachart.500.com/ssq/history/newinc/history.php"
REFERER = "https://datachart.500.com/ssq/history/history.shtml"

_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
_CELL_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.S | re.I)
_TAG_RE = re.compile(r"<[^>]+>")


def canonical_issue(raw: str) -> str:
    s = str(raw).strip()
    if len(s) == 5 and s.isdigit():
        return "20" + s
    if len(s) == 7 and s.isdigit():
        return s
    raise ValueError(f"无效期号 {raw!r}")


class Backup500Source(SourceAdapter):
    name = "chart500_backup"
    url = URL
    is_official = False
    terms_notes = "非官方备用源；仅只读公开图表页，不用于覆盖官方数据"
    coverage_note = "支持 2003 年起全量（5位期号）"

    def __init__(self, **kw: Any):
        kw.setdefault("min_interval", 1.5)
        super().__init__(**kw)

    def fetch(self, start: str, end: str) -> Any:
        """start/end 为 5 位期号（如 03001）。"""
        resp = self._get(URL, params={"start": start, "end": end},
                         headers={"Referer": REFERER})
        text = None
        for enc in ("utf-8", "gb18030"):
            try:
                text = resp.content.decode(enc)
                if "<html" in text.lower() or "<tr" in text.lower():
                    break
            except UnicodeDecodeError:
                continue
        if text is None:
            text = resp.text
        return text

    def parse(self, raw: Any) -> list[dict[str, Any]]:
        html = str(raw)
        if "tdata" not in html:
            raise SourceError("500 彩票网页面缺少 tdata 表格（结构变化或拦截页）")
        out: list[dict[str, Any]] = []
        issue_cell_re = re.compile(r"^\d{5}$|^\d{7}$")
        for row in _ROW_RE.findall(html):
            cells = [_TAG_RE.sub("", c).strip() for c in _CELL_RE.findall(row)]
            if len(cells) < 9:
                continue
            # 期号可能在第1或第2列（表头/标记列浮动），定位后取连续7格
            idx = next((i for i in range(min(3, len(cells) - 7))
                        if issue_cell_re.match(cells[i])), None)
            if idx is None:
                continue
            try:
                issue = canonical_issue(cells[idx])
                red = sorted(int(cells[idx + i]) for i in range(1, 7))
                blue = int(cells[idx + 7])
            except (ValueError, IndexError):
                continue
            out.append({"issue": issue, "draw_date": None,
                        "red": red, "blue": blue, "source": self.name})
        if not out:
            raise SourceError("500 彩票网响应解析出 0 行")
        return out

    def probe(self) -> dict[str, Any]:
        rows = self.parse(self.fetch("26100", "26116"))
        if not rows:
            raise SourceError("探针未取到数据")
        return {"n": len(rows), "sample": rows[-1]}
