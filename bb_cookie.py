# -*- coding: utf-8 -*-
"""
bb_cookie.py

从项目下的 cookie.txt 自动读取 Cookie，支持两种粘贴格式：

1. Netscape HTTP Cookie File（浏览器插件导出的制表符分隔文件，
   就是带 "# https://curl.haxx.se/rfc/cookie_spec.html" 头的那种）
2. 浏览器开发者工具里直接复制的 Cookie 请求头字符串，例如：
   BBSESSID=xxx; MORTLACH=yyy; charset=zh-cn

优先级：cookie.txt 有内容时优先，否则回退到环境变量 BB_COOKIE。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

COOKIE_FILE = Path(__file__).resolve().parent / "cookie.txt"


def _read_text(path: Path) -> str:
    """读取文本，兼容 UTF-8（含BOM）和 Windows 中文系统的 ANSI/GBK。"""
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "gbk"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _parse_netscape(text: str) -> str:
    """解析 Netscape cookie 文件：域、标志、路径、安全、过期、名称、值。"""
    pairs = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 7:
            continue
        name = fields[5].strip()
        value = "\t".join(fields[6:]).strip()
        if name:
            pairs.append(f"{name}={value}")
    return "; ".join(pairs)


def _parse_header(text: str) -> str:
    """解析直接粘贴的 Cookie 请求头（可多行）。"""
    parts = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.strip(";").strip()
        if line:
            parts.append(line)
    return "; ".join(parts)


def load_cookie(path: Optional[Path] = None) -> str:
    """读取 cookie.txt 并返回 Cookie 头字符串；文件缺失或为空返回空串。"""
    cookie_path = Path(path) if path else COOKIE_FILE
    if not cookie_path.is_file():
        return ""

    try:
        text = _read_text(cookie_path)
    except OSError:
        return ""

    if not text.strip():
        return ""

    # Netscape 格式以制表符分隔；命中即按该格式解析
    is_netscape = any(
        "\t" in line and not line.startswith("#")
        for line in text.splitlines()
    )
    cookie = _parse_netscape(text) if is_netscape else _parse_header(text)
    return cookie.strip()


def get_cookie(path: Optional[Path] = None) -> str:
    """cookie.txt 优先，其次环境变量 BB_COOKIE。"""
    return load_cookie(path) or os.getenv("BB_COOKIE", "").strip()


if __name__ == "__main__":
    result = get_cookie()
    if result:
        # 只打印 cookie 名，不打印值，避免泄露会话
        names = [p.split("=", 1)[0].strip() for p in result.split(";")]
        print(f"已解析 {len(names)} 个 Cookie：{', '.join(names)}")
    else:
        print(f"未读取到 Cookie（{COOKIE_FILE} 为空或不存在，环境变量 BB_COOKIE 也未设置）")
