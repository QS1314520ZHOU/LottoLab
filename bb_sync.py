# -*- coding: utf-8 -*-
"""
bb_sync.py

用 cookie.txt 里的新 MORTLACH 同步 BB 开奖数据到 data/bb_live_history.csv。
只打印数量与期号范围，绝不打印任何凭据内容。

用法：python bb_sync.py
"""

import sys
from pathlib import Path

import bb_zodiac_v7 as bb
from bb_cookie import load_cookie

API_URL = "https://zhigezbjs.top/pt/provider/mem/game/results/BBLT"
CSV_PATH = Path(__file__).resolve().parent / "data" / "bb_live_history.csv"


def extract_mortlach(cookie_header: str) -> str:
    for pair in cookie_header.split(";"):
        name, _, value = pair.partition("=")
        if name.strip() == "MORTLACH":
            return value.strip()
    return ""


def main() -> int:
    cookie = load_cookie()
    if not cookie:
        print("cookie.txt 为空或不存在。")
        return 1

    mortlach = extract_mortlach(cookie)
    if not mortlach:
        print("cookie 中未找到 MORTLACH。")
        return 1
    print(f"MORTLACH 已读取（长度 {len(mortlach)}）。")

    import os
    os.environ["BB_API_URL"] = API_URL
    os.environ["BB_ADD_TIMESTAMP"] = "1"
    # per_limit 会作为总数上限；10条/页固定
    os.environ["BB_API_PARAMS"] = (
        '{"MORTLACH":"' + mortlach
        + '","lang":"zh-cn","gametype":"BBLT","search_type":0,'
        '"per_limit":100000,"page":1}'
    )

    client = bb.BBWebClient()
    payload, draws = client.fetch_draws()
    print(f"首页解析 {len(draws)} 期。")

    if not draws:
        # 报错误码但不打印含token的请求
        code = payload.get("code") if isinstance(payload, dict) else None
        msg = payload.get("msg") if isinstance(payload, dict) else None
        print(f"接口返回 code={code} msg={msg} data为空。")
        return 2

    all_draws = {d.issue: d for d in draws}
    if len(draws) >= 10:
        # 可能有分页，继续翻
        pages = client.fetch_history_pages(max_pages=300, pause_seconds=0.15)
        for d in pages:
            all_draws[d.issue] = d

    fetched = sorted(all_draws.values(), key=bb.issue_sort_key)
    print(f"共取得 {len(fetched)} 期，范围 {fetched[0].issue}..{fetched[-1].issue}")

    # 合并已有 CSV
    old = {}
    if CSV_PATH.exists():
        for d in bb.load_csv(str(CSV_PATH)):
            old[d.issue] = d
    print(f"已有 CSV {len(old)} 期。")

    merged = dict(old)
    for d in fetched:
        merged[d.issue] = d
    final = sorted(merged.values(), key=bb.issue_sort_key)
    print(f"合并后 {len(final)} 期，新增 {len(final) - len(old)} 期。")

    # 全量重写
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["issue", "p1", "p2", "p3", "p4", "p5", "p6", "sp"])
        for d in final:
            writer.writerow([d.issue] + d.numbers)
    print(f"已写入 {CSV_PATH.name}。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
