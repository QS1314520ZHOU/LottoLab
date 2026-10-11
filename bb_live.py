# bb_live.py
# -*- coding: utf-8 -*-

"""
BB六合彩运行入口：

1. 随机模拟
2. CSV历史回测
3. Cookie/API实时抓取
4. 模型自动保存和加载

依赖：
    pip install numpy scikit-learn joblib requests colorama

环境变量示例：

Windows PowerShell：

$env:BB_API_URL="https://example.com/api/history"
$env:BB_API_METHOD="GET"
$env:BB_COOKIE="session=xxx; token=xxx"
$env:BB_API_PARAMS='{"gameCode":"BBLHC","pageSize":100}'
python bb_live.py --mode live

Linux/macOS：

export BB_API_URL="https://example.com/api/history"
export BB_API_METHOD="GET"
export BB_COOKIE="session=xxx; token=xxx"
export BB_API_PARAMS='{"gameCode":"BBLHC","pageSize":100}'
python bb_live.py --mode live
"""

import argparse
import csv
import json
import os
import re
import sys
import time

from typing import Dict, List, Optional

import requests
from colorama import Fore, Style, init

from bb_cookie import get_cookie
from bb_core import (
    BASE_PROBABILITY,
    DEFAULT_ZODIAC,
    MODEL_PATH,
    ZODIACS,
    BBEngine,
    normalize_draw,
    random_draw,
    special_number,
)


init(autoreset=True)


# ============================================================
# 网页/API配置
# ============================================================

BB_API_URL = os.getenv(
    "BB_API_URL",
    ""
)

BB_API_METHOD = os.getenv(
    "BB_API_METHOD",
    "GET"
).upper()

# json / form
BB_API_BODY_MODE = os.getenv(
    "BB_API_BODY_MODE",
    "json"
).lower()

BB_COOKIE = get_cookie()

BB_AUTHORIZATION = os.getenv(
    "BB_AUTHORIZATION",
    ""
)

BB_REFERER = os.getenv(
    "BB_REFERER",
    ""
)

BB_ORIGIN = os.getenv(
    "BB_ORIGIN",
    ""
)

BB_API_PARAMS = os.getenv(
    "BB_API_PARAMS",
    "{}"
)

BB_API_BODY = os.getenv(
    "BB_API_BODY",
    "{}"
)

# 额外请求头，JSON格式
BB_API_HEADERS = os.getenv(
    "BB_API_HEADERS",
    "{}"
)

POLL_INTERVAL = float(
    os.getenv(
        "BB_POLL_INTERVAL",
        "5"
    )
)

REQUEST_TIMEOUT = float(
    os.getenv(
        "BB_REQUEST_TIMEOUT",
        "20"
    )
)

# 每抓取多少期保存一次
SAVE_EVERY = int(
    os.getenv(
        "BB_SAVE_EVERY",
        "20"
    )
)


# ============================================================
# 控制台颜色
# ============================================================

class Color:

    YELLOW = Fore.YELLOW + Style.BRIGHT
    GREEN = Fore.GREEN + Style.BRIGHT
    RED = Fore.RED + Style.BRIGHT
    WHITE = Fore.WHITE + Style.BRIGHT
    CYAN = Fore.CYAN + Style.BRIGHT
    MAGENTA = Fore.MAGENTA + Style.BRIGHT
    RESET = Style.RESET_ALL


# ============================================================
# JSON与Cookie工具
# ============================================================

def parse_json_dict(
    value: str,
    default=None
) -> Dict:
    if default is None:
        default = {}

    if not value:
        return default

    try:
        result = json.loads(value)

        if isinstance(result, dict):
            return result

    except json.JSONDecodeError as error:
        print(
            f"{Color.YELLOW}"
            f"JSON环境变量解析失败：{error}"
        )

    return default


def parse_cookie_string(
    cookie_string: str
) -> Dict[str, str]:
    """
    把：
        session=abc; token=xyz

    转换为：
        {
            "session": "abc",
            "token": "xyz"
        }
    """
    cookies = {}

    if not cookie_string:
        return cookies

    for part in cookie_string.split(";"):
        part = part.strip()

        if (
            not part or
            "=" not in part
        ):
            continue

        key, value = part.split(
            "=",
            1
        )

        key = key.strip()
        value = value.strip()

        if key:
            cookies[key] = value

    return cookies


def parse_response_json(
    response: requests.Response
):
    """
    支持标准JSON和简单JSONP。
    """
    try:
        return response.json()

    except Exception:
        text = response.text.strip()

        # 尝试处理 callback({...})
        jsonp_match = re.match(
            r"^[^(]+\((.*)\)\s*;?$",
            text,
            flags=re.S
        )

        if jsonp_match:
            return json.loads(
                jsonp_match.group(1)
            )

        preview = text[:500]

        raise ValueError(
            "接口返回的不是JSON。"
            "可能Cookie失效、接口地址错误，"
            "或者返回了登录页面。\n"
            f"响应预览：{preview!r}"
        )


# ============================================================
# 期号排序
# ============================================================

def issue_sort_key(issue):
    """
    尽可能把期号转换为数字排序。
    """
    text = str(issue)

    number_parts = re.findall(
        r"\d+",
        text
    )

    if number_parts:
        try:
            return (
                0,
                int("".join(number_parts))
            )
        except ValueError:
            pass

    return (1, text)


# ============================================================
# 网页/API客户端
# ============================================================

class BBWebClient:

    ISSUE_KEYS = (
        "issue",
        "issueNo",
        "issue_no",
        "issueNum",
        "issueNumber",
        "drawNumber",
        "drawNo",
        "draw_no",
        "period",
        "periodNo",
        "expect",
        "round",
        "roundNo",
        "gameNo",
        "lotteryNo",
    )

    NUMBER_KEYS = (
        "openCode",
        "open_code",
        "openNumbers",
        "open_numbers",
        "openNumber",
        "numbers",
        "numberList",
        "resultNumbers",
        "drawResult",
        "lotteryResult",
        "resultCode",
        "winningNumbers",
        "code",
    )

    LIST_KEYS = (
        "data",
        "result",
        "rows",
        "list",
        "records",
        "items",
        "history",
        "draws",
        "openResults",
        "lotteryResults",
    )

    def __init__(
        self,
        url: str = BB_API_URL,
        method: str = BB_API_METHOD,
        cookie_string: str = BB_COOKIE
    ):
        if not url:
            raise ValueError(
                "没有配置BB_API_URL。"
                "请通过环境变量或--url指定接口地址。"
            )

        self.url = url
        self.method = method.upper()

        self.params = parse_json_dict(
            BB_API_PARAMS
        )

        self.body = parse_json_dict(
            BB_API_BODY
        )

        self.session = requests.Session()

        headers = {
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "Accept": (
                "application/json,"
                "text/plain,*/*"
            ),
            "Accept-Language": (
                "zh-CN,zh;q=0.9"
            ),
        }

        if BB_AUTHORIZATION:
            headers["Authorization"] = (
                BB_AUTHORIZATION
            )

        if BB_REFERER:
            headers["Referer"] = (
                BB_REFERER
            )

        if BB_ORIGIN:
            headers["Origin"] = (
                BB_ORIGIN
            )

        extra_headers = parse_json_dict(
            BB_API_HEADERS
        )

        headers.update(extra_headers)

        self.session.headers.update(
            headers
        )

        cookies = parse_cookie_string(
            cookie_string
        )

        if cookies:
            self.session.cookies.update(
                cookies
            )

    def fetch_payload(self):
        if self.method == "GET":
            response = self.session.get(
                self.url,
                params=self.params,
                timeout=REQUEST_TIMEOUT
            )

        elif self.method == "POST":
            if BB_API_BODY_MODE == "form":
                response = self.session.post(
                    self.url,
                    params=self.params,
                    data=self.body,
                    timeout=REQUEST_TIMEOUT
                )
            else:
                response = self.session.post(
                    self.url,
                    params=self.params,
                    json=self.body,
                    timeout=REQUEST_TIMEOUT
                )

        else:
            raise ValueError(
                f"不支持请求方法：{self.method}"
            )

        response.raise_for_status()

        return parse_response_json(
            response
        )

    @staticmethod
    def first_value(
        item: Dict,
        keys
    ):
        for key in keys:
            if key not in item:
                continue

            value = item[key]

            if value is not None:
                return value

        return None

    @staticmethod
    def parse_numbers(value):
        if value is None:
            return None

        if isinstance(value, list):
            result = []

            for item in value:
                if isinstance(item, dict):
                    item = BBWebClient.first_value(
                        item,
                        (
                            "number",
                            "num",
                            "value",
                            "code",
                        )
                    )

                try:
                    result.append(int(item))
                except (TypeError, ValueError):
                    return None

            return result

        if isinstance(value, dict):
            nested = BBWebClient.first_value(
                value,
                BBWebClient.NUMBER_KEYS
            )

            if nested is not None:
                return BBWebClient.parse_numbers(
                    nested
                )

            return None

        if isinstance(value, str):
            # 支持：
            # 01,02,03,04,05,06,07
            # 01-02-03-04-05-06-07
            # 01 02 03 04 05 06 07
            matches = re.findall(
                r"\d+",
                value
            )

            if not matches:
                return None

            return [
                int(number)
                for number in matches
            ]

        return None

    @classmethod
    def normalize_item(
        cls,
        item
    ) -> Optional[Dict]:
        if not isinstance(item, dict):
            return None

        issue = cls.first_value(
            item,
            cls.ISSUE_KEYS
        )

        # 第一种格式：
        # p1,p2,p3,p4,p5,p6,sp
        position_formats = (
            (
                "p1", "p2", "p3",
                "p4", "p5", "p6", "sp"
            ),
            (
                "ball1", "ball2", "ball3",
                "ball4", "ball5", "ball6",
                "special"
            ),
            (
                "num1", "num2", "num3",
                "num4", "num5", "num6",
                "specialNum"
            ),
        )

        numbers = None

        for keys in position_formats:
            if all(key in item for key in keys):
                try:
                    numbers = [
                        int(item[key])
                        for key in keys
                    ]
                except (TypeError, ValueError):
                    numbers = None

                if numbers is not None:
                    break

        # 第二种格式：openCode/numbers等
        if numbers is None:
            raw_numbers = cls.first_value(
                item,
                cls.NUMBER_KEYS
            )

            numbers = cls.parse_numbers(
                raw_numbers
            )

        if (
            issue is None or
            numbers is None or
            len(numbers) != 7
        ):
            return None

        try:
            return normalize_draw(
                issue,
                numbers
            )

        except ValueError:
            return None

    @classmethod
    def find_candidate_items(
        cls,
        value,
        depth=0,
        max_depth=7
    ) -> List[Dict]:
        """
        递归寻找开奖字典。

        不依赖固定的data.list路径。
        """
        if depth > max_depth:
            return []

        results = []

        if isinstance(value, dict):
            normalized = cls.normalize_item(
                value
            )

            if normalized is not None:
                results.append(value)

            # 优先检查常见列表字段
            for key in cls.LIST_KEYS:
                nested = value.get(key)

                if nested is not None:
                    results.extend(
                        cls.find_candidate_items(
                            nested,
                            depth + 1,
                            max_depth
                        )
                    )

            # 再检查其他字段
            for key, nested in value.items():
                if key in cls.LIST_KEYS:
                    continue

                if isinstance(
                    nested,
                    (dict, list)
                ):
                    results.extend(
                        cls.find_candidate_items(
                            nested,
                            depth + 1,
                            max_depth
                        )
                    )

        elif isinstance(value, list):
            for item in value:
                results.extend(
                    cls.find_candidate_items(
                        item,
                        depth + 1,
                        max_depth
                    )
                )

        return results

    def fetch_draws(self) -> List[Dict]:
        payload = self.fetch_payload()

        candidates = self.find_candidate_items(
            payload
        )

        records = {}

        for item in candidates:
            draw = self.normalize_item(item)

            if draw is None:
                continue

            records[draw["issue"]] = draw

        return sorted(
            records.values(),
            key=lambda draw:
                issue_sort_key(draw["issue"])
        )


# ============================================================
# CSV读取与保存
# ============================================================

def load_csv(path: str) -> List[Dict]:
    records = []

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        reader = csv.DictReader(file)

        required = {
            "issue",
            "p1",
            "p2",
            "p3",
            "p4",
            "p5",
            "p6",
            "sp",
        }

        fields = set(
            reader.fieldnames or []
        )

        if not required.issubset(fields):
            raise ValueError(
                "CSV字段必须包含："
                "issue,p1,p2,p3,p4,p5,p6,sp"
            )

        for row in reader:
            numbers = [
                int(row["p1"]),
                int(row["p2"]),
                int(row["p3"]),
                int(row["p4"]),
                int(row["p5"]),
                int(row["p6"]),
                int(row["sp"]),
            ]

            records.append(
                normalize_draw(
                    row["issue"],
                    numbers
                )
            )

    return sorted(
        records,
        key=lambda draw:
            issue_sort_key(draw["issue"])
    )


def append_draw_to_csv(
    path: str,
    draw: Dict
):
    """
    把实时抓取结果保存到CSV。

    如果文件不存在，自动写表头。
    """
    directory = os.path.dirname(path)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )

    exists = os.path.exists(path)
    nums = draw["nums"]

    with open(
        path,
        "a",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        writer = csv.writer(file)

        if not exists:
            writer.writerow([
                "issue",
                "p1",
                "p2",
                "p3",
                "p4",
                "p5",
                "p6",
                "sp",
            ])

        writer.writerow([
            draw["issue"],
            nums[0],
            nums[1],
            nums[2],
            nums[3],
            nums[4],
            nums[5],
            nums[6],
        ])


# ============================================================
# 控制台输出
# ============================================================

def print_prediction(
    engine: BBEngine,
    draw: Dict,
    decision: Dict
):
    hit = decision["hit"]

    result_color = (
        Color.GREEN
        if hit
        else Color.RED
    )

    actual = draw["sp"]
    special = special_number(draw)

    print("\n" + "-" * 105)

    print(
        f"{Color.CYAN}"
        f"期号 {draw['issue']} | "
        f"{Color.YELLOW}"
        f"特码 {special:02d}{actual} | "
        f"{Color.WHITE}"
        f"预测 {decision['prediction']} | "
        f"{result_color}"
        f"{'命中 √' if hit else '未中 ×'}"
    )

    print(
        f"{Color.WHITE}"
        f"预测概率："
        f"{decision['probability']:.3%} | "
        f"第二候选："
        f"{decision['second']} "
        f"{decision['second_probability']:.3%} | "
        f"概率差："
        f"{decision['margin']:.3%}"
    )

    print(
        f"AI状态："
        f"{'已启用' if decision['ai_active'] else '预热中'} | "
        f"AI训练样本：{engine.ai.train_count} | "
        f"公式漂移比例："
        f"{decision['drift_fraction']:.2%}"
    )

    overall_accuracy = (
        engine.hits / engine.total
        if engine.total
        else 0.0
    )

    print(
        f"总预测：{engine.total} | "
        f"总命中：{engine.hits} | "
        f"命中率：{overall_accuracy:.3%} | "
        f"基础概率："
        f"{BASE_PROBABILITY[DEFAULT_ZODIAC]:.3%}"
    )

    print(
        f"\n{Color.MAGENTA}"
        "当前公式TOP 5："
    )

    for index, (
        name,
        state
    ) in enumerate(
        engine.top_formulas(5),
        start=1
    ):
        print(
            f" {index}. {name:<25} "
            f"近20={state.accuracy(20):>7.2%} "
            f"近50={state.accuracy(50):>7.2%} "
            f"趋势={state.trend():>+7.2%} "
            f"评分={state.score():>+8.4f} "
            f"漂移={state.drift()}"
        )

    print("-" * 105)


def print_summary(engine: BBEngine):
    accuracy = (
        engine.hits / engine.total
        if engine.total
        else 0.0
    )

    print("\n" + "=" * 65)
    print("运行汇总")
    print("-" * 65)
    print(f"历史数据：{len(engine.history)}期")
    print(f"公式数量：{len(engine.strategies)}")
    print(f"AI样本数：{engine.ai.train_count}")
    print(f"预测期数：{engine.total}")
    print(f"命中次数：{engine.hits}")
    print(f"命中率  ：{accuracy:.4%}")
    print(f"基础生肖：{DEFAULT_ZODIAC}")
    print(
        f"基础概率："
        f"{BASE_PROBABILITY[DEFAULT_ZODIAC]:.4%}"
    )
    print(f"模型文件：{MODEL_PATH}")
    print("=" * 65)


# ============================================================
# 接口调试
# ============================================================

def debug_api(
    url: str = BB_API_URL
):
    """
    只请求并打印接口结构，不训练模型。

    不打印请求Cookie。
    """
    client = BBWebClient(url=url)

    payload = client.fetch_payload()

    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2
        )[:10000]
    )

    records = client.fetch_draws()

    print(
        f"\n成功解析："
        f"{len(records)}期开奖"
    )

    for draw in records[-10:]:
        print(draw)


# ============================================================
# 随机模拟
# ============================================================

def run_simulation(
    warmup: int,
    interval: float,
    fresh: bool
):
    if fresh:
        engine = BBEngine()
    else:
        engine = BBEngine.load()

    if len(engine.history) < warmup:
        missing = (
            warmup -
            len(engine.history)
        )

        print(
            f"{Color.CYAN}"
            f"正在随机预热{missing}期……"
        )

        base_issue = int(
            time.time()
        ) * 10000

        for index in range(missing):
            draw = random_draw(
                base_issue + index
            )

            engine.process_draw(draw)

    issue = (
        int(time.time()) *
        10000 +
        warmup
    )

    try:
        while True:
            draw = random_draw(issue)
            decision = engine.process_draw(draw)

            if decision is not None:
                print_prediction(
                    engine,
                    draw,
                    decision
                )

            if engine.total % SAVE_EVERY == 0:
                engine.save()

            issue += 1
            time.sleep(interval)

    except KeyboardInterrupt:
        engine.save()
        print_summary(engine)
        print("\n模拟程序已停止。")


# ============================================================
# CSV回测
# ============================================================

def run_backtest(
    csv_path: str,
    warmup: int,
    save_path: str
):
    records = load_csv(
        csv_path
    )

    if len(records) <= warmup:
        raise ValueError(
            f"CSV只有{len(records)}期，"
            f"不足预热期数{warmup}"
        )

    # 回测必须使用新模型，避免旧实盘状态污染
    engine = BBEngine()

    print(
        f"{Color.CYAN}"
        f"历史总数：{len(records)} | "
        f"预热：{warmup} | "
        f"正式回测：{len(records) - warmup}"
    )

    for draw in records[:warmup]:
        engine.process_draw(draw)

    for index, draw in enumerate(
        records[warmup:],
        start=1
    ):
        decision = engine.process_draw(
            draw
        )

        if decision is not None:
            print_prediction(
                engine,
                draw,
                decision
            )

        if index % 1000 == 0:
            engine.save(save_path)

    engine.save(save_path)
    print_summary(engine)


# ============================================================
# 实时运行
# ============================================================

def run_live(
    url: str,
    interval: float,
    csv_output: str,
    fresh: bool
):
    client = BBWebClient(
        url=url
    )

    if fresh:
        engine = BBEngine()
    else:
        engine = BBEngine.load()

    processed_issues = {
        str(draw["issue"])
        for draw in engine.history
    }

    unsaved_count = 0

    print(
        f"{Color.CYAN}"
        f"开始监听：{url}"
    )

    print(
        f"{Color.WHITE}"
        f"请求方式：{BB_API_METHOD} | "
        f"轮询间隔：{interval}秒 | "
        f"已有历史：{len(engine.history)}期"
    )

    print(
        f"{Color.YELLOW}"
        "Cookie不会打印。"
        "如果返回401/403，请重新登录后更新Cookie。"
    )

    try:
        while True:
            try:
                records = client.fetch_draws()

                if not records:
                    print(
                        f"{Color.YELLOW}"
                        f"{time.strftime('%H:%M:%S')} "
                        "请求成功，但未解析到7个开奖号码。"
                    )

                new_records = [
                    draw
                    for draw in records
                    if str(draw["issue"])
                    not in processed_issues
                ]

                # 确保从旧到新处理
                new_records.sort(
                    key=lambda draw:
                        issue_sort_key(
                            draw["issue"]
                        )
                )

                if not new_records:
                    print(
                        f"{Color.WHITE}"
                        f"{time.strftime('%H:%M:%S')} "
                        "暂无新开奖"
                    )

                for draw in new_records:
                    decision = engine.process_draw(
                        draw
                    )

                    processed_issues.add(
                        str(draw["issue"])
                    )

                    append_draw_to_csv(
                        csv_output,
                        draw
                    )

                    if decision is not None:
                        print_prediction(
                            engine,
                            draw,
                            decision
                        )

                    unsaved_count += 1

                if unsaved_count >= SAVE_EVERY:
                    engine.save()
                    unsaved_count = 0

            except requests.HTTPError as error:
                status_code = (
                    error.response.status_code
                    if error.response is not None
                    else "未知"
                )

                print(
                    f"{Color.RED}"
                    f"HTTP错误：{status_code}"
                )

                if status_code in (
                    401,
                    403
                ):
                    print(
                        f"{Color.YELLOW}"
                        "Cookie或Authorization可能失效。"
                    )

            except requests.Timeout:
                print(
                    f"{Color.RED}"
                    "接口请求超时。"
                )

            except requests.ConnectionError as error:
                print(
                    f"{Color.RED}"
                    f"连接失败：{error}"
                )

            except ValueError as error:
                print(
                    f"{Color.RED}"
                    f"数据解析失败：{error}"
                )

            except Exception as error:
                print(
                    f"{Color.RED}"
                    f"实时处理异常：{error}"
                )

            time.sleep(interval)

    except KeyboardInterrupt:
        engine.save()
        print_summary(engine)
        print("\n实时监听已停止。")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "BB六合彩算法、AI模型、"
            "Cookie接口抓取和CSV回测"
        )
    )

    parser.add_argument(
        "--mode",
        choices=(
            "sim",
            "backtest",
            "live",
            "debug-api",
        ),
        default="sim"
    )

    parser.add_argument(
        "--url",
        type=str,
        default=BB_API_URL,
        help="开奖API地址"
    )

    parser.add_argument(
        "--csv",
        type=str,
        default="",
        help="回测CSV文件"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/bb_live_history.csv",
        help="实时开奖保存CSV"
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=1000,
        help="预热期数"
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=POLL_INTERVAL,
        help="轮询或模拟间隔"
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help="不加载旧模型，使用新模型"
    )

    parser.add_argument(
        "--backtest-model",
        type=str,
        default="models/bb_backtest.joblib",
        help="回测模型保存位置"
    )

    args = parser.parse_args()

    print(
        f"{Color.CYAN}"
        "BB六合彩本地AI系统"
    )

    print("生肖基础概率：")

    for zodiac in ZODIACS:
        print(
            f"  {zodiac}："
            f"{BASE_PROBABILITY[zodiac]:.4%}"
        )

    print(
        f"基础概率最高："
        f"{DEFAULT_ZODIAC}\n"
    )

    if args.mode == "sim":
        run_simulation(
            warmup=args.warmup,
            interval=args.interval,
            fresh=args.fresh
        )

    elif args.mode == "backtest":
        if not args.csv:
            parser.error(
                "backtest模式必须提供--csv"
            )

        run_backtest(
            csv_path=args.csv,
            warmup=args.warmup,
            save_path=args.backtest_model
        )

    elif args.mode == "live":
        if not args.url:
            parser.error(
                "live模式必须配置"
                "BB_API_URL或--url"
            )

        run_live(
            url=args.url,
            interval=args.interval,
            csv_output=args.output,
            fresh=args.fresh
        )

    elif args.mode == "debug-api":
        if not args.url:
            parser.error(
                "debug-api模式必须提供--url"
            )

        debug_api(args.url)


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\n程序已退出。")

    except Exception as error:
        print(
            f"{Color.RED}"
            f"程序启动失败：{error}"
        )

        sys.exit(1)
