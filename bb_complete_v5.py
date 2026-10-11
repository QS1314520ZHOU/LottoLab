# -*- coding: utf-8 -*-
"""
bb_complete.py

BB六合彩多玩法研究系统（V5：自动分页、保守校准、清晰输出）：
1. 特别号：号码、单双、大小、合单双、合大小、尾大小、波色、生肖
2. 正码1-6：号码、单双、大小、尾大小
3. 连码候选
4. 自选不中候选
5. 连肖候选
6. BB公式库 + 多窗口评分 + 漂移/滞后降权
7. 7位置在线AI（SGDClassifier）
8. 模型保存/加载
9. CSV回测、随机模拟、实时API抓取、debug-api

仅用于历史数据研究。随机开奖无法保证存在稳定可预测规律。
"""

import os
import re
import csv
import json
import time
import math
import random
import argparse
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, parse_qsl
from dataclasses import dataclass
from collections import Counter, defaultdict, deque
from typing import Optional, Dict, List, Any, Tuple

import joblib
import numpy as np
import requests
from sklearn.linear_model import SGDClassifier

from bb_cookie import get_cookie


# ============================================================
# 配置
# ============================================================

MODEL_VERSION = 5

DEFAULT_MODEL_PATH = os.getenv(
    "BB_MODEL_PATH",
    "models/bb_complete.joblib"
)

DEFAULT_LIVE_CSV = os.getenv(
    "BB_LIVE_CSV",
    "data/bb_live_history.csv"
)

MAX_HISTORY = int(os.getenv("BB_MAX_HISTORY", "20000"))
AI_MIN_SAMPLES = int(os.getenv("BB_AI_MIN_SAMPLES", "300"))
SAVE_EVERY = int(os.getenv("BB_SAVE_EVERY", "20"))

# 概率校准与融合上限。彩票数据没有经过样本外验证时，不允许模型给出极端概率。
AI_WEIGHT = float(os.getenv("BB_AI_WEIGHT", "0.30"))
FORMULA_WEIGHT = float(os.getenv("BB_FORMULA_WEIGHT", "0.10"))
EMPIRICAL_WEIGHT = float(os.getenv("BB_EMPIRICAL_WEIGHT", "0.20"))
PROBABILITY_TEMPERATURE = float(os.getenv("BB_PROBABILITY_TEMPERATURE", "2.5"))
AI_EVAL_WINDOW = int(os.getenv("BB_AI_EVAL_WINDOW", "1000"))
AI_MIN_EVAL_SAMPLES = int(os.getenv("BB_AI_MIN_EVAL_SAMPLES", "300"))

NUMBERS = list(range(1, 50))
CLASSES = np.arange(1, 50)

POSITION_NAMES = [
    "正码一", "正码二", "正码三",
    "正码四", "正码五", "正码六",
    "特别号"
]

# 2026马年常见生肖排列：
# 1、13、25、37、49为马
# BB平台如果采用不同表，可使用 BB_ZODIAC_ORDER 修改
DEFAULT_ZODIAC_ORDER = [
    "马", "蛇", "龙", "兔", "虎", "牛",
    "鼠", "猪", "狗", "鸡", "猴", "羊"
]


def load_zodiac_order() -> List[str]:
    raw = os.getenv("BB_ZODIAC_ORDER", "").strip()

    if raw:
        values = [x.strip() for x in raw.split(",") if x.strip()]
        if len(values) != 12:
            raise ValueError(
                "BB_ZODIAC_ORDER必须包含12个生肖，例如："
                "马,蛇,龙,兔,虎,牛,鼠,猪,狗,鸡,猴,羊"
            )
        return values

    return DEFAULT_ZODIAC_ORDER.copy()


ZODIAC_ORDER = load_zodiac_order()
ZODIACS = list(dict.fromkeys(ZODIAC_ORDER))

RED_NUMBERS = {
    1, 2, 7, 8, 12, 13, 18, 19, 23,
    24, 29, 30, 34, 35, 40, 45, 46
}

BLUE_NUMBERS = {
    3, 4, 9, 10, 14, 15, 20, 25,
    26, 31, 36, 37, 41, 42, 47, 48
}

GREEN_NUMBERS = {
    5, 6, 11, 16, 17, 21, 22, 27,
    28, 32, 33, 38, 39, 43, 44, 49
}


# ============================================================
# 基础工具
# ============================================================

def wrap49(value: int) -> int:
    """将任意整数循环到1～49。"""
    return (int(value) - 1) % 49 + 1


def zodiac_position(value: int) -> int:
    """将任意整数转换到生肖位置1～12。"""
    return (int(value) - 1) % 12 + 1


def get_zodiac(number: int) -> str:
    return ZODIAC_ORDER[(int(number) - 1) % 12]


def zodiac_from_raw_value(value: int) -> str:
    """
    公式的生肖余12版本。
    例如原始公式结果50，不先循环49，直接映射到1～12。
    """
    return ZODIAC_ORDER[(int(value) - 1) % 12]


def zodiac_numbers(zodiac: str) -> List[int]:
    return [n for n in NUMBERS if get_zodiac(n) == zodiac]


def number_color(number: int) -> str:
    if number in RED_NUMBERS:
        return "红"
    if number in BLUE_NUMBERS:
        return "蓝"
    return "绿"


def parity(number: int) -> str:
    return "单" if number % 2 else "双"


def size_group(number: int) -> str:
    # 常见规则：01～24小，25～49大
    return "小" if number <= 24 else "大"


def tail_size(number: int) -> str:
    # 常见规则：尾数0～4尾小，5～9尾大
    return "尾小" if number % 10 <= 4 else "尾大"


def digit_sum(number: int) -> int:
    return sum(int(ch) for ch in str(abs(int(number))))


def sum_parity(number: int) -> str:
    return "和单" if digit_sum(number) % 2 else "和双"


def sum_size(number: int) -> str:
    # 常见规则：数字和1～6为和小，7～13为和大
    return "和小" if digit_sum(number) <= 6 else "和大"


def normalize_probability(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr[arr < 0] = 0.0

    total = arr.sum()
    if total <= 0:
        return np.ones(49, dtype=float) / 49.0

    return arr / total


def temperature_scale(
    probabilities: np.ndarray,
    temperature: float = PROBABILITY_TEMPERATURE
) -> np.ndarray:
    """对分类概率做温度缩放，抑制SGD在小样本上的虚假高置信度。"""
    p = normalize_probability(probabilities)
    temperature = max(1.0, float(temperature))
    return normalize_probability(np.power(np.clip(p, 1e-12, 1.0), 1.0 / temperature))


def mix_with_uniform(probabilities: np.ndarray, data_weight: float) -> np.ndarray:
    """将未经充分验证的概率向1/49均匀基线收缩。"""
    weight = min(1.0, max(0.0, float(data_weight)))
    uniform = np.ones(49, dtype=float) / 49.0
    return normalize_probability(weight * normalize_probability(probabilities) + (1.0 - weight) * uniform)


def entropy(probabilities: np.ndarray) -> float:
    p = np.asarray(probabilities, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def confidence_from_entropy(probabilities: np.ndarray) -> float:
    max_entropy = math.log(len(probabilities))
    if max_entropy <= 0:
        return 0.0
    return max(0.0, 1.0 - entropy(probabilities) / max_entropy)


# ============================================================
# 数据结构
# ============================================================

@dataclass
class Draw:
    issue: str
    numbers: List[int]

    @property
    def regulars(self) -> List[int]:
        return self.numbers[:6]

    @property
    def special(self) -> int:
        return self.numbers[6]


@dataclass
class FormulaPrediction:
    """
    number不为None：公式预测具体特码号码。
    zodiac始终存在：用于生肖公式评分和投票。
    """
    number: Optional[int]
    zodiac: str


def normalize_draw(issue: Any, numbers: Any) -> Optional[Draw]:
    if isinstance(numbers, str):
        values = [
            int(x) for x in re.findall(r"\d+", numbers)
        ]
    elif isinstance(numbers, (list, tuple)):
        values = []
        for item in numbers:
            try:
                values.append(int(item))
            except (TypeError, ValueError):
                pass
    else:
        return None

    if len(values) < 7:
        return None

    values = values[:7]

    if any(n < 1 or n > 49 for n in values):
        return None

    return Draw(
        issue=str(issue).strip(),
        numbers=values
    )


# ============================================================
# 公式库
# ============================================================

def make_number_prediction(raw_value: int) -> FormulaPrediction:
    number = wrap49(raw_value)
    return FormulaPrediction(
        number=number,
        zodiac=get_zodiac(number)
    )


def make_zodiac_prediction(raw_value: int) -> FormulaPrediction:
    return FormulaPrediction(
        number=None,
        zodiac=zodiac_from_raw_value(raw_value)
    )


def add_dual_formula(
    output: Dict[str, FormulaPrediction],
    name: str,
    raw_value: int
) -> None:
    """
    同一公式同时生成：
    1. 循环49版本
    2. 生肖余12版本
    """
    output[f"{name}/循环49"] = make_number_prediction(raw_value)
    output[f"{name}/生肖余12"] = make_zodiac_prediction(raw_value)


def build_formula_predictions(
    history: List[Draw]
) -> Dict[str, FormulaPrediction]:

    result: Dict[str, FormulaPrediction] = {}

    if not history:
        return result

    last = history[-1]
    regulars = last.regulars
    special = last.special

    # --------------------------------------------------------
    # 特码±N
    # --------------------------------------------------------
    for offset in range(1, 13):
        add_dual_formula(
            result,
            f"特+{offset}",
            special + offset
        )
        add_dual_formula(
            result,
            f"特-{offset}",
            special - offset
        )

    # --------------------------------------------------------
    # 平码位置±N
    # --------------------------------------------------------
    for position, number in enumerate(regulars, start=1):
        for offset in range(1, 7):
            add_dual_formula(
                result,
                f"平{position}+{offset}",
                number + offset
            )
            add_dual_formula(
                result,
                f"平{position}-{offset}",
                number - offset
            )

        # 平N+N、平N-N
        add_dual_formula(
            result,
            f"平{position}+位置",
            number + position
        )
        add_dual_formula(
            result,
            f"平{position}-位置",
            number - position
        )

        # 平码与特码组合
        add_dual_formula(
            result,
            f"平{position}+特",
            number + special
        )
        add_dual_formula(
            result,
            f"平{position}-特",
            number - special
        )
        add_dual_formula(
            result,
            f"特-平{position}",
            special - number
        )
        add_dual_formula(
            result,
            f"平{position}特绝对差",
            abs(number - special)
        )

    # --------------------------------------------------------
    # 平码两两组合
    # --------------------------------------------------------
    for i in range(6):
        for j in range(i + 1, 6):
            a = regulars[i]
            b = regulars[j]
            label = f"平{i + 1}平{j + 1}"

            add_dual_formula(result, f"{label}和", a + b)
            add_dual_formula(result, f"{label}差", a - b)
            add_dual_formula(result, f"{label}绝对差", abs(a - b))

    # --------------------------------------------------------
    # 求和余、最大最小等
    # --------------------------------------------------------
    regular_sum = sum(regulars)
    all_sum = sum(last.numbers)

    add_dual_formula(result, "六平和", regular_sum)
    add_dual_formula(result, "七码和", all_sum)
    add_dual_formula(result, "六平和+特", regular_sum + special)
    add_dual_formula(result, "六平和-特", regular_sum - special)
    add_dual_formula(result, "特-六平和", special - regular_sum)

    max_regular = max(regulars)
    min_regular = min(regulars)

    add_dual_formula(result, "最大平+最小平", max_regular + min_regular)
    add_dual_formula(result, "最大平-最小平", max_regular - min_regular)
    add_dual_formula(result, "最大平+特", max_regular + special)
    add_dual_formula(result, "最小平+特", min_regular + special)

    # --------------------------------------------------------
    # 尾数、合数
    # --------------------------------------------------------
    add_dual_formula(result, "特码尾数", special % 10)
    add_dual_formula(result, "特码合数", digit_sum(special))
    add_dual_formula(result, "六平尾数和", sum(n % 10 for n in regulars))
    add_dual_formula(result, "六平合数和", sum(digit_sum(n) for n in regulars))

    # --------------------------------------------------------
    # 跨期特码
    # --------------------------------------------------------
    if len(history) >= 2:
        sp1 = history[-1].special
        sp2 = history[-2].special

        add_dual_formula(result, "两期特和", sp1 + sp2)
        add_dual_formula(result, "两期特差", sp1 - sp2)
        add_dual_formula(result, "两期特绝对差", abs(sp1 - sp2))
        add_dual_formula(result, "特1倍+特2两倍", sp1 + 2 * sp2)
        add_dual_formula(result, "特1两倍+特2", 2 * sp1 + sp2)

    if len(history) >= 3:
        sp1 = history[-1].special
        sp2 = history[-2].special
        sp3 = history[-3].special

        add_dual_formula(result, "三期特和", sp1 + sp2 + sp3)
        add_dual_formula(result, "三期特加权", 3 * sp1 + 2 * sp2 + sp3)
        add_dual_formula(result, "特1+特2-特3", sp1 + sp2 - sp3)

    # --------------------------------------------------------
    # 热号/冷号
    # --------------------------------------------------------
    for window in (20, 50, 100, 200):
        recent = history[-window:]
        counts = Counter(d.special for d in recent)

        hot_number = max(NUMBERS, key=lambda n: counts[n])
        cold_number = min(NUMBERS, key=lambda n: counts[n])

        result[f"特码热{window}"] = FormulaPrediction(
            hot_number,
            get_zodiac(hot_number)
        )
        result[f"特码冷{window}"] = FormulaPrediction(
            cold_number,
            get_zodiac(cold_number)
        )

    # --------------------------------------------------------
    # 一阶特码转移
    # --------------------------------------------------------
    if len(history) >= 30:
        transitions = defaultdict(Counter)

        for previous, current in zip(history[:-1], history[1:]):
            transitions[previous.special][current.special] += 1

        current_sp = history[-1].special
        candidates = transitions.get(current_sp)

        if candidates:
            predicted = candidates.most_common(1)[0][0]
            result["特码马尔可夫1"] = FormulaPrediction(
                predicted,
                get_zodiac(predicted)
            )

    return result


# ============================================================
# 公式评分、趋势、漂移
# ============================================================

class FormulaState:
    def __init__(self):
        self.zodiac_hits = deque(maxlen=1000)
        self.zodiac_expected = deque(maxlen=1000)
        self.number_hits = deque(maxlen=1000)
        self.loss_streak = 0
        self.total = 0

    def update(
        self,
        prediction: FormulaPrediction,
        actual_special: int
    ) -> None:

        actual_zodiac = get_zodiac(actual_special)
        zodiac_hit = int(prediction.zodiac == actual_zodiac)

        expected = len(zodiac_numbers(prediction.zodiac)) / 49.0

        self.zodiac_hits.append(zodiac_hit)
        self.zodiac_expected.append(expected)

        if prediction.number is not None:
            self.number_hits.append(
                int(prediction.number == actual_special)
            )

        self.loss_streak = 0 if zodiac_hit else self.loss_streak + 1
        self.total += 1

    @staticmethod
    def _mean_last(values: deque, window: int) -> float:
        data = list(values)[-window:]
        return sum(data) / len(data) if data else 0.0

    def zodiac_rate(self, window: int) -> float:
        return self._mean_last(self.zodiac_hits, window)

    def expected_rate(self, window: int) -> float:
        return self._mean_last(self.zodiac_expected, window)

    def edge(self, window: int) -> float:
        return self.zodiac_rate(window) - self.expected_rate(window)

    def trend(self) -> float:
        return self.edge(20) - self.edge(200)

    def drifted(self) -> bool:
        if len(self.zodiac_hits) < 100:
            return False

        short_edge = self.edge(20)
        long_edge = self.edge(200)

        return (
            short_edge < -0.03 or
            short_edge < long_edge - 0.08
        )

    @staticmethod
    def _conservative_edge(hits: deque, expected: float, window: int) -> float:
        """返回扣除约95%随机波动上界后的超额命中率。"""
        data = list(hits)[-window:]
        n = len(data)
        if n < 100:
            return 0.0
        observed = sum(data) / n
        standard_error = math.sqrt(max(1e-12, expected * (1.0 - expected) / n))
        return max(0.0, observed - expected - 1.96 * standard_error)

    def zodiac_weight(self) -> float:
        expected = self.expected_rate(min(500, len(self.zodiac_expected)))
        if expected <= 0:
            return 0.0
        score = (
            0.25 * self._conservative_edge(self.zodiac_hits, expected, 100) +
            0.35 * self._conservative_edge(self.zodiac_hits, expected, 300) +
            0.40 * self._conservative_edge(self.zodiac_hits, expected, 1000)
        )
        streak_factor = math.exp(-min(self.loss_streak, 30) / 20.0)
        drift_factor = 0.10 if self.drifted() else 1.0
        return score * streak_factor * drift_factor

    def number_weight(self) -> float:
        if not self.number_hits:
            return 0.0
        expected = 1.0 / 49.0
        return (
            0.30 * self._conservative_edge(self.number_hits, expected, 300) +
            0.70 * self._conservative_edge(self.number_hits, expected, 1000)
        )

    def weight(self) -> float:
        # 排行榜仍以生肖公式的保守权重展示。
        return self.zodiac_weight()


class FormulaBoard:
    def __init__(self):
        self.states: Dict[str, FormulaState] = {}

    def update(
        self,
        predictions: Dict[str, FormulaPrediction],
        actual_special: int
    ) -> None:

        for name, prediction in predictions.items():
            if name not in self.states:
                self.states[name] = FormulaState()

            self.states[name].update(prediction, actual_special)

    def weight(self, name: str) -> float:
        state = self.states.get(name)
        return state.weight() if state else 0.0

    def top(self, limit: int = 15) -> List[Tuple[str, FormulaState]]:
        rows = list(self.states.items())
        rows.sort(key=lambda item: item[1].weight(), reverse=True)
        return rows[:limit]

    def drift_fraction(self) -> float:
        valid = [
            state for state in self.states.values()
            if state.total >= 100
        ]

        if not valid:
            return 0.0

        return sum(state.drifted() for state in valid) / len(valid)


# ============================================================
# 公式概率融合
# ============================================================

def formula_distributions(
    predictions: Dict[str, FormulaPrediction],
    board: FormulaBoard
) -> Tuple[np.ndarray, np.ndarray]:

    number_votes = np.zeros(49, dtype=float)
    zodiac_votes = {z: 0.0 for z in ZODIACS}

    zodiac_total_weight = 0.0

    for name, prediction in predictions.items():
        state = board.states.get(name)
        if state is None:
            continue

        zodiac_weight = state.zodiac_weight()
        number_weight = state.number_weight() if prediction.number is not None else 0.0

        if zodiac_weight > 0:
            zodiac_total_weight += zodiac_weight
            zodiac_votes[prediction.zodiac] += zodiac_weight

        # 精确号码只能使用精确号码命中表现，不能复用生肖权重。
        if prediction.number is not None and number_weight > 0:
            number_votes[prediction.number - 1] += number_weight

    if zodiac_total_weight <= 0 and number_votes.sum() <= 0:
        return (
            np.ones(49, dtype=float) / 49.0,
            np.array([
                len(zodiac_numbers(z)) / 49.0
                for z in ZODIACS
            ], dtype=float)
        )

    # 生肖票按该生肖包含号码平均分配
    for zodiac, weight in zodiac_votes.items():
        numbers = zodiac_numbers(zodiac)
        if not numbers:
            continue

        share = weight / len(numbers)
        for number in numbers:
            number_votes[number - 1] += share

    number_prob = normalize_probability(number_votes)

    zodiac_prob = np.array(
        [zodiac_votes[z] for z in ZODIACS],
        dtype=float
    )

    if zodiac_prob.sum() <= 0:
        zodiac_prob = np.array([
            len(zodiac_numbers(z)) / 49.0
            for z in ZODIACS
        ])
    else:
        zodiac_prob /= zodiac_prob.sum()

    return number_prob, zodiac_prob


# ============================================================
# AI特征
# ============================================================

def empirical_position_probability(
    history: List[Draw],
    position: int,
    window: int = 300
) -> np.ndarray:

    counts = np.ones(49, dtype=float)  # 拉普拉斯平滑

    for draw in history[-window:]:
        counts[draw.numbers[position] - 1] += 1.0

    return normalize_probability(counts)


def build_features(
    history: List[Draw],
    formula_number_prob: np.ndarray,
    formula_zodiac_prob: np.ndarray,
    drift_fraction: float
) -> np.ndarray:

    features: List[float] = []

    # 最近12期的7个位置
    recent = history[-12:]
    missing = 12 - len(recent)

    features.extend([0.0] * (missing * 7))

    for draw in recent:
        features.extend([n / 49.0 for n in draw.numbers])

    # 多窗口号码频率
    for window in (20, 100, 500):
        counts = np.zeros(49, dtype=float)
        selected = history[-window:]

        for draw in selected:
            for number in draw.numbers:
                counts[number - 1] += 1.0

        denominator = max(1, len(selected) * 7)
        features.extend((counts / denominator).tolist())

    # 各位置均值和标准差
    for window in (20, 100):
        selected = history[-window:]

        for position in range(7):
            values = [
                draw.numbers[position] / 49.0
                for draw in selected
            ]

            if values:
                features.append(float(np.mean(values)))
                features.append(float(np.std(values)))
            else:
                features.extend([0.0, 0.0])

    # 特码生肖频率
    for window in (20, 100, 500):
        selected = history[-window:]
        counts = Counter(get_zodiac(d.special) for d in selected)
        denominator = max(1, len(selected))

        for zodiac in ZODIACS:
            features.append(counts[zodiac] / denominator)

    # 公式聚合概率，不使用公式名作为维度，防止维度变化
    features.extend(formula_number_prob.tolist())
    features.extend(formula_zodiac_prob.tolist())
    features.append(float(drift_fraction))

    return np.asarray(features, dtype=np.float64)


# ============================================================
# 7位置在线AI
# ============================================================

class MultiPositionAI:
    def __init__(self):
        self.models = [
            SGDClassifier(
                loss="log_loss",
                alpha=0.001,
                learning_rate="optimal",
                average=True,
                random_state=100 + position
            )
            for position in range(7)
        ]
        self.initialized = [False] * 7
        self.samples = 0
        self.log_losses = [deque(maxlen=AI_EVAL_WINDOW) for _ in range(7)]
        self.top1_hits = [deque(maxlen=AI_EVAL_WINDOW) for _ in range(7)]

    @property
    def ready(self) -> bool:
        return self.samples >= AI_MIN_SAMPLES and all(self.initialized)

    def predict_position(self, features: np.ndarray, position: int) -> Optional[np.ndarray]:
        if not self.initialized[position]:
            return None
        probabilities = self.models[position].predict_proba(features.reshape(1, -1))[0]
        output = np.zeros(49, dtype=float)
        for cls, probability in zip(self.models[position].classes_, probabilities):
            output[int(cls) - 1] = probability
        return normalize_probability(output)

    def predict_all(self, features: np.ndarray) -> List[Optional[np.ndarray]]:
        return [self.predict_position(features, position) for position in range(7)]

    def record_predictions(
        self,
        probabilities: List[Optional[np.ndarray]],
        actual_numbers: List[int]
    ) -> None:
        """仅记录预测之后才揭晓的结果，用于决定AI是否真的优于均匀基线。"""
        for position, probability in enumerate(probabilities):
            if probability is None:
                continue
            # 传入值已在predict()阶段完成温度缩放，避免重复校准。
            p = normalize_probability(probability)
            actual_index = actual_numbers[position] - 1
            self.log_losses[position].append(-math.log(max(1e-12, float(p[actual_index]))))
            self.top1_hits[position].append(int(int(np.argmax(p)) == actual_index))

    def reliability(self, position: int) -> float:
        """样本外对数损失未优于均匀随机时，AI权重自动归零。"""
        losses = self.log_losses[position]
        if len(losses) < AI_MIN_EVAL_SAMPLES:
            return 0.0
        baseline = math.log(49.0)
        mean_loss = sum(losses) / len(losses)
        relative_gain = max(0.0, (baseline - mean_loss) / baseline)
        return min(AI_WEIGHT, relative_gain * 2.0)

    def top1_rate(self, position: int) -> float:
        values = self.top1_hits[position]
        return sum(values) / len(values) if values else 0.0

    def update(self, features: np.ndarray, actual_numbers: List[int]) -> None:
        x = features.reshape(1, -1)
        for position in range(7):
            y = np.array([actual_numbers[position]], dtype=int)
            if not self.initialized[position]:
                self.models[position].partial_fit(x, y, classes=CLASSES)
                self.initialized[position] = True
            else:
                self.models[position].partial_fit(x, y)
        self.samples += 1


# ============================================================
# 概率转玩法
# ============================================================

def aggregate_number_market(
    probabilities: np.ndarray
) -> Dict[str, Dict[str, float]]:

    result = {
        "号码": {},
        "单双": {"单": 0.0, "双": 0.0},
        "大小": {"大": 0.0, "小": 0.0},
        "尾大小": {"尾大": 0.0, "尾小": 0.0},
        "合单双": {"和单": 0.0, "和双": 0.0},
        "合大小": {"和大": 0.0, "和小": 0.0},
        "波色": {"红": 0.0, "蓝": 0.0, "绿": 0.0},
        "生肖": {z: 0.0 for z in ZODIACS},
        "尾数": {str(i): 0.0 for i in range(10)}
    }

    for number in NUMBERS:
        probability = float(probabilities[number - 1])

        result["号码"][f"{number:02d}"] = probability
        result["单双"][parity(number)] += probability
        result["大小"][size_group(number)] += probability
        result["尾大小"][tail_size(number)] += probability
        result["合单双"][sum_parity(number)] += probability
        result["合大小"][sum_size(number)] += probability
        result["波色"][number_color(number)] += probability
        result["生肖"][get_zodiac(number)] += probability
        result["尾数"][str(number % 10)] += probability

    return result


def regular_inclusion_probability(
    position_probabilities: List[np.ndarray]
) -> np.ndarray:
    """
    估算号码进入六个正码位置的概率：
    1 - Π(1 - p_i)

    属于近似值，因为六个位置实际是不放回的。
    """
    output = np.zeros(49, dtype=float)

    for index in range(49):
        not_selected = 1.0

        for position_prob in position_probabilities[:6]:
            not_selected *= 1.0 - float(position_prob[index])

        output[index] = 1.0 - not_selected

    return output


def build_combination_candidates(
    regular_inclusion: np.ndarray,
    special_market: Dict[str, Dict[str, float]]
) -> Dict[str, Any]:

    ranked_numbers = sorted(
        NUMBERS,
        key=lambda n: regular_inclusion[n - 1],
        reverse=True
    )

    low_numbers = list(reversed(ranked_numbers))

    zodiac_ranking = sorted(
        ZODIACS,
        key=lambda z: special_market["生肖"][z],
        reverse=True
    )

    return {
        "连码": {
            "二全中候选": ranked_numbers[:2],
            "三全中候选": ranked_numbers[:3],
            "四全中候选": ranked_numbers[:4],
            "扩展候选池": ranked_numbers[:10]
        },
        "自选不中": {
            "4个号码": low_numbers[:4],
            "5个号码": low_numbers[:5],
            "6个号码": low_numbers[:6],
            "7个号码": low_numbers[:7],
            "8个号码": low_numbers[:8],
            "9个号码": low_numbers[:9]
        },
        "连肖": {
            "二连肖候选": zodiac_ranking[:2],
            "三连肖候选": zodiac_ranking[:3],
            "四连肖候选": zodiac_ranking[:4],
            "五连肖候选": zodiac_ranking[:5]
        }
    }


# ============================================================
# 主引擎
# ============================================================

class BBEngine:
    def __init__(self):
        self.version = MODEL_VERSION
        self.history: List[Draw] = []
        self.seen_issues = set()

        self.formula_board = FormulaBoard()
        self.ai = MultiPositionAI()

        self.processed = 0
        self.last_prediction = None

    def predict(self) -> Dict[str, Any]:
        formula_predictions = build_formula_predictions(self.history)

        formula_num_prob, formula_zodiac_prob = formula_distributions(
            formula_predictions,
            self.formula_board
        )

        drift_fraction = self.formula_board.drift_fraction()

        features = build_features(
            self.history,
            formula_num_prob,
            formula_zodiac_prob,
            drift_fraction
        )

        raw_ai_predictions = self.ai.predict_all(features)
        ai_predictions = [
            temperature_scale(p) if p is not None else None
            for p in raw_ai_predictions
        ]
        ai_reliabilities = [self.ai.reliability(position) for position in range(7)]
        final_positions: List[np.ndarray] = []
        uniform = np.ones(49, dtype=float) / 49.0

        for position in range(7):
            empirical = empirical_position_probability(self.history, position)
            ai_prob = ai_predictions[position]
            ai_weight = ai_reliabilities[position] if self.ai.ready else 0.0
            formula_weight = FORMULA_WEIGHT if position == 6 else 0.0

            # 未经样本外证明的部分全部回归均���随机基线，防止80%虚假概率。
            base_weight = max(0.0, 1.0 - EMPIRICAL_WEIGHT - ai_weight - formula_weight)
            final = base_weight * uniform + EMPIRICAL_WEIGHT * empirical
            if ai_prob is not None and ai_weight > 0:
                final += ai_weight * ai_prob
            if position == 6 and formula_weight > 0:
                final += formula_weight * formula_num_prob
            final_positions.append(normalize_probability(final))

        markets = {
            POSITION_NAMES[position]: aggregate_number_market(
                final_positions[position]
            )
            for position in range(7)
        }

        inclusion = regular_inclusion_probability(final_positions)
        combinations = build_combination_candidates(
            inclusion,
            markets["特别号"]
        )

        prediction = {
            "position_probabilities": final_positions,
            "markets": markets,
            "regular_inclusion": inclusion,
            "combinations": combinations,
            "formula_predictions": formula_predictions,
            "formula_number_probability": formula_num_prob,
            "formula_zodiac_probability": formula_zodiac_prob,
            "drift_fraction": drift_fraction,
            "features": features,
            "ai_predictions": ai_predictions,
            "ai_reliabilities": ai_reliabilities
        }

        self.last_prediction = prediction
        return prediction

    def process_draw(self, draw: Draw) -> Optional[Dict[str, Any]]:
        if draw.issue in self.seen_issues:
            return None

        prediction = self.predict()

        self.ai.record_predictions(
            prediction["ai_predictions"],
            draw.numbers
        )

        self.ai.update(
            prediction["features"],
            draw.numbers
        )

        self.formula_board.update(
            prediction["formula_predictions"],
            draw.special
        )

        self.history.append(draw)

        if len(self.history) > MAX_HISTORY:
            self.history = self.history[-MAX_HISTORY:]

        self.seen_issues.add(draw.issue)
        self.processed += 1

        return prediction

    def save(self, path: str = DEFAULT_MODEL_PATH) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)

    @classmethod
    def load(cls, path: str = DEFAULT_MODEL_PATH) -> "BBEngine":
        target = Path(path)

        if not target.exists():
            return cls()

        try:
            engine = joblib.load(target)

            if getattr(engine, "version", None) != MODEL_VERSION:
                print("模型版本不一致，已创建新模型。")
                return cls()

            return engine

        except Exception as exc:
            print(f"模型加载失败，已创建新模型：{exc}")
            return cls()


# ============================================================
# 输出
# ============================================================

def top_items(
    values: Dict[str, float],
    limit: int = 3
) -> List[Tuple[str, float]]:

    return sorted(
        values.items(),
        key=lambda item: item[1],
        reverse=True
    )[:limit]


def winner(values: Dict[str, float]) -> Tuple[str, float]:
    return max(values.items(), key=lambda item: item[1])


def print_prediction(
    engine: BBEngine,
    prediction: Dict[str, Any],
    actual_draw: Optional[Draw] = None,
    heading: Optional[str] = None
) -> None:
    """使用分区式输出；actual_draw存在时表示样本外验证，否则表示真正的下一期预测。"""
    market = prediction["markets"]["特别号"]
    top_numbers = top_items(market["号码"], 5)
    top_zodiacs = top_items(market["生肖"], 3)
    sp_parity = winner(market["单双"])
    sp_size = winner(market["大小"])
    sp_color = winner(market["波色"])
    sp_tail = winner(market["尾大小"])
    concentration = confidence_from_entropy(prediction["position_probabilities"][6])
    ai_weight = prediction["ai_reliabilities"][6]

    width = 76
    print("\n" + "=" * width)
    if heading:
        print(f"【{heading}】")
    elif actual_draw:
        print("【历史逐期验证】预测在开奖前生成，再用本期开奖结果验证")
    else:
        print("【下一期预测】尚未开奖，仅显示概率排序")

    if actual_draw:
        regular_text = " ".join(f"{n:02d}" for n in actual_draw.regulars)
        print(f"期号：{actual_draw.issue}")
        print(f"实际开奖：正码 {regular_text} ｜ 特别号 {actual_draw.special:02d}（{get_zodiac(actual_draw.special)}）")

    print("-" * width)
    print("特别号候选 Top 5")
    for rank, (number, probability) in enumerate(top_numbers, start=1):
        print(f"  {rank}. {number}    模型概率 {probability:6.2%}")

    print("特别号生肖 Top 3")
    print("  " + " ｜ ".join(
        f"{rank}.{zodiac} {probability:.2%}"
        for rank, (zodiac, probability) in enumerate(top_zodiacs, start=1)
    ))

    print("属性倾向")
    print(
        f"  单双：{sp_parity[0]} {sp_parity[1]:.2%} ｜ "
        f"大小：{sp_size[0]} {sp_size[1]:.2%} ｜ "
        f"波色：{sp_color[0]} {sp_color[1]:.2%} ｜ "
        f"尾大小：{sp_tail[0]} {sp_tail[1]:.2%}"
    )

    print("模型状态")
    print(
        f"  历史样本：{engine.ai.samples} ｜ "
        f"AI：{'已启用' if engine.ai.ready and ai_weight > 0 else '未启用'} ｜ "
        f"AI有效权重：{ai_weight:.2%}"
    )
    print(
        f"  公式漂移：{prediction['drift_fraction']:.2%} ｜ "
        f"概率集中度：{concentration:.2%}（不是命中率）"
    )

    combos = prediction["combinations"]
    print("组合参考")
    print(f"  二全中候选：{combos['连码']['二全中候选']}")
    print(f"  三全中候选：{combos['连码']['三全中候选']}")
    print(f"  四全中候选：{combos['连码']['四全中候选']}")
    print(f"  自选不中6码：{combos['自选不中']['6个号码']}")
    print(f"  二连肖候选：{combos['连肖']['二连肖候选']}")
    print(f"  三连肖候选：{combos['连肖']['三连肖候选']}")

    if actual_draw:
        top1_number = int(top_numbers[0][0])
        top5_numbers = {int(number) for number, _ in top_numbers}
        top1_zodiac = top_zodiacs[0][0]
        print("-" * width)
        print("验证结果")
        print(
            f"  特别号 Top1：预测 {top1_number:02d} ｜ "
            f"实际 {actual_draw.special:02d} ｜ "
            f"{'命中' if top1_number == actual_draw.special else '未中'}"
        )
        print(f"  特别号 Top5：{'命中' if actual_draw.special in top5_numbers else '未中'}")
        print(
            f"  生肖 Top1：预测 {top1_zodiac} ｜ "
            f"实际 {get_zodiac(actual_draw.special)} ｜ "
            f"{'命中' if top1_zodiac == get_zodiac(actual_draw.special) else '未中'}"
        )
    print("=" * width)


def print_formula_ranking(
    engine: BBEngine,
    limit: int = 15
) -> None:

    print("\n公式排行榜")
    print("-" * 96)
    print(
        f"{'公式':<34}"
        f"{'样本':>8}"
        f"{'近20':>10}"
        f"{'近200':>10}"
        f"{'趋势':>10}"
        f"{'权重':>10}"
        f"{'漂移':>8}"
    )

    for name, state in engine.formula_board.top(limit):
        print(
            f"{name:<34}"
            f"{state.total:>8}"
            f"{state.zodiac_rate(20):>9.2%}"
            f"{state.zodiac_rate(200):>9.2%}"
            f"{state.trend():>+9.3f}"
            f"{state.weight():>10.5f}"
            f"{'是' if state.drifted() else '否':>8}"
        )


# ============================================================
# CSV
# ============================================================

def load_csv(path: str) -> List[Draw]:
    draws = []

    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            issue = (
                row.get("issue") or
                row.get("期号") or
                row.get("drawNumber") or
                ""
            )

            position_keys = [
                ("p1", "正码一"),
                ("p2", "正码二"),
                ("p3", "正码三"),
                ("p4", "正码四"),
                ("p5", "正码五"),
                ("p6", "正码六"),
                ("sp", "特码")
            ]

            numbers = []

            for english, chinese in position_keys:
                value = row.get(english)

                if value is None:
                    value = row.get(chinese)

                if value is None:
                    numbers = []
                    break

                numbers.append(int(value))

            draw = normalize_draw(issue, numbers)

            if draw:
                draws.append(draw)

    return draws


def append_draw_csv(path: str, draw: Draw) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    exists = target.exists()

    with open(target, "a", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)

        if not exists:
            writer.writerow([
                "issue", "p1", "p2", "p3",
                "p4", "p5", "p6", "sp"
            ])

        writer.writerow([draw.issue] + draw.numbers)


# ============================================================
# 通用网页/API客户端
# ============================================================

ISSUE_KEYS = [
    "issue", "issueNo", "issueNum", "period",
    "periodNo", "expect", "drawNumber",
    "drawNo", "round", "roundNo",
    "lotteryIssue", "gameNo", "num", "game_id"
]

NUMBER_KEYS = [
    "numbers", "number", "nums", "codes",
    "code", "openCode", "openCodes",
    "drawCode", "drawResult", "result",
    "lotteryResult", "winningNumbers",
    "openNumber", "openNumbers"
]


def parse_json_env(name: str, default: Any) -> Any:
    raw = os.getenv(name, "").strip()

    if not raw:
        return default

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name}不是有效JSON：{exc}") from exc


def issue_sort_key(draw: Draw):
    digits = re.findall(r"\d+", draw.issue)
    if digits:
        try:
            return 0, int("".join(digits))
        except ValueError:
            pass
    return 1, draw.issue


class BBWebClient:
    def __init__(self, url: Optional[str] = None):
        raw_url = url or os.getenv("BB_API_URL", "").strip()
        parts = urlsplit(raw_url)
        self.url = urlunsplit((parts.scheme, parts.netloc, parts.path, "", parts.fragment))
        url_params = dict(parse_qsl(parts.query, keep_blank_values=True))

        self.method = os.getenv("BB_API_METHOD", "GET").upper().strip()
        self.timeout = float(os.getenv("BB_REQUEST_TIMEOUT", "20"))
        self.body_mode = os.getenv("BB_API_BODY_MODE", "json").lower()

        # 支持直接粘贴完整URL；环境变量中的参数覆盖URL查询参数。
        self.params = url_params
        self.params.update(parse_json_env("BB_API_PARAMS", {}))
        self.body = parse_json_env("BB_API_BODY", {})
        self.extra_headers = parse_json_env("BB_API_HEADERS", {})

        self.session = requests.Session()

        headers = {
            "User-Agent": os.getenv(
                "BB_USER_AGENT",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/126 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "X-Requested-With": "XMLHttpRequest"
        }

        referer = os.getenv("BB_REFERER", "").strip()
        cookie = get_cookie()
        authorization = os.getenv("BB_AUTHORIZATION", "").strip()

        if referer:
            headers["Referer"] = referer

        if cookie:
            headers["Cookie"] = cookie

        if authorization:
            headers["Authorization"] = authorization

        headers.update(self.extra_headers)
        self.session.headers.update(headers)

    def fetch_raw(self, params_override: Optional[Dict[str, Any]] = None) -> Any:
        if not self.url:
            raise RuntimeError(
                "未配置BB_API_URL。请先设置真实开奖记录接口地址。"
            )

        request_params = dict(self.params)
        if params_override:
            request_params.update(params_override)
        if os.getenv("BB_ADD_TIMESTAMP", "0").strip().lower() in ("1", "true", "yes"):
            request_params["timestamp"] = int(time.time() * 1000)

        if self.method == "GET":
            response = self.session.get(
                self.url,
                params=request_params,
                timeout=self.timeout
            )
        elif self.method == "POST":
            if self.body_mode == "form":
                response = self.session.post(
                    self.url,
                    params=request_params,
                    data=self.body,
                    timeout=self.timeout
                )
            else:
                response = self.session.post(
                    self.url,
                    params=request_params,
                    json=self.body,
                    timeout=self.timeout
                )
        else:
            raise ValueError("BB_API_METHOD只支持GET或POST")

        response.raise_for_status()

        try:
            return response.json()
        except ValueError:
            text = response.text.strip()

            # 部分接口外层是JavaScript回调
            match = re.search(r"^[^(]*\((.*)\)\s*;?\s*$", text, re.S)

            if match:
                return json.loads(match.group(1))

            raise RuntimeError(
                "接口返回的不是JSON。前500字符：\n" + text[:500]
            )

    @staticmethod
    def walk_nodes(value: Any):
        yield value

        if isinstance(value, dict):
            for child in value.values():
                yield from BBWebClient.walk_nodes(child)

        elif isinstance(value, list):
            for child in value:
                yield from BBWebClient.walk_nodes(child)

    @staticmethod
    def find_value(item: Dict[str, Any], keys: List[str]) -> Any:
        lowered = {str(k).lower(): v for k, v in item.items()}

        for key in keys:
            if key in item:
                return item[key]

            value = lowered.get(key.lower())
            if value is not None:
                return value

        return None

    @staticmethod
    def extract_position_numbers(
        item: Dict[str, Any]
    ) -> Optional[List[int]]:

        key_groups = [
            ["p1", "ball1", "num1", "number1", "平码一", "正码一"],
            ["p2", "ball2", "num2", "number2", "平码二", "正码二"],
            ["p3", "ball3", "num3", "number3", "平码三", "正码三"],
            ["p4", "ball4", "num4", "number4", "平码四", "正码四"],
            ["p5", "ball5", "num5", "number5", "平码五", "正码五"],
            ["p6", "ball6", "num6", "number6", "平码六", "正码六"],
            ["sp", "special", "specialNumber", "特码", "特别号"]
        ]

        lowered = {
            str(key).lower(): value
            for key, value in item.items()
        }

        values = []

        for group in key_groups:
            found = None

            for key in group:
                if key in item:
                    found = item[key]
                    break

                if key.lower() in lowered:
                    found = lowered[key.lower()]
                    break

            if found is None:
                return None

            try:
                values.append(int(found))
            except (TypeError, ValueError):
                return None

        return values

    @classmethod
    def normalize_item(
        cls,
        item: Dict[str, Any]
    ) -> Optional[Draw]:

        issue = cls.find_value(item, ISSUE_KEYS)

        if issue is None:
            return None

        # 当前BB接口：result_parser使用01～06和sp。
        result_parser = item.get("result_parser")
        if isinstance(result_parser, dict):
            try:
                values = [int(result_parser[f"{i:02d}"]) for i in range(1, 7)]
                values.append(int(result_parser["sp"]))
                draw = normalize_draw(issue, values)
                if draw:
                    return draw
            except (KeyError, TypeError, ValueError):
                pass

        # 兼容同一接口的result结构：01～07。
        result = item.get("result")
        if isinstance(result, dict):
            try:
                values = [int(result[f"{i:02d}"]) for i in range(1, 8)]
                draw = normalize_draw(issue, values)
                if draw:
                    return draw
            except (KeyError, TypeError, ValueError):
                pass

        position_numbers = cls.extract_position_numbers(item)

        if position_numbers:
            return normalize_draw(issue, position_numbers)

        numbers = cls.find_value(item, NUMBER_KEYS)

        if isinstance(numbers, dict):
            position_numbers = cls.extract_position_numbers(numbers)

            if position_numbers:
                return normalize_draw(issue, position_numbers)

            nested_numbers = cls.find_value(numbers, NUMBER_KEYS)
            if nested_numbers is not None:
                numbers = nested_numbers

        return normalize_draw(issue, numbers)

    def parse_draws(self, payload: Any) -> List[Draw]:
        found: Dict[str, Draw] = {}

        for node in self.walk_nodes(payload):
            if not isinstance(node, dict):
                continue

            draw = self.normalize_item(node)

            if draw:
                found[draw.issue] = draw

        draws = list(found.values())

        draws.sort(key=issue_sort_key)
        return draws

    def fetch_draws(
        self,
        params_override: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, List[Draw]]:
        payload = self.fetch_raw(params_override=params_override)
        draws = self.parse_draws(payload)
        return payload, draws

    def fetch_history_pages(
        self,
        max_pages: int = 200,
        pause_seconds: float = 0.20
    ) -> List[Draw]:
        """启动时自动抓取page=1..N；空页、重复页或连续无新增时停止。"""
        found: Dict[str, Draw] = {}
        previous_signature: Optional[Tuple[str, ...]] = None
        no_new_pages = 0

        for page in range(1, max(1, max_pages) + 1):
            _, draws = self.fetch_draws({"page": page})
            signature = tuple(draw.issue for draw in draws)

            if not draws:
                print(f"历史抓取：第 {page} 页为空，停止。")
                break
            if signature == previous_signature:
                print(f"历史抓取：第 {page} 页与上一页重复，停止。")
                break

            before = len(found)
            for draw in draws:
                found[draw.issue] = draw
            added = len(found) - before
            print(f"历史抓取：第 {page} 页解析 {len(draws)} 期，新增 {added} 期，累计 {len(found)} 期")

            no_new_pages = no_new_pages + 1 if added == 0 else 0
            if no_new_pages >= 2:
                print("历史抓取：连续两页无新增期号，停止。")
                break

            previous_signature = signature
            if pause_seconds > 0:
                time.sleep(pause_seconds)

        result = list(found.values())
        result.sort(key=issue_sort_key)
        return result


# ============================================================
# 运行模式
# ============================================================

def create_engine(model_path: str, fresh: bool) -> BBEngine:
    if fresh:
        path = Path(model_path)
        if path.exists():
            path.unlink()
        return BBEngine()

    return BBEngine.load(model_path)


def run_simulation(args) -> None:
    print("警告：sim模式生成的是独立均匀随机开奖，只用于压力测试；长期命中率应接近随机基线。")
    engine = create_engine(args.model, args.fresh)

    for index in range(args.count):
        numbers = random.sample(NUMBERS, 7)
        draw = Draw(
            issue=f"SIM{engine.processed + 1:09d}",
            numbers=numbers
        )

        prediction = engine.process_draw(draw)

        if prediction and engine.processed > args.warmup:
            print_prediction(engine, prediction, draw)

        if engine.processed % SAVE_EVERY == 0:
            engine.save(args.model)

        if args.interval > 0:
            time.sleep(args.interval)

    engine.save(args.model)
    print_formula_ranking(engine)


def run_backtest(args) -> None:
    if not args.csv:
        raise RuntimeError("backtest模式必须提供 --csv")

    engine = create_engine(args.model, args.fresh)
    draws = load_csv(args.csv)
    draws.sort(key=issue_sort_key)
    print(f"读取到 {len(draws)} 期开奖。")

    number_top1_hits = 0
    number_top5_hits = 0
    zodiac_top1_hits = 0
    evaluated = 0
    log_loss_sum = 0.0
    brier_sum = 0.0

    for draw in draws:
        prediction = engine.process_draw(draw)
        if prediction is None:
            continue

        if engine.processed > args.warmup:
            probability = prediction["position_probabilities"][6]
            special_market = prediction["markets"]["特别号"]
            top5 = top_items(special_market["号码"], 5)
            predicted_number = int(top5[0][0])
            top5_numbers = {int(number) for number, _ in top5}
            predicted_zodiac = winner(special_market["生肖"])[0]

            number_top1_hits += int(predicted_number == draw.special)
            number_top5_hits += int(draw.special in top5_numbers)
            zodiac_top1_hits += int(predicted_zodiac == get_zodiac(draw.special))

            actual_index = draw.special - 1
            log_loss_sum += -math.log(max(1e-12, float(probability[actual_index])))
            target = np.zeros(49, dtype=float)
            target[actual_index] = 1.0
            brier_sum += float(np.sum((probability - target) ** 2))
            evaluated += 1

        if args.ranking_every > 0 and engine.processed % args.ranking_every == 0:
            print(
                f"已处理 {engine.processed} 期 ｜ 有效评估 {evaluated} 期 ｜ "
                f"Top1 {number_top1_hits} ｜ Top5 {number_top5_hits} ｜ "
                f"生肖Top1 {zodiac_top1_hits}"
            )

    engine.save(args.model)
    print("\n" + "=" * 76)
    print("【严格时间顺序回测结果】")
    print(f"总数据：{len(draws)} 期 ｜ 预热：{args.warmup} 期 ｜ 有效评估：{evaluated} 期")

    if evaluated:
        top1_rate = number_top1_hits / evaluated
        top5_rate = number_top5_hits / evaluated
        zodiac_rate = zodiac_top1_hits / evaluated
        mean_log_loss = log_loss_sum / evaluated
        mean_brier = brier_sum / evaluated
        baseline_log_loss = math.log(49.0)
        baseline_brier = 48.0 / 49.0

        print(f"特别号 Top1：{number_top1_hits}/{evaluated} = {top1_rate:.4%} ｜ 随机基线约 2.0408%")
        print(f"特别号 Top5：{number_top5_hits}/{evaluated} = {top5_rate:.4%} ｜ 随机基线约 10.2041%")
        print(f"生肖 Top1：{zodiac_top1_hits}/{evaluated} = {zodiac_rate:.4%} ｜ 随机基线约 8%～10%")
        print(f"Log Loss：{mean_log_loss:.6f} ｜ 均匀基线 {baseline_log_loss:.6f} ｜ {'优于基线' if mean_log_loss < baseline_log_loss else '未优于基线'}")
        print(f"Brier Score：{mean_brier:.6f} ｜ 均匀基线 {baseline_brier:.6f} ｜ {'优于基线' if mean_brier < baseline_brier else '未优于基线'}")
        print("结论：只有在足够大的样本外区间里同时改善命中率、Log Loss和Brier，才说明模型可能有增量价值。")
    else:
        print("没有有效评估期，请降低 --warmup 或提供更多历史数据。")
    print("=" * 76)
    print_formula_ranking(engine)


def run_debug_api(args) -> None:
    client = BBWebClient(args.url)
    payload, draws = client.fetch_draws()

    print("接口返回结构：")
    print(json.dumps(
        payload,
        ensure_ascii=False,
        indent=2
    )[:10000])

    print(f"\n成功解析：{len(draws)}期开奖")

    for draw in draws[-10:]:
        print(draw.issue, draw.numbers)


def run_live(args) -> None:
    engine = create_engine(args.model, args.fresh)
    client = BBWebClient(args.url)
    poll_interval = float(os.getenv("BB_POLL_INTERVAL", str(args.interval)))
    output_csv = args.output or DEFAULT_LIVE_CSV

    csv_draws = load_csv(output_csv) if Path(output_csv).exists() else []
    csv_issues = {draw.issue for draw in csv_draws}

    print("实时模式启动")
    print(f"接口：{client.url}")
    print(f"轮询间隔：{poll_interval} 秒")
    print(f"模型文件：{args.model}")
    print(f"历史CSV：{output_csv}")

    # 只在空模型时合并本地CSV和远端分页历史，保证严格按时间顺序训练。
    if engine.processed == 0:
        combined: Dict[str, Draw] = {draw.issue: draw for draw in csv_draws}
        if args.history_pages > 0:
            remote_history = client.fetch_history_pages(
                max_pages=args.history_pages,
                pause_seconds=args.page_pause
            )
            for draw in remote_history:
                combined[draw.issue] = draw

        history = sorted(combined.values(), key=issue_sort_key)
        print(f"开始预热模型：共 {len(history)} 期历史数据")
        for draw in history:
            engine.process_draw(draw)
            if draw.issue not in csv_issues:
                append_draw_csv(output_csv, draw)
                csv_issues.add(draw.issue)
        engine.save(args.model)
        print(f"预热完成：AI样本 {engine.ai.samples} 期")
    else:
        print(f"已加载模型：AI样本 {engine.ai.samples} 期；跳过全量历史回灌。")

    if engine.history:
        next_prediction = engine.predict()
        print_prediction(
            engine,
            next_prediction,
            actual_draw=None,
            heading=f"下一期预测（基于最新期 {engine.history[-1].issue}）"
        )

    while True:
        try:
            # 实时阶段只轮询第一页，历史分页只在启动预热时执行一次。
            _, draws = client.fetch_draws({"page": 1})
            new_draws = [draw for draw in draws if draw.issue not in engine.seen_issues]
            new_draws.sort(key=issue_sort_key)

            for draw in new_draws:
                prediction = engine.process_draw(draw)
                if prediction is None:
                    continue
                if draw.issue not in csv_issues:
                    append_draw_csv(output_csv, draw)
                    csv_issues.add(draw.issue)
                print_prediction(
                    engine,
                    prediction,
                    actual_draw=draw,
                    heading="新期开奖后的样本外验证"
                )

            if new_draws:
                engine.save(args.model)
                next_prediction = engine.predict()
                print_prediction(
                    engine,
                    next_prediction,
                    actual_draw=None,
                    heading=f"下一期预测（基于最新期 {engine.history[-1].issue}）"
                )

            time.sleep(poll_interval)

        except KeyboardInterrupt:
            print("\n收到退出指令，正在保存模型……")
            engine.save(args.model)
            break
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            print(f"HTTP请求失败：{status} {exc}")
            if status in (401, 403):
                print("Cookie、MORTLACH或Authorization可能已经失效。")
            time.sleep(max(5.0, poll_interval))
        except Exception as exc:
            print(f"实时抓取错误：{type(exc).__name__}: {exc}")
            time.sleep(max(5.0, poll_interval))


# ============================================================
# 命令行
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BB六合彩多玩法AI研究系统"
    )

    parser.add_argument(
        "--mode",
        choices=["sim", "backtest", "live", "debug-api"],
        default="sim"
    )

    parser.add_argument("--url", default=None)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--output", default=DEFAULT_LIVE_CSV)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)

    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--ranking-every", type=int, default=1000)
    parser.add_argument("--history-pages", type=int, default=int(os.getenv("BB_HISTORY_PAGES", "200")))
    parser.add_argument("--page-pause", type=float, default=float(os.getenv("BB_PAGE_PAUSE", "0.20")))
    parser.add_argument("--fresh", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print("生肖顺序：", ",".join(ZODIAC_ORDER))

    if args.mode == "sim":
        run_simulation(args)
    elif args.mode == "backtest":
        run_backtest(args)
    elif args.mode == "debug-api":
        run_debug_api(args)
    elif args.mode == "live":
        run_live(args)


if __name__ == "__main__":
    main()
