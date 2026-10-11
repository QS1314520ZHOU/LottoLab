# bb_core.py
# -*- coding: utf-8 -*-

import math
import os
import random
import zlib

from collections import Counter, defaultdict, deque
from functools import partial

import joblib
import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler


# ============================================================
# 配置
# ============================================================

MODEL_PATH = os.getenv(
    "BB_MODEL_PATH",
    "models/bb_engine.joblib"
)

START_CAPITAL = 200.0
BASE_BET = 2.0

# 中奖总返还赔率，包含本金
GROSS_ODDS = 1.95

MAX_HISTORY = 20000
AI_MIN_SAMPLES = 300
BATCH_MIN_SAMPLES = 1000
BATCH_RETRAIN_INTERVAL = 500
BATCH_MAX_SAMPLES = 8000
HASH_SIZE = 256

# 请按BB平台当前生肖表修改
POSITION_TO_ZODIAC = {
    1: "蛇",
    2: "龙",
    3: "兔",
    4: "虎",
    5: "牛",
    6: "鼠",
    7: "猪",
    8: "狗",
    9: "鸡",
    10: "猴",
    11: "羊",
    12: "马",
}

ZODIACS = tuple(POSITION_TO_ZODIAC.values())

ZODIAC_TO_ID = {
    zodiac: index
    for index, zodiac in enumerate(ZODIACS)
}

ID_TO_ZODIAC = {
    index: zodiac
    for zodiac, index in ZODIAC_TO_ID.items()
}


# ============================================================
# 基础函数
# ============================================================

def wrap49(value):
    return (int(value) - 1) % 49 + 1


def zodiac_position(value):
    return (int(value) - 1) % 12 + 1


def get_zodiac(value):
    return POSITION_TO_ZODIAC[
        zodiac_position(value)
    ]


def map_value(value, mode="wrap49"):
    if mode == "wrap49":
        return get_zodiac(wrap49(value))

    if mode == "mod12":
        return get_zodiac(value)

    raise ValueError(f"未知映射模式：{mode}")


def special_number(draw):
    return int(draw["nums"][-1])


def regular_numbers(draw):
    return list(draw["nums"][:6])


def digit_sum(number):
    return sum(
        int(char)
        for char in str(abs(int(number)))
    )


def normalize_draw(issue, numbers):
    numbers = [int(x) for x in numbers]

    if len(numbers) != 7:
        raise ValueError("每期开奖必须有7个号码")

    if any(x < 1 or x > 49 for x in numbers):
        raise ValueError("号码必须位于1～49")

    if len(set(numbers)) != 7:
        raise ValueError("同期开奖存在重复号码")

    zodiacs = [
        get_zodiac(number)
        for number in numbers
    ]

    return {
        "issue": str(issue),
        "nums": numbers,
        "zodiacs": zodiacs,
        "zodiacs_list": zodiacs,
        "sp": zodiacs[-1],
    }


def random_draw(issue):
    return normalize_draw(
        issue,
        random.sample(range(1, 50), 7)
    )


def build_base_probability():
    counts = Counter(
        get_zodiac(number)
        for number in range(1, 50)
    )

    return {
        zodiac: counts[zodiac] / 49.0
        for zodiac in ZODIACS
    }


BASE_PROBABILITY = build_base_probability()

DEFAULT_ZODIAC = max(
    ZODIACS,
    key=lambda z: BASE_PROBABILITY[z]
)


# ============================================================
# 基础算法
# ============================================================

class Strategies:

    @staticmethod
    def base_rate(history):
        return DEFAULT_ZODIAC

    @staticmethod
    def hot(history, window=20):
        if not history:
            return None

        values = [
            draw["sp"]
            for draw in history[-window:]
        ]

        return Counter(values).most_common(1)[0][0]

    @staticmethod
    def cold(history, window=40):
        if not history:
            return None

        counts = Counter(
            draw["sp"]
            for draw in history[-window:]
        )

        return min(
            ZODIACS,
            key=lambda z: counts[z]
        )

    @staticmethod
    def ema(history, span=30):
        if not history:
            return None

        alpha = 2.0 / (span + 1)
        scores = dict.fromkeys(ZODIACS, 0.0)

        for draw in history[-span:]:
            for zodiac in ZODIACS:
                scores[zodiac] *= 1.0 - alpha

            scores[draw["sp"]] += alpha

        return max(scores, key=scores.get)

    @staticmethod
    def omission(history):
        if not history:
            return None

        positions = dict.fromkeys(ZODIACS, -1)

        for index, draw in enumerate(history):
            positions[draw["sp"]] = index

        return min(
            ZODIACS,
            key=lambda z: positions[z]
        )

    @staticmethod
    def markov1(history):
        if len(history) < 30:
            return None

        current = history[-1]["sp"]
        counts = Counter({z: 1 for z in ZODIACS})

        for previous, following in zip(
            history,
            history[1:]
        ):
            if previous["sp"] == current:
                counts[following["sp"]] += 1

        return max(
            ZODIACS,
            key=lambda z: counts[z]
        )

    @staticmethod
    def markov2(history):
        if len(history) < 200:
            return None

        pattern = (
            history[-2]["sp"],
            history[-1]["sp"]
        )

        counts = Counter({z: 1 for z in ZODIACS})

        for index in range(2, len(history)):
            old_pattern = (
                history[index - 2]["sp"],
                history[index - 1]["sp"]
            )

            if old_pattern == pattern:
                counts[history[index]["sp"]] += 1

        return max(
            ZODIACS,
            key=lambda z: counts[z]
        )

    @staticmethod
    def ngram(history, n=2):
        if len(history) < 100:
            return None

        pattern = tuple(
            draw["sp"]
            for draw in history[-n:]
        )

        counts = Counter()

        for index in range(n, len(history)):
            old_pattern = tuple(
                history[index - n + offset]["sp"]
                for offset in range(n)
            )

            if old_pattern == pattern:
                counts[history[index]["sp"]] += 1

        if not counts:
            return None

        return counts.most_common(1)[0][0]

    @staticmethod
    def regular_sum(history, mode="wrap49"):
        if not history:
            return None

        return map_value(
            sum(regular_numbers(history[-1])),
            mode
        )

    @staticmethod
    def all_sum(history, mode="wrap49"):
        if not history:
            return None

        return map_value(
            sum(history[-1]["nums"]),
            mode
        )

    @staticmethod
    def special_sum(history, window=2, mode="wrap49"):
        if len(history) < window:
            return None

        value = sum(
            special_number(draw)
            for draw in history[-window:]
        )

        return map_value(value, mode)

    @staticmethod
    def special_offset(
        history,
        offset=1,
        lag=1,
        mode="wrap49"
    ):
        if len(history) < lag:
            return None

        value = (
            special_number(history[-lag]) +
            offset
        )

        return map_value(value, mode)

    @staticmethod
    def position_offset(
        history,
        position=0,
        offset=1,
        mode="wrap49"
    ):
        if not history:
            return None

        value = (
            history[-1]["nums"][position] +
            offset
        )

        return map_value(value, mode)

    @staticmethod
    def position_plus_position(
        history,
        position=0,
        mode="wrap49"
    ):
        if not history:
            return None

        value = (
            history[-1]["nums"][position] +
            position + 1
        )

        return map_value(value, mode)

    @staticmethod
    def position_special(
        history,
        position=0,
        operation="add",
        mode="wrap49"
    ):
        if not history:
            return None

        regular = history[-1]["nums"][position]
        special = special_number(history[-1])

        operations = {
            "add": regular + special,
            "sub": regular - special,
            "reverse_sub": special - regular,
            "abs_diff": abs(regular - special),
        }

        if operation not in operations:
            return None

        return map_value(
            operations[operation],
            mode
        )

    @staticmethod
    def position_pair(
        history,
        first=0,
        second=1,
        operation="add",
        mode="wrap49"
    ):
        if not history:
            return None

        a = history[-1]["nums"][first]
        b = history[-1]["nums"][second]

        if operation == "add":
            value = a + b
        elif operation == "abs_diff":
            value = abs(a - b)
        else:
            return None

        return map_value(value, mode)

    @staticmethod
    def special_cross(
        history,
        operation="sum2",
        mode="wrap49"
    ):
        if len(history) < 3:
            return None

        t1 = special_number(history[-1])
        t2 = special_number(history[-2])
        t3 = special_number(history[-3])

        values = {
            "sum2": t1 + t2,
            "diff2": t1 - t2,
            "abs_diff2": abs(t1 - t2),
            "sum3": t1 + t2 + t3,
            "weighted": 3 * t1 + 2 * t2 + t3,
            "tail_sum": t1 % 10 + t2 % 10,
            "digit_sum": digit_sum(t1) + digit_sum(t2),
        }

        if operation not in values:
            return None

        return map_value(
            values[operation],
            mode
        )

    @staticmethod
    def liuhe(history):
        if not history:
            return None

        mapping = {
            "鼠": "牛", "牛": "鼠",
            "虎": "猪", "猪": "虎",
            "兔": "狗", "狗": "兔",
            "龙": "鸡", "鸡": "龙",
            "蛇": "猴", "猴": "蛇",
            "马": "羊", "羊": "马",
        }

        return mapping.get(history[-1]["sp"])

    @staticmethod
    def sanhe(history):
        if len(history) < 10:
            return None

        mapping = {
            "鼠": ("龙", "猴"),
            "牛": ("蛇", "鸡"),
            "虎": ("马", "狗"),
            "兔": ("羊", "猪"),
            "龙": ("鼠", "猴"),
            "蛇": ("牛", "鸡"),
            "马": ("虎", "狗"),
            "羊": ("兔", "猪"),
            "猴": ("鼠", "龙"),
            "鸡": ("牛", "蛇"),
            "狗": ("虎", "马"),
            "猪": ("兔", "羊"),
        }

        candidates = mapping.get(history[-1]["sp"])

        if not candidates:
            return None

        recent = [
            draw["sp"]
            for draw in history[-20:]
        ]

        return max(
            candidates,
            key=lambda z: recent.count(z)
        )


# ============================================================
# 生成公式库
# ============================================================

def build_strategies():
    result = {
        "基础概率": Strategies.base_rate,
        "热20": partial(Strategies.hot, window=20),
        "热60": partial(Strategies.hot, window=60),
        "冷40": partial(Strategies.cold, window=40),
        "EMA20": partial(Strategies.ema, span=20),
        "EMA60": partial(Strategies.ema, span=60),
        "最长遗漏": Strategies.omission,
        "马尔可夫1": Strategies.markov1,
        "马尔可夫2": Strategies.markov2,
        "NGram2": partial(Strategies.ngram, n=2),
        "NGram3": partial(Strategies.ngram, n=3),
        "六合": Strategies.liuhe,
        "三合": Strategies.sanhe,
    }

    modes = (
        ("循49", "wrap49"),
        ("余12", "mod12")
    )

    positions = (
        "平1", "平2", "平3",
        "平4", "平5", "平6", "特码"
    )

    for mode_name, mode in modes:
        result[f"六平和-{mode_name}"] = partial(
            Strategies.regular_sum,
            mode=mode
        )

        result[f"七码和-{mode_name}"] = partial(
            Strategies.all_sum,
            mode=mode
        )

        for window in (2, 3, 5, 8):
            result[f"特和{window}-{mode_name}"] = partial(
                Strategies.special_sum,
                window=window,
                mode=mode
            )

        # 平1～平6和特码偏移
        for position in range(7):
            max_offset = 12 if position == 6 else 6

            for offset in range(
                -max_offset,
                max_offset + 1
            ):
                if offset == 0:
                    continue

                name = (
                    f"{positions[position]}"
                    f"{offset:+d}-{mode_name}"
                )

                result[name] = partial(
                    Strategies.position_offset,
                    position=position,
                    offset=offset,
                    mode=mode
                )

        # 上上期、上三期的特码偏移
        for lag in (2, 3):
            for offset in (-8, -5, -2, -1, 1, 2, 5, 8):
                name = (
                    f"特L{lag}{offset:+d}-"
                    f"{mode_name}"
                )

                result[name] = partial(
                    Strategies.special_offset,
                    offset=offset,
                    lag=lag,
                    mode=mode
                )

        # 平N+N
        for position in range(6):
            result[
                f"平{position + 1}+"
                f"{position + 1}-{mode_name}"
            ] = partial(
                Strategies.position_plus_position,
                position=position,
                mode=mode
            )

        # 平码和特码组合
        operation_names = {
            "add": "+特",
            "sub": "-特",
            "reverse_sub": "特-平",
            "abs_diff": "差特",
        }

        for position in range(6):
            for operation, label in operation_names.items():
                result[
                    f"平{position + 1}"
                    f"{label}-{mode_name}"
                ] = partial(
                    Strategies.position_special,
                    position=position,
                    operation=operation,
                    mode=mode
                )

        # 平码两两组合
        for first in range(6):
            for second in range(first + 1, 6):
                result[
                    f"平{first + 1}+平"
                    f"{second + 1}-{mode_name}"
                ] = partial(
                    Strategies.position_pair,
                    first=first,
                    second=second,
                    operation="add",
                    mode=mode
                )

                result[
                    f"平{first + 1}差平"
                    f"{second + 1}-{mode_name}"
                ] = partial(
                    Strategies.position_pair,
                    first=first,
                    second=second,
                    operation="abs_diff",
                    mode=mode
                )

        cross_operations = {
            "sum2": "特1+特2",
            "diff2": "特1-特2",
            "abs_diff2": "两特差",
            "sum3": "三特和",
            "weighted": "三特加权",
            "tail_sum": "两特尾和",
            "digit_sum": "两特合和",
        }

        for operation, label in cross_operations.items():
            result[f"{label}-{mode_name}"] = partial(
                Strategies.special_cross,
                operation=operation,
                mode=mode
            )

    return result


# ============================================================
# 公式状态
# ============================================================

class FormulaState:

    def __init__(self):
        self.hits = deque(maxlen=500)
        self.expected = deque(maxlen=500)
        self.win_streak = 0
        self.loss_streak = 0

    def update(self, hit, expected):
        hit = int(hit)

        self.hits.append(hit)
        self.expected.append(float(expected))

        if hit:
            self.win_streak += 1
            self.loss_streak = 0
        else:
            self.win_streak = 0
            self.loss_streak += 1

    def accuracy(self, window=None):
        values = list(self.hits)

        if window is not None:
            values = values[-window:]

        return (
            sum(values) / len(values)
            if values else 0.0
        )

    def baseline(self, window=None):
        values = list(self.expected)

        if window is not None:
            values = values[-window:]

        return (
            sum(values) / len(values)
            if values else 0.0
        )

    def edge(self, window=None):
        return (
            self.accuracy(window) -
            self.baseline(window)
        )

    def trend(self):
        return self.edge(20) - self.edge(200)

    def drift(self):
        if len(self.hits) < 120:
            return False

        values = np.asarray(
            list(self.hits),
            dtype=float
        )

        previous = values[-120:-40]
        recent = values[-40:]

        pooled = (
            previous.sum() + recent.sum()
        ) / 120.0

        standard_error = math.sqrt(
            max(
                pooled * (1 - pooled) *
                (
                    1 / len(previous) +
                    1 / len(recent)
                ),
                1e-9
            )
        )

        return (
            abs(previous.mean() - recent.mean())
            > 2 * standard_error + 0.02
        )

    def score(self):
        if len(self.hits) < 20:
            return 0.0

        value = (
            0.45 * self.edge(20) +
            0.30 * self.edge(50) +
            0.15 * self.edge(200) +
            0.10 * self.edge()
        )

        value *= min(
            1.0,
            len(self.hits) / 100.0
        )

        if self.drift():
            value *= 0.1

        return value


class FormulaBoard:

    def __init__(self):
        self.states = defaultdict(FormulaState)

    def update(self, name, prediction, actual):
        if prediction is None:
            return

        self.states[name].update(
            prediction == actual,
            BASE_PROBABILITY[prediction]
        )

    def score(self, name):
        return self.states[name].score()

    def drift_fraction(self, names):
        valid = [
            self.states[name]
            for name in names
            if len(self.states[name].hits) >= 120
        ]

        if not valid:
            return 0.0

        return (
            sum(state.drift() for state in valid)
            / len(valid)
        )


# ============================================================
# AI元模型
# ============================================================

class MetaAI:

    def __init__(self):
        self.scaler = StandardScaler()

        self.online_model = SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=0.003,
            l1_ratio=0.05,
            average=True,
            random_state=42
        )

        self.batch_model = None
        self.online_ready = False
        self.batch_ready = False
        self.train_count = 0

        self.batch_x = deque(
            maxlen=BATCH_MAX_SAMPLES
        )

        self.batch_y = deque(
            maxlen=BATCH_MAX_SAMPLES
        )

    def features(self, history, predictions, board):
        vote = np.zeros(12)
        weighted = np.zeros(12)
        hashed = np.zeros(HASH_SIZE)

        available = 0

        for name, prediction in predictions.items():
            if prediction is None:
                continue

            available += 1
            zodiac_id = ZODIAC_TO_ID[prediction]
            score = board.score(name)

            vote[zodiac_id] += 1
            weighted[zodiac_id] += max(0.0, score)

            token = f"{name}|{prediction}".encode("utf-8")
            hash_value = zlib.crc32(token)
            index = hash_value % HASH_SIZE

            hashed[index] += (
                0.01 + max(0.0, score)
            )

        if available:
            vote /= available

        if weighted.sum() > 0:
            weighted /= weighted.sum()

        features = list(vote)
        features.extend(weighted)
        features.extend(hashed)

        # 最近20期生肖
        for lag in range(1, 21):
            one_hot = np.zeros(12)

            if len(history) >= lag:
                zodiac = history[-lag]["sp"]
                one_hot[
                    ZODIAC_TO_ID[zodiac]
                ] = 1.0

            features.extend(one_hot)

        # 最近8期球位
        for lag in range(1, 9):
            if len(history) >= lag:
                features.extend([
                    number / 49.0
                    for number in history[-lag]["nums"]
                ])
            else:
                features.extend([0.0] * 7)

        # 多窗口频率
        for window in (20, 50, 200):
            frequency = np.zeros(12)
            recent = history[-window:]

            if recent:
                for draw in recent:
                    frequency[
                        ZODIAC_TO_ID[draw["sp"]]
                    ] += 1

                frequency /= len(recent)

            features.extend(frequency)

        features.append(
            board.drift_fraction(
                list(predictions)
            )
        )

        return np.asarray(
            features,
            dtype=np.float64
        )

    @staticmethod
    def normalize_classes(classes, raw):
        result = np.zeros(12)

        for class_id, probability in zip(
            classes,
            raw
        ):
            result[int(class_id)] = probability

        if result.sum() <= 0:
            result[:] = 1 / 12
        else:
            result /= result.sum()

        return result

    def predict_proba(self, x):
        if (
            not self.online_ready or
            self.train_count < AI_MIN_SAMPLES
        ):
            return None

        x2 = x.reshape(1, -1)
        scaled = self.scaler.transform(x2)

        raw = self.online_model.predict_proba(
            scaled
        )[0]

        online_probability = self.normalize_classes(
            self.online_model.classes_,
            raw
        )

        if self.batch_ready:
            batch_raw = self.batch_model.predict_proba(
                x2
            )[0]

            batch_probability = self.normalize_classes(
                self.batch_model.classes_,
                batch_raw
            )

            probability = (
                0.6 * online_probability +
                0.4 * batch_probability
            )
        else:
            probability = online_probability

        base = np.asarray([
            BASE_PROBABILITY[zodiac]
            for zodiac in ZODIACS
        ])

        base /= base.sum()

        reliability = min(
            0.85,
            self.train_count / 5000
        )

        probability = (
            reliability * probability +
            (1 - reliability) * base
        )

        return probability / probability.sum()

    def update(self, x, actual):
        x2 = x.reshape(1, -1)
        y = np.asarray([
            ZODIAC_TO_ID[actual]
        ])

        self.scaler.partial_fit(x2)
        scaled = self.scaler.transform(x2)

        if not self.online_ready:
            self.online_model.partial_fit(
                scaled,
                y,
                classes=np.arange(12)
            )

            self.online_ready = True
        else:
            self.online_model.partial_fit(
                scaled,
                y
            )

        self.batch_x.append(
            x.astype(np.float32)
        )

        self.batch_y.append(int(y[0]))
        self.train_count += 1

        if (
            len(self.batch_x) >= BATCH_MIN_SAMPLES and
            self.train_count %
            BATCH_RETRAIN_INTERVAL == 0
        ):
            self.retrain_batch()

    def retrain_batch(self):
        x = np.asarray(
            self.batch_x,
            dtype=np.float32
        )

        y = np.asarray(
            self.batch_y,
            dtype=np.int32
        )

        if len(np.unique(y)) < 2:
            return

        model = HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=120,
            max_leaf_nodes=15,
            max_depth=5,
            min_samples_leaf=30,
            l2_regularization=3.0,
            random_state=42
        )

        model.fit(x, y)

        self.batch_model = model
        self.batch_ready = True


# ============================================================
# 主引擎
# ============================================================

class BBEngine:

    def __init__(self):
        self.strategies = build_strategies()
        self.board = FormulaBoard()
        self.ai = MetaAI()

        self.history = []
        self.balance = START_CAPITAL
        self.logs = deque(maxlen=100)

        self.total = 0
        self.hits = 0

    def run_formulas(self):
        predictions = {}

        for name, function in self.strategies.items():
            try:
                prediction = function(self.history)

                predictions[name] = (
                    prediction
                    if prediction in ZODIACS
                    else None
                )
            except Exception:
                predictions[name] = None

        return predictions

    def formula_probability(self, predictions):
        base = np.asarray([
            BASE_PROBABILITY[zodiac]
            for zodiac in ZODIACS
        ])

        votes = np.zeros(12)

        for name, prediction in predictions.items():
            if prediction is None:
                continue

            score = self.board.score(name)

            if score > 0:
                votes[
                    ZODIAC_TO_ID[prediction]
                ] += score

        if votes.sum() <= 0:
            return base / base.sum()

        votes /= votes.sum()

        result = (
            0.25 * base / base.sum() +
            0.75 * votes
        )

        return result / result.sum()

    def process_draw(self, draw):
        if len(self.history) < 3:
            self.history.append(draw)
            return None

        predictions = self.run_formulas()

        meta_x = self.ai.features(
            self.history,
            predictions,
            self.board
        )

        formula_p = self.formula_probability(
            predictions
        )

        ai_p = self.ai.predict_proba(meta_x)

        if ai_p is None:
            final_p = formula_p
            ai_active = False
        else:
            ai_weight = min(
                0.7,
                0.2 + self.ai.train_count / 10000
            )

            final_p = (
                (1 - ai_weight) * formula_p +
                ai_weight * ai_p
            )

            final_p /= final_p.sum()
            ai_active = True

        ranking = np.argsort(final_p)[::-1]

        best_id = int(ranking[0])
        second_id = int(ranking[1])

        prediction = ID_TO_ZODIAC[best_id]
        actual = draw["sp"]
        hit = prediction == actual

        drift_fraction = self.board.drift_fraction(
            list(predictions)
        )

        decision = {
            "prediction": prediction,
            "probability": float(final_p[best_id]),
            "second": ID_TO_ZODIAC[second_id],
            "second_probability": float(
                final_p[second_id]
            ),
            "margin": float(
                final_p[best_id] -
                final_p[second_id]
            ),
            "drift_fraction": drift_fraction,
            "ai_active": ai_active,
            "hit": hit,
        }

        # 开奖后更新，防止未来泄漏
        for name, formula_prediction in predictions.items():
            self.board.update(
                name,
                formula_prediction,
                actual
            )

        self.ai.update(meta_x, actual)

        self.history.append(draw)
        self.history = self.history[-MAX_HISTORY:]

        self.total += 1
        self.hits += int(hit)

        self.logs.append({
            "issue": draw["issue"],
            "prediction": prediction,
            "actual": actual,
            "hit": hit,
            "probability": decision["probability"],
        })

        return decision

    def top_formulas(self, limit=10):
        names = sorted(
            self.strategies,
            key=self.board.score,
            reverse=True
        )[:limit]

        return [
            (
                name,
                self.board.states[name]
            )
            for name in names
        ]

    def save(self, path=MODEL_PATH):
        directory = os.path.dirname(path)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True
            )

        temporary = path + ".tmp"

        joblib.dump(
            {
                "history": self.history,
                "board": self.board,
                "ai": self.ai,
                "balance": self.balance,
                "logs": self.logs,
                "total": self.total,
                "hits": self.hits,
                "strategy_names": list(
                    self.strategies
                ),
            },
            temporary,
            compress=3
        )

        os.replace(temporary, path)

    @classmethod
    def load(cls, path=MODEL_PATH):
        engine = cls()

        if not os.path.exists(path):
            return engine

        try:
            state = joblib.load(path)

            engine.history = state.get(
                "history",
                []
            )[-MAX_HISTORY:]

            engine.board = state.get(
                "board",
                FormulaBoard()
            )

            engine.ai = state.get(
                "ai",
                MetaAI()
            )

            engine.balance = state.get(
                "balance",
                START_CAPITAL
            )

            engine.logs = state.get(
                "logs",
                deque(maxlen=100)
            )

            engine.total = state.get(
                "total",
                0
            )

            engine.hits = state.get(
                "hits",
                0
            )

            print(
                f"模型已加载：{path}，"
                f"历史={len(engine.history)}，"
                f"AI样本={engine.ai.train_count}"
            )

        except Exception as error:
            print(f"模型加载失败：{error}")

        return engine
