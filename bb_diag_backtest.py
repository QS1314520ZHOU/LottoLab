# -*- coding: utf-8 -*-
"""
bb_diag_backtest.py

带AI门控诊断的严格时间顺序回测：
- 指标口径与 run_backtest 完全一致（一肖/二连肖/三连肖 + 生肖LogLoss/Brier）
- 额外输出：AI样本数、各位置reliability、top1命中率、权重>0的调用次数

用法：
    BB_AI_MIN_SAMPLES=100 python bb_diag_backtest.py data/bb_live_history.csv "标签"
"""

import sys
import math
from collections import Counter

import bb_zodiac_v7 as bb


def main() -> None:
    csv_path, label = sys.argv[1], sys.argv[2]

    # 记录AI权重是否真的被reliability门控启用
    original_reliability = bb.MultiPositionAI.reliability
    weight_log = []

    def patched_reliability(self, position):
        value = original_reliability(self, position)
        if self.ready:
            weight_log.append((position, value))
        return value

    bb.MultiPositionAI.reliability = patched_reliability

    engine = bb.BBEngine()
    draws = bb.load_csv(csv_path)
    draws.sort(key=bb.issue_sort_key)

    warmup, evaluated = 30, 0
    one = two = three = 0
    base_one = base_two = base_three = 0.0
    log_loss = brier = 0.0

    for draw in draws:
        prediction = engine.process_draw(draw)
        if prediction is None:
            continue

        if engine.processed > warmup:
            market = prediction["markets"]["特别号"]["生肖"]
            ranked = bb.top_items(market, 3)
            z1 = [z for z, _ in ranked[:1]]
            z2 = [z for z, _ in ranked[:2]]
            z3 = [z for z, _ in ranked[:3]]
            actual = bb.get_zodiac(draw.special)

            one += int(actual in z1)
            two += int(actual in z2)
            three += int(actual in z3)
            base_one += sum(len(bb.zodiac_numbers(z)) / 49.0 for z in z1)
            base_two += sum(len(bb.zodiac_numbers(z)) / 49.0 for z in z2)
            base_three += sum(len(bb.zodiac_numbers(z)) / 49.0 for z in z3)
            log_loss += -math.log(max(1e-12, float(market[actual])))
            brier += sum(
                (float(market[z]) - (1.0 if z == actual else 0.0)) ** 2
                for z in bb.ZODIACS
            )
            evaluated += 1

    print(f"== {label} ==")
    print(
        f"eval={evaluated} | 一肖 {one}/{evaluated}={one / evaluated:.2%} "
        f"(基线{base_one / evaluated:.2%}) | 二连肖 {two / evaluated:.2%} "
        f"(基线{base_two / evaluated:.2%}) | 三连肖 {three / evaluated:.2%} "
        f"(基线{base_three / evaluated:.2%})"
    )
    print(f"logloss={log_loss / evaluated:.6f} | brier={brier / evaluated:.6f}")
    print(f"ai.samples={engine.ai.samples} ready={engine.ai.ready}")
    print("reliability/位置:", [round(engine.ai.reliability(i), 4) for i in range(7)])
    print("top1命中/位置:", [round(engine.ai.top1_rate(i), 4) for i in range(7)])
    nonzero = Counter(p for p, v in weight_log if v > 0)
    total = Counter(p for p, _ in weight_log)
    print("权重>0次数/位置:", dict(sorted(nonzero.items())),
          "| ready后总调用:", dict(sorted(total.items())))


if __name__ == "__main__":
    main()
