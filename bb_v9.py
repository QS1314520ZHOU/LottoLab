# -*- coding: utf-8 -*-
"""
bb_v9.py — 基于公开调研重写的 BB六合彩 选肖算法（全新设计）

玩法模型（来源：平台截图 + 公开报道 + 本地开奖数据）：
  * 高频开奖：每2分钟一期，每天约720期（与香港六合彩每周3次完全不同）
  * 49个号码开7个（正码6 + 特别号1），12生肖：11个4号肖 + 马1个5号肖
  * 一肖（全开）：7个开奖号含所选肖任一号码即中
      4号肖 赔率2.06 → 保本命中率48.54%，理论命中47.17%（平台抽水≈3.3%/注）
      马(5号肖) 赔率1.76 → 保本命中率56.82%，理论命中55.39%（抽水≈2.7%/注）

算法设计（依据2025年央视/福彩中心/网安专家公开结论：
  "每期开奖是独立随机事件，基于历史数据的预测无法命中"——
  因此本算法不假装能预测号码，而是做三件事）：
  1. 检测层：滚动统计检验（生肖均匀性、序列独立性）。开奖越"正常"，
     越没有可利用的规律；只有显著异常才提示"可能存在错价"。
  2. 候选层：统计族（热/冷/位移）+ AI族（mimo-v2.6-flash 生成候选与理由）。
     所有候选平权，AI只是一票，不迷信。
  3. 验证门：候选必须在训练窗口命中率显著超过本赔率的保本线
     （单尾二项检验，Bonferroni校正）才允许下注；否则算法输出"不下注"。
  4. 资金层：过门候选平注5元，本金200，止损开关。

用法：
  python bb_v9.py check                     # 检测层：这组开奖正常吗
  python bb_v9.py predict                   # 实盘：给本期建议（含AI候选）
  python bb_v9.py backtest                  # 回测：验证门+资金曲线三问
  python bb_v9.py backtest --ai 60          # 回测：追加评估AI候选（最后60期）
"""

import json
import math
import os
import random
import sys
import urllib.error
import urllib.request

# ---------- 玩法模型 ----------
BANKROLL0 = 200.0
STAKE = 5.0
ODDS4, ODDS_MA = 2.06, 1.76
CSV_DEFAULT = "data/bb_live_history.csv"

# 号码→生肖（n % 12）
MOD2Z = {1: "马", 2: "蛇", 3: "龙", 4: "兔", 5: "虎", 6: "牛",
         7: "鼠", 8: "猪", 9: "狗", 10: "鸡", 11: "猴", 0: "羊"}
ZODIACS = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]


def zodiac(n: int) -> str:
    return MOD2Z[n % 12]


def zsize(z: str) -> int:
    return 5 if z == "马" else 4


def odds_of(z: str) -> float:
    return ODDS_MA if z == "马" else ODDS4


def p_hit(z: str) -> float:
    """全开一肖理论命中率：1 - C(49-k,7)/C(49,7)。"""
    k = zsize(z)
    p_miss = 1.0
    for i in range(7):
        p_miss *= (49 - k - i) / (49 - i)
    return 1.0 - p_miss


# ---------- 数据 ----------
class Draw:
    __slots__ = ("issue", "nums", "sp")

    def __init__(self, issue, nums, sp):
        self.issue = issue
        self.nums = nums        # 7个号码
        self.sp = sp            # 特别号

    @property
    def sp_zodiac(self) -> str:
        return zodiac(self.sp)

    def hit(self, z: str) -> bool:
        return any(zodiac(n) == z for n in self.nums)


def load(path=CSV_DEFAULT):
    draws = []
    with open(path, encoding="utf-8-sig") as f:
        header = f.readline().strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            d = dict(zip(header, parts))
            nums = [int(d[f"p{i}"]) for i in range(1, 7)]
            sp = int(d["sp"])
            draws.append(Draw(d["issue"], nums + [sp], sp))
    draws.sort(key=lambda x: x.issue)
    return draws


# ---------- 检测层 ----------
def chi2_uniform(draws):
    """特码生肖12类频次卡方均匀性检验。"""
    counts = {z: 0 for z in ZODIACS}
    for d in draws:
        counts[d.sp_zodiac] += 1
    n = len(draws)
    exp = n / 12.0
    chi2 = sum((c - exp) ** 2 / exp for c in counts.values())
    # df=11 卡方生存函数（正则化上不完全伽马）
    p = _chi2_sf(chi2, 11)
    return chi2, p, counts


def _chi2_sf(x, k):
    """卡方分布上尾概率。"""
    return _gammainc_upper(k / 2.0, x / 2.0)


def _gammainc_upper(a, x):
    """正则化上不完全伽马 Q(a,x)（级数/连分数）。"""
    if x < a + 1.0:
        return 1.0 - _gammainc_lower_series(a, x)
    return _gammainc_cf(a, x)


def _gammainc_lower_series(a, x):
    term = 1.0 / a
    total = term
    n = a
    for _ in range(500):
        n += 1
        term *= x / n
        total += term
        if abs(term) < abs(total) * 1e-14:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gammainc_cf(a, x):
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def runs_independence(draws):
    """特码生肖序列的一致性(游程)思想检验：转移是否偏离独立。
    统计量 = 与上期同类的次数，期望 n*p̄，用正态近似双侧p。"""
    same = sum(1 for i in range(1, len(draws))
               if draws[i].sp_zodiac == draws[i - 1].sp_zodiac)
    n = len(draws) - 1
    # 独立时 P(同) = Σ p_z^2，用经验频次估计
    counts = {z: 0 for z in ZODIACS}
    for d in draws:
        counts[d.sp_zodiac] += 1
    p2 = sum((c / len(draws)) ** 2 for c in counts.values())
    mean = n * p2
    var = n * p2 * (1 - p2)
    if var <= 0:
        return same, mean, 1.0
    z = (same - mean) / math.sqrt(var)
    p = 2 * (1 - _norm_cdf(abs(z)))
    return same, mean, p


def _norm_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def detect_report(draws, window=100):
    tail = draws[-window:]
    chi2, p1, counts = chi2_uniform(tail)
    same, mean, p2 = runs_independence(tail)
    print(f"[检测层] 最近{len(tail)}期")
    print(f"  特肖频次卡方: chi2={chi2:.2f} df=11 p={p1:.3f} "
          f"-> {'异常，可能存在错价' if p1 < 0.01 else '均匀（无错价信号）'}")
    print(f"  与上期同肖次数: {same} (独立期望≈{mean:.1f}) p={p2:.3f} "
          f"-> {'异常' if p2 < 0.01 else '独立（无记忆）'}")
    print(f"  频次明细: " + " ".join(f"{z}{counts[z]}" for z in ZODIACS))


# ---------- 候选层 ----------
def cand_hot(draws, w=15):
    cnt = {z: 0 for z in ZODIACS}
    for d in draws[-w:]:
        cnt[d.sp_zodiac] += 1
    best = max(cnt.values())
    return [z for z in ZODIACS if cnt[z] == best][0]


def cand_cold(draws, w=30):
    cnt = {z: 0 for z in ZODIACS}
    for d in draws[-w:]:
        cnt[d.sp_zodiac] += 1
    best = min(cnt.values())
    for z in ZODIACS:
        if cnt[z] == best:
            return z
    return "鼠"


def cand_offset(draws, delta):
    prev = draws[-1].sp
    return zodiac((prev + delta - 1) % 49 + 1)


STAT_CANDS = [
    ("热15特肖", lambda d: cand_hot(d, 15)),
    ("冷30特肖", lambda d: cand_cold(d, 30)),
    ("特码+1", lambda d: cand_offset(d, +1)),
    ("特码-1", lambda d: cand_offset(d, -1)),
]


# ---------- AI族（mimo-v2.6-flash） ----------
AI_BASE = os.getenv("MIMO_AI_BASE", "https://token-plan-cn.xiaomimimo.com/v1")
AI_MODEL = os.getenv("MIMO_AI_MODEL", "mimo-v2.6-flash")
AI_KEY = os.getenv("MIMO_AI_KEY", "")   # 凭据只走环境变量，禁止硬编码


def ai_pick(draws, recent=30):
    """让 mimo-v2.6-flash 基于最近N期生成一个候选生肖。
    只是候选生成器之一，必须过验证门才可下注。"""
    lines = []
    for d in draws[-recent:]:
        lines.append(f"{d.issue}: {','.join(f'{n:02d}' for n in d.nums)} "
                     f"特肖={d.sp_zodiac}")
    prompt = (
        "你是彩票数据分析助手。下面是BB高频六合彩最近几期开奖（49选7，"
        "7个号码=6正码+1特码，12生肖对应号码mod12）。请分析冷热/位移/分布，"
        "给出【下一期最可能出现于7个开奖号中的一个生肖】。"
        "注意：每期独立随机，你的输出只是待检验的假设。"
        "简短思考即可，不要长篇推理。\n"
        "严格只输出JSON：{\"zodiac\":\"生肖名\",\"reason\":\"20字内理由\"}\n\n"
        + "\n".join(lines)
    )
    body = json.dumps({
        "model": AI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4000,
    }).encode()
    req = urllib.request.Request(
        AI_BASE + "/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + AI_KEY})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
        content = data["choices"][0]["message"]["content"] or ""
        start, end = content.find("{"), content.rfind("}")
        obj = json.loads(content[start:end + 1])
        z = obj.get("zodiac", "")
        if z in ZODIACS:
            return z, obj.get("reason", "")
        return None, content[:80]
    except (urllib.error.URLError, ValueError, KeyError, IndexError, TimeoutError) as e:
        return None, f"AI调用失败: {e}"


# ---------- 验证门 ----------
def gate(cand_fn, train_slice, alpha=0.05, n_cands=5):
    """训练窗口样本外评分：命中数 vs 保本所需，单尾正态近似，
    Bonferroni校正 alpha/n_cands。返回 (过门?, 命中率, z, 需要值)。"""
    wins = 0
    need = 0.0
    var = 0.0
    n = 0
    for t in train_slice:
        d = draws_all[t]
        prev = draws_all[:t]
        z = cand_fn(prev)
        o = odds_of(z)
        h = d.hit(z)
        wins += int(h)
        need += 1.0 / o
        p = p_hit(z)
        var += p * (1 - p)
        n += 1
    if n == 0 or var == 0:
        return False, 0.0, 0.0, 0.0
    zscore = (wins - need) / math.sqrt(var)
    from statistics import NormalDist
    pval = 1 - NormalDist().cdf(zscore)
    passed = (zscore > 0) and (pval < alpha / n_cands)
    return passed, wins / n, zscore, need


def ai_gate(train_range, n_cands=6):
    """AI候选在训练窗的样本外评分（并发调用），与统计候选同标准过门。"""
    from concurrent.futures import ThreadPoolExecutor
    idxs = list(train_range)

    def _ask(t):
        z, _reason = ai_pick(draws_all[:t])
        return t, z

    picks = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        for t, z in ex.map(_ask, idxs):
            picks[t] = z
            print(f"  AI训练 {t - idxs[0] + 1}/{len(idxs)}", end="\r", flush=True)
    print()
    wins = need = var = 0.0
    used = fails = 0
    for t in idxs:
        z = picks[t]
        if z is None:
            fails += 1
            continue
        d = draws_all[t]
        wins += int(d.hit(z))
        need += 1.0 / odds_of(z)
        p = p_hit(z)
        var += p * (1 - p)
        used += 1
    if used == 0 or var == 0:
        return False, 0.0, 0.0, used, fails
    zscore = (wins - need) / math.sqrt(var)
    from statistics import NormalDist
    pval = 1 - NormalDist().cdf(zscore)
    passed = (zscore > 0) and (pval < 0.05 / n_cands)
    return passed, wins / used, zscore, used, fails


# ---------- 资金层（蒙特卡洛） ----------
def sim_bankroll(nets, n_paths=40000, seed=7, cap=5000):
    rng = random.Random(seed)
    alive = [BANKROLL0] * n_paths
    ruin_t = [0] * n_paths
    alive_n = list(range(n_paths))
    period = 0
    pool = [x for x in nets if x != 0.0] or [-STAKE, STAKE * (ODDS4 - 1)]
    while alive_n and period < cap:
        period += 1
        keep = []
        for i in alive_n:
            alive[i] += pool[rng.randrange(len(pool))]
            if alive[i] < STAKE:
                ruin_t[i] = period
            else:
                keep.append(i)
        alive_n = keep
    return ruin_t, alive


def session_winrate(nets, L, trials=20000, seed=11):
    """随机起点抽窗口，返回 P(净利>0) 与期望净利。"""
    rng = random.Random(seed)
    pool_idx = [i for i, x in enumerate(nets) if x != 0.0]
    if len(pool_idx) < L:
        return float("nan"), float("nan")
    win = 0
    total = 0.0
    for _ in range(trials):
        s = rng.randrange(len(pool_idx) - L + 1)
        seg = pool_idx[s:s + L]
        net = sum(nets[i] for i in seg)
        win += int(net > 0)
        total += net
    return win / trials, total / trials


# ---------- 主流程 ----------
draws_all = []


def build_nets(cand_fn, lo, hi):
    nets = []
    for t in range(lo, hi):
        if t == 0:
            nets.append(0.0)
            continue
        z = cand_fn(draws_all[:t])
        d = draws_all[t]
        nets.append(STAKE * (odds_of(z) - 1) if d.hit(z) else -STAKE)
    return nets


def cmd_check():
    detect_report(draws_all)


def cmd_predict():
    detect_report(draws_all)
    print("\n[候选层]")
    cands = []
    for name, fn in STAT_CANDS:
        z = fn(draws_all)
        cands.append((f"统计/{name}", z))
        print(f"  {name:<8} -> {z}  (赔率{odds_of(z)}, 理论命中{p_hit(z):.2%})")
    z, reason = ai_pick(draws_all)
    if z:
        cands.append(("AI/mimo-v2.6-flash", z))
        print(f"  AI推荐    -> {z}  理由: {reason}")
    else:
        print(f"  AI推荐    -> 不可用 ({reason})")

    print("\n[验证门] 训练窗=第60~150期, 测试前仅用训练窗判定")
    lo, hi = 60, 150
    any_pass = False
    for name, fn in [(n, f) for n, f in STAT_CANDS]:
        passed, rate, zsc, need = gate(fn, range(lo, hi))
        print(f"  {name:<8} 命中{rate:.1%} z={zsc:+.2f} -> {'过门✓' if passed else '不过门✗'}")
        any_pass |= passed
    if z:
        print("  AI候选    未经训练窗评分（新面孔），默认不过门，仅记录观察")
    if any_pass:
        print("\n[本期建议] 有过门候选，可按平注5元执行（仍为小概率事件，请控损）")
    else:
        print("\n[本期建议] 无候选通过验证门 → 算法输出：本期不下注")


def cmd_backtest(use_ai=0):
    n = len(draws_all)
    WARM = 50
    print("=" * 84)
    print(f"回测 {n}期 ｜ 评估窗 {WARM}..{n-1} ｜ 本金200 每注5 "
          f"｜ 4号肖@2.06 马@1.76")
    detect_report(draws_all)

    # 门判定窗 / 表现窗
    train = range(60, 150)
    test = range(150, n)

    print(f"\n{'候选':<12} {'训练窗命中':>9} {'z':>6} {'过门':>4} "
          f"{'测试窗命中':>9} {'EV/注':>7} {'赢10期':>7} {'赢20期':>7}")
    rows = []
    for name, fn in STAT_CANDS:
        passed, tr_rate, zsc, _ = gate(fn, train)
        nets = build_nets(fn, test.start, test.stop)
        played = [x for x in nets if x != 0.0]
        if not played:
            continue
        hits = sum(1 for x in played if x > 0)
        ev = sum(played) / len(played)
        w10, e10 = session_winrate(nets, 10)
        w20, e20 = session_winrate(nets, 20)
        rows.append((name, passed, nets, ev))
        print(f"{name:<12} {tr_rate:>9.1%} {zsc:>+6.2f} "
              f"{'✓' if passed else '✗':>4} {hits/len(played):>9.1%} "
              f"{ev:>+7.3f} {w10:>7.1%} {w20:>7.1%}")

    # 随机对照
    rng = random.Random(3)
    def rand_cand(prev):
        return ZODIACS[rng.randrange(12)]
    # 固定随机序列重建（保证可复现）
    rng2 = random.Random(3)
    picks = [ZODIACS[rng2.randrange(12)] for _ in range(n)]
    rand_nets = []
    for t in range(test.start, test.stop):
        z = picks[t]
        rand_nets.append(STAKE * (odds_of(z) - 1)
                         if draws_all[t].hit(z) else -STAKE)
    rp = [x for x in rand_nets if x != 0]
    w10r, _ = session_winrate(rand_nets, 10)
    w20r, _ = session_winrate(rand_nets, 20)
    print(f"{'随机对照':<12} {'--':>9} {'--':>6} {'✗':>4} "
          f"{sum(1 for x in rp if x > 0)/len(rp):>9.1%} "
          f"{sum(rp)/len(rp):>+7.3f} {w10r:>7.1%} {w20r:>7.1%}")

    # AI候选评估（可选，最后N期；并发6路加速）
    if use_ai > 0:
        from concurrent.futures import ThreadPoolExecutor
        print(f"\n[AI族] 第一步：训练窗(60..149)与统计候选同标准过门评分", flush=True)
        ai_pass, ai_rate, ai_z, ai_used, ai_fails = ai_gate(range(60, 150))
        print(f"  AI训练窗命中 {ai_rate:.1%} z={ai_z:+.2f} "
              f"(有效{ai_used}期, 失败{ai_fails}期) -> "
              f"{'过门✓' if ai_pass else '不过门✗'}")
        print(f"[AI族] 第二步：测试后段(最后{use_ai}期)表现观察", flush=True)
        lo = n - use_ai
        idxs = list(range(lo, n))

        def _ask(t):
            z, reason = ai_pick(draws_all[:t])
            return t, z, reason

        results = {}
        with ThreadPoolExecutor(max_workers=6) as ex:
            done = 0
            for t, z, reason in ex.map(_ask, idxs):
                results[t] = (z, reason)
                done += 1
                print(f"  进度 {done}/{len(idxs)}", flush=True)
        ai_nets = []
        wins = need = 0.0
        for t in idxs:
            z, reason = results[t]
            if z is None:
                ai_nets.append(0.0)
                continue
            h = draws_all[t].hit(z)
            wins += int(h)
            need += 1.0 / odds_of(z)
            ai_nets.append(STAKE * (odds_of(z) - 1) if h else -STAKE)
            print(f"  {draws_all[t].issue} -> {z} {'中' if h else '错'} "
                  f"| {reason}", flush=True)
        played = [x for x in ai_nets if x != 0]
        if played:
            print(f"  AI命中 {wins:.0f}/{len(played)} = {wins/len(played):.1%} "
                  f"(保本需{need:.1f}次) EV/注 {sum(played)/len(played):+.3f}")
            w10a, _ = session_winrate(ai_nets, 10)
            w20a, _ = session_winrate(ai_nets, 20)
            print(f"  AI赢10期 {w10a:.1%} ｜ AI赢20期 {w20a:.1%}")

    # 验证门整体（过门者集合；门空则不下注=不亏）
    passed_rows = [(nm, nk) for nm, ps, nk, ev in rows if ps]
    print("\n[资金层] 按验证门组合（过门候选轮流平注；全不过门=不下注）")
    if passed_rows:
        combo = [0.0] * (test.stop - test.start)
        for nm, nk in passed_rows:
            for i, x in enumerate(nk):
                if combo[i] == 0.0:
                    combo[i] = x
        w10, e10 = session_winrate(combo, 10)
        w20, e20 = session_winrate(combo, 20)
        ruin_t, _ = sim_bankroll(combo)
        busted = [x for x in ruin_t if x > 0]
        med = sorted(busted)[len(busted) // 2] if busted else -1
        p200 = sum(1 for x in ruin_t if 0 < x <= 200) / len(ruin_t)
        print(f"  过门策略: {[nm for nm, _ in passed_rows]}")
        print(f"  赢10期 {w10:.1%} ｜ 赢20期 {w20:.1%} ｜ 期望/10期 {e10:+.2f} "
              f"｜ 期望/20期 {e20:+.2f}")
        print(f"  中位爆仓 {med}期 ｜ P(≤200期爆仓) {p200:.2%}")
    else:
        print("  无候选过门 → 算法全程不下注 → 本金200永远是200（这就是最优解）")
    if use_ai > 0:
        print(f"  AI族(训练窗) -> {'过门✓，可纳入下注组合' if ai_pass else '不过门✗，仅作观察项'}")

    print("\n[结论] 参照央广网/福彩中心2025年权威结论：每期独立随机，"
          "历史规律不可预测；本算法的正确行为是——没有统计证据就不出手。")


if __name__ == "__main__":
    path = CSV_DEFAULT
    args = sys.argv[1:]
    use_ai = 0
    if "--ai" in args:
        i = args.index("--ai")
        use_ai = int(args[i + 1]) if i + 1 < len(args) else 60
    if "--csv" in args:
        i = args.index("--csv")
        path = args[i + 1]
    draws_all = load(path)
    cmd = args[0] if args and not args[0].startswith("--") else "backtest"
    if cmd == "check":
        cmd_check()
    elif cmd == "predict":
        cmd_predict()
    else:
        cmd_backtest(use_ai)
