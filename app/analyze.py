# -*- coding: utf-8 -*-
"""analyze 命令：特征落库 + 规律评分卡 + 自动发现（含多重检验）。"""

from __future__ import annotations

from typing import Any

from . import db
from .backtest import _segments
from .config import SSQConfig
from .features import FEATURE_VERSION, build_features, save_features
from .patterns import (default_rules, discover_markov_cells,
                       discover_red_pairs, record_rules_to_db, score_all)


def run_analysis(cfg: SSQConfig, pattern_limit: int = 0) -> str:
    db.migrate(cfg.db_path)
    with db.connect(cfg.db_path) as conn:
        draws = db.load_draws(conn, status="ok")
    if len(draws) < cfg.min_train + 30:
        return f"数据不足：{len(draws)} 期"

    # 1) 特征
    feats = build_features(draws, cfg)
    saved = save_features(cfg, feats)

    # 2) 规律评分卡（dev 段探索，holdout 段确认）
    segs = _segments(len(draws), cfg.min_train)
    rules = default_rules(cfg)
    if pattern_limit:
        rules = rules[:pattern_limit]
    scores = score_all(rules, draws, segs["dev"].start,
                       segs["holdout"].stop, cfg,
                       holdout_start=segs["holdout"].start)
    record_rules_to_db(cfg, rules, scores)

    # 3) 自动发现（探索性）
    mk = discover_markov_cells(draws, cfg)
    pairs, n_pairs_tested = discover_red_pairs(draws, cfg)

    lines = ["=" * 78,
             f"特征落库 {saved} 期（版本 {FEATURE_VERSION}）",
             f"规律尝试总数（多重检验分母）= {len(scores)}",
             "",
             f"{'规律':<16} {'类':<12} {'n':>5} {'观测':>7} {'基线':>7} "
             f"{'差':>7} {'95%CI':>15} {'稳定':>6} {'p':>7} {'BH q':>7} {'状态':>8}"]
    for s in scores:
        ci = f"[{s.ci_lo:.1%},{s.ci_hi:.1%}]" if s.n_trigger else "-"
        stab = "-" if s.stability != s.stability else f"{s.stability:.2f}"
        lines.append(
            f"{s.name:<16} {s.category:<12} {s.n_trigger:>5} "
            f"{s.observed_p:>7.1%} {s.baseline_p:>7.1%} {s.diff:>+7.1%} "
            f"{ci:>15} {stab:>6} {s.p_value:>7.3f} {s.p_adjusted:>7.3f} "
            f"{s.status:>8}")
    lines += [
        "",
        "[自动发现] 蓝球Markov独立性："
        f"chi2={mk['chi2']:.1f} dof={mk['dof']} p={mk['p_independence']:.3f}"
        f"（{mk['n_transitions']} 转移，测试单元 {mk['n_cells_tested']}）",
        "  最高提升单元 top3："
        + "; ".join(f"{c['from']:02d}->{c['to']:02d} lift={c['lift']:.2f} "
                    f"q={c.get('p_adjusted', float('nan')):.2f}"
                    for c in mk["top_cells"][:3]),
        f"[自动发现] 红球对：测试 {n_pairs_tested} 对（min_support=30），top3："
        + "; ".join(f"{p['pair']} lift={p['lift']:.2f} q={p['p_adjusted']:.2f}"
                    for p in pairs[:3]),
        "",
        "标注：以上全部为【探索性分析】；仅 holdout 复评通过且 BH q<0.05 "
        "才标记 validated。研究分数 research_score 仅供排序，不是命中概率。",
    ]
    return "\n".join(lines)
