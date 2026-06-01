"""
Imbalance Ratio 分析与评价
分析类别不平衡对关联规则指标的影响
"""
import pandas as pd
import numpy as np
from typing import Dict, FrozenSet, List, Tuple

from src.metrics import Rule, generate_rules, format_rules_table


def compute_imbalance_ratio(matrix: pd.DataFrame) -> float:
    """计算数据集的 Imbalance Ratio (IR) = 多数类计数 / 少数类计数"""
    # 检测是否有 y=yes / y=no 列
    target_cols = [c for c in matrix.columns if c.startswith("y=")]
    if not target_cols:
        return 1.0

    counts = {}
    for col in target_cols:
        counts[col] = matrix[col].sum()

    if len(counts) < 2:
        return 1.0

    max_count = max(counts.values())
    min_count = min(counts.values())
    ir = max_count / min_count if min_count > 0 else float('inf')
    return ir


def analyze_target_rules(rules: List[Rule], matrix: pd.DataFrame) -> pd.DataFrame:
    """
    分析目标变量 y=yes 和 y=no 规则的差异
    返回对比表格
    """
    yes_rules = [r for r in rules if "y=yes" in r.consequent]
    no_rules = [r for r in rules if "y=no" in r.consequent]

    # 按 Lift 排序取 TopN
    yes_rules.sort(key=lambda r: -r.lift)
    no_rules.sort(key=lambda r: -r.lift)

    n = len(matrix)
    p_yes = matrix["y=yes"].sum() / n if "y=yes" in matrix.columns else 0
    p_no = matrix["y=no"].sum() / n if "y=no" in matrix.columns else 0

    rows = []
    rows.append({
        "Metric": "Prior Probability P(target)",
        "y=yes": f"{p_yes:.4f}",
        "y=no": f"{p_no:.4f}",
    })
    rows.append({
        "Metric": "Number of Rules Found",
        "y=yes": str(len(yes_rules)),
        "y=no": str(len(no_rules)),
    })

    if yes_rules:
        rows.append({
            "Metric": "Top-1 Antecedent (by Lift)",
            "y=yes": ", ".join(sorted(yes_rules[0].antecedent)),
            "y=no": ", ".join(sorted(no_rules[0].antecedent)) if no_rules else "-",
        })
        rows.append({
            "Metric": "Top-1 Lift",
            "y=yes": f"{yes_rules[0].lift:.4f}",
            "y=no": f"{no_rules[0].lift:.4f}" if no_rules else "-",
        })
        rows.append({
            "Metric": "Top-1 Confidence",
            "y=yes": f"{yes_rules[0].confidence:.4f}",
            "y=no": f"{no_rules[0].confidence:.4f}" if no_rules else "-",
        })

    # 统计指标分布
    for rule_set, label in [(yes_rules, "y=yes"), (no_rules, "y=no")]:
        if rule_set:
            lifts = [r.lift for r in rule_set if r.lift != float('inf')]
            confs = [r.confidence for r in rule_set]
            chis = [r.chi_square for r in rule_set]
            rows.append({
                "Metric": f"Avg Lift ({label})",
                "y=yes": f"{np.mean(lifts):.4f}" if label == "y=yes" else "",
                "y=no": f"{np.mean(lifts):.4f}" if label == "y=no" else "",
            })
            rows.append({
                "Metric": f"Avg Confidence ({label})",
                "y=yes": f"{np.mean(confs):.4f}" if label == "y=yes" else "",
                "y=no": f"{np.mean(confs):.4f}" if label == "y=no" else "",
            })
            rows.append({
                "Metric": f"Max Chi-square ({label})",
                "y=yes": f"{np.max(chis):.2f}" if label == "y=yes" else "",
                "y=no": f"{np.max(chis):.2f}" if label == "y=no" else "",
            })

    return pd.DataFrame(rows)


def analyze_metric_behavior(rules: List[Rule]) -> pd.DataFrame:
    """
    分析不同指标在类别不平衡下的行为特点
    - 对比 Lift vs Confidence 排序差异
    - 分析 Chi-square 对稀有规则的敏感性
    """
    if not rules:
        return pd.DataFrame()

    rows = []
    # 按不同指标排序取 Top-5
    by_lift = sorted([r for r in rules if r.lift != float('inf')], key=lambda r: -r.lift)[:5]
    by_conf = sorted(rules, key=lambda r: -r.confidence)[:5]
    by_chi = sorted(rules, key=lambda r: -r.chi_square)[:5]
    by_lev = sorted(rules, key=lambda r: -r.leverage)[:5]

    analysis_data = []
    labels = ["Top-5 by Lift", "Top-5 by Confidence", "Top-5 by Chi-square", "Top-5 by Leverage"]
    for label, rule_list in [
        ("Top-5 by Lift", by_lift),
        ("Top-5 by Confidence", by_conf),
        ("Top-5 by Chi-square", by_chi),
        ("Top-5 by Leverage", by_lev),
    ]:
        for r in rule_list:
            analysis_data.append({
                "Ranking Metric": label,
                "Antecedent": ", ".join(sorted(r.antecedent)),
                "Consequent": ", ".join(sorted(r.consequent)),
                "Support": f"{r.support_AB:.4f}",
                "Confidence": f"{r.confidence:.4f}",
                "Lift": f"{r.lift:.4f}",
                "Chi-square": f"{r.chi_square:.2f}",
                "Leverage": f"{r.leverage:.4f}",
            })

    return pd.DataFrame(analysis_data)
