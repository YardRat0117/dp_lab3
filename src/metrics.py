"""
关联规则评估指标
计算: Support, Confidence, Lift, Chi-Square, Conviction, Leverage, Jaccard
"""
import math
import pandas as pd
import numpy as np
from typing import Dict, FrozenSet, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Rule:
    """关联规则"""
    antecedent: FrozenSet  # 前件 A
    consequent: FrozenSet  # 后件 B (单元素)
    support_A: float       # P(A)
    support_B: float       # P(B)
    support_AB: float      # P(A ∩ B)
    confidence: float      # P(B|A)
    lift: float            # P(AB) / (P(A)*P(B))
    chi_square: float      # (O-E)^2/E
    conviction: float      # (1-P(B)) / (1-P(B|A))
    leverage: float        # P(AB) - P(A)*P(B)
    jaccard: float         # P(AB) / P(A∪B)

    def __repr__(self):
        A = ", ".join(sorted(self.antecedent))
        B = ", ".join(sorted(self.consequent))
        return (f"{{{A}}} -> {{{B}}}  "
                f"sup={self.support_AB:.4f}  "
                f"conf={self.confidence:.4f}  "
                f"lift={self.lift:.4f}  "
                f"chi2={self.chi_square:.4f}  "
                f"conv={self.conviction:.4f}")


def _count_in_matrix(matrix: pd.DataFrame, itemset: FrozenSet) -> int:
    """统计 itemset 在 boolean matrix 中出现的次数"""
    if len(itemset) == 0:
        return len(matrix)
    cols = list(itemset)
    return (matrix[list(cols)].all(axis=1)).sum()


def generate_rules(
    frequent_itemsets: Dict[FrozenSet, float],
    matrix: pd.DataFrame,
    min_confidence: float = 0.3,
    consequent_target: Optional[str] = None,
) -> List[Rule]:
    """
    从频繁项集生成关联规则

    参数:
        frequent_itemsets: {frozenset: support_ratio}
        matrix: boolean 交易矩阵
        min_confidence: 最小置信度
        consequent_target: 若指定，只生成后件包含该 item 的规则

    返回:
        rules: List[Rule]
    """
    n = len(matrix)
    support_counts = {k: int(v * n) for k, v in frequent_itemsets.items()}
    rules = []

    for itemset, sup in frequent_itemsets.items():
        if len(itemset) < 2:
            continue

        itemset_count = support_counts[itemset]

        # 对每个 item 作为后件
        for item in itemset:
            # 处理 single-item consequent
            antecedent = frozenset(it for it in itemset if it != item)
            consequent = frozenset([item])

            # 如果指定了目标后件且不匹配则跳过
            if consequent_target and item != consequent_target:
                continue

            sup_A = frequent_itemsets.get(antecedent)
            sup_B = frequent_itemsets.get(consequent)

            if sup_A is None:
                # 需要计算 antecedent 的支持度
                sup_A = _count_in_matrix(matrix, antecedent) / n
            if sup_B is None:
                sup_B = _count_in_matrix(matrix, consequent) / n

            if sup_A == 0:
                continue

            sup_AB = sup
            confidence = sup_AB / sup_A

            if confidence < min_confidence:
                continue

            # --- 计算所有指标 ---
            # Lift
            lift = sup_AB / (sup_A * sup_B) if sup_A * sup_B > 0 else 0

            # Chi-square
            # Contingency table:
            #         B    not-B
            #   A    a     b
            #  not-A c     d
            # a = sup_AB * n, b = (sup_A - sup_AB) * n, etc.
            a = sup_AB * n
            b = (sup_A - sup_AB) * n
            c = (sup_B - sup_AB) * n
            d = n - a - b - c

            chi_square = 0.0
            total = a + b + c + d
            if total > 0:
                # 期望值
                row1 = a + b
                row2 = c + d
                col1 = a + c
                col2 = b + d
                if row1 > 0 and row2 > 0 and col1 > 0 and col2 > 0:
                    e_a = row1 * col1 / total
                    e_b = row1 * col2 / total
                    e_c = row2 * col1 / total
                    e_d = row2 * col2 / total
                    chi_square = (
                        (a - e_a)**2 / e_a +
                        (b - e_b)**2 / e_b +
                        (c - e_c)**2 / e_c +
                        (d - e_d)**2 / e_d
                    )

            # Conviction
            conviction = ((1 - sup_B) / (1 - confidence)) if confidence < 1 else float('inf')

            # Leverage
            leverage = sup_AB - sup_A * sup_B

            # Jaccard
            union = sup_A + sup_B - sup_AB
            jaccard = sup_AB / union if union > 0 else 0

            # 过滤无效规则（lift < 1 表示负相关，可保留用于分析）
            rule = Rule(
                antecedent=antecedent,
                consequent=consequent,
                support_A=sup_A,
                support_B=sup_B,
                support_AB=sup_AB,
                confidence=confidence,
                lift=lift,
                chi_square=chi_square,
                conviction=conviction,
                leverage=leverage,
                jaccard=jaccard,
            )
            rules.append(rule)

    return rules


def format_rules_table(rules: List[Rule], top_n: int = 20) -> pd.DataFrame:
    """将规则列表格式化为 DataFrame 表格显示"""
    if not rules:
        return pd.DataFrame()

    data = []
    for r in rules[:top_n]:
        data.append({
            "Antecedent": ", ".join(sorted(r.antecedent)),
            "Consequent": ", ".join(sorted(r.consequent)),
            "Support": f"{r.support_AB:.4f}",
            "Confidence": f"{r.confidence:.4f}",
            "Lift": f"{r.lift:.4f}",
            "Chi-square": f"{r.chi_square:.2f}",
            "Conviction": f"{r.conviction:.2f}" if r.conviction != float('inf') else "inf",
            "Leverage": f"{r.leverage:.4f}",
            "Jaccard": f"{r.jaccard:.4f}",
        })

    return pd.DataFrame(data)


if __name__ == "__main__":
    # 简单测试
    data = pd.DataFrame({
        "a": [True, True, False, True],
        "b": [True, True, True, False],
        "c": [True, False, True, True],
        "y=yes": [True, False, True, False],
    })
    from apriori import apriori
    itemsets = apriori(data, min_support=0.25)
    rules = generate_rules(itemsets, data, min_confidence=0.3)
    print(format_rules_table(rules))
