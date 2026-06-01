"""
主入口：完整流程
1. 数据预处理
2. Apriori 和 FP-Growth 挖掘频繁项集
3. 交叉验证结果一致性
4. 生成关联规则，计算指标
5. Imbalance Ratio 分析
6. 典型模式评价
"""
import sys
import time
import pandas as pd
import numpy as np

from src.preprocess import load_and_preprocess
from src.apriori import apriori_with_counts
from src.fpgrowth import fpgrowth_with_counts
from src.metrics import generate_rules, format_rules_table
from src.imbalance import (
    compute_imbalance_ratio,
    analyze_target_rules,
    analyze_metric_behavior,
)


def cross_validate(apriori_itemsets, fpgrowth_itemsets, n_transactions):
    """交叉验证两个算法的频繁项集结果"""
    apriori_keys = set(apriori_itemsets.keys())
    fpgrowth_keys = set(fpgrowth_itemsets.keys())

    common = apriori_keys & fpgrowth_keys
    only_a = apriori_keys - fpgrowth_keys
    only_f = fpgrowth_keys - apriori_keys

    print("=" * 60)
    print("  算法交叉验证")
    print("=" * 60)
    print(f"  Apriori 频繁项集数:  {len(apriori_keys)}")
    print(f"  FP-Growth 频繁项集数: {len(fpgrowth_keys)}")
    print(f"  两者交集数量:         {len(common)}")
    if only_a:
        print(f"  仅 Apriori 有:        {len(only_a)} 个")
        for s in list(only_a)[:5]:
            print(f"    - {set(s)}")
    if only_f:
        print(f"  仅 FP-Growth 有:      {len(only_f)} 个")
        for s in list(only_f)[:5]:
            print(f"    - {set(s)}")

    # 验证支持度一致性
    max_diff = 0
    for k in common:
        diff = abs(apriori_itemsets[k] - fpgrowth_itemsets[k])
        if diff > max_diff:
            max_diff = diff
    print(f"  公共项集最大支持度差异: {max_diff:.6f}")
    print()

    return len(common) == len(apriori_keys) == len(fpgrowth_keys)


def main():
    print("=" * 60)
    print("  数据挖掘 Lab 3: 频繁模式挖掘与关联规则分析")
    print("  数据集: UCI Bank Marketing")
    print("=" * 60)
    print()

    # ========== Step 1: 数据预处理 ==========
    print("[1/5] 数据预处理 ...")
    matrix, raw_df = load_and_preprocess()
    n = len(matrix)
    print(f"      事务数: {n},  特征维度: {matrix.shape[1]}")

    # ========== Step 2: Imbalance Ratio 分析 ==========
    print()
    print("[2/5] Imbalance Ratio 分析 ...")
    ir = compute_imbalance_ratio(matrix)
    print(f"      Imbalance Ratio (no/yes) = {ir:.2f}")
    target_cols = [c for c in matrix.columns if c.startswith("y=")]
    for col in target_cols:
        cnt = matrix[col].sum()
        print(f"      {col}: {cnt} ({cnt/n*100:.1f}%)")
    print()

    # ========== Step 3: 频繁模式挖掘 ==========
    print("[3/5] 频繁模式挖掘 ...")

    # 参数设置
    MIN_SUPPORT = 0.05  # 5% 支持度
    MIN_CONFIDENCE = 0.3

    # Apriori
    print("\n  --- Apriori ---")
    t0 = time.time()
    apriori_supports, apriori_counts = apriori_with_counts(matrix, MIN_SUPPORT)
    t_apriori = time.time() - t0
    print(f"  耗时: {t_apriori:.2f}s")

    # FP-Growth
    print("\n  --- FP-Growth ---")
    t0 = time.time()
    fpgrowth_supports, fpgrowth_counts = fpgrowth_with_counts(matrix, MIN_SUPPORT)
    t_fpgrowth = time.time() - t0
    print(f"  耗时: {t_fpgrowth:.2f}s")

    # 交叉验证
    consistent = cross_validate(apriori_supports, fpgrowth_supports, n)
    if not consistent:
        print("  ⚠ 注意: 两个算法结果不完全一致，可能存在 bug")
    else:
        print("  ✅ 两个算法结果完全一致")

    # 使用 FP-Growth 结果继续分析（更高效）
    frequent_itemsets = fpgrowth_supports

    # ========== Step 4: 关联规则生成与评估 ==========
    print()
    print("[4/5] 关联规则生成与指标计算 ...")

    # 所有规则
    all_rules = generate_rules(
        frequent_itemsets, matrix,
        min_confidence=MIN_CONFIDENCE,
    )
    print(f"  生成规则总数: {len(all_rules)}")

    # 按目标变量筛选
    yes_rules = [r for r in all_rules if "y=yes" in r.consequent]
    no_rules = [r for r in all_rules if "y=no" in r.consequent]
    print(f"  → y=yes 规则: {len(yes_rules)}")
    print(f"  → y=no 规则:  {len(no_rules)}")

    if yes_rules:
        # Top-15 by Lift
        top_yes = sorted(yes_rules, key=lambda r: -r.lift)[:15]
        print("\n  Top-15 规则 (y=yes, 按 Lift 降序):")
        print(format_rules_table(top_yes, top_n=15).to_string(index=False))
        print()

    if no_rules:
        # Top-10 by Lift
        top_no = sorted(no_rules, key=lambda r: -r.lift)[:10]
        print("\n  Top-10 规则 (y=no, 按 Lift 降序):")
        print(format_rules_table(top_no, top_n=10).to_string(index=False))
        print()

    # ========== Step 5: Imbalance 影响分析 ==========
    print()
    print("[5/5] Imbalance 影响分析 ...")

    print("\n  --- 目标变量规则对比 ---")
    target_analysis = analyze_target_rules(all_rules, matrix)
    print(target_analysis.to_string(index=False))

    print("\n  --- 不同指标排序对比 ---")
    metric_analysis = analyze_metric_behavior(yes_rules)
    if not metric_analysis.empty:
        print(metric_analysis.to_string(index=False))

    # ========== 典型模式解读 ==========
    print()
    print("=" * 60)
    print("  典型模式解读与应用建议")
    print("=" * 60)

    if yes_rules:
        top5_lift = sorted(yes_rules, key=lambda r: -r.lift)[:5]
        top5_conf = sorted(yes_rules, key=lambda r: -r.confidence)[:5]
        top5_chi = sorted(yes_rules, key=lambda r: -r.chi_square)[:5]

        print("\n  【按 Lift 最高的 Top-5 规则】")
        for i, r in enumerate(top5_lift, 1):
            A = ", ".join(sorted(r.antecedent))
            B = ", ".join(sorted(r.consequent))
            print(f"  {i}. {{{A}}} → {{{B}}}")
            print(f"     Support={r.support_AB:.4f}, Confidence={r.confidence:.4f}, "
                  f"Lift={r.lift:.4f}, Chi²={r.chi_square:.2f}")

            # 解读
            if "y=yes" in r.consequent:
                interpretation = _interpret_rule(r)
                print(f"     解读: {interpretation}")

        print("\n  【按 Chi-square 最高的 Top-5 规则】")
        for i, r in enumerate(top5_chi, 1):
            A = ", ".join(sorted(r.antecedent))
            B = ", ".join(sorted(r.consequent))
            print(f"  {i}. {{{A}}} → {{{B}}}  (χ²={r.chi_square:.2f})")
    print()
    print("=" * 60)
    print("  分析完成")

    # 返回结果供后续使用
    return {
        "matrix": matrix,
        "raw": raw_df,
        "frequent_itemsets": frequent_itemsets,
        "rules": all_rules,
        "yes_rules": yes_rules,
        "no_rules": no_rules,
        "imbalance_ratio": ir,
    }


def _interpret_rule(rule) -> str:
    """对规则进行业务解读"""
    ante = set(rule.antecedent)
    parts = []

    # 检查是否有联系相关的特征
    contact_features = [a for a in ante if a.startswith("contact=")]
    if contact_features:
        parts.append(f"联系方式为 {contact_features[0].split('=')[1]}")

    month_features = [a for a in ante if a.startswith("month=")]
    if month_features:
        parts.append(f"月份为 {month_features[0].split('=')[1]}")

    job_features = [a for a in ante if a.startswith("job=")]
    if job_features:
        parts.append(f"职业为 {job_features[0].split('=')[1]}")

    edu_features = [a for a in ante if a.startswith("education=")]
    if edu_features:
        parts.append(f"学历为 {edu_features[0].split('=')[1]}")

    marital_features = [a for a in ante if a.startswith("marital=")]
    if marital_features:
        parts.append(f"婚姻状况为 {marital_features[0].split('=')[1]}")

    euro_features = [a for a in ante if a.startswith("euribor3m=")]
    if euro_features:
        parts.append(f"Euribor 3m 利率区间为 {euro_features[0].split('=')[1]}")

    emp_var = [a for a in ante if a.startswith("emp.var.rate=")]
    if emp_var:
        parts.append(f"就业变动率为 {emp_var[0].split('=')[1]}")

    nr_emp = [a for a in ante if a.startswith("nr.employed=")]
    if nr_emp:
        parts.append(f"就业人数为 {nr_emp[0].split('=')[1]}")

    poutcome = [a for a in ante if a.startswith("poutcome=")]
    if poutcome:
        parts.append(f"上次营销结果为 {poutcome[0].split('=')[1]}")

    housing = [a for a in ante if a.startswith("housing=")]
    if housing:
        parts.append(f"房贷为 {housing[0].split('=')[1]}")

    loan = [a for a in ante if a.startswith("loan=")]
    if loan:
        parts.append(f"个人贷款为 {loan[0].split('=')[1]}")

    default = [a for a in ante if a.startswith("default=")]
    if default:
        parts.append(f"信用违约为 {default[0].split('=')[1]}")

    previous = [a for a in ante if a.startswith("previous=")]
    if previous:
        parts.append(f"先前联系次数为 {previous[0].split('=')[1]}")

    campaign = [a for a in ante if a.startswith("campaign=")]
    if campaign:
        parts.append(f"本次营销联系次数为 {campaign[0].split('=')[1]}")

    if not parts:
        return "综合特征组合表明该客户群体订阅定期存款概率较高"

    return "客户" + "、".join(parts) + "时，订阅定期存款的倾向较高"

    # 预防高 lift 但低 support 的过拟合解读


if __name__ == "__main__":
    results = main()
