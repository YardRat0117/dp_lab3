"""
FP-Growth 算法 — 从零实现
使用 FP-Tree 数据结构，无需生成候选项集
"""
from collections import defaultdict
import pandas as pd
from typing import List, Set, FrozenSet, Dict, Tuple, Optional


class FPNode:
    """FP-Tree 节点"""
    __slots__ = ("item", "count", "parent", "children", "node_link")

    def __init__(self, item: Optional[str], count: int = 1, parent: Optional["FPNode"] = None):
        self.item = item          # item 名称 (根节点为 None)
        self.count = count        # 频次计数
        self.parent = parent      # 父节点
        self.children = {}        # 子节点: item -> FPNode
        self.node_link = None     # 链接到下一个相同 item 的节点（用于 header table）

    def __repr__(self):
        return f"FPNode({self.item}, count={self.count}, children={list(self.children.keys())})"


def _build_header_table(sorted_items: List[str]) -> Dict[str, List[FPNode]]:
    """构建 header table: item -> [node, node, ...] 链表"""
    return {item: [] for item in sorted_items}


def _insert_transaction(node: FPNode, transaction: List[str], count: int,
                        header_table: Dict[str, List[FPNode]]):
    """递归插入一条事务到 FP-Tree"""
    if not transaction:
        return

    item = transaction[0]
    if item in node.children:
        child = node.children[item]
        child.count += count
    else:
        child = FPNode(item, count, node)
        node.children[item] = child
        # 链接到 header table
        header_table[item].append(child)

    _insert_transaction(child, transaction[1:], count, header_table)


def _build_fptree(matrix: pd.DataFrame, min_count: int)\
        -> Tuple[Optional[FPNode], Dict[str, List[FPNode]], Dict[str, int]]:
    """
    构建 FP-Tree

    返回: (root, header_table, item_support)
        root: FP-Tree 根节点
        header_table: item -> [FPNode, ...]
        item_support: item -> 支持度计数
    """
    n_transactions = len(matrix)

    # 第一步：扫描，统计每个 item 的支持度
    item_support: Dict[str, int] = {}
    for col in matrix.columns:
        cnt = matrix[col].sum()
        if cnt >= min_count:
            item_support[col] = cnt

    if not item_support:
        return None, {}, {}

    # 按支持度降序排列 item
    sorted_items = sorted(item_support.keys(), key=lambda x: (-item_support[x], x))

    # 第二步：构建 FP-Tree
    root = FPNode(None)
    header_table = _build_header_table(sorted_items)

    # 使用 numpy 加速行遍历
    np_matrix = matrix.values
    col_indices = {item: idx for idx, item in enumerate(matrix.columns)}
    for row in np_matrix:
        transaction = [item for item in sorted_items if row[col_indices[item]]]
        if transaction:
            _insert_transaction(root, transaction, 1, header_table)

    return root, header_table, item_support


def _mine_subtree(prefix: FrozenSet, suffix_item: str,
                  conditional_pattern_base: List[Tuple[List[str], int]],
                  min_count: int) -> Dict[FrozenSet, int]:
    """
    从条件模式基构建条件 FP-Tree 并递归挖掘
    """
    itemsets: Dict[FrozenSet, int] = {}

    # 统计条件模式基中各 item 频率
    item_counts: Dict[str, int] = defaultdict(int)
    for pattern, count in conditional_pattern_base:
        for item in pattern:
            item_counts[item] += count

    # 过滤不满足 min_count 的 item
    frequent_items = {item for item, cnt in item_counts.items() if cnt >= min_count}

    if not frequent_items:
        return itemsets

    # 构建条件 FP-Tree
    sorted_items = sorted(frequent_items, key=lambda x: (-item_counts[x], x))

    # 以字典模拟节点： (prefix_path, count) -> 插入
    # 更准确的做法：构建一个临时的 FP-Tree 并递归
    root = FPNode(None)
    header_table = {item: [] for item in sorted_items}

    for pattern, count in conditional_pattern_base:
        # 过滤掉不频繁的 item，按全局顺序排序
        filtered = [item for item in pattern if item in frequent_items]
        filtered.sort(key=lambda x: (-item_counts[x], x))
        if filtered:
            _insert_transaction(root, filtered, count, header_table)

    # 递归挖掘条件树 (自底向上遍历 header table)
    itemset_suffix = frozenset(list(prefix) + [suffix_item])

    for item in reversed(sorted_items):
        # 当前项集 = item + suffix
        new_itemset = itemset_suffix
        # 获取 item 的支持度（从 header table 的节点累加）
        support = sum(node.count for node in header_table[item])
        itemsets[new_itemset] = support

        # 构建 item 的条件模式基
        item_cpb = []
        for node in header_table[item]:
            path = []
            parent = node.parent
            while parent and parent.item is not None:
                path.append(parent.item)
                parent = parent.parent
            if path:
                path.reverse()
                item_cpb.append((path, node.count))

        if item_cpb:
            sub_itemsets = _mine_subtree(
                list(prefix) + [suffix_item] if suffix_item else list(prefix),
                item,
                item_cpb,
                min_count,
            )
            itemsets.update(sub_itemsets)

    # 增加单一项
    for item in sorted_items:
        new_set = frozenset([item] + list(prefix) + ([suffix_item] if suffix_item else []))
        itemsets[new_set] = sum(node.count for node in header_table[item])

    return itemsets


def fpgrowth(matrix: pd.DataFrame, min_support: float = 0.10, verbose: bool = True) -> Dict[FrozenSet, float]:
    """
    FP-Growth 主算法

    参数:
        matrix: boolean DataFrame, 行=事务, 列=item
        min_support: 最小支持度 (比例)
        verbose: 是否打印进度

    返回:
        frequent_itemsets: dict {frozenset: support_ratio}
    """
    n_transactions = len(matrix)
    min_count = max(1, int(min_support * n_transactions))

    root, header_table, item_support = _build_fptree(matrix, min_count)
    if root is None:
        print("[FP-Growth] 无频繁项集")
        return {}

    # 存储所有频繁项集
    all_itemsets: Dict[FrozenSet, int] = {}

    # 单一项集
    for item, sup in item_support.items():
        all_itemsets[frozenset([item])] = sup

    # 自底向上挖掘
    sorted_items = sorted(item_support.keys(), key=lambda x: (-item_support[x], x))

    for item in reversed(sorted_items):
        # 构建 item 的条件模式基
        conditional_pattern_base = []
        for node in header_table[item]:
            path = []
            parent = node.parent
            while parent and parent.item is not None:
                path.append(parent.item)
                parent = parent.parent
            if path:
                path.reverse()
                conditional_pattern_base.append((path, node.count))

        if conditional_pattern_base:
            sub_itemsets = _mine_subtree(
                prefix=frozenset(),
                suffix_item=item,
                conditional_pattern_base=conditional_pattern_base,
                min_count=min_count,
            )
            all_itemsets.update(sub_itemsets)

    # 去重：有些项集会重复出现（根路径 + 递归挖掘可能重复）
    # 保留最大支持度
    dedup: Dict[FrozenSet, int] = {}
    for itemset, count in all_itemsets.items():
        if itemset not in dedup or count > dedup[itemset]:
            dedup[itemset] = count

    result = {k: v / n_transactions for k, v in dedup.items()}

    if verbose:
        print(f"[FP-Growth] 总共 {len(result)} 个频繁项集 (min_support={min_support})")

    return result


def fpgrowth_with_counts(matrix: pd.DataFrame, min_support: float = 0.10, verbose: bool = True)\
        -> Tuple[Dict[FrozenSet, float], Dict[FrozenSet, int]]:
    """返回 (support_ratio, support_count)"""
    itemsets = fpgrowth(matrix, min_support, verbose)
    n = len(matrix)
    counts = {k: int(v * n) for k, v in itemsets.items()}
    return itemsets, counts


if __name__ == "__main__":
    # 简单测试
    data = pd.DataFrame({
        "a": [True, True, False, True],
        "b": [True, True, True, False],
        "c": [True, False, True, True],
    })
    result = fpgrowth(data, min_support=0.5)
    for itemset, sup in sorted(result.items(), key=lambda x: -x[1]):
        print(f"{set(itemset)}: sup={sup:.3f}")
