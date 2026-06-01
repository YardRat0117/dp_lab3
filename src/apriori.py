"""
Apriori 算法 — 从零实现
多进程并行：Pool 全局复用，12 核全开，所有层级并行计数
"""
from itertools import combinations
import multiprocessing as mp
import os
import pandas as pd
import numpy as np
from typing import List, FrozenSet, Dict, Tuple


# —————— 并行计数全局变量 ——————
_WORKER_MATRIX = None       # numpy array (工作进程)
_WORKER_COL_TO_IDX = None   # col_name → col_index


def _init_worker(matrix_np, col_to_idx):
    """Pool 初始化：注入数据到工作进程全局变量（每进程仅调用一次）"""
    global _WORKER_MATRIX, _WORKER_COL_TO_IDX
    _WORKER_MATRIX = matrix_np
    _WORKER_COL_TO_IDX = col_to_idx


def _count_chunk(candidates: List[FrozenSet]) -> Dict[FrozenSet, int]:
    """工作进程：对一批候选集做向量化计数"""
    global _WORKER_MATRIX, _WORKER_COL_TO_IDX
    result = {}
    for cand in candidates:
        indices = [_WORKER_COL_TO_IDX[c] for c in cand]
        result[cand] = int(_WORKER_MATRIX[:, indices].all(axis=1).sum())
    return result
# ——————————————————————————


def _generate_candidates(prev_level: List[FrozenSet], k: int) -> List[FrozenSet]:
    """从 L_{k-1} 生成候选 C_k：自连接 + 剪枝"""
    candidates = []
    prev_list = list(prev_level)
    n = len(prev_list)
    for i in range(n):
        s1 = sorted(prev_list[i])
        for j in range(i + 1, n):
            s2 = sorted(prev_list[j])
            if s1[:k - 2] == s2[:k - 2]:
                candidate = frozenset(prev_list[i] | prev_list[j])
                # 剪枝
                valid = True
                for subset in combinations(candidate, k - 1):
                    if frozenset(subset) not in prev_level:
                        valid = False
                        break
                if valid:
                    candidates.append(candidate)
    return candidates


def _parallel_count(pool, candidates: List[FrozenSet], n_jobs: int) -> Dict[FrozenSet, int]:
    """将候选集分块发给工作进程池并行计数，合并结果"""
    if not candidates:
        return {}

    # 每个 worker 至少分 10 个候选，避免分太多小块
    chunk_size = max(10, len(candidates) // n_jobs)
    chunks = []
    for i in range(0, len(candidates), chunk_size):
        chunks.append(candidates[i:i + chunk_size])

    results = pool.map(_count_chunk, chunks)

    merged = {}
    for r in results:
        merged.update(r)
    return merged


def apriori(matrix: pd.DataFrame, min_support: float = 0.10,
            verbose: bool = True) -> Dict[FrozenSet, float]:
    """
    Apriori 主算法 — 全程多进程并行计数

    参数:
        matrix:      boolean DataFrame (行=事务, 列=item)
        min_support: 最小支持度 (比例)，默认 0.05
        verbose:     打印进度
    """
    n_transactions = len(matrix)
    min_count = max(1, int(min_support * n_transactions))
    n_jobs = os.cpu_count() or 4  # 全核心

    # 预处理：numpy 矩阵 + 列索引映射（一次性给所有 worker）
    matrix_np = matrix.values
    col_to_idx = {col: i for i, col in enumerate(matrix.columns)}

    # ===== 创建全局进程池（只创建一次）=====
    if verbose:
        print(f"[Apriori] 启动进程池: {n_jobs} workers")

    with mp.Pool(n_jobs, initializer=_init_worker,
                 initargs=(matrix_np, col_to_idx)) as pool:

        # L1: 频繁 1-项集（单 item 直接用列 sum，无需并行）
        L: Dict[int, Dict[FrozenSet, float]] = {}
        L[1] = {}
        for col in matrix.columns:
            count = matrix[col].sum()
            if count >= min_count:
                L[1][frozenset([col])] = count / n_transactions

        if verbose:
            print(f"[Apriori] L1: {len(L[1])} 个频繁项集")

        # L2, L3, ... 全部走进程池并行计数
        k = 2
        while L.get(k - 1):
            candidates = _generate_candidates(list(L[k - 1].keys()), k)
            if not candidates:
                break

            if verbose:
                n_cand = len(candidates)
                print(f"[Apriori]   候选 C{k}: {n_cand} 个 → 并行计数 ({n_jobs} workers)",
                      end=" ")

            support_counts = _parallel_count(pool, candidates, n_jobs)

            if verbose:
                n_freq = sum(1 for c in support_counts.values() if c >= min_count)
                print(f"→ L{k}: {n_freq}")

            L[k] = {}
            for cand, count in support_counts.items():
                if count >= min_count:
                    L[k][cand] = count / n_transactions

            k += 1

    # 合并所有层级
    result = {}
    for level in L.values():
        result.update(level)

    if verbose:
        max_k = max(L.keys()) if L else 1
        print(f"[Apriori] 总计 {len(result)} 个频繁项集, 最大长度={max_k} "
              f"(min_support={min_support})")
    return result


def apriori_with_counts(matrix: pd.DataFrame, min_support: float = 0.10,
                        verbose: bool = True)\
        -> Tuple[Dict[FrozenSet, float], Dict[FrozenSet, int]]:
    n = len(matrix)
    itemsets = apriori(matrix, min_support, verbose)
    counts = {k: int(v * n) for k, v in itemsets.items()}
    return itemsets, counts


if __name__ == "__main__":
    data = pd.DataFrame({
        "a": [True, True, False, True],
        "b": [True, True, True, False],
        "c": [True, False, True, True],
    })
    result = apriori(data, min_support=0.5)
    for itemset, sup in sorted(result.items(), key=lambda x: -x[1]):
        print(f"{set(itemset)}: sup={sup:.3f}")
