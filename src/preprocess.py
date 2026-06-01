"""
数据预处理模块
- 自动从 UCI 下载 Bank Marketing 数据集
- 连续特征离散化（等频分箱）
- 转为交易格式（boolean matrix）
"""
import pandas as pd
import numpy as np
from io import StringIO
from pathlib import Path
import zipfile
import urllib.request
import os

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/"
    "bank-additional.zip"
)
CSV_FILE = "bank-additional-full.csv"
LOCAL_CSV = DATA_DIR / CSV_FILE


def _download_data():
    """下载并解压数据集，返回 DataFrame"""
    if LOCAL_CSV.exists():
        print(f"[preprocess] 使用本地缓存: {LOCAL_CSV}")
        return pd.read_csv(LOCAL_CSV, sep=";")

    print(f"[preprocess] 从 UCI 下载数据 ...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    urllib.request.urlretrieve(DATA_URL, DATA_DIR / "bank-additional.zip")
    with zipfile.ZipFile(DATA_DIR / "bank-additional.zip", "r") as z:
        # zip 内文件在 bank-additional/ 子目录下
        zip_path = f"bank-additional/{CSV_FILE}"
        z.extract(zip_path, DATA_DIR)
        # 移动到上层方便访问
        (DATA_DIR / zip_path).rename(LOCAL_CSV)

    df = pd.read_csv(LOCAL_CSV, sep=";")
    print(f"[preprocess] 下载完成，共 {len(df)} 行, {len(df.columns)} 列")
    return df


# 列信息：名称 -> 类型
COL_TYPES = {
    "age": "numeric",
    "job": "categorical",
    "marital": "categorical",
    "education": "categorical",
    "default": "categorical",
    "housing": "categorical",
    "loan": "categorical",
    "contact": "categorical",
    "month": "categorical",
    "day_of_week": "categorical",
    "duration": "numeric",
    "campaign": "numeric",
    "pdays": "numeric",
    "previous": "numeric",
    "poutcome": "categorical",
    "emp.var.rate": "numeric",
    "cons.price.idx": "numeric",
    "cons.conf.idx": "numeric",
    "euribor3m": "numeric",
    "nr.employed": "numeric",
    "y": "target",
}

# 需要离散化的连续列（分箱数）
NUMERIC_BINS = {
    "age": 5,
    "duration": 5,
    "campaign": 4,
    "pdays": 3,
    "previous": 3,
    "emp.var.rate": 4,
    "cons.price.idx": 4,
    "cons.conf.idx": 4,
    "euribor3m": 4,
    "nr.employed": 4,
}


def _discretize(df: pd.DataFrame) -> pd.DataFrame:
    """将连续列等频离散化，替换为分段标签"""
    df = df.copy()
    for col, n_bins in NUMERIC_BINS.items():
        if col not in df.columns:
            continue
        # 处理缺失/unknown
        mask_unknown = df[col].isin(["unknown", ""])
        if mask_unknown.any():
            df.loc[mask_unknown, col] = np.nan

        try:
            # 等频分箱
            labels = [f"{col}_{i+1}" for i in range(n_bins)]
            df[col] = pd.qcut(
                df[col].astype(float),
                q=n_bins,
                labels=labels,
                duplicates="drop",
            )
        except (ValueError, TypeError):
            # 如果分箱失败（eg. 太少唯一值），用等宽
            labels = [f"{col}_{i+1}" for i in range(n_bins)]
            df[col] = pd.cut(
                df[col].astype(float),
                bins=n_bins,
                labels=labels,
            )
    return df


def _encode_as_items(df: pd.DataFrame) -> pd.DataFrame:
    """将 DataFrame 转为 boolean 交易矩阵（行=事务, 列=item）"""
    records = []
    for _, row in df.iterrows():
        items = set()
        for col in df.columns:
            val = row[col]
            if pd.isna(val) or val == "unknown" or val == "":
                continue
            item = f"{col}={val}"
            items.add(item)
        records.append(items)

    # 构建 boolean matrix
    all_items = sorted(set.union(*records)) if records else []
    matrix = pd.DataFrame(
        {item: [item in rec for rec in records] for item in all_items},
        dtype=bool,
    )
    return matrix


def load_and_preprocess(use_cache: bool = True) -> pd.DataFrame:
    """
    完整预处理流程:
      1. 下载/加载原始数据
      2. 连续列离散化
      3. 转换为 boolean 交易矩阵
    返回 (boolean_matrix, original_df)
    """
    raw = _download_data()
    # 去除 duration（domain knowledge: duration 泄露标签）
    raw = raw.drop(columns=["duration"], errors="ignore")

    # 处理 missing values: "unknown" -> NaN
    raw = raw.replace("unknown", np.nan)

    # 离散化
    df_disc = _discretize(raw)

    # 确保目标变量 y 不丢失
    # 将 y 转换为可读形式
    df_disc["y"] = raw["y"].map({"yes": "yes", "no": "no"})

    # 转为 boolean 交易矩阵
    matrix = _encode_as_items(df_disc)
    print(f"[preprocess] 交易矩阵: {matrix.shape[0]} 事务, {matrix.shape[1]} 个 item")
    print(f"[preprocess] 目标分布: yes={raw['y'].value_counts().get('yes', 0)}, "
          f"no={raw['y'].value_counts().get('no', 0)}")
    return matrix, raw


if __name__ == "__main__":
    mat, raw = load_and_preprocess()
    print(mat.head())
