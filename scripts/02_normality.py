from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["savefig.dpi"] = 1000

from granite_ml.config import (  # noqa: E402
    CHAMPION_DIR,
    FOLDS_DIR,
    RAW_DATA_FILE,
    RESULTS_DIR,
    STABILITY_DIR,
)

import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# 0. 配置
# ============================================================

# 上一步 01_foldwise_preprocessing_and_ratio_feature_construction.py 的输出目录
INPUT_ROOT = FOLDS_DIR

# 输出目录
OUT_DIR = RESULTS_DIR / "02_normality"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TYPE_COL = "Type"

NON_NUM_COLS = {
    "No.",
    "Samp1e",
    "Sample",
    "Type",
    "Type-1",
    "Type-2",
    "Reference",
}

IMPUTATION_METHODS = ["global_mean", "knn"]

N_SPLITS = 5

ALPHA = 0.05
MIN_N = 20


# ============================================================
# 1. 工具函数
# ============================================================

def get_feature_cols(df):
    """
    获取参与正态性检验的数值特征列。
    排除样品编号、类型标签等非建模列。
    """
    candidate_cols = [
        c for c in df.columns
        if c not in NON_NUM_COLS
    ]

    feature_cols = []

    for c in candidate_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if pd.api.types.is_numeric_dtype(s):
            feature_cols.append(c)

    return feature_cols


def normality_test_one_feature(x, alpha=0.05, min_n=20):
    """
    对单个特征进行正态性检验。

    使用：
    1. D’Agostino K² normality test
    2. Jarque-Bera test

    判定规则：
    - 若任一检验 p < alpha，则判为 Non-normal；
    - 若两个检验均不拒绝正态性，则判为 Normal-like；
    - 样本太少、全常数或检验失败，则标记为 Not tested。
    """
    x = pd.to_numeric(x, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).dropna().astype(float)

    n = len(x)

    mean = float(x.mean()) if n > 0 else np.nan
    std = float(x.std(ddof=1)) if n > 1 else np.nan
    skew = float(stats.skew(x, bias=False)) if n >= 3 else np.nan
    kurt_excess = float(stats.kurtosis(x, fisher=True, bias=False)) if n >= 4 else np.nan

    result = {
        "n_valid": n,
        "mean": mean,
        "std": std,
        "skewness": skew,
        "kurtosis_excess": kurt_excess,
        "K2_stat": np.nan,
        "K2_p": np.nan,
        "JB_stat": np.nan,
        "JB_p": np.nan,
        "p_min": np.nan,
        "alpha": alpha,
        "decision": "Not tested"
    }

    if n < min_n:
        result["decision"] = "Not tested (too few samples)"
        return result

    if pd.isna(std) or std == 0:
        result["decision"] = "Not tested (constant feature)"
        return result

    # D’Agostino K²
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        try:
            k2_stat, k2_p = stats.normaltest(x)
            result["K2_stat"] = float(k2_stat)
            result["K2_p"] = float(k2_p)
        except Exception:
            result["K2_stat"] = np.nan
            result["K2_p"] = np.nan

        # Jarque-Bera
        try:
            jb = stats.jarque_bera(x)
            result["JB_stat"] = float(jb.statistic)
            result["JB_p"] = float(jb.pvalue)
        except Exception:
            result["JB_stat"] = np.nan
            result["JB_p"] = np.nan

    p_values = [
        p for p in [result["K2_p"], result["JB_p"]]
        if pd.notna(p)
    ]

    if len(p_values) == 0:
        result["decision"] = "Not tested (test failed)"
        return result

    result["p_min"] = float(np.min(p_values))

    reject_k2 = (
        result["K2_p"] < alpha
        if pd.notna(result["K2_p"])
        else False
    )

    reject_jb = (
        result["JB_p"] < alpha
        if pd.notna(result["JB_p"])
        else False
    )

    if reject_k2 or reject_jb:
        result["decision"] = "Non-normal (reject normality)"
    else:
        result["decision"] = "Normal-like (fail to reject normality)"

    return result


def normality_test_dataframe(df, method, fold, alpha=0.05, min_n=20):
    """
    对一个训练折数据集的所有特征做正态性检验。
    """
    feature_cols = get_feature_cols(df)

    rows = []

    for col in feature_cols:
        test_result = normality_test_one_feature(
            df[col],
            alpha=alpha,
            min_n=min_n
        )

        row = {
            "Method": method,
            "Fold": fold,
            "Feature": col
        }

        row.update(test_result)
        rows.append(row)

    res = pd.DataFrame(rows)

    # 排序：先按 decision，再按 p_min 从大到小
    if "p_min" in res.columns:
        res = res.sort_values(
            by=["decision", "p_min", "n_valid"],
            ascending=[True, False, False]
        )

    return res


def summarize_one_fold(res):
    """
    汇总一个训练折的正态性检验结果。
    """
    total_features = len(res)

    tested_mask = res["decision"].isin([
        "Non-normal (reject normality)",
        "Normal-like (fail to reject normality)"
    ])

    tested = int(tested_mask.sum())

    normal_like = int(
        (res["decision"] == "Normal-like (fail to reject normality)").sum()
    )

    non_normal = int(
        (res["decision"] == "Non-normal (reject normality)").sum()
    )

    not_tested = total_features - tested

    return {
        "Total_features": total_features,
        "Tested_features": tested,
        "Normal_like_features": normal_like,
        "Non_normal_features": non_normal,
        "Not_tested_features": not_tested,
        "Non_normal_percent_among_tested": (
            non_normal / tested * 100 if tested > 0 else np.nan
        )
    }


def aggregate_across_folds(all_results):
    """
    汇总每个特征在所有折中的正态性判断。
    """
    df_all = pd.concat(all_results, axis=0, ignore_index=True)

    def count_decision(s, decision):
        return int((s == decision).sum())

    agg = (
        df_all
        .groupby(["Method", "Feature"], as_index=False)
        .agg(
            Folds_tested=(
                "decision",
                lambda s: int(s.isin([
                    "Non-normal (reject normality)",
                    "Normal-like (fail to reject normality)"
                ]).sum())
            ),
            Non_normal_folds=(
                "decision",
                lambda s: count_decision(s, "Non-normal (reject normality)")
            ),
            Normal_like_folds=(
                "decision",
                lambda s: count_decision(s, "Normal-like (fail to reject normality)")
            ),
            Not_tested_folds=(
                "decision",
                lambda s: int(s.str.startswith("Not tested").sum())
            ),
            Median_K2_p=("K2_p", "median"),
            Median_JB_p=("JB_p", "median"),
            Median_p_min=("p_min", "median"),
            Median_skewness=("skewness", "median"),
            Median_kurtosis_excess=("kurtosis_excess", "median")
        )
    )

    agg["Non_normal_fold_percent"] = (
        agg["Non_normal_folds"] / agg["Folds_tested"] * 100
    )

    agg["Overall_decision"] = np.where(
        agg["Non_normal_folds"] > 0,
        "Mostly/partly non-normal across folds",
        "Normal-like across tested folds"
    )

    agg = agg.sort_values(
        by=["Method", "Non_normal_fold_percent", "Median_p_min"],
        ascending=[True, False, True]
    )

    return df_all, agg


# ============================================================
# 2. 主函数
# ============================================================

def main():
    all_results = []
    fold_summary_records = []

    for method in IMPUTATION_METHODS:
        method_input_dir = os.path.join(INPUT_ROOT, method)

        if not os.path.exists(method_input_dir):
            raise FileNotFoundError(f"未找到输入文件夹：{method_input_dir}")

        method_out_dir = os.path.join(OUT_DIR, method)
        os.makedirs(method_out_dir, exist_ok=True)

        for fold in range(1, N_SPLITS + 1):
            train_path = os.path.join(
                method_input_dir,
                f"fold_{fold:02d}_train_with_ratios.xlsx"
            )

            if not os.path.exists(train_path):
                raise FileNotFoundError(f"未找到训练折文件：{train_path}")

            df_train = pd.read_excel(train_path)
            df_train.columns = [str(c).strip() for c in df_train.columns]

            res = normality_test_dataframe(
                df_train,
                method=method,
                fold=fold,
                alpha=ALPHA,
                min_n=MIN_N
            )

            out_path = os.path.join(
                method_out_dir,
                f"fold_{fold:02d}_train_normality_results.xlsx"
            )

            res.to_excel(out_path, index=False)

            fold_summary = summarize_one_fold(res)
            fold_summary.update({
                "Method": method,
                "Fold": fold,
                "Input_file": train_path,
                "Output_file": out_path
            })

            fold_summary_records.append(fold_summary)
            all_results.append(res)

            print(
                f"✅ {method} Fold {fold}: "
                f"tested={fold_summary['Tested_features']}, "
                f"non-normal={fold_summary['Non_normal_features']}"
            )

    # 汇总所有折
    all_results_df, feature_agg_df = aggregate_across_folds(all_results)

    fold_summary_df = pd.DataFrame(fold_summary_records)

    all_results_path = os.path.join(
        OUT_DIR,
        "all_fold_normality_results.xlsx"
    )

    feature_agg_path = os.path.join(
        OUT_DIR,
        "feature_normality_summary_across_folds.xlsx"
    )

    fold_summary_path = os.path.join(
        OUT_DIR,
        "fold_normality_summary.xlsx"
    )

    all_results_df.to_excel(all_results_path, index=False)
    feature_agg_df.to_excel(feature_agg_path, index=False)
    fold_summary_df.to_excel(fold_summary_path, index=False)

    print("\n全部正态性检验完成。")
    print(f"逐折详细结果：{all_results_path}")
    print(f"特征跨折汇总：{feature_agg_path}")
    print(f"折级汇总：{fold_summary_path}")

    # 最简汇总打印
    print("\n折级汇总：")
    print(
        fold_summary_df[
            [
                "Method",
                "Fold",
                "Total_features",
                "Tested_features",
                "Normal_like_features",
                "Non_normal_features",
                "Non_normal_percent_among_tested"
            ]
        ]
    )


# ============================================================
# 3. 程序入口
# ============================================================

if __name__ == "__main__":
    main()
