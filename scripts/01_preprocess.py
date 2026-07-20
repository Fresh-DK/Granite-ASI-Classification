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
from granite_ml.data import prepare_source_data  # noqa: E402

import os
import re
from itertools import combinations

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.impute import SimpleImputer, KNNImputer


# ============================================================
# 0. 配置
# ============================================================

IN_PATH = RAW_DATA_FILE

OUT_DIR = FOLDS_DIR
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

MAX_MISSING_PER_ROW = 3

RANDOM_SEED = 42

# 普通五折：N_REPEATS = 1
# 重复五折：例如 5 repeats × 5 folds，则 N_REPEATS = 5
N_SPLITS = 5
N_REPEATS = 1

# 插补方式
IMPUTATION_METHODS = ["global_mean", "knn"]

# KNN 参数
KNN_N_NEIGHBORS = 5
KNN_WEIGHTS = "distance"

# 主量元素列
MAJOR_COLS = [
    "SiO2(wt%)",
    "TiO2",
    "Al2O3",
    "Fe2O3t",
    "MgO",
    "CaO",
    "Na2O",
    "K2O",
    "MnO",
    "P2O5"
]

# 微量元素 + 稀土元素列
TRACE_COLS = [
    "Ga(ppm)",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "Pb",
    "Th",
    "U",
    "Cs"
]


# ============================================================
# 1. 特殊值清理
# ============================================================

def clean_censored_value(x):
    """
    处理地球化学数据中的特殊记录。

    规则：
    1. '<34'、'>100'、'<0.01' 等含明确数值的记录：
       提取数字部分。
       例如 '<0.01' -> 0.01，'>100' -> 100。

    2. '<d.l.'、'<dl'、'bdl'、'below detection limit'：
       视为缺失值 NaN。

    3. '> upper limit'、'>upper limit'：
       视为缺失值 NaN。

    4. 空白、nan、n.d.、na、-- 等：
       视为缺失值 NaN。

    5. 其他无法转为数字的非空字符串：
       后续作为无法解析字符串所在行删除。
    """
    if x is None:
        return np.nan

    s = str(x).strip()

    if s == "":
        return np.nan

    s_low = s.lower().replace(" ", "")

    missing_tokens = {
        "nan",
        "na",
        "n/a",
        "nd",
        "n.d.",
        "null",
        "none",
        "-",
        "--",
        "—"
    }

    if s_low in missing_tokens:
        return np.nan

    below_dl_tokens = {
        "<d.l.",
        "<d.l",
        "<dl",
        "<detectionlimit",
        "bdl",
        "<bdl",
        "belowdetectionlimit",
        "belowdl"
    }

    if s_low in below_dl_tokens:
        return np.nan

    upper_limit_tokens = {
        ">upperlimit",
        ">u.l.",
        ">ul",
        "aboveupperlimit"
    }

    if s_low in upper_limit_tokens:
        return np.nan

    # 去掉数字中的逗号，例如 1,234.5
    s2 = s.replace(",", "")

    # '<34'、'>100'、'≤0.01'、'≥100' 等：提取数字部分
    m = re.match(r"^[<>≤≥]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$", s2)

    if m:
        return m.group(1)

    return s2


def is_missing_after_clean(x):
    """
    判断清理后的值是否属于缺失。
    """
    if x is None:
        return True

    if pd.isna(x):
        return True

    s = str(x).strip().lower()

    return s in {
        "",
        "nan",
        "na",
        "n/a",
        "null",
        "none"
    }


# ============================================================
# 2. 原始基础清洗
#    注意：这里只做基础清理，不做最终插补
# ============================================================

def load_and_basic_clean(in_path):
    """
    分折前允许做的基础清洗步骤：

    1. 读取数据；
    2. 处理 <0.01、>100、<d.l.、> upper limit 等特殊值；
    3. 删除仍无法解析为数字的乱码行；
    4. 删除原始缺失值数量 > MAX_MISSING_PER_ROW 的样品；
    5. 保留 NaN，不做最终插补。

    该步骤不使用 A/S/I 标签信息，也不学习插补参数。
    """
    df_raw, source_changes, raw_workbook, header_row = prepare_source_data(in_path)
    df = df_raw.copy()

    df.columns = [str(c).strip() for c in df.columns]

    if TYPE_COL not in df.columns:
        raise ValueError(
            f"未找到类型列 {TYPE_COL}，当前列名：{df.columns.tolist()}"
        )

    num_cols = [c for c in df.columns if c not in NON_NUM_COLS]

    # 处理特殊值
    for c in num_cols:
        df[c] = df[c].apply(clean_censored_value)

    # 判断无法转成数字的非空字符串
    df_num_tmp = df[num_cols].apply(pd.to_numeric, errors="coerce")

    garbled_mask = pd.DataFrame(
        False,
        index=df.index,
        columns=num_cols
    )

    for c in num_cols:
        orig = df[c]
        conv = df_num_tmp[c]

        garbled_mask[c] = (
            (~orig.apply(is_missing_after_clean))
            &
            (conv.isna())
        )

    rows_with_garbled = garbled_mask.any(axis=1)
    garbled_rows_count = int(rows_with_garbled.sum())

    garbled_examples = []

    if garbled_rows_count > 0:
        for idx in df.index[rows_with_garbled]:
            bad_cols = garbled_mask.columns[garbled_mask.loc[idx]].tolist()

            for bc in bad_cols:
                garbled_examples.append({
                    "row_index": idx,
                    "column": bc,
                    "value_after_cleaning": df.loc[idx, bc]
                })

    df = df.loc[~rows_with_garbled].reset_index(drop=True)

    # 数值列统一转数值
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 删除缺失值过多的样品之前，统计每个特征缺失率
    original_missing_summary_raw = pd.DataFrame({
        "Feature": num_cols,
        "Original_missing_count": [
            int(df[c].isna().sum()) for c in num_cols
        ],
        "Original_missing_percent": [
            float(df[c].isna().mean() * 100) for c in num_cols
        ]
    }).sort_values(
        "Original_missing_percent",
        ascending=False
    )

    # 删除每行缺失值数量 > MAX_MISSING_PER_ROW 的样品
    missing_count_per_row = df[num_cols].isna().sum(axis=1)
    rows_too_missing = missing_count_per_row > MAX_MISSING_PER_ROW
    too_missing_count = int(rows_too_missing.sum())

    df = df.loc[~rows_too_missing].reset_index(drop=True)

    # 删除后重新统计缺失率
    original_missing_summary_after_filter = pd.DataFrame({
        "Feature": num_cols,
        "Original_missing_count_after_row_filter": [
            int(df[c].isna().sum()) for c in num_cols
        ],
        "Original_missing_percent_after_row_filter": [
            float(df[c].isna().mean() * 100) for c in num_cols
        ]
    }).sort_values(
        "Original_missing_percent_after_row_filter",
        ascending=False
    )

    basic_summary = pd.DataFrame({
        "Item": [
            "Raw worksheet row count",
            "Detected header row (zero-based)",
            "Documented source changes/exclusions",
            "Raw sample count",
            "Removed rows with unparseable strings",
            f"Removed rows with original missing values > {MAX_MISSING_PER_ROW}",
            "Final sample count after basic cleaning",
            "Numeric feature count"
        ],
        "Value": [
            len(raw_workbook),
            header_row,
            len(source_changes),
            len(df_raw),
            garbled_rows_count,
            too_missing_count,
            len(df),
            len(num_cols)
        ]
    })

    type_count = df[TYPE_COL].value_counts(dropna=False).reset_index()
    type_count.columns = [TYPE_COL, "Count"]

    garbled_examples_df = pd.DataFrame(garbled_examples)

    return {
        "df_clean": df,
        "num_cols": num_cols,
        "basic_summary": basic_summary,
        "type_count": type_count,
        "missing_raw": original_missing_summary_raw,
        "missing_after_filter": original_missing_summary_after_filter,
        "garbled_examples": garbled_examples_df,
        "source_changes": source_changes,
    }


# ============================================================
# 3. 折内 IQR 异常值边界
# ============================================================

def fit_iqr_bounds_on_train(train_df, numeric_cols):
    """
    只用训练集计算每个特征的全局 IQR 异常值边界。
    不使用 Type 标签。
    """
    records = []

    for c in numeric_cols:
        x = train_df[c].astype(float)
        x_valid = x.dropna()

        if len(x_valid) < 4:
            records.append({
                "Feature": c,
                "Q1": np.nan,
                "Q3": np.nan,
                "IQR": np.nan,
                "Lower_bound": np.nan,
                "Upper_bound": np.nan,
                "Note": "Valid values < 4, skipped"
            })
            continue

        q1 = x_valid.quantile(0.25)
        q3 = x_valid.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            records.append({
                "Feature": c,
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "Lower_bound": np.nan,
                "Upper_bound": np.nan,
                "Note": "IQR is 0 or NaN, skipped"
            })
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        records.append({
            "Feature": c,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "Lower_bound": lower,
            "Upper_bound": upper,
            "Note": "OK"
        })

    return pd.DataFrame(records)


def apply_iqr_bounds_mark_nan(df_part, numeric_cols, bounds_df):
    """
    使用训练集计算出的 IQR 边界处理 train 或 test。
    超出边界的值设为 NaN，后续由折内 imputer 填充。
    """
    df_out = df_part.copy()
    records = []

    bounds_map = bounds_df.set_index("Feature").to_dict(orient="index")

    for c in numeric_cols:
        b = bounds_map[c]
        lower = b["Lower_bound"]
        upper = b["Upper_bound"]

        if pd.isna(lower) or pd.isna(upper):
            out_n = 0
        else:
            out_mask = (df_out[c] < lower) | (df_out[c] > upper)
            out_n = int(out_mask.sum())

            if out_n > 0:
                df_out.loc[out_mask, c] = np.nan

        records.append({
            "Feature": c,
            "Outlier_count_marked_as_NaN": out_n
        })

    return df_out, pd.DataFrame(records)


# ============================================================
# 4. 折内插补
# ============================================================

def foldwise_impute(train_df, test_df, numeric_cols, method):
    """
    每一折内的插补。

    method:
    1. global_mean:
       只用训练集 fit SimpleImputer(strategy='mean')，
       然后 transform 训练集和测试集。

    2. knn:
       只用训练集计算标准化参数；
       只用训练集 fit KNNImputer；
       然后 transform 训练集和测试集。
    """
    train_out = train_df.copy()
    test_out = test_df.copy()

    if method == "global_mean":
        imputer = SimpleImputer(strategy="mean")

        train_arr = imputer.fit_transform(train_out[numeric_cols])
        test_arr = imputer.transform(test_out[numeric_cols])

        train_out[numeric_cols] = train_arr
        test_out[numeric_cols] = test_arr

        fill_info = pd.DataFrame({
            "Feature": numeric_cols,
            "Train_missing_filled": [
                int(train_df[c].isna().sum()) for c in numeric_cols
            ],
            "Test_missing_filled": [
                int(test_df[c].isna().sum()) for c in numeric_cols
            ],
            "Imputation_method": "global_mean",
            "Train_mean_used": imputer.statistics_
        })

    elif method == "knn":
        train_input = train_out[numeric_cols].astype(float)
        test_input = test_out[numeric_cols].astype(float)

        all_nan_cols = [
            c for c in numeric_cols
            if train_input[c].isna().all()
        ]

        if len(all_nan_cols) > 0:
            raise ValueError(
                f"训练集中以下特征全为缺失值，无法 KNN 插补：{all_nan_cols}"
            )

        # KNN 对量纲敏感，所以标准化参数只从训练集计算
        train_means = train_input.mean(skipna=True)
        train_stds = train_input.std(skipna=True).replace(0, 1.0)

        train_scaled = (train_input - train_means) / train_stds
        test_scaled = (test_input - train_means) / train_stds

        imputer = KNNImputer(
            n_neighbors=KNN_N_NEIGHBORS,
            weights=KNN_WEIGHTS
        )

        train_scaled_arr = imputer.fit_transform(train_scaled)
        test_scaled_arr = imputer.transform(test_scaled)

        train_filled_arr = (
            train_scaled_arr * train_stds.to_numpy()
            + train_means.to_numpy()
        )

        test_filled_arr = (
            test_scaled_arr * train_stds.to_numpy()
            + train_means.to_numpy()
        )

        train_out[numeric_cols] = train_filled_arr
        test_out[numeric_cols] = test_filled_arr

        fill_info = pd.DataFrame({
            "Feature": numeric_cols,
            "Train_missing_filled": [
                int(train_df[c].isna().sum()) for c in numeric_cols
            ],
            "Test_missing_filled": [
                int(test_df[c].isna().sum()) for c in numeric_cols
            ],
            "Imputation_method": "knn",
            "Train_mean_for_scaling": train_means.values,
            "Train_std_for_scaling": train_stds.values
        })

    else:
        raise ValueError(f"未知插补方法：{method}")

    return train_out, test_out, fill_info


# ============================================================
# 5. 折内比值特征构造
# ============================================================

def ratio_with_zero_denominator_as_zero(numer, denom):
    """
    构造比值。

    当前规则：
    - 分母为 0：比值设为 0，避免 inf；
    - 分子或分母为 NaN：结果为 NaN；
    - inf 和 -inf 替换为 NaN。

    注意：
    如果后续发现分母为 0 的情况较多，
    建议改为分母为 0 -> NaN -> 折内再次插补。
    """
    numer_num = pd.to_numeric(numer, errors="coerce")
    denom_num = pd.to_numeric(denom, errors="coerce")

    zero_mask = denom_num == 0
    zero_count = int(zero_mask.sum())

    ratio = numer_num / denom_num
    ratio = ratio.replace([np.inf, -np.inf], np.nan)

    ratio[zero_mask] = 0.0

    return ratio, zero_count


def build_pairwise_ratios(df, cols, prefix):
    """
    对指定变量列表构造两两比值，只生成组合顺序下的 a/b。

    例如 cols = [A, B, C]，
    生成 A/B, A/C, B/C。
    """
    missing_cols = [c for c in cols if c not in df.columns]

    if missing_cols:
        raise ValueError(
            f"{prefix} 以下列不存在，请检查列名：{missing_cols}"
        )

    data_dict = {}
    zero_records = []

    for a, b in combinations(cols, 2):
        new_col = f"{prefix}{a}/{b}"

        ratio, zero_count = ratio_with_zero_denominator_as_zero(
            df[a],
            df[b]
        )

        data_dict[new_col] = ratio

        if zero_count > 0:
            zero_records.append({
                "Ratio_feature": new_col,
                "Denominator": b,
                "Zero_denominator_count": zero_count
            })

    ratio_df = pd.DataFrame(data_dict, index=df.index)
    zero_summary = pd.DataFrame(zero_records)

    return ratio_df, zero_summary


def add_ratio_features(df):
    """
    添加主量元素比值和微量/稀土元素比值。
    """
    major_ratio_df, major_zero = build_pairwise_ratios(
        df,
        MAJOR_COLS,
        prefix="R_Major_"
    )

    trace_ratio_df, trace_zero = build_pairwise_ratios(
        df,
        TRACE_COLS,
        prefix="R_Trace_"
    )

    df_with_ratios = pd.concat(
        [df, major_ratio_df, trace_ratio_df],
        axis=1
    )

    zero_summary = pd.concat(
        [
            major_zero.assign(Group="Major"),
            trace_zero.assign(Group="Trace_REE")
        ],
        ignore_index=True
    )

    return df_with_ratios, zero_summary


# ============================================================
# 6. CV splitter
# ============================================================

def get_cv_splitter():
    """
    获取交叉验证划分器。
    """
    if N_REPEATS == 1:
        return StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_SEED
        )

    return RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_SEED
    )


# ============================================================
# 7. 模型特征列检查
# ============================================================

def get_model_feature_cols(df):
    """
    获取模型可用特征列。
    排除 No.、Sample、Type 等非建模列。
    """
    exclude_cols = set(NON_NUM_COLS)

    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols
    ]

    return feature_cols


def check_no_nan_inf_in_features(df, feature_cols, name):
    """
    检查建模特征中是否存在 NaN 或 inf。
    不检查 Type、Sample 等非建模列。
    """
    arr = df[feature_cols].to_numpy(dtype=float)

    nan_total = int(np.isnan(arr).sum())
    inf_total = int(np.isinf(arr).sum())

    if nan_total > 0:
        raise ValueError(f"{name} 的建模特征中仍存在 NaN：{nan_total}")

    if inf_total > 0:
        raise ValueError(f"{name} 的建模特征中仍存在 inf：{inf_total}")


# ============================================================
# 8. 主函数
# ============================================================

def main():
    # --------------------------------------------------------
    # 8.1 原始基础清洗
    # --------------------------------------------------------
    clean_result = load_and_basic_clean(IN_PATH)

    df_clean = clean_result["df_clean"]
    num_cols = clean_result["num_cols"]

    summary_clean_path = os.path.join(
        OUT_DIR,
        "basic_cleaning_summary.xlsx"
    )

    with pd.ExcelWriter(summary_clean_path, engine="openpyxl") as writer:
        clean_result["basic_summary"].to_excel(
            writer,
            sheet_name="basic_summary",
            index=False
        )

        clean_result["type_count"].to_excel(
            writer,
            sheet_name="type_count_after_cleaning",
            index=False
        )

        clean_result["source_changes"].to_excel(
            writer,
            sheet_name="source_changes",
            index=False
        )

        clean_result["missing_raw"].to_excel(
            writer,
            sheet_name="missing_raw",
            index=False
        )

        clean_result["missing_after_filter"].to_excel(
            writer,
            sheet_name="missing_after_filter",
            index=False
        )

        if not clean_result["garbled_examples"].empty:
            clean_result["garbled_examples"].to_excel(
                writer,
                sheet_name="garbled_examples",
                index=False
            )
        else:
            pd.DataFrame({
                "Message": ["No unparseable strings after cleaning."]
            }).to_excel(
                writer,
                sheet_name="garbled_examples",
                index=False
            )

    print(f"✅ 原始清洗完成：{len(df_clean)} 个样品")
    print(f"✅ 原始清洗统计已保存：{summary_clean_path}")

    # --------------------------------------------------------
    # 8.2 构造交叉验证
    # --------------------------------------------------------
    X_dummy = df_clean[num_cols]
    y = df_clean[TYPE_COL].astype(str)

    cv = get_cv_splitter()

    fold_summary_records = []
    split_index_records = []

    # --------------------------------------------------------
    # 8.3 不同插补方法分别做折内处理
    # --------------------------------------------------------
    for method in IMPUTATION_METHODS:
        method_dir = os.path.join(OUT_DIR, method)
        os.makedirs(method_dir, exist_ok=True)

        for fold_id, (train_idx, test_idx) in enumerate(
            cv.split(X_dummy, y),
            start=1
        ):
            train_df_raw = (
                df_clean.iloc[train_idx]
                .copy()
                .reset_index(drop=True)
            )

            test_df_raw = (
                df_clean.iloc[test_idx]
                .copy()
                .reset_index(drop=True)
            )

            split_index_records.append({
                "Method": method,
                "Fold": fold_id,
                "Train_indices": ",".join(map(str, train_idx.tolist())),
                "Test_indices": ",".join(map(str, test_idx.tolist()))
            })

            # ------------------------------------------------
            # 1. 只用训练集计算 IQR 异常值边界
            # ------------------------------------------------
            iqr_bounds = fit_iqr_bounds_on_train(
                train_df_raw,
                num_cols
            )

            # ------------------------------------------------
            # 2. 用训练集 IQR 边界处理 train/test 异常值
            # ------------------------------------------------
            train_nan, train_outlier_summary = apply_iqr_bounds_mark_nan(
                train_df_raw,
                num_cols,
                iqr_bounds
            )

            test_nan, test_outlier_summary = apply_iqr_bounds_mark_nan(
                test_df_raw,
                num_cols,
                iqr_bounds
            )

            # ------------------------------------------------
            # 3. 只用训练集 fit imputer，然后 transform train/test
            # ------------------------------------------------
            train_imp, test_imp, fill_info = foldwise_impute(
                train_nan,
                test_nan,
                num_cols,
                method=method
            )

            # ------------------------------------------------
            # 4. 每一折内部添加比值特征
            # ------------------------------------------------
            train_ratio, train_zero_summary = add_ratio_features(train_imp)
            test_ratio, test_zero_summary = add_ratio_features(test_imp)

            # ------------------------------------------------
            # 5. 检查建模特征中是否还有 NaN / inf
            # ------------------------------------------------
            feature_cols = get_model_feature_cols(train_ratio)

            check_no_nan_inf_in_features(
                train_ratio,
                feature_cols,
                f"{method} Fold {fold_id} train"
            )

            check_no_nan_inf_in_features(
                test_ratio,
                feature_cols,
                f"{method} Fold {fold_id} test"
            )

            # ------------------------------------------------
            # 6. 保存每折 train/test with ratios
            # ------------------------------------------------
            train_out_path = os.path.join(
                method_dir,
                f"fold_{fold_id:02d}_train_with_ratios.xlsx"
            )

            test_out_path = os.path.join(
                method_dir,
                f"fold_{fold_id:02d}_test_with_ratios.xlsx"
            )

            train_ratio.to_excel(train_out_path, index=False)
            test_ratio.to_excel(test_out_path, index=False)

            # ------------------------------------------------
            # 7. 保存每折预处理细节
            # ------------------------------------------------
            fold_stat_path = os.path.join(
                method_dir,
                f"fold_{fold_id:02d}_preprocessing_details.xlsx"
            )

            with pd.ExcelWriter(fold_stat_path, engine="openpyxl") as writer:
                iqr_bounds.to_excel(
                    writer,
                    sheet_name="train_iqr_bounds",
                    index=False
                )

                train_outlier_summary.to_excel(
                    writer,
                    sheet_name="train_outliers",
                    index=False
                )

                test_outlier_summary.to_excel(
                    writer,
                    sheet_name="test_outliers",
                    index=False
                )

                fill_info.to_excel(
                    writer,
                    sheet_name="imputation_info",
                    index=False
                )

                if not train_zero_summary.empty:
                    train_zero_summary.to_excel(
                        writer,
                        sheet_name="train_ratio_zero_den",
                        index=False
                    )
                else:
                    pd.DataFrame({
                        "Message": ["No zero denominator in train ratios."]
                    }).to_excel(
                        writer,
                        sheet_name="train_ratio_zero_den",
                        index=False
                    )

                if not test_zero_summary.empty:
                    test_zero_summary.to_excel(
                        writer,
                        sheet_name="test_ratio_zero_den",
                        index=False
                    )
                else:
                    pd.DataFrame({
                        "Message": ["No zero denominator in test ratios."]
                    }).to_excel(
                        writer,
                        sheet_name="test_ratio_zero_den",
                        index=False
                    )

            fold_summary_records.append({
                "Method": method,
                "Fold": fold_id,

                "Train_sample_count": len(train_ratio),
                "Test_sample_count": len(test_ratio),

                "Train_class_A": int(
                    (train_ratio[TYPE_COL].astype(str) == "A").sum()
                ),
                "Train_class_S": int(
                    (train_ratio[TYPE_COL].astype(str) == "S").sum()
                ),
                "Train_class_I": int(
                    (train_ratio[TYPE_COL].astype(str) == "I").sum()
                ),

                "Test_class_A": int(
                    (test_ratio[TYPE_COL].astype(str) == "A").sum()
                ),
                "Test_class_S": int(
                    (test_ratio[TYPE_COL].astype(str) == "S").sum()
                ),
                "Test_class_I": int(
                    (test_ratio[TYPE_COL].astype(str) == "I").sum()
                ),

                "Train_outliers_marked_as_NaN": int(
                    train_outlier_summary[
                        "Outlier_count_marked_as_NaN"
                    ].sum()
                ),
                "Test_outliers_marked_as_NaN": int(
                    test_outlier_summary[
                        "Outlier_count_marked_as_NaN"
                    ].sum()
                ),

                "Train_values_filled": int(
                    fill_info["Train_missing_filled"].sum()
                ),
                "Test_values_filled": int(
                    fill_info["Test_missing_filled"].sum()
                ),

                "Feature_column_count_with_ratios": len(feature_cols),
                "Final_column_count_with_metadata": train_ratio.shape[1],

                "Train_file": train_out_path,
                "Test_file": test_out_path,
                "Detail_file": fold_stat_path
            })

            print(
                f"✅ {method} Fold {fold_id}: "
                f"train={len(train_ratio)}, test={len(test_ratio)}, "
                f"features={len(feature_cols)}"
            )

    # --------------------------------------------------------
    # 8.4 保存总 summary
    # --------------------------------------------------------
    fold_summary_df = pd.DataFrame(fold_summary_records)
    split_index_df = pd.DataFrame(split_index_records)

    summary_out_path = os.path.join(
        OUT_DIR,
        "foldwise_preprocessing_summary.xlsx"
    )

    split_out_path = os.path.join(
        OUT_DIR,
        "cv_split_indices.xlsx"
    )

    fold_summary_df.to_excel(summary_out_path, index=False)
    split_index_df.to_excel(split_out_path, index=False)

    print("\n全部完成。")
    print(f"输出目录：{OUT_DIR}")
    print(f"折内预处理总表：{summary_out_path}")
    print(f"CV 分折索引：{split_out_path}")


# ============================================================
# 9. 程序入口
# ============================================================

if __name__ == "__main__":
    main()
