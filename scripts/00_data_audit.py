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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 0. 路径设置
# ============================================================

IN_PATH = RAW_DATA_FILE

OUT_DIR = RESULTS_DIR / "00_data_audit"
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


# ============================================================
# 1. 全局画图字体设置
# ============================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.titlesize"] = 22
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 15
plt.rcParams["figure.titlesize"] = 22


# ============================================================
# 2. 特殊值清理函数：与正式预处理代码保持一致
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


def normalize_type_label(x):
    """
    统一 Type 标签，避免 A-type、A、a、I-type 等写法造成统计失败。
    """
    if pd.isna(x):
        return np.nan

    s = str(x).strip().upper()
    s = s.replace(" ", "")
    s = s.replace("_", "-")

    if s in {"A", "A-TYPE", "ATYPE"}:
        return "A"

    if s in {"S", "S-TYPE", "STYPE"}:
        return "S"

    if s in {"I", "I-TYPE", "ITYPE"}:
        return "I"

    return s


# ============================================================
# 3. 读取原始数据
# ============================================================

df_raw, source_changes, raw_workbook, header_row = prepare_source_data(IN_PATH)
df = df_raw.copy()

df.columns = [str(c).strip() for c in df.columns]

if TYPE_COL not in df.columns:
    raise ValueError(f"未找到类型列 {TYPE_COL}，当前列名：{df.columns.tolist()}")

num_cols = [c for c in df.columns if c not in NON_NUM_COLS]

if len(num_cols) == 0:
    raise ValueError("没有识别到数值特征列，请检查 NON_NUM_COLS 或 Excel 表头。")

print("=" * 80)
print("原始数据读取完成")
print(f"原始样品数: {len(df_raw)}")
print(f"数值特征数: {len(num_cols)}")
print("=" * 80)


# ============================================================
# 4. 处理特殊值
# ============================================================

for c in num_cols:
    df[c] = df[c].apply(clean_censored_value)

df_num_tmp = df[num_cols].apply(pd.to_numeric, errors="coerce")


# ============================================================
# 5. 识别并删除仍无法解析为数值的非空字符串所在行
# ============================================================

garbled_mask = pd.DataFrame(False, index=df.index, columns=num_cols)

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

for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

print(f"删除无法解析字符串所在行数: {garbled_rows_count}")


# ============================================================
# 6. 删除缺失过多样品之前：统计每个特征缺失比例
# ============================================================

missing_before_row_filter = pd.DataFrame({
    "Feature": num_cols,
    "Missing_count_before_row_filter": [
        int(df[c].isna().sum()) for c in num_cols
    ],
    "Missing_percent_before_row_filter": [
        float(df[c].isna().mean() * 100) for c in num_cols
    ],
    "Valid_count_before_row_filter": [
        int(df[c].notna().sum()) for c in num_cols
    ],
}).sort_values(
    "Missing_percent_before_row_filter",
    ascending=False
).reset_index(drop=True)

missing_before_only = missing_before_row_filter[
    missing_before_row_filter["Missing_count_before_row_filter"] > 0
].copy()


# ============================================================
# 7. 删除每行缺失值数量 > 3 的样品
# ============================================================

missing_count_per_row = df[num_cols].isna().sum(axis=1)

rows_too_missing = missing_count_per_row > MAX_MISSING_PER_ROW
too_missing_count = int(rows_too_missing.sum())

df_clean = df.loc[~rows_too_missing].reset_index(drop=True)

print(f"删除缺失特征数 > {MAX_MISSING_PER_ROW} 的样品数: {too_missing_count}")
print(f"基础清洗后样品数: {len(df_clean)}")


# ============================================================
# 8. 删除缺失过多样品之后：统计每个特征缺失比例
# ============================================================

missing_after_row_filter = pd.DataFrame({
    "Feature": num_cols,
    "Missing_count_after_row_filter": [
        int(df_clean[c].isna().sum()) for c in num_cols
    ],
    "Missing_percent_after_row_filter": [
        float(df_clean[c].isna().mean() * 100) for c in num_cols
    ],
    "Valid_count_after_row_filter": [
        int(df_clean[c].notna().sum()) for c in num_cols
    ],
}).sort_values(
    "Missing_percent_after_row_filter",
    ascending=False
).reset_index(drop=True)

# 只保留有缺失值的特征
missing_after_only = missing_after_row_filter[
    missing_after_row_filter["Missing_count_after_row_filter"] > 0
].copy()

print("\n删除缺失过多样品后，仍存在缺失值的特征数:")
print(len(missing_after_only))

if len(missing_after_only) > 0:
    print("\n存在缺失值的特征:")
    print(missing_after_only[[
        "Feature",
        "Missing_count_after_row_filter",
        "Missing_percent_after_row_filter"
    ]])
else:
    print("\n删除缺失过多样品后，没有任何特征存在缺失值。")


# ============================================================
# 9. 统计基础清洗后 A/S/I 类别数量和比例
# ============================================================

df_clean[TYPE_COL] = df_clean[TYPE_COL].apply(normalize_type_label)

type_order = ["A", "S", "I"]

type_counts_raw = df_clean[TYPE_COL].value_counts(dropna=False)

type_counts = (
    df_clean[TYPE_COL]
    .value_counts()
    .reindex(type_order)
    .fillna(0)
    .astype(int)
)

type_summary = pd.DataFrame({
    "Type": type_counts.index,
    "Count": type_counts.values
})

total_type_count = int(type_summary["Count"].sum())

if total_type_count > 0:
    type_summary["Percent"] = type_summary["Count"] / total_type_count * 100
else:
    type_summary["Percent"] = 0.0

print("\n原始 Type 统计:")
print(type_counts_raw)

print("\n基础清洗后类别分布:")
print(type_summary)


# ============================================================
# 10. 保存 Excel 汇总表
# ============================================================

summary_excel_path = os.path.join(
    OUT_DIR,
    "missing_ratio_and_cleaned_type_distribution.xlsx"
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
        "Numeric feature count",
        "Features with missing values before row filter",
        "Features with missing values after row filter"
    ],
    "Value": [
        len(raw_workbook),
        header_row,
        len(source_changes),
        len(df_raw),
        garbled_rows_count,
        too_missing_count,
        len(df_clean),
        len(num_cols),
        len(missing_before_only),
        len(missing_after_only)
    ]
})

garbled_examples_df = pd.DataFrame(garbled_examples)

with pd.ExcelWriter(summary_excel_path, engine="openpyxl") as writer:
    basic_summary.to_excel(
        writer,
        sheet_name="basic_summary",
        index=False
    )

    source_changes.to_excel(
        writer,
        sheet_name="source_changes",
        index=False
    )

    missing_before_row_filter.to_excel(
        writer,
        sheet_name="missing_before_all",
        index=False
    )

    missing_before_only.to_excel(
        writer,
        sheet_name="missing_before_only",
        index=False
    )

    missing_after_row_filter.to_excel(
        writer,
        sheet_name="missing_after_all",
        index=False
    )

    missing_after_only.to_excel(
        writer,
        sheet_name="missing_after_only",
        index=False
    )

    type_counts_raw.reset_index().to_excel(
        writer,
        sheet_name="raw_type_counts",
        index=False
    )

    type_summary.to_excel(
        writer,
        sheet_name="type_distribution",
        index=False
    )

    if len(garbled_examples_df) > 0:
        garbled_examples_df.to_excel(
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

print(f"\nExcel 汇总已保存: {summary_excel_path}")


# ============================================================
# 11. 图 1：仅绘制有缺失值的特征缺失比例图
# 横轴 = 特征名
# 纵轴 = 缺失比例
# ============================================================

missing_fig_path = os.path.join(
    OUT_DIR,
    "missing_features_percent_after_basic_cleaning.png"
)

if len(missing_after_only) > 0:
    plot_df = missing_after_only.sort_values(
        "Missing_percent_after_row_filter",
        ascending=False
    ).reset_index(drop=True)

    n_features = len(plot_df)

    fig_width = max(10, n_features * 0.65)
    fig_height = 7.5

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    bars = ax.bar(
        plot_df["Feature"],
        plot_df["Missing_percent_after_row_filter"],
        edgecolor="black",
        linewidth=1.2
    )

    ax.set_xlabel(
        "Geochemical variables with missing values",
        fontsize=19,
        fontweight="bold",
        labelpad=12
    )

    ax.set_ylabel(
        "Missing values before imputation (%)",
        fontsize=19,
        fontweight="bold",
        labelpad=12
    )

    ax.set_title(
        "Feature-wise Missing Proportions After Basic Cleaning",
        fontsize=23,
        fontweight="bold",
        pad=12
    )

    ax.tick_params(axis="x", labelsize=15, width=1.5)
    ax.tick_params(axis="y", labelsize=15, width=1.5)

    plt.xticks(rotation=45, ha="right", fontweight="bold")
    plt.yticks(fontweight="bold")

    max_value = float(plot_df["Missing_percent_after_row_filter"].max())

    if max_value <= 0:
        max_value = 1.0

    ax.set_ylim(0, max_value * 1.22)

    for bar, value in zip(bars, plot_df["Missing_percent_after_row_filter"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_value * 0.025,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            rotation=0
        )

    ax.grid(axis="y", linestyle="--", alpha=0.45)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(missing_fig_path, dpi=1000, bbox_inches="tight")
    plt.close()

    print(f"缺失比例图已保存: {missing_fig_path}")

else:
    no_missing_txt = os.path.join(
        OUT_DIR,
        "no_missing_features_after_basic_cleaning.txt"
    )

    with open(no_missing_txt, "w", encoding="utf-8") as f:
        f.write(
            "After basic cleaning and row-level missingness filtering, "
            "no feature has missing values before imputation.\n"
        )

    print("没有需要绘制的缺失特征，因此未生成缺失比例图。")
    print(f"提示文件已保存: {no_missing_txt}")


# ============================================================
# 12. 图 2：基础清洗后 A/S/I 样品数量比例饼图
# ============================================================

pie_fig_path = os.path.join(
    OUT_DIR,
    "type_distribution_after_basic_cleaning_pie.png"
)

pie_df = type_summary[type_summary["Count"] > 0].copy()

if len(pie_df) > 0 and int(pie_df["Count"].sum()) > 0:
    fig, ax = plt.subplots(figsize=(7.2, 7.2))

    labels = [
        f"{row.Type}-type\nn={row.Count}\n{row.Percent:.1f}%"
        for row in pie_df.itertuples(index=False)
    ]

    wedges, texts = ax.pie(
        pie_df["Count"].values,
        labels=labels,
        startangle=90,
        counterclock=False,
        labeldistance=1.08,
        textprops={
            "fontsize": 16,
            "fontweight": "bold"
        },
        wedgeprops={
            "edgecolor": "black",
            "linewidth": 1.2
        }
    )

    ax.set_title(
        "A/S/I Type Distribution After Basic Cleaning",
        fontsize=23,
        fontweight="bold",
        pad=4
    )

    ax.axis("equal")

    # subplots_adjust 控制标题与图之间距离
    plt.subplots_adjust(top=0.91, bottom=0.04)

    plt.savefig(pie_fig_path, dpi=1000, bbox_inches="tight")
    plt.close()

    print(f"类别比例饼图已保存: {pie_fig_path}")

else:
    no_type_txt = os.path.join(
        OUT_DIR,
        "no_valid_type_distribution_for_pie.txt"
    )

    with open(no_type_txt, "w", encoding="utf-8") as f:
        f.write(
            "No valid A/S/I type counts were found after basic cleaning. "
            "Please check the Type column labels.\n"
        )

    print("没有有效 A/S/I 类别数量，因此未生成饼图。")
    print(f"提示文件已保存: {no_type_txt}")


# ============================================================
# 13. 保存基础清洗后的数据
# ============================================================

cleaned_data_path = os.path.join(
    OUT_DIR,
    "basic_cleaned_data_before_imputation.xlsx"
)

df_clean.to_excel(cleaned_data_path, index=False)

print(f"基础清洗后、插补前数据已保存: {cleaned_data_path}")


print("\n全部完成。")
print(f"输出目录: {OUT_DIR}")
