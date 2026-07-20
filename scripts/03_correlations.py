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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors


# ============================================================
# 0. 配置
# ============================================================

INPUT_ROOT = FOLDS_DIR

OUT_DIR = RESULTS_DIR / "03_correlations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMPUTATION_METHODS = ["global_mean", "knn"]
N_SPLITS = 5

NON_NUM_COLS = {
    "No.",
    "Samp1e",
    "Sample",
    "Type",
    "Type-1",
    "Type-2",
    "Reference",
}

RHO_THRESHOLDS = [0.75, 0.85, 0.90, 0.95]
METHOD_LABELS = {"global_mean": "GM", "knn": "KNN"}

SAVE_EACH_FOLD_HEATMAP = True
SAVE_MEAN_HEATMAP = True

DPI = 1000

FONT_TITLE = 28
FONT_LABEL = 24
FONT_TICK = 8
FONT_CBAR = 14

plt.rcParams.update({
    "font.weight": "bold",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "xtick.labelsize": FONT_TICK,
    "ytick.labelsize": FONT_TICK,
})


# ============================================================
# 1. 工具函数
# ============================================================

def get_feature_matrix(df):
    """
    从 train_with_ratios 数据中提取建模特征矩阵。
    排除 Type、Sample、No. 等非数值列。
    删除全 NaN 列和常数列。
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    candidate_cols = [
        c for c in df.columns
        if c not in NON_NUM_COLS
    ]

    X = df[candidate_cols].copy()

    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)

    # 删除全 NaN 列
    X = X.loc[:, X.notna().any(axis=0)]

    # 删除常数列
    X = X.loc[:, X.nunique(dropna=True) > 1]

    return X


def compute_spearman_corr(X):
    """
    计算 Spearman 相关矩阵。
    """
    return X.corr(method="spearman")


def count_high_corr_pairs(corr, thresholds):
    """
    统计不同 |rho| 阈值下的高相关特征对数量。
    只统计上三角，不重复统计，不包含对角线。
    """
    values = corr.values
    n = values.shape[0]

    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    upper_vals = values[upper_mask]
    upper_vals = upper_vals[~np.isnan(upper_vals)]

    records = {}

    for th in thresholds:
        key = f"Pair_count_abs_rho_ge_{th:.2f}"
        records[key] = int(np.sum(np.abs(upper_vals) >= th))

    records["Mean_abs_rho_upper_triangle"] = (
        float(np.mean(np.abs(upper_vals))) if len(upper_vals) > 0 else np.nan
    )

    records["Median_abs_rho_upper_triangle"] = (
        float(np.median(np.abs(upper_vals))) if len(upper_vals) > 0 else np.nan
    )

    return records


def save_high_corr_pairs(corr, out_xlsx, threshold=0.90):
    """
    保存 |rho| >= threshold 的高相关特征对明细。
    """
    cols = corr.columns.tolist()
    values = corr.values
    n = len(cols)

    rows = []

    for i in range(n):
        for j in range(i + 1, n):
            rho = values[i, j]

            if pd.isna(rho):
                continue

            if abs(rho) >= threshold:
                rows.append({
                    "Feature_1": cols[i],
                    "Feature_2": cols[j],
                    "Spearman_rho": float(rho),
                    "Abs_rho": float(abs(rho))
                })

    pair_df = pd.DataFrame(rows)

    if not pair_df.empty:
        pair_df = pair_df.sort_values(
            by="Abs_rho",
            ascending=False
        )

    pair_df.to_excel(out_xlsx, index=False)

    return pair_df


def save_heatmap(corr, out_png, title):
    """
    保存 Spearman 相关热图。
    """
    n = corr.shape[0]

    if n <= 60:
        figsize = (12, 10)
    elif n <= 120:
        figsize = (14, 12)
    else:
        figsize = (16, 14)

    plt.figure(figsize=figsize, dpi=DPI)

    norm = colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    im = plt.imshow(
        corr.values,
        aspect="auto",
        interpolation="nearest",
        cmap="Spectral_r",
        norm=norm
    )

    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label(
        "Spearman Correlation (ρ)",
        fontsize=FONT_CBAR,
        fontweight="bold"
    )
    cbar.ax.tick_params(labelsize=FONT_TICK)

    for t in cbar.ax.get_yticklabels():
        t.set_fontweight("bold")

    plt.title(
        title,
        fontsize=FONT_TITLE,
        fontweight="bold",
        pad=12
    )

    if n <= 60:
        plt.xticks(range(n), corr.columns, rotation=90)
        plt.yticks(range(n), corr.index)
    elif n <= 120:
        step = 2
        plt.xticks(range(0, n, step), corr.columns[::step], rotation=90)
        plt.yticks(range(0, n, step), corr.index[::step])
    else:
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("Features", fontsize=FONT_LABEL, fontweight="bold")
        plt.ylabel("Features", fontsize=FONT_LABEL, fontweight="bold")

    ax = plt.gca()

    for lbl in ax.get_xticklabels():
        lbl.set_fontweight("bold")

    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def save_threshold_pair_count_plot(summary_df, out_png):
    """
    保存不同 |rho| 阈值下高相关特征对数量图。
    每种插补方法分别取 5 折均值和标准差。
    """
    rows = []

    for method in summary_df["Method"].unique():
        sub = summary_df[summary_df["Method"] == method]

        for th in RHO_THRESHOLDS:
            col = f"Pair_count_abs_rho_ge_{th:.2f}"
            rows.append({
                "Method": method,
                "Threshold": th,
                "Mean_pair_count": sub[col].mean(),
                "SD_pair_count": sub[col].std(ddof=1)
            })

    plot_df = pd.DataFrame(rows)

    plt.figure(figsize=(10, 7), dpi=DPI)

    methods = plot_df["Method"].unique().tolist()
    x = np.arange(len(RHO_THRESHOLDS))
    width = 0.35 if len(methods) == 2 else 0.8 / len(methods)

    for i, method in enumerate(methods):
        sub = plot_df[plot_df["Method"] == method].sort_values("Threshold")
        offset = (i - (len(methods) - 1) / 2) * width

        plt.bar(
            x + offset,
            sub["Mean_pair_count"],
            width=width,
            yerr=sub["SD_pair_count"],
            capsize=4,
            label=METHOD_LABELS.get(method, method)
        )

    predefined_x = RHO_THRESHOLDS.index(0.90)
    plt.axvline(
        predefined_x,
        color="#333333",
        linestyle="--",
        linewidth=2,
    )
    plt.text(
        predefined_x + 0.04,
        0.97,
        "Predefined threshold",
        transform=plt.gca().get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=13,
        rotation=90,
        color="#333333",
    )

    plt.xticks(
        x,
        [f"|ρ| ≥ {th:.2f}" for th in RHO_THRESHOLDS],
        fontsize=14,
        fontweight="bold"
    )

    plt.ylabel(
        "Number of highly correlated feature pairs",
        fontsize=18,
        fontweight="bold"
    )

    plt.xlabel(
        r"Absolute Spearman correlation threshold, $|\rho_s|$",
        fontsize=18,
        fontweight="bold"
    )

    plt.title(
        "High-correlation Feature Pairs Across Thresholds",
        fontsize=20,
        fontweight="bold"
    )

    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

    return plot_df


def save_abs_rho_summary_plot(summary_df, out_png):
    """
    保存每折平均/中位数 |rho| 图。
    """
    plt.figure(figsize=(10, 7), dpi=DPI)

    for method in summary_df["Method"].unique():
        sub = summary_df[summary_df["Method"] == method].sort_values("Fold")
        method = METHOD_LABELS.get(method, method)

        plt.plot(
            sub["Fold"],
            sub["Mean_abs_rho_upper_triangle"],
            marker="o",
            linewidth=2,
            label=f"{method} mean |ρ|"
        )

        plt.plot(
            sub["Fold"],
            sub["Median_abs_rho_upper_triangle"],
            marker="s",
            linewidth=2,
            linestyle="--",
            label=f"{method} median |ρ|"
        )

    plt.xlabel("Fold", fontsize=18, fontweight="bold")
    plt.ylabel("Absolute Spearman correlation", fontsize=18, fontweight="bold")
    plt.title("Fold-wise Summary of Absolute Spearman Correlations", fontsize=20, fontweight="bold")

    plt.xticks(sorted(summary_df["Fold"].unique()), fontsize=14, fontweight="bold")
    plt.yticks(fontsize=14, fontweight="bold")

    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def average_correlation_matrices(corr_list):
    """
    对多个 fold 的相关矩阵取平均。
    要求特征列一致。
    """
    if len(corr_list) == 0:
        raise ValueError("corr_list 为空，无法计算平均相关矩阵。")

    base_cols = corr_list[0].columns.tolist()

    aligned_values = []

    for corr in corr_list:
        corr_aligned = corr.loc[base_cols, base_cols]
        aligned_values.append(corr_aligned.values)

    mean_values = np.nanmean(np.stack(aligned_values, axis=0), axis=0)

    mean_corr = pd.DataFrame(
        mean_values,
        index=base_cols,
        columns=base_cols
    )

    return mean_corr


# ============================================================
# 2. 主函数
# ============================================================

def main():
    fold_summary_records = []

    for method in IMPUTATION_METHODS:
        method_input_dir = os.path.join(INPUT_ROOT, method)

        if not os.path.exists(method_input_dir):
            raise FileNotFoundError(f"未找到输入目录：{method_input_dir}")

        method_out_dir = os.path.join(OUT_DIR, method)
        matrix_dir = os.path.join(method_out_dir, "correlation_matrices")
        heatmap_dir = os.path.join(method_out_dir, "heatmaps")
        pair_dir = os.path.join(method_out_dir, "high_corr_pairs")
        figure_dir = os.path.join(method_out_dir, "figures")

        os.makedirs(method_out_dir, exist_ok=True)
        os.makedirs(matrix_dir, exist_ok=True)
        os.makedirs(heatmap_dir, exist_ok=True)
        os.makedirs(pair_dir, exist_ok=True)
        os.makedirs(figure_dir, exist_ok=True)

        corr_list = []

        for fold in range(1, N_SPLITS + 1):
            train_path = os.path.join(
                method_input_dir,
                f"fold_{fold:02d}_train_with_ratios.xlsx"
            )

            if not os.path.exists(train_path):
                raise FileNotFoundError(f"未找到训练折文件：{train_path}")

            df_train = pd.read_excel(train_path)
            X_train = get_feature_matrix(df_train)

            corr = compute_spearman_corr(X_train)
            corr_list.append(corr)

            corr_out = os.path.join(
                matrix_dir,
                f"fold_{fold:02d}_spearman_correlation_matrix.xlsx"
            )

            corr.to_excel(corr_out, index=True)

            high_pair_out = os.path.join(
                pair_dir,
                f"fold_{fold:02d}_high_corr_pairs_abs_rho_ge_0.90.xlsx"
            )

            high_pair_df = save_high_corr_pairs(
                corr,
                high_pair_out,
                threshold=0.90
            )

            heatmap_out = os.path.join(
                heatmap_dir,
                f"fold_{fold:02d}_spearman_correlation_heatmap.png"
            )

            if SAVE_EACH_FOLD_HEATMAP:
                save_heatmap(
                    corr,
                    heatmap_out,
                    title=f"Spearman Correlation Heatmap ({method}, Fold {fold})"
                )
            else:
                heatmap_out = "Not saved"

            high_corr_stats = count_high_corr_pairs(
                corr,
                thresholds=RHO_THRESHOLDS
            )

            summary_row = {
                "Method": method,
                "Fold": fold,
                "Input_train_file": train_path,
                "Feature_count_used": X_train.shape[1],
                "Sample_count_train": X_train.shape[0],
                "Correlation_matrix_file": corr_out,
                "Heatmap_file": heatmap_out,
                "High_corr_pairs_abs_rho_ge_0.90_file": high_pair_out,
                "High_corr_pairs_abs_rho_ge_0.90_count": len(high_pair_df)
            }

            summary_row.update(high_corr_stats)

            fold_summary_records.append(summary_row)

            print(
                f"✅ {method} Fold {fold}: "
                f"features={X_train.shape[1]}, "
                f"|ρ|≥0.90 pairs={len(high_pair_df)}"
            )

        # 五折平均相关矩阵和平均热图
        mean_corr = average_correlation_matrices(corr_list)

        mean_corr_out = os.path.join(
            matrix_dir,
            f"{method}_mean_spearman_correlation_matrix_across_folds.xlsx"
        )

        mean_corr.to_excel(mean_corr_out, index=True)

        if SAVE_MEAN_HEATMAP:
            mean_heatmap_out = os.path.join(
                figure_dir,
                f"{method}_mean_spearman_correlation_heatmap_across_folds.png"
            )

            save_heatmap(
                mean_corr,
                mean_heatmap_out,
                title=f"Mean Spearman Correlation Heatmap Across Folds ({method})"
            )

            print(f"✅ {method}: 五折平均 Spearman 热图已保存：{mean_heatmap_out}")

    summary_df = pd.DataFrame(fold_summary_records)

    summary_out = os.path.join(
        OUT_DIR,
        "foldwise_spearman_correlation_summary.xlsx"
    )

    summary_df.to_excel(summary_out, index=False)

    # 汇总结果图输出目录
    summary_figure_dir = os.path.join(OUT_DIR, "summary_figures")
    os.makedirs(summary_figure_dir, exist_ok=True)

    threshold_plot_out = os.path.join(
        summary_figure_dir,
        "high_corr_pair_counts_across_thresholds.png"
    )

    plot_df = save_threshold_pair_count_plot(
        summary_df,
        threshold_plot_out
    )

    threshold_plot_data_out = os.path.join(
        summary_figure_dir,
        "high_corr_pair_counts_across_thresholds_data.xlsx"
    )

    plot_df.to_excel(threshold_plot_data_out, index=False)

    abs_rho_plot_out = os.path.join(
        summary_figure_dir,
        "foldwise_abs_spearman_rho_summary.png"
    )

    save_abs_rho_summary_plot(
        summary_df,
        abs_rho_plot_out
    )

    print("\n全部 Spearman 相关性分析完成。")
    print(f"输出目录：{OUT_DIR}")
    print(f"汇总表：{summary_out}")
    print(f"高相关特征对数量图：{threshold_plot_out}")
    print(f"折内 |ρ| 汇总图：{abs_rho_plot_out}")

    print("\n简要汇总：")

    cols_to_show = [
        "Method",
        "Fold",
        "Feature_count_used",
        "Sample_count_train",
        "Pair_count_abs_rho_ge_0.90",
        "Mean_abs_rho_upper_triangle",
        "Median_abs_rho_upper_triangle"
    ]

    existing_cols = [c for c in cols_to_show if c in summary_df.columns]

    print(summary_df[existing_cols])


# ============================================================
# 3. 程序入口
# ============================================================

if __name__ == "__main__":
    main()


