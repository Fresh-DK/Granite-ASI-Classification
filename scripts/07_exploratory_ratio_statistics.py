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
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats


# ============================================================
# 0. 路径配置
# ============================================================

RESULT_BASE_DIR = RESULTS_DIR

DATA_ROOT = FOLDS_DIR

OUT_DIR = RESULTS_DIR / "07_exploratory_ratio_statistics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_DIR = os.path.join(OUT_DIR, "figures_boxplots")
os.makedirs(FIG_DIR, exist_ok=True)

TYPE_COL = "Type"

IMPUTATION_METHODS = ["global_mean", "knn"]
N_OUTER_FOLDS = 5

# A/S/I 显示顺序
CLASS_ORDER = ["A", "S", "I"]

# 从 05 稳定候选比值中选择 Top N 个做图和统计
TOP_N_RATIOS = 20

# 最低出现次数。global_mean 5 folds + knn 5 folds = 10
MIN_APPEARANCE_COUNT = 6

# 如果你想手动指定要分析的比值，把 USE_MANUAL_FEATURES 改为 True
USE_MANUAL_FEATURES = False

MANUAL_FEATURES = [
    "R_Trace_Yb/Lu",
    "R_Major_TiO2/MgO",
    "R_Trace_Ho/Er",
    "R_Major_Fe2O3t/P2O5",
    "R_Major_TiO2/P2O5",
    "R_Trace_Y/Dy",
    "R_Trace_Y/Er",
    "R_Trace_Tm/Yb",
    "R_Trace_Dy/Ho",
    "R_Trace_Sm/Gd",
    "R_Trace_Ta/Th",
    "R_Trace_Nb/Th",
    "R_Trace_Nb/U",
    "R_Trace_Rb/U",
    "R_Trace_Ba/Eu",
]

DPI = 1000
ALPHA = 0.05


# ============================================================
# 1. 工具函数
# ============================================================

def find_dir_by_prefix(base_dir, prefix):
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"目录不存在：{base_dir}")

    candidates = []
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path) and name.startswith(prefix):
            candidates.append(path)

    if len(candidates) == 0:
        raise FileNotFoundError(f"在 {base_dir} 下未找到以 {prefix} 开头的文件夹。")

    if len(candidates) > 1:
        print(f"警告：找到多个 {prefix} 文件夹，将使用第一个：")
        for c in candidates:
            print("  ", c)

    return candidates[0]


def find_stability_summary_file(stability_root):
    candidates = [
        os.path.join(stability_root, f)
        for f in os.listdir(stability_root)
        if f.endswith(".xlsx") and "stable_champions" in f
    ]

    if len(candidates) == 0:
        candidates = [
            os.path.join(stability_root, f)
            for f in os.listdir(stability_root)
            if f.endswith(".xlsx")
        ]

    if len(candidates) == 0:
        raise FileNotFoundError(f"05 汇总目录中未找到 xlsx：{stability_root}")

    return candidates[0]


def display_feature_name(name):
    s = str(name)

    s = s.replace("10000*Ga/A1", "10000×Ga/Al")
    s = s.replace("10000*Ga/Al", "10000×Ga/Al")
    s = s.replace("A12O3", "Al2O3")

    s = s.replace("R_Major_", "")
    s = s.replace("R_Trace_", "")

    return s


def safe_filename(name):
    s = display_feature_name(name)
    s = re.sub(r"[\\/:*?\"<>|]", "_", s)
    s = s.replace("×", "x")
    s = s.replace("+", "plus")
    s = s.replace(" ", "_")
    return s


def is_constructed_ratio(feature):
    s = str(feature)
    return s.startswith("R_Major_") or s.startswith("R_Trace_")


def ratio_group(feature):
    s = str(feature)

    if not is_constructed_ratio(s):
        return "Original/classical feature"

    if s.startswith("R_Major_"):
        return "Major-element ratio"

    body = s.replace("R_Trace_", "")

    hfse = {"Nb", "Ta", "Zr", "Hf", "Ti", "Y"}
    ree = {
        "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd",
        "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"
    }
    lile = {"Rb", "Sr", "Ba", "Cs", "Pb"}
    th_u = {"Th", "U"}

    parts = re.split(r"[/_+\-*()]+", body)
    parts = [p for p in parts if p]

    has_hfse = any(p in hfse for p in parts)
    has_ree = any(p in ree for p in parts)
    has_lile = any(p in lile for p in parts)
    has_thu = any(p in th_u for p in parts)

    if has_hfse and has_ree:
        return "HFSE-REE ratio"
    if has_hfse and has_thu:
        return "HFSE-Th-U ratio"
    if has_hfse:
        return "HFSE ratio"
    if has_ree:
        return "REE ratio"
    if has_lile:
        return "LILE-related ratio"

    return "Trace-element ratio"


def benjamini_hochberg_qvalues(p_values):
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)

    valid = ~np.isnan(p)
    p_valid = p[valid]

    m = len(p_valid)

    if m == 0:
        return q

    order = np.argsort(p_valid)
    ranked_p = p_valid[order]

    ranked_q = ranked_p * m / (np.arange(1, m + 1))
    ranked_q = np.minimum.accumulate(ranked_q[::-1])[::-1]
    ranked_q = np.clip(ranked_q, 0, 1)

    q_valid = np.empty_like(ranked_q)
    q_valid[order] = ranked_q

    q[valid] = q_valid

    return q


def cliff_delta_from_u(x, y):
    """
    使用 Mann-Whitney U 计算 Cliff's delta。
    delta > 0 表示 x 整体大于 y。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    n1 = len(x)
    n2 = len(y)

    if n1 == 0 or n2 == 0:
        return np.nan

    try:
        u_stat, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
        delta = (2 * u_stat) / (n1 * n2) - 1
        return float(delta)
    except Exception:
        return np.nan


def cliff_delta_magnitude(delta):
    if pd.isna(delta):
        return "NA"

    ad = abs(delta)

    if ad < 0.147:
        return "negligible"
    elif ad < 0.330:
        return "small"
    elif ad < 0.474:
        return "medium"
    else:
        return "large"


def epsilon_squared_kruskal(H, n, k):
    """
    Kruskal-Wallis effect size.
    """
    if n <= k:
        return np.nan

    eps = (H - k + 1) / (n - k)
    return float(max(0, min(1, eps)))


def median_iqr_text(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()

    if len(x) == 0:
        return ""

    q1 = x.quantile(0.25)
    med = x.quantile(0.50)
    q3 = x.quantile(0.75)

    return f"{med:.4g} [{q1:.4g}, {q3:.4g}]"


# ============================================================
# 2. 读取稳定候选新比值
# ============================================================

def load_stable_ratio_candidates():
    if USE_MANUAL_FEATURES:
        df = pd.DataFrame({
            "Feature": MANUAL_FEATURES,
            "Display_feature": [display_feature_name(f) for f in MANUAL_FEATURES],
            "Source": "manual"
        })
        return df

    stability_root = STABILITY_DIR
    summary_file = find_stability_summary_file(stability_root)

    print("读取 05 稳定候选汇总：", summary_file)

    xls = pd.ExcelFile(summary_file)

    if "stable_ratio_candidates" not in xls.sheet_names:
        raise ValueError(f"{summary_file} 中找不到 stable_ratio_candidates sheet。")

    df = pd.read_excel(summary_file, sheet_name="stable_ratio_candidates")

    if "Feature" not in df.columns:
        raise ValueError("stable_ratio_candidates 中缺少 Feature 列。")

    # 筛选稳定候选
    if "Appearance_count" in df.columns:
        df = df[df["Appearance_count"] >= MIN_APPEARANCE_COUNT].copy()

    sort_cols = []
    ascending = []

    for c in ["Appearance_count", "Mean_champion_score", "Mean_SHAP_importance", "Mean_TopK_frequency"]:
        if c in df.columns:
            sort_cols.append(c)
            ascending.append(False)

    if sort_cols:
        df = df.sort_values(sort_cols, ascending=ascending)

    df = df.head(TOP_N_RATIOS).copy()

    df["Display_feature"] = df["Feature"].apply(display_feature_name)
    df["Ratio_group"] = df["Feature"].apply(ratio_group)

    return df.reset_index(drop=True)


# ============================================================
# 3. 读取固定五折 test 数据，构造 OOF 数据
# ============================================================

def normalize_type_value(v):
    """
    把 A-type / S-type / I-type 统一成 A / S / I。
    """
    s = str(v).strip()

    if s in ["A", "A-type", "A-Type", "A_TYPE", "A type", "A型"]:
        return "A"
    if s in ["S", "S-type", "S-Type", "S_TYPE", "S type", "S型"]:
        return "S"
    if s in ["I", "I-type", "I-Type", "I_TYPE", "I type", "I型"]:
        return "I"

    return s


def build_oof_dataset_for_method(method):
    """
    把每个 outer fold 的 test 数据拼起来。
    这样每个样品只出现一次，避免 train/test 重复计入。
    """
    method_dir = os.path.join(DATA_ROOT, method)

    if not os.path.exists(method_dir):
        raise FileNotFoundError(f"未找到数据目录：{method_dir}")

    dfs = []

    for fold in range(1, N_OUTER_FOLDS + 1):
        test_path = os.path.join(
            method_dir,
            f"fold_{fold:02d}_test_with_ratios.xlsx"
        )

        if not os.path.exists(test_path):
            raise FileNotFoundError(f"未找到 test fold 文件：{test_path}")

        df = pd.read_excel(test_path)
        df.columns = [str(c).strip() for c in df.columns]

        if TYPE_COL not in df.columns:
            raise ValueError(f"{test_path} 中没有标签列：{TYPE_COL}")

        # 关键修改：把 A-type / S-type / I-type 统一成 A / S / I
        df[TYPE_COL] = df[TYPE_COL].apply(normalize_type_value)

        df["Outer_fold"] = fold
        df["Imputation_method"] = method

        dfs.append(df)

    oof_df = pd.concat(dfs, axis=0, ignore_index=True)

    print(f"\n{method} OOF 数据 Type 分布：")
    print(oof_df[TYPE_COL].value_counts(dropna=False))

    return oof_df

def analyze_one_feature(df, feature, method):
    """
    对单个特征做：
    - A/S/I group summary
    - Kruskal-Wallis
    - pairwise Mann-Whitney U
    - Cliff's delta
    """
    rows_summary = []
    rows_pairwise = []

    work = df[[TYPE_COL, feature]].copy()
    work[feature] = pd.to_numeric(work[feature], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=[TYPE_COL, feature])

    available_classes = [c for c in CLASS_ORDER if c in work[TYPE_COL].unique()]
    if len(available_classes) < 2:
        return None, pd.DataFrame(), pd.DataFrame()

    groups = []

    for cls in available_classes:
        values = work.loc[work[TYPE_COL] == cls, feature].dropna().astype(float).values
        groups.append(values)

        rows_summary.append({
            "Method": method,
            "Feature": feature,
            "Display_feature": display_feature_name(feature),
            "Ratio_group": ratio_group(feature),
            "Class": cls,
            "N": len(values),
            "Mean": float(np.mean(values)) if len(values) else np.nan,
            "SD": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
            "Median": float(np.median(values)) if len(values) else np.nan,
            "Q1": float(np.quantile(values, 0.25)) if len(values) else np.nan,
            "Q3": float(np.quantile(values, 0.75)) if len(values) else np.nan,
            "Median_IQR": median_iqr_text(values)
        })

    # Kruskal-Wallis
    try:
        H, p_kw = stats.kruskal(*groups)
        H = float(H)
        p_kw = float(p_kw)
    except Exception:
        H, p_kw = np.nan, np.nan

    n_total = sum(len(g) for g in groups)
    k = len(groups)
    eps2 = epsilon_squared_kruskal(H, n_total, k) if pd.notna(H) else np.nan

    kruskal_row = {
        "Method": method,
        "Feature": feature,
        "Display_feature": display_feature_name(feature),
        "Ratio_group": ratio_group(feature),
        "N_total": n_total,
        "N_classes": k,
        "Kruskal_H": H,
        "Kruskal_p": p_kw,
        "Epsilon_squared": eps2,
    }

    # Pairwise Mann-Whitney U
    for i in range(len(available_classes)):
        for j in range(i + 1, len(available_classes)):
            cls1 = available_classes[i]
            cls2 = available_classes[j]

            x = work.loc[work[TYPE_COL] == cls1, feature].dropna().astype(float).values
            y = work.loc[work[TYPE_COL] == cls2, feature].dropna().astype(float).values

            if len(x) == 0 or len(y) == 0:
                u_stat, p_u = np.nan, np.nan
                delta = np.nan
            else:
                try:
                    u_stat, p_u = stats.mannwhitneyu(x, y, alternative="two-sided")
                    u_stat = float(u_stat)
                    p_u = float(p_u)
                except Exception:
                    u_stat, p_u = np.nan, np.nan

                delta = cliff_delta_from_u(x, y)

            rows_pairwise.append({
                "Method": method,
                "Feature": feature,
                "Display_feature": display_feature_name(feature),
                "Ratio_group": ratio_group(feature),
                "Comparison": f"{cls1} vs {cls2}",
                "Class_1": cls1,
                "Class_2": cls2,
                "N_1": len(x),
                "N_2": len(y),
                "Median_1": float(np.median(x)) if len(x) else np.nan,
                "Median_2": float(np.median(y)) if len(y) else np.nan,
                "Median_diff_1_minus_2": (
                    float(np.median(x) - np.median(y))
                    if len(x) and len(y)
                    else np.nan
                ),
                "MannWhitney_U": u_stat,
                "MannWhitney_p": p_u,
                "Cliffs_delta_1_vs_2": delta,
                "Cliffs_delta_magnitude": cliff_delta_magnitude(delta),
            })

    summary_df = pd.DataFrame(rows_summary)
    pairwise_df = pd.DataFrame(rows_pairwise)

    return kruskal_row, summary_df, pairwise_df


# ============================================================
# 5. 作图
# ============================================================

def plot_feature_boxplot(df, feature, method, out_path):
    work = df[[TYPE_COL, feature]].copy()
    work[feature] = pd.to_numeric(work[feature], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=[TYPE_COL, feature])

    available_classes = [c for c in CLASS_ORDER if c in work[TYPE_COL].unique()]

    data = [
        work.loc[work[TYPE_COL] == cls, feature].dropna().astype(float).values
        for cls in available_classes
    ]

    if len(data) < 2:
        return

    plt.figure(figsize=(6.5, 5.2), dpi=DPI)

    bp = plt.boxplot(
        data,
        labels=available_classes,
        showfliers=False,
        patch_artist=True
    )

    # 不指定颜色，保持默认风格；只做轻微透明
    for patch in bp["boxes"]:
        patch.set_alpha(0.6)

    rng = np.random.default_rng(42)

    for idx, values in enumerate(data, start=1):
        if len(values) == 0:
            continue

        jitter = rng.normal(loc=idx, scale=0.045, size=len(values))
        plt.scatter(
            jitter,
            values,
            s=10,
            alpha=0.35
        )

    plt.xlabel("Granite type", fontsize=12, fontweight="bold")
    plt.ylabel(display_feature_name(feature), fontsize=12, fontweight="bold")
    plt.title(
        f"{display_feature_name(feature)} by A/S/I type ({method})",
        fontsize=13,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_top_kw_significance(kruskal_df, out_path, method):
    sub = kruskal_df[kruskal_df["Method"] == method].copy()

    if sub.empty:
        return

    sub = sub.sort_values("Kruskal_q_BH", ascending=True).head(20).copy()
    sub = sub.sort_values("Kruskal_q_BH", ascending=False)

    q = sub["Kruskal_q_BH"].replace(0, np.nextafter(0, 1))
    neglogq = -np.log10(q)

    plt.figure(figsize=(8, max(5, 0.35 * len(sub))), dpi=DPI)

    plt.barh(
        sub["Display_feature"].astype(str),
        neglogq.values
    )

    plt.axvline(-np.log10(ALPHA), linestyle="--", linewidth=1)

    plt.xlabel("-log10(BH-adjusted q)", fontsize=12, fontweight="bold")
    plt.ylabel("Feature", fontsize=12, fontweight="bold")
    plt.title(
        f"Kruskal-Wallis significance of stable ratios ({method})",
        fontsize=13,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# ============================================================
# 6. 主函数
# ============================================================

def main():
    warnings.filterwarnings("ignore")

    print("========== 读取稳定候选新比值 ==========")

    candidate_df = load_stable_ratio_candidates()

    if candidate_df.empty:
        raise ValueError("未读取到任何稳定候选新比值。")

    selected_features = candidate_df["Feature"].astype(str).tolist()

    print(f"本次分析特征数：{len(selected_features)}")
    print(candidate_df[["Feature", "Display_feature"]].head(30))

    all_group_summary = []
    all_kruskal = []
    all_pairwise = []
    all_oof_data = []

    for method in IMPUTATION_METHODS:
        print(f"\n========== 构建 OOF 数据：{method} ==========")

        oof_df = build_oof_dataset_for_method(method)

        all_oof_data.append(oof_df)

        print(f"{method} OOF 样本数：{len(oof_df)}")
        print(oof_df[TYPE_COL].value_counts())

        for feature in selected_features:
            if feature not in oof_df.columns:
                print(f"警告：{method} 中找不到特征列：{feature}")
                continue

            print(f"分析 {method}: {feature}")

            kruskal_row, summary_df, pairwise_df = analyze_one_feature(
                oof_df,
                feature,
                method
            )

            if kruskal_row is not None:
                all_kruskal.append(kruskal_row)

            if not summary_df.empty:
                all_group_summary.append(summary_df)

            if not pairwise_df.empty:
                all_pairwise.append(pairwise_df)

            fig_name = f"{method}_{safe_filename(feature)}_boxplot.png"
            fig_path = os.path.join(FIG_DIR, fig_name)

            plot_feature_boxplot(
                oof_df,
                feature,
                method,
                fig_path
            )

    group_summary_df = (
        pd.concat(all_group_summary, axis=0, ignore_index=True)
        if all_group_summary else pd.DataFrame()
    )

    kruskal_df = pd.DataFrame(all_kruskal)

    pairwise_df = (
        pd.concat(all_pairwise, axis=0, ignore_index=True)
        if all_pairwise else pd.DataFrame()
    )

    # 多重检验校正
    if not kruskal_df.empty:
        kruskal_df["Kruskal_q_BH"] = benjamini_hochberg_qvalues(
            kruskal_df["Kruskal_p"].values
        )

        kruskal_df["Kruskal_significant_q05"] = kruskal_df["Kruskal_q_BH"] < ALPHA

        kruskal_df = kruskal_df.sort_values(
            ["Method", "Kruskal_q_BH", "Epsilon_squared"],
            ascending=[True, True, False]
        )

    if not pairwise_df.empty:
        pairwise_df["MannWhitney_q_BH"] = benjamini_hochberg_qvalues(
            pairwise_df["MannWhitney_p"].values
        )

        pairwise_df["Pairwise_significant_q05"] = pairwise_df["MannWhitney_q_BH"] < ALPHA

        pairwise_df = pairwise_df.sort_values(
            ["Method", "Feature", "MannWhitney_q_BH"],
            ascending=[True, True, True]
        )

    # 合并候选特征信息
    if not kruskal_df.empty:
        kruskal_df = kruskal_df.merge(
            candidate_df,
            on=["Feature", "Display_feature"],
            how="left",
            suffixes=("", "_candidate")
        )

    # 输出 Excel
    out_xlsx = os.path.join(
        OUT_DIR,
        "07_stable_novel_ratio_distribution_and_statistics.xlsx"
    )

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        candidate_df.to_excel(
            writer,
            sheet_name="selected_stable_ratios",
            index=False
        )

        if not group_summary_df.empty:
            group_summary_df.to_excel(
                writer,
                sheet_name="group_median_IQR",
                index=False
            )

        if not kruskal_df.empty:
            kruskal_df.to_excel(
                writer,
                sheet_name="kruskal_wallis",
                index=False
            )

        if not pairwise_df.empty:
            pairwise_df.to_excel(
                writer,
                sheet_name="pairwise_mannwhitney",
                index=False
            )

        # 每种插补方法的 OOF Type 数量
        oof_count_rows = []
        for method in IMPUTATION_METHODS:
            method_df = [d for d in all_oof_data if d["Imputation_method"].iloc[0] == method]
            if not method_df:
                continue
            tmp = method_df[0]
            vc = tmp[TYPE_COL].value_counts().reset_index()
            vc.columns = ["Class", "Count"]
            vc["Method"] = method
            oof_count_rows.append(vc)

        if oof_count_rows:
            pd.concat(oof_count_rows, axis=0, ignore_index=True).to_excel(
                writer,
                sheet_name="oof_class_counts",
                index=False
            )

    print("\n已保存统计结果：", out_xlsx)

    # 输出显著性图
    if not kruskal_df.empty:
        for method in IMPUTATION_METHODS:
            fig_path = os.path.join(
                OUT_DIR,
                f"{method}_stable_ratio_kruskal_significance.png"
            )

            plot_top_kw_significance(
                kruskal_df,
                fig_path,
                method
            )

            print("已保存显著性图：", fig_path)

    print("\n========== Kruskal-Wallis 显著性 Top 20 ==========")
    if not kruskal_df.empty:
        show_cols = [
            "Method",
            "Display_feature",
            "Ratio_group",
            "Kruskal_H",
            "Kruskal_p",
            "Kruskal_q_BH",
            "Epsilon_squared",
            "Kruskal_significant_q05"
        ]
        show_cols = [c for c in show_cols if c in kruskal_df.columns]
        print(kruskal_df[show_cols].head(20))

    print("\n========== 两两比较显著结果 Top 30 ==========")
    if not pairwise_df.empty:
        sig_pair = pairwise_df[pairwise_df["Pairwise_significant_q05"]].copy()
        show_cols = [
            "Method",
            "Display_feature",
            "Comparison",
            "Median_1",
            "Median_2",
            "Median_diff_1_minus_2",
            "MannWhitney_p",
            "MannWhitney_q_BH",
            "Cliffs_delta_1_vs_2",
            "Cliffs_delta_magnitude"
        ]
        show_cols = [c for c in show_cols if c in sig_pair.columns]
        print(sig_pair[show_cols].head(30))

    print("\n全部完成。")
    print("输出目录：", OUT_DIR)
    print("箱线图目录：", FIG_DIR)


# ============================================================
# 7. 程序入口
# ============================================================

if __name__ == "__main__":
    main()

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ============================================================
# 0. 路径设置
# ============================================================

IN_DIR = OUT_DIR

IN_XLSX = os.path.join(
    IN_DIR,
    "07_stable_novel_ratio_distribution_and_statistics.xlsx"
)

OUT_DIR = os.path.join(
    IN_DIR,
    "summary_outputs"
)
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 1. 读取结果表
# ============================================================

group_stats = pd.read_excel(IN_XLSX, sheet_name="group_median_IQR")
kw = pd.read_excel(IN_XLSX, sheet_name="kruskal_wallis")
pairwise = pd.read_excel(IN_XLSX, sheet_name="pairwise_mannwhitney")
selected = pd.read_excel(IN_XLSX, sheet_name="selected_stable_ratios")

print("group_median_IQR:")
print(group_stats.head())

print("\nkruskal_wallis:")
print(kw.head())


# ============================================================
# 2. 全局图形参数
# ============================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 1.4
plt.rcParams["xtick.major.width"] = 1.2
plt.rcParams["ytick.major.width"] = 1.2


# ============================================================
# 3. Representative ratio selection
#    Six ratios are retained for the compact distribution summary.
# ============================================================

REPRESENTATIVE_RATIOS = [
    "Nb/Pb",
    "Fe2O3t/P2O5",
    "TiO2/P2O5",
    "Dy/Ho",
    "Al2O3/K2O",
    "Y/Dy"
]

METHOD_FOR_FIG = "knn"   # 可改为 "global_mean"


# ============================================================
# 4. 工具函数
# ============================================================

def sci_text(x, digits=2):
    """
    科学计数法文本，例如 1.23 × 10^-5。
    """
    if pd.isna(x):
        return ""
    if x == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(x))))
    mant = x / (10 ** exp)
    return f"{mant:.{digits}f} × 10^{exp}"


def sci_text_for_plot(x, digits=1):
    """
    图内短科学计数法，例如 q=1.2e-70。
    """
    if pd.isna(x):
        return ""
    return f"{x:.{digits}e}"


def median_order_text(stats_sub):
    """
    根据 A/S/I 的中位数生成类别排序，如 A > I > S。
    """
    class_order = ["A", "I", "S"]
    med = {}

    for cls in class_order:
        row = stats_sub[stats_sub["Class"] == cls]
        if len(row) > 0:
            med[cls] = float(row.iloc[0]["Median"])

    if len(med) < 3:
        return ""

    sorted_items = sorted(med.items(), key=lambda kv: kv[1], reverse=True)
    return " > ".join([k for k, v in sorted_items])


def get_kw_value(feature_display, method, col):
    """
    从 Kruskal-Wallis 结果中取某个特征、某个方法的值。
    """
    row = kw[
        (kw["Display_feature"] == feature_display)
        &
        (kw["Method"] == method)
    ]

    if len(row) == 0:
        return np.nan

    return row.iloc[0][col]


def draw_iqr_box(ax, x, q1, median, q3, mean, width=0.50, color="#d9eaf7"):
    """
    根据 Q1、median、Q3 和 mean 画 IQR 箱形概览图。
    这里没有逐样本原始值，因此不画离群点。
    """
    iqr = q3 - q1

    # 如果 IQR 为 0，给一个很小高度，避免看不见
    if iqr == 0:
        iqr = max(abs(median) * 0.02, 1e-6)
        q1 = median - iqr / 2
        q3 = median + iqr / 2

    # 箱体
    rect = Rectangle(
        (x - width / 2, q1),
        width,
        q3 - q1,
        facecolor=color,
        edgecolor="black",
        linewidth=1.4,
        alpha=0.85
    )
    ax.add_patch(rect)

    # 中位数线
    ax.plot(
        [x - width / 2, x + width / 2],
        [median, median],
        color="black",
        linewidth=2.0
    )

    # mean 点
    ax.scatter(
        [x],
        [mean],
        marker="D",
        s=38,
        color="black",
        zorder=3
    )

    # 简短 whisker：用 Q1/Q3 外扩 0.35 IQR 作为视觉辅助，不代表真实 min/max
    whisker_low = q1 - 0.35 * iqr
    whisker_high = q3 + 0.35 * iqr

    ax.plot([x, x], [whisker_low, q1], color="black", linewidth=1.2)
    ax.plot([x, x], [q3, whisker_high], color="black", linewidth=1.2)

    cap = width * 0.45
    ax.plot([x - cap / 2, x + cap / 2], [whisker_low, whisker_low], color="black", linewidth=1.2)
    ax.plot([x - cap / 2, x + cap / 2], [whisker_high, whisker_high], color="black", linewidth=1.2)

    return whisker_low, whisker_high


def clean_feature_label(label):
    """
    图题中的标签统一规范。
    """
    s = str(label)
    s = s.replace("A12O3", "Al2O3")
    return s


# ============================================================
# 5. 生成代表性稳定新比值统计表
# ============================================================

# 以 global_mean 的 q-value 排序，取前 10
kw_global = kw[kw["Method"] == "global_mean"].copy()
kw_global = kw_global.sort_values("Kruskal_q_BH", ascending=True)

top10_features = kw_global["Display_feature"].head(10).tolist()

table_rows = []

for rank, feat in enumerate(top10_features, start=1):
    row_global = kw[
        (kw["Method"] == "global_mean")
        &
        (kw["Display_feature"] == feat)
    ].iloc[0]

    row_knn = kw[
        (kw["Method"] == "knn")
        &
        (kw["Display_feature"] == feat)
    ].iloc[0]

    stats_knn = group_stats[
        (group_stats["Method"] == "knn")
        &
        (group_stats["Display_feature"] == feat)
    ]

    pattern = median_order_text(stats_knn)

    table_rows.append({
        "Rank": rank,
        "Ratio feature": clean_feature_label(feat),
        "Ratio group": row_global["Ratio_group"],
        "global-mean H": row_global["Kruskal_H"],
        "global-mean q-value": row_global["Kruskal_q_BH"],
        "KNN H": row_knn["Kruskal_H"],
        "KNN q-value": row_knn["Kruskal_q_BH"],
        "Median order based on KNN": pattern
    })

representative_table = pd.DataFrame(table_rows)

# Human-readable formatting for exported summary tables.
formatted_table = representative_table.copy()
for c in ["global-mean H", "KNN H"]:
    formatted_table[c] = formatted_table[c].map(lambda x: f"{x:.2f}")

for c in ["global-mean q-value", "KNN q-value"]:
    formatted_table[c] = formatted_table[c].map(lambda x: sci_text(x, digits=2))

statistics_path = os.path.join(
    OUT_DIR,
    "representative_ratio_statistics.xlsx"
)

with pd.ExcelWriter(statistics_path, engine="openpyxl") as writer:
    representative_table.to_excel(writer, sheet_name="raw_values", index=False)
    formatted_table.to_excel(writer, sheet_name="formatted_values", index=False)
    kw.to_excel(writer, sheet_name="all_kruskal_results", index=False)
    group_stats.to_excel(writer, sheet_name="group_median_IQR", index=False)

print("代表性比值统计表已保存：", statistics_path)
print(formatted_table)


# ============================================================
# 6. 生成代表性 6 个比值的 2×3 类别分布图
# ============================================================

class_order = ["A", "S", "I"]
class_colors = {
    "A": "#f4a582",
    "S": "#92c5de",
    "I": "#b2abd2"
}

fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8))
axes = axes.ravel()

for ax, feat in zip(axes, REPRESENTATIVE_RATIOS):
    sub = group_stats[
        (group_stats["Method"] == METHOD_FOR_FIG)
        &
        (group_stats["Display_feature"] == feat)
    ].copy()

    if len(sub) == 0:
        ax.set_visible(False)
        continue

    y_low_all = []
    y_high_all = []

    for i, cls in enumerate(class_order, start=1):
        row = sub[sub["Class"] == cls]

        if len(row) == 0:
            continue

        row = row.iloc[0]

        q1 = float(row["Q1"])
        q3 = float(row["Q3"])
        med = float(row["Median"])
        mean = float(row["Mean"])

        low, high = draw_iqr_box(
            ax=ax,
            x=i,
            q1=q1,
            median=med,
            q3=q3,
            mean=mean,
            width=0.52,
            color=class_colors[cls]
        )

        y_low_all.append(low)
        y_high_all.append(high)

        # N 标注
        ax.text(
            i,
            low,
            f"n={int(row['N'])}",
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold"
        )

    q_bh = get_kw_value(feat, METHOD_FOR_FIG, "Kruskal_q_BH")

    ax.set_title(
        clean_feature_label(feat),
        fontsize=16,
        fontweight="bold",
        pad=8
    )

    ax.text(
        0.03,
        0.95,
        f"Kruskal q={sci_text_for_plot(q_bh, digits=1)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold"
    )

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(class_order, fontsize=13, fontweight="bold")

    ax.tick_params(axis="y", labelsize=11, width=1.2)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")

    ax.grid(axis="y", linestyle="--", alpha=0.35)

    if len(y_low_all) > 0 and len(y_high_all) > 0:
        y_min = min(y_low_all)
        y_max = max(y_high_all)
        y_range = y_max - y_min

        if y_range <= 0:
            y_range = max(abs(y_max), 1)

        ax.set_ylim(y_min - y_range * 0.15, y_max + y_range * 0.18)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

# 添加统一 y 轴说明
fig.text(
    0.04,
    0.5,
    "Ratio value",
    va="center",
    rotation="vertical",
    fontsize=16,
    fontweight="bold"
)

# 添加图例说明
legend_handles = []
for cls in class_order:
    legend_handles.append(
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor=class_colors[cls],
            edgecolor="black",
            linewidth=1.2,
            label=f"{cls}-type"
        )
    )

fig.legend(
    handles=legend_handles,
    loc="upper center",
    ncol=3,
    frameon=False,
    fontsize=13,
    bbox_to_anchor=(0.5, 1.01)
)

fig.suptitle(
    "Class-wise distributions of representative stable novel ratios",
    fontsize=20,
    fontweight="bold",
    y=1.06
)

plt.tight_layout(rect=[0.06, 0.03, 1, 0.96])

distribution_figure_path = os.path.join(
    OUT_DIR,
    "representative_ratio_distributions.png"
)

plt.savefig(distribution_figure_path, dpi=1000, bbox_inches="tight")
plt.close()

print("代表性比值分布图已保存：", distribution_figure_path)


# ============================================================
# 7. 补充图：所有稳定新比值的 q-value 排序图
# ============================================================

kw_plot = kw.copy()
kw_plot["Display_feature"] = kw_plot["Display_feature"].apply(clean_feature_label)

# 只画 KNN 和 global_mean 的 -log10(q)
kw_plot["minus_log10_q"] = -np.log10(kw_plot["Kruskal_q_BH"].clip(lower=1e-300))
kw_plot["Method_label"] = kw_plot["Method"].map({
    "global_mean": "global-mean",
    "knn": "KNN"
}).fillna(kw_plot["Method"])

# 按 global_mean q 排序
order_features = (
    kw_plot[kw_plot["Method"] == "global_mean"]
    .sort_values("Kruskal_q_BH", ascending=True)["Display_feature"]
    .tolist()
)

kw_plot["Display_feature"] = pd.Categorical(
    kw_plot["Display_feature"],
    categories=order_features,
    ordered=True
)

kw_plot = kw_plot.sort_values("Display_feature")

fig, ax = plt.subplots(figsize=(10.5, 7.5))

y_pos = np.arange(len(order_features))
height = 0.36

for method, offset, marker_label in [
    ("global-mean", -height / 2, "global-mean"),
    ("KNN", height / 2, "KNN")
]:
    sub = kw_plot[kw_plot["Method_label"] == method].set_index("Display_feature").reindex(order_features)

    ax.barh(
        y_pos + offset,
        sub["minus_log10_q"],
        height=height,
        edgecolor="black",
        linewidth=0.8,
        label=marker_label
    )

ax.set_yticks(y_pos)
ax.set_yticklabels(order_features, fontsize=11, fontweight="bold")

ax.set_xlabel(r"$-\log_{10}$(BH-adjusted Kruskal-Wallis q-value)", fontsize=14, fontweight="bold")
ax.set_ylabel("Stable novel ratio", fontsize=14, fontweight="bold")
ax.set_title("Significance ranking of stable novel ratios", fontsize=17, fontweight="bold", pad=10)

ax.grid(axis="x", linestyle="--", alpha=0.35)
ax.legend(frameon=False, fontsize=12, loc="lower right")

ax.tick_params(axis="x", labelsize=12, width=1.2)
for tick in ax.get_xticklabels():
    tick.set_fontweight("bold")

for spine in ax.spines.values():
    spine.set_linewidth(1.2)

ax.invert_yaxis()

plt.tight_layout()

ranking_figure_path = os.path.join(
    OUT_DIR,
    "selected_ratio_qvalue_ranking.png"
)

plt.savefig(ranking_figure_path, dpi=1000, bbox_inches="tight")
plt.close()

print("显著性排序图已保存：", ranking_figure_path)


# ============================================================
# 8. 输出说明
# ============================================================

print("\n全部完成。输出目录：")
print(OUT_DIR)

print("\n主要输出文件：")
print("1.", distribution_figure_path)
print("2.", statistics_path)
print("3.", ranking_figure_path)


