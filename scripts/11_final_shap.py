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


from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
from matplotlib.ticker import FuncFormatter, MaxNLocator

warnings.filterwarnings("ignore")

try:
    import shap
except ModuleNotFoundError:
    raise ModuleNotFoundError("当前环境没有 shap，请先运行：pip install shap")


# ============================================================
# 0. 路径与参数配置
# ============================================================

RESULT_BASE_DIR = RESULTS_DIR

DATA_ROOT = FOLDS_DIR

STABILITY_SUMMARY_FILE = STABILITY_DIR / "rho090_stable_champions_and_ratio_candidates_summary.xlsx"

OUT_DIR = RESULTS_DIR / "11_final_shap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TYPE_COL = "Type"
N_OUTER_FOLDS = 5

# Select one or more completed preprocessing workflows.
SHAP_METHODS = ["knn"]

CLASS_ORDER = ["A", "S", "I"]

MODEL_NAME = "ExtraTrees"
FEATURE_SET_NAME = "Stable features"

RANDOM_STATE = 42
N_ESTIMATORS = 600

TOP_N_GLOBAL = 20
TOP_N_CLASSWISE = 15

# Set an integer to subsample SHAP rows, or None to use every sample.
SHAP_MAX_SAMPLES = None

# Representative features used in the compact dependence summary.
REPRESENTATIVE_FEATURES = [
    "R_Trace_Yb/Lu",
    "R_Trace_Dy/Ho",
    "R_Trace_Nb/Pb",
    "R_Major_TiO2/P2O5",
    "R_Major_Fe2O3t/P2O5",
    "R_Major_Al2O3/K2O",
]


# ============================================================
# 1. 工具函数
# ============================================================

def normalize_type_value(v):
    s = str(v).strip()

    if s in ["A", "A-type", "A-Type", "A_TYPE", "A type", "A型"]:
        return "A"
    if s in ["S", "S-type", "S-Type", "S_TYPE", "S type", "S型"]:
        return "S"
    if s in ["I", "I-type", "I-Type", "I_TYPE", "I type", "I型"]:
        return "I"

    return s


def display_feature_name(name):
    s = str(name)

    s = s.replace("R_Major_", "")
    s = s.replace("R_Trace_", "")

    s = s.replace("A12O3", "Al2O3")
    s = s.replace("10000*Ga/A1", "10000×Ga/Al")
    s = s.replace("10000*Ga/Al", "10000×Ga/Al")

    return s


def safe_filename(name):
    s = display_feature_name(name)
    s = re.sub(r"[\\/:*?\"<>|]", "_", s)
    s = s.replace("×", "x")
    s = s.replace("+", "plus")
    s = s.replace(" ", "_")
    return s


def feature_key(name):
    s = str(name).strip()
    s = display_feature_name(s)
    s = s.replace("×", "*")
    s = s.replace("：", ":")
    s = re.sub(r"\s+", "", s)
    return s.lower()


def resolve_one_feature(feature, columns):
    columns = [str(c).strip() for c in columns]

    if feature in columns:
        return feature

    target_key = feature_key(feature)

    for c in columns:
        if feature_key(c) == target_key:
            return c

    # 修正 Al2O3 / A12O3 问题
    fixed_candidates = [
        str(feature).replace("A12O3", "Al2O3"),
        str(feature).replace("Al2O3", "A12O3"),
    ]

    for fixed in fixed_candidates:
        if fixed in columns:
            return fixed

        fixed_key = feature_key(fixed)

        for c in columns:
            if feature_key(c) == fixed_key:
                return c

    return None


def resolve_feature_list(features, columns):
    resolved = []
    missing = []

    for f in features:
        c = resolve_one_feature(f, columns)

        if c is None:
            missing.append(f)
        else:
            if c not in resolved:
                resolved.append(c)

    return resolved, missing


def clean_X(df, features):
    X = df[features].copy()

    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)

    return X


def remove_bad_features(X):
    """
    删除全空或常数列。
    这里只用于最终解释模型，不涉及性能评价。
    """
    keep_cols = []

    for c in X.columns:
        x = pd.to_numeric(X[c], errors="coerce")
        x = x.replace([np.inf, -np.inf], np.nan)
        non_na = x.dropna()

        if len(non_na) == 0:
            continue

        if non_na.nunique() <= 1:
            continue

        keep_cols.append(c)

    return X[keep_cols].copy(), keep_cols


def load_stable_interpretation_features():
    if not os.path.exists(STABILITY_SUMMARY_FILE):
        raise FileNotFoundError(f"找不到 05 汇总文件：{STABILITY_SUMMARY_FILE}")

    xls = pd.ExcelFile(STABILITY_SUMMARY_FILE)

    preferred_sheets = [
        "stable_interpretation_features",
        "core_stable_features",
        "champion_stability_summary",
    ]

    for sheet in preferred_sheets:
        if sheet in xls.sheet_names:
            df = pd.read_excel(STABILITY_SUMMARY_FILE, sheet_name=sheet)

            if "Feature" in df.columns:
                features = (
                    df["Feature"]
                    .dropna()
                    .astype(str)
                    .tolist()
                )

                if len(features) > 0:
                    print(f"从 sheet 读取稳定特征：{sheet}")
                    print(f"稳定特征数：{len(features)}")
                    return features

    raise ValueError(
        "没有在 05 汇总文件中找到 stable_interpretation_features / champion_stability_summary 等可用 Feature 表。"
    )


def read_oof_full_dataset(method):
    """
    拼接 5 个 outer-fold test 文件。
    每个样品只出现一次，作为最终解释模型的数据。
    注意：这里用于解释图，不用于报告模型泛化性能。
    """
    dfs = []

    for fold in range(1, N_OUTER_FOLDS + 1):
        test_path = os.path.join(
            DATA_ROOT,
            method,
            f"fold_{fold:02d}_test_with_ratios.xlsx"
        )

        if not os.path.exists(test_path):
            raise FileNotFoundError(test_path)

        df = pd.read_excel(test_path)
        df.columns = [str(c).strip() for c in df.columns]

        if TYPE_COL not in df.columns:
            raise ValueError(f"缺少标签列：{TYPE_COL}")

        df[TYPE_COL] = df[TYPE_COL].apply(normalize_type_value)
        df["Outer_fold"] = fold
        df["Imputation_method"] = method

        dfs.append(df)

    full_df = pd.concat(dfs, axis=0, ignore_index=True)

    print(f"\n{method} OOF 拼接数据：")
    print("样本数：", len(full_df))
    print("类别分布：")
    print(full_df[TYPE_COL].value_counts())

    return full_df


def make_final_model():
    model = ExtraTreesClassifier(
        n_estimators=N_ESTIMATORS,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight=None
    )

    return model


def normalize_shap_values(raw_shap_values, n_samples, n_features, n_classes):
    """
    兼容不同 shap 版本的输出格式。
    最终统一成：
    shap_arr.shape = (n_samples, n_features, n_classes)
    """
    if isinstance(raw_shap_values, list):
        # list length = n_classes, each = (n_samples, n_features)
        shap_arr = np.stack(raw_shap_values, axis=2)
        return shap_arr

    arr = np.asarray(raw_shap_values)

    if arr.ndim == 2:
        # 二分类时可能是 (n_samples, n_features)，这里扩一维
        arr = arr[:, :, np.newaxis]
        return arr

    if arr.ndim == 3:
        # 常见新版格式：(n_samples, n_features, n_classes)
        if arr.shape[0] == n_samples and arr.shape[1] == n_features:
            return arr

        # Some SHAP releases return (n_classes, n_samples, n_features).
        if arr.shape[0] == n_classes and arr.shape[1] == n_samples:
            arr = np.transpose(arr, (1, 2, 0))
            return arr

    raise ValueError(
        f"无法识别 SHAP 输出形状：{arr.shape}，"
        f"n_samples={n_samples}, n_features={n_features}, n_classes={n_classes}"
    )


def classify_feature_group(feature):
    s = str(feature)
    body = display_feature_name(s)

    classical_names = {
        "10000×Ga/Al",
        "A/CNK",
        "A/NK",
        "Zr+Nb+Ce+Y",
        "Sr/Y",
        "Rb/Sr",
        "K2O/Na2O",
        "Fe2O3t/MgO",
    }

    if body in classical_names:
        return "Classical geochemical index"

    if s.startswith("R_Major_"):
        return "Major-element ratio"

    if not s.startswith("R_Trace_"):
        return "Original element or classical feature"

    hfse = {"Nb", "Ta", "Zr", "Hf", "Ti", "Y"}
    ree = {
        "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd",
        "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"
    }
    lile = {"Rb", "Sr", "Ba", "Cs", "Pb"}
    thu = {"Th", "U"}

    parts = re.split(r"[/_+\-*()]+", body)
    parts = [p for p in parts if p]

    has_hfse = any(p in hfse for p in parts)
    has_ree = any(p in ree for p in parts)
    has_lile = any(p in lile for p in parts)
    has_thu = any(p in thu for p in parts)

    if has_hfse and has_ree:
        return "HFSE-REE ratio"
    if has_hfse and has_thu:
        return "HFSE-Th-U ratio"
    if has_ree:
        return "REE fractionation ratio"
    if has_hfse:
        return "HFSE-related ratio"
    if has_lile:
        return "LILE-related ratio"

    return "Trace-element ratio"


def interpretation_note(feature):
    name = display_feature_name(feature)

    notes = {
        "10000×Ga/Al": "Classical A-type granite discriminator related to Ga enrichment and Al depletion.",
        "A/CNK": "Aluminosity index commonly used to distinguish peraluminous S-type affinity.",
        "A/NK": "Alkali-normalized aluminosity index.",
        "Zr+Nb+Ce+Y": "Classical HFSE-REE enrichment index for A-type granite discrimination.",
        "Sr/Y": "Indicator related to plagioclase/amphibole fractionation and source/depth effects.",
        "Rb/Sr": "Index related to differentiation, feldspar fractionation, and crustal evolution.",
        "Yb/Lu": "Heavy REE fractionation signal.",
        "Ho/Er": "Middle-to-heavy REE fractionation signal.",
        "Dy/Ho": "HREE fractionation and REE pattern curvature signal.",
        "Y/Dy": "Y-HREE behavior and fractionation signal.",
        "Y/Er": "Y-HREE behavior and fractionation signal.",
        "Sm/Gd": "Middle REE fractionation signal.",
        "Nb/Pb": "HFSE vs crustal/LILE-related enrichment contrast.",
        "Ta/Th": "HFSE-Th contrast related to source and differentiation effects.",
        "Nb/U": "HFSE-U contrast, potentially reflecting crustal affinity and source differences.",
        "TiO2/MgO": "Major-element ratio related to mafic mineral and Fe-Ti oxide control.",
        "TiO2/P2O5": "Fe-Ti-P system, possibly linked to oxide and apatite fractionation.",
        "Fe2O3t/P2O5": "Fe-P differentiation and accessory mineral control.",
        "Al2O3/K2O": "Aluminous and feldspar/mica-related signal.",
    }

    return notes.get(name, "")


# ============================================================
# 2. 画图函数
# ============================================================

def bold_tick_labels(ax):
    """
    仅加粗 x/y 坐标刻度文字，不改变字号、颜色、布局和其他图形样式。
    """
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")

    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
        
def format_xaxis_effective_digits(ax, nbins=5):
    """
    Format x-axis tick labels using effective digits only.
    This removes unnecessary trailing zeros and reduces tick crowding.
    """
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: "0" if np.isclose(x, 0) else f"{x:.3g}")
    )

def plot_global_shap_bar(global_imp_df, out_png, top_n=20):
    sub = global_imp_df.head(top_n).copy()
    sub = sub.sort_values("Mean_abs_SHAP_overall", ascending=True)

    plt.figure(figsize=(8.5, max(5.5, 0.35 * len(sub))), dpi=1000)

    plt.barh(
        sub["Display_feature"],
        sub["Mean_abs_SHAP_overall"]
    )

    plt.xlabel("Mean |SHAP|", fontsize=12, fontweight="bold")
    plt.ylabel("Feature", fontsize=12, fontweight="bold")
    plt.title(
        f"Global SHAP importance ({MODEL_NAME}, {FEATURE_SET_NAME})",
        fontsize=14,
        fontweight="bold"
    )

    ax = plt.gca()
    format_xaxis_effective_digits(ax, nbins=6)
    bold_tick_labels(ax)

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    plt.close()


def plot_classwise_shap_bar(classwise_imp_df, out_png, top_n=15):
    n_classes = len(CLASS_ORDER)

    fig, axes = plt.subplots(
        1,
        n_classes,
        figsize=(5.2 * n_classes, max(5.5, 0.32 * top_n)),
        dpi=1000
    )

    if n_classes == 1:
        axes = [axes]

    for ax, cls in zip(axes, CLASS_ORDER):
        sub = classwise_imp_df[classwise_imp_df["Class"] == cls].copy()
        sub = sub.head(top_n)
        sub = sub.sort_values("Mean_abs_SHAP_class", ascending=True)

        ax.barh(
            sub["Display_feature"],
            sub["Mean_abs_SHAP_class"]
        )

        ax.set_title(f"{cls}-type", fontsize=13, fontweight="bold")
        ax.set_xlabel("Mean |SHAP|", fontsize=11, fontweight="bold")
        ax.tick_params(axis="y", labelsize=9)

        # Format x-axis tick labels with effective digits only
        format_xaxis_effective_digits(ax, nbins=5)

        bold_tick_labels(ax)

    fig.suptitle(
        f"Class-wise SHAP importance ({MODEL_NAME})",
        fontsize=15,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    plt.close()
    

def plot_classwise_beeswarm(shap_arr, X_shap, model_classes, method, out_dir):
    """Export one SHAP beeswarm plot per class."""
    for cls in CLASS_ORDER:
        if cls not in model_classes:
            continue

        cls_idx = list(model_classes).index(cls)

        values = shap_arr[:, :, cls_idx]

        explanation = shap.Explanation(
            values=values,
            data=X_shap.values,
            feature_names=[display_feature_name(c) for c in X_shap.columns]
        )

        plt.figure(dpi=1000)
        shap.plots.beeswarm(
            explanation,
            max_display=20,
            show=False
        )

        plt.title(
            f"SHAP beeswarm for {cls}-type ({method})",
            fontsize=13,
            fontweight="bold"
        )

        ax = plt.gca()
        bold_tick_labels(ax)

        out_png = os.path.join(
            out_dir,
            f"11_SHAP_beeswarm_{method}_{cls}_type.png"
        )

        plt.tight_layout()
        plt.savefig(out_png, bbox_inches="tight")
        plt.close()
        plt.close()


def plot_representative_dependence(
    shap_arr,
    X_shap,
    representative_features,
    feature_to_best_class,
    out_png
):
    resolved_features = []

    for f in representative_features:
        c = resolve_one_feature(f, X_shap.columns)

        if c is not None and c not in resolved_features:
            resolved_features.append(c)

    if len(resolved_features) == 0:
        print("没有找到可画 dependence 的代表性特征。")
        return

    n = len(resolved_features)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.2 * ncols, 4.2 * nrows),
        dpi=1000
    )

    axes = np.asarray(axes).reshape(-1)

    for ax_idx, feature in enumerate(resolved_features):
        ax = axes[ax_idx]

        feat_idx = list(X_shap.columns).index(feature)
        cls = feature_to_best_class.get(feature, CLASS_ORDER[0])
        cls_idx = list(model.classes_).index(cls)

        x = X_shap[feature].values
        y = shap_arr[:, feat_idx, cls_idx]

        ax.scatter(
            x,
            y,
            s=18,
            alpha=0.55
        )

        ax.axhline(0, linestyle="--", linewidth=1)

        ax.set_xlabel(display_feature_name(feature), fontsize=11, fontweight="bold")
        ax.set_ylabel(f"SHAP value for {cls}-type", fontsize=11, fontweight="bold")
        ax.set_title(
            f"{display_feature_name(feature)} ({cls}-type)",
            fontsize=12,
            fontweight="bold"
        )

        bold_tick_labels(ax)

    for j in range(len(resolved_features), len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "Representative SHAP dependence plots for stable novel ratios",
        fontsize=15,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    plt.close()


def plot_combined_shap_interpretation(
    global_imp_df,
    classwise_imp_df,
    shap_arr,
    X_shap,
    representative_features,
    feature_to_best_class,
    model_classes,
    out_png,
    out_pdf,
):
    """Create a compact multi-panel SHAP interpretation summary."""
    class_colors = {"A": "#D66B32", "S": "#2B7EA1", "I": "#7566A5"}
    blue = "#176B9C"
    dark = "#111111"
    grid = "#C7C7C7"

    fig = plt.figure(figsize=(12.8, 10.6), dpi=1000)
    outer = fig.add_gridspec(
        2, 1, height_ratios=[0.92, 1.38],
        left=0.075, right=0.985, bottom=0.065, top=0.92,
        hspace=0.34,
    )
    top = outer[0].subgridspec(1, 2, width_ratios=[0.92, 2.08], wspace=0.42)

    # Panel a: compact global ranking.
    ax_global = fig.add_subplot(top[0, 0])
    global_sub = (
        global_imp_df.head(12)
        .sort_values("Mean_abs_SHAP_overall", ascending=True)
    )
    ax_global.barh(
        global_sub["Display_feature"],
        global_sub["Mean_abs_SHAP_overall"],
        color=blue, edgecolor="none",
    )
    ax_global.set_title("Global importance", fontsize=10.5, fontweight="bold", pad=8)
    ax_global.set_xlabel("Mean |SHAP|", fontsize=8.5, fontweight="bold")
    ax_global.grid(axis="x", color=grid, linewidth=0.7, alpha=0.55)
    ax_global.set_axisbelow(True)
    format_xaxis_effective_digits(ax_global, nbins=5)

    # Panel b: class-specific rankings.
    class_grid = top[0, 1].subgridspec(1, 3, wspace=0.62)
    class_axes = []
    for idx, cls in enumerate(CLASS_ORDER):
        ax = fig.add_subplot(class_grid[0, idx])
        class_axes.append(ax)
        sub = (
            classwise_imp_df[classwise_imp_df["Class"] == cls]
            .head(8)
            .sort_values("Mean_abs_SHAP_class", ascending=True)
        )
        ax.barh(
            sub["Display_feature"], sub["Mean_abs_SHAP_class"],
            color=class_colors[cls], edgecolor="none",
        )
        ax.set_title(f"{cls}-type", fontsize=9.5, fontweight="bold", pad=7)
        ax.set_xlabel("Mean |SHAP|", fontsize=8, fontweight="bold")
        ax.grid(axis="x", color=grid, linewidth=0.7, alpha=0.55)
        ax.set_axisbelow(True)
        format_xaxis_effective_digits(ax, nbins=4)
    class_axes[1].text(
        0.5, 1.07, "Class-specific importance",
        transform=class_axes[1].transAxes, ha="center", va="bottom",
        fontsize=10.5, fontweight="bold", color=dark,
    )

    # Panel c: contribution directions for six representative ratios.
    resolved_features = []
    for feature in representative_features:
        resolved = resolve_one_feature(feature, X_shap.columns)
        if resolved is not None and resolved not in resolved_features:
            resolved_features.append(resolved)

    dep_grid = outer[1].subgridspec(2, 3, hspace=0.52, wspace=0.40)
    dep_axes = []
    for idx, feature in enumerate(resolved_features[:6]):
        ax = fig.add_subplot(dep_grid[idx // 3, idx % 3])
        dep_axes.append(ax)
        feat_idx = list(X_shap.columns).index(feature)
        cls = feature_to_best_class.get(feature, CLASS_ORDER[0])
        cls_idx = list(model_classes).index(cls)
        x_values = X_shap[feature].values
        y_values = shap_arr[:, feat_idx, cls_idx]
        ax.scatter(
            x_values, y_values, s=9, alpha=0.38,
            color=class_colors.get(cls, blue), edgecolors="none", rasterized=True,
        )
        ax.axhline(0, linestyle="--", linewidth=1.0, color="#444444")
        display = display_feature_name(feature)
        ax.set_title(f"{display} ({cls}-type)", fontsize=8.8, fontweight="bold", pad=5)
        ax.set_xlabel(display, fontsize=8, fontweight="bold")
        ax.set_ylabel(f"SHAP value for {cls}-type", fontsize=7.7, fontweight="bold")
        ax.grid(color=grid, linewidth=0.55, alpha=0.35)
        ax.set_axisbelow(True)

    for idx in range(len(resolved_features[:6]), 6):
        fig.add_subplot(dep_grid[idx // 3, idx % 3]).axis("off")

    all_axes = [ax_global, *class_axes, *dep_axes]
    for ax in all_axes:
        ax.tick_params(axis="both", labelsize=7.2, colors=dark, width=1.0, length=3.5)
        bold_tick_labels(ax)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(dark)
        ax.spines["bottom"].set_color(dark)
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)

    ax_global.text(-0.22, 1.08, "a", transform=ax_global.transAxes,
                   fontsize=13, fontweight="bold", va="top")
    class_axes[0].text(-0.34, 1.08, "b", transform=class_axes[0].transAxes,
                       fontsize=13, fontweight="bold", va="top")
    if dep_axes:
        dep_axes[0].text(-0.18, 1.16, "c", transform=dep_axes[0].transAxes,
                         fontsize=13, fontweight="bold", va="top")
        dep_axes[1].text(
            0.5, 1.22, "Representative feature contributions",
            transform=dep_axes[1].transAxes, ha="center", va="bottom",
            fontsize=10.5, fontweight="bold", color=dark,
        )

    fig.suptitle(
        "Post hoc ExtraTrees–SHAP interpretation of the 88-feature stable inventory",
        fontsize=13.5, fontweight="bold", y=0.975,
    )
    fig.savefig(out_png, dpi=1000, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ============================================================
# 3. 主程序：按插补方法生成 SHAP 解释图
# ============================================================

all_method_summary = []

for method in SHAP_METHODS:
    print("\n" + "=" * 80)
    print(f"开始最终 SHAP 解释：{method}")
    print("=" * 80)

    method_out_dir = os.path.join(OUT_DIR, method)
    os.makedirs(method_out_dir, exist_ok=True)

    # --------------------------------------------------------
    # 3.1 读取数据
    # --------------------------------------------------------

    full_df = read_oof_full_dataset(method)

    raw_features = load_stable_interpretation_features()

    resolved_features, missing_features = resolve_feature_list(
        raw_features,
        full_df.columns
    )

    print("\n稳定特征匹配结果：")
    print("原始稳定特征数：", len(raw_features))
    print("匹配成功特征数：", len(resolved_features))
    print("缺失特征数：", len(missing_features))

    if missing_features:
        print("前 20 个缺失特征：")
        print(missing_features[:20])

    if len(resolved_features) == 0:
        raise ValueError("没有匹配到任何 stable_interpretation_features 特征。")

    X_raw = clean_X(full_df, resolved_features)
    X_clean, kept_features = remove_bad_features(X_raw)

    y = full_df[TYPE_COL].astype(str).values

    print("\n最终用于 SHAP 的特征数：", len(kept_features))
    print("类别分布：")
    print(pd.Series(y).value_counts())

    # --------------------------------------------------------
    # 3.2 全数据拟合最终解释模型
    # --------------------------------------------------------

    imputer = SimpleImputer(strategy="median")

    X_imp_arr = imputer.fit_transform(X_clean)

    X_imp = pd.DataFrame(
        X_imp_arr,
        columns=X_clean.columns,
        index=X_clean.index
    )

    model = make_final_model()
    model.fit(X_imp, y)

    y_fit_pred = model.predict(X_imp)

    fit_acc = accuracy_score(y, y_fit_pred)
    fit_bacc = balanced_accuracy_score(y, y_fit_pred)
    fit_macro_p, fit_macro_r, fit_macro_f1, _ = precision_recall_fscore_support(
        y,
        y_fit_pred,
        average="macro",
        zero_division=0
    )

    print("\n注意：下面是 final refit 的 apparent fit，仅用于确认模型已拟合，不作为泛化性能汇报。")
    print(f"Apparent Accuracy: {fit_acc:.4f}")
    print(f"Apparent Balanced accuracy: {fit_bacc:.4f}")
    print(f"Apparent Macro-F1: {fit_macro_f1:.4f}")
    print("模型类别顺序：", model.classes_)

    # --------------------------------------------------------
    # 3.3 选择 SHAP 计算样本
    # --------------------------------------------------------

    if SHAP_MAX_SAMPLES is not None and len(X_imp) > SHAP_MAX_SAMPLES:
        rng = np.random.default_rng(RANDOM_STATE)
        sample_idx = rng.choice(
            np.arange(len(X_imp)),
            size=SHAP_MAX_SAMPLES,
            replace=False
        )
        sample_idx = np.sort(sample_idx)

        X_shap = X_imp.iloc[sample_idx].copy()
        y_shap = pd.Series(y).iloc[sample_idx].values
    else:
        X_shap = X_imp.copy()
        y_shap = y.copy()

    print("\nSHAP 计算样本数：", len(X_shap))

    # --------------------------------------------------------
    # 3.4 计算 SHAP
    # --------------------------------------------------------

    explainer = shap.TreeExplainer(model)

    raw_shap_values = explainer.shap_values(X_shap)

    shap_arr = normalize_shap_values(
        raw_shap_values,
        n_samples=X_shap.shape[0],
        n_features=X_shap.shape[1],
        n_classes=len(model.classes_)
    )

    print("SHAP array shape:", shap_arr.shape)

    # --------------------------------------------------------
    # 3.5 计算 global 与 class-wise SHAP 重要性
    # --------------------------------------------------------

    mean_abs_overall = np.abs(shap_arr).mean(axis=(0, 2))

    global_imp_df = pd.DataFrame({
        "Feature": X_shap.columns,
        "Display_feature": [display_feature_name(c) for c in X_shap.columns],
        "Mean_abs_SHAP_overall": mean_abs_overall,
    })

    global_imp_df["Feature_group"] = global_imp_df["Feature"].apply(classify_feature_group)
    global_imp_df["Interpretation_note"] = global_imp_df["Feature"].apply(interpretation_note)

    global_imp_df = global_imp_df.sort_values(
        "Mean_abs_SHAP_overall",
        ascending=False
    ).reset_index(drop=True)

    global_imp_df["Global_rank"] = np.arange(1, len(global_imp_df) + 1)

    classwise_rows = []

    for cls in CLASS_ORDER:
        if cls not in model.classes_:
            continue

        cls_idx = list(model.classes_).index(cls)

        mean_abs_cls = np.abs(shap_arr[:, :, cls_idx]).mean(axis=0)

        for feature, imp in zip(X_shap.columns, mean_abs_cls):
            classwise_rows.append({
                "Class": cls,
                "Feature": feature,
                "Display_feature": display_feature_name(feature),
                "Mean_abs_SHAP_class": imp,
                "Feature_group": classify_feature_group(feature),
                "Interpretation_note": interpretation_note(feature),
            })

    classwise_imp_df = pd.DataFrame(classwise_rows)

    classwise_imp_df = (
        classwise_imp_df
        .sort_values(["Class", "Mean_abs_SHAP_class"], ascending=[True, False])
        .reset_index(drop=True)
    )

    classwise_imp_df["Class_rank"] = (
        classwise_imp_df
        .groupby("Class")["Mean_abs_SHAP_class"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    # 每个特征找到它 SHAP 影响最大的类别，用于 dependence y 轴
    feature_to_best_class = {}

    for feature in X_shap.columns:
        sub = classwise_imp_df[classwise_imp_df["Feature"] == feature].copy()

        if not sub.empty:
            best_cls = sub.sort_values("Mean_abs_SHAP_class", ascending=False).iloc[0]["Class"]
            feature_to_best_class[feature] = best_cls

    # --------------------------------------------------------
    # 3.6 地球化学解释表
    # --------------------------------------------------------

    interpretation_df = global_imp_df.copy()

    interpretation_df["Use_in_text"] = np.where(
        interpretation_df["Global_rank"] <= TOP_N_GLOBAL,
        "Main text candidate",
        "Supplementary"
    )

    interpretation_df["Recommended_discussion_level"] = interpretation_df["Display_feature"].apply(
        lambda x: (
            "Core discussion"
            if x in [
                "10000×Ga/Al",
                "A/CNK",
                "Zr+Nb+Ce+Y",
                "Yb/Lu",
                "Dy/Ho",
                "Nb/Pb",
                "TiO2/P2O5",
                "Fe2O3t/P2O5",
                "Al2O3/K2O",
            ]
            else "Secondary discussion"
        )
    )

    # --------------------------------------------------------
    # 3.7 保存表格
    # --------------------------------------------------------

    model_info_df = pd.DataFrame({
        "Item": [
            "Model",
            "Feature_set",
            "Imputation_method",
            "N_samples_for_final_refit",
            "N_samples_for_SHAP",
            "N_features",
            "Class_order_in_model",
            "Apparent_accuracy_not_for_generalization",
            "Apparent_balanced_accuracy_not_for_generalization",
            "Apparent_macro_F1_not_for_generalization",
            "Note",
        ],
        "Value": [
            MODEL_NAME,
            FEATURE_SET_NAME,
            method,
            len(X_imp),
            len(X_shap),
            X_shap.shape[1],
            ", ".join(model.classes_),
            fit_acc,
            fit_bacc,
            fit_macro_f1,
            "Final full-data refit is used only for interpretation and visualization; cross-validated performance should be reported from the leakage-free outer-fold workflow.",
        ]
    })

    out_xlsx = os.path.join(
        method_out_dir,
        f"11_final_SHAP_importance_tables_{method}.xlsx"
    )

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        model_info_df.to_excel(writer, sheet_name="model_info", index=False)
        global_imp_df.to_excel(writer, sheet_name="global_SHAP_importance", index=False)
        classwise_imp_df.to_excel(writer, sheet_name="classwise_SHAP_importance", index=False)
        interpretation_df.to_excel(writer, sheet_name="geochemical_interpretation", index=False)

        pd.DataFrame({
            "Missing_feature": missing_features
        }).to_excel(writer, sheet_name="missing_features", index=False)

    print("\n已保存 SHAP 表格：", out_xlsx)

    # --------------------------------------------------------
    # 3.8 保存图件
    # --------------------------------------------------------

    global_bar_png = os.path.join(
        method_out_dir,
        f"11_SHAP_global_bar_{method}.png"
    )

    plot_global_shap_bar(
        global_imp_df,
        global_bar_png,
        top_n=TOP_N_GLOBAL
    )

    classwise_bar_png = os.path.join(
        method_out_dir,
        f"11_SHAP_classwise_bar_{method}.png"
    )

    plot_classwise_shap_bar(
        classwise_imp_df,
        classwise_bar_png,
        top_n=TOP_N_CLASSWISE
    )

    # beeswarm 建议作为补充图
    plot_classwise_beeswarm(
        shap_arr,
        X_shap,
        model.classes_,
        method,
        method_out_dir
    )

    dependence_png = os.path.join(
        method_out_dir,
        f"11_SHAP_representative_dependence_2x3_{method}.png"
    )

    plot_representative_dependence(
        shap_arr,
        X_shap,
        REPRESENTATIVE_FEATURES,
        feature_to_best_class,
        dependence_png
    )

    combined_png = os.path.join(
        method_out_dir,
        f"11_SHAP_combined_interpretation_{method}.png"
    )
    combined_pdf = os.path.join(
        method_out_dir,
        f"11_SHAP_combined_interpretation_{method}.pdf"
    )
    plot_combined_shap_interpretation(
        global_imp_df=global_imp_df,
        classwise_imp_df=classwise_imp_df,
        shap_arr=shap_arr,
        X_shap=X_shap,
        representative_features=REPRESENTATIVE_FEATURES,
        feature_to_best_class=feature_to_best_class,
        model_classes=model.classes_,
        out_png=combined_png,
        out_pdf=combined_pdf,
    )
    print("已保存合并 SHAP 主图：", combined_png)

    # --------------------------------------------------------
    # 3.9 汇总
    # --------------------------------------------------------

    top_global = global_imp_df.head(TOP_N_GLOBAL).copy()
    top_global["Imputation_method"] = method

    all_method_summary.append(top_global)

    print("\nTop 20 global SHAP features:")
    print(
        global_imp_df[
            [
                "Global_rank",
                "Display_feature",
                "Feature_group",
                "Mean_abs_SHAP_overall",
                "Interpretation_note"
            ]
        ].head(TOP_N_GLOBAL)
    )


# ============================================================
# 4. 多插补方法汇总，如果只跑 knn，也会正常输出
# ============================================================

if all_method_summary:
    combined_top_df = pd.concat(all_method_summary, axis=0, ignore_index=True)

    combined_xlsx = os.path.join(
        OUT_DIR,
        "11_combined_top_SHAP_features.xlsx"
    )

    with pd.ExcelWriter(combined_xlsx, engine="openpyxl") as writer:
        combined_top_df.to_excel(writer, sheet_name="combined_top_features", index=False)

    print("\n========== 11 最终 SHAP 解释完成 ==========")
    print("输出目录：", OUT_DIR)
    print("汇总表：", combined_xlsx)
else:
    print("没有生成任何 SHAP 汇总结果。")


