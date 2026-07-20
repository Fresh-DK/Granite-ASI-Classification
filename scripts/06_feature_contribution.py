from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["savefig.dpi"] = 1000
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from xgboost import XGBClassifier

from granite_ml.config import RESULTS_DIR
from granite_ml.features import (
    CLASSICAL_FEATURE_ALIASES, CLASS_ORDER, IMPUTATION_METHODS, N_OUTER_FOLDS,
    TYPE_COL, build_feature_metadata, build_fold_feature_sets, get_classical_features,
)
from granite_ml.io import feature_columns, load_fold, load_fold_champions
from granite_ml.excel import style_workbook
from granite_ml.metrics import classwise_rows, confusion_rows, evaluate_predictions, summarize_folds


OUT_DIR = RESULTS_DIR / "06_feature_contribution"
OUT_FILE = OUT_DIR / "06_feature_contribution_results.xlsx"
FEATURE_SET_ORDER = [
    "Non_ratio_baseline",
    "Non_ratio_plus_fold_novel",
    "Full_candidate_features",
    "Fold_cluster_champions",
]
FEATURE_LABELS = {
    "Non_ratio_baseline": "Non-ratio baseline",
    "Non_ratio_plus_fold_novel": "Baseline + fold novel ratios",
    "Full_candidate_features": "Full candidate features",
    "Fold_cluster_champions": "Fold cluster champions",
}
PAIRED_COMPARISONS = (
    ("Non_ratio_plus_fold_novel", "Non_ratio_baseline"),
    ("Fold_cluster_champions", "Full_candidate_features"),
)
METRICS = ["Accuracy", "Balanced_accuracy", "Macro_precision", "Macro_recall", "Macro_F1"]
SEED = 42
XGB_PARAMS = dict(
    n_estimators=900,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=3,
    subsample=0.85,
    colsample_bytree=0.75,
    gamma=0.0,
    reg_lambda=6.0,
    reg_alpha=0.0,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=SEED,
    n_jobs=-1,
    tree_method="hist",
)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=1000, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def numeric_pair(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train = train[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x_test = test[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if x_train.isna().any().any() or x_test.isna().any().any():
        raise ValueError("Step 06 received missing/non-finite values after fold-wise preprocessing.")
    return x_train, x_test


def fit_predict(x_train: pd.DataFrame, y_train: np.ndarray, x_test: pd.DataFrame) -> np.ndarray:
    label_to_int = {label: index for index, label in enumerate(CLASS_ORDER)}
    y_integer = np.asarray([label_to_int[label] for label in y_train], dtype=int)
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(x_train, y_integer)
    prediction = model.predict(x_test).astype(int)
    return np.asarray([CLASS_ORDER[index] for index in prediction])


def paired_delta_tests(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for method in IMPUTATION_METHODS:
        method_rows = metrics[metrics["Imputation_method"] == method]
        for minuend, subtrahend in PAIRED_COMPARISONS:
            comparison = f"{minuend} minus {subtrahend}"
            for metric in METRICS:
                left = method_rows[method_rows["Feature_set"] == minuend].set_index("Outer_fold")[metric]
                right = method_rows[method_rows["Feature_set"] == subtrahend].set_index("Outer_fold")[metric]
                folds = sorted(set(left.index) & set(right.index))
                deltas = (left.loc[folds] - right.loc[folds]).astype(float)
                for fold, delta in deltas.items():
                    detail_rows.append(
                        {
                            "Imputation_method": method, "Outer_fold": fold,
                            "Comparison": comparison, "Metric": metric, "Delta": delta,
                        }
                    )
                t_stat, t_p = stats.ttest_1samp(deltas.to_numpy(), 0.0)
                try:
                    w_stat, w_p = stats.wilcoxon(deltas.to_numpy(), zero_method="wilcox")
                except ValueError:
                    w_stat, w_p = np.nan, np.nan
                summary_rows.append(
                    {
                        "Imputation_method": method,
                        "Comparison": comparison,
                        "Metric": metric,
                        "N_paired_folds": len(deltas),
                        "Mean_delta": deltas.mean(),
                        "SD_delta": deltas.std(ddof=1),
                        "Min_delta": deltas.min(),
                        "Max_delta": deltas.max(),
                        "Paired_t_stat": t_stat,
                        "Paired_t_p": t_p,
                        "Wilcoxon_stat": w_stat,
                        "Wilcoxon_p": w_p,
                    }
                )
    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def plot_results(summary: pd.DataFrame, delta_detail: pd.DataFrame) -> None:
    from matplotlib.lines import Line2D

    metrics = pd.read_excel(OUT_FILE, sheet_name="outer_fold_metrics")
    method_specs = {
        "global_mean": {"label": "GM", "color": "#005A91", "offset": -0.055},
        "knn": {"label": "KNN", "color": "#C84E00", "offset": 0.055},
    }
    comparisons = [
        {
            "left": "Non_ratio_baseline",
            "right": "Non_ratio_plus_fold_novel",
            "title": "Incremental contribution of fold-specific\nnovel ratio champions",
            "ticks": [
                "Non-ratio baseline\nn = 47",
                "Non-ratio baseline +\nfold-specific novel ratio champions\nGM: n = 140–150; KNN: n = 125–130",
            ],
        },
        {
            "left": "Full_candidate_features",
            "right": "Fold_cluster_champions",
            "title": "Performance retention after\ncorrelation-aware compression",
            "ticks": [
                "Full candidate features\nn = 443",
                "Fold-specific cluster champions\nGM: n = 117–125; KNN: n = 100–104",
            ],
        },
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.65), sharey=True, constrained_layout=True)
    for panel_index, (ax, comparison) in enumerate(zip(axes, comparisons)):
        for method, spec in method_specs.items():
            subset = metrics[metrics["Imputation_method"] == method]
            left = subset[subset["Feature_set"] == comparison["left"]].set_index("Outer_fold")["Macro_F1"]
            right = subset[subset["Feature_set"] == comparison["right"]].set_index("Outer_fold")["Macro_F1"]
            folds = sorted(set(left.index) & set(right.index))
            x = np.array([0.0, 1.0]) + spec["offset"]
            for fold in folds:
                values = [left.loc[fold], right.loc[fold]]
                ax.plot(x, values, color=spec["color"], alpha=0.46, linewidth=1.15, zorder=1)
                ax.scatter(x, values, color=spec["color"], alpha=0.72, s=22, edgecolor="white", linewidth=0.4, zorder=2)

            means = np.array([left.loc[folds].mean(), right.loc[folds].mean()])
            stds = np.array([left.loc[folds].std(ddof=1), right.loc[folds].std(ddof=1)])
            ax.errorbar(
                x, means, yerr=stds, fmt="D", markersize=7.5,
                color=spec["color"], markerfacecolor=spec["color"],
                markeredgecolor="white", markeredgewidth=0.7,
                capsize=4, elinewidth=1.8, capthick=1.8, zorder=4,
            )
            delta = float((right.loc[folds] - left.loc[folds]).mean())
            delta_text = f"+{delta:.3f}" if delta >= 0 else f"−{abs(delta):.3f}"
            delta_y = 0.942 - (0.008 * list(method_specs).index(method))
            ax.text(
                0.995, delta_y, f'{spec["label"]}: mean ΔMacro-F1 = {delta_text}',
                transform=ax.get_yaxis_transform(), ha="right", va="top",
                color=spec["color"], fontsize=9, fontweight="bold",
            )

        ax.set_xlim(-0.25, 1.25)
        ax.set_ylim(0.83, 0.945)
        ax.set_xticks([0, 1], comparison["ticks"])
        ax.set_title(comparison["title"], pad=8, fontsize=10, fontweight="bold", loc="center")
        ax.grid(axis="y", color="#BFBFBF", linewidth=0.8, alpha=0.55)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#111111")
        ax.spines["bottom"].set_color("#111111")
        ax.spines["left"].set_linewidth(1.35)
        ax.spines["bottom"].set_linewidth(1.35)
        ax.tick_params(axis="both", colors="#111111", width=1.25, length=4.5)
        for tick_label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            tick_label.set_fontweight("bold")
            tick_label.set_color("#111111")
        ax.text(-0.19, 1.055, chr(ord("a") + panel_index), transform=ax.transAxes, fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Outer-test Macro-F1", fontweight="bold", color="#111111")
    legend_handles = [
        Line2D([0], [0], color=spec["color"], marker="D", markersize=6, linewidth=1.2, label=spec["label"])
        for spec in method_specs.values()
    ]
    legend = fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.52, 1.08), ncol=2, frameon=False)
    for legend_text in legend.get_texts():
        legend_text.set_fontweight("bold")
    save_figure(fig, "06_feature_contribution_summary")

    macro = delta_detail[delta_detail["Metric"] == "Macro_F1"].copy()
    comp_order = [f"{a} minus {b}" for a, b in PAIRED_COMPARISONS]
    comp_labels = ["Novel-ratio increment", "Champion compression"]
    fig, ax = plt.subplots(figsize=(7.6, 3.8), constrained_layout=True)
    base_y = np.arange(len(comp_order))[::-1]
    for method, spec in method_specs.items():
        offset = -0.10 if method == "global_mean" else 0.10
        for index, comparison_name in enumerate(comp_order):
            values = macro[(macro["Imputation_method"] == method) & (macro["Comparison"] == comparison_name)]["Delta"].to_numpy()
            y = base_y[index] + offset
            jitter = np.linspace(-0.035, 0.035, len(values))
            ax.scatter(values, np.full(len(values), y) + jitter, s=22, color=spec["color"], alpha=0.38, edgecolor="white", linewidth=0.35)
            ax.errorbar(values.mean(), y, xerr=values.std(ddof=1), fmt="D", markersize=6.5, color=spec["color"], markeredgecolor="white", markeredgewidth=0.7, capsize=3.5, zorder=4)
    ax.axvline(0, color="#333333", linewidth=1.0, linestyle="--")
    ax.set_yticks(base_y, comp_labels)
    ax.set_xlabel("Paired ΔMacro-F1")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(handles=legend_handles, loc="lower right", frameon=False, ncol=2)
    save_figure(fig, "06_paired_macro_f1_differences")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    confusion_rows_all: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    reference_metadata: pd.DataFrame | None = None

    for method in IMPUTATION_METHODS:
        for fold in range(1, N_OUTER_FOLDS + 1):
            train, test = load_fold(method, fold)
            candidates = feature_columns(train)
            metadata = build_feature_metadata(candidates)
            if reference_metadata is None:
                reference_metadata = metadata
            elif metadata["Feature"].tolist() != reference_metadata["Feature"].tolist():
                raise ValueError(f"Candidate feature inventory differs for {method} fold {fold}.")
            feature_sets = build_fold_feature_sets(candidates, load_fold_champions(method, fold))
            y_train = train[TYPE_COL].astype(str).to_numpy()
            y_test = test[TYPE_COL].astype(str).to_numpy()

            for feature_set in FEATURE_SET_ORDER:
                features = feature_sets[feature_set]
                feature_rows.extend(
                    {
                        "Imputation_method": method,
                        "Outer_fold": fold,
                        "Feature_set": feature_set,
                        "Feature_order": index + 1,
                        "Feature": feature,
                    }
                    for index, feature in enumerate(features)
                )
                x_train, x_test = numeric_pair(train, test, features)
                prediction = fit_predict(x_train, y_train, x_test)
                context = {"Imputation_method": method, "Outer_fold": fold, "Feature_set": feature_set}
                metrics_rows.append(
                    {
                        **context,
                        "N_features": len(features),
                        "N_train": len(train),
                        "N_test": len(test),
                        **evaluate_predictions(y_test, prediction),
                    }
                )
                class_rows.extend(classwise_rows(y_test, prediction, context))
                confusion_rows_all.extend(confusion_rows(y_test, prediction, context))
                print(f"XGBoost {method:11s} fold={fold} {feature_set:31s} done", flush=True)

    metrics = pd.DataFrame(metrics_rows)
    expected_rows = len(IMPUTATION_METHODS) * N_OUTER_FOLDS * len(FEATURE_SET_ORDER)
    if len(metrics) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} outer-test results; obtained {len(metrics)}")
    summary = summarize_folds(metrics, ["Imputation_method", "Feature_set"])
    classwise = pd.DataFrame(class_rows)
    confusion = pd.DataFrame(confusion_rows_all)
    features = pd.DataFrame(feature_rows)
    delta_detail, delta_summary = paired_delta_tests(metrics)
    classical = get_classical_features(reference_metadata["Feature"].tolist())

    with pd.ExcelWriter(OUT_FILE, engine="openpyxl") as writer:
        metrics.to_excel(writer, sheet_name="outer_fold_metrics", index=False)
        summary.to_excel(writer, sheet_name="summary_by_feature_set", index=False)
        classwise.to_excel(writer, sheet_name="classwise_metrics", index=False)
        confusion.to_excel(writer, sheet_name="confusion_matrix_long", index=False)
        delta_detail.to_excel(writer, sheet_name="paired_delta_by_fold", index=False)
        delta_summary.to_excel(writer, sheet_name="paired_delta_tests", index=False)
        features.to_excel(writer, sheet_name="fold_feature_inventory", index=False)
        reference_metadata.to_excel(writer, sheet_name="feature_metadata", index=False)
        pd.DataFrame(
            {"Classical_concept": [concept for concept, _ in CLASSICAL_FEATURE_ALIASES], "Matched_feature": classical}
        ).to_excel(writer, sheet_name="classical_feature_matches", index=False)
        pd.DataFrame(
            {
                "Item": [
                    "Feature selection scope", "Global Step 05 list used for performance",
                    "Correlation threshold", "Expected fold-result rows", "Observed fold-result rows",
                ],
                "Value": [
                    "Current outer-training partition only", False, 0.90,
                    expected_rows, len(metrics),
                ],
            }
        ).to_excel(writer, sheet_name="validation_scope", index=False)

    style_workbook(OUT_FILE)
    plot_results(summary, delta_detail)
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
