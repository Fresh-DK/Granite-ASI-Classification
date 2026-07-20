"""Create publication-ready summary figures from rerun pipeline outputs.

Every key figure is exported as a 1000-dpi PNG and a vector PDF. The best
machine-learning configuration is derived from Step 08 output, never hardcoded.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from granite_ml.excel import style_workbook
RESULTS = ROOT / "results"
OUT = RESULTS / "12_summary_figures"
OUT.mkdir(parents=True, exist_ok=True)

RHO_FILE = RESULTS / "04_cluster_champions" / "summary_across_outer_folds_by_method_and_rho.xlsx"
RHO_RAW_FILE = RESULTS / "04_cluster_champions" / "all_outer_folds_rho_sensitivity_raw.xlsx"
STABILITY_FILE = RESULTS / "05_stability" / "rho090_stable_champions_and_ratio_candidates_summary.xlsx"
MODEL_FILE = RESULTS / "08_model_comparison" / "08_four_feature_sets_seven_model_results.xlsx"
WEIGHT_FILE = RESULTS / "09_class_weight_sensitivity" / "09_class_weight_sensitivity_results.xlsx"
TRADITIONAL_FILE = RESULTS / "10_traditional_baseline" / "10_traditional_diagram_baseline_results.xlsx"

DPI = 1000
METHOD_LABELS = {"global_mean": "GM", "knn": "KNN"}
FEATURE_ORDER = [
    "Non_ratio_baseline", "Non_ratio_plus_fold_novel",
    "Full_candidate_features", "Fold_cluster_champions",
]
FEATURE_LABELS = {
    "Non_ratio_baseline": "Non-ratio baseline",
    "Non_ratio_plus_fold_novel": "Non-ratio + fold-specific novel ratios",
    "Full_candidate_features": "Full candidate features",
    "Fold_cluster_champions": "Fold cluster champions",
}
COLORS = {
    "global_mean": "#0072B2", "knn": "#D55E00",
    "Non_ratio_baseline": "#4C78A8",
    "Non_ratio_plus_fold_novel": "#F58518",
    "Full_candidate_features": "#B8B8B8",
    "Fold_cluster_champions": "#54A24B",
}
MODEL_ORDER = ["KNN", "LogisticRegression", "SVM", "RandomForest", "ExtraTrees", "GBDT", "MLP"]
MODEL_LABELS = {
    "KNN": "KNN",
    "LogisticRegression": "Logistic regression",
    "SVM": "SVM",
    "RandomForest": "Random forest",
    "ExtraTrees": "ExtraTrees",
    "GBDT": "GBDT",
    "MLP": "MLP",
}


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans", "font.size": 9,
            "axes.titlesize": 10, "axes.labelsize": 9,
            "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
            "axes.spines.top": False, "axes.spines.right": False,
            "figure.dpi": 150, "savefig.dpi": DPI,
            "pdf.fonttype": 42, "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.11, 1.04, label, transform=ax.transAxes, fontsize=11, fontweight="bold")


def correlation_threshold_tradeoff(rho: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), constrained_layout=True)
    specs = [
        ("n_high_corr_pairs_mean", "n_high_corr_pairs_std", "Highly correlated pairs"),
        ("n_champions_mean", "n_champions_std", "Retained cluster champions"),
    ]
    for ax, (mean_col, std_col, ylabel), letter in zip(axes, specs, ["a", "b"]):
        for method in ["global_mean", "knn"]:
            d = rho[rho["Method"] == method].sort_values("rho_th")
            ax.errorbar(d["rho_th"], d[mean_col], yerr=d[std_col], color=COLORS[method], marker="o", capsize=3, linewidth=1.5, label=METHOD_LABELS[method])
        ax.axvline(0.90, color="#333333", linestyle="--", linewidth=1.3)
        ax.text(0.902, 0.97, "Predefined threshold", transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=8, rotation=90)
        ax.set_xlabel(r"Absolute Spearman correlation threshold, $|\rho_s|$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(rho["rho_th"].unique()))
        ax.grid(axis="y", alpha=0.22)
        panel_label(ax, letter)
    axes[0].legend(frameon=False)
    save_figure(fig, "correlation_threshold_tradeoff")


def fold_specific_champion_counts(rho_raw: pd.DataFrame) -> pd.DataFrame:
    selected = rho_raw[np.isclose(rho_raw["rho_th"], 0.90)].sort_values(["Method", "Outer_fold"]).copy()
    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    for method in ["global_mean", "knn"]:
        d = selected[selected["Method"] == method]
        mean, sd = d["n_champions"].mean(), d["n_champions"].std(ddof=1)
        ax.plot(d["Outer_fold"], d["n_champions"], color=COLORS[method], marker="o", markersize=5, linewidth=1.6, label=f"{METHOD_LABELS[method]} ({mean:.1f} ± {sd:.1f})")
        for x, y in zip(d["Outer_fold"], d["n_champions"]):
            ax.annotate(f"{int(y)}", (x, y), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7, color=COLORS[method])
    ax.set_xlabel("Outer fold")
    ax.set_ylabel("Retained fold-specific cluster champions")
    ax.set_xticks(sorted(selected["Outer_fold"].unique()))
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    save_figure(fig, "fold_specific_champion_counts")
    return selected


def stable_feature_scores(stability: pd.DataFrame) -> pd.DataFrame:
    top = stability.sort_values(["Mean_champion_score", "Appearance_count"], ascending=False).head(20).sort_values("Mean_champion_score").copy()
    category = np.where(top["Is_classical_feature"].astype(bool), "Classical index", np.where(top["Is_candidate_novel_ratio"].astype(bool), "Constructed ratio", "Original/composite"))
    category_colors = {"Classical index": "#0072B2", "Constructed ratio": "#E69F00", "Original/composite": "#009E73"}
    labels = top.get("Display_feature", top["Feature"]).fillna(top["Feature"]).astype(str)
    fig, ax = plt.subplots(figsize=(7.8, 6.4))
    y = np.arange(len(top))
    for cat, color in category_colors.items():
        mask = category == cat
        ax.errorbar(top.loc[mask, "Mean_champion_score"], y[mask], xerr=top.loc[mask, "SD_champion_score"].fillna(0), fmt="o", color=color, capsize=2, label=cat)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Mean champion score across method–fold combinations")
    ax.grid(axis="x", alpha=0.22)
    xmax = float((top["Mean_champion_score"] + top["SD_champion_score"].fillna(0)).max())
    for yi, (_, row) in enumerate(top.iterrows()):
        ax.text(xmax * 1.02, yi, f'{int(row["Appearance_count"])}/10', va="center", fontsize=7)
    ax.set_xlim(right=xmax * 1.17)
    ax.text(0.99, 1.01, "selection frequency", transform=ax.transAxes, ha="right", fontsize=7)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3)
    fig.subplots_adjust(left=0.22, right=0.96, top=0.96, bottom=0.17)
    save_figure(fig, "stable_feature_scores")
    return top


def model_benchmark(model_summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), sharex=True, sharey=True, constrained_layout=True)
    offsets = np.linspace(-0.24, 0.24, len(FEATURE_ORDER))
    for ax, method, letter in zip(axes, ["global_mean", "knn"], ["a", "b"]):
        d = model_summary[model_summary["Imputation_method"] == method]
        ybase = np.arange(len(MODEL_ORDER))
        for offset, feature_set in zip(offsets, FEATURE_ORDER):
            sub = d[d["Feature_set"] == feature_set].set_index("Model").reindex(MODEL_ORDER)
            ax.errorbar(sub["Macro_F1_mean"], ybase + offset, xerr=sub["Macro_F1_std"], fmt="o", capsize=2, color=COLORS[feature_set], label=FEATURE_LABELS[feature_set])
        ax.set_yticks(ybase, [MODEL_LABELS[model] for model in MODEL_ORDER])
        ax.invert_yaxis()
        ax.set_xlabel("Outer-test Macro-F1 (mean ± SD)", fontweight="bold", color="#111111")
        ax.set_title(METHOD_LABELS[method], fontweight="bold", color="#111111")
        ax.grid(axis="x", color="#BFBFBF", linewidth=0.8, alpha=0.55)
        ax.spines["left"].set_color("#111111")
        ax.spines["bottom"].set_color("#111111")
        ax.spines["left"].set_linewidth(1.35)
        ax.spines["bottom"].set_linewidth(1.35)
        ax.tick_params(axis="both", colors="#111111", width=1.25, length=4.5)
        for tick_label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            tick_label.set_fontweight("bold")
            tick_label.set_color("#111111")
        panel_label(ax, letter)
    legend = axes[0].legend(frameon=False, loc="lower left", bbox_to_anchor=(0, -0.42), ncol=2)
    for legend_text in legend.get_texts():
        legend_text.set_fontweight("bold")
        legend_text.set_color("#111111")
    save_figure(fig, "model_benchmark")


def select_best_configuration(model_summary: pd.DataFrame) -> dict[str, str]:
    best = model_summary.sort_values(["Macro_F1_mean", "Balanced_accuracy_mean", "Accuracy_mean"], ascending=False).iloc[0]
    return {"Model": str(best["Model"]), "Imputation_method": str(best["Imputation_method"]), "Feature_set": str(best["Feature_set"])}


def best_model_diagnostics(model_fold: pd.DataFrame, class_summary: pd.DataFrame, confusion: pd.DataFrame, target: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = model_fold.copy()
    cm = confusion.copy()
    cls = class_summary.copy()
    for col, value in target.items():
        selected = selected[selected[col] == value]
        cm = cm[cm[col] == value]
        cls = cls[cls[col] == value]
    matrix = cm.pivot_table(index="True_class", columns="Predicted_class", values="Count", aggfunc="sum", fill_value=0)
    class_order = [c for c in ["A", "S", "I"] if c in matrix.index]
    matrix = matrix.reindex(index=class_order, columns=class_order, fill_value=0)
    normalized = matrix.div(matrix.sum(axis=1), axis=0)
    cls = cls.set_index("Class").reindex(class_order)

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), constrained_layout=True)
    im = axes[0].imshow(normalized.values, cmap="Blues", vmin=0, vmax=1)
    axes[0].set_xticks(range(len(class_order)), class_order)
    axes[0].set_yticks(range(len(class_order)), class_order)
    axes[0].set_xlabel("Predicted class", fontweight="bold")
    axes[0].set_ylabel("True class", fontweight="bold")
    axes[0].set_title("Normalized confusion matrix", fontweight="bold")
    for i in range(len(class_order)):
        for j in range(len(class_order)):
            value = normalized.iloc[i, j]
            axes[0].text(j, i, f"{value:.2f}", ha="center", va="center", fontweight="bold", color="white" if value > 0.55 else "black")
    colorbar = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    for tick_label in colorbar.ax.get_yticklabels():
        tick_label.set_fontweight("bold")

    x = np.arange(len(class_order))
    for offset, metric, color in [(-0.18, "Precision", "#0072B2"), (0, "Recall", "#D55E00"), (0.18, "F1", "#009E73")]:
        axes[1].errorbar(x + offset, cls[f"{metric}_mean"], yerr=cls[f"{metric}_std"], fmt="o", markersize=6.5, capsize=3.5, elinewidth=1.6, capthick=1.6, color=color, label=metric)
    axes[1].set_xticks(x, class_order)
    axes[1].set_ylim(0.5, 1.01)
    axes[1].set_ylabel("Score (mean ± SD)", fontweight="bold")
    axes[1].set_title("Class-wise outer-test performance", fontweight="bold")
    axes[1].grid(axis="y", color="#BFBFBF", linewidth=0.8, alpha=0.55)
    class_legend = axes[1].legend(frameon=False)
    for legend_text in class_legend.get_texts():
        legend_text.set_fontweight("bold")

    selected = selected.sort_values("Outer_fold")
    axes[2].plot(selected["Outer_fold"], selected["Macro_F1"], marker="o", markersize=6.5, linewidth=2.0, color="#005A91")
    axes[2].axhline(selected["Macro_F1"].mean(), linestyle="--", linewidth=1.6, color="#222222", label="Five-fold mean")
    axes[2].set_xticks(selected["Outer_fold"])
    axes[2].set_xlabel("Outer fold", fontweight="bold")
    axes[2].set_ylabel("Macro-F1", fontweight="bold")
    axes[2].set_title("Fold-to-fold variability", fontweight="bold")
    axes[2].grid(axis="y", color="#BFBFBF", linewidth=0.8, alpha=0.55)
    fold_legend = axes[2].legend(frameon=False)
    for legend_text in fold_legend.get_texts():
        legend_text.set_fontweight("bold")
    for ax, letter in zip(axes, ["a", "b", "c"]):
        ax.tick_params(axis="both", colors="#111111", width=1.25, length=4.5)
        for tick_label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            tick_label.set_fontweight("bold")
            tick_label.set_color("#111111")
        for spine in ax.spines.values():
            spine.set_color("#111111")
            spine.set_linewidth(1.25)
        panel_label(ax, letter)
    config_label = f"{target['Model']} | {METHOD_LABELS[target['Imputation_method']]} | {FEATURE_LABELS[target['Feature_set']]}"
    fig.suptitle(f"Best default configuration: {config_label}", fontsize=10.5, fontweight="bold")
    save_figure(fig, "best_model_diagnostics")
    return normalized.reset_index(), selected


def machine_vs_traditional(model_fold: pd.DataFrame, traditional_fold: pd.DataFrame, target: dict[str, str]) -> pd.DataFrame:
    machine = model_fold.copy()
    for col, value in target.items():
        machine = machine[machine[col] == value]
    machine = machine.assign(Method_name=f"{target['Model']} ({FEATURE_LABELS[target['Feature_set']]})", Score=machine["Macro_F1"])
    rule_labels = {
        "Rule_a_GaAl_FeOstarMgO_plus_ACNK": "Traditional rule a",
        "Rule_b_GaAl_Zr_plus_ACNK": "Traditional rule b",
        "Rule_c_GaAl_HFSE_plus_ACNK": "Traditional rule c",
        "Rule_majority_abc_plus_ACNK": "Majority vote",
    }
    trad = traditional_fold[(traditional_fold["Imputation_method"] == target["Imputation_method"]) & traditional_fold["Rule"].isin(rule_labels)].copy()
    trad["Method_name"] = trad["Rule"].map(rule_labels)
    trad["Score"] = trad["Strict_macro_F1_all"]
    combined = pd.concat([machine[["Outer_fold", "Method_name", "Score"]], trad[["Outer_fold", "Method_name", "Score"]]], ignore_index=True)
    order = [machine["Method_name"].iloc[0], "Traditional rule a", "Traditional rule b", "Traditional rule c", "Majority vote"]
    summary = combined.groupby("Method_name")["Score"].agg(["mean", "std"]).reindex(order).reset_index()
    category_specs = [
        ("SVM", "Non-ratio baseline + fold-specific\nnovel ratio champions"),
        ("Traditional rule a", "Ga/Al–FeO*/MgO\n+ A/CNK"),
        ("Traditional rule b", "Ga/Al–Zr\n+ A/CNK"),
        ("Traditional rule c", "Ga/Al–HFSE\n+ A/CNK"),
        ("Majority vote", "Rules a–c\n+ A/CNK"),
    ]
    x = np.arange(len(summary))
    point_colors = ["#005A91", "#666666", "#666666", "#C84E00", "#666666"]
    fig, ax = plt.subplots(figsize=(10.4, 5.25))

    # A vertical point-range layout keeps the method labels beneath their
    # estimates and gives the exported canvas a naturally centred footprint.
    for xi, row, color in zip(x, summary.itertuples(), point_colors):
        ax.errorbar(
            xi, row.mean, yerr=row.std, fmt="o", markersize=9,
            color=color, markeredgecolor="white", markeredgewidth=0.8,
            capsize=5, elinewidth=2.1, capthick=2.1, zorder=4,
        )
        ax.text(
            xi, min(row.mean + row.std + 0.025, 0.974),
            f"{row.mean:.3f} ± {row.std:.3f}",
            ha="center", va="bottom", fontsize=8.8,
            fontweight="bold", color=color,
        )

    ax.set_xticks(x, [main for main, _ in category_specs])
    ax.set_xlim(-0.55, len(summary) - 0.45)
    ax.set_ylim(0.38, 1.00)
    ax.set_ylabel("Outer-test Macro-F1 (mean ± SD)", fontweight="bold", color="#111111")
    ax.set_title(
        f"Machine learning versus traditional geochemical rules\n"
        f"({METHOD_LABELS[target['Imputation_method']]} imputation)",
        fontsize=12, fontweight="bold", color="#111111", pad=16,
    )
    ax.grid(axis="y", color="#B8B8B8", linewidth=0.9, alpha=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#111111")
    ax.spines["bottom"].set_color("#111111")
    ax.spines["left"].set_linewidth(1.45)
    ax.spines["bottom"].set_linewidth(1.45)
    ax.tick_params(axis="both", colors="#111111", width=1.3, length=5)
    for tick_label in ax.get_yticklabels():
        tick_label.set_fontweight("bold")
        tick_label.set_color("#111111")
    for tick_label in ax.get_xticklabels():
        tick_label.set_fontweight("bold")
        tick_label.set_color("#111111")
        tick_label.set_fontsize(9.2)
    for xi, (_, detail) in zip(x, category_specs):
        ax.text(
            xi, -0.088, detail, transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=7.5, linespacing=1.25,
            fontweight="semibold", color="#333333", clip_on=False,
        )
    fig.subplots_adjust(left=0.095, right=0.975, top=0.84, bottom=0.27)
    save_figure(fig, "machine_vs_traditional_baselines")
    return summary


def class_weight_effects(delta: pd.DataFrame, s_delta: pd.DataFrame) -> pd.DataFrame:
    keys = ["Model", "Imputation_method", "Feature_set"]
    merged = delta.merge(s_delta[keys + ["Delta_S_Recall_balanced_minus_None"]], on=keys, how="left")
    merged["Label"] = merged["Imputation_method"].map(METHOD_LABELS) + " | " + merged["Feature_set"].map(FEATURE_LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), sharey=True, constrained_layout=True)
    specs = [
        ("Delta_Macro_F1_balanced_minus_None", "Δ Macro-F1"),
        ("Delta_S_Recall_balanced_minus_None", "Δ S-type recall"),
    ]
    for ax, (column, title), letter in zip(axes, specs, ["a", "b"]):
        d = merged.sort_values(["Feature_set", "Imputation_method"])
        y = np.arange(len(d))
        ax.scatter(d[column], y, color=[COLORS[m] for m in d["Imputation_method"]])
        ax.axvline(0, color="#333333", linewidth=1)
        ax.set_yticks(y, d["Label"])
        ax.invert_yaxis()
        ax.set_xlabel("Balanced minus unweighted")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.22)
        panel_label(ax, letter)
    save_figure(fig, "class_weight_effects")
    return merged


def main() -> None:
    configure_plotting()
    rho = pd.read_excel(RHO_FILE)
    rho_raw = pd.read_excel(RHO_RAW_FILE)
    stability = pd.read_excel(STABILITY_FILE, sheet_name="stable_interpretation_features")
    model_summary = pd.read_excel(MODEL_FILE, sheet_name="summary_mean_std")
    model_fold = pd.read_excel(MODEL_FILE, sheet_name="fold_metrics")
    class_summary = pd.read_excel(MODEL_FILE, sheet_name="classwise_summary")
    confusion = pd.read_excel(MODEL_FILE, sheet_name="confusion_matrix_long")
    delta = pd.read_excel(WEIGHT_FILE, sheet_name="delta_balanced_vs_none")
    s_delta = pd.read_excel(WEIGHT_FILE, sheet_name="S_type_delta")
    traditional_fold = pd.read_excel(TRADITIONAL_FILE, sheet_name="fold_metrics")

    correlation_threshold_tradeoff(rho)
    champion_counts = fold_specific_champion_counts(rho_raw)
    top_stability = stable_feature_scores(stability)
    model_benchmark(model_summary)
    target = select_best_configuration(model_summary)
    normalized_confusion, best_fold = best_model_diagnostics(model_fold, class_summary, confusion, target)
    comparison_summary = machine_vs_traditional(model_fold, traditional_fold, target)
    weight_effect_summary = class_weight_effects(delta, s_delta)

    source_data_file = OUT / "summary_figure_data.xlsx"
    with pd.ExcelWriter(source_data_file, engine="openpyxl") as writer:
        rho.to_excel(writer, sheet_name="threshold_tradeoff", index=False)
        champion_counts.to_excel(writer, sheet_name="fold_champion_counts", index=False)
        top_stability.to_excel(writer, sheet_name="top_stable_features", index=False)
        model_summary.to_excel(writer, sheet_name="model_benchmark", index=False)
        pd.DataFrame([target]).to_excel(writer, sheet_name="best_configuration", index=False)
        normalized_confusion.to_excel(writer, sheet_name="best_normalized_cm", index=False)
        best_fold.to_excel(writer, sheet_name="best_model_folds", index=False)
        comparison_summary.to_excel(writer, sheet_name="machine_vs_traditional", index=False)
        weight_effect_summary.to_excel(writer, sheet_name="class_weight_effects", index=False)
    style_workbook(source_data_file)
    print(f"Summary figures and source data written to: {OUT}")


if __name__ == "__main__":
    main()
