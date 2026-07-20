from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["savefig.dpi"] = 1000
import matplotlib.pyplot as plt
import pandas as pd

from granite_ml.config import STABILITY_DIR
from granite_ml.features import (
    IMPUTATION_METHODS,
    N_OUTER_FOLDS,
    RHO_TAG,
    display_feature_name,
    is_candidate_novel_ratio,
    is_classical_feature,
    is_constructed_ratio,
    ratio_group,
)
from granite_ml.io import champion_workbook


MIN_APPEARANCES_STABLE = 8
OUTPUT_FILE = STABILITY_DIR / "rho090_stable_champions_and_ratio_candidates_summary.xlsx"


def stability_level(count: int) -> str:
    if count >= MIN_APPEARANCES_STABLE:
        return "Stable"
    return "Not stable"


def load_records() -> tuple[pd.DataFrame, pd.DataFrame]:
    champion_tables: list[pd.DataFrame] = []
    score_tables: list[pd.DataFrame] = []
    for method in IMPUTATION_METHODS:
        for fold in range(1, N_OUTER_FOLDS + 1):
            workbook = champion_workbook(method, fold, RHO_TAG)
            champions = pd.read_excel(workbook, sheet_name="ClusterChampions")
            if "champion" not in champions:
                raise ValueError(f"Missing champion column in {workbook}")
            champions = champions.copy()
            champions["Method"] = method
            champions["Outer_fold"] = fold
            champions["Source_file"] = str(workbook.relative_to(PROJECT_ROOT))
            champions["Feature"] = champions["champion"].astype(str)
            champion_tables.append(champions)

            scores = pd.read_excel(workbook, sheet_name="FeatureScores_SHAP_innerCV")
            scores = scores.copy()
            scores["Method"] = method
            scores["Outer_fold"] = fold
            score_tables.append(scores)

    return pd.concat(champion_tables, ignore_index=True), pd.concat(score_tables, ignore_index=True)


def summarize_champions(records: pd.DataFrame) -> pd.DataFrame:
    renamed = records.rename(
        columns={
            "champion_score": "Champion_score",
            "champion_importance_mean": "SHAP_importance",
            "champion_topk_freq_ratio": "TopK_frequency",
        }
    )
    summary = renamed.groupby("Feature", as_index=False).agg(
        Appearance_count=("Feature", "size"),
        Global_mean_count=("Method", lambda values: int((values == "global_mean").sum())),
        KNN_count=("Method", lambda values: int((values == "knn").sum())),
        Mean_champion_score=("Champion_score", "mean"),
        SD_champion_score=("Champion_score", "std"),
        Mean_SHAP_importance=("SHAP_importance", "mean"),
        Mean_TopK_frequency=("TopK_frequency", "mean"),
        Mean_cluster_size=("cluster_size", "mean"),
    )
    summary["Appearance_fraction"] = summary["Appearance_count"] / (
        len(IMPUTATION_METHODS) * N_OUTER_FOLDS
    )
    summary["Display_feature"] = summary["Feature"].map(display_feature_name)
    summary["Is_constructed_ratio"] = summary["Feature"].map(is_constructed_ratio)
    summary["Is_classical_feature"] = summary["Feature"].map(is_classical_feature)
    summary["Is_candidate_novel_ratio"] = summary["Feature"].map(is_candidate_novel_ratio)
    summary["Ratio_group"] = summary["Feature"].map(ratio_group)
    summary["Stability_level"] = summary["Appearance_count"].map(stability_level)
    return summary.sort_values(
        ["Appearance_count", "Mean_champion_score", "Feature"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def save_plot(stable_ratios: pd.DataFrame) -> None:
    plot_data = stable_ratios.head(30).sort_values(
        ["Appearance_count", "Mean_champion_score"], ascending=True
    )
    if plot_data.empty:
        return
    fig, ax = plt.subplots(figsize=(10, max(5, 0.32 * len(plot_data))), dpi=1000)
    ax.barh(plot_data["Display_feature"], plot_data["Appearance_count"])
    ax.set_xlabel("Appearance count across 10 method-fold combinations")
    ax.set_ylabel("Post hoc stable novel ratio")
    ax.set_title("Cross-fold occurrence of novel cluster champions")
    ax.set_xlim(0, len(IMPUTATION_METHODS) * N_OUTER_FOLDS + 0.5)
    fig.tight_layout()
    fig.savefig(STABILITY_DIR / "stable_novel_ratio_appearance_counts.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    STABILITY_DIR.mkdir(parents=True, exist_ok=True)
    all_champions, all_scores = load_records()
    summary = summarize_champions(all_champions)
    stable_interpretation_features = summary[
        summary["Appearance_count"] >= MIN_APPEARANCES_STABLE
    ].copy()
    stable_ratios = stable_interpretation_features[
        stable_interpretation_features["Is_candidate_novel_ratio"]
    ].copy()
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        all_champions.to_excel(writer, sheet_name="all_champions_raw", index=False)
        summary.to_excel(writer, sheet_name="champion_stability_summary", index=False)
        all_scores.to_excel(writer, sheet_name="all_feature_scores_raw", index=False)
        stable_interpretation_features.to_excel(
            writer, sheet_name="stable_interpretation_features", index=False
        )
        stable_ratios.to_excel(writer, sheet_name="stable_ratio_candidates", index=False)
        pd.DataFrame(
            {
                "Item": [
                    "Analysis scope",
                    "Allowed use",
                    "Prohibited use",
                    "Ablation performed",
                    "Stable feature minimum appearances",
                ],
                "Value": [
                    "Post hoc aggregation after outer-fold feature selection",
                    "Interpretation and final all-data model construction",
                    "Outer-fold performance estimation on the same five folds",
                    False,
                    MIN_APPEARANCES_STABLE,
                ],
            }
        ).to_excel(writer, sheet_name="scope_and_usage", index=False)

    save_plot(stable_ratios)
    manifest = {
        "rho_tag": RHO_TAG,
        "ablation_performed": False,
        "outer_cv_performance_input": False,
        "minimum_appearances_stable": MIN_APPEARANCES_STABLE,
        "output_file": str(OUTPUT_FILE.relative_to(PROJECT_ROOT)),
    }
    (STABILITY_DIR / "05_run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Stable interpretation features: {len(stable_interpretation_features)}")
    print(f"Stable novel ratio candidates: {len(stable_ratios)}")
    print("These post hoc lists are not used by steps 06, 08, or 09 for outer-fold performance.")


if __name__ == "__main__":
    main()
