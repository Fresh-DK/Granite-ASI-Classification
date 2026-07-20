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

from granite_ml.config import RESULTS_DIR
from granite_ml.features import IMPUTATION_METHODS, N_OUTER_FOLDS, TYPE_COL, build_fold_feature_sets
from granite_ml.io import feature_columns, load_fold, load_fold_champions
from granite_ml.excel import style_workbook
from granite_ml.metrics import classwise_rows, evaluate_predictions, summarize_folds
from granite_ml.models import make_model


OUT_DIR = RESULTS_DIR / "09_class_weight_sensitivity"
OUT_FILE = OUT_DIR / "09_class_weight_sensitivity_results.xlsx"
MODEL_NAME = "SVM"
FEATURE_SET_ORDER = ["Fold_cluster_champions", "Non_ratio_plus_fold_novel"]
FEATURE_LABELS = {
    "Fold_cluster_champions": "Fold cluster champions",
    "Non_ratio_plus_fold_novel": "Baseline + fold novel ratios",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, object]] = []
    classes: list[dict[str, object]] = []

    for method in IMPUTATION_METHODS:
        for fold in range(1, N_OUTER_FOLDS + 1):
            train, test = load_fold(method, fold)
            feature_sets = build_fold_feature_sets(feature_columns(train), load_fold_champions(method, fold))
            y_train = train[TYPE_COL].astype(str).to_numpy()
            y_test = test[TYPE_COL].astype(str).to_numpy()
            for feature_set in FEATURE_SET_ORDER:
                features = feature_sets[feature_set]
                x_train = train[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
                x_test = test[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
                for weight_label, class_weight in (("None", None), ("balanced", "balanced")):
                    model = make_model(MODEL_NAME, class_weight=class_weight)
                    model.fit(x_train, y_train)
                    prediction = model.predict(x_test)
                    context = {
                        "Model": MODEL_NAME,
                        "Imputation_method": method,
                        "Outer_fold": fold,
                        "Feature_set": feature_set,
                        "Class_weight": weight_label,
                    }
                    metrics.append({**context, "N_features": len(features), **evaluate_predictions(y_test, prediction)})
                    classes.extend(classwise_rows(y_test, prediction, context))
                    print(f"SVM {method:11s} fold={fold} {feature_set:31s} weight={weight_label}", flush=True)

    metric_table = pd.DataFrame(metrics)
    expected_rows = len(IMPUTATION_METHODS) * N_OUTER_FOLDS * len(FEATURE_SET_ORDER) * 2
    if len(metric_table) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} results; obtained {len(metric_table)}")
    class_table = pd.DataFrame(classes)
    keys = ["Model", "Imputation_method", "Feature_set"]
    summary = summarize_folds(metric_table, keys + ["Class_weight"])
    class_summary = class_table.groupby(keys + ["Class_weight", "Class"], as_index=False).agg(
        Precision_mean=("Precision", "mean"), Recall_mean=("Recall", "mean"), F1_mean=("F1", "mean")
    )

    none = summary[summary["Class_weight"] == "None"].copy()
    balanced = summary[summary["Class_weight"] == "balanced"].copy()
    delta = balanced.merge(none, on=keys, suffixes=("_balanced", "_none"))
    for metric in ("Accuracy", "Balanced_accuracy", "Macro_F1", "Macro_recall"):
        delta[f"Delta_{metric}_balanced_minus_None"] = delta[f"{metric}_mean_balanced"] - delta[f"{metric}_mean_none"]

    s_rows = class_summary[class_summary["Class"] == "S"].copy()
    s_none = s_rows[s_rows["Class_weight"] == "None"]
    s_bal = s_rows[s_rows["Class_weight"] == "balanced"]
    s_delta = s_bal.merge(s_none, on=keys + ["Class"], suffixes=("_balanced", "_none"))
    for metric in ("Precision", "Recall", "F1"):
        s_delta[f"Delta_S_{metric}_balanced_minus_None"] = s_delta[f"{metric}_mean_balanced"] - s_delta[f"{metric}_mean_none"]

    with pd.ExcelWriter(OUT_FILE, engine="openpyxl") as writer:
        metric_table.to_excel(writer, sheet_name="fold_metrics", index=False)
        summary.to_excel(writer, sheet_name="summary_by_weight", index=False)
        delta.to_excel(writer, sheet_name="delta_balanced_vs_none", index=False)
        class_table.to_excel(writer, sheet_name="classwise_metrics", index=False)
        class_summary.to_excel(writer, sheet_name="classwise_summary", index=False)
        s_delta.to_excel(writer, sheet_name="S_type_delta", index=False)
        pd.DataFrame(
            {
                "Item": [
                    "Model", "Feature selection scope", "Global Step 05 list used for performance",
                    "Expected fold-result rows", "Observed fold-result rows",
                ],
                "Value": [
                    MODEL_NAME, "Current outer-training partition only", False,
                    expected_rows, len(metric_table),
                ],
            }
        ).to_excel(writer, sheet_name="validation_scope", index=False)

    style_workbook(OUT_FILE)
    plot = summary.copy()
    plot["Feature_label"] = plot["Feature_set"].map(FEATURE_LABELS)
    plot["Method"] = plot["Imputation_method"].map({"global_mean": "GM", "knn": "KNN"})
    plot["Label"] = plot["Method"] + " | " + plot["Feature_label"]
    none_plot = plot[plot["Class_weight"] == "None"].set_index("Label")["Macro_F1_mean"]
    balanced_plot = plot[plot["Class_weight"] == "balanced"].set_index("Label")["Macro_F1_mean"]
    order = [f"{method} | {FEATURE_LABELS[feature]}" for method in ("GM", "KNN") for feature in FEATURE_SET_ORDER]
    x = np.arange(len(order))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width / 2, none_plot.reindex(order), width, label="None", color="#4C78A8")
    ax.bar(x + width / 2, balanced_plot.reindex(order), width, label="Balanced", color="#F58518")
    ax.set_xticks(x, order, rotation=20, ha="right")
    ax.set_ylabel("Outer-test Macro-F1")
    ax.set_title("SVM class-weight sensitivity")
    ax.legend(title="Class weight")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "09_svm_class_weight_macro_f1.png", dpi=1000, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / "09_svm_class_weight_macro_f1.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
