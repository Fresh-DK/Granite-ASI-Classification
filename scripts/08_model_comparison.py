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
from granite_ml.features import (
    CLASSICAL_FEATURE_ALIASES, CLASS_ORDER, IMPUTATION_METHODS, N_OUTER_FOLDS, TYPE_COL,
    build_feature_metadata, build_fold_feature_sets, get_classical_features,
)
from granite_ml.io import feature_columns, load_fold, load_fold_champions
from granite_ml.excel import style_workbook
from granite_ml.metrics import classwise_rows, confusion_rows, evaluate_predictions, summarize_folds
from granite_ml.models import MODEL_ORDER, make_model, model_parameter_rows


OUT_DIR = RESULTS_DIR / "08_model_comparison"
OUT_FILE = OUT_DIR / "08_four_feature_sets_seven_model_results.xlsx"
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


def numeric_pair(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train = train[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x_test = test[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return x_train, x_test


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=1000, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_plots(summary: pd.DataFrame) -> None:
    colors = ["#4C78A8", "#F58518", "#B8B8B8", "#54A24B"]
    for method in IMPUTATION_METHODS:
        sub = summary[summary["Imputation_method"] == method].copy()
        pivot = sub.pivot(index="Model", columns="Feature_set", values="Macro_F1_mean")
        pivot = pivot.reindex(index=MODEL_ORDER, columns=FEATURE_SET_ORDER)
        fig, ax = plt.subplots(figsize=(13, 6.5))
        pivot.rename(columns=FEATURE_LABELS).plot(
            kind="bar", ax=ax, ylim=(0, 1), width=0.82, color=colors,
        )
        ax.set_ylabel("Outer-test Macro-F1")
        ax.set_xlabel("Classifier")
        ax.set_title(f"Seven-classifier benchmark ({'GM' if method == 'global_mean' else 'KNN'})")
        ax.legend(title="Feature set", bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        save_figure(fig, f"08_macro_f1_{method}")


def fit_predict(model_name: str, x_train: pd.DataFrame, y_train: np.ndarray, x_test: pd.DataFrame) -> np.ndarray:
    model = make_model(model_name)
    if model_name != "MLP":
        model.fit(x_train, y_train)
        return model.predict(x_test)
    label_to_int = {label: index for index, label in enumerate(CLASS_ORDER)}
    y_train_fit = np.asarray([label_to_int[label] for label in y_train], dtype=int)
    model.fit(x_train, y_train_fit)
    prediction_int = model.predict(x_test).astype(int)
    return np.asarray([CLASS_ORDER[index] for index in prediction_int])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metric_records: list[dict[str, object]] = []
    class_records: list[dict[str, object]] = []
    confusion_records: list[dict[str, object]] = []
    feature_records: list[dict[str, object]] = []
    reference_metadata: pd.DataFrame | None = None

    for method in IMPUTATION_METHODS:
        for fold in range(1, N_OUTER_FOLDS + 1):
            train, test = load_fold(method, fold)
            candidate_columns = feature_columns(train)
            metadata = build_feature_metadata(candidate_columns)
            if reference_metadata is None:
                reference_metadata = metadata
            elif metadata["Feature"].tolist() != reference_metadata["Feature"].tolist():
                raise ValueError(f"Candidate feature order differs in {method} fold {fold}.")
            feature_sets = build_fold_feature_sets(candidate_columns, load_fold_champions(method, fold))
            y_train = train[TYPE_COL].astype(str).to_numpy()
            y_test = test[TYPE_COL].astype(str).to_numpy()

            for feature_set in FEATURE_SET_ORDER:
                features = feature_sets[feature_set]
                feature_records.extend(
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
                for model_name in MODEL_ORDER:
                    prediction = fit_predict(model_name, x_train, y_train, x_test)
                    context = {
                        "Model": model_name,
                        "Imputation_method": method,
                        "Outer_fold": fold,
                        "Feature_set": feature_set,
                    }
                    metric_records.append(
                        {
                            **context,
                            "N_features": len(features),
                            "N_train": len(train),
                            "N_test": len(test),
                            **evaluate_predictions(y_test, prediction),
                        }
                    )
                    class_records.extend(classwise_rows(y_test, prediction, context))
                    confusion_records.extend(confusion_rows(y_test, prediction, context))
                    print(f"{model_name:18s} {method:11s} fold={fold} {feature_set:31s} done", flush=True)

    metrics = pd.DataFrame(metric_records)
    expected_rows = len(IMPUTATION_METHODS) * N_OUTER_FOLDS * len(MODEL_ORDER) * len(FEATURE_SET_ORDER)
    if len(metrics) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} outer-test results; obtained {len(metrics)}")
    classwise = pd.DataFrame(class_records)
    confusion = pd.DataFrame(confusion_records)
    features = pd.DataFrame(feature_records)
    summary = summarize_folds(metrics, ["Model", "Imputation_method", "Feature_set"])
    class_summary = classwise.groupby(
        ["Model", "Imputation_method", "Feature_set", "Class"], as_index=False
    ).agg(
        Precision_mean=("Precision", "mean"), Precision_std=("Precision", "std"),
        Recall_mean=("Recall", "mean"), Recall_std=("Recall", "std"),
        F1_mean=("F1", "mean"), F1_std=("F1", "std"),
    )
    best = summary.sort_values("Macro_F1_mean", ascending=False).head(1)
    classical = get_classical_features(reference_metadata["Feature"].tolist())

    with pd.ExcelWriter(OUT_FILE, engine="openpyxl") as writer:
        metrics.to_excel(writer, sheet_name="fold_metrics", index=False)
        summary.to_excel(writer, sheet_name="summary_mean_std", index=False)
        best.to_excel(writer, sheet_name="best_overall", index=False)
        classwise.to_excel(writer, sheet_name="classwise_metrics", index=False)
        class_summary.to_excel(writer, sheet_name="classwise_summary", index=False)
        confusion.to_excel(writer, sheet_name="confusion_matrix_long", index=False)
        features.to_excel(writer, sheet_name="fold_feature_inventory", index=False)
        reference_metadata.to_excel(writer, sheet_name="feature_metadata", index=False)
        pd.DataFrame({"Classical_concept": [c for c, _ in CLASSICAL_FEATURE_ALIASES], "Matched_feature": classical}).to_excel(
            writer, sheet_name="classical_feature_matches", index=False
        )
        pd.DataFrame(model_parameter_rows()).to_excel(writer, sheet_name="model_parameters", index=False)
        pd.DataFrame(
            {
                "Item": [
                    "Feature selection scope", "Global Step 05 list used for performance",
                    "Expected fold-result rows", "Observed fold-result rows", "Primary metric",
                ],
                "Value": [
                    "Current outer-training partition only", False,
                    expected_rows, len(metrics), "Macro-F1",
                ],
            }
        ).to_excel(writer, sheet_name="validation_scope", index=False)

    style_workbook(OUT_FILE)
    save_plots(summary)
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
