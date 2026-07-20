from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .features import CLASS_ORDER


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    macro = precision_recall_fscore_support(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, labels=CLASS_ORDER, average="weighted", zero_division=0)
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Macro_precision": float(macro[0]),
        "Macro_recall": float(macro[1]),
        "Macro_F1": float(macro[2]),
        "Weighted_precision": float(weighted[0]),
        "Weighted_recall": float(weighted[1]),
        "Weighted_F1": float(weighted[2]),
    }


def classwise_rows(y_true: np.ndarray, y_pred: np.ndarray, context: dict[str, object]) -> list[dict[str, object]]:
    report = classification_report(y_true, y_pred, labels=CLASS_ORDER, target_names=CLASS_ORDER, output_dict=True, zero_division=0)
    return [
        {
            **context,
            "Class": class_name,
            "Precision": float(report[class_name]["precision"]),
            "Recall": float(report[class_name]["recall"]),
            "F1": float(report[class_name]["f1-score"]),
            "Support": int(report[class_name]["support"]),
        }
        for class_name in CLASS_ORDER
    ]


def confusion_rows(y_true: np.ndarray, y_pred: np.ndarray, context: dict[str, object]) -> list[dict[str, object]]:
    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    rows: list[dict[str, object]] = []
    for true_index, true_name in enumerate(CLASS_ORDER):
        for pred_index, pred_name in enumerate(CLASS_ORDER):
            rows.append({**context, "True_class": true_name, "Predicted_class": pred_name, "Count": int(matrix[true_index, pred_index])})
    return rows


def summarize_folds(metrics: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    value_columns = [
        "N_features", "Accuracy", "Balanced_accuracy", "Macro_precision", "Macro_recall", "Macro_F1",
        "Weighted_precision", "Weighted_recall", "Weighted_F1",
    ]
    aggregations: dict[str, tuple[str, str]] = {}
    for column in value_columns:
        aggregations[f"{column}_mean"] = (column, "mean")
        aggregations[f"{column}_std"] = (column, "std")
    aggregations["N_features_min"] = ("N_features", "min")
    aggregations["N_features_max"] = ("N_features", "max")
    aggregations["N_outer_folds"] = ("Outer_fold", "nunique")
    return metrics.groupby(group_columns, as_index=False).agg(**aggregations)

