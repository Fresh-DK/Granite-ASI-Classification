from __future__ import annotations

from itertools import combinations
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from granite_ml.config import FOLDS_DIR, RAW_DATA_FILE
from granite_ml.features import (
    IMPUTATION_METHODS,
    MAJOR_RATIO_INPUTS,
    NON_FEATURE_COLS,
    TRACE_RATIO_INPUTS,
    TYPE_COL,
)
from granite_ml.io import feature_columns, load_analysis_data


RANDOM_SEED = 42
N_SPLITS = 5
N_REPEATS = 1
KNN_N_NEIGHBORS = 5
KNN_WEIGHTS = "distance"


def fit_iqr_bounds_on_train(
    train: pd.DataFrame, numeric_columns: list[str]
) -> pd.DataFrame:
    """Estimate 1.5-IQR limits using only the outer-training partition."""
    records: list[dict[str, object]] = []
    for column in numeric_columns:
        valid = train[column].astype(float).dropna()
        if len(valid) < 4:
            records.append(
                {
                    "Feature": column,
                    "Q1": np.nan,
                    "Q3": np.nan,
                    "IQR": np.nan,
                    "Lower_bound": np.nan,
                    "Upper_bound": np.nan,
                    "Note": "Valid values < 4; bounds not applied",
                }
            )
            continue

        q1 = valid.quantile(0.25)
        q3 = valid.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            lower = np.nan
            upper = np.nan
            note = "IQR is zero or missing; bounds not applied"
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            note = "Applied"
        records.append(
            {
                "Feature": column,
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "Lower_bound": lower,
                "Upper_bound": upper,
                "Note": note,
            }
        )
    return pd.DataFrame(records)


def apply_iqr_bounds(
    frame: pd.DataFrame,
    numeric_columns: list[str],
    bounds: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mark values outside training-derived IQR limits as missing."""
    output = frame.copy()
    bound_map = bounds.set_index("Feature").to_dict(orient="index")
    records: list[dict[str, object]] = []
    for column in numeric_columns:
        lower = bound_map[column]["Lower_bound"]
        upper = bound_map[column]["Upper_bound"]
        if pd.isna(lower) or pd.isna(upper):
            count = 0
        else:
            mask = (output[column] < lower) | (output[column] > upper)
            count = int(mask.sum())
            output.loc[mask, column] = np.nan
        records.append(
            {"Feature": column, "Outlier_count_marked_as_NaN": count}
        )
    return output, pd.DataFrame(records)


def foldwise_impute(
    train: pd.DataFrame,
    test: pd.DataFrame,
    numeric_columns: list[str],
    method: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit imputation parameters on one outer-training fold only."""
    train_output = train.copy()
    test_output = test.copy()

    if method == "global_mean":
        imputer = SimpleImputer(strategy="mean")
        train_output[numeric_columns] = imputer.fit_transform(train[numeric_columns])
        test_output[numeric_columns] = imputer.transform(test[numeric_columns])
        details = pd.DataFrame(
            {
                "Feature": numeric_columns,
                "Train_missing_filled": [int(train[c].isna().sum()) for c in numeric_columns],
                "Test_missing_filled": [int(test[c].isna().sum()) for c in numeric_columns],
                "Imputation_method": method,
                "Train_mean_used": imputer.statistics_,
            }
        )
        return train_output, test_output, details

    if method != "knn":
        raise ValueError(f"Unknown imputation method: {method}")

    train_input = train[numeric_columns].astype(float)
    test_input = test[numeric_columns].astype(float)
    all_missing = [column for column in numeric_columns if train_input[column].isna().all()]
    if all_missing:
        raise ValueError(f"Training-fold features contain only missing values: {all_missing}")

    train_means = train_input.mean(skipna=True)
    train_stds = train_input.std(skipna=True).replace(0, 1.0)
    train_scaled = (train_input - train_means) / train_stds
    test_scaled = (test_input - train_means) / train_stds

    imputer = KNNImputer(n_neighbors=KNN_N_NEIGHBORS, weights=KNN_WEIGHTS)
    train_scaled_filled = imputer.fit_transform(train_scaled)
    test_scaled_filled = imputer.transform(test_scaled)
    train_output[numeric_columns] = train_scaled_filled * train_stds.to_numpy() + train_means.to_numpy()
    test_output[numeric_columns] = test_scaled_filled * train_stds.to_numpy() + train_means.to_numpy()

    details = pd.DataFrame(
        {
            "Feature": numeric_columns,
            "Train_missing_filled": [int(train[c].isna().sum()) for c in numeric_columns],
            "Test_missing_filled": [int(test[c].isna().sum()) for c in numeric_columns],
            "Imputation_method": method,
            "Train_mean_for_scaling": train_means.values,
            "Train_std_for_scaling": train_stds.values,
        }
    )
    return train_output, test_output, details


def safe_ratio(
    numerator: pd.Series, denominator: pd.Series
) -> tuple[pd.Series, int]:
    """Calculate a ratio, assigning zero when the denominator is zero."""
    zero_mask = denominator == 0
    ratio = (numerator / denominator).replace([np.inf, -np.inf], np.nan)
    ratio.loc[zero_mask] = 0.0
    return ratio, int(zero_mask.sum())


def build_pairwise_ratios(
    frame: pd.DataFrame, columns: tuple[str, ...], prefix: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Columns required for {prefix} ratios are missing: {missing}")

    ratios: dict[str, pd.Series] = {}
    zero_records: list[dict[str, object]] = []
    for numerator, denominator in combinations(columns, 2):
        name = f"{prefix}{numerator}/{denominator}"
        ratios[name], zero_count = safe_ratio(frame[numerator], frame[denominator])
        if zero_count:
            zero_records.append(
                {
                    "Ratio_feature": name,
                    "Denominator": denominator,
                    "Zero_denominator_count": zero_count,
                }
            )
    return pd.DataFrame(ratios, index=frame.index), pd.DataFrame(zero_records)


def add_ratio_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    major, major_zeros = build_pairwise_ratios(frame, MAJOR_RATIO_INPUTS, "R_Major_")
    trace, trace_zeros = build_pairwise_ratios(frame, TRACE_RATIO_INPUTS, "R_Trace_")
    output = pd.concat([frame, major, trace], axis=1)
    zero_summary = pd.concat(
        [major_zeros.assign(Group="Major"), trace_zeros.assign(Group="Trace_REE")],
        ignore_index=True,
    )
    return output, zero_summary


def cross_validator():
    if N_REPEATS == 1:
        return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    return RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_SEED,
    )


def require_finite_features(
    frame: pd.DataFrame, columns: list[str], label: str
) -> None:
    values = frame[columns].to_numpy(dtype=float)
    missing = int(np.isnan(values).sum())
    infinite = int(np.isinf(values).sum())
    if missing or infinite:
        raise ValueError(f"{label} contains {missing} NaN and {infinite} infinite feature values.")


def write_zero_summary(
    writer: pd.ExcelWriter, summary: pd.DataFrame, sheet_name: str
) -> None:
    if summary.empty:
        summary = pd.DataFrame({"Message": ["No zero denominators."]})
    summary.to_excel(writer, sheet_name=sheet_name, index=False)


def main() -> None:
    FOLDS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_analysis_data(RAW_DATA_FILE)
    numeric_columns = feature_columns(data)

    missing_summary = pd.DataFrame(
        {
            "Feature": numeric_columns,
            "Missing_count": [int(data[column].isna().sum()) for column in numeric_columns],
            "Missing_percent": [float(data[column].isna().mean() * 100.0) for column in numeric_columns],
        }
    ).sort_values(["Missing_percent", "Feature"], ascending=[False, True])
    class_summary = data[TYPE_COL].value_counts().rename_axis(TYPE_COL).reset_index(name="Count")
    input_summary = pd.DataFrame(
        {
            "Item": ["Input samples", "Input columns", "Input feature columns", "Missing feature cells"],
            "Value": [len(data), data.shape[1], len(numeric_columns), int(data[numeric_columns].isna().sum().sum())],
        }
    )
    with pd.ExcelWriter(FOLDS_DIR / "input_data_summary.xlsx", engine="openpyxl") as writer:
        input_summary.to_excel(writer, sheet_name="input_summary", index=False)
        class_summary.to_excel(writer, sheet_name="class_distribution", index=False)
        missing_summary.to_excel(writer, sheet_name="missing_values", index=False)

    y = data[TYPE_COL].astype(str)
    splitter = cross_validator()
    splits = list(splitter.split(data[numeric_columns], y))
    fold_records: list[dict[str, object]] = []
    split_records: list[dict[str, object]] = []

    for method in IMPUTATION_METHODS:
        method_dir = FOLDS_DIR / method
        method_dir.mkdir(parents=True, exist_ok=True)
        for fold, (train_indices, test_indices) in enumerate(splits, start=1):
            train_raw = data.iloc[train_indices].copy().reset_index(drop=True)
            test_raw = data.iloc[test_indices].copy().reset_index(drop=True)
            split_records.append(
                {
                    "Method": method,
                    "Fold": fold,
                    "Train_indices": ",".join(map(str, train_indices.tolist())),
                    "Test_indices": ",".join(map(str, test_indices.tolist())),
                }
            )

            bounds = fit_iqr_bounds_on_train(train_raw, numeric_columns)
            train_nan, train_outliers = apply_iqr_bounds(train_raw, numeric_columns, bounds)
            test_nan, test_outliers = apply_iqr_bounds(test_raw, numeric_columns, bounds)
            train_imputed, test_imputed, imputation = foldwise_impute(
                train_nan, test_nan, numeric_columns, method
            )
            train_ratios, train_zero = add_ratio_features(train_imputed)
            test_ratios, test_zero = add_ratio_features(test_imputed)

            candidate_features = [
                column for column in train_ratios.columns if column not in NON_FEATURE_COLS
            ]
            if len(candidate_features) != 443:
                raise ValueError(f"Expected 443 candidate features; found {len(candidate_features)}")
            require_finite_features(train_ratios, candidate_features, f"{method} fold {fold} train")
            require_finite_features(test_ratios, candidate_features, f"{method} fold {fold} test")

            train_file = method_dir / f"fold_{fold:02d}_train_with_ratios.xlsx"
            test_file = method_dir / f"fold_{fold:02d}_test_with_ratios.xlsx"
            detail_file = method_dir / f"fold_{fold:02d}_preprocessing_details.xlsx"
            train_ratios.to_excel(train_file, index=False)
            test_ratios.to_excel(test_file, index=False)
            with pd.ExcelWriter(detail_file, engine="openpyxl") as writer:
                bounds.to_excel(writer, sheet_name="train_iqr_bounds", index=False)
                train_outliers.to_excel(writer, sheet_name="train_outliers", index=False)
                test_outliers.to_excel(writer, sheet_name="test_outliers", index=False)
                imputation.to_excel(writer, sheet_name="imputation_info", index=False)
                write_zero_summary(writer, train_zero, "train_ratio_zero_den")
                write_zero_summary(writer, test_zero, "test_ratio_zero_den")

            record: dict[str, object] = {
                "Method": method,
                "Fold": fold,
                "Train_sample_count": len(train_ratios),
                "Test_sample_count": len(test_ratios),
                "Train_outliers_marked_as_NaN": int(train_outliers["Outlier_count_marked_as_NaN"].sum()),
                "Test_outliers_marked_as_NaN": int(test_outliers["Outlier_count_marked_as_NaN"].sum()),
                "Train_values_filled": int(imputation["Train_missing_filled"].sum()),
                "Test_values_filled": int(imputation["Test_missing_filled"].sum()),
                "Feature_column_count_with_ratios": len(candidate_features),
                "Final_column_count_with_metadata": train_ratios.shape[1],
                "Train_file": str(train_file),
                "Test_file": str(test_file),
                "Detail_file": str(detail_file),
            }
            for granite_type in ("A", "S", "I"):
                record[f"Train_class_{granite_type}"] = int((train_ratios[TYPE_COL] == granite_type).sum())
                record[f"Test_class_{granite_type}"] = int((test_ratios[TYPE_COL] == granite_type).sum())
            fold_records.append(record)
            print(
                f"{method} fold {fold}: train={len(train_ratios)}, "
                f"test={len(test_ratios)}, features={len(candidate_features)}"
            )

    pd.DataFrame(fold_records).to_excel(
        FOLDS_DIR / "foldwise_preprocessing_summary.xlsx", index=False
    )
    pd.DataFrame(split_records).to_excel(FOLDS_DIR / "cv_split_indices.xlsx", index=False)
    print(f"Fold data written to: {FOLDS_DIR}")


if __name__ == "__main__":
    main()
