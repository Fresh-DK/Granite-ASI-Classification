from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import CHAMPION_DIR, FOLDS_DIR, RAW_DATA_FILE
from .features import CLASS_ORDER, NON_FEATURE_COLS, RHO_TAG, TYPE_COL, normalize_type_value


def load_analysis_data(path: str | Path = RAW_DATA_FILE) -> pd.DataFrame:
    """Load the analysis-ready input table without correcting source values.

    The input contract requires canonical A/S/I labels and numeric feature
    cells (missing cells are allowed). Invalid values raise an error instead
    of being silently repaired or discarded.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing analysis-ready dataset: {path}")

    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, float_precision="round_trip")
    else:
        frame = pd.read_excel(path)
    if TYPE_COL not in frame.columns:
        raise ValueError(f"The input dataset must contain the {TYPE_COL!r} column.")
    if frame.empty:
        raise ValueError("The input dataset contains no samples.")

    labels = set(frame[TYPE_COL].dropna().astype(str))
    unexpected = sorted(labels - set(CLASS_ORDER))
    if frame[TYPE_COL].isna().any() or unexpected:
        raise ValueError(
            "The input dataset must use non-missing canonical class labels "
            f"{CLASS_ORDER}; unexpected labels: {unexpected}"
        )

    numeric_columns = feature_columns(frame)
    if not numeric_columns:
        raise ValueError("The input dataset contains no feature columns.")
    for column in numeric_columns:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Feature column {column!r} contains a non-numeric value. "
                "Provide numeric values in the analysis-ready dataset."
            ) from exc
    return frame


def fold_file(method: str, fold: int, split: str) -> Path:
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'.")
    path = FOLDS_DIR / method / f"fold_{fold:02d}_{split}_with_ratios.xlsx"
    if not path.exists():
        raise FileNotFoundError(f"Missing fold file: {path}")
    return path


def load_fold(method: str, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_excel(fold_file(method, fold, "train"))
    test = pd.read_excel(fold_file(method, fold, "test"))
    train.columns = [str(column).strip() for column in train.columns]
    test.columns = [str(column).strip() for column in test.columns]
    if TYPE_COL not in train or TYPE_COL not in test:
        raise ValueError(f"The fold files must contain the {TYPE_COL!r} column.")
    train[TYPE_COL] = train[TYPE_COL].map(normalize_type_value)
    test[TYPE_COL] = test[TYPE_COL].map(normalize_type_value)
    check_train_test_separation(train, test)
    return train, test


def champion_workbook(method: str, fold: int, rho_tag: str = RHO_TAG) -> Path:
    folder = CHAMPION_DIR / method / f"outer_fold_{fold:02d}" / f"rho_{rho_tag}"
    expected = folder / f"outer_fold_{fold:02d}_cluster_champions_SHAP_innerCV_rho{rho_tag}.xlsx"
    if expected.exists():
        return expected
    candidates = sorted(folder.glob(f"*cluster_champions*rho{rho_tag}*.xlsx"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one champion workbook in {folder}; found {candidates}")
    return candidates[0]


def load_fold_champions(method: str, fold: int, rho_tag: str = RHO_TAG) -> list[str]:
    path = champion_workbook(method, fold, rho_tag)
    table = pd.read_excel(path, sheet_name="ClusterChampions")
    if "champion" not in table:
        raise ValueError(f"Missing 'champion' column in {path}")
    champions = table["champion"].dropna().astype(str).drop_duplicates().tolist()
    if not champions:
        raise ValueError(f"No champions found in {path}")
    return champions


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [str(column) for column in frame.columns if str(column) not in NON_FEATURE_COLS]


def check_train_test_separation(train: pd.DataFrame, test: pd.DataFrame) -> None:
    for column in ("No.", "No", "Sample", "Samp1e", "Sample_ID", "SampleID", "ID"):
        if column not in train or column not in test:
            continue
        if not train[column].is_unique or not test[column].is_unique:
            continue
        overlap = set(train[column].astype(str)) & set(test[column].astype(str))
        if overlap:
            raise ValueError(f"Train/test overlap in ID column {column!r}: {sorted(overlap)[:10]}")
        return
