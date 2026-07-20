from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import CHAMPION_DIR, FOLDS_DIR
from .features import NON_FEATURE_COLS, RHO_TAG, TYPE_COL, normalize_type_value


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

