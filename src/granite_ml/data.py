from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAW_DATA_FILE
from .features import CLASS_ORDER, TYPE_COL, normalize_type_value


REQUIRED_HEADER_FIELDS = {"No.", "Type", "SiO2(wt%)", "Al2O3"}

# Corrections are deliberately tied to sample identity and the exact recorded
# token. This prevents a broad text replacement from changing unrelated data.
VALUE_CORRECTIONS = (
    {
        "No.": 225,
        "Sample": "DS-24",
        "Reference": "Liu et al. (2012)",
        "Column": "Nb",
        "Old_value": "10,0",
        "New_value": 10.0,
        "Reason": "Decimal comma in an otherwise dot-decimal numeric table.",
    },
    {
        "No.": 231,
        "Sample": "H-53",
        "Reference": "Yao et al. (2005)",
        "Column": "Tb",
        "Old_value": "l.54",
        "New_value": 1.54,
        "Reason": "Lowercase letter l is an obvious OCR/typing substitution for digit 1.",
    },
    {
        "No.": 231,
        "Sample": "H-53",
        "Reference": "Yao et al. (2005)",
        "Column": "Tm",
        "Old_value": "l.24",
        "New_value": 1.24,
        "Reason": "Lowercase letter l is an obvious OCR/typing substitution for digit 1.",
    },
)


def detect_header_row(path: str | Path, max_rows: int = 10) -> int:
    preview = pd.read_excel(path, header=None, nrows=max_rows)
    for row_index in range(len(preview)):
        values = {str(value).strip() for value in preview.iloc[row_index].dropna().tolist()}
        if REQUIRED_HEADER_FIELDS.issubset(values):
            return int(row_index)
    raise ValueError(
        f"Could not detect the data header in {path}. "
        f"Required fields: {sorted(REQUIRED_HEADER_FIELDS)}"
    )


def read_original_workbook(path: str | Path = RAW_DATA_FILE) -> tuple[pd.DataFrame, int]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    header_row = detect_header_row(path)
    frame = pd.read_excel(path, header=header_row)
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame, header_row


def _identity_mask(frame: pd.DataFrame, rule: dict[str, object]) -> pd.Series:
    no_values = pd.to_numeric(frame["No."], errors="coerce")
    return (
        no_values.eq(float(rule["No."]))
        & frame["Sample"].astype(str).str.strip().eq(str(rule["Sample"]))
        & frame["Reference"].astype(str).str.strip().eq(str(rule["Reference"]))
    )


def prepare_source_data(
    path: str | Path = RAW_DATA_FILE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    raw, header_row = read_original_workbook(path)
    processed = raw.copy()
    log_rows: list[dict[str, object]] = []

    for rule in VALUE_CORRECTIONS:
        mask = _identity_mask(processed, rule)
        if int(mask.sum()) != 1:
            raise ValueError(f"Correction rule did not match exactly one row: {rule}")
        row_index = processed.index[mask][0]
        column = str(rule["Column"])
        observed = processed.at[row_index, column]
        observed_text = str(observed).strip()
        expected_text = str(rule["Old_value"]).strip()
        new_value = float(rule["New_value"])
        if observed_text == expected_text:
            processed.at[row_index, column] = new_value
            action = "Corrected"
        elif pd.notna(observed) and np.isclose(float(observed), new_value):
            action = "Already corrected"
        else:
            raise ValueError(
                f"Unexpected value for correction rule {rule}: observed {observed!r}"
            )
        log_rows.append(
            {
                "Action": action,
                "Source_row": int(row_index) + header_row + 2,
                "No.": rule["No."],
                "Sample": rule["Sample"],
                "Type": processed.at[row_index, TYPE_COL],
                "Reference": rule["Reference"],
                "Column": column,
                "Old_value": expected_text,
                "New_value": new_value,
                "Reason": rule["Reason"],
            }
        )

    non_sample_mask = (
        processed["No."].isna()
        & processed["Sample"].isna()
        & processed[TYPE_COL].isna()
    )
    for row_index in processed.index[non_sample_mask]:
        log_rows.append(
            {
                "Action": "Excluded blank separator row",
                "Source_row": int(row_index) + header_row + 2,
                "No.": None,
                "Sample": None,
                "Type": None,
                "Reference": None,
                "Column": "Entire row",
                "Old_value": None,
                "New_value": None,
                "Reason": "Completely blank row separates the A-type and I-type sections; it is not a sample.",
            }
        )
    processed = processed.loc[~non_sample_mask].copy().reset_index(drop=True)
    processed[TYPE_COL] = processed[TYPE_COL].map(normalize_type_value)

    unexpected = sorted(set(processed[TYPE_COL].dropna().astype(str)) - set(CLASS_ORDER))
    if unexpected or processed[TYPE_COL].isna().any():
        raise ValueError(f"Unexpected or missing class labels after source preparation: {unexpected}")
    if len(processed) != 1341:
        raise ValueError(f"Expected 1,341 samples after excluding the note row; found {len(processed)}")

    return processed, pd.DataFrame(log_rows), raw, header_row


def load_source_data(path: str | Path = RAW_DATA_FILE) -> pd.DataFrame:
    processed, _, _, _ = prepare_source_data(path)
    return processed
