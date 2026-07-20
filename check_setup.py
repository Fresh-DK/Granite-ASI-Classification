from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from granite_ml.config import RAW_DATA_FILE


REQUIRED_MODULES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "scikit-learn": "sklearn",
    "openpyxl": "openpyxl",
    "xlrd": "xlrd",
    "xgboost": "xgboost",
    "shap": "shap",
    "Pillow": "PIL",
}


def main() -> None:
    missing = [package for package, module in REQUIRED_MODULES.items() if importlib.util.find_spec(module) is None]
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python: {sys.executable}")
    print(f"Source workbook: {RAW_DATA_FILE}")
    print(f"Source workbook exists: {RAW_DATA_FILE.exists()}")
    if missing:
        print("Missing Python packages: " + ", ".join(missing))
    else:
        print("All required Python packages are importable.")
    if missing:
        raise SystemExit(1)
    if not RAW_DATA_FILE.exists():
        print("Source workbook is not present; dependency check passed and data validation was skipped.")
        return
    from granite_ml.data import prepare_source_data

    processed, change_log, raw, header_row = prepare_source_data()
    print(f"Detected header row (zero-based): {header_row}")
    print(f"Raw worksheet rows: {len(raw)}")
    print(f"Prepared source samples: {len(processed)}")
    print(f"Prepared class counts: {processed['Type'].value_counts().to_dict()}")
    print(f"Documented source changes: {len(change_log)}")
    print("Setup check passed.")


if __name__ == "__main__":
    main()
