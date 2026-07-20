from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FOLDS_DIR = PROCESSED_DATA_DIR / "folds"

RESULTS_DIR = PROJECT_ROOT / "results"
CHAMPION_DIR = RESULTS_DIR / "04_cluster_champions"
STABILITY_DIR = RESULTS_DIR / "05_stability"
LOG_DIR = PROJECT_ROOT / "logs"

RAW_DATA_FILE = RAW_DATA_DIR / "SCB-Mesozoic-Granite.xls"
CORRECTED_DATA_DIR = PROCESSED_DATA_DIR / "source"
CORRECTED_DATA_FILE = CORRECTED_DATA_DIR / "SCB-Mesozoic-Granite-corrected.xlsx"


def ensure_project_directories() -> None:
    for directory in (
        RAW_DATA_DIR,
        CORRECTED_DATA_DIR,
        FOLDS_DIR,
        RESULTS_DIR,
        CHAMPION_DIR,
        STABILITY_DIR,
        LOG_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
