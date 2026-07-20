from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["savefig.dpi"] = 1000

import matplotlib.pyplot as plt
import pandas as pd

from granite_ml.config import RAW_DATA_FILE, RESULTS_DIR
from granite_ml.features import CLASS_ORDER, TYPE_COL
from granite_ml.io import feature_columns, load_analysis_data


OUTPUT_DIR = RESULTS_DIR / "00_data_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 12,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.linewidth": 1.4,
            "xtick.major.width": 1.3,
            "ytick.major.width": 1.3,
        }
    )


def main() -> None:
    data = load_analysis_data(RAW_DATA_FILE)
    features = feature_columns(data)

    class_counts = (
        data[TYPE_COL]
        .value_counts()
        .reindex(CLASS_ORDER, fill_value=0)
        .rename_axis(TYPE_COL)
        .reset_index(name="Count")
    )
    class_counts["Percent"] = class_counts["Count"] / len(data) * 100.0

    missing = pd.DataFrame(
        {
            "Feature": features,
            "Missing_count": [int(data[column].isna().sum()) for column in features],
            "Missing_percent": [float(data[column].isna().mean() * 100.0) for column in features],
        }
    ).sort_values(["Missing_percent", "Feature"], ascending=[False, True])

    summary = pd.DataFrame(
        {
            "Item": [
                "Input sample count",
                "Input column count",
                "Feature count",
                "Total missing feature cells",
                "Maximum missing features in one sample",
            ],
            "Value": [
                len(data),
                data.shape[1],
                len(features),
                int(data[features].isna().sum().sum()),
                int(data[features].isna().sum(axis=1).max()),
            ],
        }
    )

    workbook = OUTPUT_DIR / "input_data_summary.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="input_summary", index=False)
        class_counts.to_excel(writer, sheet_name="class_distribution", index=False)
        missing.to_excel(writer, sheet_name="missing_values", index=False)

    configure_plot_style()

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    bars = ax.bar(class_counts[TYPE_COL], class_counts["Count"], color=["#2166AC", "#B2182B", "#4D9221"])
    ax.set_xlabel("Granite type")
    ax.set_ylabel("Number of samples")
    ax.set_title("Class distribution in the analysis-ready dataset")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, class_counts["Count"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{int(value)}", ha="center", va="bottom", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "class_distribution.png", dpi=1000, bbox_inches="tight")
    plt.close(fig)

    missing_nonzero = missing.loc[missing["Missing_count"] > 0].copy()
    if not missing_nonzero.empty:
        fig_width = max(9.0, 0.42 * len(missing_nonzero))
        fig, ax = plt.subplots(figsize=(fig_width, 5.5))
        ax.bar(missing_nonzero["Feature"], missing_nonzero["Missing_percent"], color="#2166AC")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Missing values (%)")
        ax.set_title("Missingness in the analysis-ready dataset")
        ax.tick_params(axis="x", rotation=55)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "feature_missingness.png", dpi=1000, bbox_inches="tight")
        plt.close(fig)

    print(f"Input dataset: {RAW_DATA_FILE}")
    print(f"Samples: {len(data)}")
    print(f"Features before ratio construction: {len(features)}")
    print(f"Class counts: {dict(zip(class_counts[TYPE_COL], class_counts['Count']))}")
    print(f"Summary workbook: {workbook}")


if __name__ == "__main__":
    main()
