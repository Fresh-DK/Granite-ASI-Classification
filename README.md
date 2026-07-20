# Granite A/S/I Classification

A reproducible Python workflow for classifying A-, S-, and I-type Mesozoic
granites from whole-rock geochemical data. The code implements fold-local data
preprocessing, systematic ratio construction, correlation-aware feature
compression, model benchmarking, statistical exploration, conventional
geochemical baselines, and SHAP interpretation.

This repository distributes source code only. Generated data, model outputs,
figures, and logs are excluded from version control.

## Key properties

- Five-fold stratified outer cross-validation with global-mean and KNN
  imputation workflows.
- IQR limits, imputation parameters, correlations, feature selection, and model
  fitting are learned from the current outer-training partition only.
- Explicit feature metadata separates 47 non-systematic features from 396
  systematically constructed ratios.
- Spearman correlation clustering uses a predefined absolute threshold of
  0.90.
- Inner five-fold XGBoost–SHAP scoring selects one representative per
  correlation cluster.
- Predictive benchmarks use fold-specific feature sets and never reuse the
  post hoc recurrence inventory for outer-test evaluation.
- All PNG outputs use 1000-dpi metadata; selected summaries also have PDF
  output.

## Repository contents

```text
.
├── scripts/          # numbered analysis steps 00–12
├── src/granite_ml/  # shared configuration and utilities
├── check_setup.py   # dependency and optional data validation
├── run_all.py       # pipeline runner
├── pyproject.toml   # package metadata and dependencies
├── requirements.txt # runtime dependency constraints
└── LICENSE          # MIT license for the source code
```

## Data availability

The dataset is not distributed in this repository. Data will be made available
from the corresponding author upon reasonable request.

After obtaining the analysis-ready dataset, place it at:

```text
data/raw/SCB-Mesozoic-Granite.csv
```

The requested dataset is supplied as an analysis-ready table. It must contain
canonical `A`, `S`, and `I` class labels and numeric geochemical feature cells.
Blank feature cells are allowed and are handled by fold-local imputation.

## Installation

Python 3.9–3.12 is supported. Create and activate an isolated environment, then
run:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
python check_setup.py
```

`check_setup.py` validates installed dependencies. If the analysis-ready
dataset is present, it also checks its structure, sample count, and class
labels; otherwise it reports that data validation was skipped.

## Running the workflow

Run every step in order:

```bash
python run_all.py
```

Run selected steps:

```bash
python run_all.py --steps 00 01 04 05
```

Resume from a specific step:

```bash
python run_all.py --from-step 08
```

Generated fold data are written to `data/processed/`, step logs to `logs/`, and
analysis outputs to `results/`. These directories are created automatically
and are ignored by Git.

## Analysis steps

| Step | Script | Purpose |
|---:|---|---|
| 00 | `scripts/00_data_audit.py` | Audit source completeness and class counts |
| 01 | `scripts/01_preprocess.py` | Clean data, create outer folds, perform fold-local imputation, and construct ratios |
| 02 | `scripts/02_normality.py` | Evaluate fold-local distributional properties |
| 03 | `scripts/03_correlations.py` | Summarize fold-local Spearman correlation structure |
| 04 | `scripts/04_select_cluster_champions.py` | Form correlation clusters and select fold-specific champions |
| 05 | `scripts/05_summarize_stability.py` | Aggregate champion recurrence for interpretation |
| 06 | `scripts/06_feature_contribution.py` | Evaluate incremental ratio value and compression retention |
| 07 | `scripts/07_exploratory_ratio_statistics.py` | Compute exploratory class-wise ratio statistics |
| 08 | `scripts/08_model_comparison.py` | Benchmark seven classifiers and four feature sets |
| 09 | `scripts/09_class_weight_sensitivity.py` | Compare unweighted and balanced SVM configurations |
| 10 | `scripts/10_traditional_baseline.py` | Evaluate rule-based geochemical baselines |
| 11 | `scripts/11_final_shap.py` | Fit the interpretation model and calculate SHAP summaries |
| 12 | `scripts/12_generate_summary_figures.py` | Generate consolidated result visualizations |

## Validation boundary

Each predictive feature set is rebuilt independently for every imputation
workflow and outer fold. The recurrence inventory generated after outer-fold
evaluation is used only for exploratory statistics and full-data
interpretation, not for estimating generalization performance.

## License

The source code is available under the [MIT License](LICENSE).
