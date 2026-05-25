# Discriminants to High-Dimensional Ratio Data: An Explainable Machine Learning Study of Granite Classification

This repository contains the data and code associated with the study:

**Discriminants to High-Dimensional Ratio Data: An Explainable Machine Learning Study of Granite Classification**

The project provides a leakage-controlled and interpretable machine-learning workflow for A/S/I petrogenetic classification of Mesozoic granites from the South China Block using whole-rock geochemical data.

The repository includes data preprocessing, fold-wise imputation, systematic ratio-feature construction, Spearman correlation clustering, cluster-champion selection, stable feature-set evaluation, multi-model validation, traditional geochemical discrimination baselines, class-imbalance sensitivity analysis, and SHAP-based interpretation.

## Citation Requirement

If you use any data, code, results, figures, tables, or workflow from this repository, you must cite the accompanying paper.

The final journal citation and DOI will be updated here after publication.

Because the dataset was compiled from published geological studies, users should also cite the original data sources where appropriate. The source publications are listed in:

```text
MLGranite Source References.txt
```

## Overview

Classifying granites into A-, S-, and I-type petrogenetic categories is important for understanding continental crustal evolution and magmatic processes. Traditional geochemical discrimination diagrams are geologically interpretable, but they are limited in integrating multielement information from large, high-dimensional, and strongly correlated whole-rock geochemical datasets.

This repository implements a reproducible workflow for:

- Compiling and cleaning whole-rock geochemical data of Mesozoic granites from the South China Block;
- Constructing systematic within-group geochemical ratio features;
- Avoiding information leakage through fold-internal preprocessing and feature selection;
- Reducing feature redundancy using train-only Spearman correlation clustering;
- Selecting stable cluster champions using SHAP importance and inner-CV Top-K stability;
- Evaluating stable feature sets across multiple classifiers and imputation workflows;
- Comparing machine-learning models with traditional diagram-based baselines;
- Interpreting the final stable feature model using SHAP.

The stable novel ratios identified in this study should be interpreted as **complementary geochemical signals**, not as fixed thresholds, universal rules, or standalone petrogenetic criteria.

## Repository Structure

```text
.
├── Code/
│   ├── A_Data_Preprocessing/
│   │   ├── 01_foldwise_preprocessing_and_ratio_feature_construction.ipynb
│   │   └── 02_foldwise_normality_test.ipynb
│   │
│   ├── B_Feature_Correlation_and_Selection/
│   │   ├── 03_foldwise_spearman_correlation_analysis.ipynb
│   │   ├── 04_foldwise_correlation_clustering_champion_selection_rho_sensitivity.ipynb
│   │   └── 05_stable_cluster_champions_and_novel_ratio_candidates.ipynb
│   │
│   ├── C_Novel_Ratio_Analysis/
│   │   ├── 06_stable_novel_ratio_group_contribution_experiment.ipynb
│   │   └── 07_stable_novel_ratio_class_distribution_and_statistical_tests.ipynb
│   │
│   ├── D_Classifiers/
│   │   ├── KNN.ipynb
│   │   ├── LR.ipynb
│   │   ├── SVM.ipynb
│   │   ├── RF.ipynb
│   │   ├── ET.ipynb
│   │   ├── GBDT.ipynb
│   │   ├── MLP.ipynb
│   │   └── summarize_all_classifiers.ipynb
│   │
│   ├── E_Model_Sensitivity_and_Baselines/
│   │   ├── 09_class_weight_sensitivity_analysis.ipynb
│   │   └── 10_traditional_diagram_baseline_comparison.ipynb
│   │
│   └── F_SHAP_Interpretation/
│       └── 11_final_model_shap_interpretation.ipynb
│
├── Data/
│   └── SCB-Mesozoic-Granite.xls
│
├── MLGranite Source References.txt
├── requirements.txt
└── README.md
```

After running the notebooks, output files will be generated in a `Result/` directory.

Before public release, it is recommended to remove local Jupyter cache folders, such as:

```text
.ipynb_checkpoints/
```

## Data Description

### Main Data File

```text
Data/SCB-Mesozoic-Granite.xls
```

The dataset contains whole-rock geochemical data for Mesozoic granite samples from the South China Block.

The original compilation includes **1,341 samples** manually assembled from **204 published geological studies**. After initial data cleaning, **1,288 samples** were retained for modeling:

| Type | Number of samples |
|---|---:|
| A-type | 721 |
| S-type | 162 |
| I-type | 405 |

The dataset includes major elements, trace elements, rare earth elements, locality information, age information, lithology, and A/S/I petrogenetic labels where available.

The A/S/I labels were assigned according to the classifications reported in the source literature and checked against available petrographic and geochemical descriptions. Samples with controversial classifications, unclear sources, irrecoverable non-numeric records, or obviously inconsistent metadata were excluded from the final modeling dataset.

### Source Reference List

```text
MLGranite Source References.txt
```

This file lists the 204 published geological studies used to compile the dataset.

Users of this repository should cite both the accompanying paper and the relevant original source publications where appropriate.

## Installation

The code was tested with Python 3.9.x.

A recommended setup is:

```bash
conda create -n mlgranite python=3.9
conda activate mlgranite
pip install -r requirements.txt
```

The main dependencies include:

```text
numpy
pandas
scipy
scikit-learn
xgboost
shap
torch
matplotlib
openpyxl
xlrd
notebook
jupyterlab
ipykernel
ipython
```

For the PyTorch-based MLP classifier, users may need to install a `torch` version compatible with their local CPU or CUDA environment.

## Quick Start

Start JupyterLab from the repository root:

```bash
jupyter lab
```

Run the notebooks in numerical order.

First, run preprocessing, ratio-feature construction, correlation analysis, and stable feature selection:

```text
01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07
```

Then run the classifier notebooks:

```text
KNN.ipynb
LR.ipynb
SVM.ipynb
RF.ipynb
ET.ipynb
GBDT.ipynb
MLP.ipynb
summarize_all_classifiers.ipynb
```

Finally, run the sensitivity analysis, traditional baseline comparison, and SHAP interpretation notebooks:

```text
09_class_weight_sensitivity_analysis.ipynb
10_traditional_diagram_baseline_comparison.ipynb
11_final_model_shap_interpretation.ipynb
```

## Workflow

### 1. Data Preprocessing and Ratio-Feature Construction

Notebook:

```text
Code/A_Data_Preprocessing/01_foldwise_preprocessing_and_ratio_feature_construction.ipynb
```

Main steps:

- Standardize symbolic numerical records;
- Convert records with explicit numerical values, such as `<0.01` and `>100`, to their numerical parts;
- Treat records without recoverable numerical information, such as `<d.l.`, `bdl`, `below detection limit`, and `> upper limit`, as missing values;
- Remove sample rows containing non-empty strings that remain unparseable after cleaning;
- Remove samples with insufficient geochemical information;
- Construct an outer stratified five-fold split;
- Perform fold-internal IQR outlier handling;
- Compare two label-free imputation workflows:
  - global mean imputation, abbreviated as GM;
  - KNN imputation;
- Construct systematic ratio features within two predefined groups:
  - major elements;
  - trace elements + rare earth elements.

For each pair of variables, only one ratio direction is generated according to a predefined variable order to avoid reciprocal redundancy. For geological interpretation and visualization, selected ratios may be discussed in a more intuitive direction, whereas the original predefined computational directions are retained in all modeling analyses.

### 2. Spearman Correlation Clustering and Cluster-Champion Selection

Notebooks:

```text
Code/B_Feature_Correlation_and_Selection/03_foldwise_spearman_correlation_analysis.ipynb
Code/B_Feature_Correlation_and_Selection/04_foldwise_correlation_clustering_champion_selection_rho_sensitivity.ipynb
Code/B_Feature_Correlation_and_Selection/05_stable_cluster_champions_and_novel_ratio_candidates.ipynb
```

Main steps:

- Calculate Spearman correlation matrices only within outer training folds;
- Construct high-correlation feature networks using `|rho| >= 0.90`;
- Define correlation clusters as connected components of the high-correlation network;
- Select one representative cluster champion from each correlation cluster;
- Score candidate champions using SHAP importance and inner-CV Top-K stability.

The cluster-champion score is defined as:

```text
Champion score = mean(|SHAP|) × FreqTopK
```

where `FreqTopK` is the frequency with which a feature appears in the Top-K SHAP-importance list across inner validation folds. In this study, `K = 50`.

Stable features are defined by aggregating recurrent cluster champions across the two imputation workflows and five outer folds.

### 3. Stable Novel Ratio Analysis

Notebooks:

```text
Code/C_Novel_Ratio_Analysis/06_stable_novel_ratio_group_contribution_experiment.ipynb
Code/C_Novel_Ratio_Analysis/07_stable_novel_ratio_class_distribution_and_statistical_tests.ipynb
```

Main steps:

- Compare feature-set performance among classical indicators, stable novel ratios, classical indicators + stable novel ratios, cluster champions, and related feature subsets;
- Evaluate whether stable novel ratios provide complementary discriminatory information beyond classical geochemical indicators;
- Examine class-wise distributions of stable novel ratios among A-, S-, and I-type granites;
- Apply the Kruskal-Wallis test to evaluate class-wise distributional differences;
- Apply Benjamini-Hochberg correction for multiple testing.

The purpose of this analysis is not to establish new single-ratio thresholds or independent classification rules. Stable novel ratios are interpreted as complementary geochemical signals that may improve multivariate A/S/I discrimination when used together with classical indicators and other geochemical variables.

### 4. Multi-Model Validation

Notebooks:

```text
Code/D_Classifiers/KNN.ipynb
Code/D_Classifiers/LR.ipynb
Code/D_Classifiers/SVM.ipynb
Code/D_Classifiers/RF.ipynb
Code/D_Classifiers/ET.ipynb
Code/D_Classifiers/GBDT.ipynb
Code/D_Classifiers/MLP.ipynb
Code/D_Classifiers/summarize_all_classifiers.ipynb
```

Seven supervised classifiers are evaluated:

| Abbreviation | Model |
|---|---|
| KNN | k-nearest neighbors |
| LR | logistic regression |
| SVM | support vector machine |
| RF | random forest |
| ET | extremely randomized trees / ExtraTrees |
| GBDT | gradient boosting decision trees |
| MLP | multilayer perceptron |

All classifiers are evaluated using the same outer stratified five-fold test sets.

The main evaluation metrics are:

```text
Accuracy
Balanced accuracy
Macro Precision
Macro-F1
```

The main results use the default class-weight setting:

```text
class_weight = None
```

Class-weight sensitivity is evaluated separately.

### 5. Class-Imbalance Sensitivity and Traditional Baselines

Notebooks:

```text
Code/E_Model_Sensitivity_and_Baselines/09_class_weight_sensitivity_analysis.ipynb
Code/E_Model_Sensitivity_and_Baselines/10_traditional_diagram_baseline_comparison.ipynb
```

Class-imbalance sensitivity compares:

```text
class_weight = None
class_weight = balanced
```

The analysis focuses on:

```text
Delta Macro-F1
Delta Balanced accuracy
Delta S-type Recall
Delta S-type F1
```

Traditional diagram-based baseline rules include:

```text
Rule a: 10000×Ga/Al >= 2.6 and FeOt/MgO >= 10
Rule b: 10000×Ga/Al >= 2.6 and Zr >= 250 ppm
Rule c: 10000×Ga/Al >= 2.6 and Zr+Nb+Ce+Y >= 350 ppm
```

Samples not classified as A-type by the corresponding rule are further classified as S-type when:

```text
A/CNK >= 1.1
```

Otherwise, they are classified as I-type.

Majority and conservative combined rules are also evaluated.

These traditional rules are used only as baseline classifiers. They do not participate in machine-learning model training, feature selection, or SHAP interpretation.

### 6. SHAP Interpretation

Notebook:

```text
Code/F_SHAP_Interpretation/11_final_model_shap_interpretation.ipynb
```

The final SHAP interpretation uses:

```text
ExtraTrees + stable feature set + KNN-imputation workflow
```

The final refitted model is used only for interpretation and visualization, not for reporting apparent predictive performance.

The SHAP analysis includes:

- global SHAP feature importance;
- class-wise SHAP feature importance;
- geological comparison between model-ranked features and established A/S/I granite geochemical understanding.

Important model-ranked features include classical A-type and aluminosity-related signals, such as:

```text
Zr+Nb+Ce+Y
10000×Ga/Al
Hf
REE
Nb
Ga
P2O5
A/NK
A/CNK
```

The model also highlights complementary ratio features, such as:

```text
Y/Pb
Nb/Pb
Al2O3/K2O
Y/Cs
```

These features can be discussed in relation to HFSE/LILE fractionation, REE fractionation, alumina saturation, feldspar-mica control, Fe-Ti-P systems, Fe-Ti oxide fractionation, and apatite-related processes. However, SHAP values quantify model attribution and should not be interpreted as direct proof of causal petrogenetic mechanisms.

## Leakage-Controlled Design

A central goal of this repository is to provide a leakage-controlled workflow for geochemical machine learning.

The following steps are fitted only within the training portion of each outer fold:

```text
IQR outlier boundaries
GM/KNN imputation parameters
KNN scaling parameters
Spearman correlation matrix
Correlation clustering
SHAP-based cluster-champion scoring
Feature selection
Model training
```

The held-out test folds are used only for final model evaluation.

This workflow avoids type-stratified imputation and type-stratified outlier handling, because such procedures would use A/S/I labels during preprocessing and could introduce information leakage.

## Reproducible Outputs

Running the notebooks can reproduce the main analyses of the study, including:

- data overview after initial cleaning;
- missing-value statistics;
- fold-wise Spearman correlation structure;
- correlation-cluster structure;
- stable cluster-champion selection;
- stable novel-ratio contribution analysis;
- class-wise distributions and statistical tests of stable novel ratios;
- multi-model validation under GM and KNN imputation workflows;
- class-weight sensitivity analysis;
- traditional discrimination baseline comparison;
- global and class-wise SHAP interpretation.

## Important Notes

1. The stable feature set is a machine-learning feature subset selected under a leakage-controlled workflow.
2. Stable novel ratios should be interpreted as complementary geochemical signals.
3. Stable novel ratios should not be used as fixed-threshold rules or standalone petrogenetic criteria.
4. SHAP values describe feature contributions to model predictions, not direct causal mechanisms.
5. The trained models are primarily evaluated for Mesozoic granites from the South China Block.
6. Application to other regions or global datasets requires independent validation or retraining.
7. Whole-rock geochemical machine-learning results should be used together with petrography, mineral chemistry, isotope data, geochronology, and field geological evidence.

## Reproducibility Notes

- Please keep the original directory structure when running the notebooks.
- It is recommended to launch JupyterLab from the repository root.
- The `Result/` directory will be generated automatically by the notebooks.
- Some notebooks depend on outputs from previous notebooks, especially those generated by notebooks 03-05.
- A complete rerun may take a long time, especially for SHAP-based feature selection and final SHAP interpretation.
- Minor numerical differences may occur across operating systems, Python versions, BLAS backends, and hardware environments.

## Citation

If you use any part of this repository, please cite the accompanying paper:

```text
Hu, D., Hong, J., Wang, H., and Gan, C.
Discriminants to High-Dimensional Ratio Data:
An Explainable Machine Learning Study of Granite Classification.
```

The final journal citation and DOI will be updated after publication.

Users should also cite the original geological data sources where appropriate. The full source list is provided in:

```text
MLGranite Source References.txt
```

## License

Please check the license file before using this repository.

Suggested license options before public release:

```text
Code: MIT License or BSD-3-Clause License
Data: CC BY 4.0, subject to the reuse conditions of the original source publications
```

Because this dataset was compiled from published literature, users should cite both the accompanying paper and the relevant original source publications when reusing the data.

## Contact

For questions about the data, code, or workflow, please contact the corresponding authors listed in the paper.
