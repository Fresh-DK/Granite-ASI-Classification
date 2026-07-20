from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["savefig.dpi"] = 1000

from granite_ml.config import (  # noqa: E402
    CHAMPION_DIR,
    FOLDS_DIR,
    RAW_DATA_FILE,
    RESULTS_DIR,
    STABILITY_DIR,
)

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# ============================================================
# 0. 路径配置
# ============================================================

RESULT_BASE_DIR = RESULTS_DIR

DATA_ROOT = FOLDS_DIR

OUT_DIR = RESULTS_DIR / "10_traditional_baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_XLSX = os.path.join(
    OUT_DIR,
    "10_traditional_diagram_baseline_results.xlsx"
)

TYPE_COL = "Type"

IMPUTATION_METHODS = ["global_mean", "knn"]
N_OUTER_FOLDS = 5

CLASS_ORDER = ["A", "S", "I"]
ALL_PRED_LABELS = ["A", "S", "I", "Unclassified"]


# ============================================================
# 1. 经典判别阈值
# ============================================================

# Whalen-type A-type granite diagrams
GA_AL_THRESHOLD = 2.6
FEOSTAR_MGO_THRESHOLD = 10.0
# Conversion from total Fe reported as Fe2O3t to FeO* equivalent:
# FeO* = 0.8998 * Fe2O3t
FE2O3T_TO_FEOSTAR = 0.8998
ZR_THRESHOLD = 250.0
HFSE_SUM_THRESHOLD = 350.0

# Aluminous index for S/I separation
ACNK_S_THRESHOLD = 1.1


# ============================================================
# 2. 基础工具函数
# ============================================================

def normalize_type_value(v):
    s = str(v).strip()

    if s in ["A", "A-type", "A-Type", "A_TYPE", "A type", "A型"]:
        return "A"
    if s in ["S", "S-type", "S-Type", "S_TYPE", "S type", "S型"]:
        return "S"
    if s in ["I", "I-type", "I-Type", "I_TYPE", "I type", "I型"]:
        return "I"

    return s


def feature_key(name):
    s = str(name).strip()
    s = s.replace("×", "*")
    s = s.replace("：", ":")
    s = s.replace("（", "(").replace("）", ")")
    s = re.sub(r"\s+", "", s)
    return s.lower()


def resolve_one_feature(candidates, columns):
    columns = [str(c).strip() for c in columns]

    for f in candidates:
        if f in columns:
            return f

    candidate_keys = [feature_key(f) for f in candidates]

    for c in columns:
        if feature_key(c) in candidate_keys:
            return c

    return None


def get_numeric_series(df, candidates):
    col = resolve_one_feature(candidates, df.columns)

    if col is None:
        return None, None

    s = pd.to_numeric(df[col], errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)

    return s, col


def read_fold_test_data(method, fold):
    test_path = os.path.join(
        DATA_ROOT,
        method,
        f"fold_{fold:02d}_test_with_ratios.xlsx"
    )

    if not os.path.exists(test_path):
        raise FileNotFoundError(test_path)

    df = pd.read_excel(test_path)
    df.columns = [str(c).strip() for c in df.columns]

    if TYPE_COL not in df.columns:
        raise ValueError(f"缺少标签列：{TYPE_COL}")

    df[TYPE_COL] = df[TYPE_COL].apply(normalize_type_value)

    df["Imputation_method"] = method
    df["Outer_fold"] = fold

    return df


# ============================================================
# 3. 提取或计算传统判别指标
# ============================================================

def add_traditional_indices(df):
    out = df.copy()
    source_log = {}

    # ---------- A/CNK ----------
    acnk, acnk_col = get_numeric_series(out, ["A/CNK", "ACNK", "A_CNK"])
    if acnk is not None:
        out["TRAD_A_CNK"] = acnk
        source_log["TRAD_A_CNK"] = acnk_col
    else:
        al2o3, al_col = get_numeric_series(out, ["Al2O3", "Al2O3(wt%)", "A12O3", "A12O3(wt%)"])
        cao, ca_col = get_numeric_series(out, ["CaO", "CaO(wt%)"])
        na2o, na_col = get_numeric_series(out, ["Na2O", "Na2O(wt%)"])
        k2o, k_col = get_numeric_series(out, ["K2O", "K2O(wt%)"])

        if al2o3 is not None and cao is not None and na2o is not None and k2o is not None:
            mol_al2o3 = al2o3 / 101.9613
            mol_cao = cao / 56.0774
            mol_na2o = na2o / 61.9789
            mol_k2o = k2o / 94.1960

            out["TRAD_A_CNK"] = mol_al2o3 / (mol_cao + mol_na2o + mol_k2o)
            source_log["TRAD_A_CNK"] = f"computed from {al_col}, {ca_col}, {na_col}, {k_col}"
        else:
            out["TRAD_A_CNK"] = np.nan
            source_log["TRAD_A_CNK"] = "missing"

    # ---------- 10000*Ga/Al ----------
    gaal, gaal_col = get_numeric_series(
        out,
        [
            "10000*Ga/Al",
            "10000×Ga/Al",
            "10000*Ga/A1",
            "Ga/Al*10000"
        ]
    )

    if gaal is not None:
        out["TRAD_10000_Ga_Al"] = gaal
        source_log["TRAD_10000_Ga_Al"] = gaal_col
    else:
        ga, ga_col = get_numeric_series(out, ["Ga", "Ga(ppm)"])
        al2o3, al_col = get_numeric_series(out, ["Al2O3", "Al2O3(wt%)", "A12O3", "A12O3(wt%)"])

        if ga is not None and al2o3 is not None:
            # Al2O3 wt% -> Al ppm
            al_wt_percent = al2o3 * (2 * 26.9815 / 101.9613)
            al_ppm = al_wt_percent * 10000.0
            out["TRAD_10000_Ga_Al"] = 10000.0 * ga / al_ppm.replace(0, np.nan)
            source_log["TRAD_10000_Ga_Al"] = f"computed from {ga_col}, {al_col}"
        else:
            out["TRAD_10000_Ga_Al"] = np.nan
            source_log["TRAD_10000_Ga_Al"] = "missing"

    # ---------- FeO*/MgO ----------
    # Whalen-type diagrams use FeO*/MgO. If only total Fe as Fe2O3t is available,
    # convert it to FeO* equivalent using FeO* = 0.8998 * Fe2O3t.
    #
    # The output column TRAD_FeOstar_MgO represents FeO*/MgO throughout the script.
    fe_mg, fe_mg_col = get_numeric_series(
        out,
        [
            "FeOt/MgO",
            "FeO*/MgO",
            "FeOstar/MgO",
            "FeO/MgO",
            "R_Major_FeOt/MgO",
            "R_Major_FeOstar/MgO",
            "R_Major_FeO/MgO",
        ]
    )

    if fe_mg is not None:
        out["TRAD_FeOstar_MgO"] = fe_mg
        source_log["TRAD_FeOstar_MgO"] = f"{fe_mg_col} used directly as FeO*/MgO"
    else:
        # If a precomputed Fe2O3t/MgO ratio is present, convert the ratio.
        fe2_mg, fe2_mg_col = get_numeric_series(
            out,
            [
                "Fe2O3t/MgO",
                "R_Major_Fe2O3t/MgO",
            ]
        )

        if fe2_mg is not None:
            out["TRAD_FeOstar_MgO"] = FE2O3T_TO_FEOSTAR * fe2_mg
            source_log["TRAD_FeOstar_MgO"] = (
                f"computed as {FE2O3T_TO_FEOSTAR} * {fe2_mg_col} "
                f"to approximate FeO*/MgO"
            )
        else:
            # Otherwise compute the ratio from Fe and MgO columns.
            fe, fe_col = get_numeric_series(
                out,
                [
                    "FeOt",
                    "FeOt(wt%)",
                    "FeO*",
                    "FeO*(wt%)",
                    "FeOstar",
                    "FeOstar(wt%)",
                    "FeO",
                    "FeO(wt%)",
                    "Fe2O3t",
                    "Fe2O3t(wt%)",
                ]
            )
            mg, mg_col = get_numeric_series(out, ["MgO", "MgO(wt%)"])

            if fe is not None and mg is not None:
                if "fe2o3t" in feature_key(fe_col):
                    out["TRAD_FeOstar_MgO"] = (
                        FE2O3T_TO_FEOSTAR * fe / mg.replace(0, np.nan)
                    )
                    source_log["TRAD_FeOstar_MgO"] = (
                        f"computed as {FE2O3T_TO_FEOSTAR} * {fe_col} / {mg_col} "
                        f"to approximate FeO*/MgO"
                    )
                else:
                    out["TRAD_FeOstar_MgO"] = fe / mg.replace(0, np.nan)
                    source_log["TRAD_FeOstar_MgO"] = (
                        f"computed from {fe_col}, {mg_col}; treated as FeO*/MgO"
                    )
            else:
                out["TRAD_FeOstar_MgO"] = np.nan
                source_log["TRAD_FeOstar_MgO"] = "missing"

    # ---------- Zr ----------
    zr, zr_col = get_numeric_series(out, ["Zr", "Zr(ppm)"])
    if zr is not None:
        out["TRAD_Zr"] = zr
        source_log["TRAD_Zr"] = zr_col
    else:
        out["TRAD_Zr"] = np.nan
        source_log["TRAD_Zr"] = "missing"

    # ---------- Zr+Nb+Ce+Y ----------
    hfse_sum, hfse_col = get_numeric_series(
        out,
        [
            "Zr+Nb+Ce+Y",
            "Zr_Nb_Ce_Y",
            "Zr+Nb+Ce+Y(ppm)"
        ]
    )

    if hfse_sum is not None:
        out["TRAD_Zr_Nb_Ce_Y"] = hfse_sum
        source_log["TRAD_Zr_Nb_Ce_Y"] = hfse_col
    else:
        zr, zr_col = get_numeric_series(out, ["Zr", "Zr(ppm)"])
        nb, nb_col = get_numeric_series(out, ["Nb", "Nb(ppm)"])
        ce, ce_col = get_numeric_series(out, ["Ce", "Ce(ppm)"])
        y, y_col = get_numeric_series(out, ["Y", "Y(ppm)"])

        if zr is not None and nb is not None and ce is not None and y is not None:
            out["TRAD_Zr_Nb_Ce_Y"] = zr + nb + ce + y
            source_log["TRAD_Zr_Nb_Ce_Y"] = f"computed from {zr_col}, {nb_col}, {ce_col}, {y_col}"
        else:
            out["TRAD_Zr_Nb_Ce_Y"] = np.nan
            source_log["TRAD_Zr_Nb_Ce_Y"] = "missing"

    return out, source_log


# ============================================================
# 4. Three conventional A-type discrimination rules
# ============================================================

def diagram_a_is_A(df):
    """
    Rule a: 10000*Ga/Al vs FeO*/MgO
    A-type if:
    10000*Ga/Al >= 2.6 and FeO*/MgO >= 10
    If the dataset reports Fe2O3t instead of FeO*, Fe2O3t is converted to FeO*
    using FeO* = 0.8998 * Fe2O3t before applying this threshold.
    """
    gaal = df["TRAD_10000_Ga_Al"]
    femg = df["TRAD_FeOstar_MgO"]

    valid = gaal.notna() & femg.notna()
    is_a = valid & (gaal >= GA_AL_THRESHOLD) & (femg >= FEOSTAR_MGO_THRESHOLD)

    return is_a, valid


def diagram_b_is_A(df):
    """
    Rule b: 10000*Ga/Al vs Zr
    A-type if:
    10000*Ga/Al >= 2.6 and Zr >= 250 ppm
    """
    gaal = df["TRAD_10000_Ga_Al"]
    zr = df["TRAD_Zr"]

    valid = gaal.notna() & zr.notna()
    is_a = valid & (gaal >= GA_AL_THRESHOLD) & (zr >= ZR_THRESHOLD)

    return is_a, valid


def diagram_c_is_A(df):
    """
    Rule c: 10000*Ga/Al vs Zr+Nb+Ce+Y
    A-type if:
    10000*Ga/Al >= 2.6 and Zr+Nb+Ce+Y >= 350 ppm
    """
    gaal = df["TRAD_10000_Ga_Al"]
    hfse = df["TRAD_Zr_Nb_Ce_Y"]

    valid = gaal.notna() & hfse.notna()
    is_a = valid & (gaal >= GA_AL_THRESHOLD) & (hfse >= HFSE_SUM_THRESHOLD)

    return is_a, valid


def acnk_si_prediction(df):
    """
    A/CNK >= 1.1 -> S
    A/CNK < 1.1  -> I
    missing -> Unclassified
    """
    pred = pd.Series("Unclassified", index=df.index, dtype=object)

    acnk = df["TRAD_A_CNK"]
    valid = acnk.notna()

    pred.loc[valid & (acnk >= ACNK_S_THRESHOLD)] = "S"
    pred.loc[valid & (acnk < ACNK_S_THRESHOLD)] = "I"

    return pred, valid


# ============================================================
# 5. 传统规则 baseline
# ============================================================

def rule_a_plus_acnk(df):
    """
    Apply rule a to identify A-type samples, then use A/CNK for S/I.
    """
    pred = pd.Series("Unclassified", index=df.index, dtype=object)

    is_a, valid_a = diagram_a_is_A(df)
    si_pred, valid_acnk = acnk_si_prediction(df)

    pred.loc[is_a] = "A"

    non_a = (~is_a) & valid_acnk
    pred.loc[non_a] = si_pred.loc[non_a]

    return pred


def rule_b_plus_acnk(df):
    """
    Apply rule b to identify A-type samples, then use A/CNK for S/I.
    """
    pred = pd.Series("Unclassified", index=df.index, dtype=object)

    is_a, valid_b = diagram_b_is_A(df)
    si_pred, valid_acnk = acnk_si_prediction(df)

    pred.loc[is_a] = "A"

    non_a = (~is_a) & valid_acnk
    pred.loc[non_a] = si_pred.loc[non_a]

    return pred


def rule_c_plus_acnk(df):
    """
    Apply rule c to identify A-type samples, then use A/CNK for S/I.
    """
    pred = pd.Series("Unclassified", index=df.index, dtype=object)

    is_a, valid_c = diagram_c_is_A(df)
    si_pred, valid_acnk = acnk_si_prediction(df)

    pred.loc[is_a] = "A"

    non_a = (~is_a) & valid_acnk
    pred.loc[non_a] = si_pred.loc[non_a]

    return pred


def rule_majority_abc_plus_acnk(df):
    """
    Rule majority:
    三个 A-type 图解中至少两个判为 A-type -> A
    否则用 A/CNK 判断 S/I。
    """
    is_a1, valid_a = diagram_a_is_A(df)
    is_a2, valid_b = diagram_b_is_A(df)
    is_a3, valid_c = diagram_c_is_A(df)

    vote_count = is_a1.astype(int) + is_a2.astype(int) + is_a3.astype(int)

    pred = pd.Series("Unclassified", index=df.index, dtype=object)
    is_a_majority = vote_count >= 2

    pred.loc[is_a_majority] = "A"

    si_pred, valid_acnk = acnk_si_prediction(df)

    non_a = (~is_a_majority) & valid_acnk
    pred.loc[non_a] = si_pred.loc[non_a]

    return pred


def rule_conservative_abc_plus_acnk(df):
    """
    Conservative rule:
    至少两个有效图解判为 A-type -> A；
    如果三个图解都缺失或无法判断，则 Unclassified；
    其余样品用 A/CNK 判断 S/I。
    """
    is_a1, valid_a = diagram_a_is_A(df)
    is_a2, valid_b = diagram_b_is_A(df)
    is_a3, valid_c = diagram_c_is_A(df)

    vote_count = is_a1.astype(int) + is_a2.astype(int) + is_a3.astype(int)
    valid_count = valid_a.astype(int) + valid_b.astype(int) + valid_c.astype(int)

    pred = pd.Series("Unclassified", index=df.index, dtype=object)

    is_a_majority = (valid_count >= 2) & (vote_count >= 2)
    can_evaluate_a = valid_count >= 2

    pred.loc[is_a_majority] = "A"

    si_pred, valid_acnk = acnk_si_prediction(df)

    non_a = can_evaluate_a & (~is_a_majority) & valid_acnk
    pred.loc[non_a] = si_pred.loc[non_a]

    return pred


RULES = {
    "Rule_a_GaAl_FeOstarMgO_plus_ACNK": rule_a_plus_acnk,
    "Rule_b_GaAl_Zr_plus_ACNK": rule_b_plus_acnk,
    "Rule_c_GaAl_HFSE_plus_ACNK": rule_c_plus_acnk,
    "Rule_majority_abc_plus_ACNK": rule_majority_abc_plus_acnk,
    "Rule_conservative_abc_plus_ACNK": rule_conservative_abc_plus_acnk,
}

# Human-readable labels for figures and summary outputs.
RULE_DISPLAY_NAMES = {
    "Rule_a_GaAl_FeOstarMgO_plus_ACNK": "Rule a: 10000×Ga/Al–FeO*/MgO + A/CNK",
    "Rule_b_GaAl_Zr_plus_ACNK": "Rule b: 10000×Ga/Al–Zr + A/CNK",
    "Rule_c_GaAl_HFSE_plus_ACNK": "Rule c: 10000×Ga/Al–Zr+Nb+Ce+Y + A/CNK",
    "Rule_majority_abc_plus_ACNK": "Majority rule: ≥2 A-type diagrams + A/CNK",
    "Rule_conservative_abc_plus_ACNK": "Conservative rule: ≥2 valid A-type diagrams + A/CNK",
}

METHOD_DISPLAY_NAMES = {
    "global_mean": "GM",
    "knn": "KNN",
}

METRIC_DISPLAY_NAMES = {
    "Coverage": "Coverage",
    "Strict_macro_F1_all": "Strict Macro-F1 on all samples",
    "Strict_balanced_accuracy_all": "Strict balanced accuracy on all samples",
    "Covered_macro_F1": "Macro-F1 among covered samples",
}


# ============================================================
# 6. 评价函数
# ============================================================

def evaluate_rule(y_true, y_pred):
    y_true = pd.Series(y_true).astype(str).reset_index(drop=True)
    y_pred = pd.Series(y_pred).astype(str).reset_index(drop=True)

    covered = y_pred.isin(CLASS_ORDER)

    n_total = len(y_true)
    n_covered = int(covered.sum())

    coverage = n_covered / n_total if n_total > 0 else np.nan
    unclassified_ratio = 1.0 - coverage

    # strict: Unclassified 视为错误
    strict_accuracy = float((y_true == y_pred).mean())

    strict_balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred
    )

    strict_macro_p, strict_macro_r, strict_macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        average="macro",
        zero_division=0
    )

    strict_weighted_p, strict_weighted_r, strict_weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        average="weighted",
        zero_division=0
    )

    out = {
        "N_total": n_total,
        "N_covered": n_covered,
        "Coverage": coverage,
        "Unclassified_ratio": unclassified_ratio,

        "Strict_accuracy_all": strict_accuracy,
        "Strict_balanced_accuracy_all": strict_balanced_accuracy,
        "Strict_macro_precision_all": strict_macro_p,
        "Strict_macro_recall_all": strict_macro_r,
        "Strict_macro_F1_all": strict_macro_f1,
        "Strict_weighted_F1_all": strict_weighted_f1,
    }

    if n_covered > 0:
        yt = y_true[covered]
        yp = y_pred[covered]

        covered_accuracy = accuracy_score(yt, yp)
        covered_balanced_accuracy = balanced_accuracy_score(yt, yp)

        covered_macro_p, covered_macro_r, covered_macro_f1, _ = precision_recall_fscore_support(
            yt,
            yp,
            labels=CLASS_ORDER,
            average="macro",
            zero_division=0
        )

        out.update({
            "Covered_accuracy": covered_accuracy,
            "Covered_balanced_accuracy": covered_balanced_accuracy,
            "Covered_macro_precision": covered_macro_p,
            "Covered_macro_recall": covered_macro_r,
            "Covered_macro_F1": covered_macro_f1,
        })
    else:
        out.update({
            "Covered_accuracy": np.nan,
            "Covered_balanced_accuracy": np.nan,
            "Covered_macro_precision": np.nan,
            "Covered_macro_recall": np.nan,
            "Covered_macro_F1": np.nan,
        })

    return out


def classwise_metrics(y_true, y_pred):
    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        output_dict=True,
        zero_division=0
    )

    rows = []

    for cls in CLASS_ORDER:
        rows.append({
            "Class": cls,
            "Precision": report[cls]["precision"],
            "Recall": report[cls]["recall"],
            "F1": report[cls]["f1-score"],
            "Support": report[cls]["support"],
        })

    return rows


def confusion_long(y_true, y_pred):
    labels = CLASS_ORDER + ["Unclassified"]

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    rows = []

    for i, true_cls in enumerate(labels):
        for j, pred_cls in enumerate(labels):
            rows.append({
                "True_class": true_cls,
                "Pred_class": pred_cls,
                "Count": int(cm[i, j])
            })

    return rows


def flatten_columns(df):
    df = df.copy()

    df.columns = [
        "_".join([str(x) for x in col if str(x) != ""])
        if isinstance(col, tuple)
        else str(col)
        for col in df.columns
    ]

    return df


def save_excel_with_autowidth(path, sheets):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            if df is None or df.empty:
                continue

            safe_sheet = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_sheet, index=False)

            ws = writer.book[safe_sheet]
            ws.freeze_panes = "A2"

            for col_cells in ws.columns:
                max_len = 0
                col_letter = col_cells[0].column_letter

                for cell in col_cells:
                    try:
                        value = str(cell.value) if cell.value is not None else ""
                        max_len = max(max_len, len(value))
                    except Exception:
                        pass

                ws.column_dimensions[col_letter].width = min(max(max_len + 2, 10), 45)


# ============================================================
# 7. 主程序
# ============================================================

all_fold_rows = []
all_classwise_rows = []
all_confusion_rows = []
all_prediction_rows = []
all_source_rows = []

for method in IMPUTATION_METHODS:
    print(f"\n========== Imputation method: {method} ==========")

    for fold in range(1, N_OUTER_FOLDS + 1):
        print(f"\nOuter fold {fold}")

        df = read_fold_test_data(method, fold)
        df, source_log = add_traditional_indices(df)

        for index_name, source in source_log.items():
            all_source_rows.append({
                "Imputation_method": method,
                "Outer_fold": fold,
                "Index_name": index_name,
                "Source": source
            })

        y_true = df[TYPE_COL].astype(str)

        for rule_name, rule_func in RULES.items():
            y_pred = rule_func(df)

            metrics = evaluate_rule(y_true, y_pred)

            row = {
                "Baseline_type": "Traditional_diagram_rule",
                "Rule": rule_name,
                "Imputation_method": method,
                "Outer_fold": fold,
                "GA_AL_THRESHOLD": GA_AL_THRESHOLD,
                "FEOSTAR_MGO_THRESHOLD": FEOSTAR_MGO_THRESHOLD,
                "ZR_THRESHOLD": ZR_THRESHOLD,
                "HFSE_SUM_THRESHOLD": HFSE_SUM_THRESHOLD,
                "ACNK_S_THRESHOLD": ACNK_S_THRESHOLD,
            }

            row.update(metrics)
            all_fold_rows.append(row)

            for r in classwise_metrics(y_true, y_pred):
                rr = {
                    "Rule": rule_name,
                    "Imputation_method": method,
                    "Outer_fold": fold,
                }
                rr.update(r)
                all_classwise_rows.append(rr)

            for r in confusion_long(y_true, y_pred):
                rr = {
                    "Rule": rule_name,
                    "Imputation_method": method,
                    "Outer_fold": fold,
                }
                rr.update(r)
                all_confusion_rows.append(rr)

            pred_df = pd.DataFrame({
                "Imputation_method": method,
                "Outer_fold": fold,
                "Rule": rule_name,
                "True_type": y_true.values,
                "Pred_type": y_pred.values,
                "Covered": y_pred.isin(CLASS_ORDER).values,
                "Correct": (y_true.values == y_pred.values),

                "A_CNK": df["TRAD_A_CNK"].values,
                "Ga_Al_10000": df["TRAD_10000_Ga_Al"].values,
                "FeOstar_MgO": df["TRAD_FeOstar_MgO"].values,
                "Zr": df["TRAD_Zr"].values,
                "Zr_Nb_Ce_Y": df["TRAD_Zr_Nb_Ce_Y"].values,
            })

            all_prediction_rows.append(pred_df)

            print(
                f"{rule_name}: "
                f"Coverage={metrics['Coverage']:.3f}, "
                f"Strict Macro-F1={metrics['Strict_macro_F1_all']:.3f}, "
                f"Strict Balanced Acc={metrics['Strict_balanced_accuracy_all']:.3f}, "
                f"Covered Macro-F1={metrics['Covered_macro_F1']:.3f}"
            )


fold_metrics_df = pd.DataFrame(all_fold_rows)
classwise_df = pd.DataFrame(all_classwise_rows)
confusion_df = pd.DataFrame(all_confusion_rows)
prediction_df = pd.concat(all_prediction_rows, axis=0, ignore_index=True)
source_log_df = pd.DataFrame(all_source_rows)


# ============================================================
# 8. 汇总统计
# ============================================================

summary_df = (
    fold_metrics_df
    .groupby(["Rule", "Imputation_method"], as_index=False)
    .agg({
        "Coverage": ["mean", "std"],
        "Unclassified_ratio": ["mean", "std"],

        "Strict_accuracy_all": ["mean", "std"],
        "Strict_balanced_accuracy_all": ["mean", "std"],
        "Strict_macro_precision_all": ["mean", "std"],
        "Strict_macro_recall_all": ["mean", "std"],
        "Strict_macro_F1_all": ["mean", "std"],
        "Strict_weighted_F1_all": ["mean", "std"],

        "Covered_accuracy": ["mean", "std"],
        "Covered_balanced_accuracy": ["mean", "std"],
        "Covered_macro_precision": ["mean", "std"],
        "Covered_macro_recall": ["mean", "std"],
        "Covered_macro_F1": ["mean", "std"],
    })
)

summary_df = flatten_columns(summary_df)
summary_df = summary_df.rename(columns={
    "Rule_": "Rule",
    "Imputation_method_": "Imputation_method"
})
summary_df.insert(1, "Rule_label", summary_df["Rule"].map(RULE_DISPLAY_NAMES).fillna(summary_df["Rule"]))
summary_df.insert(3, "Imputation_label", summary_df["Imputation_method"].map(METHOD_DISPLAY_NAMES).fillna(summary_df["Imputation_method"]))


overall_summary_df = (
    fold_metrics_df
    .groupby(["Rule"], as_index=False)
    .agg({
        "Coverage": ["mean", "std"],
        "Unclassified_ratio": ["mean", "std"],

        "Strict_accuracy_all": ["mean", "std"],
        "Strict_balanced_accuracy_all": ["mean", "std"],
        "Strict_macro_precision_all": ["mean", "std"],
        "Strict_macro_recall_all": ["mean", "std"],
        "Strict_macro_F1_all": ["mean", "std"],
        "Strict_weighted_F1_all": ["mean", "std"],

        "Covered_accuracy": ["mean", "std"],
        "Covered_balanced_accuracy": ["mean", "std"],
        "Covered_macro_precision": ["mean", "std"],
        "Covered_macro_recall": ["mean", "std"],
        "Covered_macro_F1": ["mean", "std"],
    })
)

overall_summary_df = flatten_columns(overall_summary_df)
overall_summary_df = overall_summary_df.rename(columns={"Rule_": "Rule"})
overall_summary_df.insert(1, "Rule_label", overall_summary_df["Rule"].map(RULE_DISPLAY_NAMES).fillna(overall_summary_df["Rule"]))


classwise_summary_df = (
    classwise_df
    .groupby(["Rule", "Imputation_method", "Class"], as_index=False)
    .agg({
        "Precision": ["mean", "std"],
        "Recall": ["mean", "std"],
        "F1": ["mean", "std"],
        "Support": ["mean", "std"],
    })
)

classwise_summary_df = flatten_columns(classwise_summary_df)
classwise_summary_df = classwise_summary_df.rename(columns={
    "Rule_": "Rule",
    "Imputation_method_": "Imputation_method",
    "Class_": "Class",
})
classwise_summary_df.insert(1, "Rule_label", classwise_summary_df["Rule"].map(RULE_DISPLAY_NAMES).fillna(classwise_summary_df["Rule"]))
classwise_summary_df.insert(3, "Imputation_label", classwise_summary_df["Imputation_method"].map(METHOD_DISPLAY_NAMES).fillna(classwise_summary_df["Imputation_method"]))


# ============================================================
# 9. 作图
# ============================================================

def save_metric_bar(summary_df, metric, out_png, title):
    metric_col = f"{metric}_mean"

    if metric_col not in summary_df.columns:
        print("找不到指标列：", metric_col)
        return

    sub = summary_df.copy()
    sub["Rule_label"] = sub["Rule"].map(RULE_DISPLAY_NAMES).fillna(sub["Rule"].astype(str))
    sub["Method_label"] = sub["Imputation_method"].map(METHOD_DISPLAY_NAMES).fillna(sub["Imputation_method"].astype(str))
    sub["Label"] = sub["Rule_label"] + " | " + sub["Method_label"]
    sub = sub.sort_values(metric_col, ascending=True)

    plt.figure(figsize=(13, max(5, 0.55 * len(sub))), dpi=1000)

    plt.barh(sub["Label"], sub[metric_col])

    xlabel = METRIC_DISPLAY_NAMES.get(metric, metric)
    plt.xlabel(xlabel, fontsize=12, fontweight="bold")
    plt.ylabel("Traditional rule | Imputation workflow", fontsize=12, fontweight="bold")
    plt.title(title, fontsize=14, fontweight="bold")

    for i, v in enumerate(sub[metric_col].values):
        if not np.isnan(v):
            plt.text(v, i, f"{v:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    plt.close()


save_metric_bar(
    summary_df,
    "Strict_macro_F1_all",
    os.path.join(OUT_DIR, "10_traditional_diagrams_strict_macroF1.png"),
    "Traditional diagram baselines: strict Macro-F1 on all samples"
)

save_metric_bar(
    summary_df,
    "Strict_balanced_accuracy_all",
    os.path.join(OUT_DIR, "10_traditional_diagrams_strict_balanced_accuracy.png"),
    "Traditional diagram baselines: strict balanced accuracy on all samples"
)

save_metric_bar(
    summary_df,
    "Coverage",
    os.path.join(OUT_DIR, "10_traditional_diagrams_coverage.png"),
    "Traditional diagram baselines: coverage"
)

save_metric_bar(
    summary_df,
    "Covered_macro_F1",
    os.path.join(OUT_DIR, "10_traditional_diagrams_covered_macroF1.png"),
    "Traditional diagram baselines: Macro-F1 among covered samples"
)


# ============================================================
# 10. 输出 Excel
# ============================================================

sheets = {
    "fold_metrics": fold_metrics_df,
    "summary_by_method": summary_df,
    "overall_summary": overall_summary_df,
    "classwise_metrics": classwise_df,
    "classwise_summary": classwise_summary_df,
    "confusion_matrix_long": confusion_df,
    "predictions_with_indices": prediction_df,
    "traditional_index_sources": source_log_df,
}

save_excel_with_autowidth(OUT_XLSX, sheets)


# ============================================================
# 11. 打印核心结果
# ============================================================

print("\n========== 10 传统判别图 baseline 完成 ==========")
print("输出目录：", OUT_DIR)
print("Excel 文件：", OUT_XLSX)

print("\n========== overall_summary ==========")
print(overall_summary_df)

print("\n========== summary_by_method ==========")
print(summary_df)

print("\n========== classwise_summary ==========")
print(classwise_summary_df)


